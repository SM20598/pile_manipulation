from dataclasses_json import config
from torch.utils.data import Dataset
import pickle
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
import os
import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

TO_PXL = 1e3


class PileSweepData(Dataset):

    def __init__(
            self,
            paths: list[str] | str,
            run: int | None = None
        ):
        """
        Initialize dataset with either a folder containing data or a specific run.

            @param paths: list of folder paths or a single folder path containing data files
            @param run: number of a specific run
        """
        self.runs = []
        self.configs = []
        self._run_lengths = []
        self._plate_cache = {}
        self._physics = torch.zeros((3,), dtype=torch.float32)


        parentpath = Path(__file__).parent.parent
        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            full_path = parentpath / "data" / path
            if not full_path.exists():
                raise FileNotFoundError(f"Data folder not found: {full_path}")

            if run is not None:
                run_names = [str(run)]
            else:
                run_names = sorted(
                    {f.split("_data.")[0] for f in os.listdir(full_path) if f.endswith("_data.pt")}
                )

            for run_name in run_names:

                data_file = full_path / f"{run_name}_data.pt"
                if not data_file.exists():
                    raise FileNotFoundError(f"Data file not found: {data_file}")

                config_file = full_path / f"{run_name}_config.yaml"
                if not config_file.exists():
                    raise FileNotFoundError(f"Config file not found: {config_file}")

                self.runs.append(torch.load(data_file, map_location="cpu"))
                self.configs.append(yaml.safe_load(config_file.read_text()))
                self._run_lengths.append(self._count_samples_in_run(self.runs[-1]))
        
        if not self.configs:
            raise ValueError("No configs found for dataset.")

        if not self.runs:    
            raise ValueError("No data samples found for dataset.")

        # create lookup table
        self._run_lookup = []
        self._offsets = [0]
        for r, num_samples in enumerate(self._run_lengths):
            self._run_lookup.extend([r] * num_samples)
            self._offsets.append(self._offsets[-1] + num_samples)

        self._create_grids(self.configs[0])
        
    def _create_grids(
            self, 
            config
        ) -> None:
        """
        Allocates tensors for all occupancy grids.

            @param config: A config to initialize grids.
        """

        # Box grid, dimension, and center
        x_dim, y_dim, _ = config["box"]["vol"]
        x_pxl, y_pxl = int(x_dim * TO_PXL), int(y_dim * TO_PXL)

        self._grid = torch.zeros((2, x_pxl, y_pxl), dtype=torch.float32)
        self._grid_ = torch.zeros((x_pxl, y_pxl), dtype=torch.float32)

        self._box_dim = (x_dim, y_dim)
        self.ctr = (round(x_pxl / 2), round(y_pxl / 2))

        # Plate grid and dimension
        x_dim_plt, y_dim_plt, _ = self._get_plate_dims(config)
        self._precompute_plate_grid(x_dim_plt, y_dim_plt)
        self._plt_dim = (x_dim_plt, y_dim_plt)

    def _precompute_plate_grid(
            self,
            x_dim_plt,
            y_dim_plt
        ) -> None:
        """
        Draws an occupancy grid for the plate in neutral position.
        """
        
        plt_pxl = int(max(x_dim_plt, y_dim_plt) * TO_PXL)
        self._base_plate_grid = torch.zeros((plt_pxl, plt_pxl))
        
        # Draw plate on grid
        self._color_grid(
            grid=self._base_plate_grid,
            x=plt_pxl/2,
            y=plt_pxl/2,
            size=(x_dim_plt*TO_PXL, y_dim_plt*TO_PXL),
            drawing=1
        )

    def _count_samples_in_run(self, run):
        return run["states"].shape[0]

    def _get_plate_dims(self, config):
        return config["plate"]["size"]


    def _extract_sample_in_pxl(self, run, index):
        """
        Returns the sample at given index. Positions are converted from meters to pixels.

            @param run: data run.
            @param index: index of sample in run
        """
        particles = run["states"][index] * TO_PXL
        particles_ = run["states_"][index] * TO_PXL
        plate_pos = run["p_starts"][index] * TO_PXL
        plate_pos_ = run["p_stops"][index] * TO_PXL
        angle = run["angles"][index]

        return particles, particles_, plate_pos, plate_pos_, angle

    def _draw_particle_grid(self, particle_states, grid, config):

        num_particles = config["material"]["num_particles"]
        shape = config["material"]["shape"]
        particle_sizes = config["data_collection"]["sampled"]["particle_size"]
        grid_np = grid.numpy()

        for idx in range(num_particles):
            particle_state = particle_states[idx]
            dimensions = particle_sizes[idx]
            center_x = float(particle_state[0]) + self.ctr[0]
            center_y = float(particle_state[1]) + self.ctr[1]
            
            if shape == "sphere":
                radius, _, _ = dimensions * TO_PXL / 2
                cv2.circle(
                    grid_np,
                    (int(round(center_x)), int(round(center_y))),
                    int(round(radius)),
                    color=1,
                    thickness=-1,
                )
                continue

            width, height = float(dimensions[0]), float(dimensions[1])
            half_w = width * 0.5
            half_h = height * 0.5

            corners = np.array(
                [
                    [-half_w, -half_h],
                    [half_w, -half_h],
                    [half_w, half_h],
                    [-half_w, half_h],
                ],
                dtype=np.float32,
            )

            rot = Rotation.from_quat(particle_state[2:])
            yaw = float(rot.as_euler("xyz", degrees=False)[2])
            if abs(yaw) > 1e-6:
                rotation = np.array(
                    [
                        [math.cos(yaw), -math.sin(yaw)],
                        [math.sin(yaw), math.cos(yaw)],
                    ],
                    dtype=np.float32,
                )
                corners = corners @ rotation.T

            pts = np.round(corners + np.array([center_x, center_y], dtype=np.float32)).astype(np.int32)
            cv2.fillPoly(grid_np, [pts], color=1)

    def _rotate_plate_torch(self, plate_grid, angle):
        if isinstance(angle, torch.Tensor):
            angle_rad = float(angle.item())
        else:
            angle_rad = float(angle)

        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        rotation_matrix = torch.tensor(
            [[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32
        )

        grid = plate_grid.unsqueeze(0).unsqueeze(0)
        h, w = plate_grid.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, dtype=torch.float32),
            torch.linspace(-1, 1, w, dtype=torch.float32),
            indexing="ij",
        )

        coords = torch.stack([grid_x, grid_y], dim=-1)
        coords_flat = coords.reshape(-1, 2)
        rotated_flat = torch.matmul(coords_flat, rotation_matrix.t())
        rotated_coords = rotated_flat.reshape(1, h, w, 2)
        rotated_grid = F.grid_sample(
            grid, rotated_coords, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return rotated_grid.squeeze(0).squeeze(0)

    def plot_grid(self, grid: torch.Tensor, title: str = "") -> None:
        """Visualize the grid as an image"""
        from matplotlib import pyplot as plt

        plt.imshow(grid, interpolation="nearest")
        plt.title(title)
        plt.show()

    def _clear_grids(self):
        self._grid.zero_()
        self._grid_.zero_()

    def _color_grid(
            self,
            grid: torch.Tensor,
            x: float,
            y: float,
            size: tuple[float, float],
            drawing: float | torch.Tensor = 1,
        ) -> None:
        
        h, w = grid.shape[:2]
        x_size = int(round(float(size[0])))
        y_size = int(round(float(size[1])))

        x0 = int(round(float(x - x_size / 2)))
        y0 = int(round(float(y - y_size / 2)))
        x1 = x0 + x_size
        y1 = y0 + y_size

        gx0 = max(0, x0)
        gy0 = max(0, y0)
        gx1 = min(h, x1)
        gy1 = min(w, y1)
        if gx0 >= gx1 or gy0 >= gy1:
            return

        target = grid[gx0:gx1, gy0:gy1]
        mask = target == 0

        if isinstance(drawing, torch.Tensor):
            dx0 = gx0 - x0
            dy0 = gy0 - y0
            dx1 = dx0 + (gx1 - gx0)
            dy1 = dy0 + (gy1 - gy0)
            source = drawing[dx0:dx1, dy0:dy1]
            target[mask] = source[mask].to(torch.float32)
        else:
            target[mask] = float(drawing)

    def _det_physics(self, config):
        self._physics[0] = config["material"]["friction"]
        self._physics[1] = config["material"]["density"]
        self._physics[2] = config["box"]["friction"]
        # self._physics[3] = config["plate"]["speed"]

    def __getitem__(self, idx: int):
        self._clear_grids()

        # look up sample and config in table
        run_idx = self._run_lookup[idx]
        run = self.runs[run_idx]
        config = self.configs[run_idx]
        sample_index = idx - self._offsets[run_idx]

        # extract sample at given index
        particles, particles_, plate_pos, plate_pos_, angle = self._extract_sample_in_pxl(run, sample_index)

        self._draw_particle_grid(particles, self._grid[0], config)
        self._draw_particle_grid(particles_, self._grid_, config)


        # if dataset includes plates of different sizes
        x_dim_plt, y_dim_plt, _ = self._get_plate_dims(config)
        if (x_dim_plt != self._plt_dim[0] or y_dim_plt != self._plt_dim[1]):
            self._precompute_plate_grid(x_dim_plt, y_dim_plt)
            self._plate_cache.clear()

        rotated_plate = self._rotate_plate_torch(self._base_plate_grid, angle)

        self._color_grid(
            grid=self._grid[1],
            x=self.ctr[0] + float(plate_pos[0]),
            y=self.ctr[1] + float(plate_pos[1]),
            size=self._base_plate_grid.shape,
            drawing=(rotated_plate > 0.5).float() * 0.5,
        )
        self._color_grid(
            grid=self._grid[1],
            x=self.ctr[0] + float(plate_pos_[0]),
            y=self.ctr[1] + float(plate_pos_[1]),
            size=self._base_plate_grid.shape,
            drawing=(rotated_plate > 0.5).float(),
        )

        self._det_physics(config)
        return (self._grid, self._physics), self._grid_


def main():
    dataset = PileSweepData("data/cubes/chickspheres_on_glass/")

    for i in range(min(len(dataset), 3)):
        inputs, label = dataset[i]
        particle_grid, action_grid = inputs
        dataset.plot_grid(particle_grid, title=f"particles {i}")
        dataset.plot_grid(action_grid, title=f"plate action {i}")
        dataset.plot_grid(label, title=f"next state {i}")


if __name__ == "__main__":
    main()
        
        
