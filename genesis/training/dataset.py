from torch.utils.data import Dataset
import pickle
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
import os
import math
import numpy as np

TO_PXL = 1e3

class PileSweepData(Dataset):

    def __init__(self, path : str, run : int | None = None):
        
        parentpath = Path(__file__).parent.parent
        full_path = parentpath / path  # Adjust path as needed
        
        if run is not None:
            runs = [
                (full_path / f'{run}_data.pkl', full_path / f'{run}_config.yaml')
            ]
        else:
            runs = sorted([
                (full_path / f'{f}', full_path / f'{f.replace("_data.pkl", "_config.yaml")}')
                for f in os.listdir(full_path)
                if f.endswith('_data.pkl')
            ])
            for run in runs:
                print(run)

        self.samples = []
        self.configs = []
        for data_file, config_file in runs:
            with open(data_file, 'rb') as f:
                self.samples.append(pickle.load(f))

            
            with open(config_file, 'r') as f:
                self.configs.append(yaml.safe_load(f))
        
        # lookup tables for sample indexing
        self._run_lookup = []
        self._offsets = [0]
        for r, row in enumerate(self.samples):
            self._run_lookup.extend([r] * len(row))
            self._offsets.append(self._offsets[-1] + len(row))
        
        # take grid dimensions from first sample
        x_dim, y_dim, _ = self.configs[0]["sandbox"]["box"]["vol"]
        x_dim_plt, y_dim_plt, _ = self.configs[0]["plate"]["size"]
        self._create_grids(x_dim, y_dim, x_dim_plt, y_dim_plt)
        
        # Pre-compute base plate grid (unrotated)
        self._precompute_plate_grid(x_dim_plt, y_dim_plt)
        
        # Cache for rotated plates (angle -> rotated_grid)
        self._plate_cache = {}
    
    def _precompute_plate_grid(self, x_dim_plt, y_dim_plt):
        """Pre-compute the base plate grid"""
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
    
    def __len__(self):
        return sum(len(x) for x in self.samples)
    
    def __getitem__(self, idx : int):

        r = self._run_lookup[idx]
        sample = self.samples[r][idx - self._offsets[r]]
        config = self.configs[r]

        if isinstance(sample, dict):
            particles = sample["state"]
            particles_ = sample["state_"]
            action = sample["action"]
        else:
            particles, particles_, action = sample
        
        plate_pos, plate_pos_, angle = action

        # Update grid dimensions if config is different
        x_dim_new, y_dim_new, _ = config["sandbox"]["box"]["vol"]
        x_dim_plt, y_dim_plt, _ = self.configs[0]["plate"]["size"]

        if x_dim_new != self._box_dim[0] \
        or y_dim_new != self._box_dim[1] \
        or x_dim_plt != self._plt_dim[0] \
        or y_dim_plt != self._plt_dim[1] :
            self._create_grids(x_dim_new, y_dim_new, x_dim_plt, y_dim_plt)
            self._precompute_plate_grid(x_dim_plt, y_dim_plt)
            self._plate_cache.clear()  # Clear cache if grid size changed

        # Clear grids once
        self._clear_grids()

        particles = particles * TO_PXL
        particles_ = particles_ * TO_PXL
        plate_pos = plate_pos * TO_PXL
        plate_pos_ = plate_pos_ * TO_PXL

        ##################
        # Draw particles #
        ##################

        # Vectorized particle drawing for better performance
        self._draw_particles_vectorized(particles, particles_, self.ctr)
        
        ##############
        # Draw plate #
        ##############

        # Get or create rotated plate
        angle_deg = float(torch.rad2deg(angle))
        cache_key = (angle_deg, self._base_plate_grid.shape[0])
        
        if cache_key not in self._plate_cache:
            self._plate_cache[cache_key] = self._rotate_plate_torch(self._base_plate_grid, angle)
        
        rotated_plate = self._plate_cache[cache_key]

        # Draw plate at state
        self._color_grid(
            grid=self._a_grid,
            x=self.ctr[0] + plate_pos[0],
            y=self.ctr[1] + plate_pos[1],
            size=self._base_plate_grid.shape,
            drawing=(rotated_plate > 0.5).float() * 0.5
        )

        # Draw plate at state_
        self._color_grid(
            grid=self._a_grid,
            x=self.ctr[0] + plate_pos_[0],
            y=self.ctr[1] + plate_pos_[1],
            size=self._base_plate_grid.shape,
            drawing=(rotated_plate > 0.5).float()
        )

        return (self._grid, self._a_grid), self._grid_


    def _draw_particles_vectorized(self, particles, particles_, ctr):
        """Vectorized particle drawing for better performance"""
        # Convert to tensor if needed
        if not isinstance(particles, torch.Tensor):
            particles = torch.tensor(particles, dtype=torch.float32)
        else:
            particles = particles.clone().float()
            
        if not isinstance(particles_, torch.Tensor):
            particles_ = torch.tensor(particles_, dtype=torch.float32)
        else:
            particles_ = particles_.clone().float()
        
        # Extract positions and radii
        p_x, p_y, _, p_r = particles[:, 0], particles[:, 1], particles[:, 2], particles[:, 3]
        p_x_, p_y_, _, p_r_ = particles_[:, 0], particles_[:, 1], particles_[:, 2], particles_[:, 3]
        
        if not torch.allclose(p_r, p_r_):
            raise ValueError(f"Particle sizes don't match between states")
        
        # Draw particles at state
        for i in range(len(particles)):
            self._color_grid(
                grid=self._grid,
                x=p_x[i] + ctr[0],
                y=p_y[i] + ctr[1],
                size=(p_r[i], p_r[i]),
                drawing=1
            )
            
            # Draw particles at state_
            self._color_grid(
                grid=self._grid_,
                x=p_x_[i] + ctr[0],
                y=p_y_[i] + ctr[1],
                size=(p_r_[i], p_r_[i]),
                drawing=1
            )
    
    def _rotate_plate_torch(self, plate_grid, angle):
        """Rotate plate grid using PyTorch operations (faster than scipy)"""
        # Convert angle to radians if needed
        if isinstance(angle, torch.Tensor):
            angle_rad = angle.item()
        else:
            angle_rad = angle * math.pi / 180.0
        
        # Create 2D rotation matrix
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ], dtype=torch.float32)
        
        # Add batch and channel dimensions for grid_sample
        grid = plate_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Create sampling grid
        h, w = plate_grid.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, dtype=torch.float32),
            torch.linspace(-1, 1, w, dtype=torch.float32),
            indexing='ij'
        )
        
        # Apply rotation to sampling coordinates
        coords = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        
        # Reshape for matrix multiplication: [H*W, 2]
        coords_flat = coords.reshape(-1, 2)
        
        # Apply rotation: [H*W, 2] @ [2, 2] -> [H*W, 2]
        rotated_flat = torch.matmul(coords_flat, rotation_matrix.t())
        
        # Reshape back and add batch dimension: [1, H, W, 2]
        rotated_coords = rotated_flat.reshape(1, h, w, 2)
        
        # Sample using grid_sample (bilinear interpolation)
        rotated_grid = F.grid_sample(grid, rotated_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        return rotated_grid.squeeze(0).squeeze(0)


    def plot_grid(self, grid : torch.Tensor) -> None:
        """Visualize the grid as an image"""
        from matplotlib import pyplot as plt
        plt.imshow(grid, interpolation='nearest')
        plt.show()
    
    def _create_grids(self, x_dim, y_dim, x_dim_plt, y_dim_plt) -> None:
        """create new grid instances"""
        x_pxl, y_pxl = int(x_dim * TO_PXL), int(y_dim * TO_PXL)
        plt_pxl = int(max(x_dim_plt, y_dim_plt) * TO_PXL)

        self._grid = torch.zeros((x_pxl, y_pxl), dtype=torch.float32)
        self._grid_ = torch.zeros((x_pxl, y_pxl), dtype=torch.float32)
        self._a_grid = torch.zeros((x_pxl, y_pxl), dtype=torch.float32)
        self._plate_grid = torch.zeros((plt_pxl, plt_pxl), dtype=torch.float32)

        self.ctr = (round(x_pxl/2), round(y_pxl/2))

        self._box_dim = (x_dim, y_dim)
        self._plt_dim = (x_dim_plt, y_dim_plt)
    
    def _clear_grids(self):
        """clear grids efficiently"""
        self._grid.zero_()
        self._grid_.zero_()
        self._a_grid.zero_()
        self._plate_grid.zero_()

    def _color_grid(
    self,
    grid: torch.Tensor,
    x: float,
    y: float,
    size: tuple[float, float],
    drawing: float | torch.Tensor = 1
    ) -> None:

        h, w = grid.shape[:2]

        x_size = int(round(float(size[0])))
        y_size = int(round(float(size[1])))

        x0 = int(round(float(x - x_size / 2)))
        y0 = int(round(float(y - y_size / 2)))

        x1 = x0 + x_size
        y1 = y0 + y_size

        # Clip to grid bounds
        gx0 = max(0, x0)
        gy0 = max(0, y0)

        gx1 = min(h, x1)
        gy1 = min(w, y1)

        # Nothing visible
        if gx0 >= gx1 or gy0 >= gy1:
            return

        target = grid[gx0:gx1, gy0:gy1]

        # Only overwrite zeros in grid (avoid unnecessary operations)
        mask = (target == 0)

        if isinstance(drawing, torch.Tensor):
            # Corresponding crop inside drawing
            dx0 = gx0 - x0
            dy0 = gy0 - y0

            dx1 = dx0 + (gx1 - gx0)
            dy1 = dy0 + (gy1 - gy0)

            source = drawing[dx0:dx1, dy0:dy1]
            target[mask] = source[mask].to(torch.float32)
        else:
            target[mask] = float(drawing)
    

def main():
    dataset = PileSweepData("data/cubes/chickpeas_on_glass/")
    
    for i in range(len(dataset)):
        print(i)
        input, label = dataset[i]
        # dataset.plot_grid(input[0])  # Plot particle grid
        # dataset.plot_grid(input[1])  # Plot action grid
        # dataset.plot_grid(label)     # Plot next state grid


if __name__ == "__main__":
    main()
        
        
