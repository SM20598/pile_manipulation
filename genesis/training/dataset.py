from torch.utils.data import Dataset
import pickle
from pathlib import Path
import yaml
import torch
import os
import math
from scipy.ndimage import rotate

TO_PXL = 1e3

class PileSweepData(Dataset):

    def __init__(self, path : str, run : int | None = None):
        
        parentpath = Path(__file__).parent.parent
        full_path = parentpath / path  # Adjust path as needed
        
        if run is not None:
            runs = [run]
        else:
            runs = range(sum(entry.is_file() for entry in os.scandir(full_path)) // 2)

        self.samples = []
        self.configs = []
        for run in runs:
            with open(full_path / f'{run}_data.pkl', 'rb') as f:
                self.samples.append(pickle.load(f))

            
            with open(full_path / f'{run}_config.yaml', 'r') as f:
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
        else:
            self._clear_grids()

        particles = particles * TO_PXL
        particles_ = particles_ * TO_PXL
        plate_pos = plate_pos * TO_PXL
        plate_pos_ = plate_pos_ * TO_PXL

        ##################
        # Draw particles #
        ##################

        for p, p_ in zip(particles, particles_):
            p_x, p_y, _, p_r = p
            p_x_, p_y_, _, p_r_ = p_

            if p_r != p_r_:
                raise ValueError(f"Particle sizes at state {p_r} and state_ {p_r_} don't match") 

            # Draw particles at state
            self._color_grid(
                grid=self._grid,
                x=p_x + self.ctr[0],
                y=p_y + self.ctr[1],
                size=(p_r, p_r),
                drawing=1
            )

            # Draw particles at state_
            self._color_grid(
                grid=self._grid_,
                x=p_x_ + self.ctr[0],
                y=p_y_ + self.ctr[1],
                size=(p_r_, p_r_),
                drawing=1
            )
        
        ##############
        # Draw plate #
        ##############

        # Draw plate on separate grid
        self._color_grid(
            grid=self._plate_grid,
            x=self._plate_grid.shape[0]/2,
            y=self._plate_grid.shape[1]/2,
            size=(x_dim_plt*TO_PXL, y_dim_plt*TO_PXL),
            drawing=1
        )

        # rotate grid according to sample angle
        rotated_plate = rotate(self._plate_grid, angle=torch.rad2deg(angle), reshape=False, order=1)

        # Draw plate at state
        self._color_grid(
            grid=self._a_grid,
            x=self.ctr[0] + plate_pos[0],
            y=self.ctr[1] + plate_pos[1],
            size=self._plate_grid.shape,
            drawing=torch.from_numpy((rotated_plate > 0.5).astype(int)*0.5)
        )

        # Draw plate at state_
        self._color_grid(
            grid=self._a_grid,
            x=self.ctr[0] + plate_pos_[0],
            y=self.ctr[1] + plate_pos_[1],
            size=self._plate_grid.shape,
            drawing=torch.from_numpy((rotated_plate > 0.5).astype(int))
        )

        return (self._grid, self._a_grid), self._grid_


    def plot_grid(self, grid : torch.Tensor) -> None:
        """Visualize the grid as an image"""
        from matplotlib import pyplot as plt
        plt.imshow(grid, interpolation='nearest')
        plt.show()
    
    def _create_grids(self, x_dim, y_dim, x_dim_plt, y_dim_plt) -> None:
        """create new grid instances"""
        x_pxl, y_pxl = int(x_dim * TO_PXL), int(y_dim * TO_PXL)
        plt_pxl = int(max(x_dim_plt, y_dim_plt) * TO_PXL)

        self._grid = torch.zeros((x_pxl, y_pxl))
        self._grid_ = torch.zeros((x_pxl, y_pxl))
        self._a_grid = torch.zeros((x_pxl, y_pxl))
        self._plate_grid = torch.zeros((plt_pxl, plt_pxl))

        self.ctr = (round(x_pxl/2), round(y_pxl/2))

        self._box_dim = (x_dim, y_dim)
        self._plt_dim = (x_dim_plt, y_dim_plt)
    
    def _clear_grids(self):
        """clear grids"""
        self._grid *= 0
        self._grid_ *= 0
        self._a_grid *= 0
        self._plate_grid *= 0

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

        # Only overwrite zeros in grid
        mask = (target == 0)

        if isinstance(drawing, torch.Tensor):

            # Corresponding crop inside drawing
            dx0 = gx0 - x0
            dy0 = gy0 - y0

            dx1 = dx0 + (gx1 - gx0)
            dy1 = dy0 + (gy1 - gy0)

            source = drawing[dx0:dx1, dy0:dy1]

            target[mask] = source[mask].to(torch.float)

        else:
            target[mask] = drawing
    

def main():
    dataset = PileSweepData("data/cubes/chickpeas_on_glass/")
    
    for i in range(len(dataset)):
        input, label = dataset[i]
        # dataset.plot_grid(label)


if __name__ == "__main__":
    main()
        
        
