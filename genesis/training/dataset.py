from torch.utils.data import Dataset
import pickle
from pathlib import Path
import yaml
import torch
import os
import math
from scipy.ndimage import rotate

TO_PXL = 1e3

class PileData(Dataset):

    def __init__(self, path : str, run : int | None = None):
        
        parentpath = Path(__file__).parent.parent
        full_path = parentpath / path  # Adjust path as needed
        
        if run is not None:
            runs = [run]
        else:
            n_runs = int(len([name for name in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, name))])/2)

        for run in runs:
            with open(full_path / f'{run}_data.pkl', 'rb') as f:
                self.samples = pickle.load(f)
            
            with open(full_path / f'{run}_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
                
                x_dim, y_dim, _ = config["sandbox"]["box"]["vol"]
                x_dim = int(x_dim * TO_PXL)
                y_dim = int(y_dim * TO_PXL)
                c_x = round(x_dim/2)
                c_y = round(x_dim/2)
                
                plate_dim_x, plate_dim_y, _ = [round(x * TO_PXL) for x in config["plate"]["size"]]
            
            # iterate over all n samples
            inputs1 = []
            labels = []
            for sample in self.samples:
                particles = sample["state"]
                particles_ = sample["state_"]
                plate_p, plate_p_, angle = sample["action"]

                grid = torch.zeros((x_dim, y_dim))
                grid_ = torch.zeros((x_dim, y_dim))
                a_grid = torch.zeros((x_dim, y_dim))

                
                # draw each particles in the grids
                for p, p_ in zip(particles, particles_):

                    x_pos, y_pos, _, len = [x * TO_PXL for x in p]
                    x_pos_, y_pos_, _, _ = [x * TO_PXL for x in p_]

                    # draw particle at state(i)
                    len = len * TO_PXL
                    x_grid = c_x + x_pos
                    y_grid = c_y + y_pos
                    
                    grid[
                        int(torch.floor(x_grid - len/2)):int(torch.ceil(x_grid + len/2)),
                        int(torch.floor(y_grid - len/2)):int(torch.ceil(y_grid + len/2)),
                       ] = 1

                    # draw particle at state(i+1)
                    x_grid_ = c_x + x_pos_
                    y_grid_ = c_y + y_pos_
                    grid_[
                        int(torch.floor(x_grid_ - len/2)):int(torch.ceil(x_grid_ + len/2)),
                        int(torch.floor(y_grid_ - len/2)):int(torch.ceil(y_grid_ + len/2)),
                       ] = 1
                
                # draw plate in a grid
                pg_dim = max(plate_dim_x, plate_dim_y)
                plate_x, plate_y, _ = [x * TO_PXL for x in plate_p]
                plate_x_, plate_y_, _ = [x * TO_PXL for x in plate_p_]

                place_grid = torch.zeros((pg_dim, pg_dim))
                place_grid[
                    math.floor(pg_dim/2-plate_dim_x/2):math.ceil(pg_dim/2+plate_dim_x/2),
                    math.floor(pg_dim/2-plate_dim_y/2):math.ceil(pg_dim/2+plate_dim_y/2),
                ] = 1
                rotated = rotate(place_grid, angle=torch.rad2deg(angle), reshape=False, order=1)
                
                a_grid[
                    int(c_x + plate_x - pg_dim/2):int(c_x + plate_x + pg_dim/2),
                    int(c_y + plate_y - pg_dim/2):int(c_y + plate_y + pg_dim/2),
                ] = torch.from_numpy((rotated > 0.5).astype(int)*0.5)
                
                a_grid[
                    int(c_x + plate_x_ - pg_dim/2):int(c_x + plate_x_ + pg_dim/2),
                    int(c_y + plate_y_ - pg_dim/2):int(c_y + plate_y_ + pg_dim/2),
                ] = torch.from_numpy((rotated > 0.5).astype(int))


                from matplotlib import pyplot as plt
                plt.imshow(a_grid, interpolation='nearest')
                plt.show()
                return









def main():
    dataset = PileData("data/cubes/chickpeas_on_glass/", run=0)

if __name__ == "__main__":
    main()
        
        
