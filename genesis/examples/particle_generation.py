
import random
import numpy as np
from collections import defaultdict

# Random Sequential Addition (RSA) of spheres with a linked-cell (grid) acceleration

import matplotlib.pyplot as plt

def plot_particles_3d(positions, radii, box_size, resolution=16):
    """
    Visualize particles as 3D spheres.

    Parameters:
        positions (N,3): particle centers
        radii (N,): particle radii
        box_size (tuple): (Lx, Ly, Lz)
        resolution (int): sphere mesh resolution
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create sphere mesh
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    for pos, r in zip(positions, radii):
        x = r * np.outer(np.cos(u), np.sin(v)) + pos[0]
        y = r * np.outer(np.sin(u), np.sin(v)) + pos[1]
        z = r * np.outer(np.ones_like(u), np.cos(v)) + pos[2]

        ax.plot_surface(x, y, z, linewidth=0, alpha=0.6)

    # Set box limits
    Lx, Ly, Lz = box_size
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    ax.set_zlim([0, Lz])

    # Equal aspect ratio
    max_range = max(Lx, Ly, Lz)
    ax.set_box_aspect((Lx/max_range, Ly/max_range, Lz/max_range))

    ax.set_title("3D Particle Packing")
    plt.tight_layout()
    plt.show()



def generate_particles_variable_radii(
    num_particles,
    radii,
    box_size,
    max_attempts=200000
):
    """
    Generate non-overlapping spheres with variable radii using spatial partitioning.

    Parameters:
        num_particles (int)
        radii (array-like): list or array of radii (length >= num_particles)
        box_size (tuple): (Lx, Ly, Lz)
        max_attempts (int)

    Returns:
        positions (N, 3), radii (N,)
    """
    
    Lx, Ly, Lz = box_size
    radii = np.asarray(radii)

    if len(radii) < num_particles:
        raise ValueError("radii array must have at least num_particles elements")

    # Use max radius to define grid cell size
    max_r = np.max(radii)
    cell_size = 2 * max_r

    def grid_index(pos):
        return tuple((pos // cell_size).astype(int))

    # Spatial grid: maps cell -> list of particle indices
    grid = defaultdict(list)

    positions = []
    accepted_radii = []
    
    attempts = 0
    for i in range(num_particles):
        r = radii[i]
        placed = False

        while not placed and attempts < max_attempts:
            attempts += 1

            candidate = np.array([
                np.random.uniform(r, Lx - r),
                np.random.uniform(r, Ly - r),
                # r/2
                np.random.uniform(r, Lz - r)
            ])

            cell = grid_index(candidate)            
            def recursive_grid_check(neighbor_cell : list, depth : int = 0):
                if depth == 3:
                    for j in grid.get(tuple(neighbor_cell), []):
                        dist = np.linalg.norm(candidate - positions[j])
                        if dist < (r + accepted_radii[j]):
                            return True
                    return False      
                 
                for d in [-1, 0, 1]:
                    neighbor_cell.append(cell[depth] + d)
                    if recursive_grid_check(neighbor_cell, depth+1):
                        return True
                    neighbor_cell.pop()
                return False
                

            overlap = recursive_grid_check(neighbor_cell=[])
            
            # Check neighboring cells (3x3x3 neighborhood)
            # overlap = False
            # for dx in neighbor_idx:
            #     neighbor_x = cell[0] + dx
            #     for dy in neighbor_idx:
            #         neighbor_y = cell[1] + dy
                    
            #         for dz in neighbor_idx:
            #             neighbor_z = cell[2] + dz
            #             neighbor_cell = (neighbor_x, neighbor_y, neighbor_z)

            #             for j in grid.get(neighbor_cell, []):
            #                 dist = np.linalg.norm(candidate - positions[j])
            #                 if dist < (r + accepted_radii[j]):
            #                     overlap = True
            #                     break
            #             if overlap:
            #                 break
            #         if overlap:
            #             break
            #     if overlap:
            #         break

            if not overlap:
                idx = len(positions)
                positions.append(candidate)
                accepted_radii.append(r)
                grid[cell].append(idx)
                placed = True

        if attempts >= max_attempts:
            print(f"Stopped early at particle {i}")
            break

    print(f"Generated {len(positions)} particles out of {num_particles}")

    return np.array(positions), np.array(accepted_radii)




if __name__ == "__main__":
    # Example usage
    
    num_particles = 100

    # Example: random radii between 0.05 and 0.15
    radii = np.random.uniform(1., 1.1, size=num_particles)

    box_size = (1.0, 1.0, 1.0)

    positions, radii = generate_particles_variable_radii(
        num_particles,
        radii,
        box_size
    )

    print("Positions shape:", positions.shape)
    print("Radii shape:", radii.shape)
    plot_particles_3d(positions, radii, box_size)
