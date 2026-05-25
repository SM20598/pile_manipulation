from typing import Tuple
import genesis as gs
from genesis import Scene
import numpy as np

def random_sequential_addition(
    scene : Scene,
    granular_vol : Tuple[float, float, float],
    shape : str,
    num_particles : int,
    particle_size : float | list[float],
    wall_thickness : float,
):
    """Create randomly positioned and oriented particles without initial overlap."""
    width, depth, height = granular_vol

    fix_size = isinstance(particle_size, (int, float))
    if not fix_size:
        if len(particle_size) == 2 and num_particles != 2:
            particle_size = np.linspace(float(particle_size[0]), float(particle_size[1]), num_particles)
        elif len(particle_size) == num_particles:
            particle_size = [float(size) for size in particle_size]
        else:
            raise ValueError(
                "particle_size must be a scalar, a [min, max] range, or a list "
                "with the same length as num_particles."
            )

    max_particle_size = float(particle_size) if fix_size else max(particle_size)

    def get_radius(i):
        return get_length(i) / 2.0

    def get_length(i):
        return float(particle_size) if fix_size else particle_size[i]

    def get_particle_dimensions(i):
        if shape in ("cube", "box"):
            r = get_length(i)
            return (r, r, r), r
        if shape == "sphere":
            r = get_radius(i)
            return (2 * r, 2 * r, 2 * r), r
        if shape in ("rectangle", "rectangular_cube"):
            length = get_length(i)
            side = 0.5 * max_particle_size
            dimensions = (length, side, side)
            return dimensions, 0.5 * np.linalg.norm(dimensions)
        if shape == "cylinder":
            length = get_length(i)
            radius = 0.5 * max_particle_size
            dimensions = (2 * radius, 2 * radius, length)
            return dimensions, np.sqrt(radius**2 + (length / 2) ** 2)
        raise ValueError(f"Unsupported shape {shape}. Supported shapes are 'cube', 'sphere', 'rectangle', and 'cylinder'.")

    particle_dimensions = [get_particle_dimensions(i)[0] for i in range(num_particles)]
    half_extents = np.asarray(particle_dimensions, dtype=float) * 0.5
    placement_half_extents = half_extents.copy()
    if shape == "cylinder":
        placement_half_extents[:] = np.max(half_extents, axis=1, keepdims=True)
    floor_z = wall_thickness / 2.0

    lower_xy = np.array([-width / 2, -depth / 2], dtype=float) + placement_half_extents[:, :2]
    upper_xy = np.array([width / 2, depth / 2], dtype=float) - placement_half_extents[:, :2]
    if np.any(upper_xy < lower_xy):
        raise ValueError("At least one particle is too large to fit inside the granular volume.")

    positions = np.empty((num_particles, 3), dtype=float)
    order = np.argsort(-np.prod(half_extents[:, :2], axis=1))
    placed = []
    min_gap = 1e-4

    for particle_idx in order:
        placed_particle = False
        for _ in range(20000):
            xy = np.random.uniform(lower_xy[particle_idx], upper_xy[particle_idx])
            if placed:
                placed_idx = np.asarray(placed, dtype=int)
                delta = np.abs(xy - positions[placed_idx, :2])
                min_sep = placement_half_extents[particle_idx, :2] + placement_half_extents[placed_idx, :2] + min_gap
                if not np.all(np.any(delta >= min_sep, axis=1)):
                    continue
            positions[particle_idx] = (
                xy[0],
                xy[1],
                floor_z + placement_half_extents[particle_idx, 2] + min_gap,
            )
            placed.append(particle_idx)
            placed_particle = True
            break

        if not placed_particle:
            raise RuntimeError(
                "Could not randomly place all particles without overlap. "
                "Try a smaller particle size or fewer particles."
            )

    def random_euler():
        if shape == "sphere":
            return None
        if shape == "cylinder":
            yaw = float(np.random.uniform(0.0, 360.0))
            if np.random.random() < 0.5:
                return (0.0, 0.0, yaw)
            return (90.0, 0.0, yaw)
        return (0.0, 0.0, float(np.random.uniform(0.0, 360.0)))

    entities = []
    particle_sizes = []
    for i in range(num_particles):
        dimensions, r = get_particle_dimensions(i)
        size_x, size_y, size_z = dimensions
        pos = tuple(float(value) for value in positions[i])
        euler = random_euler()

        if shape in ("cube", "box", "rectangle", "rectangular_cube"):
            morph = gs.morphs.Box(pos=pos, size=(size_x, size_y, size_z), euler=euler)
        elif shape == "sphere":
            morph = gs.morphs.Sphere(pos=pos, radius=r)
        elif shape == "cylinder":
            morph = gs.morphs.Cylinder(pos=pos, radius=size_x / 2, height=size_z, euler=euler)

        entity = scene.add_entity(
            morph=morph,
            surface=gs.surfaces.Default(color=[1.0, 1.0, 0.0]),
        )
        entities.append(entity)
        particle_sizes.append(tuple(float(value) for value in dimensions))

    print(f"Generated {len(positions)} particles out of {num_particles}")
    return entities, particle_sizes
