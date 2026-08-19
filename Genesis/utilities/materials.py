from typing import Tuple
import genesis as gs
from genesis import Scene
import numpy as np


def _resolve_particle_sizes(particle_size: float | list[float], num_particles: int):
    """Expands a scalar, a [min, max] range, or a per-particle list into one size per particle."""
    if isinstance(particle_size, (int, float)):
        return np.full(num_particles, float(particle_size), dtype=float)
    if len(particle_size) == 2 and num_particles != 2:
        return np.linspace(float(particle_size[0]), float(particle_size[1]), num_particles)
    if len(particle_size) == num_particles:
        return np.asarray([float(size) for size in particle_size], dtype=float)
    raise ValueError(
        "particle_size must be a scalar, a [min, max] range, or a list "
        "with the same length as num_particles."
    )


def _particle_dimensions(shape: str, sizes: np.ndarray):
    """
    Returns (half_extents, placement_half_extents), each shape (n_particles, 3).

    placement_half_extents differs from half_extents only for cylinders, where it's
    inflated to the largest half-extent on every axis to conservatively account for
    particles that may be lying on their side (see random_euler in random_sequential_addition).
    """
    max_size = float(np.max(sizes))

    def dims(i):
        size = float(sizes[i])
        if shape in ("cube", "box"):
            return (size, size, size)
        if shape == "sphere":
            return (size, size, size)
        if shape in ("rectangle", "rectangular_cube"):
            side = 0.5 * max_size
            return (size, side, side)
        if shape == "cylinder":
            radius = 0.5 * max_size
            return (2 * radius, 2 * radius, size)
        raise ValueError(f"Unsupported shape {shape}. Supported shapes are 'cube', 'sphere', 'rectangle', and 'cylinder'.")

    dimensions = np.asarray([dims(i) for i in range(len(sizes))], dtype=float)
    half_extents = dimensions * 0.5
    placement_half_extents = half_extents.copy()
    if shape == "cylinder":
        placement_half_extents[:] = np.max(half_extents, axis=1, keepdims=True)
    return half_extents, placement_half_extents


def max_particle_height(shape: str, particle_size: float | list[float], num_particles: int) -> float:
    """
    Full z-extent (not half) the tallest particle could occupy, accounting for
    cylinders potentially lying on their side. Used to size the box height so a
    resting monolayer never sticks out above the walls.
    """
    sizes = _resolve_particle_sizes(particle_size, num_particles)
    _, placement_half_extents = _particle_dimensions(shape, sizes)
    return float(np.max(placement_half_extents[:, 2])) * 2


def random_sequential_addition(
    scene : Scene,
    granular_vol : Tuple[float, float, float],
    shape : str,
    num_particles : int,
    particle_size : float | list[float],
    wall_thickness : float,
    box_height : float,
):
    """Create randomly positioned and oriented particles without initial overlap."""
    width, depth, _ = granular_vol

    sizes = _resolve_particle_sizes(particle_size, num_particles)
    half_extents, placement_half_extents = _particle_dimensions(shape, sizes)
    floor_z = wall_thickness / 2.0

    lower_xy = np.array([-width / 2, -depth / 2], dtype=float) + placement_half_extents[:, :2]
    upper_xy = np.array([width / 2, depth / 2], dtype=float) - placement_half_extents[:, :2]
    if np.any(upper_xy < lower_xy):
        raise ValueError("At least one particle is too large to fit inside the granular volume.")

    required_height = wall_thickness + float(np.max(placement_half_extents[:, 2])) * 2
    fit_eps = 1e-6  # tolerance for float32 rounding when box height is an exact fit
    if box_height < required_height - fit_eps:
        raise ValueError(
            f"Box height ({box_height:.4f}) is too small for these particles: it must be "
            f"at least wall_thickness + particle height ({wall_thickness:.4f} + "
            f"{required_height - wall_thickness:.4f} = {required_height:.4f}), "
            "otherwise particles stick out of the box in z."
        )

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
        size_x, size_y, size_z = half_extents[i] * 2
        r = float(np.max(half_extents[i]))
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
        particle_sizes.append(tuple(float(value) for value in (size_x, size_y, size_z)))

    return entities, particle_sizes
