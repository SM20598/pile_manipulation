from typing import Tuple
import math
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


# Fraction of a layer's placement capacity above which the tray is too full to
# act in, rather than too full to fill. Corroborated by the placement-aware
# sampler's own measurements: the tool's free configuration space was fully
# available (100 % of samples) at every count up to 150 of 5 mm cubes and
# completely EMPTY (0/5 samples) at 200 - which are 67 % and 89 % of the 225
# ceiling. The collapse sits between them, so warn from 70 %.
PACKING_WARN_FRACTION = 0.70


def single_layer_capacity(shape: str, particle_size, num_particles: int,
                          box_xy: float, min_gap: float = 1e-3) -> int:
    """Most particles of this size that fit in ONE layer of a ``box_xy`` tray.

    Not a packing estimate - the exact ceiling the simulator enforces. Two
    things set it, and both are easy to get wrong:

    **The reshuffle binds, not creation.** Particles are placed twice with
    different clearances. ``random_sequential_addition`` (creation, pre-build)
    clears ``size/2``. ``SandboxManipulation.
    _sample_nonoverlapping_particle_positions`` (every reset) clears
    ``size/2 * sqrt(2)``, the footprint a cube sweeps at free yaw - 1.41x per
    axis, so about twice the area per particle. A count that creates fine can
    still fail on the first reset, and the reset runs every batch.

    **The grid fallback sets the ceiling, not rejection sampling.** Rejection
    sampling gives up well before the tray is full - 132 of a possible 225 at
    5 mm - and the reshuffle then falls through to
    ``_grid_particle_positions`` silently, so the effective capacity is the
    regular grid's.

    Verified against the simulator at the boundary: 8.5 mm cubes shuffle at
    n=81 and fail at n=82; 5 mm shuffles at n=225.

    Note what this is NOT. It is the ceiling on what the *placer can generate* -
    an axis-aligned grid with a 1 mm gap - and is far below what a *settled*
    layer can hold once gravity compacts it. Hexagonal close packing puts ~262
    spheres of 8.5 mm in this tray against a placement capacity of 169, so a
    count this function calls impossible can still settle flat. Use it to
    decide whether `shuffle_particles` will succeed and whether the tool has
    room to act (``PACKING_WARN_FRACTION``); do **not** read it as the point
    above which a pile must stack. Whether a pile stacks is a jamming property
    of the particle shape, not an occupancy threshold.
    """
    sizes = _resolve_particle_sizes(particle_size, num_particles)
    half, _ = _particle_dimensions(shape, sizes)
    ch = float(np.max(half[:, :2]))
    if shape in ("cube", "box", "rectangle", "rectangular_cube"):
        ch *= math.sqrt(2)                    # free-yaw footprint
    span = box_xy - 2 * ch
    if span < 0:
        return 0
    # _grid_particle_positions searches n_x with n_y = ceil(n / n_x) and needs
    # both spacings >= 2*ch + min_gap; the largest such product is the square
    # grid that just fits.
    per_row = int(span / (2 * ch + min_gap)) + 1
    return per_row * per_row


def capacity_table(shape: str, sizes, box_xy: float,
                   min_gap: float = 1e-3) -> dict:
    """``{particle_size: single-layer capacity}`` for a sweep of sizes.

    The lookup the collection scripts and the docs both quote, so the numbers
    cannot drift apart. In the stock 128 mm tray, cubes:
    225 at 5.00 mm, 144 at 6.75 mm, 81 at 8.50 mm, 64 at 10.25 mm, 49 at
    12.00 mm.
    """
    return {float(s): single_layer_capacity(shape, s, 1, box_xy, min_gap)
            for s in sizes}


def check_packing_fraction(shape: str, particle_size, num_particles: int,
                           box_xy: float, n_layers: int = 1,
                           log=print) -> float:
    """Warn if a layer is filled past ``PACKING_WARN_FRACTION``. Returns the fraction.

    A tray filled to its placement ceiling is not a hard pile, it is a
    *stuck* one: the action space collapses before the placement does. The
    tool needs somewhere to descend that is not already occupied, and the
    fraction of the tray it can reach falls away much faster than the
    remaining free area suggests, because the blade needs a 40 x 2 mm footprint
    clear at some yaw, not a single free cell.
    """
    cap = single_layer_capacity(shape, particle_size, num_particles, box_xy)
    if cap <= 0:
        return float("inf")
    n_layers = max(int(n_layers), 1)
    per_layer = -(-int(num_particles) // n_layers)
    frac = per_layer / cap

    # A stack is a temporary arrangement. Pushes spread the pile, so over a
    # trajectory it tends toward a single layer - and at that point the count
    # that mattered is the TOTAL, not the per-layer figure. Planning a stack to
    # sit under the threshold does not make the total fit, so this is checked
    # separately and is often the binding constraint at high object counts.
    if n_layers > 1:
        flat = int(num_particles) / cap
        if flat >= PACKING_WARN_FRACTION:
            log(
                f"WARNING: {int(num_particles)} particles is {flat*100:.0f}% of "
                f"ONE layer's placement capacity ({cap}). The {n_layers}-layer "
                f"spawn keeps each layer at {frac*100:.0f}%, but a pile spreads "
                f"as it is pushed and tends toward a monolayer over a "
                f"trajectory - and flat, this count leaves the tool nowhere to "
                f"touch down. Expect the action space to degrade as an episode "
                f"progresses. To reach this many objects usably, the pile has "
                f"to occupy less area per particle: smaller particles, a larger "
                f"tray, or a shape that packs tighter than a free-yaw cube "
                f"(a sphere of the same size needs no sqrt(2) inflation and so "
                f"gives ~2x the capacity)."
            )
    if frac >= PACKING_WARN_FRACTION:
        what = (f"{per_layer} particles per layer "
                f"({int(num_particles)} over {n_layers} layers)"
                if n_layers > 1 else f"{per_layer} particles")
        log(
            f"WARNING: {what} is {frac*100:.0f}% of a layer's "
            f"placement capacity ({cap}). Past ~{PACKING_WARN_FRACTION*100:.0f}% "
            f"the tray is too full to ACT in, not just too full to fill: the "
            f"tool has almost nowhere to touch down that is not already "
            f"occupied, so most sampled actions either land on a particle or "
            f"barely move the pile. Measured at 5 mm cubes, the "
            f"placement-aware sampler's free set was fully available at 150 "
            f"(67%) and completely empty at 200 (89%). Reduce n_particles, use "
            f"smaller particles, or use a larger tray."
        )
    return frac


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
