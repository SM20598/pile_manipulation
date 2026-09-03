"""
LAYERED VARIANT of Genesis/utilities/materials.py.

A deliberate copy, not a subclass: `random_sequential_addition` and
`max_particle_height` are module-level functions, so there is nothing to
override. Everything here is byte-identical to the original except the three
changes listed below, which are kept textually minimal so the two files stay
diffable when upstream moves:

  1. `stack_height()` is NEW - the box height a stack of `n_layers` needs.
  2. `random_sequential_addition()` takes `n_layers` and packs each layer
     independently, since overlap is only a constraint *within* a layer.
  3. Its box-height precondition accounts for the stack rather than one layer.

See Genesis/layered/README.md for why this path exists and when to use it.
"""

from typing import Tuple
import math
import sys
from pathlib import Path

# Genesis/ holds the shared capacity model; this module lives one level down.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import genesis as gs
from genesis import Scene
import numpy as np

from utilities.materials import (  # noqa: E402  shared with the monolayer path
    PACKING_WARN_FRACTION, capacity_table, check_packing_fraction,
    single_layer_capacity,
)


# Vertical clearance left between stacked layers. Large enough that a
# particle cannot start interpenetrating the layer below it, small enough
# that the drop is short and the settle that follows is cheap.
LAYER_GAP = 1e-3


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


def stack_height(shape: str, particle_size: float | list[float], num_particles: int,
                 n_layers: int = 1) -> float:
    """Box interior height a stack of ``n_layers`` needs, including the gaps.

    The single-layer original sizes the box to ``wall_thickness +
    max_particle_height(...)`` so a resting monolayer never pokes above the
    walls. A stack needs the same allowance per layer plus the inter-layer gap
    the placer leaves, and the box must be tall enough BEFORE `scene.build()`
    because the walls are geometry.
    """
    h = max_particle_height(shape, particle_size, num_particles)
    # The trailing LAYER_GAP is the floor clearance the placer leaves under the
    # FIRST layer (`inner_min[2] + min_gap`), which is easy to forget and
    # compounds nothing but is exactly what makes the fit off by one gap. The
    # single-layer original gets away without it because its fit is exact to
    # within fit_eps and a particle that pokes 1 mm above the wall at spawn
    # just settles back down; with several layers that slop has to be budgeted.
    return n_layers * (h + LAYER_GAP) + LAYER_GAP


def random_sequential_addition(
    scene : Scene,
    granular_vol : Tuple[float, float, float],
    shape : str,
    num_particles : int,
    particle_size : float | list[float],
    wall_thickness : float,
    box_height : float,
    n_layers : int = 1,
):
    """Create randomly positioned and oriented particles without initial overlap.

    ``n_layers`` spreads the particles over that many stacked layers. Overlap is
    only a constraint *within* a layer, because layers are vertically separated
    by ``LAYER_GAP``, so each is packed independently against its own members
    only - which is what makes counts above the single-layer capacity reachable.
    Layers are dropped, not interpenetrating: the caller's settle collapses them
    into a natural pile.
    """
    width, depth, _ = granular_vol

    sizes = _resolve_particle_sizes(particle_size, num_particles)
    half_extents, placement_half_extents = _particle_dimensions(shape, sizes)
    floor_z = wall_thickness / 2.0

    lower_xy = np.array([-width / 2, -depth / 2], dtype=float) + placement_half_extents[:, :2]
    upper_xy = np.array([width / 2, depth / 2], dtype=float) - placement_half_extents[:, :2]
    if np.any(upper_xy < lower_xy):
        raise ValueError("At least one particle is too large to fit inside the granular volume.")

    layer_pitch = float(np.max(placement_half_extents[:, 2])) * 2 + LAYER_GAP
    required_height = wall_thickness + n_layers * layer_pitch
    fit_eps = 1e-6  # tolerance for float32 rounding when box height is an exact fit
    if box_height < required_height - fit_eps:
        raise ValueError(
            f"Box height ({box_height:.4f}) is too small for {n_layers} layer(s) of "
            f"these particles: it must be at least wall_thickness + "
            f"n_layers * (particle height + gap) ({wall_thickness:.4f} + "
            f"{n_layers} * {layer_pitch:.4f} = {required_height:.4f}), "
            "otherwise particles stick out of the box in z."
        )

    positions = np.empty((num_particles, 3), dtype=float)
    order = np.argsort(-np.prod(half_extents[:, :2], axis=1))
    min_gap = 1e-4

    # Strided split, so each layer gets a comparable mix of particle sizes
    # rather than the largest all landing in layer 0.
    for layer_idx in range(n_layers):
        layer_order = order[layer_idx::n_layers]
        placed = []
        layer_z = floor_z + layer_idx * layer_pitch
        for particle_idx in layer_order:
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
                    layer_z + placement_half_extents[particle_idx, 2] + min_gap,
                )
                placed.append(particle_idx)
                placed_particle = True
                break

            if not placed_particle:
                raise RuntimeError(
                    f"Could not place all particles without overlap in {n_layers} "
                    f"layer(s) (failed in layer {layer_idx}). Try more layers, a "
                    f"smaller particle size, or fewer particles."
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


# --------------------------------------------------------------------------- #
# NEW in the layered variant: choosing how many layers to use.
#
# Kept at the end of the file, separate from the copied body above, so the
# copied region stays diffable against Genesis/utilities/materials.py.
# --------------------------------------------------------------------------- #
# NEW in the layered variant: choosing how many layers to use.
#
# Kept at the end of the file, separate from the copied body above, so the
# copied region stays diffable against Genesis/utilities/materials.py.
# --------------------------------------------------------------------------- #
# NEW in the layered variant: choosing how many layers to use.
#
# The single-layer capacity model itself is NOT duplicated here - it is shared
# with the monolayer path in utilities/materials.py, because it describes
# behaviour both paths have and a second copy of it would be a second thing to
# get wrong. Only the layer arithmetic is local.
# --------------------------------------------------------------------------- #

def plan_layers(shape: str, particle_size, num_particles: int, granular_vol,
                max_layers: int = 8, box_xy: float | None = None,
                target_fraction: float = PACKING_WARN_FRACTION) -> int:
    """Fewest layers that leave the tool room to act, not merely room to fit.

    ``target_fraction`` is the per-layer occupancy to plan for. It defaults to
    ``PACKING_WARN_FRACTION`` (0.70) rather than 1.0 deliberately: a layer
    filled to its placement ceiling is not a denser pile, it is a stuck one -
    the blade needs a 40 x 2 mm footprint clear at some yaw to touch down, and
    that vanishes well before the last particle stops fitting. Pass 1.0 to get
    the old "fewest that fit" behaviour.

    Worked example, 150 cubes of 8.5 mm in the stock tray (capacity 81):
    fewest-that-fit gives 2 layers at 93 % occupancy, which still warns;
    planning for 70 % gives 3 layers at 62 %, which does not.

    Analytic rather than a trial pack, because the binding capacity is the grid
    fallback's and that is a closed form - see ``single_layer_capacity``.
    """
    width = granular_vol[0] if box_xy is None else box_xy
    cap = single_layer_capacity(shape, particle_size, num_particles, width)
    if cap <= 0:
        raise ValueError(
            f"A particle of size {particle_size} does not fit in a "
            f"{width:.3f} m tray at all.")
    usable = max(1, int(cap * target_fraction))
    layers = -(-num_particles // usable)          # ceil
    if layers > max_layers:
        raise ValueError(
            f"Cannot place {num_particles} particles of size {particle_size} in "
            f"{max_layers} layers of a {width:.3f} m tray at "
            f"{target_fraction*100:.0f}% occupancy: one layer holds {cap} "
            f"({usable} at that occupancy), so {layers} layers would be needed. "
            f"Use smaller particles, a larger tray, or fewer particles."
        )
    return layers
