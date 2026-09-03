"""Tests for Genesis/layered/materials_layered.py — the stacked-spawn variant.

Genesis-free: only the planning and geometry logic is covered here, which is
also the only genuinely new logic in the layered path. The placement itself
needs a built scene, so it is verified live (see Genesis/layered/README.md).

The most important test in this file is
`test_one_layer_matches_the_single_layer_original`: the layered module is a
*copy*, so the thing most likely to go wrong is the copy quietly diverging from
upstream in the n_layers=1 case that everybody actually uses.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Genesis"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Genesis" / "layered"))

import materials_layered as ML  # noqa: E402
from utilities import materials as MO  # noqa: E402

TRAY = (0.127, 0.127, 0.05)
WALL = 0.02


# --------------------------------------------------------------------------- #
# the copy has not diverged
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("size", [0.005, 0.0085, 0.012])
@pytest.mark.parametrize("n", [10, 50, 200])
def test_one_layer_matches_the_single_layer_original(size, n):
    """max_particle_height is untouched by the copy, so it must agree exactly."""
    assert ML.max_particle_height("cube", size, n) == MO.max_particle_height("cube", size, n)


def test_shared_helpers_are_untouched():
    sizes_a = ML._resolve_particle_sizes([0.005, 0.012], 7)
    sizes_b = MO._resolve_particle_sizes([0.005, 0.012], 7)
    np.testing.assert_allclose(sizes_a, sizes_b)
    for shape in ("cube", "sphere", "cylinder", "rectangle"):
        a = ML._particle_dimensions(shape, sizes_a)
        b = MO._particle_dimensions(shape, sizes_b)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])


# --------------------------------------------------------------------------- #
# stack_height — the box must be tall enough BEFORE the scene is built
# --------------------------------------------------------------------------- #

def test_stack_height_grows_with_layers():
    h = [ML.stack_height("cube", 0.005, 100, L) for L in (1, 2, 3, 4)]
    assert h == sorted(h)
    # each extra layer costs exactly one particle height plus one gap
    steps = {round(b - a, 9) for a, b in zip(h, h[1:])}
    assert steps == {round(0.005 + ML.LAYER_GAP, 9)}


@pytest.mark.parametrize("size", [0.005, 0.0085, 0.012])
@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_stack_height_budgets_what_the_placer_actually_needs(size, n_layers):
    """The regression this guards against was found by the placer's own check.

    The placer leaves a floor gap under the first layer and then one gap per
    layer. stack_height originally budgeted only the per-layer gaps, so the fit
    was short by exactly one LAYER_GAP — invisible at one layer (where upstream
    is an exact fit anyway and a particle just settles back down) and a hard
    failure at two.
    """
    interior = ML.stack_height("cube", size, 100, n_layers)
    # what _sample_nonoverlapping_particle_positions computes as stack_top,
    # measured from inner_min[2]
    placer_needs = ML.LAYER_GAP + n_layers * (size + ML.LAYER_GAP)
    assert placer_needs <= interior + 1e-9


def test_stack_height_at_one_layer_still_clears_a_single_particle():
    assert ML.stack_height("cube", 0.005, 50, 1) >= MO.max_particle_height("cube", 0.005, 50)


# --------------------------------------------------------------------------- #
# single_layer_capacity — the number that decides whether layering is needed
#
# These two values are pinned against the SIMULATOR, not against the formula:
# 8.5 mm cubes shuffle at n=81 and fail at n=82, and 5 mm shuffles at n=225.
# If a change to the placer moves the real ceiling, these must move with it.
# --------------------------------------------------------------------------- #

BOX_XY = 0.128     # box.vol width, which is what the reshuffle bounds use


@pytest.mark.parametrize("size,expected", [(0.005, 225), (0.0085, 81)])
def test_capacity_matches_the_simulator_at_the_boundary(size, expected):
    assert ML.single_layer_capacity("cube", size, 100, BOX_XY) == expected


def test_capacity_uses_the_free_yaw_footprint_not_the_bare_size():
    """The reshuffle clears size/2*sqrt(2); creation only clears size/2.

    The reshuffle is the binding path because it runs on every reset, so the
    capacity must be computed from the larger clearance. Getting this wrong
    over-estimated the ceiling by ~2.4x.
    """
    cube = ML.single_layer_capacity("cube", 0.0085, 100, BOX_XY)
    sphere = ML.single_layer_capacity("sphere", 0.0085, 100, BOX_XY)
    # a sphere of the same size gets no sqrt(2) inflation, so it packs more
    assert sphere > cube


def test_capacity_falls_with_particle_size():
    caps = [ML.single_layer_capacity("cube", s, 100, BOX_XY)
            for s in (0.005, 0.00675, 0.0085, 0.01025, 0.012)]
    assert caps == sorted(caps, reverse=True)


def test_capacity_grows_with_tray_size():
    small = ML.single_layer_capacity("cube", 0.0085, 100, 0.128)
    big = ML.single_layer_capacity("cube", 0.0085, 100, 0.256)
    assert big > small


def test_capacity_is_zero_for_a_particle_that_cannot_fit():
    assert ML.single_layer_capacity("cube", 0.5, 1, BOX_XY) == 0


# --------------------------------------------------------------------------- #
# plan_layers
# --------------------------------------------------------------------------- #

def test_plan_layers_targets_room_to_act_not_just_fit():
    """The default plans for 70 % occupancy, not 100 %.

    150 cubes of 8.5 mm fit in 2 layers (93 % each, which warns) but need 3 to
    sit under the threshold (62 %). Planning for the ceiling would hand back a
    configuration the packing check immediately complains about.
    """
    fits = ML.plan_layers("cube", 0.0085, 150, TRAY, box_xy=BOX_XY,
                          target_fraction=1.0)
    usable = ML.plan_layers("cube", 0.0085, 150, TRAY, box_xy=BOX_XY)
    assert fits == 2 and usable == 3


def test_plan_layers_result_is_under_the_warning_threshold():
    """Whatever it returns must not itself trip check_packing_fraction."""
    for size in (0.005, 0.00675, 0.0085, 0.01025, 0.012):
        for n in (50, 70, 100, 150, 200):
            L = ML.plan_layers("cube", size, n, TRAY, box_xy=BOX_XY)
            cap = ML.single_layer_capacity("cube", size, 1, BOX_XY)
            assert (-(-n // L)) / cap < ML.PACKING_WARN_FRACTION


def test_plan_layers_is_one_while_a_layer_stays_under_threshold():
    cap = ML.single_layer_capacity("cube", 0.005, 1, BOX_XY)   # 225
    under = int(ML.PACKING_WARN_FRACTION * cap) - 1            # 156
    assert ML.plan_layers("cube", 0.005, under, TRAY, box_xy=BOX_XY) == 1


def test_plan_layers_target_fraction_of_one_recovers_fewest_that_fit():
    cap = ML.single_layer_capacity("cube", 0.005, 1, BOX_XY)
    assert ML.plan_layers("cube", 0.005, cap, TRAY, box_xy=BOX_XY,
                          target_fraction=1.0) == 1
    assert ML.plan_layers("cube", 0.005, cap + 1, TRAY, box_xy=BOX_XY,
                          target_fraction=1.0) == 2


def test_plan_layers_is_monotone_in_count():
    seen = [ML.plan_layers("cube", 0.0085, n, TRAY, box_xy=BOX_XY)
            for n in (50, 100, 200, 400)]
    assert seen == sorted(seen)


def test_plan_layers_is_monotone_in_particle_size():
    seen = [ML.plan_layers("cube", s, 200, TRAY, box_xy=BOX_XY)
            for s in (0.005, 0.00675, 0.0085, 0.01025, 0.012)]
    assert seen == sorted(seen)


def test_plan_layers_raises_rather_than_returning_something_infeasible():
    with pytest.raises(ValueError, match="Cannot place"):
        ML.plan_layers("cube", 0.012, 5000, TRAY, max_layers=3, box_xy=BOX_XY)


def test_plan_layers_defaults_box_xy_to_the_granular_volume():
    a = ML.plan_layers("cube", 0.0085, 200, (0.128, 0.128, 0.05))
    b = ML.plan_layers("cube", 0.0085, 200, (0.128, 0.128, 0.05), box_xy=0.128)
    assert a == b
