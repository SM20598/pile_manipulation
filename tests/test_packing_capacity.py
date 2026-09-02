"""Tests for the single-layer capacity model in Genesis/utilities/materials.py.

Two distinct questions live here, and conflating them is the mistake this
module exists to prevent:

  *Will it fit?*  -> single_layer_capacity, the placement ceiling.
  *Can we act?*   -> check_packing_fraction, which is a much lower bar.

The capacity numbers are pinned against the SIMULATOR, not against the
formula: 8.5 mm cubes shuffle at n=81 and fail at n=82, and 5 mm shuffles at
n=225. If a change to the placer moves the real ceiling, these must move too.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Genesis"))

from utilities.materials import (  # noqa: E402
    PACKING_WARN_FRACTION, capacity_table, check_packing_fraction,
    single_layer_capacity,
)

BOX = 0.128            # box.vol width — what the reshuffle bounds against
SWEEP = (0.005, 0.00675, 0.0085, 0.01025, 0.012)


# --------------------------------------------------------------------------- #
# the ceiling, pinned to measured simulator behaviour
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("size,expected", [(0.005, 225), (0.0085, 81)])
def test_capacity_matches_the_simulator(size, expected):
    assert single_layer_capacity("cube", size, 1, BOX) == expected


def test_capacity_table_covers_the_historical_sweep():
    table = capacity_table("cube", SWEEP, BOX)
    assert table == {0.005: 225, 0.00675: 144, 0.0085: 81,
                     0.01025: 64, 0.012: 49}


def test_capacity_uses_the_free_yaw_footprint_for_cubes_only():
    """A cube needs size/2*sqrt(2) of clearance; a sphere needs size/2.

    This is the factor that made the first version of this model ~2.4x too
    generous, so it is asserted directly rather than left implicit.
    """
    assert single_layer_capacity("sphere", 0.0085, 1, BOX) > \
           single_layer_capacity("cube", 0.0085, 1, BOX)


def test_capacity_falls_monotonically_with_size():
    caps = [single_layer_capacity("cube", s, 1, BOX) for s in SWEEP]
    assert caps == sorted(caps, reverse=True)
    assert len(set(caps)) == len(caps), "each size should give a distinct ceiling"


def test_capacity_grows_with_tray_width():
    assert single_layer_capacity("cube", 0.0085, 1, 0.256) > \
           single_layer_capacity("cube", 0.0085, 1, 0.128)


def test_capacity_is_zero_when_the_particle_cannot_fit():
    assert single_layer_capacity("cube", 0.5, 1, BOX) == 0


# --------------------------------------------------------------------------- #
# the action-space threshold — a much lower bar than the ceiling
# --------------------------------------------------------------------------- #

def _warns(shape, size, n, layers=1):
    msgs = []
    check_packing_fraction(shape, size, n, BOX, n_layers=layers, log=msgs.append)
    return msgs


def test_threshold_brackets_the_measured_free_set_collapse():
    """The 70 % figure is not arbitrary.

    The placement-aware sampler's free set was measured fully available at 150
    of 5 mm cubes and completely empty at 200 — 67 % and 89 % of the 225
    ceiling. The threshold must sit between those, or it is not describing the
    thing it claims to.
    """
    cap = single_layer_capacity("cube", 0.005, 1, BOX)
    assert 150 / cap < PACKING_WARN_FRACTION <= 200 / cap


def test_warns_at_89_percent_and_stays_quiet_at_44():
    assert _warns("cube", 0.005, 200)          # 89 %
    assert not _warns("cube", 0.005, 100)      # 44 %


def test_warning_is_quiet_just_below_and_fires_just_above():
    cap = single_layer_capacity("cube", 0.005, 1, BOX)
    just_under = int(PACKING_WARN_FRACTION * cap) - 1
    just_over = int(PACKING_WARN_FRACTION * cap) + 1
    assert not _warns("cube", 0.005, just_under)
    assert _warns("cube", 0.005, just_over)


def test_returned_fraction_is_the_occupancy():
    cap = single_layer_capacity("cube", 0.005, 1, BOX)
    frac = check_packing_fraction("cube", 0.005, 150, BOX, log=lambda m: None)
    assert frac == pytest.approx(150 / cap)


def test_layers_are_counted_per_layer_not_in_total():
    """Stacking raises total capacity, so occupancy must be divided by layers."""
    one = check_packing_fraction("cube", 0.0085, 150, BOX, n_layers=1,
                                 log=lambda m: None)
    two = check_packing_fraction("cube", 0.0085, 150, BOX, n_layers=2,
                                 log=lambda m: None)
    assert two == pytest.approx(one / 2, rel=0.02)


def test_multi_layer_message_names_both_numbers():
    # order-independent: a multi-layer case emits both the per-layer warning
    # and the collapse warning, and which comes first is not a contract
    joined = " ".join(_warns("cube", 0.0085, 150, layers=2))
    assert "per layer" in joined and "150 over 2 layers" in joined


def test_warning_says_what_to_do_about_it():
    msg = " ".join(_warns("cube", 0.005, 200))
    for hint in ("Reduce n_particles", "smaller particles", "larger tray"):
        assert hint in msg


def test_an_impossible_particle_reports_infinite_occupancy_without_crashing():
    assert check_packing_fraction("cube", 0.5, 10, BOX,
                                  log=lambda m: None) == float("inf")


# --------------------------------------------------------------------------- #
# what this means for the historical sweep
# --------------------------------------------------------------------------- #

def test_no_size_reaches_200_in_one_layer_with_room_to_act():
    """Documents the practical consequence, so a regression is visible.

    Not even 5 mm: 200 is 89 % of its 225 ceiling. The guide's claim that
    "5 mm is the only size that reaches 200 objects" is true of *placement* and
    false of *usability* — at 200 the tool has nowhere left to touch down.
    """
    ok = [s for s in SWEEP
          if 200 / single_layer_capacity("cube", s, 1, BOX) < PACKING_WARN_FRACTION]
    assert ok == []


def test_five_mm_reaches_150_with_room_to_act():
    frac = 150 / single_layer_capacity("cube", 0.005, 1, BOX)
    assert frac < PACKING_WARN_FRACTION


# --------------------------------------------------------------------------- #
# collapse to a monolayer — usually the binding constraint at high counts
# --------------------------------------------------------------------------- #

def test_a_stack_under_threshold_still_warns_about_flattening():
    """Planning a stack to 62 % per layer does not make the TOTAL fit.

    A pile spreads as it is pushed, so over a trajectory it tends toward a
    single layer, and at that point the total count is what matters. This is
    the constraint that actually blocks 150-200 objects.
    """
    msgs = []
    check_packing_fraction("cube", 0.0085, 150, BOX, n_layers=3,
                           log=msgs.append)
    joined = " ".join(msgs)
    assert "ONE layer's placement capacity" in joined
    assert "tends toward a monolayer" in joined


def test_no_collapse_warning_when_the_total_would_fit_flat():
    msgs = []
    check_packing_fraction("cube", 0.005, 100, BOX, n_layers=2, log=msgs.append)
    assert not any("tends toward a monolayer" in m for m in msgs)


def test_single_layer_never_gets_the_collapse_warning():
    """It is already flat; there is nothing to collapse into."""
    msgs = []
    check_packing_fraction("cube", 0.005, 200, BOX, n_layers=1, log=msgs.append)
    assert not any("tends toward a monolayer" in m for m in msgs)


def test_spheres_have_about_twice_a_cubes_capacity():
    """The escape route the collapse warning points at.

    A sphere needs no free-yaw inflation, so it occupies ~half the area of a
    cube of the same size. That is the one lever that changes the count a flat
    pile can hold without changing particle size or tray.
    """
    for size in SWEEP:
        ratio = (single_layer_capacity("sphere", size, 1, BOX)
                 / single_layer_capacity("cube", size, 1, BOX))
        assert 1.6 <= ratio <= 2.2
