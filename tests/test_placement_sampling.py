"""Tests for Genesis/placement_sampling.py — configuration-space sampling of
collision-free tool touchdown poses.

Genesis-free: the module depends only on torch/numpy/scipy.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Genesis"))

from placement_sampling import (  # noqa: E402
    build_occupancy, clearance_map, free_placements, nearest_free_placement,
    sample_free_placements, _rotated_rect_kernel,
)

BOX = (0.128, 0.128)
RES = 0.002
TOOL_L, TOOL_W = 0.04, 0.002


def _occupancy(positions, half=0.004, n_envs=1):
    pos = torch.tensor(positions, dtype=torch.float32).view(n_envs, -1, 2)
    half_xy = torch.full((pos.shape[1], 2), half)
    return build_occupancy(pos, half_xy, BOX, RES)


def test_empty_box_is_entirely_free():
    # one particle parked far outside the tray
    occ, meta = _occupancy([[10.0, 10.0]])
    assert not occ.any()

    angles = torch.tensor([0.0])
    free = free_placements(occ, meta, angles, TOOL_L, TOOL_W)
    # everything except the wall margin should be available
    assert free.float().mean() > 0.5


def test_particle_blocks_placements_around_itself():
    occ, meta = _occupancy([[0.0, 0.0]])
    assert occ.any()

    angles = torch.tensor([0.0])
    free = free_placements(occ, meta, angles, TOOL_L, TOOL_W)

    centre_row, centre_col = meta["H"] // 2, meta["W"] // 2
    assert not free[0, 0, centre_row, centre_col], "tool centre on a particle must be blocked"
    # a horizontal tool is blocked far away in x but not in y
    dx = int(round(0.015 / RES))
    dy = int(round(0.015 / RES))
    assert not free[0, 0, centre_row, centre_col + dx], "blocked along the blade"
    assert free[0, 0, centre_row + dy, centre_col], "free perpendicular to the blade"


def test_orientation_changes_the_free_set():
    """A long thin tool must be orientation-sensitive, or the C-space is wrong."""
    occ, meta = _occupancy([[0.0, 0.0]])
    angles = torch.tensor([0.0, math.pi / 2])
    free = free_placements(occ, meta, angles, TOOL_L, TOOL_W)

    r, c = meta["H"] // 2, meta["W"] // 2
    off = int(round(0.015 / RES))
    # offset along +x: blocked when the blade lies along x, free when across
    assert not free[0, 0, r, c + off]
    assert free[0, 1, r, c + off]


def test_wall_margin_excludes_poses_that_poke_through():
    occ, meta = _occupancy([[10.0, 10.0]])          # empty tray
    angles = torch.tensor([0.0])
    free = free_placements(occ, meta, angles, TOOL_L, TOOL_W)

    # with the blade along x, its half-length is 0.02 -> the outer 0.02 m of
    # the x range must be unusable
    margin_cells = int(math.ceil((TOOL_L / 2) / RES))
    assert not free[0, 0, :, :margin_cells].any()
    assert not free[0, 0, :, meta["W"] - margin_cells:].any()


def test_wall_margin_widens_the_excluded_border():
    """placement-aware sampling must keep the same distance from the wall that
    the blind sampler does, or it quietly shifts touchdowns toward the rim."""
    occ, meta = _occupancy([[10.0, 10.0]])          # empty tray
    angles = torch.tensor([0.0])
    margin = 0.02

    bare = free_placements(occ, meta, angles, TOOL_L, TOOL_W)
    with_margin = free_placements(occ, meta, angles, TOOL_L, TOOL_W,
                                  wall_margin=margin)

    cells = int(math.ceil((TOOL_L / 2 + margin) / RES))
    assert not with_margin[0, 0, :, :cells].any()
    assert not with_margin[0, 0, :, meta["W"] - cells:].any()
    # and it is strictly stricter, not merely different
    assert with_margin[0, 0].sum() < bare[0, 0].sum()
    assert (bare[0, 0] | with_margin[0, 0]).equal(bare[0, 0])


def test_wall_margin_applies_to_both_axes_at_any_yaw():
    occ, meta = _occupancy([[10.0, 10.0]])
    angles = torch.tensor([math.pi / 2])            # blade along y
    margin = 0.015
    free = free_placements(occ, meta, angles, TOOL_L, TOOL_W, wall_margin=margin)

    cells = int(math.ceil((TOOL_L / 2 + margin) / RES))
    assert not free[0, 0, :cells, :].any()
    assert not free[0, 0, meta["H"] - cells:, :].any()


def test_clearance_is_zero_on_obstacles_and_positive_in_free_space():
    occ, meta = _occupancy([[0.0, 0.0]])
    dist = clearance_map(occ, meta)

    r, c = meta["H"] // 2, meta["W"] // 2
    assert dist[0, r, c] == pytest.approx(0.0)
    assert dist[0, 0, 0] > 0.0
    # a cell ~10 mm from a 4 mm-half particle should read roughly that far
    probe = dist[0, r, c + int(round(0.02 / RES))]
    assert 0.010 < float(probe) < 0.020


def test_fully_blocked_env_reports_not_ok():
    """The documented degradation path: dense pile -> caller falls back."""
    occ, meta = _occupancy([[10.0, 10.0]])
    occ[:] = True                                   # nothing is free anywhere
    angles = torch.tensor([0.0])
    free = free_placements(occ, meta, angles, TOOL_L, TOOL_W)
    assert not free.any()

    xy, yaw, ok = sample_free_placements(free, meta, angles, n_samples=5)
    assert ok.shape == (1, 5)
    assert not ok.any(), "a blocked env must report ok=False, not raise"
    assert torch.isfinite(xy).all(), "outputs must stay finite so callers can mask"


def test_sampled_placements_land_in_free_space():
    occ, meta = _occupancy([[0.0, 0.0], [0.03, 0.02], [-0.025, 0.03]])
    angles = torch.tensor([0.0, math.pi / 4, math.pi / 2])
    free = free_placements(occ, meta, angles, TOOL_L, TOOL_W)

    xy, yaw, ok = sample_free_placements(free, meta, angles, n_samples=64)
    assert ok.all()

    # every sample must map back to a free cell of its own orientation
    col = ((xy[..., 0] + BOX[0] / 2) / RES).long().clamp(0, meta["W"] - 1)
    row = ((xy[..., 1] + BOX[1] / 2) / RES).long().clamp(0, meta["H"] - 1)
    a_i = torch.stack([(angles - y).abs().argmin() for y in yaw.flatten()]).view(yaw.shape)
    assert free[0, a_i[0], row[0], col[0]].all()


def test_sampled_yaws_come_from_the_requested_bins():
    occ, meta = _occupancy([[0.0, 0.0]])
    angles = torch.tensor([-0.5, 0.0, 0.7])
    free = free_placements(occ, meta, angles, TOOL_L, TOOL_W)
    _, yaw, ok = sample_free_placements(free, meta, angles, n_samples=32)
    assert torch.isin(yaw[ok], angles).all()


def test_rotated_kernel_covers_the_tool_area():
    kernel = _rotated_rect_kernel(TOOL_L, TOOL_W, 0.0, RES, torch.device("cpu"))
    assert kernel.sum() > 0
    # rotating by 90 deg transposes the footprint
    k90 = _rotated_rect_kernel(TOOL_L, TOOL_W, math.pi / 2, RES, torch.device("cpu"))
    assert torch.allclose(kernel.sum(), k90.sum(), rtol=0.15)
    assert kernel.shape == k90.shape


def test_multi_env_grids_are_independent():
    pos = torch.tensor([[[0.0, 0.0]], [[10.0, 10.0]]], dtype=torch.float32)
    half_xy = torch.full((1, 2), 0.004)
    occ, meta = build_occupancy(pos, half_xy, BOX, RES)
    assert occ[0].any() and not occ[1].any()


def test_active_limits_which_particles_are_rasterized():
    """Parked particles must not be treated as obstacles inside the tray."""
    pos = torch.tensor([[[0.0, 0.0], [0.02, 0.02]]], dtype=torch.float32)
    half_xy = torch.full((2, 2), 0.004)
    full, _ = build_occupancy(pos, half_xy, BOX, RES)
    limited, _ = build_occupancy(pos, half_xy, BOX, RES, active=1)
    assert full.sum() > limited.sum()


# --------------------------------------------------------------------------- #
# nearest_free_placement — the composed sampler
#
# This is the one piece of sampling logic that exists in neither historical
# branch: it is what lets the density-weighted start sampler and the free-space
# sampler compose instead of competing. So it is tested on its own terms rather
# than by analogy with sample_free_placements.
# --------------------------------------------------------------------------- #

def _one_env_free_grid(box=0.128, res=0.004, n_angles=2):
    """A free set with a single free column on the +x side of the tray."""
    H = W = int(round(box / res))
    free = torch.zeros((1, n_angles, H, W), dtype=torch.bool)
    free[0, :, :, W - 1] = True           # rightmost column free at every yaw
    meta = {"resolution": res, "width": box, "depth": box, "H": H, "W": W}
    return free, meta


def test_nearest_free_placement_snaps_to_the_only_free_region():
    free, meta = _one_env_free_grid()
    angles = torch.tensor([-math.pi / 4, 0.0])
    # target sits on the far LEFT; the only legal placement is on the right
    targets = torch.tensor([[[-0.06, 0.0]]])

    xy, yaw, ok = nearest_free_placement(free, meta, angles, targets)

    assert ok.all()
    # snapped into the free column, i.e. near +width/2
    assert xy[0, 0, 0] > 0.05
    assert float(yaw[0, 0]) in set(angles.tolist())


def test_nearest_free_placement_picks_the_closest_of_several():
    """The whole point: minimum displacement from the density-chosen point."""
    free, meta = _one_env_free_grid()
    angles = torch.tensor([0.0, 0.0])
    H = meta["H"]
    # free column spans all rows, so y should track the target's y
    targets = torch.tensor([[[0.0, -0.05], [0.0, 0.0], [0.0, 0.05]]])

    xy, yaw, ok = nearest_free_placement(free, meta, angles, targets)

    assert ok.all()
    ys = xy[0, :, 1]
    assert ys[0] < ys[1] < ys[2], "each sample snaps to its own nearest cell"
    for i, want in enumerate([-0.05, 0.0, 0.05]):
        assert abs(float(ys[i]) - want) <= meta["resolution"]


def test_nearest_free_placement_reports_not_ok_for_a_blocked_env():
    """A full tray must degrade to the caller's blind draw, not raise."""
    free, meta = _one_env_free_grid()
    free[:] = False

    xy, yaw, ok = nearest_free_placement(
        free, meta, torch.tensor([0.0, 0.0]), torch.tensor([[[0.0, 0.0]]]))

    assert not ok.any()


def test_nearest_free_placement_is_per_env():
    free_a, meta = _one_env_free_grid()
    free_b = free_a.clone()
    free_b[:] = False                      # env 1 entirely blocked
    free = torch.cat([free_a, free_b], dim=0)
    targets = torch.zeros((2, 1, 2))

    xy, yaw, ok = nearest_free_placement(free, meta, torch.tensor([0.0, 0.0]), targets)

    assert bool(ok[0, 0]) and not bool(ok[1, 0])


def test_nearest_free_placement_subsamples_large_free_sets():
    """max_candidates bounds the work without changing the contract."""
    free, meta = _one_env_free_grid()
    free[:] = True                         # everything free -> large candidate set
    targets = torch.tensor([[[0.02, -0.03]]])

    xy, yaw, ok = nearest_free_placement(
        free, meta, torch.tensor([0.0, 0.0]), targets, max_candidates=64)

    assert ok.all()
    # with everything free the nearest cell is the target's own cell; a 64-cell
    # subsample cannot guarantee that, but it must stay inside the tray
    assert abs(float(xy[0, 0, 0])) <= meta["width"] / 2
    assert abs(float(xy[0, 0, 1])) <= meta["depth"] / 2


def test_nearest_free_placement_displacement_beats_uniform_free_draw():
    """Composition must move the pose LESS than an unbiased free draw would."""
    free, meta = _one_env_free_grid(res=0.002)
    angles = torch.tensor([0.0, 0.0])
    target = torch.tensor([[[0.0, 0.04]]])

    xy_near, _, ok_near = nearest_free_placement(free, meta, angles, target)
    torch.manual_seed(0)
    xy_unif, _, ok_unif = sample_free_placements(free, meta, angles, 1)

    assert ok_near.all() and ok_unif.all()
    d_near = (xy_near[0, 0] - target[0, 0]).norm()
    d_unif = (xy_unif[0, 0] - target[0, 0]).norm()
    assert d_near <= d_unif
