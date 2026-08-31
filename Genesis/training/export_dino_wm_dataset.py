#####################################################################
# EXPORTS GENESIS ROLLOUT DATA (see sandbox_manipulation.py's       #
# "_rollout.pt" files) INTO THE FLAT-DIRECTORY FORMAT DINO-WM'S     #
# DATASET LOADERS EXPECT (states.pth / actions.pth / proprios.pth / #
# seq_lengths.pth / obses/episode_NNN.pth).                         #
#                                                                    #
# Writes two parallel, directly-comparable observation variants     #
# from the same underlying trajectories:                            #
#   occupancy - particles rasterized top-down (reuses               #
#               PileSweepData's cv2 rasterizer, no camera needed)   #
#   rendered  - real camera frames captured during collection       #
#               (requires data_collection.py --render-images)       #
#####################################################################

import argparse
import sys
from pathlib import Path
import re

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from dataset import PileSweepData  # noqa: E402


def _resolve_input_path(path: str, data_root: Path) -> Path:
    path = Path(path)

    return path if path.is_absolute() else data_root / path


def find_rollout_files(root: Path):
    """
    Find all rollout files named _[NUM]_rollout.pt, e.g.:

        _0_rollout.pt
        _1_rollout.pt
        _25_rollout.pt
        _1000_rollout.pt

    The files are returned in numeric order.
    """
    pattern = re.compile(r"^_(\d+)_rollout\.pt$")
    files = []

    for data_file in root.rglob("*_rollout.pt"):
        match = pattern.match(data_file.name)
        if match is None:
            continue

        config_file = data_file.with_name(
            data_file.name.replace("_rollout.pt", "_config.yaml")
        )

        if config_file.exists():
            rollout_idx = int(match.group(1))
            files.append((rollout_idx, data_file, config_file))

    # Sort numerically rather than lexicographically:
    # _2, _10 rather than _10, _2
    files.sort(key=lambda x: x[0])

    return [(data_file, config_file) for _, data_file, config_file in files]



def make_rasterizer(config: dict, resolution_scale: float, soft_particle_occupancy: bool) -> PileSweepData:
    """
    Builds a bare PileSweepData instance for its rasterization methods only
    (`_create_grids`, `_draw_particle_grid`), without going through its
    file-loading `__init__` - avoids re-reading every run file in a folder
    just to obtain a pixel grid.
    """
    rasterizer = object.__new__(PileSweepData)
    rasterizer.resolution_scale = float(resolution_scale)
    rasterizer.to_pxl = 1e3 * rasterizer.resolution_scale
    rasterizer.soft_particle_occupancy = bool(soft_particle_occupancy)
    rasterizer.include_sweep_removed = False
    rasterizer._create_grids(config)
    return rasterizer


def rasterize_particle_frame(rasterizer: PileSweepData, particle_state_m: torch.Tensor, config: dict) -> torch.Tensor:
    """particle_state_m: (n_particles, 7) in meters. Returns uint8 (H, W, 3)."""
    particles = particle_state_m.clone()
    particles[:, :3] = particles[:, :3] * rasterizer.to_pxl + rasterizer.ctr_in_PXL
    grid = torch.zeros_like(rasterizer._output_grid)
    rasterizer._draw_particle_grid(particles, grid, config)
    img = (grid.clamp(0, 1) * 255).to(torch.uint8)
    return img.unsqueeze(-1).repeat(1, 1, 3)


def rasterize_occupancy_frames(rasterizer: PileSweepData, states: torch.Tensor, states_: torch.Tensor, config: dict) -> torch.Tensor:
    """states/states_: (n_samples, n_particles, 7) meters. Returns uint8 (T, H, W, 3), T = n_samples + 1."""
    n_samples = states.shape[0]
    frames = [rasterize_particle_frame(rasterizer, states[t], config) for t in range(n_samples)]
    frames.append(rasterize_particle_frame(rasterizer, states_[-1], config))
    return torch.stack(frames, dim=0)


def build_state_sequence(states: torch.Tensor, states_: torch.Tensor) -> torch.Tensor:
    """states/states_: (n_samples, n_particles, 7). Returns flattened (T, n_particles*7)."""
    frames = torch.cat([states[0:1], states_], dim=0)  # (T, P, 7)
    return frames.reshape(frames.shape[0], -1)


def build_action_and_proprio(p_starts: torch.Tensor, p_stops: torch.Tensor, angles: torch.Tensor):
    """
    p_starts/p_stops: (n_samples, 3), angles: (n_samples,).

    Returns action (T, 5) = [x_start, y_start, x_end, y_end, angle] of the
    push taken FROM that frame (last frame has no push, so it's
    zero-padded), and proprio (T, 3) = tool pose [x, y, angle] at that frame
    - the position the plate is about to sweep from (last frame reuses the
    final push's stop pose and angle, since no further push follows it).

    `angle` is the plate's own rotation about z, sampled independently from
    the start->stop travel direction (see sandbox_manipulation.py's
    generate_action_samples) - it's NOT recoverable from start/stop alone,
    so it must be carried as its own action dimension rather than
    reconstructed downstream via atan2(travel direction).
    """
    n_samples = p_starts.shape[0]
    T = n_samples + 1

    action = torch.zeros((T, 5), dtype=torch.float32)
    action[:n_samples, 0:2] = p_starts[:, 0:2]
    action[:n_samples, 2:4] = p_stops[:, 0:2]
    action[:n_samples, 4] = angles

    proprio = torch.zeros((T, 3), dtype=torch.float32)
    proprio[:n_samples, 0:2] = p_starts[:, 0:2]
    proprio[:n_samples, 2] = angles
    proprio[n_samples, 0:2] = p_stops[-1, 0:2]
    proprio[n_samples, 2] = angles[-1]

    return action, proprio


def pad_stack(tensors: list) -> torch.Tensor:
    T_max = max(t.shape[0] for t in tensors)
    D = tensors[0].shape[-1]
    out = torch.zeros((len(tensors), T_max) + tensors[0].shape[1:], dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, :t.shape[0]] = t
    return out


def export(input_paths, output_roots: dict, resolution_scale: float, soft_particle_occupancy: bool):
    """
    output_roots: {"occupancy": Path or None, "rendered": Path or None}
    """
    obs_types = [k for k, v in output_roots.items() if v is not None]
    for obs_type in obs_types:
        (output_roots[obs_type] / "obses").mkdir(parents=True, exist_ok=True)

    all_states, all_actions, all_proprios, seq_lengths = [], [], [], []
    rasterizer = None
    episode_idx = 0

    for folder in input_paths:
        run_files = find_rollout_files(folder)
        if not run_files:
            raise FileNotFoundError(f"No _rollout.pt files found under {folder}")

        for data_file, config_file in run_files:
            rollout = torch.load(data_file, map_location="cpu")
            config = yaml.full_load(config_file.read_text())

            if "occupancy" in obs_types and rasterizer is None:
                rasterizer = make_rasterizer(config, resolution_scale, soft_particle_occupancy)

            if "rendered" in obs_types and "frames" not in rollout:
                raise ValueError(
                    f"{data_file} has no camera frames; re-collect with "
                    "--render-images to export obs_type='rendered', or drop "
                    "'rendered' from --obs-types."
                )

            n_envs = config["statistics"]["n_envs"]

            for env_idx in range(n_envs):
                states = rollout["states"][env_idx]      # (n_samples, P, 7)
                states_ = rollout["states_"][env_idx]
                p_starts = rollout["p_starts"][env_idx]   # (n_samples, 3)
                p_stops = rollout["p_stops"][env_idx]
                angles = rollout["angles"][env_idx]        # (n_samples,)

                state_seq = build_state_sequence(states, states_)
                action_seq, proprio_seq = build_action_and_proprio(p_starts, p_stops, angles)

                if "occupancy" in obs_types:
                    frames = rasterize_occupancy_frames(rasterizer, states, states_, config)
                    torch.save(frames, output_roots["occupancy"] / "obses" / f"episode_{episode_idx:06d}.pth")

                if "rendered" in obs_types:
                    frames = torch.cat(
                        [rollout["frames"][env_idx, 0:1], rollout["frames_"][env_idx]], dim=0
                    )
                    torch.save(frames, output_roots["rendered"] / "obses" / f"episode_{episode_idx:06d}.pth")

                all_states.append(state_seq)
                all_actions.append(action_seq)
                all_proprios.append(proprio_seq)
                seq_lengths.append(state_seq.shape[0])
                episode_idx += 1

        print(f" > exported {len(run_files)} run file(s) from {folder} ({episode_idx} episodes so far)")

    states_t = pad_stack(all_states)
    actions_t = pad_stack(all_actions)
    proprios_t = pad_stack(all_proprios)
    seq_lengths_t = torch.tensor(seq_lengths)

    for obs_type in obs_types:
        root = output_roots[obs_type]
        torch.save(states_t, root / "states.pth")
        torch.save(actions_t, root / "actions.pth")
        torch.save(proprios_t, root / "proprios.pth")
        torch.save(seq_lengths_t, root / "seq_lengths.pth")
        print(f" > wrote {episode_idx} trajectories to {root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export Genesis rollouts into DINO-WM's dataset format.")
    parser.add_argument("--input-paths", nargs="+", required=True, help=(
        "Folders containing _rollout.pt run files (e.g. 'oneset/cube/n30/size0.012'). "
        "Relative paths are resolved under Genesis/data; absolute paths are used as-is."
    ))
    parser.add_argument("--output-root", required=True, help=(
        "Base output directory. Gets an 'occupancy/' and/or 'rendered/' subdirectory "
        "per --obs-types, each in the layout datasets/genesis_granular_dset.py expects."
    ))
    parser.add_argument("--obs-types", nargs="+", choices=["occupancy", "rendered"], default=["occupancy", "rendered"])
    parser.add_argument("--resolution-scale", type=float, default=1.0, help="Occupancy grid pixels-per-mm scale.")
    parser.add_argument("--soft-particle-occupancy", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(__file__).parent.parent / "data"
    input_paths = [_resolve_input_path(p, data_root) for p in args.input_paths]

    output_root = Path(args.output_root)
    output_roots = {
        obs_type: output_root / obs_type if obs_type in args.obs_types else None
        for obs_type in ("occupancy", "rendered")
    }

    export(
        input_paths,
        output_roots,
        resolution_scale=args.resolution_scale,
        soft_particle_occupancy=args.soft_particle_occupancy,
    )


if __name__ == "__main__":
    main()
