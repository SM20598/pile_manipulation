#####################################################################
# CONVERTS A DINO-WM-FORMAT GRANULAR DATASET (as written by            #
# export_dino_wm_dataset.py: states.pth / actions.pth / proprios.pth / #
# seq_lengths.pth / obses/episode_NNNNNN.pth) INTO A SINGLE .h5 FILE   #
# consumable by le-wm's stable_worldmodel.data.HDF5Dataset reader.     #
#                                                                       #
# Column mapping (dino_wm layout -> le-wm HDF5 columns):                #
#   obses/episode_{i}.pth (T, H, W, 3) uint8   -> pixels               #
#   actions.pth[i]         (T, action_dim)      -> action              #
#   proprios.pth[i]        (T, proprio_dim)      -> proprio             #
#   states.pth[i]          (T, state_dim)        -> state               #
#                                                                        #
# le-wm's own convention for "no action taken from this frame" (the     #
# last frame of an episode) is NaN, not the zero-padding dino_wm uses -  #
# see train.py's `torch.nan_to_num(batch["action"], 0.0)` and           #
# get_column_normalizer's NaN-row filtering before computing z-score     #
# stats. This script overwrites dino_wm's zero-padded last action row   #
# with NaN to match that convention.                                    #
#                                                                        #
# Usage:                                                                #
#   python export_leworldmodel_dataset.py \                             #
#       --input-dir /path/to/dino_wm_dataset/occupancy \                #
#       --output-path ~/.stable_worldmodel/datasets/granular_test10.h5 \#
#       --num-episodes 10                                               #
#####################################################################

import argparse
from pathlib import Path

import numpy as np
import torch


def export(input_dir: Path, output_path: Path, num_episodes: int | None, mode: str):
    from stable_worldmodel.data import HDF5Writer

    states = torch.load(input_dir / "states.pth", map_location="cpu")
    actions = torch.load(input_dir / "actions.pth", map_location="cpu")
    proprios = torch.load(input_dir / "proprios.pth", map_location="cpu")
    seq_lengths = torch.load(input_dir / "seq_lengths.pth", map_location="cpu")

    n_available = states.shape[0]
    n_episodes = n_available if num_episodes is None else min(num_episodes, n_available)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with HDF5Writer(output_path, mode=mode) as writer:
        for i in range(n_episodes):
            T = int(seq_lengths[i])

            pixels = torch.load(input_dir / "obses" / f"episode_{i:06d}.pth", map_location="cpu")
            pixels = pixels[:T].numpy().astype(np.uint8)

            action = actions[i, :T].numpy().astype(np.float32).copy()
            action[-1] = np.nan  # le-wm's "no action from the last frame" sentinel

            proprio = proprios[i, :T].numpy().astype(np.float32)
            state = states[i, :T].numpy().astype(np.float32)

            writer.write_episode({
                "pixels": pixels,
                "action": action,
                "proprio": proprio,
                "state": state,
            })

    print(f" > wrote {n_episodes} episodes to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a dino_wm-format granular dataset into an HDF5 file for le-wm."
    )
    parser.add_argument("--input-dir", required=True, help=(
        "Path to a dino_wm dataset's obs_type directory (the one directly "
        "containing states.pth/actions.pth/proprios.pth/seq_lengths.pth/obses/), "
        "e.g. .../dino_wm_angle_density_2000/occupancy"
    ))
    parser.add_argument("--output-path", required=True, help=(
        "Destination .h5 file. To be loadable via the le-wm data config's "
        "`name: <basename>.h5`, place this under "
        "$STABLEWM_HOME/datasets/ (default ~/.stable_worldmodel/datasets/)."
    ))
    parser.add_argument("--num-episodes", type=int, default=None, help=(
        "Only export the first N episodes (for a quick training-loop smoke "
        "test). Default: export all episodes in the source dataset."
    ))
    parser.add_argument("--mode", choices=["append", "overwrite", "error"], default="overwrite", help=(
        "HDF5Writer write mode - see stable_worldmodel.data.format.WRITE_MODES. "
        "Defaults to 'overwrite' since re-running this script during iteration "
        "is the common case, unlike export_dino_wm_dataset.py's from-scratch export."
    ))
    return parser.parse_args()


def main():
    args = parse_args()
    export(
        Path(args.input_dir),
        Path(args.output_path).expanduser(),
        args.num_episodes,
        args.mode,
    )


if __name__ == "__main__":
    main()
