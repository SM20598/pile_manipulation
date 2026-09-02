################################
# A SCRIPT FOR DATA COLLECTION #
################################

import argparse
import itertools
import yaml
import numpy as np
import torch
from pathlib import Path

from sandbox_manipulation import SandboxManipulation
from state_library import default_library_path, load_or_build_state_library
from training import export_dino_wm_dataset

##################################
# PARAMS THAT REQUIRE RESTARTING #
##################################
BASIC_SETTING = "basic"
DEFAULT_SHAPES = ["cube"]
DEFAULT_NUM_PARTICLES = [50]
PARTICLE_SIZES = np.linspace(0.005, 0.012, 5).tolist()

PARTICLE_FRICTIONS = np.linspace(0.05, 0.5, 5).tolist()
PARTICLE_DENSITIES = np.linspace(750, 5000, 5).tolist()
BOX_FRICTION = np.linspace(0.05, 0.5, 4).tolist()
PER_PARTICLE_VALUE_PROBABILITY = 0.5

PARTICLE_SIZES = [0.005]
# len(PARTICLE_FRICTIONS) = number of material-batches; total episodes =
# that * --n-envs. All entries here are identical (0.12), so batches don't
# actually vary material properties - this is purely a lever for total
# episode count, with per-episode diversity coming from shuffle_particles()'
# own randomness. x5 with --n-envs 100 = 500 episodes (confirmed n_envs=100
# runs cleanly on this GPU, ~4.7GB peak vs 16.3GB total - far cheaper than
# 100 batches * n_envs=10 for the same episode count).
PARTICLE_FRICTIONS = [0.12] * 5
PARTICLE_DENSITIES = [750]
BOX_FRICTION = [0.12]
PER_PARTICLE_VALUE_PROBABILITY = 0



def scalar_or_particle_values(value: float, n_particles: int, rng: np.random.Generator):
    if rng.random() >= PER_PARTICLE_VALUE_PROBABILITY:
        return float(value), None
    return float(value), rng.uniform(0.8 * value, 1.2 * value, n_particles).tolist()


def build_particle_size_settings(sizes: list[float], n_particles: int, rng: np.random.Generator):
    settings = []
    for size in sizes:
        base, sampled = scalar_or_particle_values(size, n_particles, rng)
        settings.append({"base": base, "sampled": sampled})
    return settings


def build_property_env_settings(n_particles: int, rng: np.random.Generator):
    env_settings = []
    for particle_friction, particle_density, box_friction in itertools.product(
        PARTICLE_FRICTIONS,
        PARTICLE_DENSITIES,
        BOX_FRICTION,
    ):
        friction_base, friction_sampled = scalar_or_particle_values(particle_friction, n_particles, rng)
        density_base, density_sampled = scalar_or_particle_values(particle_density, n_particles, rng)
        env_settings.append(
            {
                "particle_friction": friction_base,
                "sampled_particle_friction": friction_sampled,
                "particle_density": density_base,
                "sampled_particle_density": density_sampled,
                "box_friction": float(box_friction),
            }
        )
    return env_settings


def read_yaml(path: str):
    base_dir = Path(__file__).parent
    full_path = base_dir / path
    with open(full_path) as stream:
        return yaml.safe_load(stream)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect sandbox manipulation data.")
    parser.add_argument("--settings", nargs="+", default=BASIC_SETTING)
    parser.add_argument("--particle-shape", choices=["config", "cube", "sphere", "cylinder", "rectangle"], default=DEFAULT_SHAPES)
    parser.add_argument("--num-particles", nargs="+", type=int, default=DEFAULT_NUM_PARTICLES)
    parser.add_argument("--particle-sizes", nargs="+", type=float, default=PARTICLE_SIZES)
    parser.add_argument("--n-envs", type=int, default=10)
    parser.add_argument("--samples-per-env", type=int, default=5)
    parser.add_argument("--center-bias", type=float, default=0.0, help=(
        "Forwarded to SandboxManipulation.collect_data_samples()'s center_bias: "
        "when > 0, each push's STOP is pulled toward the box center by a random "
        "fraction in [0, center_bias] instead of sampled uniformly, so episodes "
        "demonstrate gathering material inward rather than a systematic edge/wall "
        "drift (see generate_action_samples() docstring). 0 (default) = unchanged."
    ))
    parser.add_argument("--seed", type=int, default=None, help=(
        "Seed for BOTH generators. Previously this script called "
        "np.random.default_rng() with no argument and left every torch draw "
        "(spawn poses, orientations, every action) unseeded, so a run could "
        "not be repeated - which also meant a run that produced something odd "
        "could not be replayed to look at it. The seed is recorded in each "
        "batch's saved config."
    ))
    parser.add_argument("--state-library", type=int, default=0, metavar="N", help=(
        "Settle N piles once per build, expand each by the container's "
        "symmetry group, save settled_states.pt beside the data, and reset by "
        "RESTORING a state instead of re-settling. shuffle_particles() itself "
        "runs zero simulation steps, so all of a reset's cost is the settle "
        "that follows it: measured 54x faster at n=50 and 6184x at n=200. A "
        "square tray admits the full dihedral group D4, so each settle yields "
        "8 variants, and each settle randomizes every env independently - the "
        "bank is N x n_envs x 8 states for the cost of N settles. "
        "0 (default) disables it and reproduces the previous behaviour."
    ))
    parser.add_argument("--state-library-damping", type=float, default=0.0, help=(
        "Temporary viscous damping applied DURING library settles only. A "
        "numerical convergence aid, not a physical model - real air drag on a "
        "5 mm cube at 50 mm/s is ~3e-5 of its weight. Deliberately never "
        "applied to the post-push settle, where cutting the relaxation short "
        "would bias the recorded s' toward smaller displacements."
    ))
    parser.add_argument("--start-sampling",
                        choices=["auto", "uniform", "density", "free", "composed"],
                        default="auto", help=(
        "How each push's touchdown pose is drawn. 'auto' (default) keeps the "
        "current behaviour: density-weighted, which aims the tool at material "
        "but can materialize the blade inside a cube. 'free' draws from the "
        "tool's free configuration space instead, cutting touchdown overlap "
        "from ~89%% to ~16%% but drifting toward empty tray. 'composed' lets "
        "density choose the neighbourhood and then moves the pose the shortest "
        "distance that makes it legal - measured to keep density's spatial "
        "distribution while cutting overlap to ~28%%. See "
        "generate_action_samples()."
    ))
    parser.add_argument("--shared-travel-distance", action="store_true", help=(
        "Give every env in a batch the same push LENGTH for a given sample, "
        "keeping its own start point, direction and blade yaw. Envs step in "
        "lockstep and the sweep is sized from the longest travel in the batch, "
        "so independent lengths make every env run for the longest one's "
        "duration - worth up to 12x of end-to-end batch time at 150 objects. "
        "Costs the within-batch spread of one of five action dimensions, and "
        "truncates pushes that would leave the sampling box."
    ))
    parser.add_argument("--output-root", default="data/corl")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--viewer-type", choices=["observer", "bird", "leveled"], default=None)
    parser.add_argument("--render-images", action=argparse.BooleanOptionalAction, default=True, help=(
        "Render an RGB frame per env at every state snapshot and save it into each run's "
        "_rollout.pt file alongside states/actions/config, on by default like the rest of "
        "the collected data (one camera per env, so this adds n_envs render calls per "
        "sample; pass --no-render-images to skip it, e.g. for large sweeps that only need "
        "the UNet's occupancy-grid path)."
    ))
    parser.add_argument("--render-resolution", nargs=2, type=int, default=[128, 128], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--export-dino-wm", action="store_true", help=(
        "After collecting each (shape, n_particles, particle_size) run, also export it into "
        "DINO-WM's dataset format (see training/export_dino_wm_dataset.py), written under "
        "--dino-wm-output-root with the same shape/n<N>/size<S> layout as --output-root."
    ))
    parser.add_argument("--dino-wm-output-root", default=None, help=(
        "Base output dir for the DINO-WM export. Defaults to '<output-root>_dino_wm'."
    ))
    parser.add_argument("--dino-wm-obs-types", nargs="+", choices=["occupancy", "rendered"], default=["occupancy", "rendered"])
    parser.add_argument("--dino-wm-resolution-scale", type=float, default=1.0, help="Occupancy grid pixels-per-mm scale.")
    return parser.parse_args()


def main():
    args = parse_args()
    # Seed BOTH generators: the numpy rng below draws the material/size
    # settings, while torch draws spawn poses, orientations and every action.
    # Seeding one and not the other does not make a run reproducible.
    rng = np.random.default_rng(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    
    if args.samples_per_env <= 0:
        raise ValueError("--samples-per-env must be positive")

    if args.export_dino_wm and "rendered" in args.dino_wm_obs_types and not args.render_images:
        raise ValueError(
            "--dino-wm-obs-types includes 'rendered' but --render-images was not passed; "
            "either add --render-images or drop 'rendered' from --dino-wm-obs-types."
        )
    dino_wm_output_root = args.dino_wm_output_root or f"{args.output_root}_dino_wm"

    config = read_yaml(f"configs/{BASIC_SETTING}.yaml")
    config.setdefault("data_collection", {}).update({
        "render_images": args.render_images,
        "render_resolution": args.render_resolution,
        "seed": args.seed,
    })

    # Iterate shapes
    shapes = [args.particle_shape] if args.particle_shape != DEFAULT_SHAPES else DEFAULT_SHAPES
    for shape in shapes:
        config["material"]["shape"] = shape

        # Iterate number of particles
        for n_p in args.num_particles:
            config["material"]["n_particles"] = n_p
            env_settings = build_property_env_settings(n_p, rng)
            
            # Iterate particle sizes: each value in --particle-sizes is a nominal/base
            # size, independently expanded to a per-particle sampled list (or left
            # scalar) by scalar_or_particle_values - same handling whether it's the
            # default sweep or a custom list, and correct for every n_p in --num-particles.
            sizes = build_particle_size_settings(args.particle_sizes, n_p, rng)
            for size_setting in sizes:
                config["material"]["particle_size"] = size_setting["base"]
                config.setdefault("data_collection", {})["sampled"] = {}
                if size_setting["sampled"] is not None:
                    config["data_collection"]["sampled"]["particle_size"] = size_setting["sampled"]


                # iterate through material settings
                print(f"\n+++ shape={shape}, size={size_setting['base']}, n_particles={n_p} +++")

                sm = SandboxManipulation(
                    config=config,
                    n_envs=args.n_envs,
                    debug=args.debug,
                    viewer_type=args.viewer_type,
                )

                sm.build()

                leaf_subpath = f"{shape}/n{n_p}/size{size_setting['base']}"

                # Build (or reuse) the settled-state library once per BUILD.
                # It is specific to this (shape, n_particles, particle_size),
                # which is exactly the granularity a rebuild happens at, so it
                # lives beside the data it was settled for and is found again
                # on a later run without being pointed at.
                state_library = None
                if args.state_library > 0:
                    sm.set_material_properties(env_settings[0])
                    lib_path = default_library_path(
                        Path(__file__).parent / args.output_root,
                        shape, n_p, size_setting["base"])
                    print(f"\n--- settled-state library -> {lib_path}", flush=True)
                    state_library = load_or_build_state_library(
                        sm, lib_path, n_settles=args.state_library,
                        damping=args.state_library_damping)

                for property_idx, property_setting in enumerate(env_settings):
                    print(f"\n--- material batch {property_idx + 1}/{len(env_settings)}", flush=True)

                    sm.set_material_properties(property_setting)
                    try:
                        if state_library is None:
                            sm.shuffle_particles()
                        sm.collect_data_samples(
                            n_samples=args.samples_per_env,
                            path=f"{args.output_root}/{leaf_subpath}",
                            center_bias=args.center_bias,
                            start_sampling=args.start_sampling,
                            shared_travel_distance=args.shared_travel_distance,
                            state_library=state_library,
                        )
                    except RuntimeError as e:
                        print(f"Maximum attempts reached, stopped retrying to shuffle, skipping: {e}")

                if state_library is not None:
                    state_library.save(
                        Path(__file__).parent / args.output_root / leaf_subpath)


                sm.destroy()

                if args.export_dino_wm:
                    # gs.init() (inside SandboxManipulation.__init__) sets torch's
                    # process-wide default device to cuda as a side effect; the
                    # exporter's rasterizer allocates bare CPU tensors (it runs
                    # cv2-based drawing on numpy views) and doesn't expect that
                    # default - see the same fix in env/granular/granular_env.py.
                    torch.set_default_device("cpu")
                    # matches how SandboxManipulation.collect_data_samples() resolved `path` above:
                    # relative to Genesis/, not Genesis/data/ (--output-root already includes "data/").
                    leaf_input_path = Path(__file__).parent / args.output_root / leaf_subpath
                    output_roots = {
                        obs_type: Path(dino_wm_output_root) / leaf_subpath / obs_type
                        if obs_type in args.dino_wm_obs_types else None
                        for obs_type in ("occupancy", "rendered")
                    }
                    print(f"\n>>> Exporting DINO-WM format for {leaf_subpath} to {dino_wm_output_root}/{leaf_subpath}")
                    export_dino_wm_dataset.export(
                        [leaf_input_path],
                        output_roots,
                        resolution_scale=args.dino_wm_resolution_scale,
                        soft_particle_occupancy=False,
                    )


if __name__ == "__main__":
    main()
