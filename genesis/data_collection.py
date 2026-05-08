################################
# A SCRIPT FOR DATA COLLECTION #
################################

import argparse
import copy
import time
import yaml
from pathlib import Path

from sandbox_manipulation import SandboxManipulation

##################################
# PARAMS THAT REQUIRE RESTARTING #
##################################
DEFAULT_NUM_PARTICLES = [40, 50]
DEFAULT_SETTINGS = ["chickpeas_on_glass", "chickpeas_on_wood"]


def read_yaml(path: str):
    base_dir = Path(__file__).parent
    full_path = base_dir / path
    with open(full_path) as stream:
        return yaml.safe_load(stream)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect sandbox manipulation data.")
    parser.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS)
    parser.add_argument("--num-particles", nargs="+", type=int, default=DEFAULT_NUM_PARTICLES)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--samples-per-env", type=int, default=1000)
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--speed", type=float, default=0.125)
    parser.add_argument("--output-root", default="data/cubes")
    parser.add_argument("--settle-steps", type=int, default=None)
    parser.add_argument("--sweep-steps", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=1)
    parser.add_argument("--phase-progress-interval", type=int, default=10)
    parser.add_argument("--trace-scene-steps", action="store_true")
    parser.add_argument("--lower-steps", type=int, default=None)
    parser.add_argument("--lift-steps", type=int, default=None)
    parser.add_argument("--substeps", type=int, default=None)
    parser.add_argument("--rigid-iterations", type=int, default=None)
    parser.add_argument("--rigid-ls-iterations", type=int, default=None)
    parser.add_argument("--rigid-tolerance", type=float, default=None)
    parser.add_argument("--rigid-ls-tolerance", type=float, default=None)
    parser.add_argument("--box-box-detection", action="store_true")
    parser.add_argument("--use-contact-island", action="store_true")
    parser.add_argument("--use-hibernation", action="store_true")
    parser.add_argument("--goal-threshold", type=float, default=None)
    parser.add_argument("--particle-shape", choices=["config", "cube", "sphere"], default="config")
    parser.add_argument("--settle-angular-damping", type=float, default=None)
    parser.add_argument("--settle-linear-damping", type=float, default=None)
    parser.add_argument("--settle-sleep-threshold", type=float, default=None)
    parser.add_argument("--disable-settle-stabilization", action="store_true")
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--viewer-type", choices=["observer", "bird", "leveled"], default=None)
    parser.add_argument("--update-visualizer", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    for setting in args.settings:
        base_config = read_yaml(f"configs/{setting}.yaml")

        for n_p in args.num_particles:
            config = copy.deepcopy(base_config)
            config["simulation"]["backend"] = args.backend
            config["simulation"]["performance_mode"] = True
            if args.substeps is not None:
                config["simulation"]["substeps"] = args.substeps
            viewer_options = config["simulation"].setdefault("viewer_options", {})
            viewer_options["show_viewer"] = args.show_viewer
            if args.viewer_type is not None:
                viewer_options["viewer_type"] = args.viewer_type
            config["sandbox"]["material"]["properties"]["n_particles"] = n_p
            if args.particle_shape != "config":
                config["sandbox"]["material"]["properties"]["cubes"] = args.particle_shape == "cube"
            config.setdefault("data_collection", {})
            config["data_collection"]["progress"] = not args.quiet_progress
            config["data_collection"]["sample_progress_interval"] = args.progress_interval
            config["data_collection"]["phase_progress_interval"] = args.phase_progress_interval
            config["data_collection"]["trace_scene_steps"] = args.trace_scene_steps
            config["data_collection"]["update_visualizer"] = args.update_visualizer
            if args.disable_settle_stabilization:
                config["data_collection"]["settle_stabilization"] = False
            if args.settle_angular_damping is not None:
                config["data_collection"]["settle_angular_damping"] = args.settle_angular_damping
            if args.settle_linear_damping is not None:
                config["data_collection"]["settle_linear_damping"] = args.settle_linear_damping
            if args.settle_sleep_threshold is not None:
                config["data_collection"]["settle_sleep_threshold"] = args.settle_sleep_threshold
            if args.lower_steps is not None:
                config["data_collection"]["lower_steps"] = args.lower_steps
            if args.lift_steps is not None:
                config["data_collection"]["lift_steps"] = args.lift_steps
            if args.goal_threshold is not None:
                config["data_collection"]["goal_threshold"] = args.goal_threshold
            config.setdefault("rigid_options", {})
            if args.rigid_iterations is not None:
                config["rigid_options"]["iterations"] = args.rigid_iterations
            if args.rigid_ls_iterations is not None:
                config["rigid_options"]["ls_iterations"] = args.rigid_ls_iterations
            if args.rigid_tolerance is not None:
                config["rigid_options"]["tolerance"] = args.rigid_tolerance
            if args.rigid_ls_tolerance is not None:
                config["rigid_options"]["ls_tolerance"] = args.rigid_ls_tolerance
            if args.box_box_detection:
                config["rigid_options"]["box_box_detection"] = True
            if args.use_contact_island:
                config["rigid_options"]["use_contact_island"] = True
            if args.use_hibernation:
                config["rigid_options"]["use_hibernation"] = True

            print(
                f"\n=== setting={setting}, n_particles={n_p}, "
                f"n_envs={args.n_envs}, samples_per_env={args.samples_per_env}, "
                f"backend={args.backend} ===",
                flush=True,
            )
            run_start = time.monotonic()
            sm = SandboxManipulation(config=config, n_envs=args.n_envs)
            try:
                build_start = time.monotonic()
                print("Building Genesis scene...", flush=True)
                sm.build()
                print(f"Build complete in {time.monotonic() - build_start:.1f}s", flush=True)
                sm.collect_data_samples(
                    n_samples=args.samples_per_env,
                    speed=args.speed,
                    path=f"{args.output_root}/{setting}",
                    settle_steps=args.settle_steps,
                    sweep_steps=args.sweep_steps,
                )
            finally:
                sm.destroy()
                print(f"Run finished in {time.monotonic() - run_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
