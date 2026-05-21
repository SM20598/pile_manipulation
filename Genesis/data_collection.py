################################
# A SCRIPT FOR DATA COLLECTION #
################################

import argparse
import copy
import itertools
import os
import yaml
import numpy as np
from pathlib import Path

from sandbox_manipulation import SandboxManipulation

##################################
# PARAMS THAT REQUIRE RESTARTING #
##################################
DEFAULT_SETTINGS = ["basic_example"]
depth = 5

# Requires rebuilding the scene
SHAPES = ["cube"]
DEFAULT_NUM_PARTICLES = [10]
CUBE_PARTICLE_SIZES = np.linspace(0.005, 0.012, depth)
SPHERE_PARTICLE_SIZES = np.linspace(0.005, 0.012, depth)
RECTANGLE_PARTICLE_SIZES = np.linspace(0.005, 0.012, depth)
CYLINDER_PARTICLE_SIZES = np.linspace(0.005, 0.012, depth)

# Can be changed without rebuilding the scene
PARTICLE_FRICTIONS = np.linspace(0.03, 0.5, depth)
PARTICLE_DENSITIES = np.linspace(750, 5000, depth)
TABLE_FRICTION = np.linspace(0.02, 0.5, 4)


PARTICLE_SIZES_BY_SHAPE = {
    "cube": CUBE_PARTICLE_SIZES,
    "sphere": SPHERE_PARTICLE_SIZES,
    "rectangle": RECTANGLE_PARTICLE_SIZES,
    "cylinder": CYLINDER_PARTICLE_SIZES,
}


def property_value_or_range(value: float, rng: np.random.Generator, min_value: float):
    """50/50 scalar-or-range switch around a sweep value."""
    value = float(value)
    if rng.random() < 0.5:
        return max(value, min_value), "constant"
    return [max(value * 0.8, min_value), max(value * 1.2, min_value)], "range_pm20"


def config_shape_name(shape: str):
    return "rectangular_cube" if shape == "rectangle" else shape


def scale_property_value(value, scale: float):
    if isinstance(value, (int, float)):
        return float(value) * scale
    return [float(v) * scale for v in value]


def geometry_batch_dir_name(shape: str, n_particles: int, particle_size):
    if isinstance(particle_size, (int, float)):
        size_label = f"{float(particle_size):.5f}"
    else:
        size_label = "_".join(f"{float(value):.5f}" for value in particle_size)
    size_label = size_label.replace(".", "p")
    return f"{shape}/n{n_particles}/size{size_label}"


def set_particle_size_config(material_properties: dict, shape: str, particle_size):
    material_properties["particle_size"] = particle_size
    if shape == "rectangle":
        material_properties["particle_length"] = particle_size
        material_properties["particle_width"] = scale_property_value(particle_size, 0.6)
        material_properties["particle_height"] = scale_property_value(particle_size, 0.6)
    elif shape == "cylinder":
        material_properties["particle_height"] = particle_size
        material_properties["particle_radius"] = scale_property_value(particle_size, 0.5)


def build_property_env_settings(shape, particle_size, particle_size_mode, rng):
    env_settings = []
    for particle_friction, particle_density, table_friction in itertools.product(
        PARTICLE_FRICTIONS,
        PARTICLE_DENSITIES,
        TABLE_FRICTION,
    ):
        particle_friction_cfg, particle_friction_mode = property_value_or_range(
            particle_friction, rng, min_value=1e-2
        )
        particle_density_cfg, particle_density_mode = property_value_or_range(
            particle_density, rng, min_value=1e-6
        )
        env_settings.append(
            {
                "particle_friction": particle_friction_cfg,
                "particle_density": particle_density_cfg,
                "table_friction": float(table_friction),
                "metadata": {
                    "property_env_idx": len(env_settings),
                    "particle_shape": config_shape_name(shape),
                    "particle_size_center": float(particle_size),
                    "particle_size_mode": particle_size_mode,
                    "particle_friction_center": float(particle_friction),
                    "particle_friction_mode": particle_friction_mode,
                    "particle_density_center": float(particle_density),
                    "particle_density_mode": particle_density_mode,
                    "table_friction_center": float(table_friction),
                },
            }
        )
    return env_settings


def parse_n_envs(value: str | None):
    if value is None:
        return None
    value = value.strip().lower()
    if value in ("none", "all"):
        return None
    n_envs = int(value)
    if n_envs <= 0:
        raise argparse.ArgumentTypeError("--n-envs must be positive, 'none', or 'all'")
    return n_envs


def configure_viewer_for_host(args, config):
    viewer_options = config["simulation"].setdefault("viewer_options", {})
    show_viewer = args.show_viewer
    update_visualizer = args.update_visualizer
    if show_viewer and os.name == "posix":
        try:
            import pyglet

            pyglet.display.get_display()
        except Exception as exc:
            print(
                f"Could not connect to a display ({exc}); disabling Genesis viewer and visualizer updates.",
                flush=True,
            )
            show_viewer = False
            update_visualizer = False
    viewer_options["show_viewer"] = show_viewer
    if args.viewer_type is not None:
        viewer_options["viewer_type"] = args.viewer_type
    return update_visualizer


def read_yaml(path: str):
    base_dir = Path(__file__).parent
    full_path = base_dir / path
    with open(full_path) as stream:
        return yaml.safe_load(stream)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect sandbox manipulation data.")
    parser.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS)
    parser.add_argument("--num-particles", nargs="+", type=int, default=DEFAULT_NUM_PARTICLES)
    parser.add_argument(
        "--n-envs",
        type=parse_n_envs,
        default=10,
        help=(
            "Parallel environments per material batch. "
            "Use 'none' or 'all' to use the default batch size of 64."
        ),
    )
    parser.add_argument("--samples-per-env", type=int, default=1000)
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--speed", type=float, default=0.125)
    parser.add_argument("--output-root", default="data/test")
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
    parser.add_argument("--particle-shape", choices=["config", "cube", "sphere", "rectangle", "cylinder"], default="config")
    parser.add_argument("--particle-shapes", nargs="+", choices=SHAPES, default=SHAPES)
    parser.add_argument("--property-seed", type=int, default=None)
    parser.add_argument("--settle-angular-damping", type=float, default=None)
    parser.add_argument("--settle-linear-damping", type=float, default=None)
    parser.add_argument("--settle-sleep-threshold", type=float, default=None)
    parser.add_argument("--disable-settle-stabilization", action="store_true")
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--viewer-type", choices=["observer", "bird", "leveled"], default=None)
    parser.add_argument("--update-visualizer", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument(
        "--shared-env-values",
        action="store_true",
        help=(
            "Mirror old_data_collection.py: build one scene from the config, "
            "do not run the property sweep, and give every environment the same material values."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.property_seed)

    for setting in args.settings:
        base_config = read_yaml(f"configs/{setting}.yaml")
        if args.shared_env_values:
            for n_p in args.num_particles:
                config = copy.deepcopy(base_config)
                config.setdefault("simulation", {})
                config["simulation"]["backend"] = args.backend
                config["simulation"]["performance_mode"] = True
                if args.substeps is not None:
                    config["simulation"]["substeps"] = args.substeps

                update_visualizer = configure_viewer_for_host(args, config)

                config.setdefault("sandbox", {}).setdefault("material", {}).setdefault("properties", {})
                material_properties = config["sandbox"]["material"]["properties"]
                material_properties["n_particles"] = n_p
                if args.particle_shape != "config":
                    material_properties["shape"] = config_shape_name(args.particle_shape)
                    material_properties["cubes"] = args.particle_shape == "cube"

                config.setdefault("data_collection", {})
                config["data_collection"]["progress"] = not args.quiet_progress
                config["data_collection"]["sample_progress_interval"] = args.progress_interval
                config["data_collection"]["phase_progress_interval"] = args.phase_progress_interval
                config["data_collection"]["trace_scene_steps"] = args.trace_scene_steps
                config["data_collection"]["update_visualizer"] = update_visualizer
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

                n_envs = 64 if args.n_envs is None else args.n_envs
                print(
                    f"\n=== shared-env-values setting={setting}, n_particles={n_p}, "
                    f"n_envs={n_envs}, samples_per_env={args.samples_per_env}, "
                    f"backend={args.backend} ===",
                    flush=True,
                )
                sm = SandboxManipulation(config=config, n_envs=n_envs)
                try:
                    print("Building Genesis scene...", flush=True)
                    sm.build()
                    print("Build complete", flush=True)
                    sm.collect_data_samples(
                        n_samples=args.samples_per_env,
                        speed=args.speed,
                        path=f"{args.output_root}/{setting}",
                        settle_steps=args.settle_steps,
                        sweep_steps=args.sweep_steps,
                    )
                finally:
                    sm.destroy()
                    print("Run finished", flush=True)
            continue

        shapes = args.particle_shapes if args.particle_shape == "config" else [args.particle_shape]

        i = 0
        i_total = len(shapes) * len(args.num_particles) * max(len(PARTICLE_SIZES_BY_SHAPE[s]) for s in shapes)
        for shape in shapes:
            for n_p in args.num_particles:
                for particle_size in PARTICLE_SIZES_BY_SHAPE[shape]:

                    print(f'\n++++++++ Collecting batch {i+1}/{i_total} ++++++++')
                    print(f' > shape="{shape}", n_particles={n_p}, particle_size={particle_size} <\n', flush=True)
                    particle_size_cfg, particle_size_mode = property_value_or_range(
                        particle_size, rng, min_value=1e-6
                    )

                    config = copy.deepcopy(base_config)
                    config.setdefault("simulation", {})
                    config["simulation"]["backend"] = args.backend
                    config["simulation"]["performance_mode"] = True
                    if args.substeps is not None:
                        config["simulation"]["substeps"] = args.substeps

                    update_visualizer = configure_viewer_for_host(args, config)

                    config.setdefault("sandbox", {}).setdefault("material", {}).setdefault("properties", {})
                    material_properties = config["sandbox"]["material"]["properties"]
                    material_properties["n_particles"] = n_p
                    material_properties["shape"] = config_shape_name(shape)
                    set_particle_size_config(material_properties, shape, particle_size_cfg)

                    config.setdefault("data_collection", {})
                    config["data_collection"]["progress"] = not args.quiet_progress
                    config["data_collection"]["sample_progress_interval"] = args.progress_interval
                    config["data_collection"]["phase_progress_interval"] = args.phase_progress_interval
                    config["data_collection"]["trace_scene_steps"] = args.trace_scene_steps
                    config["data_collection"]["update_visualizer"] = update_visualizer
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
                    env_settings = build_property_env_settings(
                        shape, particle_size, particle_size_mode, rng
                    )
                    n_envs = 64 if args.n_envs is None else args.n_envs
                    if n_envs <= 0:
                        raise ValueError("--n-envs must be positive when provided")

                    print(
                        f"\n=== building setting={setting}, shape={shape}, n_particles={n_p}, "
                        f"particle_size={particle_size_cfg}, n_envs={n_envs}, "
                        f"material_batches={len(env_settings)}, backend={args.backend} ===",
                        flush=True,
                    )
                    sm = SandboxManipulation(config=config, n_envs=n_envs)
                    try:
                        print("Building Genesis scene...", flush=True)
                        sm.build()
                        print("Build complete", flush=True)

                        for property_idx, property_setting in enumerate(env_settings):
                            print(
                                f"\n--- collecting shape={shape}, n_particles={n_p}, "
                                f"particle_size={particle_size_cfg}, "
                                f"material_batch={property_idx + 1}/{len(env_settings)}, "
                                f"samples_per_env={args.samples_per_env} ---",
                                flush=True,
                            )
                            sm.set_material_properties(property_setting)
                            sm.shuffle_particles()
                            sm.collect_data_samples(
                                n_samples=args.samples_per_env,
                                speed=args.speed,
                                path=(
                                    f"{args.output_root}/"
                                    f"{geometry_batch_dir_name(shape, n_p, particle_size_cfg)}"
                                ),
                                settle_steps=args.settle_steps,
                                sweep_steps=args.sweep_steps,
                                env_index_offset=property_idx * n_envs,
                            )
                    finally:
                        sm.destroy()
                        print("Geometry run finished", flush=True)
                    i += 1


if __name__ == "__main__":
    main()
