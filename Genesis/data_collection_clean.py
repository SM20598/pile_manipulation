################################
# A SCRIPT FOR DATA COLLECTION #
################################

import argparse
import itertools
import yaml
import numpy as np
from pathlib import Path

from sandbox_manipulation_clean import SandboxManipulation

##################################
# PARAMS THAT REQUIRE RESTARTING #
##################################
BASIC_SETTING = "basic"
DEFAULT_SHAPES = ["cube"]
DEFAULT_NUM_PARTICLES = [10, 20]
PARTICLE_SIZES = np.linspace(0.005, 0.012, 5).tolist()

PARTICLE_FRICTIONS = np.linspace(0.05, 0.5, 5).tolist()
PARTICLE_DENSITIES = np.linspace(750, 5000, 5).tolist()
BOX_FRICTION = np.linspace(0.05, 0.5, 4).tolist()
PER_PARTICLE_VALUE_PROBABILITY = 0.5


def scalar_or_particle_values(value: float, n_particles: int, rng: np.random.Generator):
    if rng.random() >= PER_PARTICLE_VALUE_PROBABILITY:
        return float(value), None
    return float(value), rng.uniform(0.8 * value, 1.2 * value, n_particles).tolist()


def build_particle_size_settings(n_particles: int, rng: np.random.Generator):
    settings = []
    for size in PARTICLE_SIZES:
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
    parser.add_argument("--output-root", default="data/corl")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--viewer-type", choices=["observer", "bird", "leveled"], default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng()

    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    
    if args.samples_per_env <= 0:
        raise ValueError("--samples-per-env must be positive")

    config = read_yaml(f"configs/{BASIC_SETTING}.yaml")

    # Iterate shapes
    shapes = [args.particle_shape] if args.particle_shape != DEFAULT_SHAPES else DEFAULT_SHAPES
    for shape in shapes:
        config["material"]["shape"] = shape

        # Iterate number of particles
        for n_p in args.num_particles:
            config["material"]["n_particles"] = n_p
            env_settings = build_property_env_settings(n_p, rng)
            
            # Iterate particle sizes
            sizes = (
                [{"base": args.particle_sizes[0], "sampled": args.particle_sizes}]
                if args.particle_sizes != PARTICLE_SIZES
                else build_particle_size_settings(n_p, rng)
            )
            if n_p == 40: 
                sizes = sizes[2:] # TEMPORARY BECAUSE FIRST TWO SIZE HAVE BEEN COLLECTED FOR 40
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

                for property_idx, property_setting in enumerate(env_settings):
                    print(f"\n--- material batch {property_idx + 1}/{len(env_settings)}", flush=True)
                    
                    sm.set_material_properties(property_setting)
                    sm.shuffle_particles()
                    sm.collect_data_samples(
                        n_samples=args.samples_per_env,
                        path=(
                            f"{args.output_root}/"
                            f"{shape}/n{n_p}/size{size_setting['base']}"
                        ),
                    )

                sm.destroy()


if __name__ == "__main__":
    main()
