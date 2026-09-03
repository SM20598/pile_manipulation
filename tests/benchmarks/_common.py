"""Shared setup for the benchmark/probe scripts.

Kept separate so each script is a thin, readable wrapper around one question.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def add_paths(layered: bool = False):
    """Put Genesis/ (and optionally Genesis/layered/) on sys.path.

    The package uses flat sibling imports, matching upstream's convention, so
    scripts outside it have to add the directory rather than import a package.
    """
    if layered:
        sys.path.insert(0, str(ROOT / "Genesis" / "layered"))
    sys.path.insert(0, str(ROOT / "Genesis"))


def banner():
    """Print the backend actually in use.

    Not decoration: Genesis falls back to CPU silently, and a CPU run is
    otherwise indistinguishable from a GPU one in the output. Any timing
    quoted without this line is meaningless.
    """
    import torch
    import genesis as gs
    dev = getattr(gs, "device", None)
    print("=" * 72)
    print(f"  genesis {gs.__version__}   backend={gs.backend}   device={dev}")
    print(f"  torch {torch.__version__}   cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  !! NO GPU - these timings are CPU numbers and are not")
        print("     comparable with GPU figures.")
    print("=" * 72)


def make_sim(shape="cube", size=0.005, n=100, friction=0.4, density=750,
             n_envs=1, layered=False, n_layers=None, config_overrides=None):
    """Build a scene from the shipped config with the usual sweep parameters."""
    import yaml
    if layered:
        from sandbox_manipulation_layered import SandboxManipulation
        cfg_path = ROOT / "Genesis" / "layered" / "configs" / "basic_layered.yaml"
    else:
        from sandbox_manipulation import SandboxManipulation
        cfg_path = ROOT / "Genesis" / "configs" / "basic.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    cfg["material"].update(shape=shape, particle_size=size, n_particles=n,
                           density=density, friction=friction)
    if n_layers is not None:
        cfg["material"]["n_layers"] = n_layers
    cfg["box"]["friction"] = friction
    for k, v in (config_overrides or {}).items():
        cfg.setdefault(k, {}).update(v) if isinstance(v, dict) else cfg.update({k: v})
    sim = SandboxManipulation(config=cfg, n_envs=n_envs)
    sim.build()
    sim.set_material_properties(dict(
        particle_friction=friction, particle_density=density,
        box_friction=friction,
        sampled_particle_friction=None, sampled_particle_density=None))
    return sim
