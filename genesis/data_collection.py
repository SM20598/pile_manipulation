################################
# A SCRIPT FOR DATA COLLECTION #
################################

# inputs
import genesis as gs
import numpy as np
import yaml
from pathlib import Path

from sandbox_manipulation import SandboxManipulation

##################################
# PARAMS THAT REQUIRE RESTARTING #
##################################
num_particles = [10, 20, 30]
settings = ["chickpeas_on_glass", "chickpeas_on_wood"]


def read_yaml(path : str):
    base_dir = Path(__file__).parent
    full_path = base_dir / path
    with open(full_path) as stream:
        return yaml.safe_load(stream)

for setting in settings:
    config = read_yaml(f"configs/{setting}.yaml")
    for n_p in num_particles:
        config['sandbox']['material']['properties']['n_particles'] = n_p
        
        sm = SandboxManipulation(config)
        sm.build()
        sm.collect_data_samples(n_samples=2000)
        sm.export_data_samples(path=f"data/cubes/{setting}")
        sm.destroy()