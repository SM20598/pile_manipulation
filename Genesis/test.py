import torch
from pathlib import Path
from scipy.spatial.transform import Rotation

path = Path(__file__).parent

data = torch.load(path / "data/test/cube/n40/size0.0085/_30_data.pt")

for state, state_ in zip(data["states"], data["states_"]):
    for particle, particle_ in zip(state, state_):
        if not torch.equal(particle, particle_):
            print("NOT EQUAL")
            print("Positions:", particle[:3])
            print("Quaternions:", particle[3:])
            rot = Rotation.from_quat(particle[3:], scalar_first = True)
            print("Euler angles", rot.as_euler("xyz", degrees=True))
            print("\n")