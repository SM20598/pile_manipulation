<<<<<<< HEAD
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
=======
import argparse
import pickle
from pathlib import Path
from pprint import pprint


DEFAULT_PKL = Path("data/test_old/basic/17_data.pkl")


def read_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def main():
        parser = argparse.ArgumentParser(description="Print the contents of a data pickle file.")
        parser.add_argument(
                "path",
                nargs="?",
                default=DEFAULT_PKL,
                help="Path to a .pkl file, relative to Genesis/ or absolute.",
        )
        args = parser.parse_args()

        base_dir = Path(__file__).parent
        path = Path(args.path)
        if not path.is_absolute():
                path = base_dir / path

        data = read_pickle(path)
        print(f"Loaded: {path}")
        print(f"Type: {type(data).__name__}")
        if hasattr(data, "__len__"):
                print(f"Length: {len(data)}")

        print(data[0]["state"])

if __name__ == "__main__":
    main()
>>>>>>> a31e3f7 (whatever)
