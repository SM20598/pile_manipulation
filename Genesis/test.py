import torch
from pathlib import Path

path = Path(__file__).parent

data = torch.load(path / "data/test/basic/0_data.pt")

print((data["states"].shape))