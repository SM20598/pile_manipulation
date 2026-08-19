import torch
from PIL import Image
from pathlib import Path

import pathlib
path = pathlib.Path(__file__).parent.parent.resolve()

root = path / Path("data/dino_wm_preview_dino_wm/cube/n20/size0.012")
preview_dir = root / "png_previews"
preview_dir.mkdir(parents=True, exist_ok=True)

for obs_type in ("occupancy", "rendered"):
    states = torch.load(root / obs_type / "states.pth")
    for ep in range(states.shape[0]):
        frames = torch.load(root / obs_type / "obses" / f"episode_{ep:06d}.pth")
        for t in range(frames.shape[0]):
            img = Image.fromarray(frames[t].numpy()).resize((256, 256), Image.NEAREST)
            img.save(preview_dir / f"{obs_type}_ep{ep}_t{t}.png")