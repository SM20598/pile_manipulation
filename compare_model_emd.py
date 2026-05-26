import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Genesis.training.dataset import PileSweepData
from GranularDynamics2.myClasses.NFDUNetFilm import NFDUNetFiLM, NFDUNetFiLMShallow


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHANGE_THRESHOLD = 1e-3


def resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    best = path / "unet_best.pth"
    final = path / "unet.pth"
    if best.exists():
        return best
    if final.exists():
        return final
    raise FileNotFoundError(f"No unet_best.pth or unet.pth found in {path}")


def model_name(path: Path) -> str:
    return path.parent.name if path.is_file() else path.name


def load_model(checkpoint: Path, model_variant: str, input_mode: str):
    in_channels = 3 if input_mode in ("sweep-removed-input", "sweep-removed-residual") else 2
    residual_channel = 2 if input_mode == "sweep-removed-residual" else 0
    if model_variant in ("shallow", "shallow-lowres"):
        model = NFDUNetFiLMShallow(in_channels=in_channels, residual_channel=residual_channel).to(DEVICE)
    else:
        model = NFDUNetFiLM(in_channels=in_channels, residual_channel=residual_channel).to(DEVICE)
    try:
        state_dict = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def projection_cache(height: int, width: int, n_projections: int, device: str, resolution_scale: float):
    ys = torch.arange(height, dtype=torch.float32, device=device)
    xs = torch.arange(width, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1) / resolution_scale

    angles = torch.linspace(0.0, torch.pi, n_projections + 1, device=device)[:-1]
    directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    projections = coords @ directions.T
    sorted_proj, order = torch.sort(projections, dim=0)
    deltas = (sorted_proj[1:] - sorted_proj[:-1]).T
    return order.T.contiguous(), deltas.contiguous()


def sliced_wasserstein_batch(a, b, order, deltas, eps=1e-8):
    """Average 1D Wasserstein distances over fixed projection directions."""
    batch_size = a.shape[0]
    a = a.clamp_min(0.0).reshape(batch_size, -1)
    b = b.clamp_min(0.0).reshape(batch_size, -1)
    mass_a = a.sum(dim=1, keepdim=True)
    mass_b = b.sum(dim=1, keepdim=True)
    valid = (mass_a.squeeze(1) > eps) & (mass_b.squeeze(1) > eps)
    if not bool(valid.any().item()):
        return 0.0, 0

    a = a[valid] / mass_a[valid]
    b = b[valid] / mass_b[valid]
    distances = torch.zeros(a.shape[0], device=a.device)

    for proj_idx in range(order.shape[0]):
        idx = order[proj_idx]
        a_sorted = a[:, idx]
        b_sorted = b[:, idx]
        cdf_diff = torch.cumsum(a_sorted - b_sorted, dim=1).abs()
        distances += (cdf_diff[:, :-1] * deltas[proj_idx].unsqueeze(0)).sum(dim=1)

    distances /= order.shape[0]
    return distances.sum().item(), int(valid.sum().item())


def empty_totals():
    return {
        "samples": 0,
        "mse": 0.0,
        "copy_mse": 0.0,
        "changed_sse": 0.0,
        "changed_copy_sse": 0.0,
        "changed_pixels": 0,
        "hard_iou": 0.0,
        "hard_dice": 0.0,
        "mass_loss": 0.0,
        "swd": 0.0,
        "swd_count": 0,
        "changed_swd": 0.0,
        "changed_swd_count": 0,
        "added_swd": 0.0,
        "added_swd_count": 0,
    }


def update_standard_metrics(totals, probs, outputs, current_state):
    batch_size = probs.shape[0]
    pred_mask = probs > 0.5
    target_mask = outputs > 0.5
    changed_mask = (outputs - current_state).abs() > CHANGE_THRESHOLD
    intersection = (pred_mask & target_mask).sum(dim=(1, 2)).float()
    pred_area = pred_mask.sum(dim=(1, 2)).float()
    target_area = target_mask.sum(dim=(1, 2)).float()
    union = pred_area + target_area - intersection
    changed_count = changed_mask.sum().item()

    totals["samples"] += batch_size
    totals["mse"] += F.mse_loss(probs, outputs).item() * batch_size
    totals["copy_mse"] += F.mse_loss(current_state, outputs).item() * batch_size
    totals["hard_iou"] += ((intersection + 1e-6) / (union + 1e-6)).sum().item()
    totals["hard_dice"] += ((2.0 * intersection + 1e-6) / (pred_area + target_area + 1e-6)).sum().item()
    totals["mass_loss"] += (
        (probs.sum(dim=(1, 2)) - outputs.sum(dim=(1, 2))).abs().mean().item()
        / outputs[0].numel()
        * batch_size
    )
    if changed_count > 0:
        totals["changed_sse"] += ((probs - outputs).pow(2) * changed_mask).sum().item()
        totals["changed_copy_sse"] += ((current_state - outputs).pow(2) * changed_mask).sum().item()
        totals["changed_pixels"] += changed_count

    return changed_mask


def evaluate_checkpoint(model, loader, n_projections: int, resolution_scale: float, max_batches: int | None = None):
    totals = empty_totals()
    order = None
    deltas = None

    with torch.no_grad():
        for batch_idx, (inputs_, outputs) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs, physics = inputs_
            inputs = inputs.to(DEVICE)
            physics = physics.to(DEVICE)
            outputs = outputs.to(DEVICE)
            current_state = inputs[:, 0]

            probs = model(inputs, physics).squeeze(1).float().clamp(0.0, 1.0)

            if order is None:
                order, deltas = projection_cache(
                    outputs.shape[-2],
                    outputs.shape[-1],
                    n_projections,
                    DEVICE,
                    resolution_scale,
                )

            changed_mask = update_standard_metrics(totals, probs, outputs, current_state)

            value, count = sliced_wasserstein_batch(probs, outputs, order, deltas)
            totals["swd"] += value
            totals["swd_count"] += count

            value, count = sliced_wasserstein_batch(probs * changed_mask, outputs * changed_mask, order, deltas)
            totals["changed_swd"] += value
            totals["changed_swd_count"] += count

            pred_added = (probs - current_state).clamp_min(0.0)
            target_added = (outputs - current_state).clamp_min(0.0)
            value, count = sliced_wasserstein_batch(pred_added, target_added, order, deltas)
            totals["added_swd"] += value
            totals["added_swd_count"] += count

    samples = totals["samples"]
    changed_pixels = totals["changed_pixels"]
    return {
        "mse": totals["mse"] / samples,
        "copy_mse": totals["copy_mse"] / samples,
        "changed_mse": totals["changed_sse"] / changed_pixels if changed_pixels else 0.0,
        "changed_copy_mse": totals["changed_copy_sse"] / changed_pixels if changed_pixels else 0.0,
        "iou": totals["hard_iou"] / samples,
        "dice": totals["hard_dice"] / samples,
        "mass_loss": totals["mass_loss"] / samples,
        "swd": totals["swd"] / totals["swd_count"] if totals["swd_count"] else 0.0,
        "changed_swd": totals["changed_swd"] / totals["changed_swd_count"] if totals["changed_swd_count"] else 0.0,
        "added_swd": totals["added_swd"] / totals["added_swd_count"] if totals["added_swd_count"] else 0.0,
    }


def print_results(results):
    headers = [
        "model",
        "mse",
        "changed_mse",
        "iou",
        "dice",
        "mass",
        "sliced_emd",
        "changed_sliced_emd",
        "added_sliced_emd",
    ]
    print("\t".join(headers))
    for name, metrics in results:
        print(
            "\t".join(
                [
                    name,
                    f"{metrics['mse']:.6f}",
                    f"{metrics['changed_mse']:.6f}",
                    f"{metrics['iou']:.4f}",
                    f"{metrics['dice']:.4f}",
                    f"{metrics['mass_loss']:.6f}",
                    f"{metrics['swd']:.4f}",
                    f"{metrics['changed_swd']:.4f}",
                    f"{metrics['added_swd']:.4f}",
                ]
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Compare trained occupancy models with sliced Wasserstein metrics.")
    parser.add_argument("models", nargs="+", type=Path, help="Run directories or checkpoint files.")
    parser.add_argument("--data-folders", nargs="+", default=["corl/cube"])
    parser.add_argument(
        "--model-variant",
        choices=["full", "shallow", "lowres", "shallow-lowres"],
        default="full",
    )
    parser.add_argument(
        "--resolution-scale",
        type=float,
        default=None,
        help="Defaults to 0.25 for lowres variants and 1.0 otherwise.",
    )
    parser.add_argument(
        "--input-mode",
        choices=["standard", "sweep-removed-input", "sweep-removed-residual"],
        default="standard",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-projections", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None, help="Optional quick-check limit.")
    return parser.parse_args()


def main():
    args = parse_args()
    resolution_scale = (
        args.resolution_scale
        if args.resolution_scale is not None
        else (0.25 if args.model_variant in ("lowres", "shallow-lowres") else 1.0)
    )
    dataset = PileSweepData(
        args.data_folders,
        split="test",
        resolution_scale=resolution_scale,
        include_sweep_removed=args.input_mode in ("sweep-removed-input", "sweep-removed-residual"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=DEVICE == "cuda",
    )

    results = []
    for model_path in args.models:
        checkpoint = resolve_checkpoint(model_path)
        print(f"Evaluating {model_name(model_path)} from {checkpoint}", flush=True)
        model = load_model(checkpoint, args.model_variant, args.input_mode)
        metrics = evaluate_checkpoint(model, loader, args.n_projections, resolution_scale, args.max_batches)
        results.append((model_name(model_path), metrics))

    print_results(results)


if __name__ == "__main__":
    main()
