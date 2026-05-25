#############################################################
# OVERFIT A TINY GENESIS SUBSET TO DEBUG THE UNET TRAINING  #
#############################################################

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from Genesis.training.dataset import PileSweepData
from GranularDynamics2.myClasses.NFDUNetFilm import NFDUNetFiLM


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 500
BATCH_SIZE = 8
LR = 1e-3
N_SAMPLES = 8
DATA_FOLDERS = ["corl/cube/n20/size0.0085"]
LOG_DIR = Path("runs/overfit")


def estimate_pos_weight(dataset: Dataset) -> torch.Tensor:
    positives = 0.0
    total = 0.0
    for _, output in dataset:
        positives += output.sum().item()
        total += output.numel()

    negatives = total - positives
    if positives == 0:
        raise ValueError("Cannot use BCEWithLogitsLoss: overfit subset has no occupied pixels.")

    return torch.tensor([negatives / positives], dtype=torch.float32, device=DEVICE)


def evaluate(model, loader, criterion):
    model.eval()
    total_bce = 0.0
    total_mse = 0.0
    total_zero_mse = 0.0
    total_copy_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs_, outputs in loader:
            inputs, physics = inputs_
            inputs = inputs.to(DEVICE)
            physics = physics.to(DEVICE)
            outputs = outputs.to(DEVICE)

            logits = model(inputs, physics).squeeze(1).float()
            probs = torch.sigmoid(logits)
            batch_size = inputs.size(0)

            total_bce += criterion(logits, outputs).item() * batch_size
            total_mse += F.mse_loss(probs, outputs).item() * batch_size
            total_zero_mse += F.mse_loss(torch.zeros_like(outputs), outputs).item() * batch_size
            total_copy_mse += F.mse_loss(inputs[:, 0], outputs).item() * batch_size
            total_samples += batch_size

    return {
        "bce": total_bce / total_samples,
        "mse": total_mse / total_samples,
        "zero_mse": total_zero_mse / total_samples,
        "copy_mse": total_copy_mse / total_samples,
    }


if __name__ == "__main__":
    full_dataset = PileSweepData(DATA_FOLDERS, split=None)
    overfit_dataset: Dataset = Subset(full_dataset, indices=range(min(N_SAMPLES, len(full_dataset))))
    train_loader = DataLoader(
        overfit_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=DEVICE == "cuda",
    )

    model = NFDUNetFiLM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    pos_weight = estimate_pos_weight(overfit_dataset)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    writer = SummaryWriter(log_dir=LOG_DIR)

    print(f"Overfit samples: {len(overfit_dataset)}")
    print(f"Positive-pixel weight: {pos_weight.item():.3f}")

    with trange(EPOCHS, desc="Overfit Epochs") as tbar:
        for epoch in tbar:
            model.train()
            total_loss = 0.0
            train_size = 0

            for inputs_, outputs in train_loader:
                inputs, physics = inputs_
                inputs = inputs.to(DEVICE)
                physics = physics.to(DEVICE)
                outputs = outputs.to(DEVICE)

                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs, physics).squeeze(1).float()
                loss = criterion(logits, outputs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                train_size += inputs.size(0)

            avg_train_loss = total_loss / train_size
            metrics = evaluate(model, train_loader, criterion)

            writer.add_scalar("Loss/TrainBCE", avg_train_loss, epoch)
            writer.add_scalar("Metric/TrainProbMSE", metrics["mse"], epoch)
            writer.add_scalar("Baseline/ZeroMSE", metrics["zero_mse"], epoch)
            writer.add_scalar("Baseline/CopyInputMSE", metrics["copy_mse"], epoch)

            tbar.set_postfix(
                {
                    "BCE": metrics["bce"],
                    "MSE": metrics["mse"],
                    "Copy": metrics["copy_mse"],
                }
            )

            if epoch == 0 or (epoch + 1) % 25 == 0:
                print(
                    "Epoch",
                    epoch + 1,
                    f"BCE={metrics['bce']:.6f}",
                    f"MSE={metrics['mse']:.6f}",
                    f"ZeroMSE={metrics['zero_mse']:.6f}",
                    f"CopyMSE={metrics['copy_mse']:.6f}",
                )

    writer.close()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), LOG_DIR / "unet_overfit.pth")
