#############################################################
# THIS SCRIPT IS USED TO TRAIN A UNET WITH THE GENESIS DATA #
#############################################################

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from Genesis.training.dataset import PileSweepData
from GranularDynamics2.myClasses.NFDUNetFilm import NFDUNetFiLM


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-4
POS_WEIGHT = 20.0
DICE_WEIGHT = 1
SHARPNESS_WEIGHT = 0.05
TV_WEIGHT = 0.01
MASS_WEIGHT = 0.1
PATIENCE = 10
CHANGE_THRESHOLD = 1e-3
DEFAULT_DATA_FOLDERS = ["corl"]
DEFAULT_LOG_DIR = Path("runs/nfdunetfilm_bce_dice_tunedmultiloss")


def parse_args():
   parser = argparse.ArgumentParser(description="Train or evaluate the Genesis UNet model.")
   parser.add_argument("--eval-only", action="store_true", help="Load a checkpoint and evaluate without training.")
   parser.add_argument(
      "--checkpoint",
      type=Path,
      default=None,
      help="Checkpoint to load for --eval-only. Defaults to <log-dir>/unet.pth.",
   )
   parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
   parser.add_argument("--data-folders", nargs="+", default=DEFAULT_DATA_FOLDERS)
   parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
   parser.add_argument("--num-workers", type=int, default=4)
   parser.add_argument("--sharpness-weight", type=float, default=SHARPNESS_WEIGHT)
   parser.add_argument("--tv-weight", type=float, default=TV_WEIGHT)
   parser.add_argument("--mass-weight", type=float, default=MASS_WEIGHT)
   return parser.parse_args()


def augment_batch(inputs, outputs, physics):
   inputs_rot = [torch.rot90(inputs, k, dims=(-2, -1)) for k in range(4)]
   inputs_mir = [torch.flip(r, dims=[-1]) for r in inputs_rot]
   inputs = torch.cat(inputs_rot + inputs_mir, dim=0)

   outputs_rot = [torch.rot90(outputs, k, dims=(-2, -1)) for k in range(4)]
   outputs_mir = [torch.flip(r, dims=[-1]) for r in outputs_rot]
   outputs = torch.cat(outputs_rot + outputs_mir, dim=0)

   physics = physics.repeat(8, 1)
   return inputs, outputs, physics


def soft_dice_loss(logits, targets, eps=1e-6):
   probs = torch.sigmoid(logits)
   dims = tuple(range(1, probs.ndim))
   intersection = (probs * targets).sum(dim=dims)
   denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
   dice = (2.0 * intersection + eps) / (denominator + eps)
   return 1.0 - dice.mean()


def combined_loss(logits, outputs, criterion):
   probs = torch.sigmoid(logits)
   bce = criterion(logits, outputs)
   dice = soft_dice_loss(logits, outputs)
   sharpness = (probs * (1.0 - probs)).mean()
   tv_h = (probs[:, 1:, :] - probs[:, :-1, :]).abs().mean()
   tv_w = (probs[:, :, 1:] - probs[:, :, :-1]).abs().mean()
   tv = tv_h + tv_w
   mass = (probs.sum(dim=(1, 2)) - outputs.sum(dim=(1, 2))).abs().mean() / outputs[0].numel()
   loss = (
      bce
      + DICE_WEIGHT * dice
      + SHARPNESS_WEIGHT * sharpness
      + TV_WEIGHT * tv
      + MASS_WEIGHT * mass
   )
   return loss, bce, dice, sharpness, tv, mass


def update_metric_totals(totals, logits, outputs, inputs, loss, bce_loss, dice_loss, sharpness_loss, tv_loss, mass_loss):
   probs = torch.sigmoid(logits)
   current_state = inputs[:, 0]
   pred_mask = probs > 0.5
   target_mask = outputs > 0.5
   changed_mask = (outputs - current_state).abs() > CHANGE_THRESHOLD
   batch_size = inputs.size(0)
   intersection = (pred_mask & target_mask).sum(dim=(1, 2)).float()
   pred_area = pred_mask.sum(dim=(1, 2)).float()
   target_area = target_mask.sum(dim=(1, 2)).float()
   union = pred_area + target_area - intersection
   changed_count = changed_mask.sum().item()

   totals["loss"] += loss.item() * batch_size
   totals["bce"] += bce_loss.item() * batch_size
   totals["dice_loss"] += dice_loss.item() * batch_size
   totals["sharpness_loss"] += sharpness_loss.item() * batch_size
   totals["tv_loss"] += tv_loss.item() * batch_size
   totals["mass_loss"] += mass_loss.item() * batch_size
   totals["prob_mse"] += F.mse_loss(probs, outputs).item() * batch_size
   totals["zero_mse"] += F.mse_loss(torch.zeros_like(outputs), outputs).item() * batch_size
   totals["copy_mse"] += F.mse_loss(current_state, outputs).item() * batch_size
   if changed_count > 0:
      totals["changed_prob_sse"] += ((probs - outputs).pow(2) * changed_mask).sum().item()
      totals["changed_zero_sse"] += outputs.pow(2).mul(changed_mask).sum().item()
      totals["changed_copy_sse"] += ((current_state - outputs).pow(2) * changed_mask).sum().item()
      totals["changed_pixels"] += changed_count
   totals["total_pixels"] += changed_mask.numel()
   totals["hard_iou"] += ((intersection + 1e-6) / (union + 1e-6)).sum().item()
   totals["hard_dice"] += ((2.0 * intersection + 1e-6) / (pred_area + target_area + 1e-6)).sum().item()
   totals["size"] += batch_size


def average_metrics(totals):
   metrics = {
      key: value / totals["size"]
      for key, value in totals.items()
      if key not in ("size", "changed_prob_sse", "changed_zero_sse", "changed_copy_sse", "changed_pixels", "total_pixels")
   }
   changed_pixels = totals["changed_pixels"]
   metrics["changed_mse"] = totals["changed_prob_sse"] / changed_pixels if changed_pixels else 0.0
   metrics["changed_zero_mse"] = totals["changed_zero_sse"] / changed_pixels if changed_pixels else 0.0
   metrics["changed_copy_mse"] = totals["changed_copy_sse"] / changed_pixels if changed_pixels else 0.0
   metrics["changed_pixel_frac"] = totals["changed_pixels"] / totals["total_pixels"] if totals["total_pixels"] else 0.0
   return metrics


def empty_totals():
   return {
      "loss": 0.0,
      "bce": 0.0,
      "dice_loss": 0.0,
      "sharpness_loss": 0.0,
      "tv_loss": 0.0,
      "mass_loss": 0.0,
      "prob_mse": 0.0,
      "zero_mse": 0.0,
      "copy_mse": 0.0,
      "changed_prob_sse": 0.0,
      "changed_zero_sse": 0.0,
      "changed_copy_sse": 0.0,
      "changed_pixels": 0,
      "total_pixels": 0,
      "hard_iou": 0.0,
      "hard_dice": 0.0,
      "size": 0,
   }


def evaluate_model(model, loader, criterion):
   model.eval()
   totals = empty_totals()
   with torch.no_grad():
      for inputs_, outputs in loader:
         inputs, physics = inputs_
         inputs = inputs.to(DEVICE)
         physics = physics.to(DEVICE)
         outputs = outputs.to(DEVICE)

         logits = model(inputs, physics).squeeze(1).float()
         loss, bce_loss, dice_loss, sharpness_loss, tv_loss, mass_loss = combined_loss(logits, outputs, criterion)
         update_metric_totals(
            totals,
            logits,
            outputs,
            inputs,
            loss,
            bce_loss,
            dice_loss,
            sharpness_loss,
            tv_loss,
            mass_loss,
         )

   return average_metrics(totals)


def print_test_metrics(test_metrics):
   print(
      f"Test Loss: {test_metrics['loss']:.6f}, "
      f"Test BCE: {test_metrics['bce']:.6f}, "
      f"Test DiceLoss: {test_metrics['dice_loss']:.6f}, "
      f"Test SharpnessLoss: {test_metrics['sharpness_loss']:.6f}, "
      f"Test TVLoss: {test_metrics['tv_loss']:.6f}, "
      f"Test MassLoss: {test_metrics['mass_loss']:.6f}, "
      f"Test MSE: {test_metrics['prob_mse']:.6f}, "
      f"Test IoU: {test_metrics['hard_iou']:.4f}, "
      f"Test Dice: {test_metrics['hard_dice']:.4f}, "
      f"Zero MSE: {test_metrics['zero_mse']:.6f}, "
      f"Copy MSE: {test_metrics['copy_mse']:.6f}, "
      f"Changed Pixel Frac: {test_metrics['changed_pixel_frac']:.6f}, "
      f"Changed MSE: {test_metrics['changed_mse']:.6f}, "
      f"Changed Zero MSE: {test_metrics['changed_zero_mse']:.6f}, "
      f"Changed Copy MSE: {test_metrics['changed_copy_mse']:.6f}"
   )


if __name__ == "__main__":
   args = parse_args()
   SHARPNESS_WEIGHT = args.sharpness_weight
   TV_WEIGHT = args.tv_weight
   MASS_WEIGHT = args.mass_weight
   data_folders = args.data_folders
   log_dir = args.log_dir
   data_aug = True
   log_dir.mkdir(parents=True, exist_ok=True)

   test_dataset: Dataset = PileSweepData(data_folders, split="test")

   model = NFDUNetFiLM().to(DEVICE)

   pos_weight = torch.tensor([POS_WEIGHT], device=DEVICE)
   criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

   test_loader = DataLoader(
      test_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=DEVICE == "cuda",
   )

   if args.eval_only:
      checkpoint = args.checkpoint if args.checkpoint is not None else log_dir / "unet.pth"
      state_dict = torch.load(checkpoint, map_location=DEVICE)
      model.load_state_dict(state_dict)
      print(f"Evaluating checkpoint: {checkpoint}")
      test_metrics = evaluate_model(model, test_loader, criterion)
      print_test_metrics(test_metrics)
      raise SystemExit(0)

   train_dataset: Dataset = PileSweepData(data_folders, split="train")
   val_dataset: Dataset = PileSweepData(data_folders, split="val")
   optimizer = torch.optim.Adam(model.parameters(), lr=LR)
   writer = SummaryWriter(log_dir=log_dir)
   scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

   batch_size = args.batch_size // 8 if data_aug else args.batch_size
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=DEVICE == "cuda")
   val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=DEVICE == "cuda")
   test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=DEVICE == "cuda")

   best_val_loss = float("inf")
   best_epoch = 0
   epochs_without_improvement = 0
   val_loss_history = []
   IMPROVEMENT_WINDOW = 5  # epochs over which to measure improvement rate
   with trange(EPOCHS, desc="Training Epochs") as tbar:
      for epoch in tbar:
         model.train()
         train_totals = empty_totals()

         for inputs_, outputs in train_loader:
            inputs, physics = inputs_
            inputs = inputs.to(DEVICE)
            physics = physics.to(DEVICE)
            outputs = outputs.to(DEVICE)

            if data_aug:
               inputs, outputs, physics = augment_batch(inputs, outputs, physics)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=DEVICE, dtype=torch.bfloat16):
               logits = model(inputs, physics).squeeze(1)
               loss, bce_loss, dice_loss, sharpness_loss, tv_loss, mass_loss = combined_loss(logits.float(), outputs, criterion)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            update_metric_totals(
               train_totals,
               logits.detach().float(),
               outputs,
               inputs,
               loss.detach(),
               bce_loss.detach(),
               dice_loss.detach(),
               sharpness_loss.detach(),
               tv_loss.detach(),
               mass_loss.detach(),
            )

         train_metrics = average_metrics(train_totals)

         model.eval()
         val_totals = empty_totals()
         with torch.no_grad():
            for inputs_, outputs in val_loader:
               inputs, physics = inputs_
               inputs = inputs.to(DEVICE)
               physics = physics.to(DEVICE)
               outputs = outputs.to(DEVICE)

               logits = model(inputs, physics).squeeze(1).float()
               loss, bce_loss, dice_loss, sharpness_loss, tv_loss, mass_loss = combined_loss(logits, outputs, criterion)
               update_metric_totals(
                  val_totals,
                  logits,
                  outputs,
                  inputs,
                  loss,
                  bce_loss,
                  dice_loss,
                  sharpness_loss,
                  tv_loss,
                  mass_loss,
               )

         val_metrics = average_metrics(val_totals)
         avg_val_loss = val_metrics["loss"]
         print(
            f"Epoch {epoch + 1}: "
            f"train loss={train_metrics['loss']:.6f}, train MSE={train_metrics['prob_mse']:.6f}, "
            f"val loss={val_metrics['loss']:.6f}, val MSE={val_metrics['prob_mse']:.6f}, "
            f"val IoU={val_metrics['hard_iou']:.4f}, val Dice={val_metrics['hard_dice']:.4f}, "
            f"val copy MSE={val_metrics['copy_mse']:.6f}, "
            f"val changed MSE={val_metrics['changed_mse']:.6f}, "
            f"val changed copy MSE={val_metrics['changed_copy_mse']:.6f}"
         )

         val_loss_history.append(avg_val_loss)
         if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), log_dir / "unet_best.pth")
         else:
            epochs_without_improvement += 1

         # improvement rate over the last IMPROVEMENT_WINDOW epochs (positive = still improving)
         if len(val_loss_history) >= IMPROVEMENT_WINDOW + 1:
            improvement_rate = (val_loss_history[-IMPROVEMENT_WINDOW - 1] - val_loss_history[-1]) / IMPROVEMENT_WINDOW
         else:
            improvement_rate = float("nan")
         train_val_gap = train_metrics["loss"] - val_metrics["loss"]

         writer.add_scalar("Loss/TrainCombined", train_metrics["loss"], epoch)
         writer.add_scalar("Loss/ValCombined", val_metrics["loss"], epoch)
         writer.add_scalar("Loss/TrainBCE", train_metrics["bce"], epoch)
         writer.add_scalar("Loss/ValBCE", val_metrics["bce"], epoch)
         writer.add_scalar("Loss/TrainDice", train_metrics["dice_loss"], epoch)
         writer.add_scalar("Loss/ValDice", val_metrics["dice_loss"], epoch)
         writer.add_scalar("Loss/TrainSharpness", train_metrics["sharpness_loss"], epoch)
         writer.add_scalar("Loss/ValSharpness", val_metrics["sharpness_loss"], epoch)
         writer.add_scalar("Loss/TrainTV", train_metrics["tv_loss"], epoch)
         writer.add_scalar("Loss/ValTV", val_metrics["tv_loss"], epoch)
         writer.add_scalar("Loss/TrainMass", train_metrics["mass_loss"], epoch)
         writer.add_scalar("Loss/ValMass", val_metrics["mass_loss"], epoch)
         writer.add_scalar("Metric/TrainProbMSE", train_metrics["prob_mse"], epoch)
         writer.add_scalar("Metric/ValProbMSE", val_metrics["prob_mse"], epoch)
         writer.add_scalar("Metric/TrainChangedMSE", train_metrics["changed_mse"], epoch)
         writer.add_scalar("Metric/ValChangedMSE", val_metrics["changed_mse"], epoch)
         writer.add_scalar("Metric/TrainChangedPixelFrac", train_metrics["changed_pixel_frac"], epoch)
         writer.add_scalar("Metric/ValChangedPixelFrac", val_metrics["changed_pixel_frac"], epoch)
         writer.add_scalar("Metric/ValHardIoU", val_metrics["hard_iou"], epoch)
         writer.add_scalar("Metric/ValHardDice", val_metrics["hard_dice"], epoch)
         writer.add_scalar("Baseline/ValZeroMSE", val_metrics["zero_mse"], epoch)
         writer.add_scalar("Baseline/ValCopyInputMSE", val_metrics["copy_mse"], epoch)
         writer.add_scalar("Baseline/ValChangedZeroMSE", val_metrics["changed_zero_mse"], epoch)
         writer.add_scalar("Baseline/ValChangedCopyInputMSE", val_metrics["changed_copy_mse"], epoch)
         writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
         if not (improvement_rate != improvement_rate):  # not nan
            writer.add_scalar("Convergence/ValImprovementRate", improvement_rate, epoch)
         writer.add_scalar("Convergence/TrainValGap", train_val_gap, epoch)
         writer.add_scalar("Convergence/BestEpoch", best_epoch, epoch)
         writer.add_scalar("Convergence/EpochsWithoutImprovement", epochs_without_improvement, epoch)

         tbar.set_postfix(
            {
               "Train Loss": f"{train_metrics['loss']:.4f}",
               "Val Loss": f"{val_metrics['loss']:.4f}",
               "IoU": f"{val_metrics['hard_iou']:.3f}",
               "Best": best_epoch,
               "No Improv": epochs_without_improvement,
            }
         )

         if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping after {PATIENCE} epochs without validation loss improvement.")
            break

         if (epoch + 1) % 10 == 0:
            save_path = log_dir / f"unet_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), save_path)

         scheduler.step()

      writer.close()
      save_path = log_dir / "unet.pth"
      torch.save(model.state_dict(), save_path)

      epochs_run = len(val_loss_history)
      print(
         f"\n=== Convergence summary ==="
         f"\n  Epochs run:        {epochs_run} / {EPOCHS}"
         f"\n  Best val loss:     {best_val_loss:.6f}  (epoch {best_epoch})"
         f"\n  Suggested budget:  {best_epoch + PATIENCE} epochs  "
         f"(best epoch + early-stop patience)"
      )

      model.eval()
      test_metrics = evaluate_model(model, test_loader, criterion)
      print_test_metrics(test_metrics)
