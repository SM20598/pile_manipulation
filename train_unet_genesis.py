#############################################################
# THIS SCRIPT IS USED TO TRAIN A UNET WITH THE GENESIS DATA #
#############################################################

import argparse
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from Genesis.training.dataset import PileSweepData
from GranularDynamics2.myClasses.NFDUNetFilm import NFDUNetFiLM, NFDUNetFiLMShallow


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 70
BATCH_SIZE = 64
LR = 1e-4

MSE_WEIGHT = 1.0
MASS_WEIGHT = 0.2
PATIENCE = 10
CHANGE_THRESHOLD = 1e-3


def parse_args():
   parser = argparse.ArgumentParser(description="Train or evaluate the FiLM-UNet model.")

   ###################
   # EXPORT SETTINGS #
   ###################
   parser.add_argument("--log-dir", type=str, default="runs")
   parser.add_argument("--ident", type=str, default=None, help="Adds an identifier to log dir and pth files: logdir_{log-ident}. ")

   ##################
   # MODEL SETTINGS #
   ##################
   parser.add_argument("--data-folders", nargs="+", default=["corl/cubes"])
   parser.add_argument("--model-variant", choices=["full", "shallow", "lowres", "shallow-lowres"], default="full", help=(
      "full: UNet of depth 3 with 128x128 grid input;"
      "shallow: UNet of depth 2 with 128x128 grid input;"
      "lowres: UNet of depth 3 with 32x32 grid input;"
      "shallow-lowres: UNet of depth 2 with 32x32 grid input;"
      ),
   )
   parser.add_argument("--input-mode", choices=["standard", "sweep-removed-input", "sweep-removed-residual"], default="standard", help=(
         "standard: 2 channels, residual to current occupancy; "
         "sweep-removed-input: 3 channels, residual to current occupancy; "
         "sweep-removed-residual: 3 channels, residual to sweep-removed occupancy."
      ),
   )

   ###################
   # TRAINING PARAMS #
   ###################
   parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
   parser.add_argument("--epochs", type=int, default=EPOCHS, help="Total epoch budget, including resumed epochs.")
   parser.add_argument("--num-workers", type=int, default=4)
   parser.add_argument("--fresh-start", action="store_true", help="Do not resume from an existing checkpoint.")
   parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Checkpoint to resume training from.")
   parser.add_argument("--start-epoch", type=int, default=None, help="Epoch index already completed by the resume checkpoint.")
   parser.add_argument("--mse-weight", type=float, default=MSE_WEIGHT)
   parser.add_argument("--mass-weight", type=float, default=MASS_WEIGHT)
   
   return parser.parse_args()


def checkpoint_epoch(path: Path) -> int | None:
   match = re.search(r"epoch_(\d+)", path.stem)
   return int(match.group(1)) if match else None


def latest_epoch_checkpoint(log_dir: Path) -> Path | None:
   checkpoints = []
   for path in log_dir.glob("unet_epoch_*.pth"):
      epoch = checkpoint_epoch(path)
      if epoch is not None:
         checkpoints.append((epoch, path))
   if not checkpoints:
      return None
   return max(checkpoints, key=lambda item: item[0])[1]


def default_resume_checkpoint(log_dir: Path) -> Path | None:
   """
   Returns path to latest model, best model, final model or None.
   If returns None no model has been found in the dir.
   """
   latest = latest_epoch_checkpoint(log_dir)
   if latest is not None:
      return latest
   best = log_dir / "unet_best.pth"
   if best.exists():
      return best
   final = log_dir / "unet.pth"
   if final.exists():
      return final
   return None


def augment_batch(inputs, outputs, physics):
   """Rotate and mirror samples: 1 sample --> 8 samples"""
   inputs_rot = [torch.rot90(inputs, k, dims=(-2, -1)) for k in range(4)]
   inputs_mir = [torch.flip(r, dims=[-1]) for r in inputs_rot]
   inputs = torch.cat(inputs_rot + inputs_mir, dim=0)

   outputs_rot = [torch.rot90(outputs, k, dims=(-2, -1)) for k in range(4)]
   outputs_mir = [torch.flip(r, dims=[-1]) for r in outputs_rot]
   outputs = torch.cat(outputs_rot + outputs_mir, dim=0)

   physics = physics.repeat(8, 1)
   return inputs, outputs, physics



def combined_loss(predictions: torch.Tensor, outputs:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
   mse = F.mse_loss(predictions, outputs)
   mass = (predictions.sum(dim=(1, 2)) - outputs.sum(dim=(1, 2))).abs().mean() / outputs[0].numel()
   loss = MSE_WEIGHT * mse + MASS_WEIGHT * mass
   return (loss, mass)


def update_metric_totals(
   totals,
   predictions,
   outputs,
   inputs,
   loss,
   mass_loss,
):
   current_state = inputs[:, 0]
   pred_mask = predictions > 0.5
   target_mask = outputs > 0.5
   changed_mask = (outputs - current_state).abs() > CHANGE_THRESHOLD
   batch_size = inputs.size(0)
   intersection = (pred_mask & target_mask).sum(dim=(1, 2)).float()
   pred_area = pred_mask.sum(dim=(1, 2)).float()
   target_area = target_mask.sum(dim=(1, 2)).float()
   union = pred_area + target_area - intersection
   changed_count = changed_mask.sum().item()

   totals["loss"] += loss.item() * batch_size
   totals["mass_loss"] += mass_loss.item() * batch_size
   totals["prob_mse"] += F.mse_loss(predictions, outputs).item() * batch_size
   totals["zero_mse"] += F.mse_loss(torch.zeros_like(outputs), outputs).item() * batch_size
   totals["copy_mse"] += F.mse_loss(current_state, outputs).item() * batch_size
   if changed_count > 0:
      totals["changed_prob_sse"] += ((predictions - outputs).pow(2) * changed_mask).sum().item()
      totals["changed_zero_sse"] += outputs.pow(2).mul(changed_mask).sum().item()
      totals["changed_copy_sse"] += ((current_state - outputs).pow(2) * changed_mask).sum().item()
      totals["changed_pixels"] += changed_count
   totals["total_pixels"] += changed_mask.numel()
   totals["hard_iou"] += ((intersection + 1e-6) / (union + 1e-6)).sum().item()
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
      "size": 0,
   }


def evaluate_model(model, loader):
   model.eval()
   totals = empty_totals()
   with torch.no_grad():
      for inputs_, outputs in loader:
         inputs, physics = inputs_
         inputs = inputs.to(DEVICE)
         physics = physics.to(DEVICE)
         outputs = outputs.to(DEVICE)

         predictions = model(inputs, physics).squeeze(1).float()
         loss, mass_loss = combined_loss(predictions, outputs)
         update_metric_totals(
            totals,
            predictions,
            outputs,
            inputs,
            loss,
            mass_loss
         )

   return average_metrics(totals)


def print_test_metrics(test_metrics):
   print(
      f"Test Loss: {test_metrics['loss']:.6f}, "
      f"Test MassLoss: {test_metrics['mass_loss']:.6f}, "
      f"Test MSE: {test_metrics['prob_mse']:.6f}, "
      f"Test IoU: {test_metrics['hard_iou']:.4f}, "
      f"Zero MSE: {test_metrics['zero_mse']:.6f}, "
      f"Copy MSE: {test_metrics['copy_mse']:.6f}, "
      f"Changed Pixel Frac: {test_metrics['changed_pixel_frac']:.6f}, "
      f"Changed MSE: {test_metrics['changed_mse']:.6f}, "
      f"Changed Zero MSE: {test_metrics['changed_zero_mse']:.6f}, "
      f"Changed Copy MSE: {test_metrics['changed_copy_mse']:.6f}"
   )


if __name__ == "__main__":
   args = parse_args()
   data_aug    = True
   EPOCHS      = args.epochs
   MSE_WEIGHT  = args.mse_weight
   MASS_WEIGHT = args.mass_weight
   

   #################
   # Log Directory #
   #################
   log_dir = args.log_dir
   log_dir = log_dir + "_" + str(args.model_variant) + "_" + str(args.input_mode)
   if args.ident:
      log_dir = log_dir + "_" + args.ident
   log_dir = Path(log_dir)
   log_dir.mkdir(parents=True, exist_ok=True)


   ##############
   # Init Model #
   ##############
   model_variant = args.model_variant.lower()
   input_mode    = args.input_mode.lower()
   remove_sweep  = input_mode in ("sweep-removed-input", "sweep-removed-residual")
   params = {
      "in_channels": 3 if remove_sweep else 2,
      "residual_channel": 2 if input_mode == "sweep-removed-residual" else 0,
   }
   model = NFDUNetFiLMShallow(**params) if "shallow" in model_variant else NFDUNetFiLM(**params).to(DEVICE)

   start_epoch = 0
   if not args.fresh_start:
      resume_checkpoint = args.resume_checkpoint or default_resume_checkpoint(log_dir)
      if resume_checkpoint is not None:
         state_dict = torch.load(resume_checkpoint, map_location=DEVICE)
         model.load_state_dict(state_dict)
         start_epoch = args.start_epoch if args.start_epoch is not None else (checkpoint_epoch(resume_checkpoint) or 0)
         print(f"Resuming from {resume_checkpoint} at epoch {start_epoch}. Training to epoch {EPOCHS}.")
      else:
         print(f"No checkpoint found in {log_dir}; starting from scratch.")

   print( "=========== Model settings ===========")
   print(f"Model variant: {model_variant}")
   print(f"Input mode: {input_mode}")
   print(f"Log dir: {log_dir}")
   print( "=========== -------------- ===========")


   ################
   # Load Dataset #
   ################
   num_workers      = args.num_workers
   data_folders     = args.data_folders
   resolution_scale = 0.25 if model_variant in ("lowres", "shallow-lowres") else 1.0
   batch_size       = args.batch_size // 8 if data_aug else args.batch_size

   train_dataset = PileSweepData(data_folders, split="train", resolution_scale=resolution_scale, include_sweep_removed=remove_sweep)
   val_dataset   = PileSweepData(data_folders, split="val",   resolution_scale=resolution_scale, include_sweep_removed=remove_sweep)
   test_dataset  = PileSweepData(data_folders, split="test",  resolution_scale=resolution_scale, include_sweep_removed=remove_sweep)

   train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=DEVICE == "cuda")
   val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=DEVICE == "cuda")
   test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=DEVICE == "cuda")


   #################
   # Init Training #
   #################
   best_val_loss              = float("inf")
   best_epoch                 = start_epoch
   val_loss_history           = []
   improvement_window         = 5  # epochs over which to measure improvement rate
   epochs_without_improvement = 0
   
   writer = SummaryWriter(log_dir=log_dir)
   scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
   optimizer = torch.optim.Adam(model.parameters(), lr=LR)
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
   if start_epoch > 0:
      resumed_lr = LR * (0.5 ** (start_epoch // 10))
      for param_group in optimizer.param_groups:
         param_group["lr"] = resumed_lr
      print(f"Resumed optimizer learning rate: {resumed_lr:.8f}")

      resume_val_metrics = evaluate_model(model, val_loader)
      best_val_loss = resume_val_metrics["loss"]
      val_loss_history.append(best_val_loss)
      print(
         f"Resume checkpoint val loss={best_val_loss:.6f}, "
         f"val MSE={resume_val_metrics['prob_mse']:.6f}, "
         f"val changed MSE={resume_val_metrics['changed_mse']:.6f}"
      )


   with trange(start_epoch, EPOCHS, desc="Training Epochs") as tbar:
      for epoch in tbar:

         ### TRAINING LOOP ###
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
               logits: torch.Tensor = model(inputs, physics).squeeze(1)
               loss, mass_loss = combined_loss(logits.float(), outputs)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            update_metric_totals(train_totals, logits.detach().float(), outputs, inputs, loss.detach(), mass_loss.detach())

         train_metrics = average_metrics(train_totals)

         ### VALIDATION LOOP ###
         model.eval()
         val_totals = empty_totals()
         with torch.no_grad():
            for inputs_, outputs in val_loader:
               inputs, physics = inputs_
               inputs = inputs.to(DEVICE)
               physics = physics.to(DEVICE)
               outputs = outputs.to(DEVICE)

               logits = model(inputs, physics).squeeze(1).float()
               loss, mass_loss = combined_loss(logits, outputs)
               update_metric_totals(val_totals, logits, outputs, inputs, loss, mass_loss)

         val_metrics = average_metrics(val_totals)
         avg_val_loss = val_metrics["loss"]
         
         
         print(
            f"Epoch {epoch + 1}: "
            f"train loss={train_metrics['loss']:.6f}, train MSE={train_metrics['prob_mse']:.6f}, "
            f"val loss={val_metrics['loss']:.6f}, val MSE={val_metrics['prob_mse']:.6f}, "
            f"val IoU={val_metrics['hard_iou']:.4f}, "
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
         if len(val_loss_history) >= improvement_window + 1:
            improvement_rate = (val_loss_history[-improvement_window - 1] - val_loss_history[-1]) / improvement_window
         else:
            improvement_rate = float("nan")
         train_val_gap = train_metrics["loss"] - val_metrics["loss"]

         writer.add_scalar("Loss/TrainCombined", train_metrics["loss"], epoch)
         writer.add_scalar("Loss/ValCombined", val_metrics["loss"], epoch)
         writer.add_scalar("Loss/TrainMass", train_metrics["mass_loss"], epoch)
         writer.add_scalar("Loss/ValMass", val_metrics["mass_loss"], epoch)
         writer.add_scalar("Metric/TrainProbMSE", train_metrics["prob_mse"], epoch)
         writer.add_scalar("Metric/ValProbMSE", val_metrics["prob_mse"], epoch)
         writer.add_scalar("Metric/TrainChangedMSE", train_metrics["changed_mse"], epoch)
         writer.add_scalar("Metric/ValChangedMSE", val_metrics["changed_mse"], epoch)
         writer.add_scalar("Metric/TrainChangedPixelFrac", train_metrics["changed_pixel_frac"], epoch)
         writer.add_scalar("Metric/ValChangedPixelFrac", val_metrics["changed_pixel_frac"], epoch)
         writer.add_scalar("Metric/ValHardIoU", val_metrics["hard_iou"], epoch)
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

      resume_baseline_count = 1 if start_epoch > 0 else 0
      epochs_run = len(val_loss_history) - resume_baseline_count
      print(
         f"\n=== Convergence summary ==="
         f"\n  Epochs run:        {start_epoch + epochs_run} / {EPOCHS}"
         f"\n  Resumed from:      {start_epoch}"
         f"\n  Best val loss:     {best_val_loss:.6f}  (epoch {best_epoch})"
         f"\n  Suggested budget:  {best_epoch + PATIENCE} epochs  "
         f"(best epoch + early-stop patience)"
      )

      model.eval()
      test_metrics = evaluate_model(model, test_loader)
      print_test_metrics(test_metrics)
