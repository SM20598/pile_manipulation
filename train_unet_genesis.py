#############################################################
# THIS SCRIPT IS USED TO TRAIN A UNET WITH THE GENESIS DATA #
#############################################################

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
PATIENCE = 10


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
   bce = criterion(logits, outputs)
   dice = soft_dice_loss(logits, outputs)
   return bce + DICE_WEIGHT * dice, bce, dice


def update_metric_totals(totals, logits, outputs, inputs, loss, bce_loss, dice_loss):
   probs = torch.sigmoid(logits)
   pred_mask = probs > 0.5
   target_mask = outputs > 0.5
   batch_size = inputs.size(0)
   intersection = (pred_mask & target_mask).sum(dim=(1, 2)).float()
   pred_area = pred_mask.sum(dim=(1, 2)).float()
   target_area = target_mask.sum(dim=(1, 2)).float()
   union = pred_area + target_area - intersection

   totals["loss"] += loss.item() * batch_size
   totals["bce"] += bce_loss.item() * batch_size
   totals["dice_loss"] += dice_loss.item() * batch_size
   totals["prob_mse"] += F.mse_loss(probs, outputs).item() * batch_size
   totals["zero_mse"] += F.mse_loss(torch.zeros_like(outputs), outputs).item() * batch_size
   totals["copy_mse"] += F.mse_loss(inputs[:, 0], outputs).item() * batch_size
   totals["hard_iou"] += ((intersection + 1e-6) / (union + 1e-6)).sum().item()
   totals["hard_dice"] += ((2.0 * intersection + 1e-6) / (pred_area + target_area + 1e-6)).sum().item()
   totals["size"] += batch_size


def average_metrics(totals):
   return {
      key: value / totals["size"]
      for key, value in totals.items()
      if key != "size"
   }


def empty_totals():
   return {
      "loss": 0.0,
      "bce": 0.0,
      "dice_loss": 0.0,
      "prob_mse": 0.0,
      "zero_mse": 0.0,
      "copy_mse": 0.0,
      "hard_iou": 0.0,
      "hard_dice": 0.0,
      "size": 0,
   }


if __name__ == "__main__":
   data_folders = ["corl"]
   log_dir = Path("runs/nfdunetfilm_bce_dice_change")
   data_aug = True
   log_dir.mkdir(parents=True, exist_ok=True)

   train_dataset: Dataset = PileSweepData(data_folders, split="train")
   val_dataset: Dataset = PileSweepData(data_folders, split="val")
   test_dataset: Dataset = PileSweepData(data_folders, split="test")

   model = NFDUNetFiLM().to(DEVICE)

   optimizer = torch.optim.Adam(model.parameters(), lr=LR)
   pos_weight = torch.tensor([POS_WEIGHT], device=DEVICE)
   criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
   writer = SummaryWriter(log_dir=log_dir)
   scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

   batch_size = BATCH_SIZE // 8 if data_aug else BATCH_SIZE
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
   val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
   test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

   best_val_loss = float("inf")
   epochs_without_improvement = 0
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
               loss, bce_loss, dice_loss = combined_loss(logits.float(), outputs, criterion)

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
               loss, bce_loss, dice_loss = combined_loss(logits, outputs, criterion)
               update_metric_totals(val_totals, logits, outputs, inputs, loss, bce_loss, dice_loss)

         val_metrics = average_metrics(val_totals)
         avg_val_loss = val_metrics["loss"]
         print(
            f"Epoch {epoch + 1}: "
            f"train loss={train_metrics['loss']:.6f}, train MSE={train_metrics['prob_mse']:.6f}, "
            f"val loss={val_metrics['loss']:.6f}, val MSE={val_metrics['prob_mse']:.6f}, "
            f"val IoU={val_metrics['hard_iou']:.4f}, val Dice={val_metrics['hard_dice']:.4f}, "
            f"val copy MSE={val_metrics['copy_mse']:.6f}"
         )

         if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), log_dir / "unet_best.pth")
         else:
            epochs_without_improvement += 1

         writer.add_scalar("Loss/TrainCombined", train_metrics["loss"], epoch)
         writer.add_scalar("Loss/ValCombined", val_metrics["loss"], epoch)
         writer.add_scalar("Loss/TrainBCE", train_metrics["bce"], epoch)
         writer.add_scalar("Loss/ValBCE", val_metrics["bce"], epoch)
         writer.add_scalar("Loss/TrainDice", train_metrics["dice_loss"], epoch)
         writer.add_scalar("Loss/ValDice", val_metrics["dice_loss"], epoch)
         writer.add_scalar("Metric/TrainProbMSE", train_metrics["prob_mse"], epoch)
         writer.add_scalar("Metric/ValProbMSE", val_metrics["prob_mse"], epoch)
         writer.add_scalar("Metric/ValHardIoU", val_metrics["hard_iou"], epoch)
         writer.add_scalar("Metric/ValHardDice", val_metrics["hard_dice"], epoch)
         writer.add_scalar("Baseline/ValZeroMSE", val_metrics["zero_mse"], epoch)
         writer.add_scalar("Baseline/ValCopyInputMSE", val_metrics["copy_mse"], epoch)
         writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

         tbar.set_postfix(
            {
               "Train Loss": train_metrics["loss"],
               "Val Loss": val_metrics["loss"],
               "Val MSE": val_metrics["prob_mse"],
               "IoU": val_metrics["hard_iou"],
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

      model.eval()
      test_totals = empty_totals()
      with torch.no_grad():
         for inputs_, outputs in test_loader:
            inputs, physics = inputs_
            inputs = inputs.to(DEVICE)
            physics = physics.to(DEVICE)
            outputs = outputs.to(DEVICE)

            logits = model(inputs, physics).squeeze(1).float()
            loss, bce_loss, dice_loss = combined_loss(logits, outputs, criterion)
            update_metric_totals(test_totals, logits, outputs, inputs, loss, bce_loss, dice_loss)

      test_metrics = average_metrics(test_totals)
      print(
         f"Test Loss: {test_metrics['loss']:.6f}, "
         f"Test BCE: {test_metrics['bce']:.6f}, "
         f"Test DiceLoss: {test_metrics['dice_loss']:.6f}, "
         f"Test MSE: {test_metrics['prob_mse']:.6f}, "
         f"Test IoU: {test_metrics['hard_iou']:.4f}, "
         f"Test Dice: {test_metrics['hard_dice']:.4f}, "
         f"Zero MSE: {test_metrics['zero_mse']:.6f}, "
         f"Copy MSE: {test_metrics['copy_mse']:.6f}"
      )
