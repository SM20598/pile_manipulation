#############################################################
# THIS SCRIPT IS USED TO TRAIN A UNET WITH THE GENESIS DATA #
#############################################################

# Inputs
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from Genesis.training.dataset import PileSweepData
from GranularDynamics2.myClasses.UNetModels_modular import UNet
from GranularDynamics2.TrainUnet_modular import data_augmentation
from tqdm import trange
import torch
from torch.utils.tensorboard import SummaryWriter
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-4

def load_genesis_dataset(data_folder):
   base_dir = Path(__file__).parent
   full_path = base_dir / "Genesis" / "data" / data_folder
   return PileSweepData(full_path)

def chose_loss(loss : str):
   if "mse":
      return torch.nn.MSELoss()
   else:
      from GranularDynamics2.utils import output_dice_loss
      return output_dice_loss
   

if __name__ == "__main__":

   continue_training = False
   data_folder = "cubes/chickpeas_on_glass"
   log_dir = "runs/unet_train"
   structure_parameters = {
    "in_channels": 2,
    "out_channels": 1,
    "features": [4,8],
    "kernel_size": 3,
    "activation": 'relu',
    "activation_list": ['relu'],
    "residual": True,
    "bottleneck_type": "None",
    "mixed_blocks": []
    }
   data_aug = True
   
   dataset : Dataset = load_genesis_dataset(data_folder)
   model = UNet(structure_parameters).to(DEVICE)
   optimizer = torch.optim.Adam(model.parameters(), lr=LR)
   criterion = torch.nn.MSELoss()
   writer = SummaryWriter(log_dir=log_dir)
   scaler = torch.amp.GradScaler()
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
   
   
   avg_val_losses = []
   with trange(EPOCHS, desc="Training Epochs") as tbar:
      for epoch in tbar:
            model.train()

            total_loss = 0.0
            val_loss = 0.0
            train_size = 0
            val_size = 0

            file_idx = 0
            n_samples = len(dataset)
            while file_idx < n_samples:
               n_train = int(0.8 * n_samples)
               n_val = int(0.1 * n_samples)
               train_data = torch.utils.data.Subset(dataset, range(0, n_train))
               val_data   = torch.utils.data.Subset(dataset, range(n_train, n_train+ n_val))

               val_loader = DataLoader(
                  val_data,
                  batch_size=BATCH_SIZE // 4 if data_aug else BATCH_SIZE,
                  shuffle=False,
                  num_workers=4,
                  pin_memory=True
               )
               train_loader = DataLoader(
                  train_data,
                  batch_size=BATCH_SIZE // 4 if data_aug else BATCH_SIZE,
                  shuffle=True,
                  num_workers=4,
                  pin_memory=True
               )
               
               # TRAINING LOOP
               for inputs, outputs in train_loader:
                  
                  if data_aug:
                        inputs, outputs = data_augmentation(inputs, outputs)
                  inputs.to(DEVICE)
                  outputs.to(DEVICE)
               
                  optimizer.zero_grad(set_to_none=True)

                  with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16):
                     pred_next = model(inputs)
                     loss = criterion(pred_next.squeeze(1), outputs)  # (B, 1, H, W) -> (B, H, W)
                  
                  scaler.scale(loss).backward()
                  scaler.step(optimizer)
                  scaler.update()

                  total_loss += loss.item() * inputs.size(0)
                  train_size += inputs.size(0)

               # Validation
               model.eval()
               
               # VALIDATION LOOP
               with torch.no_grad():
                  for inputs, outputs in val_loader:
                     inputs, outputs = inputs.to(DEVICE), outputs.to(DEVICE)
                     pred_next = model(inputs)
                     val_loss += criterion(pred_next.squeeze(1), outputs.squeeze(1)).item() * inputs.size(0)
                     val_size += inputs.size(0)

            avg_val_loss = val_loss / val_size
            avg_val_losses.append(avg_val_loss)

            if len(avg_val_losses) > 5 and avg_val_loss > max(avg_val_losses[-5:]):
               print("Early stopping due to no improvement in validation loss.")
               break

            writer.add_scalar("Loss/Val", avg_val_loss, epoch)
            avg_train_loss = total_loss / train_size
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            tbar.set_postfix({"Train Loss": avg_train_loss, "Val Loss": avg_val_loss})

            # Save model every 10 epochs
            if (epoch + 1) % 10 == 0:
               save_path = os.path.join(log_dir, f"unet_epoch_{epoch+1}.pth")
               model.save_checkpoint(save_path)
            scheduler.step()
            pass
   writer.close()                
   save_path = os.path.join(log_dir, "unet.pth")
   model.save_checkpoint(save_path)

   

   