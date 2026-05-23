#############################################################
# THIS SCRIPT IS USED TO TRAIN A UNET WITH THE GENESIS DATA #
#############################################################

# Inputs
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from Genesis.training.dataset import PileSweepData
from GranularDynamics2.myClasses.UNetModels_modular import UNet
from GranularDynamics2.myClasses.UNetModels_conditioned import UNetConditioned, UNetFiLM
from tqdm import trange
import torch
from torch.utils.tensorboard import SummaryWriter
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-4


def chose_loss(loss : str):
   if loss == "mse":
      return torch.nn.MSELoss()
   else:
      from GranularDynamics2.utils import output_dice_loss
      return output_dice_loss
   

if __name__ == "__main__":

   continue_training = False
   data_folders = ["corl"]
   log_dir = "runs/unet_unconditioned"
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
   
   dataset : Dataset = PileSweepData(data_folders)
   
   # +++ INSPECT DATA +++
   # for i in range(len(dataset)):
   #    inputs, label = dataset[i]
   #    inputt, _ = inputs
   #    dataset.plot_input_and_output(inputt.cpu(), label.cpu())
      
   # +++ MODEL CHOICE +++
   # model = UNetConditioned(structure_parameters).to(DEVICE)
   # model = UNetConditioned().to(DEVICE)
   model = UNetFiLM().to(DEVICE)

   optimizer = torch.optim.Adam(model.parameters(), lr=LR)
   criterion = torch.nn.MSELoss()
   writer = SummaryWriter(log_dir=log_dir)
   scaler = torch.amp.GradScaler()
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
   
   
   avg_val_losses = []
   with trange(EPOCHS, desc="Training Epochs") as tbar:
   
      n_samples = len(dataset)
      
      n_train = int(0.88 * n_samples)
      n_val = int(0.02 * n_samples)
      train_data = torch.utils.data.Subset(dataset, range(0, n_train))
      val_data   = torch.utils.data.Subset(dataset, range(n_train, n_train+ n_val))
      test_data   = torch.utils.data.Subset(dataset, range(n_train+ n_val, n_samples))
      
      val_loader = DataLoader(
            val_data,
            batch_size=BATCH_SIZE // 8 if data_aug else BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
         )
      train_loader = DataLoader(
         train_data,
         batch_size= BATCH_SIZE // 8 if data_aug else BATCH_SIZE,
         shuffle=True,
         num_workers=4,
         pin_memory=True
      )
      test_loader = DataLoader(
         test_data,
         batch_size=BATCH_SIZE // 8 if data_aug else BATCH_SIZE,
         shuffle=False,
         num_workers=4,
         pin_memory=True
      )
      
      for epoch in tbar:
         model.train()

         total_loss = 0.0
         val_loss = 0.0
         train_size = 0
         val_size = 0
         
         for inputs_, outputs in train_loader:
               inputs, physics = inputs_
               
               inputs = inputs.to(DEVICE)
               physics = physics.to(DEVICE)
               outputs = outputs.to(DEVICE)
               
               if data_aug:
                  inputs_rot = [torch.rot90(inputs, k, dims=(-2, -1)) for k in range(4)]
                  inputs_mir = [torch.flip(r, dims=[-1]) for r in inputs_rot]
                  inputs = torch.cat(inputs_rot + inputs_mir, dim=0)
                  
                  outputs_rot = [torch.rot90(outputs, k, dims=(-2, -1)) for k in range(4)]
                  outputs_mir = [torch.flip(r, dims=[-1]) for r in outputs_rot]
                  outputs = torch.cat(outputs_rot + outputs_mir, dim=0)
                  
                  physics = physics.repeat(8, 1)
                  
               optimizer.zero_grad(set_to_none=True)

               with torch.amp.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                  pred_next = model(inputs, physics)
               loss = criterion(pred_next.squeeze(1).float(), outputs)  # (B, 1, H, W) -> (B, H, W)
               
               scaler.scale(loss).backward()
               scaler.unscale_(optimizer)
               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
               scaler.step(optimizer)
               scaler.update()

               total_loss += loss.item() * inputs.size(0)
               train_size += inputs.size(0)
         print("Total loss", total_loss, "train size", train_size)
         
         # VALIDATION LOOP
         model.eval()
         with torch.no_grad():
            for inputs_, outputs in val_loader:
               inputs, physics = inputs_
               
               inputs = inputs.to(DEVICE)
               physics = physics.to(DEVICE)
               outputs = outputs.to(DEVICE)
               
               pred_next = model(inputs, physics)
               val_loss += criterion(pred_next.squeeze(1), outputs.squeeze(1)).item() * inputs.size(0)
               val_size += inputs.size(0)
         print("Avg val loss", avg_val_losses, "train size", val_size)
         
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
            torch.save(model.state_dict(), save_path)
         scheduler.step()
         
      writer.close()                
      save_path = os.path.join(log_dir, "unet.pth")
      torch.save(model.state_dict(), save_path)

      criterion = torch.nn.MSELoss()
      test_loss = 0.0
      with torch.no_grad():
            for inputs_, outputs in test_loader:
               inputs, physics = inputs_
               
               inputs = inputs.to(DEVICE)
               physics = physics.to(DEVICE)
               outputs = outputs.to(DEVICE)
               
               pred_next = model(inputs, physics)
               test_inputs = pred_next.squeeze(1)
            # test_inputs = inputs[:,0:1,:,:].squeeze(1)
            test_loss += criterion(test_inputs, outputs).item() * inputs.size(0)#
      avg_test_loss = test_loss / len(test_loader.dataset)
      print(f"Test Loss: {avg_test_loss}")

   

   