import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from Genesis.training.dataset import PileSweepData
from GranularDynamics2.myClasses.UNetModels_conditioned import UNetConditioned
            
            
def plot_grids_4x4(input, action, ground_truth, prediction) -> None:
        """Visualize the grid as an image"""
        from matplotlib import pyplot as plt
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(input.cpu().numpy(), interpolation='nearest')
        axs[0, 0].set_title("Input State")
        axs[0, 1].imshow(action.cpu().numpy(), interpolation='nearest')
        axs[0, 1].set_title("Action")
        axs[1, 0].imshow(ground_truth.cpu().numpy(), interpolation='nearest')
        axs[1, 0].set_title("Ground Truth")
        axs[1, 1].imshow(prediction.cpu().numpy(), interpolation='nearest')
        axs[1, 1].set_title("Model Prediction")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    from Genesis.training.dataset import PileSweepData
    model = UNetConditioned()
    model.load_state_dict(torch.load("runs/unetfilm/unet.pth", weights_only=True))
    model.eval()
    data_folders = ["chickpeas/chickspheres_on_glass", "chickpeas/chickspheres_on_wood"]
    dataset = PileSweepData(data_folders)
   
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(DEVICE)
   
    n_samples = len(dataset)
    test_data = torch.utils.data.Subset(dataset, range(int(0.9 * n_samples), n_samples))
    test_loader = DataLoader(
        test_data,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    with torch.no_grad():
        for inputs_, outputs in test_loader:
            inputs, physics = inputs_
            
            # inputs = inputs.to(DEVICE)
            # physics = physics.to(DEVICE)
            # outputs = outputs.to(DEVICE)
            
            pred_next = model(inputs, physics)

            plot_grids_4x4(inputs[0][0], inputs[0][1], outputs[0], pred_next[0][0])