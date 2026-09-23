import sys, os
import torch
sys.path.append(os.path.abspath("."))

from src.data.dataset import SatellitePatchDataset
from src.models.unet import UNet
from src.training.trainer import train_model

dataset = SatellitePatchDataset(
    patches_dir="data/patches",
    damaged_dir="data/damaged",
    masks_dir="data/masks",
    num_patches=15,
)

model = UNet(input_channels=2, output_channels=1, base_features=32)

trained_model = train_model(model, dataset, epochs=20, batch_size=4, lr=1e-3)

os.makedirs("experiments", exist_ok=True)
torch.save(trained_model.state_dict(), "experiments/unet_v1.pth")
print("Model saved to experiments/unet_v1.pth")