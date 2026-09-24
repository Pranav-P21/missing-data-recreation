import sys, os
sys.path.append(os.path.abspath("."))

import random
import torch
import glob

torch.manual_seed(42)

from src.data.dataset import SatellitePatchDataset
from src.models.unet import UNet
from src.training.trainer import train_model


def grid_position_of(idx):
    return idx % 15


num_patches = len(glob.glob("data/patches/patch_*.npy"))
all_idx = list(range(num_patches))

random.seed(42)
positions = list(range(15))
random.shuffle(positions)
val_positions = set(positions[:3])
train_positions = set(positions[3:])

train_ids = [i for i in all_idx if grid_position_of(i) in train_positions]
val_ids = [i for i in all_idx if grid_position_of(i) in val_positions]
print(f"Total: {num_patches} patches -> Train: {len(train_ids)}, Val: {len(val_ids)}")

train_dataset = SatellitePatchDataset("data/patches", "data/damaged", "data/masks", train_ids, augment=True)
val_dataset = SatellitePatchDataset("data/patches", "data/damaged", "data/masks", val_ids, augment=False)

model = UNet(input_channels=2, output_channels=1, base_features=32)

trained_model = train_model(model, train_dataset, val_dataset, epochs=20, batch_size=4, lr=1e-3)

os.makedirs("experiments", exist_ok=True)

print("Model saved to experiments/unet_v2.pth")