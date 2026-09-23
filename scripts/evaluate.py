import sys, os
sys.path.append(os.path.abspath("."))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.unet import UNet
from src.data.dataset import SatellitePatchDataset

# load trained model
model = UNet(input_channels=2, output_channels=1, base_features=32)
model.load_state_dict(torch.load("experiments/unet_v1.pth"))
model.eval()

# load one sample
dataset = SatellitePatchDataset(
    patches_dir="data/patches",
    damaged_dir="data/damaged",
    masks_dir="data/masks",
    num_patches=15,
)

idx = 5  # same patch we visualized back in Phase 4
x, y = dataset[idx]

with torch.no_grad():
    pred = model(x.unsqueeze(0))  # add batch dimension

# convert back to plain numpy for plotting
damaged_img = x[0].numpy()
mask_img = x[1].numpy()
clean_img = y[0].numpy()
pred_img = pred[0, 0].numpy()

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(clean_img, cmap="gray"); axes[0].set_title("Original (ground truth)")
axes[1].imshow(damaged_img, cmap="gray"); axes[1].set_title("Damaged input")
axes[2].imshow(mask_img, cmap="gray"); axes[2].set_title("Mask")
axes[3].imshow(pred_img, cmap="gray"); axes[3].set_title("Model prediction")
plt.tight_layout()
plt.savefig("notebooks/evaluation_preview.png")
plt.show()
