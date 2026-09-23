import sys, os
sys.path.append(os.path.abspath("."))

import torch
import numpy as np
import matplotlib.pyplot as plt
import glob

from src.models.unet import UNet
from src.data.dataset import SatellitePatchDataset

# ---- CHANGE THESE TWO LINES FOR EACH RUN ----
MODEL_PATH = "experiments/unet_v1.pth"
OUTPUT_IMAGE = "notebooks/evaluation_75patches.png"
# ----------------------------------------------

num_patches = len(glob.glob("data/patches/patch_*.npy"))

model = UNet(input_channels=2, output_channels=1, base_features=32)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

dataset = SatellitePatchDataset(
    patches_dir="data/patches",
    damaged_dir="data/damaged",
    masks_dir="data/masks",
    num_patches=num_patches,
)

idx = 5  # same patch used every time, for fair comparison
x, y = dataset[idx]

with torch.no_grad():
    pred = model(x.unsqueeze(0))

damaged_img = x[0].numpy()
mask_img = x[1].numpy()
clean_img = y[0].numpy()
pred_img = pred[0, 0].numpy()

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(clean_img, cmap="gray"); axes[0].set_title("Original (ground truth)")
axes[1].imshow(damaged_img, cmap="gray"); axes[1].set_title("Damaged input")
axes[2].imshow(mask_img, cmap="gray"); axes[2].set_title("Mask")
axes[3].imshow(pred_img, cmap="gray"); axes[3].set_title(f"Prediction ({MODEL_PATH})")
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE)
plt.show()

print(f"Saved evaluation to {OUTPUT_IMAGE}")