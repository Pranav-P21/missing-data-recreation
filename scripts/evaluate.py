import sys, os
sys.path.append(os.path.abspath("."))

import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob

from src.models.unet import UNet
from src.data.dataset import SatellitePatchDataset


import math

def psnr(pred, target, max_val=1.0):
    mse = ((pred - target) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(max_val) - 10 * math.log10(mse)

# ---- CHANGE THESE FOR EACH RUN ----
MODEL_PATH = "experiments/unet_best.pth"
OUTPUT_IMAGE = "notebooks/evaluation_val_patch.png"
# ------------------------------------


def grid_position_of(idx):
    return idx % 15


num_patches = len(glob.glob("data/patches/patch_*.npy"))
all_idx = list(range(num_patches))

# must match scripts/train.py exactly - same seed, same logic - so this
# reconstructs the identical val set the model was actually validated on
random.seed(42)
positions = list(range(15))
random.shuffle(positions)
val_positions = set(positions[:3])

val_ids = [i for i in all_idx if grid_position_of(i) in val_positions]

model = UNet(input_channels=2, output_channels=1, base_features=32)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

val_dataset = SatellitePatchDataset(
    patches_dir="data/patches",
    damaged_dir="data/damaged",
    masks_dir="data/masks",
    patch_ids=val_ids,
    augment=False,
)

total_psnr = 0.0
with torch.no_grad():
    for x, y in val_dataset:
        pred = model(x.unsqueeze(0))
        total_psnr += psnr(pred, y.unsqueeze(0))
avg_psnr = total_psnr / len(val_dataset)
print(f"Average PSNR over {len(val_dataset)} val patches: {avg_psnr:.2f} dB")

i = 0  # index into val_dataset (0 = first held-out patch); not a raw patch idx
x, y = val_dataset[i]

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
plt.suptitle(f"Held-out val patch (patch_id={val_ids[i]:03d}, grid_pos={grid_position_of(val_ids[i])})")
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE)
plt.show()

print(f"Evaluated on val patch_{val_ids[i]:03d} (grid position {grid_position_of(val_ids[i])})")
print(f"Saved evaluation to {OUTPUT_IMAGE}")