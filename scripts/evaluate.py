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

# --- average PSNR over full val set ---
total_psnr = 0.0
with torch.no_grad():
    for x, y in val_dataset:
        pred = model(x.unsqueeze(0))
        total_psnr += psnr(pred, y.unsqueeze(0))
avg_psnr = total_psnr / len(val_dataset)
print(f"Average PSNR over {len(val_dataset)} val patches: {avg_psnr:.2f} dB")

# --- 3 randomly selected val patches, shown side by side ---
random.seed()  # true randomness this time, not seed 42 - we want variety, not reproducibility here
sample_indices = random.sample(range(len(val_dataset)), min(3, len(val_dataset)))

fig, axes = plt.subplots(len(sample_indices), 4, figsize=(16, 4 * len(sample_indices)))

for row, i in enumerate(sample_indices):
    x, y = val_dataset[i]
    with torch.no_grad():
        pred = model(x.unsqueeze(0))

    damaged_img = x[0].numpy()
    mask_img = x[1].numpy()
    clean_img = y[0].numpy()
    pred_img = pred[0, 0].numpy()
    patch_psnr = psnr(pred, y.unsqueeze(0))

    titles = ["Original (ground truth)", "Damaged input", "Mask", f"Prediction ({patch_psnr:.2f} dB)"]
    imgs = [clean_img, damaged_img, mask_img, pred_img]

    for col in range(4):
        ax = axes[row, col] if len(sample_indices) > 1 else axes[col]
        ax.imshow(imgs[col], cmap="gray")
        ax.set_title(titles[col])
        if col == 0:
            ax.set_ylabel(f"patch_{val_ids[i]:03d} (grid_pos={grid_position_of(val_ids[i])})", fontsize=10)

plt.suptitle(f"3 random held-out val patches - avg PSNR: {avg_psnr:.2f} dB")
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE)
plt.show()

print(f"Saved evaluation to {OUTPUT_IMAGE}")