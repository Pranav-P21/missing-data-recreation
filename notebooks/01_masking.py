import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def apply_random_block_mask(patch, rng, n_holes=(1, 3), hole_size=(20, 80)):
    mask = np.ones_like(patch)
    n = rng.integers(n_holes[0], n_holes[1] + 1)
    for _ in range(n):
        h = rng.integers(hole_size[0], hole_size[1])
        w = rng.integers(hole_size[0], hole_size[1])
        y0 = rng.integers(0, patch.shape[0] - h)
        x0 = rng.integers(0, patch.shape[1] - w)
        mask[y0:y0+h, x0:x0+w] = 0
    damaged = patch * mask
    return damaged, mask


# --- find how many patches actually exist ---
num_patches = len(glob.glob("data/patches/patch_*.npy"))
print(f"Found {num_patches} patches")

# --- single-patch visual test (kept for reference) ---
patch = np.load("data/patches/patch_005.npy")
print("Loaded patch shape:", patch.shape)

rng_test = np.random.default_rng(seed=42)
hole_h, hole_w = 60, 60
y0 = rng_test.integers(0, patch.shape[0] - hole_h)
x0 = rng_test.integers(0, patch.shape[1] - hole_w)
mask = np.ones_like(patch)
mask[y0:y0+hole_h, x0:x0+hole_w] = 0
damaged = patch * mask

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(patch, cmap="gray"); axes[0].set_title("Original")
axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Mask (1=keep, 0=missing)")
axes[2].imshow(damaged, cmap="gray"); axes[2].set_title("Damaged")
plt.savefig("notebooks/masking_preview.png")
plt.show()

# --- apply masking to ALL patches and save ---
os.makedirs("data/damaged", exist_ok=True)
os.makedirs("data/masks", exist_ok=True)

rng = np.random.default_rng(seed=42)

for idx in range(num_patches):
    try:
        print(f"Processing patch {idx}...")
        p = np.load(f"data/patches/patch_{idx:03d}.npy")
        dmg, m = apply_random_block_mask(p, rng)
        np.save(f"data/damaged/damaged_{idx:03d}.npy", dmg)
        np.save(f"data/masks/mask_{idx:03d}.npy", m)
    except Exception as e:
        print(f"ERROR on patch {idx}: {e}")
        break

print(f"Saved damaged patches and masks for all {num_patches} patches")