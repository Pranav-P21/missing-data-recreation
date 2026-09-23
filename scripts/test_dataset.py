import sys, os
sys.path.append(os.path.abspath("."))

from src.data.dataset import SatellitePatchDataset

dataset = SatellitePatchDataset(
    patches_dir="data/patches",
    damaged_dir="data/damaged",
    masks_dir="data/masks",
    num_patches=15,
)

print("Dataset length:", len(dataset))

x, y = dataset[0]
print("Input shape:", x.shape)
print("Target shape:", y.shape)
print("Input dtype:", x.dtype, "Target dtype:", y.dtype)