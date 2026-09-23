import sys
print("Using Python at:", sys.executable)

from pystac_client import Client
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds

print("Starting search...")

bbox = [72.8062, 19.0084, 72.9492, 19.1436]  # Mumbai, 15km box

catalog = Client.open("https://earth-search.aws.element84.com/v1")

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2025-12-01/2026-02-28",
    query={"eo:cloud_cover": {"lt": 10}},
    limit=10,
)

items = list(search.items())
print(f"Found {len(items)} items")

for item in items:
    print(item.id, item.properties["datetime"], item.properties["eo:cloud_cover"])

# --- pick one scene and inspect the band ---
target_id = "S2B_42QZG_20260210_0_L2A"
item = next(i for i in items if i.id == target_id)

print("Available assets:", list(item.assets.keys()))

band_url = item.assets["red"].href
print("Band URL:", band_url)

with rasterio.open(band_url) as src:
    print("bands:", src.count)
    print("dtype:", src.dtypes)
    print("nodata:", src.nodata)
    print("crs:", src.crs)
    print("size:", src.width, src.height)
    print("bounds:", src.bounds)

# --- Step 4: reproject bbox and crop to it ---
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32642", always_xy=True)
min_x, min_y = transformer.transform(bbox[0], bbox[1])
max_x, max_y = transformer.transform(bbox[2], bbox[3])

with rasterio.open(band_url) as src:
    window = from_bounds(min_x, min_y, max_x, max_y, transform=src.transform)
    data = src.read(1, window=window)
    print("Cropped shape:", data.shape)
    print("Min/max pixel value:", data.min(), data.max())

nodata_count = (data == 0).sum()
total_pixels = data.size
print(f"NoData pixels: {nodata_count} / {total_pixels} ({100*nodata_count/total_pixels:.4f}%)")

import numpy as np

CLIP_MAX = 10000.0
normalized = np.clip(data, 0, CLIP_MAX) / CLIP_MAX
print("Normalized min/max:", normalized.min(), normalized.max())

patch_size = 256
h, w = normalized.shape

n_rows = h // patch_size
n_cols = w // patch_size

patches = []
for i in range(n_rows):
    for j in range(n_cols):
        y0 = i * patch_size
        x0 = j * patch_size
        patch = normalized[y0:y0+patch_size, x0:x0+patch_size]
        patches.append(patch)

print(f"Extracted {len(patches)} patches of shape {patches[0].shape}")

import os

save_dir = "data/patches"
os.makedirs(save_dir, exist_ok=True)

for idx, patch in enumerate(patches):
    save_path = os.path.join(save_dir, f"patch_{idx:03d}.npy")
    np.save(save_path, patch)

print(f"Saved {len(patches)} patches to {save_dir}")

import matplotlib.pyplot as plt

plt.imshow(data, cmap="gray")
plt.title(f"Band B04 - {target_id}")
plt.colorbar(label="Reflectance (uint16)")
plt.savefig("notebooks/patch_preview.png")
plt.show()