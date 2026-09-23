import sys, os
sys.path.append(os.path.abspath("."))

import numpy as np
from pystac_client import Client
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

bbox = [72.8062, 19.0084, 72.9492, 19.1436]
CLIP_MAX = 10000.0
patch_size = 256

scene_ids = [
    "S2B_42QZG_20260210_0_L2A",
    "S2C_42QZG_20260225_0_L2A",
    "S2C_42QZG_20260215_0_L2A",
    "S2C_42QZG_20260116_0_L2A",
    "S2B_42QZG_20251212_0_L2A",
]

catalog = Client.open("https://earth-search.aws.element84.com/v1")
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32642", always_xy=True)
min_x, min_y = transformer.transform(bbox[0], bbox[1])
max_x, max_y = transformer.transform(bbox[2], bbox[3])

os.makedirs("data/patches", exist_ok=True)

patch_counter = 0

for scene_id in scene_ids:
    print(f"Processing scene: {scene_id}")

    search = catalog.search(collections=["sentinel-2-l2a"], ids=[scene_id])
    item = next(search.items())

    band_url = item.assets["red"].href

    with rasterio.open(band_url) as src:
        window = from_bounds(min_x, min_y, max_x, max_y, transform=src.transform)
        data = src.read(1, window=window)

    normalized = np.clip(data, 0, CLIP_MAX) / CLIP_MAX

    h, w = normalized.shape
    n_rows = h // patch_size
    n_cols = w // patch_size

    for i in range(n_rows):
        for j in range(n_cols):
            y0, x0 = i * patch_size, j * patch_size
            patch = normalized[y0:y0+patch_size, x0:x0+patch_size]
            np.save(f"data/patches/patch_{patch_counter:03d}.npy", patch)
            patch_counter += 1

    print(f"  -> total patches so far: {patch_counter}")

print(f"\nDone. Total patches saved: {patch_counter}")