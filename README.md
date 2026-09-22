# Satellite Missing-Data Reconstruction — V1 Prototype

Machine learning prototype for reconstructing missing satellite pixel values,
inspired by the ISRO NM391 problem. V1 uses a small regional area, one
Sentinel-2 band, 256x256 patches, synthetic missing-data masks, and a
U-Net reconstruction model.

## Project structure

- `configs/` — experiment configuration (patch size, model, training params)
- `data/` — raw, processed, and patched data (not tracked in git)
- `src/data/` — dataset loading, patching, preprocessing, masking
- `src/models/` — U-Net and building blocks
- `src/training/` — training loop and loss functions
- `src/evaluation/` — metrics and visualization
- `src/utils/` — shared utilities (e.g. seeding)
- `scripts/` — entry points: `prepare_data.py`, `train.py`, `evaluate.py`
- `notebooks/` — exploration and inspection only, not pipeline code
- `experiments/` — logged results per run

## Status

Dataset selected: Sentinel-2 L2A (AWS Earth Search STAC), Mumbai region,
15km x 15km box, band B04, dry-season low-cloud scene. Data loading and
inspection not yet implemented.
