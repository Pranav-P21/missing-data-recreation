import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.masking import random_augment


class SatellitePatchDataset(Dataset):
    def __init__(self, patches_dir, damaged_dir, masks_dir, patch_ids, augment=False):
        self.patches_dir = patches_dir
        self.damaged_dir = damaged_dir
        self.masks_dir = masks_dir
        self.patch_ids = patch_ids
        self.augment = augment

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, i):
        idx = self.patch_ids[i]
        clean = np.load(f"{self.patches_dir}/patch_{idx:03d}.npy")
        damaged = np.load(f"{self.damaged_dir}/damaged_{idx:03d}.npy")
        mask = np.load(f"{self.masks_dir}/mask_{idx:03d}.npy")

        if self.augment:
            damaged, mask, clean = random_augment(damaged, mask, clean)

        input_tensor = np.stack([damaged, mask], axis=0)
        target_tensor = clean[np.newaxis, :, :]

        return (
            torch.from_numpy(input_tensor.copy()).float(),
            torch.from_numpy(target_tensor.copy()).float(),
        )