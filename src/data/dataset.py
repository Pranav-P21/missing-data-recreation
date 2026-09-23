import numpy as np
import torch
from torch.utils.data import Dataset


class SatellitePatchDataset(Dataset):
    def __init__(self, patches_dir, damaged_dir, masks_dir, num_patches):
        self.patches_dir = patches_dir
        self.damaged_dir = damaged_dir
        self.masks_dir = masks_dir
        self.num_patches = num_patches

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        clean = np.load(f"{self.patches_dir}/patch_{idx:03d}.npy")
        damaged = np.load(f"{self.damaged_dir}/damaged_{idx:03d}.npy")
        mask = np.load(f"{self.masks_dir}/mask_{idx:03d}.npy")

        # stack damaged + mask into a 2-channel input
        input_tensor = np.stack([damaged, mask], axis=0)   # shape: (2, H, W)
        target_tensor = clean[np.newaxis, :, :]              # shape: (1, H, W)

        return (
            torch.from_numpy(input_tensor).float(),
            torch.from_numpy(target_tensor).float(),
        )