import torch.nn as nn
from pytorch_msssim import SSIM


class CombinedLoss(nn.Module):
    def __init__(self, w_mse=0.4, w_l1=0.3, w_ssim=0.3):
        super().__init__()
        self.w_mse = w_mse
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=1)

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        ssim_score = self.ssim(pred, target)   # 1.0 = identical, 0.0 = totally different
        ssim_loss = 1 - ssim_score              # convert to a loss (lower = better)
        return self.w_mse * mse_loss + self.w_l1 * l1_loss + self.w_ssim * ssim_loss