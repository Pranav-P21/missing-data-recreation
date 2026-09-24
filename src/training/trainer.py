import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def psnr(pred, target, max_val=1.0):
    mse = ((pred - target) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(max_val) - 10 * math.log10(mse)


def train_model(model, train_dataset, val_dataset, epochs=20, batch_size=4, lr=1e-3, device="cpu"):
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    from src.training.losses import CombinedLoss
    criterion = CombinedLoss(alpha=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
                val_psnr += psnr(pred, y)
        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f} - Val: {val_loss:.6f} - PSNR: {val_psnr:.2f} dB")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "experiments/unet_best.pth")

    torch.save(model.state_dict(), "experiments/unet_final.pth")
    return model