import torch
import sys
import os

# Allow importing from src/
sys.path.append(os.path.abspath("."))

from src.models.unet import UNet


# Create model
model = UNet(
    input_channels=2,
    output_channels=1,
    base_features=32
)

print(model)


# Fake input
x = torch.randn(
    1,              # batch size
    2,              # damaged image + mask
    256,
    256
)


# Forward pass
with torch.no_grad():
    output = model(x)


print("\nInput shape: ", x.shape)
print("Output shape:", output.shape)