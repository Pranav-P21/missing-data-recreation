import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolutions with ReLU activation.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        input_channels=2,
        output_channels=1,
        base_features=32
    ):
        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------

        self.enc1 = DoubleConv(
            input_channels,
            base_features
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(
            base_features,
            base_features * 2
        )

        self.pool2 = nn.MaxPool2d(2)

        # -------------------------
        # Bottleneck
        # -------------------------

        self.bottleneck = DoubleConv(
            base_features * 2,
            base_features * 4
        )

        # -------------------------
        # Decoder
        # -------------------------

        self.up2 = nn.ConvTranspose2d(
            base_features * 4,
            base_features * 2,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            base_features * 4,
            base_features * 2
        )

        self.up1 = nn.ConvTranspose2d(
            base_features * 2,
            base_features,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            base_features * 2,
            base_features
        )

        # -------------------------
        # Output
        # -------------------------

        self.output = nn.Conv2d(
            base_features,
            output_channels,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool2(e2)
        )

        # Decoder
        d2 = self.up2(b)

        # Skip connection
        d2 = torch.cat([d2, e2], dim=1)

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        # Skip connection
        d1 = torch.cat([d1, e1], dim=1)

        d1 = self.dec1(d1)

        # Output
        return self.output(d1)