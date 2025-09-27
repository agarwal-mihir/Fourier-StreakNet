"""UNet with Fourier attention for ultrasound streak restoration."""

import torch
import torch.nn as nn
from .blocks.unet_blocks import DoubleConv, Down, Up
from .blocks.attention import FourierAttentionLayer


class UNetWithFourierAttention(nn.Module):
    """UNet architecture with Fourier attention layers for enhanced feature processing.

    This model incorporates Fourier attention layers in the bottleneck to better
    handle frequency-domain characteristics of ultrasound images and streaks.
    """

    def __init__(self, n_channels=2, n_classes=1, bilinear=True):
        """Initialize the UNet with Fourier attention.

        Args:
            n_channels: Number of input channels (e.g., 2 for image + mask)
            n_classes: Number of output channels
            bilinear: Whether to use bilinear upsampling
        """
        super(UNetWithFourierAttention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # Bottleneck with Fourier Attention Layers
        self.fal1 = FourierAttentionLayer(512, 512)
        self.fal2 = FourierAttentionLayer(512, 512)
        self.fal3 = FourierAttentionLayer(512, 512)

        # Decoder
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, n_channels, height, width)

        Returns:
            Output tensor of shape (batch, n_classes, height, width)
        """
        # Encoder
        x1 = self.inc(x)    # [B, 64, 256, 256]
        x2 = self.down1(x1) # [B, 128, 128, 128]
        x3 = self.down2(x2) # [B, 256, 64, 64]
        x4 = self.down3(x3) # [B, 512, 32, 32]
        x5 = self.down4(x4) # [B, 512, 16, 16]

        # Bottleneck with Fourier Attention Layers
        x5 = self.fal1(x5)
        x5 = self.fal2(x5)
        x5 = self.fal3(x5)

        # Decoder
        x = self.up1(x5, x4)  # [B, 256, 32, 32]
        x = self.up2(x, x3)   # [B, 128, 64, 64]
        x = self.up3(x, x2)   # [B, 64, 128, 128]
        x = self.up4(x, x1)   # [B, 64, 256, 256]
        logits = self.outc(x) # [B, 1, 256, 256]
        return logits


def create_fourier_attention_unet(n_channels=2, n_classes=1, **kwargs):
    """Factory function to create a UNet with Fourier attention model."""
    return UNetWithFourierAttention(n_channels=n_channels, n_classes=n_classes, **kwargs)


