"""Self-attention and Fourier attention blocks."""

import torch
import torch.nn as nn
from .fourier import FourierUnit


class SelfAttention2D(nn.Module):
    """2D self-attention mechanism."""

    def __init__(self, in_channels):
        super(SelfAttention2D, self).__init__()
        self.in_channels = in_channels

        # Query, Key, Value convolutions
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Apply self-attention to input tensor."""
        batch_size, C, H, W = x.size()

        # Compute queries, keys, and values
        proj_query = self.query_conv(x).view(batch_size, -1, H*W).permute(0, 2, 1)  # B, N, C'
        proj_key = self.key_conv(x).view(batch_size, -1, H*W)  # B, C', N
        energy = torch.bmm(proj_query, proj_key)  # B, N, N
        attention = self.softmax(energy)  # B, N, N

        proj_value = self.value_conv(x).view(batch_size, -1, H*W)  # B, C, N

        # Apply attention
        out = torch.bmm(attention, proj_value.permute(0, 2, 1))  # B, N, C
        out = out.permute(0, 2, 1).view(batch_size, C, H, W)

        # Residual connection
        out = self.gamma * out + x
        return out


class FourierAttentionLayer(nn.Module):
    """Layer combining self-attention with Fourier transforms."""

    def __init__(self, in_channels, out_channels, groups=1, fft_norm='ortho'):
        super(FourierAttentionLayer, self).__init__()
        self.attention = SelfAttention2D(in_channels)
        self.fourier_unit = FourierUnit(
            in_channels, out_channels, groups=groups, fft_norm=fft_norm
        )

    def forward(self, x):
        """Apply attention followed by Fourier processing."""
        # Apply attention
        x = self.attention(x)
        # Apply Fourier transform
        x = self.fourier_unit(x)
        return x


