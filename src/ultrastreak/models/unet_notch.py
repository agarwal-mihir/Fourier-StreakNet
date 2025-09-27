"""UNet with integrated notch filter preprocessing for enhanced streak segmentation."""

import cv2
import numpy as np
import torch
import torch.nn as nn
from .unet import UNet


class CenterNotchFilter:
    """Center notch filter for frequency domain processing."""

    def __init__(self, width=10):
        """Initialize centered notch filter parameters.

        Args:
            width: width of the notch filter
        """
        self.width = width

    def apply(self, image):
        """Apply notch filter centered on the image.

        Args:
            image: Input image as numpy array

        Returns:
            Filtered image
        """
        # Convert to grayscale if image is RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Convert to float and normalize
        gray = gray.astype(np.float32) / 255.0

        # Take Fourier transform
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

        # Create notch filter mask
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        # Create the filter - Centered Notch Filter
        mask = np.ones((rows, cols), np.float32)

        # Create a circle mask centered on the image
        y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
        radius = np.sqrt(x**2 + y**2)

        # Apply center notch (remove low frequencies in the center)
        center_mask = radius < self.width
        mask[center_mask] = 0

        # Apply mask and perform inverse FFT
        fshift_filtered = fshift * mask
        f_filtered = np.fft.ifftshift(fshift_filtered)
        img_filtered = np.fft.ifft2(f_filtered)
        img_filtered = np.abs(img_filtered)

        # Normalize the filtered image
        img_filtered = (img_filtered - img_filtered.min()) / (img_filtered.max() - img_filtered.min())

        return img_filtered


class UNetWithNotchFilter(nn.Module):
    """UNet architecture with integrated notch filter preprocessing.

    This model applies a center notch filter to input images before
    passing them through the UNet for improved streak segmentation.
    """

    def __init__(self, in_channels=1, out_channels=1, notch_width=10, features=(64, 128, 256, 512)):
        super(UNetWithNotchFilter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.notch_filter = CenterNotchFilter(width=notch_width)
        self.unet = UNet(in_channels=2, out_channels=out_channels, features=features)  # 2 channels: original + filtered

    def forward(self, x):
        """Forward pass with notch filter preprocessing.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Model predictions
        """
        batch_size = x.shape[0]

        # Apply notch filter to each image in batch
        filtered_images = []
        for i in range(batch_size):
            # Convert tensor to numpy for filtering
            img_np = x[i].cpu().numpy()
            if self.in_channels == 1:
                img_np = img_np[0]  # Remove channel dimension for grayscale
            else:
                img_np = np.transpose(img_np, (1, 2, 0))  # CHW to HWC

            # Apply notch filter
            filtered = self.notch_filter.apply(img_np)

            # Convert back to tensor
            if self.in_channels == 1:
                filtered_tensor = torch.from_numpy(filtered).unsqueeze(0)  # Add channel dimension
            else:
                filtered_tensor = torch.from_numpy(np.transpose(filtered, (2, 0, 1)))  # HWC to CHW

            filtered_images.append(filtered_tensor)

        # Stack filtered images
        filtered_batch = torch.stack(filtered_images).to(x.device)

        # Concatenate original and filtered images
        combined_input = torch.cat([x, filtered_batch], dim=1)

        # Pass through UNet
        return self.unet(combined_input)


def create_unet_notch(in_channels=1, out_channels=1, notch_width=10, **kwargs):
    """Factory function to create a UNet with notch filter model."""
    return UNetWithNotchFilter(
        in_channels=in_channels,
        out_channels=out_channels,
        notch_width=notch_width,
        **kwargs
    )


