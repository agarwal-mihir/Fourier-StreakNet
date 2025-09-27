"""Basic UNet architecture for image segmentation - matches original working implementation."""

import torch
import torch.nn as nn


class UNet(nn.Module):
    """Standard UNet architecture for image segmentation.
    
    This implementation matches the original working code structure.
    """

    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.encoder_conv1 = self.conv_block(in_channels, 64)
        self.encoder_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv2 = self.conv_block(64, 128)
        self.encoder_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv3 = self.conv_block(128, 256)
        self.encoder_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv4 = self.conv_block(256, 512)
        self.encoder_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_conv = self.conv_block(512, 1024)

        # Decoder
        self.decoder_upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_conv4 = self.conv_block(1024, 512)
        self.decoder_upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_conv3 = self.conv_block(512, 256)
        self.decoder_upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_conv2 = self.conv_block(256, 128)
        self.decoder_upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv1 = self.conv_block(128, 64)

        # Output layer
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass through the UNet."""
        # Encoder
        encoder1 = self.encoder_conv1(x)
        encoder2 = self.encoder_conv2(self.encoder_pool1(encoder1))
        encoder3 = self.encoder_conv3(self.encoder_pool2(encoder2))
        encoder4 = self.encoder_conv4(self.encoder_pool3(encoder3))

        # Bottleneck
        bottleneck = self.bottleneck_conv(self.encoder_pool4(encoder4))

        # Decoder
        decoder4 = self.decoder_upconv4(bottleneck)
        decoder4 = torch.cat((encoder4, decoder4), dim=1)
        decoder4 = self.decoder_conv4(decoder4)

        decoder3 = self.decoder_upconv3(decoder4)
        decoder3 = torch.cat((encoder3, decoder3), dim=1)
        decoder3 = self.decoder_conv3(decoder3)

        decoder2 = self.decoder_upconv2(decoder3)
        decoder2 = torch.cat((encoder2, decoder2), dim=1)
        decoder2 = self.decoder_conv2(decoder2)

        decoder1 = self.decoder_upconv1(decoder2)
        decoder1 = torch.cat((encoder1, decoder1), dim=1)
        decoder1 = self.decoder_conv1(decoder1)

        # Output layer (raw logits)
        output = self.output_conv(decoder1)
        return output

    def conv_block(self, in_channels, out_channels):
        """Create a convolutional block with two conv layers and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )


def create_unet(in_channels=1, out_channels=1, **kwargs):
    """Factory function to create a UNet model."""
    return UNet(in_channels=in_channels, out_channels=out_channels)