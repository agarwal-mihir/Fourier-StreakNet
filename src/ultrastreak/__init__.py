"""Ultrasound streak removal using deep learning.

This package provides tools for:
- Generating synthetic streak data for training
- Training UNet models for streak segmentation
- Training Fourier attention networks for streak restoration
- Running inference pipelines on ultrasound images
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from . import cli, models, utils

__all__ = ["cli", "models", "utils"]
