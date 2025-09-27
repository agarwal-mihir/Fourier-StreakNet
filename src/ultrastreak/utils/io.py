"""Input/Output utilities for image processing and file handling."""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_tensor_as_image(tensor: torch.Tensor,
                        save_path: Union[str, Path],
                        normalize: bool = True) -> None:
    """Save a tensor as an image file.

    Args:
        tensor: Input tensor
        save_path: Path to save the image
        normalize: Whether to normalize tensor values to [0, 255]
    """
    ensure_dir(Path(save_path).parent)

    # Convert tensor to numpy
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension

    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)  # Remove channel dimension for grayscale

    image_array = tensor.detach().cpu().numpy()

    if normalize:
        image_array = (image_array * 255).astype(np.uint8)

    # Save image
    if image_array.ndim == 2:
        Image.fromarray(image_array, mode='L').save(save_path)
    else:
        Image.fromarray(image_array).save(save_path)


def load_image_as_tensor(image_path: Union[str, Path],
                        input_size: Optional[Tuple[int, int]] = None,
                        normalize: bool = True) -> torch.Tensor:
    """Load an image as a tensor.

    Args:
        image_path: Path to the image
        input_size: Target size for resizing
        normalize: Whether to normalize to [0, 1]

    Returns:
        Image tensor
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale

    if input_size is not None:
        image = image.resize(input_size, Image.BILINEAR)

    tensor = torch.from_numpy(np.array(image)).float()

    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)  # Add channel dimension

    if normalize:
        tensor = tensor / 255.0

    return tensor.unsqueeze(0)  # Add batch dimension


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   metrics: dict,
                   filepath: Union[str, Path]) -> None:
    """Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
    """
    ensure_dir(Path(filepath).parent)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: Union[str, Path],
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: str = 'cpu') -> dict:
    """Load model checkpoint.

    Args:
        filepath: Path to checkpoint
        model: PyTorch model
        optimizer: Optimizer (optional)
        device: Device to load on

    Returns:
        Dictionary containing checkpoint info
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def get_file_list(directory: Union[str, Path],
                 extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.tif')) -> list:
    """Get list of files with specified extensions from directory.

    Args:
        directory: Directory path
        extensions: File extensions to include

    Returns:
        List of file paths
    """
    directory = Path(directory)
    files = []

    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))

    return sorted(files)


def create_output_dirs(base_dir: Union[str, Path],
                      subdirs: list = None) -> dict:
    """Create output directory structure.

    Args:
        base_dir: Base output directory
        subdirs: List of subdirectories to create

    Returns:
        Dictionary mapping subdir names to paths
    """
    if subdirs is None:
        subdirs = ['checkpoints', 'logs', 'visualizations', 'results']

    base_path = ensure_dir(base_dir)
    dirs = {}

    for subdir in subdirs:
        dirs[subdir] = ensure_dir(base_path / subdir)

    return dirs


