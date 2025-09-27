"""Visualization utilities for model predictions and results."""

import os
from pathlib import Path
from typing import Optional, Union, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def tensor_to_numpy(tensor: torch.Tensor,
                   denormalize: bool = True,
                   to_uint8: bool = True) -> np.ndarray:
    """Convert tensor to numpy array for visualization.

    Args:
        tensor: Input tensor
        denormalize: Whether to denormalize from [-1,1] or [0,1] to [0,1]
        to_uint8: Whether to convert to uint8 (0-255 range)

    Returns:
        Numpy array ready for visualization
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Convert to numpy
    array = tensor.detach().cpu().numpy()

    # Denormalize if needed
    if denormalize:
        if array.min() < 0:  # Assume [-1, 1] normalization
            array = (array + 1) / 2
        array = np.clip(array, 0, 1)

    # Convert to uint8 if requested
    if to_uint8:
        array = (array * 255).astype(np.uint8)

    return array


def create_comparison_grid(images: list,
                          titles: list,
                          figsize: Tuple[int, int] = (15, 10),
                          save_path: Optional[Union[str, Path]] = None) -> None:
    """Create a grid comparison of multiple images.

    Args:
        images: List of images (numpy arrays or tensors)
        titles: List of titles for each image
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    n_images = len(images)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // n_cols
        col = i % n_cols

        if row < len(axes) and col < len(axes[row]):
            ax = axes[row][col]

            # Convert tensor to numpy if needed
            if torch.is_tensor(img):
                img = tensor_to_numpy(img)

            # Handle different channel configurations
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            elif img.ndim == 3:
                if img.shape[0] == 1:  # Single channel
                    ax.imshow(img[0], cmap='gray')
                elif img.shape[0] == 3:  # RGB
                    ax.imshow(img.transpose(1, 2, 0))
                else:  # Other configurations
                    ax.imshow(img[0], cmap='gray')
            else:
                ax.imshow(img)

            ax.set_title(title)
            ax.axis('off')

    # Hide unused subplots
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if row < len(axes) and col < len(axes[row]):
            axes[row][col].axis('off')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_segmentation_results(input_image: Union[np.ndarray, torch.Tensor],
                            mask: Union[np.ndarray, torch.Tensor],
                            prediction: Union[np.ndarray, torch.Tensor],
                            save_path: Optional[Union[str, Path]] = None) -> None:
    """Plot segmentation results with input, ground truth mask, and prediction.

    Args:
        input_image: Input image
        mask: Ground truth mask
        prediction: Model prediction
        save_path: Path to save the figure (optional)
    """
    images = [input_image, mask, prediction]
    titles = ['Input Image', 'Ground Truth Mask', 'Prediction']

    create_comparison_grid(images, titles, figsize=(15, 5), save_path=save_path)


def plot_restoration_results(input_image: Union[np.ndarray, torch.Tensor],
                           mask: Union[np.ndarray, torch.Tensor],
                           prediction: Union[np.ndarray, torch.Tensor],
                           ground_truth: Union[np.ndarray, torch.Tensor],
                           save_path: Optional[Union[str, Path]] = None) -> None:
    """Plot restoration results with input, mask, prediction, and ground truth.

    Args:
        input_image: Input image with artifacts
        mask: Artifact mask
        prediction: Restored image
        ground_truth: Ground truth clean image
        save_path: Path to save the figure (optional)
    """
    images = [input_image, mask, prediction, ground_truth]
    titles = ['Input (with artifacts)', 'Artifact Mask', 'Restored Image', 'Ground Truth']

    create_comparison_grid(images, titles, figsize=(20, 5), save_path=save_path)


def plot_training_history(train_losses: list,
                         val_losses: list,
                         metrics: Optional[dict] = None,
                         save_path: Optional[Union[str, Path]] = None) -> None:
    """Plot training history including losses and metrics.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        metrics: Dictionary of additional metrics to plot
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot metrics if provided
    if metrics:
        axes[1].set_title('Training Metrics')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Value')
        axes[1].grid(True)

        for metric_name, metric_values in metrics.items():
            if len(metric_values) == len(epochs):
                axes[1].plot(epochs, metric_values, label=metric_name)

        axes[1].legend()
    else:
        axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def create_overlay_visualization(image: Union[np.ndarray, torch.Tensor],
                               mask: Union[np.ndarray, torch.Tensor],
                               prediction: Optional[Union[np.ndarray, torch.Tensor]] = None,
                               alpha: float = 0.5,
                               save_path: Optional[Union[str, Path]] = None) -> None:
    """Create overlay visualization of mask/prediction on image.

    Args:
        image: Base image
        mask: Mask to overlay
        prediction: Optional prediction to overlay
        alpha: Transparency of overlay
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 3 if prediction is not None else 2, figsize=(15, 5))

    # Convert tensors to numpy
    if torch.is_tensor(image):
        image = tensor_to_numpy(image, to_uint8=True)
    if torch.is_tensor(mask):
        mask = tensor_to_numpy(mask, to_uint8=True)
    if prediction is not None and torch.is_tensor(prediction):
        prediction = tensor_to_numpy(prediction, to_uint8=True)

    # Plot base image
    axes[0].imshow(image if image.ndim == 2 else image.transpose(1, 2, 0))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot mask overlay
    axes[1].imshow(image if image.ndim == 2 else image.transpose(1, 2, 0))
    mask_overlay = np.zeros_like(image)
    if image.ndim == 2:
        mask_overlay[mask > 0] = [255, 0, 0]  # Red for mask
    else:
        mask_overlay[mask > 0] = [255, 0, 0]  # Red for mask
    axes[1].imshow(mask_overlay, alpha=alpha)
    axes[1].set_title('Mask Overlay')
    axes[1].axis('off')

    # Plot prediction overlay if provided
    if prediction is not None:
        axes[2].imshow(image if image.ndim == 2 else image.transpose(1, 2, 0))
        pred_overlay = np.zeros_like(image)
        if image.ndim == 2:
            pred_overlay[prediction > 0] = [0, 255, 0]  # Green for prediction
        else:
            pred_overlay[prediction > 0] = [0, 255, 0]  # Green for prediction
        axes[2].imshow(pred_overlay, alpha=alpha)
        axes[2].set_title('Prediction Overlay')
        axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def save_image_grid(
    images: List[Union[np.ndarray, torch.Tensor, Image.Image]],
    titles: List[str],
    save_path: Union[str, Path],
    figsize: Tuple[int, int] = (12, 12),
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    dpi: int = 150,
    normalize: bool = False
) -> None:
    """Save a grid of images with titles.
    
    Args:
        images: List of images (numpy arrays, torch tensors, or PIL Images)
        titles: List of titles for each image
        save_path: Path to save the figure
        figsize: Figure size (width, height)
        rows: Number of rows (auto-calculated if None)
        cols: Number of columns (auto-calculated if None)
        dpi: DPI for the saved image
        normalize: Whether to normalize images to 0-255 range
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate grid layout
    n_images = len(images)
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    elif rows is None:
        rows = int(np.ceil(n_images / cols))
    elif cols is None:
        cols = int(np.ceil(n_images / rows))
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    # Plot each image
    for i, (img, title) in enumerate(zip(images, titles)):
        # Convert to numpy array
        if torch.is_tensor(img):
            img = img.cpu().numpy()
            if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                img = img.transpose(1, 2, 0)  # CHW -> HWC
        elif isinstance(img, Image.Image):
            img = np.array(img)
        
        # Normalize if requested
        if normalize and img.size > 0:
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        
        # Display image
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            axes[i].imshow(img.squeeze(), cmap='gray', vmin=0, vmax=255)
        else:
            axes[i].imshow(img)
        
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def create_comparison_plot(
    original: Union[np.ndarray, torch.Tensor, Image.Image],
    segmentation_mask: Union[np.ndarray, torch.Tensor, Image.Image],
    restored: Union[np.ndarray, torch.Tensor, Image.Image],
    ground_truth: Optional[Union[np.ndarray, torch.Tensor, Image.Image]] = None,
    save_path: Union[str, Path] = "comparison.png",
    figsize: Tuple[int, int] = (12, 10),
    normalize_restored: bool = True
) -> None:
    """Create a comparison plot for the complete pipeline.
    
    Args:
        original: Original input image
        segmentation_mask: Predicted segmentation mask
        restored: Restored output image
        ground_truth: Ground truth mask (optional)
        save_path: Path to save the figure
        figsize: Figure size (width, height)
        normalize_restored: Whether to normalize the restored image
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert all images to numpy arrays
    def to_numpy(img):
        if torch.is_tensor(img):
            img = img.cpu().numpy()
            if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                img = img.transpose(1, 2, 0)
        elif isinstance(img, Image.Image):
            img = np.array(img)
        return img
    
    original_np = to_numpy(original)
    segmentation_np = to_numpy(segmentation_mask)
    restored_np = to_numpy(restored)
    
    # Normalize restored image if requested
    if normalize_restored:
        restored_np = ((restored_np - restored_np.min()) / (restored_np.max() - restored_np.min()) * 255).astype(np.uint8)
    
    # Create figure
    if ground_truth is not None:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        gt_np = to_numpy(ground_truth)
        images = [original_np, segmentation_np, restored_np, gt_np]
        titles = ['Original Image', 'Segmentation Mask', 'Restored Image', 'Ground Truth Mask']
    else:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes = np.array(axes)
        
        images = [original_np, segmentation_np, restored_np]
        titles = ['Original Image', 'Segmentation Mask', 'Restored Image']
    
    # Plot each image
    for i, (img, title) in enumerate(zip(images, titles)):
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            axes[i].imshow(img.squeeze(), cmap='gray', vmin=0, vmax=255)
        else:
            axes[i].imshow(img)
        
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_restoration_comparison(
    original: Union[np.ndarray, torch.Tensor, Image.Image],
    restored: Union[np.ndarray, torch.Tensor, Image.Image],
    save_path: Union[str, Path] = "restoration_comparison.png",
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """Create a comparison showing original vs restored images.
    
    Args:
        original: Original input image
        restored: Restored output image
        save_path: Path to save the figure
        figsize: Figure size (width, height)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy arrays
    def to_numpy(img):
        if torch.is_tensor(img):
            img = img.cpu().numpy()
            if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                img = img.transpose(1, 2, 0)
        elif isinstance(img, Image.Image):
            img = np.array(img)
        return img
    
    original_np = to_numpy(original)
    restored_np = to_numpy(restored)
    restored_normalized = ((restored_np - restored_np.min()) / (restored_np.max() - restored_np.min()) * 255).astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot images
    axes[0].imshow(original_np.squeeze(), cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(restored_np.squeeze(), cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Restored Image (Original Range)')
    axes[1].axis('off')
    
    axes[2].imshow(restored_normalized.squeeze(), cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Restored Image (Normalized)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


