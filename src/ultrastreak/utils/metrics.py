"""Metrics and evaluation utilities for model performance."""

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from sklearn.metrics import jaccard_score


def calculate_psnr_masked(input_image: np.ndarray,
                         output_image: np.ndarray,
                         mask: np.ndarray,
                         max_pixel_value: float = 255.0) -> float:
    """Compute PSNR for the masked pixels.

    Args:
        input_image: Input image (H x W or H x W x C)
        output_image: Output image (H x W or H x W x C)
        mask: Binary mask (H x W), where 1 indicates pixels to evaluate
        max_pixel_value: Maximum pixel value (e.g., 255 for 8-bit images)

    Returns:
        PSNR value for the masked region
    """
    # Ensure input arrays are numpy arrays
    input_image = np.asarray(input_image, dtype=np.float32)
    output_image = np.asarray(output_image, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)

    # Select masked pixels
    masked_input = input_image[mask == 1]
    masked_output = output_image[mask == 1]

    # Check if there are any masked pixels
    if len(masked_input) == 0:
        print("Warning: No masked pixels found for PSNR calculation")
        return float('nan')

    # Calculate MSE manually to handle edge cases
    mse = np.mean((masked_input - masked_output) ** 2)

    # Handle case where images are identical (MSE = 0)
    if mse == 0:
        print("Warning: MSE is 0 (identical images), returning high PSNR value")
        return 100.0

    # Calculate PSNR manually
    psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    # Cap PSNR at reasonable maximum to avoid infinity
    if np.isinf(psnr_value) or psnr_value > 100:
        print(f"Warning: PSNR calculated as {psnr_value}, capping at 100 dB")
        return 100.0

    return psnr_value


def calculate_mse_unmasked(input_image: np.ndarray,
                          output_image: np.ndarray,
                          mask: np.ndarray) -> float:
    """Compute MSE for the unmasked pixels.

    Args:
        input_image: Input image (H x W or H x W x C)
        output_image: Output image (H x W or H x W x C)
        mask: Binary mask (H x W), where 1 indicates masked regions and 0 indicates unmasked regions

    Returns:
        MSE value for the unmasked region
    """
    # Ensure inputs are numpy arrays
    input_image = np.asarray(input_image, dtype=np.float32)
    output_image = np.asarray(output_image, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)

    # Select unmasked pixels
    unmasked_input = input_image[mask == 0]
    unmasked_output = output_image[mask == 0]

    # Check if there are any unmasked pixels
    if len(unmasked_input) == 0:
        print("Warning: No unmasked pixels found for MSE calculation")
        return float('nan')

    # Compute MSE for unmasked region
    mse = np.mean((unmasked_input - unmasked_output) ** 2)
    return mse


def calculate_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate IoU (Intersection over Union) / Jaccard Index.

    Args:
        pred: Binary prediction tensor
        target: Binary target tensor

    Returns:
        IoU score
    """
    # Convert to binary
    pred_binary = (pred > 0.5).float()

    # Flatten
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()

    # Calculate IoU
    intersection = np.logical_and(pred_flat, target_flat).sum()
    union = np.logical_or(pred_flat, target_flat).sum()

    # Avoid division by zero
    if union == 0:
        return 1.0

    return intersection / union


def calculate_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Dice coefficient.

    Args:
        pred: Prediction tensor
        target: Target tensor

    Returns:
        Dice coefficient
    """
    smooth = 1.0

    # Convert prediction to binary
    pred_binary = (pred > 0.5).float()

    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()

    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Alias for calculate_dice for backward compatibility."""
    return calculate_dice(pred, target)


def seam_intensity_match(input_img: torch.Tensor,
                        pred_img: torch.Tensor,
                        mask: torch.Tensor,
                        ring_size: int = 5) -> torch.Tensor:
    """Adjust predicted intensities inside the masked region so they match the
    surrounding unmasked region along the seam.

    Args:
        input_img: [1,1,H,W] in [0,1]
        pred_img: [1,1,H,W] in [0,1]
        mask: [1,1,H,W] in {0,1}
        ring_size: Size of the ring around the mask for intensity matching

    Returns:
        Adjusted prediction tensor
    """
    img_np = input_img.squeeze().detach().cpu().numpy().astype(np.float32)
    pred_np = pred_img.squeeze().detach().cpu().numpy().astype(np.float32)
    mask_np = (mask.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8)

    if mask_np.sum() == 0:
        return pred_img

    k = np.ones((ring_size, ring_size), np.uint8)
    outer_ring = cv2.dilate(mask_np, k, iterations=1) - mask_np
    inner_ring = mask_np - cv2.erode(mask_np, k, iterations=1)

    outer_vals = img_np[outer_ring > 0]
    inner_vals = pred_np[inner_ring > 0]

    # If not enough pixels, fall back to simple gain matching using medians
    if len(outer_vals) < 50 or len(inner_vals) < 50:
        m_out = float(np.median(img_np[mask_np == 0]))
        m_in = float(np.median(pred_np[mask_np == 1]))
        gain = 1.0 if m_in == 0 else np.clip(m_out / m_in, 0.5, 2.0)
        bias = 0.0
    else:
        # Robust linear fit using medians for scale and bias
        m_out = float(np.median(outer_vals))
        m_in = float(np.median(inner_vals))
        mad_in = float(np.median(np.abs(inner_vals - m_in)) + 1e-6)
        # Scale to match spread; avoid over-scaling
        s_out = float(np.percentile(outer_vals, 90) - np.percentile(outer_vals, 10) + 1e-6)
        s_in = float(np.percentile(inner_vals, 90) - np.percentile(inner_vals, 10) + 1e-6)
        gain = np.clip(s_out / s_in, 0.5, 2.0)
        bias = np.clip(m_out - gain * m_in, -0.25, 0.25)

    adjusted = pred_img.clone()
    mask_bool = mask > 0.5
    adjusted[mask_bool] = torch.clamp(adjusted[mask_bool] * gain + bias, 0.0, 1.0)
    return adjusted


def evaluate_segmentation_metrics(pred: torch.Tensor,
                                target: torch.Tensor,
                                threshold: float = 0.5) -> dict:
    """Evaluate segmentation metrics.

    Args:
        pred: Prediction tensor
        target: Target tensor
        threshold: Threshold for binary classification

    Returns:
        Dictionary of metrics
    """
    pred_binary = (pred > threshold).float()

    iou = calculate_iou(pred_binary, target)
    dice = calculate_dice(pred_binary, target)

    return {
        'iou': iou,
        'dice': dice,
        'accuracy': ((pred_binary == target).float().mean()).item()
    }


def evaluate_restoration_metrics(input_img: np.ndarray,
                               output_img: np.ndarray,
                               gt_img: np.ndarray,
                               mask: np.ndarray) -> dict:
    """Evaluate restoration metrics.

    Args:
        input_img: Input image with artifacts
        output_img: Restored image
        gt_img: Ground truth image
        mask: Binary mask of artifact regions

    Returns:
        Dictionary of metrics
    """
    # PSNR on masked regions (artifact areas)
    psnr_masked = calculate_psnr_masked(gt_img, output_img, mask)

    # PSNR on unmasked regions (clean areas)
    psnr_unmasked = calculate_psnr_masked(gt_img, output_img, 1 - mask)

    # MSE on unmasked regions
    mse_unmasked = calculate_mse_unmasked(gt_img, output_img, mask)

    return {
        'psnr_masked': psnr_masked,
        'psnr_unmasked': psnr_unmasked,
        'mse_unmasked': mse_unmasked
    }


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_pixel_value: float = 1.0) -> float:
    """Calculate PSNR between prediction and target tensors."""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Ensure arrays are in the right range
    if pred_np.max() <= 1.0:
        pred_np = pred_np * 255
    if target_np.max() <= 1.0:
        target_np = target_np * 255
    
    # Handle edge cases similar to archive implementation
    mse = np.mean((pred_np - target_np) ** 2)
    if mse == 0:
        return 100.0
    
    psnr_value = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Cap PSNR at reasonable maximum
    if np.isinf(psnr_value) or psnr_value > 100:
        return 100.0
    
    return float(psnr_value)


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, max_pixel_value: float = 1.0) -> float:
    """Calculate SSIM between prediction and target tensors."""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Ensure arrays are in the right range
    if pred_np.max() <= 1.0:
        pred_np = pred_np * 255
    if target_np.max() <= 1.0:
        target_np = target_np * 255
    
    # For single channel images
    if pred_np.ndim == 4:  # Batch dimension
        pred_np = pred_np[0, 0]
    elif pred_np.ndim == 3:  # Channel dimension
        pred_np = pred_np[0]
        
    if target_np.ndim == 4:
        target_np = target_np[0, 0]
    elif target_np.ndim == 3:
        target_np = target_np[0]
    
    return float(ssim(target_np, pred_np, data_range=255))
