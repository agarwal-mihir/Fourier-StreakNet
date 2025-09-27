#!/usr/bin/env python3
"""Testing script for UNet segmentation models - based on original unet_notch_test.py."""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json

# Import from the ultrastreak package
import sys
sys.path.append('src')
from ultrastreak.models.unet import UNet
from ultrastreak.models.unet_notch import UNetWithNotchFilter
from ultrastreak.data.datasets import StreakDataset
from ultrastreak.utils.metrics import calculate_dice, calculate_iou
from ultrastreak.utils.logging import setup_logger
from ultrastreak.utils.seed import setup_reproducibility
from ultrastreak.utils.io import load_checkpoint, ensure_dir


def test_model(args):
    """Test the trained UNet model."""
    # Setup reproducibility
    setup_reproducibility(args.seed)
    
    # Setup logging
    logger = setup_logger('test_segmentation')
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Create test dataset
    test_dataset = StreakDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        transform=transform,
        target_transform=transform,
        augment=False,
        notch_filter=(args.model_type == 'unet_notch')
    )
    
    # Use subset if specified
    if args.subset_ratio < 1.0:
        subset_size = int(args.subset_ratio * len(test_dataset))
        indices = torch.randperm(len(test_dataset))[:subset_size]
        test_subset = torch.utils.data.Subset(test_dataset, indices)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
        logger.info(f"Using subset of {len(test_subset)} samples from {len(test_dataset)} total")
    else:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        logger.info(f"Testing on full dataset: {len(test_dataset)} samples")
    
    # Create model
    if args.model_type == 'unet_notch':
        model = UNetWithNotchFilter(in_channels=1, out_channels=1).to(device)
    else:
        model = UNet(in_channels=1, out_channels=1).to(device)
    
    # Load trained model
    checkpoint = load_checkpoint(args.model_path, model, device=device)
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'metrics' in checkpoint:
        logger.info(f"Model metrics: {checkpoint['metrics']}")
    
    # Define loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Evaluation
    model.eval()
    test_losses = []
    iou_scores = []
    dice_scores = []
    
    # Result storage
    results = {}
    
    # Create output directory
    if args.output_dir:
        ensure_dir(args.output_dir)
    
    with torch.no_grad():
        for batch_idx, (inputs, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            inputs, masks = inputs.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(inputs)
            outputs_sigmoid = torch.sigmoid(outputs)
            
            # Compute loss
            loss = criterion(outputs, masks).item()
            test_losses.append(loss)
            
            # Calculate metrics for each sample in batch
            for i in range(inputs.size(0)):
                sample_idx = batch_idx * args.batch_size + i
                
                iou = calculate_iou(outputs_sigmoid[i], masks[i])
                dice = calculate_dice(outputs_sigmoid[i], masks[i]).item()
                
                iou_scores.append(iou)
                dice_scores.append(dice)
                
                # Store results
                results[f'sample_{sample_idx}'] = {
                    'loss': loss,
                    'iou': iou,
                    'dice': dice
                }
                
                # Save visualization if output directory specified
                if args.output_dir and args.save_visualizations:
                    save_prediction_visualization(
                        inputs[i], masks[i], outputs_sigmoid[i],
                        os.path.join(args.output_dir, f'prediction_{sample_idx}.png'),
                        args.model_type
                    )
    
    # Calculate summary statistics
    avg_loss = np.mean(test_losses)
    avg_iou = np.mean(iou_scores)
    avg_dice = np.mean(dice_scores)
    
    # Calculate percentiles
    iou_percentiles = {
        '25th': np.percentile(iou_scores, 25),
        '50th': np.percentile(iou_scores, 50),
        '75th': np.percentile(iou_scores, 75)
    }
    
    dice_percentiles = {
        '25th': np.percentile(dice_scores, 25),
        '50th': np.percentile(dice_scores, 50),
        '75th': np.percentile(dice_scores, 75)
    }
    
    # Print results
    logger.info("===== Test Results =====")
    logger.info(f"Average Loss: {avg_loss:.4f}")
    logger.info(f"Average IoU: {avg_iou:.4f}")
    logger.info(f"Average Dice: {avg_dice:.4f}")
    logger.info(f"IoU Percentiles - 25th: {iou_percentiles['25th']:.4f}, "
               f"50th: {iou_percentiles['50th']:.4f}, 75th: {iou_percentiles['75th']:.4f}")
    logger.info(f"Dice Percentiles - 25th: {dice_percentiles['25th']:.4f}, "
               f"50th: {dice_percentiles['50th']:.4f}, 75th: {dice_percentiles['75th']:.4f}")
    
    # Save results to JSON
    summary_results = {
        'model_type': args.model_type,
        'model_path': args.model_path,
        'test_samples': len(iou_scores),
        'avg_loss': avg_loss,
        'avg_iou': avg_iou,
        'avg_dice': avg_dice,
        'iou_percentiles': iou_percentiles,
        'dice_percentiles': dice_percentiles,
        'individual_results': results
    }
    
    results_file = os.path.join(args.output_dir if args.output_dir else '.', 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump(summary_results, f, indent=4)
    
    logger.info(f"Results saved to {results_file}")
    
    # Create histogram plots
    if args.output_dir:
        create_metric_histograms(iou_scores, dice_scores, avg_iou, avg_dice, args.output_dir)
        logger.info(f"Histogram plots saved to {args.output_dir}")


def save_prediction_visualization(input_tensor, mask_tensor, pred_tensor, save_path, model_type):
    """Save visualization of prediction results."""
    # Convert tensors to numpy
    if input_tensor.dim() == 3 and input_tensor.shape[0] == 2:  # UNet with notch filter
        input_img = input_tensor[0].cpu().numpy()
        filtered_img = input_tensor[1].cpu().numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0, 0].imshow(input_img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(filtered_img, cmap='gray')
        axes[0, 1].set_title('Notch Filtered Image')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(mask_tensor[0].cpu().numpy(), cmap='gray')
        axes[1, 0].set_title('Ground Truth Mask')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(pred_tensor[0].cpu().numpy(), cmap='gray')
        axes[1, 1].set_title('Prediction')
        axes[1, 1].axis('off')
    else:
        input_img = input_tensor[0].cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(input_img, cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask_tensor[0].cpu().numpy(), cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        axes[2].imshow(pred_tensor[0].cpu().numpy(), cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
    
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()


def create_metric_histograms(iou_scores, dice_scores, avg_iou, avg_dice, output_dir):
    """Create histogram plots of IoU and Dice scores."""
    # IoU histogram
    plt.figure(figsize=(10, 6))
    plt.hist(iou_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(avg_iou, color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean IoU: {avg_iou:.4f}')
    plt.title('Histogram of IoU Scores')
    plt.xlabel('IoU Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_histogram.png'))
    plt.close()
    
    # Dice histogram
    plt.figure(figsize=(10, 6))
    plt.hist(dice_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(avg_dice, color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean Dice: {avg_dice:.4f}')
    plt.title('Histogram of Dice Scores')
    plt.xlabel('Dice Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_histogram.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test UNet for streak segmentation")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing test mask images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--model_type", type=str, choices=['unet', 'unet_notch'], default='unet_notch', help="Model type")
    parser.add_argument("--output_dir", type=str, default="test_results", help="Directory to save test results")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--subset_ratio", type=float, default=1.0, help="Fraction of dataset to use (0.0-1.0)")
    parser.add_argument("--save_visualizations", action="store_true", help="Save prediction visualizations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    
    args = parser.parse_args()
    test_model(args)


if __name__ == "__main__":
    main()


