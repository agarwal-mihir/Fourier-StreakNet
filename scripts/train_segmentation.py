#!/usr/bin/env python3
"""Training script for UNet segmentation models."""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Import from the ultrastreak package
import sys
sys.path.append('src')
from ultrastreak.models.unet import UNet
from ultrastreak.models.unet_notch import UNetWithNotchFilter
from ultrastreak.data.datasets import StreakDataset
from ultrastreak.utils.metrics import calculate_dice, calculate_iou
from ultrastreak.utils.logging import setup_logger
from ultrastreak.utils.seed import setup_reproducibility
from ultrastreak.utils.io import save_checkpoint, ensure_dir


def dice_coefficient(pred, target):
    """Compute Dice coefficient."""
    smooth = 1.0
    pred_binary = (pred > 0.5).float()
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def evaluate_model(model, val_loader, device, criterion):
    """Evaluate model performance."""
    model.eval()
    val_loss = 0.0
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        for inputs, masks in val_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            
            outputs = model(inputs)
            outputs_sigmoid = torch.sigmoid(outputs)
            
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            # Calculate metrics
            dice = dice_coefficient(outputs_sigmoid, masks)
            dice_scores.append(dice.item())
            
            # Calculate IoU for each sample in batch
            for i in range(inputs.size(0)):
                iou = calculate_iou(outputs_sigmoid[i], masks[i])
                iou_scores.append(iou)
    
    avg_val_loss = val_loss / len(val_loader)
    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_iou = sum(iou_scores) / len(iou_scores)
    
    return avg_val_loss, avg_dice, avg_iou


def visualize_predictions(model, val_loader, device, save_dir, num_samples=5):
    """Visualize model predictions."""
    ensure_dir(save_dir)
    model.eval()
    
    with torch.no_grad():
        for i, (inputs, masks) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            
            # Convert to numpy for visualization
            if inputs.shape[1] == 2:  # UNet with notch filter
                input_img = inputs[:, 0, :, :].cpu().numpy()
                filtered_img = inputs[:, 1, :, :].cpu().numpy()
            else:
                input_img = inputs[:, 0, :, :].cpu().numpy()
                filtered_img = None
            
            mask = masks[:, 0, :, :].cpu().numpy()
            pred = outputs[:, 0, :, :].cpu().numpy()
            
            # Plot results for first sample in batch
            if filtered_img is not None:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes[0, 0].imshow(input_img[0], cmap='gray')
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(filtered_img[0], cmap='gray')
                axes[0, 1].set_title('Notch Filtered Image')
                axes[0, 1].axis('off')
                
                axes[1, 0].imshow(mask[0], cmap='gray')
                axes[1, 0].set_title('Ground Truth Mask')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(pred[0], cmap='gray')
                axes[1, 1].set_title('Prediction')
                axes[1, 1].axis('off')
            else:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(input_img[0], cmap='gray')
                axes[0].set_title('Input Image')
                axes[0].axis('off')
                
                axes[1].imshow(mask[0], cmap='gray')
                axes[1].set_title('Ground Truth Mask')
                axes[1].axis('off')
                
                axes[2].imshow(pred[0], cmap='gray')
                axes[2].set_title('Prediction')
                axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'prediction_{i}.png'))
            plt.close()


def load_config(config_path=None):
    """Load configuration from YAML file."""
    default_config = {
        'training': {
            'epochs': 100,
            'learning_rate': 1e-4,
            'batch_size': 16,
            'eval_interval': 5,
            'save_interval': 10
        },
        'model': {
            'in_channels': 1,
            'out_channels': 1,
            'features': [64, 128, 256, 512]
        }
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        # Merge configs
        for key, value in user_config.items():
            if key in default_config:
                default_config[key].update(value)
            else:
                default_config[key] = value
    
    return default_config


def main():
    parser = argparse.ArgumentParser(description="Train UNet for streak segmentation")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing mask images")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--model_type", type=str, choices=['unet', 'unet_notch'], default='unet_notch', help="Model type")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--eval_interval", type=int, help="Evaluation interval (epochs)")
    parser.add_argument("--save_interval", type=int, help="Checkpoint saving interval (epochs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.eval_interval:
        config['training']['eval_interval'] = args.eval_interval
    if args.save_interval:
        config['training']['save_interval'] = args.save_interval
    
    # Setup reproducibility
    setup_reproducibility(args.seed)
    
    # Setup logging
    logger = setup_logger('train_segmentation')
    logger.info(f"Starting training with model type: {args.model_type}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Create dataset
    dataset = StreakDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        transform=transform,
        target_transform=transform,
        augment=True,
        notch_filter=False  # Always use single channel for regular UNet
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    logger.info(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    # Create model
    if args.model_type == 'unet_notch':
        model = UNetWithNotchFilter(in_channels=1, out_channels=1).to(device)
    else:
        model = UNet(in_channels=1, out_channels=1).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Create output directories
    ensure_dir(args.output_dir)
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    ensure_dir(viz_dir)
    
    # Training loop
    best_dice = 0.0
    train_losses = []
    val_losses = []
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}") as pbar:
            for inputs, masks in pbar:
                inputs, masks = inputs.to(device), masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=epoch_loss/(pbar.n+1))
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate
        if (epoch + 1) % config['training']['eval_interval'] == 0:
            val_loss, val_dice, val_iou = evaluate_model(model, val_loader, device, criterion)
            val_losses.append(val_loss)
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                save_checkpoint(
                    model, optimizer, epoch, val_loss, 
                    {'dice': val_dice, 'iou': val_iou},
                    os.path.join(args.output_dir, 'best_model.pth')
                )
                logger.info(f"New best model saved with Dice: {best_dice:.4f}")
                
                # Visualize predictions for best model
                visualize_predictions(model, val_loader, device, 
                                    os.path.join(viz_dir, f"epoch_{epoch+1}"))
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            save_checkpoint(
                model, optimizer, epoch, avg_train_loss, 
                {'dice': best_dice},
                os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth')
            )
    
    logger.info("Training completed!")
    logger.info(f"Best Dice score: {best_dice:.4f}")


if __name__ == "__main__":
    main()


