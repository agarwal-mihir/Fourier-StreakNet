#!/usr/bin/env python3
"""Training script for Fourier attention restoration models."""

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

# Import from the ultrastreak package
import sys
sys.path.append('src')
from ultrastreak.models.fourier_attention_net import UNetWithFourierAttention
from ultrastreak.data.datasets import HIFUDataset
from ultrastreak.utils.metrics import calculate_psnr, calculate_ssim
from ultrastreak.utils.logging import setup_logger
from ultrastreak.utils.seed import setup_reproducibility
from ultrastreak.utils.io import save_checkpoint, ensure_dir


def psnr_loss(pred, target):
    """Compute PSNR-based loss."""
    mse = nn.MSELoss()(pred, target)
    return -10 * torch.log10(mse + 1e-8)


def evaluate_restoration_model(model, val_loader, device, criterion):
    """Evaluate restoration model performance."""
    model.eval()
    val_loss = 0.0
    psnr_scores = []
    ssim_scores = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['img'].to(device)
            masks = batch['mask'].to(device)
            targets = batch['gt'].to(device)
            
            # Convert RGB to grayscale
            inputs_gray = inputs[:, 0:1, :, :]
            targets_gray = targets[:, 0:1, :, :]
            
            # Concatenate grayscale image and mask as input
            combined_input = torch.cat([inputs_gray, masks], dim=1)
            
            outputs = model(combined_input)
            
            loss = criterion(outputs, targets_gray)
            val_loss += loss.item()
            
            # Calculate metrics
            for i in range(inputs.size(0)):
                psnr = calculate_psnr(outputs[i], targets_gray[i])
                ssim = calculate_ssim(outputs[i], targets_gray[i])
                psnr_scores.append(psnr)
                ssim_scores.append(ssim)
    
    avg_val_loss = val_loss / len(val_loader)
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    
    return avg_val_loss, avg_psnr, avg_ssim


def visualize_restoration_results(model, val_loader, device, save_dir, num_samples=3):
    """Visualize restoration results."""
    ensure_dir(save_dir)
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
                
            inputs = batch['img'].to(device)
            masks = batch['mask'].to(device)
            targets = batch['gt'].to(device)
            
            # Convert RGB to grayscale
            inputs_gray = inputs[:, 0:1, :, :]
            targets_gray = targets[:, 0:1, :, :]
            
            # Concatenate grayscale image and mask as input
            combined_input = torch.cat([inputs_gray, masks], dim=1)
            
            outputs = model(combined_input)
            
            # Convert to numpy for visualization
            input_img = inputs_gray[0, 0].cpu().numpy()
            mask_img = masks[0, 0].cpu().numpy()
            target_img = targets_gray[0, 0].cpu().numpy()
            output_img = outputs[0, 0].cpu().numpy()
            
            # Plot results
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            axes[0, 0].imshow(input_img, cmap='gray')
            axes[0, 0].set_title('Input Image (with streaks)')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(mask_img, cmap='gray')
            axes[0, 1].set_title('Streak Mask')
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(target_img, cmap='gray')
            axes[1, 0].set_title('Ground Truth')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(output_img, cmap='gray')
            axes[1, 1].set_title('Restored Image')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'restoration_{i}.png'))
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train Fourier Attention Network for restoration")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing mask images")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth images")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--eval_interval", type=int, default=5, help="Evaluation interval (epochs)")
    parser.add_argument("--save_interval", type=int, default=10, help="Checkpoint saving interval (epochs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    
    args = parser.parse_args()
    
    # Setup reproducibility
    setup_reproducibility(args.seed)
    
    # Setup logging
    logger = setup_logger('train_restoration')
    logger.info("Starting Fourier Attention Network training")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Create dataset
    dataset = HIFUDataset(
        img_dir=args.input_dir,
        mask_dir=args.mask_dir,
        gt_dir=args.gt_dir,
        transform=transform
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    # Create model (1 channel for grayscale image + 1 channel for mask = 2 channels input, 1 channel for grayscale output)
    model = UNetWithFourierAttention(n_channels=2, n_classes=1, bilinear=True).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create output directories
    ensure_dir(args.output_dir)
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    ensure_dir(viz_dir)
    
    # Training loop
    best_psnr = 0.0
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                inputs = batch['img'].to(device)
                masks = batch['mask'].to(device)
                targets = batch['gt'].to(device)
                
                # Convert RGB to grayscale (take first channel or use weighted sum)
                inputs_gray = inputs[:, 0:1, :, :]  # Take first channel
                targets_gray = targets[:, 0:1, :, :]  # Take first channel
                
                # Concatenate grayscale image and mask as input
                combined_input = torch.cat([inputs_gray, masks], dim=1)
                
                optimizer.zero_grad()
                outputs = model(combined_input)
                loss = criterion(outputs, targets_gray)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=epoch_loss/(pbar.n+1))
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate
        if (epoch + 1) % args.eval_interval == 0:
            val_loss, val_psnr, val_ssim = evaluate_restoration_model(model, val_loader, device, criterion)
            val_losses.append(val_loss)
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}")
            
            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    model, optimizer, epoch, val_loss, 
                    {'psnr': val_psnr, 'ssim': val_ssim},
                    os.path.join(args.output_dir, 'best_model.pth')
                )
                logger.info(f"New best model saved with PSNR: {best_psnr:.4f}")
                
                # Visualize results for best model
                visualize_restoration_results(model, val_loader, device, 
                                            os.path.join(viz_dir, f"epoch_{epoch+1}"))
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, avg_train_loss, 
                {'psnr': best_psnr},
                os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth')
            )
    
    logger.info("Training completed!")
    logger.info(f"Best PSNR score: {best_psnr:.4f}")


if __name__ == "__main__":
    main()