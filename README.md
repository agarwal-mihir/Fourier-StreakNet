# Ultrasound Streak Removal

A deep learning toolkit for removing streak artifacts from ultrasound images using UNet segmentation and Fourier attention-based restoration.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)

## Overview

This package provides a complete pipeline for ultrasound image streak removal:

1. **Data Generation**: Create synthetic training data with realistic streak artifacts
2. **Segmentation**: Train UNet models to detect streak locations
3. **Restoration**: Train Fourier attention networks to remove detected streaks
4. **Inference**: Run the complete pipeline on new ultrasound images
5. **Evaluation**: Comprehensive metrics for model performance assessment

## Features

- **Modular Architecture**: Clean, professional code structure
- **Multiple Models**: UNet, UNet with notch filtering, and Fourier attention networks
- **Comprehensive Evaluation**: IoU, Dice, PSNR, SSIM metrics
- **CLI Interface**: Easy-to-use command line tools with full pipeline support
- **Configuration-Driven**: YAML-based configuration for reproducible experiments
- **Experiment Tracking**: Built-in logging and visualization
- **Reproducible**: Configurable random seeds and deterministic operations
- **Complete Pipeline**: End-to-end workflow from data generation to evaluation

## Installation

### From Source

```bash
git clone https://github.com/yourusername/ultrastreak.git
cd ultrastreak
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- OpenCV >= 4.5.0
- NumPy, SciPy, scikit-image
- Pillow, matplotlib, tqdm

## Quick Start

### 1. Prepare Training Data

Prepare your dataset with the following structure:

```
data/
├── train/
│   ├── images/    # Input images with streaks
│   ├── masks/     # Binary masks of streak locations
│   └── gt/        # Ground truth clean images (for restoration)
├── val/           # Validation data (same structure)
└── test/          # Test data
```

Supported formats:
- Images: PNG, JPG, JPEG, TIFF
- Masks: Binary images (0=background, 255=foreground)

### 2. Train Segmentation Model

```bash
# Train UNet for streak detection
ultrastreak train-seg \
  --image-dir ./data/synthetic/images \
  --mask-dir ./data/synthetic/masks \
  --output-dir ./checkpoints/segmentation \
  --model-type unet \
  --epochs 100 \
  --config configs/train/segmentation.yaml
```

### 3. Train Restoration Model

```bash
# Train Fourier attention network for streak removal
ultrastreak train-restore \
  --input-dir ./data/train/streaked \
  --mask-dir ./data/train/masks \
  --gt-dir ./data/train/clean \
  --output-dir ./checkpoints/restoration \
  --epochs 100 \
  --config configs/train/restoration.yaml
```

### 4. Run Inference

```bash
# Process new images with the complete pipeline
ultrastreak infer \
  --input-dir ./data/test \
  --output-dir ./results \
  --seg-checkpoint ./checkpoints/segmentation/best_model.pth \
  --restore-checkpoint ./checkpoints/restoration/best_model.pth \
  --mode pipeline
```

### 5. Evaluate Results

```bash
# Evaluate segmentation performance
ultrastreak eval \
  --pred-dir ./results/segmentation \
  --gt-dir ./data/test/masks \
  --task segmentation \
  --output-file ./evaluation_results.json
```

## Project Structure

```
ultrastreak/
├── src/ultrastreak/          # Main package
│   ├── cli/                  # Command line interface
│   │   ├── cli.py           # Main CLI entry point
│   │   └── commands.py      # Command implementations
│   ├── data/                # Data loading and augmentation
│   │   ├── datasets.py      # Dataset classes
│   │   └── transforms/      # Image transformations
│   ├── models/              # Neural network models
│   │   ├── unet.py         # Basic UNet
│   │   ├── unet_notch.py   # UNet with notch filter
│   │   ├── fourier_attention_net.py  # Restoration model
│   │   └── blocks/         # Model components
│   └── utils/               # Utility functions
├── configs/                 # Configuration files
│   ├── data/               # Data configurations
│   ├── model/              # Model configurations
│   ├── train/              # Training configurations
│   └── infer/              # Inference configurations
├── scripts/                # Training and evaluation scripts
│   ├── train_segmentation.py
│   └── train_restoration.py
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── pyproject.toml         # Package configuration
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Models

### Segmentation Models

1. **UNet**: Standard UNet architecture for binary segmentation
2. **UNet with Notch Filter**: Enhanced UNet that applies frequency-domain filtering to better detect streaks

### Restoration Models

1. **Fourier Attention Network**: UNet with Fourier attention layers that leverage frequency-domain information for better artifact removal

## Configuration

The package uses YAML configuration files for reproducible experiments. See `configs/data/ultrasound.yaml` for a complete example.

Key configuration sections:
- **Data**: Dataset paths, preprocessing, augmentation
- **Model**: Architecture parameters
- **Training**: Optimizer, scheduler, loss function
- **Inference**: Post-processing, tiling options

## Data Format

### Training Data Structure

```
data/
├── train/
│   ├── images/          # Input images with streaks
│   ├── masks/           # Binary masks of streak locations
│   └── gt/             # Ground truth clean images (for restoration)
├── val/                # Validation data (same structure)
└── test/               # Test data
```

### Supported Formats

- Images: PNG, JPG, JPEG, TIFF
- Masks: Binary images (0=background, 255=foreground)

## CLI Commands

### Available Commands

The `ultrastreak` CLI provides the following commands:

- **`make-data`**: Generate synthetic training data with streak artifacts (optional)
- **`train-seg`**: Train segmentation models for streak detection
- **`train-restore`**: Train restoration models for streak removal
- **`infer`**: Run inference on new images
- **`eval`**: Evaluate model performance
- **`visualize`**: Create visualizations and comparison plots

### Configuration Files

The project uses YAML configuration files for reproducible experiments:

- `configs/train/segmentation.yaml`: Segmentation training configuration
- `configs/train/restoration.yaml`: Restoration training configuration
- `configs/model/unet.yaml`: UNet model configuration
- `configs/model/fourier_attention.yaml`: Fourier attention model configuration

### Command Examples

```bash
# Show help for any command
ultrastreak train-seg --help

# Train with custom configuration
ultrastreak train-seg \
  --config custom_config.yaml \
  --image-dir ./data/images \
  --mask-dir ./data/masks \
  --epochs 50 \
  --batch-size 32

# Run inference with different modes
ultrastreak infer --mode segmentation    # Only segmentation
ultrastreak infer --mode restoration   # Only restoration  
ultrastreak infer --mode pipeline       # Full pipeline

# Evaluate restoration results
ultrastreak eval \
  --pred-dir ./results \
  --gt-dir ./data/test/gt \
  --mask-dir ./data/test/masks \
  --task restoration
```

## Evaluation Metrics

### Segmentation
- **IoU (Intersection over Union)**: Overlap between predicted and ground truth masks
- **Dice Coefficient**: F1 score for pixel-wise classification
- **Precision/Recall**: Standard classification metrics

### Restoration
- **PSNR (Peak Signal-to-Noise Ratio)**: Image quality metric
- **SSIM (Structural Similarity Index)**: Perceptual image quality
- **MSE**: Mean squared error in masked regions

