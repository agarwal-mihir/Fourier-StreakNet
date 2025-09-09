"""CLI command implementations."""

import logging
from pathlib import Path
from typing import Dict, Any

# Import statements for available modules
# Training and inference modules will be implemented when needed


def make_data_cmd(args, config: Dict[str, Any], logger: logging.Logger):
    """Generate synthetic training data."""
    logger.info("Generating synthetic training data...")

    from ..data.transforms.streaks import StreakAugmentation, VerticalStreakAugmentation
    from ..utils.io import get_file_list, ensure_dir

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Create output directories
    img_output = ensure_dir(output_dir / 'images')
    mask_output = ensure_dir(output_dir / 'masks')

    # Get input files
    input_files = get_file_list(input_dir)
    if not input_files:
        logger.error(f"No image files found in {input_dir}")
        return

    # Choose augmentation type
    if args.vertical_only:
        augmenter = VerticalStreakAugmentation(
            num_streaks_range=tuple(args.num_streaks),
            blend_factor_range=tuple(args.blend_factor),
            max_streak_width=args.max_streak_width
        )
    else:
        augmenter = StreakAugmentation(
            num_streaks_range=tuple(args.num_streaks),
            blend_factor_range=tuple(args.blend_factor),
            max_streak_width=args.max_streak_width
        )

    logger.info(f"Processing {len(input_files)} images...")

    for i, img_path in enumerate(input_files):
        logger.info(f"Processing {i+1}/{len(input_files)}: {img_path.name}")

        # Load and augment image
        from PIL import Image
        import numpy as np

        image = Image.open(img_path).convert('L')
        image_np = np.array(image)

        # Apply augmentation
        augmented_img, mask = augmenter(image_np)

        # Save results
        base_name = img_path.stem
        Image.fromarray(augmented_img).save(img_output / f"{base_name}.png")
        Image.fromarray(mask).save(mask_output / f"{base_name}_mask.png")

    logger.info(f"Synthetic data generation completed. Results saved to {output_dir}")


def train_seg_cmd(args, config: Dict[str, Any], logger: logging.Logger):
    """Train segmentation model."""
    logger.info("Training segmentation model...")
    
    import subprocess
    import sys
    from pathlib import Path
    
    # Build command to run training script
    script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "train_segmentation.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--image_dir", args.image_dir,
        "--mask_dir", args.mask_dir,
        "--output_dir", args.output_dir,
        "--model_type", args.model_type,
        "--epochs", str(args.epochs or config['training']['epochs']),
        "--batch_size", str(args.batch_size or config['data']['batch_size']),
        "--learning_rate", str(args.learning_rate or config['training']['learning_rate']),
        "--seed", str(args.seed)
    ]
    
    # Add validation directories if provided
    if args.val_image_dir and args.val_mask_dir:
        cmd.extend(["--val_image_dir", args.val_image_dir, "--val_mask_dir", args.val_mask_dir])
    
    # Add CUDA flag if available
    import torch
    if torch.cuda.is_available():
        cmd.append("--cuda")
    
    logger.info(f"Running segmentation training: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Segmentation training completed successfully")
        if result.stdout:
            logger.debug(f"Training output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        raise


def train_restore_cmd(args, config: Dict[str, Any], logger: logging.Logger):
    """Train restoration model."""
    logger.info("Training restoration model...")
    
    import subprocess
    import sys
    from pathlib import Path
    
    # Build command to run training script
    script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "train_restoration.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--input_dir", args.input_dir,
        "--mask_dir", args.mask_dir,
        "--gt_dir", args.gt_dir,
        "--output_dir", args.output_dir,
        "--epochs", str(args.epochs or config['training']['epochs']),
        "--batch_size", str(args.batch_size or config['data']['batch_size']),
        "--learning_rate", str(args.learning_rate or config['training']['learning_rate']),
        "--seed", str(args.seed)
    ]
    
    # Add CUDA flag if available
    import torch
    if torch.cuda.is_available():
        cmd.append("--cuda")
    
    logger.info(f"Running restoration training: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Restoration training completed successfully")
        if result.stdout:
            logger.debug(f"Training output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        raise


def infer_cmd(args, config: Dict[str, Any], logger: logging.Logger):
    """Run inference on images."""
    logger.info("Running inference...")

    from ..inference import InferencePipeline
    from ..utils.io import ensure_dir, get_file_list
    
    # Validate inputs
    if not args.input_dir and not args.input_image:
        logger.error("Must specify either --input-dir or --input-image")
        return

    if args.mode == 'pipeline' and (not args.seg_checkpoint or not args.restore_checkpoint):
        logger.error("Pipeline mode requires both --seg-checkpoint and --restore-checkpoint")
        return

    # Get input files
    if args.input_image:
        input_files = [Path(args.input_image)]
    else:
        input_files = get_file_list(args.input_dir)
        if not input_files:
            logger.error(f"No image files found in {args.input_dir}")
            return

    # Create output directory
    output_dir = ensure_dir(args.output_dir)

    # Update config
    if args.batch_size:
        config['inference']['batch_size'] = args.batch_size

    # Create inference pipeline
    pipeline = InferencePipeline(
        seg_checkpoint=args.seg_checkpoint,
        restore_checkpoint=args.restore_checkpoint if hasattr(args, 'restore_checkpoint') else None,
        config=config,
        logger=logger
    )

    # Run inference
    results = pipeline.run(
        input_files=input_files,
        output_dir=output_dir,
        mode=args.mode,
        save_intermediates=args.save_intermediates
    )

    logger.info(f"Inference completed. Results saved to {output_dir}")


def eval_cmd(args, config: Dict[str, Any], logger: logging.Logger):
    """Evaluate model performance."""
    logger.info("Evaluating model performance...")

    from ..utils.metrics import evaluate_segmentation_metrics, evaluate_restoration_metrics
    from ..utils.io import get_file_list
    import json

    # Get file lists
    pred_files = get_file_list(args.pred_dir)
    gt_files = get_file_list(args.gt_dir)
    mask_files = get_file_list(args.mask_dir) if args.mask_dir else None

    if len(pred_files) != len(gt_files):
        logger.error("Number of prediction and ground truth files must match")
        return

    logger.info(f"Evaluating {len(pred_files)} samples...")

    results = []
    summary_stats = {}

    for pred_file, gt_file in zip(pred_files, gt_files):
        # Load images
        from PIL import Image
        import numpy as np

        pred_img = np.array(Image.open(pred_file))
        gt_img = np.array(Image.open(gt_file))

        if args.task == 'segmentation':
            # For segmentation, assume binary masks
            metrics = evaluate_segmentation_metrics(
                pred=torch.from_numpy(pred_img).float(),
                target=torch.from_numpy(gt_img).float()
            )
        else:  # restoration
            mask_img = None
            if mask_files:
                # Find corresponding mask file
                mask_name = pred_file.name.replace('_pred', '_mask')
                mask_path = args.mask_dir / mask_name
                if mask_path.exists():
                    mask_img = np.array(Image.open(mask_path))

            if mask_img is None:
                mask_img = np.ones_like(pred_img[:, :, 0])  # Use full image if no mask

            metrics = evaluate_restoration_metrics(
                input_img=pred_img,
                output_img=pred_img,
                gt_img=gt_img,
                mask=mask_img
            )

        results.append({
            'filename': pred_file.name,
            **metrics
        })

    # Compute summary statistics
    if results:
        metric_keys = list(results[0].keys())
        metric_keys.remove('filename')

        summary_stats = {}
        for key in metric_keys:
            values = [r[key] for r in results if not np.isnan(r[key])]
            if values:
                summary_stats[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }

    # Save results
    output_data = {
        'summary': summary_stats,
        'individual_results': results
    }

    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Evaluation completed. Results saved to {args.output_file}")


def visualize_cmd(args, config: Dict[str, Any], logger: logging.Logger):
    """Create visualizations."""
    logger.info("Creating visualizations...")

    from ..utils.visualization import create_comparison_grid
    from ..utils.io import get_file_list, ensure_dir

    # Get input files
    input_files = get_file_list(args.input_dir)
    if not input_files:
        logger.error(f"No image files found in {args.input_dir}")
        return

    # Limit number of samples
    input_files = input_files[:args.num_samples]

    # Create output directory
    output_dir = ensure_dir(args.output_dir)

    logger.info(f"Creating {args.type} visualizations for {len(input_files)} samples...")

    if args.type == 'grid':
        # Create image grids
        from PIL import Image
        import numpy as np

        images = []
        for img_file in input_files:
            img = Image.open(img_file)
            images.append(np.array(img))

        # Save grid
        grid_path = output_dir / 'image_grid.png'
        create_comparison_grid(
            images=images,
            titles=[f.name for f in input_files],
            save_path=str(grid_path)
        )

    elif args.type == 'overlay':
        # Create overlay visualizations (requires masks)
        logger.warning("Overlay visualization requires mask files - not yet implemented")

    elif args.type == 'comparison':
        # Create comparison visualizations (requires multiple sets)
        logger.warning("Comparison visualization requires multiple image sets - not yet implemented")

    logger.info(f"Visualizations saved to {output_dir}")
