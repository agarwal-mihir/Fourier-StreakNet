"""Command line interface for ultrastreak package."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml

from ..utils.logging import setup_logger
from ..utils.seed import setup_reproducibility


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file or return defaults.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    default_config = {
        'data': {
            'input_size': [256, 256],
            'batch_size': 16,
            'num_workers': 4,
        },
        'model': {
            'type': 'unet',
            'in_channels': 1,
            'out_channels': 1,
            'features': [64, 128, 256, 512],
        },
        'training': {
            'epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'patience': 20,
            'save_interval': 10,
        },
        'inference': {
            'batch_size': 8,
            'threshold': 0.5,
            'save_intermediates': False,
        }
    }

    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        # Merge configs
        for key, value in user_config.items():
            if key in default_config:
                default_config[key].update(value)
            else:
                default_config[key] = value

    return default_config


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to parser."""
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')


def create_data_parser(subparsers) -> argparse.ArgumentParser:
    """Create parser for data generation commands."""
    data_parser = subparsers.add_parser('make-data', help='Generate synthetic training data')
    add_common_args(data_parser)

    data_parser.add_argument('--input-dir', type=str, required=True,
                           help='Directory containing clean images')
    data_parser.add_argument('--output-dir', type=str, required=True,
                           help='Directory to save generated data')
    data_parser.add_argument('--num-streaks', type=int, nargs=2, default=[5, 15],
                           help='Range for number of streaks [min, max]')
    data_parser.add_argument('--blend-factor', type=float, nargs=2, default=[0.2, 0.6],
                           help='Range for streak blending factor [min, max]')
    data_parser.add_argument('--max-streak-width', type=int, default=20,
                           help='Maximum streak width')
    data_parser.add_argument('--vertical-only', action='store_true',
                           help='Generate only vertical streaks')

    return data_parser


def create_train_seg_parser(subparsers) -> argparse.ArgumentParser:
    """Create parser for segmentation training."""
    train_parser = subparsers.add_parser('train-seg', help='Train segmentation model')
    add_common_args(train_parser)

    train_parser.add_argument('--image-dir', type=str, required=True,
                            help='Directory containing training images')
    train_parser.add_argument('--mask-dir', type=str, required=True,
                            help='Directory containing training masks')
    train_parser.add_argument('--val-image-dir', type=str,
                            help='Directory containing validation images')
    train_parser.add_argument('--val-mask-dir', type=str,
                            help='Directory containing validation masks')
    train_parser.add_argument('--output-dir', type=str, default='checkpoints/segmentation',
                            help='Directory to save checkpoints')
    train_parser.add_argument('--model-type', type=str, choices=['unet', 'unet_notch'],
                            default='unet_notch', help='Model type to train')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')

    return train_parser


def create_train_restore_parser(subparsers) -> argparse.ArgumentParser:
    """Create parser for restoration training."""
    train_parser = subparsers.add_parser('train-restore', help='Train restoration model')
    add_common_args(train_parser)

    train_parser.add_argument('--input-dir', type=str, required=True,
                            help='Directory containing input images with artifacts')
    train_parser.add_argument('--mask-dir', type=str, required=True,
                            help='Directory containing artifact masks')
    train_parser.add_argument('--gt-dir', type=str, required=True,
                            help='Directory containing ground truth clean images')
    train_parser.add_argument('--output-dir', type=str, default='checkpoints/restoration',
                            help='Directory to save checkpoints')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')

    return train_parser


def create_infer_parser(subparsers) -> argparse.ArgumentParser:
    """Create parser for inference."""
    infer_parser = subparsers.add_parser('infer', help='Run inference on images')
    add_common_args(infer_parser)

    infer_parser.add_argument('--input-dir', type=str,
                            help='Directory containing input images')
    infer_parser.add_argument('--input-image', type=str,
                            help='Path to single input image')
    infer_parser.add_argument('--output-dir', type=str, required=True,
                            help='Directory to save results')
    infer_parser.add_argument('--seg-checkpoint', type=str,
                            help='Path to segmentation model checkpoint')
    infer_parser.add_argument('--restore-checkpoint', type=str,
                            help='Path to restoration model checkpoint')
    infer_parser.add_argument('--mode', type=str, choices=['seg', 'restore', 'pipeline'],
                            default='pipeline', help='Inference mode')
    infer_parser.add_argument('--batch-size', type=int, help='Batch size for inference')
    infer_parser.add_argument('--save-intermediates', action='store_true',
                            help='Save intermediate results')

    return infer_parser


def create_eval_parser(subparsers) -> argparse.ArgumentParser:
    """Create parser for evaluation."""
    eval_parser = subparsers.add_parser('eval', help='Evaluate model performance')
    add_common_args(eval_parser)

    eval_parser.add_argument('--pred-dir', type=str, required=True,
                           help='Directory containing predictions')
    eval_parser.add_argument('--gt-dir', type=str, required=True,
                           help='Directory containing ground truth')
    eval_parser.add_argument('--mask-dir', type=str,
                           help='Directory containing masks (for restoration eval)')
    eval_parser.add_argument('--output-file', type=str, default='evaluation_results.json',
                           help='Output file for results')
    eval_parser.add_argument('--task', type=str, choices=['segmentation', 'restoration'],
                           default='segmentation', help='Evaluation task type')

    return eval_parser


def create_visualize_parser(subparsers) -> argparse.ArgumentParser:
    """Create parser for visualization."""
    viz_parser = subparsers.add_parser('visualize', help='Create visualizations')
    add_common_args(viz_parser)

    viz_parser.add_argument('--input-dir', type=str, required=True,
                          help='Directory containing images')
    viz_parser.add_argument('--output-dir', type=str, required=True,
                          help='Directory to save visualizations')
    viz_parser.add_argument('--type', type=str, choices=['grid', 'overlay', 'comparison'],
                          default='grid', help='Type of visualization')
    viz_parser.add_argument('--num-samples', type=int, default=10,
                          help='Number of samples to visualize')

    return viz_parser


def setup_main_parser() -> argparse.ArgumentParser:
    """Set up the main argument parser."""
    parser = argparse.ArgumentParser(
        description='Ultrasound Streak Removal Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate synthetic training data
  ultrastreak make-data --input-dir ./data/clean --output-dir ./data/synthetic

  # Train segmentation model
  ultrastreak train-seg --image-dir ./data/train/images --mask-dir ./data/train/masks

  # Train restoration model
  ultrastreak train-restore --input-dir ./data/train/input --mask-dir ./data/train/masks --gt-dir ./data/train/gt

  # Run full pipeline inference
  ultrastreak infer --input-dir ./data/test --output-dir ./results --mode pipeline

  # Evaluate segmentation results
  ultrastreak eval --pred-dir ./results/predictions --gt-dir ./data/test/masks --task segmentation
        """
    )

    parser.add_argument('--version', action='version', version='ultrastreak 0.1.0')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True

    # Create subcommand parsers
    create_data_parser(subparsers)
    create_train_seg_parser(subparsers)
    create_train_restore_parser(subparsers)
    create_infer_parser(subparsers)
    create_eval_parser(subparsers)
    create_visualize_parser(subparsers)

    return parser


def main():
    """Main CLI entry point."""
    parser = setup_main_parser()
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup reproducibility
    setup_reproducibility(args.seed)

    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logger('ultrastreak', level=getattr(__import__('logging'), log_level))

    logger.info(f"Starting ultrastreak command: {args.command}")

    try:
        # Import here to avoid circular imports
        if args.command == 'make-data':
            from .commands import make_data_cmd
            make_data_cmd(args, config, logger)
        elif args.command == 'train-seg':
            from .commands import train_seg_cmd
            train_seg_cmd(args, config, logger)
        elif args.command == 'train-restore':
            from .commands import train_restore_cmd
            train_restore_cmd(args, config, logger)
        elif args.command == 'infer':
            from .commands import infer_cmd
            infer_cmd(args, config, logger)
        elif args.command == 'eval':
            from .commands import eval_cmd
            eval_cmd(args, config, logger)
        elif args.command == 'visualize':
            from .commands import visualize_cmd
            visualize_cmd(args, config, logger)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)

        logger.info("Command completed successfully")

    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
