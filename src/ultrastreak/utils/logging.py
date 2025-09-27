"""Logging utilities for training and inference."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def setup_logger(name: str = 'ultrastreak',
                log_file: Optional[Union[str, Path]] = None,
                level: int = logging.INFO,
                format_string: Optional[str] = None) -> logging.Logger:
    """Set up a logger with file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Custom format string (optional)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def create_experiment_logger(experiment_name: Optional[str] = None,
                           log_dir: Union[str, Path] = 'logs') -> logging.Logger:
    """Create a logger for an experiment with timestamped log file.

    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to save logs

    Returns:
        Configured logger
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = f"experiment_{timestamp}"

    log_dir = Path(log_dir) / experiment_name
    log_file = log_dir / "training.log"

    return setup_logger(
        name=f'ultrastreak_{experiment_name}',
        log_file=log_file,
        level=logging.INFO
    )


class TrainingLogger:
    """Logger for training progress and metrics."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.epoch_losses = []
        self.val_losses = []
        self.metrics_history = {}

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  metrics: Optional[dict] = None, lr: Optional[float] = None):
        """Log training progress for an epoch.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
            metrics: Additional metrics
            lr: Learning rate
        """
        self.epoch_losses.append(train_loss)
        self.val_losses.append(val_loss)

        log_msg = f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"

        if lr is not None:
            log_msg += f", LR = {lr:.6f}"

        if metrics:
            for metric_name, metric_value in metrics.items():
                log_msg += f", {metric_name} = {metric_value:.4f}"
                if metric_name not in self.metrics_history:
                    self.metrics_history[metric_name] = []
                self.metrics_history[metric_name].append(metric_value)

        self.logger.info(log_msg)

    def log_best_model(self, epoch: int, metric_name: str, metric_value: float):
        """Log when a new best model is found.

        Args:
            epoch: Epoch when best model was found
            metric_name: Name of the metric
            metric_value: Value of the metric
        """
        self.logger.info(f"New best model at epoch {epoch}: {metric_name} = {metric_value:.4f}")

    def log_training_start(self, config: dict):
        """Log training configuration at start.

        Args:
            config: Training configuration dictionary
        """
        self.logger.info("Starting training with configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

    def log_training_end(self, total_epochs: int, best_epoch: int, best_metric: dict):
        """Log training completion summary.

        Args:
            total_epochs: Total number of epochs trained
            best_epoch: Epoch with best performance
            best_metric: Best metric values
        """
        self.logger.info(f"Training completed after {total_epochs} epochs")
        self.logger.info(f"Best performance at epoch {best_epoch}:")
        for metric_name, value in best_metric.items():
            self.logger.info(f"  {metric_name}: {value:.4f}")


class InferenceLogger:
    """Logger for inference progress and results."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_inference_start(self, num_samples: int, batch_size: int):
        """Log inference start.

        Args:
            num_samples: Number of samples to process
            batch_size: Batch size for inference
        """
        self.logger.info(f"Starting inference on {num_samples} samples with batch size {batch_size}")

    def log_batch_progress(self, processed: int, total: int):
        """Log batch processing progress.

        Args:
            processed: Number of samples processed
            total: Total number of samples
        """
        progress = processed / total * 100
        self.logger.info(f"Processed {processed}/{total} samples ({progress:.1f}%)")

    def log_inference_results(self, results: dict):
        """Log inference results summary.

        Args:
            results: Dictionary of inference results
        """
        self.logger.info("Inference completed. Results:")
        for metric_name, value in results.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {metric_name}: {value:.4f}")
            else:
                self.logger.info(f"  {metric_name}: {value}")

    def log_error(self, sample_id: str, error: str):
        """Log inference error for a specific sample.

        Args:
            sample_id: Identifier of the problematic sample
            error: Error message
        """
        self.logger.error(f"Error processing sample {sample_id}: {error}")


def log_model_info(model, logger: logging.Logger):
    """Log model architecture information.

    Args:
        model: PyTorch model
        logger: Logger instance
    """
    logger.info(f"Model: {model.__class__.__name__}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Log model structure
    logger.info("Model structure:")
    logger.info(str(model))


