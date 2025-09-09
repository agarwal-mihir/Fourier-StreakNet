"""Utilities for ensuring reproducible results."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_deterministic_mode():
    """Set PyTorch to deterministic mode for reproducible results."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For newer PyTorch versions
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)


def get_worker_init_fn(seed: int = 42):
    """Get worker initialization function for DataLoader reproducibility.

    Args:
        seed: Base seed value

    Returns:
        Worker initialization function
    """
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn


def setup_reproducibility(seed: int = 42, deterministic: bool = True):
    """Setup complete reproducibility environment.

    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic mode
    """
    set_seed(seed)

    if deterministic:
        set_deterministic_mode()

    print(f"Random seed set to: {seed}")
    if deterministic:
        print("Deterministic mode enabled")


class ReproducibilityContext:
    """Context manager for reproducible code blocks."""

    def __init__(self, seed: int = 42, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self.original_states = {}

    def __enter__(self):
        # Save current random states
        self.original_states = {
            'python_random': random.getstate(),
            'numpy_random': np.random.get_state(),
            'torch_random': torch.get_rng_state(),
            'torch_cuda_random': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
        }

        # Set reproducible state
        set_seed(self.seed)
        if self.deterministic:
            set_deterministic_mode()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original random states
        random.setstate(self.original_states['python_random'])
        np.random.set_state(self.original_states['numpy_random'])
        torch.set_rng_state(self.original_states['torch_random'])

        if self.original_states['torch_cuda_random'] is not None:
            torch.cuda.set_rng_state_all(self.original_states['torch_cuda_random'])

        torch.backends.cudnn.deterministic = self.original_states['cudnn_deterministic']
        torch.backends.cudnn.benchmark = self.original_states['cudnn_benchmark']


def make_reproducible(func):
    """Decorator to make a function reproducible.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    def wrapper(*args, seed=42, deterministic=True, **kwargs):
        with ReproducibilityContext(seed=seed, deterministic=deterministic):
            return func(*args, **kwargs)
    return wrapper
