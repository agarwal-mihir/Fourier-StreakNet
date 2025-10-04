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
