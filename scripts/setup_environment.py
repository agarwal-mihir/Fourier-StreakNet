#!/usr/bin/env python3
"""Setup script for ultrastreak development environment."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e.stderr}")
        return False


def main():
    """Set up the development environment."""
    print("Setting up ultrastreak development environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install package in development mode
    success = run_command("pip install -e .", "Installing ultrastreak in development mode")
    if not success:
        sys.exit(1)
    
    # Install development dependencies
    success = run_command("pip install -e '.[dev]'", "Installing development dependencies")
    if not success:
        print("Warning: Development dependencies installation failed")
    
    # Set up pre-commit hooks (if available)
    success = run_command("pre-commit install", "Setting up pre-commit hooks")
    if not success:
        print("Warning: Pre-commit hooks setup failed (optional)")
    
    # Create necessary directories
    dirs_to_create = [
        "logs",
        "outputs",
        "experiments"
    ]
    
    for dir_name in dirs_to_create:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created directory: {dir_name}")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Verify installation: ultrastreak --help")
    print("2. Generate sample data: ultrastreak make-data --help")
    print("3. Train a model: ultrastreak train-seg --help")


if __name__ == "__main__":
    main()
