"""Inference pipeline for ultrasound streak removal."""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm

from ..models import UNet, UNetWithFourierAttention
from ..utils.io import ensure_dir, get_file_list
from ..utils.metrics import seam_intensity_match


class InferencePipeline:
    """Complete inference pipeline for streak detection and removal."""

    def __init__(self, seg_checkpoint: str, restore_checkpoint: Optional[str] = None, 
                 config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        """Initialize the inference pipeline.

        Args:
            seg_checkpoint: Path to segmentation model checkpoint
            restore_checkpoint: Path to restoration model checkpoint (optional)
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load models
        self.seg_model = self._load_segmentation_model(seg_checkpoint)
        self.restore_model = None
        if restore_checkpoint:
            self.restore_model = self._load_restoration_model(restore_checkpoint)

        self.logger.info(f"Inference pipeline initialized on device: {self.device}")

    def _load_segmentation_model(self, checkpoint_path: str) -> UNet:
        """Load the segmentation model."""
        self.logger.info(f"Loading segmentation model from {checkpoint_path}")
        
        model = UNet(in_channels=1, out_channels=1)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        return model

    def _load_restoration_model(self, checkpoint_path: str) -> UNetWithFourierAttention:
        """Load the restoration model."""
        self.logger.info(f"Loading restoration model from {checkpoint_path}")
        
        model = UNetWithFourierAttention(n_channels=2, n_classes=1)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        return model

    def _preprocess_image(self, image_path: Path) -> torch.Tensor:
        """Preprocess image for model input."""
        image = Image.open(image_path).convert('L')
        image_np = np.array(image, dtype=np.float32) / 255.0
        
        # Resize to expected size if needed
        if image_np.shape != (256, 256):
            image = Image.fromarray((image_np * 255).astype(np.uint8))
            image = image.resize((256, 256), Image.BICUBIC)
            image_np = np.array(image, dtype=np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        return image_tensor.to(self.device)

    def _segment_streaks(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Run segmentation to detect streaks."""
        with torch.no_grad():
            logits = self.seg_model(image_tensor)
            # Apply sigmoid and threshold to get binary mask
            mask = torch.sigmoid(logits) > 0.5
            mask = mask.float()
        return mask

    def _restore_image(self, image_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
        """Restore image using detected streak mask."""
        # Concatenate image and mask as input to restoration model
        input_tensor = torch.cat([image_tensor, mask_tensor], dim=1)  # [1, 2, H, W]

        with torch.no_grad():
            pred = torch.sigmoid(self.restore_model(input_tensor))
            # Blend prediction with input based on mask (keep outside region as original)
            pred = pred * mask_tensor + image_tensor * (1 - mask_tensor)
            # Seam intensity match to reduce boundary artifacts
            pred = seam_intensity_match(image_tensor, pred, mask_tensor, ring_size=5)
            pred = torch.clamp(pred, 0.0, 1.0)

        return pred

    def _postprocess_output(self, output_tensor: torch.Tensor) -> np.ndarray:
        """Postprocess model output to save as image."""
        output_np = output_tensor.squeeze().cpu().numpy()
        output_np = (output_np * 255).astype(np.uint8)
        return output_np

    def run_single_image(self, image_path: Path, output_dir: Path, 
                        mode: str = 'pipeline', save_intermediates: bool = False) -> Dict[str, Any]:
        """Run inference on a single image."""
        self.logger.info(f"Processing {image_path.name} with mode: {mode}")

        # Load and preprocess image
        image_tensor = self._preprocess_image(image_path)
        self.logger.info(f"Image tensor shape: {image_tensor.shape}")

        results = {}

        if mode in ['seg', 'segmentation', 'pipeline']:
            self.logger.info("Running segmentation...")
            # Run segmentation
            mask_tensor = self._segment_streaks(image_tensor)
            mask_np = self._postprocess_output(mask_tensor)
            
            # Save segmentation mask
            mask_path = output_dir / f"{image_path.stem}_mask.png"
            self.logger.info(f"Saving segmentation mask to {mask_path}")
            Image.fromarray(mask_np).save(mask_path)
            results['segmentation_mask'] = str(mask_path)

            if save_intermediates:
                # Save masked image for visualization
                masked_image = image_tensor.squeeze().cpu().numpy() * (mask_tensor.squeeze().cpu().numpy() > 0.5)
                masked_image = (masked_image * 255).astype(np.uint8)
                masked_path = output_dir / f"{image_path.stem}_masked.png"
                Image.fromarray(masked_image).save(masked_path)

        if mode in ['restoration', 'pipeline']:
            if mode == 'pipeline':
                # Use the segmentation mask for restoration
                mask_tensor = self._segment_streaks(image_tensor)
            else:
                # For restoration-only mode, create a dummy mask (full image)
                mask_tensor = torch.ones_like(image_tensor)

            # Run restoration
            if self.restore_model is not None:
                restored_tensor = self._restore_image(image_tensor, mask_tensor)
                restored_np = self._postprocess_output(restored_tensor)
                
                # Save restored image
                restored_path = output_dir / f"{image_path.stem}_restored.png"
                Image.fromarray(restored_np).save(restored_path)
                results['restored_image'] = str(restored_path)
            else:
                self.logger.warning("Restoration model not loaded, skipping restoration")

        return results

    def run(self, input_files: List[Path], output_dir: Path, 
           mode: str = 'pipeline', save_intermediates: bool = False) -> List[Dict[str, Any]]:
        """Run inference on multiple images."""
        self.logger.info(f"Running {mode} inference on {len(input_files)} images")

        # Create output directory
        output_dir = ensure_dir(output_dir)

        all_results = []

        for image_path in tqdm(input_files, desc="Processing images"):
            try:
                result = self.run_single_image(
                    image_path, output_dir, mode, save_intermediates
                )
                result['input_image'] = str(image_path)
                all_results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {str(e)}")
                all_results.append({
                    'input_image': str(image_path),
                    'error': str(e)
                })

        self.logger.info(f"Inference completed. Processed {len(all_results)} images.")
        return all_results


def run_inference(seg_checkpoint: str, restore_checkpoint: str, 
                 input_dir: str, output_dir: str, mode: str = 'pipeline',
                 config: Optional[Dict[str, Any]] = None, 
                 save_intermediates: bool = False) -> List[Dict[str, Any]]:
    """Convenience function to run inference."""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Default configuration
    if config is None:
        config = {
            'inference': {
                'batch_size': 1,
                'device': 'auto'
            }
        }

    # Get input files
    input_path = Path(input_dir)
    if input_path.is_file():
        input_files = [input_path]
    else:
        input_files = get_file_list(input_path)

    if not input_files:
        raise ValueError(f"No image files found in {input_dir}")

    # Create pipeline
    pipeline = InferencePipeline(
        seg_checkpoint=seg_checkpoint,
        restore_checkpoint=restore_checkpoint,
        config=config,
        logger=logger
    )

    # Run inference
    results = pipeline.run(
        input_files=input_files,
        output_dir=Path(output_dir),
        mode=mode,
        save_intermediates=save_intermediates
    )

    return results