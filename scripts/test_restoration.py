#!/usr/bin/env python3
"""Run inference and compute PSNR, MSE, and full-image SIR (energy & mean).

This script uses the src ultrastreak pipeline and metrics utilities. It does not
modify or import any code from archive/old_scripts.
"""

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image
import pandas as pd

from ultrastreak.inference import run_inference
from ultrastreak.utils import (
    calculate_psnr_masked,
    calculate_mse_unmasked,
    calculate_sir_full,
)


def _load_grayscale(path: Path, size=None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def compute_metrics_for_item(input_path: Path, restored_path: Path, mask_path: Path) -> Dict[str, float]:
    restored = _load_grayscale(restored_path)
    input_img = _load_grayscale(input_path, size=restored.shape[::-1])
    mask = _load_grayscale(mask_path, size=restored.shape[::-1])
    mask_bin = (mask > 128).astype(np.float32)

    if np.any(mask_bin > 0):
        psnr_v = float(calculate_psnr_masked(input_img, restored, mask_bin))
    else:
        psnr_v = float("nan")

    mse_v = float(calculate_mse_unmasked(input_img, restored, mask_bin))
    sir = calculate_sir_full(restored, mask_bin)

    return {
        "PSNR": psnr_v,
        "MSE": mse_v,
        "SIR_ENERGY_FULL": float(sir["sir_energy_full"]),
        "SIR_MEAN_FULL": float(sir["sir_mean_full"]),
    }


def main():
    p = argparse.ArgumentParser(description="Inference + metrics for restoration")
    p.add_argument("--seg-checkpoint", required=True, help="Path to segmentation checkpoint")
    p.add_argument("--restore-checkpoint", required=True, help="Path to restoration checkpoint")
    p.add_argument("--input-dir", required=True, help="Directory with input images")
    p.add_argument("--output-dir", required=True, help="Directory to save outputs")
    p.add_argument("--csv", default=None, help="Optional path to write per-image metrics CSV")
    args = p.parse_args()

    results = run_inference(
        seg_checkpoint=args.seg_checkpoint,
        restore_checkpoint=args.restore_checkpoint,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode="pipeline",
        config=None,
        save_intermediates=False,
    )

    rows: List[Dict[str, float]] = []
    for item in results:
        input_path = Path(item.get("input_image", ""))
        mask_path = item.get("segmentation_mask")
        restored_path = item.get("restored_image")
        if not (input_path and mask_path and restored_path):
            continue
        m = compute_metrics_for_item(input_path, Path(restored_path), Path(mask_path))
        rows.append({"Filename": input_path.name, **m})

    if not rows:
        print("No metrics computed (no restored outputs/masks found).")
        return

    df = pd.DataFrame(rows)
    mean_row = {
        "Filename": "MEAN_VALUES",
        "PSNR": pd.to_numeric(df["PSNR"], errors="coerce").mean(),
        "MSE": pd.to_numeric(df["MSE"], errors="coerce").mean(),
        "SIR_ENERGY_FULL": pd.to_numeric(df["SIR_ENERGY_FULL"], errors="coerce").mean(),
        "SIR_MEAN_FULL": pd.to_numeric(df["SIR_MEAN_FULL"], errors="coerce").mean(),
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    if args.csv:
        out_csv = Path(args.csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Saved metrics to {out_csv}")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
