"""Cache Chemprop atom-surface (Path C) predictions for v4 I3.

Uses the existing chemprop_as_feat checkpoint that achieves:
  γ₁=0.826, γ₂=0.892, G_E=0.649, H_E=0.661, G_mix=0.639, H_vap=0.621, P=0.760

Runs chemprop_predict on train/val/test splits and saves (N, 7) standardized
predictions to cosmobridge_v4/data/preds_atom_surface_{split}.npy

This produces the third frozen path for the 3-path router (I3).
"""

import sys
import numpy as np
import pandas as pd
import subprocess
import tempfile
import pickle
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.preprocessing import TARGET_COLUMNS


def run_chemprop_predict(split):
    """Run chemprop_predict on a split with atom-surface features."""
    data_dir = Path("data/chemprop_atom_surface")
    out_file = tempfile.mktemp(suffix=".csv")

    cmd = [
        "chemprop_predict",
        "--test_path", str(data_dir / f"{split}.csv"),
        "--features_path", str(data_dir / f"{split}_features.csv"),
        "--atom_descriptors", "feature",
        "--atom_descriptors_path", str(data_dir / f"{split}_atom_descriptors.npz"),
        "--checkpoint_dir", "checkpoints/chemprop_atom_surface/chemprop_as_feat",
        "--preds_path", out_file,
        "--num_workers", "0",
    ]
    print(f"  Running: {' '.join(cmd[:4])} ... --preds_path {out_file}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-500:]}")
        raise RuntimeError(f"chemprop_predict failed for {split}")

    # Read predictions
    df = pd.read_csv(out_file)
    # Expected columns: smiles, then 7 target columns
    cols = [c for c in df.columns if c in TARGET_COLUMNS]
    if len(cols) != 7:
        # Sometimes Chemprop uses different naming
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 7:
            cols = list(numeric.columns)
        else:
            raise ValueError(f"Unexpected columns: {df.columns.tolist()}")
    preds = df[TARGET_COLUMNS].values.astype(np.float32)
    Path(out_file).unlink(missing_ok=True)
    return preds


def main():
    output_dir = Path("cosmobridge_v4/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        print(f"\n=== {split.upper()} ===")
        preds = run_chemprop_predict(split)
        print(f"  Predictions shape: {preds.shape}")
        print(f"  Mean: {preds.mean():.3f}, Std: {preds.std():.3f}")

        # Save
        out_path = output_dir / f"preds_atom_surface_{split}.npy"
        np.save(out_path, preds)
        print(f"  Saved: {out_path}")

        # Sanity check: compute R² against cached targets
        cached = np.load(f"cosmobridge_v4/data/cached_{split}.npz")
        targets = cached["targets"]
        assert preds.shape == targets.shape, f"Shape mismatch: {preds.shape} vs {targets.shape}"

        # Check if predictions are already in standardized space or raw space
        # Targets are already standardized. Check if AS preds are too.
        ss_res = ((preds - targets) ** 2).sum(axis=0)
        ss_tot = ((targets - targets.mean(axis=0)) ** 2).sum(axis=0)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        print(f"  Per-property R² (if standardized):")
        for i, p in enumerate(TARGET_COLUMNS):
            print(f"    {p}: {r2[i]:.4f}")

    print("\nDone. Next: run train_v4_triple.py")


if __name__ == "__main__":
    main()
