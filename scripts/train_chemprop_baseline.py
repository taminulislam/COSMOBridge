"""Chemprop (D-MPNN) baseline for IL property prediction.

Uses chemprop CLI to train Directed Message Passing Neural Network.
Reference: Yang et al., J. Chem. Inf. Model., 2019.
"""

import sys
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.training.metrics import compute_metrics, format_metrics

TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]
EXTRA_FEATURES = ["temperature", "x1", "inv_temperature", "temp_squared", "temp_cubed"]


def prepare_data():
    """Prepare chemprop-compatible CSV files."""
    tmp_dir = Path("data/chemprop_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        df = pd.read_csv(f"data/processed/splits/{split}.csv")
        out = pd.DataFrame()
        out["smiles"] = df["smiles"]
        for feat in EXTRA_FEATURES:
            if feat in df.columns:
                out[feat] = df[feat]
        for t in TARGET_COLUMNS:
            out[t] = df[t]
        out.to_csv(tmp_dir / f"{split}.csv", index=False)
        print(f"  {split}: {len(out)} samples")

    # Features-only files (chemprop needs separate features files)
    for split in ["train", "val", "test"]:
        df = pd.read_csv(f"data/processed/splits/{split}.csv")
        feat_df = df[EXTRA_FEATURES].copy()
        feat_df.to_csv(tmp_dir / f"{split}_features.csv", index=False)

    return tmp_dir


def main():
    print("=== Chemprop (D-MPNN) Baseline ===\n")

    print("Preparing data...")
    tmp_dir = prepare_data()

    ckpt_dir = Path("checkpoints/chemprop")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build chemprop command
    cmd = [
        "chemprop_train",
        "--data_path", str(tmp_dir / "train.csv"),
        "--separate_val_path", str(tmp_dir / "val.csv"),
        "--separate_test_path", str(tmp_dir / "test.csv"),
        "--features_path", str(tmp_dir / "train_features.csv"),
        "--separate_val_features_path", str(tmp_dir / "val_features.csv"),
        "--separate_test_features_path", str(tmp_dir / "test_features.csv"),
        "--save_dir", str(ckpt_dir),
        "--dataset_type", "regression",
        "--smiles_columns", "smiles",
        "--target_columns", *TARGET_COLUMNS,
        "--epochs", "100",
        "--batch_size", "32",
        "--hidden_size", "300",
        "--depth", "3",
        "--ffn_num_layers", "2",
        "--ffn_hidden_size", "300",
        "--dropout", "0.2",
        "--metric", "rmse",
        "--extra_metrics", "r2", "mae",
        "--seed", "42",
        "--num_folds", "1",
        "--gpu", "0",
        "--quiet",
    ]

    print(f"Running chemprop training...")
    print(f"  Command: {' '.join(cmd[:10])}...")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    print(result.stdout[-2000:] if result.stdout else "")
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr[-2000:]}")

    # Find and load test predictions
    test_preds_path = None
    for f in ckpt_dir.rglob("test_preds.csv"):
        test_preds_path = f
        break

    if test_preds_path and test_preds_path.exists():
        pred_df = pd.read_csv(test_preds_path)
        test_df = pd.read_csv(tmp_dir / "test.csv")

        preds = pred_df[TARGET_COLUMNS].values
        targets = test_df[TARGET_COLUMNS].values

        metrics = compute_metrics(preds, targets)
        print(f"\n{'='*60}")
        print("CHEMPROP TEST RESULTS")
        print(f"{'='*60}")
        print(format_metrics(metrics, "Chemprop D-MPNN"))

        print(f"\n  Per-property R²:")
        for p in TARGET_COLUMNS:
            print(f"    {p:15s} R² = {metrics[f'{p}_r2']:.4f}")
        print(f"    {'AVERAGE':15s} R² = {metrics['avg_r2']:.4f}")

        results = {
            "model": "chemprop_dmpnn",
            "reference": "Yang et al., J. Chem. Inf. Model., 2019",
            "test_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                             for k, v in metrics.items()},
        }
        with open("results/chemprop_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: results/chemprop_results.json")
    else:
        print("WARNING: Could not find test predictions file")
        print(f"  Searched in: {ckpt_dir}")
        for f in ckpt_dir.rglob("*"):
            print(f"    {f}")


if __name__ == "__main__":
    main()
