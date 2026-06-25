"""Chemprop variants with gamma1 masked for ILThermo.

Trains a specific variant based on command-line argument.
Usage: python train_chemprop_variants.py --variant [b|c|d]
"""

import sys
import json
import subprocess
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessing import TARGET_COLUMNS

THERMO_FEATURES = ["temperature", "x1", "inv_temperature", "temp_squared", "temp_cubed"]
SURFACE_FEATURES = [
    "surface_area", "volume", "sphericity", "aspect_ratio",
    "curv_mean", "curv_std", "curv_skew",
    "gcurv_mean", "gcurv_std", "gcurv_skew",
    "esp_mean", "esp_std", "esp_min", "esp_max", "esp_skew", "esp_kurtosis",
    "esp_pos_frac", "esp_neg_frac", "esp_charge_segregation", "esp_range",
]


def prepare_data(merged_csv, output_dir, feature_cols, mask_targets_ilthermo):
    """Prepare balanced data with specified targets masked for ILThermo."""
    df = pd.read_csv(merged_csv)
    orig = df[df["source"] == "original"].copy()
    ilth = df[df["source"] != "original"].copy()

    for t in mask_targets_ilthermo:
        n_before = ilth[t].notna().sum()
        ilth[t] = np.nan
        print(f"    Masked ILThermo {t}: {n_before} → NaN")

    repeat = max(1, round(len(ilth) / max(len(orig), 1)))
    orig_rep = pd.concat([orig] * repeat, ignore_index=True)
    df_bal = pd.concat([orig_rep, ilth], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"    {len(orig)} orig × {repeat} + {len(ilth)} ILTh = {len(df_bal)} total")

    out = pd.DataFrame()
    out["smiles"] = df_bal["smiles"]
    for t in TARGET_COLUMNS:
        out[t] = df_bal[t] if t in df_bal.columns else np.nan
    out.to_csv(output_dir / "train.csv", index=False)

    feat_df = pd.DataFrame()
    for f in feature_cols:
        feat_df[f] = df_bal[f].fillna(0.0) if f in df_bal.columns else 0.0
    feat_df.to_csv(output_dir / "train_features.csv", index=False)
    return len(df_bal)


def prepare_eval(csv_path, output_dir, prefix, feature_cols):
    df = pd.read_csv(csv_path)
    out = pd.DataFrame()
    out["smiles"] = df["smiles"]
    for t in TARGET_COLUMNS:
        out[t] = df[t] if t in df.columns else np.nan
    out.to_csv(output_dir / f"{prefix}.csv", index=False)
    feat_df = pd.DataFrame()
    for f in feature_cols:
        feat_df[f] = df[f].fillna(0.0) if f in df.columns else 0.0
    feat_df.to_csv(output_dir / f"{prefix}_features.csv", index=False)


def train(data_dir, ckpt_dir, hidden=300, depth=3, ffn_hidden=300, epochs=100, seed=42):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "chemprop_train",
        "--data_path", str(data_dir / "train.csv"),
        "--separate_val_path", str(data_dir / "val.csv"),
        "--separate_test_path", str(data_dir / "test.csv"),
        "--features_path", str(data_dir / "train_features.csv"),
        "--separate_val_features_path", str(data_dir / "val_features.csv"),
        "--separate_test_features_path", str(data_dir / "test_features.csv"),
        "--save_dir", str(ckpt_dir),
        "--dataset_type", "regression",
        "--smiles_columns", "smiles",
        "--target_columns", *TARGET_COLUMNS,
        "--epochs", str(epochs),
        "--batch_size", "32",
        "--hidden_size", str(hidden),
        "--depth", str(depth),
        "--ffn_num_layers", "2",
        "--ffn_hidden_size", str(ffn_hidden),
        "--dropout", "0.2",
        "--metric", "rmse",
        "--extra_metrics", "r2", "mae",
        "--seed", str(seed),
        "--num_folds", "1",
        "--gpu", "0",
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-300:]}")
        return None
    scores_path = Path(ckpt_dir) / "fold_0" / "test_scores.json"
    if scores_path.exists():
        scores = json.load(open(scores_path))
        metrics = {}
        for i, p in enumerate(TARGET_COLUMNS):
            metrics[f"{p}_r2"] = scores["r2"][i]
        metrics["avg_r2"] = np.mean(scores["r2"])
        return metrics
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["b", "c", "d"])
    args = parser.parse_args()

    merged_dir = Path("data/merged_v5")

    configs = {
        "b": {
            "name": "v3b: Thermo only + gamma1 masked",
            "features": THERMO_FEATURES,
            "mask": ["gamma1"],
            "hidden": 300, "depth": 3, "ffn_hidden": 300,
        },
        "c": {
            "name": "v3c: Surface + gamma1 masked + deeper",
            "features": THERMO_FEATURES + SURFACE_FEATURES,
            "mask": ["gamma1"],
            "hidden": 400, "depth": 4, "ffn_hidden": 400,
        },
        "d": {
            "name": "v3d: Surface + gamma1 & H_E masked",
            "features": THERMO_FEATURES + SURFACE_FEATURES,
            "mask": ["gamma1", "H_E"],
            "hidden": 300, "depth": 3, "ffn_hidden": 300,
        },
    }

    cfg = configs[args.variant]
    print(f"=== {cfg['name']} ===\n")

    data_dir = Path(f"data/chemprop_v3{args.variant}")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing data...")
    n = prepare_data(merged_dir / "splits/train.csv", data_dir, cfg["features"], cfg["mask"])
    prepare_eval(merged_dir / "splits/val.csv", data_dir, "val", cfg["features"])
    prepare_eval(merged_dir / "splits/test.csv", data_dir, "test", cfg["features"])

    print(f"\nTraining {cfg['name']}...")
    print(f"  Hidden: {cfg['hidden']}, Depth: {cfg['depth']}, FFN: {cfg['ffn_hidden']}")
    m = train(data_dir, f"checkpoints/chemprop_v3{args.variant}",
              hidden=cfg["hidden"], depth=cfg["depth"], ffn_hidden=cfg["ffn_hidden"])

    if m:
        print(f"\n  Results:")
        for p in TARGET_COLUMNS:
            print(f"    {p:15s} R² = {m[f'{p}_r2']:.4f}")
        print(f"    {'AVERAGE':15s} R² = {m['avg_r2']:.4f}")

        # Compare with base Chemprop
        try:
            base = json.load(open("results/chemprop_results.json"))["test_metrics"]
            print(f"\n  vs Chemprop base:")
            for p in TARGET_COLUMNS:
                d = m[f"{p}_r2"] - base[f"{p}_r2"]
                s = "+" if d > 0 else ""
                print(f"    {p:15s}: {m[f'{p}_r2']:.4f} vs {base[f'{p}_r2']:.4f} ({s}{d:.4f})")
            d = m["avg_r2"] - base["avg_r2"]
            s = "+" if d > 0 else ""
            print(f"    {'AVERAGE':15s}: {m['avg_r2']:.4f} vs {base['avg_r2']:.4f} ({s}{d:.4f})")
        except:
            pass

        with open(f"results/chemprop_v3{args.variant}_results.json", "w") as f:
            json.dump({"name": cfg["name"], "metrics": m, "config": {
                "features": len(cfg["features"]), "masked": cfg["mask"],
                "hidden": cfg["hidden"], "depth": cfg["depth"]}}, f, indent=2)
        print(f"\nSaved: results/chemprop_v3{args.variant}_results.json")


if __name__ == "__main__":
    main()
