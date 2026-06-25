"""Chemprop Unified v3: Mask gamma1 for ILThermo rows.

Fix: Set gamma1 to NaN for all ILThermo rows so the model learns gamma1
ONLY from original data. ILThermo still contributes H_E (692 values) and
molecular diversity for the D-MPNN backbone.

Combined with oversampling of original data for balanced training.
"""

import sys
import json
import subprocess
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


def prepare_train_data(merged_csv, output_dir, feature_cols, mask_gamma1_ilthermo=True):
    """Prepare balanced training data with optional gamma1 masking."""
    df = pd.read_csv(merged_csv)

    orig = df[df["source"] == "original"].copy()
    ilth = df[df["source"] != "original"].copy()
    n_orig = len(orig)
    n_ilth = len(ilth)

    # Mask gamma1 for ILThermo rows
    if mask_gamma1_ilthermo:
        gamma1_before = ilth["gamma1"].notna().sum()
        ilth["gamma1"] = np.nan
        print(f"    Masked {gamma1_before} ILThermo gamma1 values → NaN")

    # Oversample original to balance
    repeat = max(1, round(n_ilth / n_orig))
    orig_repeated = pd.concat([orig] * repeat, ignore_index=True)
    df_balanced = pd.concat([orig_repeated, ilth], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"    {n_orig} original × {repeat} = {len(orig_repeated)}, "
          f"{n_ilth} ILThermo → {len(df_balanced)} total")

    # Count non-NaN per target
    for t in TARGET_COLUMNS:
        n_valid = df_balanced[t].notna().sum()
        print(f"    {t:15s}: {n_valid} non-NaN values")

    # Save data
    out = pd.DataFrame()
    out["smiles"] = df_balanced["smiles"]
    for t in TARGET_COLUMNS:
        out[t] = df_balanced[t]
    out.to_csv(output_dir / "train.csv", index=False)

    feat_df = pd.DataFrame()
    for f in feature_cols:
        feat_df[f] = df_balanced[f].fillna(0.0) if f in df_balanced.columns else 0.0
    feat_df.to_csv(output_dir / "train_features.csv", index=False)

    return len(df_balanced)


def prepare_eval_data(csv_path, output_dir, prefix, feature_cols):
    """Prepare val/test data (no oversampling, no masking)."""
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
    return len(out)


def train_chemprop(name, data_dir, ckpt_dir, epochs=100, seed=42):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
        "--hidden_size", "300",
        "--depth", "3",
        "--ffn_num_layers", "2",
        "--ffn_hidden_size", "300",
        "--dropout", "0.2",
        "--metric", "rmse",
        "--extra_metrics", "r2", "mae",
        "--seed", str(seed),
        "--num_folds", "1",
        "--gpu", "0",
        "--quiet",
    ]

    print(f"\n  Training {name}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-300:]}")
        return None

    scores_path = ckpt_dir / "fold_0" / "test_scores.json"
    if scores_path.exists():
        scores = json.load(open(scores_path))
        metrics = {}
        for i, p in enumerate(TARGET_COLUMNS):
            metrics[f"{p}_r2"] = scores["r2"][i]
            metrics[f"{p}_mae"] = scores["mae"][i]
            metrics[f"{p}_rmse"] = scores["rmse"][i]
        metrics["avg_r2"] = np.mean(scores["r2"])
        return metrics
    return None


def main():
    print("=== Chemprop Unified v3: Gamma1 Masked for ILThermo ===\n")

    merged_dir = Path("data/merged_v5")
    feats_all = THERMO_FEATURES + SURFACE_FEATURES

    data_dir = Path("data/chemprop_unified_v3")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    print("Preparing training data (gamma1 masked for ILThermo, original oversampled)...")
    n_train = prepare_train_data(
        merged_dir / "splits/train.csv", data_dir, feats_all, mask_gamma1_ilthermo=True)

    n_val = prepare_eval_data(merged_dir / "splits/val.csv", data_dir, "val", feats_all)
    n_test = prepare_eval_data(merged_dir / "splits/test.csv", data_dir, "test", feats_all)
    print(f"\n  Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # Train
    print(f"\n{'='*60}")
    print("UNIFIED v3: D-MPNN + Surface + ILThermo + Oversampling + Gamma1 Mask")
    print(f"{'='*60}")

    m = train_chemprop("Unified v3", data_dir, "checkpoints/chemprop_unified_v3")

    if m:
        print(f"\n  Results:")
        for p in TARGET_COLUMNS:
            print(f"    {p:15s} R² = {m[f'{p}_r2']:.4f}")
        print(f"    {'AVERAGE':15s} R² = {m['avg_r2']:.4f}")

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    prev = {}
    for name, path, key in [
        ("Chemprop (base)", "results/chemprop_results.json", "test_metrics"),
        ("Unified v2", "results/chemprop_unified_v2_results.json", "metrics"),
        ("PointCloud", "results/pointcloud_results.json", None),
        ("MoE Fix6", "results/moe_fix6_results.json", "metrics"),
        ("Ens (top-2)", "results/ensemble_all_models_results.json", "ENS"),
    ]:
        try:
            data = json.load(open(path))
            if key == "ENS":
                p_m = data.get("top2_average", {}).get("metrics", {})
            elif key:
                p_m = data.get(key, {})
            else:
                for k in ['metrics', 'test_metrics']:
                    if k in data: p_m = data[k]; break
            if p_m:
                prev[name] = p_m
        except:
            pass

    if m:
        prev["Unified v3"] = m

    header = "  {:<12s}".format("Property")
    for name in prev:
        header += " {:>14s}".format(name[:14])
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<12s}".format(p)
        for name, pm in prev.items():
            line += " {:14.4f}".format(pm.get(key, float('nan')))
        print(line)

    line = "  {:<12s}".format("AVERAGE")
    for name, pm in prev.items():
        line += " {:14.4f}".format(pm.get('avg_r2', float('nan')))
    print(line)

    # Highlight vs Chemprop base
    if m and "Chemprop (base)" in prev:
        base = prev["Chemprop (base)"]
        print(f"\n  vs Chemprop (base):")
        for p in TARGET_COLUMNS:
            key = f"{p}_r2"
            delta = m[key] - base[key]
            sign = "+" if delta > 0 else ""
            marker = "WIN" if delta > 0 else "LOSE"
            print(f"    {p:15s}: {m[key]:.4f} vs {base[key]:.4f} ({sign}{delta:.4f}) {marker}")
        delta_avg = m['avg_r2'] - base['avg_r2']
        sign = "+" if delta_avg > 0 else ""
        print(f"    {'AVERAGE':15s}: {m['avg_r2']:.4f} vs {base['avg_r2']:.4f} ({sign}{delta_avg:.4f})")

    # Save
    results = {
        "model": "chemprop_unified_v3",
        "description": "Chemprop D-MPNN + 25 features (5 thermo + 20 surface) + "
                       "ILThermo merged_v5 with gamma1 masked for ILThermo rows + "
                       "original oversampled 24x",
        "metrics": {k: float(v) for k, v in m.items()} if m else {},
    }
    with open("results/chemprop_unified_v3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/chemprop_unified_v3_results.json")


if __name__ == "__main__":
    main()
