"""Chemprop Unified: D-MPNN + Surface Descriptors + ILThermo Transfer.

Single model combining all winning ingredients:
1. Chemprop's D-MPNN (best single-model architecture)
2. 20 COSMO surface descriptors as molecule-level features (not per-atom)
3. ILThermo merged_v5 data with balanced sampling
4. 5 thermodynamic features

Total features: 25 (5 thermo + 20 surface)
Training data: 3,806 samples (merged_v5, balanced 50/50)

Also trains ablations:
A. Chemprop + surface descriptors (original data only)
B. Chemprop + ILThermo (thermo features only)
C. Chemprop + surface + ILThermo (full unified)
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
from src.training.metrics import compute_metrics, format_metrics

THERMO_FEATURES = ["temperature", "x1", "inv_temperature", "temp_squared", "temp_cubed"]
SURFACE_FEATURES = [
    "surface_area", "volume", "sphericity", "aspect_ratio",
    "curv_mean", "curv_std", "curv_skew",
    "gcurv_mean", "gcurv_std", "gcurv_skew",
    "esp_mean", "esp_std", "esp_min", "esp_max", "esp_skew", "esp_kurtosis",
    "esp_pos_frac", "esp_neg_frac", "esp_charge_segregation", "esp_range",
]


def prepare_chemprop_data(csv_path, output_dir, prefix, feature_cols):
    """Prepare data + feature files for Chemprop."""
    df = pd.read_csv(csv_path)

    # Main data: SMILES + targets
    out = pd.DataFrame()
    out["smiles"] = df["smiles"]
    for t in TARGET_COLUMNS:
        if t in df.columns:
            out[t] = df[t]
        else:
            out[t] = np.nan
    out.to_csv(output_dir / f"{prefix}.csv", index=False)

    # Features
    feat_df = pd.DataFrame()
    for f in feature_cols:
        if f in df.columns:
            feat_df[f] = df[f].fillna(0.0)
        else:
            feat_df[f] = 0.0
    feat_df.to_csv(output_dir / f"{prefix}_features.csv", index=False)

    return len(out)


def train_chemprop(name, data_dir, ckpt_dir, epochs=100, seed=42):
    """Train Chemprop and return test metrics."""
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
        metrics["avg_mae"] = np.mean(scores["mae"])
        metrics["avg_rmse"] = np.mean(scores["rmse"])
        return metrics
    return None


def main():
    print("=== Chemprop Unified: D-MPNN + Surface Descriptors + ILThermo ===\n")

    orig_splits = Path("data/processed/splits")
    merged_dir = Path("data/merged_v5")

    # ══════════════════════════════════════════════════════════
    # Prepare data variants
    # ══════════════════════════════════════════════════════════

    # A: Original data + surface descriptors (25 features)
    dir_a = Path("data/chemprop_unified/orig_surface")
    dir_a.mkdir(parents=True, exist_ok=True)
    print("Preparing A: Original + surface descriptors...")
    feats_a = THERMO_FEATURES + SURFACE_FEATURES
    for split in ["train", "val", "test"]:
        n = prepare_chemprop_data(orig_splits / f"{split}.csv", dir_a, split, feats_a)
        print(f"  {split}: {n} samples, {len(feats_a)} features")

    # B: Merged_v5 + thermo only (5 features) — tests ILThermo value
    dir_b = Path("data/chemprop_unified/merged_thermo")
    dir_b.mkdir(parents=True, exist_ok=True)
    print("\nPreparing B: Merged_v5 + thermo only...")
    feats_b = THERMO_FEATURES
    for split in ["train", "val", "test"]:
        n = prepare_chemprop_data(merged_dir / f"splits/{split}.csv", dir_b, split, feats_b)
        print(f"  {split}: {n} samples, {len(feats_b)} features")

    # C: Merged_v5 + surface + thermo (25 features) — full unified
    dir_c = Path("data/chemprop_unified/merged_surface")
    dir_c.mkdir(parents=True, exist_ok=True)
    print("\nPreparing C: Merged_v5 + surface + thermo (UNIFIED)...")
    feats_c = THERMO_FEATURES + SURFACE_FEATURES
    for split in ["train", "val", "test"]:
        n = prepare_chemprop_data(merged_dir / f"splits/{split}.csv", dir_c, split, feats_c)
        print(f"  {split}: {n} samples, {len(feats_c)} features")

    # ══════════════════════════════════════════════════════════
    # Train all variants
    # ══════════════════════════════════════════════════════════
    results = {}

    # Baseline: Chemprop original (already trained, load scores)
    print(f"\n{'='*60}")
    print("BASELINE: Chemprop (thermo only, original data)")
    print(f"{'='*60}")
    try:
        base = json.load(open("results/chemprop_results.json"))["test_metrics"]
        results["baseline"] = {"name": "Chemprop (base)", "metrics": base}
        print(f"  avg R² = {base['avg_r2']:.4f}")
    except:
        pass

    # A: Original + surface
    print(f"\n{'='*60}")
    print("A: Chemprop + Surface Descriptors (original data, 25 features)")
    print(f"{'='*60}")
    m_a = train_chemprop("A", dir_a, "checkpoints/chemprop_unified/orig_surface")
    if m_a:
        results["orig_surface"] = {"name": "CP + Surface (orig)", "metrics": m_a}
        print(f"  avg R² = {m_a['avg_r2']:.4f}")
        for p in TARGET_COLUMNS:
            print(f"    {p:15s} R² = {m_a[f'{p}_r2']:.4f}")

    # B: Merged + thermo
    print(f"\n{'='*60}")
    print("B: Chemprop + ILThermo (merged_v5, thermo only, 5 features)")
    print(f"{'='*60}")
    m_b = train_chemprop("B", dir_b, "checkpoints/chemprop_unified/merged_thermo")
    if m_b:
        results["merged_thermo"] = {"name": "CP + ILThermo (thermo)", "metrics": m_b}
        print(f"  avg R² = {m_b['avg_r2']:.4f}")
        for p in TARGET_COLUMNS:
            print(f"    {p:15s} R² = {m_b[f'{p}_r2']:.4f}")

    # C: Merged + surface + thermo (UNIFIED)
    print(f"\n{'='*60}")
    print("C: UNIFIED — Chemprop + Surface + ILThermo (25 features, 3806 samples)")
    print(f"{'='*60}")
    m_c = train_chemprop("C (unified)", dir_c, "checkpoints/chemprop_unified/merged_surface")
    if m_c:
        results["unified"] = {"name": "CP Unified", "metrics": m_c}
        print(f"  avg R² = {m_c['avg_r2']:.4f}")
        for p in TARGET_COLUMNS:
            print(f"    {p:15s} R² = {m_c[f'{p}_r2']:.4f}")

    # ══════════════════════════════════════════════════════════
    # COMPARISON
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")

    # Add our best models for context
    for name, path, key in [
        ("PointCloud", "results/pointcloud_results.json", None),
        ("MoE Fix6", "results/moe_fix6_results.json", "metrics"),
    ]:
        try:
            data = json.load(open(path))
            m = data.get(key) if key else None
            if m is None:
                for k in ['metrics', 'test_metrics']:
                    if k in data: m = data[k]; break
            results[name.lower().replace(" ", "_")] = {"name": name, "metrics": m}
        except:
            pass

    header = "  {:<12s}".format("Property")
    for key, info in results.items():
        header += " {:>16s}".format(info["name"][:16])
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        k = f"{p}_r2"
        line = "  {:<12s}".format(p)
        for key, info in results.items():
            v = info["metrics"].get(k, float("nan"))
            line += " {:16.4f}".format(v)
        print(line)

    line = "  {:<12s}".format("AVERAGE")
    for key, info in results.items():
        v = info["metrics"].get("avg_r2", float("nan"))
        line += " {:16.4f}".format(v)
    print(line)

    # Save
    save_results = {}
    for key, info in results.items():
        save_results[key] = {
            "name": info["name"],
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                        for k, v in info["metrics"].items()},
        }
    with open("results/chemprop_unified_results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved: results/chemprop_unified_results.json")


if __name__ == "__main__":
    main()
