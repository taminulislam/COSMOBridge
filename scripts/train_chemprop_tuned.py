"""Chemprop tuned variants: partial gamma1 masking + oversampling ratios.

Variant A: Partial gamma1 mask — keep 50% of ILThermo gamma1 (random subset)
Variant B: Lower oversampling — 10x instead of 24x (more ILThermo influence)
Variant C: Higher oversampling — 48x (even more original influence)
Variant D: Partial mask + lower oversample (A+B combined)
Variant E: v2 recipe (no mask) but with 10x oversample (sweet spot)

All use thermo-only features (5D) since surface descriptors hurt.
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


def prepare_data(merged_csv, output_dir, gamma1_mask_frac=1.0, oversample_ratio=None):
    """Prepare balanced data.

    gamma1_mask_frac: fraction of ILThermo gamma1 to mask (1.0 = all masked, 0.0 = none)
    oversample_ratio: how many times to repeat original (None = auto-balance)
    """
    df = pd.read_csv(merged_csv)
    orig = df[df["source"] == "original"].copy()
    ilth = df[df["source"] != "original"].copy()

    # Partial gamma1 masking
    if gamma1_mask_frac > 0:
        g1_valid = ilth["gamma1"].notna()
        n_to_mask = int(g1_valid.sum() * gamma1_mask_frac)
        mask_idx = ilth[g1_valid].sample(n=n_to_mask, random_state=42).index
        ilth.loc[mask_idx, "gamma1"] = np.nan
        print(f"    Gamma1 mask: {n_to_mask}/{g1_valid.sum()} ILThermo values masked ({gamma1_mask_frac*100:.0f}%)")

    # Oversampling
    if oversample_ratio is None:
        repeat = max(1, round(len(ilth) / max(len(orig), 1)))
    else:
        repeat = oversample_ratio

    orig_rep = pd.concat([orig] * repeat, ignore_index=True)
    df_bal = pd.concat([orig_rep, ilth], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    orig_frac = len(orig_rep) / len(df_bal) * 100
    print(f"    Oversample: {len(orig)} × {repeat} = {len(orig_rep)} original "
          f"({orig_frac:.0f}%), {len(ilth)} ILThermo → {len(df_bal)} total")

    # Save
    out = pd.DataFrame()
    out["smiles"] = df_bal["smiles"]
    for t in TARGET_COLUMNS:
        out[t] = df_bal[t] if t in df_bal.columns else np.nan
    out.to_csv(output_dir / "train.csv", index=False)

    feat_df = df_bal[THERMO_FEATURES].fillna(0.0)
    feat_df.to_csv(output_dir / "train_features.csv", index=False)
    return len(df_bal)


def prepare_eval(csv_path, output_dir, prefix):
    df = pd.read_csv(csv_path)
    out = pd.DataFrame()
    out["smiles"] = df["smiles"]
    for t in TARGET_COLUMNS:
        out[t] = df[t] if t in df.columns else np.nan
    out.to_csv(output_dir / f"{prefix}.csv", index=False)
    df[THERMO_FEATURES].fillna(0.0).to_csv(output_dir / f"{prefix}_features.csv", index=False)


def train(data_dir, ckpt_dir, seed=42):
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
        "--epochs", "100",
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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-200:]}")
        return None
    scores_path = Path(ckpt_dir) / "fold_0" / "test_scores.json"
    if scores_path.exists():
        scores = json.load(open(scores_path))
        m = {}
        for i, p in enumerate(TARGET_COLUMNS):
            m[f"{p}_r2"] = scores["r2"][i]
        m["avg_r2"] = np.mean(scores["r2"])
        return m
    return None


def main():
    print("=== Chemprop Tuned: Partial Masking + Oversampling Ratios ===\n")

    merged_dir = Path("data/merged_v5")

    variants = [
        ("A: 50% gamma1 mask, 24x OS",   0.50, None),
        ("B: full mask, 10x OS",          1.00, 10),
        ("C: full mask, 48x OS",          1.00, 48),
        ("D: 50% mask, 10x OS",           0.50, 10),
        ("E: no mask, 10x OS",            0.00, 10),
    ]

    all_results = {}

    for name, mask_frac, os_ratio in variants:
        tag = name.split(":")[0].strip().lower()
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")

        data_dir = Path(f"data/chemprop_tuned/{tag}")
        data_dir.mkdir(parents=True, exist_ok=True)

        print("  Preparing data...")
        prepare_data(merged_dir / "splits/train.csv", data_dir, mask_frac, os_ratio)
        prepare_eval(merged_dir / "splits/val.csv", data_dir, "val")
        prepare_eval(merged_dir / "splits/test.csv", data_dir, "test")

        m = train(data_dir, f"checkpoints/chemprop_tuned/{tag}")
        if m:
            all_results[name] = m
            print(f"\n  Results:")
            for p in TARGET_COLUMNS:
                print(f"    {p:15s} R² = {m[f'{p}_r2']:.4f}")
            print(f"    {'AVERAGE':15s} R² = {m['avg_r2']:.4f}")

    # ══════════════════════════════════════════════════════════
    # COMPARISON
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")

    # Add baselines
    baselines = {}
    for bname, path, key in [
        ("Chemprop (base)", "results/chemprop_results.json", "test_metrics"),
        ("v2 (no mask, 24x)", "results/chemprop_unified_v2_results.json", "metrics"),
        ("v3b (full mask, 24x)", "results/chemprop_v3b_results.json", "metrics"),
    ]:
        try:
            data = json.load(open(path))
            baselines[bname] = data.get(key, {})
        except:
            pass

    all_models = {**baselines, **all_results}

    header = "  {:<12s}".format("Property")
    for name in all_models:
        header += " {:>16s}".format(name[:16])
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<12s}".format(p)
        for name, m in all_models.items():
            line += " {:16.4f}".format(m.get(key, float("nan")))
        print(line)

    line = "  {:<12s}".format("AVERAGE")
    for name, m in all_models.items():
        line += " {:16.4f}".format(m.get("avg_r2", float("nan")))
    print(line)

    # Find best
    best_name = max(all_results.keys(), key=lambda n: all_results[n]["avg_r2"])
    best_m = all_results[best_name]
    base_m = baselines.get("Chemprop (base)", {})

    if base_m:
        print(f"\n  Best variant: {best_name}")
        print(f"  vs Chemprop base:")
        wins = 0
        for p in TARGET_COLUMNS:
            key = f"{p}_r2"
            d = best_m[key] - base_m[key]
            s = "+" if d > 0 else ""
            w = "WIN" if d > 0 else "LOSE"
            if d > 0: wins += 1
            print(f"    {p:15s}: {best_m[key]:.4f} vs {base_m[key]:.4f} ({s}{d:.4f}) {w}")
        d = best_m["avg_r2"] - base_m["avg_r2"]
        s = "+" if d > 0 else ""
        print(f"    {'AVERAGE':15s}: {best_m['avg_r2']:.4f} vs {base_m['avg_r2']:.4f} ({s}{d:.4f}) — wins {wins}/7 properties")

    # Save
    with open("results/chemprop_tuned_results.json", "w") as f:
        json.dump({name: {"metrics": m} for name, m in all_results.items()}, f, indent=2)
    print(f"\nSaved: results/chemprop_tuned_results.json")


if __name__ == "__main__":
    main()
