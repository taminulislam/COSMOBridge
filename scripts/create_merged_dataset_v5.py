"""Create merged dataset v5 with aggressive gamma1 filtering.

Key fix: Filter ILThermo gamma1 to original's mean +/- 2*std [0.03, 1.52].
This aligns the gamma1 distributions so the unified scaler no longer
compresses/shifts original data. Both sources end up with nearly identical
gamma1 statistics (mean ~0.77, std ~0.37).

Also keeps unified scaler (which works correctly when distributions match).

Output: data/merged_v5/
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem

TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]
THERMO_FEATURES = ["temperature", "x1", "inv_temperature", "temp_squared", "temp_cubed"]
SURFACE_FEATURES = [
    "surface_area", "volume", "sphericity", "aspect_ratio",
    "curv_mean", "curv_std", "curv_skew",
    "gcurv_mean", "gcurv_std", "gcurv_skew",
    "esp_mean", "esp_std", "esp_min", "esp_max", "esp_skew", "esp_kurtosis",
    "esp_pos_frac", "esp_neg_frac", "esp_charge_segregation", "esp_range",
]


def main():
    output_dir = Path("data/merged_v5")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load original ──
    print("Loading original dataset...")
    orig = pd.read_csv("data/processed/il_data_raw.csv")
    orig["source"] = "original"
    g1_orig = orig["gamma1"].dropna()
    orig_mean = g1_orig.mean()
    orig_std = g1_orig.std()
    print(f"  Original: {len(orig)} rows, {orig['il_short_name'].nunique()} ILs")
    print(f"  gamma1: mean={orig_mean:.4f}, std={orig_std:.4f}, "
          f"range [{g1_orig.min():.4f}, {g1_orig.max():.4f}]")

    # ── Load ILThermo with aggressive gamma1 filter ──
    print("\nLoading ILThermo...")
    ilth = pd.read_csv("data/augmented/ilthermo_data.csv")
    ilth["source"] = "ilthermo"

    # FIX: Filter gamma1 to original's mean +/- 2*std
    lo = orig_mean - 2 * orig_std
    hi = orig_mean + 2 * orig_std
    print(f"  Gamma1 filter: mean +/- 2*std = [{lo:.4f}, {hi:.4f}]")

    before = len(ilth)
    g1_before = ilth["gamma1"].notna().sum()
    gamma1_mask = ilth["gamma1"].isna() | ((ilth["gamma1"] >= lo) & (ilth["gamma1"] <= hi))
    ilth = ilth[gamma1_mask].reset_index(drop=True)
    removed = before - len(ilth)
    g1_after = ilth["gamma1"].notna().sum()
    print(f"  ILThermo: {len(ilth)} rows kept ({removed} removed)")
    print(f"  gamma1: {g1_after} values (was {g1_before})")
    g1_ilth = ilth["gamma1"].dropna()
    if len(g1_ilth) > 0:
        print(f"  gamma1: mean={g1_ilth.mean():.4f}, std={g1_ilth.std():.4f}, "
              f"range [{g1_ilth.min():.4f}, {g1_ilth.max():.4f}]")

    print(f"\n  Distribution comparison after filter:")
    print(f"    Original:  mean={orig_mean:.4f}, std={orig_std:.4f}")
    print(f"    ILThermo:  mean={g1_ilth.mean():.4f}, std={g1_ilth.std():.4f}")
    print(f"    Diff:      mean_diff={abs(orig_mean - g1_ilth.mean()):.4f}, "
          f"std_ratio={g1_ilth.std()/orig_std:.3f}")

    # ── Standard prep ──
    for col in TARGET_COLUMNS:
        if col not in ilth.columns:
            ilth[col] = np.nan

    ilth["il_id"] = "ILThermo"
    if "cation_smiles" not in ilth.columns:
        ilth["cation_smiles"] = ilth["smiles"].apply(lambda s: s.split(".")[0] if "." in s else s)
    if "anion_smiles" not in ilth.columns:
        ilth["anion_smiles"] = ilth["smiles"].apply(
            lambda s: s.split(".")[1] if "." in str(s) and len(str(s).split(".")) > 1 else "")

    for df in [orig, ilth]:
        df["inv_temperature"] = 1.0 / df["temperature"]
        df["temp_squared"] = df["temperature"] ** 2
        df["temp_cubed"] = df["temperature"] ** 3

    # ── Merge surface descriptors ──
    print("\nMerging surface descriptors...")
    desc_orig = pd.read_csv("data/pipeline/surface_descriptors.csv")
    orig = orig.merge(desc_orig, on="il_short_name", how="left", suffixes=("", "_dup"))
    orig = orig[[c for c in orig.columns if not c.endswith("_dup")]]

    desc_ilth_path = Path("data/pipeline/surface_descriptors_ilthermo.csv")
    if desc_ilth_path.exists():
        desc_ilth = pd.read_csv(desc_ilth_path)
        if "smiles" in desc_ilth.columns:
            ilth = ilth.merge(desc_ilth.drop(columns=["il_short_name"], errors="ignore"),
                              on="smiles", how="left", suffixes=("", "_dup"))
            ilth = ilth[[c for c in ilth.columns if not c.endswith("_dup")]]

    for col in SURFACE_FEATURES:
        for df in [orig, ilth]:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)

    FEATURE_COLUMNS = THERMO_FEATURES + SURFACE_FEATURES

    # ── Merge ──
    common = ["smiles", "il_short_name", "temperature", "x1", "source",
              "cation_smiles", "anion_smiles"] + TARGET_COLUMNS + FEATURE_COLUMNS

    for col in common:
        if col not in orig.columns:
            orig[col] = np.nan if col in TARGET_COLUMNS else 0.0
        if col not in ilth.columns:
            ilth[col] = np.nan if col in TARGET_COLUMNS else 0.0

    merged = pd.concat([orig[common], ilth[common]], ignore_index=True)

    # Assign indices
    unique_smiles = sorted(merged["smiles"].unique())
    merged["il_idx"] = merged["smiles"].map({s: i for i, s in enumerate(unique_smiles)})
    unique_cations = sorted(merged["cation_smiles"].dropna().unique())
    merged["cation_idx"] = merged["cation_smiles"].map({s: i for i, s in enumerate(unique_cations)}).fillna(0).astype(int)
    unique_anions = sorted(merged["anion_smiles"].dropna().unique())
    merged["anion_idx"] = merged["anion_smiles"].map({s: i for i, s in enumerate(unique_anions)}).fillna(0).astype(int)

    print(f"\nMerged: {len(merged)} rows, {len(unique_smiles)} ILs")

    # ── Normalize (unified scaler — safe now that distributions match) ──
    print("Normalizing...")
    feature_scaler = StandardScaler()
    merged[FEATURE_COLUMNS] = feature_scaler.fit_transform(merged[FEATURE_COLUMNS])

    target_scalers = {}
    for col in TARGET_COLUMNS:
        valid = merged[col].notna()
        if valid.sum() > 1:
            scaler = StandardScaler()
            merged.loc[valid, col] = scaler.fit_transform(merged.loc[valid, [col]]).flatten()
            target_scalers[col] = scaler
            print(f"  {col}: n={valid.sum()}, raw_mean={scaler.mean_[0]:.4f}, raw_std={scaler.scale_[0]:.4f}")

    # Verify gamma1 per source after unified normalization
    print("\nPost-normalization gamma1 per source:")
    for src in ["original", "ilthermo"]:
        vals = merged.loc[(merged["source"] == src) & merged["gamma1"].notna(), "gamma1"]
        if len(vals) > 0:
            print(f"  {src}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                  f"range [{vals.min():.3f}, {vals.max():.3f}], n={len(vals)}")

    # ── Splits ──
    split_info = json.load(open("data/processed/splits/split_info.json"))
    test_ils = set(split_info["test_ils"])
    val_ils = set(split_info["val_ils"])

    test_mask = merged["il_short_name"].isin(test_ils) & (merged["source"] == "original")
    val_mask = merged["il_short_name"].isin(val_ils) & (merged["source"] == "original")
    train_mask = ~test_mask & ~val_mask

    test_df = merged[test_mask].reset_index(drop=True)
    val_df = merged[val_mask].reset_index(drop=True)
    train_df = merged[train_mask].reset_index(drop=True)

    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    print(f"\n  Train: {len(train_df)} ({train_df['smiles'].nunique()} ILs, "
          f"{(train_df['source']=='original').sum()} original, "
          f"{(train_df['source']=='ilthermo').sum()} ILThermo)")
    print(f"  Val: {len(val_df)}, Test: {len(test_df)}")

    # ── Save ──
    meta = {
        "total_rows": len(merged),
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "gamma1_filter": f"mean+/-2std [{lo:.4f}, {hi:.4f}]",
        "log_transform": "none",
        "unique_smiles": len(unique_smiles),
        "normalization": "unified_scaler",
        "description": "Aggressive gamma1 filter aligns ILThermo distribution to original "
                       "before unified normalization. Both sources have similar gamma1 stats.",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(output_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump(feature_scaler, f)

    with open(output_dir / "target_scalers.pkl", "wb") as f:
        pickle.dump(target_scalers, f)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
