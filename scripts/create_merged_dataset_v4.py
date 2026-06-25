"""Create merged dataset v4 with source-specific target scalers.

Key fix: Instead of fitting ONE StandardScaler on all data combined,
fit SEPARATE scalers per source (original vs ILThermo) for each target.
This preserves each distribution's natural statistics and prevents
the original data from being compressed/shifted by ILThermo's range.

Each sample is normalized using its own source's scaler.
At test time, predictions are inverse-transformed using the original scaler.

Output: data/merged_v4/
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
    output_dir = Path("data/merged_v4")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load original ──
    print("Loading original dataset...")
    orig = pd.read_csv("data/processed/il_data_raw.csv")
    orig["source"] = "original"
    print(f"  Original: {len(orig)} rows, {orig['il_short_name'].nunique()} ILs")
    print(f"  gamma1 range: [{orig['gamma1'].min():.3f}, {orig['gamma1'].max():.3f}]")
    print(f"  gamma1 mean={orig['gamma1'].mean():.3f}, std={orig['gamma1'].std():.3f}")

    # ── Load ILThermo with gamma1 filter ──
    print("\nLoading ILThermo with gamma1 filter...")
    ilth = pd.read_csv("data/augmented/ilthermo_data.csv")
    ilth["source"] = "ilthermo"

    # Filter gamma1 to match original range [0.05, 5.0]
    before = len(ilth)
    gamma1_mask = ilth["gamma1"].isna() | ((ilth["gamma1"] >= 0.05) & (ilth["gamma1"] <= 5.0))
    ilth = ilth[gamma1_mask].reset_index(drop=True)
    removed = before - len(ilth)
    print(f"  ILThermo after gamma1 filter [0.05, 5.0]: {len(ilth)} ({removed} removed)")
    if ilth["gamma1"].dropna().shape[0] > 0:
        print(f"  gamma1 range: [{ilth['gamma1'].dropna().min():.3f}, {ilth['gamma1'].dropna().max():.3f}]")
        print(f"  gamma1 mean={ilth['gamma1'].dropna().mean():.3f}, std={ilth['gamma1'].dropna().std():.3f}")

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

    # ── Normalize features (unified scaler is fine for features) ──
    print("Normalizing features (unified scaler)...")
    feature_scaler = StandardScaler()
    merged[FEATURE_COLUMNS] = feature_scaler.fit_transform(merged[FEATURE_COLUMNS])

    # ── KEY FIX: Source-specific target scalers ──
    print("Normalizing targets with SOURCE-SPECIFIC scalers...")
    orig_mask = merged["source"] == "original"
    ilth_mask = merged["source"] == "ilthermo"

    target_scalers_original = {}
    target_scalers_ilthermo = {}

    for col in TARGET_COLUMNS:
        # Fit scaler on original data only
        valid_orig = orig_mask & merged[col].notna()
        if valid_orig.sum() > 1:
            scaler_orig = StandardScaler()
            scaler_orig.fit(merged.loc[valid_orig, [col]])
            target_scalers_original[col] = scaler_orig
            # Normalize original samples with original scaler
            merged.loc[valid_orig, col] = scaler_orig.transform(merged.loc[valid_orig, [col]]).flatten()
            print(f"  {col} (original): n={valid_orig.sum()}, "
                  f"mean={scaler_orig.mean_[0]:.4f}, std={scaler_orig.scale_[0]:.4f}")

        # Fit scaler on ILThermo data only
        valid_ilth = ilth_mask & merged[col].notna()
        if valid_ilth.sum() > 1:
            scaler_ilth = StandardScaler()
            scaler_ilth.fit(merged.loc[valid_ilth, [col]])
            target_scalers_ilthermo[col] = scaler_ilth
            # Normalize ILThermo samples with ILThermo scaler
            merged.loc[valid_ilth, col] = scaler_ilth.transform(merged.loc[valid_ilth, [col]]).flatten()
            print(f"  {col} (ilthermo): n={valid_ilth.sum()}, "
                  f"mean={scaler_ilth.mean_[0]:.4f}, std={scaler_ilth.scale_[0]:.4f}")

    # Verify: both sources should now have mean~0, std~1 for each target
    print("\nPost-normalization verification:")
    for col in TARGET_COLUMNS:
        for src in ["original", "ilthermo"]:
            vals = merged.loc[(merged["source"] == src) & merged[col].notna(), col]
            if len(vals) > 0:
                print(f"  {col} ({src}): mean={vals.mean():.4f}, std={vals.std():.4f}, n={len(vals)}")

    # ── Splits (preserve original test ILs) ──
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

    # Check gamma1 distribution in train
    for src in ["original", "ilthermo"]:
        g1 = train_df.loc[(train_df["source"] == src) & train_df["gamma1"].notna(), "gamma1"]
        if len(g1) > 0:
            print(f"  gamma1 in train ({src}): n={len(g1)}, "
                  f"mean={g1.mean():.4f}, std={g1.std():.4f}, "
                  f"range [{g1.min():.3f}, {g1.max():.3f}]")

    # ── Save metadata and scalers ──
    meta = {
        "total_rows": len(merged),
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "gamma1_filter": "[0.05, 5.0]",
        "log_transform": "none",
        "unique_smiles": len(unique_smiles),
        "normalization": "source_specific_target_scalers",
        "description": "Each source (original/ilthermo) has its own target scaler. "
                       "Both sources are normalized to mean=0, std=1 independently.",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(output_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump(feature_scaler, f)

    with open(output_dir / "target_scalers_original.pkl", "wb") as f:
        pickle.dump(target_scalers_original, f)

    with open(output_dir / "target_scalers_ilthermo.pkl", "wb") as f:
        pickle.dump(target_scalers_ilthermo, f)

    print(f"\nSaved to {output_dir}")
    print("  - feature_scaler.pkl (unified)")
    print("  - target_scalers_original.pkl (per-target, original source)")
    print("  - target_scalers_ilthermo.pkl (per-target, ILThermo source)")


if __name__ == "__main__":
    main()
