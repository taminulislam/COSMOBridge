"""Create merged dataset for Strategies A-E.

Merges original (223 samples, 28 ILs) + ILThermo (5622 samples, 115 ILs)
into data/merged/ with:
  - Morgan fingerprints (Strategy C)
  - Log-transformed heavy-tailed targets (Strategy D)
  - Surface descriptors from Phase 1
  - GroupKFold splits for CV ensemble (Strategy B)
  - Proper normalization

The original test set ILs (5 ILs, 39 samples) are HELD OUT from all
training — they never appear in merged training data.

Output:
  data/merged/merged_full.csv          — all data before splits
  data/merged/splits/train.csv         — training (original train + all ILThermo)
  data/merged/splits/val.csv           — validation (original val ILs only)
  data/merged/splits/test.csv          — test (original test ILs only, untouched)
  data/merged/cv_folds/fold_{i}/       — 5-fold CV splits
  data/merged/metadata.json            — dataset info
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem


# ── Configuration ────────────────────────────────────────────────────────────

TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]
THERMO_FEATURES = ["temperature", "x1", "inv_temperature", "temp_squared", "temp_cubed"]

SURFACE_FEATURES = [
    "surface_area", "volume", "sphericity", "aspect_ratio",
    "curv_mean", "curv_std", "curv_skew",
    "gcurv_mean", "gcurv_std", "gcurv_skew",
    "esp_mean", "esp_std", "esp_min", "esp_max", "esp_skew", "esp_kurtosis",
    "esp_pos_frac", "esp_neg_frac", "esp_charge_segregation", "esp_range",
]

MORGAN_BITS = 256  # Compact Morgan fingerprint
MORGAN_RADIUS = 2

# Heavy-tailed targets to log-transform
LOG_TARGETS = ["gamma1", "P"]


# ── Morgan Fingerprints ──────────────────────────────────────────────────────

def compute_morgan_fp(smiles, n_bits=MORGAN_BITS, radius=MORGAN_RADIUS):
    """Compute Morgan fingerprint as a numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def add_morgan_fingerprints(df, n_bits=MORGAN_BITS):
    """Add Morgan fingerprint columns to dataframe."""
    fp_cols = [f"morgan_{i}" for i in range(n_bits)]
    fps = []
    for smi in df["smiles"]:
        fps.append(compute_morgan_fp(smi, n_bits))
    fp_df = pd.DataFrame(fps, columns=fp_cols, index=df.index)
    return pd.concat([df, fp_df], axis=1), fp_cols


# ── Target Transformation ────────────────────────────────────────────────────

def log_transform_targets(df, columns=LOG_TARGETS):
    """Apply log(1+|x|)*sign(x) transform to heavy-tailed targets."""
    for col in columns:
        if col in df.columns:
            valid = df[col].notna()
            df.loc[valid, col] = np.sign(df.loc[valid, col]) * np.log1p(np.abs(df.loc[valid, col]))
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    base_dir = Path(".")
    output_dir = base_dir / "data" / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load original dataset ──
    print("Loading original dataset...")
    orig = pd.read_csv("data/processed/il_data_raw.csv")
    orig["source"] = "original"
    print(f"  Original: {len(orig)} rows, {orig['il_short_name'].nunique()} ILs")

    # ── Load original split info to preserve test set ──
    split_info_path = Path("data/processed/splits/split_info.json")
    with open(split_info_path) as f:
        split_info = json.load(f)
    test_ils = set(split_info["test_ils"])
    val_ils = set(split_info["val_ils"])
    train_ils = set(split_info["train_ils"])
    print(f"  Original splits: train={len(train_ils)}, val={len(val_ils)}, test={len(test_ils)}")

    # ── Load ILThermo dataset ──
    print("\nLoading ILThermo dataset...")
    ilth = pd.read_csv("data/augmented/ilthermo_data.csv")
    ilth["source"] = "ilthermo"

    # Filter ILThermo outliers (Strategy D)
    before = len(ilth)
    ilth = ilth[~((ilth["gamma1"].notna()) & (ilth["gamma1"].abs() > 100))]
    print(f"  ILThermo: {len(ilth)} rows after outlier filter ({before - len(ilth)} removed)")

    # ── Ensure consistent columns ──
    for col in TARGET_COLUMNS:
        if col not in ilth.columns:
            ilth[col] = np.nan

    # Standardize ILThermo columns to match original
    ilth["il_id"] = "ILThermo"
    if "cation_smiles" not in ilth.columns:
        ilth["cation_smiles"] = ilth["smiles"].apply(lambda s: s.split(".")[0] if "." in s else s)
    if "anion_smiles" not in ilth.columns:
        ilth["anion_smiles"] = ilth["smiles"].apply(lambda s: s.split(".")[1] if "." in str(s) and len(str(s).split(".")) > 1 else "")

    # ── Engineer thermo features ──
    for df in [orig, ilth]:
        df["inv_temperature"] = 1.0 / df["temperature"]
        df["temp_squared"] = df["temperature"] ** 2
        df["temp_cubed"] = df["temperature"] ** 3

    # ── Log-transform heavy-tailed targets (Strategy D) ──
    print("\nApplying log transform to heavy-tailed targets...")
    orig = log_transform_targets(orig.copy())
    ilth = log_transform_targets(ilth.copy())
    for col in LOG_TARGETS:
        v = orig[col].dropna()
        print(f"  {col} (original): range [{v.min():.3f}, {v.max():.3f}]")
        v2 = ilth[col].dropna()
        if len(v2) > 0:
            print(f"  {col} (ilthermo): range [{v2.min():.3f}, {v2.max():.3f}]")

    # ── Merge surface descriptors ──
    print("\nMerging surface descriptors...")
    # Original descriptors (keyed by il_short_name)
    desc_orig = pd.read_csv("data/pipeline/surface_descriptors.csv")
    orig = orig.merge(desc_orig, on="il_short_name", how="left", suffixes=("", "_dup"))
    orig = orig[[c for c in orig.columns if not c.endswith("_dup")]]

    # ILThermo descriptors (keyed by smiles)
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

    n_orig_desc = (orig["surface_area"] != 0).sum()
    n_ilth_desc = (ilth["surface_area"] != 0).sum()
    print(f"  Original with descriptors: {n_orig_desc}/{len(orig)}")
    print(f"  ILThermo with descriptors: {n_ilth_desc}/{len(ilth)}")

    # ── Add Morgan fingerprints (Strategy C) ──
    print("\nComputing Morgan fingerprints...")
    orig, morgan_cols = add_morgan_fingerprints(orig)
    ilth, _ = add_morgan_fingerprints(ilth)
    print(f"  Morgan FP: {len(morgan_cols)} bits (radius={MORGAN_RADIUS})")

    # ── Define full feature list ──
    FEATURE_COLUMNS = THERMO_FEATURES + SURFACE_FEATURES + morgan_cols

    # ── Merge datasets ──
    print("\nMerging datasets...")
    common_cols = ["smiles", "il_short_name", "temperature", "x1", "source",
                   "cation_smiles", "anion_smiles"] + TARGET_COLUMNS + FEATURE_COLUMNS

    # Ensure all columns exist
    for col in common_cols:
        if col not in orig.columns:
            orig[col] = np.nan if col in TARGET_COLUMNS else 0.0
        if col not in ilth.columns:
            ilth[col] = np.nan if col in TARGET_COLUMNS else 0.0

    merged = pd.concat([orig[common_cols], ilth[common_cols]], ignore_index=True)
    print(f"  Merged: {len(merged)} rows, {merged['smiles'].nunique()} unique SMILES")

    # ── Assign IL indices for all unique SMILES ──
    unique_smiles = sorted(merged["smiles"].unique())
    smi_to_idx = {s: i for i, s in enumerate(unique_smiles)}
    merged["il_idx"] = merged["smiles"].map(smi_to_idx)

    unique_cations = sorted(merged["cation_smiles"].dropna().unique())
    cat_to_idx = {s: i for i, s in enumerate(unique_cations)}
    merged["cation_idx"] = merged["cation_smiles"].map(cat_to_idx).fillna(0).astype(int)

    unique_anions = sorted(merged["anion_smiles"].dropna().unique())
    an_to_idx = {s: i for i, s in enumerate(unique_anions)}
    merged["anion_idx"] = merged["anion_smiles"].map(an_to_idx).fillna(0).astype(int)

    print(f"  Unique ILs: {len(unique_smiles)}")
    print(f"  Unique cations: {len(unique_cations)}")
    print(f"  Unique anions: {len(unique_anions)}")

    # ── Normalize features ──
    print("\nNormalizing features...")
    feature_scaler = StandardScaler()
    merged[FEATURE_COLUMNS] = feature_scaler.fit_transform(merged[FEATURE_COLUMNS])

    # Normalize targets per-column (only on non-NaN)
    target_scalers = {}
    for col in TARGET_COLUMNS:
        valid_mask = merged[col].notna()
        if valid_mask.sum() > 1:
            scaler = StandardScaler()
            merged.loc[valid_mask, col] = scaler.fit_transform(
                merged.loc[valid_mask, [col]]
            ).flatten()
            target_scalers[col] = scaler
            print(f"  {col}: normalized {valid_mask.sum()} values")

    # ── Save full merged dataset ──
    merged.to_csv(output_dir / "merged_full.csv", index=False)
    print(f"\nSaved merged_full.csv: {len(merged)} rows")

    # ── Create splits (preserve original test ILs) ──
    print("\nCreating splits...")
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Test: only original test ILs
    test_mask = merged["il_short_name"].isin(test_ils) & (merged["source"] == "original")
    test_df = merged[test_mask].reset_index(drop=True)

    # Val: only original val ILs
    val_mask = merged["il_short_name"].isin(val_ils) & (merged["source"] == "original")
    val_df = merged[val_mask].reset_index(drop=True)

    # Train: everything else (original train ILs + ALL ILThermo data)
    train_mask = ~test_mask & ~val_mask
    train_df = merged[train_mask].reset_index(drop=True)

    test_df.to_csv(splits_dir / "test.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    train_df.to_csv(splits_dir / "train.csv", index=False)

    print(f"  Train: {len(train_df)} rows ({train_df['smiles'].nunique()} ILs)")
    print(f"  Val: {len(val_df)} rows ({val_df['smiles'].nunique()} ILs)")
    print(f"  Test: {len(test_df)} rows ({test_df['smiles'].nunique()} ILs)")
    print(f"  ILThermo in train: {(train_df['source'] == 'ilthermo').sum()}")

    # ── Create 5-fold CV splits (Strategy B) ──
    print("\nCreating 5-fold CV splits...")
    cv_dir = output_dir / "cv_folds"
    cv_dir.mkdir(parents=True, exist_ok=True)

    # For CV, use only original data (to evaluate on original ILs)
    orig_merged = merged[merged["source"] == "original"].reset_index(drop=True)
    ilthermo_merged = merged[merged["source"] == "ilthermo"].reset_index(drop=True)

    gkf = GroupKFold(n_splits=5)
    groups = orig_merged["il_short_name"].values

    for fold, (train_idx, val_idx) in enumerate(gkf.split(orig_merged, groups=groups)):
        fold_dir = cv_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_val = orig_merged.iloc[val_idx].reset_index(drop=True)
        fold_train_orig = orig_merged.iloc[train_idx].reset_index(drop=True)

        # Add ILThermo to training
        fold_train = pd.concat([fold_train_orig, ilthermo_merged], ignore_index=True)

        fold_train.to_csv(fold_dir / "train.csv", index=False)
        fold_val.to_csv(fold_dir / "val.csv", index=False)

        val_ils_fold = sorted(fold_val["il_short_name"].unique())
        print(f"  Fold {fold}: train={len(fold_train)} ({fold_train['smiles'].nunique()} ILs), "
              f"val={len(fold_val)} ({len(val_ils_fold)} ILs: {val_ils_fold[:3]}...)")

    # ── Save scalers and metadata ──
    with open(output_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump(feature_scaler, f)
    with open(output_dir / "target_scalers.pkl", "wb") as f:
        pickle.dump(target_scalers, f)

    metadata = {
        "total_rows": len(merged),
        "original_rows": int((merged["source"] == "original").sum()),
        "ilthermo_rows": int((merged["source"] == "ilthermo").sum()),
        "unique_smiles": len(unique_smiles),
        "unique_cations": len(unique_cations),
        "unique_anions": len(unique_anions),
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "morgan_bits": MORGAN_BITS,
        "morgan_radius": MORGAN_RADIUS,
        "log_transformed_targets": LOG_TARGETS,
        "test_ils": sorted(test_ils),
        "val_ils": sorted(val_ils),
        "train_split": len(train_df),
        "val_split": len(val_df),
        "test_split": len(test_df),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"MERGED DATASET COMPLETE")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}")
    print(f"  Total features: {len(FEATURE_COLUMNS)} "
          f"(5 thermo + 20 surface + {len(morgan_cols)} Morgan FP)")
    print(f"  Targets: {TARGET_COLUMNS} (log-transformed: {LOG_TARGETS})")
    print(f"  Total rows: {len(merged)} ({len(unique_smiles)} unique ILs)")


if __name__ == "__main__":
    main()
