"""Data preprocessing: Excel -> CSV, IL-to-image mapping, normalization, splits."""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import json


# ── Column name mapping (Excel -> clean Python names) ──────────────────────
COLUMN_MAP = {
    "IL ID": "il_id",
    "SMILES": "smiles",
    "CATION": "cation_smiles",
    "ANION": "anion_smiles",
    "T (K)": "temperature",
    "x1": "x1",
    "γ1": "gamma1",
    "γ2": "gamma2",
    "G^E (kcal/mol)": "G_E",
    "H^E (kcal/mol)": "H_E",
    "G^mix (kcal/mol)": "G_mix",
    "H_vap (kcal/mol)": "H_vap",
    "P (bar)": "P",
}

TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]

# Thermodynamic features
THERMO_FEATURES = ["temperature", "x1", "inv_temperature", "temp_squared", "temp_cubed"]

# Surface descriptor features (from step4b_surface_descriptors.py)
SURFACE_FEATURES = [
    "surface_area", "volume", "sphericity", "aspect_ratio",
    "curv_mean", "curv_std", "curv_skew",
    "gcurv_mean", "gcurv_std", "gcurv_skew",
    "esp_mean", "esp_std", "esp_min", "esp_max", "esp_skew", "esp_kurtosis",
    "esp_pos_frac", "esp_neg_frac", "esp_charge_segregation", "esp_range",
]

FEATURE_COLUMNS = THERMO_FEATURES + SURFACE_FEATURES


# ── IL-to-Image mapping ───────────────────────────────────────────────────
# Maps IL short name (from IL ID) to image filenames.
# COSMO: sigma surface of the ion pair
# EP: electrostatic potential surface (files ending in _EP)
# Structure: ball-and-stick 3D structure (in EP folder, without _EP suffix)
# Some ILs lack COSMO ion pair images; we use cation-only COSMO where available.

IL_IMAGE_MAP = {
    "AMIMCl": {
        "cosmo": "COSMO files/AMIM Cl.png",
        "ep": "Electrostatic Potential/AMIMCL_EP.png",
        "structure": "Electrostatic Potential/AMIMCl.png",
    },
    "BMIMBr": {
        "cosmo": "COSMO files/BMIM Br.png",
        "ep": "Electrostatic Potential/BMIMBr.png",
        "structure": "Electrostatic Potential/BMIMBr .png",
    },
    "BMIMCl": {
        "cosmo": "COSMO files/BMIM Cl png.png",
        "ep": "Electrostatic Potential/BMIMCl.png",
        "structure": "Electrostatic Potential/BMIMCl .png",
    },
    "BMIMHSO4": {
        "cosmo": "COSMO files/BMIM HSO4.png",
        "ep": "Electrostatic Potential/BMIMHSO4 .png",  # no _EP variant; using structure view
        "structure": "Electrostatic Potential/BMIMHSO4.png",
    },
    "BMIMLAC": {
        "cosmo": "COSMO files/BMIM LAC.png",
        "ep": "Electrostatic Potential/BMIMLAC.png",
        "structure": "Electrostatic Potential/BMIMLAC .png",
    },
    "BMIMMESO4": {
        "cosmo": "COSMO files/BMIM MSO4.png",
        "ep": "Electrostatic Potential/BMIMMSO4 .png",  # no _EP variant; using structure view
        "structure": "Electrostatic Potential/BMIMMSO4.png",
    },
    "BMIMOAc": {
        "cosmo": "COSMO files/BMIM OAc.png",
        "ep": "Electrostatic Potential/BMIMOAc_EP.png",
        "structure": "Electrostatic Potential/BMIMOAc .png",
    },
    "ChCl": {
        "cosmo": "COSMO files/Cholinium.png",  # cation-only COSMO
        "ep": "Electrostatic Potential/CHCL_EP.png",
        "structure": "Electrostatic Potential/CHCL.png",
    },
    "ChHSO4": {
        "cosmo": "COSMO files/Cholinium.png",  # cation-only COSMO
        "ep": "Electrostatic Potential/CHHSO4_EP.png",
        "structure": "Electrostatic Potential/CHHSO4.png",
    },
    "ChLAC": {
        "cosmo": "COSMO files/Cholinium.png",  # cation-only COSMO
        "ep": "Electrostatic Potential/CHLAC_EP.png",
        "structure": "Electrostatic Potential/CHLAC.png",
    },
    "ChLys": {
        "cosmo": "COSMO files/Cholinium.png",  # cation-only COSMO
        "ep": "Electrostatic Potential/CHLYS_EP.png",
        "structure": "Electrostatic Potential/CHLYS.png",
    },
    "ChOAc": {
        "cosmo": "COSMO files/Cholinium.png",  # cation-only COSMO
        "ep": "Electrostatic Potential/CHOAC_EP.png",
        "structure": "Electrostatic Potential/CHOAc.png",
    },
    "DMBACl": {
        "cosmo": "COSMO files/DMBA.png",  # cation-only COSMO
        "ep": "Electrostatic Potential/DMBACL_EP.png",
        "structure": "Electrostatic Potential/DMBACl.png",
    },
    "DMBAHSO4": {
        "cosmo": "COSMO files/DMBA.png",  # cation-only COSMO
        "ep": "Electrostatic Potential/DMBAHSO4_EP.png",
        "structure": "Electrostatic Potential/DMBAHSO4.png",
    },
    "EMIMBr": {
        "cosmo": "COSMO files/EMIM BR.png",
        "ep": "Electrostatic Potential/EMIMBr_EP.png",
        "structure": "Electrostatic Potential/EMIMBr.png",
    },
    "EMIMCl": {
        "cosmo": "COSMO files/EMIM CL.png",
        "ep": "Electrostatic Potential/EMIMCl_EP.png",
        "structure": "Electrostatic Potential/EMIMCl .png",
    },
    "EMIMHSO4": {
        "cosmo": "COSMO files/EMIM HSO4 PNG.png",
        "ep": "Electrostatic Potential/EMIMHSO4_EP.png",
        "structure": "Electrostatic Potential/EMIMHSO4 .png",
    },
    "EMIMLAC": {
        "cosmo": "COSMO files/EMIM LAC PNG.png",
        "ep": "Electrostatic Potential/EMIMLAC_EP.png",
        "structure": "Electrostatic Potential/EMIMLAC .png",
    },
    "EMIMMeSO4": {
        "cosmo": "COSMO files/EMIM MSO4  PNG.png",
        "ep": "Electrostatic Potential/EMIMMSO4_EP.png",
        "structure": "Electrostatic Potential/EMIMMSO4.png",
    },
    "EMIMOAc": {
        "cosmo": "COSMO files/EMIM OAC.png",
        "ep": "Electrostatic Potential/EMIMOAc_EP.png",
        "structure": "Electrostatic Potential/EMIMOAc .png",
    },
    "EOAOAc": {
        "cosmo": "COSMO files/EOA.png",  # cation-only COSMO
        "ep": "Electrostatic Potential/EOAOAC_EP.png",
        "structure": "Electrostatic Potential/EOAOAC.png",
    },
    "MMIMLAC": {
        "cosmo": "COSMO files/MIM LAC PNG.png",
        "ep": "Electrostatic Potential/MIMLAC_EP.png",
        "structure": "Electrostatic Potential/MIMLAC.png",
    },
    "N11H2OHLAC": {
        "cosmo": None,  # no COSMO image available
        "ep": "Electrostatic Potential/N112OH(OH)LAC.png",
        "structure": "Electrostatic Potential/N112OH(OH)LAC .png",
    },
    "N11H2OHOAc": {
        "cosmo": None,  # no COSMO image available
        "ep": "Electrostatic Potential/N112OH(OH)OAc_EP.png",
        "structure": "Electrostatic Potential/N112OH(OH)OAc.png",
    },
    "TEACl": {
        "cosmo": None,  # no COSMO image available
        "ep": "Electrostatic Potential/TEACL_EP.png",
        "structure": "Electrostatic Potential/TEACl.png",
    },
    "TEAHSO4": {
        "cosmo": None,  # no COSMO image available
        "ep": "Electrostatic Potential/TEAHSO4_EP.png",
        "structure": "Electrostatic Potential/TEAHSO4.png",
    },
    "TEALAC": {
        "cosmo": None,  # no COSMO image available
        "ep": "Electrostatic Potential/TEALAC_EP.png",
        "structure": "Electrostatic Potential/TEALAC.png",
    },
    "TEALOAc": {
        "cosmo": None,  # no COSMO image available
        "ep": "Electrostatic Potential/TEAOAC_EP.png",
        "structure": "Electrostatic Potential/TEAOAC.png",
    },
}


def extract_il_short_name(il_id: str) -> str:
    """Extract short name from IL ID string, e.g. 'IL-1 (AMIMCl)' -> 'AMIMCl'.

    Handles unicode en-dash (U+2011) and regular hyphen in IL IDs.
    Also normalizes subscript characters.
    """
    # Extract text within parentheses
    start = il_id.index("(") + 1
    end = il_id.index(")")
    name = il_id[start:end]
    # Normalize subscript characters
    name = name.replace("\u2084", "4").replace("\u2083", "3")
    return name


def load_excel_data(excel_path: str, sheet_name: str = "Datasheet") -> pd.DataFrame:
    """Load and clean Excel data into a pandas DataFrame."""
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Keep only the named columns (drop unnamed/empty columns)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Rename columns
    rename_map = {}
    for col in df.columns:
        if col in COLUMN_MAP:
            rename_map[col] = COLUMN_MAP[col]
    df = df.rename(columns=rename_map)

    # Drop rows with missing IL IDs or missing target values
    df = df.dropna(subset=["il_id"])
    df = df.dropna(subset=TARGET_COLUMNS, how="any")

    # Extract short name for image mapping
    df["il_short_name"] = df["il_id"].apply(extract_il_short_name)

    # Assign integer IL index (for embeddings)
    unique_ils = sorted(df["il_short_name"].unique())
    il_to_idx = {name: idx for idx, name in enumerate(unique_ils)}
    df["il_idx"] = df["il_short_name"].map(il_to_idx)

    # Engineered thermodynamic features (computed from raw temperature before normalization)
    df["inv_temperature"] = 1.0 / df["temperature"]  # Arrhenius-type 1/T
    df["temp_squared"] = df["temperature"] ** 2       # Polynomial T²
    df["temp_cubed"] = df["temperature"] ** 3          # Polynomial T³

    # Assign cation and anion indices
    unique_cations = sorted(df["cation_smiles"].unique())
    unique_anions = sorted(df["anion_smiles"].unique())
    cat_to_idx = {s: i for i, s in enumerate(unique_cations)}
    an_to_idx = {s: i for i, s in enumerate(unique_anions)}
    df["cation_idx"] = df["cation_smiles"].map(cat_to_idx)
    df["anion_idx"] = df["anion_smiles"].map(an_to_idx)

    return df


def add_image_paths(df: pd.DataFrame, image_dir: str) -> pd.DataFrame:
    """Add resolved image file paths to DataFrame."""
    image_dir = Path(image_dir)

    cosmo_paths = []
    ep_paths = []
    structure_paths = []

    for _, row in df.iterrows():
        short_name = row["il_short_name"]
        mapping = IL_IMAGE_MAP.get(short_name, {})

        # COSMO image
        cosmo_rel = mapping.get("cosmo")
        if cosmo_rel:
            cosmo_full = image_dir / cosmo_rel
            cosmo_paths.append(str(cosmo_full) if cosmo_full.exists() else None)
        else:
            cosmo_paths.append(None)

        # EP image
        ep_rel = mapping.get("ep")
        if ep_rel:
            ep_full = image_dir / ep_rel
            ep_paths.append(str(ep_full) if ep_full.exists() else None)
        else:
            ep_paths.append(None)

        # Structure image
        struct_rel = mapping.get("structure")
        if struct_rel:
            struct_full = image_dir / struct_rel
            structure_paths.append(str(struct_full) if struct_full.exists() else None)
        else:
            structure_paths.append(None)

    df["cosmo_image_path"] = cosmo_paths
    df["ep_image_path"] = ep_paths
    df["structure_image_path"] = structure_paths

    return df


def merge_surface_descriptors(df: pd.DataFrame, descriptors_csv: str = None) -> pd.DataFrame:
    """Merge surface descriptors into the dataframe by il_short_name.

    If descriptors_csv is not provided, searches default location.
    Missing descriptors are filled with 0.0.
    """
    if descriptors_csv is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        descriptors_csv = base_dir / "data" / "pipeline" / "surface_descriptors.csv"

    descriptors_csv = Path(descriptors_csv)
    if not descriptors_csv.exists():
        print(f"  WARNING: Surface descriptors not found at {descriptors_csv}")
        print(f"  Setting surface features to 0. Run step4b_surface_descriptors.py first.")
        for col in SURFACE_FEATURES:
            df[col] = 0.0
        return df

    desc_df = pd.read_csv(descriptors_csv)
    # Merge on il_short_name (many rows per IL share the same surface descriptors)
    df = df.merge(desc_df, on="il_short_name", how="left", suffixes=("", "_dup"))
    # Drop any duplicate columns from merge
    df = df[[c for c in df.columns if not c.endswith("_dup")]]
    # Fill NaN for any ILs missing descriptors
    for col in SURFACE_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = 0.0

    n_with = df[SURFACE_FEATURES[0]].ne(0).sum()
    print(f"  Surface descriptors merged: {n_with}/{len(df)} rows have descriptors")
    return df


def normalize_features(df: pd.DataFrame, fit: bool = True, scaler: StandardScaler = None):
    """Normalize numerical features and targets.

    Returns: (df_normalized, feature_scaler, target_scaler)
    """
    feature_scaler = scaler if scaler else StandardScaler()
    target_scaler = StandardScaler()

    # Ensure all feature columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    if fit:
        df[FEATURE_COLUMNS] = feature_scaler.fit_transform(df[FEATURE_COLUMNS])
        df[TARGET_COLUMNS] = target_scaler.fit_transform(df[TARGET_COLUMNS])
    else:
        df[FEATURE_COLUMNS] = feature_scaler.transform(df[FEATURE_COLUMNS])
        df[TARGET_COLUMNS] = target_scaler.transform(df[TARGET_COLUMNS])

    return df, feature_scaler, target_scaler


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Create train/val/test splits stratified by IL ID.

    Uses leave-N-ILs-out strategy so the model is evaluated on unseen ionic liquids.
    """
    rng = np.random.RandomState(seed)
    unique_ils = sorted(df["il_short_name"].unique())
    n_ils = len(unique_ils)

    n_train = int(n_ils * train_ratio)
    n_val = int(n_ils * val_ratio)

    perm = rng.permutation(unique_ils)
    train_ils = set(perm[:n_train])
    val_ils = set(perm[n_train : n_train + n_val])
    test_ils = set(perm[n_train + n_val :])

    splits = {
        "train": df[df["il_short_name"].isin(train_ils)].reset_index(drop=True),
        "val": df[df["il_short_name"].isin(val_ils)].reset_index(drop=True),
        "test": df[df["il_short_name"].isin(test_ils)].reset_index(drop=True),
        "train_ils": sorted(train_ils),
        "val_ils": sorted(val_ils),
        "test_ils": sorted(test_ils),
    }

    print(f"Split: {len(train_ils)} train ILs ({len(splits['train'])} rows), "
          f"{len(val_ils)} val ILs ({len(splits['val'])} rows), "
          f"{len(test_ils)} test ILs ({len(splits['test'])} rows)")

    return splits


def create_kfold_splits(df: pd.DataFrame, n_folds: int = 5) -> list:
    """Create k-fold cross-validation splits grouped by IL ID."""
    gkf = GroupKFold(n_splits=n_folds)
    groups = df["il_short_name"].values

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        folds.append({
            "fold": fold_idx,
            "train": df.iloc[train_idx].reset_index(drop=True),
            "val": df.iloc[val_idx].reset_index(drop=True),
            "train_ils": sorted(df.iloc[train_idx]["il_short_name"].unique()),
            "val_ils": sorted(df.iloc[val_idx]["il_short_name"].unique()),
        })
    return folds


def preprocess_and_save(
    excel_path: str,
    image_dir: str,
    output_dir: str,
    config: dict = None,
):
    """Full preprocessing pipeline: load, clean, map images, normalize, split, save."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading Excel data...")
    df = load_excel_data(excel_path)
    print(f"  Loaded {len(df)} rows, {df['il_short_name'].nunique()} unique ILs")

    # Map images
    print("Mapping image paths...")
    df = add_image_paths(df, image_dir)
    n_cosmo = df["cosmo_image_path"].notna().sum()
    n_ep = df["ep_image_path"].notna().sum()
    print(f"  COSMO images found: {n_cosmo}/{len(df)} rows")
    print(f"  EP images found: {n_ep}/{len(df)} rows")

    # Merge surface descriptors
    print("Merging surface descriptors...")
    df = merge_surface_descriptors(df)

    # Save raw CSV (before normalization)
    raw_csv_path = output_dir / "il_data_raw.csv"
    df.to_csv(raw_csv_path, index=False)
    print(f"  Saved raw data to {raw_csv_path}")

    # Normalize
    print("Normalizing features...")
    df_norm, feat_scaler, target_scaler = normalize_features(df.copy())

    # Save normalized CSV
    norm_csv_path = output_dir / "il_data_normalized.csv"
    df_norm.to_csv(norm_csv_path, index=False)

    # Save scalers
    import pickle
    with open(output_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump(feat_scaler, f)
    with open(output_dir / "target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    # Create splits
    seed = config.get("data", {}).get("random_seed", 42) if config else 42
    print("Creating train/val/test splits...")
    splits = create_splits(df_norm, seed=seed)

    # Save splits
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    for split_name in ["train", "val", "test"]:
        splits[split_name].to_csv(splits_dir / f"{split_name}.csv", index=False)

    # Save split info
    split_info = {
        "train_ils": splits["train_ils"],
        "val_ils": splits["val_ils"],
        "test_ils": splits["test_ils"],
    }
    with open(splits_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print("Preprocessing complete!")
    return df, splits


def verify_image_mapping(image_dir: str):
    """Verify that all mapped image files exist."""
    image_dir = Path(image_dir)
    print("Verifying image mapping...")
    missing = []
    found = []

    for il_name, paths in IL_IMAGE_MAP.items():
        for img_type, rel_path in paths.items():
            if rel_path is None:
                missing.append(f"  {il_name}/{img_type}: NOT AVAILABLE (no image)")
                continue
            full_path = image_dir / rel_path
            if full_path.exists():
                found.append(f"  {il_name}/{img_type}: OK")
            else:
                missing.append(f"  {il_name}/{img_type}: MISSING -> {rel_path}")

    print(f"\nFound: {len(found)} images")
    print(f"Missing: {len(missing)} images")
    if missing:
        print("\nMissing images:")
        for m in missing:
            print(m)

    return len(missing) == 0


if __name__ == "__main__":
    import sys
    base_dir = Path(__file__).resolve().parent.parent.parent

    # Verify images first
    verify_image_mapping(base_dir / "Image")

    # Run full preprocessing
    preprocess_and_save(
        excel_path=str(base_dir / "Activity coeff and Excess Enthalpy for Imaging.xlsx"),
        image_dir=str(base_dir / "Image"),
        output_dir=str(base_dir / "data" / "processed"),
    )
