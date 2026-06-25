"""Phase #1 Step F: Assemble the expanded cached_*.npz files.

Inputs (already exist):
    - data/merged_v4/splits/{train,val,test}.csv            : the expanded splits
    - data/merged_v4/target_scalers_{original,ilthermo}.pkl : source-specific target scalers
    - cosmobridge_v5/data/precomputed_chemprop_features.npz : chemprop_fp + surface_feat for 243 SMILES
    - cosmobridge_v4/data/cached_{train,val,test}.npz       : original 152/32/39 samples with full features

Outputs:
    - cosmobridge_v4/data/cached_{train,val,test}_expanded.npz
        keys: smiles, il_ids, chemprop_fp (N, 300), surface_fp (N, 256),
              thermo_feat (N, 25), targets (N, 7), target_mask (N, 7),
              preds_fusion (N, 7), preds_chemprop (N, 7),
              source (N,) in {original, ilthermo}

Design notes:
    - For new (ilthermo) samples we cannot produce true v4 preds_fusion
      and preds_chemprop. As a pragmatic substitute we use the source
      mean per property (computed from the original 152 samples).
      This gives the PerPropHead a sensible "starting point" base that
      won't inject outliers, but means ilthermo samples contribute as
      "predict-from-scratch" training signal for the residual head.
    - target_mask is 1 where the ground-truth target is present,
      0 where NaN. The PerPropHead training loop must multiply its
      MSE loss by target_mask so missing labels are ignored.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parent.parent
MERGED = PROJECT / "data/merged_v4"
V5 = PROJECT / "cosmobridge_v5"
CACHED_DIR = PROJECT / "cosmobridge_v4/data"

FEAT_COLS = [
    "temperature", "x1", "inv_temperature", "temp_squared", "temp_cubed",
    "surface_area", "volume", "sphericity", "aspect_ratio",
    "curv_mean", "curv_std", "curv_skew",
    "gcurv_mean", "gcurv_std", "gcurv_skew",
    "esp_mean", "esp_std", "esp_min", "esp_max",
    "esp_skew", "esp_kurtosis", "esp_pos_frac", "esp_neg_frac",
    "esp_charge_segregation", "esp_range",
]
PROPS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


def build_smiles_lookup():
    """Map each unique SMILES to its chemprop_fp (300) and surface_fp (256)."""
    lookup = {}

    # Source 1: original cached files (152 train + 32 val + 39 test samples).
    # Note: same SMILES appears multiple times across temperature rows, so we
    # just store the first occurrence per SMILES.
    for split in ("train", "val", "test"):
        c = np.load(CACHED_DIR / f"cached_{split}.npz", allow_pickle=True)
        for i, s in enumerate(c["smiles"]):
            s = str(s)
            if s not in lookup:
                lookup[s] = (c["chemprop_fp"][i].astype(np.float32),
                              c["surface_fp"][i].astype(np.float32))

    # Source 2: precomputed chemprop features for the expansion compounds.
    pc = np.load(V5 / "data/precomputed_chemprop_features.npz", allow_pickle=True)
    for i, s in enumerate(pc["smiles"]):
        s = str(s)
        if s not in lookup:
            lookup[s] = (pc["graph_feat"][i].astype(np.float32),
                          pc["surface_feat"][i].astype(np.float32))

    return lookup


def build_original_v4_stats():
    """Per-property mean of preds_fusion and preds_chemprop on the original
    152-sample training set. Used as the fallback base for new ilthermo
    samples where we can't run the v4 model."""
    c = np.load(CACHED_DIR / "cached_train.npz", allow_pickle=True)
    return {
        "fusion_mean": c["preds_fusion"].mean(axis=0).astype(np.float32),
        "chemprop_mean": c["preds_chemprop"].mean(axis=0).astype(np.float32),
    }


def fit_thermo_scaler(train_df):
    """Z-score the 25 thermo features using the training subset only."""
    return StandardScaler().fit(train_df[FEAT_COLS].values.astype(np.float32))


def source_specific_target_scale(df, scaler_orig, scaler_ilth):
    """Return (targets, mask) from pre-scaled target columns in merged_v4.

    IMPORTANT: the merged_v4 CSVs already contain source-specific
    z-scored targets (verified on 2026-04-11: merged_v4/test.csv matches
    cached_test.npz targets to ~4 decimal places). We copy the values
    directly and set mask = 1 where non-NaN. The scaler_orig / scaler_ilth
    arguments are kept for API compatibility but are not applied.
    """
    n = len(df)
    targets = np.zeros((n, 7), dtype=np.float32)
    mask = np.zeros((n, 7), dtype=np.float32)
    for i, p in enumerate(PROPS):
        if p not in df.columns:
            continue
        col = df[p].values
        present = ~np.isnan(col)
        targets[present, i] = col[present].astype(np.float32)
        mask[present, i] = 1.0
    return targets, mask


def build_split(split_name, split_df, smi_lookup, thermo_scaler,
                scaler_orig, scaler_ilth, v4_stats):
    n = len(split_df)
    print(f"\nBuilding {split_name}: {n} rows")

    chemprop_fp = np.zeros((n, 300), dtype=np.float32)
    surface_fp = np.zeros((n, 256), dtype=np.float32)
    miss = 0
    for i, s in enumerate(split_df["smiles"].astype(str)):
        if s in smi_lookup:
            chemprop_fp[i], surface_fp[i] = smi_lookup[s]
        else:
            miss += 1
    if miss:
        print(f"  WARNING: {miss}/{n} SMILES missing from lookup")

    # Scale the 25-D thermo features
    thermo_feat = thermo_scaler.transform(split_df[FEAT_COLS].values.astype(np.float32)).astype(np.float32)

    # Source-specific target scaling + availability mask
    targets, target_mask = source_specific_target_scale(split_df, scaler_orig, scaler_ilth)

    # v4 base predictions:
    #   - For original source: take from cached_*.npz (indexed by row order).
    #   - For ilthermo source: use fallback mean per property.
    preds_fusion = np.tile(v4_stats["fusion_mean"], (n, 1))
    preds_chemprop = np.tile(v4_stats["chemprop_mean"], (n, 1))

    # Pull real v4 preds for the original rows (they preserve order in split).
    if (split_df["source"] == "original").any():
        orig_cached_path = CACHED_DIR / f"cached_{split_name}.npz"
        if orig_cached_path.exists():
            oc = np.load(orig_cached_path, allow_pickle=True)
            orig_mask = (split_df["source"] == "original").values
            n_orig = int(orig_mask.sum())
            if n_orig == len(oc["targets"]):
                preds_fusion[orig_mask] = oc["preds_fusion"].astype(np.float32)
                preds_chemprop[orig_mask] = oc["preds_chemprop"].astype(np.float32)
                print(f"  filled {n_orig} original-source v4 preds from cached_{split_name}.npz")
            else:
                print(f"  NOTE: {split_name} original count mismatch ({n_orig} vs cached {len(oc['targets'])}); using fallback means")

    out = dict(
        smiles=split_df["smiles"].astype(str).values,
        il_ids=split_df["il_short_name"].astype(str).values,
        chemprop_fp=chemprop_fp,
        surface_fp=surface_fp,
        thermo_feat=thermo_feat,
        targets=targets,
        target_mask=target_mask,
        preds_fusion=preds_fusion.astype(np.float32),
        preds_chemprop=preds_chemprop.astype(np.float32),
        source=split_df["source"].astype(str).values,
    )

    # Report
    print(f"  chemprop_fp: {chemprop_fp.shape}")
    print(f"  thermo_feat: {thermo_feat.shape}  (z-scored: mean={thermo_feat.mean():.3f} std={thermo_feat.std():.3f})")
    print(f"  target availability per property:")
    for i, p in enumerate(PROPS):
        print(f"    {p:8s}: {int(target_mask[:, i].sum())}/{n}")
    return out


def main():
    with open(MERGED / "target_scalers_original.pkl", "rb") as f:
        scaler_orig = pickle.load(f)
    with open(MERGED / "target_scalers_ilthermo.pkl", "rb") as f:
        scaler_ilth = pickle.load(f)

    print("Target scalers loaded:")
    print(f"  original  : {list(scaler_orig.keys())}")
    print(f"  ilthermo  : {list(scaler_ilth.keys())}")

    smi_lookup = build_smiles_lookup()
    print(f"\nSMILES feature lookup: {len(smi_lookup)} unique SMILES")

    v4_stats = build_original_v4_stats()
    print(f"\nv4 base means (for ilthermo fallback):")
    print(f"  preds_fusion : {v4_stats['fusion_mean']}")
    print(f"  preds_chemprop: {v4_stats['chemprop_mean']}")

    train_df = pd.read_csv(MERGED / "splits/train.csv")
    val_df = pd.read_csv(MERGED / "splits/val.csv")
    test_df = pd.read_csv(MERGED / "splits/test.csv")

    thermo_scaler = fit_thermo_scaler(train_df)

    out_dir = CACHED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = build_split(name, df, smi_lookup, thermo_scaler,
                           scaler_orig, scaler_ilth, v4_stats)
        out_path = out_dir / f"cached_{name}_expanded.npz"
        np.savez(out_path, **out)
        print(f"  saved to {out_path.name}")


if __name__ == "__main__":
    main()
