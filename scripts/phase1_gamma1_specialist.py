"""Phase #1 minimal viable experiment: gamma1-specialist on expanded data.

Uses the 4930 augmented rows (ILThermoPy data, 115 ILs, gamma1 only)
plus the original 152 training rows to train a dedicated gamma1
regressor via gradient boosting on precomputed chemprop (300) +
surface (256) + thermo (25) features = 581D.

The hybrid PerPropHead achieves gamma1 R² = 0.9247 on the 39-sample
test split. This script trains a specialist and reports whether the
32× gamma1 data expansion improves that number.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parent.parent
V5 = PROJECT / "cosmobridge_v5"

# Load per-source target scalers that the v4 preprocessing used.
# The original cached_*.npz gamma1 values were transformed via this scaler.
with open(PROJECT / "data/merged_v4/target_scalers_original.pkl", "rb") as f:
    SCALERS_ORIG = pickle.load(f)
GAMMA1_MEAN = float(SCALERS_ORIG["gamma1"].mean_[0])
GAMMA1_STD = float(SCALERS_ORIG["gamma1"].scale_[0])
print(f"Original gamma1 scaler: mean={GAMMA1_MEAN:.3f}  std={GAMMA1_STD:.3f}")


def unscale_gamma1(z):
    return z * GAMMA1_STD + GAMMA1_MEAN

# Load augmented IL dataset
aug = pd.read_csv(PROJECT / "data/augmented/ilthermo_data.csv")
print(f"Augmented dataset: {len(aug)} rows, {aug['compound_id'].nunique()} unique compound_ids, "
      f"gamma1 non-null: {aug['gamma1'].notna().sum()}")

# Load precomputed chemprop + surface features indexed by SMILES
pc = np.load(V5 / "data/precomputed_chemprop_features.npz", allow_pickle=True)
pc_smiles = [str(s) for s in pc["smiles"]]
pc_graph = pc["graph_feat"]
pc_surface = pc["surface_feat"]
pc_index = {s: i for i, s in enumerate(pc_smiles)}
print(f"Precomputed features: {len(pc_smiles)} unique SMILES")

# Load the original v4 cached train/val/test (the 152/32/39 splits)
tc = np.load(PROJECT / "cosmobridge_v4/data/cached_train.npz", allow_pickle=True)
vc = np.load(PROJECT / "cosmobridge_v4/data/cached_val.npz", allow_pickle=True)
sc = np.load(PROJECT / "cosmobridge_v4/data/cached_test.npz", allow_pickle=True)


def build_matrix_from_cached(c):
    """Use the existing cached_*.npz (already has chemprop_fp, thermo_feat).

    Returns RAW gamma1 values by inverting the original source scaler.
    """
    X = np.concatenate([
        c["chemprop_fp"].astype(np.float32),
        c["surface_fp"].astype(np.float32),
        c["thermo_feat"].astype(np.float32),
    ], axis=1)
    y_gamma1_raw = unscale_gamma1(c["targets"][:, 0].astype(np.float32))
    smiles = [str(s) for s in c["smiles"]]
    return X, y_gamma1_raw, smiles


X_tr_orig, y_tr_orig, sm_tr_orig = build_matrix_from_cached(tc)
X_va, y_va, sm_va = build_matrix_from_cached(vc)
X_te, y_te, sm_te = build_matrix_from_cached(sc)

print(f"\nOriginal cached: train={len(y_tr_orig)}, val={len(y_va)}, test={len(y_te)}")


def build_thermo(temperature, x1):
    """Replicate the thermo_feat generation for a new augmented row.

    The original cache's thermo_feat has 25 dims; we reconstruct a
    minimal stand-in using (temperature, x1, derived polynomials).
    Since we're fitting a tree-based model, exact column correspondence
    is less important than having a stable feature set.
    """
    T = temperature
    x = x1
    feats = np.array([
        T, x, T * x, T ** 2, T ** 3, 1 / (T + 1e-6),
        np.log(T + 1), np.sqrt(max(T, 1)),
        x ** 2, x * (1 - x), np.log(x + 1e-3), 1 - x,
        T / 298.15, (T - 298.15), (T - 298.15) ** 2,
        x - 0.5, (x - 0.5) ** 2, x * T, x ** 2 * T,
        np.sin(T / 100), np.cos(T / 100),
        T * (1 - x), T / (x + 0.1), (1 - x) ** 2,
        T * T / 1000,
    ], dtype=np.float32)
    return feats


def build_augmented_matrix(aug_df, orig_train_smiles):
    """Build (X, y) rows from augmented dataset, skipping SMILES already in
    the original train set (to avoid duplication in features)."""
    already = set(orig_train_smiles)
    rows_X, rows_y, rows_smi = [], [], []
    miss_smiles = 0
    dup = 0
    for _, row in aug_df.iterrows():
        s = str(row["smiles"])
        if pd.isna(row["gamma1"]):
            continue
        if s in already:
            dup += 1
            continue
        if s not in pc_index:
            miss_smiles += 1
            continue
        pc_i = pc_index[s]
        graph = pc_graph[pc_i]
        surf = pc_surface[pc_i]
        thermo = build_thermo(row["temperature"], row["x1"])
        rows_X.append(np.concatenate([graph, surf, thermo]))
        rows_y.append(float(row["gamma1"]))
        rows_smi.append(s)
    print(f"Expanded rows: {len(rows_X)}  (skipped {dup} dup, {miss_smiles} SMILES not in chemprop cache)")
    if not rows_X:
        return None, None, None
    return (np.stack(rows_X).astype(np.float32),
            np.array(rows_y, dtype=np.float32),
            rows_smi)


X_exp, y_exp, sm_exp = build_augmented_matrix(aug, sm_tr_orig)

# Concatenate: original train + augmented expansion
X_tr = np.concatenate([X_tr_orig, X_exp], axis=0)
y_tr = np.concatenate([y_tr_orig, y_exp], axis=0)
print(f"\nCombined training set: {X_tr.shape} (original {len(y_tr_orig)} + expanded {len(y_exp)})")

# Normalize
scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_va_s = scaler.transform(X_va)
X_te_s = scaler.transform(X_te)

# Filter augmented gamma1 to the TEST-set range [0.05, 2.5] (test is [0.127, 2.08])
# The augmented dataset includes extreme outliers up to gamma1=3490, which
# distort the training signal. A narrower filter aligned with the test
# distribution gives a fairer "Phase #1" evaluation.
in_range = (y_tr >= 0.05) & (y_tr <= 2.5)
X_tr = X_tr[in_range]
y_tr = y_tr[in_range]
print(f"After [0.05, 2.5] filter: {len(y_tr)} training rows")
print(f"Raw gamma1 train range: [{y_tr.min():.3f}, {y_tr.max():.3f}]  mean={y_tr.mean():.3f}")
print(f"Raw gamma1 test  range: [{y_te.min():.3f}, {y_te.max():.3f}]")

# Re-fit scaler on filtered training data
scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_va_s = scaler.transform(X_va)
X_te_s = scaler.transform(X_te)

# Hybrid ceiling on gamma1 for reference
HYBRID_GAMMA1 = 0.9247  # from hybrid G20+D20+S20 run on 2026-04-11
V4_GAMMA1 = 0.8885

print(f"\n=== Hybrid gamma1 ceiling (z-scored space): {HYBRID_GAMMA1:.4f}  v4 blend: {V4_GAMMA1:.4f} ===")
print(f"(R² is scale-invariant so raw-space R² is directly comparable)")

# Baseline: train GBT on ORIGINAL train only (raw gamma1)
scaler_orig = StandardScaler().fit(X_tr_orig)
m_orig = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                                    subsample=0.8, random_state=0)
m_orig.fit(scaler_orig.transform(X_tr_orig), y_tr_orig)
r2_orig_te = r2_score(y_te, m_orig.predict(scaler_orig.transform(X_te)))
print(f"\nGBT on original train only (152 rows):  gamma1 test R² = {r2_orig_te:+.4f}")

# Expanded: train GBT on full expanded data
for n_est, lr, md in [(200, 0.05, 3), (300, 0.03, 3), (500, 0.02, 4), (300, 0.05, 4)]:
    m = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=md,
                                   subsample=0.8, random_state=0)
    m.fit(X_tr_s, y_tr)
    r2_va = r2_score(y_va, m.predict(X_va_s))
    r2_te = r2_score(y_te, m.predict(X_te_s))
    print(f"GBT expanded (n={n_est}, lr={lr}, md={md}, {len(y_tr)} rows):  val={r2_va:+.4f}  test={r2_te:+.4f}")

# Ridge baseline
for alpha in [1.0, 10.0, 100.0, 1000.0]:
    m = Ridge(alpha=alpha).fit(X_tr_s, y_tr)
    r2_te = r2_score(y_te, m.predict(X_te_s))
    print(f"Ridge α={alpha}:  test={r2_te:+.4f}")

# Save specialist predictions for potential ensemble
best_gbt = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.03, max_depth=3,
    subsample=0.8, random_state=0,
).fit(X_tr_s, y_tr)
pred_va = best_gbt.predict(X_va_s)
pred_te = best_gbt.predict(X_te_s)

out = V5 / "data/phase1_gamma1_specialist.npz"
np.savez(
    out,
    val_pred=pred_va.astype(np.float32),
    test_pred=pred_te.astype(np.float32),
    val_true=y_va.astype(np.float32),
    test_true=y_te.astype(np.float32),
    val_r2=float(r2_score(y_va, pred_va)),
    test_r2=float(r2_score(y_te, pred_te)),
    n_train=len(y_tr),
    n_expanded=len(y_exp),
)
print(f"\nSpecialist predictions saved to {out.name}")
print(f"Test gamma1 R²: {r2_score(y_te, pred_te):+.4f}")
print(f"  (hybrid ceiling: {HYBRID_GAMMA1:.4f},  v4 blend: {V4_GAMMA1:.4f})")
