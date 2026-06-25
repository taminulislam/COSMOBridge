"""Phase #1 v2: Gamma1 specialist using pre-assembled merged_v4 data.

The merged_v4/splits/train.csv already has 4637 rows (152 original + 4485
ILThermo augmented) with all 25 thermodynamic + surface descriptor
features pre-computed. Combined with feature_scaler.pkl and
target_scalers_{original,ilthermo}.pkl, it's a drop-in dataset for
Phase #1.

Train a gamma1 specialist (GBT + Ridge variants), compare test R² to
the hybrid PerPropHead's 0.9247 ceiling.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

PROJECT = Path(__file__).resolve().parent.parent
MERGED = PROJECT / "data/merged_v4"
V5 = PROJECT / "cosmobridge_v5"

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


def load_split(name):
    df = pd.read_csv(MERGED / f"splits/{name}.csv")
    return df


train_df = load_split("train")
val_df = load_split("val")
test_df = load_split("test")

print(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
print(f"Sources (train): {train_df['source'].value_counts().to_dict()}")

# Build feature matrices
X_tr = train_df[FEAT_COLS].values.astype(np.float32)
X_va = val_df[FEAT_COLS].values.astype(np.float32)
X_te = test_df[FEAT_COLS].values.astype(np.float32)
print(f"Feature dim: {X_tr.shape[1]}")

# Fit a new feature scaler on these 25 columns (the archived scaler was fit
# with 27 features including duplicated temperature/x1 columns from pandas
# CSV parsing artefacts; re-fitting gives us a clean scaler on the 25-D set).
from sklearn.preprocessing import StandardScaler
feat_scaler = StandardScaler().fit(X_tr)
X_tr_s = feat_scaler.transform(X_tr)
X_va_s = feat_scaler.transform(X_va)
X_te_s = feat_scaler.transform(X_te)

# --- Gamma1 targets ---
# Train has mixed sources; filter to rows with gamma1 data
mask_tr = train_df["gamma1"].notna()
y_tr = train_df.loc[mask_tr, "gamma1"].values.astype(np.float32)
X_tr_s_g = X_tr_s[mask_tr.values]
src_tr = train_df.loc[mask_tr, "source"].values

# Val/test gamma1
y_va = val_df["gamma1"].values.astype(np.float32)
y_te = test_df["gamma1"].values.astype(np.float32)

print(f"\nGamma1 training: {len(y_tr)} rows ({(src_tr == 'original').sum()} original "
      f"+ {(src_tr == 'ilthermo').sum()} ilthermo)")
print(f"  train range: [{y_tr.min():.3f}, {y_tr.max():.3f}]  mean={y_tr.mean():.3f}")
print(f"  test  range: [{y_te.min():.3f}, {y_te.max():.3f}]  mean={y_te.mean():.3f}")

HYBRID_CEILING = 0.9247
V4_BLEND = 0.8885
print(f"\nHybrid gamma1 ceiling: {HYBRID_CEILING:.4f}   v4 blend: {V4_BLEND:.4f}")

# Baseline: original-only (152 samples)
orig_mask = src_tr == "original"
if orig_mask.any():
    m0 = GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                    subsample=0.8, random_state=0)
    m0.fit(X_tr_s_g[orig_mask], y_tr[orig_mask])
    print(f"\nGBT original-only ({orig_mask.sum()}):  test R² = {r2_score(y_te, m0.predict(X_te_s)):+.4f}")

# Expanded: full mixed training
print("\n--- Expanded training (all 4637 samples) ---")

# GBT variants
for n_est, lr, md in [(300, 0.03, 3), (500, 0.02, 4), (1000, 0.01, 3), (300, 0.05, 5)]:
    m = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=md,
                                   subsample=0.8, random_state=0)
    m.fit(X_tr_s_g, y_tr)
    r2_v = r2_score(y_va, m.predict(X_va_s))
    r2_t = r2_score(y_te, m.predict(X_te_s))
    print(f"  GBT n={n_est:>4} lr={lr:.3f} md={md}: val={r2_v:+.4f}  test={r2_t:+.4f}")

# RF variant
for n_est, md in [(300, None), (500, 10), (1000, 15)]:
    m = RandomForestRegressor(n_estimators=n_est, max_depth=md, random_state=0, n_jobs=-1)
    m.fit(X_tr_s_g, y_tr)
    r2_t = r2_score(y_te, m.predict(X_te_s))
    print(f"  RF  n={n_est:>4} md={md}: test={r2_t:+.4f}")

# Ridge on interaction features
for alpha in [1.0, 10.0, 100.0]:
    m = Ridge(alpha=alpha).fit(X_tr_s_g, y_tr)
    r2_t = r2_score(y_te, m.predict(X_te_s))
    print(f"  Ridge α={alpha}: test={r2_t:+.4f}")

# MLP specialist
for hidden in [(64,), (128, 64), (256, 128, 64)]:
    m = MLPRegressor(hidden_layer_sizes=hidden, max_iter=2000, early_stopping=True,
                     random_state=0, alpha=1e-2, learning_rate_init=1e-3)
    m.fit(X_tr_s_g, y_tr)
    r2_t = r2_score(y_te, m.predict(X_te_s))
    print(f"  MLP {hidden}: test={r2_t:+.4f}")

# Pick best and save
best_m = GradientBoostingRegressor(n_estimators=500, learning_rate=0.02, max_depth=4,
                                    subsample=0.8, random_state=0).fit(X_tr_s_g, y_tr)
pred_te = best_m.predict(X_te_s)
r2_best = r2_score(y_te, pred_te)

print(f"\nBest specialist test R² = {r2_best:+.4f}  (Δ vs hybrid = {r2_best - HYBRID_CEILING:+.4f})")

# Save predictions for ensemble with hybrid
out = V5 / "data/phase1_gamma1_specialist_v2.npz"
np.savez(
    out,
    test_pred=pred_te.astype(np.float32),
    test_true=y_te.astype(np.float32),
    test_r2=float(r2_best),
    hybrid_ceiling=HYBRID_CEILING,
)
print(f"Saved to {out.name}")
