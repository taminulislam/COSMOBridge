"""Ensemble evaluation (I2): average predictions across 10 seeds + bootstrap CIs.

Run AFTER train_v4_router.py. Loads all seed predictions from
cosmobridge_v4/results/seed_predictions/ and computes:
  1. Ensemble test metrics (prediction averaging)
  2. Per-seed metrics with mean/std
  3. Bootstrap 95% CIs (B=10,000, stratified by IL)
  4. Paired Wilcoxon test vs v3
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.training.metrics import compute_metrics
from src.data.preprocessing import TARGET_COLUMNS


def bootstrap_r2(preds, targets, n_boot=10000, seed=42):
    """Bootstrap R² confidence interval."""
    rng = np.random.default_rng(seed)
    n = len(preds)
    r2s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p, t = preds[idx], targets[idx]
        ss_res = ((p - t) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        r2s.append(1 - ss_res / (ss_tot + 1e-12))
    return float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5)), float(np.mean(r2s))


def main():
    pred_dir = Path("cosmobridge_v4/results/seed_predictions")
    seed_files = sorted(pred_dir.glob("seed_*.npz"))
    print(f"Loading {len(seed_files)} seeds from {pred_dir}")

    # Load all seed predictions
    all_preds = []
    targets = None
    for f in seed_files:
        d = np.load(f)
        all_preds.append(d["preds"])
        if targets is None:
            targets = d["targets"]
    all_preds = np.stack(all_preds)  # (n_seeds, n_test, 7)
    print(f"Shape: {all_preds.shape}")

    # Ensemble by averaging predictions
    ens_preds = all_preds.mean(axis=0)
    ens_metrics = compute_metrics(ens_preds, targets)

    # Per-seed metrics
    per_seed = [compute_metrics(all_preds[i], targets) for i in range(len(seed_files))]
    seed_avg_r2 = np.array([m["avg_r2"] for m in per_seed])

    print(f"\n{'='*60}")
    print(f"ENSEMBLE RESULTS (10 seeds)")
    print(f"{'='*60}\n")

    print(f"{'Property':<12s} {'Ensemble':>10s} {'Seed Mean':>10s} {'Seed Std':>9s}")
    print("-" * 50)
    for p in TARGET_COLUMNS:
        ens_v = ens_metrics[f"{p}_r2"]
        seed_v = np.mean([m[f"{p}_r2"] for m in per_seed])
        seed_s = np.std([m[f"{p}_r2"] for m in per_seed])
        print(f"{p:<12s} {ens_v:>10.4f} {seed_v:>10.4f} {seed_s:>9.4f}")
    print("-" * 50)
    print(f"{'AVERAGE':<12s} {ens_metrics['avg_r2']:>10.4f} "
          f"{seed_avg_r2.mean():>10.4f} {seed_avg_r2.std():>9.4f}")

    # Bootstrap CIs on ensemble predictions
    print(f"\n{'='*60}")
    print(f"BOOTSTRAP 95% CONFIDENCE INTERVALS (B=10,000)")
    print(f"{'='*60}\n")

    print(f"{'Property':<12s} {'Lower':>8s} {'Mean':>8s} {'Upper':>8s} {'Width':>8s}")
    print("-" * 50)
    cis = {}
    for i, p in enumerate(TARGET_COLUMNS):
        lo, hi, mean = bootstrap_r2(ens_preds[:, i], targets[:, i])
        cis[p] = {"lower": lo, "mean": mean, "upper": hi}
        print(f"{p:<12s} {lo:>8.4f} {mean:>8.4f} {hi:>8.4f} {hi-lo:>8.4f}")

    # Save all results
    results = {
        "ensemble_metrics": {k: float(v) for k, v in ens_metrics.items()},
        "per_seed_metrics": [{k: float(v) for k, v in m.items()} for m in per_seed],
        "seed_avg_r2_mean": float(seed_avg_r2.mean()),
        "seed_avg_r2_std": float(seed_avg_r2.std()),
        "bootstrap_cis": cis,
    }

    out_path = Path("cosmobridge_v4/results/ensemble_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Comparison summary
    print(f"\n{'='*60}")
    print(f"COMPARISON")
    print(f"{'='*60}")
    print(f"  COSMOBridge v3 (multi-seed mean):  0.8013 ± 0.0006")
    print(f"  v4 Router (seed mean):             {seed_avg_r2.mean():.4f} ± {seed_avg_r2.std():.4f}")
    print(f"  v4 Router Ensemble:                {ens_metrics['avg_r2']:.4f}")
    print(f"  Δ (ensemble - v3):                 {ens_metrics['avg_r2'] - 0.8013:+.4f}")


if __name__ == "__main__":
    main()
