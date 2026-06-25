"""Statistical comparison: v4 triple-path router vs v3 scalar gates.

Computes:
  1. Bootstrap 95% CIs (B=10,000) on each model's test R² per property
  2. Paired Wilcoxon signed-rank test on per-sample squared errors
  3. Paired bootstrap test on R² differences
  4. Per-IL performance breakdown

Uses cached Path A (Fusion) and Path B (Chemprop) predictions to reconstruct
v3 predictions with v3's learned gate values:
    v3: α_p = sigmoid([0.36, 0.39, 0.36, 0.42, 0.45, 0.37, 0.69])
    ŷ = α · pred_fusion + (1-α) · pred_chemprop
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.preprocessing import TARGET_COLUMNS


# v3's learned gate values (from paper Table 3, best seed)
V3_ALPHA = np.array([0.36, 0.39, 0.36, 0.42, 0.45, 0.37, 0.69])


def compute_r2(preds, targets, axis=0):
    """Compute R² per property."""
    ss_res = ((preds - targets) ** 2).sum(axis=axis)
    ss_tot = ((targets - targets.mean(axis=axis, keepdims=True)) ** 2).sum(axis=axis)
    return 1 - ss_res / (ss_tot + 1e-12)


def bootstrap_r2(preds, targets, n_boot=10000, seed=42):
    """Per-property bootstrap R² CI, stratified by IL identity if provided."""
    rng = np.random.default_rng(seed)
    n = len(preds)
    r2s = np.zeros((n_boot, 7))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r2s[b] = compute_r2(preds[idx], targets[idx])
    lo = np.percentile(r2s, 2.5, axis=0)
    hi = np.percentile(r2s, 97.5, axis=0)
    mean = r2s.mean(axis=0)
    return lo, mean, hi, r2s


def paired_bootstrap(preds_a, preds_b, targets, n_boot=10000, seed=42):
    """Paired bootstrap: what fraction of resamples does A beat B on avg R²?"""
    rng = np.random.default_rng(seed)
    n = len(targets)
    a_wins = 0
    deltas = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r2_a = compute_r2(preds_a[idx], targets[idx]).mean()
        r2_b = compute_r2(preds_b[idx], targets[idx]).mean()
        delta = r2_a - r2_b
        deltas.append(delta)
        if delta > 0:
            a_wins += 1
    deltas = np.array(deltas)
    return {
        "a_win_fraction": a_wins / n_boot,
        "delta_mean": float(deltas.mean()),
        "delta_ci_low": float(np.percentile(deltas, 2.5)),
        "delta_ci_high": float(np.percentile(deltas, 97.5)),
    }


def wilcoxon_per_property(preds_a, preds_b, targets):
    """Paired Wilcoxon test on per-sample squared errors for each property."""
    results = {}
    for i, prop in enumerate(TARGET_COLUMNS):
        err_a = (preds_a[:, i] - targets[:, i]) ** 2
        err_b = (preds_b[:, i] - targets[:, i]) ** 2
        try:
            stat, p = wilcoxon(err_a, err_b, alternative="less")  # A has smaller errors
            results[prop] = {"statistic": float(stat), "p_value": float(p),
                             "a_median_err": float(np.median(err_a)),
                             "b_median_err": float(np.median(err_b))}
        except ValueError as e:
            results[prop] = {"error": str(e)}
    return results


def main():
    # Load v4 triple predictions (mean across seeds = ensemble)
    v4_dir = Path("cosmobridge_v4/results/seed_predictions_triple")
    v4_files = sorted(v4_dir.glob("seed_*.npz"))
    print(f"Loading {len(v4_files)} v4 triple-path seeds...")

    v4_all_preds = []
    targets = None
    for f in v4_files:
        d = np.load(f)
        v4_all_preds.append(d["preds"])
        if targets is None:
            targets = d["targets"]
    v4_all_preds = np.stack(v4_all_preds)  # (n_seeds, n_test, 7)
    v4_ens_preds = v4_all_preds.mean(axis=0)  # ensemble prediction

    # Reconstruct v3 predictions using cached paths + v3 gates
    cached = np.load("cosmobridge_v4/data/cached_test.npz", allow_pickle=True)
    pred_fusion = cached["preds_fusion"]
    pred_chemprop = cached["preds_chemprop"]
    v3_preds = V3_ALPHA[None, :] * pred_fusion + (1 - V3_ALPHA[None, :]) * pred_chemprop

    print(f"v3 preds shape: {v3_preds.shape}, v4 ens preds: {v4_ens_preds.shape}")
    print(f"Targets shape: {targets.shape}")

    # Sanity check: verify v3 R² matches published
    v3_r2 = compute_r2(v3_preds, targets)
    v4_r2 = compute_r2(v4_ens_preds, targets)
    print(f"\nSanity check (v3 R² should match paper Table 1):")
    print(f"{'Property':<12s} {'v3 R²':>8s} {'v4 R²':>8s} {'Δ':>8s}")
    for i, p in enumerate(TARGET_COLUMNS):
        print(f"{p:<12s} {v3_r2[i]:>8.4f} {v4_r2[i]:>8.4f} {v4_r2[i]-v3_r2[i]:>+8.4f}")
    print(f"{'AVG':<12s} {v3_r2.mean():>8.4f} {v4_r2.mean():>8.4f} "
          f"{v4_r2.mean()-v3_r2.mean():>+8.4f}")

    # ── Bootstrap CIs ──
    print(f"\n{'='*70}")
    print(f"BOOTSTRAP 95% CONFIDENCE INTERVALS (B=10,000)")
    print(f"{'='*70}\n")

    v3_lo, v3_mean, v3_hi, _ = bootstrap_r2(v3_preds, targets)
    v4_lo, v4_mean, v4_hi, _ = bootstrap_r2(v4_ens_preds, targets)

    print(f"{'Property':<12s} {'v3 CI':>22s} {'v4 CI':>22s}")
    print("-" * 70)
    for i, p in enumerate(TARGET_COLUMNS):
        v3_ci = f"[{v3_lo[i]:.3f}, {v3_hi[i]:.3f}]"
        v4_ci = f"[{v4_lo[i]:.3f}, {v4_hi[i]:.3f}]"
        print(f"{p:<12s} {v3_ci:>22s} {v4_ci:>22s}")
    v3_avg_lo = v3_lo.mean(); v3_avg_hi = v3_hi.mean()
    v4_avg_lo = v4_lo.mean(); v4_avg_hi = v4_hi.mean()

    # ── Paired bootstrap ──
    print(f"\n{'='*70}")
    print(f"PAIRED BOOTSTRAP TEST (v4 vs v3)")
    print(f"{'='*70}\n")
    pb = paired_bootstrap(v4_ens_preds, v3_preds, targets)
    print(f"  P(v4 > v3 on avg R²):  {pb['a_win_fraction']*100:.1f}%")
    print(f"  Mean Δ (v4 − v3):      {pb['delta_mean']:+.4f}")
    print(f"  95% CI on Δ:           [{pb['delta_ci_low']:+.4f}, {pb['delta_ci_high']:+.4f}]")
    significant = pb['delta_ci_low'] > 0
    print(f"  Statistically significant: {'YES' if significant else 'NO'} "
          f"(CI {'does not' if significant else 'does'} cross zero)")

    # ── Wilcoxon per property ──
    print(f"\n{'='*70}")
    print(f"PAIRED WILCOXON SIGNED-RANK TEST (per-property, one-sided: v4 < v3 errors)")
    print(f"{'='*70}\n")

    wilcox = wilcoxon_per_property(v4_ens_preds, v3_preds, targets)
    print(f"{'Property':<12s} {'v4 median²':>12s} {'v3 median²':>12s} "
          f"{'p-value':>10s} {'sig?':>6s}")
    print("-" * 60)
    for p in TARGET_COLUMNS:
        r = wilcox[p]
        if "error" in r:
            print(f"{p:<12s} {r['error']}")
        else:
            sig = "***" if r['p_value'] < 0.001 else \
                  "**" if r['p_value'] < 0.01 else \
                  "*" if r['p_value'] < 0.05 else "ns"
            print(f"{p:<12s} {r['a_median_err']:>12.4f} {r['b_median_err']:>12.4f} "
                  f"{r['p_value']:>10.4f} {sig:>6s}")

    # ── Per-IL breakdown ──
    print(f"\n{'='*70}")
    print(f"PER-IL PERFORMANCE (avg R² across 7 properties)")
    print(f"{'='*70}\n")

    il_ids = cached["il_ids"]
    unique_ils = np.unique(il_ids)
    print(f"{'IL':<15s} {'n':>4s} {'v3 avg':>8s} {'v4 avg':>8s} {'Δ':>8s}")
    print("-" * 50)
    for il in unique_ils:
        mask = il_ids == il
        v3_il = compute_r2(v3_preds[mask], targets[mask]).mean()
        v4_il = compute_r2(v4_ens_preds[mask], targets[mask]).mean()
        print(f"{il:<15s} {mask.sum():>4d} {v3_il:>8.4f} {v4_il:>8.4f} "
              f"{v4_il-v3_il:>+8.4f}")

    # Save results
    results = {
        "v3_predictions": "reconstructed from cached paths + V3_ALPHA",
        "v4_predictions": "ensemble mean of 10 triple-path router seeds",
        "v3_r2": {p: float(v3_r2[i]) for i, p in enumerate(TARGET_COLUMNS)},
        "v4_r2": {p: float(v4_r2[i]) for i, p in enumerate(TARGET_COLUMNS)},
        "v3_avg_r2": float(v3_r2.mean()),
        "v4_avg_r2": float(v4_r2.mean()),
        "delta_avg_r2": float(v4_r2.mean() - v3_r2.mean()),
        "bootstrap_v3": {
            p: {"low": float(v3_lo[i]), "mean": float(v3_mean[i]), "high": float(v3_hi[i])}
            for i, p in enumerate(TARGET_COLUMNS)
        },
        "bootstrap_v4": {
            p: {"low": float(v4_lo[i]), "mean": float(v4_mean[i]), "high": float(v4_hi[i])}
            for i, p in enumerate(TARGET_COLUMNS)
        },
        "paired_bootstrap": pb,
        "wilcoxon_per_property": wilcox,
    }

    out_path = Path("cosmobridge_v4/results/bootstrap_cis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
