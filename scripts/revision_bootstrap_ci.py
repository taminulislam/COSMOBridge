"""Per-property bootstrap 95% CIs for the COSMOBridge headline (review Crit-1b),
and per-property paired Delta vs Chemprop with Bonferroni flags (review Crit-1a backing).

Pure NumPy on results/cosmobridge_test_preds.npz (produced by revision_savepreds.py).
Safe to run on a login node (no model forward). B=10000 bootstrap resamples of the
39 test rows; the COSMOBridge prediction is the 5-seed ensemble mean.
"""
import numpy as np
from pathlib import Path

d = np.load(Path(__file__).resolve().parent.parent / "results" / "cosmobridge_test_preds.npz",
            allow_pickle=True)
sp = d["seed_preds"]            # (5,39,7)
y = d["targets"]               # (39,7)
cp = d["chemprop_preds"]        # (39,7)
cols = [str(c) for c in d["target_cols"]]
ens = sp.mean(axis=0)          # (39,7) 5-seed ensemble mean


def r2_cols(pred, true):
    res = ((true - pred) ** 2).sum(0)
    tot = ((true - true.mean(0)) ** 2).sum(0)
    return 1.0 - res / tot


n = len(y)
B = 10000
rng = np.random.default_rng(0)
be = np.empty((B, len(cols)))
bc = np.empty((B, len(cols)))
for b in range(B):
    idx = rng.integers(0, n, n)
    be[b] = r2_cols(ens[idx], y[idx])
    bc[b] = r2_cols(cp[idx], y[idx])

pt = r2_cols(ens, y)
print("=== COSMOBridge per-property R^2 with 95% bootstrap CI (B=10000) ===")
for i, c in enumerate(cols):
    lo, hi = np.percentile(be[:, i], [2.5, 97.5])
    print(f"  {c:7s} {pt[i]:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

m = len(cols)
print("\n=== Paired Delta (COSMOBridge - Chemprop), per property ===")
for i, c in enumerate(cols):
    dd = be[:, i] - bc[:, i]
    lo, hi = np.percentile(dd, [2.5, 97.5])
    p_le0 = (dd <= 0).mean()
    p_two = 2 * min(p_le0, 1 - p_le0)
    print(f"  {c:7s} d={dd.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"P(d<=0)={p_le0:.4f}  Bonferroni-sig(m={m})={p_two * m < 0.05}")

davg = be.mean(1) - bc.mean(1)
lo, hi = np.percentile(davg, [2.5, 97.5])
print(f"\n=== Average Delta vs Chemprop ===\n  d={davg.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
      f"P(d<=0)={(davg <= 0).mean():.4f}")
