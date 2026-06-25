"""Crit-2: 5-seed evaluation of STILT (chemprop_tuned config C = full mask, 48x OS),
the nearest competitor to COSMOBridge, to show the ranking is not seed-fragile.

STILT uses thermo-only features (no COSMO clouds), so it is fully retrainable in
our env. PointCloud and GBH+STILT both run PointNet on the raw .npz clouds, which
are owner-only (0600) in this working copy and cannot be re-seeded here.
"""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.preprocessing import TARGET_COLUMNS
from scripts.train_chemprop_tuned import train

DATA = Path("data/chemprop_tuned/c")        # config C = STILT (full mask, 48x oversampling)
SEEDS = [42, 123, 456, 789, 1024]

rows = []
for sd in SEEDS:
    ck = f"checkpoints/revision/stilt_seed{sd}"
    m = train(DATA, ck, seed=sd)
    if m:
        rows.append(m)
        print(f"  seed {sd}: avg_r2={m['avg_r2']:.4f}")
    else:
        print(f"  seed {sd}: FAILED")

avg = [m["avg_r2"] for m in rows]
out = {
    "model": "STILT (chemprop_tuned C: full mask, 48x OS)",
    "seeds": SEEDS,
    "per_seed": rows,
    "avg_mean": float(np.mean(avg)), "avg_std": float(np.std(avg)),
    "per_prop_mean": {p: float(np.mean([m[f"{p}_r2"] for m in rows])) for p in TARGET_COLUMNS},
    "per_prop_std": {p: float(np.std([m[f"{p}_r2"] for m in rows])) for p in TARGET_COLUMNS},
}
Path("results").mkdir(exist_ok=True)
json.dump(out, open("results/stilt_multiseed.json", "w"), indent=2)
print(f"\nSTILT 5-seed avg R^2 = {out['avg_mean']:.4f} +/- {out['avg_std']:.4f}")
print(f"(COSMOBridge 5-seed = 0.801 +/- 0.001)")
print("Saved results/stilt_multiseed.json")
