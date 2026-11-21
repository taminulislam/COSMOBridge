"""B6: paired bootstrap of COSMOBridge - Chemprop on the held-out test set,
at two levels of resampling.

  row-level     : resample the 39 test rows independently (what the paper reports)
  cluster-level : resample the 5 held-out ionic liquids with replacement, keeping
                  each IL's 7-8 temperature points together

The 39 rows are 5 ILs x ~8 temperatures, so they are not independent: the
row-level interval ignores within-IL correlation and is anti-conservative.
Both are reported.

gamma1 and gamma2 were pre-specified as the surface-dependent properties before
the test set was scored; p-values are Bonferroni-corrected for those two tests.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/work/nvme/bgte/Dataset_Chemistry")
N_BOOT = 20000
RNG = np.random.default_rng(0)
PRESPECIFIED = ["gamma1", "gamma2"]


def r2(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def main():
    # the "official" footing: the same predictions Table 3 and the paper's
    # row-level bootstrap (results/c1_official_bootstrap.json) are computed from.
    d = np.load(ROOT / "results/cosmobridge_c1_official.npz", allow_pickle=True)
    cols = [str(c) for c in d["target_cols"]]
    targets = d["targets"].astype(np.float64)
    seed_preds = d["seed_preds"].astype(np.float64)        # (5, 39, 7)
    cp = d["chemprop_official_preds"].astype(np.float64)
    seeds = [int(s) for s in d["seeds"]]

    test = pd.read_csv(ROOT / "data/processed/splits/test.csv")
    assert len(test) == targets.shape[0], (len(test), targets.shape)
    il = test["il_short_name"].to_numpy()
    ils = np.unique(il)
    groups = [np.where(il == u)[0] for u in ils]
    print(f"{len(test)} test rows, {len(ils)} ILs: "
          + ", ".join(f"{u}({len(g)})" for u, g in zip(ils, groups)))

    out = {"n_rows": int(len(test)), "n_ils": int(len(ils)),
           "n_boot": N_BOOT, "prespecified": PRESPECIFIED, "properties": {}}

    def delta(idx, j):
        """mean over seeds of (R2_seed - R2_chemprop), matching Table 3's convention."""
        t = targets[idx, j]
        ss_tot = np.sum((t - t.mean()) ** 2)
        if ss_tot <= 0:
            return np.nan
        r2_cp = 1.0 - np.sum((t - cp[idx, j]) ** 2) / ss_tot
        r2_sd = 1.0 - np.sum((t[None, :] - seed_preds[:, idx, j]) ** 2, axis=1) / ss_tot
        return float(r2_sd.mean() - r2_cp)

    all_idx = np.arange(len(test))
    for j, c in enumerate(cols):
        obs = delta(all_idx, j)
        rec = {"delta_r2": float(obs),
               "r2_cosmobridge": float(np.mean([r2(seed_preds[s, :, j], targets[:, j])
                                                for s in range(seed_preds.shape[0])])),
               "r2_chemprop": float(r2(cp[:, j], targets[:, j]))}

        for level in ("row", "cluster"):
            diffs = np.empty(N_BOOT)
            for b in range(N_BOOT):
                if level == "row":
                    idx = RNG.integers(0, len(test), len(test))
                else:
                    pick = RNG.integers(0, len(groups), len(groups))
                    idx = np.concatenate([groups[k] for k in pick])
                diffs[b] = delta(idx, j)
            diffs = diffs[np.isfinite(diffs)]
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            # two-sided p: proportion of resamples on the other side of zero
            p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
            p = min(p, 1.0)
            p_adj = min(p * len(PRESPECIFIED), 1.0) if c in PRESPECIFIED else None
            rec[level] = {"ci_lo": float(lo), "ci_hi": float(hi),
                          "p": float(p),
                          "p_bonferroni": None if p_adj is None else float(p_adj),
                          "n_valid": int(len(diffs))}
        out["properties"][c] = rec

    # average across the 7 properties
    obs_avg = np.mean([out["properties"][c]["delta_r2"] for c in cols])
    out["avg_delta_r2"] = float(obs_avg)

    json.dump(out, open(ROOT / "results/b6_cluster_bootstrap.json", "w"), indent=2)

    print(f"\n{'property':10s} {'dR2':>8} | {'row 95% CI':>22} {'p':>8} | "
          f"{'cluster 95% CI':>22} {'p':>8}")
    for c in cols:
        r = out["properties"][c]
        rw, cl = r["row"], r["cluster"]
        star = "*" if c in PRESPECIFIED else " "
        print(f"{c:9s}{star} {r['delta_r2']:8.3f} | "
              f"[{rw['ci_lo']:8.3f},{rw['ci_hi']:8.3f}] {rw['p']:8.3f} | "
              f"[{cl['ci_lo']:8.3f},{cl['ci_hi']:8.3f}] {cl['p']:8.3f}")
    print("\n* = pre-specified; Bonferroni-adjusted p over the two pre-specified tests:")
    for c in PRESPECIFIED:
        r = out["properties"][c]
        print(f"   {c}: row {r['row']['p_bonferroni']:.4f}   "
              f"cluster {r['cluster']['p_bonferroni']:.4f}")
    print(f"\nSaved results/b6_cluster_bootstrap.json")


if __name__ == "__main__":
    main()
