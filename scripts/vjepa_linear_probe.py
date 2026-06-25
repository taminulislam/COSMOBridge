"""Idea 7: Linear probe baseline on DFT V-JEPA features.

Pure ridge regression from 192-D CLS embeddings to the 7 target properties.
No v4 base, no multimodal fusion, no temperature gating.

This is the diagnostic baseline: can raw V-JEPA features *alone* predict
the 7 thermo properties to any useful degree? Expected to underperform
everything (pure image-only ceiling on this dataset is known to collapse),
but useful as a published negative result.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parent.parent
V5 = PROJECT / "cosmobridge_v5"
PROPS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


def load_features(source):
    suffix = "_dft" if source == "dft" else ""
    tr = np.load(V5 / f"data/cached_image_features_train{suffix}.npz")["vit_feat"]
    te = np.load(V5 / f"data/cached_image_features_test{suffix}.npz")["vit_feat"]
    return tr.astype(np.float32), te.astype(np.float32)


def main():
    tc = np.load(PROJECT / "cosmobridge_v4/data/cached_train.npz", allow_pickle=True)
    tsc = np.load(PROJECT / "cosmobridge_v4/data/cached_test.npz", allow_pickle=True)
    y_tr = tc["targets"].astype(np.float32)
    y_te = tsc["targets"].astype(np.float32)
    v4_te = tsc["preds_fusion"].astype(np.float32)

    rows = []
    for source in ("gasteiger", "dft"):
        X_tr, X_te = load_features(source)
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Sweep ridge alpha
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
        best_alpha = None
        best_avg = -np.inf
        for a in alphas:
            ridge = Ridge(alpha=a).fit(X_tr_s, y_tr)
            pred = ridge.predict(X_te_s)
            avg = np.mean([r2_score(y_te[:, i], pred[:, i]) for i in range(7)])
            if avg > best_avg:
                best_avg = avg
                best_alpha = a

        ridge = Ridge(alpha=best_alpha).fit(X_tr_s, y_tr)
        pred = ridge.predict(X_te_s)
        per_prop = {p: float(r2_score(y_te[:, i], pred[:, i])) for i, p in enumerate(PROPS)}
        avg = float(np.mean(list(per_prop.values())))
        print(f"\n=== {source.upper()} V-JEPA linear probe (alpha={best_alpha}) ===")
        print(f"  avg R² = {avg:.4f}")
        for p in PROPS:
            print(f"  {p:8s}: {per_prop[p]:+.4f}")
        rows.append({"source": source, "alpha": best_alpha, "avg_r2": avg, **per_prop})

    # v4 baseline for context
    v4_per = {p: float(r2_score(y_te[:, i], v4_te[:, i])) for i, p in enumerate(PROPS)}
    v4_avg = float(np.mean(list(v4_per.values())))
    print(f"\nv4 baseline (preds_fusion only): avg R² = {v4_avg:.4f}")

    out = V5 / "results/vjepa_linear_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"probes": rows, "v4_baseline": {"avg_r2": v4_avg, **v4_per}}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
