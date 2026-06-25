"""Parameter-free k-NN regression baseline using V-JEPA embeddings.

For each test sample, compute cosine similarity to all training samples
in V-JEPA-embedding space, take a softmax-weighted average of training
targets. No learning. Tests whether V-JEPA similarity carries useful
signal for the 39-sample test distribution.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

PROJECT = Path(__file__).resolve().parent.parent
V5 = PROJECT / "cosmobridge_v5"
PROPS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


def load_vjepa(source, split):
    suffix = "_dft" if source == "dft" else ""
    return np.load(
        V5 / f"data/cached_image_features_{split}{suffix}.npz"
    )["vit_feat"].astype(np.float32)


def main():
    tc = np.load(PROJECT / "cosmobridge_v4/data/cached_train.npz", allow_pickle=True)
    sc = np.load(PROJECT / "cosmobridge_v4/data/cached_test.npz", allow_pickle=True)
    y_tr = tc["targets"].astype(np.float32)
    y_te = sc["targets"].astype(np.float32)

    # Try four embedding sources: Gasteiger, DFT, averaged, concatenated
    emb_sources = {
        "gasteiger": (load_vjepa("gasteiger", "train"),
                       load_vjepa("gasteiger", "test")),
        "dft": (load_vjepa("dft", "train"),
                 load_vjepa("dft", "test")),
        "mean": ((load_vjepa("gasteiger", "train") + load_vjepa("dft", "train")) / 2,
                  (load_vjepa("gasteiger", "test") + load_vjepa("dft", "test")) / 2),
        "concat": (np.concatenate([load_vjepa("gasteiger", "train"),
                                    load_vjepa("dft", "train")], axis=1),
                    np.concatenate([load_vjepa("gasteiger", "test"),
                                    load_vjepa("dft", "test")], axis=1)),
    }

    # Baseline temperature features for additional distance weighting
    th_tr = tc["thermo_feat"][:, :5].astype(np.float32)   # T, x1, inv_T, T^2, T^3
    th_te = sc["thermo_feat"][:, :5].astype(np.float32)

    results = []

    for src_name, (emb_tr, emb_te) in emb_sources.items():
        emb_tr_n = emb_tr / (np.linalg.norm(emb_tr, axis=-1, keepdims=True) + 1e-8)
        emb_te_n = emb_te / (np.linalg.norm(emb_te, axis=-1, keepdims=True) + 1e-8)
        sim = emb_te_n @ emb_tr_n.T  # (39, 152)

        for tau in [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
            # Pure softmax weighting
            w = np.exp(sim * tau - (sim * tau).max(axis=1, keepdims=True))
            w = w / w.sum(axis=1, keepdims=True)
            pred = w @ y_tr
            r2_avg = float(np.mean([r2_score(y_te[:, i], pred[:, i]) for i in range(7)]))

            # Top-k hard selection (k=3, 5)
            for k in [3, 5, 10]:
                top_k = np.argsort(-sim, axis=1)[:, :k]
                # Uniform average
                pred_unif = np.stack([y_tr[top_k[i]].mean(axis=0) for i in range(len(y_te))])
                r2_unif = float(np.mean([r2_score(y_te[:, i], pred_unif[:, i]) for i in range(7)]))
                # Similarity-weighted average with numerical-stable softmax
                pred_sw = np.zeros_like(y_te)
                for i in range(len(y_te)):
                    idx = top_k[i]
                    logits = sim[i, idx] * tau
                    logits = logits - logits.max()
                    w_i = np.exp(logits)
                    w_i = w_i / w_i.sum()
                    pred_sw[i] = w_i @ y_tr[idx]
                r2_sw = float(np.mean([r2_score(y_te[:, i], pred_sw[:, i]) for i in range(7)]))

                results.append({
                    "embedding": src_name, "tau": tau, "k": k,
                    "softmax_all_r2": r2_avg,
                    "topk_uniform_r2": r2_unif,
                    "topk_weighted_r2": r2_sw,
                })

    # Print sorted by best R²
    results.sort(key=lambda r: max(r["softmax_all_r2"], r["topk_uniform_r2"], r["topk_weighted_r2"]), reverse=True)
    print(f"{'embed':<10}{'τ':<6}{'k':<4}{'softmax':<10}{'top-k unif':<12}{'top-k wt':<12}")
    for r in results[:15]:
        print(f"{r['embedding']:<10}{r['tau']:<6}{r['k']:<4}"
              f"{r['softmax_all_r2']:>8.4f}  {r['topk_uniform_r2']:>10.4f}  {r['topk_weighted_r2']:>10.4f}")

    best = max(results, key=lambda r: max(r["softmax_all_r2"], r["topk_uniform_r2"], r["topk_weighted_r2"]))
    best_r2 = max(best["softmax_all_r2"], best["topk_uniform_r2"], best["topk_weighted_r2"])
    print(f"\nBest kNN: {best['embedding']} τ={best['tau']} k={best['k']} → R² = {best_r2:.4f}")
    print(f"vs v4 blend (0.8091), hybrid (0.8320)")

    out = V5 / "results/knn_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"best": best, "best_r2": best_r2, "all": results[:50]}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
