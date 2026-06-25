"""Phase #1 (decoupled): train a dedicated gamma1 specialist on the
3945-row expanded gamma1 dataset, then ensemble its gamma1 predictions
with the 0.8320 hybrid's gamma1 predictions. Keep the other 6 property
predictions from the hybrid unchanged.

The specialist uses the same hybrid feature stack (PCA(Gasteiger,20) +
PCA(DFT,20) + PCA(Supervised,20) = 60D) plus the 25 thermo+surface
descriptors (to match the archived recipe). It's trained as a standalone
regressor (no residual correction) because we cannot trust the v4 base
fallback for new ilthermo rows.

Output: cosmobridge_v5/results/perprop_dft/phase1_gamma1_only.json
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parent.parent
V5 = PROJECT / "cosmobridge_v5"
CACHED_DIR = PROJECT / "cosmobridge_v4/data"


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_vjepa(source, split):
    suffix = "_dft" if source == "dft" else ""
    return np.load(
        V5 / f"data/cached_image_features_{split}_expanded{suffix}.npz"
    )["vit_feat"].astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Expanded caches
    tc = np.load(CACHED_DIR / "cached_train_expanded.npz", allow_pickle=True)
    vc = np.load(CACHED_DIR / "cached_val_expanded.npz", allow_pickle=True)
    sc = np.load(CACHED_DIR / "cached_test_expanded.npz", allow_pickle=True)

    # Filter training rows to those with gamma1 label
    g1_mask = tc["target_mask"][:, 0] == 1.0
    print(f"Train rows with gamma1: {int(g1_mask.sum())} / {len(g1_mask)}")

    y_tr = tc["targets"][g1_mask, 0].astype(np.float32)
    y_va = vc["targets"][:, 0].astype(np.float32)
    y_te = sc["targets"][:, 0].astype(np.float32)

    # Feature stack: same as hybrid
    gast_tr = load_vjepa("gasteiger", "train")[g1_mask]
    gast_va = load_vjepa("gasteiger", "val")
    gast_te = load_vjepa("gasteiger", "test")
    dft_tr = load_vjepa("dft", "train")[g1_mask]
    dft_va = load_vjepa("dft", "val")
    dft_te = load_vjepa("dft", "test")

    pca_g = PCA(20).fit(gast_tr)
    pca_d = PCA(20).fit(dft_tr)
    G_tr = pca_g.transform(gast_tr).astype(np.float32)
    G_va = pca_g.transform(gast_va).astype(np.float32)
    G_te = pca_g.transform(gast_te).astype(np.float32)
    D_tr = pca_d.transform(dft_tr).astype(np.float32)
    D_va = pca_d.transform(dft_va).astype(np.float32)
    D_te = pca_d.transform(dft_te).astype(np.float32)

    # Supervised ViT features: 152 + 32 + 39 = 223. For expanded rows we
    # fall back to the per-column mean of the original 152 samples.
    sup = np.load(V5 / "data/supervised_vit_features.npz")["features"]
    sup_tr_orig = sup[:152].astype(np.float32)
    sup_va = sup[152:152 + 32].astype(np.float32)
    sup_te = sup[152 + 32:].astype(np.float32)
    pca_sup = PCA(20).fit(sup_tr_orig)

    # Build expanded Supervised stream aligned with tc rows (before masking)
    n_train_total = len(tc["source"])
    sup_tr_raw_full = np.zeros((n_train_total, sup_tr_orig.shape[1]), dtype=np.float32)
    orig_mask = tc["source"] == "original"
    sup_tr_raw_full[orig_mask] = sup_tr_orig
    sup_tr_raw_full[~orig_mask] = sup_tr_orig.mean(axis=0, keepdims=True)
    sup_tr_raw = sup_tr_raw_full[g1_mask]
    S_tr = pca_sup.transform(sup_tr_raw).astype(np.float32)
    S_va = pca_sup.transform(sup_va).astype(np.float32)
    S_te = pca_sup.transform(sup_te).astype(np.float32)

    # 25-D thermo+surface descriptors from cache
    th_tr = tc["thermo_feat"][g1_mask].astype(np.float32)
    th_va = vc["thermo_feat"].astype(np.float32)
    th_te = sc["thermo_feat"].astype(np.float32)

    X_tr = np.concatenate([G_tr, D_tr, S_tr, th_tr], axis=1).astype(np.float32)
    X_va = np.concatenate([G_va, D_va, S_va, th_va], axis=1).astype(np.float32)
    X_te = np.concatenate([G_te, D_te, S_te, th_te], axis=1).astype(np.float32)
    print(f"Feature dim: {X_tr.shape[1]}  Train rows: {X_tr.shape[0]}")

    class Gamma1MLP(nn.Module):
        def __init__(self, in_dim, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    # Baseline: hybrid reported gamma1 on test = 0.9247
    HYBRID_GAMMA1 = 0.9247
    HYBRID_GAMMA1_PREDS = None  # will load from hybrid's per-seed outputs if possible

    seed_results = []
    for seed in range(10):
        set_seed(seed)
        model = Gamma1MLP(X_tr.shape[1]).to(device)
        opt = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
        sch = CosineAnnealingLR(opt, T_max=300)
        dl = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=64, shuffle=True,
        )
        best_val = float("inf")
        best_state = None
        patience = 0
        for ep in range(300):
            model.train()
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                loss = ((model(xb) - yb) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sch.step()
            model.eval()
            with torch.no_grad():
                vl = ((model(torch.from_numpy(X_va).to(device))
                       - torch.from_numpy(y_va).to(device)) ** 2).mean().item()
            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 30:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred_te = model(torch.from_numpy(X_te).to(device)).cpu().numpy()
        r2 = r2_score(y_te, pred_te)
        seed_results.append({"seed": seed, "gamma1_r2": float(r2), "preds": pred_te.tolist()})
        print(f"  seed {seed}: gamma1 test R² = {r2:+.4f}")

    # Ensemble: mean predictions across seeds
    all_preds = np.stack([np.array(r["preds"]) for r in seed_results], axis=0)
    mean_pred = all_preds.mean(axis=0)
    ensemble_r2 = r2_score(y_te, mean_pred)
    print(f"\n=== Gamma1 specialist ensemble (10 seeds): R² = {ensemble_r2:+.4f} ===")
    print(f"vs hybrid gamma1 ceiling: {HYBRID_GAMMA1:.4f}  Δ = {ensemble_r2 - HYBRID_GAMMA1:+.4f}")

    # Optionally, blend with the hybrid's gamma1 predictions (loaded from the
    # hybrid's saved json via per-seed preds if present)
    hybrid_preds_path = V5 / "results/perprop_dft/hybrid_G20_D20_S20.json"
    blend_r2 = None
    if hybrid_preds_path.exists():
        with open(hybrid_preds_path) as f:
            hd = json.load(f)
        # The archived json only stored per-seed summary stats, not per-sample preds.
        # Without per-sample hybrid predictions, we cannot ensemble cleanly here.
        print(f"(hybrid per-sample preds not available in json — cannot blend live)")
    else:
        print("(hybrid preds file not found)")

    out = V5 / "results/perprop_dft/phase1_gamma1_only.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "method": "gamma1_specialist_on_expanded",
            "n_train": int(X_tr.shape[0]),
            "feature_dim": int(X_tr.shape[1]),
            "hybrid_gamma1_ceiling": HYBRID_GAMMA1,
            "ensemble_r2": float(ensemble_r2),
            "per_seed": [{"seed": r["seed"], "gamma1_r2": r["gamma1_r2"]} for r in seed_results],
            "mean_pred": mean_pred.tolist(),
            "y_test": y_te.tolist(),
        }, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
