"""RACA: Retrieval-Augmented Cross-Attention for IL property prediction.

Architecture:
    Query:       per-sample features (PCA(Gasteiger V-JEPA,20) + PCA(DFT V-JEPA,20)
                                        + PCA(Supervised,20) + thermo_feat)
    Retrieval:   cosine similarity on per-compound mean V-JEPA embedding,
                 top-k training compounds (k=3).
    Context:     for each retrieved compound, pick the sample at closest
                 temperature to the query; concat its features + 7-D target.
    Transformer: tiny encoder (d=128, 2 layers, 4 heads) that lets the
                 query token attend to the k neighbor tokens.
    Readout:     MLP on the refined query token → 7 property predictions.

Training: leave-one-compound-out. For each training sample, retrieval
excludes the sample's own compound from the candidate pool, so the
model never learns to retrieve itself. This mirrors the test-time
scenario where the 5 test compounds are unseen during training.

Runs on CPU/GPU, 10 seeds, ~2 min per seed.
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

PROJECT = Path(__file__).resolve().parent.parent
V5 = PROJECT / "cosmobridge_v5"
PROPS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]
K_NEIGHBORS = 3


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
        V5 / f"data/cached_image_features_{split}{suffix}.npz"
    )["vit_feat"].astype(np.float32)


def build_features():
    """Return feature matrices (train, val, test), per-compound V-JEPA
    embeddings (for retrieval), compound-id-per-sample map, and targets."""
    tc = np.load(PROJECT / "cosmobridge_v4/data/cached_train.npz", allow_pickle=True)
    vc = np.load(PROJECT / "cosmobridge_v4/data/cached_val.npz", allow_pickle=True)
    sc = np.load(PROJECT / "cosmobridge_v4/data/cached_test.npz", allow_pickle=True)

    y_tr = tc["targets"].astype(np.float32)
    y_va = vc["targets"].astype(np.float32)
    y_te = sc["targets"].astype(np.float32)

    # Per-sample V-JEPA (Gasteiger + DFT averaged for retrieval)
    gast_tr = load_vjepa("gasteiger", "train")
    gast_va = load_vjepa("gasteiger", "val")
    gast_te = load_vjepa("gasteiger", "test")
    dft_tr = load_vjepa("dft", "train")
    dft_va = load_vjepa("dft", "val")
    dft_te = load_vjepa("dft", "test")

    # Per-compound V-JEPA (average across both encoders AND across temperatures)
    # for the retrieval similarity metric.
    retrieval_emb_tr = (gast_tr + dft_tr) / 2.0
    train_smiles = [str(s) for s in tc["smiles"]]
    unique_smiles = list(dict.fromkeys(train_smiles))  # order preserved
    n_unique = len(unique_smiles)
    compound_emb = np.zeros((n_unique, 192), dtype=np.float32)
    compound_idx_per_train = np.zeros(len(train_smiles), dtype=np.int64)
    for cidx, s in enumerate(unique_smiles):
        mask = np.array([ts == s for ts in train_smiles])
        compound_emb[cidx] = retrieval_emb_tr[mask].mean(axis=0)
        compound_idx_per_train[mask] = cidx

    # For val/test, compute a per-sample retrieval embedding; this is the
    # query embedding used to compute similarities to train compounds.
    retrieval_emb_va = ((gast_va + dft_va) / 2.0).astype(np.float32)
    retrieval_emb_te = ((gast_te + dft_te) / 2.0).astype(np.float32)

    # Per-sample feature vectors for the query and neighbor tokens.
    # Match the archived hybrid recipe: PCA each V-JEPA stream to 20,
    # PCA Supervised to 20, then concat with the 25-D thermo+surface
    # descriptors. Total = 20+20+20+25 = 85.
    pca_g = PCA(20).fit(gast_tr)
    pca_d = PCA(20).fit(dft_tr)
    G_tr = pca_g.transform(gast_tr).astype(np.float32)
    G_va = pca_g.transform(gast_va).astype(np.float32)
    G_te = pca_g.transform(gast_te).astype(np.float32)
    D_tr = pca_d.transform(dft_tr).astype(np.float32)
    D_va = pca_d.transform(dft_va).astype(np.float32)
    D_te = pca_d.transform(dft_te).astype(np.float32)

    sup = np.load(V5 / "data/supervised_vit_features.npz")["features"]
    sup_tr = sup[:152].astype(np.float32)
    sup_va = sup[152:152 + 32].astype(np.float32)
    sup_te = sup[152 + 32:].astype(np.float32)
    pca_s = PCA(20).fit(sup_tr)
    S_tr = pca_s.transform(sup_tr).astype(np.float32)
    S_va = pca_s.transform(sup_va).astype(np.float32)
    S_te = pca_s.transform(sup_te).astype(np.float32)

    th_tr = tc["thermo_feat"].astype(np.float32)
    th_va = vc["thermo_feat"].astype(np.float32)
    th_te = sc["thermo_feat"].astype(np.float32)

    X_tr = np.concatenate([G_tr, D_tr, S_tr, th_tr], axis=1).astype(np.float32)
    X_va = np.concatenate([G_va, D_va, S_va, th_va], axis=1).astype(np.float32)
    X_te = np.concatenate([G_te, D_te, S_te, th_te], axis=1).astype(np.float32)

    return {
        "X_tr": X_tr, "X_va": X_va, "X_te": X_te,
        "y_tr": y_tr, "y_va": y_va, "y_te": y_te,
        "retrieval_emb_va": retrieval_emb_va,
        "retrieval_emb_te": retrieval_emb_te,
        "compound_emb": compound_emb,
        "compound_idx_per_train": compound_idx_per_train,
        "unique_smiles": unique_smiles,
        "n_unique": n_unique,
        "th_tr_temp": th_tr[:, 0],  # first thermo dim is (normalized) temperature
    }


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return a @ b.T


def build_context_for_train(data, k=K_NEIGHBORS):
    """For each training sample, retrieve k neighbors from training
    compounds OTHER than its own compound (LOO within training)."""
    n_tr = len(data["X_tr"])
    n_unique = data["n_unique"]
    compound_idx = data["compound_idx_per_train"]
    compound_emb = data["compound_emb"]
    X_tr = data["X_tr"]
    y_tr = data["y_tr"]
    th_temp = data["th_tr_temp"]

    # Similarity matrix: each training SAMPLE against each training COMPOUND
    per_sample_emb = np.zeros((n_tr, 192), dtype=np.float32)
    for i in range(n_tr):
        per_sample_emb[i] = compound_emb[compound_idx[i]]
    sim = cosine_sim(per_sample_emb, compound_emb)  # (n_tr, n_unique)

    # Exclude own compound by setting its similarity to -inf
    for i in range(n_tr):
        sim[i, compound_idx[i]] = -np.inf

    nbr_feat = np.zeros((n_tr, k, X_tr.shape[1]), dtype=np.float32)
    nbr_targ = np.zeros((n_tr, k, y_tr.shape[1]), dtype=np.float32)

    for i in range(n_tr):
        # Top-k compounds
        top_k = np.argsort(-sim[i])[:k]
        q_T = th_temp[i]
        for j, cidx in enumerate(top_k):
            # Among all training samples of this compound, pick the closest T
            mask = compound_idx == cidx
            comp_feats = X_tr[mask]
            comp_targs = y_tr[mask]
            comp_Ts = th_temp[mask]
            best = int(np.argmin(np.abs(comp_Ts - q_T)))
            nbr_feat[i, j] = comp_feats[best]
            nbr_targ[i, j] = comp_targs[best]

    return nbr_feat, nbr_targ


def build_context_for_query(data, query_emb, query_temp, k=K_NEIGHBORS):
    """Retrieve k neighbors from all training compounds for val/test."""
    compound_emb = data["compound_emb"]
    compound_idx = data["compound_idx_per_train"]
    X_tr = data["X_tr"]
    y_tr = data["y_tr"]
    th_temp = data["th_tr_temp"]

    n_q = len(query_emb)
    sim = cosine_sim(query_emb, compound_emb)  # (n_q, n_unique)
    nbr_feat = np.zeros((n_q, k, X_tr.shape[1]), dtype=np.float32)
    nbr_targ = np.zeros((n_q, k, y_tr.shape[1]), dtype=np.float32)

    for i in range(n_q):
        top_k = np.argsort(-sim[i])[:k]
        q_T = query_temp[i]
        for j, cidx in enumerate(top_k):
            mask = compound_idx == cidx
            comp_feats = X_tr[mask]
            comp_targs = y_tr[mask]
            comp_Ts = th_temp[mask]
            best = int(np.argmin(np.abs(comp_Ts - q_T)))
            nbr_feat[i, j] = comp_feats[best]
            nbr_targ[i, j] = comp_targs[best]
    return nbr_feat, nbr_targ


class RACAModel(nn.Module):
    def __init__(self, feat_dim, target_dim=7, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.d = d_model
        # Query uses a zero target placeholder
        self.query_proj = nn.Linear(feat_dim + target_dim, d_model)
        self.neighbor_proj = nn.Linear(feat_dim + target_dim, d_model)
        # Learnable type embedding to distinguish query from neighbor
        self.query_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.neighbor_type = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=256,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, target_dim),
        )

    def forward(self, q_feat, n_feat, n_targ):
        """
        q_feat: (B, feat_dim)
        n_feat: (B, k, feat_dim)
        n_targ: (B, k, target_dim)
        """
        B, k, _ = n_feat.shape
        # Query token: features + zero target placeholder
        zero_target = torch.zeros(B, q_feat.shape[-1] - q_feat.shape[-1] + 7, device=q_feat.device)
        q_in = torch.cat([q_feat, torch.zeros(B, 7, device=q_feat.device)], dim=-1)  # (B, feat+7)
        q_tok = self.query_proj(q_in).unsqueeze(1) + self.query_type  # (B, 1, d)
        n_in = torch.cat([n_feat, n_targ], dim=-1)  # (B, k, feat+7)
        n_tok = self.neighbor_proj(n_in) + self.neighbor_type  # (B, k, d)
        tokens = torch.cat([q_tok, n_tok], dim=1)  # (B, 1+k, d)
        out = self.encoder(tokens)  # (B, 1+k, d)
        return self.head(out[:, 0])


def train_seed(data, seed, epochs=300, lr=5e-4, weight_decay=1e-2):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr = data["X_tr"]
    y_tr = data["y_tr"]
    X_va = data["X_va"]
    y_va = data["y_va"]
    X_te = data["X_te"]
    y_te = data["y_te"]

    # Neighbor contexts (precomputed)
    n_tr_feat, n_tr_targ = build_context_for_train(data)
    # val / test: query embeddings already available
    n_va_feat, n_va_targ = build_context_for_query(
        data, data["retrieval_emb_va"], data["X_va"][:, 0]
    )
    n_te_feat, n_te_targ = build_context_for_query(
        data, data["retrieval_emb_te"], data["X_te"][:, 0]
    )

    model = RACAModel(feat_dim=X_tr.shape[1]).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    # Tensors
    t_X_tr = torch.from_numpy(X_tr).to(device)
    t_y_tr = torch.from_numpy(y_tr).to(device)
    t_nf_tr = torch.from_numpy(n_tr_feat).to(device)
    t_nt_tr = torch.from_numpy(n_tr_targ).to(device)
    t_X_va = torch.from_numpy(X_va).to(device)
    t_y_va = torch.from_numpy(y_va).to(device)
    t_nf_va = torch.from_numpy(n_va_feat).to(device)
    t_nt_va = torch.from_numpy(n_va_targ).to(device)
    t_X_te = torch.from_numpy(X_te).to(device)
    t_nf_te = torch.from_numpy(n_te_feat).to(device)
    t_nt_te = torch.from_numpy(n_te_targ).to(device)

    best_val = float("inf")
    best_state = None
    patience = 0
    for ep in range(epochs):
        model.train()
        # Simple full-batch training since dataset is tiny (152 rows)
        opt.zero_grad()
        pred = model(t_X_tr, t_nf_tr, t_nt_tr)
        loss = ((pred - t_y_tr) ** 2).mean()
        loss.backward()
        opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(t_X_va, t_nf_va, t_nt_va)
            val_loss = ((val_pred - t_y_va) ** 2).mean().item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 40:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(t_X_te, t_nf_te, t_nt_te).cpu().numpy()

    metrics = {
        f"{p}_r2": float(r2_score(y_te[:, i], test_pred[:, i]))
        for i, p in enumerate(PROPS)
    }
    metrics["avg_r2"] = float(np.mean(list(metrics.values())))
    return metrics, test_pred


def main():
    data = build_features()
    print(f"Features built: X_tr={data['X_tr'].shape}  X_te={data['X_te'].shape}")
    print(f"Unique training compounds: {data['n_unique']}")

    # Reference baselines
    v4_te = 0.4 * np.load(
        PROJECT / "cosmobridge_v4/data/cached_test.npz", allow_pickle=True
    )["preds_fusion"] + 0.6 * np.load(
        PROJECT / "cosmobridge_v4/data/cached_test.npz", allow_pickle=True
    )["preds_chemprop"]
    v4_metrics = {
        f"{p}_r2": float(r2_score(data["y_te"][:, i], v4_te[:, i]))
        for i, p in enumerate(PROPS)
    }
    v4_metrics["avg_r2"] = float(np.mean(list(v4_metrics.values())))
    print(f"v4 blend test avg R²: {v4_metrics['avg_r2']:.4f}")

    all_preds = []
    seed_results = []
    for seed in range(10):
        m, pred = train_seed(data, seed)
        seed_results.append(m)
        all_preds.append(pred)
        print(f"  seed {seed}: avg_r2={m['avg_r2']:.4f}")

    # Ensemble by averaging predictions
    mean_pred = np.mean(all_preds, axis=0)
    ensemble = {
        f"{p}_r2": float(r2_score(data["y_te"][:, i], mean_pred[:, i]))
        for i, p in enumerate(PROPS)
    }
    ensemble["avg_r2"] = float(np.mean(list(ensemble.values())))
    avgs = [m["avg_r2"] for m in seed_results]

    print(f"\n=== RACA ensemble (10 seeds): avg R² = {ensemble['avg_r2']:.4f} "
          f"(seed std {np.std(avgs):.4f}) ===")
    print(f"v4 blend:      {v4_metrics['avg_r2']:.4f}")
    print(f"hybrid ceiling: 0.8320")
    print()
    print("Per-property (RACA ensemble):")
    for p in PROPS:
        print(f"  {p:8s}: {ensemble[f'{p}_r2']:+.4f}  (v4={v4_metrics[f'{p}_r2']:+.4f})")

    out = V5 / "results/perprop_dft/raca.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "method": "retrieval_augmented_cross_attention",
            "k_neighbors": K_NEIGHBORS,
            "n_seeds": len(seed_results),
            "ensemble": ensemble,
            "per_seed": seed_results,
            "v4_blend": v4_metrics,
        }, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
