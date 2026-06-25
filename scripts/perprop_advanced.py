"""Advanced PerPropHead variants for beating 0.8320:

  --frame-level-train : Idea β — expand training data 36× by using
                         individual frames; infer via frame-average.
  --tta-eval          : Idea ζ — train on mean-pooled features (as in
                         the hybrid recipe) but evaluate by averaging
                         predictions across the 36 rotation frames.
  --film              : Idea α — replace PCA on V-JEPA streams with a
                         temperature-conditioned FiLM projection:
                             y = γ(T) ⊙ MLP(features) + β(T)
                         joint-trained with the PerPropHead.

All three variants reuse the archived PerPropHead architecture and the
same v4 blend baseline (0.4·fusion + 0.6·chemprop for train, seed
ensemble for test). Per-frame features must exist under
  cosmobridge_v5/data/cached_image_features_{split}_{source}_perframe.npz.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parent.parent
V5 = PROJECT / "cosmobridge_v5"
PROPS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]
N_FRAMES = 36


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def metrics(pred, targ):
    out = {}
    for i, n in enumerate(PROPS):
        sr = ((targ[:, i] - pred[:, i]) ** 2).sum()
        st = ((targ[:, i] - targ[:, i].mean()) ** 2).sum()
        out[f"{n}_r2"] = float(1 - sr / (st + 1e-8))
    out["avg_r2"] = float(np.mean([out[f"{n}_r2"] for n in PROPS]))
    return out


def load_perframe(source, split):
    return np.load(
        V5 / f"data/cached_image_features_{split}_{source}_perframe.npz"
    )["vit_feat"].astype(np.float32)


class PerPropHead(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(5, 32), nn.GELU(), nn.Linear(32, nf), nn.Sigmoid()
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(nf + 5, 32), nn.GELU(), nn.Linear(32, 1))
            for _ in range(7)
        ])
        self.alphas = nn.Parameter(torch.full((7,), -3.0))
        for h in self.heads:
            with torch.no_grad():
                h[-1].weight.mul_(0.01)
                h[-1].bias.zero_()

    def forward(self, v, i, t):
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        inp = torch.cat([g, tmp], dim=-1)
        res = torch.cat([h(inp) for h in self.heads], dim=-1)
        return v + torch.sigmoid(self.alphas) * res


class FiLMProjection(nn.Module):
    """Temperature-conditioned projection for a single V-JEPA stream.

    Maps (features: 192, T_feat: 5) -> out_dim via
        hidden = ReLU(Linear(features))
        γ, β   = Linear(T_feat) split
        out    = γ ⊙ hidden + β
    """
    def __init__(self, in_dim=192, out_dim=20, hidden=64):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                   nn.Linear(hidden, out_dim))
        self.film = nn.Linear(5, 2 * out_dim)

    def forward(self, x, t):
        h = self.proj(x)  # (B, out_dim)
        gb = self.film(t[:, :5])  # (B, 2*out_dim)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma * h + beta


class FiLMPerPropHead(nn.Module):
    """PerPropHead with FiLM projections for each V-JEPA stream (instead of
    PCA). Supervised stream still uses frozen PCA features."""
    def __init__(self, n_vjepa_streams=2, vjepa_dim=192, sup_pca_dim=20,
                  film_dim=20):
        super().__init__()
        self.film_layers = nn.ModuleList([
            FiLMProjection(vjepa_dim, film_dim) for _ in range(n_vjepa_streams)
        ])
        total = film_dim * n_vjepa_streams + sup_pca_dim
        self.head = PerPropHead(total)

    def forward(self, vjepa_list, sup_pca, t):
        vjepa_proj = [layer(v, t) for layer, v in zip(self.film_layers, vjepa_list)]
        i = torch.cat(vjepa_proj + [sup_pca], dim=-1)
        # reuse PerPropHead gate/head: pass a dummy "v" (base predictions handled
        # outside), and pass i as the feature vector.
        return self.head(
            torch.zeros(i.shape[0], 7, device=i.device),  # dummy base
            i, t,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frame-level", "tta", "film"], required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--tag", type=str, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Mode: {args.mode}")

    # Shared data loading
    tc = np.load(PROJECT / "cosmobridge_v4/data/cached_train.npz", allow_pickle=True)
    tsc = np.load(PROJECT / "cosmobridge_v4/data/cached_test.npz", allow_pickle=True)

    sup = np.load(V5 / "data/supervised_vit_features.npz")["features"]
    sup_tr_raw = sup[:152].astype(np.float32)
    sup_te_raw = sup[152 + 32:].astype(np.float32)
    pca_sup = PCA(20).fit(sup_tr_raw)
    sup_tr = pca_sup.transform(sup_tr_raw).astype(np.float32)
    sup_te = pca_sup.transform(sup_te_raw).astype(np.float32)

    # Per-frame V-JEPA features (N, 36, 192)
    gast_tr = load_perframe("gasteiger", "train")
    gast_te = load_perframe("gasteiger", "test")
    dft_tr = load_perframe("dft", "train")
    dft_te = load_perframe("dft", "test")

    # Mean-pooled versions
    gast_tr_mean = gast_tr.mean(axis=1)
    gast_te_mean = gast_te.mean(axis=1)
    dft_tr_mean = dft_tr.mean(axis=1)
    dft_te_mean = dft_te.mean(axis=1)

    # v4 baseline
    pdir = PROJECT / "cosmobridge_v4/results/seed_predictions"
    sf = sorted(pdir.glob("seed_*.npz"))
    if sf:
        v4p = np.mean([
            np.load(f)["preds" if "preds" in np.load(f).files else "predictions"]
            for f in sf
        ], axis=0).astype(np.float32)
    else:
        v4p = tsc["preds_fusion"].astype(np.float32)
    v4_tr = (0.4 * tc["preds_fusion"] + 0.6 * tc["preds_chemprop"]).astype(np.float32)
    tr_th = tc["thermo_feat"].astype(np.float32)
    te_th = tsc["thermo_feat"].astype(np.float32)
    tr_tgt = tc["targets"].astype(np.float32)
    te_tgt = tsc["targets"].astype(np.float32)

    mv4 = metrics(v4p, te_tgt)
    print(f"v4 router baseline: avg_r2={mv4['avg_r2']:.4f}")

    seed_results = []

    for seed in range(args.seeds):
        set_seed(seed)

        if args.mode in ("tta", "frame-level"):
            # Same PCA-based feature path as the hybrid recipe
            pca_g = PCA(20).fit(gast_tr_mean)
            pca_d = PCA(20).fit(dft_tr_mean)

            if args.mode == "tta":
                # Train on mean-pooled features, evaluate per-frame
                g_tr_p = pca_g.transform(gast_tr_mean).astype(np.float32)
                d_tr_p = pca_d.transform(dft_tr_mean).astype(np.float32)
                trf = np.concatenate([g_tr_p, d_tr_p, sup_tr], axis=1)
                tr_tgt_used = tr_tgt
                tr_th_used = tr_th
                v4_tr_used = v4_tr

            else:  # frame-level: expand training data 36x
                g_tr_frames_p = np.stack([
                    pca_g.transform(gast_tr[:, f]).astype(np.float32)
                    for f in range(N_FRAMES)
                ], axis=1)  # (N, 36, 20)
                d_tr_frames_p = np.stack([
                    pca_d.transform(dft_tr[:, f]).astype(np.float32)
                    for f in range(N_FRAMES)
                ], axis=1)  # (N, 36, 20)
                # Expand: broadcast sup/target/thermo/v4 across frames
                n = len(tr_tgt)
                expanded_feats = np.concatenate([
                    g_tr_frames_p.reshape(n * N_FRAMES, 20),
                    d_tr_frames_p.reshape(n * N_FRAMES, 20),
                    np.broadcast_to(sup_tr[:, None, :], (n, N_FRAMES, 20)).reshape(n * N_FRAMES, 20),
                ], axis=1).astype(np.float32)
                trf = expanded_feats
                tr_tgt_used = np.broadcast_to(tr_tgt[:, None, :], (n, N_FRAMES, 7)).reshape(n * N_FRAMES, 7).astype(np.float32)
                tr_th_used = np.broadcast_to(tr_th[:, None, :], (n, N_FRAMES, 25)).reshape(n * N_FRAMES, 25).astype(np.float32)
                v4_tr_used = np.broadcast_to(v4_tr[:, None, :], (n, N_FRAMES, 7)).reshape(n * N_FRAMES, 7).astype(np.float32)

            model = PerPropHead(trf.shape[1]).to(device)
            opt = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
            sch = CosineAnnealingLR(opt, T_max=args.epochs)
            dl = DataLoader(
                TensorDataset(
                    torch.from_numpy(v4_tr_used), torch.from_numpy(trf),
                    torch.from_numpy(tr_th_used), torch.from_numpy(tr_tgt_used),
                ),
                batch_size=32, shuffle=True,
            )
            best, best_state, patience = float("inf"), None, 0
            for ep in range(args.epochs):
                model.train()
                for v, i, t, y in dl:
                    v, i, t, y = [x.to(device) for x in (v, i, t, y)]
                    l = ((model(v, i, t) - y) ** 2).mean()
                    opt.zero_grad()
                    l.backward()
                    opt.step()
                sch.step()
                model.eval()
                with torch.no_grad():
                    # Use mean-pooled training features for train-loss-based early stop
                    g_tr_m_p = pca_g.transform(gast_tr_mean).astype(np.float32)
                    d_tr_m_p = pca_d.transform(dft_tr_mean).astype(np.float32)
                    trf_mean = np.concatenate([g_tr_m_p, d_tr_m_p, sup_tr], axis=1).astype(np.float32)
                    tl = ((
                        model(
                            torch.from_numpy(v4_tr).to(device),
                            torch.from_numpy(trf_mean).to(device),
                            torch.from_numpy(tr_th).to(device),
                        ) - torch.from_numpy(tr_tgt).to(device)
                    ) ** 2).mean().item()
                if tl < best:
                    best, best_state, patience = tl, {k: v.clone() for k, v in model.state_dict().items()}, 0
                else:
                    patience += 1
                    if patience >= 50:
                        break
            model.load_state_dict(best_state)
            model.eval()

            # Evaluate: average predictions across 36 frames per test sample
            with torch.no_grad():
                frame_preds = []
                for f in range(N_FRAMES):
                    g_te_p = pca_g.transform(gast_te[:, f]).astype(np.float32)
                    d_te_p = pca_d.transform(dft_te[:, f]).astype(np.float32)
                    tef = np.concatenate([g_te_p, d_te_p, sup_te], axis=1).astype(np.float32)
                    pred = model(
                        torch.from_numpy(v4p).to(device),
                        torch.from_numpy(tef).to(device),
                        torch.from_numpy(te_th).to(device),
                    ).cpu().numpy()
                    frame_preds.append(pred)
                pred_mean = np.mean(np.stack(frame_preds, axis=0), axis=0)

        else:  # FiLM
            model = FiLMPerPropHead(n_vjepa_streams=2, vjepa_dim=192,
                                     sup_pca_dim=20, film_dim=20).to(device)
            opt = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
            sch = CosineAnnealingLR(opt, T_max=args.epochs)

            tr_g = torch.from_numpy(gast_tr_mean)
            tr_d = torch.from_numpy(dft_tr_mean)
            tr_s = torch.from_numpy(sup_tr)
            tr_v = torch.from_numpy(v4_tr)
            tr_t = torch.from_numpy(tr_th)
            tr_y = torch.from_numpy(tr_tgt)
            dl = DataLoader(
                TensorDataset(tr_v, tr_g, tr_d, tr_s, tr_t, tr_y),
                batch_size=32, shuffle=True,
            )
            best, best_state, patience = float("inf"), None, 0
            for ep in range(args.epochs):
                model.train()
                for v, g, d, s, t, y in dl:
                    v, g, d, s, t, y = [x.to(device) for x in (v, g, d, s, t, y)]
                    delta = model([g, d], s, t)
                    pred = v + delta
                    l = ((pred - y) ** 2).mean()
                    opt.zero_grad()
                    l.backward()
                    opt.step()
                sch.step()
                model.eval()
                with torch.no_grad():
                    delta = model([tr_g.to(device), tr_d.to(device)], tr_s.to(device), tr_t.to(device))
                    tl = ((tr_v.to(device) + delta - tr_y.to(device)) ** 2).mean().item()
                if tl < best:
                    best, best_state, patience = tl, {k: v.clone() for k, v in model.state_dict().items()}, 0
                else:
                    patience += 1
                    if patience >= 50:
                        break
            model.load_state_dict(best_state)
            model.eval()
            te_g = torch.from_numpy(gast_te_mean).to(device)
            te_d = torch.from_numpy(dft_te_mean).to(device)
            te_s = torch.from_numpy(sup_te).to(device)
            te_t = torch.from_numpy(te_th).to(device)
            te_v = torch.from_numpy(v4p).to(device)
            with torch.no_grad():
                delta = model([te_g, te_d], te_s, te_t)
                pred_mean = (te_v + delta).cpu().numpy()

        m = metrics(pred_mean, te_tgt)
        seed_results.append(m)
        print(f"  seed {seed}: avg_r2={m['avg_r2']:.4f}")

    avgs = [x["avg_r2"] for x in seed_results]
    print(f"\n=== {args.tag} ensemble: avg R² = {np.mean(avgs):.4f} ± {np.std(avgs):.4f} ===")
    print(f"v4 baseline: {mv4['avg_r2']:.4f}  Δ = {np.mean(avgs) - mv4['avg_r2']:+.4f}")
    for prop in PROPS:
        vs = [x[f"{prop}_r2"] for x in seed_results]
        print(f"  {prop:8s}: {np.mean(vs):.4f} ± {np.std(vs):.4f} "
              f"(v4={mv4[f'{prop}_r2']:.4f} Δ={np.mean(vs) - mv4[f'{prop}_r2']:+.4f})")

    out_dir = V5 / "results/perprop_dft"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.tag}.json", "w") as f:
        json.dump({
            "tag": args.tag,
            "mode": args.mode,
            "seeds": args.seeds,
            "baseline_avg_r2": mv4["avg_r2"],
            "ensemble_avg_r2": float(np.mean(avgs)),
            "ensemble_std": float(np.std(avgs)),
            "per_seed": seed_results,
        }, f, indent=2)
    print(f"\nSaved to {out_dir / (args.tag + '.json')}")


if __name__ == "__main__":
    main()
