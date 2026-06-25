"""Phase #1 Step G: Run the hybrid PerPropHead on the expanded
(4637/32/39) cached_*_expanded.npz splits with NaN-masked loss.

The expansion adds ~4485 ilthermo samples with gamma1 (and a subset of
H_E) labels only. For these, the multi-property MSE loss is masked via
a per-sample target_mask so missing labels are ignored.

Outputs per-seed test metrics to
    cosmobridge_v5/results/perprop_dft/phase1_expanded_*.json
"""

import argparse
import json
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
CACHED_DIR = PROJECT / "cosmobridge_v4/data"
PROPS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


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


def load_expanded(split):
    c = np.load(CACHED_DIR / f"cached_{split}_expanded.npz", allow_pickle=True)
    return c


def load_vjepa_expanded(source, split):
    suffix = "_dft" if source == "dft" else ""
    return np.load(
        V5 / f"data/cached_image_features_{split}_expanded{suffix}.npz"
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


def build_features(vjepa_sources, sup_pca_dim):
    """Return (train, val, test) 2D feature matrices by loading the
    expanded V-JEPA streams, fitting PCA on training rows, and stacking
    with PCA(Supervised, 20)."""
    parts_tr, parts_va, parts_te = [], [], []
    for src in vjepa_sources:
        tr = load_vjepa_expanded(src, "train")
        va = load_vjepa_expanded(src, "val")
        te = load_vjepa_expanded(src, "test")
        pca = PCA(20).fit(tr)
        parts_tr.append(pca.transform(tr).astype(np.float32))
        parts_va.append(pca.transform(va).astype(np.float32))
        parts_te.append(pca.transform(te).astype(np.float32))

    # Supervised ViT features come from the archived 223-row file; we need
    # a slot for each expanded row. The archived file has 152 train + 32 val
    # + 39 test = 223 rows. For the 4485 new ilthermo rows we use the
    # per-column mean of the original 152 training rows as a conservative
    # fallback. This means the Supervised ViT stream contributes only to the
    # original-source rows during training and always to val/test (which are
    # all-original anyway).
    sup = np.load(V5 / "data/supervised_vit_features.npz")["features"]
    sup_tr_orig = sup[:152].astype(np.float32)
    sup_va = sup[152:152 + 32].astype(np.float32)
    sup_te = sup[152 + 32:].astype(np.float32)
    pca_sup = PCA(sup_pca_dim).fit(sup_tr_orig)

    # Build expanded Supervised stream
    train_c = load_expanded("train")
    n_train = len(train_c["source"])
    sup_tr_expanded_raw = np.zeros((n_train, sup_tr_orig.shape[1]), dtype=np.float32)
    orig_mask = train_c["source"] == "original"
    n_orig = int(orig_mask.sum())
    if n_orig == sup_tr_orig.shape[0]:
        sup_tr_expanded_raw[orig_mask] = sup_tr_orig
    else:
        # Safety: if row counts don't align, skip Supervised for originals too
        print(f"WARN: original train rows ({n_orig}) != supervised train rows "
              f"({sup_tr_orig.shape[0]}); using per-column mean everywhere")
        sup_tr_expanded_raw[:] = sup_tr_orig.mean(axis=0, keepdims=True)
    # Fill ilthermo rows with the per-column mean of the original train set
    sup_mean = sup_tr_orig.mean(axis=0)
    sup_tr_expanded_raw[~orig_mask] = sup_mean

    parts_tr.append(pca_sup.transform(sup_tr_expanded_raw).astype(np.float32))
    parts_va.append(pca_sup.transform(sup_va).astype(np.float32))
    parts_te.append(pca_sup.transform(sup_te).astype(np.float32))

    trf = np.concatenate(parts_tr, axis=1).astype(np.float32)
    vaf = np.concatenate(parts_va, axis=1).astype(np.float32)
    tef = np.concatenate(parts_te, axis=1).astype(np.float32)
    return trf, vaf, tef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--tag", type=str, default="phase1_expanded_hybrid")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tc = load_expanded("train")
    vc = load_expanded("val")
    sc = load_expanded("test")

    print(f"Train: {len(tc['source'])} rows  "
          f"({int((tc['source'] == 'original').sum())} original + "
          f"{int((tc['source'] == 'ilthermo').sum())} ilthermo)")
    print(f"Val:   {len(vc['source'])} rows")
    print(f"Test:  {len(sc['source'])} rows")

    # Feature matrix (hybrid G20 + D20 + S20 = 60-D)
    trf, vaf, tef = build_features(("gasteiger", "dft"), sup_pca_dim=20)
    print(f"Feature dim: {trf.shape[1]}")

    # Base predictions
    v4_tr = (0.4 * tc["preds_fusion"] + 0.6 * tc["preds_chemprop"]).astype(np.float32)
    v4_te = (0.4 * sc["preds_fusion"] + 0.6 * sc["preds_chemprop"]).astype(np.float32)
    tr_th = tc["thermo_feat"].astype(np.float32)
    te_th = sc["thermo_feat"].astype(np.float32)
    tr_tgt = tc["targets"].astype(np.float32)
    tr_mask = tc["target_mask"].astype(np.float32)
    te_tgt = sc["targets"].astype(np.float32)

    # Reference metrics on test
    mv4 = metrics(v4_te, te_tgt)
    print(f"\nv4 blend test avg R² (reference): {mv4['avg_r2']:.4f}")

    seed_results = []
    for seed in range(args.seeds):
        set_seed(seed)
        model = PerPropHead(trf.shape[1]).to(device)
        opt = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
        sched = CosineAnnealingLR(opt, T_max=args.epochs)
        ds = TensorDataset(
            torch.from_numpy(v4_tr), torch.from_numpy(trf),
            torch.from_numpy(tr_th), torch.from_numpy(tr_tgt),
            torch.from_numpy(tr_mask),
        )
        dl = DataLoader(ds, batch_size=64, shuffle=True)

        best, best_state, patience = float("inf"), None, 0
        for ep in range(args.epochs):
            model.train()
            for v, i, t, y, m in dl:
                v, i, t, y, m = [x.to(device) for x in (v, i, t, y, m)]
                pred = model(v, i, t)
                # Per-property masked MSE, averaged across properties.
                # This prevents gamma1 (many labels) from dominating the
                # shared representation over properties with fewer labels.
                diff = (pred - y) ** 2 * m  # (B, 7)
                # Per-prop mean over samples that have the label
                mask_sum = m.sum(dim=0).clamp_min(1.0)  # (7,)
                per_prop = diff.sum(dim=0) / mask_sum   # (7,)
                loss = per_prop.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                preds = model(
                    torch.from_numpy(v4_tr).to(device),
                    torch.from_numpy(trf).to(device),
                    torch.from_numpy(tr_th).to(device),
                )
                m_tensor = torch.from_numpy(tr_mask).to(device)
                y_tensor = torch.from_numpy(tr_tgt).to(device)
                # Per-property mean over samples with labels, then average
                # across properties (same loss used in training).
                diff_tr = (preds - y_tensor) ** 2 * m_tensor
                mask_sum_tr = m_tensor.sum(dim=0).clamp_min(1.0)
                tl = (diff_tr.sum(dim=0) / mask_sum_tr).mean().item()
            if tl < best:
                best, best_state, patience = tl, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 50:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred = model(
                torch.from_numpy(v4_te).to(device),
                torch.from_numpy(tef).to(device),
                torch.from_numpy(te_th).to(device),
            ).cpu().numpy()
        m = metrics(pred, te_tgt)
        seed_results.append(m)
        print(f"  seed {seed}: avg_r2={m['avg_r2']:.4f}")

    avgs = [x["avg_r2"] for x in seed_results]
    print(f"\n=== {args.tag} ensemble: avg R² = {np.mean(avgs):.4f} ± {np.std(avgs):.4f} ===")
    print(f"v4 blend reference: {mv4['avg_r2']:.4f}  Δ = {np.mean(avgs) - mv4['avg_r2']:+.4f}")
    for prop in PROPS:
        vs = [x[f"{prop}_r2"] for x in seed_results]
        print(f"  {prop:8s}: {np.mean(vs):.4f} ± {np.std(vs):.4f} "
              f"(v4={mv4[f'{prop}_r2']:.4f} Δ={np.mean(vs) - mv4[f'{prop}_r2']:+.4f})")

    out_dir = V5 / "results/perprop_dft"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.tag}.json", "w") as f:
        json.dump({
            "tag": args.tag,
            "seeds": args.seeds,
            "baseline_avg_r2": mv4["avg_r2"],
            "ensemble_avg_r2": float(np.mean(avgs)),
            "ensemble_std": float(np.std(avgs)),
            "per_seed": seed_results,
        }, f, indent=2)
    print(f"\nSaved to {out_dir / (args.tag + '.json')}")


if __name__ == "__main__":
    main()
