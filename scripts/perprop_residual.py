"""Reproduce the 0.831 PerPropHead recipe with pluggable V-JEPA features.

Flags:
    --vjepa-source {gasteiger,dft}    which V-JEPA cache to use
    --vjepa-pca INT                   PCA dim for V-JEPA (0 = no PCA)
    --add-surface                     add PCA(surface_fp, 20) as third stream
    --tag STR                         output filename tag

The recipe (from cosmobridge_v5/scripts/slurm_combined_sigma.sh):
    v4_base  = 0.4 * preds_fusion + 0.6 * preds_chemprop   (train)
    v4_base  = mean of cosmobridge_v4/results/seed_predictions (test)
    features = concat(PCA(V-JEPA), PCA(Supervised_ViT) [, PCA(surface_fp)])
    head     = PerPropHead (temperature-gated, 7 heads, sigmoid-alpha)
    10 seeds, AdamW 5e-4, 300 epochs, early stop patience 50
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

PROJECT = Path(__file__).resolve().parent.parent
V5 = PROJECT / "cosmobridge_v5"
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


def load_vjepa(source):
    if source == "dft":
        suffix = "_dft"
    elif source == "gasteiger":
        suffix = ""
    else:
        raise ValueError(source)
    tr = np.load(V5 / f"data/cached_image_features_train{suffix}.npz")["vit_feat"]
    te = np.load(V5 / f"data/cached_image_features_test{suffix}.npz")["vit_feat"]
    return tr.astype(np.float32), te.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vjepa-source", choices=["gasteiger", "dft"], default="dft")
    ap.add_argument("--vjepa-pca", type=int, default=20,
                    help="PCA dim for V-JEPA (0 = no PCA, use raw 192)")
    ap.add_argument("--hybrid-vjepa", action="store_true",
                    help="Use BOTH Gasteiger + DFT V-JEPA, each PCA'd separately")
    ap.add_argument("--hybrid-pca-each", type=int, default=10,
                    help="Per-encoder PCA dim in hybrid mode (so total = 2*this)")
    ap.add_argument("--add-dinov2", action="store_true",
                    help="Add PCA(DINOv2, N) as an extra image stream")
    ap.add_argument("--dinov2-pca", type=int, default=20)
    ap.add_argument("--add-chemberta", action="store_true",
                    help="Add PCA(ChemBERTa, N) as an extra SMILES-text stream")
    ap.add_argument("--chemberta-pca", type=int, default=20)
    ap.add_argument("--drop-sup", action="store_true",
                    help="Omit the Supervised ViT stream entirely (useful for "
                         "replacement experiments where chemberta takes its role)")
    ap.add_argument("--base-source", choices=["v4_blend", "stronger"],
                    default="v4_blend",
                    help="Base predictions for residual correction. 'v4_blend' "
                         "is 0.4·fusion+0.6·chemprop; 'stronger' loads from "
                         "cosmobridge_v5/data/stronger_base_preds.npz.")
    ap.add_argument("--sup-pca", type=int, default=20)
    ap.add_argument("--add-surface", action="store_true")
    ap.add_argument("--surface-pca", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--tag", type=str, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"V-JEPA source: {args.vjepa_source}  V-JEPA PCA: {args.vjepa_pca}  "
          f"surface stream: {args.add_surface}")

    tc = np.load(PROJECT / "cosmobridge_v4/data/cached_train.npz", allow_pickle=True)
    tsc = np.load(PROJECT / "cosmobridge_v4/data/cached_test.npz", allow_pickle=True)

    # V-JEPA streams (one or two depending on --hybrid-vjepa)
    vjepa_streams_tr = []
    vjepa_streams_te = []
    if args.hybrid_vjepa:
        for src in ("gasteiger", "dft"):
            tr, te = load_vjepa(src)
            pca = PCA(args.hybrid_pca_each).fit(tr)
            vjepa_streams_tr.append(pca.transform(tr).astype(np.float32))
            vjepa_streams_te.append(pca.transform(te).astype(np.float32))
    else:
        tr, te = load_vjepa(args.vjepa_source)
        if args.vjepa_pca > 0:
            pca = PCA(args.vjepa_pca).fit(tr)
            vjepa_streams_tr.append(pca.transform(tr).astype(np.float32))
            vjepa_streams_te.append(pca.transform(te).astype(np.float32))
        else:
            vjepa_streams_tr.append(tr.astype(np.float32))
            vjepa_streams_te.append(te.astype(np.float32))

    parts_tr = list(vjepa_streams_tr)
    parts_te = list(vjepa_streams_te)

    if not args.drop_sup:
        sup = np.load(V5 / "data/supervised_vit_features.npz")["features"]
        sup_tr = sup[:152].astype(np.float32)
        sup_te = sup[152 + 32:].astype(np.float32)
        pca_sup = PCA(args.sup_pca).fit(sup_tr)
        parts_tr.append(pca_sup.transform(sup_tr).astype(np.float32))
        parts_te.append(pca_sup.transform(sup_te).astype(np.float32))

    if args.add_dinov2:
        dv_tr = np.load(V5 / "data/cached_image_features_train_dinov2.npz")["vit_feat"].astype(np.float32)
        dv_te = np.load(V5 / "data/cached_image_features_test_dinov2.npz")["vit_feat"].astype(np.float32)
        pca_dv = PCA(args.dinov2_pca).fit(dv_tr)
        parts_tr.append(pca_dv.transform(dv_tr).astype(np.float32))
        parts_te.append(pca_dv.transform(dv_te).astype(np.float32))

    if args.add_chemberta:
        cb_tr = np.load(V5 / "data/cached_image_features_train_chemberta.npz")["vit_feat"].astype(np.float32)
        cb_te = np.load(V5 / "data/cached_image_features_test_chemberta.npz")["vit_feat"].astype(np.float32)
        pca_cb = PCA(args.chemberta_pca).fit(cb_tr)
        parts_tr.append(pca_cb.transform(cb_tr).astype(np.float32))
        parts_te.append(pca_cb.transform(cb_te).astype(np.float32))

    if args.add_surface:
        s_tr = tc["surface_fp"].astype(np.float32)
        s_te = tsc["surface_fp"].astype(np.float32)
        pca_s = PCA(args.surface_pca).fit(s_tr)
        parts_tr.append(pca_s.transform(s_tr).astype(np.float32))
        parts_te.append(pca_s.transform(s_te).astype(np.float32))

    trf = np.concatenate(parts_tr, axis=1).astype(np.float32)
    tef = np.concatenate(parts_te, axis=1).astype(np.float32)
    print(f"Feature dim: {trf.shape[1]}")

    # Base predictions — either v4 blend or a stronger custom base (Phase #2)
    if args.base_source == "stronger":
        sb = np.load(V5 / "data/stronger_base_preds.npz", allow_pickle=True)
        v4_tr = sb["train"].astype(np.float32)
        v4p = sb["test"].astype(np.float32)
        print(f"  Stronger base loaded (source={sb['source']})")
    else:
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
        model = PerPropHead(trf.shape[1]).to(device)
        opt = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
        sched = CosineAnnealingLR(opt, T_max=args.epochs)
        dl = DataLoader(
            TensorDataset(
                torch.from_numpy(v4_tr), torch.from_numpy(trf),
                torch.from_numpy(tr_th), torch.from_numpy(tr_tgt),
            ),
            batch_size=32, shuffle=True,
        )
        best = float("inf")
        best_state = None
        patience = 0
        for ep in range(args.epochs):
            model.train()
            for v, i, t, y in dl:
                v, i, t, y = [x.to(device) for x in (v, i, t, y)]
                l = ((model(v, i, t) - y) ** 2).mean()
                opt.zero_grad()
                l.backward()
                opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                tl = ((
                    model(
                        torch.from_numpy(v4_tr).to(device),
                        torch.from_numpy(trf).to(device),
                        torch.from_numpy(tr_th).to(device),
                    )
                    - torch.from_numpy(tr_tgt).to(device)
                ) ** 2).mean().item()
            if tl < best:
                best = tl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 50:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred = model(
                torch.from_numpy(v4p).to(device),
                torch.from_numpy(tef).to(device),
                torch.from_numpy(te_th).to(device),
            ).cpu().numpy()
        m = metrics(pred, te_tgt)
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
            "vjepa_source": args.vjepa_source,
            "vjepa_pca": args.vjepa_pca,
            "add_surface": args.add_surface,
            "seeds": args.seeds,
            "baseline_avg_r2": mv4["avg_r2"],
            "ensemble_avg_r2": float(np.mean(avgs)),
            "ensemble_std": float(np.std(avgs)),
            "per_seed": seed_results,
        }, f, indent=2)
    print(f"\nSaved to {out_dir / (args.tag + '.json')}")


if __name__ == "__main__":
    main()
