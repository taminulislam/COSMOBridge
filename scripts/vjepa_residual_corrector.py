"""Idea 4: V-JEPA residual corrector on v4 outputs.

Pipeline:
    1. For every unique SMILES in the v4 cached splits, extract a
       192-D V-JEPA CLS embedding by averaging the encoder's CLS output
       across the 36 rotation frames for that compound.
    2. Train a small MLP that maps the 192-D embedding to a 7-D
       residual (targets - preds_fusion) using the train split.
    3. Evaluate the corrected prediction (preds_fusion + MLP output)
       against targets on the test split.

Runs as a single script on one GPU. No distributed training, no multi-seed
ensemble — the goal is a fast signal on whether the DFT V-JEPA features
can close any of v4's residual on the seven thermo properties.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V5_ROOT = PROJECT_ROOT / "cosmobridge_v5"
sys.path.insert(0, str(V5_ROOT))

from models.multiview_vit import PatchEmbedding, ViTBlock  # noqa: E402


CACHED_DIR = PROJECT_ROOT / "cosmobridge_v4" / "data"
VJEPA_CKPT = V5_ROOT / "checkpoints" / "vjepa" / "vit_pretrained_vjepa.pt"
FRAMES_DIR_V5 = V5_ROOT / "data" / "cosmo_images"
FRAMES_DIR_PIPELINE = PROJECT_ROOT / "data" / "pipeline" / "cosmo_images"
RESULTS_DIR = V5_ROOT / "results" / "vjepa_residual"
EMBED_DIM = 192
FEATURE_DIMS = {"vjepa": 192, "chemprop": 300, "thermo": 25, "surface": 256}


class ViTTinyEncoder(nn.Module):
    """Mirror of pretrain_vjepa.ViTTinyEncoder so we can load the checkpoint."""

    def __init__(self, embed_dim=192, img_size=224, patch_size=16,
                 n_layers=6, n_heads=3, mlp_ratio=4, dropout=0.1,
                 stochastic_depth=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, n_layers)]
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, n_heads, mlp_ratio, dropout, dpr[i])
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        patches = self.patch_embed(x)
        B = patches.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = self.pos_dropout(
            torch.cat([cls, patches], dim=1) + self.pos_embed
        )
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens[:, 0])


def smi_hash(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


def find_frames(smiles):
    """Return a sorted list of frame PNG paths for this compound or None."""
    h = smi_hash(str(smiles))
    d = FRAMES_DIR_V5 / f"{h}_frames"
    if d.exists():
        frames = sorted(d.glob("frame_*.png"))
        if len(frames) >= 4:
            return frames
    # Fallback: compound_id-keyed DFT dir (from 2026-04-10 render)
    for p in FRAMES_DIR_PIPELINE.glob("*_frames"):
        if smi_hash(str(smiles)) in str(p):
            frames = sorted(p.glob("frame_*.png"))
            if len(frames) >= 4:
                return frames
    return None


def load_frames_tensor(frame_paths, transform):
    imgs = []
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(transform(img))
    return torch.stack(imgs)  # (N, 3, H, W)


def extract_cls_embeddings(encoder, device):
    """Compute a 192-D embedding per unique SMILES across all three splits."""
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    all_smiles = set()
    for split in ("train", "val", "test"):
        c = np.load(CACHED_DIR / f"cached_{split}.npz", allow_pickle=True)
        for s in c["smiles"]:
            all_smiles.add(str(s))

    smiles_list = sorted(all_smiles)
    print(f"Extracting V-JEPA CLS for {len(smiles_list)} unique compounds...")

    emb = {}
    n_hit = 0
    n_miss = 0
    with torch.no_grad():
        for i, s in enumerate(smiles_list):
            frames = find_frames(s)
            if frames is None:
                n_miss += 1
                emb[s] = np.zeros(EMBED_DIM, dtype=np.float32)
                continue
            views = load_frames_tensor(frames, tfm).to(device)
            cls = encoder(views)  # (N, 192)
            emb[s] = cls.mean(dim=0).cpu().numpy().astype(np.float32)
            n_hit += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(smiles_list)}] hit={n_hit} miss={n_miss}")

    print(f"  CLS extraction complete: {n_hit} hit, {n_miss} miss")
    return emb


def build_matrices(embed_dict, split, features=("vjepa",)):
    """Stack per-sample (feature-vector, residual, v4-pred, target) for a split.

    Parameters
    ----------
    features : tuple of str
        Any subset of {"vjepa","chemprop","thermo","surface"}; the output
        feature vector is the concatenation in the order given.
    """
    c = np.load(CACHED_DIR / f"cached_{split}.npz", allow_pickle=True)
    smiles = [str(s) for s in c["smiles"]]
    targets = c["targets"]
    v4_pred = c["preds_fusion"]
    residual = targets - v4_pred

    parts = []
    for feat in features:
        if feat == "vjepa":
            parts.append(np.stack([embed_dict[s] for s in smiles]))
        elif feat == "chemprop":
            parts.append(c["chemprop_fp"].astype(np.float32))
        elif feat == "thermo":
            parts.append(c["thermo_feat"].astype(np.float32))
        elif feat == "surface":
            parts.append(c["surface_fp"].astype(np.float32))
        else:
            raise ValueError(f"unknown feature: {feat}")
    X = np.concatenate(parts, axis=1).astype(np.float32)
    return X, residual.astype(np.float32), v4_pred, targets


class ResidualMLP(nn.Module):
    def __init__(self, in_dim=192, hidden=128, out_dim=7, dropout=0.5):
        super().__init__()
        # Width scales with input dimension so the 517-D variant still has a
        # reasonable bottleneck without blowing up parameter count.
        h = max(hidden, min(in_dim // 2, 256))
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_corrector(X_tr, r_tr, X_val, r_val, device,
                    epochs=300, lr=3e-4, weight_decay=1e-2, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ResidualMLP(in_dim=X_tr.shape[1]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    X_tr_t = torch.from_numpy(X_tr).to(device)
    r_tr_t = torch.from_numpy(r_tr).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    r_val_t = torch.from_numpy(r_val).to(device)
    best_val = float("inf")
    best_state = None
    for ep in range(epochs):
        model.train()
        optim.zero_grad()
        pred = model(X_tr_t)
        loss = F.mse_loss(pred, r_tr_t)
        loss.backward()
        optim.step()
        sched.step()
        if (ep + 1) % 20 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_loss = F.mse_loss(model(X_val_t), r_val_t).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def eval_residual(model, X, v4_pred, targets, device):
    model.eval()
    with torch.no_grad():
        delta = model(torch.from_numpy(X).to(device)).cpu().numpy()
    corrected = v4_pred + delta
    prop_names = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]
    baseline = {n: r2_score(targets[:, i], v4_pred[:, i]) for i, n in enumerate(prop_names)}
    corrected_r2 = {n: r2_score(targets[:, i], corrected[:, i]) for i, n in enumerate(prop_names)}
    return baseline, corrected_r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--features", type=str, default="vjepa",
                        help="Comma-separated subset of {vjepa,chemprop,thermo,surface}")
    parser.add_argument("--tag", type=str, default=None,
                        help="Filename tag for metrics output (defaults to feature list).")
    parser.add_argument("--skip-properties", type=str, default="",
                        help="Comma-separated property names to skip (use v4 as-is).")
    parser.add_argument("--per-property", action="store_true",
                        help="Train 7 separate MLPs, one per target property.")
    args = parser.parse_args()

    prop_names = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]
    skip = set(p.strip() for p in args.skip_properties.split(",") if p.strip())
    skip_mask = np.array([1.0 if p not in skip else 0.0 for p in prop_names],
                         dtype=np.float32)
    if skip:
        print(f"Skipping residual correction for: {sorted(skip)}")

    features = tuple(args.features.split(","))
    tag = args.tag or "_".join(features)
    results_out = RESULTS_DIR / f"metrics_{tag}.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Features: {features}  (tag={tag})")

    # V-JEPA CLS extraction is only needed if "vjepa" is in the feature list.
    if "vjepa" in features:
        encoder = ViTTinyEncoder().to(device)
        ckpt = torch.load(VJEPA_CKPT, map_location=device, weights_only=False)
        state = ckpt.get("encoder_state_dict", ckpt)
        encoder.load_state_dict(state, strict=True)
        encoder.eval()
        print(f"V-JEPA encoder loaded from {VJEPA_CKPT.name}")
        emb_dict = extract_cls_embeddings(encoder, device)
    else:
        emb_dict = {}

    X_tr, r_tr, v4_tr, y_tr = build_matrices(emb_dict, "train", features)
    X_val, r_val, v4_val, y_val = build_matrices(emb_dict, "val", features)
    X_te, r_te, v4_te, y_te = build_matrices(emb_dict, "test", features)
    print(f"Input dim: {X_tr.shape[1]}")

    baseline = {n: r2_score(y_te[:, i], v4_te[:, i]) for i, n in enumerate(prop_names)}
    print(f"\nv4 baseline (test): avg R²={np.mean(list(baseline.values())):.4f}")

    # Train 10 seeds, average their residual corrections.
    # In --per-property mode, train a separate MLP for each target column.
    all_deltas = []
    for seed in range(args.seeds):
        if args.per_property:
            delta_cols = []
            for k in range(7):
                r_tr_k = r_tr[:, k:k + 1]
                r_val_k = r_val[:, k:k + 1]

                class _Single(nn.Module):
                    def __init__(self, d):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.5),
                            nn.Linear(64, 1),
                        )

                    def forward(self, x):
                        return self.net(x)

                torch.manual_seed(seed * 100 + k)
                m = _Single(X_tr.shape[1]).to(device)
                opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-2)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
                X_tr_t = torch.from_numpy(X_tr).to(device)
                r_tr_kt = torch.from_numpy(r_tr_k).to(device)
                X_val_t = torch.from_numpy(X_val).to(device)
                r_val_kt = torch.from_numpy(r_val_k).to(device)
                best_val = float("inf")
                best_state = None
                for ep in range(args.epochs):
                    m.train()
                    opt.zero_grad()
                    loss = F.mse_loss(m(X_tr_t), r_tr_kt)
                    loss.backward()
                    opt.step()
                    sch.step()
                    if (ep + 1) % 20 == 0:
                        m.eval()
                        with torch.no_grad():
                            v = F.mse_loss(m(X_val_t), r_val_kt).item()
                        if v < best_val:
                            best_val = v
                            best_state = {kk: vv.detach().cpu().clone() for kk, vv in m.state_dict().items()}
                m.load_state_dict(best_state)
                m.eval()
                with torch.no_grad():
                    delta_cols.append(m(torch.from_numpy(X_te).to(device)).cpu().numpy())
            delta = np.concatenate(delta_cols, axis=1)
        else:
            model = train_corrector(
                X_tr, r_tr, X_val, r_val, device,
                epochs=args.epochs, seed=seed
            )
            model.eval()
            with torch.no_grad():
                delta = model(torch.from_numpy(X_te).to(device)).cpu().numpy()

        # Zero out skipped properties so v4 predictions are kept as-is.
        delta = delta * skip_mask[None, :]
        all_deltas.append(delta)
        per_seed_corrected = v4_te + delta
        avg = np.mean([r2_score(y_te[:, i], per_seed_corrected[:, i]) for i in range(7)])
        print(f"  seed {seed}: corrected test avg R²={avg:.4f}")

    # Ensemble: average residual predictions across seeds
    mean_delta = np.mean(all_deltas, axis=0)
    corrected = v4_te + mean_delta
    corrected_r2 = {n: r2_score(y_te[:, i], corrected[:, i]) for i, n in enumerate(prop_names)}
    avg_corrected = float(np.mean(list(corrected_r2.values())))
    avg_baseline = float(np.mean(list(baseline.values())))

    print(f"\n=== ENSEMBLE RESULT ({args.seeds} seeds) ===")
    print(f"{'property':10s} {'v4 base':>10s} {'+residual':>10s} {'delta':>10s}")
    for n in prop_names:
        print(f"{n:10s} {baseline[n]:10.4f} {corrected_r2[n]:10.4f} "
              f"{corrected_r2[n]-baseline[n]:+10.4f}")
    print(f"{'avg':10s} {avg_baseline:10.4f} {avg_corrected:10.4f} "
          f"{avg_corrected-avg_baseline:+10.4f}")

    results_out.parent.mkdir(parents=True, exist_ok=True)
    with open(results_out, "w") as f:
        json.dump({
            "features": list(features),
            "input_dim": int(X_tr.shape[1]),
            "baseline": baseline,
            "corrected": corrected_r2,
            "avg_baseline": avg_baseline,
            "avg_corrected": avg_corrected,
            "seeds": args.seeds,
        }, f, indent=2)
    print(f"\nSaved to {results_out}")


if __name__ == "__main__":
    main()
