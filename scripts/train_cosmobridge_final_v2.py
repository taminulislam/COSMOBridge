"""COSMOBridge Final v2: Pre-cached fingerprints + pre-loaded fusion.

Fixes from v1:
1. Pre-cache Chemprop fingerprints → batch_size=32, no online overhead
2. Pre-load CP-GBH Hybrid fusion weights → gamma1=0.908 guaranteed
3. Only train direct path + gates (fusion frozen from CP-GBH)
"""

import sys
import json
import subprocess
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.models.fusion.cosmobridge import COSMOBridge
from src.models.fusion.multimodal_pointcloud import MultimodalPointCloudModel
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud

# Chemprop
from chemprop.utils import load_checkpoint as load_chemprop


class CachedDataset(Dataset):
    def __init__(self, g, s, t, y):
        self.g = torch.tensor(g, dtype=torch.float32)
        self.s = torch.tensor(s, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.g[i], self.s[i], self.t[i], self.y[i]


def identity_collate(b): return b


def extract_all_features(device):
    """Extract and cache all features from both frozen encoders."""
    pc_dir = "data/pipeline/point_clouds"
    orig_splits = Path("data/processed/splits")

    # PointNet
    config = load_config("configs/default.yaml")
    pc_model = MultimodalPointCloudModel(config=config, pretrained_gnn_path=None)
    ckpt = torch.load("checkpoints/pointcloud/best_model.pt", map_location=device, weights_only=False)
    pc_model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    pc_model.to(device).eval()

    # Chemprop fingerprints via CLI (fastest, uses their optimized code)
    data = {}
    for split in ["train", "val", "test"]:
        # PointNet features
        ds = PointCloudMultimodalDataset(str(orig_splits / f"{split}.csv"), pc_dir, is_train=False)
        sf, tf, tgt = [], [], []
        with torch.no_grad():
            for items in DataLoader(ds, batch_size=32, shuffle=False, collate_fn=identity_collate):
                pcs = torch.stack([x["point_cloud"] for x in items]).to(device)
                sf.append(pc_model.pointnet(pcs).cpu().numpy())
                tf.append(torch.stack([x["features"] for x in items]).numpy())
                tgt.append(torch.stack([x["targets"] for x in items]).numpy())

        # Chemprop fingerprints
        out = tempfile.mktemp(suffix=".csv")
        subprocess.run(["chemprop_fingerprint",
                         "--test_path", f"data/chemprop_tmp/{split}.csv",
                         "--features_path", f"data/chemprop_tmp/{split}_features.csv",
                         "--checkpoint_dir", "checkpoints/chemprop",
                         "--fingerprint_type", "MPN",
                         "--preds_path", out],
                        capture_output=True, text=True, timeout=120)
        gf = pd.read_csv(out).select_dtypes(include=[np.number]).values.astype(np.float32)

        data[split] = {
            "g": gf, "s": np.concatenate(sf),
            "t": np.concatenate(tf), "y": np.concatenate(tgt)
        }
        print(f"  {split}: graph={gf.shape} surface={data[split]['s'].shape}")

    return data


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    # ══════════════════════════════════════════════════════════
    print("Step 1: Cache all encoder features...\n")
    data = extract_all_features(device)
    graph_dim = data["train"]["g"].shape[1]

    train_ds = CachedDataset(data["train"]["g"], data["train"]["s"], data["train"]["t"], data["train"]["y"])
    val_ds = CachedDataset(data["val"]["g"], data["val"]["s"], data["val"]["t"], data["val"]["y"])
    test_ds = CachedDataset(data["test"]["g"], data["test"]["s"], data["test"]["t"], data["test"]["y"])
    train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False)

    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 2: Build COSMOBridge with pre-loaded fusion")
    print(f"{'='*60}")

    model = COSMOBridge(graph_dim=graph_dim, surface_dim=256, thermo_dim=len(FEATURE_COLUMNS),
                         fused_dim=256, rank=32, hyper_hidden=64, dropout=0.3)
    model.to(device)

    # Pre-load CP-GBH Hybrid fusion weights
    from scripts.train_chemprop_gbh_hybrid import ChempropGBHFusion
    cpgbh = ChempropGBHFusion(graph_dim=graph_dim, surface_dim=256,
                                thermo_dim=len(FEATURE_COLUMNS), fused_dim=256,
                                rank=32, hyper_hidden=64, dropout=0.3)
    cpgbh.load_state_dict(torch.load("checkpoints/chemprop_gbh_hybrid/best.pt",
                                       map_location=device, weights_only=True))

    # Transfer fusion weights
    my_sd = model.state_dict()
    loaded = 0
    for cpgbh_prefix, cosmo_prefix in [("graph_proj", "graph_proj"),
                                         ("surface_proj", "surface_proj"),
                                         ("fusion", "fusion"),
                                         ("prediction_head", "fused_head")]:
        for k, v in cpgbh.state_dict().items():
            if k.startswith(cpgbh_prefix + "."):
                new_k = k.replace(cpgbh_prefix + ".", cosmo_prefix + ".", 1)
                if new_k in my_sd and my_sd[new_k].shape == v.shape:
                    my_sd[new_k] = v; loaded += 1
    model.load_state_dict(my_sd)
    print(f"  Pre-loaded {loaded} fusion params from CP-GBH Hybrid")

    # Verify fusion works
    model.eval()
    with torch.no_grad():
        g = torch.tensor(data["test"]["g"], dtype=torch.float32).to(device)
        s = torch.tensor(data["test"]["s"], dtype=torch.float32).to(device)
        t = torch.tensor(data["test"]["t"], dtype=torch.float32).to(device)
        y = data["test"]["y"]
        preds_check, aux = model(g, s, t)
        fused_check = aux["preds_fused"].cpu().numpy()
        m_f = compute_metrics(fused_check, y)
        print(f"  Fusion verification: gamma1={m_f['gamma1_r2']:.4f} gamma2={m_f['gamma2_r2']:.4f} "
              f"P={m_f['P_r2']:.4f}")

    # FREEZE fusion path
    for name, p in model.named_parameters():
        if any(name.startswith(pf) for pf in ["graph_proj.", "surface_proj.", "fusion.", "fused_head."]):
            p.requires_grad = False

    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Frozen (fusion): {n_frozen:,}")
    print(f"  Trainable (direct + gates): {n_train:,}")

    ckpt_dir = Path("checkpoints/cosmobridge_final_v2")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 3: Train direct path only")
    print(f"{'='*60}")

    model.gate_logits.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

    best, no_imp = float("inf"), 0
    for ep in range(200):
        model.train()
        tl, n = 0, 0
        for g, s, t, y in train_ldr:
            g, s, t, y = g.to(device), s.to(device), t.to(device), y.to(device)
            optimizer.zero_grad()
            preds, _ = model(g, s, t)
            loss = ((preds - y)**2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            tl += loss.item(); n += 1
        scheduler.step()

        model.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for g, s, t, y in val_ldr:
                g, s, t, y = g.to(device), s.to(device), t.to(device), y.to(device)
                preds, _ = model(g, s, t)
                vl += ((preds - y)**2).mean().item(); vn += 1
        avg = vl / max(vn, 1)
        if avg < best: best = avg; no_imp = 0; torch.save(model.state_dict(), ckpt_dir / "s2.pt")
        else: no_imp += 1
        if ep % 30 == 0:
            print(f"    Ep {ep:3d} | T:{tl/max(n,1):.4f} V:{avg:.4f} B:{best:.4f} P:{no_imp}/30")
        if no_imp >= 30: print(f"    Early stop ep {ep}"); break
    model.load_state_dict(torch.load(ckpt_dir / "s2.pt", map_location=device, weights_only=True))

    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 4: Train gates only")
    print(f"{'='*60}")

    for p in model.direct_head.parameters(): p.requires_grad = False
    model.gate_logits.requires_grad = True

    optimizer2 = AdamW([model.gate_logits], lr=0.1)
    best2, no_imp2 = float("inf"), 0
    for ep in range(100):
        model.train()
        tl, n = 0, 0
        for g, s, t, y in train_ldr:
            g, s, t, y = g.to(device), s.to(device), t.to(device), y.to(device)
            optimizer2.zero_grad()
            preds, _ = model(g, s, t)
            loss = ((preds - y)**2).mean()
            loss.backward()
            optimizer2.step()
            tl += loss.item(); n += 1

        model.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for g, s, t, y in val_ldr:
                g, s, t, y = g.to(device), s.to(device), t.to(device), y.to(device)
                preds, _ = model(g, s, t)
                vl += ((preds - y)**2).mean().item(); vn += 1
        avg = vl / max(vn, 1)
        if avg < best2: best2 = avg; no_imp2 = 0; torch.save(model.state_dict(), ckpt_dir / "s3.pt")
        else: no_imp2 += 1
        if ep % 20 == 0:
            gates = torch.sigmoid(model.gate_logits).detach().cpu().numpy()
            print(f"    Ep {ep:3d} | T:{tl/max(n,1):.4f} V:{avg:.4f} B:{best2:.4f} P:{no_imp2}/30 "
                  f"| [{' '.join(f'{x:.2f}' for x in gates)}]")
        if no_imp2 >= 30: print(f"    Early stop ep {ep}"); break
    model.load_state_dict(torch.load(ckpt_dir / "s3.pt", map_location=device, weights_only=True))

    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")

    model.eval()
    all_p, all_t, all_f, all_d = [], [], [], []
    with torch.no_grad():
        for g, s, t, y in test_ldr:
            g, s, t, y = g.to(device), s.to(device), t.to(device), y.to(device)
            preds, aux = model(g, s, t)
            all_p.append(preds.cpu().numpy()); all_t.append(y.cpu().numpy())
            all_f.append(aux["preds_fused"].cpu().numpy())
            all_d.append(aux["preds_direct"].cpu().numpy())

    preds = np.concatenate(all_p); targets = np.concatenate(all_t)
    preds_f = np.concatenate(all_f); preds_d = np.concatenate(all_d)

    metrics = compute_metrics(preds, targets)
    mf = compute_metrics(preds_f, targets)
    md = compute_metrics(preds_d, targets)

    print(format_metrics(metrics, "COSMOBridge Final v2"))

    gates = torch.sigmoid(model.gate_logits).detach().cpu().numpy()
    print(f"\n  Per-property routing:")
    for i, p in enumerate(TARGET_COLUMNS):
        path = "FUSION" if gates[i] > 0.6 else ("DIRECT" if gates[i] < 0.4 else "MIXED")
        print(f"    {p:15s}: α={gates[i]:.3f} {path:>6s}  "
              f"fused={mf[f'{p}_r2']:.3f}  direct={md[f'{p}_r2']:.3f}  → routed={metrics[f'{p}_r2']:.3f}")

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    prev = {}
    for name, path, key in [
        ("Chemprop", "results/chemprop_results.json", "test_metrics"),
        ("STILT", "results/chemprop_tuned_results.json", "STILT_C"),
        ("3-Model Router", "results/per_property_router_results.json", "metrics"),
        ("CP-GBH Hybrid", "results/chemprop_gbh_hybrid_results.json", "metrics"),
        ("Final v1", "results/cosmobridge_final_results.json", "metrics"),
    ]:
        try:
            data = json.load(open(path))
            if key == "STILT_C": m = data.get("C: full mask, 48x OS", {}).get("metrics", {})
            elif key: m = data[key]
            prev[name] = m
        except: pass

    header = f"  {'Property':<15s}"
    for n in prev: header += f" {n[:14]:>14s}"
    header += f" {'Final v2':>14s}"
    print(header)
    print("  " + "-" * len(header))
    for p in TARGET_COLUMNS:
        line = f"  {p:<15s}"
        for n in prev: line += f" {prev[n].get(f'{p}_r2', 0):14.4f}"
        line += f" {metrics[f'{p}_r2']:14.4f}"
        print(line)
    line = f"  {'AVERAGE':<15s}"
    for n in prev: line += f" {prev[n].get('avg_r2', 0):14.4f}"
    line += f" {metrics['avg_r2']:14.4f}"
    print(line)

    # vs Chemprop
    base = prev.get("Chemprop", {})
    wins = sum(1 for p in TARGET_COLUMNS if metrics[f"{p}_r2"] > base.get(f"{p}_r2", 0))
    d = metrics['avg_r2'] - base.get('avg_r2', 0)
    print(f"\n  vs Chemprop: {metrics['avg_r2']:.4f} vs {base.get('avg_r2',0):.4f} "
          f"({'+' if d>0 else ''}{d:.4f}) wins {wins}/7")

    # Save
    results = {
        "model": "COSMOBridge_Final_v2",
        "description": "Embedded Chemprop (cached fingerprints) + pre-loaded CP-GBH fusion (frozen) "
                       "+ trainable direct FFN + learned gates. True single model.",
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items()},
        "gate_values": {p: float(g) for p, g in zip(TARGET_COLUMNS, gates)},
    }
    with open("results/cosmobridge_final_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/cosmobridge_final_v2_results.json")


if __name__ == "__main__":
    main()
