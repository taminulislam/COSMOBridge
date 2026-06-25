"""Train COSMOBridge: single-model with built-in per-property routing.

Extracts frozen features from Chemprop + PointNet, then trains:
  - GBH bilinear fusion path (for γ₁, γ₂, P)
  - Direct graph FFN path (for G_E, H_E, G_mix)
  - 7 per-property gates (learned routing)
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.models.fusion.cosmobridge import COSMOBridge
from src.models.fusion.multimodal_pointcloud import MultimodalPointCloudModel
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_pointcloud import PointCloudMultimodalDataset


class PrecomputedDataset(Dataset):
    def __init__(self, graph_feats, surface_feats, thermo_feats, targets):
        self.g = torch.tensor(graph_feats, dtype=torch.float32)
        self.s = torch.tensor(surface_feats, dtype=torch.float32)
        self.t = torch.tensor(thermo_feats, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return self.g[idx], self.s[idx], self.t[idx], self.y[idx]


def identity_collate(batch_list):
    return batch_list


def extract_pointnet_features(device, split_path, pc_dir):
    """Extract frozen PointNet features."""
    config = load_config("configs/default.yaml")
    model = MultimodalPointCloudModel(config=config, pretrained_gnn_path=None)
    ckpt = torch.load("checkpoints/pointcloud/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(device).eval()

    ds = PointCloudMultimodalDataset(str(split_path), pc_dir, is_train=False)
    sf_list, tf_list, tgt_list = [], [], []
    with torch.no_grad():
        for items in DataLoader(ds, batch_size=32, shuffle=False, collate_fn=identity_collate):
            pcs = torch.stack([x["point_cloud"] for x in items]).to(device)
            feats = torch.stack([x["features"] for x in items])
            tgts = torch.stack([x["targets"] for x in items])
            sf_list.append(model.pointnet(pcs).cpu().numpy())
            tf_list.append(feats.numpy())
            tgt_list.append(tgts.numpy())
    return np.concatenate(sf_list), np.concatenate(tf_list), np.concatenate(tgt_list)


def extract_chemprop_features(split):
    """Extract frozen Chemprop fingerprints."""
    out_path = tempfile.mktemp(suffix=".csv")
    cmd = ["chemprop_fingerprint",
           "--test_path", f"data/chemprop_tmp/{split}.csv",
           "--features_path", f"data/chemprop_tmp/{split}_features.csv",
           "--checkpoint_dir", "checkpoints/chemprop",
           "--fingerprint_type", "MPN",
           "--preds_path", out_path]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return pd.read_csv(out_path).select_dtypes(include=[np.number]).values.astype(np.float32)


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    pc_dir = "data/pipeline/point_clouds"
    orig_splits = Path("data/processed/splits")

    # ══════════════════════════════════════════════════════════
    print("Step 1: Extracting frozen encoder features...\n")
    # ══════════════════════════════════════════════════════════

    data = {}
    for split in ["train", "val", "test"]:
        print(f"  {split}:")
        sf, tf, tgt = extract_pointnet_features(device, orig_splits / f"{split}.csv", pc_dir)
        gf = extract_chemprop_features(split)
        data[split] = {"graph": gf, "surface": sf, "thermo": tf, "targets": tgt}
        print(f"    Graph: {gf.shape}, Surface: {sf.shape}, Thermo: {tf.shape}")

    graph_dim = data["train"]["graph"].shape[1]

    # Build datasets
    train_ds = PrecomputedDataset(data["train"]["graph"], data["train"]["surface"],
                                   data["train"]["thermo"], data["train"]["targets"])
    val_ds = PrecomputedDataset(data["val"]["graph"], data["val"]["surface"],
                                 data["val"]["thermo"], data["val"]["targets"])
    test_ds = PrecomputedDataset(data["test"]["graph"], data["test"]["surface"],
                                  data["test"]["thermo"], data["test"]["targets"])

    train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False)

    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 2: Training COSMOBridge (dual-path + per-property gates)")
    print(f"{'='*60}")
    # ══════════════════════════════════════════════════════════

    model = COSMOBridge(graph_dim=graph_dim, surface_dim=256, thermo_dim=len(FEATURE_COLUMNS),
                         fused_dim=256, rank=32, hyper_hidden=64, dropout=0.3)
    model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_fusion = sum(p.numel() for p in model.fusion.parameters()) + \
               sum(p.numel() for p in model.graph_proj.parameters()) + \
               sum(p.numel() for p in model.surface_proj.parameters()) + \
               sum(p.numel() for p in model.fused_head.parameters())
    n_direct = sum(p.numel() for p in model.direct_head.parameters())

    print(f"  Total params: {n_total:,}")
    print(f"  Fusion path: {n_fusion:,}")
    print(f"  Direct path: {n_direct:,}")
    print(f"  Gates: 7")
    gates = torch.sigmoid(model.gate_logits).detach().cpu().numpy()
    print(f"  Initial gates: {' '.join(f'{g:.2f}' for g in gates)}")
    print(f"  (γ₁={gates[0]:.2f}→fusion, G_E={gates[2]:.2f}→direct, H_vap={gates[5]:.2f}→mixed)")

    ckpt_dir = Path("checkpoints/cosmobridge")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, eta_min=1e-5)

    best_loss, no_improve = float("inf"), 0
    for epoch in range(300):
        model.train()
        tl, n = 0, 0
        for g, s, t, y in train_ldr:
            g, s, t, y = g.to(device), s.to(device), t.to(device), y.to(device)
            optimizer.zero_grad()
            preds, _ = model(g, s, t)
            loss = ((preds - y)**2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        avg_val = vl / max(vn, 1)

        if avg_val < best_loss:
            best_loss = avg_val; no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_improve += 1

        if epoch % 30 == 0:
            gates = torch.sigmoid(model.gate_logits).detach().cpu().numpy()
            g_str = " ".join(f"{g:.2f}" for g in gates)
            print(f"  Epoch {epoch:3d}/300 | Train:{tl/max(n,1):.4f} Val:{avg_val:.4f} "
                  f"Best:{best_loss:.4f} Pat:{no_improve}/40 | Gates:[{g_str}]")
        if no_improve >= 40:
            print(f"  Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True))

    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    # ══════════════════════════════════════════════════════════

    model.eval()
    all_preds, all_tgts = [], []
    all_fused, all_direct = [], []
    with torch.no_grad():
        for g, s, t, y in test_ldr:
            g, s, t, y = g.to(device), s.to(device), t.to(device), y.to(device)
            preds, aux = model(g, s, t)
            all_preds.append(preds.cpu().numpy())
            all_tgts.append(y.cpu().numpy())
            all_fused.append(aux["preds_fused"].cpu().numpy())
            all_direct.append(aux["preds_direct"].cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_tgts)
    preds_fused = np.concatenate(all_fused)
    preds_direct = np.concatenate(all_direct)

    metrics = compute_metrics(preds, targets)
    metrics_fused = compute_metrics(preds_fused, targets)
    metrics_direct = compute_metrics(preds_direct, targets)

    print(format_metrics(metrics, "COSMOBridge (routed)"))

    # Gate analysis
    gates = torch.sigmoid(model.gate_logits).detach().cpu().numpy()
    print(f"\n  Learned per-property routing (α: 1=fusion, 0=direct):")
    for i, p in enumerate(TARGET_COLUMNS):
        path = "FUSION" if gates[i] > 0.6 else ("DIRECT" if gates[i] < 0.4 else "MIXED")
        bar_f = "█" * int(gates[i] * 20)
        bar_d = "░" * (20 - int(gates[i] * 20))
        r2_f = metrics_fused.get(f"{p}_r2", 0)
        r2_d = metrics_direct.get(f"{p}_r2", 0)
        r2_r = metrics.get(f"{p}_r2", 0)
        print(f"    {p:15s}: α={gates[i]:.3f} [{bar_f}{bar_d}] {path:>6s}  "
              f"(fused={r2_f:.3f} direct={r2_d:.3f} routed={r2_r:.3f})")

    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    prev = {}
    for name, path, key in [
        ("Chemprop", "results/chemprop_results.json", "test_metrics"),
        ("STILT", "results/chemprop_tuned_results.json", "STILT_C"),
        ("Ens Top-2", "results/ensemble_all_models_results.json", "ENS"),
        ("3-Model Router", "results/per_property_router_results.json", "metrics"),
        ("CP-GBH (no route)", "results/chemprop_gbh_hybrid_results.json", "metrics"),
    ]:
        try:
            data = json.load(open(path))
            if key == "STILT_C": m = data.get("C: full mask, 48x OS", {}).get("metrics", {})
            elif key == "ENS": m = data.get("top2_average", {}).get("metrics", {})
            elif key: m = data[key]
            prev[name] = m
        except: pass

    header = "  {:<15s}".format("Property")
    for name in prev:
        header += " {:>14s}".format(name[:14])
    header += " {:>14s}".format("COSMOBridge")
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<15s}".format(p)
        for name in prev:
            line += " {:14.4f}".format(prev[name].get(key, float('nan')))
        line += " {:14.4f}".format(metrics[key])
        print(line)

    line = "  {:<15s}".format("AVERAGE")
    for name in prev:
        line += " {:14.4f}".format(prev[name].get('avg_r2', float('nan')))
    line += " {:14.4f}".format(metrics['avg_r2'])
    print(line)

    # vs Chemprop
    base = prev.get("Chemprop", {})
    if base:
        print(f"\n  vs Chemprop:")
        wins = 0
        for p in TARGET_COLUMNS:
            key = f"{p}_r2"
            d = metrics[key] - base[key]
            if d > 0: wins += 1
            s = "+" if d > 0 else ""
            w = "WIN" if d > 0 else ("~tied" if abs(d) < 0.01 else "lose")
            print(f"    {p:15s}: {metrics[key]:.4f} vs {base[key]:.4f} ({s}{d:.4f}) {w}")
        d = metrics['avg_r2'] - base['avg_r2']
        s = "+" if d > 0 else ""
        print(f"    {'AVERAGE':15s}: {metrics['avg_r2']:.4f} vs {base['avg_r2']:.4f} ({s}{d:.4f}) wins {wins}/7")

    # Save
    results = {
        "model": "COSMOBridge",
        "description": "Single multimodal model with dual-path (GBH bilinear fusion + "
                       "direct graph FFN) and learned per-property routing gates. "
                       "Frozen Chemprop D-MPNN (300D) + frozen PointNet COSMO (256D).",
        "n_total_params": n_total,
        "n_fusion_path": n_fusion,
        "n_direct_path": n_direct,
        "gate_values": {p: float(g) for p, g in zip(TARGET_COLUMNS, gates)},
        "metrics_routed": {k: float(v) if isinstance(v, (float, np.floating)) else v
                           for k, v in metrics.items()},
        "metrics_fusion_only": {k: float(v) if isinstance(v, (float, np.floating)) else v
                                for k, v in metrics_fused.items()},
        "metrics_direct_only": {k: float(v) if isinstance(v, (float, np.floating)) else v
                                for k, v in metrics_direct.items()},
    }
    with open("results/cosmobridge_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/cosmobridge_results.json")


if __name__ == "__main__":
    main()
