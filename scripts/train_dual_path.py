"""Dual-Path Fusion: Cross-Attention + Low-Rank Bilinear.

Hybrid architecture targeting both gamma1 (cross-attention) and gamma2 (bilinear).
Uses per-property gates to route each property to its optimal fusion path.
Trained with STILT data strategy (gamma1 masked, 48x oversample).
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.dual_path_fusion import DualPathFusion
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_joint import MergedDataset, collate_merged


class DualPathModel(nn.Module):
    """PointCloud model with dual-path fusion (cross-attention + bilinear)."""

    def __init__(self, feature_dim=25, pretrained_gnn_path=None):
        super().__init__()
        dropout = 0.3

        self.pointnet = PointNetEncoder(in_channels=7, feature_dim=256, dropout=dropout)
        self.gnn = MolecularGNN(
            atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
            dropout=dropout, pooling="mean", num_targets=0)

        if pretrained_gnn_path and Path(pretrained_gnn_path).exists():
            ckpt = torch.load(pretrained_gnn_path, map_location="cpu", weights_only=True)
            gnn_state = {k: v for k, v in ckpt.items()
                         if any(k.startswith(p) for p in ["atom_projection", "convs", "batch_norms", "pool"])}
            if gnn_state:
                self.gnn.load_state_dict(gnn_state, strict=False)
                print(f"  Loaded pre-trained GNN: {len(gnn_state)} params")

        self.fusion = DualPathFusion(
            pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
            fused_dim=256, bilinear_rank=32, num_heads=8,
            n_properties=7, dropout=dropout)

        # Per-property prediction heads (shared base + per-prop output)
        self.shared_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.prop_heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(7)])

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        pc_feat = self.pointnet(point_cloud)
        graph_feat = self.gnn.get_features(atom_features, edge_index, bond_features, batch)

        h_fused, h_per_prop, gate_values = self.fusion(pc_feat, graph_feat, features)

        # Per-property prediction using routed representations
        preds = []
        for p in range(7):
            h_p = h_per_prop[:, :, p]  # (B, 256) — property-specific fused repr
            h_p = self.shared_head(h_p)
            preds.append(self.prop_heads[p](h_p))
        predictions = torch.cat(preds, dim=1)  # (B, 7)

        return predictions, {"gate_values": gate_values}


def prepare_stilt_data(merged_csv, output_path):
    df = pd.read_csv(merged_csv)
    orig = df[df["source"] == "original"].copy()
    ilth = df[df["source"] != "original"].copy()
    ilth["gamma1"] = np.nan
    orig_rep = pd.concat([orig] * 48, ignore_index=True)
    balanced = pd.concat([orig_rep, ilth], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    balanced.to_csv(output_path, index=False)
    print(f"  STILT data: {len(orig)}×48 + {len(ilth)} = {len(balanced)} samples")
    return len(balanced)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            preds, _ = model(point_cloud=batch["point_cloud"], features=batch["features"],
                             atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                             bond_features=batch["bond_features"], batch=batch["batch"])
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    pc_dir = "data/pipeline/point_clouds"
    merged_dir = Path("data/merged_v5")
    meta = json.load(open(merged_dir / "metadata.json"))
    features = meta["feature_columns"]

    # STILT data
    stilt_dir = Path("data/dual_path_stilt")
    stilt_dir.mkdir(parents=True, exist_ok=True)
    train_path = stilt_dir / "train.csv"
    if not train_path.exists():
        prepare_stilt_data(merged_dir / "splits/train.csv", train_path)

    train_ds = MergedDataset(str(train_path), pc_dir, features, is_train=True)
    val_ds = MergedDataset(str(merged_dir / "splits/val.csv"), pc_dir, features, is_train=False)
    test_ds = MergedDataset(str(merged_dir / "splits/test.csv"), pc_dir, features, is_train=False)

    train_ldr = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_merged)
    val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)

    # Build model
    print(f"\n{'='*60}")
    print("Dual-Path Fusion: Cross-Attention + Low-Rank Bilinear")
    print(f"{'='*60}")

    model = DualPathModel(feature_dim=len(features),
                           pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
    model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_fusion = sum(p.numel() for p in model.fusion.parameters())
    n_crossattn = sum(p.numel() for n, p in model.fusion.named_parameters()
                      if 'attn' in n or 'ln_pc' in n or 'ln_graph' in n or 'crossattn' in n)
    n_bilinear = sum(p.numel() for n, p in model.fusion.named_parameters()
                     if 'bilinear' in n)
    print(f"  Total params: {n_total:,}")
    print(f"  Fusion params: {n_fusion:,}")
    print(f"    Cross-attention path: ~{n_crossattn:,}")
    print(f"    Bilinear path: ~{n_bilinear:,}")
    print(f"    Per-property gates: 7")
    print(f"  Initial gates: gamma1→crossattn (α=0.73), gamma2→bilinear (α=0.27)")

    ckpt_dir = Path("checkpoints/dual_path")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=60, eta_min=1e-6)

    best_loss, no_improve = float("inf"), 0
    for epoch in range(200):
        model.train()
        tl, n = 0, 0
        for batch in train_ldr:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            optimizer.zero_grad()
            preds, aux = model(point_cloud=batch["point_cloud"], features=batch["features"],
                               atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                               bond_features=batch["bond_features"], batch=batch["batch"])
            mask = ~torch.isnan(batch["targets"])
            safe = batch["targets"].clone(); safe[~mask] = 0.0
            loss = ((preds - safe)**2 * mask.float()).sum() / mask.float().sum().clamp(min=1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item(); n += 1
        scheduler.step()

        model.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for batch in val_ldr:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor): batch[k] = v.to(device)
                preds, _ = model(point_cloud=batch["point_cloud"], features=batch["features"],
                                 atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                                 bond_features=batch["bond_features"], batch=batch["batch"])
                mask = ~torch.isnan(batch["targets"])
                safe = batch["targets"].clone(); safe[~mask] = 0.0
                vl += ((preds-safe)**2*mask.float()).sum().item()/mask.float().sum().clamp(min=1).item()
                vn += 1
        avg_val = vl / max(vn, 1)
        if avg_val < best_loss:
            best_loss = avg_val; no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_improve += 1
        if epoch % 20 == 0:
            gates = torch.sigmoid(model.fusion.gate_logits).detach().cpu().numpy()
            gate_str = " ".join(f"{g:.2f}" for g in gates)
            print(f"  Epoch {epoch:3d}/200 | Train:{tl/max(n,1):.4f} | Val:{avg_val:.4f} | "
                  f"Best:{best_loss:.4f} | Pat:{no_improve}/30 | Gates:[{gate_str}]")
        if no_improve >= 30:
            print(f"  Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True))

    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")

    preds, targets = evaluate(model, test_ldr, device)
    metrics = compute_metrics(preds, targets)
    print(format_metrics(metrics, "Dual-Path Fusion"))

    # Gate analysis
    gates = torch.sigmoid(model.fusion.gate_logits).detach().cpu().numpy()
    print(f"\n  Learned per-property gates (α: 1=cross-attn, 0=bilinear):")
    for i, p in enumerate(TARGET_COLUMNS):
        path = "cross-attn" if gates[i] > 0.6 else ("bilinear" if gates[i] < 0.4 else "mixed")
        bar_ca = "█" * int(gates[i] * 20)
        bar_bi = "░" * (20 - int(gates[i] * 20))
        print(f"    {p:15s}: α={gates[i]:.3f} [{bar_ca}{bar_bi}] → {path}")

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    prev = {}
    for name, path, key in [
        ("Chemprop", "results/chemprop_results.json", "test_metrics"),
        ("CrossAttn PC", "results/pointcloud_results.json", None),
        ("GBH v2+STILT", "results/gbh_v2_stilt_results.json", "metrics"),
        ("GBH v3", "results/gbh_v3_results.json", "metrics"),
        ("STILT", "results/chemprop_tuned_results.json", "STILT_C"),
    ]:
        try:
            data = json.load(open(path))
            if key == "STILT_C": m = data.get("C: full mask, 48x OS", {}).get("metrics", {})
            elif key: m = data[key]
            else:
                for k in ['metrics', 'test_metrics']:
                    if k in data: m = data[k]; break
            prev[name] = m
        except: pass

    header = "  {:<12s}".format("Property")
    for name in prev:
        header += " {:>12s}".format(name[:12])
    header += " {:>12s}".format("Dual-Path")
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<12s}".format(p)
        for name in prev:
            line += " {:12.4f}".format(prev[name].get(key, float('nan')))
        line += " {:12.4f}".format(metrics[key])
        print(line)

    line = "  {:<12s}".format("AVERAGE")
    for name in prev:
        line += " {:12.4f}".format(prev[name].get('avg_r2', float('nan')))
    line += " {:12.4f}".format(metrics['avg_r2'])
    print(line)

    # vs Chemprop
    if "Chemprop" in prev:
        base = prev["Chemprop"]
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
        "model": "dual_path_fusion",
        "description": "Cross-attention + low-rank bilinear in parallel with per-property gates. "
                       "STILT data (gamma1 masked, 48x oversample). GAT-GNN backbone.",
        "n_params": n_total,
        "gate_values": {p: float(g) for p, g in zip(TARGET_COLUMNS, gates)},
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items()},
    }
    with open("results/dual_path_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/dual_path_results.json")


if __name__ == "__main__":
    main()
