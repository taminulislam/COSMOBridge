"""MoE-A + Fix5: Aggressive gamma1 filtering (mean +/- 2*std).

Uses merged_v5 where ILThermo gamma1 is filtered to match original's
distribution. With aligned distributions, the unified scaler works
correctly and the model sees consistent gamma1 signals from both sources.

No domain conditioning or separate heads needed.
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
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.moe import ExpertHead, PropertyConditionedGating
from src.models.fusion.multimodal_pointcloud import PointCloudFusion
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_joint import MergedDataset, collate_merged

TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


class MoEA(nn.Module):
    def __init__(self, feature_dim, fused_dim=256, dropout=0.3, pretrained_gnn_path=None):
        super().__init__()
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
        self.fusion = PointCloudFusion(
            pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
            fused_dim=fused_dim, num_heads=8, dropout=dropout)
        self.experts = nn.ModuleList([
            ExpertHead(fused_dim, hidden_dim=128, num_targets=7, dropout=dropout)
            for _ in range(4)])
        self.gating = PropertyConditionedGating(
            input_dim=fused_dim, num_experts=4, num_properties=7, hidden_dim=64)

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        pc_feat = self.pointnet(point_cloud)
        graph_feat = self.gnn.get_features(atom_features, edge_index, bond_features, batch)
        fused = self.fusion(pc_feat, graph_feat, features)
        expert_preds = torch.stack([e(fused) for e in self.experts], dim=2)
        gate_weights, lb_loss = self.gating(fused)
        predictions = (expert_preds * gate_weights).sum(dim=2)
        return predictions, {"load_balance_loss": lb_loss, "gate_weights": gate_weights.detach()}


def evaluate_single(model, loader, device):
    model.eval()
    all_preds, all_targets, all_gw = [], [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            preds, aux = model(
                point_cloud=batch["point_cloud"], features=batch["features"],
                atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                bond_features=batch["bond_features"], batch=batch["batch"])
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())
            all_gw.append(aux["gate_weights"].cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_gw)


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    merged_dir = Path("data/merged_v5")
    if not merged_dir.exists():
        print("ERROR: Run create_merged_dataset_v5.py first")
        return

    meta = json.load(open(merged_dir / "metadata.json"))
    merged_features = meta["feature_columns"]
    pc_dir = "data/pipeline/point_clouds"

    # ── Load data ──
    print(f"\n{'='*60}")
    print("MoE-A + Fix5: Aggressive Gamma1 Filter (mean +/- 2*std)")
    print(f"{'='*60}")

    train_ds = MergedDataset(str(merged_dir / "splits/train.csv"), pc_dir, merged_features, is_train=True)
    val_ds = MergedDataset(str(merged_dir / "splits/val.csv"), pc_dir, merged_features, is_train=False)
    test_ds = MergedDataset(str(merged_dir / "splits/test.csv"), pc_dir, merged_features, is_train=False)

    train_ldr = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_merged)
    val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)

    model = MoEA(feature_dim=len(merged_features),
                 pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
    model.to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_dir = Path("checkpoints/moe_fix5")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=60, eta_min=1e-6)

    # ── Train ──
    best_loss, no_improve = float("inf"), 0
    for epoch in range(200):
        model.train()
        tl, n = 0, 0
        for batch in train_ldr:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            optimizer.zero_grad()
            preds, aux = model(point_cloud=batch["point_cloud"], features=batch["features"],
                               atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                               bond_features=batch["bond_features"], batch=batch["batch"])
            mask = ~torch.isnan(batch["targets"])
            safe = batch["targets"].clone(); safe[~mask] = 0.0
            loss = ((preds - safe)**2 * mask.float()).sum() / mask.float().sum().clamp(min=1)
            loss = loss + aux["load_balance_loss"]
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
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                preds, _ = model(point_cloud=batch["point_cloud"], features=batch["features"],
                                 atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                                 bond_features=batch["bond_features"], batch=batch["batch"])
                mask = ~torch.isnan(batch["targets"])
                safe = batch["targets"].clone(); safe[~mask] = 0.0
                vl += ((preds - safe)**2 * mask.float()).sum().item() / mask.float().sum().clamp(min=1).item()
                vn += 1
        avg_val = vl / max(vn, 1)
        if avg_val < best_loss:
            best_loss = avg_val; no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_improve += 1
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/200 | Train: {tl/max(n,1):.4f} | "
                  f"Val: {avg_val:.4f} | Best: {best_loss:.4f} | Pat: {no_improve}/25")
        if no_improve >= 25:
            print(f"  Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True))

    # ── Evaluate ──
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")

    preds, targets, gate_weights = evaluate_single(model, test_ldr, device)
    metrics = compute_metrics(preds, targets)
    print(format_metrics(metrics, "MoE-A Fix5 (aggressive gamma1 filter)"))

    # Per-property
    print(f"\n  Per-property R²:")
    for p in TARGET_COLUMNS:
        print(f"    {p:15s} R² = {metrics[f'{p}_r2']:.4f}")
    print(f"    {'AVERAGE':15s} R² = {metrics['avg_r2']:.4f}")

    # Compare with all previous fixes
    print(f"\n{'='*60}")
    print("COMPARISON WITH ALL FIXES")
    print(f"{'='*60}")

    prev = {}
    for fix_name, path in [("Fix2", "results/moe_fix2_results.json"),
                            ("Fix3", "results/moe_fix3_results.json"),
                            ("Fix4", "results/moe_fix4_results.json"),
                            ("PointCloud", "results/pointcloud_results.json")]:
        try:
            data = json.load(open(path))
            m = data.get("metrics", data.get("metrics_normalized",
                  data.get("hybrid", data.get("test_metrics", data.get("single_model", {})))))
            prev[fix_name] = m
        except Exception:
            pass

    header = f"  {'Property':15s}"
    for name in prev:
        header += f" {name:>10s}"
    header += f" {'Fix5':>10s}"
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = f"  {p:15s}"
        for name in prev:
            v = prev[name].get(key, float("nan"))
            line += f" {v:10.4f}"
        line += f" {metrics[key]:10.4f}"
        print(line)

    line = f"  {'AVERAGE':15s}"
    for name in prev:
        v = prev[name].get("avg_r2", float("nan"))
        line += f" {v:10.4f}"
    line += f" {metrics['avg_r2']:10.4f}"
    print(line)

    # ── Save ──
    results = {
        "fix": "aggressive_gamma1_filter_2std",
        "description": "ILThermo gamma1 filtered to original mean+/-2*std before unified normalization",
        "gamma1_filter": meta["gamma1_filter"],
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items()},
        "gate_weights": gate_weights.mean(axis=0).tolist(),
    }
    with open("results/moe_fix5_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/moe_fix5_results.json")


if __name__ == "__main__":
    main()
