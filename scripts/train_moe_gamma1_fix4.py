"""MoE-A + Fix4: Source-specific target scalers.

Root cause fix: The unified scaler in merged_v3 compressed original data
into a different statistical space (mean=-0.4, std=0.35) while ILThermo
dominated (96.7% of data). Both sources now have mean~0, std~1 independently,
so the model sees consistent distributions from both sources.

Uses merged_v4 dataset where each source is normalized with its own scaler.
No domain conditioning needed — the normalization fix removes the distribution
mismatch that caused gamma1 to fail.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
    """Standard MoE-A (no domain conditioning needed with proper normalization)."""

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
    """Evaluate single model."""
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

    merged_dir = Path("data/merged_v4")
    if not merged_dir.exists():
        print("ERROR: Run create_merged_dataset_v4.py first")
        return

    meta = json.load(open(merged_dir / "metadata.json"))
    merged_features = meta["feature_columns"]
    pc_dir = "data/pipeline/point_clouds"

    # Load scalers for reporting
    with open(merged_dir / "target_scalers_original.pkl", "rb") as f:
        scalers_orig = pickle.load(f)
    with open(merged_dir / "target_scalers_ilthermo.pkl", "rb") as f:
        scalers_ilth = pickle.load(f)

    print(f"\nScaler comparison for gamma1:")
    if "gamma1" in scalers_orig:
        print(f"  Original: mean={scalers_orig['gamma1'].mean_[0]:.4f}, "
              f"std={scalers_orig['gamma1'].scale_[0]:.4f}")
    if "gamma1" in scalers_ilth:
        print(f"  ILThermo: mean={scalers_ilth['gamma1'].mean_[0]:.4f}, "
              f"std={scalers_ilth['gamma1'].scale_[0]:.4f}")

    # ── Load data ──
    print(f"\n{'='*60}")
    print("MoE-A + Fix4: Source-Specific Scalers")
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

    ckpt_dir = Path("checkpoints/moe_fix4")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

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
    print("EVALUATION (test set = original domain, normalized with original scaler)")
    print(f"{'='*60}")

    preds, targets, gate_weights = evaluate_single(model, test_ldr, device)

    # Metrics in normalized space (consistent with how model was trained)
    metrics_norm = compute_metrics(preds, targets)
    print("\n[Normalized space]")
    print(format_metrics(metrics_norm, "MoE-A Fix4 (source-specific scalers)"))

    # Inverse-transform to original scale using original scaler for interpretability
    preds_raw = preds.copy()
    targets_raw = targets.copy()
    for i, col in enumerate(TARGET_COLUMNS):
        if col in scalers_orig:
            sc = scalers_orig[col]
            valid = ~np.isnan(targets[:, i])
            preds_raw[valid, i] = preds[valid, i] * sc.scale_[0] + sc.mean_[0]
            targets_raw[valid, i] = targets[valid, i] * sc.scale_[0] + sc.mean_[0]

    metrics_raw = compute_metrics(preds_raw, targets_raw)
    print("\n[Original scale (inverse-transformed)]")
    print(format_metrics(metrics_raw, "MoE-A Fix4 (original units)"))

    # Per-property summary
    print(f"\n  Per-property R² (normalized):")
    for p in TARGET_COLUMNS:
        print(f"    {p:15s} R² = {metrics_norm[f'{p}_r2']:.4f}")
    print(f"    {'AVERAGE':15s} R² = {metrics_norm['avg_r2']:.4f}")

    # Compare with fix2 and fix3
    print(f"\n{'='*60}")
    print("COMPARISON WITH PREVIOUS FIXES")
    print(f"{'='*60}")
    print(f"  {'Property':15s} {'Fix2':>8s} {'Fix3':>8s} {'Fix4':>8s}")
    fix2_path = Path("results/moe_fix2_results.json")
    fix3_path = Path("results/moe_fix3_results.json")
    fix2, fix3 = {}, {}
    if fix2_path.exists():
        data = json.load(open(fix2_path))
        fix2 = data.get("hybrid", data.get("after_ft_orig", {}))
    if fix3_path.exists():
        data = json.load(open(fix3_path))
        fix3 = data.get("metrics", {})
    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        f2 = fix2.get(key, float("nan"))
        f3 = fix3.get(key, float("nan"))
        f4 = metrics_norm.get(key, float("nan"))
        print(f"  {p:15s} {f2:8.4f} {f3:8.4f} {f4:8.4f}")
    f2_avg = fix2.get("avg_r2", float("nan"))
    f3_avg = fix3.get("avg_r2", float("nan"))
    print(f"  {'AVERAGE':15s} {f2_avg:8.4f} {f3_avg:8.4f} {metrics_norm['avg_r2']:8.4f}")

    # ── Save results ──
    results = {
        "fix": "source_specific_scalers",
        "description": "Each data source normalized with its own target scaler",
        "metrics_normalized": {k: float(v) if isinstance(v, (float, np.floating)) else v
                               for k, v in metrics_norm.items()},
        "metrics_original_scale": {k: float(v) if isinstance(v, (float, np.floating)) else v
                                   for k, v in metrics_raw.items()},
        "gate_weights": gate_weights.mean(axis=0).tolist(),
    }
    with open("results/moe_fix4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/moe_fix4_results.json")


if __name__ == "__main__":
    main()
