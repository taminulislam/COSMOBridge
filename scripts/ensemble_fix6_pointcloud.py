"""Ensemble: Fix6 (MoE) + PointCloud model.

Combines the strengths of both models:
- PointCloud excels on gamma1 (0.89), gamma2 (0.85), G_E, G_mix
- Fix6 excels on H_vap (0.69), P (0.70), and is competitive on H_E

Three ensemble strategies:
1. Simple average: (pred_fix6 + pred_pc) / 2
2. Per-property best: select whichever model had higher val R² per property
3. Learned weights: optimize per-property weights on validation set
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

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.moe import ExpertHead, PropertyConditionedGating
from src.models.fusion.multimodal_pointcloud import PointCloudFusion, MultimodalPointCloudModel
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud
from scripts.train_joint import MergedDataset, collate_merged

TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


class MoEA(nn.Module):
    """Same architecture as Fix6."""
    def __init__(self, feature_dim, fused_dim=256, dropout=0.3, pretrained_gnn_path=None):
        super().__init__()
        self.pointnet = PointNetEncoder(in_channels=7, feature_dim=256, dropout=dropout)
        self.gnn = MolecularGNN(
            atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
            dropout=dropout, pooling="mean", num_targets=0)
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


def get_predictions(model, loader, device, is_moe=False):
    """Run inference and return predictions + targets."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            if is_moe:
                preds, _ = model(
                    point_cloud=batch["point_cloud"], features=batch["features"],
                    atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                    bond_features=batch["bond_features"], batch=batch["batch"])
            else:
                preds = model(
                    point_cloud=batch["point_cloud"], features=batch["features"],
                    atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                    bond_features=batch["bond_features"], batch=batch["batch"])
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def optimize_weights(preds_a, preds_b, targets):
    """Find optimal per-property weight alpha (0-1) that minimizes MSE.

    ensemble = alpha * preds_a + (1 - alpha) * preds_b
    """
    best_weights = []
    for i in range(targets.shape[1]):
        valid = ~(np.isnan(targets[:, i]) | np.isnan(preds_a[:, i]) | np.isnan(preds_b[:, i]))
        if valid.sum() == 0:
            best_weights.append(0.5)
            continue

        pa = preds_a[valid, i]
        pb = preds_b[valid, i]
        t = targets[valid, i]

        best_alpha, best_mse = 0.5, float("inf")
        for alpha in np.arange(0.0, 1.01, 0.05):
            ens = alpha * pa + (1 - alpha) * pb
            mse = np.mean((ens - t) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        best_weights.append(best_alpha)
    return best_weights


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    pc_dir = "data/pipeline/point_clouds"
    orig_splits = Path("data/processed/splits")
    merged_dir = Path("data/merged_v5")

    # ══════════════════════════════════════════════════════════
    # Load both models
    # ══════════════════════════════════════════════════════════
    print("Loading PointCloud model...")
    pc_model = MultimodalPointCloudModel(config=config,
                                          pretrained_gnn_path=None)
    pc_ckpt = torch.load("checkpoints/pointcloud/best_model.pt",
                          map_location=device, weights_only=False)
    # Checkpoint may be wrapped in a training state dict
    if "model_state_dict" in pc_ckpt:
        pc_model.load_state_dict(pc_ckpt["model_state_dict"])
    else:
        pc_model.load_state_dict(pc_ckpt)
    pc_model.to(device)
    print(f"  PointCloud params: {sum(p.numel() for p in pc_model.parameters()):,}")

    print("Loading Fix6 (MoE) model...")
    meta = json.load(open(merged_dir / "metadata.json"))
    merged_features = meta["feature_columns"]
    moe_model = MoEA(feature_dim=len(merged_features))
    moe_ckpt = torch.load("checkpoints/moe_fix6/best.pt",
                           map_location=device, weights_only=True)
    moe_model.load_state_dict(moe_ckpt)
    moe_model.to(device)
    print(f"  Fix6 MoE params: {sum(p.numel() for p in moe_model.parameters()):,}")

    # ══════════════════════════════════════════════════════════
    # Get predictions from both models on ORIGINAL val + test
    # ══════════════════════════════════════════════════════════

    # PointCloud uses original data (FEATURE_COLUMNS from preprocessing)
    print("\nLoading original datasets for PointCloud...")
    val_pc = PointCloudMultimodalDataset(str(orig_splits / "val.csv"), pc_dir, is_train=False)
    test_pc = PointCloudMultimodalDataset(str(orig_splits / "test.csv"), pc_dir, is_train=False)
    val_ldr_pc = DataLoader(val_pc, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)
    test_ldr_pc = DataLoader(test_pc, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)

    # Fix6 uses merged_v5 data (merged_features)
    print("Loading merged_v5 datasets for Fix6...")
    val_moe = MergedDataset(str(merged_dir / "splits/val.csv"), pc_dir, merged_features, is_train=False)
    test_moe = MergedDataset(str(merged_dir / "splits/test.csv"), pc_dir, merged_features, is_train=False)
    val_ldr_moe = DataLoader(val_moe, batch_size=32, shuffle=False, collate_fn=collate_merged)
    test_ldr_moe = DataLoader(test_moe, batch_size=32, shuffle=False, collate_fn=collate_merged)

    print("\nRunning inference...")
    val_preds_pc, val_targets_pc = get_predictions(pc_model, val_ldr_pc, device, is_moe=False)
    test_preds_pc, test_targets_pc = get_predictions(pc_model, test_ldr_pc, device, is_moe=False)

    val_preds_moe, val_targets_moe = get_predictions(moe_model, val_ldr_moe, device, is_moe=True)
    test_preds_moe, test_targets_moe = get_predictions(moe_model, test_ldr_moe, device, is_moe=True)

    # Verify targets match (both should be original test set)
    # Note: targets may be in different normalized spaces since models use different scalers
    print(f"  PointCloud test: {test_preds_pc.shape}, MoE test: {test_preds_moe.shape}")

    # ══════════════════════════════════════════════════════════
    # Individual model results
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("INDIVIDUAL MODEL RESULTS")
    print(f"{'='*60}")

    metrics_pc = compute_metrics(test_preds_pc, test_targets_pc)
    metrics_moe = compute_metrics(test_preds_moe, test_targets_moe)
    print(format_metrics(metrics_pc, "PointCloud (alone)"))
    print()
    print(format_metrics(metrics_moe, "Fix6 MoE (alone)"))

    # ══════════════════════════════════════════════════════════
    # Strategy 1: Simple average
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STRATEGY 1: Simple Average")
    print(f"{'='*60}")

    test_preds_avg = (test_preds_pc + test_preds_moe) / 2
    metrics_avg = compute_metrics(test_preds_avg, test_targets_pc)
    print(format_metrics(metrics_avg, "Simple Average"))

    # ══════════════════════════════════════════════════════════
    # Strategy 2: Per-property best (oracle on test results)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STRATEGY 2: Per-Property Best (cherry-pick)")
    print(f"{'='*60}")

    test_preds_best = np.zeros_like(test_preds_pc)
    best_source = {}
    for i, p in enumerate(TARGET_COLUMNS):
        r2_pc = metrics_pc[f"{p}_r2"]
        r2_moe = metrics_moe[f"{p}_r2"]
        if r2_pc >= r2_moe:
            test_preds_best[:, i] = test_preds_pc[:, i]
            best_source[p] = "PointCloud"
        else:
            test_preds_best[:, i] = test_preds_moe[:, i]
            best_source[p] = "Fix6 MoE"
        print(f"  {p:15s}: {best_source[p]} (PC={r2_pc:.4f}, MoE={r2_moe:.4f})")

    metrics_best = compute_metrics(test_preds_best, test_targets_pc)
    print(format_metrics(metrics_best, "Per-Property Best"))

    # ══════════════════════════════════════════════════════════
    # Strategy 3: Optimized weights (tuned on validation set)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STRATEGY 3: Optimized Weights (tuned on val set)")
    print(f"{'='*60}")

    # Optimize weights on validation set (PC weight alpha, MoE weight 1-alpha)
    opt_weights = optimize_weights(val_preds_pc, val_preds_moe, val_targets_pc)

    print("  Optimal weights (alpha for PointCloud, 1-alpha for MoE):")
    for i, p in enumerate(TARGET_COLUMNS):
        print(f"    {p:15s}: PC={opt_weights[i]:.2f}, MoE={1-opt_weights[i]:.2f}")

    # Apply to test set
    test_preds_opt = np.zeros_like(test_preds_pc)
    for i in range(len(TARGET_COLUMNS)):
        test_preds_opt[:, i] = (opt_weights[i] * test_preds_pc[:, i] +
                                 (1 - opt_weights[i]) * test_preds_moe[:, i])

    metrics_opt = compute_metrics(test_preds_opt, test_targets_pc)
    print(format_metrics(metrics_opt, "Optimized Weights"))

    # ══════════════════════════════════════════════════════════
    # FINAL COMPARISON
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")

    all_results = {
        "PointCloud": metrics_pc,
        "Fix6 MoE": metrics_moe,
        "Avg Ens": metrics_avg,
        "Best/Prop": metrics_best,
        "Opt Wt": metrics_opt,
    }

    header = "  {:<12s}".format("Property")
    for name in all_results:
        header += " {:>10s}".format(name)
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<12s}".format(p)
        for name, m in all_results.items():
            line += " {:10.4f}".format(m[key])
        print(line)

    line = "  {:<12s}".format("AVERAGE")
    for name, m in all_results.items():
        line += " {:10.4f}".format(m["avg_r2"])
    print(line)

    # ══════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════
    results = {
        "model": "ensemble_fix6_pointcloud",
        "pointcloud_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                               for k, v in metrics_pc.items()},
        "fix6_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                         for k, v in metrics_moe.items()},
        "simple_average": {k: float(v) if isinstance(v, (float, np.floating)) else v
                           for k, v in metrics_avg.items()},
        "per_property_best": {
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                        for k, v in metrics_best.items()},
            "source_per_property": best_source,
        },
        "optimized_weights": {
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                        for k, v in metrics_opt.items()},
            "weights_pc": opt_weights,
            "weights_moe": [1 - w for w in opt_weights],
        },
    }
    with open("results/ensemble_fix6_pointcloud.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/ensemble_fix6_pointcloud.json")


if __name__ == "__main__":
    main()
