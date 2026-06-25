"""Enhanced PointCloud and MoE models with Chemprop-inspired improvements.

Key changes from original:
1. Replace GAT-GNN with D-MPNN (directed message passing, bond-level states)
2. Sum pooling instead of mean pooling (preserves molecular size)
3. Larger FFN heads (300D, 2 layers — matching Chemprop capacity)
4. Lower dropout (0.2 vs 0.3)

Trains two models:
A. Enhanced PointCloud: D-MPNN + PointNet + cross-attention fusion
B. Enhanced MoE: D-MPNN backbone + balanced sampling on merged_v5
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.dmpnn import DirectedMPNN
from src.models.fusion.multimodal_pointcloud import PointCloudFusion
from src.models.fusion.moe import ExpertHead, PropertyConditionedGating
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud
from scripts.train_joint import MergedDataset, collate_merged


class EnhancedPointCloudModel(nn.Module):
    """PointCloud model with D-MPNN replacing GAT-GNN.

    Changes from original MultimodalPointCloudModel:
    - D-MPNN for graph encoding (directed, bond-level messages)
    - Larger prediction FFN (300D, 2 hidden layers)
    - Dropout 0.2
    """

    def __init__(self, feature_dim=25, fused_dim=288, dropout=0.2):
        super().__init__()
        self.pointnet = PointNetEncoder(in_channels=7, feature_dim=300, dropout=dropout)
        self.dmpnn = DirectedMPNN(
            atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=300, num_layers=3, dropout=dropout, num_targets=0)
        self.fusion = PointCloudFusion(
            pointcloud_dim=300, graph_dim=300, tabular_dim=feature_dim,
            fused_dim=fused_dim, num_heads=8, dropout=dropout)
        self.prediction_head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, 7),
        )

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        pc_feat = self.pointnet(point_cloud)
        graph_feat = self.dmpnn.get_features(atom_features, edge_index, bond_features, batch)
        fused = self.fusion(pc_feat, graph_feat, features)
        return self.prediction_head(fused)


class EnhancedMoE(nn.Module):
    """MoE with D-MPNN backbone and larger expert heads.

    Changes from MoEA:
    - D-MPNN for graph encoding
    - Expert heads: 300D hidden, 2 layers
    - Dropout 0.2
    """

    def __init__(self, feature_dim, fused_dim=288, dropout=0.2):
        super().__init__()
        self.pointnet = PointNetEncoder(in_channels=7, feature_dim=300, dropout=dropout)
        self.dmpnn = DirectedMPNN(
            atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=300, num_layers=3, dropout=dropout, num_targets=0)
        self.fusion = PointCloudFusion(
            pointcloud_dim=300, graph_dim=300, tabular_dim=feature_dim,
            fused_dim=fused_dim, num_heads=8, dropout=dropout)
        # Larger expert heads (matching Chemprop FFN capacity)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fused_dim, fused_dim),
                nn.BatchNorm1d(fused_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fused_dim, fused_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fused_dim // 2, 7),
            ) for _ in range(4)])
        self.gating = PropertyConditionedGating(
            input_dim=fused_dim, num_experts=4, num_properties=7, hidden_dim=64)

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        pc_feat = self.pointnet(point_cloud)
        graph_feat = self.dmpnn.get_features(atom_features, edge_index, bond_features, batch)
        fused = self.fusion(pc_feat, graph_feat, features)
        expert_preds = torch.stack([e(fused) for e in self.experts], dim=2)
        gate_weights, lb_loss = self.gating(fused)
        predictions = (expert_preds * gate_weights).sum(dim=2)
        return predictions, {"load_balance_loss": lb_loss, "gate_weights": gate_weights.detach()}


def train_model(model, train_ldr, val_ldr, device, ckpt_path, is_moe=False,
                num_epochs=200, lr=1e-4, patience=25):
    """Generic training loop."""
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnualingWR = CosineAnnealingWarmRestarts(optimizer, T_0=60, eta_min=1e-6)

    best_loss, no_improve = float("inf"), 0
    for epoch in range(num_epochs):
        model.train()
        tl, n = 0, 0
        for batch in train_ldr:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            optimizer.zero_grad()

            if is_moe:
                preds, aux = model(point_cloud=batch["point_cloud"], features=batch["features"],
                                   atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                                   bond_features=batch["bond_features"], batch=batch["batch"])
                mask = ~torch.isnan(batch["targets"])
                safe = batch["targets"].clone(); safe[~mask] = 0.0
                loss = ((preds - safe)**2 * mask.float()).sum() / mask.float().sum().clamp(min=1)
                loss = loss + aux["load_balance_loss"]
            else:
                preds = model(point_cloud=batch["point_cloud"], features=batch["features"],
                              atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                              bond_features=batch["bond_features"], batch=batch["batch"])
                loss = ((preds - batch["targets"])**2).mean()

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
                if is_moe:
                    preds, _ = model(point_cloud=batch["point_cloud"], features=batch["features"],
                                     atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                                     bond_features=batch["bond_features"], batch=batch["batch"])
                    mask = ~torch.isnan(batch["targets"])
                    safe = batch["targets"].clone(); safe[~mask] = 0.0
                    vl += ((preds - safe)**2 * mask.float()).sum().item() / mask.float().sum().clamp(min=1).item()
                else:
                    preds = model(point_cloud=batch["point_cloud"], features=batch["features"],
                                  atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                                  bond_features=batch["bond_features"], batch=batch["batch"])
                    vl += ((preds - batch["targets"])**2).mean().item()
                vn += 1
        avg_val = vl / max(vn, 1)

        if avg_val < best_loss:
            best_loss = avg_val; no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/{num_epochs} | Train: {tl/max(n,1):.4f} | "
                  f"Val: {avg_val:.4f} | Best: {best_loss:.4f} | Pat: {no_improve}/{patience}")
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return model


def evaluate(model, loader, device, is_moe=False):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            if is_moe:
                preds, _ = model(point_cloud=batch["point_cloud"], features=batch["features"],
                                 atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                                 bond_features=batch["bond_features"], batch=batch["batch"])
            else:
                preds = model(point_cloud=batch["point_cloud"], features=batch["features"],
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

    # ══════════════════════════════════════════════════════════
    # Model A: Enhanced PointCloud (D-MPNN + PointNet)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("MODEL A: Enhanced PointCloud (D-MPNN backbone)")
    print(f"{'='*60}")

    orig_splits = Path("data/processed/splits")
    train_ds = PointCloudMultimodalDataset(str(orig_splits / "train.csv"), pc_dir, is_train=True)
    val_ds = PointCloudMultimodalDataset(str(orig_splits / "val.csv"), pc_dir, is_train=False)
    test_ds = PointCloudMultimodalDataset(str(orig_splits / "test.csv"), pc_dir, is_train=False)

    train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_pointcloud)
    val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)

    model_a = EnhancedPointCloudModel(feature_dim=len(FEATURE_COLUMNS), fused_dim=288, dropout=0.2)
    model_a.to(device)
    n_params_a = sum(p.numel() for p in model_a.parameters())
    print(f"  Params: {n_params_a:,}")

    ckpt_a = Path("checkpoints/enhanced_pointcloud")
    ckpt_a.mkdir(parents=True, exist_ok=True)

    model_a = train_model(model_a, train_ldr, val_ldr, device,
                           ckpt_a / "best.pt", is_moe=False, lr=1e-4)

    preds_a, targets_a = evaluate(model_a, test_ldr, device, is_moe=False)
    metrics_a = compute_metrics(preds_a, targets_a)
    print(f"\n{format_metrics(metrics_a, 'Enhanced PointCloud (D-MPNN)')}")

    # ══════════════════════════════════════════════════════════
    # Model B: Enhanced MoE (D-MPNN + balanced sampling)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("MODEL B: Enhanced MoE (D-MPNN + balanced sampling)")
    print(f"{'='*60}")

    merged_dir = Path("data/merged_v5")
    meta = json.load(open(merged_dir / "metadata.json"))
    merged_features = meta["feature_columns"]

    train_csv = str(merged_dir / "splits/train.csv")
    train_ds_m = MergedDataset(train_csv, pc_dir, merged_features, is_train=True)
    val_ds_m = MergedDataset(str(merged_dir / "splits/val.csv"), pc_dir, merged_features, is_train=False)
    test_ds_m = MergedDataset(str(merged_dir / "splits/test.csv"), pc_dir, merged_features, is_train=False)

    # Balanced sampler
    df_train = pd.read_csv(train_csv)
    is_orig = (df_train["source"] == "original").values
    n_orig, n_ilth = is_orig.sum(), len(df_train) - is_orig.sum()
    weights = np.where(is_orig, 0.5 / max(n_orig, 1), 0.5 / max(n_ilth, 1))
    sampler = WeightedRandomSampler(torch.from_numpy(weights).double(),
                                     num_samples=len(df_train), replacement=True)
    print(f"  Balanced sampler: {n_orig} original, {n_ilth} ILThermo")

    train_ldr_m = DataLoader(train_ds_m, batch_size=64, sampler=sampler, collate_fn=collate_merged)
    val_ldr_m = DataLoader(val_ds_m, batch_size=32, shuffle=False, collate_fn=collate_merged)
    test_ldr_m = DataLoader(test_ds_m, batch_size=32, shuffle=False, collate_fn=collate_merged)

    model_b = EnhancedMoE(feature_dim=len(merged_features), fused_dim=288, dropout=0.2)
    model_b.to(device)
    n_params_b = sum(p.numel() for p in model_b.parameters())
    print(f"  Params: {n_params_b:,}")

    ckpt_b = Path("checkpoints/enhanced_moe")
    ckpt_b.mkdir(parents=True, exist_ok=True)

    model_b = train_model(model_b, train_ldr_m, val_ldr_m, device,
                           ckpt_b / "best.pt", is_moe=True, lr=1e-4)

    preds_b, targets_b = evaluate(model_b, test_ldr_m, device, is_moe=True)
    metrics_b = compute_metrics(preds_b, targets_b)
    print(f"\n{format_metrics(metrics_b, 'Enhanced MoE (D-MPNN)')}")

    # ══════════════════════════════════════════════════════════
    # Comparison
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("COMPARISON: Original vs Enhanced vs Chemprop")
    print(f"{'='*60}")

    prev = {}
    for name, path, key in [
        ("Chemprop", "results/chemprop_results.json", "test_metrics"),
        ("PointCloud (orig)", "results/pointcloud_results.json", None),
        ("MoE Fix6 (orig)", "results/moe_fix6_results.json", "metrics"),
    ]:
        try:
            data = json.load(open(path))
            m = data.get(key, data.get("metrics", data.get("test_metrics", {})))
            prev[name] = m
        except Exception:
            pass

    header = "  {:<25s}".format("Property")
    for name in prev:
        header += " {:>12s}".format(name)
    header += " {:>12s} {:>12s}".format("PC+DMPNN", "MoE+DMPNN")
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<25s}".format(p)
        for name in prev:
            line += " {:12.4f}".format(prev[name].get(key, float('nan')))
        line += " {:12.4f} {:12.4f}".format(metrics_a[key], metrics_b[key])
        print(line)

    line = "  {:<25s}".format("AVERAGE")
    for name in prev:
        line += " {:12.4f}".format(prev[name].get('avg_r2', float('nan')))
    line += " {:12.4f} {:12.4f}".format(metrics_a['avg_r2'], metrics_b['avg_r2'])
    print(line)

    # Save
    results = {
        "enhanced_pointcloud": {
            "n_params": n_params_a,
            "changes": "D-MPNN(300D,3L), PointNet(300D), fusion(300D), FFN(300-150-7), dropout=0.2",
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                        for k, v in metrics_a.items()},
        },
        "enhanced_moe": {
            "n_params": n_params_b,
            "changes": "D-MPNN(300D,3L), PointNet(300D), fusion(300D), experts(300-150-7)x4, dropout=0.2, balanced sampling",
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                        for k, v in metrics_b.items()},
        },
    }
    with open("results/enhanced_models_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/enhanced_models_results.json")


if __name__ == "__main__":
    main()
