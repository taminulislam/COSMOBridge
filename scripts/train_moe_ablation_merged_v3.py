"""MoE Fusion Ablation on merged_v3 (gamma1-fixed). Called with --variant A/B/C/D."""

import sys
import json
import copy
import numpy as np
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
from src.models.fusion.fusion_variants import (
    PhysicsBottleneckFusion, HierarchicalMultiScaleFusion, GatedResidualFusion)
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_joint import MergedDataset, collate_merged
from scripts.train_moe import evaluate_single

TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]

VARIANT_NAMES = {
    "A": "Cross-Attention",
    "B": "Physics Bottleneck",
    "C": "Hierarchical Multi-Scale",
    "D": "Gated Residual",
}


class MoEVariant(nn.Module):
    def __init__(self, feature_dim, fusion_module, num_experts=4,
                 fused_dim=256, dropout=0.3, pretrained_gnn_path=None):
        super().__init__()
        self.pointnet = PointNetEncoder(in_channels=7, feature_dim=256, dropout=dropout)
        self.gnn = MolecularGNN(
            atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
            dropout=dropout, pooling="mean", num_targets=0)
        if pretrained_gnn_path and Path(pretrained_gnn_path).exists():
            ckpt = torch.load(pretrained_gnn_path, map_location="cpu", weights_only=True)
            gnn_state = {k: v for k, v in ckpt.items()
                         if any(k.startswith(p) for p in
                                ["atom_projection", "convs", "batch_norms", "pool"])}
            if gnn_state:
                self.gnn.load_state_dict(gnn_state, strict=False)
        self.fusion = fusion_module
        self.experts = nn.ModuleList([
            ExpertHead(fused_dim, hidden_dim=128, num_targets=7, dropout=dropout)
            for _ in range(num_experts)])
        self.gating = PropertyConditionedGating(
            input_dim=fused_dim, num_experts=num_experts,
            num_properties=7, hidden_dim=64)

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        pc_feat = self.pointnet(point_cloud)
        graph_feat = self.gnn.get_features(atom_features, edge_index, bond_features, batch)
        fused = self.fusion(pc_feat, graph_feat, features)
        expert_preds = torch.stack([e(fused) for e in self.experts], dim=2)
        gate_weights, lb_loss = self.gating(fused)
        predictions = (expert_preds * gate_weights).sum(dim=2)
        return predictions, {"load_balance_loss": lb_loss, "gate_weights": gate_weights.detach()}


def build_fusion(variant, feature_dim, fused_dim=256):
    if variant == "A":
        return PointCloudFusion(pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
                                fused_dim=fused_dim, num_heads=8, dropout=0.3)
    elif variant == "B":
        return PhysicsBottleneckFusion(pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
                                       fused_dim=fused_dim, n_bottlenecks=6, num_heads=4, dropout=0.3)
    elif variant == "C":
        return HierarchicalMultiScaleFusion(pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
                                             fused_dim=fused_dim, num_heads=4, dropout=0.3)
    elif variant == "D":
        return GatedResidualFusion(pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
                                    fused_dim=fused_dim, dropout=0.3)


class MaskedMSEWithWeighting(nn.Module):
    """Masked MSE with 5x importance weighting for original gamma1 samples."""
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, aux_losses=None):
        mask = ~torch.isnan(targets)
        if mask.sum() == 0:
            return {"total": torch.tensor(0.0, device=preds.device, requires_grad=True)}
        safe = targets.clone()
        safe[~mask] = 0.0
        diff2 = (preds - safe) ** 2 * mask.float()
        per_task = diff2.sum(dim=0) / mask.sum(dim=0).float().clamp(min=1)
        active = mask.any(dim=0)
        total = per_task[active].mean()
        if aux_losses and "load_balance_loss" in aux_losses:
            total = total + aux_losses["load_balance_loss"]
        losses = {"total": total}
        for i, name in enumerate(TARGET_COLUMNS):
            losses[name] = per_task[i]
        return losses


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=["A", "B", "C", "D"])
    args = parser.parse_args()

    vid = args.variant
    vname = VARIANT_NAMES[vid]

    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)

    merged_dir = Path("data/merged_v3")
    meta = json.load(open(merged_dir / "metadata.json"))
    feature_columns = meta["feature_columns"]
    n_features = len(feature_columns)

    pc_dir = "data/pipeline/point_clouds"
    splits = merged_dir / "splits"

    print(f"{'='*60}")
    print(f"MoE-{vid}: {vname} (merged_v3, gamma1 fixed, {meta['total_rows']} rows)")
    print(f"Device: {device}")
    print(f"{'='*60}")

    train_ds = MergedDataset(str(splits / "train.csv"), pc_dir, feature_columns, is_train=True)
    val_ds = MergedDataset(str(splits / "val.csv"), pc_dir, feature_columns, is_train=False)
    test_ds = MergedDataset(str(splits / "test.csv"), pc_dir, feature_columns, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_merged)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)

    fusion = build_fusion(vid, n_features)
    model = MoEVariant(
        feature_dim=n_features, fusion_module=fusion, num_experts=4,
        fused_dim=256, dropout=0.3,
        pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    print(f"Params: {n_params:,} (fusion: {fusion_params:,})")

    # Train with masked loss
    criterion = MaskedMSEWithWeighting().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=60, T_mult=1, eta_min=1e-6)

    ckpt_dir = Path(f"checkpoints/moe_v3_{vid}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    no_improve = 0
    patience = 25

    for epoch in range(200):
        model.train()
        total_loss, n = 0, 0
        for batch in train_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            optimizer.zero_grad()
            preds, aux = model(point_cloud=batch["point_cloud"], features=batch["features"],
                               atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                               bond_features=batch["bond_features"], batch=batch["batch"])
            losses = criterion(preds, batch["targets"], aux)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += losses["total"].item()
            n += 1
        scheduler.step()

        model.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                preds, aux = model(point_cloud=batch["point_cloud"], features=batch["features"],
                                   atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                                   bond_features=batch["bond_features"], batch=batch["batch"])
                losses = criterion(preds, batch["targets"], aux)
                vl += losses["total"].item()
                vn += 1
        avg_val = vl / max(vn, 1)

        if avg_val < best_loss:
            best_loss = avg_val
            no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
        else:
            no_improve += 1

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/200 | Train: {total_loss/max(n,1):.4f} | "
                  f"Val: {avg_val:.4f} | Best: {best_loss:.4f} | Pat: {no_improve}/{patience}")
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=True))

    preds, targets, gate_weights = evaluate_single(model, test_loader, device)
    metrics = compute_metrics(preds, targets)
    print(f"\n{format_metrics(metrics, f'MoE-{vid} ({vname}) merged_v3')}")

    result = {
        "variant": vid, "name": vname, "dataset": "merged_v3",
        "total_params": n_params, "fusion_params": fusion_params,
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items()},
        "gate_weights": gate_weights.mean(axis=0).tolist(),
    }
    with open(f"results/moe_v3_{vid}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: results/moe_v3_{vid}.json")


if __name__ == "__main__":
    main()
