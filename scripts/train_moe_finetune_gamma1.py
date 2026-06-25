"""MoE with gamma1 head fine-tuning.

Stage 1: Train full MoE on merged_v3 data (all 7 targets)
Stage 2: Freeze entire model, unfreeze gamma1 output weights only
Stage 3: Fine-tune gamma1 on original 223 samples for 30 epochs

Tests all 4 fusion variants with this approach.
"""

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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import TARGET_COLUMNS, FEATURE_COLUMNS
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.moe import ExpertHead, PropertyConditionedGating
from src.models.fusion.multimodal_pointcloud import PointCloudFusion
from src.models.fusion.fusion_variants import (
    PhysicsBottleneckFusion, HierarchicalMultiScaleFusion, GatedResidualFusion)
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_joint import MergedDataset, collate_merged
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud
from scripts.train_moe import evaluate_single

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

    def get_gamma1_params(self):
        """Get only the parameters that affect gamma1 output (index 0).

        This includes:
        - The last linear layer's weight[0] and bias[0] in each expert
        - The gating network's property embedding for property 0
        """
        params = []
        for expert in self.experts:
            # Last layer of each expert
            last_layer = expert.net[-1]  # nn.Linear
            # We can't easily isolate row 0, so we fine-tune the full last layer
            params.extend(last_layer.parameters())
        # Gating property embedding for gamma1
        params.append(self.gating.property_embed.weight)
        return params


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


class MaskedMSE(nn.Module):
    def forward(self, preds, targets, aux=None):
        mask = ~torch.isnan(targets)
        if mask.sum() == 0:
            return {"total": torch.tensor(0.0, device=preds.device, requires_grad=True)}
        safe = targets.clone()
        safe[~mask] = 0.0
        diff2 = (preds - safe) ** 2 * mask.float()
        per_task = diff2.sum(dim=0) / mask.sum(dim=0).float().clamp(min=1)
        total = per_task[mask.any(dim=0)].mean()
        if aux and "load_balance_loss" in aux:
            total = total + aux["load_balance_loss"]
        return {"total": total}


def stage1_train_merged(model, train_loader, val_loader, device, ckpt_dir,
                        num_epochs=200, lr=1e-4, patience=25):
    """Stage 1: Train on full merged data."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    criterion = MaskedMSE()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=60, T_mult=1, eta_min=1e-6)

    best_loss = float("inf")
    no_improve = 0

    for epoch in range(num_epochs):
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
            print(f"    Epoch {epoch:3d}/{num_epochs} | Train: {total_loss/max(n,1):.4f} | "
                  f"Val: {avg_val:.4f} | Best: {best_loss:.4f} | Pat: {no_improve}/{patience}")
        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=True))
    return model


def stage2_finetune_gamma1(model, train_loader, val_loader, device, ckpt_dir,
                           num_epochs=30, lr=5e-5):
    """Stage 2: Freeze all, fine-tune gamma1 output weights on original data."""
    ckpt_dir = Path(ckpt_dir)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze gamma1-related params
    gamma1_params = model.get_gamma1_params()
    for param in gamma1_params:
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"    Fine-tuning {n_trainable:,} / {n_total:,} params for gamma1")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Gamma1-only MSE loss (index 0)
    best_gamma1_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss, n = 0, 0
        for batch in train_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            optimizer.zero_grad()
            preds, _ = model(point_cloud=batch["point_cloud"], features=batch["features"],
                             atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                             bond_features=batch["bond_features"], batch=batch["batch"])
            # Only backprop gamma1 loss
            gamma1_loss = ((preds[:, 0] - batch["targets"][:, 0]) ** 2).mean()
            gamma1_loss.backward()
            optimizer.step()
            total_loss += gamma1_loss.item()
            n += 1
        scheduler.step()

        avg_loss = total_loss / max(n, 1)
        if avg_loss < best_gamma1_loss:
            best_gamma1_loss = avg_loss
            torch.save(model.state_dict(), ckpt_dir / "best_finetuned.pt")

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"    FT Epoch {epoch:3d}/{num_epochs} | gamma1 loss: {avg_loss:.4f}")

    model.load_state_dict(torch.load(ckpt_dir / "best_finetuned.pt",
                                      map_location=device, weights_only=True))

    # Unfreeze all for future use
    for param in model.parameters():
        param.requires_grad = True

    return model


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    # ── Load merged_v3 data ──
    merged_dir = Path("data/merged_v3")
    meta = json.load(open(merged_dir / "metadata.json"))
    merged_features = meta["feature_columns"]

    pc_dir = "data/pipeline/point_clouds"
    merged_splits = merged_dir / "splits"
    orig_splits = Path("data/processed/splits")

    # Merged data loaders (Stage 1)
    print("\nLoading merged_v3 data...")
    train_merged = MergedDataset(str(merged_splits / "train.csv"), pc_dir, merged_features, is_train=True)
    val_merged = MergedDataset(str(merged_splits / "val.csv"), pc_dir, merged_features, is_train=False)
    test_merged = MergedDataset(str(merged_splits / "test.csv"), pc_dir, merged_features, is_train=False)

    train_loader_m = DataLoader(train_merged, batch_size=64, shuffle=True, collate_fn=collate_merged)
    val_loader_m = DataLoader(val_merged, batch_size=32, shuffle=False, collate_fn=collate_merged)
    test_loader_m = DataLoader(test_merged, batch_size=32, shuffle=False, collate_fn=collate_merged)

    # Original data loaders (Stage 2 fine-tuning)
    print("Loading original data for gamma1 fine-tuning...")
    train_orig = PointCloudMultimodalDataset(str(orig_splits / "train.csv"), pc_dir, is_train=True)
    val_orig = PointCloudMultimodalDataset(str(orig_splits / "val.csv"), pc_dir, is_train=False)
    test_orig = PointCloudMultimodalDataset(str(orig_splits / "test.csv"), pc_dir, is_train=False)

    train_loader_o = DataLoader(train_orig, batch_size=32, shuffle=True, collate_fn=collate_pointcloud)
    val_loader_o = DataLoader(val_orig, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)
    test_loader_o = DataLoader(test_orig, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)

    # ── Train all 4 variants ──
    all_results = {}
    n_merged_features = len(merged_features)

    for vid in ["A", "B", "C", "D"]:
        vname = VARIANT_NAMES[vid]
        print(f"\n{'='*60}")
        print(f"  MoE-{vid}: {vname}")
        print(f"{'='*60}")

        set_seed(42)
        fusion = build_fusion(vid, n_merged_features)
        model = MoEVariant(
            feature_dim=n_merged_features, fusion_module=fusion, num_experts=4,
            fused_dim=256, dropout=0.3,
            pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
        model.to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        fusion_params = sum(p.numel() for p in model.fusion.parameters())
        print(f"  Params: {n_params:,} (fusion: {fusion_params:,})")

        ckpt_dir = f"checkpoints/moe_ft_{vid}"

        # Stage 1: Train on merged data
        print(f"\n  Stage 1: Training on merged_v3 ({len(train_merged)} samples)...")
        model = stage1_train_merged(model, train_loader_m, val_loader_m, device,
                                     ckpt_dir, num_epochs=200, lr=1e-4, patience=25)

        # Evaluate before fine-tuning (on merged test — same scale)
        preds_before, targets_m, gw = evaluate_single(model, test_loader_m, device)
        metrics_before = compute_metrics(preds_before, targets_m)
        print(f"\n  Before gamma1 fine-tune:")
        print(f"    gamma1 R²: {metrics_before.get('gamma1_r2', 'N/A'):.4f}")
        print(f"    Avg R² (all 7): {metrics_before.get('avg_r2', 'N/A'):.4f}")

        # Stage 2: Fine-tune gamma1 on original data
        print(f"\n  Stage 2: Fine-tuning gamma1 on original ({len(train_orig)} samples)...")
        model = stage2_finetune_gamma1(model, train_loader_o, val_loader_o, device,
                                        ckpt_dir, num_epochs=30, lr=5e-5)

        # Final evaluation on ORIGINAL test set (same scale as gamma1 training)
        preds_orig, targets_orig, gw_orig = evaluate_single(model, test_loader_o, device)
        metrics_orig = compute_metrics(preds_orig, targets_orig)

        # Also evaluate non-gamma1 on merged test (correct scale for those)
        preds_merged, targets_merged, _ = evaluate_single(model, test_loader_m, device)
        metrics_merged = compute_metrics(preds_merged, targets_merged)

        # Best of both: gamma1 from original eval, rest from merged eval
        hybrid_metrics = {}
        hybrid_metrics["gamma1_r2"] = metrics_orig.get("gamma1_r2", 0)
        for prop in TARGET_COLUMNS[1:]:
            hybrid_metrics[f"{prop}_r2"] = metrics_merged.get(f"{prop}_r2", 0)
        hybrid_metrics["avg_r2"] = np.mean([hybrid_metrics[f"{p}_r2"] for p in TARGET_COLUMNS])

        print(f"\n  After gamma1 fine-tune (hybrid eval):")
        for prop in TARGET_COLUMNS:
            src = "orig" if prop == "gamma1" else "merged"
            print(f"    {prop:15s} R²: {hybrid_metrics[f'{prop}_r2']:.4f}  (from {src} test)")
        print(f"    {'AVERAGE':15s} R²: {hybrid_metrics['avg_r2']:.4f}")

        all_results[vid] = {
            "name": vname,
            "total_params": n_params,
            "fusion_params": fusion_params,
            "before_finetune": {k: float(v) if isinstance(v, (float, np.floating)) else v
                               for k, v in metrics_before.items()},
            "after_finetune_orig": {k: float(v) if isinstance(v, (float, np.floating)) else v
                                    for k, v in metrics_orig.items()},
            "after_finetune_merged": {k: float(v) if isinstance(v, (float, np.floating)) else v
                                      for k, v in metrics_merged.items()},
            "hybrid": {k: float(v) if isinstance(v, (float, np.floating)) else v
                      for k, v in hybrid_metrics.items()},
            "gate_weights": gw_orig.mean(axis=0).tolist(),
        }

    # ── Final comparison ──
    print(f"\n{'='*60}")
    print("FINAL COMPARISON: MoE + Gamma1 Fine-Tuning")
    print(f"{'='*60}")

    header = f"{'Model':35s} {'Avg R²':>10s}"
    for p in TARGET_COLUMNS:
        header += f" {p:>8s}"
    print(header)
    print("-" * len(header))

    # Baselines
    for name, path in [("Hard Ensemble (prev best)", "results/ensemble_phase23_results.json"),
                       ("Phase 3 PointCloud", "results/pointcloud_results.json")]:
        try:
            with open(path) as f:
                data = json.load(f)
            m = data.get("hard_ensemble", data.get("test_metrics", {}))
            row = f"{name:35s} {m.get('avg_r2',0):>10.4f}"
            for p in TARGET_COLUMNS:
                row += f" {m.get(f'{p}_r2',0):>8.4f}"
            print(row)
        except Exception:
            pass

    print("-" * len(header))
    for vid in ["A", "B", "C", "D"]:
        r = all_results[vid]
        m = r["hybrid"]
        row = f"MoE-{vid} ({r['name']})+FT{' ':8s} {m['avg_r2']:>10.4f}"
        for p in TARGET_COLUMNS:
            row += f" {m[f'{p}_r2']:>8.4f}"
        print(row)

    best_vid = max(all_results.keys(), key=lambda k: all_results[k]["hybrid"]["avg_r2"])
    best_r2 = all_results[best_vid]["hybrid"]["avg_r2"]
    print(f"\nBest: MoE-{best_vid} ({all_results[best_vid]['name']}) + gamma1 FT = R² {best_r2:.4f}")

    # Save
    with open("results/moe_gamma1_finetune_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: results/moe_gamma1_finetune_results.json")


if __name__ == "__main__":
    main()
