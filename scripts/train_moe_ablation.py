"""MoE Fusion Ablation Study.

Trains 4 MoE variants with identical setup except for the fusion module:
  A: Cross-Attention (baseline)
  B: Physics-Informed Bottleneck
  C: Hierarchical Multi-Scale
  D: Gated Residual

Produces a comparison table and interpretability analysis.
"""

import sys
import json
import copy
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device, set_seed
from src.models.fusion.moe import (
    MixtureOfExpertsModel, SharedBackbone, ExpertHead,
    PropertyConditionedGating,
)
from src.models.fusion.fusion_variants import (
    PhysicsBottleneckFusion,
    HierarchicalMultiScaleFusion,
    GatedResidualFusion,
)
from src.models.fusion.multimodal_pointcloud import PointCloudFusion
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_joint import MergedDataset, collate_merged
from scripts.train_moe import MoEMaskedLoss, train_moe, evaluate_single

TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]

from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.cross_attention import CrossAttention


class MoEWithFusionVariant(nn.Module):
    """MoE model with swappable fusion module."""

    def __init__(self, feature_dim, fusion_module, num_experts=4,
                 fused_dim=256, dropout=0.3, pretrained_gnn_path=None):
        super().__init__()

        # PointNet
        self.pointnet = PointNetEncoder(in_channels=7, feature_dim=256, dropout=dropout)

        # GNN
        self.gnn = MolecularGNN(
            atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
            dropout=dropout, pooling="mean", num_targets=0)

        # Load pre-trained GNN
        if pretrained_gnn_path and Path(pretrained_gnn_path).exists():
            ckpt = torch.load(pretrained_gnn_path, map_location="cpu", weights_only=True)
            gnn_state = {k: v for k, v in ckpt.items()
                         if any(k.startswith(p) for p in
                                ["atom_projection", "convs", "batch_norms", "pool"])}
            if gnn_state:
                self.gnn.load_state_dict(gnn_state, strict=False)

        # Swappable fusion module
        self.fusion = fusion_module

        # Expert heads
        self.experts = nn.ModuleList([
            ExpertHead(fused_dim, hidden_dim=128, num_targets=7, dropout=dropout)
            for _ in range(num_experts)
        ])

        # Gating
        self.gating = PropertyConditionedGating(
            input_dim=fused_dim, num_experts=num_experts,
            num_properties=7, hidden_dim=64)

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        # Encode
        pc_feat = self.pointnet(point_cloud)
        graph_feat = self.gnn.get_features(atom_features, edge_index,
                                            bond_features, batch)

        # Fuse (swappable)
        fused = self.fusion(pc_feat, graph_feat, features)

        # Expert predictions
        expert_preds = torch.stack(
            [expert(fused) for expert in self.experts], dim=2)

        # Gating
        gate_weights, lb_loss = self.gating(fused)

        # Weighted combination
        predictions = (expert_preds * gate_weights).sum(dim=2)

        return predictions, {
            "load_balance_loss": lb_loss,
            "gate_weights": gate_weights.detach(),
        }


def build_fusion_variant(variant, feature_dim, fused_dim=256):
    """Build a fusion module by variant name."""
    if variant == "A":
        return PointCloudFusion(
            pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
            fused_dim=fused_dim, num_heads=8, dropout=0.3)
    elif variant == "B":
        return PhysicsBottleneckFusion(
            pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
            fused_dim=fused_dim, n_bottlenecks=6, num_heads=4, dropout=0.3)
    elif variant == "C":
        return HierarchicalMultiScaleFusion(
            pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
            fused_dim=fused_dim, num_heads=4, dropout=0.3)
    elif variant == "D":
        return GatedResidualFusion(
            pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
            fused_dim=fused_dim, dropout=0.3)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    # Load data
    merged_dir = Path("data/merged")
    meta = json.load(open(merged_dir / "metadata.json"))
    feature_columns = meta["feature_columns"]
    n_features = len(feature_columns)

    pc_dir = "data/pipeline/point_clouds"
    splits = merged_dir / "splits"

    print("Loading datasets...")
    train_ds = MergedDataset(str(splits / "train.csv"), pc_dir, feature_columns, is_train=True)
    val_ds = MergedDataset(str(splits / "val.csv"), pc_dir, feature_columns, is_train=False)
    test_ds = MergedDataset(str(splits / "test.csv"), pc_dir, feature_columns, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_merged)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)

    # ── Train each variant ──
    variants = {
        "A": "Cross-Attention",
        "B": "Physics Bottleneck",
        "C": "Hierarchical Multi-Scale",
        "D": "Gated Residual",
    }

    all_results = {}
    all_gate_weights = {}

    for variant_id, variant_name in variants.items():
        print(f"\n{'='*70}")
        print(f"  VARIANT {variant_id}: {variant_name}")
        print(f"{'='*70}")

        set_seed(42)  # Same init for fair comparison

        fusion = build_fusion_variant(variant_id, n_features)
        model = MoEWithFusionVariant(
            feature_dim=n_features,
            fusion_module=fusion,
            num_experts=4,
            fused_dim=256,
            dropout=0.3,
            pretrained_gnn_path="checkpoints/transfer/pretrained.pt",
        )
        model.to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        fusion_params = sum(p.numel() for p in model.fusion.parameters())
        print(f"  Total params: {n_params:,}  |  Fusion params: {fusion_params:,}")

        model, snapshots = train_moe(
            model, train_loader, val_loader, device,
            ckpt_dir=f"checkpoints/moe_ablation_{variant_id}",
            num_epochs=200, lr=1e-4, patience=25)

        # Evaluate
        preds, targets, gate_weights = evaluate_single(model, test_loader, device)
        metrics = compute_metrics(preds, targets)

        print(f"\n  Variant {variant_id} ({variant_name}) Results:")
        print(format_metrics(metrics, f"MoE-{variant_id}"))

        all_results[variant_id] = {
            "name": variant_name,
            "total_params": n_params,
            "fusion_params": fusion_params,
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                       for k, v in metrics.items()},
        }
        all_gate_weights[variant_id] = gate_weights

    # ── Comparison Table ──
    print(f"\n{'='*70}")
    print("ABLATION STUDY: FUSION VARIANT COMPARISON")
    print(f"{'='*70}")

    # Header
    header = f"{'Variant':30s} {'Params':>10s} {'Fus.Params':>10s} {'Avg R²':>10s}"
    for prop in TARGET_COLUMNS:
        header += f" {prop:>8s}"
    print(header)
    print("-" * len(header))

    for vid in ["A", "B", "C", "D"]:
        r = all_results[vid]
        m = r["metrics"]
        row = f"MoE-{vid}: {r['name']:24s} {r['total_params']:>10,d} {r['fusion_params']:>10,d} {m.get('avg_r2', 0):>10.4f}"
        for prop in TARGET_COLUMNS:
            row += f" {m.get(f'{prop}_r2', 0):>8.4f}"
        print(row)

    # Best variant
    best_id = max(all_results.keys(), key=lambda k: all_results[k]["metrics"].get("avg_r2", -999))
    print(f"\n  Best: MoE-{best_id} ({all_results[best_id]['name']}) "
          f"with avg R² = {all_results[best_id]['metrics']['avg_r2']:.4f}")

    # Efficiency analysis
    print(f"\n  Efficiency (fusion params):")
    for vid in ["A", "B", "C", "D"]:
        r = all_results[vid]
        r2 = r["metrics"].get("avg_r2", 0)
        fp = r["fusion_params"]
        efficiency = r2 / fp * 100000 if fp > 0 else 0
        print(f"    MoE-{vid}: {fp:>8,d} fusion params, R²/param efficiency: {efficiency:.4f}")

    # ── Save results ──
    with open("results/moe_ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results/moe_ablation_results.json")

    # ── Generate comparison figure ──
    print("\nGenerating ablation comparison figure...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: Average R² comparison
        names = [f"MoE-{v}\n({all_results[v]['name']})" for v in ["A", "B", "C", "D"]]
        r2s = [all_results[v]["metrics"].get("avg_r2", 0) for v in ["A", "B", "C", "D"]]
        colors = ['#3498DB', '#E74C3C', '#27AE60', '#F39C12']

        bars = axes[0].bar(range(4), r2s, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
        axes[0].set_xticks(range(4))
        axes[0].set_xticklabels(names, fontsize=9)
        axes[0].set_ylabel('Average $R^2$')
        axes[0].set_title('(a) Fusion Variant Comparison')
        for bar, val in zip(bars, r2s):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        # Panel B: Per-property comparison
        prop_labels = [r'$\gamma_1$', r'$\gamma_2$', r'$G^E$', r'$H^E$',
                       r'$G_{mix}$', r'$H_{vap}$', r'$P$']
        x = np.arange(7)
        width = 0.2
        for i, (vid, color) in enumerate(zip(["A", "B", "C", "D"], colors)):
            r2_vals = [all_results[vid]["metrics"].get(f"{p}_r2", 0) for p in TARGET_COLUMNS]
            axes[1].bar(x + i * width, r2_vals, width, label=f'MoE-{vid}',
                       color=color, alpha=0.85)

        axes[1].set_xticks(x + 1.5 * width)
        axes[1].set_xticklabels(prop_labels)
        axes[1].set_ylabel('$R^2$')
        axes[1].set_title('(b) Per-Property $R^2$ by Fusion Variant')
        axes[1].legend(fontsize=8)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig('paper/figures/moe_ablation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: paper/figures/moe_ablation.png")
    except Exception as e:
        print(f"  Figure generation failed: {e}")


if __name__ == "__main__":
    main()
