"""Train MoE on original 223 samples only — fair comparison with Hard Ensemble.

Uses the same data splits as all Phase 1-3 experiments.
No ILThermo contamination, no gamma1 distribution mismatch.
Tests whether MoE's learned routing matches/beats hard-coded ensemble routing.
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

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import TARGET_COLUMNS, FEATURE_COLUMNS
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud
from scripts.train_moe import evaluate_single

from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.moe import ExpertHead, PropertyConditionedGating
from src.models.fusion.multimodal_pointcloud import PointCloudFusion
from src.models.fusion.fusion_variants import PhysicsBottleneckFusion
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class MoEOriginal(nn.Module):
    """MoE model for original dataset (25 features, not 281)."""

    def __init__(self, feature_dim, fusion_type="cross_attention",
                 num_experts=4, fused_dim=256, dropout=0.3,
                 pretrained_gnn_path=None):
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
                print(f"  Loaded pre-trained GNN: {len(gnn_state)} params")

        # Fusion
        if fusion_type == "bottleneck":
            self.fusion = PhysicsBottleneckFusion(
                pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
                fused_dim=fused_dim, n_bottlenecks=6, num_heads=4, dropout=dropout)
        else:
            self.fusion = PointCloudFusion(
                pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
                fused_dim=fused_dim, num_heads=8, dropout=dropout)

        self.experts = nn.ModuleList([
            ExpertHead(fused_dim, hidden_dim=128, num_targets=7, dropout=dropout)
            for _ in range(num_experts)
        ])

        self.gating = PropertyConditionedGating(
            input_dim=fused_dim, num_experts=num_experts,
            num_properties=7, hidden_dim=64)

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        pc_feat = self.pointnet(point_cloud)
        graph_feat = self.gnn.get_features(atom_features, edge_index,
                                            bond_features, batch)
        fused = self.fusion(pc_feat, graph_feat, features)

        expert_preds = torch.stack(
            [expert(fused) for expert in self.experts], dim=2)
        gate_weights, lb_loss = self.gating(fused)
        predictions = (expert_preds * gate_weights).sum(dim=2)

        return predictions, {
            "load_balance_loss": lb_loss,
            "gate_weights": gate_weights.detach(),
        }


def train_model(model, train_loader, val_loader, device, ckpt_dir,
                num_epochs=300, lr=1e-4, patience=30):
    """Train with standard MSE loss (no masking needed — all 7 targets present)."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=60, T_mult=1, eta_min=1e-6)

    best_loss = float("inf")
    no_improve = 0
    snapshots = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        n = 0

        for batch in train_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            optimizer.zero_grad()
            preds, aux = model(
                point_cloud=batch["point_cloud"], features=batch["features"],
                atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                bond_features=batch["bond_features"], batch=batch["batch"])

            loss = criterion(preds, batch["targets"]) + aux["load_balance_loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n += 1

        scheduler.step()
        avg_loss = total_loss / max(n, 1)

        # Validate
        model.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                preds, _ = model(
                    point_cloud=batch["point_cloud"], features=batch["features"],
                    atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                    bond_features=batch["bond_features"], batch=batch["batch"])
                vl += criterion(preds, batch["targets"]).item()
                vn += 1

        avg_val = vl / max(vn, 1)

        if avg_val < best_loss:
            best_loss = avg_val
            no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
        else:
            no_improve += 1

        # Snapshot at cycle boundaries
        if (epoch + 1) % 60 == 0:
            snapshots.append(copy.deepcopy(model.state_dict()))

        if epoch % 20 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch:3d}/{num_epochs} | Train: {avg_loss:.4f} | "
                  f"Val: {avg_val:.4f} | Best: {best_loss:.4f} | Pat: {no_improve}/{patience}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(ckpt_dir / "best_model.pt",
                                      map_location=device, weights_only=True))
    return model, snapshots


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    n_features = len(FEATURE_COLUMNS)
    pc_dir = "data/pipeline/point_clouds"
    splits_dir = Path("data/processed/splits")

    print("Loading original dataset...")
    train_ds = PointCloudMultimodalDataset(str(splits_dir / "train.csv"), pc_dir, is_train=True)
    val_ds = PointCloudMultimodalDataset(str(splits_dir / "val.csv"), pc_dir, is_train=False)
    test_ds = PointCloudMultimodalDataset(str(splits_dir / "test.csv"), pc_dir, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_pointcloud)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)

    # ── Train both fusion variants ──
    all_results = {}

    for fusion_type, fusion_name in [("cross_attention", "MoE-CrossAttn"),
                                      ("bottleneck", "MoE-Bottleneck")]:
        print(f"\n{'='*60}")
        print(f"  {fusion_name} (original data, 223 samples)")
        print(f"{'='*60}")

        set_seed(42)
        model = MoEOriginal(
            feature_dim=n_features, fusion_type=fusion_type,
            num_experts=4, fused_dim=256, dropout=0.3,
            pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
        model.to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        fusion_params = sum(p.numel() for p in model.fusion.parameters())
        print(f"  Params: {n_params:,} (fusion: {fusion_params:,})")

        model, snapshots = train_model(
            model, train_loader, val_loader, device,
            f"checkpoints/moe_original_{fusion_type}",
            num_epochs=300, lr=1e-4, patience=30)

        # Single model evaluation
        preds, targets, gate_weights = evaluate_single(model, test_loader, device)
        metrics = compute_metrics(preds, targets)
        print(f"\n  {fusion_name} Single Model:")
        print(format_metrics(metrics, fusion_name))

        # Snapshot ensemble
        if len(snapshots) >= 2:
            all_preds = [preds]
            for state in snapshots:
                model.load_state_dict(state)
                model.to(device).eval()
                sp, _, _ = evaluate_single(model, test_loader, device)
                all_preds.append(sp)
            ens_preds = np.mean(all_preds, axis=0)
            ens_metrics = compute_metrics(ens_preds, targets)
            print(f"\n  {fusion_name} Snapshot Ensemble ({len(all_preds)} models):")
            print(format_metrics(ens_metrics, f"{fusion_name} (snapshot)"))
        else:
            ens_metrics = metrics

        best_metrics = ens_metrics if ens_metrics.get("avg_r2", -999) > metrics.get("avg_r2", -999) else metrics
        all_results[fusion_name] = {
            "total_params": n_params,
            "fusion_params": fusion_params,
            "single": {k: float(v) if isinstance(v, (float, np.floating)) else v
                       for k, v in metrics.items()},
            "snapshot_ensemble": {k: float(v) if isinstance(v, (float, np.floating)) else v
                                  for k, v in ens_metrics.items()},
            "best": {k: float(v) if isinstance(v, (float, np.floating)) else v
                     for k, v in best_metrics.items()},
            "gate_weights": gate_weights.mean(axis=0).tolist(),
        }

    # ── Fair Comparison ──
    print(f"\n{'='*60}")
    print("FAIR COMPARISON (all on original 223 samples)")
    print(f"{'='*60}")

    # Load previous results
    baselines = {}
    for name, path in [
        ("Baseline GNN", "results/gnn_results.json"),
        ("Phase 3 PointCloud", "results/pointcloud_results.json"),
        ("Hard Ensemble", "results/ensemble_phase23_results.json"),
    ]:
        try:
            with open(path) as f:
                data = json.load(f)
            baselines[name] = data.get("hard_ensemble", data.get("test_metrics", {}))
        except Exception:
            pass

    # Print comparison table
    header = f"{'Model':30s} {'Avg R²':>10s}"
    for p in TARGET_COLUMNS:
        header += f" {p:>8s}"
    print(header)
    print("-" * len(header))

    for name, m in baselines.items():
        row = f"{name:30s} {m.get('avg_r2', 0):>10.4f}"
        for p in TARGET_COLUMNS:
            row += f" {m.get(f'{p}_r2', 0):>8.4f}"
        print(row)

    for name, data in all_results.items():
        m = data["best"]
        row = f"{name:30s} {m.get('avg_r2', 0):>10.4f}"
        for p in TARGET_COLUMNS:
            row += f" {m.get(f'{p}_r2', 0):>8.4f}"
        print(row)

    # ── Visualize gating weights ──
    print("\nGenerating gating visualization...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        for fusion_name, data in all_results.items():
            gw = np.array(data["gate_weights"])  # (7, 4)
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(gw, cmap='YlOrRd', aspect='auto', vmin=0)

            expert_labels = ['Expert 1', 'Expert 2', 'Expert 3', 'Expert 4']
            prop_labels = [r'$\gamma_1$', r'$\gamma_2$', r'$G^E$', r'$H^E$',
                           r'$G_{mix}$', r'$H_{vap}$', r'$P$']

            ax.set_xticks(range(min(4, gw.shape[1])))
            ax.set_xticklabels(expert_labels[:gw.shape[1]])
            ax.set_yticks(range(7))
            ax.set_yticklabels(prop_labels)

            for i in range(gw.shape[0]):
                for j in range(gw.shape[1]):
                    color = 'white' if gw[i, j] > 0.35 else 'black'
                    ax.text(j, i, f'{gw[i, j]:.2f}', ha='center', va='center',
                           fontsize=10, fontweight='bold', color=color)

            plt.colorbar(im, ax=ax, shrink=0.8, label='Gating Weight')
            safe_name = fusion_name.replace(" ", "_").replace("-", "_").lower()
            ax.set_title(f'{fusion_name}: Property-Expert Routing', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'paper/figures/gating_{safe_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: paper/figures/gating_{safe_name}.png")
    except Exception as e:
        print(f"  Visualization failed: {e}")

    # Save
    with open("results/moe_original_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results/moe_original_results.json")


if __name__ == "__main__":
    main()
