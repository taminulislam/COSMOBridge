"""Fine-tune MoE+DMPNN on original data for PointCloud-style prediction.

Uses the MoE+DMPNN checkpoint (avg R²=0.672, trained on merged_v5) as
initialization, then fine-tunes on original 152 samples.

Key differences from Fix8 (which failed via catastrophic forgetting):
- Better starting point: MoE+DMPNN (0.672) vs Fix6 (0.650)
- D-MPNN backbone already captures better molecular representations
- Differential LR: backbone gets very low LR (1e-6), head gets higher (5e-5)
- This prevents forgetting while allowing the head to adapt
- Freeze strategy: freeze backbone for first 10 epochs, then unfreeze with low LR
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
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.dmpnn import DirectedMPNN
from src.models.fusion.multimodal_pointcloud import PointCloudFusion
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud


class EnhancedPointCloudFromMoE(nn.Module):
    """PointCloud model using D-MPNN, initialized from MoE+DMPNN checkpoint.

    Takes the shared components (PointNet, D-MPNN, Fusion) from the MoE
    and replaces the expert heads with a single deeper prediction head.
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


def load_moe_backbone(model, moe_checkpoint_path, device):
    """Load shared backbone weights from MoE+DMPNN checkpoint.

    Maps MoE weights → PointCloud model:
      pointnet.* → pointnet.*
      dmpnn.* → dmpnn.*
      fusion.* → fusion.*
    Skips: experts.*, gating.* (MoE-specific)
    """
    moe_state = torch.load(moe_checkpoint_path, map_location=device, weights_only=True)

    model_state = model.state_dict()
    loaded = 0
    skipped = 0

    for k, v in moe_state.items():
        # Transfer shared backbone weights
        if k.startswith(("pointnet.", "dmpnn.", "fusion.")):
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
            else:
                skipped += 1
        # Skip MoE-specific weights (experts, gating)

    model.load_state_dict(model_state)
    print(f"  Loaded {loaded} params from MoE+DMPNN backbone, skipped {skipped}")
    return model


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
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
    orig_splits = Path("data/processed/splits")

    # Load data
    print("Loading original datasets...")
    train_ds = PointCloudMultimodalDataset(str(orig_splits / "train.csv"), pc_dir, is_train=True)
    val_ds = PointCloudMultimodalDataset(str(orig_splits / "val.csv"), pc_dir, is_train=False)
    test_ds = PointCloudMultimodalDataset(str(orig_splits / "test.csv"), pc_dir, is_train=False)

    train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_pointcloud)
    val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)

    # Build model and load MoE backbone
    print("\nBuilding model...")
    model = EnhancedPointCloudFromMoE(
        feature_dim=len(FEATURE_COLUMNS), fused_dim=288, dropout=0.2)
    model = load_moe_backbone(model, "checkpoints/enhanced_moe/best.pt", device)
    model.to(device)
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_dir = Path("checkpoints/dmpnn_finetune")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate before fine-tuning (zero-shot from MoE backbone)
    preds_0, targets_0 = evaluate(model, test_ldr, device)
    metrics_0 = compute_metrics(preds_0, targets_0)
    print(f"\n  Before fine-tuning (MoE backbone + random head):")
    print(f"  avg R²={metrics_0['avg_r2']:.4f}, gamma1={metrics_0['gamma1_r2']:.4f}")

    # ══════════════════════════════════════════════════════════
    # PHASE 1: Freeze backbone, train head only (10 epochs)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 1: Freeze backbone, train prediction head (10 epochs)")
    print(f"{'='*60}")

    # Freeze backbone
    for name, p in model.named_parameters():
        if not name.startswith("prediction_head"):
            p.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params (head only): {n_trainable:,}")

    optimizer1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=5e-4, weight_decay=1e-4)

    best_loss, no_improve = float("inf"), 0
    for epoch in range(20):
        model.train()
        tl, n = 0, 0
        for batch in train_ldr:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            optimizer1.zero_grad()
            preds = model(point_cloud=batch["point_cloud"], features=batch["features"],
                          atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                          bond_features=batch["bond_features"], batch=batch["batch"])
            loss = ((preds - batch["targets"])**2).mean()
            loss.backward()
            optimizer1.step()
            tl += loss.item(); n += 1

        model.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for batch in val_ldr:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                preds = model(point_cloud=batch["point_cloud"], features=batch["features"],
                              atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                              bond_features=batch["bond_features"], batch=batch["batch"])
                vl += ((preds - batch["targets"])**2).mean().item()
                vn += 1
        avg_val = vl / max(vn, 1)
        if avg_val < best_loss:
            best_loss = avg_val; no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "phase1.pt")
        else:
            no_improve += 1
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/20 | Train: {tl/max(n,1):.4f} | Val: {avg_val:.4f} | Best: {best_loss:.4f}")
        if no_improve >= 10:
            print(f"  Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(ckpt_dir / "phase1.pt", map_location=device, weights_only=True))

    preds_1, targets_1 = evaluate(model, test_ldr, device)
    metrics_1 = compute_metrics(preds_1, targets_1)
    print(f"\n  After Phase 1 (head only):")
    print(format_metrics(metrics_1, "Phase 1"))

    # ══════════════════════════════════════════════════════════
    # PHASE 2: Unfreeze all, differential LR (backbone=1e-6, head=5e-5)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 2: Unfreeze all, differential LR")
    print(f"{'='*60}")

    # Unfreeze everything
    for p in model.parameters():
        p.requires_grad = True

    # Differential LR: backbone gets very low LR to prevent forgetting
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if name.startswith("prediction_head"):
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer2 = AdamW([
        {"params": backbone_params, "lr": 1e-6},   # very low for backbone
        {"params": head_params, "lr": 5e-5},         # moderate for head
    ], weight_decay=1e-4)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=100, eta_min=1e-7)

    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params (all): {n_total:,}")
    print(f"  Backbone LR: 1e-6, Head LR: 5e-5")

    best_loss2, no_improve2 = float("inf"), 0
    for epoch in range(100):
        model.train()
        tl, n = 0, 0
        for batch in train_ldr:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            optimizer2.zero_grad()
            preds = model(point_cloud=batch["point_cloud"], features=batch["features"],
                          atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                          bond_features=batch["bond_features"], batch=batch["batch"])
            loss = ((preds - batch["targets"])**2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer2.step()
            tl += loss.item(); n += 1
        scheduler2.step()

        model.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for batch in val_ldr:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                preds = model(point_cloud=batch["point_cloud"], features=batch["features"],
                              atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                              bond_features=batch["bond_features"], batch=batch["batch"])
                vl += ((preds - batch["targets"])**2).mean().item()
                vn += 1
        avg_val = vl / max(vn, 1)
        if avg_val < best_loss2:
            best_loss2 = avg_val; no_improve2 = 0
            torch.save(model.state_dict(), ckpt_dir / "phase2.pt")
        else:
            no_improve2 += 1
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/100 | Train: {tl/max(n,1):.4f} | Val: {avg_val:.4f} | "
                  f"Best: {best_loss2:.4f} | Pat: {no_improve2}/25")
        if no_improve2 >= 25:
            print(f"  Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(ckpt_dir / "phase2.pt", map_location=device, weights_only=True))

    # ══════════════════════════════════════════════════════════
    # FINAL EVALUATION
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")

    preds_2, targets_2 = evaluate(model, test_ldr, device)
    metrics_2 = compute_metrics(preds_2, targets_2)
    print(format_metrics(metrics_2, "D-MPNN Fine-tuned"))

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    prev = {}
    for name, path, key in [
        ("Chemprop", "results/chemprop_results.json", "test_metrics"),
        ("PointCloud (orig)", "results/pointcloud_results.json", None),
        ("MoE Fix6", "results/moe_fix6_results.json", "metrics"),
        ("MoE+DMPNN", "results/enhanced_models_results.json", "ENHANCED_MOE"),
    ]:
        try:
            data = json.load(open(path))
            if key == "ENHANCED_MOE":
                m = data["enhanced_moe"]["metrics"]
            elif key:
                m = data[key]
            else:
                for k in ['metrics', 'test_metrics']:
                    if k in data:
                        m = data[k]; break
            prev[name] = m
        except Exception:
            pass

    header = "  {:<15s}".format("Property")
    for name in prev:
        header += " {:>14s}".format(name)
    header += " {:>14s} {:>14s}".format("Phase1(head)", "Phase2(full)")
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<15s}".format(p)
        for name in prev:
            line += " {:14.4f}".format(prev[name].get(key, float('nan')))
        line += " {:14.4f} {:14.4f}".format(metrics_1[key], metrics_2[key])
        print(line)

    line = "  {:<15s}".format("AVERAGE")
    for name in prev:
        line += " {:14.4f}".format(prev[name].get('avg_r2', float('nan')))
    line += " {:14.4f} {:14.4f}".format(metrics_1['avg_r2'], metrics_2['avg_r2'])
    print(line)

    # Save
    results = {
        "model": "dmpnn_finetune",
        "description": "MoE+DMPNN backbone fine-tuned on original data. "
                       "Phase1: head only (5e-4). Phase2: differential LR (backbone=1e-6, head=5e-5).",
        "before_finetune": {k: float(v) if isinstance(v, (float, np.floating)) else v
                            for k, v in metrics_0.items()},
        "phase1_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                           for k, v in metrics_1.items()},
        "phase2_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                           for k, v in metrics_2.items()},
    }
    with open("results/dmpnn_finetune_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/dmpnn_finetune_results.json")


if __name__ == "__main__":
    main()
