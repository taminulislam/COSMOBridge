"""MoE-A + Fix8: Two-phase training.

Phase 1: Train on merged_v5 with balanced sampling (same as Fix6).
         This learns H_vap, P, H_E from ILThermo + baseline for all properties.

Phase 2: Fine-tune ENTIRE model on original data only with lower LR.
         This adapts all representations (backbone + experts + gating)
         to the original domain, improving gamma1/gamma2 without losing
         the ILThermo-informed features for H_vap/P.

Key differences from Fix2 (which failed):
- Fix2 used merged_v3 (misaligned normalization) -> Fix8 uses merged_v5 (aligned)
- Fix2 froze backbone, unfroze only 2,268 output params -> Fix8 unfreezes everything
- Fix2 fine-tuned only gamma1 -> Fix8 fine-tunes all 7 properties
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR

from src.utils.config import load_config, get_device, set_seed
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.moe import ExpertHead, PropertyConditionedGating
from src.models.fusion.multimodal_pointcloud import PointCloudFusion
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_joint import MergedDataset, collate_merged
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud

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


def make_balanced_sampler(csv_path):
    df = pd.read_csv(csv_path)
    is_original = (df["source"] == "original").values
    n_orig = is_original.sum()
    n_ilth = len(df) - n_orig
    weights = np.where(is_original, 0.5 / max(n_orig, 1), 0.5 / max(n_ilth, 1))
    weights = torch.from_numpy(weights).double()
    sampler = WeightedRandomSampler(weights, num_samples=len(df), replacement=True)
    print(f"  Balanced sampler: {n_orig} original, {n_ilth} ILThermo, "
          f"oversample {n_ilth/max(n_orig,1):.1f}x")
    return sampler


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

    ckpt_dir = Path("checkpoints/moe_fix8")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # PHASE 1: Train on merged_v5 with balanced sampling
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 1: Train on merged_v5 (balanced sampling)")
    print(f"{'='*60}")

    train_csv = str(merged_dir / "splits/train.csv")
    train_ds = MergedDataset(train_csv, pc_dir, merged_features, is_train=True)
    val_ds = MergedDataset(str(merged_dir / "splits/val.csv"), pc_dir, merged_features, is_train=False)
    test_ds = MergedDataset(str(merged_dir / "splits/test.csv"), pc_dir, merged_features, is_train=False)

    sampler = make_balanced_sampler(train_csv)
    train_ldr = DataLoader(train_ds, batch_size=64, sampler=sampler, collate_fn=collate_merged)
    val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)

    model = MoEA(feature_dim=len(merged_features),
                 pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
    model.to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=60, eta_min=1e-6)

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
            torch.save(model.state_dict(), ckpt_dir / "phase1.pt")
        else:
            no_improve += 1
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/200 | Train: {tl/max(n,1):.4f} | "
                  f"Val: {avg_val:.4f} | Best: {best_loss:.4f} | Pat: {no_improve}/25")
        if no_improve >= 25:
            print(f"  Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(ckpt_dir / "phase1.pt", map_location=device, weights_only=True))

    # Evaluate after Phase 1
    preds_p1, targets_p1, _ = evaluate_single(model, test_ldr, device)
    metrics_p1 = compute_metrics(preds_p1, targets_p1)
    print(f"\n  Phase 1 results: avg R²={metrics_p1['avg_r2']:.4f}, "
          f"gamma1={metrics_p1['gamma1_r2']:.4f}")

    # ══════════════════════════════════════════════════════════
    # PHASE 2: Fine-tune ENTIRE model on original data only
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 2: Fine-tune on original data (all params, lower LR)")
    print(f"{'='*60}")

    orig_splits = Path("data/processed/splits")
    train_orig = PointCloudMultimodalDataset(str(orig_splits / "train.csv"), pc_dir, is_train=True)
    val_orig = PointCloudMultimodalDataset(str(orig_splits / "val.csv"), pc_dir, is_train=False)
    test_orig = PointCloudMultimodalDataset(str(orig_splits / "test.csv"), pc_dir, is_train=False)

    train_ldr_orig = DataLoader(train_orig, batch_size=32, shuffle=True, collate_fn=collate_pointcloud)
    val_ldr_orig = DataLoader(val_orig, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)
    test_ldr_orig = DataLoader(test_orig, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)

    print(f"  Original train: {len(train_orig)}, val: {len(val_orig)}, test: {len(test_orig)}")

    # Lower LR for fine-tuning, all params unfrozen
    optimizer2 = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=50, eta_min=1e-7)

    best_loss2, no_improve2 = float("inf"), 0
    for epoch in range(50):
        model.train()
        tl, n = 0, 0
        for batch in train_ldr_orig:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            optimizer2.zero_grad()
            preds, aux = model(point_cloud=batch["point_cloud"], features=batch["features"],
                               atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                               bond_features=batch["bond_features"], batch=batch["batch"])
            # Train on ALL 7 properties (original data has all)
            loss = ((preds - batch["targets"])**2).mean() + aux["load_balance_loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer2.step()
            tl += loss.item(); n += 1
        scheduler2.step()

        # Validate on original val set
        model.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for batch in val_ldr_orig:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                preds, _ = model(point_cloud=batch["point_cloud"], features=batch["features"],
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

        if epoch % 10 == 0:
            print(f"  FT Epoch {epoch:3d}/50 | Train: {tl/max(n,1):.4f} | "
                  f"Val: {avg_val:.4f} | Best: {best_loss2:.4f} | Pat: {no_improve2}/15")
        if no_improve2 >= 15:
            print(f"  Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(ckpt_dir / "phase2.pt", map_location=device, weights_only=True))

    # ══════════════════════════════════════════════════════════
    # FINAL EVALUATION
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FINAL EVALUATION (after Phase 2 fine-tuning)")
    print(f"{'='*60}")

    # Evaluate on original test set (same as PointCloud uses)
    preds_p2, targets_p2, gate_weights = evaluate_single(model, test_ldr_orig, device)
    metrics_p2 = compute_metrics(preds_p2, targets_p2)
    print(format_metrics(metrics_p2, "MoE-A Fix8 (two-phase)"))

    # Also evaluate on merged test set for comparison
    preds_m, targets_m, _ = evaluate_single(model, test_ldr, device)
    metrics_merged = compute_metrics(preds_m, targets_m)

    print(f"\n  Per-property R² (original test):")
    for p in TARGET_COLUMNS:
        p1 = metrics_p1.get(f'{p}_r2', float('nan'))
        p2 = metrics_p2[f'{p}_r2']
        delta = p2 - p1
        sign = "+" if delta > 0 else ""
        print(f"    {p:15s} Phase1={p1:7.4f}  Phase2={p2:7.4f}  ({sign}{delta:.4f})")
    p1_avg = metrics_p1['avg_r2']
    p2_avg = metrics_p2['avg_r2']
    delta = p2_avg - p1_avg
    sign = "+" if delta > 0 else ""
    print(f"    {'AVERAGE':15s} Phase1={p1_avg:7.4f}  Phase2={p2_avg:7.4f}  ({sign}{delta:.4f})")

    # Compare with Fix6 and PointCloud
    print(f"\n{'='*60}")
    print("COMPARISON: Fix6 vs Fix8 vs PointCloud")
    print(f"{'='*60}")

    prev = {}
    for name, path in [("Fix6", "results/moe_fix6_results.json"),
                        ("PointCloud", "results/pointcloud_results.json")]:
        try:
            data = json.load(open(path))
            m = data.get("metrics", data.get("test_metrics", data.get("single_model", {})))
            prev[name] = m
        except Exception:
            pass

    header = "  {:<15s}".format("Property")
    for name in prev:
        header += " {:>10s}".format(name)
    header += " {:>10s} {:>10s}".format("Fix8", "Fix8-Fix6")
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = p + "_r2"
        f8 = metrics_p2[key]
        f6 = prev.get("Fix6", {}).get(key, float("nan"))
        delta = f8 - f6
        sign = "+" if delta > 0 else ""
        line = "  {:<15s}".format(p)
        for name in prev:
            line += " {:10.4f}".format(prev[name].get(key, float("nan")))
        line += " {:10.4f} {:>9s}".format(f8, "{}{:.4f}".format(sign, delta))
        print(line)

    f8_avg = metrics_p2["avg_r2"]
    f6_avg = prev.get("Fix6", {}).get("avg_r2", float("nan"))
    delta = f8_avg - f6_avg
    sign = "+" if delta > 0 else ""
    line = "  {:<15s}".format("AVERAGE")
    for name in prev:
        line += " {:10.4f}".format(prev[name].get("avg_r2", float("nan")))
    line += " {:10.4f} {:>9s}".format(f8_avg, "{}{:.4f}".format(sign, delta))
    print(line)

    # ── Save ──
    results = {
        "fix": "two_phase_training",
        "description": "Phase1: merged_v5 + balanced sampling (like Fix6). "
                       "Phase2: fine-tune entire model on original data only (LR=2e-5, 50 epochs).",
        "phase1_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                           for k, v in metrics_p1.items()},
        "phase2_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                           for k, v in metrics_p2.items()},
        "merged_test_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                                for k, v in metrics_merged.items()},
        "gate_weights": gate_weights.mean(axis=0).tolist(),
    }
    with open("results/moe_fix8_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/moe_fix8_results.json")


if __name__ == "__main__":
    main()
