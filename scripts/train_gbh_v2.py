"""Train GBH v2: Low-rank bilinear + residual + deeper HyperNetwork.

Trains PointCloud and MoE variants.
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
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.gated_bilinear_hyper_v2 import GatedBilinearHyperFusionV2
from src.models.fusion.moe import ExpertHead, PropertyConditionedGating
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud
from scripts.train_joint import MergedDataset, collate_merged


class GBHv2PointCloud(nn.Module):
    def __init__(self, feature_dim=25, pretrained_gnn_path=None):
        super().__init__()
        dropout = 0.3
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
                print(f"  Loaded pre-trained GNN: {len(gnn_state)} params")

        self.fusion = GatedBilinearHyperFusionV2(
            pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
            fused_dim=256, rank=32, thermo_dim=5, hyper_hidden=64, dropout=dropout)

        self.prediction_head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 7))

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        pc_feat = self.pointnet(point_cloud)
        graph_feat = self.gnn.get_features(atom_features, edge_index, bond_features, batch)
        fused = self.fusion(pc_feat, graph_feat, features)
        return self.prediction_head(fused)


class GBHv2MoE(nn.Module):
    def __init__(self, feature_dim, pretrained_gnn_path=None):
        super().__init__()
        dropout = 0.3
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
                print(f"  Loaded pre-trained GNN: {len(gnn_state)} params")

        self.fusion = GatedBilinearHyperFusionV2(
            pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
            fused_dim=256, rank=32, thermo_dim=5, hyper_hidden=64, dropout=dropout)

        self.experts = nn.ModuleList([
            ExpertHead(256, hidden_dim=128, num_targets=7, dropout=dropout)
            for _ in range(4)])
        self.gating = PropertyConditionedGating(
            input_dim=256, num_experts=4, num_properties=7, hidden_dim=64)

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        pc_feat = self.pointnet(point_cloud)
        graph_feat = self.gnn.get_features(atom_features, edge_index, bond_features, batch)
        fused = self.fusion(pc_feat, graph_feat, features)
        expert_preds = torch.stack([e(fused) for e in self.experts], dim=2)
        gate_weights, lb_loss = self.gating(fused)
        predictions = (expert_preds * gate_weights).sum(dim=2)
        return predictions, {"load_balance_loss": lb_loss, "gate_weights": gate_weights.detach()}


def evaluate(model, loader, device, is_moe=False):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
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


def train_loop(model, train_ldr, val_ldr, device, ckpt_path, is_moe=False,
               num_epochs=200, lr=1e-4, patience=30):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=60, eta_min=1e-6)

    best_loss, no_improve = float("inf"), 0
    for epoch in range(num_epochs):
        model.train()
        tl, n = 0, 0
        for batch in train_ldr:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
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
                    if isinstance(v, torch.Tensor): batch[k] = v.to(device)
                if is_moe:
                    preds, _ = model(point_cloud=batch["point_cloud"], features=batch["features"],
                                     atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                                     bond_features=batch["bond_features"], batch=batch["batch"])
                    mask = ~torch.isnan(batch["targets"])
                    safe = batch["targets"].clone(); safe[~mask] = 0.0
                    vl += ((preds-safe)**2*mask.float()).sum().item()/mask.float().sum().clamp(min=1).item()
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


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")
    pc_dir = "data/pipeline/point_clouds"
    gnn_path = "checkpoints/transfer/pretrained.pt"

    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("MODEL A: GBH v2 PointCloud (low-rank bilinear + residual)")
    print(f"{'='*60}")

    orig_splits = Path("data/processed/splits")
    train_ds = PointCloudMultimodalDataset(str(orig_splits / "train.csv"), pc_dir, is_train=True)
    val_ds = PointCloudMultimodalDataset(str(orig_splits / "val.csv"), pc_dir, is_train=False)
    test_ds = PointCloudMultimodalDataset(str(orig_splits / "test.csv"), pc_dir, is_train=False)
    train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_pointcloud)
    val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)

    model_a = GBHv2PointCloud(feature_dim=len(FEATURE_COLUMNS), pretrained_gnn_path=gnn_path)
    model_a.to(device)
    n_a = sum(p.numel() for p in model_a.parameters())
    n_fusion = sum(p.numel() for p in model_a.fusion.parameters())
    n_hyper = sum(p.numel() for p in model_a.fusion.hypernet.parameters())
    n_bilinear = sum(p.numel() for p in model_a.fusion.bilinear.parameters())
    print(f"  Total params: {n_a:,}")
    print(f"  Fusion params: {n_fusion:,} (HyperNet: {n_hyper:,}, Bilinear: {n_bilinear:,})")
    print(f"  Params/sample: {n_a/152:.0f}")

    ckpt_a = Path("checkpoints/gbh_v2_pointcloud"); ckpt_a.mkdir(parents=True, exist_ok=True)
    model_a = train_loop(model_a, train_ldr, val_ldr, device, ckpt_a / "best.pt",
                          is_moe=False, patience=30)

    preds_a, targets_a = evaluate(model_a, test_ldr, device, is_moe=False)
    metrics_a = compute_metrics(preds_a, targets_a)
    print(f"\n{format_metrics(metrics_a, 'GBH v2 PointCloud')}")

    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("MODEL B: GBH v2 MoE (merged_v5 + balanced sampling)")
    print(f"{'='*60}")

    merged_dir = Path("data/merged_v5")
    meta = json.load(open(merged_dir / "metadata.json"))
    merged_features = meta["feature_columns"]

    train_csv = str(merged_dir / "splits/train.csv")
    train_ds_m = MergedDataset(train_csv, pc_dir, merged_features, is_train=True)
    val_ds_m = MergedDataset(str(merged_dir / "splits/val.csv"), pc_dir, merged_features, is_train=False)
    test_ds_m = MergedDataset(str(merged_dir / "splits/test.csv"), pc_dir, merged_features, is_train=False)

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

    model_b = GBHv2MoE(feature_dim=len(merged_features), pretrained_gnn_path=gnn_path)
    model_b.to(device)
    n_b = sum(p.numel() for p in model_b.parameters())
    print(f"  Total params: {n_b:,}")

    ckpt_b = Path("checkpoints/gbh_v2_moe"); ckpt_b.mkdir(parents=True, exist_ok=True)
    model_b = train_loop(model_b, train_ldr_m, val_ldr_m, device, ckpt_b / "best.pt",
                          is_moe=True, patience=30)

    preds_b, targets_b = evaluate(model_b, test_ldr_m, device, is_moe=True)
    metrics_b = compute_metrics(preds_b, targets_b)
    print(f"\n{format_metrics(metrics_b, 'GBH v2 MoE')}")

    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("COMPARISON: All Fusion Mechanisms")
    print(f"{'='*60}")

    prev = {}
    for name, path, key in [
        ("Chemprop", "results/chemprop_results.json", "test_metrics"),
        ("CrossAttn", "results/pointcloud_results.json", None),
        ("FiLM", "results/film_models_results.json", "FILM"),
        ("GBH v1", "results/gbh_results.json", "GBH_V1"),
        ("MoE Fix6", "results/moe_fix6_results.json", "metrics"),
    ]:
        try:
            data = json.load(open(path))
            if key == "FILM": m = data["film_pointcloud"]["metrics"]
            elif key == "GBH_V1": m = data["gbh_pointcloud"]["metrics"]
            elif key: m = data[key]
            else:
                for k in ['metrics','test_metrics']:
                    if k in data: m = data[k]; break
            prev[name] = m
        except: pass

    header = "  {:<10s}".format("Property")
    for name in prev:
        header += " {:>10s}".format(name[:10])
    header += " {:>10s} {:>10s}".format("GBHv2 PC", "GBHv2 MoE")
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<10s}".format(p)
        for name in prev:
            line += " {:10.4f}".format(prev[name].get(key, float('nan')))
        line += " {:10.4f} {:10.4f}".format(metrics_a[key], metrics_b[key])
        print(line)

    line = "  {:<10s}".format("AVERAGE")
    for name in prev:
        line += " {:10.4f}".format(prev[name].get('avg_r2', float('nan')))
    line += " {:10.4f} {:10.4f}".format(metrics_a['avg_r2'], metrics_b['avg_r2'])
    print(line)

    print(f"\n  Fusion params: GBH v1={sum(p.numel() for p in GBHv2PointCloud(len(FEATURE_COLUMNS)).fusion.parameters()):,}, "
          f"GBH v2={n_fusion:,}")

    # Save
    results = {
        "gbh_v2_pointcloud": {
            "n_params": n_a, "n_fusion_params": n_fusion,
            "n_hypernet_params": n_hyper, "n_bilinear_params": n_bilinear,
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                        for k, v in metrics_a.items()},
        },
        "gbh_v2_moe": {
            "n_params": n_b,
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                        for k, v in metrics_b.items()},
        },
    }
    with open("results/gbh_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/gbh_v2_results.json")


if __name__ == "__main__":
    main()
