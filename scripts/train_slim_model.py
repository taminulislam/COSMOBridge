"""Slim multimodal model: D-MPNN + PointNet + concat fusion.

Inspired by Chemprop's success principle: fewer params, deeper FFN.
Replaces 863K cross-attention fusion with simple concatenation + deeper FFN.

Target: ~400K total params (vs 1.3M current, vs 300K Chemprop)

Architecture:
  PointNet (COSMO surface) → 128D
  D-MPNN (molecular graph) → 128D
  Thermo features → 25D
  ────────────────────────────
  Concat → 281D → FFN(256→128→7)

Trains two versions:
A. Slim PointCloud: on original 152 samples
B. Slim MoE: on merged_v5 with balanced sampling
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
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud
from scripts.train_joint import MergedDataset, collate_merged


class SlimPointCloudModel(nn.Module):
    """Lightweight multimodal model with concat fusion.

    ~400K params vs 1.3M in original PointCloud model.
    """

    def __init__(self, feature_dim=25, hidden=128, dropout=0.2):
        super().__init__()
        # Smaller encoders (128D instead of 256/300D)
        self.pointnet = PointNetEncoder(in_channels=7, feature_dim=hidden, dropout=dropout)
        self.dmpnn = DirectedMPNN(
            atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=hidden, num_layers=3, dropout=dropout, num_targets=0)

        concat_dim = hidden + hidden + feature_dim  # 128 + 128 + 25 = 281

        # Deeper FFN (where the real capacity goes)
        self.ffn = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 7),
        )

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        pc_feat = self.pointnet(point_cloud)
        graph_feat = self.dmpnn.get_features(atom_features, edge_index, bond_features, batch)
        combined = torch.cat([pc_feat, graph_feat, features], dim=-1)
        return self.ffn(combined)


class SlimMoE(nn.Module):
    """Lightweight MoE with concat fusion and smaller experts."""

    def __init__(self, feature_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.pointnet = PointNetEncoder(in_channels=7, feature_dim=hidden, dropout=dropout)
        self.dmpnn = DirectedMPNN(
            atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=hidden, num_layers=3, dropout=dropout, num_targets=0)

        concat_dim = hidden + hidden + feature_dim

        # Project concat to fused dim
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 4 lightweight expert heads
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 7),
            ) for _ in range(4)])

        # Simple gating (no property conditioning — saves params)
        self.gate = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4 * 7),  # 4 experts × 7 properties
        )

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        pc_feat = self.pointnet(point_cloud)
        graph_feat = self.dmpnn.get_features(atom_features, edge_index, bond_features, batch)
        combined = torch.cat([pc_feat, graph_feat, features], dim=-1)
        h = self.proj(combined)

        expert_preds = torch.stack([e(h) for e in self.experts], dim=2)  # (B, 7, 4)
        gate_logits = self.gate(h).view(-1, 7, 4)  # (B, 7, 4)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        predictions = (expert_preds * gate_weights).sum(dim=2)

        # Load balance loss
        avg_w = gate_weights.mean(dim=(0, 1))
        lb_loss = 0.01 * ((avg_w - 0.25) ** 2).sum()

        return predictions, {"load_balance_loss": lb_loss, "gate_weights": gate_weights.detach()}


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


def train_loop(model, train_ldr, val_ldr, device, ckpt_path, is_moe=False,
               num_epochs=200, lr=1e-4, patience=25):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=60, eta_min=1e-6)

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


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")
    pc_dir = "data/pipeline/point_clouds"

    # ══════════════════════════════════════════════════════════
    # MODEL A: Slim PointCloud (original data)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("MODEL A: Slim PointCloud (D-MPNN + PointNet + Concat)")
    print(f"{'='*60}")

    orig_splits = Path("data/processed/splits")
    train_ds = PointCloudMultimodalDataset(str(orig_splits / "train.csv"), pc_dir, is_train=True)
    val_ds = PointCloudMultimodalDataset(str(orig_splits / "val.csv"), pc_dir, is_train=False)
    test_ds = PointCloudMultimodalDataset(str(orig_splits / "test.csv"), pc_dir, is_train=False)
    train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_pointcloud)
    val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)

    model_a = SlimPointCloudModel(feature_dim=len(FEATURE_COLUMNS), hidden=128, dropout=0.2)
    model_a.to(device)
    n_a = sum(p.numel() for p in model_a.parameters())
    print(f"  Params: {n_a:,} (vs Chemprop ~300K, vs original PointCloud 3.7M)")

    ckpt_a = Path("checkpoints/slim_pointcloud"); ckpt_a.mkdir(parents=True, exist_ok=True)
    model_a = train_loop(model_a, train_ldr, val_ldr, device, ckpt_a / "best.pt", is_moe=False)

    preds_a, targets_a = evaluate(model_a, test_ldr, device, is_moe=False)
    metrics_a = compute_metrics(preds_a, targets_a)
    print(f"\n{format_metrics(metrics_a, 'Slim PointCloud')}")

    # ══════════════════════════════════════════════════════════
    # MODEL B: Slim MoE (merged_v5 + balanced sampling)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("MODEL B: Slim MoE (D-MPNN + PointNet + Concat + MoE)")
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

    model_b = SlimMoE(feature_dim=len(merged_features), hidden=128, dropout=0.2)
    model_b.to(device)
    n_b = sum(p.numel() for p in model_b.parameters())
    print(f"  Params: {n_b:,}")

    ckpt_b = Path("checkpoints/slim_moe"); ckpt_b.mkdir(parents=True, exist_ok=True)
    model_b = train_loop(model_b, train_ldr_m, val_ldr_m, device, ckpt_b / "best.pt", is_moe=True)

    preds_b, targets_b = evaluate(model_b, test_ldr_m, device, is_moe=True)
    metrics_b = compute_metrics(preds_b, targets_b)
    print(f"\n{format_metrics(metrics_b, 'Slim MoE')}")

    # ══════════════════════════════════════════════════════════
    # COMPARISON
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    prev = {}
    for name, path, key in [
        ("Chemprop", "results/chemprop_results.json", "test_metrics"),
        ("PointCloud", "results/pointcloud_results.json", None),
        ("MoE Fix6", "results/moe_fix6_results.json", "metrics"),
        ("MoE+DMPNN", "results/enhanced_models_results.json", "EMOE"),
    ]:
        try:
            data = json.load(open(path))
            if key == "EMOE":
                m = data["enhanced_moe"]["metrics"]
            elif key:
                m = data[key]
            else:
                for k in ['metrics', 'test_metrics']:
                    if k in data: m = data[k]; break
            prev[name] = m
        except:
            pass

    header = "  {:<15s}".format("Property")
    for name in prev:
        header += " {:>11s}".format(name)
    header += " {:>11s} {:>11s}".format("Slim PC", "Slim MoE")
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<15s}".format(p)
        for name in prev:
            line += " {:11.4f}".format(prev[name].get(key, float('nan')))
        line += " {:11.4f} {:11.4f}".format(metrics_a[key], metrics_b[key])
        print(line)

    line = "  {:<15s}".format("AVERAGE")
    for name in prev:
        line += " {:11.4f}".format(prev[name].get('avg_r2', float('nan')))
    line += " {:11.4f} {:11.4f}".format(metrics_a['avg_r2'], metrics_b['avg_r2'])
    print(line)

    print(f"\n  Param counts: Slim PC={n_a:,}, Slim MoE={n_b:,}, Chemprop=~300K")

    # Save
    results = {
        "slim_pointcloud": {
            "n_params": n_a,
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                        for k, v in metrics_a.items()},
        },
        "slim_moe": {
            "n_params": n_b,
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                        for k, v in metrics_b.items()},
        },
    }
    with open("results/slim_models_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/slim_models_results.json")


if __name__ == "__main__":
    main()
