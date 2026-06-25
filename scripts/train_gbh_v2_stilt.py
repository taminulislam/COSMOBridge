"""GBH v2 + STILT data strategy: maximum data for HyperNetwork fusion.

Hypothesis: GBH v2's T-dependent fusion benefits more from data diversity
than static fusion. With STILT's 10,950 samples (48x oversample + gamma1 mask),
the HyperNetwork sees 115 unique molecules × 8+ temperatures = 900+ unique
fusion configurations, vs 19×8=152 with original data only.

Architecture: GBH v2 PointCloud (low-rank bilinear + residual + deeper HyperNet)
Data: STILT strategy (gamma1 masked for ILThermo, 48x oversample)
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import TARGET_COLUMNS
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.gated_bilinear_hyper_v2 import GatedBilinearHyperFusionV2
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_joint import MergedDataset, collate_merged


class GBHv2PointCloud(nn.Module):
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
        self.prediction_head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 7))

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        pc_feat = self.pointnet(point_cloud)
        graph_feat = self.gnn.get_features(atom_features, edge_index, bond_features, batch)
        fused = self.fusion(pc_feat, graph_feat, features)
        return self.prediction_head(fused)


def prepare_stilt_data(merged_csv, output_dir):
    """STILT data strategy: mask gamma1, oversample 48x."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(merged_csv)
    orig = df[df["source"] == "original"].copy()
    ilth = df[df["source"] != "original"].copy()

    # Mask gamma1 for ILThermo
    n_masked = ilth["gamma1"].notna().sum()
    ilth["gamma1"] = np.nan
    print(f"  Masked {n_masked} ILThermo gamma1 values")

    # 48x oversample
    repeat = 48
    orig_rep = pd.concat([orig] * repeat, ignore_index=True)
    balanced = pd.concat([orig_rep, ilth], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  {len(orig)} × {repeat} = {len(orig_rep)} original ({len(orig_rep)/len(balanced)*100:.0f}%) "
          f"+ {len(ilth)} ILThermo → {len(balanced)} total")

    balanced.to_csv(output_dir / "train.csv", index=False)
    return len(balanced)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            preds = model(point_cloud=batch["point_cloud"], features=batch["features"],
                          atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                          bond_features=batch["bond_features"], batch=batch["batch"])
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def main():
    import os
    config = load_config("configs/default.yaml")
    seed = int(os.environ.get("SEED", "42"))
    set_seed(seed)
    print(f"[seed override] using seed={seed}")
    device = get_device(config)
    print(f"Device: {device}")

    pc_dir = "data/pipeline/point_clouds"
    merged_dir = Path("data/merged_v5")
    meta = json.load(open(merged_dir / "metadata.json"))
    features = meta["feature_columns"]

    # Prepare STILT data
    print("\nPreparing STILT data (gamma1 masked, 48x oversample)...")
    stilt_dir = Path("data/gbh_v2_stilt")
    stilt_dir.mkdir(parents=True, exist_ok=True)
    n_train = prepare_stilt_data(merged_dir / "splits/train.csv", stilt_dir)

    # Load datasets
    train_ds = MergedDataset(str(stilt_dir / "train.csv"), pc_dir, features, is_train=True)
    val_ds = MergedDataset(str(merged_dir / "splits/val.csv"), pc_dir, features, is_train=False)
    test_ds = MergedDataset(str(merged_dir / "splits/test.csv"), pc_dir, features, is_train=False)

    train_ldr = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_merged)
    val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)

    # Build model
    print(f"\n{'='*60}")
    print("GBH v2 + STILT Data (10,950 samples, 115 unique molecules)")
    print(f"{'='*60}")

    model = GBHv2PointCloud(feature_dim=len(features),
                             pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_fusion = sum(p.numel() for p in model.fusion.parameters())
    print(f"  Params: {n_params:,} (fusion: {n_fusion:,})")
    print(f"  Params/sample: {n_params/n_train:.0f} (vs {n_params/152:.0f} with original only)")
    print(f"  Unique fusion configs: 115 ILs × ~8 temps ≈ 900+ (vs 152 with original)")

    ckpt_dir = Path("checkpoints/gbh_v2_stilt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=60, eta_min=1e-6)

    # Train
    best_loss, no_improve = float("inf"), 0
    for epoch in range(200):
        model.train()
        tl, n = 0, 0
        for batch in train_ldr:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            optimizer.zero_grad()
            preds = model(point_cloud=batch["point_cloud"], features=batch["features"],
                          atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                          bond_features=batch["bond_features"], batch=batch["batch"])
            mask = ~torch.isnan(batch["targets"])
            safe = batch["targets"].clone(); safe[~mask] = 0.0
            loss = ((preds - safe)**2 * mask.float()).sum() / mask.float().sum().clamp(min=1)
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
                preds = model(point_cloud=batch["point_cloud"], features=batch["features"],
                              atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                              bond_features=batch["bond_features"], batch=batch["batch"])
                mask = ~torch.isnan(batch["targets"])
                safe = batch["targets"].clone(); safe[~mask] = 0.0
                vl += ((preds-safe)**2*mask.float()).sum().item()/mask.float().sum().clamp(min=1).item()
                vn += 1
        avg_val = vl / max(vn, 1)
        if avg_val < best_loss:
            best_loss = avg_val; no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_improve += 1
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/200 | Train: {tl/max(n,1):.4f} | "
                  f"Val: {avg_val:.4f} | Best: {best_loss:.4f} | Pat: {no_improve}/30")
        if no_improve >= 30:
            print(f"  Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True))

    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")

    preds, targets = evaluate(model, test_ldr, device)
    metrics = compute_metrics(preds, targets)
    print(format_metrics(metrics, "GBH v2 + STILT"))

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON: Does more data help GBH?")
    print(f"{'='*60}")

    prev = {}
    for name, path, key in [
        ("Chemprop", "results/chemprop_results.json", "test_metrics"),
        ("CrossAttn PC", "results/pointcloud_results.json", None),
        ("FiLM PC", "results/film_models_results.json", "FILM"),
        ("GBHv2 (orig)", "results/gbh_v2_results.json", "GBH_PC"),
        ("GBHv2 (MoE)", "results/gbh_v2_results.json", "GBH_MOE"),
        ("MoE Fix6", "results/moe_fix6_results.json", "metrics"),
        ("STILT", "results/chemprop_tuned_results.json", "STILT_C"),
    ]:
        try:
            data = json.load(open(path))
            if key == "FILM": m = data["film_pointcloud"]["metrics"]
            elif key == "GBH_PC": m = data["gbh_v2_pointcloud"]["metrics"]
            elif key == "GBH_MOE": m = data["gbh_v2_moe"]["metrics"]
            elif key == "STILT_C": m = data.get("C: full mask, 48x OS", {}).get("metrics", {})
            elif key: m = data[key]
            else:
                for k in ['metrics', 'test_metrics']:
                    if k in data: m = data[k]; break
            prev[name] = m
        except: pass

    prev["GBHv2+STILT"] = {k: float(v) if isinstance(v, (float, np.floating)) else v
                            for k, v in metrics.items()}

    header = "  {:<14s}".format("Property")
    for name in prev:
        header += " {:>12s}".format(name[:12])
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<14s}".format(p)
        for name in prev:
            line += " {:12.4f}".format(prev[name].get(key, float('nan')))
        print(line)

    line = "  {:<14s}".format("AVERAGE")
    for name in prev:
        line += " {:12.4f}".format(prev[name].get('avg_r2', float('nan')))
    print(line)

    # Data scaling analysis
    print(f"\n  Data scaling effect on GBH v2:")
    gbh_orig = prev.get("GBHv2 (orig)", {}).get("avg_r2", 0)
    gbh_moe = prev.get("GBHv2 (MoE)", {}).get("avg_r2", 0)
    gbh_stilt = metrics["avg_r2"]
    print(f"    Original (152 samples):    avg R² = {gbh_orig:.4f}")
    print(f"    MoE balanced (3,806):      avg R² = {gbh_moe:.4f}")
    print(f"    STILT (10,950):            avg R² = {gbh_stilt:.4f}")
    print(f"    Improvement orig→STILT:    +{gbh_stilt - gbh_orig:.4f}")

    # Save
    results = {
        "model": "gbh_v2_stilt",
        "description": "GBH v2 (low-rank bilinear + residual + deeper HyperNet) "
                       "with STILT data strategy (gamma1 masked, 48x oversample, 10950 samples)",
        "n_params": n_params,
        "n_train": n_train,
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items()},
    }
    results["seed"] = seed
    out_path = f"results/gbh_v2_stilt_results_s{seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
