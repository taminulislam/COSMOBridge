"""MoE-A + Fix7: Fix6 + augmentation on oversampled original data.

Builds on Fix6 (aggressive filter + balanced sampling) and adds:
- Gaussian noise on tabular features for original samples (std=0.05)
- Enhanced point cloud jitter (translation + scaling + extra rotation axes)
- Feature dropout (randomly zero 10% of surface descriptors)

This creates diversity in the oversampled original data (seen ~24x per epoch)
to reduce overfitting and improve generalization on gamma1/gamma2.
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.utils.config import load_config, get_device, set_seed
from src.data.graph_builder import smiles_to_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.moe import ExpertHead, PropertyConditionedGating
from src.models.fusion.multimodal_pointcloud import PointCloudFusion
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_joint import collate_merged

TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]

# Thermo features (indices 0-4) get lighter noise; surface features (5-24) get heavier
THERMO_FEATURE_COUNT = 5  # temperature, x1, inv_temperature, temp_squared, temp_cubed


class AugmentedMergedDataset(Dataset):
    """MergedDataset with augmentation for original samples.

    When is_train=True and the sample is from original data:
    - Tabular features: Gaussian noise (small for thermo, larger for surface)
    - Point cloud: random jitter, scale perturbation, multi-axis rotation
    - Feature dropout: randomly zero 10% of surface descriptors
    - Atom features: small Gaussian noise on continuous features
    """

    def __init__(self, csv_path, pc_dir, feature_columns, is_train=True, n_points=1024,
                 feature_noise_std=0.05, surface_noise_std=0.10,
                 pc_jitter_std=0.02, pc_scale_range=(0.95, 1.05),
                 feature_dropout=0.10):
        self.df = pd.read_csv(csv_path)
        self.pc_dir = Path(pc_dir)
        self.feature_columns = feature_columns
        self.is_train = is_train
        self.n_points = n_points

        # Augmentation params
        self.feature_noise_std = feature_noise_std
        self.surface_noise_std = surface_noise_std
        self.pc_jitter_std = pc_jitter_std
        self.pc_scale_range = pc_scale_range
        self.feature_dropout = feature_dropout

        # Track which samples are original (for selective augmentation)
        self.is_original = (self.df["source"] == "original").values

        # Point cloud index
        idx_path = self.pc_dir / "index.csv"
        self.pc_index = {}
        if idx_path.exists():
            idx_df = pd.read_csv(idx_path)
            self.pc_index = dict(zip(idx_df["smiles"], idx_df["filename"]))

        # Pre-build graphs
        self.graphs = {}
        for smi in self.df["smiles"].unique():
            try:
                self.graphs[smi] = smiles_to_graph(smi)
            except Exception:
                self.graphs[smi] = None

        n_pc = sum(1 for s in self.df["smiles"].unique() if s in self.pc_index)
        n_g = sum(1 for s in self.df["smiles"].unique() if self.graphs.get(s) is not None)
        n_orig = self.is_original.sum()
        print(f"  {csv_path}: {len(self.df)} rows, {n_pc} with PC, {n_g} with graphs")
        print(f"  Augmentation enabled for {n_orig} original samples "
              f"(feature_noise={feature_noise_std}, surface_noise={surface_noise_std}, "
              f"pc_jitter={pc_jitter_std}, feat_dropout={feature_dropout})")

    def __len__(self):
        return len(self.df)

    def _load_pc(self, smiles, augment=False):
        fn = self.pc_index.get(smiles)
        if fn:
            path = self.pc_dir / fn
            if path.exists():
                pts = np.load(path)["points"]
                if len(pts) >= self.n_points:
                    idx = np.random.choice(len(pts), self.n_points, replace=False) if self.is_train else np.arange(self.n_points)
                    pts = pts[idx]
                else:
                    extra = np.random.choice(len(pts), self.n_points - len(pts), replace=True)
                    pts = np.concatenate([pts, pts[extra]])

                if self.is_train:
                    # Standard z-axis rotation (all training samples)
                    angle = np.random.uniform(0, 2 * np.pi)
                    c, s = np.cos(angle), np.sin(angle)
                    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    pts[:, :3] = pts[:, :3] @ R.T
                    pts[:, 3:6] = pts[:, 3:6] @ R.T

                    if augment:
                        # Additional augmentation for original samples
                        # Random jitter on coordinates
                        pts[:, :3] += np.random.normal(0, self.pc_jitter_std, pts[:, :3].shape)

                        # Random scale perturbation
                        scale = np.random.uniform(*self.pc_scale_range)
                        pts[:, :3] *= scale

                        # Additional rotation around x or y axis (small angle)
                        angle2 = np.random.normal(0, 0.1)  # ~6 degrees std
                        axis = np.random.choice([0, 1])
                        if axis == 0:  # x-axis
                            c2, s2 = np.cos(angle2), np.sin(angle2)
                            R2 = np.array([[1, 0, 0], [0, c2, -s2], [0, s2, c2]])
                        else:  # y-axis
                            c2, s2 = np.cos(angle2), np.sin(angle2)
                            R2 = np.array([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
                        pts[:, :3] = pts[:, :3] @ R2.T
                        pts[:, 3:6] = pts[:, 3:6] @ R2.T

                        # Small noise on ESP values (column 6)
                        pts[:, 6] += np.random.normal(0, 0.02, pts[:, 6].shape)

                return torch.tensor(pts, dtype=torch.float32)
        return torch.zeros(self.n_points, 7)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row["smiles"]
        augment = self.is_train and self.is_original[idx]

        features = np.array([row[c] for c in self.feature_columns], dtype=np.float32)

        if augment:
            # Gaussian noise on features (lighter on thermo, heavier on surface)
            noise = np.zeros_like(features)
            noise[:THERMO_FEATURE_COUNT] = np.random.normal(
                0, self.feature_noise_std, THERMO_FEATURE_COUNT)
            noise[THERMO_FEATURE_COUNT:] = np.random.normal(
                0, self.surface_noise_std, len(features) - THERMO_FEATURE_COUNT)
            features = features + noise

            # Feature dropout: randomly zero some surface descriptors
            if self.feature_dropout > 0:
                drop_mask = np.random.random(len(features) - THERMO_FEATURE_COUNT) < self.feature_dropout
                features[THERMO_FEATURE_COUNT:][drop_mask] = 0.0

        features = torch.tensor(features, dtype=torch.float32)
        pc = self._load_pc(smiles, augment=augment)

        g = self.graphs.get(smiles)
        if g is not None:
            af = np.array(g["atom_features"], dtype=np.float32)
            if augment:
                # Small noise on continuous atom features
                af += np.random.normal(0, 0.02, af.shape)
            af = torch.tensor(af, dtype=torch.float32)
            ei = torch.tensor(g["edge_index"], dtype=torch.long)
            bf = torch.tensor(g["bond_features"], dtype=torch.float32)
        else:
            af = torch.zeros(1, ATOM_FEATURE_DIM)
            ei = torch.zeros(2, 0, dtype=torch.long)
            bf = torch.zeros(0, BOND_FEATURE_DIM)

        targets = torch.tensor(
            [row[c] if pd.notna(row.get(c)) else float("nan") for c in TARGET_COLUMNS],
            dtype=torch.float32)

        return {"point_cloud": pc, "features": features,
                "atom_features": af, "edge_index": ei, "bond_features": bf,
                "num_atoms": af.shape[0], "targets": targets}


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

    effective_ratio = (0.5 / max(n_orig, 1)) / (0.5 / max(n_ilth, 1))
    print(f"  Balanced sampler: {n_orig} original, {n_ilth} ILThermo")
    print(f"  Original oversample factor: {effective_ratio:.1f}x")
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

    # ── Load data ──
    print(f"\n{'='*60}")
    print("MoE-A + Fix7: Filter + Balanced Sampling + Augmentation")
    print(f"{'='*60}")

    train_csv = str(merged_dir / "splits/train.csv")

    # Use augmented dataset for training
    train_ds = AugmentedMergedDataset(
        train_csv, pc_dir, merged_features, is_train=True,
        feature_noise_std=0.05, surface_noise_std=0.10,
        pc_jitter_std=0.02, pc_scale_range=(0.95, 1.05),
        feature_dropout=0.10)

    # Standard dataset for val/test (no augmentation)
    from scripts.train_joint import MergedDataset
    val_ds = MergedDataset(str(merged_dir / "splits/val.csv"), pc_dir, merged_features, is_train=False)
    test_ds = MergedDataset(str(merged_dir / "splits/test.csv"), pc_dir, merged_features, is_train=False)

    print("\nCreating balanced sampler...")
    sampler = make_balanced_sampler(train_csv)

    train_ldr = DataLoader(train_ds, batch_size=64, sampler=sampler, collate_fn=collate_merged)
    val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)

    model = MoEA(feature_dim=len(merged_features),
                 pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
    model.to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_dir = Path("checkpoints/moe_fix7")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=60, eta_min=1e-6)

    # ── Train ──
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
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_improve += 1
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/200 | Train: {tl/max(n,1):.4f} | "
                  f"Val: {avg_val:.4f} | Best: {best_loss:.4f} | Pat: {no_improve}/25")
        if no_improve >= 25:
            print(f"  Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True))

    # ── Evaluate ──
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")

    preds, targets, gate_weights = evaluate_single(model, test_ldr, device)
    metrics = compute_metrics(preds, targets)
    print(format_metrics(metrics, "MoE-A Fix7 (filter + balanced + augmentation)"))

    print(f"\n  Per-property R²:")
    for p in TARGET_COLUMNS:
        print(f"    {p:15s} R² = {metrics[f'{p}_r2']:.4f}")
    print(f"    {'AVERAGE':15s} R² = {metrics['avg_r2']:.4f}")

    # Compare
    print(f"\n{'='*60}")
    print("COMPARISON: Fix6 vs Fix7 vs PointCloud")
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
    header += " {:>10s} {:>10s}".format("Fix7", "Fix7-Fix6")
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = p + "_r2"
        line = "  {:<15s}".format(p)
        for name in prev:
            line += " {:10.4f}".format(prev[name].get(key, float("nan")))
        f7 = metrics[key]
        f6 = prev.get("Fix6", {}).get(key, float("nan"))
        delta = f7 - f6 if not (np.isnan(f7) or np.isnan(f6)) else float("nan")
        sign = "+" if delta > 0 else ""
        line += " {:10.4f} {:>9s}".format(f7, "{}{:.4f}".format(sign, delta))
        print(line)

    f7_avg = metrics["avg_r2"]
    f6_avg = prev.get("Fix6", {}).get("avg_r2", float("nan"))
    delta = f7_avg - f6_avg
    sign = "+" if delta > 0 else ""
    line = "  {:<15s}".format("AVERAGE")
    for name in prev:
        line += " {:10.4f}".format(prev[name].get("avg_r2", float("nan")))
    line += " {:10.4f} {:>9s}".format(f7_avg, "{}{:.4f}".format(sign, delta))
    print(line)

    # ── Save ──
    results = {
        "fix": "filter_balanced_augmented",
        "description": "merged_v5 + balanced sampling + augmentation on original samples "
                       "(feature noise, PC jitter/scale, feature dropout, atom noise)",
        "augmentation": {
            "feature_noise_std": 0.05,
            "surface_noise_std": 0.10,
            "pc_jitter_std": 0.02,
            "pc_scale_range": [0.95, 1.05],
            "feature_dropout": 0.10,
        },
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items()},
        "gate_weights": gate_weights.mean(axis=0).tolist(),
    }
    with open("results/moe_fix7_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/moe_fix7_results.json")


if __name__ == "__main__":
    main()
