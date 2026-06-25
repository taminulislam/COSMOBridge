"""Strategy A+C+D: Joint training on merged dataset.

Trains PointCloud+GNN+Tabular on all 5845 samples with masked loss,
Morgan fingerprints, log-transformed targets, and surface descriptors.

Strategy B: Also trains 5-fold CV ensemble and averages predictions.

Strategy E: After A+B, trains a stacking meta-learner.
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
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.utils.config import load_config, get_device, set_seed
from src.data.graph_builder import smiles_to_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.multimodal_pointcloud import PointCloudFusion
from src.training.metrics import compute_metrics, format_metrics

import hashlib


TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


def smiles_to_hash(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


# ── Dataset ──────────────────────────────────────────────────────────────────

class MergedDataset(Dataset):
    """Dataset for merged training with point clouds + graphs + extended features."""

    def __init__(self, csv_path, pc_dir, feature_columns, is_train=True, n_points=1024):
        self.df = pd.read_csv(csv_path)
        self.pc_dir = Path(pc_dir)
        self.feature_columns = feature_columns
        self.is_train = is_train
        self.n_points = n_points

        # Load point cloud index
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
        print(f"  {csv_path}: {len(self.df)} rows, {n_pc} with PC, {n_g} with graphs")

    def __len__(self):
        return len(self.df)

    def _load_pc(self, smiles):
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
                    angle = np.random.uniform(0, 2 * np.pi)
                    c, s = np.cos(angle), np.sin(angle)
                    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    pts[:, :3] = pts[:, :3] @ R.T
                    pts[:, 3:6] = pts[:, 3:6] @ R.T
                return torch.tensor(pts, dtype=torch.float32)
        return torch.zeros(self.n_points, 7)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row["smiles"]

        features = torch.tensor([row[c] for c in self.feature_columns], dtype=torch.float32)
        pc = self._load_pc(smiles)

        g = self.graphs.get(smiles)
        if g is not None:
            af = torch.tensor(g["atom_features"], dtype=torch.float32)
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


def collate_merged(batch):
    pcs = torch.stack([b["point_cloud"] for b in batch])
    feats = torch.stack([b["features"] for b in batch])
    targets = torch.stack([b["targets"] for b in batch])

    af_list, ei_list, bf_list, gb = [], [], [], []
    offset = 0
    for i, b in enumerate(batch):
        n = b["atom_features"].shape[0]
        af_list.append(b["atom_features"])
        gb.extend([i] * n)
        ei = b["edge_index"].clone()
        if ei.shape[1] > 0:
            ei += offset
        ei_list.append(ei)
        bf_list.append(b["bond_features"])
        offset += n

    return {
        "point_cloud": pcs, "features": feats, "targets": targets,
        "atom_features": torch.cat(af_list),
        "edge_index": torch.cat(ei_list, dim=1),
        "bond_features": torch.cat(bf_list),
        "batch": torch.tensor(gb, dtype=torch.long),
    }


# ── Model ────────────────────────────────────────────────────────────────────

class JointModel(nn.Module):
    """PointNet + GNN + Tabular for joint training on merged dataset."""

    def __init__(self, feature_dim, pretrained_gnn_path=None):
        super().__init__()

        self.pointnet = PointNetEncoder(in_channels=7, feature_dim=256, dropout=0.3)

        self.gnn = MolecularGNN(
            atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
            dropout=0.3, pooling="mean", num_targets=0)

        if pretrained_gnn_path and Path(pretrained_gnn_path).exists():
            ckpt = torch.load(pretrained_gnn_path, map_location="cpu", weights_only=True)
            gnn_state = {k: v for k, v in ckpt.items()
                         if any(k.startswith(p) for p in ["atom_projection", "convs", "batch_norms", "pool"])}
            if gnn_state:
                self.gnn.load_state_dict(gnn_state, strict=False)
                print(f"  Loaded pre-trained GNN: {len(gnn_state)} params")

        self.fusion = PointCloudFusion(
            pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
            fused_dim=256, num_heads=8, dropout=0.3)

        self.prediction_head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 7))

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        pc = self.pointnet(point_cloud)
        g = self.gnn.get_features(atom_features, edge_index, bond_features, batch)
        fused = self.fusion(pc, g, features)
        return self.prediction_head(fused)


# ── Masked Loss ──────────────────────────────────────────────────────────────

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        mask = ~torch.isnan(targets)
        if mask.sum() == 0:
            return {"total": torch.tensor(0.0, device=predictions.device, requires_grad=True)}
        safe_targets = targets.clone()
        safe_targets[~mask] = 0.0
        diff2 = (predictions - safe_targets) ** 2 * mask.float()
        per_task = diff2.sum(dim=0) / mask.sum(dim=0).float().clamp(min=1)
        active = mask.any(dim=0)
        total = per_task[active].mean()
        losses = {"total": total}
        for i, name in enumerate(TARGET_COLUMNS):
            losses[name] = per_task[i]
        return losses


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, device, ckpt_dir,
                num_epochs=200, lr=1e-4, patience=25, use_masked_loss=True):
    """Train with optional masked loss."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    criterion = MaskedMSELoss().to(device) if use_masked_loss else nn.MSELoss().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=10)
    cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - 10, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[10])

    best_loss = float("inf")
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        n = 0
        for batch in train_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            optimizer.zero_grad()
            preds = model(point_cloud=batch["point_cloud"], features=batch["features"],
                          atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                          bond_features=batch["bond_features"], batch=batch["batch"])

            if use_masked_loss:
                losses = criterion(preds, batch["targets"])
                loss = losses["total"]
            else:
                loss = criterion(preds, batch["targets"])

            if torch.isnan(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        scheduler.step()
        avg_loss = total_loss / max(n, 1)

        # Validate
        model.eval()
        vl = 0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                preds = model(point_cloud=batch["point_cloud"], features=batch["features"],
                              atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                              bond_features=batch["bond_features"], batch=batch["batch"])
                # Val always uses clean MSE on non-NaN
                mask = ~torch.isnan(batch["targets"])
                safe = batch["targets"].clone()
                safe[~mask] = 0.0
                diff = ((preds - safe) ** 2 * mask.float()).sum() / mask.float().sum().clamp(min=1)
                vl += diff.item()
                vn += 1

        avg_val = vl / max(vn, 1)

        if avg_val < best_loss:
            best_loss = avg_val
            no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch:3d}/{num_epochs} | Train: {avg_loss:.4f} | "
                  f"Val: {avg_val:.4f} | Best: {best_loss:.4f} | Pat: {no_improve}/{patience}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=True))
    return model


def evaluate(model, loader, device):
    """Run inference, return predictions and targets."""
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    merged_dir = Path("data/merged")
    if not merged_dir.exists():
        print("ERROR: Run create_merged_dataset.py first")
        return

    meta = json.load(open(merged_dir / "metadata.json"))
    feature_columns = meta["feature_columns"]
    n_features = len(feature_columns)
    print(f"Features: {n_features} ({meta.get('morgan_bits', 0)} Morgan bits)")

    pc_dir = "data/pipeline/point_clouds"

    # ══════════════════════════════════════════════════════════════════
    # STRATEGY A: Joint training on merged dataset
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STRATEGY A: Joint Training on Merged Dataset")
    print(f"{'='*60}")

    model_a = JointModel(feature_dim=n_features, pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
    model_a.to(device)
    print(f"Model params: {sum(p.numel() for p in model_a.parameters() if p.requires_grad):,}")

    splits = merged_dir / "splits"
    train_ds = MergedDataset(str(splits / "train.csv"), pc_dir, feature_columns, is_train=True)
    val_ds = MergedDataset(str(splits / "val.csv"), pc_dir, feature_columns, is_train=False)
    test_ds = MergedDataset(str(splits / "test.csv"), pc_dir, feature_columns, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_merged)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)

    model_a = train_model(model_a, train_loader, val_loader, device,
                          "checkpoints/joint", num_epochs=200, lr=1e-4, patience=25,
                          use_masked_loss=True)

    preds_a, targets_a = evaluate(model_a, test_loader, device)
    metrics_a = compute_metrics(preds_a, targets_a)
    print(f"\n  Strategy A Test Results:")
    print(format_metrics(metrics_a, "Joint"))

    # ══════════════════════════════════════════════════════════════════
    # STRATEGY B: 5-Fold CV Ensemble
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STRATEGY B: 5-Fold CV Ensemble")
    print(f"{'='*60}")

    cv_preds = []
    for fold in range(5):
        print(f"\n--- Fold {fold} ---")
        fold_dir = merged_dir / "cv_folds" / f"fold_{fold}"

        model_fold = JointModel(feature_dim=n_features, pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
        model_fold.to(device)

        fold_train = MergedDataset(str(fold_dir / "train.csv"), pc_dir, feature_columns, is_train=True)
        fold_val = MergedDataset(str(fold_dir / "val.csv"), pc_dir, feature_columns, is_train=False)

        fold_train_loader = DataLoader(fold_train, batch_size=64, shuffle=True, collate_fn=collate_merged)
        fold_val_loader = DataLoader(fold_val, batch_size=32, shuffle=False, collate_fn=collate_merged)

        model_fold = train_model(model_fold, fold_train_loader, fold_val_loader, device,
                                 f"checkpoints/cv_fold_{fold}", num_epochs=150, lr=1e-4,
                                 patience=20, use_masked_loss=True)

        # Evaluate this fold on the ORIGINAL test set
        fold_preds, _ = evaluate(model_fold, test_loader, device)
        cv_preds.append(fold_preds)

        fold_metrics = compute_metrics(fold_preds, targets_a)
        print(f"  Fold {fold} R²: {fold_metrics['avg_r2']:.4f}")

    # Average CV predictions
    ensemble_preds = np.mean(cv_preds, axis=0)
    metrics_b = compute_metrics(ensemble_preds, targets_a)
    print(f"\n  Strategy B (CV Ensemble) Test Results:")
    print(format_metrics(metrics_b, "CV Ensemble"))

    # ══════════════════════════════════════════════════════════════════
    # STRATEGY E: Stacking Meta-Learner
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STRATEGY E: Stacking Meta-Learner")
    print(f"{'='*60}")

    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor

        # Stack predictions from all CV folds + Strategy A as features
        # Use val set for training the meta-learner
        val_preds_a, val_targets = evaluate(model_a, val_loader, device)

        val_cv_preds = []
        for fold in range(5):
            model_fold = JointModel(feature_dim=n_features)
            ckpt = Path(f"checkpoints/cv_fold_{fold}/best_model.pt")
            if ckpt.exists():
                model_fold.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            model_fold.to(device)
            vp, _ = evaluate(model_fold, val_loader, device)
            val_cv_preds.append(vp)

        # Meta features: [strategy_a_preds, cv_fold_preds_0..4]
        val_meta = np.column_stack([val_preds_a] + val_cv_preds)
        test_meta = np.column_stack([preds_a] + cv_preds)

        # Train per-target GBR
        meta_preds = np.zeros_like(targets_a)
        for t in range(7):
            valid = ~np.isnan(val_targets[:, t])
            if valid.sum() < 5:
                meta_preds[:, t] = ensemble_preds[:, t]
                continue
            gbr = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1)
            gbr.fit(val_meta[valid], val_targets[valid, t])
            meta_preds[:, t] = gbr.predict(test_meta)

        metrics_e = compute_metrics(meta_preds, targets_a)
        print(f"  Strategy E (Stacking) Test Results:")
        print(format_metrics(metrics_e, "Stacking"))
    except Exception as e:
        print(f"  Strategy E failed: {e}")
        metrics_e = {}

    # ══════════════════════════════════════════════════════════════════
    # FINAL COMPARISON
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FINAL COMPARISON — ALL STRATEGIES")
    print(f"{'='*60}")

    all_results = {}
    # Load previous best
    for name, path in [("Best Previous (Hard Ensemble)", "results/ensemble_phase23_results.json"),
                       ("Phase 3 PointCloud", "results/pointcloud_results.json")]:
        try:
            with open(path) as f:
                d = json.load(f)
            r2 = d.get("hard_ensemble", d.get("test_metrics", {})).get("avg_r2",
                 d.get("test_metrics", {}).get("avg_r2", "N/A"))
            all_results[name] = r2
        except Exception:
            pass

    all_results["Strategy A (Joint)"] = metrics_a.get("avg_r2", "N/A")
    all_results["Strategy B (CV Ensemble)"] = metrics_b.get("avg_r2", "N/A")
    if metrics_e:
        all_results["Strategy E (Stacking)"] = metrics_e.get("avg_r2", "N/A")

    for name, r2 in sorted(all_results.items(), key=lambda x: -x[1] if isinstance(x[1], float) else 0):
        print(f"  {name:40s}  R² = {r2:.4f}" if isinstance(r2, float) else f"  {name:40s}  R² = {r2}")

    # Per-property comparison
    print(f"\n  Per-property R²:")
    print(f"  {'Property':15s} {'Joint':>10s} {'CV Ens':>10s} {'Stacking':>10s}")
    for prop in TARGET_COLUMNS:
        key = f"{prop}_r2"
        ja = metrics_a.get(key, float("nan"))
        jb = metrics_b.get(key, float("nan"))
        je = metrics_e.get(key, float("nan")) if metrics_e else float("nan")
        print(f"  {prop:15s} {ja:10.4f} {jb:10.4f} {je:10.4f}")

    # Save all results
    results = {
        "strategy_a_joint": {k: float(v) if isinstance(v, (float, np.floating)) else v
                             for k, v in metrics_a.items()},
        "strategy_b_cv_ensemble": {k: float(v) if isinstance(v, (float, np.floating)) else v
                                   for k, v in metrics_b.items()},
    }
    if metrics_e:
        results["strategy_e_stacking"] = {k: float(v) if isinstance(v, (float, np.floating)) else v
                                           for k, v in metrics_e.items()}
    with open("results/strategies_abcde_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/strategies_abcde_results.json")


if __name__ == "__main__":
    main()
