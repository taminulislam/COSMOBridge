"""Phase 3: Train multimodal PointNet + GNN + Tabular model.

Loads pre-trained GNN from Phase 2, adds PointNet for COSMO surface point
clouds, and trains with cross-attention fusion.

Usage:
    python scripts/train_pointcloud.py --config configs/default.yaml
"""

import argparse
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import TARGET_COLUMNS, FEATURE_COLUMNS
from src.data.graph_builder import smiles_to_graph, smiles_to_combined_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.data.image_loader import get_train_transforms, get_eval_transforms
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics, format_metrics


def smiles_to_hash(smiles):
    return hashlib.md5(smiles.encode()).hexdigest()[:12]


class PointCloudMultimodalDataset(Dataset):
    """Dataset providing point clouds + graphs + tabular features."""

    def __init__(
        self,
        csv_path,
        point_cloud_dir,
        is_train=True,
        n_points=1024,
    ):
        self.df = pd.read_csv(csv_path)
        self.pc_dir = Path(point_cloud_dir)
        self.is_train = is_train
        self.n_points = n_points

        # Load point cloud index
        index_path = self.pc_dir / "index.csv"
        self.pc_index = {}
        if index_path.exists():
            idx_df = pd.read_csv(index_path)
            self.pc_index = dict(zip(idx_df["smiles"], idx_df["filename"]))

        # Pre-build graphs
        self.graphs = {}
        unique_smiles = self.df["smiles"].unique() if "smiles" in self.df.columns else []
        for smi in unique_smiles:
            try:
                self.graphs[smi] = smiles_to_graph(smi)
            except Exception:
                self.graphs[smi] = None

        # Count available point clouds
        n_with_pc = sum(1 for smi in self.df["smiles"].unique()
                        if smi in self.pc_index and
                        (self.pc_dir / self.pc_index[smi]).exists())
        print(f"  Point clouds: {n_with_pc}/{self.df['smiles'].nunique()} unique ILs")

    def __len__(self):
        return len(self.df)

    def _load_point_cloud(self, smiles):
        """Load pre-computed point cloud or return zeros."""
        filename = self.pc_index.get(smiles)
        if filename:
            path = self.pc_dir / filename
            if path.exists():
                data = np.load(path)
                points = data["points"]  # (N, 7)

                # Ensure correct size
                if len(points) >= self.n_points:
                    if self.is_train:
                        idx = np.random.choice(len(points), self.n_points, replace=False)
                    else:
                        idx = np.arange(self.n_points)
                    points = points[idx]
                else:
                    extra = self.n_points - len(points)
                    extra_idx = np.random.choice(len(points), extra, replace=True)
                    points = np.concatenate([points, points[extra_idx]])

                # Random rotation augmentation during training
                if self.is_train:
                    angle = np.random.uniform(0, 2 * np.pi)
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    R = np.array([[cos_a, -sin_a, 0],
                                  [sin_a, cos_a, 0],
                                  [0, 0, 1]])
                    points[:, :3] = points[:, :3] @ R.T  # Rotate xyz
                    points[:, 3:6] = points[:, 3:6] @ R.T  # Rotate normals

                return torch.tensor(points, dtype=torch.float32)

        return torch.zeros(self.n_points, 7)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Features (already normalized in the CSV)
        features = torch.tensor(
            [row[col] for col in FEATURE_COLUMNS], dtype=torch.float32
        )

        # Point cloud
        smiles = row.get("smiles", "")
        point_cloud = self._load_point_cloud(smiles)

        # Graph
        g = self.graphs.get(smiles)
        if g is not None:
            atom_features = torch.tensor(g["atom_features"], dtype=torch.float32)
            edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
            bond_features = torch.tensor(g["bond_features"], dtype=torch.float32)
        else:
            atom_features = torch.zeros(1, ATOM_FEATURE_DIM)
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            bond_features = torch.zeros(0, BOND_FEATURE_DIM)

        # Targets
        targets = torch.tensor(
            [row[col] for col in TARGET_COLUMNS], dtype=torch.float32
        )

        return {
            "point_cloud": point_cloud,
            "features": features,
            "atom_features": atom_features,
            "edge_index": edge_index,
            "bond_features": bond_features,
            "num_atoms": atom_features.shape[0],
            "targets": targets,
        }


def collate_pointcloud(batch):
    """Custom collate for point cloud + graph batches."""
    point_clouds = torch.stack([b["point_cloud"] for b in batch])
    features = torch.stack([b["features"] for b in batch])
    targets = torch.stack([b["targets"] for b in batch])

    # Graph batching (same as collate_multimodal)
    atom_features_list = []
    edge_index_list = []
    bond_features_list = []
    graph_batch = []
    atom_offset = 0

    for i, b in enumerate(batch):
        n_atoms = b["atom_features"].shape[0]
        atom_features_list.append(b["atom_features"])
        graph_batch.extend([i] * n_atoms)
        ei = b["edge_index"].clone()
        if ei.shape[1] > 0:
            ei += atom_offset
        edge_index_list.append(ei)
        bond_features_list.append(b["bond_features"])
        atom_offset += n_atoms

    atom_features = torch.cat(atom_features_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    bond_features = torch.cat(bond_features_list, dim=0)
    graph_batch_t = torch.tensor(graph_batch, dtype=torch.long)

    return {
        "point_cloud": point_clouds,
        "features": features,
        "atom_features": atom_features,
        "edge_index": edge_index,
        "bond_features": bond_features,
        "batch": graph_batch_t,
        "targets": targets,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--pretrained-gnn", type=str,
                        default="checkpoints/transfer/pretrained.pt")
    parser.add_argument("--save-results", type=str,
                        default="results/pointcloud_results.json")
    parser.add_argument("--seed", type=int, default=None,
                        help="override experiment seed (for multi-seed error bars)")
    args = parser.parse_args()

    config = load_config(args.config)
    config["training"]["num_epochs"] = args.epochs
    config["training"]["learning_rate"] = args.lr
    config["training"]["early_stopping_patience"] = 25
    if args.seed is not None:
        config.setdefault("experiment", {})["seed"] = args.seed
        config["experiment"]["checkpoint_dir"] = f"checkpoints/pointcloud_s{args.seed}"
    else:
        config["experiment"]["checkpoint_dir"] = "checkpoints/pointcloud"

    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    device = get_device(config)
    print(f"Device: {device}")

    # ── Build model ──
    from src.models.fusion.multimodal_pointcloud import MultimodalPointCloudModel

    model = MultimodalPointCloudModel(
        config=config,
        pretrained_gnn_path=args.pretrained_gnn,
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    # ── Build data loaders ──
    processed_dir = Path("data/processed")
    splits_dir = processed_dir / "splits"
    pc_dir = Path("data/pipeline/point_clouds")

    print("\nLoading datasets...")
    train_ds = PointCloudMultimodalDataset(
        str(splits_dir / "train.csv"), str(pc_dir),
        is_train=True, n_points=args.n_points,
    )
    val_ds = PointCloudMultimodalDataset(
        str(splits_dir / "val.csv"), str(pc_dir),
        is_train=False, n_points=args.n_points,
    )
    test_ds = PointCloudMultimodalDataset(
        str(splits_dir / "test.csv"), str(pc_dir),
        is_train=False, n_points=args.n_points,
    )

    batch_size = config.get("training", {}).get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_pointcloud)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_pointcloud)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_pointcloud)

    # ── Train ──
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
    )

    history = trainer.train(verbose=True)

    # ── Save results ──
    results = {
        "model": "pointcloud_multimodal",
        "n_params": n_params,
        "n_points": args.n_points,
        "pretrained_gnn": args.pretrained_gnn,
        "best_val_loss": trainer.best_val_loss,
        "epochs_trained": len(history["train_loss"]),
    }
    if "test_metrics" in history:
        results["test_metrics"] = {
            k: float(v) if not (isinstance(v, float) and np.isnan(v)) else None
            for k, v in history["test_metrics"].items()
        }

    Path(args.save_results).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_results, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.save_results}")

    # ── Comparison with all baselines ──
    print(f"\n{'='*60}")
    print("COMPARISON WITH ALL MODELS")
    print(f"{'='*60}")
    baselines = [
        ("Baseline GNN (Phase 0)", "results/gnn_results.json"),
        ("GNN+Surface (Phase 1)", "results/gnn_surface_results.json"),
        ("Transfer GNN (Phase 2)", "results/transfer_results.json"),
    ]
    for name, path in baselines:
        try:
            with open(path) as f:
                base = json.load(f)
            bm = base.get("test_metrics", {})
            tm = results.get("test_metrics", {})
            r2_b = bm.get("avg_r2", 0)
            r2_t = tm.get("avg_r2", 0)
            print(f"\n  vs {name}:")
            print(f"    R²:  {r2_b:.4f} -> {r2_t:.4f}  ({r2_t - r2_b:+.4f})")
            for key in sorted(bm.keys()):
                if key.endswith("_r2"):
                    prop = key.replace("_r2", "")
                    b = bm[key]
                    t = tm.get(key, 0)
                    print(f"    {prop:15s} R²: {b:.4f} -> {t:.4f}  ({t - b:+.4f})")
        except Exception:
            pass


if __name__ == "__main__":
    main()
