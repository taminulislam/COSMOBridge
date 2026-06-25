"""Atom-Surface Cross-Attention model.

Local cross-attention between GNN atom nodes and nearby surface points.
Each atom attends to surface points, capturing how atomic environment
manifests on the molecular surface.
"""

import sys
import json
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import TARGET_COLUMNS, FEATURE_COLUMNS
from src.data.graph_builder import smiles_to_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.training.trainer import Trainer
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud

try:
    from torch_geometric.nn import GATConv, BatchNorm, global_mean_pool
except ImportError:
    raise RuntimeError("torch_geometric required")


class AtomSurfaceAttention(nn.Module):
    """Cross-attention: atom nodes query surface point features.

    Each atom representation attends to all surface points, producing
    surface-informed atom features that are then pooled globally.
    """

    def __init__(self, atom_dim=256, point_dim=64, out_dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        # Point feature encoder (shared MLP on raw point features)
        self.point_encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, point_dim),
            nn.ReLU(),
        )

        # Cross-attention projections
        self.q_proj = nn.Linear(atom_dim, out_dim)
        self.k_proj = nn.Linear(point_dim, out_dim)
        self.v_proj = nn.Linear(point_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

        self.scale = self.head_dim ** 0.5
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, atom_features, point_cloud, graph_batch):
        """
        Args:
            atom_features: (N_atoms, atom_dim) — per-atom GNN features
            point_cloud: (B, N_points, 7) — surface points
            graph_batch: (N_atoms,) — maps atoms to batch index

        Returns:
            (B, out_dim) — surface-informed global features
        """
        B = point_cloud.shape[0]
        N_points = point_cloud.shape[1]

        # Encode surface points: (B, N_points, point_dim)
        point_feat = self.point_encoder(point_cloud)

        # For each sample in batch, gather its atom features
        out_features = []
        for b in range(B):
            atom_mask = (graph_batch == b)
            atoms_b = atom_features[atom_mask]  # (n_atoms_b, atom_dim)
            points_b = point_feat[b]  # (N_points, point_dim)

            if atoms_b.shape[0] == 0:
                out_features.append(torch.zeros(self.q_proj.out_features, device=atom_features.device))
                continue

            n_atoms = atoms_b.shape[0]

            # Cross-attention: atoms query surface points
            Q = self.q_proj(atoms_b).view(n_atoms, self.num_heads, self.head_dim)
            K = self.k_proj(points_b).view(N_points, self.num_heads, self.head_dim)
            V = self.v_proj(points_b).view(N_points, self.num_heads, self.head_dim)

            # (n_atoms, heads, head_dim) x (heads, head_dim, N_points)
            Q = Q.permute(1, 0, 2)  # (heads, n_atoms, head_dim)
            K = K.permute(1, 2, 0)  # (heads, head_dim, N_points)
            V = V.permute(1, 0, 2)  # (heads, N_points, head_dim)

            attn = torch.matmul(Q, K) / self.scale  # (heads, n_atoms, N_points)
            attn = F.softmax(attn, dim=-1)
            attended = torch.matmul(attn, V)  # (heads, n_atoms, head_dim)

            # Reshape: (n_atoms, out_dim)
            attended = attended.permute(1, 0, 2).reshape(n_atoms, -1)
            attended = self.out_proj(attended)

            # Mean pool over atoms
            pooled = attended.mean(dim=0)  # (out_dim,)
            out_features.append(pooled)

        return self.norm(torch.stack(out_features))  # (B, out_dim)


class AtomSurfaceModel(nn.Module):
    """GNN with atom-level surface cross-attention + tabular features."""

    def __init__(self, pretrained_gnn_path=None):
        super().__init__()
        hidden_dim = 256

        # GNN layers (no prediction head)
        self.atom_proj = nn.Linear(ATOM_FEATURE_DIM, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(4):
            self.convs.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True, dropout=0.3))
            self.bns.append(BatchNorm(hidden_dim))

        # Load pre-trained GNN weights
        if pretrained_gnn_path and Path(pretrained_gnn_path).exists():
            ckpt = torch.load(pretrained_gnn_path, map_location="cpu", weights_only=True)
            own_state = self.state_dict()
            loaded = 0
            for k, v in ckpt.items():
                if k in own_state and v.shape == own_state[k].shape:
                    own_state[k] = v
                    loaded += 1
                elif k.replace("atom_projection", "atom_proj") in own_state:
                    key = k.replace("atom_projection", "atom_proj")
                    if v.shape == own_state[key].shape:
                        own_state[key] = v
                        loaded += 1
            self.load_state_dict(own_state, strict=False)
            print(f"  Loaded {loaded} pre-trained GNN params")

        # Atom-surface cross-attention
        self.atom_surface_attn = AtomSurfaceAttention(
            atom_dim=hidden_dim, point_dim=64, out_dim=hidden_dim, num_heads=4
        )

        # Global graph pooling (parallel path)
        self.pool = global_mean_pool

        # Prediction: graph_pool(256) + atom_surface(256) + thermo(5) = 517
        thermo_dim = 5
        pred_dim = hidden_dim * 2 + thermo_dim
        self.prediction_head = nn.Sequential(
            nn.Linear(pred_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 7),
        )

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        # GNN forward (get per-atom features)
        x = self.atom_proj(atom_features)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)

        # Global graph representation
        graph_feat = self.pool(x, batch)  # (B, 256)

        # Atom-surface cross-attention
        surface_feat = self.atom_surface_attn(x, point_cloud, batch)  # (B, 256)

        # Thermo features (first 5)
        thermo = features[:, :5]

        combined = torch.cat([graph_feat, surface_feat, thermo], dim=-1)
        return self.prediction_head(combined)


def main():
    config = load_config("configs/default.yaml")
    config["training"]["num_epochs"] = 200
    config["training"]["learning_rate"] = 1e-4
    config["training"]["early_stopping_patience"] = 25
    config["experiment"]["checkpoint_dir"] = "checkpoints/atom_surface"

    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    model = AtomSurfaceModel(pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Atom-Surface model params: {n_params:,}")

    splits_dir = Path("data/processed/splits")
    pc_dir = "data/pipeline/point_clouds"

    print("\nLoading datasets...")
    train_ds = PointCloudMultimodalDataset(str(splits_dir / "train.csv"), pc_dir, is_train=True)
    val_ds = PointCloudMultimodalDataset(str(splits_dir / "val.csv"), pc_dir, is_train=False)
    test_ds = PointCloudMultimodalDataset(str(splits_dir / "test.csv"), pc_dir, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_pointcloud)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)

    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                      test_loader=test_loader, config=config, device=device)
    history = trainer.train(verbose=True)

    results = {"model": "atom_surface_crossattn", "n_params": n_params,
               "best_val_loss": trainer.best_val_loss}
    if "test_metrics" in history:
        results["test_metrics"] = {k: float(v) if not (isinstance(v, float) and np.isnan(v)) else None
                                   for k, v in history["test_metrics"].items()}
    with open("results/atom_surface_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/atom_surface_results.json")


if __name__ == "__main__":
    main()
