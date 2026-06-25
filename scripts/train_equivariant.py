"""SE(3)-Equivariant point cloud encoder for COSMO surfaces.

Uses E(n) Equivariant Graph Neural Network (EGNN) layers that respect
rotational symmetry natively. No rotation augmentation needed.

Invariant scalar features (ESP, distances) and equivariant vector features
(coordinates, normals) are processed separately and combined.
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
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.multimodal_pointcloud import PointCloudFusion
from src.training.trainer import Trainer
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud


class EGNNLayer(nn.Module):
    """E(n) Equivariant Graph Neural Network layer.

    Updates node features (invariant) and coordinates (equivariant)
    using message passing that respects rotational symmetry.

    Reference: Satorras et al., "E(n) Equivariant Graph Neural Networks", 2021.
    """

    def __init__(self, node_dim, hidden_dim=64, coord_update=True):
        super().__init__()
        self.coord_update = coord_update

        # Message MLP: [hi, hj, ||xi-xj||², eij] -> mij
        edge_input = node_dim * 2 + 1 + 1  # +1 for distance, +1 for ESP diff
        self.message_mlp = nn.Sequential(
            nn.Linear(edge_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Node update: [hi, agg_messages] -> hi'
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )

        # Coordinate update (equivariant)
        if coord_update:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, h, x, edge_index, edge_attr=None):
        """
        Args:
            h: (N, node_dim) node features (invariant)
            x: (N, 3) coordinates (equivariant)
            edge_index: (2, E) edges
            edge_attr: (E, 1) optional edge features (ESP difference)
        Returns:
            h', x' updated features and coordinates
        """
        src, dst = edge_index

        # Compute squared distances (invariant under rotation)
        diff = x[src] - x[dst]
        dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)  # (E, 1)

        # Build edge input
        edge_input = [h[src], h[dst], dist_sq]
        if edge_attr is not None:
            edge_input.append(edge_attr)
        else:
            edge_input.append(torch.zeros_like(dist_sq))
        edge_input = torch.cat(edge_input, dim=-1)

        # Messages
        messages = self.message_mlp(edge_input)  # (E, hidden)

        # Aggregate (sum)
        agg = torch.zeros(h.shape[0], messages.shape[1], device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)

        # Update nodes
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))

        # Update coordinates (equivariant: weighted sum of direction vectors)
        if self.coord_update:
            coord_weights = self.coord_mlp(messages)  # (E, 1)
            coord_agg = torch.zeros_like(x)
            weighted_diff = diff * coord_weights
            coord_agg.scatter_add_(0, dst.unsqueeze(1).expand_as(weighted_diff), weighted_diff)
            x = x + coord_agg

        return h, x


class EquivariantSurfaceEncoder(nn.Module):
    """Equivariant encoder for molecular surface point clouds.

    Builds a k-NN graph on the point cloud and processes with EGNN layers.
    """

    def __init__(self, feature_dim=256, n_layers=3, k=16, dropout=0.3):
        super().__init__()
        self.k = k

        # Initial node features from invariants: ESP, normal magnitude, etc.
        # Input: 4 invariant features (esp, nx², ny², nz²→normal_mag)
        node_input_dim = 4
        self.node_embed = nn.Sequential(
            nn.Linear(node_input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )

        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(64, hidden_dim=64, coord_update=(i < n_layers - 1))
            for i in range(n_layers)
        ])

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(64, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self._feature_dim = feature_dim

    def _build_knn_graph(self, x, batch_idx, k):
        """Build k-NN edges within each batch element."""
        edges_src = []
        edges_dst = []
        B = batch_idx.max().item() + 1

        for b in range(B):
            mask = (batch_idx == b)
            idx = torch.where(mask)[0]
            pts = x[mask]
            n = pts.shape[0]
            kk = min(k, n - 1)
            if kk <= 0:
                continue

            # Pairwise distances
            dists = torch.cdist(pts, pts)
            dists.fill_diagonal_(float("inf"))
            _, nn_idx = dists.topk(kk, largest=False, dim=-1)

            for i in range(n):
                for j_local in range(kk):
                    edges_src.append(idx[i])
                    edges_dst.append(idx[nn_idx[i, j_local]])

        if not edges_src:
            return torch.zeros(2, 0, dtype=torch.long, device=x.device)
        return torch.stack([
            torch.tensor(edges_src, device=x.device),
            torch.tensor(edges_dst, device=x.device)
        ])

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, N, 7) — [x, y, z, nx, ny, nz, esp]
        Returns:
            (B, feature_dim)
        """
        B, N, _ = point_cloud.shape
        device = point_cloud.device

        # Flatten batch
        coords = point_cloud[:, :, :3].reshape(B * N, 3)
        normals = point_cloud[:, :, 3:6].reshape(B * N, 3)
        esp = point_cloud[:, :, 6].reshape(B * N, 1)

        # Invariant features: ESP, normal magnitude, mean coord distance from center
        normal_mag = normals.norm(dim=-1, keepdim=True)

        # Per-batch center distance
        batch_idx = torch.arange(B, device=device).repeat_interleave(N)
        centers = torch.zeros(B, 3, device=device)
        for b in range(B):
            mask = batch_idx == b
            centers[b] = coords[mask].mean(dim=0)
        center_dist = (coords - centers[batch_idx]).norm(dim=-1, keepdim=True)

        node_invariants = torch.cat([esp, normal_mag, center_dist,
                                      (esp ** 2).clamp(max=1)], dim=-1)
        h = self.node_embed(node_invariants)

        # Build k-NN graph
        edge_index = self._build_knn_graph(coords, batch_idx, self.k)

        # ESP difference as edge attribute
        if edge_index.shape[1] > 0:
            edge_attr = (esp[edge_index[0]] - esp[edge_index[1]]).abs()
        else:
            edge_attr = None

        # EGNN layers
        x = coords
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)

        # Pool per batch
        out = torch.zeros(B, h.shape[-1], device=device)
        for b in range(B):
            mask = batch_idx == b
            out[b] = h[mask].mean(dim=0)

        return self.out_proj(out)

    @property
    def feature_dim(self):
        return self._feature_dim


class EquivariantMultimodalModel(nn.Module):
    """Equivariant surface encoder + GNN + Tabular."""

    def __init__(self, config=None, pretrained_gnn_path=None):
        super().__init__()
        config = config or {}

        self.surface_encoder = EquivariantSurfaceEncoder(feature_dim=256, n_layers=3, k=16)

        gc = config.get("model", {}).get("graph", {})
        self.gnn = MolecularGNN(
            atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
            dropout=0.3, pooling="mean", num_targets=0,
        )

        if pretrained_gnn_path and Path(pretrained_gnn_path).exists():
            ckpt = torch.load(pretrained_gnn_path, map_location="cpu", weights_only=True)
            gnn_state = {k: v for k, v in ckpt.items()
                         if any(k.startswith(p) for p in ["atom_projection", "convs", "batch_norms", "pool"])}
            if gnn_state:
                self.gnn.load_state_dict(gnn_state, strict=False)

        self.fusion = PointCloudFusion(
            pointcloud_dim=256, graph_dim=256, tabular_dim=len(FEATURE_COLUMNS),
            fused_dim=256, num_heads=8, dropout=0.3,
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 7),
        )

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        pc_feat = self.surface_encoder(point_cloud)
        graph_feat = self.gnn.get_features(atom_features, edge_index, bond_features, batch)
        fused = self.fusion(pc_feat, graph_feat, features)
        return self.prediction_head(fused)


def main():
    config = load_config("configs/default.yaml")
    config["training"]["num_epochs"] = 200
    config["training"]["learning_rate"] = 1e-4
    config["training"]["early_stopping_patience"] = 25
    config["experiment"]["checkpoint_dir"] = "checkpoints/equivariant"

    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    # Use fewer points for equivariant model (k-NN is O(N²))
    n_points = 512

    model = EquivariantMultimodalModel(config=config, pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Equivariant model params: {n_params:,}")

    splits_dir = Path("data/processed/splits")
    pc_dir = "data/pipeline/point_clouds"

    print("\nLoading datasets...")
    train_ds = PointCloudMultimodalDataset(str(splits_dir / "train.csv"), pc_dir, is_train=True, n_points=n_points)
    val_ds = PointCloudMultimodalDataset(str(splits_dir / "val.csv"), pc_dir, is_train=False, n_points=n_points)
    test_ds = PointCloudMultimodalDataset(str(splits_dir / "test.csv"), pc_dir, is_train=False, n_points=n_points)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_pointcloud)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_pointcloud)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_pointcloud)

    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                      test_loader=test_loader, config=config, device=device)
    history = trainer.train(verbose=True)

    results = {"model": "equivariant_multimodal", "n_params": n_params, "n_points": n_points,
               "best_val_loss": trainer.best_val_loss}
    if "test_metrics" in history:
        results["test_metrics"] = {k: float(v) if not (isinstance(v, float) and np.isnan(v)) else None
                                   for k, v in history["test_metrics"].items()}
    with open("results/equivariant_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/equivariant_results.json")


if __name__ == "__main__":
    main()
