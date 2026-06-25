"""Train DGCNN + GNN + Tabular multimodal model.

Replaces PointNet with DGCNN (EdgeConv) for local surface topology.
"""

import sys
import json
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import TARGET_COLUMNS, FEATURE_COLUMNS
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.pointcloud.dgcnn import DGCNNEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.multimodal_pointcloud import PointCloudFusion
from src.training.trainer import Trainer
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud

import torch.nn as nn


class DGCNNMultimodalModel(nn.Module):
    """DGCNN + GNN + Tabular with cross-attention fusion."""

    def __init__(self, config=None, pretrained_gnn_path=None):
        super().__init__()
        config = config or {}
        mc = config.get("model", {})

        # DGCNN encoder (replaces PointNet)
        pc_config = mc.get("pointcloud", {})
        pc_feat_dim = pc_config.get("feature_dim", 256)
        self.surface_encoder = DGCNNEncoder(
            in_channels=7,
            feature_dim=pc_feat_dim,
            k=min(20, 1024 // 4),  # k neighbors
            dropout=0.3,
        )

        # GNN encoder
        gc = mc.get("graph", {})
        graph_hidden = gc.get("hidden_dim", 256)
        self.gnn = MolecularGNN(
            atom_feature_dim=gc.get("atom_feature_dim", 22),
            bond_feature_dim=gc.get("bond_feature_dim", 7),
            hidden_dim=graph_hidden,
            num_layers=gc.get("num_layers", 4),
            conv_type=gc.get("conv_type", "GAT"),
            heads=gc.get("heads", 4),
            dropout=gc.get("dropout", 0.3),
            pooling=gc.get("pooling", "mean"),
            num_targets=0,
        )

        # Load pre-trained GNN
        if pretrained_gnn_path and Path(pretrained_gnn_path).exists():
            ckpt = torch.load(pretrained_gnn_path, map_location="cpu", weights_only=True)
            gnn_state = {k: v for k, v in ckpt.items()
                         if any(k.startswith(p) for p in ["atom_projection", "convs", "batch_norms", "pool"])}
            if gnn_state:
                self.gnn.load_state_dict(gnn_state, strict=False)
                print(f"  Loaded pre-trained GNN: {len(gnn_state)} params")

        # Fusion
        fc = mc.get("fusion", {})
        fused_dim = fc.get("fused_dim", 256)
        self.fusion = PointCloudFusion(
            pointcloud_dim=pc_feat_dim,
            graph_dim=graph_hidden,
            tabular_dim=len(FEATURE_COLUMNS),
            fused_dim=fused_dim,
            num_heads=8,
            dropout=0.3,
        )

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.BatchNorm1d(fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fused_dim // 2, 7),
        )

    def forward(self, point_cloud, features, atom_features, edge_index, bond_features, batch, **kwargs):
        pc_feat = self.surface_encoder(point_cloud)
        graph_feat = self.gnn.get_features(atom_features, edge_index, bond_features, batch)
        fused = self.fusion(pc_feat, graph_feat, features)
        return self.prediction_head(fused)


def main():
    config = load_config("configs/default.yaml")
    config["training"]["num_epochs"] = 200
    config["training"]["learning_rate"] = 1e-4
    config["training"]["early_stopping_patience"] = 25
    config["experiment"]["checkpoint_dir"] = "checkpoints/dgcnn"

    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    model = DGCNNMultimodalModel(config=config, pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DGCNN Model params: {n_params:,}")

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

    results = {"model": "dgcnn_multimodal", "n_params": n_params,
               "best_val_loss": trainer.best_val_loss,
               "epochs_trained": len(history["train_loss"])}
    if "test_metrics" in history:
        results["test_metrics"] = {k: float(v) if not (isinstance(v, float) and np.isnan(v)) else None
                                   for k, v in history["test_metrics"].items()}

    with open("results/dgcnn_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/dgcnn_results.json")


if __name__ == "__main__":
    main()
