"""Multi-fidelity ESP learning.

Uses two point cloud channels — Gasteiger ESP (fast/approximate, all ILs)
and learns a fidelity-aware representation. When DFT ESP becomes available,
the model can leverage both fidelities.

Currently: augments training by adding Gaussian noise to ESP as "low fidelity"
and clean ESP as "high fidelity", with a fidelity embedding that teaches
the model to bridge quality gaps.
"""

import sys
import json
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import TARGET_COLUMNS, FEATURE_COLUMNS
from src.data.graph_builder import smiles_to_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.multimodal_pointcloud import PointCloudFusion
from src.training.trainer import Trainer
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud


class MultiFidelityPointCloudDataset(PointCloudMultimodalDataset):
    """Extends point cloud dataset with fidelity augmentation.

    During training, randomly presents either:
    - High fidelity (clean Gasteiger ESP) with fidelity_id=1
    - Low fidelity (noisy ESP) with fidelity_id=0
    This teaches the model to be robust to ESP quality variations.
    """

    def __init__(self, csv_path, point_cloud_dir, is_train=True,
                 n_points=1024, noise_std=0.05):
        super().__init__(csv_path, point_cloud_dir, is_train, n_points)
        self.noise_std = noise_std

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        if self.is_train and np.random.random() < 0.5:
            # Low fidelity: add noise to ESP channel
            pc = sample["point_cloud"].clone()
            pc[:, 6] += torch.randn(pc.shape[0]) * self.noise_std
            sample["point_cloud"] = pc
            sample["fidelity"] = torch.tensor(0, dtype=torch.long)
        else:
            sample["fidelity"] = torch.tensor(1, dtype=torch.long)

        return sample


def collate_multifidelity(batch):
    """Collate with fidelity indicator."""
    base = collate_pointcloud(batch)
    base["fidelity"] = torch.stack([b["fidelity"] for b in batch])
    return base


class MultiFidelityModel(nn.Module):
    """PointNet + GNN + Tabular with fidelity-aware encoding."""

    def __init__(self, config=None, pretrained_gnn_path=None):
        super().__init__()
        config = config or {}
        mc = config.get("model", {})

        pc_feat_dim = 256
        graph_hidden = 256

        # PointNet encoder
        self.pointnet = PointNetEncoder(in_channels=7, feature_dim=pc_feat_dim, dropout=0.3)

        # Fidelity embedding (learned correction)
        self.fidelity_embed = nn.Embedding(2, pc_feat_dim)
        self.fidelity_gate = nn.Sequential(
            nn.Linear(pc_feat_dim * 2, pc_feat_dim),
            nn.Sigmoid(),
        )

        # GNN
        gc = mc.get("graph", {})
        self.gnn = MolecularGNN(
            atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=graph_hidden, num_layers=4, conv_type="GAT", heads=4,
            dropout=0.3, pooling="mean", num_targets=0,
        )

        if pretrained_gnn_path and Path(pretrained_gnn_path).exists():
            ckpt = torch.load(pretrained_gnn_path, map_location="cpu", weights_only=True)
            gnn_state = {k: v for k, v in ckpt.items()
                         if any(k.startswith(p) for p in ["atom_projection", "convs", "batch_norms", "pool"])}
            if gnn_state:
                self.gnn.load_state_dict(gnn_state, strict=False)
                print(f"  Loaded pre-trained GNN: {len(gnn_state)} params")

        # Fusion
        self.fusion = PointCloudFusion(
            pointcloud_dim=pc_feat_dim, graph_dim=graph_hidden,
            tabular_dim=len(FEATURE_COLUMNS), fused_dim=256,
            num_heads=8, dropout=0.3,
        )

        # Prediction
        self.prediction_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 7),
        )

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, fidelity=None, **kwargs):
        pc_feat = self.pointnet(point_cloud)

        # Apply fidelity-aware gating
        if fidelity is not None:
            fid_emb = self.fidelity_embed(fidelity)  # (B, pc_feat_dim)
            gate = self.fidelity_gate(torch.cat([pc_feat, fid_emb], dim=-1))
            pc_feat = pc_feat * gate  # Fidelity-modulated features

        graph_feat = self.gnn.get_features(atom_features, edge_index, bond_features, batch)
        fused = self.fusion(pc_feat, graph_feat, features)
        return self.prediction_head(fused)


def main():
    config = load_config("configs/default.yaml")
    config["training"]["num_epochs"] = 200
    config["training"]["learning_rate"] = 1e-4
    config["training"]["early_stopping_patience"] = 25
    config["experiment"]["checkpoint_dir"] = "checkpoints/multifidelity"

    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    model = MultiFidelityModel(config=config, pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Multi-fidelity model params: {n_params:,}")

    splits_dir = Path("data/processed/splits")
    pc_dir = "data/pipeline/point_clouds"

    print("\nLoading datasets...")
    train_ds = MultiFidelityPointCloudDataset(str(splits_dir / "train.csv"), pc_dir, is_train=True)
    val_ds = MultiFidelityPointCloudDataset(str(splits_dir / "val.csv"), pc_dir, is_train=False)
    test_ds = MultiFidelityPointCloudDataset(str(splits_dir / "test.csv"), pc_dir, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_multifidelity)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_multifidelity)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_multifidelity)

    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                      test_loader=test_loader, config=config, device=device)
    history = trainer.train(verbose=True)

    results = {"model": "multifidelity_pointcloud", "n_params": n_params,
               "best_val_loss": trainer.best_val_loss}
    if "test_metrics" in history:
        results["test_metrics"] = {k: float(v) if not (isinstance(v, float) and np.isnan(v)) else None
                                   for k, v in history["test_metrics"].items()}
    with open("results/multifidelity_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/multifidelity_results.json")


if __name__ == "__main__":
    main()
