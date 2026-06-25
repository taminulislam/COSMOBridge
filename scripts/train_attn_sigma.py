"""Attention-weighted sigma profile + GNN model.

Instead of treating sigma profile bins as fixed features, learns which
charge density regions matter most for each property via attention.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import TARGET_COLUMNS, FEATURE_COLUMNS, THERMO_FEATURES
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.graph.gnn import MolecularGNN
from src.data.dataset import ILMultimodalDataset, collate_multimodal
from src.training.trainer import Trainer

N_SIGMA_BINS = 50  # Finer histogram for attention


class SigmaProfileAttention(nn.Module):
    """Learns property-aware weighting of sigma profile bins.

    Takes raw sigma profile histogram and produces a weighted feature
    vector where the weights are learned per-property context.
    """

    def __init__(self, n_bins=50, hidden_dim=64, output_dim=32):
        super().__init__()
        # Bin embedding (treat bins as a sequence)
        self.bin_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Attention over bins
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        self.n_bins = n_bins

    def forward(self, sigma_profile):
        """
        Args:
            sigma_profile: (B, n_bins) histogram values
        Returns:
            (B, output_dim) attended sigma features
        """
        B = sigma_profile.shape[0]
        # (B, n_bins, 1) -> embed each bin value
        x = sigma_profile.unsqueeze(-1)  # (B, N, 1)
        x = self.bin_embed(x)  # (B, N, hidden)

        # Self-attention over bins
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attended = torch.matmul(attn, V)  # (B, N, hidden)

        # Pool across bins
        pooled = attended.mean(dim=1)  # (B, hidden)
        return self.out_proj(pooled)


class AttnSigmaGNN(nn.Module):
    """GNN with attention-weighted sigma profile features."""

    def __init__(self, n_sigma_bins=50, sigma_dim=32, pretrained_gnn_path=None):
        super().__init__()

        # GNN
        self.gnn = MolecularGNN(
            atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
            dropout=0.3, pooling="mean", num_targets=0,
        )

        # Load pre-trained GNN
        if pretrained_gnn_path and Path(pretrained_gnn_path).exists():
            ckpt = torch.load(pretrained_gnn_path, map_location="cpu", weights_only=True)
            gnn_state = {k: v for k, v in ckpt.items()
                         if any(k.startswith(p) for p in ["atom_projection", "convs", "batch_norms", "pool"])}
            if gnn_state:
                self.gnn.load_state_dict(gnn_state, strict=False)
                print(f"  Loaded pre-trained GNN: {len(gnn_state)} params")

        # Sigma profile attention
        self.sigma_attn = SigmaProfileAttention(n_sigma_bins, hidden_dim=64, output_dim=sigma_dim)

        # Thermo feature dim
        thermo_dim = len(THERMO_FEATURES)

        # Prediction head: GNN(256) + sigma_attn(32) + thermo(5) = 293
        pred_dim = 256 + sigma_dim + thermo_dim
        self.prediction_head = nn.Sequential(
            nn.Linear(pred_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 7),
        )

        self.n_sigma_bins = n_sigma_bins

    def forward(self, atom_features, edge_index, bond_features, batch,
                features=None, sigma_profile=None, **kwargs):
        """
        features: (B, feature_dim) — all features including sigma bins appended
        """
        graph_feat = self.gnn.get_features(atom_features, edge_index, bond_features, batch)

        # Split features into thermo and sigma profile
        thermo_feat = features[:, :len(THERMO_FEATURES)]

        # Sigma profile is appended after FEATURE_COLUMNS
        sigma_start = len(FEATURE_COLUMNS)
        sigma_prof = features[:, sigma_start:sigma_start + self.n_sigma_bins]

        sigma_feat = self.sigma_attn(sigma_prof)

        combined = torch.cat([graph_feat, sigma_feat, thermo_feat], dim=-1)
        return self.prediction_head(combined)


def extract_sigma_profiles_fine(pc_dir, n_bins=N_SIGMA_BINS):
    """Extract fine-grained sigma profiles from point clouds."""
    pc_dir = Path(pc_dir)
    index_path = pc_dir / "index.csv"
    profiles = {}

    if not index_path.exists():
        return profiles

    idx_df = pd.read_csv(index_path)
    for _, row in idx_df.iterrows():
        npz_path = pc_dir / row["filename"]
        if npz_path.exists():
            try:
                data = np.load(npz_path)
                esp = data["points"][:, 6]
                hist, _ = np.histogram(esp, bins=n_bins, range=(-0.03, 0.03), density=True)
                hist = hist / (hist.sum() + 1e-10)
                profiles[row["smiles"]] = hist.astype(np.float32)
            except Exception:
                pass

    print(f"  Fine sigma profiles ({n_bins} bins): {len(profiles)} ILs")
    return profiles


def augment_csv_with_sigma(csv_path, profiles, output_path, n_bins):
    """Add sigma profile columns to CSV."""
    df = pd.read_csv(csv_path)
    sigma_cols = [f"sigma_bin_{i}" for i in range(n_bins)]
    for col in sigma_cols:
        df[col] = 0.0
    for idx, row in df.iterrows():
        smi = row.get("smiles", "")
        if smi in profiles:
            for j, col in enumerate(sigma_cols):
                df.at[idx, col] = profiles[smi][j]
    df.to_csv(output_path, index=False)
    return sigma_cols


def main():
    config = load_config("configs/default.yaml")
    config["training"]["num_epochs"] = 200
    config["training"]["learning_rate"] = 1e-4
    config["training"]["early_stopping_patience"] = 25
    config["experiment"]["checkpoint_dir"] = "checkpoints/attn_sigma"

    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    # Extract fine sigma profiles
    print("\nExtracting sigma profiles...")
    profiles = extract_sigma_profiles_fine("data/pipeline/point_clouds", N_SIGMA_BINS)

    # Augment CSVs
    splits_dir = Path("data/processed/splits")
    aug_dir = Path("data/processed/splits_attn_sigma")
    aug_dir.mkdir(parents=True, exist_ok=True)

    sigma_cols = None
    for split in ["train", "val", "test"]:
        sigma_cols = augment_csv_with_sigma(
            splits_dir / f"{split}.csv", profiles,
            aug_dir / f"{split}.csv", N_SIGMA_BINS)

    # Extend FEATURE_COLUMNS for dataset loading
    import src.data.preprocessing as prep
    import src.data.dataset as ds
    extended = FEATURE_COLUMNS + sigma_cols
    prep.FEATURE_COLUMNS = extended
    ds.FEATURE_COLUMNS = extended

    model = AttnSigmaGNN(
        n_sigma_bins=N_SIGMA_BINS, sigma_dim=32,
        pretrained_gnn_path="checkpoints/transfer/pretrained.pt",
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    graph_cache = "data/processed/graphs.pkl"
    graph_path = graph_cache if Path(graph_cache).exists() else None

    train_ds = ILMultimodalDataset(str(aug_dir / "train.csv"), graph_path, is_train=True, config=config)
    val_ds = ILMultimodalDataset(str(aug_dir / "val.csv"), graph_path, is_train=False, config=config)
    test_ds = ILMultimodalDataset(str(aug_dir / "test.csv"), graph_path, is_train=False, config=config)

    from src.data.dataset import collate_multimodal
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_multimodal)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_multimodal)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_multimodal)

    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                      test_loader=test_loader, config=config, device=device)
    history = trainer.train(verbose=True)

    # Restore
    prep.FEATURE_COLUMNS = FEATURE_COLUMNS
    ds.FEATURE_COLUMNS = FEATURE_COLUMNS

    results = {"model": "attn_sigma_gnn", "n_params": n_params, "n_sigma_bins": N_SIGMA_BINS,
               "best_val_loss": trainer.best_val_loss}
    if "test_metrics" in history:
        results["test_metrics"] = {k: float(v) if not (isinstance(v, float) and np.isnan(v)) else None
                                   for k, v in history["test_metrics"].items()}
    with open("results/attn_sigma_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/attn_sigma_results.json")


if __name__ == "__main__":
    main()
