"""COSMO-RS sigma profile features + GNN model.

Extracts sigma profiles (charge density histograms) from the COSMO surface
ESP values and uses them as tabular features alongside the GNN.

Sigma profiles are the standard representation in COSMO-RS thermodynamic
models and directly encode solvation thermodynamics.
"""

import sys
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import TARGET_COLUMNS, FEATURE_COLUMNS
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.graph.gnn import MolecularGNN
from src.data.dataset import ILMultimodalDataset, collate_multimodal
from src.training.trainer import Trainer


N_SIGMA_BINS = 20  # Number of bins for sigma profile histogram
SIGMA_RANGE = (-0.03, 0.03)  # Charge density range (e/Å²)


def extract_sigma_profile(npz_path, n_bins=N_SIGMA_BINS):
    """Extract sigma profile from a point cloud .npz file.

    The sigma profile p(sigma) is a histogram of surface charge densities,
    where sigma = ESP value at each surface point, weighted by the surface
    area element of each point.

    Returns:
        (n_bins,) array — normalized sigma profile histogram
    """
    data = np.load(npz_path)
    points = data["points"]  # (N, 7): x,y,z,nx,ny,nz,esp
    esp = points[:, 6]  # ESP values

    # Compute histogram (area-weighted would be better but uniform is reasonable
    # since points are from farthest-point sampling = ~uniform area)
    hist, _ = np.histogram(esp, bins=n_bins, range=SIGMA_RANGE, density=True)

    # Normalize to sum to 1
    hist = hist / (hist.sum() + 1e-10)
    return hist.astype(np.float32)


def compute_all_sigma_profiles(pc_dir, index_path):
    """Compute sigma profiles for all point clouds."""
    pc_dir = Path(pc_dir)
    profiles = {}  # smiles -> sigma_profile

    if not index_path.exists():
        print("  WARNING: No point cloud index found")
        return profiles

    idx_df = pd.read_csv(index_path)
    for _, row in idx_df.iterrows():
        smiles = row["smiles"]
        npz_path = pc_dir / row["filename"]
        if npz_path.exists():
            try:
                profiles[smiles] = extract_sigma_profile(str(npz_path))
            except Exception:
                pass

    print(f"  Sigma profiles computed: {len(profiles)} ILs")
    return profiles


def add_sigma_profiles_to_csv(csv_path, profiles, output_path):
    """Add sigma profile columns to a data CSV."""
    df = pd.read_csv(csv_path)
    sigma_cols = [f"sigma_bin_{i}" for i in range(N_SIGMA_BINS)]

    for col in sigma_cols:
        df[col] = 0.0

    for idx, row in df.iterrows():
        smiles = row.get("smiles", "")
        if smiles in profiles:
            for j, col in enumerate(sigma_cols):
                df.at[idx, col] = profiles[smiles][j]

    df.to_csv(output_path, index=False)
    n_with = (df[sigma_cols[0]] != 0).sum()
    print(f"  Saved {output_path}: {n_with}/{len(df)} rows with sigma profiles")
    return sigma_cols


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    # ── Step 1: Compute sigma profiles ──
    print("\nComputing sigma profiles...")
    pc_dir = Path("data/pipeline/point_clouds")
    profiles = compute_all_sigma_profiles(pc_dir, pc_dir / "index.csv")

    # ── Step 2: Add to split CSVs ──
    print("\nAugmenting split CSVs with sigma profiles...")
    splits_dir = Path("data/processed/splits")
    augmented_dir = Path("data/processed/splits_sigma")
    augmented_dir.mkdir(parents=True, exist_ok=True)

    sigma_cols = None
    for split in ["train", "val", "test"]:
        sigma_cols = add_sigma_profiles_to_csv(
            splits_dir / f"{split}.csv",
            profiles,
            augmented_dir / f"{split}.csv",
        )

    # ── Step 3: Build GNN with extended features ──
    # Original FEATURE_COLUMNS (25) + sigma profile bins (20) = 45 features
    extended_features = FEATURE_COLUMNS + sigma_cols
    aux_dim = len(extended_features)
    print(f"\nExtended feature dim: {aux_dim} (25 original + {N_SIGMA_BINS} sigma bins)")

    model = MolecularGNN(
        atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
        hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
        dropout=0.3, pooling="mean", num_targets=7,
        aux_feature_dim=aux_dim,
    )

    # Load Phase 2 pre-trained weights (partial match)
    pretrained_path = Path("checkpoints/transfer/pretrained.pt")
    if pretrained_path.exists():
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        gnn_state = {k: v for k, v in ckpt.items()
                     if any(k.startswith(p) for p in ["atom_projection", "convs", "batch_norms", "pool"])}
        if gnn_state:
            model.load_state_dict(gnn_state, strict=False)
            print(f"  Loaded pre-trained GNN backbone: {len(gnn_state)} params")

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    # ── Step 4: Build custom dataset that reads extended features ──
    # We need a modified dataset that reads sigma columns too
    from src.data.dataset import ILMultimodalDataset

    # Monkey-patch FEATURE_COLUMNS for this run
    import src.data.preprocessing as prep
    import src.data.dataset as ds
    original_fc = prep.FEATURE_COLUMNS
    prep.FEATURE_COLUMNS = extended_features
    ds.FEATURE_COLUMNS = extended_features

    config["training"]["num_epochs"] = 200
    config["training"]["learning_rate"] = 1e-4
    config["training"]["early_stopping_patience"] = 25
    config["experiment"]["checkpoint_dir"] = "checkpoints/sigma_profile"

    graph_cache = "data/processed/graphs.pkl"
    graph_path = graph_cache if Path(graph_cache).exists() else None

    train_ds = ILMultimodalDataset(str(augmented_dir / "train.csv"), graph_path, is_train=True, config=config)
    val_ds = ILMultimodalDataset(str(augmented_dir / "val.csv"), graph_path, is_train=False, config=config)
    test_ds = ILMultimodalDataset(str(augmented_dir / "test.csv"), graph_path, is_train=False, config=config)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_multimodal)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_multimodal)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_multimodal)

    # ── Step 5: Train ──
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                      test_loader=test_loader, config=config, device=device)
    history = trainer.train(verbose=True)

    # Restore original
    prep.FEATURE_COLUMNS = original_fc
    ds.FEATURE_COLUMNS = original_fc

    results = {"model": "sigma_profile_gnn", "n_params": n_params,
               "n_sigma_bins": N_SIGMA_BINS, "total_features": aux_dim,
               "best_val_loss": trainer.best_val_loss,
               "epochs_trained": len(history["train_loss"])}
    if "test_metrics" in history:
        results["test_metrics"] = {k: float(v) if not (isinstance(v, float) and np.isnan(v)) else None
                                   for k, v in history["test_metrics"].items()}

    with open("results/sigma_profile_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/sigma_profile_results.json")


if __name__ == "__main__":
    main()
