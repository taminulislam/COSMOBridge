"""Chemprop + COSMO surface features (Strategy #1).

Extracts PointNet features from COSMO surface point clouds and feeds them
as auxiliary features to Chemprop alongside thermodynamic features.

This combines:
- Chemprop's efficient D-MPNN (best avg R²=0.770)
- Our COSMO surface information (best gamma1 R²=0.887)

Step 1: Extract PointNet features from trained PointCloud model
Step 2: Save as feature files for Chemprop
Step 3: Train Chemprop with SMILES + thermo features + PointNet features
"""

import sys
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.models.fusion.multimodal_pointcloud import MultimodalPointCloudModel
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud

THERMO_FEATURES = ["temperature", "x1", "inv_temperature", "temp_squared", "temp_cubed"]


def extract_pointnet_features(model, loader, device):
    """Extract PointNet encoder features (before fusion) for each sample."""
    model.eval()
    all_pc_feats = []
    all_graph_feats = []

    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Extract modality-specific features (before fusion)
            pc_feat = model.pointnet(batch["point_cloud"])  # (B, 256)
            graph_feat = model.gnn.get_features(
                batch["atom_features"], batch["edge_index"],
                batch["bond_features"], batch["batch"])  # (B, 256)

            all_pc_feats.append(pc_feat.cpu().numpy())
            all_graph_feats.append(graph_feat.cpu().numpy())

    return np.concatenate(all_pc_feats), np.concatenate(all_graph_feats)


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    pc_dir = "data/pipeline/point_clouds"
    orig_splits = Path("data/processed/splits")
    tmp_dir = Path("data/chemprop_cosmo")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # STEP 1: Load trained PointCloud model, extract features
    # ══════════════════════════════════════════════════════════
    print("Step 1: Extracting PointNet + GNN features from trained model...")

    model = MultimodalPointCloudModel(config=config, pretrained_gnn_path=None)
    ckpt = torch.load("checkpoints/pointcloud/best_model.pt",
                       map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    print(f"  Loaded PointCloud model")

    for split in ["train", "val", "test"]:
        ds = PointCloudMultimodalDataset(str(orig_splits / f"{split}.csv"), pc_dir, is_train=False)
        ldr = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)
        pc_feats, graph_feats = extract_pointnet_features(model, ldr, device)
        print(f"  {split}: PointNet={pc_feats.shape}, GNN={graph_feats.shape}")

        # Save features
        np.save(tmp_dir / f"{split}_pointnet_feats.npy", pc_feats)
        np.save(tmp_dir / f"{split}_gnn_feats.npy", graph_feats)

    # ══════════════════════════════════════════════════════════
    # STEP 2: Prepare Chemprop data with extracted features
    # ══════════════════════════════════════════════════════════
    print("\nStep 2: Preparing Chemprop data files...")

    for split in ["train", "val", "test"]:
        df = pd.read_csv(orig_splits / f"{split}.csv")

        # Main data file (SMILES + targets)
        out = pd.DataFrame()
        out["smiles"] = df["smiles"]
        for t in TARGET_COLUMNS:
            out[t] = df[t]
        out.to_csv(tmp_dir / f"{split}.csv", index=False)

        # Features file: thermo + PointNet features + GNN features
        pc_feats = np.load(tmp_dir / f"{split}_pointnet_feats.npy")
        gnn_feats = np.load(tmp_dir / f"{split}_gnn_feats.npy")

        feat_df = pd.DataFrame()
        # Thermo features
        for f in THERMO_FEATURES:
            if f in df.columns:
                feat_df[f] = df[f]
        # PointNet features (256D)
        for i in range(pc_feats.shape[1]):
            feat_df[f"pn_{i}"] = pc_feats[:, i]
        # GNN features (256D)
        for i in range(gnn_feats.shape[1]):
            feat_df[f"gnn_{i}"] = gnn_feats[:, i]

        feat_df.to_csv(tmp_dir / f"{split}_features.csv", index=False)
        print(f"  {split}: {len(out)} samples, {feat_df.shape[1]} features "
              f"({len(THERMO_FEATURES)} thermo + {pc_feats.shape[1]} PointNet + {gnn_feats.shape[1]} GNN)")

    # ══════════════════════════════════════════════════════════
    # STEP 3: Train Chemprop variants
    # ══════════════════════════════════════════════════════════

    # Also prepare thermo-only features for comparison
    for split in ["train", "val", "test"]:
        df = pd.read_csv(orig_splits / f"{split}.csv")
        feat_df = df[THERMO_FEATURES].copy()
        feat_df.to_csv(tmp_dir / f"{split}_thermo_only.csv", index=False)

    variants = [
        ("Chemprop (thermo only)", "thermo_only", "chemprop_thermo"),
        ("Chemprop + PointNet + GNN", "features", "chemprop_cosmo"),
    ]

    results = {}

    for name, feat_suffix, save_name in variants:
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")

        ckpt_dir = Path(f"checkpoints/{save_name}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "chemprop_train",
            "--data_path", str(tmp_dir / "train.csv"),
            "--separate_val_path", str(tmp_dir / "val.csv"),
            "--separate_test_path", str(tmp_dir / "test.csv"),
            "--features_path", str(tmp_dir / f"train_{feat_suffix}.csv"),
            "--separate_val_features_path", str(tmp_dir / f"val_{feat_suffix}.csv"),
            "--separate_test_features_path", str(tmp_dir / f"test_{feat_suffix}.csv"),
            "--save_dir", str(ckpt_dir),
            "--dataset_type", "regression",
            "--smiles_columns", "smiles",
            "--target_columns", *TARGET_COLUMNS,
            "--epochs", "100",
            "--batch_size", "32",
            "--hidden_size", "300",
            "--depth", "3",
            "--ffn_num_layers", "2",
            "--ffn_hidden_size", "300",
            "--dropout", "0.2",
            "--metric", "rmse",
            "--extra_metrics", "r2", "mae",
            "--seed", "42",
            "--num_folds", "1",
            "--gpu", "0",
            "--quiet",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-500:]}")
            continue

        # Load scores
        scores_path = ckpt_dir / "fold_0" / "test_scores.json"
        if scores_path.exists():
            scores = json.load(open(scores_path))
            metrics = {}
            for i, p in enumerate(TARGET_COLUMNS):
                metrics[f"{p}_r2"] = scores["r2"][i]
                metrics[f"{p}_mae"] = scores["mae"][i]
                metrics[f"{p}_rmse"] = scores["rmse"][i]
            metrics["avg_r2"] = np.mean(scores["r2"])
            metrics["avg_mae"] = np.mean(scores["mae"])
            metrics["avg_rmse"] = np.mean(scores["rmse"])

            print(f"\n  Results:")
            for p in TARGET_COLUMNS:
                print(f"    {p:15s} R² = {metrics[f'{p}_r2']:.4f}")
            print(f"    {'AVERAGE':15s} R² = {metrics['avg_r2']:.4f}")

            results[save_name] = {
                "name": name,
                "metrics": {k: float(v) for k, v in metrics.items()},
            }
        else:
            print(f"  WARNING: No scores found at {scores_path}")

    # ══════════════════════════════════════════════════════════
    # FINAL COMPARISON
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")

    prev = {}
    for name, path, key in [
        ("Chemprop (orig)", "results/chemprop_results.json", "test_metrics"),
        ("PointCloud", "results/pointcloud_results.json", None),
        ("Ens Fix6+PC", "results/ensemble_fix6_pointcloud.json", "simple_average"),
    ]:
        try:
            data = json.load(open(path))
            m = data.get(key) if key else None
            if m is None:
                for k in ['metrics', 'test_metrics']:
                    if k in data: m = data[k]; break
            prev[name] = m
        except:
            pass

    header = "  {:<20s}".format("Property")
    for name in prev:
        header += " {:>14s}".format(name)
    for save_name, info in results.items():
        header += " {:>14s}".format(info["name"][:14])
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<20s}".format(p)
        for name in prev:
            line += " {:14.4f}".format(prev[name].get(key, float('nan')))
        for save_name, info in results.items():
            line += " {:14.4f}".format(info["metrics"].get(key, float('nan')))
        print(line)

    line = "  {:<20s}".format("AVERAGE")
    for name in prev:
        line += " {:14.4f}".format(prev[name].get('avg_r2', float('nan')))
    for save_name, info in results.items():
        line += " {:14.4f}".format(info["metrics"].get('avg_r2', float('nan')))
    print(line)

    # Save
    with open("results/chemprop_cosmo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/chemprop_cosmo_results.json")


if __name__ == "__main__":
    main()
