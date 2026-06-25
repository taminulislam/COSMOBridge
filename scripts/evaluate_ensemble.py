"""Per-property ensemble: Phase 2 (H_vap, P) + Phase 3 (rest).

Loads both trained models, runs inference on test set, and combines
predictions per-property using the best model for each target.

Also optimizes per-property weights on the validation set.

Usage:
    python scripts/evaluate_ensemble.py
"""

import sys
import json
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
from src.training.metrics import compute_metrics, format_metrics


def load_phase2_model(config, device):
    """Load Phase 2 transfer learning GNN."""
    from src.models.graph.gnn import MolecularGNN
    mc = config.get("model", {}).get("graph", {})
    model = MolecularGNN(
        atom_feature_dim=ATOM_FEATURE_DIM,
        bond_feature_dim=BOND_FEATURE_DIM,
        hidden_dim=mc.get("hidden_dim", 256),
        num_layers=mc.get("num_layers", 4),
        conv_type=mc.get("conv_type", "GAT"),
        heads=mc.get("heads", 4),
        dropout=mc.get("dropout", 0.3),
        pooling=mc.get("pooling", "mean"),
        num_targets=7,
        aux_feature_dim=len(FEATURE_COLUMNS),
    )
    ckpt_path = Path("checkpoints/transfer/finetune/best_model.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Phase 2 model loaded from {ckpt_path}")
    else:
        print(f"  WARNING: {ckpt_path} not found")
    model.to(device).eval()
    return model


def load_phase3_model(config, device):
    """Load Phase 3 PointCloud multimodal model."""
    from src.models.fusion.multimodal_pointcloud import MultimodalPointCloudModel
    # Load with temp_skip=False to match the original Phase 3 checkpoint
    config_p3 = {**config}
    config_p3.setdefault("model", {})["temp_skip"] = False
    model = MultimodalPointCloudModel(config=config_p3)
    ckpt_path = Path("checkpoints/pointcloud/best_model.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Phase 3 model loaded from {ckpt_path}")
    else:
        print(f"  WARNING: {ckpt_path} not found")
    model.to(device).eval()
    return model


def predict_phase2(model, loader, device):
    """Run Phase 2 GNN inference."""
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            preds = model(
                atom_features=batch["atom_features"],
                edge_index=batch["edge_index"],
                bond_features=batch["bond_features"],
                batch=batch["graph_batch"],
                features=batch["features"],
            )
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def predict_phase3(model, loader, device):
    """Run Phase 3 PointCloud multimodal inference."""
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            preds = model(
                point_cloud=batch["point_cloud"],
                features=batch["features"],
                atom_features=batch["atom_features"],
                edge_index=batch["edge_index"],
                bond_features=batch["bond_features"],
                batch=batch["batch"],
            )
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def optimize_weights(preds_p2, preds_p3, targets):
    """Find optimal per-property weights on validation set via grid search."""
    n_targets = targets.shape[1]
    best_weights = np.zeros(n_targets)  # weight for Phase 3 (1-w for Phase 2)

    for t in range(n_targets):
        best_r2 = -np.inf
        best_w = 0.5
        for w in np.arange(0.0, 1.01, 0.05):
            blend = w * preds_p3[:, t] + (1 - w) * preds_p2[:, t]
            ss_res = np.sum((targets[:, t] - blend) ** 2)
            ss_tot = np.sum((targets[:, t] - targets[:, t].mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            if r2 > best_r2:
                best_r2 = r2
                best_w = w
        best_weights[t] = best_w

    return best_weights


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    # ── Load models ──
    print("\nLoading models...")
    model_p2 = load_phase2_model(config, device)
    model_p3 = load_phase3_model(config, device)

    # ── Build data loaders ──
    # Phase 2 uses multimodal dataset (for graph + features)
    from src.data.dataset import ILMultimodalDataset, collate_multimodal
    processed_dir = Path("data/processed")
    splits_dir = processed_dir / "splits"
    graph_cache = str(processed_dir / "graphs.pkl")
    graph_path = graph_cache if Path(graph_cache).exists() else None

    val_ds_p2 = ILMultimodalDataset(str(splits_dir / "val.csv"), graph_path, is_train=False, config=config)
    test_ds_p2 = ILMultimodalDataset(str(splits_dir / "test.csv"), graph_path, is_train=False, config=config)
    val_loader_p2 = DataLoader(val_ds_p2, batch_size=32, shuffle=False, collate_fn=collate_multimodal)
    test_loader_p2 = DataLoader(test_ds_p2, batch_size=32, shuffle=False, collate_fn=collate_multimodal)

    # Phase 3 uses point cloud dataset
    from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud
    pc_dir = "data/pipeline/point_clouds"
    val_ds_p3 = PointCloudMultimodalDataset(str(splits_dir / "val.csv"), pc_dir, is_train=False)
    test_ds_p3 = PointCloudMultimodalDataset(str(splits_dir / "test.csv"), pc_dir, is_train=False)
    val_loader_p3 = DataLoader(val_ds_p3, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)
    test_loader_p3 = DataLoader(test_ds_p3, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)

    # ── Run inference ──
    print("\nRunning inference...")
    val_preds_p2, val_targets = predict_phase2(model_p2, val_loader_p2, device)
    val_preds_p3, _ = predict_phase3(model_p3, val_loader_p3, device)

    test_preds_p2, test_targets = predict_phase2(model_p2, test_loader_p2, device)
    test_preds_p3, _ = predict_phase3(model_p3, test_loader_p3, device)

    # ── Strategy 1: Hard assignment (Phase 3 for all except H_vap, P) ──
    print(f"\n{'='*60}")
    print("STRATEGY 1: Hard Per-Property Assignment")
    print(f"{'='*60}")

    # Phase 3 for indices 0-4 (gamma1, gamma2, G_E, H_E, G_mix)
    # Phase 2 for indices 5-6 (H_vap, P)
    hard_preds = test_preds_p3.copy()
    hard_preds[:, 5] = test_preds_p2[:, 5]  # H_vap
    hard_preds[:, 6] = test_preds_p2[:, 6]  # P

    hard_metrics = compute_metrics(hard_preds, test_targets)
    print(format_metrics(hard_metrics, "Hard Ensemble"))

    # ── Strategy 2: Optimized weights per property ──
    print(f"\n{'='*60}")
    print("STRATEGY 2: Optimized Blending Weights (val set)")
    print(f"{'='*60}")

    weights = optimize_weights(val_preds_p2, val_preds_p3, val_targets)
    print("\nOptimal Phase 3 weights per property:")
    for i, name in enumerate(TARGET_COLUMNS):
        print(f"  {name:15s}: Phase3={weights[i]:.2f}, Phase2={1-weights[i]:.2f}")

    # Apply optimized weights to test set
    opt_preds = np.zeros_like(test_preds_p2)
    for t in range(len(TARGET_COLUMNS)):
        opt_preds[:, t] = weights[t] * test_preds_p3[:, t] + (1 - weights[t]) * test_preds_p2[:, t]

    opt_metrics = compute_metrics(opt_preds, test_targets)
    print(format_metrics(opt_metrics, "Optimized Ensemble"))

    # ── Comparison table ──
    print(f"\n{'='*60}")
    print("FULL COMPARISON")
    print(f"{'='*60}")

    # Load all previous results
    all_results = {
        "Baseline GNN": "results/gnn_results.json",
        "Phase 1 (+Surface)": "results/gnn_surface_results.json",
        "Phase 2 (Transfer)": "results/transfer_results.json",
        "Phase 3 (PointCloud)": "results/pointcloud_results.json",
    }

    header = f"{'Property':15s}"
    for name in all_results:
        header += f" {name:>18s}"
    header += f" {'Hard Ensemble':>18s} {'Opt Ensemble':>18s}"
    print(header)
    print("-" * len(header))

    prev_metrics = {}
    for name, path in all_results.items():
        try:
            with open(path) as f:
                prev_metrics[name] = json.load(f).get("test_metrics", {})
        except Exception:
            prev_metrics[name] = {}

    for prop in TARGET_COLUMNS + ["avg"]:
        key = f"{prop}_r2"
        row = f"{prop:15s}"
        for name in all_results:
            val = prev_metrics.get(name, {}).get(key, float("nan"))
            row += f" {val:18.4f}"
        row += f" {hard_metrics.get(key, float('nan')):18.4f}"
        row += f" {opt_metrics.get(key, float('nan')):18.4f}"
        print(row)

    # ── Save results ──
    ensemble_results = {
        "model": "ensemble_phase2_phase3",
        "hard_ensemble": {
            k: float(v) if not (isinstance(v, float) and np.isnan(v)) else None
            for k, v in hard_metrics.items()
        },
        "optimized_ensemble": {
            k: float(v) if not (isinstance(v, float) and np.isnan(v)) else None
            for k, v in opt_metrics.items()
        },
        "optimal_weights": {
            name: float(weights[i]) for i, name in enumerate(TARGET_COLUMNS)
        },
    }
    with open("results/ensemble_phase23_results.json", "w") as f:
        json.dump(ensemble_results, f, indent=2)
    print(f"\nResults saved to results/ensemble_phase23_results.json")


if __name__ == "__main__":
    main()
