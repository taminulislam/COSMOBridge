"""Ensemble predictions from GNN and Tabular models with per-target weights.

The GNN excels on identity-driven targets (gamma1-G_mix) while Tabular v2
excels on temperature-driven targets (H_vap, P). This script learns optimal
per-target blending weights on the validation set, then evaluates on test.

Usage:
    python scripts/ensemble.py \
        --gnn-checkpoint checkpoints/gnn_improved/best_model.pt \
        --gnn-config configs/gnn_improved.yaml \
        --tabular-checkpoint checkpoints/tabular_improved/best_model.pt \
        --tabular-config configs/tabular_improved.yaml \
        --save-results results/ensemble_results.json
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device, set_seed
from src.data.dataset import ILTabularDataset, ILMultimodalDataset, collate_multimodal
from src.training.metrics import compute_metrics, format_metrics
from scripts.train import build_model


def get_predictions(model, loader, device):
    """Get model predictions on a data loader."""
    model.eval()
    all_preds = []
    all_targets = []

    import inspect
    sig = inspect.signature(model.forward)
    params = list(sig.parameters.keys())
    key_aliases = {"batch": "graph_batch"}

    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            if "atom_features" in params:
                kwargs = {}
                for p in params:
                    if p == "kwargs":
                        continue
                    if p in batch:
                        kwargs[p] = batch[p]
                    elif p in key_aliases and key_aliases[p] in batch:
                        kwargs[p] = batch[key_aliases[p]]
                pred = model(**kwargs)
            else:
                pred = model(
                    features=batch["features"],
                    il_idx=batch["il_idx"],
                    cation_idx=batch["cation_idx"],
                    anion_idx=batch["anion_idx"],
                )

            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def optimize_weights(gnn_preds, tab_preds, targets, n_targets=7):
    """Find optimal per-target blending weight via grid search on val set.

    weight_i: how much GNN contributes for target i (0=all tabular, 1=all GNN)
    """
    best_weights = np.ones(n_targets) * 0.5
    best_r2 = np.zeros(n_targets) - 999

    for i in range(n_targets):
        for w in np.arange(0.0, 1.05, 0.05):
            blended = w * gnn_preds[:, i] + (1 - w) * tab_preds[:, i]
            ss_res = np.sum((blended - targets[:, i]) ** 2)
            ss_tot = np.sum((targets[:, i] - targets[:, i].mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            if r2 > best_r2[i]:
                best_r2[i] = r2
                best_weights[i] = w

    return best_weights


def main():
    parser = argparse.ArgumentParser(description="Ensemble GNN + Tabular")
    parser.add_argument("--gnn-checkpoint", type=str, required=True)
    parser.add_argument("--gnn-config", type=str, required=True)
    parser.add_argument("--tabular-checkpoint", type=str, required=True)
    parser.add_argument("--tabular-config", type=str, required=True)
    parser.add_argument("--save-results", type=str, default=None)
    args = parser.parse_args()

    # Load configs
    gnn_config = load_config(args.gnn_config)
    tab_config = load_config(args.tabular_config)
    device = get_device(gnn_config)
    set_seed(42)

    processed_dir = Path(gnn_config.get("data", {}).get("processed_dir", "data/processed"))
    splits_dir = processed_dir / "splits"

    # Build GNN model + data
    print("Loading GNN model...")
    gnn_model = build_model("gnn", gnn_config)
    ckpt = torch.load(args.gnn_checkpoint, map_location=device, weights_only=False)
    gnn_model.load_state_dict(ckpt["model_state_dict"])
    gnn_model.to(device)

    graph_cache = str(processed_dir / "graphs.pkl")
    graph_path = graph_cache if Path(graph_cache).exists() else None
    gnn_val_ds = ILMultimodalDataset(str(splits_dir / "val.csv"), graph_path, is_train=False, config=gnn_config)
    gnn_test_ds = ILMultimodalDataset(str(splits_dir / "test.csv"), graph_path, is_train=False, config=gnn_config)
    gnn_val_loader = DataLoader(gnn_val_ds, batch_size=32, shuffle=False, collate_fn=collate_multimodal)
    gnn_test_loader = DataLoader(gnn_test_ds, batch_size=32, shuffle=False, collate_fn=collate_multimodal)

    # Build Tabular model + data
    print("Loading Tabular model...")
    tab_model = build_model("tabular", tab_config)
    ckpt = torch.load(args.tabular_checkpoint, map_location=device, weights_only=False)
    tab_model.load_state_dict(ckpt["model_state_dict"])
    tab_model.to(device)

    tab_val_ds = ILTabularDataset(str(splits_dir / "val.csv"))
    tab_test_ds = ILTabularDataset(str(splits_dir / "test.csv"))
    tab_val_loader = DataLoader(tab_val_ds, batch_size=32, shuffle=False)
    tab_test_loader = DataLoader(tab_test_ds, batch_size=32, shuffle=False)

    # Get predictions
    print("Getting predictions...")
    gnn_val_preds, val_targets = get_predictions(gnn_model, gnn_val_loader, device)
    gnn_test_preds, test_targets = get_predictions(gnn_model, gnn_test_loader, device)
    tab_val_preds, _ = get_predictions(tab_model, tab_val_loader, device)
    tab_test_preds, _ = get_predictions(tab_model, tab_test_loader, device)

    # Optimize per-target weights on validation set
    print("\nOptimizing ensemble weights on validation set...")
    weights = optimize_weights(gnn_val_preds, tab_val_preds, val_targets)

    target_names = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]
    print("Per-target GNN weights:")
    for name, w in zip(target_names, weights):
        print(f"  {name:>8s}: GNN={w:.2f}, Tabular={1-w:.2f}")

    # Apply to val
    val_ensemble = weights[None, :] * gnn_val_preds + (1 - weights[None, :]) * tab_val_preds
    val_metrics = compute_metrics(val_ensemble, val_targets)
    print(f"\nVal Ensemble: R²={val_metrics['avg_r2']:.4f}")

    # Apply to test
    test_ensemble = weights[None, :] * gnn_test_preds + (1 - weights[None, :]) * tab_test_preds
    test_metrics = compute_metrics(test_ensemble, test_targets)

    print(f"\nTest Results (Ensemble):")
    print(format_metrics(test_metrics, "Test"))

    # Compare with individual models
    gnn_test_metrics = compute_metrics(gnn_test_preds, test_targets)
    tab_test_metrics = compute_metrics(tab_test_preds, test_targets)

    print(f"\nComparison:")
    print(f"  {'Model':>12s}  {'Avg R²':>8s}  {'Avg MAE':>8s}  {'Avg RMSE':>9s}")
    print(f"  {'GNN':>12s}  {gnn_test_metrics['avg_r2']:>8.4f}  {gnn_test_metrics['avg_mae']:>8.4f}  {gnn_test_metrics['avg_rmse']:>9.4f}")
    print(f"  {'Tabular':>12s}  {tab_test_metrics['avg_r2']:>8.4f}  {tab_test_metrics['avg_mae']:>8.4f}  {tab_test_metrics['avg_rmse']:>9.4f}")
    print(f"  {'Ensemble':>12s}  {test_metrics['avg_r2']:>8.4f}  {test_metrics['avg_mae']:>8.4f}  {test_metrics['avg_rmse']:>9.4f}")

    # Save
    if args.save_results:
        results = {
            "model": "ensemble_gnn_tabular",
            "weights": {n: float(w) for n, w in zip(target_names, weights)},
            "test_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                             for k, v in test_metrics.items()},
            "gnn_test_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                                  for k, v in gnn_test_metrics.items()},
            "tabular_test_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                                      for k, v in tab_test_metrics.items()},
        }
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()
