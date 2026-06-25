"""Evaluation metrics for ionic liquid property prediction."""

import numpy as np
import torch


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, target_names: list = None) -> dict:
    """Compute regression metrics for each target property.

    Args:
        predictions: (N, num_targets)
        targets: (N, num_targets)
        target_names: list of target column names

    Returns:
        dict with MAE, RMSE, R2, MAPE per target and overall
    """
    target_names = target_names or [
        "gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"
    ]

    metrics = {}
    all_mae, all_rmse, all_r2 = [], [], []

    for i, name in enumerate(target_names):
        pred = predictions[:, i]
        true = targets[:, i]

        # Filter NaN values
        valid = ~(np.isnan(pred) | np.isnan(true))
        pred_v = pred[valid]
        true_v = true[valid]

        if len(pred_v) == 0:
            mae = rmse = r2 = mape = float("nan")
            metrics[f"{name}_mae"] = mae
            metrics[f"{name}_rmse"] = rmse
            metrics[f"{name}_r2"] = r2
            metrics[f"{name}_mape"] = mape
            continue

        mae = np.mean(np.abs(pred_v - true_v))
        rmse = np.sqrt(np.mean((pred_v - true_v) ** 2))

        ss_res = np.sum((true_v - pred_v) ** 2)
        ss_tot = np.sum((true_v - np.mean(true_v)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

        # MAPE (avoid division by zero)
        nonzero = np.abs(true_v) > 1e-8
        if nonzero.sum() > 0:
            mape = np.mean(np.abs((true_v[nonzero] - pred_v[nonzero]) / true_v[nonzero])) * 100
        else:
            mape = float("nan")

        metrics[f"{name}_mae"] = mae
        metrics[f"{name}_rmse"] = rmse
        metrics[f"{name}_r2"] = r2
        metrics[f"{name}_mape"] = mape

        all_mae.append(mae)
        all_rmse.append(rmse)
        all_r2.append(r2)

    metrics["avg_mae"] = np.mean(all_mae)
    metrics["avg_rmse"] = np.mean(all_rmse)
    metrics["avg_r2"] = np.mean(all_r2)

    return metrics


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """Format metrics dict into a readable string."""
    lines = []
    if prefix:
        lines.append(f"--- {prefix} ---")

    # Overall
    lines.append(f"  Avg MAE: {metrics.get('avg_mae', 0):.4f}  "
                  f"Avg RMSE: {metrics.get('avg_rmse', 0):.4f}  "
                  f"Avg R2: {metrics.get('avg_r2', 0):.4f}")

    # Per-target
    target_names = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]
    for name in target_names:
        mae = metrics.get(f"{name}_mae", 0)
        rmse = metrics.get(f"{name}_rmse", 0)
        r2 = metrics.get(f"{name}_r2", 0)
        lines.append(f"  {name:>8s}: MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

    return "\n".join(lines)
