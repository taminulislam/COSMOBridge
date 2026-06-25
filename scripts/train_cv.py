"""5-fold cross-validation training for reliable model evaluation.

Usage:
    python scripts/train_cv.py --config configs/gnn_improved.yaml --model gnn
    python scripts/train_cv.py --config configs/tabular_improved.yaml --model tabular
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
from src.data.preprocessing import create_kfold_splits, load_excel_data, add_image_paths, normalize_features
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics, format_metrics
from scripts.train import build_model


def run_fold(fold_idx, train_df, val_df, model_type, config, device):
    """Train and evaluate a single fold."""
    tc = config.get("training", {})
    batch_size = tc.get("batch_size", 32)
    num_workers = config.get("experiment", {}).get("num_workers", 0)

    processed_dir = Path(config.get("data", {}).get("processed_dir", "data/processed"))

    # Save fold splits to temp CSVs
    fold_dir = processed_dir / "cv_folds" / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(fold_dir / "train.csv", index=False)
    val_df.to_csv(fold_dir / "val.csv", index=False)

    if model_type == "tabular":
        train_ds = ILTabularDataset(str(fold_dir / "train.csv"))
        val_ds = ILTabularDataset(str(fold_dir / "val.csv"))
        collate = None
    else:
        graph_cache = str(processed_dir / "graphs.pkl")
        graph_path = graph_cache if Path(graph_cache).exists() else None
        smiles_aug = config.get("training", {}).get("smiles_augment", False)
        train_ds = ILMultimodalDataset(
            str(fold_dir / "train.csv"), graph_path,
            is_train=True, config=config, smiles_augment=smiles_aug,
        )
        val_ds = ILMultimodalDataset(
            str(fold_dir / "val.csv"), graph_path,
            is_train=False, config=config,
        )
        collate = collate_multimodal

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate)

    # Override checkpoint dir per fold
    fold_config = config.copy()
    fold_ckpt = str(Path(config.get("experiment", {}).get("checkpoint_dir", "checkpoints")) / f"fold_{fold_idx}")
    fold_config.setdefault("experiment", {})["checkpoint_dir"] = fold_ckpt

    model = build_model(model_type, fold_config)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        config=fold_config,
        device=device,
    )

    history = trainer.train(verbose=False)

    # Evaluate on val (this fold's test set)
    val_loss, val_metrics, val_preds, val_targets = trainer.evaluate(val_loader)

    return val_metrics, val_preds, val_targets, len(history["train_loss"])


def main():
    parser = argparse.ArgumentParser(description="5-fold CV for IL models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="tabular",
                        choices=["tabular", "gnn", "vision", "multimodal"])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--save-results", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    device = get_device(config)
    print(f"Device: {device}")
    print(f"Running {args.n_folds}-fold CV for {args.model} model\n")

    # Load full normalized data
    processed_dir = Path(config.get("data", {}).get("processed_dir", "data/processed"))
    norm_csv = processed_dir / "il_data_normalized.csv"
    import pandas as pd
    df = pd.read_csv(norm_csv)

    # Create k-fold splits grouped by IL
    folds = create_kfold_splits(df, n_folds=args.n_folds)

    all_metrics = []
    for fold_idx, fold_data in enumerate(folds):
        train_df = fold_data["train"]
        val_df = fold_data["val"]
        n_train_ils = len(fold_data["train_ils"])
        n_val_ils = len(fold_data["val_ils"])

        print(f"Fold {fold_idx+1}/{args.n_folds}: "
              f"Train={len(train_df)} ({n_train_ils} ILs), "
              f"Val={len(val_df)} ({n_val_ils} ILs) — "
              f"Val ILs: {fold_data['val_ils']}")

        metrics, _, _, epochs = run_fold(fold_idx, train_df, val_df, args.model, config, device)
        all_metrics.append(metrics)
        print(f"  -> R²={metrics['avg_r2']:.4f}  MAE={metrics['avg_mae']:.4f}  "
              f"RMSE={metrics['avg_rmse']:.4f}  (epochs={epochs})")

    # Aggregate metrics
    print(f"\n{'='*60}")
    print(f"{args.n_folds}-Fold CV Results for {args.model.upper()}")
    print(f"{'='*60}")

    target_names = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]
    summary = {}

    for metric_type in ["mae", "rmse", "r2"]:
        for target in target_names + ["avg"]:
            key = f"{target}_{metric_type}"
            values = [m[key] for m in all_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            summary[f"{key}_mean"] = float(mean_val)
            summary[f"{key}_std"] = float(std_val)

    # Print per-target results
    print(f"\n{'Target':>8s}  {'MAE':>12s}  {'RMSE':>12s}  {'R²':>12s}")
    print("-" * 50)
    for target in target_names:
        mae = f"{summary[f'{target}_mae_mean']:.4f}±{summary[f'{target}_mae_std']:.4f}"
        rmse = f"{summary[f'{target}_rmse_mean']:.4f}±{summary[f'{target}_rmse_std']:.4f}"
        r2 = f"{summary[f'{target}_r2_mean']:.4f}±{summary[f'{target}_r2_std']:.4f}"
        print(f"{target:>8s}  {mae:>12s}  {rmse:>12s}  {r2:>12s}")
    print("-" * 50)
    avg_mae = f"{summary['avg_mae_mean']:.4f}±{summary['avg_mae_std']:.4f}"
    avg_rmse = f"{summary['avg_rmse_mean']:.4f}±{summary['avg_rmse_std']:.4f}"
    avg_r2 = f"{summary['avg_r2_mean']:.4f}±{summary['avg_r2_std']:.4f}"
    print(f"{'AVG':>8s}  {avg_mae:>12s}  {avg_rmse:>12s}  {avg_r2:>12s}")

    # Save
    if args.save_results:
        results = {
            "model": args.model,
            "n_folds": args.n_folds,
            "summary": summary,
            "per_fold": [{k: float(v) if isinstance(v, (float, np.floating)) else v
                          for k, v in m.items()} for m in all_metrics],
        }
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_results}")

    print("\nDone!")


if __name__ == "__main__":
    main()
