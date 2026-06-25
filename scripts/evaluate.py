"""Evaluate trained models and generate comprehensive results.

Usage:
    python scripts/evaluate.py --model tabular --checkpoint checkpoints/best_model.pt
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd

from src.utils.config import load_config, get_device, set_seed
from src.training.metrics import compute_metrics, format_metrics
from scripts.train import build_model, build_dataloaders


def main():
    parser = argparse.ArgumentParser(description="Evaluate IL property prediction model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="tabular",
                        choices=["tabular", "gnn", "vision", "multimodal"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config)
    set_seed(config.get("experiment", {}).get("seed", 42))

    # Build model and load checkpoint
    model = build_model(args.model, config)
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}, using random weights")

    model.to(device)
    model.eval()

    # Build data
    _, val_loader, test_loader = build_dataloaders(args.model, config)
    loader = test_loader if args.split == "test" else val_loader

    # Evaluate
    from src.training.losses import MultiTaskMSELoss
    criterion = MultiTaskMSELoss().to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            import inspect
            sig = inspect.signature(model.forward)
            params = list(sig.parameters.keys())

            if "cosmo_image" in params:
                pred = model(**{k: batch[k] for k in params if k in batch})
            else:
                pred = model(
                    features=batch["features"],
                    il_idx=batch["il_idx"],
                    cation_idx=batch["cation_idx"],
                    anion_idx=batch["anion_idx"],
                )

            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    metrics = compute_metrics(predictions, targets)
    print(f"\n{args.split.upper()} Results ({args.model} model):")
    print(format_metrics(metrics, args.split.upper()))

    # Save predictions
    if args.output:
        target_names = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]
        results_df = pd.DataFrame(predictions, columns=[f"pred_{n}" for n in target_names])
        for i, name in enumerate(target_names):
            results_df[f"true_{name}"] = targets[:, i]
        results_df.to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")


if __name__ == "__main__":
    main()
