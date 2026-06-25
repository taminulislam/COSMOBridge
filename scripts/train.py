"""Main training entry point for IL property prediction models.

Usage:
    python scripts/train.py --config configs/default.yaml --model tabular
    python scripts/train.py --config configs/default.yaml --model multimodal
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device, set_seed
from src.data.dataset import ILTabularDataset, ILMultimodalDataset, collate_multimodal
from src.training.trainer import Trainer


def build_model(model_type: str, config: dict):
    """Build model based on type string."""
    if model_type == "tabular":
        from src.models.tabular.dnn import TabularDNN
        from src.data.preprocessing import FEATURE_COLUMNS
        mc = config.get("model", {}).get("tabular", {})
        return TabularDNN(
            num_ils=28,
            num_cations=9,
            num_anions=7,
            feature_dim=len(FEATURE_COLUMNS),
            il_embed_dim=mc.get("il_embed_dim", 64),
            cation_embed_dim=mc.get("cation_embed_dim", 32),
            anion_embed_dim=mc.get("anion_embed_dim", 32),
            hidden_dims=mc.get("hidden_dims", [128, 64, 32]),
            dropout=mc.get("dropout", 0.4),
            num_targets=config.get("model", {}).get("prediction", {}).get("num_targets", 7),
        )
    elif model_type == "gnn":
        from src.models.graph.gnn import MolecularGNN
        from src.data.preprocessing import FEATURE_COLUMNS
        mc = config.get("model", {}).get("graph", {})
        aux_dim = len(FEATURE_COLUMNS) if mc.get("use_aux_features", True) else 0
        return MolecularGNN(
            atom_feature_dim=22,
            bond_feature_dim=7,
            hidden_dim=mc.get("hidden_dim", 256),
            num_layers=mc.get("num_layers", 4),
            conv_type=mc.get("conv_type", "GAT"),
            heads=mc.get("heads", 4),
            dropout=mc.get("dropout", 0.3),
            pooling=mc.get("pooling", "mean"),
            num_targets=config.get("model", {}).get("prediction", {}).get("num_targets", 7),
            aux_feature_dim=aux_dim,
        )
    elif model_type == "vision":
        from src.models.vision.vit import DualImageEncoder
        mc = config.get("model", {}).get("vision", {})
        return DualImageEncoder(
            backbone=mc.get("backbone", "resnet34"),
            pretrained=mc.get("pretrained", True),
            dropout=mc.get("dropout", 0.3),
            num_targets=config.get("model", {}).get("prediction", {}).get("num_targets", 7),
        )
    elif model_type == "multimodal":
        from src.models.fusion.multimodal import MultimodalILModel
        return MultimodalILModel(config=config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def build_dataloaders(model_type: str, config: dict):
    """Build train/val/test DataLoaders."""
    tc = config.get("training", {})
    batch_size = tc.get("batch_size", 32)
    num_workers = config.get("experiment", {}).get("num_workers", 0)

    processed_dir = Path(config.get("data", {}).get("processed_dir", "data/processed"))
    splits_dir = processed_dir / "splits"

    if model_type == "tabular":
        train_ds = ILTabularDataset(str(splits_dir / "train.csv"))
        val_ds = ILTabularDataset(str(splits_dir / "val.csv"))
        test_ds = ILTabularDataset(str(splits_dir / "test.csv"))
        collate = None
    else:
        graph_cache = str(processed_dir / "graphs.pkl")
        graph_path = graph_cache if Path(graph_cache).exists() else None
        smiles_aug = tc.get("smiles_augment", False)
        train_ds = ILMultimodalDataset(str(splits_dir / "train.csv"), graph_path, is_train=True, config=config, smiles_augment=smiles_aug)
        val_ds = ILMultimodalDataset(str(splits_dir / "val.csv"), graph_path, is_train=False, config=config)
        test_ds = ILMultimodalDataset(str(splits_dir / "test.csv"), graph_path, is_train=False, config=config)
        collate = collate_multimodal

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Train IL property prediction model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="tabular",
                        choices=["tabular", "gnn", "vision", "multimodal"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-results", type=str, default=None,
                        help="Path to save results JSON")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Override checkpoint directory")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.checkpoint_dir:
        config["experiment"]["checkpoint_dir"] = args.checkpoint_dir

    # Setup
    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    device = get_device(config)
    print(f"Device: {device}")

    # Build model and data
    print(f"\nBuilding {args.model} model...")
    model = build_model(args.model, config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    print("Building data loaders...")
    train_loader, val_loader, test_loader = build_dataloaders(args.model, config)

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
    )

    history = trainer.train(verbose=True)

    # Save results
    if args.save_results:
        import json
        results = {
            "model": args.model,
            "n_params": n_params,
            "best_val_loss": trainer.best_val_loss,
            "train_loss_final": history["train_loss"][-1] if history["train_loss"] else None,
            "val_loss_final": history["val_loss"][-1] if history["val_loss"] else None,
            "epochs_trained": len(history["train_loss"]),
        }
        if "test_metrics" in history:
            results["test_metrics"] = {
                k: float(v) if not (isinstance(v, float) and np.isnan(v)) else None
                for k, v in history["test_metrics"].items()
            }
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_results}")

    print("\nDone!")


if __name__ == "__main__":
    main()
