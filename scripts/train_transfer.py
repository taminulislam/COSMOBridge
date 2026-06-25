"""Phase 2: Transfer learning — pre-train on ILThermo, fine-tune on original dataset.

Pre-training stage:
  - 5,622 ILThermo samples (4,930 gamma1, 692 H_E)
  - Masked MSE loss (ignores NaN targets)
  - GNN learns general molecular graph representations

Fine-tuning stage:
  - 223 original samples with all 7 targets
  - Lower learning rate, shorter training
  - Evaluates on same test split as baseline

Usage:
    python scripts/train_transfer.py --config configs/default.yaml
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import TARGET_COLUMNS, FEATURE_COLUMNS, THERMO_FEATURES, SURFACE_FEATURES
from src.data.graph_builder import smiles_to_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.data.dataset import ILMultimodalDataset, collate_multimodal
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics, format_metrics


# ── Masked loss for partial targets ──────────────────────────────────────────

class MaskedMSELoss(nn.Module):
    """MSE loss that ignores NaN targets."""

    def __init__(self, target_names=None):
        super().__init__()
        self.target_names = target_names or TARGET_COLUMNS

    def forward(self, predictions, targets):
        mask = ~torch.isnan(targets)
        if mask.sum() == 0:
            return {"total": torch.tensor(0.0, device=predictions.device, requires_grad=True)}

        # Replace NaN with 0 BEFORE computing diff to avoid NaN gradients
        safe_targets = targets.clone()
        safe_targets[~mask] = 0.0

        diff2 = (predictions - safe_targets) ** 2
        # Zero out positions where target was NaN
        diff2 = diff2 * mask.float()

        # Per-task loss (mean over valid samples only)
        per_task = diff2.sum(dim=0) / mask.sum(dim=0).float().clamp(min=1)
        # Only average over tasks that have at least one valid sample
        active_tasks = mask.any(dim=0)
        total = per_task[active_tasks].mean()

        losses = {"total": total}
        for i, name in enumerate(self.target_names):
            losses[name] = per_task[i]
        return losses


# ── ILThermo Dataset ─────────────────────────────────────────────────────────

class ILThermoDataset(Dataset):
    """Dataset for ILThermo pre-training with partial targets."""

    def __init__(self, csv_path, surface_desc_path=None, feature_scaler=None,
                 target_scaler=None, fit_scalers=False):
        self.df = pd.read_csv(csv_path)

        # Engineer thermo features
        self.df["inv_temperature"] = 1.0 / self.df["temperature"]
        self.df["temp_squared"] = self.df["temperature"] ** 2
        self.df["temp_cubed"] = self.df["temperature"] ** 3

        # Merge surface descriptors if available
        if surface_desc_path and Path(surface_desc_path).exists():
            desc_df = pd.read_csv(surface_desc_path)
            if "smiles" in desc_df.columns:
                self.df = self.df.merge(desc_df.drop(columns=["il_short_name"], errors="ignore"),
                                        on="smiles", how="left", suffixes=("", "_dup"))
                self.df = self.df[[c for c in self.df.columns if not c.endswith("_dup")]]

        # Fill missing surface features with 0
        for col in SURFACE_FEATURES:
            if col not in self.df.columns:
                self.df[col] = 0.0
            else:
                self.df[col] = self.df[col].fillna(0.0)

        # Normalize features
        if fit_scalers:
            self.feature_scaler = StandardScaler()
            self.df[FEATURE_COLUMNS] = self.feature_scaler.fit_transform(self.df[FEATURE_COLUMNS])
            # Normalize available targets (keeping NaN as NaN)
            self.target_scaler = StandardScaler()
            # Fit on non-NaN values
            target_data = self.df[TARGET_COLUMNS].copy()
            # For fitting, fill NaN with 0 temporarily
            for col in TARGET_COLUMNS:
                valid = target_data[col].dropna()
                if len(valid) > 1:
                    mean, std = valid.mean(), valid.std()
                    if std > 0:
                        self.df[col] = (self.df[col] - mean) / std
        else:
            self.feature_scaler = feature_scaler
            self.target_scaler = target_scaler
            if feature_scaler:
                self.df[FEATURE_COLUMNS] = feature_scaler.transform(self.df[FEATURE_COLUMNS])

        # Build graphs (cache by SMILES)
        self._build_graphs()

    def _build_graphs(self):
        """Build molecular graphs for unique SMILES."""
        self.graphs = {}
        unique_smiles = self.df["smiles"].unique()
        failed = 0
        for smi in unique_smiles:
            try:
                self.graphs[smi] = smiles_to_graph(smi)
            except Exception:
                self.graphs[smi] = None
                failed += 1
        print(f"  Graphs: {len(unique_smiles) - failed}/{len(unique_smiles)} built")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Features
        features = torch.tensor(
            [row[col] for col in FEATURE_COLUMNS], dtype=torch.float32
        )

        # Graph
        smiles = row["smiles"]
        g = self.graphs.get(smiles)
        if g is not None:
            atom_features = torch.tensor(g["atom_features"], dtype=torch.float32)
            edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
            bond_features = torch.tensor(g["bond_features"], dtype=torch.float32)
        else:
            atom_features = torch.zeros(1, ATOM_FEATURE_DIM)
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            bond_features = torch.zeros(0, BOND_FEATURE_DIM)

        # Targets (NaN preserved — MaskedMSELoss handles them)
        targets = torch.tensor(
            [row[col] if pd.notna(row.get(col)) else float("nan") for col in TARGET_COLUMNS],
            dtype=torch.float32,
        )

        # Dummy image tensors (not used in GNN)
        dummy_img = torch.zeros(3, 224, 224)

        return {
            "features": features,
            "il_idx": torch.tensor(0, dtype=torch.long),
            "cation_idx": torch.tensor(0, dtype=torch.long),
            "anion_idx": torch.tensor(0, dtype=torch.long),
            "cosmo_image": dummy_img,
            "ep_image": dummy_img,
            "atom_features": atom_features,
            "edge_index": edge_index,
            "bond_features": bond_features,
            "num_atoms": atom_features.shape[0],
            "targets": targets,
            "il_name": smiles[:20],
        }


# ── Pre-training loop with masked loss ───────────────────────────────────────

def pretrain(model, train_loader, config, device, ckpt_dir):
    """Pre-train on ILThermo with masked loss."""
    tc = config.get("training", {})
    num_epochs = tc.get("pretrain_epochs", 100)
    lr = tc.get("pretrain_lr", 3e-4)
    patience = tc.get("pretrain_patience", 15)

    criterion = MaskedMSELoss().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - 5, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[5])

    best_loss = float("inf")
    no_improve = 0

    print(f"\n{'='*60}")
    print(f"PRE-TRAINING on ILThermo ({len(train_loader.dataset)} samples)")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        n = 0
        for batch in train_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            optimizer.zero_grad()

            # Forward through GNN
            import inspect
            sig = inspect.signature(model.forward)
            params = list(sig.parameters.keys())
            kwargs = {}
            key_aliases = {"batch": "graph_batch"}
            for p in params:
                if p == "kwargs":
                    continue
                if p in batch:
                    kwargs[p] = batch[p]
                elif p in key_aliases and key_aliases[p] in batch:
                    kwargs[p] = batch[key_aliases[p]]

            preds = model(**kwargs)
            targets = batch["targets"]

            losses = criterion(preds, targets)
            loss = losses["total"]

            if torch.isnan(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n += 1

        scheduler.step()
        avg_loss = total_loss / max(n, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "pretrained.pt")
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{num_epochs} | Loss: {avg_loss:.4f} | "
                  f"Best: {best_loss:.4f} | LR: {lr_now:.2e} | Patience: {no_improve}/{patience}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Load best
    model.load_state_dict(torch.load(ckpt_dir / "pretrained.pt", map_location=device, weights_only=True))
    print(f"  Pre-training complete. Best loss: {best_loss:.4f}")
    return model


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=200)
    parser.add_argument("--pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--finetune-lr", type=float, default=5e-5)
    parser.add_argument("--save-results", type=str, default="results/transfer_results.json")
    args = parser.parse_args()

    config = load_config(args.config)
    config["training"]["pretrain_epochs"] = args.pretrain_epochs
    config["training"]["pretrain_lr"] = args.pretrain_lr
    config["training"]["pretrain_patience"] = 15

    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    device = get_device(config)
    print(f"Device: {device}")

    # ── Build model ──
    from src.models.graph.gnn import MolecularGNN
    mc = config.get("model", {}).get("graph", {})
    aux_dim = len(FEATURE_COLUMNS)
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
        aux_feature_dim=aux_dim,
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    ckpt_dir = Path("checkpoints/transfer")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Pre-train on ILThermo ──
    base_dir = Path(".")
    ilthermo_csv = base_dir / "data" / "augmented" / "ilthermo_data.csv"
    surface_desc_ilthermo = base_dir / "data" / "pipeline" / "surface_descriptors_ilthermo.csv"

    print("\nLoading ILThermo dataset...")
    pretrain_ds = ILThermoDataset(
        str(ilthermo_csv),
        surface_desc_path=str(surface_desc_ilthermo),
        fit_scalers=True,
    )

    pretrain_loader = DataLoader(
        pretrain_ds, batch_size=64, shuffle=True,
        num_workers=0, collate_fn=collate_multimodal,
    )

    model = pretrain(model, pretrain_loader, config, device, ckpt_dir)

    # ── Stage 2: Fine-tune on original dataset ──
    print(f"\n{'='*60}")
    print(f"FINE-TUNING on original dataset (223 samples)")
    print(f"{'='*60}")

    # Override config for fine-tuning
    config["training"]["learning_rate"] = args.finetune_lr
    config["training"]["num_epochs"] = args.finetune_epochs
    config["training"]["early_stopping_patience"] = 25
    config["experiment"]["checkpoint_dir"] = str(ckpt_dir / "finetune")

    processed_dir = Path(config.get("data", {}).get("processed_dir", "data/processed"))
    splits_dir = processed_dir / "splits"
    graph_cache = str(processed_dir / "graphs.pkl")
    graph_path = graph_cache if Path(graph_cache).exists() else None

    train_ds = ILMultimodalDataset(str(splits_dir / "train.csv"), graph_path, is_train=True, config=config)
    val_ds = ILMultimodalDataset(str(splits_dir / "val.csv"), graph_path, is_train=False, config=config)
    test_ds = ILMultimodalDataset(str(splits_dir / "test.csv"), graph_path, is_train=False, config=config)

    batch_size = config.get("training", {}).get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_multimodal)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
    )

    history = trainer.train(verbose=True)

    # ── Save results ──
    results = {
        "model": "gnn_transfer",
        "n_params": n_params,
        "pretrain_samples": len(pretrain_ds),
        "finetune_samples": len(train_ds),
        "best_val_loss": trainer.best_val_loss,
        "epochs_trained": len(history["train_loss"]),
    }
    if "test_metrics" in history:
        results["test_metrics"] = {
            k: float(v) if not (isinstance(v, float) and np.isnan(v)) else None
            for k, v in history["test_metrics"].items()
        }

    Path(args.save_results).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_results, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.save_results}")

    # ── Comparison ──
    print(f"\n{'='*60}")
    print("COMPARISON WITH BASELINES")
    print(f"{'='*60}")
    for baseline_name, baseline_path in [
        ("Baseline GNN", "results/gnn_results.json"),
        ("GNN+Surface", "results/gnn_surface_results.json"),
    ]:
        try:
            with open(baseline_path) as f:
                base = json.load(f)
            bm = base.get("test_metrics", {})
            tm = results.get("test_metrics", {})
            r2_b = bm.get("avg_r2", 0)
            r2_t = tm.get("avg_r2", 0)
            mae_b = bm.get("avg_mae", 0)
            mae_t = tm.get("avg_mae", 0)
            print(f"\n  vs {baseline_name}:")
            print(f"    R²:  {r2_b:.4f} -> {r2_t:.4f}  ({r2_t - r2_b:+.4f})")
            print(f"    MAE: {mae_b:.4f} -> {mae_t:.4f}  ({mae_t - mae_b:+.4f})")
            for key in sorted(bm.keys()):
                if key.endswith("_r2"):
                    prop = key.replace("_r2", "")
                    b = bm[key]
                    t = tm.get(key, 0)
                    print(f"    {prop:15s} R²: {b:.4f} -> {t:.4f}  ({t - b:+.4f})")
        except Exception:
            pass


if __name__ == "__main__":
    main()
