"""Training loop for ionic liquid property prediction models."""

import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.training.losses import MultiTaskMSELoss, UncertaintyWeightedLoss, PhysicsInformedLoss
from src.training.metrics import compute_metrics, format_metrics


def build_criterion(config: dict):
    """Build loss function from config."""
    tc = config.get("training", {})
    loss_type = tc.get("loss_type", "mse")
    task_weights = tc.get("task_weights", {})

    if loss_type == "uncertainty":
        return UncertaintyWeightedLoss()
    elif loss_type == "physics":
        return PhysicsInformedLoss(
            task_weights=task_weights,
            use_uncertainty=tc.get("physics_use_uncertainty", True),
            physics_weight=tc.get("physics_weight", 0.1),
        )
    else:
        return MultiTaskMSELoss(task_weights=task_weights)


class Trainer:
    """Handles training, validation, and evaluation for all model types."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device = None,
        test_loader: DataLoader = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device or torch.device("cpu")

        self.model.to(self.device)

        # Training config
        tc = config.get("training", {})
        self.num_epochs = tc.get("num_epochs", 200)
        self.gradient_clip = tc.get("gradient_clip", 1.0)
        self.patience = tc.get("early_stopping_patience", 20)

        # Loss (configurable)
        self.criterion = build_criterion(config).to(self.device)

        # Optimizer — include loss params if learnable (uncertainty weights)
        lr = tc.get("learning_rate", 1e-4)
        wd = tc.get("weight_decay", 1e-4)
        params = list(model.parameters()) + list(self.criterion.parameters())
        self.optimizer = AdamW(params, lr=lr, weight_decay=wd)

        # Scheduler
        warmup_epochs = tc.get("warmup_epochs", 10)
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs - warmup_epochs, eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        # Tracking
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.history = {"train_loss": [], "val_loss": [], "val_metrics": []}

        # Checkpoint dir
        ckpt_dir = config.get("experiment", {}).get("checkpoint_dir", "checkpoints")
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _forward_batch(self, batch: dict) -> torch.Tensor:
        """Run forward pass. Handles both tabular-only and multimodal batches."""
        # Move tensors to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        # Check what the model expects
        import inspect
        sig = inspect.signature(self.model.forward)
        params = list(sig.parameters.keys())

        # Map collated keys to model parameter names
        # The collate function uses 'graph_batch' but GNN expects 'batch'
        key_aliases = {"batch": "graph_batch"}

        if "cosmo_image" in params or ("atom_features" in params and "cosmo_image" not in params):
            # GNN or Multimodal model — pass all matching params
            kwargs = {}
            for p in params:
                if p == "kwargs":
                    continue
                if p in batch:
                    kwargs[p] = batch[p]
                elif p in key_aliases and key_aliases[p] in batch:
                    kwargs[p] = batch[key_aliases[p]]
            return self.model(**kwargs)
        elif "features" in params and "il_idx" in params and "atom_features" not in params:
            # Tabular model
            return self.model(
                features=batch["features"],
                il_idx=batch["il_idx"],
                cation_idx=batch["cation_idx"],
                anion_idx=batch["anion_idx"],
            )
        else:
            # Generic: try passing what we have
            return self.model(**batch)

    def train_epoch(self) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            predictions = self._forward_batch(batch)
            targets = batch["targets"].to(self.device)

            losses = self.criterion(predictions, targets)
            loss = losses["total"]

            loss.backward()

            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple:
        """Evaluate model on a data loader.

        Returns: (avg_loss, metrics_dict, all_predictions, all_targets)
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []

        for batch in loader:
            predictions = self._forward_batch(batch)
            targets = batch["targets"].to(self.device)

            losses = self.criterion(predictions, targets)
            total_loss += losses["total"].item()
            n_batches += 1

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        metrics = compute_metrics(all_preds, all_targets)

        return avg_loss, metrics, all_preds, all_targets

    def train(self, verbose: bool = True) -> dict:
        """Full training loop with early stopping.

        Returns training history.
        """
        if verbose:
            print(f"Training on {self.device} for up to {self.num_epochs} epochs")
            print(f"Train: {len(self.train_loader.dataset)} samples, "
                  f"Val: {len(self.val_loader.dataset)} samples")
            print(f"Loss: {self.criterion.__class__.__name__}")

        start_time = time.time()

        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)

            # Validate
            val_loss, val_metrics, _, _ = self.evaluate(self.val_loader)
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(val_metrics)

            # Scheduler step
            self.scheduler.step()

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint("best_model.pt")
            else:
                self.epochs_without_improvement += 1

            # Logging
            if verbose and (epoch % 10 == 0 or epoch == self.num_epochs - 1):
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch:3d}/{self.num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val R2: {val_metrics['avg_r2']:.4f} | "
                      f"LR: {lr:.2e} | "
                      f"Patience: {self.epochs_without_improvement}/{self.patience}")

            if self.epochs_without_improvement >= self.patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                break

        elapsed = time.time() - start_time
        if verbose:
            print(f"\nTraining complete in {elapsed:.1f}s")
            print(f"Best val loss: {self.best_val_loss:.4f}")

        # Load best model and evaluate on test set
        self._load_checkpoint("best_model.pt")

        if self.test_loader:
            test_loss, test_metrics, _, _ = self.evaluate(self.test_loader)
            if verbose:
                print(f"\nTest Results:")
                print(format_metrics(test_metrics, "Test"))
            self.history["test_metrics"] = test_metrics

        return self.history

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.ckpt_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "criterion_state_dict": self.criterion.state_dict(),
            "best_val_loss": self.best_val_loss,
        }, path)

    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.ckpt_dir / filename
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "criterion_state_dict" in checkpoint:
                self.criterion.load_state_dict(checkpoint["criterion_state_dict"])
