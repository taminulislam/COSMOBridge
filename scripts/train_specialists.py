"""Per-property specialist models.

Trains two separate models optimized for different property groups:
  - Structure specialist: PointCloud+GNN for gamma1, gamma2, G_E, H_E, G_mix (5 targets)
  - Temperature specialist: GNN+thermo features for H_vap, P (2 targets)

Combines predictions at inference for the full 7-target evaluation.
"""

import sys
import json
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import TARGET_COLUMNS, FEATURE_COLUMNS
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.multimodal_pointcloud import MultimodalPointCloudModel
from src.data.dataset import ILMultimodalDataset, collate_multimodal
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud

STRUCTURE_TARGETS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix"]  # indices 0-4
TEMP_TARGETS = ["H_vap", "P"]  # indices 5-6


class TargetSubsetWrapper(nn.Module):
    """Wraps a model to only predict a subset of targets."""

    def __init__(self, base_model, target_indices):
        super().__init__()
        self.base_model = base_model
        self.target_indices = target_indices

    def forward(self, **kwargs):
        full_pred = self.base_model(**kwargs)
        return full_pred  # Return full predictions; loss masking handles the rest


class SubsetLoss(nn.Module):
    """MSE loss on only a subset of target indices."""

    def __init__(self, target_indices, target_names=None):
        super().__init__()
        self.target_indices = target_indices
        self.target_names = target_names or TARGET_COLUMNS

    def forward(self, predictions, targets):
        idx = self.target_indices
        pred_sub = predictions[:, idx]
        tgt_sub = targets[:, idx]
        per_task = torch.mean((pred_sub - tgt_sub) ** 2, dim=0)
        total = per_task.mean()

        losses = {"total": total}
        for i, ti in enumerate(idx):
            losses[self.target_names[ti]] = per_task[i]
        return losses


def train_specialist(model, train_loader, val_loader, config, device,
                     target_indices, ckpt_dir, name):
    """Train a specialist model on a subset of targets."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    tc = config.get("training", {})
    num_epochs = tc.get("num_epochs", 200)
    lr = tc.get("learning_rate", 1e-4)
    patience = tc.get("early_stopping_patience", 25)

    criterion = SubsetLoss(target_indices).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=10)
    cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - 10, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[10])

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    no_improve = 0

    print(f"\n{'='*60}")
    print(f"Training {name} specialist (targets: {[TARGET_COLUMNS[i] for i in target_indices]})")
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

            # Forward — build kwargs dynamically
            import inspect
            sig = inspect.signature(model.forward)
            params = list(sig.parameters.keys())
            key_aliases = {"batch": "graph_batch"}
            kwargs = {}
            for p in params:
                if p == "kwargs":
                    continue
                if p in batch:
                    kwargs[p] = batch[p]
                elif p in key_aliases and key_aliases[p] in batch:
                    kwargs[p] = batch[key_aliases[p]]

            preds = model(**kwargs)
            losses = criterion(preds, batch["targets"])
            loss = losses["total"]

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        scheduler.step()
        avg_loss = total_loss / max(n, 1)

        # Validation
        model.eval()
        val_loss = 0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                sig = inspect.signature(model.forward)
                params = list(sig.parameters.keys())
                kwargs = {}
                for p in params:
                    if p == "kwargs":
                        continue
                    if p in batch:
                        kwargs[p] = batch[p]
                    elif p in key_aliases and key_aliases[p] in batch:
                        kwargs[p] = batch[key_aliases[p]]
                preds = model(**kwargs)
                losses = criterion(preds, batch["targets"])
                val_loss += losses["total"].item()
                vn += 1

        avg_val = val_loss / max(vn, 1)

        if avg_val < best_loss:
            best_loss = avg_val
            no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{num_epochs} | Train: {avg_loss:.4f} | "
                  f"Val: {avg_val:.4f} | Best: {best_loss:.4f} | Patience: {no_improve}/{patience}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=True))
    print(f"  {name} specialist training complete. Best val loss: {best_loss:.4f}")
    return model


def main():
    config = load_config("configs/default.yaml")
    config["training"]["num_epochs"] = 200
    config["training"]["learning_rate"] = 1e-4
    config["training"]["early_stopping_patience"] = 25

    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    # ── Structure Specialist: PointCloud + GNN ──
    config_struct = {**config}
    config_struct.setdefault("model", {})["temp_skip"] = False
    struct_model = MultimodalPointCloudModel(
        config=config_struct,
        pretrained_gnn_path="checkpoints/transfer/pretrained.pt",
    )
    struct_model.to(device)
    print(f"Structure model params: {sum(p.numel() for p in struct_model.parameters() if p.requires_grad):,}")

    splits_dir = Path("data/processed/splits")
    pc_dir = "data/pipeline/point_clouds"

    train_ds_pc = PointCloudMultimodalDataset(str(splits_dir / "train.csv"), pc_dir, is_train=True)
    val_ds_pc = PointCloudMultimodalDataset(str(splits_dir / "val.csv"), pc_dir, is_train=False)
    test_ds_pc = PointCloudMultimodalDataset(str(splits_dir / "test.csv"), pc_dir, is_train=False)

    train_loader_pc = DataLoader(train_ds_pc, batch_size=32, shuffle=True, collate_fn=collate_pointcloud)
    val_loader_pc = DataLoader(val_ds_pc, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)
    test_loader_pc = DataLoader(test_ds_pc, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)

    struct_model = train_specialist(
        struct_model, train_loader_pc, val_loader_pc, config, device,
        target_indices=[0, 1, 2, 3, 4],
        ckpt_dir="checkpoints/specialist_structure",
        name="Structure",
    )

    # ── Temperature Specialist: GNN + thermo features ──
    temp_model = MolecularGNN(
        atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
        hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
        dropout=0.3, pooling="mean", num_targets=7,
        aux_feature_dim=len(FEATURE_COLUMNS),
    )
    # Load Phase 2 pre-trained weights
    pretrained = Path("checkpoints/transfer/pretrained.pt")
    if pretrained.exists():
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=True)
        temp_model.load_state_dict(ckpt, strict=False)
        print("  Loaded Phase 2 pre-trained GNN for temperature specialist")
    temp_model.to(device)
    print(f"Temperature model params: {sum(p.numel() for p in temp_model.parameters() if p.requires_grad):,}")

    graph_cache = "data/processed/graphs.pkl"
    graph_path = graph_cache if Path(graph_cache).exists() else None

    train_ds_gnn = ILMultimodalDataset(str(splits_dir / "train.csv"), graph_path, is_train=True, config=config)
    val_ds_gnn = ILMultimodalDataset(str(splits_dir / "val.csv"), graph_path, is_train=False, config=config)
    test_ds_gnn = ILMultimodalDataset(str(splits_dir / "test.csv"), graph_path, is_train=False, config=config)

    train_loader_gnn = DataLoader(train_ds_gnn, batch_size=32, shuffle=True, collate_fn=collate_multimodal)
    val_loader_gnn = DataLoader(val_ds_gnn, batch_size=32, shuffle=False, collate_fn=collate_multimodal)
    test_loader_gnn = DataLoader(test_ds_gnn, batch_size=32, shuffle=False, collate_fn=collate_multimodal)

    temp_model = train_specialist(
        temp_model, train_loader_gnn, val_loader_gnn, config, device,
        target_indices=[5, 6],
        ckpt_dir="checkpoints/specialist_temp",
        name="Temperature",
    )

    # ── Combine predictions on test set ──
    print(f"\n{'='*60}")
    print("COMBINED SPECIALIST EVALUATION")
    print(f"{'='*60}")

    struct_model.eval()
    temp_model.eval()

    all_preds = []
    all_targets = []

    # Structure specialist predictions
    struct_preds = []
    with torch.no_grad():
        for batch in test_loader_pc:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            preds = struct_model(
                point_cloud=batch["point_cloud"], features=batch["features"],
                atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                bond_features=batch["bond_features"], batch=batch["batch"],
            )
            struct_preds.append(preds.cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())
    struct_preds = np.concatenate(struct_preds)
    all_targets = np.concatenate(all_targets)

    # Temperature specialist predictions
    temp_preds = []
    with torch.no_grad():
        for batch in test_loader_gnn:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            preds = temp_model(
                atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                bond_features=batch["bond_features"], batch=batch["graph_batch"],
                features=batch["features"],
            )
            temp_preds.append(preds.cpu().numpy())
    temp_preds = np.concatenate(temp_preds)

    # Combine: structure for 0-4, temperature for 5-6
    combined = struct_preds.copy()
    combined[:, 5:7] = temp_preds[:, 5:7]

    metrics = compute_metrics(combined, all_targets)
    print(format_metrics(metrics, "Combined Specialists"))

    results = {"model": "specialist_combined",
               "best_struct_targets": STRUCTURE_TARGETS,
               "best_temp_targets": TEMP_TARGETS}
    results["test_metrics"] = {k: float(v) if not (isinstance(v, float) and np.isnan(v)) else None
                               for k, v in metrics.items()}

    with open("results/specialist_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/specialist_results.json")


if __name__ == "__main__":
    main()
