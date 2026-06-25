"""Train Property-Conditioned Mixture of Experts model.

Single model that learns internal expert routing per property.
Uses merged dataset with Morgan FP + surface descriptors.
Includes snapshot ensemble at inference for variance reduction.
"""

import sys
import json
import copy
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.utils.config import load_config, get_device, set_seed
from src.models.fusion.moe import MixtureOfExpertsModel
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_joint import MergedDataset, collate_merged

TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


class MoEMaskedLoss(nn.Module):
    """Masked MSE + load balancing loss for MoE training.

    Also applies per-source weighting: original samples get higher weight
    for gamma1 to prevent ILThermo distribution domination.
    """

    def __init__(self, gamma1_original_weight=5.0):
        super().__init__()
        self.gamma1_weight = gamma1_original_weight

    def forward(self, predictions, targets, aux_losses=None, is_original=None):
        mask = ~torch.isnan(targets)
        if mask.sum() == 0:
            total = torch.tensor(0.0, device=predictions.device, requires_grad=True)
            return {"total": total}

        safe_targets = targets.clone()
        safe_targets[~mask] = 0.0

        diff2 = (predictions - safe_targets) ** 2 * mask.float()

        # Apply importance weighting for gamma1 (index 0) from original data
        if is_original is not None:
            gamma1_weight = torch.ones(predictions.shape[0], device=predictions.device)
            gamma1_weight[is_original] = self.gamma1_weight
            diff2[:, 0] = diff2[:, 0] * gamma1_weight

        per_task = diff2.sum(dim=0) / mask.sum(dim=0).float().clamp(min=1)
        active = mask.any(dim=0)
        mse_loss = per_task[active].mean()

        # Add load balance loss from MoE gating
        total = mse_loss
        if aux_losses and "load_balance_loss" in aux_losses:
            total = total + aux_losses["load_balance_loss"]

        losses = {"total": total, "mse": mse_loss}
        for i, name in enumerate(TARGET_COLUMNS):
            losses[name] = per_task[i]
        if aux_losses:
            losses["load_balance"] = aux_losses.get("load_balance_loss", torch.tensor(0.0))
        return losses


def train_moe(model, train_loader, val_loader, device, ckpt_dir,
              num_epochs=300, lr=1e-4, patience=30):
    """Train MoE with cosine restarts (for snapshot ensemble)."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    criterion = MoEMaskedLoss(gamma1_original_weight=5.0).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing with restarts — snapshots are taken at each restart
    T_0 = 60  # First cycle length
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=1e-6)

    best_loss = float("inf")
    no_improve = 0
    snapshots = []  # Save model states at cycle boundaries

    print(f"\n{'='*60}")
    print(f"TRAINING MIXTURE OF EXPERTS ({len(train_loader.dataset)} samples)")
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

            predictions, aux_losses = model(
                point_cloud=batch["point_cloud"],
                features=batch["features"],
                atom_features=batch["atom_features"],
                edge_index=batch["edge_index"],
                bond_features=batch["bond_features"],
                batch=batch["batch"],
            )

            losses = criterion(predictions, batch["targets"], aux_losses)
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

        # Validate
        model.eval()
        val_loss = 0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                preds, _ = model(
                    point_cloud=batch["point_cloud"], features=batch["features"],
                    atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                    bond_features=batch["bond_features"], batch=batch["batch"])
                mask = ~torch.isnan(batch["targets"])
                safe = batch["targets"].clone()
                safe[~mask] = 0.0
                diff = ((preds - safe) ** 2 * mask.float()).sum() / mask.float().sum().clamp(min=1)
                val_loss += diff.item()
                vn += 1

        avg_val = val_loss / max(vn, 1)

        if avg_val < best_loss:
            best_loss = avg_val
            no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
        else:
            no_improve += 1

        # Save snapshot at cycle boundaries
        if (epoch + 1) % T_0 == 0:
            snapshot_state = copy.deepcopy(model.state_dict())
            snapshots.append(snapshot_state)
            torch.save(snapshot_state, ckpt_dir / f"snapshot_{len(snapshots)}.pt")
            print(f"  ** Snapshot {len(snapshots)} saved at epoch {epoch}")

        if epoch % 20 == 0 or epoch == num_epochs - 1:
            lr_now = optimizer.param_groups[0]["lr"]
            lb = losses.get("load_balance", torch.tensor(0.0)).item()
            print(f"  Epoch {epoch:3d}/{num_epochs} | Train: {avg_loss:.4f} | "
                  f"Val: {avg_val:.4f} | Best: {best_loss:.4f} | "
                  f"LB: {lb:.4f} | LR: {lr_now:.2e} | Pat: {no_improve}/{patience}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Save all snapshots list
    torch.save({"n_snapshots": len(snapshots)}, ckpt_dir / "snapshot_info.pt")

    # Load best single model
    model.load_state_dict(torch.load(ckpt_dir / "best_model.pt",
                                      map_location=device, weights_only=True))
    print(f"  Training complete. Best loss: {best_loss:.4f}, Snapshots: {len(snapshots)}")
    return model, snapshots


def evaluate_single(model, loader, device):
    """Evaluate single model."""
    model.eval()
    all_preds, all_targets = [], []
    all_gate_weights = []

    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            preds, aux = model(
                point_cloud=batch["point_cloud"], features=batch["features"],
                atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                bond_features=batch["bond_features"], batch=batch["batch"])
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())
            all_gate_weights.append(aux["gate_weights"].cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    gate_weights = np.concatenate(all_gate_weights)
    return preds, targets, gate_weights


def evaluate_snapshot_ensemble(model, snapshots, loader, device, ckpt_dir):
    """Average predictions from multiple snapshots."""
    all_snapshot_preds = []

    # Best model predictions
    preds, targets, _ = evaluate_single(model, loader, device)
    all_snapshot_preds.append(preds)

    # Snapshot predictions
    for i, state in enumerate(snapshots):
        model.load_state_dict(state)
        model.to(device).eval()
        sp, _, _ = evaluate_single(model, loader, device)
        all_snapshot_preds.append(sp)

    # Average
    ensemble_preds = np.mean(all_snapshot_preds, axis=0)
    return ensemble_preds, targets


def visualize_gating(gate_weights, output_path):
    """Visualize expert-property gating weights."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Average gating weights: (P, K)
    mean_weights = gate_weights.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mean_weights, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.6)

    expert_labels = ['Expert 1\n(surface)', 'Expert 2\n(mixing)',
                     'Expert 3\n(thermo)', 'Expert 4\n(generalist)']
    prop_labels = [r'$\gamma_1$', r'$\gamma_2$', r'$G^E$', r'$H^E$',
                   r'$G_{mix}$', r'$H_{vap}$', r'$P$']

    ax.set_xticks(range(len(expert_labels)))
    ax.set_xticklabels(expert_labels[:mean_weights.shape[1]], fontsize=10)
    ax.set_yticks(range(len(prop_labels)))
    ax.set_yticklabels(prop_labels, fontsize=11)

    for i in range(mean_weights.shape[0]):
        for j in range(mean_weights.shape[1]):
            color = 'white' if mean_weights[i, j] > 0.35 else 'black'
            ax.text(j, i, f'{mean_weights[i, j]:.2f}', ha='center', va='center',
                   fontsize=10, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label='Gating Weight')
    ax.set_title('Property-Expert Routing Weights (MoE Gating)', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Gating visualization saved: {output_path}")


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    # ── Load merged dataset ──
    merged_dir = Path("data/merged")
    if not merged_dir.exists():
        print("ERROR: Run create_merged_dataset.py first")
        return

    meta = json.load(open(merged_dir / "metadata.json"))
    feature_columns = meta["feature_columns"]
    n_features = len(feature_columns)
    print(f"Features: {n_features}")

    pc_dir = "data/pipeline/point_clouds"
    splits = merged_dir / "splits"

    print("\nLoading datasets...")
    train_ds = MergedDataset(str(splits / "train.csv"), pc_dir, feature_columns, is_train=True)
    val_ds = MergedDataset(str(splits / "val.csv"), pc_dir, feature_columns, is_train=False)
    test_ds = MergedDataset(str(splits / "test.csv"), pc_dir, feature_columns, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_merged)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)

    # ── Build MoE model ──
    model = MixtureOfExpertsModel(
        feature_dim=n_features,
        num_experts=4,
        num_targets=7,
        fused_dim=256,
        dropout=0.3,
        pretrained_gnn_path="checkpoints/transfer/pretrained.pt",
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MoE Model params: {n_params:,}")

    # ── Train ──
    model, snapshots = train_moe(
        model, train_loader, val_loader, device,
        ckpt_dir="checkpoints/moe",
        num_epochs=300, lr=1e-4, patience=30)

    # ── Evaluate: Single best model ──
    print(f"\n{'='*60}")
    print("SINGLE MODEL EVALUATION")
    print(f"{'='*60}")

    preds, targets, gate_weights = evaluate_single(model, test_loader, device)
    single_metrics = compute_metrics(preds, targets)
    print(format_metrics(single_metrics, "MoE (single)"))

    # ── Evaluate: Snapshot ensemble ──
    if len(snapshots) >= 2:
        print(f"\n{'='*60}")
        print(f"SNAPSHOT ENSEMBLE ({len(snapshots)+1} models)")
        print(f"{'='*60}")

        # Reload best model first
        model.load_state_dict(torch.load("checkpoints/moe/best_model.pt",
                                          map_location=device, weights_only=True))
        ens_preds, ens_targets = evaluate_snapshot_ensemble(
            model, snapshots, test_loader, device, "checkpoints/moe")
        ens_metrics = compute_metrics(ens_preds, ens_targets)
        print(format_metrics(ens_metrics, "MoE (snapshot ensemble)"))
    else:
        ens_metrics = single_metrics

    # ── Visualize gating ──
    print("\nVisualizing expert-property routing...")
    # Reload best for visualization
    model.load_state_dict(torch.load("checkpoints/moe/best_model.pt",
                                      map_location=device, weights_only=True))
    model.to(device)
    _, _, gate_weights = evaluate_single(model, test_loader, device)
    visualize_gating(gate_weights, "paper/figures/moe_gating_weights.png")

    # ── Comparison with all previous models ──
    print(f"\n{'='*60}")
    print("COMPARISON WITH ALL MODELS")
    print(f"{'='*60}")

    best_metric = ens_metrics if ens_metrics["avg_r2"] > single_metrics["avg_r2"] else single_metrics
    best_label = "snapshot" if ens_metrics["avg_r2"] > single_metrics["avg_r2"] else "single"

    baselines = [
        ("Baseline GNN", "results/gnn_results.json"),
        ("Phase 3 PointCloud", "results/pointcloud_results.json"),
        ("Hard Ensemble (P2+P3)", "results/ensemble_phase23_results.json"),
    ]

    print(f"\n  MoE ({best_label}): avg R² = {best_metric['avg_r2']:.4f}")
    for name, path in baselines:
        try:
            with open(path) as f:
                data = json.load(f)
            bm = data.get("hard_ensemble", data.get("test_metrics", {}))
            r2 = bm.get("avg_r2", "N/A")
            print(f"  {name:30s}: avg R² = {r2:.4f}" if isinstance(r2, float) else f"  {name}: {r2}")
        except Exception:
            pass

    # Per-property
    print(f"\n  {'Property':15s} {'MoE single':>12s} {'MoE snapshot':>12s} {'Hard Ens':>12s}")
    try:
        with open("results/ensemble_phase23_results.json") as f:
            ens_data = json.load(f).get("hard_ensemble", {})
    except Exception:
        ens_data = {}

    for prop in TARGET_COLUMNS:
        key = f"{prop}_r2"
        s = single_metrics.get(key, float("nan"))
        e = ens_metrics.get(key, float("nan"))
        h = ens_data.get(key, float("nan"))
        print(f"  {prop:15s} {s:12.4f} {e:12.4f} {h:12.4f}")

    # ── Save results ──
    results = {
        "model": "mixture_of_experts",
        "n_params": n_params,
        "num_experts": 4,
        "n_snapshots": len(snapshots),
        "single_model": {k: float(v) if isinstance(v, (float, np.floating)) else v
                         for k, v in single_metrics.items()},
        "snapshot_ensemble": {k: float(v) if isinstance(v, (float, np.floating)) else v
                              for k, v in ens_metrics.items()},
        "mean_gating_weights": gate_weights.mean(axis=0).tolist(),
    }
    with open("results/moe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/moe_results.json")


if __name__ == "__main__":
    main()
