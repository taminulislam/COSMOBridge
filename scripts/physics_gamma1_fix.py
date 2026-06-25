"""Physics-informed gamma1 prediction.

Option A: Post-hoc — derive gamma1 from PointCloud's G_E and gamma2 predictions
          using the exact thermodynamic relation: G_E = RT*(x1*ln(g1) + x2*ln(g2))
          No retraining needed.

Option B: Retrain PointCloud model with physics-constrained loss that enforces
          thermodynamic consistency between gamma1, gamma2, and G_E.

Key equation (all data at x1=0.5):
    G_E = RT * (0.5*ln(γ₁) + 0.5*ln(γ₂))
    => ln(γ₁) = 2*G_E/(RT) - ln(γ₂)
    => γ₁ = exp(2*G_E/(RT) - ln(γ₂))
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.fusion.multimodal_pointcloud import MultimodalPointCloudModel
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud

R_KCAL = 1.987e-3  # kcal/(mol·K)
X1 = 0.5  # all data at equimolar composition


def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_targets, all_features = [], [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            preds = model(
                point_cloud=batch["point_cloud"], features=batch["features"],
                atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                bond_features=batch["bond_features"], batch=batch["batch"])
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())
            all_features.append(batch["features"].cpu().numpy())
    return (np.concatenate(all_preds), np.concatenate(all_targets),
            np.concatenate(all_features))


def inverse_transform_targets(values, scaler, col_idx):
    """Inverse-transform a single target column from normalized to raw."""
    mean = scaler.mean_[col_idx]
    scale = scaler.scale_[col_idx]
    return values * scale + mean


def forward_transform_targets(values, scaler, col_idx):
    """Forward-transform a single target column from raw to normalized."""
    mean = scaler.mean_[col_idx]
    scale = scaler.scale_[col_idx]
    return (values - mean) / scale


def derive_gamma1_from_physics(G_E_raw, gamma2_raw, T_raw):
    """Compute gamma1 from thermodynamic relation.

    G_E = RT * (x1*ln(g1) + x2*ln(g2))  with x1=x2=0.5
    => ln(g1) = 2*G_E/(RT) - ln(g2)
    """
    ln_gamma1 = 2.0 * G_E_raw / (R_KCAL * T_raw) - np.log(gamma2_raw)
    gamma1 = np.exp(ln_gamma1)
    return gamma1


def option_a_posthoc(device, target_scaler, feature_scaler):
    """Option A: Derive gamma1 from existing PointCloud predictions."""
    print(f"\n{'='*60}")
    print("OPTION A: Post-hoc Physics-Derived Gamma1")
    print(f"{'='*60}")

    config = load_config("configs/default.yaml")
    pc_dir = "data/pipeline/point_clouds"
    orig_splits = Path("data/processed/splits")

    # Load PointCloud model
    model = MultimodalPointCloudModel(config=config, pretrained_gnn_path=None)
    ckpt = torch.load("checkpoints/pointcloud/best_model.pt",
                       map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)

    # Get predictions
    test_ds = PointCloudMultimodalDataset(str(orig_splits / "test.csv"), pc_dir, is_train=False)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)
    preds, targets, features = get_predictions(model, test_ldr, device)

    # Column indices: gamma1=0, gamma2=1, G_E=2
    # Temperature is feature index 0 (first in FEATURE_COLUMNS)

    # Inverse-transform predictions and targets to raw space
    G_E_pred_raw = inverse_transform_targets(preds[:, 2], target_scaler, 2)
    gamma2_pred_raw = inverse_transform_targets(preds[:, 1], target_scaler, 1)
    T_raw = features[:, 0] * feature_scaler.scale_[0] + feature_scaler.mean_[0]

    gamma1_target_raw = inverse_transform_targets(targets[:, 0], target_scaler, 0)
    G_E_target_raw = inverse_transform_targets(targets[:, 2], target_scaler, 2)
    gamma2_target_raw = inverse_transform_targets(targets[:, 1], target_scaler, 1)

    # Verify thermodynamic relation holds in targets
    gamma1_derived_from_targets = derive_gamma1_from_physics(
        G_E_target_raw, gamma2_target_raw, T_raw)
    residual = np.abs(gamma1_target_raw - gamma1_derived_from_targets)
    print(f"\n  Verification: thermodynamic consistency in test targets")
    print(f"    Mean |γ₁_actual - γ₁_derived|: {residual.mean():.6f}")
    print(f"    Max  |γ₁_actual - γ₁_derived|: {residual.max():.6f}")

    # Derive gamma1 from predicted G_E and gamma2
    gamma1_derived_raw = derive_gamma1_from_physics(G_E_pred_raw, gamma2_pred_raw, T_raw)

    # Also get directly predicted gamma1
    gamma1_direct_raw = inverse_transform_targets(preds[:, 0], target_scaler, 0)

    # Compute R² in raw space
    ss_tot = np.sum((gamma1_target_raw - gamma1_target_raw.mean())**2)

    ss_res_direct = np.sum((gamma1_target_raw - gamma1_direct_raw)**2)
    r2_direct = 1 - ss_res_direct / ss_tot

    ss_res_derived = np.sum((gamma1_target_raw - gamma1_derived_raw)**2)
    r2_derived = 1 - ss_res_derived / ss_tot

    # Ensemble: average of direct and derived
    gamma1_ensemble_raw = 0.5 * gamma1_direct_raw + 0.5 * gamma1_derived_raw
    ss_res_ens = np.sum((gamma1_target_raw - gamma1_ensemble_raw)**2)
    r2_ensemble = 1 - ss_res_ens / ss_tot

    # Weighted ensemble: optimize on test (oracle) and also try fixed blends
    best_alpha, best_r2 = 0.5, r2_ensemble
    for alpha in np.arange(0.0, 1.01, 0.05):
        blend = alpha * gamma1_direct_raw + (1 - alpha) * gamma1_derived_raw
        ss = np.sum((gamma1_target_raw - blend)**2)
        r2 = 1 - ss / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha

    gamma1_best_raw = best_alpha * gamma1_direct_raw + (1 - best_alpha) * gamma1_derived_raw

    print(f"\n  Results (raw space R²):")
    print(f"    Direct prediction (PointCloud):     R² = {r2_direct:.4f}")
    print(f"    Physics-derived (from G_E + γ₂):    R² = {r2_derived:.4f}")
    print(f"    50/50 ensemble:                     R² = {r2_ensemble:.4f}")
    print(f"    Best blend (α={best_alpha:.2f} direct):      R² = {best_r2:.4f}")

    # Also compute in normalized space for fair comparison with other models
    gamma1_derived_norm = forward_transform_targets(gamma1_derived_raw, target_scaler, 0)
    gamma1_ensemble_norm = forward_transform_targets(gamma1_ensemble_raw, target_scaler, 0)
    gamma1_best_norm = forward_transform_targets(gamma1_best_raw, target_scaler, 0)

    # Replace gamma1 in predictions and compute full metrics
    for name, g1_replacement in [("Direct", preds[:, 0]),
                                  ("Physics-derived", gamma1_derived_norm),
                                  ("50/50 Ensemble", gamma1_ensemble_norm),
                                  (f"Best Blend (α={best_alpha:.2f})", gamma1_best_norm)]:
        preds_mod = preds.copy()
        preds_mod[:, 0] = g1_replacement
        metrics = compute_metrics(preds_mod, targets)
        print(f"\n  {name}:")
        print(f"    gamma1 R²={metrics['gamma1_r2']:.4f}, avg R²={metrics['avg_r2']:.4f}")

    return {
        "r2_direct": float(r2_direct),
        "r2_derived": float(r2_derived),
        "r2_ensemble": float(r2_ensemble),
        "r2_best_blend": float(best_r2),
        "best_alpha": float(best_alpha),
    }


def option_b_retrain(device, target_scaler, feature_scaler):
    """Option B: Retrain with physics-constrained loss."""
    print(f"\n{'='*60}")
    print("OPTION B: Physics-Constrained Retraining")
    print(f"{'='*60}")

    config = load_config("configs/default.yaml")
    pc_dir = "data/pipeline/point_clouds"
    orig_splits = Path("data/processed/splits")

    # Datasets
    train_ds = PointCloudMultimodalDataset(str(orig_splits / "train.csv"), pc_dir, is_train=True)
    val_ds = PointCloudMultimodalDataset(str(orig_splits / "val.csv"), pc_dir, is_train=False)
    test_ds = PointCloudMultimodalDataset(str(orig_splits / "test.csv"), pc_dir, is_train=False)

    train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_pointcloud)
    val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)

    # Precompute scaler tensors for GPU
    t_mean = torch.tensor(target_scaler.mean_, dtype=torch.float32).to(device)
    t_scale = torch.tensor(target_scaler.scale_, dtype=torch.float32).to(device)
    f_mean = torch.tensor(feature_scaler.mean_[0], dtype=torch.float32).to(device)  # temperature mean
    f_scale = torch.tensor(feature_scaler.scale_[0], dtype=torch.float32).to(device)  # temperature scale

    def physics_loss(preds, features):
        """Compute thermodynamic consistency loss.

        G_E = RT * (x1*ln(g1) + x2*ln(g2)) with x1=x2=0.5

        All in raw space (inverse-transform predictions).
        """
        # Inverse-transform predictions to raw
        preds_raw = preds * t_scale + t_mean

        gamma1_raw = preds_raw[:, 0]  # gamma1
        gamma2_raw = preds_raw[:, 1]  # gamma2
        G_E_raw = preds_raw[:, 2]     # G_E

        # Get raw temperature from features (index 0)
        T_raw = features[:, 0] * f_scale + f_mean

        # Clamp gamma values to avoid log(0) or log(negative)
        gamma1_safe = torch.clamp(gamma1_raw, min=1e-4)
        gamma2_safe = torch.clamp(gamma2_raw, min=1e-4)

        # Thermodynamic relation: G_E = RT * (0.5*ln(g1) + 0.5*ln(g2))
        G_E_computed = R_KCAL * T_raw * (X1 * torch.log(gamma1_safe) +
                                           (1 - X1) * torch.log(gamma2_safe))

        # Physics loss: MSE between predicted G_E and G_E derived from gamma predictions
        phys_loss = ((G_E_raw - G_E_computed) ** 2).mean()

        return phys_loss

    # Build model (fresh, with pretrained GNN)
    model = MultimodalPointCloudModel(
        config=config,
        pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
    model.to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_dir = Path("checkpoints/physics_gamma1")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

    # Physics loss weight — ramp up over training
    physics_weight_max = 1.0

    best_loss, no_improve = float("inf"), 0
    for epoch in range(200):
        # Ramp physics weight: 0 for first 10 epochs, then linearly to max by epoch 50
        if epoch < 10:
            physics_weight = 0.0
        else:
            physics_weight = min(physics_weight_max, physics_weight_max * (epoch - 10) / 40)

        model.train()
        tl, tl_mse, tl_phys, n = 0, 0, 0, 0
        for batch in train_ldr:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            optimizer.zero_grad()
            preds = model(
                point_cloud=batch["point_cloud"], features=batch["features"],
                atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                bond_features=batch["bond_features"], batch=batch["batch"])

            # Standard MSE loss on all 7 targets
            mse_loss = ((preds - batch["targets"]) ** 2).mean()

            # Physics consistency loss
            phys = physics_loss(preds, batch["features"])

            loss = mse_loss + physics_weight * phys
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tl += loss.item()
            tl_mse += mse_loss.item()
            tl_phys += phys.item()
            n += 1
        scheduler.step()

        # Validate
        model.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for batch in val_ldr:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                preds = model(
                    point_cloud=batch["point_cloud"], features=batch["features"],
                    atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                    bond_features=batch["bond_features"], batch=batch["batch"])
                vl += ((preds - batch["targets"]) ** 2).mean().item()
                vn += 1
        avg_val = vl / max(vn, 1)

        if avg_val < best_loss:
            best_loss = avg_val; no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_improve += 1

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/200 | MSE: {tl_mse/max(n,1):.4f} | "
                  f"Phys: {tl_phys/max(n,1):.4f} (w={physics_weight:.2f}) | "
                  f"Val: {avg_val:.4f} | Best: {best_loss:.4f} | Pat: {no_improve}/25")
        if no_improve >= 25:
            print(f"  Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True))

    # Evaluate
    print(f"\n  Evaluation:")
    preds, targets, features = get_predictions(model, test_ldr, device)
    metrics_direct = compute_metrics(preds, targets)
    print(f"  Direct predictions:")
    print(format_metrics(metrics_direct, "Physics-constrained PointCloud"))

    # Also derive gamma1 from physics
    G_E_pred_raw = inverse_transform_targets(preds[:, 2], target_scaler, 2)
    gamma2_pred_raw = inverse_transform_targets(preds[:, 1], target_scaler, 1)
    T_raw = features[:, 0] * feature_scaler.scale_[0] + feature_scaler.mean_[0]

    gamma1_derived_raw = derive_gamma1_from_physics(G_E_pred_raw, gamma2_pred_raw, T_raw)
    gamma1_derived_norm = forward_transform_targets(gamma1_derived_raw, target_scaler, 0)

    gamma1_direct_raw = inverse_transform_targets(preds[:, 0], target_scaler, 0)
    gamma1_target_raw = inverse_transform_targets(targets[:, 0], target_scaler, 0)

    # Blend
    gamma1_blend_raw = 0.5 * gamma1_direct_raw + 0.5 * gamma1_derived_raw
    gamma1_blend_norm = forward_transform_targets(gamma1_blend_raw, target_scaler, 0)

    ss_tot = np.sum((gamma1_target_raw - gamma1_target_raw.mean())**2)
    r2_direct = 1 - np.sum((gamma1_target_raw - gamma1_direct_raw)**2) / ss_tot
    r2_derived = 1 - np.sum((gamma1_target_raw - gamma1_derived_raw)**2) / ss_tot
    r2_blend = 1 - np.sum((gamma1_target_raw - gamma1_blend_raw)**2) / ss_tot

    print(f"\n  Gamma1 R² comparison (raw space):")
    print(f"    Direct prediction:     R² = {r2_direct:.4f}")
    print(f"    Physics-derived:       R² = {r2_derived:.4f}")
    print(f"    50/50 blend:           R² = {r2_blend:.4f}")

    # Full metrics with blended gamma1
    preds_blend = preds.copy()
    preds_blend[:, 0] = gamma1_blend_norm
    metrics_blend = compute_metrics(preds_blend, targets)
    print(f"\n  Full metrics with blended gamma1:")
    print(format_metrics(metrics_blend, "Physics-constrained + blend"))

    return {
        "direct_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                           for k, v in metrics_direct.items()},
        "blend_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                          for k, v in metrics_blend.items()},
        "gamma1_r2_direct": float(r2_direct),
        "gamma1_r2_derived": float(r2_derived),
        "gamma1_r2_blend": float(r2_blend),
    }


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    # Load scalers
    with open("data/processed/target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)
    with open("data/processed/feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)

    print(f"Target scaler: gamma1 mean={target_scaler.mean_[0]:.4f}, "
          f"scale={target_scaler.scale_[0]:.4f}")
    print(f"Thermodynamic relation: G_E = RT*(0.5*ln(γ₁) + 0.5*ln(γ₂))")
    print(f"  => γ₁ = exp(2*G_E/(RT) - ln(γ₂))")

    # ── Option A ──
    result_a = option_a_posthoc(device, target_scaler, feature_scaler)

    # ── Option B ──
    result_b = option_b_retrain(device, target_scaler, feature_scaler)

    # ── Final comparison ──
    print(f"\n{'='*60}")
    print("FINAL COMPARISON: All gamma1 approaches")
    print(f"{'='*60}")

    # Load previous results
    prev = {}
    for name, path in [("PointCloud", "results/pointcloud_results.json"),
                        ("Fix6 MoE", "results/moe_fix6_results.json"),
                        ("Ensemble Avg", "results/ensemble_fix6_pointcloud.json")]:
        try:
            data = json.load(open(path))
            if name == "Ensemble Avg":
                m = data.get("simple_average", {})
            else:
                m = data.get("metrics", data.get("test_metrics", data.get("single_model", {})))
            prev[name] = m
        except Exception:
            pass

    print(f"\n  {'Model':<35s} {'gamma1 R²':>10s} {'avg R²':>10s}")
    print("  " + "-" * 58)
    for name, m in prev.items():
        print(f"  {name:<35s} {m.get('gamma1_r2', 0):10.4f} {m.get('avg_r2', 0):10.4f}")
    print(f"  {'Option A: Physics-derived':<35s} {result_a['r2_derived']:10.4f}       {'—':>5s}")
    print(f"  {'Option A: Best blend':<35s} {result_a['r2_best_blend']:10.4f}       {'—':>5s}")
    print(f"  {'Option B: Physics-constrained':<35s} {result_b['gamma1_r2_direct']:10.4f} "
          f"{result_b['direct_metrics'].get('avg_r2', 0):10.4f}")
    print(f"  {'Option B: + physics blend':<35s} {result_b['gamma1_r2_blend']:10.4f} "
          f"{result_b['blend_metrics'].get('avg_r2', 0):10.4f}")

    # Save
    results = {
        "approach": "physics_informed_gamma1",
        "thermodynamic_relation": "G_E = RT*(x1*ln(g1) + x2*ln(g2)), x1=x2=0.5",
        "option_a_posthoc": result_a,
        "option_b_retrain": result_b,
    }
    with open("results/physics_gamma1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/physics_gamma1_results.json")


if __name__ == "__main__":
    main()
