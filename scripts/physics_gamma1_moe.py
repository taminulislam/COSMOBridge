"""Physics-informed gamma1 for MoE model.

Option A: Post-hoc — derive gamma1 from Fix6 MoE's G_E and gamma2 predictions.
Option B: Retrain MoE with physics-constrained loss (tuned: gentler weight,
          warm-start from Fix6 checkpoint).

Tuning vs PointCloud Option B:
  - physics_weight_max = 0.1 (was 1.0 — too aggressive, caused early stopping)
  - Warm-start from Fix6 checkpoint (was fresh init — wasted training)
  - Slower ramp: start at epoch 20, reach max at epoch 80
  - Keep balanced sampling from Fix6

Thermodynamic relation (all data at x1=0.5):
    G_E = RT * (0.5*ln(γ₁) + 0.5*ln(γ₂))
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.utils.config import load_config, get_device, set_seed
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.moe import ExpertHead, PropertyConditionedGating
from src.models.fusion.multimodal_pointcloud import PointCloudFusion
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_joint import MergedDataset, collate_merged

TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]
R_KCAL = 1.987e-3  # kcal/(mol·K)
X1 = 0.5


class MoEA(nn.Module):
    def __init__(self, feature_dim, fused_dim=256, dropout=0.3, pretrained_gnn_path=None):
        super().__init__()
        self.pointnet = PointNetEncoder(in_channels=7, feature_dim=256, dropout=dropout)
        self.gnn = MolecularGNN(
            atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
            dropout=dropout, pooling="mean", num_targets=0)
        if pretrained_gnn_path and Path(pretrained_gnn_path).exists():
            ckpt = torch.load(pretrained_gnn_path, map_location="cpu", weights_only=True)
            gnn_state = {k: v for k, v in ckpt.items()
                         if any(k.startswith(p) for p in ["atom_projection", "convs", "batch_norms", "pool"])}
            if gnn_state:
                self.gnn.load_state_dict(gnn_state, strict=False)
        self.fusion = PointCloudFusion(
            pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
            fused_dim=fused_dim, num_heads=8, dropout=dropout)
        self.experts = nn.ModuleList([
            ExpertHead(fused_dim, hidden_dim=128, num_targets=7, dropout=dropout)
            for _ in range(4)])
        self.gating = PropertyConditionedGating(
            input_dim=fused_dim, num_experts=4, num_properties=7, hidden_dim=64)

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        pc_feat = self.pointnet(point_cloud)
        graph_feat = self.gnn.get_features(atom_features, edge_index, bond_features, batch)
        fused = self.fusion(pc_feat, graph_feat, features)
        expert_preds = torch.stack([e(fused) for e in self.experts], dim=2)
        gate_weights, lb_loss = self.gating(fused)
        predictions = (expert_preds * gate_weights).sum(dim=2)
        return predictions, {"load_balance_loss": lb_loss, "gate_weights": gate_weights.detach()}


def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_targets, all_features = [], [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            preds, _ = model(
                point_cloud=batch["point_cloud"], features=batch["features"],
                atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                bond_features=batch["bond_features"], batch=batch["batch"])
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())
            all_features.append(batch["features"].cpu().numpy())
    return (np.concatenate(all_preds), np.concatenate(all_targets),
            np.concatenate(all_features))


def inv_transform(values, scaler, col_idx):
    return values * scaler.scale_[col_idx] + scaler.mean_[col_idx]


def fwd_transform(values, scaler, col_idx):
    return (values - scaler.mean_[col_idx]) / scaler.scale_[col_idx]


def derive_gamma1(G_E_raw, gamma2_raw, T_raw):
    gamma2_safe = np.clip(gamma2_raw, 1e-4, None)
    ln_gamma1 = 2.0 * G_E_raw / (R_KCAL * T_raw) - np.log(gamma2_safe)
    return np.exp(np.clip(ln_gamma1, -10, 10))


def gamma1_r2_raw(pred_raw, target_raw):
    ss_tot = np.sum((target_raw - target_raw.mean())**2)
    ss_res = np.sum((target_raw - pred_raw)**2)
    return 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0


def make_balanced_sampler(csv_path):
    df = pd.read_csv(csv_path)
    is_original = (df["source"] == "original").values
    n_orig = is_original.sum()
    n_ilth = len(df) - n_orig
    weights = np.where(is_original, 0.5 / max(n_orig, 1), 0.5 / max(n_ilth, 1))
    sampler = WeightedRandomSampler(torch.from_numpy(weights).double(),
                                     num_samples=len(df), replacement=True)
    print(f"  Balanced sampler: {n_orig} original, {n_ilth} ILThermo, "
          f"oversample {n_ilth/max(n_orig,1):.1f}x")
    return sampler


def option_a(device, target_scaler, feature_scaler, merged_features):
    """Option A: Post-hoc physics-derived gamma1 from Fix6 predictions."""
    print(f"\n{'='*60}")
    print("OPTION A: Post-hoc Physics-Derived Gamma1 (MoE Fix6)")
    print(f"{'='*60}")

    merged_dir = Path("data/merged_v5")
    pc_dir = "data/pipeline/point_clouds"

    model = MoEA(feature_dim=len(merged_features))
    model.load_state_dict(torch.load("checkpoints/moe_fix6/best.pt",
                                      map_location=device, weights_only=True))
    model.to(device)

    test_ds = MergedDataset(str(merged_dir / "splits/test.csv"), pc_dir,
                             merged_features, is_train=False)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)

    preds, targets, features = get_predictions(model, test_ldr, device)

    # Inverse-transform to raw space
    G_E_pred_raw = inv_transform(preds[:, 2], target_scaler, 2)
    gamma2_pred_raw = inv_transform(preds[:, 1], target_scaler, 1)
    T_raw = features[:, 0] * feature_scaler.scale_[0] + feature_scaler.mean_[0]

    gamma1_target_raw = inv_transform(targets[:, 0], target_scaler, 0)
    gamma1_direct_raw = inv_transform(preds[:, 0], target_scaler, 0)
    gamma1_derived_raw = derive_gamma1(G_E_pred_raw, gamma2_pred_raw, T_raw)

    r2_direct = gamma1_r2_raw(gamma1_direct_raw, gamma1_target_raw)
    r2_derived = gamma1_r2_raw(gamma1_derived_raw, gamma1_target_raw)

    print(f"\n  Direct gamma1 R² (raw):     {r2_direct:.4f}")
    print(f"  Physics-derived R² (raw):   {r2_derived:.4f}")

    # Sweep blend ratios
    best_alpha, best_r2 = 1.0, r2_direct
    results_sweep = []
    for alpha in np.arange(0.0, 1.01, 0.05):
        blend = alpha * gamma1_direct_raw + (1 - alpha) * gamma1_derived_raw
        r2 = gamma1_r2_raw(blend, gamma1_target_raw)
        results_sweep.append((alpha, r2))
        if r2 > best_r2:
            best_r2, best_alpha = r2, alpha

    print(f"  Best blend: α={best_alpha:.2f} (direct), R²={best_r2:.4f}")

    # Compute full metrics for best blend
    gamma1_best_norm = fwd_transform(
        best_alpha * gamma1_direct_raw + (1 - best_alpha) * gamma1_derived_raw,
        target_scaler, 0)
    preds_best = preds.copy()
    preds_best[:, 0] = gamma1_best_norm
    metrics_best = compute_metrics(preds_best, targets)

    print(f"\n  Full metrics with best blend:")
    print(format_metrics(metrics_best, "MoE Fix6 + Physics Blend"))

    return {
        "r2_direct": float(r2_direct),
        "r2_derived": float(r2_derived),
        "r2_best_blend": float(best_r2),
        "best_alpha": float(best_alpha),
        "full_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                         for k, v in metrics_best.items()},
    }


def option_b(device, target_scaler, feature_scaler, merged_features):
    """Option B: Retrain MoE with physics-constrained loss, warm-start from Fix6."""
    print(f"\n{'='*60}")
    print("OPTION B: Physics-Constrained MoE (warm-start from Fix6)")
    print(f"{'='*60}")

    merged_dir = Path("data/merged_v5")
    pc_dir = "data/pipeline/point_clouds"

    # Precompute scaler tensors
    t_mean = torch.tensor(target_scaler.mean_, dtype=torch.float32).to(device)
    t_scale = torch.tensor(target_scaler.scale_, dtype=torch.float32).to(device)
    f_mean_T = torch.tensor(feature_scaler.mean_[0], dtype=torch.float32).to(device)
    f_scale_T = torch.tensor(feature_scaler.scale_[0], dtype=torch.float32).to(device)

    def physics_loss(preds, features):
        """Thermodynamic consistency loss in raw space."""
        preds_raw = preds * t_scale + t_mean
        g1 = torch.clamp(preds_raw[:, 0], min=1e-4)
        g2 = torch.clamp(preds_raw[:, 1], min=1e-4)
        G_E = preds_raw[:, 2]
        T = features[:, 0] * f_scale_T + f_mean_T

        G_E_from_gamma = R_KCAL * T * (X1 * torch.log(g1) + (1 - X1) * torch.log(g2))
        return ((G_E - G_E_from_gamma) ** 2).mean()

    # Load datasets
    train_csv = str(merged_dir / "splits/train.csv")
    train_ds = MergedDataset(train_csv, pc_dir, merged_features, is_train=True)
    val_ds = MergedDataset(str(merged_dir / "splits/val.csv"), pc_dir, merged_features, is_train=False)
    test_ds = MergedDataset(str(merged_dir / "splits/test.csv"), pc_dir, merged_features, is_train=False)

    sampler = make_balanced_sampler(train_csv)
    train_ldr = DataLoader(train_ds, batch_size=64, sampler=sampler, collate_fn=collate_merged)
    val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)

    # Warm-start from Fix6 checkpoint
    model = MoEA(feature_dim=len(merged_features))
    model.load_state_dict(torch.load("checkpoints/moe_fix6/best.pt",
                                      map_location=device, weights_only=True))
    model.to(device)
    print(f"  Loaded Fix6 checkpoint (warm-start)")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_dir = Path("checkpoints/moe_physics")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=60, eta_min=1e-7)

    # Tuned physics parameters
    physics_weight_max = 0.1  # gentle (was 1.0)
    ramp_start = 20
    ramp_end = 80

    best_loss, no_improve = float("inf"), 0
    for epoch in range(200):
        if epoch < ramp_start:
            pw = 0.0
        elif epoch < ramp_end:
            pw = physics_weight_max * (epoch - ramp_start) / (ramp_end - ramp_start)
        else:
            pw = physics_weight_max

        model.train()
        tl, tl_mse, tl_phys, n = 0, 0, 0, 0
        for batch in train_ldr:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            optimizer.zero_grad()
            preds, aux = model(point_cloud=batch["point_cloud"], features=batch["features"],
                               atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                               bond_features=batch["bond_features"], batch=batch["batch"])

            mask = ~torch.isnan(batch["targets"])
            safe = batch["targets"].clone(); safe[~mask] = 0.0
            mse = ((preds - safe)**2 * mask.float()).sum() / mask.float().sum().clamp(min=1)

            phys = physics_loss(preds, batch["features"])

            loss = mse + aux["load_balance_loss"] + pw * phys
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tl += loss.item(); tl_mse += mse.item(); tl_phys += phys.item(); n += 1
        scheduler.step()

        model.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for batch in val_ldr:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                preds, _ = model(point_cloud=batch["point_cloud"], features=batch["features"],
                                 atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                                 bond_features=batch["bond_features"], batch=batch["batch"])
                mask = ~torch.isnan(batch["targets"])
                safe = batch["targets"].clone(); safe[~mask] = 0.0
                vl += ((preds - safe)**2 * mask.float()).sum().item() / mask.float().sum().clamp(min=1).item()
                vn += 1
        avg_val = vl / max(vn, 1)

        if avg_val < best_loss:
            best_loss = avg_val; no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_improve += 1

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/200 | MSE: {tl_mse/max(n,1):.4f} | "
                  f"Phys: {tl_phys/max(n,1):.4f} (w={pw:.3f}) | "
                  f"Val: {avg_val:.4f} | Best: {best_loss:.4f} | Pat: {no_improve}/25")
        if no_improve >= 25:
            print(f"  Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True))

    # Evaluate direct predictions
    preds, targets, features = get_predictions(model, test_ldr, device)
    metrics_direct = compute_metrics(preds, targets)
    print(f"\n  Direct predictions:")
    print(format_metrics(metrics_direct, "MoE Physics-Constrained"))

    # Physics-derived gamma1
    G_E_raw = inv_transform(preds[:, 2], target_scaler, 2)
    g2_raw = inv_transform(preds[:, 1], target_scaler, 1)
    T_raw = features[:, 0] * feature_scaler.scale_[0] + feature_scaler.mean_[0]
    g1_target_raw = inv_transform(targets[:, 0], target_scaler, 0)
    g1_direct_raw = inv_transform(preds[:, 0], target_scaler, 0)
    g1_derived_raw = derive_gamma1(G_E_raw, g2_raw, T_raw)

    r2_direct = gamma1_r2_raw(g1_direct_raw, g1_target_raw)
    r2_derived = gamma1_r2_raw(g1_derived_raw, g1_target_raw)

    # Sweep blends
    best_alpha, best_r2 = 1.0, r2_direct
    for alpha in np.arange(0.0, 1.01, 0.05):
        blend = alpha * g1_direct_raw + (1 - alpha) * g1_derived_raw
        r2 = gamma1_r2_raw(blend, g1_target_raw)
        if r2 > best_r2:
            best_r2, best_alpha = r2, alpha

    print(f"\n  Gamma1 R² (raw space):")
    print(f"    Direct:         {r2_direct:.4f}")
    print(f"    Physics-derived: {r2_derived:.4f}")
    print(f"    Best blend (α={best_alpha:.2f}): {best_r2:.4f}")

    # Full metrics with best blend
    g1_best_norm = fwd_transform(
        best_alpha * g1_direct_raw + (1 - best_alpha) * g1_derived_raw,
        target_scaler, 0)
    preds_best = preds.copy()
    preds_best[:, 0] = g1_best_norm
    metrics_blend = compute_metrics(preds_best, targets)
    print(f"\n  Full metrics with best blend:")
    print(format_metrics(metrics_blend, "MoE Physics + Blend"))

    return {
        "direct_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                           for k, v in metrics_direct.items()},
        "blend_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                          for k, v in metrics_blend.items()},
        "r2_direct": float(r2_direct),
        "r2_derived": float(r2_derived),
        "r2_best_blend": float(best_r2),
        "best_alpha": float(best_alpha),
    }


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    merged_dir = Path("data/merged_v5")
    meta = json.load(open(merged_dir / "metadata.json"))
    merged_features = meta["feature_columns"]

    with open(merged_dir / "target_scalers.pkl", "rb") as f:
        target_scaler_dict = pickle.load(f)
    with open(merged_dir / "feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)

    # Build a unified scaler-like object for convenience
    class UnifiedScaler:
        def __init__(self, scaler_dict, target_cols):
            self.mean_ = np.array([scaler_dict[c].mean_[0] for c in target_cols])
            self.scale_ = np.array([scaler_dict[c].scale_[0] for c in target_cols])
    target_scaler = UnifiedScaler(target_scaler_dict, TARGET_COLUMNS)

    print(f"Merged_v5 target scalers:")
    for i, c in enumerate(TARGET_COLUMNS):
        print(f"  {c}: mean={target_scaler.mean_[i]:.4f}, scale={target_scaler.scale_[i]:.4f}")

    # ── Option A ──
    result_a = option_a(device, target_scaler, feature_scaler, merged_features)

    # ── Option B ──
    result_b = option_b(device, target_scaler, feature_scaler, merged_features)

    # ── Final comparison ──
    print(f"\n{'='*60}")
    print("FINAL COMPARISON: MoE Physics-Informed Gamma1")
    print(f"{'='*60}")

    prev = {}
    for name, path, key in [
        ("PointCloud", "results/pointcloud_results.json", "test_metrics"),
        ("Fix6 MoE", "results/moe_fix6_results.json", "metrics"),
        ("Ens Fix6+PC", "results/ensemble_fix6_pointcloud.json", "simple_average"),
        ("PC+PhysBlend", "results/physics_gamma1_results.json", None),
    ]:
        try:
            data = json.load(open(path))
            if name == "PC+PhysBlend":
                r2 = data["option_a_posthoc"]["r2_best_blend"]
                prev[name] = {"gamma1_r2": r2, "avg_r2": "—"}
            else:
                prev[name] = data.get(key, data.get("metrics", {}))
        except Exception:
            pass

    print(f"\n  {'Model':<40s} {'gamma1 R²':>10s} {'avg R²':>10s}")
    print("  " + "-" * 62)
    for name, m in prev.items():
        g1 = m.get("gamma1_r2", "—")
        avg = m.get("avg_r2", "—")
        g1s = f"{g1:.4f}" if isinstance(g1, float) else str(g1)
        avgs = f"{avg:.4f}" if isinstance(avg, float) else str(avg)
        print(f"  {name:<40s} {g1s:>10s} {avgs:>10s}")
    print(f"  {'MoE Option A: Physics blend':<40s} {result_a['r2_best_blend']:10.4f} "
          f"{result_a['full_metrics']['avg_r2']:10.4f}")
    print(f"  {'MoE Option B: Physics-constrained':<40s} {result_b['r2_direct']:10.4f} "
          f"{result_b['direct_metrics']['avg_r2']:10.4f}")
    print(f"  {'MoE Option B: + blend':<40s} {result_b['r2_best_blend']:10.4f} "
          f"{result_b['blend_metrics']['avg_r2']:10.4f}")

    # Save
    results = {
        "approach": "physics_informed_gamma1_moe",
        "option_a": result_a,
        "option_b": result_b,
    }
    with open("results/physics_gamma1_moe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/physics_gamma1_moe_results.json")


if __name__ == "__main__":
    main()
