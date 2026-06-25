"""Train GBH v3: Property-Adaptive HyperNet + D-MPNN + Physics Loss.

Four targeted fixes:
1. D-MPNN backbone (replaces GAT)
2. Compact architecture (~400K params)
3. Property-adaptive gate (learn which props need surface fusion)
4. Physics auxiliary loss (G_E consistency)
"""

import sys
import json
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
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.models.fusion.gbh_v3 import GBHv3
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_joint import MergedDataset, collate_merged

R_KCAL = 1.987e-3
X1 = 0.5


def prepare_stilt_data(merged_csv, output_path):
    df = pd.read_csv(merged_csv)
    orig = df[df["source"] == "original"].copy()
    ilth = df[df["source"] != "original"].copy()
    ilth["gamma1"] = np.nan
    orig_rep = pd.concat([orig] * 48, ignore_index=True)
    balanced = pd.concat([orig_rep, ilth], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    balanced.to_csv(output_path, index=False)
    print(f"  STILT data: {len(orig)}×48 + {len(ilth)} = {len(balanced)} samples")
    return len(balanced)


def physics_loss(preds, features, target_scaler_mean, target_scaler_scale, feature_scale):
    """Thermodynamic consistency: G_E = RT(x1 ln g1 + x2 ln g2)."""
    # Inverse-transform to raw
    g1_raw = preds[:, 0] * target_scaler_scale[0] + target_scaler_mean[0]
    g2_raw = preds[:, 1] * target_scaler_scale[1] + target_scaler_mean[1]
    GE_raw = preds[:, 2] * target_scaler_scale[2] + target_scaler_mean[2]
    T_raw = features[:, 0] * feature_scale[0] + feature_scale[1]

    g1_safe = torch.clamp(g1_raw, min=1e-4)
    g2_safe = torch.clamp(g2_raw, min=1e-4)

    GE_computed = R_KCAL * T_raw * (X1 * torch.log(g1_safe) + (1-X1) * torch.log(g2_safe))
    return ((GE_raw - GE_computed) ** 2).mean()


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            preds, _ = model(point_cloud=batch["point_cloud"], features=batch["features"],
                             atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                             bond_features=batch["bond_features"], batch=batch["batch"])
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    pc_dir = "data/pipeline/point_clouds"
    merged_dir = Path("data/merged_v5")
    meta = json.load(open(merged_dir / "metadata.json"))
    features = meta["feature_columns"]

    # Prepare STILT data
    stilt_dir = Path("data/gbh_v3_stilt")
    stilt_dir.mkdir(parents=True, exist_ok=True)
    train_path = stilt_dir / "train.csv"
    if not train_path.exists():
        prepare_stilt_data(merged_dir / "splits/train.csv", train_path)
    else:
        print(f"  Using existing STILT data: {train_path}")

    # Load datasets
    train_ds = MergedDataset(str(train_path), pc_dir, features, is_train=True)
    val_ds = MergedDataset(str(merged_dir / "splits/val.csv"), pc_dir, features, is_train=False)
    test_ds = MergedDataset(str(merged_dir / "splits/test.csv"), pc_dir, features, is_train=False)

    train_ldr = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_merged)
    val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)

    # Load scalers for physics loss
    import pickle
    with open(merged_dir / "target_scalers.pkl", "rb") as f:
        ts = pickle.load(f)
    with open(merged_dir / "feature_scaler.pkl", "rb") as f:
        fs = pickle.load(f)

    ts_mean = torch.tensor([ts[c].mean_[0] for c in TARGET_COLUMNS], dtype=torch.float32).to(device)
    ts_scale = torch.tensor([ts[c].scale_[0] for c in TARGET_COLUMNS], dtype=torch.float32).to(device)
    fs_params = torch.tensor([fs.scale_[0], fs.mean_[0]], dtype=torch.float32).to(device)

    # Build model
    print(f"\n{'='*60}")
    print("GBH v3: Property-Adaptive + D-MPNN + Compact + Physics")
    print(f"{'='*60}")

    ckpt_dir = Path("checkpoints/gbh_v3")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = GBHv3(feature_dim=len(features), thermo_dim=5, dropout=0.25)
    model.to(device)

    # Resume from checkpoint if exists
    ckpt_path = ckpt_dir / "best.pt"
    if ckpt_path.exists():
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            print(f"  Resumed from checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"  Could not load checkpoint ({e}), training from scratch")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {n_params:,} (target: ~400K, Chemprop: ~300K)")
    print(f"  PointNet: {sum(p.numel() for p in model.pointnet.parameters()):,}")
    print(f"  D-MPNN: {sum(p.numel() for p in model.dmpnn.parameters()):,}")
    print(f"  Fusion: {sum(p.numel() for p in model.bilinear.parameters()) + sum(p.numel() for p in model.hypernet.parameters()):,}")
    print(f"  Prop gate: {sum(p.numel() for p in model.prop_gate.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=60, eta_min=1e-6)

    # Physics loss weight ramp
    phys_max = 0.1

    best_loss, no_improve = float("inf"), 0
    for epoch in range(200):
        phys_w = min(phys_max, phys_max * max(0, epoch - 15) / 45) if epoch > 15 else 0.0

        model.train()
        tl, tl_mse, tl_phys, n = 0, 0, 0, 0
        for batch in train_ldr:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            optimizer.zero_grad()
            preds, aux = model(point_cloud=batch["point_cloud"], features=batch["features"],
                               atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                               bond_features=batch["bond_features"], batch=batch["batch"])

            mask = ~torch.isnan(batch["targets"])
            safe = batch["targets"].clone(); safe[~mask] = 0.0
            mse = ((preds - safe)**2 * mask.float()).sum() / mask.float().sum().clamp(min=1)

            phys = physics_loss(preds, batch["features"], ts_mean, ts_scale, fs_params)

            loss = mse + phys_w * phys
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
                    if isinstance(v, torch.Tensor): batch[k] = v.to(device)
                preds, _ = model(point_cloud=batch["point_cloud"], features=batch["features"],
                                 atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                                 bond_features=batch["bond_features"], batch=batch["batch"])
                mask = ~torch.isnan(batch["targets"])
                safe = batch["targets"].clone(); safe[~mask] = 0.0
                vl += ((preds-safe)**2*mask.float()).sum().item()/mask.float().sum().clamp(min=1).item()
                vn += 1
        avg_val = vl / max(vn, 1)
        if avg_val < best_loss:
            best_loss = avg_val; no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_improve += 1
        if epoch % 20 == 0:
            gate_vals = torch.sigmoid(model.prop_gate.gate_logits).detach().cpu().numpy()
            gate_str = " ".join(f"{g:.2f}" for g in gate_vals)
            print(f"  Epoch {epoch:3d}/200 | MSE:{tl_mse/max(n,1):.4f} Phys:{tl_phys/max(n,1):.4f}(w={phys_w:.3f}) | "
                  f"Val:{avg_val:.4f} Best:{best_loss:.4f} Pat:{no_improve}/30 | Gates:[{gate_str}]")
        if no_improve >= 30:
            print(f"  Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True))

    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")

    preds, targets = evaluate(model, test_ldr, device)
    metrics = compute_metrics(preds, targets)
    print(format_metrics(metrics, "GBH v3"))

    # Property gate analysis
    gate_vals = torch.sigmoid(model.prop_gate.gate_logits).detach().cpu().numpy()
    print(f"\n  Learned property-adaptive gates (higher = more fusion):")
    for i, p in enumerate(TARGET_COLUMNS):
        bar = "█" * int(gate_vals[i] * 20)
        print(f"    {p:15s}: {gate_vals[i]:.3f} {bar}")

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    prev = {}
    for name, path, key in [
        ("Chemprop", "results/chemprop_results.json", "test_metrics"),
        ("CrossAttn PC", "results/pointcloud_results.json", None),
        ("GBHv2+STILT", "results/gbh_v2_stilt_results.json", "metrics"),
        ("STILT", "results/chemprop_tuned_results.json", "STILT_C"),
    ]:
        try:
            data = json.load(open(path))
            if key == "STILT_C": m = data.get("C: full mask, 48x OS", {}).get("metrics", {})
            elif key: m = data[key]
            else:
                for k in ['metrics','test_metrics']:
                    if k in data: m = data[k]; break
            prev[name] = m
        except: pass

    header = "  {:<12s}".format("Property")
    for name in prev:
        header += " {:>12s}".format(name[:12])
    header += " {:>12s}".format("GBH v3")
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<12s}".format(p)
        for name in prev:
            line += " {:12.4f}".format(prev[name].get(key, float('nan')))
        line += " {:12.4f}".format(metrics[key])
        print(line)

    line = "  {:<12s}".format("AVERAGE")
    for name in prev:
        line += " {:12.4f}".format(prev[name].get('avg_r2', float('nan')))
    line += " {:12.4f}".format(metrics['avg_r2'])
    print(line)

    # vs Chemprop detail
    if "Chemprop" in prev:
        base = prev["Chemprop"]
        print(f"\n  vs Chemprop:")
        wins = 0
        for p in TARGET_COLUMNS:
            key = f"{p}_r2"
            d = metrics[key] - base[key]
            if d > 0: wins += 1
            s = "+" if d > 0 else ""
            w = "WIN" if d > 0 else ("~tied" if abs(d) < 0.01 else "lose")
            print(f"    {p:15s}: {metrics[key]:.4f} vs {base[key]:.4f} ({s}{d:.4f}) {w}")
        d = metrics['avg_r2'] - base['avg_r2']
        s = "+" if d > 0 else ""
        print(f"    {'AVERAGE':15s}: {metrics['avg_r2']:.4f} vs {base['avg_r2']:.4f} ({s}{d:.4f}) wins {wins}/7")

    # Save
    results = {
        "model": "gbh_v3",
        "description": "Property-adaptive HyperNet fusion + D-MPNN + compact arch + physics loss + STILT data",
        "n_params": n_params,
        "gate_values": {p: float(g) for p, g in zip(TARGET_COLUMNS, gate_vals)},
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items()},
    }
    with open("results/gbh_v3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/gbh_v3_results.json")


if __name__ == "__main__":
    main()
