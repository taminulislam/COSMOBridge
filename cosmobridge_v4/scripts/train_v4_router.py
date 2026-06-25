"""Train COSMOBridge v4 with per-molecule router (I1).

Uses cached features (run cache_features.py first). Trains 10 seeds, saves
per-seed checkpoints and test predictions.

Usage:
    python cosmobridge_v4/scripts/train_v4_router.py
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cosmobridge_v4"))

from cosmobridge_v4.models.cosmobridge_v4_router import COSMOBridgeV4Router
from src.training.metrics import compute_metrics
from src.data.preprocessing import TARGET_COLUMNS


def load_cached(split):
    d = np.load(f"cosmobridge_v4/data/cached_{split}.npz", allow_pickle=True)
    return {
        "chemprop_fp": torch.from_numpy(d["chemprop_fp"]).float(),
        "surface_fp": torch.from_numpy(d["surface_fp"]).float(),
        "thermo_feat": torch.from_numpy(d["thermo_feat"]).float(),
        "targets": torch.from_numpy(d["targets"]).float(),
        "preds_fusion": torch.from_numpy(d["preds_fusion"]).float(),
        "preds_chemprop": torch.from_numpy(d["preds_chemprop"]).float(),
    }


def make_loader(data, batch_size=32, shuffle=False):
    ds = TensorDataset(
        data["chemprop_fp"], data["surface_fp"], data["thermo_feat"],
        data["preds_fusion"], data["preds_chemprop"], data["targets"],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_seed(seed, train_data, val_data, test_data, device, config):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = COSMOBridgeV4Router(
        graph_dim=train_data["chemprop_fp"].shape[1],
        surface_dim=train_data["surface_fp"].shape[1],
        thermo_dim=train_data["thermo_feat"].shape[1],
        hidden=config["hidden"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"],
                       weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    train_ldr = make_loader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_ldr = make_loader(val_data, batch_size=config["batch_size"], shuffle=False)

    best_val = float("inf")
    best_state = None
    no_imp = 0

    for epoch in range(config["epochs"]):
        # Anchor weight decays from 0.1 to 0.01 linearly
        anchor_weight = config["anchor_init"] * (1.0 - epoch / config["epochs"]) + \
                        config["anchor_final"] * (epoch / config["epochs"])

        model.train()
        for g_fp, s_fp, t_feat, pf, pc, y in train_ldr:
            g_fp, s_fp, t_feat = g_fp.to(device), s_fp.to(device), t_feat.to(device)
            pf, pc, y = pf.to(device), pc.to(device), y.to(device)

            optimizer.zero_grad()
            preds, aux = model(g_fp, s_fp, t_feat, pf, pc)
            mse = ((preds - y) ** 2).mean()
            anchor = model.anchor_loss(aux["logits"].clone().requires_grad_(False))
            # Re-run router to get differentiable logits for anchor loss
            router_logits = model.router(g_fp, s_fp, t_feat)
            anchor = model.anchor_loss(router_logits)
            loss = mse + anchor_weight * anchor
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for g_fp, s_fp, t_feat, pf, pc, y in val_ldr:
                g_fp, s_fp, t_feat = g_fp.to(device), s_fp.to(device), t_feat.to(device)
                pf, pc, y = pf.to(device), pc.to(device), y.to(device)
                preds, _ = model(g_fp, s_fp, t_feat, pf, pc)
                val_losses.append(((preds - y) ** 2).mean().item())
        val_mse = np.mean(val_losses)

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= config["patience"]:
            break

    # Load best and evaluate on test
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_targets, test_alphas = [], [], []
    with torch.no_grad():
        for g_fp, s_fp, t_feat, pf, pc, y in make_loader(test_data, batch_size=64):
            g_fp, s_fp, t_feat = g_fp.to(device), s_fp.to(device), t_feat.to(device)
            pf, pc, y = pf.to(device), pc.to(device), y.to(device)
            preds, aux = model(g_fp, s_fp, t_feat, pf, pc)
            test_preds.append(preds.cpu().numpy())
            test_targets.append(y.cpu().numpy())
            test_alphas.append(aux["alpha"].cpu().numpy())

    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)
    test_alphas = np.concatenate(test_alphas)

    metrics = compute_metrics(test_preds, test_targets)
    return metrics, test_preds, test_targets, test_alphas, best_state


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading cached data...")
    train_data = load_cached("train")
    val_data = load_cached("val")
    test_data = load_cached("test")
    print(f"  Train: {train_data['targets'].shape}, Val: {val_data['targets'].shape}, "
          f"Test: {test_data['targets'].shape}")

    config = {
        "hidden": 64,
        "dropout": 0.3,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "batch_size": 32,
        "epochs": 300,
        "patience": 40,
        "anchor_init": 0.1,
        "anchor_final": 0.01,
    }

    seeds = list(range(10))
    all_metrics = []

    pred_dir = Path("cosmobridge_v4/results/seed_predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("cosmobridge_v4/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        metrics, preds, targets, alphas, state = train_one_seed(
            seed, train_data, val_data, test_data, device, config,
        )
        print(f"  Test avg R²: {metrics['avg_r2']:.4f}")
        for p in TARGET_COLUMNS:
            print(f"    {p}: R²={metrics[f'{p}_r2']:.4f}")
        all_metrics.append(metrics)

        # Save predictions and checkpoint
        np.savez(
            pred_dir / f"seed_{seed}.npz",
            preds=preds, targets=targets, alphas=alphas,
        )
        torch.save(state, ckpt_dir / f"router_seed_{seed}.pt")

    # Summary
    print(f"\n{'='*70}")
    print(f"MULTI-SEED RESULTS (10 seeds)")
    print(f"{'='*70}")
    print(f"\n  {'Property':<12s} {'Mean R²':>9s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    print("  " + "-" * 50)
    for p in TARGET_COLUMNS:
        vals = [m[f"{p}_r2"] for m in all_metrics]
        print(f"  {p:<12s} {np.mean(vals):9.4f} {np.std(vals):8.4f} "
              f"{min(vals):8.4f} {max(vals):8.4f}")
    avgs = [m["avg_r2"] for m in all_metrics]
    print(f"  {'AVERAGE':<12s} {np.mean(avgs):9.4f} {np.std(avgs):8.4f} "
          f"{min(avgs):8.4f} {max(avgs):8.4f}")

    # Save results
    results = {
        "config": config,
        "seeds": seeds,
        "per_seed": [{k: float(v) for k, v in m.items()} for m in all_metrics],
        "mean": {p: float(np.mean([m[f"{p}_r2"] for m in all_metrics])) for p in TARGET_COLUMNS},
        "std": {p: float(np.std([m[f"{p}_r2"] for m in all_metrics])) for p in TARGET_COLUMNS},
        "avg_mean": float(np.mean(avgs)),
        "avg_std": float(np.std(avgs)),
    }
    with open("cosmobridge_v4/results/router_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: cosmobridge_v4/results/router_metrics.json")

    # Comparison with v3
    print(f"\n  COSMOBridge v3: 0.8013 ± 0.0006")
    print(f"  v4 Router:     {np.mean(avgs):.4f} ± {np.std(avgs):.4f}")
    print(f"  Δ: {np.mean(avgs) - 0.8013:+.4f}")


if __name__ == "__main__":
    main()
