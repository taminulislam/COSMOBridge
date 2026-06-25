"""Train COSMOBridge v4 Triple-Path router (I3).

3-path softmax routing: Fusion + Chemprop + Atom-Surface D-MPNN.
Requires cached features AND atom-surface predictions (run cache_features.py
and cache_atom_surface_preds.py first).
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cosmobridge_v4"))

from cosmobridge_v4.models.cosmobridge_v4_triple import COSMOBridgeV4Triple
from src.training.metrics import compute_metrics
from src.data.preprocessing import TARGET_COLUMNS


def load_cached(split):
    d = np.load(f"cosmobridge_v4/data/cached_{split}.npz", allow_pickle=True)
    as_preds = np.load(f"cosmobridge_v4/data/preds_atom_surface_{split}.npy")
    return {
        "chemprop_fp": torch.from_numpy(d["chemprop_fp"]).float(),
        "surface_fp": torch.from_numpy(d["surface_fp"]).float(),
        "thermo_feat": torch.from_numpy(d["thermo_feat"]).float(),
        "targets": torch.from_numpy(d["targets"]).float(),
        "preds_fusion": torch.from_numpy(d["preds_fusion"]).float(),
        "preds_chemprop": torch.from_numpy(d["preds_chemprop"]).float(),
        "preds_atom_surface": torch.from_numpy(as_preds).float(),
    }


def make_loader(data, batch_size=32, shuffle=False):
    ds = TensorDataset(
        data["chemprop_fp"], data["surface_fp"], data["thermo_feat"],
        data["preds_fusion"], data["preds_chemprop"], data["preds_atom_surface"],
        data["targets"],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_seed(seed, train_data, val_data, test_data, device, config):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = COSMOBridgeV4Triple(
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
        anchor_weight = config["anchor_init"] * (1.0 - epoch / config["epochs"]) + \
                        config["anchor_final"] * (epoch / config["epochs"])

        model.train()
        for g_fp, s_fp, t_feat, pf, pc, pas, y in train_ldr:
            g_fp, s_fp, t_feat = g_fp.to(device), s_fp.to(device), t_feat.to(device)
            pf, pc, pas, y = pf.to(device), pc.to(device), pas.to(device), y.to(device)

            optimizer.zero_grad()
            preds, _ = model(g_fp, s_fp, t_feat, pf, pc, pas)
            mse = ((preds - y) ** 2).mean()
            router_logits = model.router(g_fp, s_fp, t_feat)
            anchor = model.anchor_loss(router_logits)
            loss = mse + anchor_weight * anchor
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for g_fp, s_fp, t_feat, pf, pc, pas, y in val_ldr:
                g_fp, s_fp, t_feat = g_fp.to(device), s_fp.to(device), t_feat.to(device)
                pf, pc, pas, y = pf.to(device), pc.to(device), pas.to(device), y.to(device)
                preds, _ = model(g_fp, s_fp, t_feat, pf, pc, pas)
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

    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_targets, test_weights = [], [], []
    with torch.no_grad():
        for g_fp, s_fp, t_feat, pf, pc, pas, y in make_loader(test_data, batch_size=64):
            g_fp, s_fp, t_feat = g_fp.to(device), s_fp.to(device), t_feat.to(device)
            pf, pc, pas, y = pf.to(device), pc.to(device), pas.to(device), y.to(device)
            preds, aux = model(g_fp, s_fp, t_feat, pf, pc, pas)
            test_preds.append(preds.cpu().numpy())
            test_targets.append(y.cpu().numpy())
            test_weights.append(aux["weights"].cpu().numpy())

    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)
    test_weights = np.concatenate(test_weights)

    metrics = compute_metrics(test_preds, test_targets)
    return metrics, test_preds, test_targets, test_weights, best_state


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading cached data + atom-surface predictions...")
    train_data = load_cached("train")
    val_data = load_cached("val")
    test_data = load_cached("test")
    print(f"  Train: {train_data['targets'].shape}, Val: {val_data['targets'].shape}, "
          f"Test: {test_data['targets'].shape}")
    print(f"  AS preds available: {train_data['preds_atom_surface'].shape}")

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
    all_weights = []

    pred_dir = Path("cosmobridge_v4/results/seed_predictions_triple")
    pred_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("cosmobridge_v4/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        metrics, preds, targets, weights, state = train_one_seed(
            seed, train_data, val_data, test_data, device, config,
        )
        print(f"  Test avg R²: {metrics['avg_r2']:.4f}")
        for p in TARGET_COLUMNS:
            print(f"    {p}: R²={metrics[f'{p}_r2']:.4f}")
        all_metrics.append(metrics)
        all_weights.append(weights.mean(axis=0))  # mean weights per property

        np.savez(
            pred_dir / f"seed_{seed}.npz",
            preds=preds, targets=targets, weights=weights,
        )
        torch.save(state, ckpt_dir / f"triple_seed_{seed}.pt")

    # Summary
    print(f"\n{'='*70}")
    print(f"TRIPLE-PATH ROUTER RESULTS (10 seeds)")
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

    # Average routing weights across seeds
    mean_weights = np.stack(all_weights).mean(axis=0)  # (7, 3)
    print(f"\n  Average routing weights (across seeds):")
    print(f"  {'Property':<12s} {'Fusion':>8s} {'Chemprop':>10s} {'AtomSurf':>10s}")
    for i, p in enumerate(TARGET_COLUMNS):
        print(f"  {p:<12s} {mean_weights[i,0]:>8.3f} {mean_weights[i,1]:>10.3f} "
              f"{mean_weights[i,2]:>10.3f}")

    results = {
        "config": config,
        "seeds": seeds,
        "per_seed": [{k: float(v) for k, v in m.items()} for m in all_metrics],
        "mean": {p: float(np.mean([m[f"{p}_r2"] for m in all_metrics])) for p in TARGET_COLUMNS},
        "std": {p: float(np.std([m[f"{p}_r2"] for m in all_metrics])) for p in TARGET_COLUMNS},
        "avg_mean": float(np.mean(avgs)),
        "avg_std": float(np.std(avgs)),
        "routing_weights": {
            p: {"fusion": float(mean_weights[i, 0]),
                "chemprop": float(mean_weights[i, 1]),
                "atom_surface": float(mean_weights[i, 2])}
            for i, p in enumerate(TARGET_COLUMNS)
        },
    }
    with open("cosmobridge_v4/results/triple_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: cosmobridge_v4/results/triple_metrics.json")

    print(f"\n  COSMOBridge v3: 0.8013")
    print(f"  v4 I1 (2-path router): 0.8078")
    print(f"  v4 I3 (3-path router): {np.mean(avgs):.4f} ± {np.std(avgs):.4f}")
    print(f"  Δ vs v3: {np.mean(avgs) - 0.8013:+.4f}")
    print(f"  Δ vs I1: {np.mean(avgs) - 0.8078:+.4f}")


if __name__ == "__main__":
    main()
