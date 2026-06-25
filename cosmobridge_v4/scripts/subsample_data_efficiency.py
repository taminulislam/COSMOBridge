"""Subsample the v4 training set to produce a real data-efficiency curve.

For each subsample size n in [30, 50, 75, 100, 125, 150] and each seed in 0..9,
train the v4 router on a random subsample of n training ILs, keep val and test
fixed, evaluate on the 39-sample test set, and collect avg R² across 7 props.

Outputs:
  cosmobridge_v4/results/data_efficiency_curve.json
  cosmobridge_v4/results/data_efficiency_curve.npz
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

DATA_DIR = project_root / "cosmobridge_v4" / "data"
RESULTS_DIR = project_root / "cosmobridge_v4" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_cached(split):
    d = np.load(DATA_DIR / f"cached_{split}.npz", allow_pickle=True)
    return {
        "chemprop_fp": torch.from_numpy(d["chemprop_fp"]).float(),
        "surface_fp": torch.from_numpy(d["surface_fp"]).float(),
        "thermo_feat": torch.from_numpy(d["thermo_feat"]).float(),
        "targets": torch.from_numpy(d["targets"]).float(),
        "preds_fusion": torch.from_numpy(d["preds_fusion"]).float(),
        "preds_chemprop": torch.from_numpy(d["preds_chemprop"]).float(),
    }


def subsample(data, indices):
    return {k: v[indices] for k, v in data.items()}


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

    # Use a small batch size for small subsamples
    bs = min(config["batch_size"], max(4, len(train_data["targets"]) // 4))
    train_ldr = make_loader(train_data, batch_size=bs, shuffle=True)
    val_ldr = make_loader(val_data, batch_size=32, shuffle=False)

    best_val = float("inf")
    best_state = None
    no_imp = 0

    for epoch in range(config["epochs"]):
        anchor_weight = (
            config["anchor_init"] * (1.0 - epoch / config["epochs"]) +
            config["anchor_final"] * (epoch / config["epochs"])
        )

        model.train()
        for g_fp, s_fp, t_feat, pf, pc, y in train_ldr:
            g_fp, s_fp, t_feat = g_fp.to(device), s_fp.to(device), t_feat.to(device)
            pf, pc, y = pf.to(device), pc.to(device), y.to(device)
            optimizer.zero_grad()
            preds, aux = model(g_fp, s_fp, t_feat, pf, pc)
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
            for g_fp, s_fp, t_feat, pf, pc, y in val_ldr:
                g_fp, s_fp, t_feat = g_fp.to(device), s_fp.to(device), t_feat.to(device)
                pf, pc, y = pf.to(device), pc.to(device), y.to(device)
                preds, _ = model(g_fp, s_fp, t_feat, pf, pc)
                val_losses.append(((preds - y) ** 2).mean().item())
        val_mse = float(np.mean(val_losses))

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
    test_preds, test_targets = [], []
    with torch.no_grad():
        for g_fp, s_fp, t_feat, pf, pc, y in make_loader(test_data, batch_size=64):
            g_fp, s_fp, t_feat = g_fp.to(device), s_fp.to(device), t_feat.to(device)
            pf, pc, y = pf.to(device), pc.to(device), y.to(device)
            preds, _ = model(g_fp, s_fp, t_feat, pf, pc)
            test_preds.append(preds.cpu().numpy())
            test_targets.append(y.cpu().numpy())

    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)
    metrics = compute_metrics(test_preds, test_targets)
    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading cached data...")
    train_data_full = load_cached("train")
    val_data = load_cached("val")
    test_data = load_cached("test")
    n_train_full = len(train_data_full["targets"])
    print(f"  Full train: {n_train_full}, Val: {len(val_data['targets'])}, "
          f"Test: {len(test_data['targets'])}")

    config = {
        "hidden": 64, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-3,
        "batch_size": 32, "epochs": 300, "patience": 40,
        "anchor_init": 0.1, "anchor_final": 0.01,
    }

    subsample_sizes = [30, 50, 75, 100, 125, 150]
    seeds = list(range(10))

    curve = {}  # {n: [avg_r2 per seed]}
    per_prop_curve = {p: {} for p in TARGET_COLUMNS}

    for n in subsample_sizes:
        if n > n_train_full:
            continue
        print(f"\n{'='*60}")
        print(f"Subsample size n = {n}")
        print(f"{'='*60}")
        curve[n] = []
        for p in TARGET_COLUMNS:
            per_prop_curve[p][n] = []

        for seed in seeds:
            rng = np.random.RandomState(seed * 1000 + n)
            idx = rng.choice(n_train_full, size=n, replace=False)
            idx_t = torch.from_numpy(idx).long()
            sub_train = subsample(train_data_full, idx_t)
            try:
                m = train_one_seed(seed, sub_train, val_data, test_data,
                                    device, config)
                avg_r2 = m["avg_r2"]
                print(f"  n={n:3d} seed={seed}: avg_r2 = {avg_r2:.4f}")
                curve[n].append(avg_r2)
                for p in TARGET_COLUMNS:
                    per_prop_curve[p][n].append(m[f"{p}_r2"])
            except Exception as e:
                print(f"  n={n} seed={seed} FAILED: {e}")

    # Save JSON summary
    summary = {
        "subsample_sizes": subsample_sizes,
        "seeds": seeds,
        "avg_r2_curve": {
            str(n): {
                "mean": float(np.mean(v)) if v else None,
                "std":  float(np.std(v))  if v else None,
                "n_success": len(v),
                "values": [float(x) for x in v],
            } for n, v in curve.items()
        },
        "per_property_curve": {
            p: {
                str(n): {
                    "mean": float(np.mean(v)) if v else None,
                    "std":  float(np.std(v))  if v else None,
                } for n, v in per_prop_curve[p].items()
            } for p in TARGET_COLUMNS
        },
    }

    out_json = RESULTS_DIR / "data_efficiency_curve.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")

    # Also save raw .npz for downstream plotting
    ns = sorted(curve.keys())
    means = np.array([np.mean(curve[n]) if curve[n] else np.nan for n in ns])
    stds  = np.array([np.std(curve[n])  if curve[n] else np.nan for n in ns])
    np.savez(
        RESULTS_DIR / "data_efficiency_curve.npz",
        n=np.array(ns), mean=means, std=stds,
    )
    print(f"Wrote {RESULTS_DIR / 'data_efficiency_curve.npz'}")
    print("\n=== SUMMARY ===")
    for n, m, s in zip(ns, means, stds):
        print(f"  n={n:3d}: {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
