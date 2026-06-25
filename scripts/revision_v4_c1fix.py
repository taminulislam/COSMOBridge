"""Corrected COSMOBridge-v4: rerun the published triple-path router with the
chemprop-path UNIT BUG fixed.

v4's cached preds_chemprop are in standardized target space (manual chemprop forward
without inverse_transform), while preds_fusion, preds_atom_surface, and targets are in
original units. We load the published cache, inverse-transform preds_chemprop to original
units (so all three paths + targets are consistent), then reuse the published
train_one_seed (cosmobridge_v4_triple model, identical config) for 10 seeds. We report
seed-mean and the 10-seed ensemble, with a per-property bootstrap vs official chemprop.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cosmobridge_v4"))

from chemprop.utils import load_scalers
from src.training.metrics import compute_metrics
from src.data.preprocessing import TARGET_COLUMNS
from cosmobridge_v4.scripts.train_v4_triple import load_cached, train_one_seed

CKPT = "checkpoints/chemprop/fold_0/model_0/model.pt"
CONFIG = {"hidden": 64, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-3,
          "batch_size": 32, "epochs": 300, "patience": 40,
          "anchor_init": 0.1, "anchor_final": 0.01}


def fix_chemprop(data, scaler):
    """Inverse-transform the standardized chemprop path into original units."""
    pc = data["preds_chemprop"].numpy()
    pc_inv = np.asarray(scaler.inverse_transform(pc), dtype=np.float32)
    data["preds_chemprop"] = torch.from_numpy(pc_inv).float()
    return data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    scaler = load_scalers(CKPT)[0]

    train_data = fix_chemprop(load_cached("train"), scaler)
    val_data = fix_chemprop(load_cached("val"), scaler)
    test_data = fix_chemprop(load_cached("test"), scaler)
    y = test_data["targets"].numpy()

    # sanity: corrected chemprop path avg R^2 should be ~0.770
    pc = test_data["preds_chemprop"].numpy()
    def r2(yt, yp): return 1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    print("corrected chemprop-path avg R2 (expect ~0.770):",
          round(np.mean([r2(y[:, i], pc[:, i]) for i in range(7)]), 4))

    seeds = list(range(10))
    all_metrics, seed_preds, all_weights = [], [], []
    for seed in seeds:
        m, preds, targets, weights, _ = train_one_seed(seed, train_data, val_data, test_data, device, CONFIG)
        all_metrics.append(m); seed_preds.append(preds); all_weights.append(weights.mean(axis=0))
        print(f"  seed {seed}: avg_r2={m['avg_r2']:.4f}")

    mean_w = np.stack(all_weights).mean(0)  # (7,3): Fusion, Chemprop, AtomSurface
    print("\n  Corrected routing weights (Fusion / Chemprop / AtomSurface):")
    for i, p in enumerate(TARGET_COLUMNS):
        print(f"    {p:8s} {mean_w[i,0]:.3f} {mean_w[i,1]:.3f} {mean_w[i,2]:.3f}")

    seed_preds = np.stack(seed_preds)            # (10, N, 7)
    ens = seed_preds.mean(0)
    ens_metrics = compute_metrics(ens, y)
    avgs = [m["avg_r2"] for m in all_metrics]

    print(f"\n{'='*60}")
    print("CORRECTED v4 (unit bug fixed)")
    print(f"{'='*60}")
    print(f"  seed-mean avg R2 : {np.mean(avgs):.4f} ± {np.std(avgs):.4f}  (was 0.8182 buggy)")
    print(f"  ENSEMBLE avg R2  : {ens_metrics['avg_r2']:.4f}")
    print(f"\n  {'prop':8s} {'v4_ens':>8s}")
    for i, p in enumerate(TARGET_COLUMNS):
        print(f"  {p:8s} {ens_metrics[f'{p}_r2']:8.3f}")

    out = project_root / "results" / "cosmobridge_v4_c1fix.npz"
    np.savez(out, seed_preds=seed_preds.astype(np.float32), targets=y.astype(np.float32),
             ens_preds=ens.astype(np.float32), seeds=np.array(seeds),
             target_cols=np.array(TARGET_COLUMNS, dtype=object))
    json.dump({"seed_mean_avg": float(np.mean(avgs)), "seed_std_avg": float(np.std(avgs)),
               "ensemble_avg": float(ens_metrics["avg_r2"]),
               "ensemble_per_prop": {p: float(ens_metrics[f"{p}_r2"]) for p in TARGET_COLUMNS},
               "per_seed_avg": [float(a) for a in avgs],
               "routing_weights": {p: {"fusion": float(mean_w[i,0]), "chemprop": float(mean_w[i,1]),
                                        "atom_surface": float(mean_w[i,2])} for i, p in enumerate(TARGET_COLUMNS)}},
              open(project_root / "results" / "cosmobridge_v4_c1fix.json", "w"), indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
