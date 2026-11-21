"""B5: run the Chemprop D-MPNN baseline over the same 5 seeds used for
COSMOBridge / STILT / PointCloud, so the comparison in Table 3 is seed-matched.

Reuses the exact configuration of scripts/train_chemprop_baseline.py; only the
seed varies. Writes results/b5_chemprop_multiseed.json.
"""
import json, subprocess, sys, shutil
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/work/nvme/bgte/Dataset_Chemistry")
sys.path.insert(0, str(ROOT))
from src.training.metrics import compute_metrics

TARGETS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]
EXTRA = ["temperature", "x1", "inv_temperature", "temp_squared", "temp_cubed"]
SEEDS = [42, 123, 456, 789, 1024]
BIN = "/work/nvme/bgte/tislam6/envs/cosmo/bin/chemprop_train"


def prepare(tmp):
    tmp.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        df = pd.read_csv(ROOT / f"data/processed/splits/{split}.csv")
        out = pd.DataFrame({"smiles": df["smiles"]})
        for f in EXTRA:
            if f in df.columns:
                out[f] = df[f]
        for t in TARGETS:
            out[t] = df[t]
        out.to_csv(tmp / f"{split}.csv", index=False)
        df[EXTRA].to_csv(tmp / f"{split}_features.csv", index=False)
    return tmp


def run_seed(tmp, seed):
    ckpt = ROOT / f"checkpoints/b5_chemprop_seed{seed}"
    if ckpt.exists():
        shutil.rmtree(ckpt)
    ckpt.mkdir(parents=True)
    cmd = [BIN,
           "--data_path", str(tmp / "train.csv"),
           "--separate_val_path", str(tmp / "val.csv"),
           "--separate_test_path", str(tmp / "test.csv"),
           "--features_path", str(tmp / "train_features.csv"),
           "--separate_val_features_path", str(tmp / "val_features.csv"),
           "--separate_test_features_path", str(tmp / "test_features.csv"),
           "--save_dir", str(ckpt),
           "--dataset_type", "regression",
           "--smiles_columns", "smiles",
           "--target_columns", *TARGETS,
           "--epochs", "100", "--batch_size", "32",
           "--hidden_size", "300", "--depth", "3",
           "--ffn_num_layers", "2", "--ffn_hidden_size", "300",
           "--dropout", "0.2",
           "--metric", "rmse", "--extra_metrics", "r2", "mae",
           "--seed", str(seed), "--num_folds", "1",
           "--save_preds", "--no_cuda", "--quiet"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        print(f"  seed {seed} FAILED\n{r.stderr[-3000:]}")
        return None
    pred_path = next(ckpt.rglob("test_preds.csv"), None)
    if pred_path is not None:
        preds = pd.read_csv(pred_path)[TARGETS].values
        targets = pd.read_csv(tmp / "test.csv")[TARGETS].values
        return {k: float(v) for k, v in compute_metrics(preds, targets).items()}
    # chemprop >=1.6 writes per-task scores even without --save_preds
    score_path = next(ckpt.rglob("test_scores.json"), None)
    if score_path is None:
        print(f"  seed {seed}: no predictions and no test_scores.json")
        return None
    sc = json.load(open(score_path))
    m = {f"{t}_r2": float(v) for t, v in zip(TARGETS, sc["r2"])}
    m.update({f"{t}_rmse": float(v) for t, v in zip(TARGETS, sc["rmse"])})
    m.update({f"{t}_mae": float(v) for t, v in zip(TARGETS, sc["mae"])})
    m["avg_r2"] = float(np.mean(sc["r2"]))
    m["avg_rmse"] = float(np.mean(sc["rmse"]))
    m["avg_mae"] = float(np.mean(sc["mae"]))
    return m


def main():
    tmp = prepare(ROOT / "data/chemprop_b5_tmp")
    per_seed = {}
    for s in SEEDS:
        print(f"--- seed {s} ---", flush=True)
        m = run_seed(tmp, s)
        if m:
            per_seed[s] = m
            print(f"  avg R2 = {m['avg_r2']:.4f}", flush=True)
    if not per_seed:
        print("no seeds succeeded"); return
    summary = {"seeds": SEEDS, "per_seed": per_seed, "aggregate": {}}
    for p in TARGETS + ["avg"]:
        key = f"{p}_r2"
        vals = [per_seed[s][key] for s in per_seed]
        summary["aggregate"][key] = {"mean": float(np.mean(vals)),
                                     "std": float(np.std(vals, ddof=0)),
                                     "n": len(vals)}
    out = ROOT / "results/b5_chemprop_multiseed.json"
    json.dump(summary, open(out, "w"), indent=2)
    print("\n=== Chemprop, 5 seeds ===")
    for p in TARGETS + ["avg"]:
        a = summary["aggregate"][f"{p}_r2"]
        print(f"  {p:9s} R2 = {a['mean']:.4f} +/- {a['std']:.4f}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
