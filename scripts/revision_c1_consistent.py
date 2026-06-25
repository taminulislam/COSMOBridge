"""C1 (reviewer 2nd round): resolve the chemprop-baseline evaluation inconsistency.

The published COSMOBridge blends its fusion path against a Chemprop path called as
  chemprop_model(smi, features_batch=tr)
with RAW (unscaled) thermo features, even though the checkpoint was trained with
features_scaling=True. That direct call bypasses the saved features_scaler, giving a
Chemprop path that scores avg R^2 = 0.804 on test, while the *correct* standalone
Chemprop baseline (chemprop_predict CLI, scaler applied) is avg R^2 = 0.770 -- the
number the paper's Table reports. The +0.031 'surpass on 5/7' headline therefore
compares two different evaluations of the same checkpoint.

This script recomputes everything CONSISTENTLY: it loads the trained features_scaler,
applies it to the thermo features so the Chemprop path is evaluated the same way it was
trained, re-fits the 7 per-property gates (5 seeds) against that correctly-scaled path,
and re-evaluates COSMOBridge. It saves per-row test predictions for:
  chemprop_scaled (correct baseline)  -- expected ~0.770
  chemprop_raw    (old in-pipeline)   -- expected ~0.804
  fusion          (GBH fusion path)
  seed_preds      (COSMOBridge re-fit on the scaled chemprop path, 5 seeds)
so the per-property bootstrap (review C1) can be run on a self-consistent comparison.

Run via jobs/revision_c1_consistent.sh (cosmo env, 1 GPU).
"""
import sys, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
from src.utils.config import load_config, get_device
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from scripts.train_chemprop_gbh_hybrid import ChempropGBHFusion
import scripts.train_pointcloud as tp
from scripts.train_pointcloud import PointCloudMultimodalDataset
from chemprop.utils import load_checkpoint as load_chemprop, load_scalers
from scripts.cosmobridge_multiseed import train_gates_one_seed, identity_collate

PC_DIR = "data/pipeline/point_clouds_dft_apr11"
CKPT = "checkpoints/chemprop/fold_0/model_0/model.pt"

tp.PointCloudMultimodalDataset._load_point_cloud = lambda self, smiles: torch.zeros(1)


def r2_cols(y, p):
    out = []
    for i in range(y.shape[1]):
        ss_res = np.sum((y[:, i] - p[:, i]) ** 2)
        ss_tot = np.sum((y[:, i] - y[:, i].mean()) ** 2)
        out.append(1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0)
    return np.array(out)


def extract_features(device, features_scaler):
    """Identical to revision_savepreds, but stores BOTH raw and scaler-transformed
    thermo features so we can build a correctly-scaled Chemprop path."""
    orig = Path("data/processed/splits")
    data = {}
    for split in ["train", "val", "test"]:
        ds = PointCloudMultimodalDataset(str(orig / f"{split}.csv"), PC_DIR, is_train=False)
        df = pd.read_csv(orig / f"{split}.csv")
        tf, tgt = [], []
        for items in DataLoader(ds, batch_size=32, shuffle=False, collate_fn=identity_collate):
            tf.append(torch.stack([x["features"] for x in items]).numpy())
            tgt.append(torch.stack([x["targets"] for x in items]).numpy())
        s = np.load(f"data/chemprop_cosmo/{split}_pointnet_feats.npy").astype(np.float32)
        out = tempfile.mktemp(suffix=".csv")
        subprocess.run(["chemprop_fingerprint", "--test_path", f"data/chemprop_tmp/{split}.csv",
                        "--features_path", f"data/chemprop_tmp/{split}_features.csv",
                        "--checkpoint_dir", "checkpoints/chemprop", "--fingerprint_type", "MPN",
                        "--preds_path", out], capture_output=True, text=True, timeout=300)
        gf = pd.read_csv(out).select_dtypes(include=[np.number]).values.astype(np.float32)
        t = np.concatenate(tf); y = np.concatenate(tgt)
        raw_tr = df[FEATURE_COLUMNS[:5]].values.astype(np.float32)
        scaled_tr = features_scaler.transform(raw_tr).astype(np.float32) if features_scaler is not None else raw_tr
        assert s.shape[0] == t.shape[0] == len(df)
        data[split] = {"g": gf, "s": s, "t": t, "y": y, "smi": df["smiles"].tolist(),
                       "tr": scaled_tr, "tr_raw": raw_tr}
    return data


def chemprop_path(data, split, chemprop_model, use_scaled):
    smi = [[x] for x in data[split]["smi"]]
    key = "tr" if use_scaled else "tr_raw"
    tr = [data[split][key][i] for i in range(len(data[split]["smi"]))]
    with torch.no_grad():
        return chemprop_model(smi, features_batch=tr).cpu().numpy()


def main():
    device = get_device(load_config("configs/default.yaml"))
    print(f"Device: {device}")

    scaler, features_scaler, *_ = load_scalers(CKPT)
    print(f"features_scaler loaded: {features_scaler is not None}")

    print("Extracting features...")
    data = extract_features(device, features_scaler)
    graph_dim = data["train"]["g"].shape[1]
    y = data["test"]["y"]

    fusion_model = ChempropGBHFusion(graph_dim=graph_dim, surface_dim=256,
                                     thermo_dim=len(FEATURE_COLUMNS), fused_dim=256,
                                     rank=32, hyper_hidden=64, dropout=0.3)
    fusion_model.load_state_dict(torch.load("checkpoints/chemprop_gbh_hybrid/best.pt",
                                            map_location=device, weights_only=True))
    fusion_model.to(device).eval()
    chemprop_model = load_chemprop(CKPT); chemprop_model.to(device).eval()

    # Fusion path test preds
    g = torch.tensor(data["test"]["g"], dtype=torch.float32, device=device)
    s = torch.tensor(data["test"]["s"], dtype=torch.float32, device=device)
    t = torch.tensor(data["test"]["t"], dtype=torch.float32, device=device)
    with torch.no_grad():
        pf = fusion_model(g, s, t).cpu().numpy()

    pc_scaled = chemprop_path(data, "test", chemprop_model, use_scaled=True)
    pc_raw    = chemprop_path(data, "test", chemprop_model, use_scaled=False)

    print("\n=== Standalone Chemprop path, per-property R^2 ===")
    cols = TARGET_COLUMNS
    rs = r2_cols(y, pc_scaled); rr = r2_cols(y, pc_raw)
    for i, c in enumerate(cols):
        print(f"  {c:8s}  scaled={rs[i]:.3f}   raw={rr[i]:.3f}")
    print(f"  {'AVG':8s}  scaled={rs.mean():.4f}   raw={rr.mean():.4f}")
    print("  (scaled should reproduce the paper's 0.770 baseline; raw is the in-pipeline 0.804)")

    # Re-fit gates against the CORRECTLY-SCALED chemprop path (data['tr'] is scaled)
    seeds = [42, 123, 456, 789, 1024]
    all_gates, all_preds = [], []
    print("\n=== Re-fitting COSMOBridge gates on the scaled chemprop path ===")
    for sd in seeds:
        metrics, gates = train_gates_one_seed(sd, data, fusion_model, chemprop_model, device)
        a = np.asarray(gates, dtype=np.float64).reshape(1, -1)
        all_gates.append(np.asarray(gates, dtype=np.float64))
        all_preds.append(a * pf + (1.0 - a) * pc_scaled)
        print(f"  seed {sd}: avg_r2={metrics['avg_r2']:.4f}  gates={np.round(gates,3).tolist()}")

    seed_preds = np.stack(all_preds).astype(np.float32)
    cb = seed_preds.mean(0)
    rc = r2_cols(y, cb)
    print("\n=== COSMOBridge (re-fit, scaled chemprop path) per-property R^2 ===")
    for i, c in enumerate(cols):
        print(f"  {c:8s}  CB={rc[i]:.3f}  chemprop_scaled={rs[i]:.3f}  d={rc[i]-rs[i]:+.3f}  {'WIN' if rc[i]>rs[i] else 'lose'}")
    print(f"  {'AVG':8s}  CB={rc.mean():.4f}  chemprop_scaled={rs.mean():.4f}  d={rc.mean()-rs.mean():+.4f}")
    print(f"  COSMOBridge wins on {int((rc>rs).sum())} of 7 vs the correctly-scaled Chemprop baseline")

    out = project_root / "results" / "cosmobridge_c1_consistent.npz"
    np.savez(out, seed_preds=seed_preds, targets=y.astype(np.float32),
             fusion_preds=pf.astype(np.float32),
             chemprop_scaled_preds=pc_scaled.astype(np.float32),
             chemprop_raw_preds=pc_raw.astype(np.float32),
             seed_gates=np.stack(all_gates).astype(np.float32),
             seeds=np.array(seeds), target_cols=np.array(TARGET_COLUMNS, dtype=object))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
