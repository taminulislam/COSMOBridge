"""C1 definitive (faithful gates): rebuild COSMOBridge on OFFICIAL chemprop predictions
using the PUBLISHED gate-training protocol verbatim.

The published COSMOBridge blends fusion + a chemprop path called as
chemprop_model(smi, features_batch=tr) -- a manual forward that disagrees with
chemprop's official chemprop_predict CLI by up to 0.37 per prediction on the energy
properties (same checkpoint, identical targets); in-pipeline avg R^2 = 0.804 vs the
official CLI baseline 0.770. COSMOBridge (0.801) is built on the manual path.

Here we swap ONLY the chemprop path for the official CLI predictions and reuse the
published train_gates_one_seed (gradient gate fit, train MSE, val early-stop) so the
gate methodology is identical to the paper -- isolating the effect of using the
correct chemprop. A lookup wrapper returns the precomputed official prediction for
each (smiles, thermo-feature) row, so train_gates_one_seed runs unchanged.
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
from scripts.cosmobridge_multiseed import train_gates_one_seed, identity_collate

PC_DIR = "data/pipeline/point_clouds_dft_apr11"
tp.PointCloudMultimodalDataset._load_point_cloud = lambda self, smiles: torch.zeros(1)


def r2(yt, yp):
    return 1 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2)


def key_of(smiles, tr_row):
    return (smiles, tuple(np.round(np.asarray(tr_row, dtype=np.float64), 5).tolist()))


class OfficialChempropLookup:
    """Drop-in for the loaded chemprop model: returns precomputed official CLI
    predictions for each (smiles, thermo) row passed in the batch."""
    def __init__(self, table, device):
        self.table = table; self.device = device
    def to(self, *a, **k): return self
    def eval(self, *a, **k): return self
    def __call__(self, smi, features_batch=None):
        rows = []
        for j, entry in enumerate(smi):
            s = entry[0] if isinstance(entry, (list, tuple)) else entry
            tr_row = features_batch[j]
            rows.append(self.table[key_of(s, tr_row)])
        return torch.tensor(np.stack(rows), dtype=torch.float32, device=self.device)


def cli_chemprop(split):
    out = tempfile.mktemp(suffix=".csv")
    subprocess.run(["chemprop_predict", "--test_path", f"data/chemprop_tmp/{split}.csv",
                    "--features_path", f"data/chemprop_tmp/{split}_features.csv",
                    "--checkpoint_dir", "checkpoints/chemprop",
                    "--preds_path", out, "--num_workers", "0"],
                   capture_output=True, text=True, timeout=600)
    return pd.read_csv(out)[TARGET_COLUMNS].values.astype(np.float32)


def build_data(device):
    orig = Path("data/processed/splits")
    data, table = {}, {}
    for split in ["train", "val", "test"]:
        ds = PointCloudMultimodalDataset(str(orig / f"{split}.csv"), PC_DIR, is_train=False)
        df = pd.read_csv(orig / f"{split}.csv")
        gf_path = tempfile.mktemp(suffix=".csv")
        subprocess.run(["chemprop_fingerprint", "--test_path", f"data/chemprop_tmp/{split}.csv",
                        "--features_path", f"data/chemprop_tmp/{split}_features.csv",
                        "--checkpoint_dir", "checkpoints/chemprop", "--fingerprint_type", "MPN",
                        "--preds_path", gf_path], capture_output=True, text=True, timeout=300)
        gf = pd.read_csv(gf_path).select_dtypes(include=[np.number]).values.astype(np.float32)
        s = np.load(f"data/chemprop_cosmo/{split}_pointnet_feats.npy").astype(np.float32)
        tf, tgt = [], []
        for items in DataLoader(ds, batch_size=32, shuffle=False, collate_fn=identity_collate):
            tf.append(torch.stack([x["features"] for x in items]).numpy())
            tgt.append(torch.stack([x["targets"] for x in items]).numpy())
        t = np.concatenate(tf); y = np.concatenate(tgt)
        tr = df[FEATURE_COLUMNS[:5]].values.astype(np.float32)
        data[split] = {"g": gf, "s": s, "t": t, "y": y, "smi": df["smiles"].tolist(), "tr": tr}
        official = cli_chemprop(split)
        for i, smiles in enumerate(df["smiles"].tolist()):
            table[key_of(smiles, tr[i])] = official[i]
        data[split]["_official"] = official
    return data, table


def main():
    device = get_device(load_config("configs/default.yaml"))
    print(f"Device: {device}")
    data, table = build_data(device)
    y = data["test"]["y"]; cols = TARGET_COLUMNS

    fusion_model = ChempropGBHFusion(graph_dim=data["train"]["g"].shape[1], surface_dim=256,
                                     thermo_dim=len(FEATURE_COLUMNS), fused_dim=256,
                                     rank=32, hyper_hidden=64, dropout=0.3)
    fusion_model.load_state_dict(torch.load("checkpoints/chemprop_gbh_hybrid/best.pt",
                                            map_location=device, weights_only=True))
    fusion_model.to(device).eval()

    chemprop_lookup = OfficialChempropLookup(table, device)

    # fusion path on test (for saved preds)
    g = torch.tensor(data["test"]["g"], dtype=torch.float32, device=device)
    s = torch.tensor(data["test"]["s"], dtype=torch.float32, device=device)
    t = torch.tensor(data["test"]["t"], dtype=torch.float32, device=device)
    with torch.no_grad():
        pf = fusion_model(g, s, t).cpu().numpy()
    pc = data["test"]["_official"]

    rc_chemprop = np.array([r2(y[:, i], pc[:, i]) for i in range(7)])
    print("\n=== Official chemprop (CLI) test per-property R^2 ===")
    for i, c in enumerate(cols): print(f"  {c:8s} {rc_chemprop[i]:.3f}")
    print(f"  AVG {rc_chemprop.mean():.4f}  (expect ~0.770)")

    seeds = [42, 123, 456, 789, 1024]
    all_preds, all_gates = [], []
    print("\n=== Re-fitting gates (published protocol) on OFFICIAL chemprop ===")
    for sd in seeds:
        metrics, gates = train_gates_one_seed(sd, data, fusion_model, chemprop_lookup, device)
        a = np.asarray(gates, dtype=np.float64).reshape(1, -1)
        all_gates.append(np.asarray(gates, dtype=np.float64))
        all_preds.append(a * pf + (1.0 - a) * pc)
        print(f"  seed {sd}: avg_r2={metrics['avg_r2']:.4f}  gates={np.round(gates,3).tolist()}")

    seed_preds = np.stack(all_preds).astype(np.float32)
    cb = seed_preds.mean(0)
    rc_cb = np.array([r2(y[:, i], cb[:, i]) for i in range(7)])
    print("\n=== COSMOBridge (faithful gates, OFFICIAL chemprop) vs official chemprop ===")
    wins = 0
    for i, c in enumerate(cols):
        w = "WIN" if rc_cb[i] > rc_chemprop[i] else "lose"; wins += rc_cb[i] > rc_chemprop[i]
        print(f"  {c:8s} CB={rc_cb[i]:.3f}  chemprop={rc_chemprop[i]:.3f}  d={rc_cb[i]-rc_chemprop[i]:+.3f}  {w}")
    print(f"  AVG      CB={rc_cb.mean():.4f}  chemprop={rc_chemprop.mean():.4f}  d={rc_cb.mean()-rc_chemprop.mean():+.4f}")
    print(f"  5-seed-mean gates: {np.round(np.stack(all_gates).mean(0),3).tolist()}")
    print(f"  COSMOBridge wins on {int(wins)} of 7 vs the OFFICIAL chemprop baseline")

    out = project_root / "results" / "cosmobridge_c1_official.npz"
    np.savez(out, seed_preds=seed_preds, targets=y.astype(np.float32),
             chemprop_official_preds=pc.astype(np.float32), fusion_preds=pf.astype(np.float32),
             seed_gates=np.stack(all_gates).astype(np.float32),
             seeds=np.array(seeds), target_cols=np.array(TARGET_COLUMNS, dtype=object))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
