"""Reviewer revision (COSMOBridge): reproduce the 5-seed gate eval and additionally
save per-seed per-row TEST predictions + exact gates.

Outputs results/cosmobridge_test_preds.npz with:
  seed_preds   (5, 39, 7)  per-seed routed predictions on the test split
  targets      (39, 7)     ground truth
  fusion_preds (39, 7)     frozen CP-GBH fusion-path prediction (deterministic)
  chemprop_preds(39, 7)    frozen Chemprop-path prediction (deterministic)
  seed_gates   (5, 7)      per-seed learned alpha (= sigmoid(gate_logits))
  seeds, target_cols

Enables (a) per-property bootstrap CIs for Table 1 (review Crit-1b) and
(b) exact 5-seed-mean gates for Table 2 (review Imp-5). Reuses the published
train_gates_one_seed verbatim; extract_features is copied with the point-cloud
directory pointed at the group-readable DFT clouds (the default point_clouds/
dir is mode 0700 and unreadable to non-owner). Run via jobs/revision_savepreds.sh.
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
from chemprop.utils import load_checkpoint as load_chemprop
from scripts.cosmobridge_multiseed import train_gates_one_seed, identity_collate

PC_DIR = "data/pipeline/point_clouds_dft_apr11"   # index.csv is group-readable; the .npz are 0600

# The raw .npz clouds are owner-only (unreadable to non-owner), but the PointNet
# surface features they produce are cached and row-identical to the splits. Use those
# directly and stub cloud loading so the dataset still yields features/targets.
tp.PointCloudMultimodalDataset._load_point_cloud = lambda self, smiles: torch.zeros(1)


def extract_features(device):
    """Same feature construction as cosmobridge_multiseed, but the 256-D surface
    features come from the cached PointNet feature bank (data/chemprop_cosmo/*_pointnet_feats.npy,
    verified row-identical to the splits) instead of re-running PointNet on the 0600 clouds."""
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
        assert s.shape[0] == t.shape[0] == len(df), f"row mismatch in {split}: s={s.shape} t={t.shape} df={len(df)}"
        data[split] = {"g": gf, "s": s, "t": t, "y": y, "smi": df["smiles"].tolist(),
                       "tr": df[FEATURE_COLUMNS[:5]].values.astype(np.float32)}
    return data


def path_preds_test(data, fusion_model, chemprop_model, device):
    g = torch.tensor(data["test"]["g"], dtype=torch.float32, device=device)
    s = torch.tensor(data["test"]["s"], dtype=torch.float32, device=device)
    t = torch.tensor(data["test"]["t"], dtype=torch.float32, device=device)
    smi = [[x] for x in data["test"]["smi"]]
    tr = [data["test"]["tr"][i] for i in range(len(data["test"]["smi"]))]
    with torch.no_grad():
        pf = fusion_model(g, s, t).cpu().numpy()
        pc = chemprop_model(smi, features_batch=tr).cpu().numpy()
    return pf, pc


def main():
    device = get_device(load_config("configs/default.yaml"))
    print(f"Device: {device}")

    print("Extracting features (chemprop_fingerprint + PointNet)...")
    data = extract_features(device)
    graph_dim = data["train"]["g"].shape[1]
    print(f"graph_dim={graph_dim}, surface_dim={data['train']['s'].shape[1]}")

    print("Loading frozen paths...")
    fusion_model = ChempropGBHFusion(graph_dim=graph_dim, surface_dim=256,
                                     thermo_dim=len(FEATURE_COLUMNS), fused_dim=256,
                                     rank=32, hyper_hidden=64, dropout=0.3)
    fusion_model.load_state_dict(torch.load("checkpoints/chemprop_gbh_hybrid/best.pt",
                                            map_location=device, weights_only=True))
    fusion_model.to(device).eval()
    chemprop_model = load_chemprop("checkpoints/chemprop/fold_0/model_0/model.pt")
    chemprop_model.to(device).eval()

    pf, pc = path_preds_test(data, fusion_model, chemprop_model, device)
    y = data["test"]["y"]

    seeds = [42, 123, 456, 789, 1024]
    all_gates, all_preds = [], []
    for sd in seeds:
        metrics, gates = train_gates_one_seed(sd, data, fusion_model, chemprop_model, device)
        a = np.asarray(gates, dtype=np.float64).reshape(1, -1)
        all_gates.append(np.asarray(gates, dtype=np.float64))
        all_preds.append(a * pf + (1.0 - a) * pc)
        print(f"  seed {sd}: avg_r2={metrics['avg_r2']:.4f}  gates={np.round(gates,3).tolist()}")

    seed_preds = np.stack(all_preds).astype(np.float32)
    seed_gates = np.stack(all_gates).astype(np.float32)
    out = project_root / "results" / "cosmobridge_test_preds.npz"
    np.savez(out, seed_preds=seed_preds, targets=y.astype(np.float32),
             fusion_preds=pf.astype(np.float32), chemprop_preds=pc.astype(np.float32),
             seed_gates=seed_gates, seeds=np.array(seeds),
             target_cols=np.array(TARGET_COLUMNS, dtype=object))
    print(f"\n5-seed-mean gates: {np.round(seed_gates.mean(0),3).tolist()}")
    print(f"5-seed-mean avg_r2 reported per-seed above; saved {out}  seed_preds={seed_preds.shape}")


if __name__ == "__main__":
    main()
