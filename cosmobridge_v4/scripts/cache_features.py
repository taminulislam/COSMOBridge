"""Cache frozen features and path predictions for v4 training.

Runs expensive frozen models ONCE and saves:
  - chemprop_fp: Chemprop MPN fingerprint (300D)
  - surface_fp: PointNet fingerprint (256D)
  - thermo_feat: 25 thermodynamic features
  - targets: 7 property values (standardized)
  - preds_fusion: Path A predictions (frozen CP-GBH hybrid output)
  - preds_chemprop: Path B predictions (frozen Chemprop MPN+FFN output)
  - smiles, il_ids: metadata

Output: cosmobridge_v4/data/cached_{train,val,test}.npz
"""

import sys
import numpy as np
import pandas as pd
import torch
import subprocess
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_device
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.models.fusion.multimodal_pointcloud import MultimodalPointCloudModel
from scripts.train_chemprop_gbh_hybrid import ChempropGBHFusion
from scripts.train_pointcloud import PointCloudMultimodalDataset
from chemprop.utils import load_checkpoint as load_chemprop


def identity_collate(batch):
    return batch


def extract_chemprop_fingerprints(split):
    """Extract Chemprop MPN fingerprints via CLI."""
    out = tempfile.mktemp(suffix=".csv")
    subprocess.run([
        "chemprop_fingerprint",
        "--test_path", f"data/chemprop_tmp/{split}.csv",
        "--features_path", f"data/chemprop_tmp/{split}_features.csv",
        "--checkpoint_dir", "checkpoints/chemprop",
        "--fingerprint_type", "MPN",
        "--preds_path", out,
    ], capture_output=True, text=True, timeout=180, check=True)
    fp = pd.read_csv(out).select_dtypes(include=[np.number]).values.astype(np.float32)
    return fp


def main():
    device = get_device(load_config("configs/default.yaml"))
    print(f"Device: {device}")

    # --- Load frozen PointNet ---
    config = load_config("configs/default.yaml")
    pc_model = MultimodalPointCloudModel(config=config, pretrained_gnn_path=None)
    ckpt = torch.load("checkpoints/pointcloud/best_model.pt",
                      map_location=device, weights_only=False)
    pc_model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    pc_model.to(device).eval()
    print("Loaded PointNet")

    # --- Load frozen CP-GBH fusion ---
    # graph_dim is known from Chemprop: 300
    graph_dim = 300
    fusion_model = ChempropGBHFusion(
        graph_dim=graph_dim, surface_dim=256, thermo_dim=len(FEATURE_COLUMNS),
        fused_dim=256, rank=32, hyper_hidden=64, dropout=0.3,
    )
    fusion_model.load_state_dict(torch.load(
        "checkpoints/chemprop_gbh_hybrid/best.pt",
        map_location=device, weights_only=True,
    ))
    fusion_model.to(device).eval()
    print("Loaded CP-GBH fusion")

    # --- Load frozen Chemprop ---
    chemprop_model = load_chemprop("checkpoints/chemprop/fold_0/model_0/model.pt")
    chemprop_model.to(device).eval()
    print("Loaded Chemprop")

    # --- Process each split ---
    orig = Path("data/processed/splits")
    pc_dir = "data/pipeline/point_clouds"
    output_dir = Path("cosmobridge_v4/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        print(f"\n--- {split.upper()} ---")

        # PointNet features via dataset
        ds = PointCloudMultimodalDataset(
            str(orig / f"{split}.csv"), pc_dir, is_train=False,
        )
        df = pd.read_csv(orig / f"{split}.csv")

        surface_fps = []
        thermo_feats = []
        targets = []
        smiles_list = df["smiles"].tolist()

        with torch.no_grad():
            for items in DataLoader(ds, batch_size=32, shuffle=False, collate_fn=identity_collate):
                pc_batch = torch.stack([x["point_cloud"] for x in items]).to(device)
                surface_fps.append(pc_model.pointnet(pc_batch).cpu().numpy())
                thermo_feats.append(torch.stack([x["features"] for x in items]).numpy())
                targets.append(torch.stack([x["targets"] for x in items]).numpy())

        surface_fp = np.concatenate(surface_fps).astype(np.float32)
        thermo_feat = np.concatenate(thermo_feats).astype(np.float32)
        target = np.concatenate(targets).astype(np.float32)

        # Chemprop fingerprint (MPN output)
        chemprop_fp = extract_chemprop_fingerprints(split)
        print(f"  chemprop_fp: {chemprop_fp.shape}, surface_fp: {surface_fp.shape}, "
              f"thermo_feat: {thermo_feat.shape}, targets: {target.shape}")

        # Thermo features (first 5) for Chemprop prediction
        tr_feat = df[FEATURE_COLUMNS[:5]].values.astype(np.float32)

        # Path A predictions (fusion)
        preds_fusion = []
        with torch.no_grad():
            for i in range(0, len(chemprop_fp), 32):
                g = torch.from_numpy(chemprop_fp[i:i+32]).to(device)
                s = torch.from_numpy(surface_fp[i:i+32]).to(device)
                t = torch.from_numpy(thermo_feat[i:i+32]).to(device)
                p = fusion_model(g, s, t).cpu().numpy()
                preds_fusion.append(p)
        preds_fusion = np.concatenate(preds_fusion).astype(np.float32)

        # Path B predictions (Chemprop directly)
        preds_chemprop = []
        with torch.no_grad():
            for i in range(0, len(smiles_list), 32):
                smi_batch = [[s] for s in smiles_list[i:i+32]]
                tr_batch = tr_feat[i:i+32]
                p = chemprop_model(smi_batch, features_batch=tr_batch).cpu().numpy()
                preds_chemprop.append(p)
        preds_chemprop = np.concatenate(preds_chemprop).astype(np.float32)

        print(f"  preds_fusion: {preds_fusion.shape}, preds_chemprop: {preds_chemprop.shape}")

        # Save
        out_path = output_dir / f"cached_{split}.npz"
        np.savez(
            out_path,
            chemprop_fp=chemprop_fp,
            surface_fp=surface_fp,
            thermo_feat=thermo_feat,
            targets=target,
            preds_fusion=preds_fusion,
            preds_chemprop=preds_chemprop,
            smiles=np.array(smiles_list),
            il_ids=df["il_short_name"].values,
        )
        print(f"  Saved: {out_path}")

    print("\nDone. Next: run train_v4_router.py")


if __name__ == "__main__":
    main()
