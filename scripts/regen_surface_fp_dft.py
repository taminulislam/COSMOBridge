"""Idea 1: Regenerate surface_fp features using DFT point clouds.

Loads the frozen PointNet encoder and runs it over the DFT-derived
point clouds (data/pipeline/point_clouds/{hash}.npz) to produce a new
256-D surface_fp vector for every sample in cached_{train,val,test}.npz.
Preserves all other fields (chemprop_fp, thermo_feat, targets, preds_*,
smiles, il_ids) unchanged.

Output files: cosmobridge_v4/data/cached_{split}_dft.npz

Usage:
    python scripts/regen_surface_fp_dft.py
"""

import hashlib
from pathlib import Path

import numpy as np
import torch

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.pointcloud.pointnet import PointNetEncoder


CACHED_DIR = PROJECT_ROOT / "cosmobridge_v4" / "data"
POINTCLOUD_DIR = PROJECT_ROOT / "data" / "pipeline" / "point_clouds"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "pointcloud" / "best_model.pt"
N_POINTS = 1024


def smi_hash(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


def load_encoder(device):
    encoder = PointNetEncoder(in_channels=7, feature_dim=256).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    enc_state = {
        k[len("pointnet."):]: v
        for k, v in sd.items()
        if k.startswith("pointnet.")
    }
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    print(f"PointNet encoder loaded: {len(enc_state)} params, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    encoder.eval()
    return encoder


def load_point_cloud(smiles):
    """Return a (1024, 7) point cloud or None if missing."""
    h = smi_hash(str(smiles))
    p = POINTCLOUD_DIR / f"{h}.npz"
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=True)
    if "points" in data.files:
        pts = data["points"]
    elif "surface" in data.files:
        pts = data["surface"]
    else:
        return None
    pts = pts.astype(np.float32)
    if pts.shape[0] < N_POINTS:
        idx = np.random.choice(pts.shape[0], N_POINTS, replace=True)
    else:
        idx = np.random.choice(pts.shape[0], N_POINTS, replace=False)
    return pts[idx, :7]  # (N_POINTS, 7)


def regen_split(split, encoder, device):
    src = CACHED_DIR / f"cached_{split}.npz"
    dst = CACHED_DIR / f"cached_{split}_dft.npz"
    cached = np.load(src, allow_pickle=True)

    smiles = cached["smiles"]
    old_surface = cached["surface_fp"]
    n = len(smiles)

    new_surface = np.zeros_like(old_surface)
    hits = 0
    misses = 0
    with torch.no_grad():
        pts_batch = []
        batch_idx = []
        BATCH = 32
        for i, s in enumerate(smiles):
            pts = load_point_cloud(s)
            if pts is None:
                # Fallback: keep old surface_fp feature
                new_surface[i] = old_surface[i]
                misses += 1
                continue
            pts_batch.append(pts)
            batch_idx.append(i)
            if len(pts_batch) == BATCH or i == n - 1:
                x = torch.from_numpy(np.stack(pts_batch)).to(device)
                feats = encoder(x).cpu().numpy()
                for j, idx in enumerate(batch_idx):
                    new_surface[idx] = feats[j]
                hits += len(pts_batch)
                pts_batch, batch_idx = [], []

    # Write new cache, preserving all other fields
    out = {k: cached[k] for k in cached.files}
    out["surface_fp"] = new_surface.astype(np.float32)
    np.savez(dst, **out)
    print(f"{split}: {hits}/{n} regenerated ({misses} fallback), saved to {dst.name}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    encoder = load_encoder(device)
    for split in ("train", "val", "test"):
        regen_split(split, encoder, device)
    print("\nDone. To use, update dataset.py to read cached_{split}_dft.npz,")
    print("or rename the _dft files over the originals.")


if __name__ == "__main__":
    main()
