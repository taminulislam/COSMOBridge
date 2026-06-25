"""A3: regenerate surface_fp in LignoIL_A1 cache using newly-available DFT
point clouds.

Variant of scripts/regen_surface_fp_dft.py that targets
  cosmobridge_v5/data/LignoIL_A1/cached_{train,val,test}.npz
instead of the v4 cache. Writes:
  cosmobridge_v5/data/LignoIL_A1/cached_{train,val,test}_dft.npz

Every row whose SMILES has a point cloud under
  data/pipeline/point_clouds/{md5(smiles)[:12]}.npz
gets its surface_fp replaced with a freshly-computed PointNet embedding.
Rows without a matching point cloud keep their existing surface_fp
(usually zeros for pre-training rows).

This is the real "DFT expansion" step — turns the 465 completed Psi4
DFT runs into actual model features for the ~3900 ILThermo pre-training
rows that previously had zero surface_fp.
"""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.pointcloud.pointnet import PointNetEncoder  # noqa

CACHED_DIR = PROJECT_ROOT / "cosmobridge_v5" / "data" / "LignoIL_A1"
POINTCLOUD_DIR = PROJECT_ROOT / "data" / "pipeline" / "point_clouds"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "pointcloud" / "best_model.pt"
N_POINTS = 1024


def smi_hash(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


def load_encoder(device):
    encoder = PointNetEncoder(in_channels=7, feature_dim=256).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    enc_state = {k[len("pointnet."):]: v for k, v in sd.items()
                 if k.startswith("pointnet.")}
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    print(f"PointNet encoder: {len(enc_state)} params, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    encoder.eval()
    return encoder


def load_point_cloud(smiles):
    h = smi_hash(str(smiles))
    p = POINTCLOUD_DIR / f"{h}.npz"
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=True)
    pts = data["points"] if "points" in data.files else (
        data["surface"] if "surface" in data.files else None)
    if pts is None:
        return None
    pts = pts.astype(np.float32)
    if pts.shape[0] < N_POINTS:
        idx = np.random.choice(pts.shape[0], N_POINTS, replace=True)
    else:
        idx = np.random.choice(pts.shape[0], N_POINTS, replace=False)
    return pts[idx, :7]


def regen_split(split, encoder, device):
    src = CACHED_DIR / f"cached_{split}.npz"
    if not src.exists():
        print(f"{split}: source {src.name} missing; skip.")
        return
    dst = CACHED_DIR / f"cached_{split}_dft.npz"
    cached = np.load(src, allow_pickle=True)

    smiles = cached["smiles"]
    old_surface = cached["surface_fp"]
    n = len(smiles)

    new_surface = np.zeros_like(old_surface)
    hits = misses = 0
    pts_batch, batch_idx = [], []
    BATCH = 32
    with torch.no_grad():
        for i, s in enumerate(smiles):
            pts = load_point_cloud(s)
            if pts is None:
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

    out = {k: cached[k] for k in cached.files}
    out["surface_fp"] = new_surface.astype(np.float32)
    np.savez(dst, **out)
    nz_before = int(((old_surface != 0).any(axis=1)).sum())
    nz_after = int(((new_surface != 0).any(axis=1)).sum())
    print(f"{split}: hits={hits} misses={misses}  "
          f"non-zero rows: before={nz_before}/{n}  after={nz_after}/{n}  → {dst.name}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Cache dir: {CACHED_DIR}")
    print(f"Point clouds dir: {POINTCLOUD_DIR}")
    encoder = load_encoder(device)
    for split in ("train", "val", "test"):
        regen_split(split, encoder, device)
    print("\nDone. Outputs at cached_{train,val,test}_dft.npz in LignoIL_A1.")


if __name__ == "__main__":
    main()
