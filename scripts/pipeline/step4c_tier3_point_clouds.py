"""Step 4c (tier3 variant): extract point clouds from TIER3 DFT surfaces.

The original step4c_extract_point_clouds.py reads geometry_status.csv (which
tracks the original ILThermo DFT campaign). Our tier-3 campaign uses a
different registry: tier3_compounds.csv. This script iterates that list,
loads TIER3_XXX_pair.npz from dft_surface/, samples a 1024-point cloud per
SMILES, and appends to data/pipeline/point_clouds/index.csv. Rows that
already exist in the index are skipped (idempotent).

Run this AFTER the tier3 DFT array has produced the _pair.npz files.
"""
from __future__ import annotations
import csv, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PIPELINE = PROJECT_ROOT / "data" / "pipeline"
DFT_SURF = PIPELINE / "dft_surface"
PC_DIR = PIPELINE / "point_clouds"
IDX_CSV = PC_DIR / "index.csv"


# Inlined from step4c_extract_point_clouds.py to avoid the step4b import chain
# (skimage/rdkit) which isn't available in every env.
def smiles_to_hash(smiles: str) -> str:
    return hashlib.md5(smiles.encode()).hexdigest()[:12]


def _farthest_point_sample(pts, n_samples):
    n = len(pts)
    selected = np.zeros(n_samples, dtype=np.int64)
    distances = np.full(n, np.inf)
    selected[0] = np.random.randint(n)
    for i in range(1, n_samples):
        last = selected[i - 1]
        dist_to_last = np.sum((pts - pts[last]) ** 2, axis=1)
        distances = np.minimum(distances, dist_to_last)
        selected[i] = np.argmax(distances)
    return selected


def _sample_point_cloud(verts, normals, esp, n_points=1024):
    n = len(verts)
    if n >= n_points:
        idx = _farthest_point_sample(verts, n_points)
    else:
        extra = n_points - n
        extra_idx = np.random.choice(n, size=extra, replace=True)
        idx = np.concatenate([np.arange(n), extra_idx])
    return np.column_stack([verts[idx], normals[idx], esp[idx, None]]).astype(np.float32)


def load_dft_surface(npz_path, n_points=1024):
    """Load a Psi4 dft_surface .npz and return a centered, scaled (N,7) cloud."""
    data = np.load(npz_path, allow_pickle=True)
    surface = data["surface"]  # (M, 7)
    verts = surface[:, :3]
    normals = surface[:, 3:6]
    esp = surface[:, 6]
    center = verts.mean(axis=0)
    verts = verts - center
    scale = np.max(np.linalg.norm(verts, axis=1)) + 1e-8
    verts = verts / scale
    return _sample_point_cloud(verts, normals, esp, n_points)


def main():
    tier3_csv = PIPELINE / "tier3_compounds.csv"
    if not tier3_csv.exists():
        print(f"Missing {tier3_csv}")
        return 1

    df = pd.read_csv(tier3_csv)
    PC_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing index
    existing = {}
    if IDX_CSV.exists():
        with open(IDX_CSV) as f:
            for row in csv.DictReader(f):
                existing[row["smiles"]] = row

    n_ok = n_missing_pair = n_existing = n_failed = 0
    for _, row in df.iterrows():
        cid = row["compound_id"]          # TIER3_xxx
        smiles = row["smiles"]
        pair_npz = DFT_SURF / f"{cid}_pair.npz"
        h = smiles_to_hash(smiles)
        out_pc = PC_DIR / f"{h}.npz"

        if out_pc.exists():
            existing[smiles] = {"smiles": smiles, "filename": f"{h}.npz",
                                 "il_name": row.get("name", cid)}
            n_existing += 1
            continue
        if not pair_npz.exists():
            print(f"  [skip] {cid}: pair DFT missing ({pair_npz.name})")
            n_missing_pair += 1
            continue
        try:
            points = load_dft_surface(pair_npz, n_points=1024)
        except Exception as e:
            print(f"  [fail] {cid}: {e}")
            n_failed += 1
            continue
        np.savez_compressed(out_pc, points=points)
        existing[smiles] = {"smiles": smiles, "filename": f"{h}.npz",
                             "il_name": row.get("name", cid)}
        n_ok += 1
        if n_ok % 25 == 0:
            print(f"  [progress] {n_ok} new point clouds written")

    # Rewrite the full index (sorted for stability)
    with open(IDX_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["smiles", "filename", "il_name"])
        w.writeheader()
        for smi in sorted(existing):
            w.writerow(existing[smi])

    print(f"\nTIER3 point-cloud extraction complete:")
    print(f"  new written        : {n_ok}")
    print(f"  already on disk    : {n_existing}")
    print(f"  pair DFT missing   : {n_missing_pair}")
    print(f"  conversion failed  : {n_failed}")
    print(f"Index now has        : {len(existing)} total SMILES")


if __name__ == "__main__":
    sys.exit(main() or 0)
