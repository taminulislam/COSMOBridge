"""Step 4c: Extract point clouds from COSMO isosurface meshes.

Samples fixed-size point clouds (vertices + normals + ESP) from the
marching-cubes mesh for each IL. Saved as .npz files for fast loading
during training.

Input:  data/processed/il_data_raw.csv  (original 28 ILs)
        data/augmented/ilthermo_data.csv (ILThermo ILs)
Output: data/pipeline/point_clouds/{smiles_hash}.npz
        data/pipeline/point_clouds/index.csv  (smiles -> filename mapping)

Each .npz contains:
  points: (N, 7) array — [x, y, z, nx, ny, nz, esp]
"""

import csv
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

from step4b_surface_descriptors import (
    prepare_molecule, get_mol_data, separate_ions,
    build_isosurface, compute_surface_esp,
)


DFT_SURFACE_DIR = Path("data/pipeline/dft_surface")
GEOMETRY_STATUS_CSV = Path("data/pipeline/geometry_status.csv")


def build_dft_smiles_index():
    """Map each SMILES to its matching dft_surface/{compound_id}.npz file.

    The DFT campaign is keyed by compound_id with suffix _cation/_anion/_pair;
    geometry_status.csv stores compound_id and its constituent SMILES. Returns
    {smiles: Path} for every .npz that currently exists on disk.
    """
    base = Path(__file__).resolve().parent.parent.parent
    csv_path = base / GEOMETRY_STATUS_CSV
    surf_dir = base / DFT_SURFACE_DIR
    if not csv_path.exists() or not surf_dir.exists():
        return {}
    mapping = {}
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        cid = row["compound_id"]
        for suffix, key in [("_pair", "smiles"),
                             ("_cation", "cation_smiles"),
                             ("_anion", "anion_smiles")]:
            npz = surf_dir / f"{cid}{suffix}.npz"
            if npz.exists() and isinstance(row.get(key), str):
                mapping[row[key]] = npz
    return mapping


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
    return sample_point_cloud(verts, normals, esp, n_points)


def smiles_to_hash(smiles):
    """Create a short hash from SMILES for filenames."""
    return hashlib.md5(smiles.encode()).hexdigest()[:12]


def sample_point_cloud(verts, normals, esp, n_points=1024):
    """Sample a fixed number of points from the mesh surface.

    Uses farthest point sampling for uniform coverage when downsampling,
    or random duplication when upsampling.
    """
    n = len(verts)

    if n >= n_points:
        # Farthest point sampling for uniform coverage
        indices = farthest_point_sample(verts, n_points)
    else:
        # Upsample: use all points + random duplicates
        extra = n_points - n
        extra_idx = np.random.choice(n, size=extra, replace=True)
        indices = np.concatenate([np.arange(n), extra_idx])

    # Combine into (N, 7): [x, y, z, nx, ny, nz, esp]
    points = np.column_stack([
        verts[indices],
        normals[indices],
        esp[indices, None],
    ])

    return points.astype(np.float32)


def farthest_point_sample(pts, n_samples):
    """Farthest point sampling for uniform coverage."""
    n = len(pts)
    selected = np.zeros(n_samples, dtype=np.int64)
    distances = np.full(n, np.inf)

    # Start from random point
    selected[0] = np.random.randint(n)

    for i in range(1, n_samples):
        last = selected[i - 1]
        dist_to_last = np.sum((pts - pts[last]) ** 2, axis=1)
        distances = np.minimum(distances, dist_to_last)
        selected[i] = np.argmax(distances)

    return selected


def extract_point_cloud(smiles, n_points=1024, dft_index=None):
    """Extract point cloud from SMILES.

    If dft_index is provided and contains this SMILES, the Psi4 DFT
    surface is loaded directly, bypassing the Gasteiger marching-cubes
    fallback. Returns (N, 7) array or None on failure.
    """
    if dft_index is not None and smiles in dft_index:
        return load_dft_surface(dft_index[smiles], n_points=n_points)
    mol = prepare_molecule(smiles)
    if mol is None:
        return None

    pos, charges, radii = get_mol_data(mol)
    pos = separate_ions(mol, pos, separation=2.5)

    verts, faces, normals = build_isosurface(pos, radii, grid_res=0.20)
    if verts is None or len(verts) < 10:
        return None

    esp = compute_surface_esp(verts, pos, charges, radii, mol=mol)

    # Center the point cloud
    center = verts.mean(axis=0)
    verts_centered = verts - center

    # Normalize scale (unit sphere)
    scale = np.max(np.linalg.norm(verts_centered, axis=1)) + 1e-8
    verts_centered = verts_centered / scale

    points = sample_point_cloud(verts_centered, normals, esp, n_points)
    return points


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--prefer-dft", action="store_true",
                        help="Prefer Psi4 dft_surface/*.npz when available")
    args = parser.parse_args()

    dft_index = build_dft_smiles_index() if args.prefer_dft else None
    if dft_index:
        print(f"DFT surfaces available for {len(dft_index)} SMILES")

    base_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = base_dir / "data" / "pipeline" / "point_clouds"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all unique SMILES from both datasets
    all_smiles = {}

    # Original dataset
    raw_csv = base_dir / "data" / "processed" / "il_data_raw.csv"
    if raw_csv.exists():
        df = pd.read_csv(raw_csv)
        for _, row in df.drop_duplicates(subset=["smiles"]).iterrows():
            all_smiles[row["smiles"]] = row.get("il_short_name", "unknown")

    # ILThermo dataset
    ilthermo_csv = base_dir / "data" / "augmented" / "ilthermo_data.csv"
    if ilthermo_csv.exists():
        df2 = pd.read_csv(ilthermo_csv)
        for _, row in df2.drop_duplicates(subset=["smiles"]).iterrows():
            if row["smiles"] not in all_smiles:
                all_smiles[row["smiles"]] = row.get("il_short_name", "unknown")[:30]

    smiles_list = sorted(all_smiles.keys())
    end = args.end if args.end is not None else len(smiles_list)
    smiles_list = smiles_list[args.start:end]

    print(f"Extracting point clouds for {len(smiles_list)} unique ILs "
          f"[{args.start}:{end}], {args.n_points} points each\n")

    index = []
    success = 0
    failed = 0

    for i, smiles in enumerate(smiles_list):
        name = all_smiles[smiles]
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(smiles_list)}] {name} ({success} ok, {failed} fail)")

        # Skip if already computed
        file_hash = smiles_to_hash(smiles)
        npz_path = output_dir / f"{file_hash}.npz"
        if npz_path.exists():
            index.append({"smiles": smiles, "filename": f"{file_hash}.npz", "il_name": name})
            success += 1
            continue

        points = extract_point_cloud(smiles, n_points=args.n_points, dft_index=dft_index)
        if points is not None:
            np.savez_compressed(npz_path, points=points)
            index.append({"smiles": smiles, "filename": f"{file_hash}.npz", "il_name": name})
            success += 1
        else:
            failed += 1

    # Save/update index
    index_path = output_dir / "index.csv"

    # Merge with existing index if present
    existing_index = {}
    if index_path.exists():
        with open(index_path) as f:
            for row in csv.DictReader(f):
                existing_index[row["smiles"]] = row

    for entry in index:
        existing_index[entry["smiles"]] = entry

    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles", "filename", "il_name"])
        writer.writeheader()
        for smiles in sorted(existing_index.keys()):
            writer.writerow(existing_index[smiles])

    print(f"\nDone: {success} success, {failed} failed")
    print(f"Index: {len(existing_index)} total entries at {index_path}")


if __name__ == "__main__":
    main()
