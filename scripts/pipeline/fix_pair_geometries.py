"""Repair broken _pair.xyz files by rebuilding them from cation+anion.

Some pair geometries from step 2 have either duplicated atoms (cation
appears twice) or overlapping ion centroids (atoms within 0.001 A),
which makes Psi4's molecule parser reject the geometry. We rebuild
each affected pair from the matching {cation,anion}.xyz fragments,
re-centering each ion on its centroid and translating the anion along
+x by a fixed separation so that no atom-pair distance is below the
qcelemental "too close" threshold.

Inputs are read from data/pipeline/geometries/. Original pair files are
backed up to .xyz.bak before being overwritten.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

GEOM_DIR = Path("data/pipeline/geometries")
SEPARATION = 4.0  # angstroms between centroids of cation and anion


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].strip())
    atoms, coords = [], []
    for i in range(n):
        parts = lines[2 + i].split()
        atoms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, np.asarray(coords)


def write_xyz(path, atoms, coords, comment=""):
    lines = [str(len(atoms)), comment]
    for a, xyz in zip(atoms, coords):
        lines.append(f"{a} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
    Path(path).write_text("\n".join(lines) + "\n")


def rebuild_pair(cid, geom_dir):
    cat_atoms, cat_xyz = read_xyz(geom_dir / f"{cid}_cation.xyz")
    an_atoms, an_xyz = read_xyz(geom_dir / f"{cid}_anion.xyz")

    cat_xyz = cat_xyz - cat_xyz.mean(axis=0)
    an_xyz = an_xyz - an_xyz.mean(axis=0)

    cat_radius = np.linalg.norm(cat_xyz, axis=1).max() if len(cat_xyz) > 1 else 0.0
    an_radius = np.linalg.norm(an_xyz, axis=1).max() if len(an_xyz) > 1 else 0.0
    offset = cat_radius + an_radius + SEPARATION

    an_xyz = an_xyz + np.array([offset, 0.0, 0.0])
    return cat_atoms + an_atoms, np.vstack([cat_xyz, an_xyz])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cids", nargs="+", required=True,
                    help="Compound IDs (without _pair suffix) to repair.")
    ap.add_argument("--geom-dir", default=str(GEOM_DIR))
    args = ap.parse_args()

    geom_dir = Path(args.geom_dir)
    repaired = 0
    for cid in args.cids:
        pair_path = geom_dir / f"{cid}_pair.xyz"
        cat_path = geom_dir / f"{cid}_cation.xyz"
        an_path = geom_dir / f"{cid}_anion.xyz"
        if not (cat_path.exists() and an_path.exists()):
            print(f"SKIP {cid}: cation or anion missing")
            continue

        if pair_path.exists():
            backup = pair_path.with_suffix(".xyz.bak")
            if not backup.exists():
                shutil.copy2(pair_path, backup)

        atoms, coords = rebuild_pair(cid, geom_dir)
        write_xyz(pair_path, atoms, coords,
                  comment=f"{cid} pair rebuilt from cation+anion (sep={SEPARATION}A)")
        print(f"OK {cid}: {len(atoms)} atoms")
        repaired += 1

    print(f"\nRepaired {repaired} / {len(args.cids)} pairs")


if __name__ == "__main__":
    main()
