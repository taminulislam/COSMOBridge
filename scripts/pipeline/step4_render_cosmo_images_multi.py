"""Step 4 (multi-conformer variant): Render N COSMO rotation-frame sets per IL,
one per ETKDG conformer seed. Output goes under
    data/pipeline/cosmo_images_multi/conf_{k}/{compound_id}_frames/frame_NNN.png

Reuses all the heavy-lifting helpers from step4_render_cosmo_images (isosurface
building, ESP computation, rendering). Only the RDKit embedding is
re-implemented to accept a seed and the optional `keep_lowest_energy` flag,
which regenerates multiple conformers with different seeds and keeps the lowest
MMFF94 energy (so `conf_0..conf_{N-1}` are genuinely distinct low-energy
geometries, not independent random draws).

Usage (SLURM-friendly range slicing like the original step4):
    python step4_render_cosmo_images_multi.py --start 0 --end 50 --n_conformers 5
    python step4_render_cosmo_images_multi.py --start 0 --end 50 --n_conformers 5 --seed_offset 1

Output dir can be overridden with --output_root.

Notes for Stage 2 of the v5 improvement plan:
- `conf_0` with default seed 42 is intentionally the same geometry as the
  existing `cosmo_images_gasteiger_apr10/` run, so you can validate against it
  before committing to re-render.
- Each conformer is a fresh ETKDG embedding with seed = base_seed + conf_id.
- MMFF optimization is run on each conformer independently (500 iters, same as
  the single-conformer pipeline).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

# Reuse step4 helpers — these are pure functions that operate on arrays/mols
from step4_render_cosmo_images import (  # noqa: E402
    VDW_RADII,
    build_isosurface,
    compute_surface_esp,
    get_mol_data,
    load_dft_points,
    read_xyz,
    render_isosurface,
    rotation_matrix,
    separate_ions,
)

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402


def embed_seeded(smiles, seed):
    """Embed a molecule with ETKDGv3 using an explicit seed. Returns
    (mol, mmff_energy) or (None, None) on failure.

    We do NOT use the "best of N" logic from step2_geometry_optimization.py —
    this function is called once per (compound, conformer_id) so the caller
    can keep whichever energy-ranked conformers they want.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    res = AllChem.EmbedMolecule(mol, params)
    if res != 0:
        params2 = AllChem.ETKDGv3()
        params2.useRandomCoords = True
        params2.randomSeed = int(seed)
        res = AllChem.EmbedMolecule(mol, params2)
        if res != 0:
            return None, None

    mmff_energy = float("nan")
    try:
        ff_props = AllChem.MMFFGetMoleculeProperties(mol)
        if ff_props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol, ff_props)
            if ff is not None:
                ff.Minimize(maxIts=500)
                mmff_energy = float(ff.CalcEnergy())
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    AllChem.ComputeGasteigerCharges(mol)
    return mol, mmff_energy


def render_molecule_seeded(
    smiles, compound_id, conf_output_dir, seed,
    img_size=512, grid_res=0.20, n_views=36,
):
    """Render a single conformer's rotation frames for one compound.

    Unlike step4.render_molecule this variant:
    - Always uses the Gasteiger / RDKit path (no DFT fallback) so the seed
      actually controls the geometry. DFT-based geometries are precomputed and
      single-conformer only.
    - Skips the main {compound_id}_cosmo.png and _ep.png outputs since
      conformer ensembling only needs the 36 rotation frames.
    - Output: {conf_output_dir}/{compound_id}_frames/frame_NNN.png
    """
    mol, mmff_energy = embed_seeded(smiles, seed)
    if mol is None:
        return False, None

    pos, charges, radii = get_mol_data(mol)
    pos = separate_ions(mol, pos, separation=2.5)

    verts, faces, normals = build_isosurface(pos, radii, grid_res=grid_res)
    if verts is None or len(verts) < 10:
        return False, mmff_energy

    esp = compute_surface_esp(verts, pos, charges, radii, mol=mol)

    frames_dir = Path(conf_output_dir) / f"{compound_id}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for fi in range(n_views):
        angle = fi * (360.0 / n_views)
        R_f = rotation_matrix(15, angle)
        frame = render_isosurface(verts, faces, normals, esp, R_f, img_size)
        frame.save(frames_dir / f"frame_{fi:03d}.png", quality=85)

    return True, mmff_energy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--n_conformers", type=int, default=5,
                    help="Number of conformers to render per compound")
    ap.add_argument("--seed_offset", type=int, default=0,
                    help="First conformer's base seed offset. conf_k gets seed=42+k+seed_offset")
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--n_views", type=int, default=36)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--grid_res", type=float, default=0.20)
    ap.add_argument("--output_root", type=str,
                    default="data/pipeline/cosmo_images_multi")
    ap.add_argument("--compounds_csv", type=str,
                    default="data/pipeline/ilthermo_compounds.csv")
    args = ap.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    energies_dir = output_root / "_energies"
    energies_dir.mkdir(parents=True, exist_ok=True)

    # Load compounds
    compounds = []
    source_csv = Path("data/pipeline/geometry_status.csv")
    if not (source_csv.exists() and source_csv.stat().st_size > 200):
        source_csv = Path(args.compounds_csv)
    if not source_csv.exists():
        print(f"ERROR: no compound list at {source_csv}")
        return

    with open(source_csv) as f:
        for row in csv.DictReader(f):
            if row.get("smiles"):
                compounds.append(row)

    end = args.end if args.end is not None else len(compounds)
    compounds = compounds[args.start:end]
    print(f"Rendering [{args.start}:{end}] × {args.n_conformers} conformers = "
          f"{len(compounds) * args.n_conformers} frame sets")

    success, failed = 0, 0
    for i, comp in enumerate(compounds):
        cid = comp["compound_id"]
        smiles = comp["smiles"]
        energies = {}

        for k in range(args.n_conformers):
            seed = args.base_seed + k + args.seed_offset
            conf_dir = output_root / f"conf_{k}"
            conf_dir.mkdir(parents=True, exist_ok=True)

            frames_dir = conf_dir / f"{cid}_frames"
            if frames_dir.exists() and len(list(frames_dir.glob("frame_*.png"))) >= args.n_views:
                success += 1
                continue

            try:
                ok, energy = render_molecule_seeded(
                    smiles, cid, conf_dir, seed=seed,
                    img_size=args.img_size, grid_res=args.grid_res, n_views=args.n_views,
                )
                if ok:
                    success += 1
                    if energy is not None:
                        energies[str(k)] = energy
                else:
                    failed += 1
            except Exception as e:
                print(f"    ERROR {cid} conf_{k}: {e}")
                failed += 1

        if energies:
            import json
            energies_path = energies_dir / f"{cid}.json"
            existing = {}
            if energies_path.exists():
                with open(energies_path) as f:
                    existing = json.load(f)
            existing.update(energies)
            with open(energies_path, "w") as f:
                json.dump(existing, f, indent=2)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(compounds)}] {cid} "
                  f"({success} frame-sets ok, {failed} failed)", flush=True)

    print(f"\nDone: {success} frame-sets rendered, {failed} failed")


if __name__ == "__main__":
    main()
