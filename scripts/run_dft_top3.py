"""Run full COSMO pipeline for top 3 IL candidates.

Step 1: SMILES → 3D geometry (RDKit ETKDG + MMFF94 + xTB if available)
Step 2: Generate NWChem input files (B3LYP/def2-SVP + COSMO)
Step 3: Run NWChem DFT (if available) or use semi-empirical ESP approximation
Step 4: Extract point cloud from surface
Step 5: Render COSMO images
"""

import sys
import os
import subprocess
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from rdkit import Chem
from rdkit.Chem import AllChem
from scripts.pipeline.step2_geometry_optimization import smiles_to_3d, mol_to_xyz_string

TOP3 = [
    ("mmim_oac", "C[n+]1ccn(C)c1.CC(=O)[O-]", "1,3-Dimethylimidazolium Acetate"),
    ("tmg_oac", "CN(C)C(=[NH2+])N(C)C.CC(=O)[O-]", "Tetramethylguanidinium Acetate"),
    ("mmim_lev", "C[n+]1ccn(C)c1.CC(=O)CCC(=O)[O-]", "1,3-Dimethylimidazolium Levulinate"),
]


def step1_geometry(compound_id, smiles, geom_dir):
    """Generate and optimize 3D geometry."""
    print(f"\n  Step 1: Geometry optimization for {compound_id}")
    geom_dir = Path(geom_dir)
    geom_dir.mkdir(parents=True, exist_ok=True)

    # Split into cation and anion
    parts = smiles.split(".")
    cation_smi = parts[0]
    anion_smi = parts[1] if len(parts) > 1 else None

    # Process each ion
    for label, smi in [("cation", cation_smi), ("anion", anion_smi)]:
        if smi is None:
            continue
        print(f"    {label}: {smi}")
        result = smiles_to_3d(smi, n_conformers=10)
        if result is None:
            print(f"    ERROR: Could not generate 3D for {label}")
            continue
        mol, conf_id = result
        xyz_str = mol_to_xyz_string(mol, conf_id)
        xyz_path = geom_dir / f"{compound_id}_{label}.xyz"
        xyz_path.write_text(xyz_str)
        print(f"    Saved: {xyz_path} ({mol.GetNumAtoms()} atoms)")

    # Combined ion pair
    mol_full = Chem.MolFromSmiles(smiles)
    if mol_full:
        mol_full = Chem.AddHs(mol_full)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol_full, params)
        try:
            AllChem.MMFFOptimizeMolecule(mol_full, maxIters=500)
        except:
            pass
        conf = mol_full.GetConformer()
        n = mol_full.GetNumAtoms()
        lines = [str(n), f"{compound_id} ion pair"]
        for i in range(n):
            atom = mol_full.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            lines.append(f"{atom.GetSymbol():2s}  {pos.x:12.6f}  {pos.y:12.6f}  {pos.z:12.6f}")
        pair_path = geom_dir / f"{compound_id}_pair.xyz"
        pair_path.write_text("\n".join(lines))
        print(f"    Saved ion pair: {pair_path} ({n} atoms)")
        return mol_full
    return None


def step2_nwchem_input(compound_id, geom_dir, nwchem_dir):
    """Generate NWChem input for DFT ESP + COSMO."""
    print(f"\n  Step 2: Generating NWChem input for {compound_id}")
    nwchem_dir = Path(nwchem_dir)
    nwchem_dir.mkdir(parents=True, exist_ok=True)

    xyz_path = Path(geom_dir) / f"{compound_id}_pair.xyz"
    if not xyz_path.exists():
        print(f"    ERROR: {xyz_path} not found")
        return

    from scripts.pipeline.step3_dft_esp import generate_nwchem_input, get_charge_from_smiles

    smiles = [s for cid, s, _ in TOP3 if cid == compound_id][0]
    charge = get_charge_from_smiles(smiles)

    output_path = nwchem_dir / f"{compound_id}.nw"
    generate_nwchem_input(compound_id, str(xyz_path), str(output_path),
                           charge=charge, method="b3lyp", basis="def2-svp",
                           use_cosmo=True, dielectric=15.0)
    print(f"    Saved: {output_path} (charge={charge})")


def step3_extract_point_cloud(compound_id, mol, pc_dir, n_points=1024):
    """Extract point cloud from molecular surface (approximate COSMO)."""
    print(f"\n  Step 3: Extracting point cloud for {compound_id}")
    pc_dir = Path(pc_dir)
    pc_dir.mkdir(parents=True, exist_ok=True)

    if mol is None:
        print(f"    ERROR: No molecule")
        return

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    # Compute Gasteiger charges for ESP
    AllChem.ComputeGasteigerCharges(mol)

    # Generate surface points
    vdw_radii = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 16: 1.80, 17: 1.75, 9: 1.47}
    atom_pos = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                           conf.GetAtomPosition(i).z] for i in range(n_atoms)])
    atom_radii = np.array([vdw_radii.get(mol.GetAtomWithIdx(i).GetAtomicNum(), 1.70)
                           for i in range(n_atoms)])
    atom_charges = []
    for i in range(n_atoms):
        c = float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge'))
        atom_charges.append(0.0 if np.isnan(c) else c)
    atom_charges = np.array(atom_charges)

    # Generate surface points on atomic spheres
    all_points = []
    probe = 1.4 * 0.3

    for i in range(n_atoms):
        radius = atom_radii[i] + probe
        n_pts = max(50, int(n_points * 2 * radius**2 / max(sum(r**2 for r in atom_radii), 1)))
        phi = np.random.uniform(0, 2 * np.pi, n_pts)
        costheta = np.random.uniform(-1, 1, n_pts)
        theta = np.arccos(costheta)

        pts = np.column_stack([
            atom_pos[i, 0] + radius * np.sin(theta) * np.cos(phi),
            atom_pos[i, 1] + radius * np.sin(theta) * np.sin(phi),
            atom_pos[i, 2] + radius * np.cos(theta),
        ])

        # Remove points inside other atoms
        keep = np.ones(len(pts), dtype=bool)
        for j in range(n_atoms):
            if i == j: continue
            dists = np.linalg.norm(pts - atom_pos[j], axis=1)
            keep &= dists > (atom_radii[j] + probe) * 0.9
        pts = pts[keep]

        if len(pts) > 0:
            # Compute normals (outward from atom center)
            normals = pts - atom_pos[i]
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

            # Compute ESP at each point
            esp = np.zeros(len(pts))
            for j in range(n_atoms):
                dists = np.linalg.norm(pts - atom_pos[j], axis=1)
                dists = np.clip(dists, 0.5, None)
                esp += atom_charges[j] / dists

            # Stack: xyz(3) + normals(3) + ESP(1) = 7D
            points = np.column_stack([pts, normals, esp])
            all_points.append(points)

    if all_points:
        all_pts = np.concatenate(all_points)

        # Subsample to n_points
        if len(all_pts) > n_points:
            idx = np.random.choice(len(all_pts), n_points, replace=False)
            all_pts = all_pts[idx]
        elif len(all_pts) < n_points:
            extra = np.random.choice(len(all_pts), n_points - len(all_pts), replace=True)
            all_pts = np.concatenate([all_pts, all_pts[extra]])

        # Normalize coordinates to unit sphere
        center = all_pts[:, :3].mean(axis=0)
        all_pts[:, :3] -= center
        scale = np.abs(all_pts[:, :3]).max()
        if scale > 0:
            all_pts[:, :3] /= scale

        out_path = pc_dir / f"{compound_id}.npz"
        np.savez(out_path, points=all_pts.astype(np.float32))
        print(f"    Saved: {out_path} ({all_pts.shape})")
        print(f"    ESP range: [{all_pts[:, 6].min():.4f}, {all_pts[:, 6].max():.4f}]")
        return out_path
    else:
        print(f"    ERROR: No surface points generated")
        return None


def main():
    print("=== DFT Pipeline for Top 3 IL Candidates ===\n")

    geom_dir = Path("data/pipeline/geometries_novel")
    nwchem_dir = Path("data/pipeline/nwchem_novel")
    pc_dir = Path("data/pipeline/point_clouds_novel")

    molecules = {}

    for compound_id, smiles, name in TOP3:
        print(f"\n{'='*60}")
        print(f"Processing: {name} ({compound_id})")
        print(f"SMILES: {smiles}")
        print(f"{'='*60}")

        # Step 1: Geometry
        mol = step1_geometry(compound_id, smiles, geom_dir)
        molecules[compound_id] = mol

        # Step 2: NWChem input
        step2_nwchem_input(compound_id, geom_dir, nwchem_dir)

        # Step 3: Extract point cloud (semi-empirical approximation)
        step3_extract_point_cloud(compound_id, mol, pc_dir)

    # Step 4: Check if NWChem is available and run
    print(f"\n{'='*60}")
    print("NWChem DFT Calculations")
    print(f"{'='*60}")

    nwchem_available = subprocess.run(
        ["which", "nwchem"], capture_output=True).returncode == 0
    module_check = subprocess.run(
        ["bash", "-c", "module avail nwchem 2>&1"], capture_output=True, text=True)

    if nwchem_available:
        print("  NWChem found! Running DFT calculations...")
        for compound_id, _, name in TOP3:
            nw_file = nwchem_dir / f"{compound_id}.nw"
            if nw_file.exists():
                out_dir = Path(f"data/pipeline/dft_output/{compound_id}")
                out_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n  Running NWChem for {compound_id}...")
                result = subprocess.run(
                    ["nwchem", str(nw_file)],
                    capture_output=True, text=True, timeout=3600,
                    cwd=str(out_dir))
                if result.returncode == 0:
                    print(f"    DFT completed successfully")
                else:
                    print(f"    DFT error: {result.stderr[-200:]}")
    elif "nwchem" in module_check.stdout.lower() or "nwchem" in module_check.stderr.lower():
        print("  NWChem available as module but not loaded.")
        print("  To run full DFT, submit with: module load nwchem")
        print("  NWChem input files saved to:", nwchem_dir)

        # Generate SLURM script for NWChem
        compound_list = nwchem_dir / "compound_list.txt"
        compound_list.write_text("\n".join(cid for cid, _, _ in TOP3))

        slurm_script = f"""#!/bin/bash
#SBATCH --job-name=il_dft
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=1-3
#SBATCH --output=jobs/logs/dft_%A_%a.out
#SBATCH --error=jobs/logs/dft_%A_%a.err

module load nwchem/7.2.3.PrgEnv-gnu 2>/dev/null || module load nwchem 2>/dev/null

CID=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {compound_list})
echo "Running DFT for $CID"

mkdir -p data/pipeline/dft_output/$CID
cd data/pipeline/dft_output/$CID
mpirun -np 4 nwchem {nwchem_dir}/$CID.nw > nwchem.out 2>&1
echo "Done: $CID"
"""
        slurm_path = Path("jobs/dft_top3.sh")
        slurm_path.write_text(slurm_script)
        print(f"  SLURM script saved: {slurm_path}")
        print(f"  Submit with: sbatch {slurm_path}")
    else:
        print("  NWChem not available on this node.")
        print("  Semi-empirical point clouds generated as approximation.")
        print(f"  NWChem inputs saved to {nwchem_dir}/ for later DFT runs.")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Geometries:     {geom_dir}/")
    print(f"  NWChem inputs:  {nwchem_dir}/")
    print(f"  Point clouds:   {pc_dir}/")
    for compound_id, smiles, name in TOP3:
        pc_file = pc_dir / f"{compound_id}.npz"
        nw_file = nwchem_dir / f"{compound_id}.nw"
        geom_file = geom_dir / f"{compound_id}_pair.xyz"
        print(f"\n  {name}:")
        print(f"    Geometry:    {'OK' if geom_file.exists() else 'MISSING'}")
        print(f"    NWChem input:{'OK' if nw_file.exists() else 'MISSING'}")
        print(f"    Point cloud: {'OK' if pc_file.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
