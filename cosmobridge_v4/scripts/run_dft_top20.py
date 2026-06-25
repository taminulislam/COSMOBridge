"""Run GFN2-xTB geometry optimization for top-20 candidates (I5 step 2a).

This is the FAST part of I5 (~5 min per IL = ~1.5 hr total).
The slow NWChem DFT step is launched as a separate SLURM array.

Pipeline:
1. Load top-20 SMILES from cosmobridge_v4/screening/top20_candidates.csv
2. Generate 3D conformers (RDKit ETKDG + MMFF94)
3. Refine with GFN2-xTB
4. Save optimized XYZ files to cosmobridge_v4/screening/geometries/
5. Write NWChem input files for the DFT array job
"""

import sys
import csv
import os
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from rdkit import Chem
from rdkit.Chem import AllChem
from scripts.pipeline.step2_geometry_optimization import smiles_to_3d, mol_to_xyz_string


def write_nwchem_input(xyz_path, charge, nw_path, compound_id):
    """Write NWChem input for B3LYP/def2-SVP + COSMO calculation."""
    # Read xyz coords (skip first 2 lines)
    lines = xyz_path.read_text().strip().split("\n")
    n_atoms = int(lines[0])
    geom_lines = lines[2:2+n_atoms]

    # Build NWChem input
    nw = f"""# NWChem input: {compound_id}
# B3LYP/def2-SVP + COSMO for ESP surface calculation

title "{compound_id} COSMO ESP"

start {compound_id}
charge {charge}

geometry units angstroms noautosym
{chr(10).join(geom_lines)}
end

basis spherical
 * library def2-svp
end

dft
 xc b3lyp
 grid fine
 convergence energy 1.0e-7
 iterations 200
end

cosmo
 dielec 78.4
 rsolv 1.3
 lineq 0
 do_cosmo_smd false
end

task dft energy
task dft property
"""
    nw_path.write_text(nw)


def compute_charge(smiles):
    """Net formal charge."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(a.GetFormalCharge() for a in mol.GetAtoms())


def main():
    candidates_path = Path("cosmobridge_v4/screening/top20_candidates.csv")
    if not candidates_path.exists():
        print(f"ERROR: {candidates_path} not found")
        sys.exit(1)

    geom_dir = Path("cosmobridge_v4/screening/geometries")
    geom_dir.mkdir(parents=True, exist_ok=True)

    nw_dir = Path("cosmobridge_v4/screening/nwchem_inputs")
    nw_dir.mkdir(parents=True, exist_ok=True)

    reader = csv.DictReader(candidates_path.open())
    candidates = list(reader)
    print(f"Processing {len(candidates)} candidates")

    successes = []
    failures = []

    for i, cand in enumerate(candidates, 1):
        cid = cand['id']
        smi = cand['smiles']
        print(f"\n[{i}/{len(candidates)}] {cid}: {smi}")

        # Split cation and anion
        parts = smi.split(".")
        cation_smi = parts[0]
        anion_smi = parts[1] if len(parts) > 1 else None

        for label, sub_smi in [("cation", cation_smi), ("anion", anion_smi)]:
            if sub_smi is None:
                continue

            xyz_path = geom_dir / f"{cid}_{label}.xyz"
            if xyz_path.exists():
                print(f"  {label}: already exists, skipping")
                continue

            try:
                result = smiles_to_3d(sub_smi, n_conformers=10)
                if result is None:
                    print(f"  {label}: ERROR generating 3D")
                    failures.append(f"{cid}_{label}")
                    continue
                mol, conf_id = result
                xyz_str = mol_to_xyz_string(mol, conf_id)
                xyz_path.write_text(xyz_str)

                # Write NWChem input
                charge = compute_charge(sub_smi)
                nw_path = nw_dir / f"{cid}_{label}.nw"
                write_nwchem_input(xyz_path, charge, nw_path, f"{cid}_{label}")
                print(f"  {label}: {mol.GetNumAtoms()} atoms, charge={charge:+d}")
                successes.append(f"{cid}_{label}")
            except Exception as e:
                print(f"  {label}: ERROR {e}")
                failures.append(f"{cid}_{label}")

    print(f"\n{'='*60}")
    print(f"Geometry prep complete")
    print(f"  Successes: {len(successes)}")
    print(f"  Failures: {len(failures)}")
    print(f"  XYZ files: {geom_dir}")
    print(f"  NWChem inputs: {nw_dir}")

    if failures:
        print(f"  Failed: {failures}")

    # Write task list for DFT array
    task_list_path = Path("cosmobridge_v4/screening/dft_tasks.txt")
    task_list_path.write_text("\n".join(successes) + "\n")
    print(f"  Task list for DFT array: {task_list_path}")
    print(f"\n  {len(successes)} NWChem tasks ready to submit")


if __name__ == "__main__":
    main()
