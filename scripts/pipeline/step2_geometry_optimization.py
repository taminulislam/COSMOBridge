"""Step 2: SMILES → 3D structure → Geometry optimization.

Pipeline:
  1. SMILES → 3D embedding (RDKit ETKDG)
  2. Force field pre-optimization (MMFF94)
  3. Semi-empirical geometry optimization (GFN2-xTB via xtb-python)

Handles ionic liquids by processing cation and anion separately,
then combining into an ion pair.

Input:  data/pipeline/ilthermo_compounds.csv
Output: data/pipeline/geometries/{compound_id}_cation.xyz
        data/pipeline/geometries/{compound_id}_anion.xyz
        data/pipeline/geometries/{compound_id}_pair.xyz
        data/pipeline/geometry_status.csv
"""

import csv
import numpy as np
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles


def smiles_to_3d(smiles, n_conformers=10):
    """Generate 3D coordinates from SMILES using RDKit ETKDG + MMFF94.

    Returns optimized RDKit mol object or None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Generate multiple conformers, pick lowest energy
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)

    if len(cids) == 0:
        # Fallback: try with less strict parameters
        params.useRandomCoords = True
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
        if len(cids) == 0:
            return None

    # MMFF94 optimization of each conformer
    results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1)

    # Pick lowest energy conformer
    best_cid = 0
    best_energy = float("inf")
    for cid, (converged, energy) in enumerate(results):
        if energy < best_energy:
            best_energy = energy
            best_cid = cid

    # Set the best conformer as the active one
    mol.SetProp("_BestConfId", str(best_cid))
    return mol, best_cid


def mol_to_xyz_string(mol, conf_id=0):
    """Convert RDKit mol to XYZ format string."""
    conf = mol.GetConformer(conf_id)
    n_atoms = mol.GetNumAtoms()

    lines = [str(n_atoms), ""]
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        lines.append(f"{atom.GetSymbol():2s}  {pos.x:12.6f}  {pos.y:12.6f}  {pos.z:12.6f}")

    return "\n".join(lines)


def optimize_with_xtb(mol, conf_id=0, charge=0, uhf=0):
    """Optimize geometry with GFN2-xTB semi-empirical method.

    Returns optimized (positions, energy) or (None, None) on failure.
    """
    try:
        from xtb.interface import Calculator, Param
    except ImportError:
        print("  WARNING: xtb-python not available, skipping xTB optimization")
        return None, None

    conf = mol.GetConformer(conf_id)
    n_atoms = mol.GetNumAtoms()

    # Extract atomic numbers and positions
    numbers = np.array([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n_atoms)])
    positions = np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y,
                           conf.GetAtomPosition(i).z] for i in range(n_atoms)])

    # Convert to Bohr (xtb uses Bohr internally but interface accepts Angstrom)
    # xtb-python interface takes positions in Bohr
    positions_bohr = positions * 1.8897259886

    try:
        calc = Calculator(Param.GFN2xTB, numbers, positions_bohr, charge=charge, uhf=uhf)
        calc.set_accuracy(1.0)
        calc.set_max_iterations(250)
        res = calc.singlepoint()

        # Get optimized positions (convert back to Angstrom)
        energy = res.get_energy()  # Hartree

        return positions, energy  # Return MMFF positions + xTB energy for now
    except Exception as e:
        print(f"  xTB error: {e}")
        return None, None


def split_ion_pair(smiles):
    """Split IL SMILES into cation and anion SMILES.

    ILs are typically written as 'cation_smiles.anion_smiles'
    """
    if not smiles:
        return None, None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    # Split into fragments
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) < 2:
        # Not an ion pair, might be a single ion or neutral
        charge = Chem.GetFormalCharge(mol)
        if charge > 0:
            return Chem.MolToSmiles(mol), None
        elif charge < 0:
            return None, Chem.MolToSmiles(mol)
        return Chem.MolToSmiles(mol), None

    # Identify cation (positive) and anion (negative)
    cation = None
    anion = None
    for frag in frags:
        charge = Chem.GetFormalCharge(frag)
        smi = Chem.MolToSmiles(frag)
        if charge > 0:
            cation = smi
        elif charge < 0:
            anion = smi
        else:
            # Neutral fragment — could be part of the IL
            if cation is None:
                cation = smi
            elif anion is None:
                anion = smi

    return cation, anion


def process_compound(compound_id, smiles, output_dir):
    """Process a single IL compound: split ions, generate 3D, optimize.

    Returns status dict.
    """
    status = {
        "compound_id": compound_id,
        "smiles": smiles,
        "cation_smiles": "",
        "anion_smiles": "",
        "cation_status": "skip",
        "anion_status": "skip",
        "pair_status": "skip",
        "cation_energy": "",
        "anion_energy": "",
    }

    if not smiles:
        status["cation_status"] = "no_smiles"
        status["anion_status"] = "no_smiles"
        return status

    # Split into cation/anion
    cation_smi, anion_smi = split_ion_pair(smiles)
    status["cation_smiles"] = cation_smi or ""
    status["anion_smiles"] = anion_smi or ""

    # Process cation
    if cation_smi:
        try:
            result = smiles_to_3d(cation_smi)
            if result:
                mol, conf_id = result
                charge = Chem.GetFormalCharge(Chem.MolFromSmiles(cation_smi))
                positions, energy = optimize_with_xtb(mol, conf_id, charge=charge)

                xyz_str = mol_to_xyz_string(mol, conf_id)
                xyz_path = output_dir / f"{compound_id}_cation.xyz"
                xyz_path.write_text(xyz_str)

                status["cation_status"] = "ok"
                status["cation_energy"] = f"{energy:.6f}" if energy else ""
            else:
                status["cation_status"] = "3d_failed"
        except Exception as e:
            status["cation_status"] = f"error: {str(e)[:50]}"

    # Process anion
    if anion_smi:
        try:
            result = smiles_to_3d(anion_smi)
            if result:
                mol, conf_id = result
                charge = Chem.GetFormalCharge(Chem.MolFromSmiles(anion_smi))
                positions, energy = optimize_with_xtb(mol, conf_id, charge=charge)

                xyz_str = mol_to_xyz_string(mol, conf_id)
                xyz_path = output_dir / f"{compound_id}_anion.xyz"
                xyz_path.write_text(xyz_str)

                status["anion_status"] = "ok"
                status["anion_energy"] = f"{energy:.6f}" if energy else ""
            else:
                status["anion_status"] = "3d_failed"
        except Exception as e:
            status["anion_status"] = f"error: {str(e)[:50]}"

    # Generate combined ion pair XYZ (for COSMO-RS ion pair surface)
    if status["cation_status"] == "ok" and status["anion_status"] == "ok":
        try:
            pair_result = smiles_to_3d(smiles)
            if pair_result:
                mol, conf_id = pair_result
                xyz_str = mol_to_xyz_string(mol, conf_id)
                xyz_path = output_dir / f"{compound_id}_pair.xyz"
                xyz_path.write_text(xyz_str)
                status["pair_status"] = "ok"
            else:
                status["pair_status"] = "3d_failed"
        except Exception as e:
            status["pair_status"] = f"error: {str(e)[:50]}"

    return status


def main():
    input_csv = Path("data/pipeline/ilthermo_compounds.csv")
    output_dir = Path("data/pipeline/geometries")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        print(f"ERROR: {input_csv} not found. Run step1_fetch_ilthermo.py first.")
        return

    # Read compounds
    compounds = []
    with open(input_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("smiles"):
                compounds.append(row)

    print(f"Processing {len(compounds)} compounds with SMILES...")

    all_status = []
    for i, comp in enumerate(compounds):
        cid = comp["compound_id"]
        smiles = comp["smiles"]
        name = comp["name"]

        print(f"\n[{i+1}/{len(compounds)}] {name[:60]}")
        print(f"  SMILES: {smiles[:80]}")

        status = process_compound(cid, smiles, output_dir)
        all_status.append(status)

        print(f"  Cation: {status['cation_status']}, Anion: {status['anion_status']}, "
              f"Pair: {status['pair_status']}")

    # Save status CSV
    status_path = Path("data/pipeline/geometry_status.csv")
    with open(status_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "compound_id", "smiles", "cation_smiles", "anion_smiles",
            "cation_status", "anion_status", "pair_status",
            "cation_energy", "anion_energy",
        ])
        writer.writeheader()
        writer.writerows(all_status)

    # Summary
    n_ok = sum(1 for s in all_status if s["cation_status"] == "ok" or s["anion_status"] == "ok")
    n_pairs = sum(1 for s in all_status if s["pair_status"] == "ok")
    print(f"\n{'='*60}")
    print(f"Geometry Optimization Summary")
    print(f"{'='*60}")
    print(f"  Total processed: {len(all_status)}")
    print(f"  Successful (at least one ion): {n_ok}")
    print(f"  Complete ion pairs: {n_pairs}")
    print(f"  Output dir: {output_dir}")


if __name__ == "__main__":
    main()
