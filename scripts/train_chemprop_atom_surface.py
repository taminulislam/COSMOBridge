"""Chemprop with per-atom COSMO surface features.

Feeds per-atom surface descriptors (10D) into Chemprop's D-MPNN as
atom-level features. Chemprop concatenates these with its default 133D
atom features BEFORE message passing, giving the D-MPNN direct access
to local electrostatic surface information.

This combines:
- Chemprop's optimized D-MPNN (best avg R²=0.770)
- Per-atom COSMO surface patches (10D: ESP stats + curvature + area)
- Thermodynamic features (T, x₁)

Step 1: Pre-compute per-atom surface features for all molecules
Step 2: Save as Chemprop-compatible atom descriptor files
Step 3: Train Chemprop with --atom_descriptors feature
"""

import sys
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessing import TARGET_COLUMNS
from src.data.graph_builder import smiles_to_graph
from src.models.graph.atom_surface_dmpnn import compute_atom_surface_features, N_SURFACE_FEATURES
from src.training.metrics import compute_metrics, format_metrics

THERMO_FEATURES = ["temperature", "x1", "inv_temperature", "temp_squared", "temp_cubed"]


def precompute_surface_features(smiles_list, pc_dir):
    """Compute per-atom surface features for a list of SMILES.

    Returns list of (num_atoms_i, 10) arrays — one per molecule.
    Chemprop expects this format for atom_descriptors.
    """
    pc_idx = pd.read_csv(Path(pc_dir) / "index.csv")
    pc_map = dict(zip(pc_idx["smiles"], pc_idx["filename"]))

    # Cache per unique SMILES
    cache = {}
    all_features = []
    n_with_surface = 0

    for smi in smiles_list:
        if smi in cache:
            all_features.append(cache[smi])
            continue

        fn = pc_map.get(smi)
        if fn and (Path(pc_dir) / fn).exists():
            pc = np.load(Path(pc_dir) / fn)["points"]
            sf = compute_atom_surface_features(smi, pc)
            if sf is not None:
                # Chemprop uses RDKit without Hs for atom count
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smi)
                n_atoms_no_h = mol.GetNumAtoms() if mol else 0

                if sf.shape[0] != n_atoms_no_h:
                    # Our function uses AddHs — need to aggregate H surface features to heavy atoms
                    mol_h = Chem.AddHs(mol)
                    heavy_features = np.zeros((n_atoms_no_h, N_SURFACE_FEATURES))

                    for h_idx in range(mol_h.GetNumAtoms()):
                        atom = mol_h.GetAtomWithIdx(h_idx)
                        if atom.GetAtomicNum() == 1:
                            # H atom — add its surface features to its parent heavy atom
                            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
                            if neighbors:
                                parent_idx = neighbors[0]
                                # Map parent_idx in H-mol to heavy-atom index
                                # Heavy atoms keep their original indices in AddHs
                                if parent_idx < n_atoms_no_h:
                                    heavy_features[parent_idx] += sf[h_idx]
                        else:
                            if h_idx < n_atoms_no_h:
                                heavy_features[h_idx] += sf[h_idx]

                    # Re-normalize area fractions
                    total_area = heavy_features[:, 0].sum()
                    if total_area > 0:
                        heavy_features[:, 0] /= total_area

                    sf = heavy_features

                cache[smi] = sf.astype(np.float32)
                all_features.append(cache[smi])
                n_with_surface += 1
            else:
                # Fallback: zeros
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smi)
                n_atoms = mol.GetNumAtoms() if mol else 1
                zeros = np.zeros((n_atoms, N_SURFACE_FEATURES), dtype=np.float32)
                cache[smi] = zeros
                all_features.append(zeros)
        else:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smi)
            n_atoms = mol.GetNumAtoms() if mol else 1
            zeros = np.zeros((n_atoms, N_SURFACE_FEATURES), dtype=np.float32)
            cache[smi] = zeros
            all_features.append(zeros)

    print(f"    {n_with_surface}/{len(set(smiles_list))} unique molecules with surface features")
    return all_features


def main():
    print("=== Chemprop + Per-Atom COSMO Surface Features ===\n")

    pc_dir = "data/pipeline/point_clouds"
    orig_splits = Path("data/processed/splits")
    tmp_dir = Path("data/chemprop_atom_surface")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # STEP 1: Pre-compute and save atom surface features
    # ══════════════════════════════════════════════════════════
    print("Step 1: Pre-computing per-atom surface features...")

    for split in ["train", "val", "test"]:
        df = pd.read_csv(orig_splits / f"{split}.csv")
        smiles_list = df["smiles"].tolist()

        print(f"  {split} ({len(smiles_list)} samples):")
        atom_feats = precompute_surface_features(smiles_list, pc_dir)

        # Verify shapes
        from rdkit import Chem
        for i, (smi, af) in enumerate(zip(smiles_list[:3], atom_feats[:3])):
            mol = Chem.MolFromSmiles(smi)
            print(f"    {smi[:35]:35s}: chemprop_atoms={mol.GetNumAtoms()}, "
                  f"surface_feats={af.shape}, nonzero={af.sum():.2f}")

        # Save as .npz (Chemprop format: one 2D array per molecule, keyed by index)
        npz_dict = {str(i): af for i, af in enumerate(atom_feats)}
        np.savez(tmp_dir / f"{split}_atom_descriptors.npz", **npz_dict)

        # Also save main data and thermo features
        out = pd.DataFrame()
        out["smiles"] = df["smiles"]
        for t in TARGET_COLUMNS:
            out[t] = df[t]
        out.to_csv(tmp_dir / f"{split}.csv", index=False)

        feat_df = df[THERMO_FEATURES].copy()
        feat_df.to_csv(tmp_dir / f"{split}_features.csv", index=False)

    # ══════════════════════════════════════════════════════════
    # STEP 2: Train Chemprop with atom surface features
    # ══════════════════════════════════════════════════════════
    ckpt_base = Path("checkpoints/chemprop_atom_surface")

    variants = [
        ("Chemprop + atom_surface (feature mode)",
         "feature",  # concatenated BEFORE message passing
         "chemprop_as_feat"),
        ("Chemprop + atom_surface (descriptor mode)",
         "descriptor",  # concatenated AFTER message passing
         "chemprop_as_desc"),
    ]

    results = {}

    for name, mode, save_name in variants:
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")

        ckpt_dir = ckpt_base / save_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "chemprop_train",
            "--data_path", str(tmp_dir / "train.csv"),
            "--separate_val_path", str(tmp_dir / "val.csv"),
            "--separate_test_path", str(tmp_dir / "test.csv"),
            "--features_path", str(tmp_dir / "train_features.csv"),
            "--separate_val_features_path", str(tmp_dir / "val_features.csv"),
            "--separate_test_features_path", str(tmp_dir / "test_features.csv"),
            "--atom_descriptors", mode,
            "--atom_descriptors_path", str(tmp_dir / "train_atom_descriptors.npz"),
            "--separate_val_atom_descriptors_path", str(tmp_dir / "val_atom_descriptors.npz"),
            "--separate_test_atom_descriptors_path", str(tmp_dir / "test_atom_descriptors.npz"),
            "--no_atom_descriptor_scaling",
            "--save_dir", str(ckpt_dir),
            "--dataset_type", "regression",
            "--smiles_columns", "smiles",
            "--target_columns", *TARGET_COLUMNS,
            "--epochs", "100",
            "--batch_size", "32",
            "--hidden_size", "300",
            "--depth", "3",
            "--ffn_num_layers", "2",
            "--ffn_hidden_size", "300",
            "--dropout", "0.2",
            "--metric", "rmse",
            "--extra_metrics", "r2", "mae",
            "--seed", "42",
            "--num_folds", "1",
            "--gpu", "0",
            "--quiet",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-500:]}")
            continue

        scores_path = ckpt_dir / "fold_0" / "test_scores.json"
        if scores_path.exists():
            scores = json.load(open(scores_path))
            metrics = {}
            for i, p in enumerate(TARGET_COLUMNS):
                metrics[f"{p}_r2"] = scores["r2"][i]
                metrics[f"{p}_mae"] = scores["mae"][i]
                metrics[f"{p}_rmse"] = scores["rmse"][i]
            metrics["avg_r2"] = np.mean(scores["r2"])

            print(f"\n  Results ({mode} mode):")
            for p in TARGET_COLUMNS:
                print(f"    {p:15s} R² = {metrics[f'{p}_r2']:.4f}")
            print(f"    {'AVERAGE':15s} R² = {metrics['avg_r2']:.4f}")

            results[save_name] = {"name": name, "mode": mode,
                                   "metrics": {k: float(v) for k, v in metrics.items()}}
        else:
            print(f"  WARNING: No scores at {scores_path}")

    # ══════════════════════════════════════════════════════════
    # FINAL COMPARISON
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")

    prev = {}
    for name, path, key in [
        ("Chemprop (base)", "results/chemprop_results.json", "test_metrics"),
        ("PointCloud", "results/pointcloud_results.json", None),
        ("MoE Fix6", "results/moe_fix6_results.json", "metrics"),
    ]:
        try:
            data = json.load(open(path))
            m = data.get(key) if key else None
            if m is None:
                for k in ['metrics', 'test_metrics']:
                    if k in data: m = data[k]; break
            prev[name] = m
        except:
            pass

    header = "  {:<12s}".format("Property")
    for name in prev:
        header += " {:>14s}".format(name)
    for sn, info in results.items():
        header += " {:>14s}".format(info["mode"][:14])
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<12s}".format(p)
        for name in prev:
            line += " {:14.4f}".format(prev[name].get(key, float('nan')))
        for sn, info in results.items():
            line += " {:14.4f}".format(info["metrics"].get(key, float('nan')))
        print(line)

    line = "  {:<12s}".format("AVERAGE")
    for name in prev:
        line += " {:14.4f}".format(prev[name].get('avg_r2', float('nan')))
    for sn, info in results.items():
        line += " {:14.4f}".format(info["metrics"].get('avg_r2', float('nan')))
    print(line)

    # Save
    with open("results/chemprop_atom_surface_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/chemprop_atom_surface_results.json")


if __name__ == "__main__":
    main()
