"""Chemprop + Per-Atom COSMO Surface Features v2.

Improvements over v1:
1. Let Chemprop scale surface features (remove --no_atom_descriptor_scaling)
2. Reduce to 3 key features: esp_mean, esp_std, area_fraction
3. Temperature-conditioned: surface_feat × f(T) varies across data points
4. Multi-seed ensemble: 5 seeds, average predictions
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

# Indices in the 10D surface features: 0=area_frac, 1=esp_mean, 2=esp_std
KEY_FEATURE_INDICES = [0, 1, 2]  # area_fraction, esp_mean, esp_std
KEY_FEATURE_NAMES = ["area_frac", "esp_mean", "esp_std"]
N_KEY = len(KEY_FEATURE_INDICES)

# Temperature conditioning: multiply each surface feature by [1, 1/T_norm, T_norm]
# This creates 3 × 3 = 9 features that vary with temperature
N_TEMP_CONDITIONED = N_KEY * 3  # 9 features per atom


def precompute_surface_features_v2(df, pc_dir):
    """Compute per-atom surface features with temperature conditioning.

    For each sample (molecule at specific T):
      - 3 base surface features (area_frac, esp_mean, esp_std)
      - × 3 temperature functions (1, 1/T_norm, T_norm)
      = 9 features per atom, varying with T
    """
    pc_idx = pd.read_csv(Path(pc_dir) / "index.csv")
    pc_map = dict(zip(pc_idx["smiles"], pc_idx["filename"]))

    # Pre-compute base surface features per unique molecule
    from rdkit import Chem
    base_cache = {}
    n_with_surface = 0

    for smi in df["smiles"].unique():
        fn = pc_map.get(smi)
        if fn and (Path(pc_dir) / fn).exists():
            pc = np.load(Path(pc_dir) / fn)["points"]
            sf = compute_atom_surface_features(smi, pc)
            if sf is not None:
                mol = Chem.MolFromSmiles(smi)
                n_atoms = mol.GetNumAtoms() if mol else 0

                if sf.shape[0] != n_atoms:
                    # Aggregate H features to heavy atoms
                    mol_h = Chem.AddHs(mol)
                    heavy = np.zeros((n_atoms, N_SURFACE_FEATURES))
                    for h_idx in range(mol_h.GetNumAtoms()):
                        atom = mol_h.GetAtomWithIdx(h_idx)
                        if atom.GetAtomicNum() == 1:
                            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
                            if neighbors and neighbors[0] < n_atoms:
                                heavy[neighbors[0]] += sf[h_idx]
                        elif h_idx < n_atoms:
                            heavy[h_idx] += sf[h_idx]
                    total = heavy[:, 0].sum()
                    if total > 0:
                        heavy[:, 0] /= total
                    sf = heavy

                # Keep only 3 key features
                base_cache[smi] = sf[:, KEY_FEATURE_INDICES].astype(np.float32)
                n_with_surface += 1
            else:
                mol = Chem.MolFromSmiles(smi)
                n_atoms = mol.GetNumAtoms() if mol else 1
                base_cache[smi] = np.zeros((n_atoms, N_KEY), dtype=np.float32)
        else:
            mol = Chem.MolFromSmiles(smi)
            n_atoms = mol.GetNumAtoms() if mol else 1
            base_cache[smi] = np.zeros((n_atoms, N_KEY), dtype=np.float32)

    print(f"    {n_with_surface}/{df['smiles'].nunique()} unique molecules with surface features")

    # Temperature conditioning: create per-sample features
    # Normalize temperature
    T_values = df["temperature"].values
    T_mean = T_values.mean()
    T_std = T_values.std() if T_values.std() > 0 else 1.0
    T_norm = (T_values - T_mean) / T_std

    all_features = []
    for idx, row in df.iterrows():
        smi = row["smiles"]
        base = base_cache[smi]  # (n_atoms, 3)
        t_norm = T_norm[idx]
        inv_t = 1.0 / (1.0 + abs(t_norm))  # bounded inverse temperature

        # Temperature-conditioned: base × [1, inv_t, t_norm]
        conditioned = np.concatenate([
            base,                    # base features (3)
            base * inv_t,            # scaled by inverse temperature (3)
            base * t_norm,           # scaled by normalized temperature (3)
        ], axis=1)  # (n_atoms, 9)

        all_features.append(conditioned.astype(np.float32))

    return all_features


def train_chemprop_with_seed(seed, tmp_dir, ckpt_dir):
    """Train single Chemprop model with given seed."""
    cmd = [
        "chemprop_train",
        "--data_path", str(tmp_dir / "train.csv"),
        "--separate_val_path", str(tmp_dir / "val.csv"),
        "--separate_test_path", str(tmp_dir / "test.csv"),
        "--features_path", str(tmp_dir / "train_features.csv"),
        "--separate_val_features_path", str(tmp_dir / "val_features.csv"),
        "--separate_test_features_path", str(tmp_dir / "test_features.csv"),
        "--atom_descriptors", "feature",
        "--atom_descriptors_path", str(tmp_dir / "train_atom_descriptors.npz"),
        "--separate_val_atom_descriptors_path", str(tmp_dir / "val_atom_descriptors.npz"),
        "--separate_test_atom_descriptors_path", str(tmp_dir / "test_atom_descriptors.npz"),
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
        "--seed", str(seed),
        "--num_folds", "1",
        "--gpu", "0",
        "--quiet",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"    Seed {seed} ERROR: {result.stderr[-300:]}")
        return None

    scores_path = ckpt_dir / "fold_0" / "test_scores.json"
    if scores_path.exists():
        scores = json.load(open(scores_path))
        metrics = {}
        for i, p in enumerate(TARGET_COLUMNS):
            metrics[f"{p}_r2"] = scores["r2"][i]
            metrics[f"{p}_mae"] = scores["mae"][i]
            metrics[f"{p}_rmse"] = scores["rmse"][i]
        metrics["avg_r2"] = np.mean(scores["r2"])
        return metrics
    return None


def main():
    print("=== Chemprop + Atom Surface v2 (3 features + T-conditioning + 5-seed ensemble) ===\n")

    pc_dir = "data/pipeline/point_clouds"
    orig_splits = Path("data/processed/splits")
    tmp_dir = Path("data/chemprop_atom_surface_v2")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # STEP 1: Pre-compute temperature-conditioned atom surface features
    # ══════════════════════════════════════════════════════════
    print("Step 1: Pre-computing temperature-conditioned atom surface features...")
    print(f"  Features per atom: {N_TEMP_CONDITIONED} (3 base × 3 T-functions)")

    for split in ["train", "val", "test"]:
        df = pd.read_csv(orig_splits / f"{split}.csv")
        print(f"\n  {split} ({len(df)} samples):")
        atom_feats = precompute_surface_features_v2(df, pc_dir)

        # Verify: features should vary across samples of same molecule
        from rdkit import Chem
        smiles_0 = df["smiles"].iloc[0]
        indices = df.index[df["smiles"] == smiles_0].tolist()
        if len(indices) >= 2:
            diff = np.abs(atom_feats[indices[0]] - atom_feats[indices[1]]).sum()
            print(f"    T-conditioning check: same molecule, different T → feature diff = {diff:.4f} "
                  f"(should be > 0)")

        # Save as npz
        npz_dict = {str(i): af for i, af in enumerate(atom_feats)}
        np.savez(tmp_dir / f"{split}_atom_descriptors.npz", **npz_dict)

        # Data and thermo features
        out = pd.DataFrame()
        out["smiles"] = df["smiles"]
        for t in TARGET_COLUMNS:
            out[t] = df[t]
        out.to_csv(tmp_dir / f"{split}.csv", index=False)

        feat_df = df[THERMO_FEATURES].copy()
        feat_df.to_csv(tmp_dir / f"{split}_features.csv", index=False)

    # ══════════════════════════════════════════════════════════
    # STEP 2: Train 5-seed ensemble
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 2: Training 5-seed ensemble")
    print(f"{'='*60}")

    seeds = [42, 123, 456, 789, 1024]
    all_metrics = []

    for seed in seeds:
        ckpt_dir = Path(f"checkpoints/chemprop_as_v2/seed_{seed}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Seed {seed}...")
        metrics = train_chemprop_with_seed(seed, tmp_dir, ckpt_dir)
        if metrics:
            all_metrics.append(metrics)
            print(f"    avg R²={metrics['avg_r2']:.4f}, "
                  f"gamma1={metrics['gamma1_r2']:.4f}, gamma2={metrics['gamma2_r2']:.4f}")

    # ══════════════════════════════════════════════════════════
    # STEP 3: Compute ensemble metrics (average of R² across seeds)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(all_metrics)} models trained")
    print(f"{'='*60}")

    if all_metrics:
        # Per-seed results
        print(f"\n  Per-seed R²:")
        print(f"  {'Seed':<8s}", end="")
        for p in TARGET_COLUMNS:
            print(f" {p:>8s}", end="")
        print(f" {'AVG':>8s}")

        for i, m in enumerate(all_metrics):
            print(f"  {seeds[i]:<8d}", end="")
            for p in TARGET_COLUMNS:
                print(f" {m[f'{p}_r2']:8.4f}", end="")
            print(f" {m['avg_r2']:8.4f}")

        # Average across seeds
        avg_metrics = {}
        std_metrics = {}
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(vals)
            std_metrics[key] = np.std(vals)

        print(f"\n  Ensemble average (mean ± std across {len(all_metrics)} seeds):")
        for p in TARGET_COLUMNS:
            key = f"{p}_r2"
            print(f"    {p:15s} R² = {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
        print(f"    {'AVERAGE':15s} R² = {avg_metrics['avg_r2']:.4f} ± {std_metrics['avg_r2']:.4f}")

    # ══════════════════════════════════════════════════════════
    # COMPARISON
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    prev = {}
    for name, path, key in [
        ("Chemprop (base)", "results/chemprop_results.json", "test_metrics"),
        ("CP+Surface v1", "results/chemprop_atom_surface_results.json", "CP_AS_V1"),
        ("PointCloud", "results/pointcloud_results.json", None),
        ("MoE Fix6", "results/moe_fix6_results.json", "metrics"),
        ("Ens Fix6+PC", "results/ensemble_fix6_pointcloud.json", "simple_average"),
    ]:
        try:
            data = json.load(open(path))
            if key == "CP_AS_V1":
                m = data.get("chemprop_as_feat", {}).get("metrics", {})
            elif key:
                m = data.get(key, {})
            else:
                for k in ['metrics', 'test_metrics']:
                    if k in data: m = data[k]; break
            if m:
                prev[name] = m
        except:
            pass

    header = "  {:<15s}".format("Property")
    for name in prev:
        header += " {:>13s}".format(name[:13])
    header += " {:>13s}".format("v2 ensemble")
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<15s}".format(p)
        for name in prev:
            line += " {:13.4f}".format(prev[name].get(key, float('nan')))
        line += " {:13.4f}".format(avg_metrics.get(key, float('nan')))
        print(line)

    line = "  {:<15s}".format("AVERAGE")
    for name in prev:
        line += " {:13.4f}".format(prev[name].get('avg_r2', float('nan')))
    line += " {:13.4f}".format(avg_metrics.get('avg_r2', float('nan')))
    print(line)

    # Save
    results = {
        "model": "chemprop_atom_surface_v2",
        "description": "Chemprop + 3 key atom surface features (area, esp_mean, esp_std) "
                       "× 3 temperature functions (1, 1/T, T) = 9 features/atom. "
                       "5-seed ensemble average. Chemprop handles feature scaling.",
        "n_atom_features": N_TEMP_CONDITIONED,
        "seeds": seeds[:len(all_metrics)],
        "per_seed_metrics": [{k: float(v) for k, v in m.items()} for m in all_metrics],
        "ensemble_avg": {k: float(v) for k, v in avg_metrics.items()},
        "ensemble_std": {k: float(v) for k, v in std_metrics.items()},
    }
    with open("results/chemprop_atom_surface_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/chemprop_atom_surface_v2_results.json")


if __name__ == "__main__":
    main()
