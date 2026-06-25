"""COSMOBridge v3 Prediction Pipeline for Novel Ionic Liquids.

Complete pipeline:
1. Generate SMILES for novel cation-anion combinations
2. Compute COSMO-style point clouds (Gasteiger ESP approximation)
3. Extract Chemprop MPN fingerprints (frozen encoder)
4. Extract PointNet surface features (frozen encoder)
5. Run COSMOBridge v3 (frozen fusion + frozen Chemprop readout + learned gates)
6. Inverse-transform predictions to raw physical units
7. Rank by Derringer-Suich desirability
8. Output top candidates with full property predictions

Usage:
    python scripts/predict_cosmobridge_v3.py
"""

import sys
import json
import subprocess
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.data.graph_builder import smiles_to_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.fusion.cosmobridge import COSMOBridge
from src.models.fusion.multimodal_pointcloud import MultimodalPointCloudModel
from src.training.metrics import compute_metrics
from scripts.train_chemprop_gbh_hybrid import ChempropGBHFusion
from scripts.rank_ils_desirability import (
    compute_desirability, is_pareto_optimal, TRAIN_STATS,
    desirability_minimize, desirability_target)

from rdkit import Chem
from rdkit.Chem import AllChem
from chemprop.utils import load_checkpoint as load_chemprop

# ══════════════════════════════════════════════════════════
# CANDIDATE IONIC LIQUIDS
# ══════════════════════════════════════════════════════════
CATIONS = {
    "TMG": ("CN(C)C(=[NH2+])N(C)C", "Tetramethylguanidinium", True, 5),
    "DEA": ("C(CO)[NH2+]CCO", "Diethanolammonium", True, 4),
    "TEtA": ("CC[NH+](CC)CC", "Triethylammonium", True, 5),
    "EMPy": ("CC[n+]1cccc(C)c1", "1-Ethyl-3-methylpyridinium", True, 3),
    "HMIM": ("CCCCCC[n+]1ccn(C)c1", "1-Hexyl-3-methylimidazolium", True, 3),
    "MIM": ("C[n+]1ccn(C)c1", "1,3-Dimethylimidazolium", True, 4),
    "DBUH": ("C1CCC2=[NH+]CCC2C1", "DBU-H", True, 3),
    "PyrrH": ("C1CC[NH2+]C1", "Pyrrolidinium", True, 4),
    "BMIM": ("CCCCn1cc[n+](C)c1", "1-Butyl-3-methylimidazolium", False, 4),
    "EMIM": ("CCn1cc[n+](C)c1", "1-Ethyl-3-methylimidazolium", False, 4),
    "Ch": ("C[N+](C)(C)CCO", "Cholinium", False, 5),
}

ANIONS = {
    "For": ("C(=O)[O-]", "Formate", True, 5),
    "Pro": ("CCC(=O)[O-]", "Propanoate", True, 5),
    "Gly": ("NCC(=O)[O-]", "Glycinate", True, 4),
    "Lev": ("CC(=O)CCC(=O)[O-]", "Levulinate", True, 3),
    "DCA": ("N#C[N-]C#N", "Dicyanamide", True, 3),
    "TFA": ("FC(F)(F)C(=O)[O-]", "Trifluoroacetate", True, 4),
    "OAc": ("CC(=O)[O-]", "Acetate", False, 5),
    "Lac": ("CC(O)C(=O)[O-]", "Lactate", False, 5),
    "Cl": ("[Cl-]", "Chloride", False, 5),
    "HSO4": ("OS(=O)(=O)[O-]", "Hydrogen sulfate", False, 4),
}

THERMO_FEATURES = ["temperature", "x1", "inv_temperature", "temp_squared", "temp_cubed"]


def generate_point_cloud(smiles, n_points=1024):
    """Generate approximate COSMO point cloud from SMILES using Gasteiger charges."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_points, 7), dtype=np.float32)

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except:
        pass

    AllChem.ComputeGasteigerCharges(mol)
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    vdw = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 16: 1.80, 17: 1.75, 9: 1.47, 35: 1.85}
    atom_pos = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                           conf.GetAtomPosition(i).z] for i in range(n_atoms)])
    atom_radii = np.array([vdw.get(mol.GetAtomWithIdx(i).GetAtomicNum(), 1.70) for i in range(n_atoms)])
    atom_charges = []
    for i in range(n_atoms):
        c = float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge'))
        atom_charges.append(0.0 if np.isnan(c) else c)
    atom_charges = np.array(atom_charges)

    # Generate surface points
    all_pts = []
    probe = 0.42
    for i in range(n_atoms):
        radius = atom_radii[i] + probe
        n = max(30, int(n_points * 2 * radius**2 / max(sum(r**2 for r in atom_radii), 1)))
        phi = np.random.uniform(0, 2*np.pi, n)
        costheta = np.random.uniform(-1, 1, n)
        theta = np.arccos(costheta)
        pts = np.column_stack([
            atom_pos[i,0] + radius*np.sin(theta)*np.cos(phi),
            atom_pos[i,1] + radius*np.sin(theta)*np.sin(phi),
            atom_pos[i,2] + radius*np.cos(theta)])

        keep = np.ones(len(pts), dtype=bool)
        for j in range(n_atoms):
            if i == j: continue
            keep &= np.linalg.norm(pts - atom_pos[j], axis=1) > (atom_radii[j] + probe) * 0.9
        pts = pts[keep]
        if len(pts) == 0: continue

        normals = pts - atom_pos[i]
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        esp = np.zeros(len(pts))
        for j in range(n_atoms):
            d = np.clip(np.linalg.norm(pts - atom_pos[j], axis=1), 0.5, None)
            esp += atom_charges[j] / d
        all_pts.append(np.column_stack([pts, normals, esp]))

    if not all_pts:
        return np.zeros((n_points, 7), dtype=np.float32)

    all_pts = np.concatenate(all_pts)
    if len(all_pts) > n_points:
        idx = np.random.choice(len(all_pts), n_points, replace=False)
        all_pts = all_pts[idx]
    elif len(all_pts) < n_points:
        extra = np.random.choice(len(all_pts), n_points - len(all_pts), replace=True)
        all_pts = np.concatenate([all_pts, all_pts[extra]])

    # Normalize
    center = all_pts[:, :3].mean(axis=0)
    all_pts[:, :3] -= center
    scale = max(np.abs(all_pts[:, :3]).max(), 1e-6)
    all_pts[:, :3] /= scale

    return all_pts.astype(np.float32)


def main():
    set_seed(42)
    device = get_device(load_config("configs/default.yaml"))
    print(f"Device: {device}")

    # Load scalers
    with open("data/processed/target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)
    with open("data/processed/feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)

    # ══════════════════════════════════════════════════════════
    # Step 1: Generate candidates
    # ══════════════════════════════════════════════════════════
    print("\nStep 1: Generating IL candidates...\n")
    T = 348.15; x1 = 0.5
    candidates = []
    for ck, (cs, cn, cnovel, csynth) in CATIONS.items():
        for ak, (asmi, an, anovel, asynth) in ANIONS.items():
            smi = f"{cs}.{asmi}"
            mol = Chem.MolFromSmiles(smi)
            if mol is None: continue
            candidates.append({
                "smiles": smi, "name": f"{cn} {an.lower()}",
                "cation": cn, "anion": an,
                "novel": "NEW" if (cnovel and anovel) else ("new" if (cnovel or anovel) else ""),
                "fully_novel": cnovel and anovel,
                "synth_score": (csynth + asynth) / 2,
            })
    print(f"  {len(candidates)} valid candidates")

    # ══════════════════════════════════════════════════════════
    # Step 2: Generate COSMO point clouds
    # ══════════════════════════════════════════════════════════
    print("\nStep 2: Generating COSMO point clouds...")
    pc_cache = {}
    for c in candidates:
        if c["smiles"] not in pc_cache:
            pc_cache[c["smiles"]] = generate_point_cloud(c["smiles"])
    print(f"  Generated {len(pc_cache)} unique point clouds")

    # ══════════════════════════════════════════════════════════
    # Step 3: Extract PointNet surface features
    # ══════════════════════════════════════════════════════════
    print("\nStep 3: Extracting PointNet features...")
    config = load_config("configs/default.yaml")
    pc_model = MultimodalPointCloudModel(config=config, pretrained_gnn_path=None)
    ckpt = torch.load("checkpoints/pointcloud/best_model.pt", map_location=device, weights_only=False)
    pc_model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    pc_model.to(device).eval()

    surface_cache = {}
    with torch.no_grad():
        for smi, pc in pc_cache.items():
            pc_tensor = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).to(device)
            sf = pc_model.pointnet(pc_tensor).cpu().numpy()[0]
            surface_cache[smi] = sf
    print(f"  Extracted {len(surface_cache)} surface feature vectors (256D)")

    # ══════════════════════════════════════════════════════════
    # Step 4: Extract Chemprop fingerprints
    # ══════════════════════════════════════════════════════════
    print("\nStep 4: Extracting Chemprop fingerprints...")

    # Prepare temp CSV for Chemprop
    tmp_dir = Path("data/cosmobridge_predict_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Normalize thermo features
    thermo_raw = np.array([T, x1, 1/T, T**2, T**3])
    thermo_norm = (thermo_raw - feature_scaler.mean_[:5]) / feature_scaler.scale_[:5]

    unique_smiles = list(pc_cache.keys())
    pred_df = pd.DataFrame({"smiles": unique_smiles})
    for t in TARGET_COLUMNS: pred_df[t] = ""
    pred_df.to_csv(tmp_dir / "predict.csv", index=False)

    feat_df = pd.DataFrame([thermo_norm[:5]] * len(unique_smiles),
                            columns=THERMO_FEATURES)
    feat_df.to_csv(tmp_dir / "predict_features.csv", index=False)

    out_fp = tempfile.mktemp(suffix=".csv")
    subprocess.run(["chemprop_fingerprint",
                     "--test_path", str(tmp_dir / "predict.csv"),
                     "--features_path", str(tmp_dir / "predict_features.csv"),
                     "--checkpoint_dir", "checkpoints/chemprop",
                     "--fingerprint_type", "MPN",
                     "--preds_path", out_fp],
                    capture_output=True, text=True, timeout=120)

    graph_df = pd.read_csv(out_fp).select_dtypes(include=[np.number])
    graph_cache = {smi: graph_df.iloc[i].values.astype(np.float32)
                   for i, smi in enumerate(unique_smiles)}
    graph_dim = graph_df.shape[1]
    print(f"  Extracted {len(graph_cache)} graph fingerprints ({graph_dim}D)")

    # ══════════════════════════════════════════════════════════
    # Step 5: Get Chemprop full predictions (for Path B)
    # ══════════════════════════════════════════════════════════
    print("\nStep 5: Getting Chemprop full predictions...")
    out_pred = tempfile.mktemp(suffix=".csv")
    subprocess.run(["chemprop_predict",
                     "--test_path", str(tmp_dir / "predict.csv"),
                     "--features_path", str(tmp_dir / "predict_features.csv"),
                     "--checkpoint_dir", "checkpoints/chemprop",
                     "--preds_path", out_pred],
                    capture_output=True, text=True, timeout=120)
    chemprop_preds = pd.read_csv(out_pred)
    chemprop_cache = {smi: chemprop_preds[TARGET_COLUMNS].iloc[i].values.astype(np.float32)
                      for i, smi in enumerate(unique_smiles)}
    print(f"  Got Chemprop predictions for {len(chemprop_cache)} molecules")

    # ══════════════════════════════════════════════════════════
    # Step 6: Load COSMOBridge v3 and predict
    # ══════════════════════════════════════════════════════════
    print("\nStep 6: Running COSMOBridge v3...")

    # Load fusion model
    fusion_model = ChempropGBHFusion(graph_dim=graph_dim, surface_dim=256,
                                       thermo_dim=len(FEATURE_COLUMNS), fused_dim=256,
                                       rank=32, hyper_hidden=64, dropout=0.3)
    fusion_model.load_state_dict(torch.load("checkpoints/chemprop_gbh_hybrid/best.pt",
                                              map_location=device, weights_only=True))
    fusion_model.to(device).eval()

    # Load gates
    cosmobridge_ckpt = torch.load("checkpoints/cosmobridge_final_v2/s3.pt",
                                    map_location=device, weights_only=True)
    gate_logits = cosmobridge_ckpt.get("gate_logits",
                  torch.tensor([2.0, 2.0, -2.0, -2.0, -2.0, 0.0, 1.5]))
    gates = torch.sigmoid(gate_logits).cpu().numpy()
    print(f"  Gates: {' '.join(f'{g:.2f}' for g in gates)}")

    # Build full thermo feature vector (25D)
    full_thermo_norm = np.zeros(len(FEATURE_COLUMNS))
    full_thermo_norm[:5] = thermo_norm[:5]
    # Surface descriptors would be filled per-molecule if available

    # Predict for all candidates
    results = []
    for c in candidates:
        smi = c["smiles"]
        g_feat = torch.tensor(graph_cache[smi], dtype=torch.float32).unsqueeze(0).to(device)
        s_feat = torch.tensor(surface_cache[smi], dtype=torch.float32).unsqueeze(0).to(device)
        t_feat = torch.tensor(full_thermo_norm, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            preds_fused = fusion_model(g_feat, s_feat, t_feat).cpu().numpy()[0]

        preds_chemprop = chemprop_cache[smi]

        # Apply gates
        preds_routed = gates * preds_fused + (1 - gates) * preds_chemprop

        # Inverse transform to raw
        props = {}
        for i, col in enumerate(TARGET_COLUMNS):
            try:
                raw = float(preds_routed[i]) * target_scaler.scale_[i] + target_scaler.mean_[i]
                props[col] = raw
            except:
                props[col] = None

        # Desirability
        d_values, D, weights = compute_desirability(props, c["synth_score"])

        results.append({
            **c, **{f"{t}_pred": props.get(t) for t in TARGET_COLUMNS},
            "D_combined": D,
            **{f"d_{k}": v for k, v in d_values.items()},
        })

    results_df = pd.DataFrame(results).sort_values("D_combined", ascending=False)

    # ══════════════════════════════════════════════════════════
    # Step 7: Output rankings
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("COSMOBridge v3 — TOP 15 IL CANDIDATES (Derringer-Suich Desirability)")
    print(f"{'='*110}")
    print(f"{'Rank':>4s}  {'Ionic Liquid':<40s} {'Nov':>3s} {'γ₁':>7s} {'γ₂':>7s} "
          f"{'G_mix':>7s} {'H_vap':>7s} {'P':>7s} {'D':>6s}")
    print("-" * 110)

    top15 = results_df.head(15)
    for rank, (_, row) in enumerate(top15.iterrows(), 1):
        g1 = f"{row.get('gamma1_pred',0):.3f}" if row.get('gamma1_pred') else "?"
        g2 = f"{row.get('gamma2_pred',0):.3f}" if row.get('gamma2_pred') else "?"
        gm = f"{row.get('G_mix_pred',0):.3f}" if row.get('G_mix_pred') else "?"
        hv = f"{row.get('H_vap_pred',0):.2f}" if row.get('H_vap_pred') else "?"
        p = f"{row.get('P_pred',0):.3f}" if row.get('P_pred') else "?"
        print(f"{rank:4d}  {row['name']:<40s} {row.get('novel',''):>3s} {g1:>7s} {g2:>7s} "
              f"{gm:>7s} {hv:>7s} {p:>7s} {row['D_combined']:6.3f}")

    # Pareto front
    valid = results_df.dropna(subset=["gamma1_pred", "G_mix_pred", "P_pred"])
    if len(valid) > 0:
        costs = valid[["gamma1_pred", "G_mix_pred", "P_pred"]].values
        pareto = is_pareto_optimal(costs)
        print(f"\n  Pareto-optimal ({pareto.sum()}/{len(valid)}):")
        for _, row in valid[pareto].iterrows():
            print(f"    {row['name']:<40s} γ₁={row['gamma1_pred']:.3f} "
                  f"G_mix={row['G_mix_pred']:.3f} P={row['P_pred']:.3f} D={row['D_combined']:.3f}")

    # Detailed top 3
    print(f"\n{'='*110}")
    print("DETAILED TOP 3")
    print(f"{'='*110}")
    for rank, (_, row) in enumerate(top15.head(3).iterrows(), 1):
        print(f"\n  #{rank}: {row['name']}")
        print(f"  SMILES: {row['smiles']}")
        print(f"  Novelty: {'Fully novel' if row.get('fully_novel') else 'Partially novel' if row.get('novel') else 'Known'}")
        print(f"  Synthetic accessibility: {row.get('synth_score',0):.1f}/5")
        print(f"  Combined desirability D = {row['D_combined']:.3f}")
        print(f"  Predicted properties (T={T:.0f}K, x₁={x1}):")
        for t in TARGET_COLUMNS:
            v = row.get(f"{t}_pred")
            if v is not None:
                print(f"    {t:15s}: {v:.4f}")

    # Save
    results_df.to_csv("results/cosmobridge_v3_candidates.csv", index=False)
    top15.to_csv("results/cosmobridge_v3_top15.csv", index=False)

    summary = {
        "model": "COSMOBridge_v3",
        "n_candidates": len(results_df),
        "gates_used": {p: float(g) for p, g in zip(TARGET_COLUMNS, gates)},
        "top_5": [
            {"rank": i+1, "name": row["name"], "D": float(row["D_combined"]),
             "gamma1": row.get("gamma1_pred"), "G_mix": row.get("G_mix_pred"),
             "P": row.get("P_pred")}
            for i, (_, row) in enumerate(top15.head(5).iterrows())
        ],
    }
    with open("results/cosmobridge_v3_screening.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved: results/cosmobridge_v3_candidates.csv ({len(results_df)} candidates)")
    print(f"Saved: results/cosmobridge_v3_top15.csv")
    print(f"Saved: results/cosmobridge_v3_screening.json")


if __name__ == "__main__":
    main()
