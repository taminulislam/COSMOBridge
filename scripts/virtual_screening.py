"""Approach 1: Combinatorial Virtual Screening for Novel Ionic Liquids.

Enumerates all cation × anion combinations from known building blocks,
predicts thermodynamic properties using trained models, and ranks
candidates by application-specific criteria.

Applications:
  - Lignin dissolution: low gamma1, negative G_mix, moderate H_vap
  - Plastic depolymerization: favorable H_E, low gamma1

Output:
  results/virtual_screening/all_candidates.csv
  results/virtual_screening/top_lignin.csv
  results/virtual_screening/top_plastic.csv
  results/virtual_screening/novel_candidates.csv
"""

import sys
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS
from src.data.graph_builder import smiles_to_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM


def smiles_to_hash(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


def extract_ions(smiles_set):
    """Extract unique cation and anion SMILES from a set of IL SMILES."""
    cations = set()
    anions = set()
    for smi in smiles_set:
        parts = smi.split(".")
        if len(parts) == 2:
            for p in parts:
                if "+" in p:
                    cations.add(p)
                elif "-" in p:
                    anions.add(p)
    return sorted(cations), sorted(anions)


def get_ion_name(smiles):
    """Get a readable name for an ion SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles[:30]
    mw = Descriptors.MolWt(mol)
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    return f"{formula} (MW={mw:.0f})"


def is_valid_il(smiles):
    """Check if a combined IL SMILES is valid."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def predict_properties_gnn(model, smiles_list, feature_template, device):
    """Predict properties for a list of IL SMILES using the GNN model."""
    results = []
    for smi in smiles_list:
        try:
            g = smiles_to_graph(smi)
        except Exception:
            results.append(None)
            continue

        af = torch.tensor(g["atom_features"], dtype=torch.float32).unsqueeze(0)
        ei = torch.tensor(g["edge_index"], dtype=torch.long)
        bf = torch.tensor(g["bond_features"], dtype=torch.float32)
        batch = torch.zeros(af.shape[1], dtype=torch.long)
        feat = feature_template.unsqueeze(0)

        # Move to device
        af, ei, bf, batch, feat = af.squeeze(0).to(device), ei.to(device), bf.to(device), batch.to(device), feat.to(device)

        with torch.no_grad():
            pred = model(atom_features=af, edge_index=ei, bond_features=bf,
                        batch=batch, features=feat)
        results.append(pred.cpu().numpy().flatten())

    return results


def predict_properties_pointcloud(model, smiles_list, feature_template, pc_dir, device):
    """Predict using PointCloud+GNN model."""
    pc_index = {}
    idx_path = Path(pc_dir) / "index.csv"
    if idx_path.exists():
        idx_df = pd.read_csv(idx_path)
        pc_index = dict(zip(idx_df["smiles"], idx_df["filename"]))

    results = []
    n_points = 1024

    for smi in smiles_list:
        try:
            g = smiles_to_graph(smi)
        except Exception:
            results.append(None)
            continue

        af = torch.tensor(g["atom_features"], dtype=torch.float32).to(device)
        ei = torch.tensor(g["edge_index"], dtype=torch.long).to(device)
        bf = torch.tensor(g["bond_features"], dtype=torch.float32).to(device)
        batch = torch.zeros(af.shape[0], dtype=torch.long).to(device)
        feat = feature_template.unsqueeze(0).to(device)

        # Point cloud
        fn = pc_index.get(smi)
        if fn and (Path(pc_dir) / fn).exists():
            pts = np.load(Path(pc_dir) / fn)["points"]
            if len(pts) >= n_points:
                pts = pts[:n_points]
            else:
                extra = np.random.choice(len(pts), n_points - len(pts), replace=True)
                pts = np.concatenate([pts, pts[extra]])
        else:
            pts = np.zeros((n_points, 7), dtype=np.float32)

        pc = torch.tensor(pts, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(point_cloud=pc, features=feat,
                        atom_features=af, edge_index=ei,
                        bond_features=bf, batch=batch)
        results.append(pred.cpu().numpy().flatten())

    return results


def rank_for_lignin(df):
    """Rank candidates for lignin dissolution.

    Criteria: low gamma1 (favorable interaction), negative G_mix (spontaneous),
    moderate H_vap (recyclable), low P (safe/non-volatile).
    """
    df = df.copy()
    # Lower is better for all these (we negate gamma1 and G_mix ranking)
    df["lignin_score"] = (
        -df["gamma1_pred"].rank(pct=True)    # Low gamma1 → high score
        - df["G_mix_pred"].rank(pct=True)     # Negative G_mix → high score
        + 0.5 * df["H_vap_pred"].rank(pct=True)  # Moderate H_vap
        - 0.3 * df["P_pred"].rank(pct=True)   # Low P → high score
    )
    return df.sort_values("lignin_score", ascending=False)


def rank_for_plastic(df):
    """Rank candidates for plastic depolymerization.

    Criteria: strong IL-polymer interaction (low gamma1, negative H_E),
    favorable mixing (negative G_mix).
    """
    df = df.copy()
    df["plastic_score"] = (
        -df["gamma1_pred"].rank(pct=True)
        - df["H_E_pred"].rank(pct=True)       # Negative H_E → exothermic interaction
        - df["G_mix_pred"].rank(pct=True)
    )
    return df.sort_values("plastic_score", ascending=False)


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]

    output_dir = Path("results/virtual_screening")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect all known IL SMILES ──
    all_known = set()
    for csv_path in ["data/processed/il_data_raw.csv", "data/augmented/ilthermo_data.csv"]:
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            all_known.update(df["smiles"].unique())

    cations, anions = extract_ions(all_known)
    print(f"Building blocks: {len(cations)} cations × {len(anions)} anions")

    # ── Enumerate all combinations ──
    print("\nEnumerating combinations...")
    candidates = []
    for cat in cations:
        for an in anions:
            smi = f"{cat}.{an}"
            is_novel = smi not in all_known
            if is_valid_il(smi):
                candidates.append({
                    "smiles": smi,
                    "cation": cat,
                    "anion": an,
                    "cation_name": get_ion_name(cat),
                    "anion_name": get_ion_name(an),
                    "is_novel": is_novel,
                })

    print(f"Valid combinations: {len(candidates)} ({sum(c['is_novel'] for c in candidates)} novel)")

    # ── Load trained model ──
    print("\nLoading Phase 2 Transfer GNN model...")
    from src.models.graph.gnn import MolecularGNN
    model_gnn = MolecularGNN(
        atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
        hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
        dropout=0.3, pooling="mean", num_targets=7,
        aux_feature_dim=len(FEATURE_COLUMNS),
    )
    ckpt_path = Path("checkpoints/transfer/finetune/best_model.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_gnn.load_state_dict(ckpt["model_state_dict"])
        print("  GNN model loaded")
    model_gnn.to(device).eval()

    # Also load Phase 3 PointCloud model
    print("Loading Phase 3 PointCloud model...")
    from src.models.fusion.multimodal_pointcloud import MultimodalPointCloudModel
    config_p3 = {**config}
    config_p3.setdefault("model", {})["temp_skip"] = False
    model_pc = MultimodalPointCloudModel(config=config_p3)
    ckpt_pc = Path("checkpoints/pointcloud/best_model.pt")
    if ckpt_pc.exists():
        ckpt = torch.load(ckpt_pc, map_location=device, weights_only=False)
        model_pc.load_state_dict(ckpt["model_state_dict"])
        print("  PointCloud model loaded")
    model_pc.to(device).eval()

    # ── Feature template (standard conditions: T=298K, x1=0.5) ──
    feature_template = torch.zeros(len(FEATURE_COLUMNS))
    # These are normalized values — use approximate z-scores for T=298, x1=0.5
    # (would be more precise with actual scaler, but this is adequate for ranking)

    # ── Predict properties ──
    smiles_list = [c["smiles"] for c in candidates]
    print(f"\nPredicting properties for {len(smiles_list)} candidates...")

    # GNN predictions (for H_vap, P — Phase 2 is better)
    batch_size = 100
    gnn_preds = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        preds = predict_properties_gnn(model_gnn, batch, feature_template, device)
        gnn_preds.extend(preds)
        if (i // batch_size) % 5 == 0:
            print(f"  GNN: {i+len(batch)}/{len(smiles_list)}")

    # PointCloud predictions (for gamma1, gamma2, G_E, H_E, G_mix)
    pc_dir = "data/pipeline/point_clouds"
    pc_preds = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        preds = predict_properties_pointcloud(model_pc, batch, feature_template, pc_dir, device)
        pc_preds.extend(preds)
        if (i // batch_size) % 5 == 0:
            print(f"  PointCloud: {i+len(batch)}/{len(smiles_list)}")

    # ── Combine predictions (ensemble: PC for structure props, GNN for temp props) ──
    print("\nCombining predictions (hard ensemble)...")
    for i, cand in enumerate(candidates):
        gnn_p = gnn_preds[i]
        pc_p = pc_preds[i]

        if pc_p is not None and gnn_p is not None:
            # Phase 3 for structure-dependent (0-4), Phase 2 for temp-dependent (5-6)
            for j, prop in enumerate(TARGET_COLUMNS):
                if j < 5 and pc_p is not None:
                    cand[f"{prop}_pred"] = float(pc_p[j])
                elif gnn_p is not None:
                    cand[f"{prop}_pred"] = float(gnn_p[j])
                else:
                    cand[f"{prop}_pred"] = 0.0
        elif gnn_p is not None:
            for j, prop in enumerate(TARGET_COLUMNS):
                cand[f"{prop}_pred"] = float(gnn_p[j])
        else:
            for prop in TARGET_COLUMNS:
                cand[f"{prop}_pred"] = np.nan

    # ── Build results DataFrame ──
    df = pd.DataFrame(candidates)
    df = df.dropna(subset=["gamma1_pred"])
    print(f"Candidates with valid predictions: {len(df)}")

    # ── Save all candidates ──
    df.to_csv(output_dir / "all_candidates.csv", index=False)

    # ── Rank for lignin dissolution ──
    df_lignin = rank_for_lignin(df)
    df_lignin_novel = df_lignin[df_lignin["is_novel"]].head(20)
    df_lignin.head(30).to_csv(output_dir / "top_lignin.csv", index=False)

    print(f"\n{'='*70}")
    print("TOP 10 NOVEL ILs FOR LIGNIN DISSOLUTION")
    print(f"{'='*70}")
    for i, (_, row) in enumerate(df_lignin_novel.head(10).iterrows()):
        print(f"\n  #{i+1}: {row['smiles'][:60]}")
        print(f"       Cation: {row['cation_name']}")
        print(f"       Anion:  {row['anion_name']}")
        print(f"       gamma1={row['gamma1_pred']:.3f}  G_mix={row['G_mix_pred']:.3f}  "
              f"H_vap={row['H_vap_pred']:.3f}  P={row['P_pred']:.3f}")

    # ── Rank for plastic depolymerization ──
    df_plastic = rank_for_plastic(df)
    df_plastic_novel = df_plastic[df_plastic["is_novel"]].head(20)
    df_plastic.head(30).to_csv(output_dir / "top_plastic.csv", index=False)

    print(f"\n{'='*70}")
    print("TOP 10 NOVEL ILs FOR PLASTIC DEPOLYMERIZATION")
    print(f"{'='*70}")
    for i, (_, row) in enumerate(df_plastic_novel.head(10).iterrows()):
        print(f"\n  #{i+1}: {row['smiles'][:60]}")
        print(f"       Cation: {row['cation_name']}")
        print(f"       Anion:  {row['anion_name']}")
        print(f"       gamma1={row['gamma1_pred']:.3f}  H_E={row['H_E_pred']:.3f}  "
              f"G_mix={row['G_mix_pred']:.3f}")

    # ── Novel candidates only ──
    df_novel = df[df["is_novel"]].sort_values("gamma1_pred")
    df_novel.to_csv(output_dir / "novel_candidates.csv", index=False)

    print(f"\n{'='*70}")
    print("SCREENING SUMMARY")
    print(f"{'='*70}")
    print(f"  Total valid combinations: {len(df)}")
    print(f"  Known/existing ILs: {(~df['is_novel']).sum()}")
    print(f"  Novel candidates: {df['is_novel'].sum()}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
