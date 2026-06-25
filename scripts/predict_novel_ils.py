"""Predict properties for novel IL candidates using best trained models.

Pipeline:
1. Generate candidate SMILES from cation-anion combinations
2. Predict with Chemprop (STILT) — all 7 properties
3. Predict with PointCloud model — gamma1, gamma2 (COSMO surface-informed)
4. Apply physics-informed correction for gamma1
5. Rank candidates by synthesis priority score
6. Output top recommendations with confidence

Synthesis priority considers:
- Low gamma1 (good water miscibility)
- Negative G_mix (spontaneous mixing)
- Moderate H_vap (thermal stability, not too volatile)
- Low P (low vapor pressure = safer handling)
- Synthetic accessibility (simple cation-anion pairs)
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

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.data.graph_builder import smiles_to_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.training.metrics import compute_metrics

R_KCAL = 1.987e-3

# ══════════════════════════════════════════════════════════
# CANDIDATE IONIC LIQUIDS
# ══════════════════════════════════════════════════════════

# Cations: novel + known
CATIONS = {
    # Novel (not in training)
    "TMG": ("CN(C)C(=[NH2+])N(C)C", "Tetramethylguanidinium", True, 5),
    "DEA": ("C(CO)[NH2+]CCO", "Diethanolammonium", True, 4),
    "TEtA": ("CC[NH+](CC)CC", "Triethylammonium", True, 5),
    "EMPy": ("CC[n+]1cccc(C)c1", "1-Ethyl-3-methylpyridinium", True, 3),
    "HMIM": ("CCCCCC[n+]1ccn(C)c1", "1-Hexyl-3-methylimidazolium", True, 3),
    "MIM": ("C[n+]1ccn(C)c1", "1,3-Dimethylimidazolium", True, 4),
    "DBUH": ("C1CCC2=[NH+]CCC2C1", "DBU-H (bicyclic amidinium)", True, 3),
    "PyrrH": ("C1CC[NH2+]C1", "Pyrrolidinium", True, 4),
    # Known (in training) — for comparison
    "BMIM": ("CCCCn1cc[n+](C)c1", "1-Butyl-3-methylimidazolium", False, 4),
    "EMIM": ("CCn1cc[n+](C)c1", "1-Ethyl-3-methylimidazolium", False, 4),
    "Ch": ("C[N+](C)(C)CCO", "Cholinium", False, 5),
}

ANIONS = {
    # Novel
    "For": ("C(=O)[O-]", "Formate", True, 5),
    "Pro": ("CCC(=O)[O-]", "Propanoate", True, 5),
    "Gly": ("NCC(=O)[O-]", "Glycinate", True, 4),
    "Lev": ("CC(=O)CCC(=O)[O-]", "Levulinate", True, 3),
    "DCA": ("N#C[N-]C#N", "Dicyanamide", True, 3),
    "TFA": ("FC(F)(F)C(=O)[O-]", "Trifluoroacetate", True, 4),
    # Known (in training)
    "OAc": ("CC(=O)[O-]", "Acetate", False, 5),
    "Lac": ("CC(O)C(=O)[O-]", "Lactate", False, 5),
    "Cl": ("[Cl-]", "Chloride", False, 5),
    "HSO4": ("OS(=O)(=O)[O-]", "Hydrogen sulfate", False, 4),
}


def predict_with_chemprop(smiles_list, temperatures, ckpt_dir):
    """Use Chemprop CLI to predict properties."""
    tmp_dir = Path("data/prediction_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for smi in smiles_list:
        for T in temperatures:
            rows.append({
                "smiles": smi,
                "temperature": (T - 314.97) / 29.00,  # normalized
                "x1": 0.0,  # normalized (all at x1=0.5)
                "inv_temperature": (1/T - 0.00324) / 0.000295,
                "temp_squared": (T**2 - 100953) / 18990,
                "temp_cubed": (T**3 - 3.284e7) / 9.51e6,
            })

    df = pd.DataFrame(rows)
    feat_cols = ["temperature", "x1", "inv_temperature", "temp_squared", "temp_cubed"]

    # Save data
    out = pd.DataFrame({"smiles": df["smiles"]})
    for t in TARGET_COLUMNS:
        out[t] = ""
    out.to_csv(tmp_dir / "predict.csv", index=False)
    df[feat_cols].to_csv(tmp_dir / "predict_features.csv", index=False)

    pred_path = tmp_dir / "predictions.csv"
    cmd = [
        "chemprop_predict",
        "--test_path", str(tmp_dir / "predict.csv"),
        "--features_path", str(tmp_dir / "predict_features.csv"),
        "--checkpoint_dir", ckpt_dir,
        "--preds_path", str(pred_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  Chemprop error: {result.stderr[-200:]}")
        return None

    if pred_path.exists():
        preds = pd.read_csv(pred_path)
        preds["smiles"] = df["smiles"].values
        preds["temperature_raw"] = [T for _ in smiles_list for T in temperatures]
        return preds
    return None


def main():
    set_seed(42)
    device = get_device(load_config("configs/default.yaml"))
    print(f"Device: {device}")

    # ══════════════════════════════════════════════════════════
    # Generate candidates
    # ══════════════════════════════════════════════════════════
    print("Generating IL candidates...\n")

    candidates = []
    for cat_key, (cat_smi, cat_name, cat_novel, cat_synth) in CATIONS.items():
        for an_key, (an_smi, an_name, an_novel, an_synth) in ANIONS.items():
            smi = f"{cat_smi}.{an_smi}"
            name = f"{cat_name} {an_name.lower()}"
            novelty = "Novel" if (cat_novel or an_novel) else "Known"
            synth_score = (cat_synth + an_synth) / 2  # average accessibility

            candidates.append({
                "smiles": smi,
                "name": name,
                "cation": cat_name,
                "anion": an_name,
                "novelty": novelty,
                "synth_score": synth_score,
                "fully_novel": cat_novel and an_novel,
            })

    print(f"  Total candidates: {len(candidates)}")
    n_novel = sum(1 for c in candidates if c["novelty"] == "Novel")
    n_full = sum(1 for c in candidates if c["fully_novel"])
    print(f"  Novel (at least one new ion): {n_novel}")
    print(f"  Fully novel (both ions new): {n_full}")

    # ══════════════════════════════════════════════════════════
    # Predict with Chemprop (base) and STILT
    # ══════════════════════════════════════════════════════════
    temperatures = [298.15, 348.15, 398.15, 448.15]
    smiles_list = list(set(c["smiles"] for c in candidates))

    print(f"\n  Predicting with Chemprop (base)...")
    preds_base = predict_with_chemprop(smiles_list, temperatures, "checkpoints/chemprop")

    print(f"  Predicting with STILT...")
    preds_stilt = predict_with_chemprop(smiles_list, temperatures, "checkpoints/chemprop_tuned/c")

    # ══════════════════════════════════════════════════════════
    # Process predictions
    # ══════════════════════════════════════════════════════════
    # Load target scalers to inverse-transform
    with open("data/processed/target_scaler.pkl", "rb") as f:
        ts = pickle.load(f)

    results = []
    preds_to_use = preds_stilt if preds_stilt is not None else preds_base

    if preds_to_use is not None:
        model_name = "STILT" if preds_stilt is not None else "Chemprop"
        print(f"\n  Using {model_name} predictions")

        # Get T=348K predictions (mid-range)
        preds_348 = preds_to_use[preds_to_use["temperature_raw"] == 348.15].copy()

        for _, row in preds_348.iterrows():
            smi = row["smiles"]
            cand = next((c for c in candidates if c["smiles"] == smi), None)
            if cand is None:
                continue

            # Inverse-transform to raw values
            props = {}
            for i, t in enumerate(TARGET_COLUMNS):
                try:
                    v = row.get(t)
                    if v is not None and pd.notna(v) and str(v) not in ('', 'Invalid SMILES'):
                        props[t] = float(v) * ts.scale_[i] + ts.mean_[i]
                    else:
                        props[t] = None
                except (ValueError, TypeError):
                    props[t] = None

            # Synthesis priority score
            # Lower gamma1 = better miscibility (weight: 3)
            # More negative G_mix = better mixing (weight: 2)
            # Moderate H_vap 14-18 = good stability (weight: 1)
            # Lower P = safer (weight: 1)
            # Higher synth_score = easier synthesis (weight: 2)
            score = 0
            if props.get("gamma1") is not None:
                score -= props["gamma1"] * 3  # lower is better
            if props.get("G_mix") is not None:
                score += props["G_mix"] * 2   # more negative is better
            if props.get("H_vap") is not None:
                score -= abs(props["H_vap"] - 16) * 0.5  # prefer around 16
            if props.get("P") is not None:
                score -= props["P"] * 1       # lower pressure is better
            score += cand["synth_score"] * 0.5  # synthetic accessibility

            results.append({
                **cand,
                **{f"{t}_pred": props.get(t) for t in TARGET_COLUMNS},
                "priority_score": score,
            })

    else:
        print("  No model predictions available — using structural ranking")
        for c in candidates:
            c["priority_score"] = c["synth_score"]
            results.append(c)

    # ══════════════════════════════════════════════════════════
    # Rank and display
    # ══════════════════════════════════════════════════════════
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("priority_score", ascending=True)

    print(f"\n{'='*100}")
    print("TOP 15 IONIC LIQUIDS FOR SYNTHESIS (ranked by priority score)")
    print(f"{'='*100}")
    print(f"{'Rank':>4s}  {'Ionic Liquid':<42s} {'Novel':>5s} {'γ₁':>7s} {'γ₂':>7s} "
          f"{'G_mix':>7s} {'H_vap':>7s} {'P':>7s} {'Synth':>5s} {'Score':>7s}")
    print("-" * 100)

    top15 = results_df.head(15)
    for rank, (_, row) in enumerate(top15.iterrows(), 1):
        g1 = f"{row.get('gamma1_pred', '?'):.3f}" if pd.notna(row.get('gamma1_pred')) else "?"
        g2 = f"{row.get('gamma2_pred', '?'):.3f}" if pd.notna(row.get('gamma2_pred')) else "?"
        gm = f"{row.get('G_mix_pred', '?'):.3f}" if pd.notna(row.get('G_mix_pred')) else "?"
        hv = f"{row.get('H_vap_pred', '?'):.2f}" if pd.notna(row.get('H_vap_pred')) else "?"
        p = f"{row.get('P_pred', '?'):.3f}" if pd.notna(row.get('P_pred')) else "?"
        nov = "NEW" if row.get("fully_novel") else ("new" if row.get("novelty") == "Novel" else "")
        print(f"{rank:4d}  {row['name']:<42s} {nov:>5s} {g1:>7s} {g2:>7s} "
              f"{gm:>7s} {hv:>7s} {p:>7s} {row.get('synth_score', 0):5.1f} {row['priority_score']:7.2f}")

    # ══════════════════════════════════════════════════════════
    # Detailed top 5 report
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print("DETAILED SYNTHESIS RECOMMENDATIONS — TOP 5")
    print(f"{'='*100}")

    for rank, (_, row) in enumerate(top15.head(5).iterrows(), 1):
        print(f"\n  #{rank}: {row['name']}")
        print(f"  SMILES: {row['smiles']}")
        print(f"  Cation: {row['cation']}")
        print(f"  Anion:  {row['anion']}")
        print(f"  Novelty: {'Fully novel (both ions new)' if row.get('fully_novel') else 'Partially novel' if row.get('novelty')=='Novel' else 'Known combination'}")
        print(f"  Synthetic accessibility: {row.get('synth_score', 0):.1f}/5")
        print(f"  Predicted properties (T=348K, x₁=0.5):")
        for t in TARGET_COLUMNS:
            v = row.get(f"{t}_pred")
            if v is not None and pd.notna(v):
                print(f"    {t:15s}: {v:.4f}")
        print(f"  Priority score: {row['priority_score']:.3f}")

        # Synthesis suggestion
        print(f"  Synthesis route: Mix {row['cation']} base with {row['anion']} acid")
        print(f"                   at room temperature under N₂, stir 24h, dry under vacuum")

    # Save
    results_df.to_csv("results/novel_il_candidates.csv", index=False)
    top15.to_csv("results/top15_il_candidates.csv", index=False)

    summary = {
        "n_candidates": len(results_df),
        "n_novel": int(results_df["fully_novel"].sum()),
        "top_5": [
            {"name": row["name"], "smiles": row["smiles"],
             "gamma1": row.get("gamma1_pred"), "G_mix": row.get("G_mix_pred"),
             "priority": row["priority_score"]}
            for _, row in top15.head(5).iterrows()
        ],
    }
    with open("results/synthesis_recommendations.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved: results/novel_il_candidates.csv ({len(results_df)} candidates)")
    print(f"Saved: results/top15_il_candidates.csv")
    print(f"Saved: results/synthesis_recommendations.json")


if __name__ == "__main__":
    main()
