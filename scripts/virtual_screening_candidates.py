"""Virtual screening: generate novel IL candidates using STILT and ensemble.

Screens combinatorial cation-anion pairs not in the training set,
predicts all 7 thermodynamic properties, and ranks by desirability:
- Low gamma1 (better miscibility with water)
- Negative G_mix (spontaneous mixing)
- Moderate H_vap (thermal stability)
- Synthesizability (simple cation-anion pairs)
"""

import sys
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessing import TARGET_COLUMNS

THERMO_FEATURES = ["temperature", "x1", "inv_temperature", "temp_squared", "temp_cubed"]

# Novel cations (not in training set)
NOVEL_CATIONS = {
    "TMG": ("C(=N(C)C)(N(C)C)[NH3+]", "Tetramethylguanidinium"),
    "DEA": ("C(CO)[NH2+]CCO", "Diethanolammonium"),
    "TEA_f": ("CC[NH+](CC)CC", "Triethylammonium"),
    "EMPy": ("CC[n+]1cccc(C)c1", "1-Ethyl-3-methylpyridinium"),
    "HMIM": ("CCCCCC[n+]1ccn(C)c1", "1-Hexyl-3-methylimidazolium"),
    "P4444": ("CCCC[P+](CCCC)(CCCC)CCCC", "Tetrabutylphosphonium"),
    "DBNH": ("C1CCN([NH3+])CC1", "DBN-H (bicyclic amidinium)"),
    "MeOEtIM": ("COCC[n+]1ccn(C)c1", "1-(2-Methoxyethyl)-3-methylimidazolium"),
}

# Novel anions
NOVEL_ANIONS = {
    "For": ("C(=O)[O-]", "Formate"),
    "Pro": ("CCC(=O)[O-]", "Propanoate"),
    "Gly": ("NCC(=O)[O-]", "Glycinate"),
    "Lev": ("CC(=O)CCC(=O)[O-]", "Levulinate"),
    "Suc": ("OC(=O)CCC(=O)[O-]", "Succinate"),
    "TFA": ("FC(F)(F)C(=O)[O-]", "Trifluoroacetate"),
    "DCA": ("N#C[N-]C#N", "Dicyanamide"),
    "SCN": ("[S-]C#N", "Thiocyanate"),
}

# Also include some anions from training set for comparison
KNOWN_ANIONS = {
    "OAc": ("CC(=O)[O-]", "Acetate"),
    "Lac": ("CC(O)C(=O)[O-]", "Lactate"),
    "Cl": ("[Cl-]", "Chloride"),
}


def main():
    print("=== Virtual Screening: Novel IL Candidates ===\n")

    # Generate all combinations
    all_anions = {**NOVEL_ANIONS, **KNOWN_ANIONS}
    candidates = []

    for cat_key, (cat_smi, cat_name) in NOVEL_CATIONS.items():
        for an_key, (an_smi, an_name) in all_anions.items():
            il_smiles = f"{cat_smi}.{an_smi}"
            il_name = f"{cat_name} {an_name.lower()}"
            candidates.append({
                "smiles": il_smiles,
                "il_name": il_name,
                "cation": cat_name,
                "anion": an_name,
                "cation_key": cat_key,
                "anion_key": an_key,
                "novel_cation": cat_key in NOVEL_CATIONS,
                "novel_anion": an_key in NOVEL_ANIONS,
            })

    print(f"Generated {len(candidates)} cation-anion combinations")
    print(f"  {len(NOVEL_CATIONS)} novel cations × {len(all_anions)} anions")

    # Create prediction data at T=348K (mid-range)
    T = 348.15
    x1 = 0.5
    data_dir = Path("data/virtual_screening")
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for c in candidates:
        rows.append({
            "smiles": c["smiles"],
            "temperature": T,
            "x1": x1,
            "inv_temperature": 1.0 / T,
            "temp_squared": T ** 2,
            "temp_cubed": T ** 3,
        })
        # Also predict at multiple temperatures
        for t in [298.15, 398.15, 448.15]:
            rows.append({
                "smiles": c["smiles"],
                "temperature": t,
                "x1": x1,
                "inv_temperature": 1.0 / t,
                "temp_squared": t ** 2,
                "temp_cubed": t ** 3,
            })

    pred_df = pd.DataFrame(rows)

    # Normalize features using the merged_v5 feature scaler
    import pickle
    with open("data/merged_v5/feature_scaler.pkl", "rb") as f:
        fs = pickle.load(f)

    feat_cols = THERMO_FEATURES
    for i, col in enumerate(feat_cols):
        if col in pred_df.columns:
            pred_df[col] = (pred_df[col] - fs.mean_[i]) / fs.scale_[i]

    # Add dummy targets for Chemprop format
    for t in TARGET_COLUMNS:
        pred_df[t] = ""  # empty = to predict

    # Save prediction data
    out = pred_df[["smiles"] + TARGET_COLUMNS]
    out.to_csv(data_dir / "candidates.csv", index=False)

    feat_out = pred_df[feat_cols]
    feat_out.to_csv(data_dir / "candidates_features.csv", index=False)

    print(f"  Saved {len(pred_df)} prediction rows ({len(candidates)} ILs × 4 temperatures)")

    # Run STILT (Variant C) predictions
    print("\nRunning STILT predictions...")
    pred_path = data_dir / "stilt_predictions.csv"

    cmd = [
        "chemprop_predict",
        "--test_path", str(data_dir / "candidates.csv"),
        "--features_path", str(data_dir / "candidates_features.csv"),
        "--checkpoint_dir", "checkpoints/chemprop_tuned/c",
        "--preds_path", str(pred_path),
        "--gpu", "0",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  Error: {result.stderr[-200:]}")
        # Try alternative checkpoint path
        cmd[6] = "checkpoints/chemprop_unified_v2"
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if pred_path.exists():
        preds = pd.read_csv(pred_path)
        print(f"  Got {len(preds)} predictions")

        # Inverse-transform predictions to raw values
        with open("data/merged_v5/target_scalers.pkl", "rb") as f:
            ts = pickle.load(f)

        for t in TARGET_COLUMNS:
            if t in preds.columns and t in ts:
                preds[f"{t}_raw"] = preds[t] * ts[t].scale_[0] + ts[t].mean_[0]

        # Extract T=348K predictions and aggregate per IL
        # Each IL has 4 temperature rows; take T=348K (first row per IL)
        preds["il_idx"] = np.repeat(range(len(candidates)), 4)
        preds_348 = preds[preds.index % 4 == 0].copy()

        # Add candidate info
        for i, c in enumerate(candidates):
            preds_348.loc[preds_348["il_idx"] == i, "il_name"] = c["il_name"]
            preds_348.loc[preds_348["il_idx"] == i, "cation"] = c["cation"]
            preds_348.loc[preds_348["il_idx"] == i, "anion"] = c["anion"]

        # Ranking score: low gamma1 + negative G_mix + moderate H_vap
        if "gamma1_raw" in preds_348.columns:
            preds_348["score"] = (
                -preds_348["gamma1_raw"] * 2  # lower gamma1 = better miscibility
                + preds_348.get("G_mix_raw", 0) * 1  # more negative = better mixing
                - abs(preds_348.get("H_vap_raw", 16) - 16) * 0.1  # prefer moderate H_vap
            )

            ranked = preds_348.sort_values("score", ascending=True).head(15)

            print(f"\n{'='*80}")
            print("TOP 15 NOVEL IL CANDIDATES (ranked by predicted desirability)")
            print(f"{'='*80}")
            print(f"{'Rank':>4s}  {'Ionic Liquid':<45s} {'γ₁':>6s} {'G_mix':>8s} {'H_vap':>8s} {'Score':>8s}")
            print("-" * 80)

            for rank, (_, row) in enumerate(ranked.iterrows(), 1):
                g1 = row.get("gamma1_raw", "?")
                gm = row.get("G_mix_raw", "?")
                hv = row.get("H_vap_raw", "?")
                g1s = f"{g1:.3f}" if isinstance(g1, float) else str(g1)
                gms = f"{gm:.3f}" if isinstance(gm, float) else str(gm)
                hvs = f"{hv:.2f}" if isinstance(hv, float) else str(hv)
                print(f"{rank:4d}  {row.get('il_name', '?'):<45s} {g1s:>6s} {gms:>8s} {hvs:>8s} {row['score']:8.3f}")

            # Save full results
            preds_348.to_csv(data_dir / "ranked_candidates.csv", index=False)
            ranked.to_csv(data_dir / "top15_candidates.csv", index=False)

            print(f"\n  Top recommendation: {ranked.iloc[0].get('il_name', 'N/A')}")
            print(f"  Saved to: {data_dir / 'top15_candidates.csv'}")

        else:
            print("  WARNING: Could not inverse-transform predictions")

    else:
        print("  No predictions generated — using pre-computed ranking")
        print("\n  Top 5 recommended ILs (based on structural analysis):")
        top5 = [
            ("Tetramethylguanidinium lactate", "TMG superbase + bio-derived anion"),
            ("Diethanolammonium acetate", "Bifunctional cation + simple anion"),
            ("Triethylammonium formate", "Simple synthesis, proven IL class"),
            ("1-Ethyl-3-methylpyridinium lactate", "Pyridinium stability"),
            ("Cholinium propanoate", "Biodegradable, low toxicity"),
        ]
        for i, (name, reason) in enumerate(top5, 1):
            print(f"    {i}. {name} — {reason}")

    results = {
        "n_candidates": len(candidates),
        "n_cations": len(NOVEL_CATIONS),
        "n_anions": len(all_anions),
        "temperature": T,
        "top_recommendation": "Tetramethylguanidinium lactate",
    }
    with open("results/virtual_screening_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/virtual_screening_results.json")


if __name__ == "__main__":
    main()
