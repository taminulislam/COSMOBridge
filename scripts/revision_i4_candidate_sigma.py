"""I4: candidate uncertainty bands for Table 7.

Re-runs the COSMOBridge-v3 virtual screen on a C1-consistent footing and reports
per-candidate mean +/- std (sigma) across the 5 published v3 gate seeds
{42,123,456,789,1024}.

Why this is the honest version of the screen:
  * The published screen (predict_cosmobridge_v3.py) blended the STANDARDIZED fusion
    output with the RAW chemprop CLI output and then inverse-transformed the whole
    blend -- double-scaling the chemprop term (the same class of unit bug as C1).
  * Here every path is in ORIGINAL units before blending, exactly as the C1-corrected
    gates were fit (verified: seed_preds == gate*fusion_raw + (1-gate)*chemprop_raw,
    avg R2 = 0.777).  The candidate fusion/chemprop predictions are seed-independent
    (single fusion checkpoint, single chemprop model); only the 7 gates vary by seed,
    so the seed-to-seed spread is a genuine uncertainty band on the reported means.

Outputs results/cosmobridge_v3_candidates_sigma.csv and a short JSON summary, and
checks that the locked top-5 (TMG propanoate/acetate/formate/glycinate/lactate, in
order) is preserved.  It does NOT edit the manuscript.
"""
import sys, json, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.models.fusion.multimodal_pointcloud import MultimodalPointCloudModel
from scripts.train_chemprop_gbh_hybrid import ChempropGBHFusion
from scripts.rank_ils_desirability import compute_desirability, is_pareto_optimal
from scripts.predict_cosmobridge_v3 import (
    CATIONS, ANIONS, THERMO_FEATURES, generate_point_cloud)

LOCKED_TOP5 = ["Tetramethylguanidinium propanoate", "Tetramethylguanidinium acetate",
               "Tetramethylguanidinium formate", "Tetramethylguanidinium glycinate",
               "Tetramethylguanidinium lactate"]


def main():
    set_seed(42)  # fixes the Gasteiger point-cloud sampling so fusion preds are reproducible
    device = get_device(load_config("configs/default.yaml"))
    print(f"Device: {device}")

    with open("data/processed/target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)
    with open("data/processed/feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    tscale, tmean = target_scaler.scale_, target_scaler.mean_

    # 5 published v3 gate seeds (raw-space blend), from the C1 rebuild
    npz = np.load("results/cosmobridge_c1_official.npz", allow_pickle=True)
    seed_gates = npz["seed_gates"]          # (5, 7), gate = weight on fusion path
    seeds = list(npz["seeds"])
    print(f"  Loaded {len(seeds)} gate seeds {seeds}; mean gates {np.round(seed_gates.mean(0),3)}")

    # ---- candidates ----
    T = 348.15; x1 = 0.5
    candidates = []
    for ck, (cs, cn, cnovel, csynth) in CATIONS.items():
        for ak, (asmi, an, anovel, asynth) in ANIONS.items():
            from rdkit import Chem
            smi = f"{cs}.{asmi}"
            if Chem.MolFromSmiles(smi) is None:
                continue
            candidates.append({"smiles": smi, "name": f"{cn} {an.lower()}",
                               "cation": cn, "anion": an,
                               "fully_novel": cnovel and anovel,
                               "synth_score": (csynth + asynth) / 2})
    print(f"  {len(candidates)} candidates")

    # ---- point clouds + surface (256D) ----
    pc_cache = {c["smiles"]: generate_point_cloud(c["smiles"]) for c in candidates}
    config = load_config("configs/default.yaml")
    pc_model = MultimodalPointCloudModel(config=config, pretrained_gnn_path=None)
    ckpt = torch.load("checkpoints/pointcloud/best_model.pt", map_location=device, weights_only=False)
    pc_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    pc_model.to(device).eval()
    surface_cache = {}
    with torch.no_grad():
        for smi, pc in pc_cache.items():
            t = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).to(device)
            surface_cache[smi] = pc_model.pointnet(t).cpu().numpy()[0]

    # ---- chemprop fingerprints + full predictions (raw units via CLI) ----
    tmp = Path("data/cosmobridge_predict_tmp"); tmp.mkdir(parents=True, exist_ok=True)
    thermo_raw = np.array([T, x1, 1/T, T**2, T**3])
    thermo_norm = (thermo_raw - feature_scaler.mean_[:5]) / feature_scaler.scale_[:5]
    uniq = list(pc_cache.keys())
    pd.DataFrame({"smiles": uniq, **{t: "" for t in TARGET_COLUMNS}}).to_csv(tmp/"predict.csv", index=False)
    pd.DataFrame([thermo_norm[:5]]*len(uniq), columns=THERMO_FEATURES).to_csv(tmp/"predict_features.csv", index=False)

    fp_out = tempfile.mktemp(suffix=".csv")
    subprocess.run(["chemprop_fingerprint", "--test_path", str(tmp/"predict.csv"),
                    "--features_path", str(tmp/"predict_features.csv"),
                    "--checkpoint_dir", "checkpoints/chemprop", "--fingerprint_type", "MPN",
                    "--preds_path", fp_out], capture_output=True, text=True, timeout=300)
    gdf = pd.read_csv(fp_out).select_dtypes(include=[np.number])
    graph_cache = {smi: gdf.iloc[i].values.astype(np.float32) for i, smi in enumerate(uniq)}
    graph_dim = gdf.shape[1]

    pr_out = tempfile.mktemp(suffix=".csv")
    subprocess.run(["chemprop_predict", "--test_path", str(tmp/"predict.csv"),
                    "--features_path", str(tmp/"predict_features.csv"),
                    "--checkpoint_dir", "checkpoints/chemprop", "--preds_path", pr_out],
                   capture_output=True, text=True, timeout=300)
    cdf = pd.read_csv(pr_out)
    chemprop_raw = {smi: cdf[TARGET_COLUMNS].iloc[i].values.astype(np.float32)
                    for i, smi in enumerate(uniq)}  # RAW units (CLI applies inverse_transform)

    # ---- fusion model ----
    fusion = ChempropGBHFusion(graph_dim=graph_dim, surface_dim=256,
                               thermo_dim=len(FEATURE_COLUMNS), fused_dim=256,
                               rank=32, hyper_hidden=64, dropout=0.3)
    fusion.load_state_dict(torch.load("checkpoints/chemprop_gbh_hybrid/best.pt",
                                      map_location=device, weights_only=True))
    fusion.to(device).eval()
    full_thermo = np.zeros(len(FEATURE_COLUMNS)); full_thermo[:5] = thermo_norm[:5]

    # ---- per-candidate prediction: blend each seed in RAW units ----
    rows = []
    for c in candidates:
        smi = c["smiles"]
        g = torch.tensor(graph_cache[smi]).unsqueeze(0).to(device)
        s = torch.tensor(surface_cache[smi]).unsqueeze(0).to(device)
        tf = torch.tensor(full_thermo, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            fused_std = fusion(g, s, tf).cpu().numpy()[0]
        fused_raw = fused_std * tscale + tmean              # to original units
        cp_raw = chemprop_raw[smi]
        # (5,7) raw predictions, one per gate seed
        per_seed = np.stack([gate * fused_raw + (1 - gate) * cp_raw for gate in seed_gates])
        mean_p = per_seed.mean(0); std_p = per_seed.std(0)
        props = {col: float(mean_p[i]) for i, col in enumerate(TARGET_COLUMNS)}
        stds = {col: float(std_p[i]) for i, col in enumerate(TARGET_COLUMNS)}
        d_values, D, _ = compute_desirability(props, c["synth_score"])
        rows.append({**c, **{f"{t}_pred": props[t] for t in TARGET_COLUMNS},
                     **{f"{t}_std": stds[t] for t in TARGET_COLUMNS},
                     "D_combined": D})

    df = pd.DataFrame(rows).sort_values("D_combined", ascending=False).reset_index(drop=True)
    # Pareto flag on the (gamma1, G_mix, P) front (minimize gamma1, P; G_mix already negative)
    out_csv = "results/cosmobridge_v3_candidates_sigma.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "="*88)
    print("TOP 10 (C1-consistent, 5-seed gate ensemble; mean +/- std)")
    print("="*88)
    for i, r in df.head(10).iterrows():
        print(f"{i+1:2d}. {r['name']:<34s} g1={r['gamma1_pred']:.3f}+/-{r['gamma1_std']:.3f} "
              f"Gmix={r['G_mix_pred']:.2f}+/-{r['G_mix_std']:.2f} "
              f"P={r['P_pred']:.3f}+/-{r['P_std']:.3f}  D={r['D_combined']:.3f}")

    top5 = list(df.head(5)["name"])
    ok = top5 == LOCKED_TOP5
    print("\nLocked top-5 preserved?", ok)
    if not ok:
        print("  expected:", LOCKED_TOP5)
        print("  got     :", top5)

    summary = {"seeds": [int(s) for s in seeds], "top5_preserved": bool(ok),
               "top5": top5,
               "table7": [{"name": r["name"],
                           "gamma1": round(r["gamma1_pred"], 3), "gamma1_std": round(r["gamma1_std"], 3),
                           "G_mix": round(r["G_mix_pred"], 2), "G_mix_std": round(r["G_mix_std"], 2),
                           "P": round(r["P_pred"], 3), "P_std": round(r["P_std"], 3),
                           "D": round(r["D_combined"], 3)}
                          for _, r in df.head(5).iterrows()]}
    json.dump(summary, open("results/cosmobridge_v3_candidates_sigma.json", "w"), indent=2)
    print(f"\nsaved {out_csv} and results/cosmobridge_v3_candidates_sigma.json")


if __name__ == "__main__":
    main()
