"""Rank comparison: v3 Gasteiger-based desirability vs DFT COSMO solvation energy.

Tests whether the ML-based (Gasteiger ESP + desirability) ranking is consistent
with first-principles DFT COSMO solvation predictions for the top-20 candidates.

A high Spearman ρ validates that the ML rankings are physically meaningful.
A low ρ suggests the Gasteiger approximation is lossy and DFT-based surfaces
could shift the top-5 recommendations.
"""

import sys
import json
import re
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, kendalltau


def main():
    # Load candidates with their v3 ranks
    candidates_path = Path("cosmobridge_v4/screening/top20_candidates.csv")
    lines = candidates_path.read_text().strip().split("\n")[1:]  # skip header
    candidates = []
    for line in lines:
        parts = line.split(",")
        rank = int(parts[0])
        il_id = parts[1]
        d_score = float(parts[5]) if parts[5] else None
        candidates.append({"rank": rank, "id": il_id, "v3_d": d_score})

    print(f"Loaded {len(candidates)} candidates")

    # Load DFT energies
    nw_dir = Path("cosmobridge_v4/screening/nwchem_outputs")
    dft_data = {}
    for cand in candidates:
        cid = cand["id"]
        cat_out = nw_dir / f"{cid}_cation.out"
        an_out = nw_dir / f"{cid}_anion.out"

        if not (cat_out.exists() and an_out.exists()):
            continue

        cat_text = cat_out.read_text()
        an_text = an_out.read_text()

        # Extract COSMO solvation energy (in Hartree)
        cat_cosmo = re.findall(r"COSMO energy =\s+(-?\d+\.\d+)", cat_text)
        an_cosmo = re.findall(r"COSMO energy =\s+(-?\d+\.\d+)", an_text)
        cat_dft = re.findall(r"Total DFT energy =\s+(-?\d+\.\d+)", cat_text)
        an_dft = re.findall(r"Total DFT energy =\s+(-?\d+\.\d+)", an_text)

        if cat_cosmo and an_cosmo:
            dft_data[cid] = {
                "cosmo_total_Ha": float(cat_cosmo[-1]) + float(an_cosmo[-1]),
                "dft_total_Ha": float(cat_dft[-1]) + float(an_dft[-1]),
                "cation_cosmo": float(cat_cosmo[-1]),
                "anion_cosmo": float(an_cosmo[-1]),
            }

    # Build comparison table
    comparison = []
    for cand in candidates:
        if cand["id"] in dft_data:
            row = {**cand, **dft_data[cand["id"]]}
            comparison.append(row)

    # Rank by COSMO energy (more negative = stronger solvation = better water miscibility)
    sorted_by_cosmo = sorted(comparison, key=lambda x: x["cosmo_total_Ha"])
    for i, row in enumerate(sorted_by_cosmo, 1):
        row["dft_rank"] = i

    print(f"\n{'='*80}")
    print(f"RANKING COMPARISON: v3 Desirability vs DFT COSMO Energy")
    print(f"{'='*80}\n")
    print(f"{'IL':<15s} {'v3 Rank':>9s} {'v3 D':>8s} {'DFT COSMO (Ha)':>16s} {'DFT Rank':>10s}")
    print("-" * 70)

    sorted_by_v3 = sorted(comparison, key=lambda x: x["rank"])
    for row in sorted_by_v3:
        v3_d = f"{row['v3_d']:.3f}" if row.get('v3_d') else "  -  "
        print(f"{row['id']:<15s} {row['rank']:>9d} {v3_d:>8s} "
              f"{row['cosmo_total_Ha']:>16.4f} {row['dft_rank']:>10d}")

    # Compute rank correlations (only for candidates with v3_d scores)
    scored = [r for r in comparison if r.get("v3_d") is not None]
    if len(scored) >= 3:
        v3_ranks = np.array([r["rank"] for r in scored])
        dft_ranks = np.array([r["dft_rank"] for r in scored])
        cosmo_energies = np.array([r["cosmo_total_Ha"] for r in scored])
        v3_ds = np.array([r["v3_d"] for r in scored])

        rho_r, p_r = spearmanr(v3_ranks, dft_ranks)
        tau_r, p_t = kendalltau(v3_ranks, dft_ranks)
        rho_e, p_e = spearmanr(-v3_ds, cosmo_energies)  # lower COSMO = better

        print(f"\n{'='*60}")
        print(f"RANK CORRELATION (n={len(scored)} scored candidates)")
        print(f"{'='*60}")
        print(f"  Spearman ρ (v3_rank vs DFT_rank):        {rho_r:.3f} (p={p_r:.3f})")
        print(f"  Kendall τ (v3_rank vs DFT_rank):         {tau_r:.3f} (p={p_t:.3f})")
        print(f"  Spearman ρ (v3_D vs COSMO energy):       {rho_e:.3f} (p={p_e:.3f})")

        # Top-5 overlap
        v3_top5 = set([r["id"] for r in sorted_by_v3[:5]])
        dft_top5 = set([r["id"] for r in sorted_by_cosmo[:5]])
        overlap = v3_top5 & dft_top5
        print(f"\n  v3 top-5:  {v3_top5}")
        print(f"  DFT top-5: {dft_top5}")
        print(f"  Overlap:   {len(overlap)}/5 = {overlap}")

    # Full sorted-by-DFT ranking
    print(f"\n{'='*80}")
    print(f"FULL RANKING BY DFT COSMO SOLVATION ENERGY (more negative = better)")
    print(f"{'='*80}\n")
    print(f"{'DFT Rank':>9s} {'IL':<15s} {'COSMO (Ha)':>14s} {'v3 Rank':>10s}")
    print("-" * 60)
    for r in sorted_by_cosmo:
        print(f"{r['dft_rank']:>9d} {r['id']:<15s} {r['cosmo_total_Ha']:>14.4f} "
              f"{r['rank']:>10d}")

    # Save
    results = {
        "n_candidates": len(comparison),
        "rank_comparison": [
            {"id": r["id"], "v3_rank": r["rank"], "v3_d": r.get("v3_d"),
             "dft_cosmo_Ha": r["cosmo_total_Ha"], "dft_rank": r["dft_rank"]}
            for r in sorted_by_v3
        ],
    }
    if len(scored) >= 3:
        results["correlations"] = {
            "spearman_rho_ranks": float(rho_r),
            "spearman_p": float(p_r),
            "kendall_tau": float(tau_r),
            "kendall_p": float(p_t),
            "spearman_desirability_vs_cosmo": float(rho_e),
        }
        results["top5_overlap"] = len(overlap)
        results["v3_top5"] = list(v3_top5)
        results["dft_top5"] = list(dft_top5)

    out_path = Path("cosmobridge_v4/results/screening_dft_vs_gasteiger.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
