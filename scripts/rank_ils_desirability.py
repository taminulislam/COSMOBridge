"""Improved IL candidate ranking using Derringer-Suich desirability functions.

Replaces the ad-hoc linear scoring equation with:
1. Per-property desirability mapping d_i ∈ [0,1] via sigmoidal/linear functions
2. Geometric mean combination: D = (∏ d_i^w_i)^(1/Σw_i)
3. Pareto front identification (no weights needed)

References:
- Derringer & Suich, J. Quality Technology, 1980
- Harrington, Industrial Quality Control, 1965
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path("paper/figures")
TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]

# Training data statistics for reference bounds
TRAIN_STATS = {
    "gamma1": {"min": 0.127, "max": 2.122, "mean": 0.776, "std": 0.372},
    "gamma2": {"min": 0.030, "max": 2.917, "mean": 0.504, "std": 0.611},
    "G_mix":  {"min": -2.68, "max": 0.10, "mean": -1.280, "std": 0.765},
    "H_vap":  {"min": 8.26, "max": 27.36, "mean": 16.662, "std": 5.099},
    "P":      {"min": 0.001, "max": 4.45, "mean": 0.651, "std": 1.154},
}


def desirability_minimize(value, low, high):
    """Desirability for minimize objective: d=1 at low, d=0 at high."""
    if value <= low: return 1.0
    if value >= high: return 0.0
    return (high - value) / (high - low)


def desirability_target(value, target, low, high):
    """Desirability for target objective: d=1 at target, d=0 at bounds."""
    if value <= low or value >= high: return 0.0
    if value <= target:
        return (value - low) / (target - low)
    else:
        return (high - value) / (high - target)


def compute_desirability(props, synth_score):
    """Compute per-property desirability and combined score.

    Desirability functions based on physical requirements:
    - gamma1: minimize (lower = better miscibility)
    - G_mix: minimize (more negative = better mixing)
    - H_vap: target range (moderate = good stability, not too volatile)
    - P: minimize (lower vapor pressure = safer handling)
    - synth: maximize (easier synthesis = more practical)
    """
    d = {}

    # gamma1: minimize, ideal ≤ 0.2, unacceptable ≥ 1.0
    g1 = props.get("gamma1")
    d["gamma1"] = desirability_minimize(g1, 0.15, 1.0) if g1 is not None else 0.5

    # gamma2: minimize (lower = less non-ideal solvent behavior)
    g2 = props.get("gamma2")
    d["gamma2"] = desirability_minimize(g2, 0.0, 1.5) if g2 is not None else 0.5

    # G_mix: minimize (more negative = spontaneous mixing)
    gm = props.get("G_mix")
    d["G_mix"] = desirability_minimize(gm, -3.0, -0.3) if gm is not None else 0.5

    # H_vap: target 14-18 kcal/mol (moderate stability)
    hv = props.get("H_vap")
    d["H_vap"] = desirability_target(hv, 16.0, 8.0, 28.0) if hv is not None else 0.5

    # P: minimize (lower = safer)
    p = props.get("P")
    d["P"] = desirability_minimize(p, 0.0, 2.0) if p is not None else 0.5

    # Synthesis accessibility: maximize (1-5 scale → 0-1)
    d["synth"] = (synth_score - 1) / 4  # maps [1,5] → [0,1]

    # Weights (application-specific: biomass processing focus)
    weights = {
        "gamma1": 3.0,   # most important for miscibility
        "G_mix": 2.0,    # thermodynamic driving force
        "gamma2": 1.0,   # secondary miscibility
        "H_vap": 0.5,    # moderate importance
        "P": 1.0,        # safety
        "synth": 1.0,    # practicality
    }

    # Geometric mean (Derringer-Suich)
    # D = (∏ d_i^w_i)^(1/Σw_i)
    log_D = 0.0
    total_w = 0.0
    for key, w in weights.items():
        di = max(d[key], 1e-10)  # avoid log(0)
        log_D += w * np.log(di)
        total_w += w
    D = np.exp(log_D / total_w)

    return d, D, weights


def is_pareto_optimal(costs):
    """Find Pareto-optimal points (minimize all objectives)."""
    n = len(costs)
    is_optimal = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_optimal[i]: continue
        for j in range(n):
            if i == j: continue
            if all(costs[j] <= costs[i]) and any(costs[j] < costs[i]):
                is_optimal[i] = False
                break
    return is_optimal


def main():
    print("=== Improved IL Ranking: Derringer-Suich Desirability + Pareto ===\n")

    # Load predictions
    candidates_path = Path("results/novel_il_candidates.csv")
    if not candidates_path.exists():
        print("ERROR: Run predict_novel_ils.py first")
        return

    df = pd.read_csv(candidates_path)
    print(f"Loaded {len(df)} candidates")

    # Compute desirability for each candidate
    results = []
    for _, row in df.iterrows():
        props = {}
        for t in TARGET_COLUMNS:
            v = row.get(f"{t}_pred")
            if pd.notna(v):
                props[t] = float(v)

        synth = row.get("synth_score", 3.0)
        d_values, D_combined, weights = compute_desirability(props, synth)

        results.append({
            "name": row.get("name", "?"),
            "smiles": row.get("smiles", "?"),
            "cation": row.get("cation", "?"),
            "anion": row.get("anion", "?"),
            "novelty": row.get("novelty", "?"),
            "fully_novel": row.get("fully_novel", False),
            "synth_score": synth,
            **{f"d_{k}": v for k, v in d_values.items()},
            "D_combined": D_combined,
            **{f"{t}_pred": props.get(t) for t in TARGET_COLUMNS},
            "old_score": row.get("priority_score", 0),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("D_combined", ascending=False)

    # Print top 15
    print(f"\n{'='*110}")
    print("TOP 15 IL CANDIDATES — Derringer-Suich Desirability Ranking")
    print(f"{'='*110}")
    print(f"{'Rank':>4s}  {'Ionic Liquid':<40s} {'Novel':>5s} {'d(γ₁)':>6s} {'d(G_m)':>6s} "
          f"{'d(P)':>6s} {'d(syn)':>6s} {'D':>6s} {'OldRk':>5s}")
    print("-" * 110)

    top15 = results_df.head(15)
    # Get old ranking for comparison
    old_ranked = results_df.sort_values("old_score", ascending=True)
    old_rank_map = {name: i+1 for i, name in enumerate(old_ranked["name"])}

    for rank, (_, row) in enumerate(top15.iterrows(), 1):
        nov = "NEW" if row.get("fully_novel") else ("new" if row.get("novelty") == "Novel" else "")
        old_rk = str(old_rank_map.get(row["name"], "?"))
        print(f"{rank:4d}  {row['name']:<40s} {nov:>5s} {row['d_gamma1']:6.3f} {row['d_G_mix']:6.3f} "
              f"{row['d_P']:6.3f} {row['d_synth']:6.3f} {row['D_combined']:6.3f} {old_rk:>5s}")

    # Pareto front analysis
    print(f"\n{'='*110}")
    print("PARETO FRONT (gamma1, G_mix, P — no weights)")
    print(f"{'='*110}")

    valid = results_df.dropna(subset=["gamma1_pred", "G_mix_pred", "P_pred"])
    if len(valid) > 0:
        costs = valid[["gamma1_pred", "G_mix_pred", "P_pred"]].values
        # G_mix is already negative (minimize = more negative), need to flip for Pareto
        # We want: min gamma1, min G_mix (most negative), min P
        pareto_mask = is_pareto_optimal(costs)
        pareto_candidates = valid[pareto_mask]

        print(f"  {pareto_mask.sum()} Pareto-optimal candidates (out of {len(valid)}):")
        for _, row in pareto_candidates.iterrows():
            print(f"    {row['name']:<40s} γ₁={row['gamma1_pred']:.3f} "
                  f"G_mix={row['G_mix_pred']:.3f} P={row['P_pred']:.3f} "
                  f"D={row['D_combined']:.3f}")

    # Comparison: old vs new ranking
    print(f"\n{'='*110}")
    print("OLD vs NEW RANKING COMPARISON (top 5)")
    print(f"{'='*110}")
    print(f"  {'Old':>4s}  {'New':>4s}  {'Ionic Liquid':<40s} {'Old Score':>9s} {'Desirability':>12s}")
    print("  " + "-" * 75)

    new_top5 = set(top15.head(5)["name"])
    old_top5 = set(old_ranked.head(5)["name"])
    all_top = new_top5 | old_top5

    for name in sorted(all_top, key=lambda n: old_rank_map.get(n, 999)):
        row = results_df[results_df["name"] == name].iloc[0]
        new_rank = list(top15["name"]).index(name) + 1 if name in top15["name"].values else ">15"
        old_rk = str(old_rank_map.get(name, "?"))
        print(f"  {old_rk:>4s}  {str(new_rank):>4s}  {name:<40s} {row['old_score']:9.2f} {row['D_combined']:12.3f}")

    # ══════════════════════════════════════════════════════════
    # Generate figures
    # ══════════════════════════════════════════════════════════

    # Figure 1: Desirability breakdown for top 5
    fig, ax = plt.subplots(figsize=(12, 6))
    top5 = results_df.head(5)
    props_to_show = ["d_gamma1", "d_G_mix", "d_P", "d_H_vap", "d_synth"]
    prop_labels = [r"$d(\gamma_1)$", r"$d(G_{mix})$", r"$d(P)$", r"$d(H_{vap})$", r"$d(A_{synth})$"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#795548"]

    x = np.arange(len(top5))
    width = 0.15
    for i, (col, label, color) in enumerate(zip(props_to_show, prop_labels, colors)):
        vals = top5[col].values
        ax.bar(x + i * width, vals, width * 0.9, label=label, color=color, alpha=0.85)

    # Add combined D as diamond markers
    ax.scatter(x + 2 * width, top5["D_combined"].values, s=100, color="red",
               marker="D", zorder=5, label="D (combined)")

    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels([n[:25] for n in top5["name"]], fontsize=8, rotation=15)
    ax.set_ylabel("Desirability [0-1]", fontsize=11)
    ax.set_title("Derringer-Suich Desirability Breakdown — Top 5 IL Candidates", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "desirability_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: desirability_breakdown.png")

    # Figure 2: Pareto front (gamma1 vs G_mix)
    if len(valid) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(valid["gamma1_pred"], valid["G_mix_pred"],
                   c=valid["D_combined"], cmap="RdYlGn", s=50, alpha=0.7,
                   edgecolors="gray", linewidth=0.5)

        # Highlight Pareto front
        pareto_pts = valid[pareto_mask]
        ax.scatter(pareto_pts["gamma1_pred"], pareto_pts["G_mix_pred"],
                   s=120, facecolors="none", edgecolors="red", linewidth=2.5,
                   label="Pareto-optimal", zorder=5)

        # Label top 3
        for _, row in results_df.head(3).iterrows():
            if pd.notna(row.get("gamma1_pred")) and pd.notna(row.get("G_mix_pred")):
                ax.annotate(row["name"][:20], (row["gamma1_pred"], row["G_mix_pred"]),
                            fontsize=7, fontweight="bold",
                            xytext=(5, 5), textcoords="offset points")

        ax.set_xlabel(r"$\gamma_1$ (lower = better miscibility)", fontsize=11)
        ax.set_ylabel(r"$G_{mix}$ (kcal/mol, more negative = better)", fontsize=11)
        ax.set_title("Pareto Front: Activity Coefficient vs Mixing Energy", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, valid["D_combined"].max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Combined Desirability D", fontsize=9)
        plt.tight_layout()
        plt.savefig(OUT / "pareto_front.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: pareto_front.png")

    # Save results
    results_df.to_csv("results/il_candidates_desirability.csv", index=False)
    top15.to_csv("results/top15_desirability.csv", index=False)

    summary = {
        "method": "Derringer-Suich desirability (geometric mean) + Pareto front",
        "reference": "Derringer & Suich, J. Quality Technology, 1980",
        "weights": {"gamma1": 3.0, "G_mix": 2.0, "gamma2": 1.0, "H_vap": 0.5, "P": 1.0, "synth": 1.0},
        "top_5": [
            {"rank": i+1, "name": row["name"], "D": float(row["D_combined"]),
             "gamma1": row.get("gamma1_pred"), "G_mix": row.get("G_mix_pred")}
            for i, (_, row) in enumerate(top15.head(5).iterrows())
        ],
        "n_pareto_optimal": int(pareto_mask.sum()) if len(valid) > 0 else 0,
    }
    with open("results/desirability_ranking.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Saved: results/il_candidates_desirability.csv")
    print(f"  Saved: results/desirability_ranking.json")


if __name__ == "__main__":
    main()
