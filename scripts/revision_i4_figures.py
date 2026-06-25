"""Regenerate the two candidate figures (desirability_breakdown.png, candidate_esp_panel.png)
with the C1-consistent corrected screen values, so they match the corrected Table 7.

Keeps the FIVE candidates forwarded to the chemistry team, ordered by corrected
desirability (TMG acetate > formate > propanoate > lactate > glycinate).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from src.utils.config import set_seed
from src.data.preprocessing import TARGET_COLUMNS
from scripts.rank_ils_desirability import compute_desirability
from scripts.predict_cosmobridge_v3 import generate_point_cloud

OUTS = [Path("paper/jcim/figures"), Path("paper/cosmobridge_Overleaf/figures")]
FORWARDED = ["Tetramethylguanidinium acetate", "Tetramethylguanidinium formate",
             "Tetramethylguanidinium propanoate", "Tetramethylguanidinium lactate",
             "Tetramethylguanidinium glycinate"]
SHORT = {"Tetramethylguanidinium acetate": "TMG acetate",
         "Tetramethylguanidinium formate": "TMG formate",
         "Tetramethylguanidinium propanoate": "TMG propanoate",
         "Tetramethylguanidinium lactate": "TMG lactate",
         "Tetramethylguanidinium glycinate": "TMG glycinate"}


def main():
    set_seed(42)
    df = pd.read_csv("results/cosmobridge_v3_candidates_sigma.csv")
    rows = {r["name"]: r for _, r in df.iterrows()}

    # ---- compute corrected d-components for the 5 forwarded candidates ----
    recs = []
    for name in FORWARDED:
        r = rows[name]
        props = {c: float(r[f"{c}_pred"]) for c in TARGET_COLUMNS}
        d_values, D, _ = compute_desirability(props, float(r["synth_score"]))
        recs.append({"name": name, "smiles": r["smiles"], "D": D, **d_values})
    cand = pd.DataFrame(recs)
    print("Corrected d-components:")
    print(cand[["name", "D", "gamma1", "G_mix", "P", "H_vap", "synth"]].to_string(index=False))

    # ===================== Figure 1: desirability breakdown =====================
    comps = [("gamma1", r"$d(\gamma_1)$", "#3498db"), ("G_mix", r"$d(G_{mix})$", "#2ecc71"),
             ("P", r"$d(P)$", "#f39c12"), ("H_vap", r"$d(H_{vap})$", "#9b59b6"),
             ("synth", r"$d(A_{synth})$", "#8d6e63")]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(cand)); width = 0.16
    for i, (col, lab, c) in enumerate(comps):
        ax.bar(x + (i - 2) * width, cand[col].values, width, label=lab, color=c, alpha=0.9)
    ax.scatter(x, cand["D"].values, s=130, color="red", marker="D",
               zorder=5, label="D (combined)", edgecolor="white", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[n] for n in cand["name"]], fontsize=10)
    ax.set_ylabel("Desirability [0--1]", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_title("Derringer--Suich Desirability --- Five Forwarded COSMOBridge Candidates",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for o in OUTS:
        bak = o / "desirability_breakdown.PRE_I4FIX.png"
        src = o / "desirability_breakdown.png"
        if src.exists() and not bak.exists():
            src.rename(bak)
        plt.savefig(o / "desirability_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("saved desirability_breakdown.png")

    # ===================== Figure 2: ESP panel (corrected top-3) =====================
    top3 = cand.head(3)  # acetate, formate, propanoate
    labels = [f"[TMG][{a}]\nRank #{i+1}{n}" for i, (a, n) in enumerate(
        [("OAc", ""), ("For", " (fully novel)"), ("Pro", " (fully novel)")])]
    fig = plt.figure(figsize=(13, 3.6))
    sc = None
    for i, (_, r) in enumerate(top3.iterrows()):
        pc = generate_point_cloud(r["smiles"])
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 6], cmap="coolwarm",
                        s=6, vmin=-0.3, vmax=0.3, alpha=0.8)
        ax.set_title(labels[i], fontsize=10)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.view_init(elev=20, azim=45)
    fig.suptitle("COSMO-style ESP Surfaces of Top COSMOBridge v3 Candidates",
                 fontsize=13, fontweight="bold")
    cb = fig.colorbar(sc, ax=fig.axes, shrink=0.6, pad=0.02)
    cb.set_label("ESP (a.u.)")
    for o in OUTS:
        bak = o / "candidate_esp_panel.PRE_I4FIX.png"
        src = o / "candidate_esp_panel.png"
        if src.exists() and not bak.exists():
            src.rename(bak)
        plt.savefig(o / "candidate_esp_panel.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("saved candidate_esp_panel.png")


if __name__ == "__main__":
    main()
