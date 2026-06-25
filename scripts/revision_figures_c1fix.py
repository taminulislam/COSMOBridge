"""Regenerate radar_top_models.png and model_progression.png with the corrected
(unit-bug-fixed) COSMOBridge numbers so the figures match Table 3."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTS = [Path("paper/jcim/figures"), Path("paper/cosmobridge_Overleaf/figures")]
TARGET_LABELS = [r"$\gamma_1$", r"$\gamma_2$", r"$G^E$", r"$H^E$", r"$G_{mix}$", r"$H_{vap}$", r"$P$"]

# corrected per-property R^2 (all official chemprop, original units)
PERPROP = {
    "Chemprop":    [0.828, 0.858, 0.748, 0.731, 0.725, 0.675, 0.825],
    "PointCloud":  [0.765, 0.792, 0.571, 0.538, 0.546, 0.410, 0.422],  # 5-seed means (C2)
    "STILT":       [0.765, 0.791, 0.672, 0.630, 0.639, 0.628, 0.768],
    "COSMOBridge": [0.907, 0.938, 0.704, 0.727, 0.680, 0.648, 0.837],
    "COSMOBridge-v4": [0.906, 0.931, 0.726, 0.728, 0.703, 0.681, 0.841],
}
STYLE = {  # color, linestyle, lw
    "Chemprop": ("#e74c3c", "--", 1.5),
    "PointCloud": ("#2ecc71", "-.", 1.5),
    "STILT": ("#1abc9c", ":", 1.5),
    "COSMOBridge": ("#1f3a93", "-", 3.0),
    "COSMOBridge-v4": ("#e67e22", "-", 2.0),
}


def radar():
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(TARGET_LABELS), endpoint=False).tolist()
    angles += angles[:1]
    for name, vals in PERPROP.items():
        c, ls, lw = STYLE[name]
        v = [max(x, 0) for x in vals] + [max(vals[0], 0)]
        ax.plot(angles, v, ls, linewidth=lw, label=name, color=c)
        ax.fill(angles, v, alpha=0.05, color=c)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(TARGET_LABELS, fontsize=12)
    ax.set_ylim(0, 1.0); ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8)
    ax.legend(loc="lower right", bbox_to_anchor=(1.32, 0), fontsize=9)
    ax.set_title(r"Top Models: Per-Property $R^2$", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    for o in OUTS:
        plt.savefig(o / "radar_top_models.png", dpi=300, bbox_inches="tight")
    plt.close(); print("saved radar_top_models.png")


def progression():
    fig, ax = plt.subplots(figsize=(12, 5))
    prog = [
        ("COSMO-SAC\n(physics)", -0.772, "#95a5a6"),
        ("Tabular\n(baseline)", 0.339, "#bdc3c7"),
        ("GNN v2\n(graph)", 0.578, "#3498db"),
        ("PointCloud\n(3D surface)", 0.578, "#2ecc71"),
        ("MoE\n(ILThermo)", 0.650, "#f39c12"),
        ("CP+AtomSurf\n(per-atom)", 0.721, "#9b59b6"),
        ("STILT\n(transfer)", 0.699, "#1abc9c"),
        ("Chemprop\n(D-MPNN)", 0.770, "#e74c3c"),
        ("COSMOBridge\n(v3)", 0.777, "#1f3a93"),
        ("COSMOBridge-v4\n(router)", 0.788, "#e67e22"),
    ]
    names = [p[0] for p in prog]; vals = [p[1] for p in prog]; colors = [p[2] for p in prog]
    bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
    for i, (v, b) in enumerate(zip(vals, bars)):
        ax.text(i, v + 0.015 if v > 0 else 0.02, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    bars[-1].set_edgecolor("#006400"); bars[-1].set_linewidth(3)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(r"Average $R^2$", fontsize=12)
    ax.set_title("Model Progression: From Physics Baseline to COSMOBridge", fontsize=13, fontweight="bold")
    ax.axhline(y=0.770, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(len(names) - 0.5, 0.782, "Chemprop baseline", fontsize=7, color="red", ha="right")
    ax.set_ylim(-0.1, 0.85); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for o in OUTS:
        plt.savefig(o / "model_progression.png", dpi=300, bbox_inches="tight")
    plt.close(); print("saved model_progression.png")


if __name__ == "__main__":
    radar(); progression()
