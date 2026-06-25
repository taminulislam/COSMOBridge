#!/usr/bin/env python3
"""
Generate a publication-quality architecture diagram for
Variant C: Chemprop D-MPNN + ILThermo Transfer.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# ── colour palette ──────────────────────────────────────────────────────
BLUE   = "#3B82F6"   # original data
BLUE_L = "#DBEAFE"
ORANGE = "#F97316"   # ILThermo
ORANGE_L = "#FFF7ED"
GREEN  = "#10B981"   # D-MPNN
GREEN_L = "#D1FAE5"
PURPLE = "#8B5CF6"   # FFN
PURPLE_L = "#EDE9FE"
GRAY   = "#6B7280"
GRAY_L = "#F3F4F6"
RED    = "#EF4444"
WHITE  = "#FFFFFF"
BLACK  = "#111827"


def _box(ax, xy, w, h, colour, text, fontsize=8, lw=1.2, alpha=0.92,
         text_colour=BLACK, bold=False, zorder=3):
    """Draw a rounded box with centred text."""
    fb = FancyBboxPatch(xy, w, h,
                        boxstyle="round,pad=0.02",
                        facecolor=colour, edgecolor="gray",
                        linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(fb)
    weight = "bold" if bold else "normal"
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color=text_colour, weight=weight, zorder=zorder + 1,
            linespacing=1.35)
    return fb


def _arrow(ax, xy_from, xy_to, colour=GRAY, lw=1.5, style="-|>",
           connectionstyle="arc3,rad=0.0", zorder=2):
    """Draw a simple arrow between two points."""
    ar = FancyArrowPatch(xy_from, xy_to,
                         arrowstyle=style, color=colour,
                         lw=lw, connectionstyle=connectionstyle,
                         zorder=zorder, mutation_scale=12)
    ax.add_patch(ar)
    return ar


def main():
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    # ====================================================================
    # TITLE
    # ====================================================================
    ax.text(9, 10.55, "Variant C: Chemprop D-MPNN + ILThermo Transfer",
            ha="center", va="center", fontsize=16, weight="bold",
            color=BLACK)
    ax.plot([1, 17], [10.3, 10.3], color=GRAY, lw=0.6, zorder=1)

    # ====================================================================
    # LEFT SIDE  --  Data Sources
    # ====================================================================
    # --- Original Dataset ---
    _box(ax, (0.3, 7.8), 3.6, 2.2, BLUE_L,
         "Original Dataset\n\n152 samples  |  19 ILs\n\n"
         "7 targets:\n"
         r"$\gamma_1,\;\gamma_2,\;G^E,\;H^E,$"
         "\n"
         r"$G_{mix},\;H_{vap},\;P$",
         fontsize=8.5, lw=1.6)
    # header stripe
    _box(ax, (0.3, 9.35), 3.6, 0.65, BLUE,
         "Original Dataset", fontsize=10, bold=True,
         text_colour=WHITE, lw=1.6)

    # --- ILThermo Database ---
    _box(ax, (0.3, 4.4), 3.6, 2.6, ORANGE_L,
         "ILThermo Database\n\n3,654 samples  |  96 ILs\n\n"
         r"$\gamma_1$  MASKED  "
         + "\u2716\n"
         r"$H^E$  only",
         fontsize=8.5, lw=1.6)
    _box(ax, (0.3, 6.35), 3.6, 0.65, ORANGE,
         "ILThermo Database", fontsize=10, bold=True,
         text_colour=WHITE, lw=1.6)

    # gamma1 MASKED X highlight
    ax.text(2.1, 5.1, "\u2716", fontsize=16, color=RED,
            ha="center", va="center", weight="bold", zorder=5)

    # ====================================================================
    # CENTER  --  Data Processing / Merging
    # ====================================================================
    _box(ax, (5.0, 8.1), 3.0, 1.5, BLUE_L,
         "Oversampled 48\u00d7\n\n7,296 rows (67%)",
         fontsize=9, lw=1.2)

    _box(ax, (5.0, 5.3), 3.0, 1.3, ORANGE_L,
         "ILThermo\n\n3,654 rows (33%)",
         fontsize=9, lw=1.2)

    # Merged box
    _box(ax, (5.0, 3.2), 3.0, 1.4, GRAY_L,
         "Merged Training Set\n\n10,950 samples",
         fontsize=9, bold=True, lw=1.6)

    # Arrows: sources -> processing
    _arrow(ax, (3.9, 9.0), (5.0, 9.0), colour=BLUE, lw=2)
    _arrow(ax, (3.9, 5.9), (5.0, 5.9), colour=ORANGE, lw=2)

    # Arrows: processing -> merged
    _arrow(ax, (6.5, 8.1), (6.5, 4.6), colour=BLUE, lw=1.5)
    _arrow(ax, (6.5, 5.3), (6.5, 4.6), colour=ORANGE, lw=1.5)

    # ====================================================================
    # RIGHT SIDE  --  Chemprop D-MPNN Architecture
    # ====================================================================
    rx = 9.2  # left edge of right column

    # -- SMILES input --
    _box(ax, (rx, 9.3), 2.4, 0.6, GRAY_L,
         "SMILES Input", fontsize=9, bold=True, lw=1.0)

    _arrow(ax, (rx + 1.2, 9.3), (rx + 1.2, 8.95), colour=GRAY)

    # -- RDKit Molecular Graph --
    _box(ax, (rx, 8.3), 2.4, 0.6, GREEN_L,
         "RDKit Molecular\nGraph", fontsize=8.5, lw=1.0)

    _arrow(ax, (rx + 1.2, 8.3), (rx + 1.2, 7.95), colour=GREEN)

    # -- Atom + Bond features side by side --
    _box(ax, (rx - 0.3, 7.15), 1.6, 0.7, GREEN_L,
         "Atom Features\n(133D)", fontsize=8, lw=1.0)
    _box(ax, (rx + 1.5, 7.15), 1.6, 0.7, GREEN_L,
         "Bond Features\n(147D)", fontsize=8, lw=1.0)

    _arrow(ax, (rx + 0.5, 7.15), (rx + 1.2, 6.75), colour=GREEN, lw=1.2)
    _arrow(ax, (rx + 2.3, 7.15), (rx + 1.2, 6.75), colour=GREEN, lw=1.2)

    # -- Directed Message Passing --
    _box(ax, (rx - 0.1, 5.8), 2.8, 0.9, GREEN,
         "Directed Message\nPassing\n3 iter, hidden=300D",
         fontsize=8.5, bold=True, text_colour=WHITE, lw=1.6)

    # "avoids tottering" annotation
    ax.annotate("avoids\ntottering",
                xy=(rx + 2.7, 6.25), xytext=(rx + 3.6, 6.6),
                fontsize=7.5, color=GREEN, style="italic",
                ha="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))

    _arrow(ax, (rx + 1.2, 5.8), (rx + 1.2, 5.45), colour=GREEN)

    # -- Sum pooling --
    _box(ax, (rx + 0.1, 4.8), 2.2, 0.6, GREEN_L,
         "Sum Pooling\nGraph FP (300D)", fontsize=8.5, lw=1.0)

    _arrow(ax, (rx + 1.2, 4.8), (rx + 1.2, 4.45), colour=GREEN)

    # -- Thermo features branch --
    _box(ax, (rx + 3.2, 4.8), 2.8, 0.6, GRAY_L,
         "5 Thermo Features\n"
         r"$[T,\;x_1,\;1/T,\;T^2,\;T^3]$",
         fontsize=8, lw=1.0)

    # arrow from merged set to thermo features
    _arrow(ax, (8.0, 3.9), (rx + 3.2, 5.1),
           colour=GRAY, lw=1.3,
           connectionstyle="arc3,rad=-0.15")

    # arrow from merged set to SMILES
    _arrow(ax, (8.0, 3.9), (rx, 9.6),
           colour=GRAY, lw=1.3,
           connectionstyle="arc3,rad=-0.25")

    # -- Concatenate --
    _box(ax, (rx + 0.1, 3.9), 2.2, 0.5, GRAY_L,
         "Concatenate (305D)", fontsize=8.5, lw=1.0)

    # arrows into concat
    _arrow(ax, (rx + 4.6, 4.8), (rx + 1.2, 4.4),
           colour=GRAY, lw=1.2, connectionstyle="arc3,rad=0.2")

    _arrow(ax, (rx + 1.2, 3.9), (rx + 1.2, 3.55), colour=PURPLE)

    # -- FFN --
    _box(ax, (rx - 0.1, 2.65), 2.8, 0.85, PURPLE,
         "Feed-Forward Network\n"
         r"305 $\rightarrow$ 300 $\rightarrow$ 300 $\rightarrow$ 7",
         fontsize=8.5, bold=True, text_colour=WHITE, lw=1.6)

    _arrow(ax, (rx + 1.2, 2.65), (rx + 1.2, 2.3), colour=PURPLE)

    # -- Output --
    _box(ax, (rx - 0.3, 1.55), 3.2, 0.7, PURPLE_L,
         "7 IL Properties\n"
         r"$\gamma_1,\;\gamma_2,\;G^E,\;H^E,\;G_{mix},\;H_{vap},\;P$",
         fontsize=8.5, bold=True, lw=1.6)

    # ====================================================================
    # BOTTOM  --  Three innovation callouts
    # ====================================================================
    callout_y = 0.25
    callout_h = 0.85

    _box(ax, (0.3, callout_y), 4.8, callout_h, GREEN_L,
         "ILThermo backbone diversity\n115 unique molecular graphs",
         fontsize=8.5, bold=False, lw=1.2)
    ax.text(0.55, callout_y + callout_h - 0.15, "1",
            fontsize=8, weight="bold", color=WHITE,
            bbox=dict(boxstyle="circle", facecolor=GREEN, edgecolor="none",
                      pad=0.25),
            ha="center", va="center", zorder=5)

    _box(ax, (5.6, callout_y), 5.2, callout_h, ORANGE_L,
         r"$\gamma_1$ masking: removes conflicting"
         "\nILThermo gradients",
         fontsize=8.5, bold=False, lw=1.2)
    ax.text(5.85, callout_y + callout_h - 0.15, "2",
            fontsize=8, weight="bold", color=WHITE,
            bbox=dict(boxstyle="circle", facecolor=ORANGE, edgecolor="none",
                      pad=0.25),
            ha="center", va="center", zorder=5)

    _box(ax, (11.3, callout_y), 5.8, callout_h, BLUE_L,
         "48\u00d7 oversampling:\n67% original dominance in training",
         fontsize=8.5, bold=False, lw=1.2)
    ax.text(11.55, callout_y + callout_h - 0.15, "3",
            fontsize=8, weight="bold", color=WHITE,
            bbox=dict(boxstyle="circle", facecolor=BLUE, edgecolor="none",
                      pad=0.25),
            ha="center", va="center", zorder=5)

    # ====================================================================
    # Section labels
    # ====================================================================
    ax.text(2.1, 10.1, "DATA SOURCES", ha="center", fontsize=9,
            weight="bold", color=GRAY, style="italic")
    ax.text(6.5, 10.1, "PROCESSING", ha="center", fontsize=9,
            weight="bold", color=GRAY, style="italic")
    ax.text(11.5, 10.1, "D-MPNN  +  FFN", ha="center", fontsize=9,
            weight="bold", color=GRAY, style="italic")

    # ====================================================================
    # Save
    # ====================================================================
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "paper", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "variant_c_architecture.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=WHITE, edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
