"""Generate architecture diagram for COSMOBridge model (CP-GBH Hybrid).

COSMOBridge: Bridging 2D Molecular Graphs and 3D COSMO Surfaces
via Temperature-Conditioned Bilinear Fusion
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = "paper/figures/cosmobridge_architecture.png"


def draw_box(ax, xy, w, h, text, color='#E8F4FD', edge_color='#2196F3',
             fontsize=9, fontweight='normal', text_color='black',
             subtext=None, subtext_size=6.5, alpha=1.0):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor=edge_color,
                         linewidth=1.8, alpha=alpha)
    ax.add_patch(box)
    cx, cy = xy[0] + w/2, xy[1] + h/2
    if subtext:
        ax.text(cx, cy + 0.18, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color)
        ax.text(cx, cy - 0.22, subtext, ha='center', va='center',
                fontsize=subtext_size, color='#555555', style='italic')
    else:
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color)


def draw_arrow(ax, start, end, color='#666666', lw=1.5, style='->',
               connectionstyle=None):
    if connectionstyle:
        arrow = FancyArrowPatch(start, end, arrowstyle=style,
                                connectionstyle=connectionstyle,
                                color=color, lw=lw, mutation_scale=14)
    else:
        arrow = FancyArrowPatch(start, end, arrowstyle=style,
                                color=color, lw=lw, mutation_scale=14)
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(-0.5, 20)
    ax.set_ylim(-0.5, 13.5)
    ax.axis('off')

    # ── Title ──
    ax.text(10, 13.0, 'COSMOBridge', ha='center', va='center',
            fontsize=22, fontweight='bold', color='#1a237e',
            fontfamily='serif')
    ax.text(10, 12.4, 'Bridging 2D Molecular Graphs and 3D COSMO Surfaces\n'
            'via Temperature-Conditioned Bilinear Fusion',
            ha='center', va='center', fontsize=10, color='#333333', style='italic')

    # ── FROZEN BADGE ──
    def frozen_badge(x, y):
        ax.text(x, y, '❄ FROZEN', fontsize=7, fontweight='bold', color='#0D47A1',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#E3F2FD',
                          edgecolor='#1565C0', linewidth=1.2))

    # ══════════════════════════════════════════════════════════
    # LEFT: Input Modalities
    # ══════════════════════════════════════════════════════════
    ax.text(1.5, 11.5, 'Input Modalities', ha='center', fontsize=12,
            fontweight='bold', color='#B71C1C')

    # SMILES input
    draw_box(ax, (0.2, 9.5), 2.6, 1.5, 'SMILES\nMolecular Graph',
             color='#FFEBEE', edge_color='#D32F2F', fontsize=10, fontweight='bold',
             subtext='Atoms, bonds, connectivity')

    # COSMO Point Cloud
    draw_box(ax, (0.2, 7.2), 2.6, 1.5, 'COSMO Surface\nPoint Cloud',
             color='#E8F5E9', edge_color='#2E7D32', fontsize=10, fontweight='bold',
             subtext='1024 × 7: xyz + normals + ESP')

    # Thermo features
    draw_box(ax, (0.2, 4.9), 2.6, 1.5, 'Thermodynamic\nFeatures',
             color='#FFF3E0', edge_color='#E65100', fontsize=10, fontweight='bold',
             subtext='T, x₁, 1/T, T², T³ + descriptors')

    # ══════════════════════════════════════════════════════════
    # CENTER-LEFT: Frozen Encoders
    # ══════════════════════════════════════════════════════════
    ax.text(5.5, 11.5, 'Pre-trained Encoders', ha='center', fontsize=12,
            fontweight='bold', color='#1565C0')

    # Chemprop D-MPNN
    draw_box(ax, (3.8, 9.5), 3.4, 1.5, 'Chemprop D-MPNN',
             color='#E3F2FD', edge_color='#1565C0', fontsize=10, fontweight='bold',
             subtext='Directed bond-level messages → 300D')
    frozen_badge(6.8, 10.8)

    # Details inside
    ax.text(5.5, 9.85, '3 iterations · sum pool · ~300K params',
            ha='center', fontsize=6.5, color='#1565C0')

    # PointNet
    draw_box(ax, (3.8, 7.2), 3.4, 1.5, 'PointNet Encoder',
             color='#E8F5E9', edge_color='#2E7D32', fontsize=10, fontweight='bold',
             subtext='Shared MLPs → max pool → 256D')
    frozen_badge(6.8, 8.5)

    ax.text(5.5, 7.55, 'Surface electrostatic features',
            ha='center', fontsize=6.5, color='#2E7D32')

    # Arrows: inputs → encoders
    draw_arrow(ax, (2.8, 10.25), (3.8, 10.25), color='#D32F2F', lw=2)
    draw_arrow(ax, (2.8, 7.95), (3.8, 7.95), color='#2E7D32', lw=2)

    # Feature dimension labels
    ax.text(7.5, 10.5, '300D', fontsize=8, fontweight='bold', color='#1565C0')
    ax.text(7.5, 8.2, '256D', fontsize=8, fontweight='bold', color='#2E7D32')
    ax.text(3.3, 5.65, '25D', fontsize=8, fontweight='bold', color='#E65100')

    # ══════════════════════════════════════════════════════════
    # CENTER: GBH Fusion (the novel part)
    # ══════════════════════════════════════════════════════════

    # Main fusion box
    fusion_bg = FancyBboxPatch((8.0, 3.8), 5.5, 7.8, boxstyle="round,pad=0.2",
                                facecolor='#F3E5F5', edgecolor='#6A1B9A',
                                linewidth=2.5, alpha=0.3)
    ax.add_patch(fusion_bg)

    ax.text(10.75, 11.2, 'GBH Bilinear Fusion', ha='center', fontsize=13,
            fontweight='bold', color='#4A148C')
    ax.text(10.75, 10.7, 'TRAINABLE · 471K params', ha='center', fontsize=8,
            fontweight='bold', color='#7B1FA2')

    # Projection layers
    draw_box(ax, (8.3, 9.3), 2.0, 0.8, 'Graph Proj',
             color='#CE93D8', edge_color='#7B1FA2', fontsize=8, fontweight='bold',
             subtext='300 → 256D')

    draw_box(ax, (8.3, 7.9), 2.0, 0.8, 'Surface Proj',
             color='#CE93D8', edge_color='#7B1FA2', fontsize=8, fontweight='bold',
             subtext='256 → 256D')

    # Low-rank bilinear
    draw_box(ax, (8.3, 6.3), 4.8, 1.2, 'Low-Rank Bilinear\n(U · h_graph) ⊙ (V · h_surface)',
             color='#E1BEE7', edge_color='#7B1FA2', fontsize=9, fontweight='bold',
             subtext='Rank 32 · captures cross-modal interaction')

    # HyperNetwork
    draw_box(ax, (8.3, 4.5), 2.3, 1.4, 'HyperNetwork\n(T, x₁)',
             color='#FCE4EC', edge_color='#C62828', fontsize=9, fontweight='bold',
             subtext='5→64→64→ W(T), gate(T)')

    # Gated fusion
    draw_box(ax, (11.0, 4.5), 2.2, 1.4, 'Gated Fusion\n+ Residual',
             color='#E1BEE7', edge_color='#7B1FA2', fontsize=9, fontweight='bold',
             subtext='gate⊙bilinear + (1-gate)⊙thermo')

    # Arrows inside fusion
    draw_arrow(ax, (7.2, 10.0), (8.3, 9.7), color='#1565C0', lw=2)
    draw_arrow(ax, (7.2, 8.0), (8.3, 8.3), color='#2E7D32', lw=2)
    draw_arrow(ax, (10.3, 9.3), (10.3, 7.5), color='#7B1FA2', lw=1.5)
    draw_arrow(ax, (10.3, 7.9), (10.3, 7.5), color='#7B1FA2', lw=1.5)
    draw_arrow(ax, (10.7, 6.3), (12.1, 5.9), color='#7B1FA2', lw=1.5)
    draw_arrow(ax, (9.5, 5.9), (9.5, 7.5), color='#C62828', lw=1.5,
               connectionstyle="arc3,rad=0.3")
    draw_arrow(ax, (2.8, 5.65), (8.3, 5.2), color='#E65100', lw=2,
               connectionstyle="arc3,rad=-0.15")

    # Residual paths
    ax.annotate('', xy=(8.3, 4.2), xytext=(7.5, 8.0),
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1, ls='--'))
    ax.annotate('', xy=(8.5, 4.0), xytext=(7.5, 10.0),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1, ls='--'))
    ax.text(7.3, 6.0, 'α·residual', fontsize=6, color='#2E7D32', rotation=70)
    ax.text(7.0, 7.0, 'β·residual', fontsize=6, color='#1565C0', rotation=70)

    # ══════════════════════════════════════════════════════════
    # RIGHT: Output
    # ══════════════════════════════════════════════════════════

    # Prediction head
    draw_box(ax, (14.2, 6.0), 2.5, 1.5, 'Prediction Head',
             color='#E8EAF6', edge_color='#283593', fontsize=10, fontweight='bold',
             subtext='256→128→BN→GELU→7')

    draw_arrow(ax, (13.2, 5.2), (14.2, 6.75), color='#7B1FA2', lw=2)

    # Output properties
    draw_box(ax, (14.0, 3.5), 3.0, 2.0, '7 IL Properties',
             color='#F5F5F5', edge_color='#424242', fontsize=10, fontweight='bold')

    props = ['γ₁ = 0.908', 'γ₂ = 0.936', 'G_E', 'H_E', 'G_mix', 'H_vap', 'P = 0.829']
    colors_p = ['#D32F2F', '#D32F2F', '#666', '#666', '#666', '#666', '#D32F2F']
    for i, (p, c) in enumerate(zip(props, colors_p)):
        row = i // 3
        col = i % 3
        ax.text(14.3 + col * 1.0, 5.0 - row * 0.5, p, fontsize=7,
                fontweight='bold' if c == '#D32F2F' else 'normal', color=c)

    draw_arrow(ax, (15.5, 6.0), (15.5, 5.5), color='#283593', lw=2)

    ax.text(17.3, 4.5, 'NEW BEST', fontsize=9, fontweight='bold', color='#D32F2F',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFCDD2', edgecolor='#D32F2F'))
    ax.text(17.3, 3.9, 'γ₁ = 0.908\nγ₂ = 0.936\nP = 0.829',
            fontsize=8, fontweight='bold', color='#B71C1C', ha='center')

    # ══════════════════════════════════════════════════════════
    # BOTTOM: Innovation callouts
    # ══════════════════════════════════════════════════════════
    innovations = [
        (1.0, 2.5, '① Frozen specialist encoders\n    (Chemprop + PointNet)', '#1565C0'),
        (6.0, 2.5, '② Low-rank bilinear captures\n    graph × surface interaction', '#7B1FA2'),
        (11.0, 2.5, '③ HyperNet generates\n    T-dependent fusion weights', '#C62828'),
        (16.0, 2.5, '④ Record γ₁=0.908, γ₂=0.936\n    Best activity coefficients', '#D32F2F'),
    ]
    for x, y, text, color in innovations:
        ax.text(x, y, text, fontsize=7.5, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.9, linewidth=1.5))

    # ── Model name badge ──
    ax.text(10, 1.2, 'COSMOBridge = Chemprop D-MPNN (300D) + PointNet COSMO (256D) '
            '+ GBH Bilinear Fusion (471K trainable)',
            ha='center', fontsize=8, color='#333',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#EDE7F6',
                      edgecolor='#4A148C', linewidth=2))

    ax.text(10, 0.5, 'Frozen encoders (pre-trained) + Trainable fusion = '
            'Best of both worlds for low-data molecular multimodal learning',
            ha='center', fontsize=7.5, color='#555', style='italic')

    plt.tight_layout()
    plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
