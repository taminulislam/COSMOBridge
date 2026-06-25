"""Generate publication-quality architecture diagram for COSMOBridge v3.

Style matches moe_architecture.png: multimodal samples on left,
encoders in center, fusion + routing on right.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path

OUT = "paper/figures/cosmobridge_v3_architecture.png"


def draw_box(ax, xy, w, h, text, color='#E8F4FD', edge='#2196F3',
             fs=9, fw='normal', tc='black', sub=None, ss=6.5, alpha=1.0):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor=edge, linewidth=1.8, alpha=alpha)
    ax.add_patch(box)
    cx, cy = xy[0] + w/2, xy[1] + h/2
    if sub:
        ax.text(cx, cy + 0.22, text, ha='center', va='center', fontsize=fs, fontweight=fw, color=tc)
        ax.text(cx, cy - 0.22, sub, ha='center', va='center', fontsize=ss, color='#555', style='italic')
    else:
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, fontweight=fw, color=tc)


def draw_arrow(ax, s, e, color='#666', lw=1.5, style='->', cs=None):
    kw = dict(arrowstyle=style, color=color, lw=lw, mutation_scale=13)
    if cs: kw['connectionstyle'] = cs
    ax.add_patch(FancyArrowPatch(s, e, **kw))


def frozen_badge(ax, x, y):
    ax.text(x, y, '❄ FROZEN', fontsize=6.5, fontweight='bold', color='#0D47A1',
            ha='center', bbox=dict(boxstyle='round,pad=0.15', facecolor='#E3F2FD',
                                    edgecolor='#1565C0', linewidth=1))


def main():
    fig, ax = plt.subplots(1, 1, figsize=(22, 13))
    ax.set_xlim(-1, 22)
    ax.set_ylim(-1, 13)
    ax.axis('off')

    # ── Title ──
    ax.text(11, 12.5, 'COSMOBridge', fontsize=24, fontweight='bold',
            color='#1A237E', ha='center', fontfamily='serif')
    ax.text(11, 11.9, 'Bridging Molecular Graphs and COSMO Surfaces via\n'
            'Dual-Path Property-Adaptive Fusion', fontsize=10,
            color='#333', ha='center', style='italic')

    # ══════════════════════════════════════════════════════════
    # LEFT: Multimodal Input Samples
    # ══════════════════════════════════════════════════════════
    ax.text(1.5, 10.8, 'Multimodal Inputs', fontsize=12, fontweight='bold',
            color='#B71C1C', ha='center')

    # 3D COSMO Surface
    draw_box(ax, (0, 8.5), 3.0, 1.8, 'COSMO Surface\nPoint Cloud',
             color='#E8F5E9', edge='#2E7D32', fs=10, fw='bold',
             sub='1024 × 7: xyz + normals + ESP')
    # Small 3D scatter hint
    np.random.seed(42)
    xs = np.random.randn(30) * 0.3 + 0.6
    ys = np.random.randn(30) * 0.2 + 9.6
    colors_sc = plt.cm.RdBu_r(np.random.rand(30))
    ax.scatter(xs, ys, c=colors_sc, s=8, alpha=0.6, zorder=5)

    # Molecular Graph
    draw_box(ax, (0, 6.2), 3.0, 1.8, 'Molecular Graph\n(SMILES)',
             color='#FFEBEE', edge='#D32F2F', fs=10, fw='bold',
             sub='Atoms, bonds, connectivity')
    # Mini graph hint
    nodes_x = [0.5, 1.0, 1.5, 1.0, 2.0]
    nodes_y = [7.3, 7.6, 7.3, 7.0, 7.0]
    for i, (nx, ny) in enumerate(zip(nodes_x, nodes_y)):
        ax.plot(nx, ny, 'o', color='#D32F2F', markersize=5, zorder=5)
    for i, j in [(0,1),(1,2),(1,3),(2,4)]:
        ax.plot([nodes_x[i],nodes_x[j]], [nodes_y[i],nodes_y[j]], '-', color='#D32F2F', lw=1, zorder=4)

    # Thermodynamic Features
    draw_box(ax, (0, 3.9), 3.0, 1.8, 'Thermodynamic\nFeatures',
             color='#FFF3E0', edge='#E65100', fs=10, fw='bold',
             sub='T, x₁, 1/T, T², T³ + 20 descriptors')
    # Mini table hint
    for i, (txt, val) in enumerate([('T', '348K'), ('x₁', '0.5'), ('1/T', '0.003')]):
        ax.text(0.4, 5.3 - i*0.25, f'{txt}={val}', fontsize=5.5, color='#E65100', fontfamily='monospace')

    # ══════════════════════════════════════════════════════════
    # CENTER-LEFT: Frozen Encoders
    # ══════════════════════════════════════════════════════════
    ax.text(5.5, 10.8, 'Pre-trained Encoders', fontsize=12, fontweight='bold',
            color='#1565C0', ha='center')

    # PointNet
    draw_box(ax, (4, 8.5), 3.0, 1.8, 'PointNet Encoder',
             color='#E3F2FD', edge='#1565C0', fs=10, fw='bold',
             sub='SharedMLP → MaxPool → 256D')
    frozen_badge(ax, 6.5, 10.1)

    # Chemprop D-MPNN
    draw_box(ax, (4, 6.2), 3.0, 1.8, 'Chemprop D-MPNN',
             color='#E3F2FD', edge='#1565C0', fs=10, fw='bold',
             sub='Directed bonds, 3 iter → 300D')
    frozen_badge(ax, 6.5, 7.8)
    ax.text(5.5, 6.45, 'Yang et al. 2019 · 265K params',
            fontsize=6, color='#1565C0', ha='center')

    # Arrows: inputs → encoders
    draw_arrow(ax, (3.0, 9.4), (4.0, 9.4), color='#2E7D32', lw=2.5)
    draw_arrow(ax, (3.0, 7.1), (4.0, 7.1), color='#D32F2F', lw=2.5)

    # Dim labels
    ax.text(7.3, 9.6, '256D', fontsize=9, fontweight='bold', color='#2E7D32')
    ax.text(7.3, 7.3, '300D', fontsize=9, fontweight='bold', color='#D32F2F')
    ax.text(3.3, 4.8, '25D', fontsize=9, fontweight='bold', color='#E65100')

    # ══════════════════════════════════════════════════════════
    # CENTER: Dual Path
    # ══════════════════════════════════════════════════════════

    # PATH A: GBH Fusion (top)
    path_a_bg = FancyBboxPatch((8, 7.8), 5.5, 3.0, boxstyle="round,pad=0.15",
                                 facecolor='#F3E5F5', edgecolor='#6A1B9A',
                                 linewidth=2.0, alpha=0.3)
    ax.add_patch(path_a_bg)
    ax.text(10.75, 10.5, 'Path A: GBH Bilinear Fusion', fontsize=11,
            fontweight='bold', color='#4A148C', ha='center')
    frozen_badge(ax, 12.8, 10.5)

    draw_box(ax, (8.3, 9.0), 2.2, 1.0, 'Low-Rank\nBilinear',
             color='#CE93D8', edge='#7B1FA2', fs=8, fw='bold',
             sub='(U·h_graph)⊙(V·h_surface)')

    draw_box(ax, (10.8, 9.0), 2.4, 1.0, 'HyperNet(T)\n+ Residual',
             color='#E1BEE7', edge='#7B1FA2', fs=8, fw='bold',
             sub='T-dependent weights')

    draw_box(ax, (8.3, 8.0), 4.9, 0.7, 'Fused Head: 256→128→GELU→7',
             color='#E1BEE7', edge='#7B1FA2', fs=8)

    # PATH B: Chemprop Readout (bottom)
    path_b_bg = FancyBboxPatch((8, 4.5), 5.5, 2.8, boxstyle="round,pad=0.15",
                                 facecolor='#FFEBEE', edgecolor='#C62828',
                                 linewidth=2.0, alpha=0.3)
    ax.add_patch(path_b_bg)
    ax.text(10.75, 7.0, 'Path B: Chemprop Full Readout', fontsize=11,
            fontweight='bold', color='#B71C1C', ha='center')
    frozen_badge(ax, 12.8, 7.0)

    draw_box(ax, (8.3, 5.5), 2.2, 1.0, 'Chemprop\nFFN',
             color='#FFCDD2', edge='#C62828', fs=8, fw='bold',
             sub='305→300→ReLU→7')

    draw_box(ax, (10.8, 5.5), 2.4, 1.0, 'Jointly-trained\nMPN+FFN',
             color='#FFCDD2', edge='#C62828', fs=8, fw='bold',
             sub='G_E=0.748, H_E=0.731')

    draw_box(ax, (8.3, 4.7), 4.9, 0.6, 'Chemprop predictions (7 properties)',
             color='#FFCDD2', edge='#C62828', fs=8)

    # Arrows: encoders → paths
    draw_arrow(ax, (7.0, 9.4), (8.3, 9.5), color='#2E7D32', lw=2)
    draw_arrow(ax, (7.0, 7.3), (8.3, 9.3), color='#D32F2F', lw=2,
               cs="arc3,rad=-0.2")
    draw_arrow(ax, (7.0, 7.0), (8.3, 6.0), color='#D32F2F', lw=2,
               cs="arc3,rad=0.1")
    draw_arrow(ax, (3.0, 4.8), (8.3, 5.8), color='#E65100', lw=2,
               cs="arc3,rad=-0.1")
    draw_arrow(ax, (3.0, 5.2), (8.3, 9.2), color='#E65100', lw=1.5,
               cs="arc3,rad=-0.3")

    # ══════════════════════════════════════════════════════════
    # RIGHT: Per-Property Gate + Output
    # ══════════════════════════════════════════════════════════

    # Gate box
    draw_box(ax, (14.5, 6.5), 3.0, 4.0, '', color='#E8EAF6', edge='#283593',
             alpha=0.5)
    ax.text(16.0, 10.2, 'Per-Property Gate', fontsize=11, fontweight='bold',
            color='#1A237E', ha='center')
    ax.text(16.0, 9.7, 'α_p · fusion + (1-α_p) · chemprop',
            fontsize=7, color='#333', ha='center', style='italic')
    ax.text(16.0, 9.3, '7 TRAINABLE PARAMS', fontsize=8, fontweight='bold',
            color='#D32F2F', ha='center')

    # Gate values
    props = ['γ₁', 'γ₂', 'G_E', 'H_E', 'G_mix', 'H_vap', 'P']
    gates = [0.37, 0.39, 0.36, 0.42, 0.45, 0.37, 0.69]
    r2s = [0.889, 0.913, 0.785, 0.775, 0.758, 0.737, 0.839]

    for i, (prop, gate, r2) in enumerate(zip(props, gates, r2s)):
        y = 8.8 - i * 0.35
        # Bar
        bar_w = gate * 1.8
        ax.barh(y, bar_w, height=0.25, left=14.7, color='#7B1FA2', alpha=0.6)
        ax.barh(y, (1-gate)*1.8, height=0.25, left=14.7+bar_w, color='#C62828', alpha=0.6)
        ax.text(14.6, y, f'{prop}', fontsize=7, ha='right', va='center', fontweight='bold')
        ax.text(16.7, y, f'R²={r2:.3f}', fontsize=6.5, ha='left', va='center',
                fontweight='bold', color='#1A237E')

    # Legend for bars
    ax.barh(6.7, 0.3, height=0.2, left=14.7, color='#7B1FA2', alpha=0.6)
    ax.text(15.1, 6.7, 'Fusion', fontsize=6, va='center')
    ax.barh(6.7, 0.3, height=0.2, left=15.6, color='#C62828', alpha=0.6)
    ax.text(16.0, 6.7, 'Chemprop', fontsize=6, va='center')

    # Arrows into gate
    draw_arrow(ax, (13.2, 8.3), (14.5, 8.5), color='#7B1FA2', lw=2)
    draw_arrow(ax, (13.2, 5.0), (14.5, 7.5), color='#C62828', lw=2,
               cs="arc3,rad=-0.2")

    # Output
    draw_box(ax, (18, 7.0), 3.0, 3.5, '', color='#F5F5F5', edge='#333', alpha=0.8)
    ax.text(19.5, 10.2, 'Output', fontsize=12, fontweight='bold', color='#333', ha='center')
    ax.text(19.5, 9.6, '7 IL Properties', fontsize=10, fontweight='bold',
            color='#1A237E', ha='center')

    out_props = [('γ₁ = 0.889', True), ('γ₂ = 0.913', True),
                  ('G_E = 0.785', True), ('H_E = 0.775', True),
                  ('G_mix = 0.758', True), ('H_vap = 0.737', True),
                  ('P = 0.839', True)]
    for i, (txt, win) in enumerate(out_props):
        y = 9.1 - i * 0.35
        c = '#D32F2F' if 'γ' in txt or 'P' in txt else '#1A237E'
        ax.text(19.5, y, txt, fontsize=8, ha='center', fontweight='bold', color=c)

    draw_arrow(ax, (17.5, 8.5), (18.0, 8.5), color='#283593', lw=2.5)

    # Result badge
    ax.text(19.5, 6.8, 'avg R² = 0.814', fontsize=12, fontweight='bold',
            color='white', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#D32F2F', edgecolor='#B71C1C',
                      linewidth=2))

    # ══════════════════════════════════════════════════════════
    # BOTTOM: Innovation callouts
    # ══════════════════════════════════════════════════════════
    innovations = [
        (0.5, 1.8, '① PointNet on 3D COSMO\n    surfaces (novel)', '#2E7D32'),
        (4.5, 1.8, '② Chemprop D-MPNN\n    as frozen sub-module', '#1565C0'),
        (8.5, 1.8, '③ GBH bilinear captures\n    surface×graph interaction', '#7B1FA2'),
        (12.5, 1.8, '④ Learned per-property routing\n    (7 trainable params)', '#283593'),
        (16.5, 1.8, '⑤ Beats Chemprop on ALL\n    7 properties (0.814 avg)', '#D32F2F'),
    ]
    for x, y, text, color in innovations:
        ax.text(x, y, text, fontsize=7.5, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.9, linewidth=1.5))

    # Model summary
    ax.text(11, 0.5, 'COSMOBridge = Chemprop D-MPNN (frozen, 265K) + PointNet COSMO (frozen, 109K) + '
            'GBH Bilinear Fusion (frozen, 471K) + Chemprop Readout (frozen, 94K) + 7 Learned Gates',
            ha='center', fontsize=8, color='#333',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#EDE7F6',
                      edgecolor='#4A148C', linewidth=2))

    plt.tight_layout()
    plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
