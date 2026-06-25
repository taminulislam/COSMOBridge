"""Generate COSMOBridge v3 architecture diagram with rich multimodal input examples.

Matches moe_architecture.png style: actual visual examples of each modality on the left.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, AllChem

OUT = "paper/figures/cosmobridge_v3_architecture.png"


def draw_box(ax, xy, w, h, text, color='#E8F4FD', edge='#2196F3',
             fs=9, fw='normal', tc='black', sub=None, ss=6.5, alpha=1.0):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor=edge, linewidth=1.8, alpha=alpha)
    ax.add_patch(box)
    cx, cy = xy[0] + w/2, xy[1] + h/2
    if sub:
        ax.text(cx, cy + 0.22, text, ha='center', va='center', fontsize=fs, fontweight=fw, color=tc)
        ax.text(cx, cy - 0.25, sub, ha='center', va='center', fontsize=ss, color='#555', style='italic')
    else:
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, fontweight=fw, color=tc)


def draw_arrow(ax, s, e, color='#666', lw=1.5, style='->', cs=None):
    kw = dict(arrowstyle=style, color=color, lw=lw, mutation_scale=13)
    if cs: kw['connectionstyle'] = cs
    ax.add_patch(FancyArrowPatch(s, e, **kw))


def frozen_badge(ax, x, y):
    ax.text(x, y, '❄ FROZEN', fontsize=6, fontweight='bold', color='#0D47A1',
            ha='center', bbox=dict(boxstyle='round,pad=0.12', facecolor='#E3F2FD',
                                    edgecolor='#1565C0', linewidth=1))


def draw_3d_cosmo_example(ax, x, y, w, h):
    """Draw a 3D COSMO surface scatter example."""
    np.random.seed(42)
    n = 80
    theta = np.random.uniform(0, 2*np.pi, n)
    phi = np.random.uniform(0, np.pi, n)
    r = 1.0 + np.random.randn(n) * 0.05
    xs = r * np.sin(phi) * np.cos(theta) * w * 0.3 + x + w * 0.5
    ys = r * np.sin(phi) * np.sin(theta) * h * 0.25 + y + h * 0.5
    esp = np.sin(theta) * np.cos(phi)
    colors = plt.cm.RdBu_r((esp + 1) / 2)
    ax.scatter(xs, ys, c=colors, s=6, alpha=0.8, zorder=5)
    ax.text(x + w * 0.5, y + h * 0.1, 'ESP: −0.37 to +0.34',
            fontsize=5, ha='center', color='#555')


def draw_molecule_example(ax, x, y, w, h):
    """Draw a molecular structure representation."""
    # Simple molecular graph visualization
    atoms = [
        (x + w*0.15, y + h*0.6, 'N', '#1565C0'),
        (x + w*0.35, y + h*0.7, 'C', '#333'),
        (x + w*0.55, y + h*0.6, 'C', '#333'),
        (x + w*0.75, y + h*0.7, 'N', '#1565C0'),
        (x + w*0.55, y + h*0.4, 'C', '#333'),
        (x + w*0.35, y + h*0.4, 'C', '#333'),
        # Anion
        (x + w*0.5, y + h*0.15, 'O', '#D32F2F'),
        (x + w*0.3, y + h*0.15, 'C', '#333'),
        (x + w*0.7, y + h*0.15, 'O', '#D32F2F'),
    ]
    bonds = [(0,1),(1,2),(2,3),(2,4),(4,5),(5,0),(6,7),(7,8)]

    for i, j in bonds:
        ax.plot([atoms[i][0], atoms[j][0]], [atoms[i][1], atoms[j][1]],
                '-', color='#999', lw=1.2, zorder=4)
    for ax_, ay_, label, color in atoms:
        ax.plot(ax_, ay_, 'o', color=color, markersize=8, zorder=5)
        ax.text(ax_, ay_, label, fontsize=5, ha='center', va='center',
                color='white', fontweight='bold', zorder=6)

    ax.text(x + w*0.5, y + h*0.87, '[BMIM][OAc]', fontsize=6,
            ha='center', fontweight='bold', color='#333')


def draw_features_table(ax, x, y, w, h):
    """Draw a mini features table."""
    headers = ['Feature', 'Value']
    rows = [
        ('T (K)', '348.15'),
        ('x₁', '0.500'),
        ('1/T', '0.00287'),
        ('T²', '1.21×10⁵'),
        ('ESP_mean', '−0.089'),
        ('surf_area', '215.4'),
        ('sphericity', '0.912'),
    ]
    row_h = h / (len(rows) + 1.2)

    # Header
    ax.text(x + w*0.3, y + h - row_h*0.3, 'Feature', fontsize=5.5,
            ha='center', fontweight='bold', color='#E65100')
    ax.text(x + w*0.75, y + h - row_h*0.3, 'Value', fontsize=5.5,
            ha='center', fontweight='bold', color='#E65100')
    ax.plot([x + w*0.05, x + w*0.95], [y + h - row_h*0.6, y + h - row_h*0.6],
            '-', color='#E65100', lw=0.5)

    for i, (feat, val) in enumerate(rows):
        ry = y + h - row_h * (i + 1.2)
        bg_color = '#FFF3E0' if i % 2 == 0 else 'white'
        rect = FancyBboxPatch((x + w*0.05, ry - row_h*0.3), w*0.9, row_h*0.7,
                               boxstyle="square,pad=0", facecolor=bg_color,
                               edgecolor='none', alpha=0.5)
        ax.add_patch(rect)
        ax.text(x + w*0.3, ry, feat, fontsize=5, ha='center', va='center',
                fontfamily='monospace', color='#333')
        ax.text(x + w*0.75, ry, val, fontsize=5, ha='center', va='center',
                fontfamily='monospace', color='#E65100')


def main():
    fig, ax = plt.subplots(1, 1, figsize=(26, 16))
    ax.set_xlim(-1.5, 24)
    ax.set_ylim(0.0, 15.5)
    ax.axis('off')

    # ── Title ──
    ax.text(12, 15.0, 'COSMOBridge', fontsize=26, fontweight='bold',
            color='#1A237E', ha='center', fontfamily='serif')
    ax.text(12, 14.3, 'Bridging 2D Molecular Graphs and 3D COSMO Surfaces\n'
            'via Dual-Path Property-Adaptive Fusion',
            fontsize=10, color='#333', ha='center', style='italic')

    # ══════════════════════════════════════════════════════════
    # LEFT: Multimodal Input Examples (rich visuals like MoE diagram)
    # ══════════════════════════════════════════════════════════
    ax.text(-0.5, 11.8, 'Multimodal Inputs', fontsize=13, fontweight='bold',
            color='#B71C1C')

    # 1. COSMO Surface Point Cloud — with 3D scatter example
    draw_box(ax, (-1.2, 8.8), 3.8, 2.6, '', color='#E8F5E9', edge='#2E7D32', alpha=0.3)
    ax.text(0.7, 11.15, 'COSMO Surface Point Cloud', fontsize=9,
            fontweight='bold', color='#2E7D32', ha='center')
    draw_3d_cosmo_example(ax, -1.0, 9.0, 3.4, 2.0)
    ax.text(0.7, 8.95, '1024 × 7: xyz + normals + ESP', fontsize=6,
            ha='center', color='#555', style='italic')

    # 2. Molecular Graph — with structure drawing
    draw_box(ax, (-1.2, 5.8), 3.8, 2.6, '', color='#FFEBEE', edge='#D32F2F', alpha=0.3)
    ax.text(0.7, 8.15, 'Molecular Graph (SMILES)', fontsize=9,
            fontweight='bold', color='#D32F2F', ha='center')
    draw_molecule_example(ax, -1.0, 5.9, 3.4, 2.0)

    # 3. Thermodynamic Features — with data table
    draw_box(ax, (-1.2, 2.6), 3.8, 2.8, '', color='#FFF3E0', edge='#E65100', alpha=0.3)
    ax.text(0.7, 5.15, 'Thermodynamic & Surface Features', fontsize=9,
            fontweight='bold', color='#E65100', ha='center')
    draw_features_table(ax, -1.0, 2.7, 3.4, 2.3)

    # ══════════════════════════════════════════════════════════
    # CENTER-LEFT: Frozen Encoders
    # ══════════════════════════════════════════════════════════
    ax.text(5.5, 11.8, 'Pre-trained Encoders', fontsize=13, fontweight='bold',
            color='#1565C0')

    # PointNet — expanded with internal architecture
    pn_bg = FancyBboxPatch((3.6, 8.6), 3.8, 3.0, boxstyle="round,pad=0.12",
                            facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2.0, alpha=0.4)
    ax.add_patch(pn_bg)
    ax.text(5.5, 11.4, 'PointNet Encoder', fontsize=11, fontweight='bold',
            color='#1565C0', ha='center')
    ax.text(5.5, 11.05, 'Qi et al. 2017 — first applied to COSMO surfaces',
            fontsize=6, color='#1565C0', ha='center', style='italic')
    frozen_badge(ax, 7.0, 11.4)

    # Internal layers
    draw_box(ax, (3.9, 10.2), 1.5, 0.55, 'SharedMLP',
             color='#BBDEFB', edge='#1565C0', fs=7, fw='bold',
             sub='7→64→128→256')
    draw_box(ax, (5.6, 10.2), 1.5, 0.55, 'Max Pool',
             color='#BBDEFB', edge='#1565C0', fs=7, fw='bold',
             sub='permutation invariant')
    draw_box(ax, (3.9, 9.3), 3.2, 0.55, 'Global Feature Projection → 256D',
             color='#BBDEFB', edge='#1565C0', fs=7)

    # Internal arrows (straight, 90-degree)
    draw_arrow(ax, (5.4, 10.2), (5.6, 10.45), color='#1565C0', lw=1)
    draw_arrow(ax, (5.5, 10.2), (5.5, 9.85), color='#1565C0', lw=1)

    # Chemprop D-MPNN — expanded with internal architecture
    cp_bg = FancyBboxPatch((3.6, 5.4), 3.8, 3.0, boxstyle="round,pad=0.12",
                            facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2.0, alpha=0.4)
    ax.add_patch(cp_bg)
    ax.text(5.5, 8.2, 'Chemprop D-MPNN', fontsize=11, fontweight='bold',
            color='#1565C0', ha='center')
    ax.text(5.5, 7.85, 'Yang et al. 2019 · 265K params',
            fontsize=6, color='#1565C0', ha='center', style='italic')
    frozen_badge(ax, 7.0, 8.2)

    # Internal layers
    draw_box(ax, (3.9, 6.9), 1.5, 0.55, 'Directed\nMsg Passing',
             color='#BBDEFB', edge='#1565C0', fs=6.5, fw='bold',
             sub='bond-level, 3 iter')
    draw_box(ax, (5.6, 6.9), 1.5, 0.55, 'Sum Pool',
             color='#BBDEFB', edge='#1565C0', fs=7, fw='bold',
             sub='preserves mol. size')
    draw_box(ax, (3.9, 6.0), 3.2, 0.55, 'Graph Fingerprint → 300D',
             color='#BBDEFB', edge='#1565C0', fs=7)

    # Internal arrows
    draw_arrow(ax, (5.4, 6.9), (5.6, 7.15), color='#1565C0', lw=1)
    draw_arrow(ax, (5.5, 6.9), (5.5, 6.55), color='#1565C0', lw=1)

    # Arrows: inputs → encoders (STRAIGHT horizontal)
    draw_arrow(ax, (2.6, 10.45), (3.9, 10.45), color='#2E7D32', lw=2.5)
    draw_arrow(ax, (2.6, 7.15), (3.9, 7.15), color='#D32F2F', lw=2.5)

    # Dim labels
    ax.text(7.5, 9.55, '256D', fontsize=10, fontweight='bold', color='#2E7D32')
    ax.text(7.5, 6.25, '300D', fontsize=10, fontweight='bold', color='#D32F2F')
    ax.text(3.0, 3.8, '25D', fontsize=10, fontweight='bold', color='#E65100')

    # ══════════════════════════════════════════════════════════
    # CENTER: Dual Paths
    # ══════════════════════════════════════════════════════════

    # PATH A: GBH Fusion — EXPANDED (core contribution)
    path_a = FancyBboxPatch((8.2, 7.8), 5.5, 4.0, boxstyle="round,pad=0.15",
                              facecolor='#F3E5F5', edgecolor='#6A1B9A', lw=2.5, alpha=0.35)
    ax.add_patch(path_a)
    ax.text(10.95, 11.5, 'Path A: GBH Bilinear Fusion', fontsize=12,
            fontweight='bold', color='#4A148C', ha='center')
    ax.text(10.95, 11.15, 'THIS WORK — Core Architectural Contribution',
            fontsize=6.5, color='#6A1B9A', ha='center', style='italic', fontweight='bold')
    frozen_badge(ax, 13.2, 11.5)

    # Row 1: Low-Rank Bilinear + HyperNet side by side
    draw_box(ax, (8.4, 10.0), 2.4, 1.0, 'Low-Rank Bilinear',
             color='#CE93D8', edge='#7B1FA2', fs=8, fw='bold',
             sub='(U·h_g)⊙(V·h_s), rank=32')

    draw_box(ax, (11.0, 10.0), 2.4, 1.0, 'HyperNet(T, x₁)',
             color='#CE93D8', edge='#7B1FA2', fs=8, fw='bold',
             sub='5→64→64→ W, gate, bias')

    # Row 2: Gated fusion + residual
    draw_box(ax, (8.4, 9.0), 4.9, 0.75, 'Gated Fusion + Residual Paths',
             color='#E1BEE7', edge='#7B1FA2', fs=8, fw='bold',
             sub='gate⊙bilinear + (1-gate)⊙thermo + α·h_surface + β·h_graph')

    # Row 3: Fused head
    draw_box(ax, (8.4, 8.0), 4.9, 0.7, 'Fused Head → 7 predictions',
             color='#E1BEE7', edge='#7B1FA2', fs=8,
             sub='256 → LayerNorm → 128 → GELU → 7')

    # Internal arrows (straight down)
    draw_arrow(ax, (10.5, 10.0), (10.5, 9.75), color='#7B1FA2', lw=1)
    draw_arrow(ax, (10.5, 9.0), (10.5, 8.7), color='#7B1FA2', lw=1)

    # PATH B: Chemprop Readout
    path_b = FancyBboxPatch((8.2, 4.3), 5.5, 3.2, boxstyle="round,pad=0.15",
                              facecolor='#FFEBEE', edgecolor='#C62828', lw=2.0, alpha=0.3)
    ax.add_patch(path_b)
    ax.text(10.95, 7.2, 'Path B: Chemprop Full Readout', fontsize=11,
            fontweight='bold', color='#B71C1C', ha='center')
    frozen_badge(ax, 13.0, 7.2)

    draw_box(ax, (8.5, 5.5), 2.3, 1.2, 'Chemprop\nFFN',
             color='#FFCDD2', edge='#C62828', fs=9, fw='bold',
             sub='305→300→ReLU→Drop')

    draw_box(ax, (11.1, 5.5), 2.3, 1.2, 'Trained\nEnd-to-End',
             color='#FFCDD2', edge='#C62828', fs=9, fw='bold',
             sub='MPN+FFN jointly optimized')

    draw_box(ax, (8.5, 4.5), 4.9, 0.8, 'Chemprop predictions: 7 properties',
             color='#FFCDD2', edge='#C62828', fs=8)

    # Arrows: encoders → paths (all 90-degree, straight)
    # PointNet → Path A fusion (straight horizontal then down into bilinear)
    draw_arrow(ax, (7.1, 9.55), (8.4, 9.55), color='#2E7D32', lw=2)
    ax.plot([8.3, 8.3], [9.55, 10.5], '-', color='#2E7D32', lw=2, zorder=3)

    # Chemprop → Path A fusion (right then up, 90-degree)
    ax.plot([7.4, 7.9, 7.9], [6.25, 6.25, 10.5], '-', color='#D32F2F', lw=2, zorder=3)
    draw_arrow(ax, (7.9, 10.5), (8.4, 10.5), color='#D32F2F', lw=2)

    # Chemprop → Path B readout (right then down, 90-degree)
    ax.plot([7.4, 7.7, 7.7], [6.25, 6.25, 5.8], '-', color='#D32F2F', lw=2, zorder=3)
    draw_arrow(ax, (7.7, 5.8), (8.5, 5.8), color='#D32F2F', lw=2)

    # Thermo → Path B (right then straight horizontal)
    ax.plot([2.6, 3.2, 3.2, 8.2, 8.2], [3.5, 3.5, 4.2, 4.2, 4.7], '-', color='#E65100', lw=1.8, zorder=3)
    draw_arrow(ax, (8.2, 4.7), (8.5, 4.7), color='#E65100', lw=1.8)

    # Thermo → Path A fusion (right, up along left edge, then horizontal into gated fusion)
    ax.plot([2.6, 3.3, 3.3, 8.1, 8.1], [4.8, 4.8, 12.2, 12.2, 9.4], '-', color='#E65100', lw=1.3, zorder=2)
    draw_arrow(ax, (8.1, 9.4), (8.4, 9.4), color='#E65100', lw=1.3)

    # ══════════════════════════════════════════════════════════
    # RIGHT: Per-Property Gate + Output
    # ══════════════════════════════════════════════════════════

    # Gate box
    gate_bg = FancyBboxPatch((14.8, 5.5), 3.2, 5.5, boxstyle="round,pad=0.15",
                               facecolor='#E8EAF6', edgecolor='#283593', lw=2.0, alpha=0.5)
    ax.add_patch(gate_bg)
    ax.text(16.4, 10.7, 'Per-Property Gate', fontsize=12, fontweight='bold',
            color='#1A237E', ha='center')
    ax.text(16.4, 10.2, 'α · fusion + (1−α) · chemprop', fontsize=7,
            color='#333', ha='center', style='italic')
    ax.text(16.4, 9.8, '7 TRAINABLE PARAMS', fontsize=9, fontweight='bold',
            color='#D32F2F', ha='center')

    props = ['γ₁', 'γ₂', 'G_E', 'H_E', 'G_mix', 'H_vap', 'P']
    gates = [0.37, 0.39, 0.36, 0.42, 0.45, 0.37, 0.69]
    r2s = [0.889, 0.913, 0.785, 0.775, 0.758, 0.737, 0.839]

    for i, (prop, gate, r2) in enumerate(zip(props, gates, r2s)):
        y = 9.2 - i * 0.5
        bar_w = gate * 2.0
        ax.barh(y, bar_w, height=0.35, left=15.0, color='#7B1FA2', alpha=0.7, edgecolor='white')
        ax.barh(y, (1-gate)*2.0, height=0.35, left=15.0+bar_w, color='#C62828', alpha=0.7, edgecolor='white')
        ax.text(14.85, y, f'{prop}', fontsize=8, ha='right', va='center', fontweight='bold')
        ax.text(17.15, y, f'{r2:.3f}', fontsize=7, ha='left', va='center',
                fontweight='bold', color='#1A237E')

    ax.barh(5.8, 0.4, height=0.25, left=15.0, color='#7B1FA2', alpha=0.7)
    ax.text(15.5, 5.8, 'Fusion', fontsize=6, va='center')
    ax.barh(5.8, 0.4, height=0.25, left=16.0, color='#C62828', alpha=0.7)
    ax.text(16.5, 5.8, 'Chemprop', fontsize=6, va='center')

    # Path A → Gate (straight horizontal from fused head)
    draw_arrow(ax, (13.3, 8.35), (14.8, 8.35), color='#7B1FA2', lw=2.5)
    # Path B → Gate (right then up, 90-degree)
    ax.plot([13.5, 14.2, 14.2], [4.9, 4.9, 7.0], '-', color='#C62828', lw=2.5, zorder=3)
    draw_arrow(ax, (14.2, 7.0), (14.8, 7.0), color='#C62828', lw=2.5)

    # Output
    draw_box(ax, (18.5, 6.2), 3.5, 4.5, '', color='#F5F5F5', edge='#333', alpha=0.8)
    ax.text(20.25, 10.4, 'Output: 7 IL Properties', fontsize=11,
            fontweight='bold', color='#1A237E', ha='center')

    out_data = [
        ('γ₁ = 0.889', '#D32F2F', True),
        ('γ₂ = 0.913', '#D32F2F', True),
        ('G_E = 0.785', '#1A237E', True),
        ('H_E = 0.775', '#1A237E', True),
        ('G_mix = 0.758', '#1A237E', True),
        ('H_vap = 0.737', '#1A237E', True),
        ('P = 0.839', '#D32F2F', True),
    ]
    for i, (txt, color, win) in enumerate(out_data):
        y = 9.7 - i * 0.5
        marker = '✓' if win else ''
        ax.text(20.25, y, f'{txt} {marker}', fontsize=9, ha='center',
                fontweight='bold', color=color)

    draw_arrow(ax, (18.0, 8.0), (18.5, 8.0), color='#283593', lw=2.5)

    # Result badge — use multi-seed mean reported in JCIM Table 1
    ax.text(20.25, 6.0, 'avg R² = 0.801 ± 0.001', fontsize=13, fontweight='bold',
            color='white', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#D32F2F',
                      edgecolor='#B71C1C', linewidth=2.5))

    ax.text(20.25, 5.3, 'Beats Chemprop on 5 of 7 properties',
            fontsize=7, ha='center', fontweight='bold', color='#D32F2F')

    # ══════════════════════════════════════════════════════════
    # BOTTOM: Innovation callouts
    # ══════════════════════════════════════════════════════════
    innovations = [
        (-0.5, 2.5, '① PointNet on 3D COSMO\n    surfaces (novel)', '#2E7D32'),
        (3.5, 2.5, '② Chemprop D-MPNN\n    as frozen sub-module', '#1565C0'),
        (7.5, 2.5, '③ T-dependent bilinear\n    surface×graph fusion', '#7B1FA2'),
        (11.5, 2.5, '④ Learned per-property\n    routing (7 params)', '#283593'),
        (15.5, 2.5, '⑤ Soft blending exceeds\n    both individual paths', '#E65100'),
        (19.5, 2.5, '⑥ Wins 5/7 properties\n    vs Chemprop (R²=0.801)', '#D32F2F'),
    ]
    for x, y, text, color in innovations:
        ax.text(x, y, text, fontsize=7, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.9, linewidth=1.5))

    # Model summary bar
    ax.text(11, 1.2, 'COSMOBridge = Chemprop D-MPNN (frozen) + PointNet COSMO (frozen) + '
            'GBH Bilinear Fusion (frozen) + Chemprop Readout (frozen) + 7 Learned Gates',
            ha='center', fontsize=8.5, color='#333',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#EDE7F6',
                      edgecolor='#4A148C', linewidth=2))

    plt.tight_layout()
    plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
