"""Generate architecture diagram for the novel multimodal framework.

Shows the complete pipeline:
1. Three input modalities (COSMO surface, Molecular Graph, Thermodynamic features)
2. Modality-specific encoders (PointNet, GAT-GNN, Feature MLP)
3. Cross-attention fusion
4. Two prediction pathways:
   a. Direct prediction head (PointCloud model)
   b. Property-Conditioned MoE with expert routing (MoE model)
5. Ensemble combination
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_box(ax, xy, w, h, text, color='#E8F4FD', edge_color='#2196F3',
             fontsize=9, fontweight='normal', text_color='black', alpha=1.0,
             subtext=None, subtext_size=6.5):
    """Draw a rounded box with text."""
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor=edge_color,
                         linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    cx, cy = xy[0] + w/2, xy[1] + h/2
    if subtext:
        ax.text(cx, cy + 0.15, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color)
        ax.text(cx, cy - 0.2, subtext, ha='center', va='center',
                fontsize=subtext_size, color='#555555', style='italic')
    else:
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color)


def draw_arrow(ax, start, end, color='#666666', lw=1.2, style='->', connectionstyle=None):
    """Draw an arrow between two points."""
    if connectionstyle:
        arrow = FancyArrowPatch(start, end, arrowstyle=style,
                                connectionstyle=connectionstyle,
                                color=color, lw=lw, mutation_scale=12)
    else:
        arrow = FancyArrowPatch(start, end, arrowstyle=style,
                                color=color, lw=lw, mutation_scale=12)
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(22, 14))
    ax.set_xlim(-1, 21)
    ax.set_ylim(-1, 13)
    ax.axis('off')

    # Title
    ax.text(10, 12.5, 'Multimodal COSMO-Surface Framework for Ionic Liquid Property Prediction',
            ha='center', va='center', fontsize=16, fontweight='bold', color='#1a1a1a')
    ax.text(10, 12.0, 'Novel integration: PointNet (3D COSMO surface) + GAT (molecular graph) + '
            'thermodynamic features → cross-attention fusion → MoE routing',
            ha='center', va='center', fontsize=8.5, color='#555555', style='italic')

    # ══════════════════════════════════════════════
    # INPUT MODALITIES (left side)
    # ══════════════════════════════════════════════

    # Section label
    ax.text(1.1, 11.2, 'Input Modalities', ha='center', fontsize=11,
            fontweight='bold', color='#D32F2F')

    # 1. COSMO Surface Point Cloud
    draw_box(ax, (0, 9.2), 2.2, 1.6, 'COSMO Surface\nPoint Cloud',
             color='#FFEBEE', edge_color='#E53935', fontsize=9, fontweight='bold',
             subtext='(N × 7): xyz + normals + ESP')

    # 2. Molecular Graph
    draw_box(ax, (0, 7.0), 2.2, 1.6, 'Molecular\nGraph',
             color='#FFF3E0', edge_color='#FF9800', fontsize=9, fontweight='bold',
             subtext='Atoms, bonds, connectivity')

    # 3. Thermodynamic Features
    draw_box(ax, (0, 4.8), 2.2, 1.6, 'Thermodynamic\n& Surface Features',
             color='#E8F5E9', edge_color='#4CAF50', fontsize=9, fontweight='bold',
             subtext='T, x₁, 1/T, T², T³ + 20 descriptors')

    # ══════════════════════════════════════════════
    # ENCODERS
    # ══════════════════════════════════════════════

    ax.text(4.3, 11.2, 'Modality Encoders', ha='center', fontsize=11,
            fontweight='bold', color='#1565C0')

    # PointNet Encoder
    draw_box(ax, (3.2, 9.2), 2.2, 1.6, 'PointNet\nEncoder',
             color='#E3F2FD', edge_color='#1976D2', fontsize=9, fontweight='bold',
             subtext='SharedMLP → MaxPool → 256D')

    # GAT-GNN Encoder
    draw_box(ax, (3.2, 7.0), 2.2, 1.6, 'GAT-GNN\nEncoder',
             color='#E3F2FD', edge_color='#1976D2', fontsize=9, fontweight='bold',
             subtext='4-layer GAT, 4 heads → 256D')

    # Feature MLP
    draw_box(ax, (3.2, 4.8), 2.2, 1.6, 'Feature\nProjection',
             color='#E3F2FD', edge_color='#1976D2', fontsize=9, fontweight='bold',
             subtext='Linear → ReLU → Dropout → 256D')

    # Arrows: inputs → encoders
    draw_arrow(ax, (2.2, 10.0), (3.2, 10.0), color='#E53935', lw=1.8)
    draw_arrow(ax, (2.2, 7.8), (3.2, 7.8), color='#FF9800', lw=1.8)
    draw_arrow(ax, (2.2, 5.6), (3.2, 5.6), color='#4CAF50', lw=1.8)

    # "Pre-trained" label on GNN
    ax.text(4.3, 6.85, '(transfer-learned)', ha='center', fontsize=6.5,
            color='#1565C0', style='italic')

    # ══════════════════════════════════════════════
    # CROSS-ATTENTION FUSION (center)
    # ══════════════════════════════════════════════

    ax.text(7.8, 11.2, 'Cross-Attention Fusion', ha='center', fontsize=11,
            fontweight='bold', color='#6A1B9A')

    # Main fusion box
    fusion_box = FancyBboxPatch((6.3, 5.5), 3.0, 5.0, boxstyle="round,pad=0.15",
                                 facecolor='#F3E5F5', edgecolor='#7B1FA2',
                                 linewidth=2.0, alpha=0.4)
    ax.add_patch(fusion_box)

    # Sub-components inside fusion
    draw_box(ax, (6.6, 9.2), 2.4, 0.9, 'Surface → Graph\nCross-Attention',
             color='#CE93D8', edge_color='#7B1FA2', fontsize=8,
             subtext='8-head attention')

    draw_box(ax, (6.6, 8.0), 2.4, 0.9, 'Graph → Surface\nCross-Attention',
             color='#CE93D8', edge_color='#7B1FA2', fontsize=8,
             subtext='8-head attention')

    draw_box(ax, (6.6, 6.7), 2.4, 0.9, 'Learnable\nModality Weights',
             color='#E1BEE7', edge_color='#7B1FA2', fontsize=8,
             subtext='softmax(w₁, w₂, w₃)')

    draw_box(ax, (6.6, 5.6), 2.4, 0.8, 'Concat → Fusion MLP',
             color='#E1BEE7', edge_color='#7B1FA2', fontsize=8,
             subtext='768D → LayerNorm → GELU → 256D')

    # Arrows: encoders → fusion
    draw_arrow(ax, (5.4, 10.0), (6.6, 9.65), color='#1976D2', lw=1.5)
    draw_arrow(ax, (5.4, 7.8), (6.6, 8.45), color='#1976D2', lw=1.5)
    draw_arrow(ax, (5.4, 5.6), (6.6, 6.0), color='#1976D2', lw=1.5)

    # Residual connection labels
    ax.text(6.45, 9.0, '+', fontsize=10, fontweight='bold', color='#7B1FA2')
    ax.text(6.45, 7.8, '+', fontsize=10, fontweight='bold', color='#7B1FA2')

    # Output label
    ax.text(7.8, 5.3, 'Fused Representation h ∈ ℝ²⁵⁶', ha='center',
            fontsize=8, fontweight='bold', color='#4A148C')

    # ══════════════════════════════════════════════
    # TWO PREDICTION PATHWAYS (right side)
    # ══════════════════════════════════════════════

    # PATH A: Direct Prediction (PointCloud model)
    ax.text(12.5, 11.2, 'Path A: Direct Prediction', ha='center', fontsize=10,
            fontweight='bold', color='#00695C')

    draw_box(ax, (11.0, 9.5), 3.0, 1.3, 'Prediction Head',
             color='#E0F2F1', edge_color='#00897B', fontsize=9, fontweight='bold',
             subtext='256 → BN → GELU → 128 → 7 targets')

    draw_arrow(ax, (9.3, 7.5), (11.0, 10.1), color='#00897B', lw=2.0,
               connectionstyle="arc3,rad=-0.2")

    # Result box
    draw_box(ax, (11.3, 8.3), 2.4, 0.8, 'PointCloud Model',
             color='#B2DFDB', edge_color='#00695C', fontsize=8, fontweight='bold',
             subtext='Avg R² = 0.655')

    # PATH B: MoE (Fix6 model)
    ax.text(17.5, 11.2, 'Path B: Mixture of Experts', ha='center', fontsize=10,
            fontweight='bold', color='#BF360C')

    # Expert heads
    expert_colors = ['#FFCCBC', '#FFE0B2', '#FFF9C4', '#DCEDC8']
    expert_names = ['Expert 1\n(Surface)', 'Expert 2\n(Mixing)', 'Expert 3\n(Thermo)', 'Expert 4\n(General)']
    for i in range(4):
        y = 9.8 - i * 1.0
        draw_box(ax, (15.5, y), 1.6, 0.7, expert_names[i],
                 color=expert_colors[i], edge_color='#E64A19', fontsize=7, fontweight='bold')

    # Gating network
    draw_box(ax, (15.5, 5.6), 1.6, 1.2, 'Property-\nConditioned\nGating',
             color='#FCE4EC', edge_color='#C62828', fontsize=8, fontweight='bold',
             subtext='h + prop_embed → softmax')

    # Weighted sum
    draw_box(ax, (17.8, 8.0), 1.5, 1.5, '  Σ  \nWeighted\nSum',
             color='#FBE9E7', edge_color='#BF360C', fontsize=9, fontweight='bold')

    # Arrow: fusion → experts
    draw_arrow(ax, (9.3, 6.5), (15.5, 10.1), color='#E64A19', lw=2.0,
               connectionstyle="arc3,rad=-0.15")
    draw_arrow(ax, (9.3, 6.0), (15.5, 6.2), color='#C62828', lw=1.5,
               connectionstyle="arc3,rad=0.1")

    # Arrows: experts → weighted sum
    for i in range(4):
        y = 10.15 - i * 1.0
        draw_arrow(ax, (17.1, y), (17.8, 8.75), color='#E64A19', lw=1.0)

    # Arrow: gating → weighted sum
    draw_arrow(ax, (17.1, 6.2), (17.8, 8.0), color='#C62828', lw=1.5,
               connectionstyle="arc3,rad=-0.2")

    # Gate weight labels
    ax.text(17.5, 7.6, 'w₁...w₄ per property', ha='center',
            fontsize=6.5, color='#C62828', style='italic')

    # Result box
    draw_box(ax, (17.5, 5.6), 2.0, 0.8, 'MoE Fix6 Model',
             color='#FFCCBC', edge_color='#BF360C', fontsize=8, fontweight='bold',
             subtext='Avg R² = 0.650')

    # ══════════════════════════════════════════════
    # OUTPUT (predictions)
    # ══════════════════════════════════════════════

    # Output targets box
    draw_box(ax, (11.0, 1.0), 3.0, 1.8, '7 IL Properties',
             color='#F5F5F5', edge_color='#424242', fontsize=10, fontweight='bold')

    targets = ['γ₁', 'γ₂', 'Gᴱ', 'Hᴱ', 'Gₘᵢₓ', 'Hᵥₐₚ', 'P']
    for i, t in enumerate(targets):
        ax.text(11.3 + (i % 4) * 0.7, 2.2 - (i // 4) * 0.5, t,
                fontsize=8, ha='center', fontweight='bold', color='#333333')

    # Arrows from both paths to output
    draw_arrow(ax, (12.5, 8.3), (12.5, 2.8), color='#00897B', lw=2.0, style='->')
    draw_arrow(ax, (18.5, 5.6), (14.0, 2.8), color='#BF360C', lw=2.0,
               style='->', connectionstyle="arc3,rad=0.3")

    # ══════════════════════════════════════════════
    # ENSEMBLE BOX (bottom)
    # ══════════════════════════════════════════════

    ens_box = FancyBboxPatch((15.5, 1.0), 4.0, 1.8, boxstyle="round,pad=0.15",
                              facecolor='#E8EAF6', edgecolor='#283593',
                              linewidth=2.0, alpha=0.8)
    ax.add_patch(ens_box)
    ax.text(17.5, 2.3, 'Ensemble Strategy', ha='center', fontsize=9,
            fontweight='bold', color='#1A237E')
    ax.text(17.5, 1.8, 'Per-property optimized weights:', ha='center',
            fontsize=7.5, color='#333333')
    ax.text(17.5, 1.4, 'γ₁,γ₂,Gᴱ,Gₘᵢₓ → PointCloud\nHᵥₐₚ, P → MoE Fix6',
            ha='center', fontsize=7, color='#555555', linespacing=1.4)

    # Arrow from output to ensemble
    draw_arrow(ax, (14.0, 1.9), (15.5, 1.9), color='#283593', lw=1.5)

    # ══════════════════════════════════════════════
    # NOVEL CONTRIBUTION HIGHLIGHTS
    # ══════════════════════════════════════════════

    # Innovation callouts
    innovations = [
        (0.5, 3.5, '① PointNet on COSMO\n    surfaces (novel)'),
        (0.5, 2.5, '② Tri-modal cross-attention\n    fusion (novel combination)'),
        (0.5, 1.5, '③ Property-conditioned\n    MoE gating (novel)'),
        (0.5, 0.5, '④ ILThermo transfer with\n    distribution alignment'),
    ]
    for x, y, text in innovations:
        ax.text(x, y, text, fontsize=7.5, color='#1A237E', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8EAF6',
                          edgecolor='#3F51B5', alpha=0.7))

    # ══════════════════════════════════════════════
    # DATA FLOW ANNOTATIONS
    # ══════════════════════════════════════════════

    # Dimension annotations
    ax.text(2.7, 10.3, '(B,1024,7)', fontsize=6, color='#888888', rotation=0)
    ax.text(2.7, 8.1, 'atom/bond\nfeatures', fontsize=6, color='#888888')
    ax.text(2.7, 5.9, '(B, 25)', fontsize=6, color='#888888')

    ax.text(5.6, 10.3, '(B, 256)', fontsize=6, color='#888888')
    ax.text(5.6, 8.1, '(B, 256)', fontsize=6, color='#888888')
    ax.text(5.6, 5.9, '(B, 256)', fontsize=6, color='#888888')

    plt.tight_layout()
    out_path = "paper/figures/full_architecture.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_path}")

    # Also save a simplified version
    print("Done!")


if __name__ == "__main__":
    main()
