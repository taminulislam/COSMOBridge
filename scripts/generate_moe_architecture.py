"""Generate MoE architecture diagram in the style of wavepropnet_architecture.png.

Clean boxes with colored backgrounds, readable fonts, no arrow-text overlap.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(22, 13))
ax.set_xlim(0, 22)
ax.set_ylim(0, 13)
ax.axis('off')
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# ── Color scheme ──
C_INPUT = '#E8D5B7'      # Warm beige (inputs)
C_POINTNET = '#FADBD8'   # Light pink (PointNet)
C_GNN = '#D5F5E3'        # Light green (GNN)
C_TABULAR = '#D6EAF8'    # Light blue (Tabular)
C_FUSION = '#F9E79F'     # Light yellow (Fusion)
C_EXPERT = '#D2B4DE'     # Light purple (Experts)
C_GATE = '#F5CBA7'       # Light orange (Gating)
C_OUTPUT = '#A9DFBF'     # Green (Output)
C_TITLE = '#2C3E50'      # Dark title

BORDER = '#555555'
FONT_TITLE = 13
FONT_BOX = 10
FONT_DETAIL = 8


def draw_box(x, y, w, h, label, detail, color, border=BORDER, fontsize=FONT_BOX):
    """Draw a rounded box with label and detail text."""
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                           facecolor=color, edgecolor=border, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.35, label, ha='center', va='top',
            fontsize=fontsize, fontweight='bold', color='#2C3E50')
    if detail:
        ax.text(x + w/2, y + 0.35, detail, ha='center', va='bottom',
                fontsize=FONT_DETAIL, color='#555555', linespacing=1.3)


def draw_arrow(x1, y1, x2, y2, color='#555555', style='->', lw=1.5):
    """Draw arrow without overlapping text."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle='arc3,rad=0'))


def draw_arrow_curved(x1, y1, x2, y2, color='#555555', rad=0.2, lw=1.5):
    """Draw curved arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle=f'arc3,rad={rad}'))


# ══════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════
title_rect = FancyBboxPatch((4.5, 12.0), 13, 0.85, boxstyle="round,pad=0.1",
                             facecolor='white', edgecolor=C_TITLE, linewidth=2.5)
ax.add_patch(title_rect)
ax.text(11, 12.42, 'Multimodal Property-Conditioned Mixture of Experts (MoE) for IL Property Prediction',
        ha='center', va='center', fontsize=13.5, fontweight='bold', color=C_TITLE)
# Modality labels above inputs
ax.text(1.9, 11.0, '3D Surface\nModality', ha='center', va='center',
        fontsize=9, fontweight='bold', fontstyle='italic', color='#C0392B',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FADBD8', edgecolor='#C0392B', alpha=0.8))
ax.text(1.9, 7.95, 'Molecular Graph\nModality', ha='center', va='center',
        fontsize=9, fontweight='bold', fontstyle='italic', color='#27AE60',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#D5F5E3', edgecolor='#27AE60', alpha=0.8))
ax.text(1.9, 4.85, 'Tabular\nModality', ha='center', va='center',
        fontsize=9, fontweight='bold', fontstyle='italic', color='#2980B9',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#D6EAF8', edgecolor='#2980B9', alpha=0.8))

# ══════════════════════════════════════════════════════════════════════
# INPUT LAYER (left side)
# ══════════════════════════════════════════════════════════════════════

# Input: COSMO Point Cloud
draw_box(0.3, 8.5, 3.2, 2.2,
         'COSMO Point Cloud',
         '(B, 1024, 7)\nx, y, z, nx, ny, nz, ESP\nFrom marching-cubes mesh',
         C_INPUT)

# Input: Molecular Graph
draw_box(0.3, 5.5, 3.2, 2.2,
         'Molecular Graph',
         'Atom features: 22-dim\nBond features: 7-dim\nFrom SMILES (RDKit)',
         C_INPUT)

# Input: Tabular Features
draw_box(0.3, 2.3, 3.2, 2.2,
         'Tabular Features',
         '281 features:\n5 thermo + 20 surface\n+ 256 Morgan FP',
         C_INPUT)

# ══════════════════════════════════════════════════════════════════════
# ENCODERS
# ══════════════════════════════════════════════════════════════════════

# PointNet Encoder
draw_box(4.5, 8.5, 3.2, 2.2,
         'PointNet Encoder',
         'Shared MLPs:\n7→64→128→256\nMax Pool → 256-dim\n(Pre-trained)',
         C_POINTNET)

# GNN Encoder (GAT)
draw_box(4.5, 5.5, 3.2, 2.2,
         'GNN Encoder (GAT)',
         '4-layer GAT, 4 heads\nHidden: 256-dim\nMean Pool → 256-dim\n(Pre-trained Phase 2)',
         C_GNN)

# Tabular Projection
draw_box(4.5, 2.3, 3.2, 2.2,
         'Feature Projection',
         'Linear → ReLU → Dropout\n281 → 256-dim',
         C_TABULAR)

# Arrows: Input → Encoders
draw_arrow(3.5, 9.6, 4.5, 9.6)
draw_arrow(3.5, 6.6, 4.5, 6.6)
draw_arrow(3.5, 3.4, 4.5, 3.4)

# ══════════════════════════════════════════════════════════════════════
# CROSS-ATTENTION FUSION
# ══════════════════════════════════════════════════════════════════════

# Fusion box (large)
fusion_rect = FancyBboxPatch((8.5, 4.0, ), 4.0, 6.5, boxstyle="round,pad=0.2",
                              facecolor=C_FUSION, edgecolor='#B7950B', linewidth=2)
ax.add_patch(fusion_rect)
ax.text(10.5, 10.1, 'Cross-Attention Fusion', ha='center', va='center',
        fontsize=FONT_TITLE, fontweight='bold', color='#7D6608')

# Sub-boxes inside fusion
sub_w, sub_h = 3.4, 1.2

# Cross-attention PC↔Graph
ca_rect1 = FancyBboxPatch((8.8, 8.0), sub_w, sub_h, boxstyle="round,pad=0.1",
                           facecolor='#FEF9E7', edgecolor='#B7950B', linewidth=1)
ax.add_patch(ca_rect1)
ax.text(10.5, 8.8, 'Surface ↔ Graph', ha='center', va='top',
        fontsize=FONT_BOX, fontweight='bold', color='#7D6608')
ax.text(10.5, 8.2, 'Multi-head Cross-Attention\n(8 heads, bidirectional)', ha='center', va='bottom',
        fontsize=FONT_DETAIL, color='#555555')

# Concatenate + MLP
ca_rect2 = FancyBboxPatch((8.8, 6.2), sub_w, sub_h, boxstyle="round,pad=0.1",
                           facecolor='#FEF9E7', edgecolor='#B7950B', linewidth=1)
ax.add_patch(ca_rect2)
ax.text(10.5, 7.0, 'Concat + Fusion MLP', ha='center', va='top',
        fontsize=FONT_BOX, fontweight='bold', color='#7D6608')
ax.text(10.5, 6.4, '[PC; Graph; Tabular]\n768 → 256-dim + LayerNorm + GELU', ha='center', va='bottom',
        fontsize=FONT_DETAIL, color='#555555')

# Output annotation
ax.text(10.5, 4.7, 'Fused Representation\n(B, 256)', ha='center', va='center',
        fontsize=FONT_BOX, fontweight='bold', fontstyle='italic', color='#7D6608')

# Arrows: Encoders → Fusion
draw_arrow(7.7, 9.6, 8.5, 9.0)
draw_arrow(7.7, 6.6, 8.5, 7.0)
draw_arrow(7.7, 3.4, 8.8, 4.5)

# Internal fusion arrows
draw_arrow(10.5, 8.0, 10.5, 7.4)
draw_arrow(10.5, 6.2, 10.5, 5.2)

# ══════════════════════════════════════════════════════════════════════
# MIXTURE OF EXPERTS (right side)
# ══════════════════════════════════════════════════════════════════════

# Expert box container
expert_rect = FancyBboxPatch((13.3, 5.5), 4.2, 5.5, boxstyle="round,pad=0.2",
                              facecolor='#F4ECF7', edgecolor='#7D3C98', linewidth=2)
ax.add_patch(expert_rect)
ax.text(15.4, 10.65, '4 Expert Heads', ha='center', va='center',
        fontsize=FONT_TITLE, fontweight='bold', color='#6C3483')

# Individual experts
expert_names = [
    ('Expert 1', 'Surface\nSpecialist', '#E8DAEF'),
    ('Expert 2', 'Mixing\nSpecialist', '#D7BDE2'),
    ('Expert 3', 'Thermo\nSpecialist', '#C39BD3'),
    ('Expert 4', 'Generalist', '#BB8FCE'),
]
ey = 9.3
for i, (name, spec, color) in enumerate(expert_names):
    rect = FancyBboxPatch((13.6, ey), 3.6, 0.95, boxstyle="round,pad=0.08",
                           facecolor=color, edgecolor='#7D3C98', linewidth=1)
    ax.add_patch(rect)
    ax.text(14.5, ey + 0.48, name, ha='center', va='center',
            fontsize=9, fontweight='bold', color='#4A235A')
    ax.text(16.3, ey + 0.48, spec, ha='center', va='center',
            fontsize=8, color='#555555')
    ey -= 1.05

# "Each: 256→128→64→7" annotation
ax.text(15.4, 5.75, 'Each: Linear(256→128→64→7)', ha='center', va='center',
        fontsize=FONT_DETAIL, fontstyle='italic', color='#555555')

# Arrow: Fusion → Experts
draw_arrow(12.5, 7.5, 13.3, 7.5)

# ══════════════════════════════════════════════════════════════════════
# GATING NETWORK
# ══════════════════════════════════════════════════════════════════════

draw_box(13.3, 2.0, 4.2, 2.7,
         'Property-Conditioned\nGating Network',
         '7 learned property embeddings\n+ fused repr → MLP → softmax\nOutputs: (B, 7, 4) weights\n+ Load balancing loss',
         C_GATE, border='#CA6F1E')

# Arrow: Fusion → Gating
draw_arrow(12.5, 5.0, 13.3, 3.8)

# Arrow label
ax.text(12.7, 4.5, '256-dim', ha='center', va='center',
        fontsize=8, fontstyle='italic', color='#777777', rotation=50)

# ══════════════════════════════════════════════════════════════════════
# WEIGHTED COMBINATION
# ══════════════════════════════════════════════════════════════════════

# Multiplication symbol
mult_rect = FancyBboxPatch((18.2, 5.8), 1.5, 3.8, boxstyle="round,pad=0.15",
                            facecolor='#FDEBD0', edgecolor='#E67E22', linewidth=2)
ax.add_patch(mult_rect)
ax.text(18.95, 8.7, 'Weighted', ha='center', va='center',
        fontsize=FONT_BOX, fontweight='bold', color='#A04000')
ax.text(18.95, 7.7, 'Σ', ha='center', va='center',
        fontsize=24, fontweight='bold', color='#E67E22')
ax.text(18.95, 6.7, 'experts ×\ngating', ha='center', va='center',
        fontsize=FONT_DETAIL, color='#555555')
ax.text(18.95, 6.1, '(B, 7, 4)×(B, 7, 4)\n→ (B, 7)', ha='center', va='center',
        fontsize=7, color='#777777')

# Arrows: Experts → Weighted Sum
draw_arrow(17.5, 7.8, 18.2, 7.8)

# Arrows: Gating → Weighted Sum
draw_arrow(17.5, 3.8, 18.95, 5.8)

# ══════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════

draw_box(20.2, 5.8, 1.5, 3.8,
         'Output',
         '7 Properties:\nγ₁, γ₂\nGᴱ, Hᴱ, Gₘᵢₓ\nHᵥₐₚ, P\n\n(B, 7)',
         C_OUTPUT, border='#1E8449')

# Arrow: Weighted Sum → Output
draw_arrow(19.7, 7.7, 20.2, 7.7)

# ══════════════════════════════════════════════════════════════════════
# SNAPSHOT ENSEMBLE annotation (bottom)
# ══════════════════════════════════════════════════════════════════════

snap_rect = FancyBboxPatch((0.3, 0.3), 21.4, 1.3, boxstyle="round,pad=0.15",
                            facecolor='#EBF5FB', edgecolor='#2980B9', linewidth=1.5,
                            linestyle='--')
ax.add_patch(snap_rect)
ax.text(11, 1.25, 'Snapshot Ensemble at Inference', ha='center', va='top',
        fontsize=11, fontweight='bold', color='#2471A3')
ax.text(11, 0.6, 'Cosine Annealing with Restarts (T₀=60 epochs) → Save model at each cycle boundary → '
        'Average predictions from best model + N snapshots → Variance reduction without separate training',
        ha='center', va='center', fontsize=8.5, color='#555555')

# ══════════════════════════════════════════════════════════════════════

plt.savefig('paper/figures/moe_architecture.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: paper/figures/moe_architecture.png")
