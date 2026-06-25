"""Generate MoE architecture diagram with example modality visuals."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from rdkit import Chem
from rdkit.Chem import Draw


# ── Create example modality images first ──────────────────────────────────

def create_pointcloud_image(output_path):
    """Create a 3D point cloud scatter plot from real data."""
    pc_dir = Path("data/pipeline/point_clouds")
    idx_df = pd.read_csv(pc_dir / "index.csv")
    pts = np.load(pc_dir / idx_df.iloc[0]["filename"])["points"]

    fig = plt.figure(figsize=(3, 3), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pts[::2, 0], pts[::2, 1], pts[::2, 2],
                    c=pts[::2, 6], cmap='RdBu_r', s=4, alpha=0.7)
    ax.set_xlabel('x', fontsize=7, labelpad=-3)
    ax.set_ylabel('y', fontsize=7, labelpad=-3)
    ax.set_zlabel('z', fontsize=7, labelpad=-3)
    ax.tick_params(labelsize=5, pad=-2)
    ax.view_init(elev=25, azim=45)
    ax.set_title('COSMO Surface\nPoint Cloud', fontsize=8, fontweight='bold', pad=1)
    plt.colorbar(sc, ax=ax, shrink=0.5, label='ESP', pad=0.01)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white',
                transparent=False, pad_inches=0.05)
    plt.close()


def create_graph_image(output_path):
    """Create a molecular graph visualization from RDKit."""
    smi = "CCCCn1cc[n+](C)c1.[Cl-]"
    mol = Chem.MolFromSmiles(smi)
    img = Draw.MolToImage(mol, size=(350, 280),
                          kekulize=True, wedgeBonds=True)
    # Add title
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((60, 5), "Molecular Graph", fill="black", font=font)
    draw.text((40, 250), "BMIMCl: 22-dim atom features", fill="gray",
              font=ImageFont.load_default())
    img.save(output_path)


def create_tabular_image(output_path):
    """Create a visual representation of tabular features."""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=150)
    ax.axis('off')

    table_data = [
        ['Feature', 'Value'],
        ['T (K)', '298.15'],
        ['x₁', '0.50'],
        ['1/T', '0.00335'],
        ['Morgan FP', '[0,1,0,1...]'],
        ['Surf. Area', '225.0'],
        ['ESP mean', '-0.12'],
        ['Sphericity', '0.87'],
        ['...', '(281 total)'],
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.45, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Style header
    for j in range(2):
        table[0, j].set_facecolor('#D6EAF8')
        table[0, j].set_text_props(fontweight='bold')

    # Style rows
    for i in range(1, len(table_data)):
        for j in range(2):
            table[i, j].set_facecolor('#EBF5FB' if i % 2 == 0 else 'white')

    table.scale(1.0, 1.6)
    ax.set_title('Tabular Features\n(Thermo + Surface + Morgan FP)',
                 fontsize=9, fontweight='bold', pad=8)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white',
                pad_inches=0.05)
    plt.close()


# ── Main architecture diagram ─────────────────────────────────────────────

def main():
    # Generate example images
    example_dir = Path("paper/figures/examples")
    example_dir.mkdir(parents=True, exist_ok=True)

    print("Creating example modality images...")
    create_pointcloud_image(example_dir / "pc_example.png")
    create_graph_image(example_dir / "graph_example.png")
    create_tabular_image(example_dir / "tabular_example.png")
    print("  Done")

    # ── Build main figure with inset images ──
    fig = plt.figure(figsize=(26, 13))

    # Main architecture axes (right portion)
    ax = fig.add_axes([0.17, 0.0, 0.83, 1.0])
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 13)
    ax.axis('off')

    # ── Color scheme ──
    C_INPUT = '#E8D5B7'
    C_POINTNET = '#FADBD8'
    C_GNN = '#D5F5E3'
    C_TABULAR = '#D6EAF8'
    C_FUSION = '#F9E79F'
    C_EXPERT = '#D2B4DE'
    C_GATE = '#F5CBA7'
    C_OUTPUT = '#A9DFBF'
    C_TITLE = '#2C3E50'
    BORDER = '#555555'
    FONT_BOX = 10
    FONT_DETAIL = 8

    def draw_box(x, y, w, h, label, detail, color, border=BORDER):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                               facecolor=color, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.35, label, ha='center', va='top',
                fontsize=FONT_BOX, fontweight='bold', color='#2C3E50')
        if detail:
            ax.text(x + w/2, y + 0.35, detail, ha='center', va='bottom',
                    fontsize=FONT_DETAIL, color='#555555', linespacing=1.3)

    def draw_arrow(x1, y1, x2, y2, color='#555555', lw=1.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw))

    # ══════════════════════════════════════════════════════════════
    # TITLE
    # ══════════════════════════════════════════════════════════════
    title_rect = FancyBboxPatch((3.5, 12.0), 15, 0.85, boxstyle="round,pad=0.1",
                                 facecolor='white', edgecolor=C_TITLE, linewidth=2.5)
    ax.add_patch(title_rect)
    ax.text(11, 12.42, 'Multimodal Property-Conditioned Mixture of Experts (MoE)',
            ha='center', va='center', fontsize=14, fontweight='bold', color=C_TITLE)

    # ══════════════════════════════════════════════════════════════
    # MODALITY LABELS (above inputs)
    # ══════════════════════════════════════════════════════════════
    ax.text(1.9, 11.0, '3D Surface\nModality', ha='center', va='center',
            fontsize=9, fontweight='bold', fontstyle='italic', color='#C0392B',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FADBD8',
                      edgecolor='#C0392B', alpha=0.8))
    ax.text(1.9, 7.95, 'Molecular Graph\nModality', ha='center', va='center',
            fontsize=9, fontweight='bold', fontstyle='italic', color='#27AE60',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#D5F5E3',
                      edgecolor='#27AE60', alpha=0.8))
    ax.text(1.9, 4.85, 'Tabular\nModality', ha='center', va='center',
            fontsize=9, fontweight='bold', fontstyle='italic', color='#2980B9',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#D6EAF8',
                      edgecolor='#2980B9', alpha=0.8))

    # ══════════════════════════════════════════════════════════════
    # INPUT BOXES
    # ══════════════════════════════════════════════════════════════
    draw_box(0.3, 8.5, 3.2, 2.2,
             'COSMO Point Cloud',
             '(B, 1024, 7)\nx, y, z, nx, ny, nz, ESP\nFrom marching-cubes mesh',
             C_INPUT)

    draw_box(0.3, 5.5, 3.2, 2.2,
             'Molecular Graph',
             'Atom features: 22-dim\nBond features: 7-dim\nFrom SMILES (RDKit)',
             C_INPUT)

    draw_box(0.3, 2.3, 3.2, 2.2,
             'Tabular Features',
             '281 features:\n5 thermo + 20 surface\n+ 256 Morgan FP',
             C_INPUT)

    # ══════════════════════════════════════════════════════════════
    # ENCODERS
    # ══════════════════════════════════════════════════════════════
    draw_box(4.5, 8.5, 3.2, 2.2,
             'PointNet Encoder',
             'Shared MLPs:\n7→64→128→256\nMax Pool → 256-dim\n(Pre-trained)',
             C_POINTNET)

    draw_box(4.5, 5.5, 3.2, 2.2,
             'GNN Encoder (GAT)',
             '4-layer GAT, 4 heads\nHidden: 256-dim\nMean Pool → 256-dim\n(Pre-trained Phase 2)',
             C_GNN)

    draw_box(4.5, 2.3, 3.2, 2.2,
             'Feature Projection',
             'Linear → ReLU → Dropout\n281 → 256-dim',
             C_TABULAR)

    # Arrows: Input → Encoders
    draw_arrow(3.5, 9.6, 4.5, 9.6)
    draw_arrow(3.5, 6.6, 4.5, 6.6)
    draw_arrow(3.5, 3.4, 4.5, 3.4)

    # ══════════════════════════════════════════════════════════════
    # CROSS-ATTENTION FUSION
    # ══════════════════════════════════════════════════════════════
    fusion_rect = FancyBboxPatch((8.5, 4.0), 4.0, 6.5, boxstyle="round,pad=0.2",
                                  facecolor=C_FUSION, edgecolor='#B7950B', linewidth=2)
    ax.add_patch(fusion_rect)
    ax.text(10.5, 10.1, 'Cross-Attention Fusion', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#7D6608')

    sub_w, sub_h = 3.4, 1.2
    ca_rect1 = FancyBboxPatch((8.8, 8.0), sub_w, sub_h, boxstyle="round,pad=0.1",
                               facecolor='#FEF9E7', edgecolor='#B7950B', linewidth=1)
    ax.add_patch(ca_rect1)
    ax.text(10.5, 8.8, 'Surface ↔ Graph', ha='center', va='top',
            fontsize=FONT_BOX, fontweight='bold', color='#7D6608')
    ax.text(10.5, 8.2, 'Multi-head Cross-Attention\n(8 heads, bidirectional)', ha='center', va='bottom',
            fontsize=FONT_DETAIL, color='#555555')

    ca_rect2 = FancyBboxPatch((8.8, 6.2), sub_w, sub_h, boxstyle="round,pad=0.1",
                               facecolor='#FEF9E7', edgecolor='#B7950B', linewidth=1)
    ax.add_patch(ca_rect2)
    ax.text(10.5, 7.0, 'Concat + Fusion MLP', ha='center', va='top',
            fontsize=FONT_BOX, fontweight='bold', color='#7D6608')
    ax.text(10.5, 6.4, '[PC; Graph; Tabular]\n768 → 256-dim + LayerNorm + GELU', ha='center', va='bottom',
            fontsize=FONT_DETAIL, color='#555555')

    ax.text(10.5, 4.7, 'Fused Representation\n(B, 256)', ha='center', va='center',
            fontsize=FONT_BOX, fontweight='bold', fontstyle='italic', color='#7D6608')

    # Arrows: Encoders → Fusion
    draw_arrow(7.7, 9.6, 8.5, 9.0)
    draw_arrow(7.7, 6.6, 8.5, 7.0)
    draw_arrow(7.7, 3.4, 8.8, 4.5)

    # Internal fusion arrows
    draw_arrow(10.5, 8.0, 10.5, 7.4)
    draw_arrow(10.5, 6.2, 10.5, 5.2)

    # ══════════════════════════════════════════════════════════════
    # EXPERT HEADS
    # ══════════════════════════════════════════════════════════════
    expert_rect = FancyBboxPatch((13.3, 5.5), 4.2, 5.5, boxstyle="round,pad=0.2",
                                  facecolor='#F4ECF7', edgecolor='#7D3C98', linewidth=2)
    ax.add_patch(expert_rect)
    ax.text(15.4, 10.65, '4 Expert Heads', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#6C3483')

    expert_names = [
        ('Expert 1', 'Surface\nSpecialist', '#E8DAEF'),
        ('Expert 2', 'Mixing\nSpecialist', '#D7BDE2'),
        ('Expert 3', 'Thermo\nSpecialist', '#C39BD3'),
        ('Expert 4', 'Generalist', '#BB8FCE'),
    ]
    ey = 9.3
    for name, spec, color in expert_names:
        rect = FancyBboxPatch((13.6, ey), 3.6, 0.95, boxstyle="round,pad=0.08",
                               facecolor=color, edgecolor='#7D3C98', linewidth=1)
        ax.add_patch(rect)
        ax.text(14.5, ey + 0.48, name, ha='center', va='center',
                fontsize=9, fontweight='bold', color='#4A235A')
        ax.text(16.3, ey + 0.48, spec, ha='center', va='center',
                fontsize=8, color='#555555')
        ey -= 1.05

    ax.text(15.4, 5.75, 'Each: Linear(256→128→64→7)', ha='center', va='center',
            fontsize=FONT_DETAIL, fontstyle='italic', color='#555555')

    draw_arrow(12.5, 7.5, 13.3, 7.5)

    # ══════════════════════════════════════════════════════════════
    # GATING NETWORK
    # ══════════════════════════════════════════════════════════════
    draw_box(13.3, 2.0, 4.2, 2.7,
             'Property-Conditioned\nGating Network',
             '7 learned property embeddings\n+ fused repr → MLP → softmax\nOutputs: (B, 7, 4) weights\n+ Load balancing loss',
             C_GATE, border='#CA6F1E')

    draw_arrow(12.5, 5.0, 13.3, 3.8)
    ax.text(12.7, 4.5, '256-dim', ha='center', va='center',
            fontsize=8, fontstyle='italic', color='#777777', rotation=50)

    # ══════════════════════════════════════════════════════════════
    # WEIGHTED COMBINATION
    # ══════════════════════════════════════════════════════════════
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

    draw_arrow(17.5, 7.8, 18.2, 7.8)
    draw_arrow(17.5, 3.8, 18.95, 5.8)

    # ══════════════════════════════════════════════════════════════
    # OUTPUT
    # ══════════════════════════════════════════════════════════════
    draw_box(20.2, 5.8, 1.5, 3.8,
             'Output',
             '7 Properties:\nγ₁, γ₂\nGᴱ, Hᴱ, Gₘᵢₓ\nHᵥₐₚ, P\n\n(B, 7)',
             C_OUTPUT, border='#1E8449')

    draw_arrow(19.7, 7.7, 20.2, 7.7)

    # ══════════════════════════════════════════════════════════════
    # SNAPSHOT ENSEMBLE (bottom)
    # ══════════════════════════════════════════════════════════════
    snap_rect = FancyBboxPatch((0.3, 0.3), 21.4, 1.3, boxstyle="round,pad=0.15",
                                facecolor='#EBF5FB', edgecolor='#2980B9',
                                linewidth=1.5, linestyle='--')
    ax.add_patch(snap_rect)
    ax.text(11, 1.25, 'Snapshot Ensemble at Inference', ha='center', va='top',
            fontsize=11, fontweight='bold', color='#2471A3')
    ax.text(11, 0.6, 'Cosine Annealing with Restarts (T₀=60) → Save model at each cycle '
            '→ Average predictions from best + N snapshots → Variance reduction',
            ha='center', va='center', fontsize=8.5, color='#555555')

    # ══════════════════════════════════════════════════════════════
    # EXAMPLE MODALITY IMAGES (left panel, using separate axes)
    # ══════════════════════════════════════════════════════════════

    # Point cloud example (top-left)
    ax_pc = fig.add_axes([0.0, 0.62, 0.17, 0.33])
    pc_img = Image.open(example_dir / "pc_example.png")
    ax_pc.imshow(pc_img)
    ax_pc.axis('off')
    # Border
    for spine in ax_pc.spines.values():
        spine.set_visible(True)
        spine.set_color('#C0392B')
        spine.set_linewidth(2)

    # Molecular graph example (middle-left)
    ax_graph = fig.add_axes([0.0, 0.32, 0.17, 0.28])
    graph_img = Image.open(example_dir / "graph_example.png")
    ax_graph.imshow(graph_img)
    ax_graph.axis('off')
    for spine in ax_graph.spines.values():
        spine.set_visible(True)
        spine.set_color('#27AE60')
        spine.set_linewidth(2)

    # Tabular example (bottom-left)
    ax_tab = fig.add_axes([0.0, 0.05, 0.17, 0.25])
    tab_img = Image.open(example_dir / "tabular_example.png")
    ax_tab.imshow(tab_img)
    ax_tab.axis('off')
    for spine in ax_tab.spines.values():
        spine.set_visible(True)
        spine.set_color('#2980B9')
        spine.set_linewidth(2)

    # ══════════════════════════════════════════════════════════════

    plt.savefig('paper/figures/moe_architecture.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: paper/figures/moe_architecture.png")


if __name__ == "__main__":
    main()
