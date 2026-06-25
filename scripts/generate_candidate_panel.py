"""Generate publication-quality panel figure of top 3 IL candidates.

Creates:
1. 3-panel COSMO ESP surface comparison (top 3 candidates)
2. Training set reference comparison
3. Sigma-profile comparison
4. Property prediction bar chart for candidates
"""

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    ("mmim_oac", "[MMIM][OAc]\nRank #1", "C[n+]1ccn(C)c1.CC(=O)[O-]"),
    ("tmg_oac", "[TMG][OAc]\nRank #3", "CN(C)C(=[NH2+])N(C)C.CC(=O)[O-]"),
    ("mmim_lev", "[MMIM][Lev]\nRank #5 (fully novel)", "C[n+]1ccn(C)c1.CC(=O)CCC(=O)[O-]"),
]

# Predicted properties from STILT
PREDICTIONS = {
    "mmim_oac": {"gamma1": 0.326, "gamma2": 0.027, "G_mix": -2.513, "H_vap": 25.63, "P": 0.054},
    "tmg_oac":  {"gamma1": 0.172, "gamma2": -0.134, "G_mix": -2.819, "H_vap": 24.94, "P": 0.083},
    "mmim_lev": {"gamma1": 0.420, "gamma2": 0.173, "G_mix": -2.150, "H_vap": 24.14, "P": 0.180},
}


def load_point_cloud(cid):
    """Load point cloud from novel or training set."""
    novel_path = Path(f"data/pipeline/point_clouds_novel/{cid}.npz")
    if novel_path.exists():
        return np.load(novel_path)["points"]
    # Try training set
    import pandas as pd
    idx = pd.read_csv("data/pipeline/point_clouds/index.csv")
    return None


def render_esp_surface(ax, points, title, elev=20, azim=45):
    """Render ESP surface on a 3D axis."""
    xyz = points[:, :3]
    esp = points[:, 6]

    # Center
    xyz = xyz - xyz.mean(axis=0)

    # Color by ESP
    vmax = max(abs(esp.min()), abs(esp.max()), 0.01)
    colors = plt.cm.RdBu_r((esp / vmax + 1) / 2)

    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=4, alpha=0.85)

    r = np.abs(xyz).max() * 1.1
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    ax.set_xlabel('X', fontsize=7, labelpad=1)
    ax.set_ylabel('Y', fontsize=7, labelpad=1)
    ax.set_zlabel('Z', fontsize=7, labelpad=1)
    ax.tick_params(labelsize=5)
    ax.set_title(title, fontsize=9, fontweight='bold', pad=2)
    ax.view_init(elev=elev, azim=azim)

    return vmax


def fig1_candidate_panel():
    """3-panel COSMO surface comparison of top 3 candidates."""
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])

    vmaxes = []
    for i, (cid, label, smi) in enumerate(CANDIDATES):
        ax = fig.add_subplot(gs[0, i], projection='3d')
        pc = load_point_cloud(cid)
        if pc is not None:
            vmax = render_esp_surface(ax, pc, label, elev=15, azim=45 + i*30)
            vmaxes.append(vmax)
        else:
            ax.set_title(f"{label}\n(no point cloud)")

    # Shared colorbar
    if vmaxes:
        vmax = max(vmaxes)
        cax = fig.add_subplot(gs[0, 3])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(-vmax, vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label('ESP (a.u.)', fontsize=9)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle('COSMO-style Electrostatic Potential Surfaces of Top IL Candidates',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "candidate_esp_panel.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved candidate_esp_panel.png")


def fig2_sigma_profiles():
    """Sigma-profile comparison of candidates."""
    fig, ax = plt.subplots(figsize=(8, 5))

    sigma_bins = np.linspace(-0.3, 0.3, 51)
    sigma_centers = 0.5 * (sigma_bins[:-1] + sigma_bins[1:])
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    for i, (cid, label, _) in enumerate(CANDIDATES):
        pc = load_point_cloud(cid)
        if pc is not None:
            esp = pc[:, 6]
            hist, _ = np.histogram(esp, bins=sigma_bins, density=True)
            ax.plot(sigma_centers, hist, '-', linewidth=2, color=colors[i],
                    label=label.split('\n')[0])
            ax.fill_between(sigma_centers, hist, alpha=0.15, color=colors[i])

    ax.set_xlabel(r'Surface Charge Density $\sigma$ (a.u.)', fontsize=11)
    ax.set_ylabel('Probability Density p(σ)', fontsize=11)
    ax.set_title('Sigma-Profiles of Top IL Candidates', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.annotate('← negative\n(H-bond acceptor)', xy=(-0.15, 0), fontsize=7,
                color='blue', ha='center', va='bottom')
    ax.annotate('positive →\n(H-bond donor)', xy=(0.10, 0), fontsize=7,
                color='red', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(OUT / "candidate_sigma_profiles.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved candidate_sigma_profiles.png")


def fig3_property_comparison():
    """Bar chart of predicted properties for top 3 candidates."""
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))

    props = ['gamma1', 'gamma2', 'G_mix', 'H_vap', 'P']
    prop_labels = [r'$\gamma_1$', r'$\gamma_2$', r'$G_{mix}$ (kcal/mol)',
                   r'$H_{vap}$ (kcal/mol)', r'$P$ (bar)']
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    names = [label.split('\n')[0] for _, label, _ in CANDIDATES]

    for j, (prop, plabel) in enumerate(zip(props, prop_labels)):
        ax = axes[j]
        vals = [PREDICTIONS[cid][prop] for cid, _, _ in CANDIDATES]
        bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.85,
                      edgecolor='white', linewidth=1.5)
        for k, v in enumerate(vals):
            ax.text(k, v + (0.02 if v >= 0 else -0.04), f'{v:.3f}',
                    ha='center', fontsize=7, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7, rotation=15)
        ax.set_title(plabel, fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        if prop in ['gamma1', 'G_mix', 'P']:
            ax.axhline(y=0, color='black', linewidth=0.5)

    fig.suptitle('Predicted Thermodynamic Properties of Top IL Candidates (T=348K, x₁=0.5)',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "candidate_properties.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved candidate_properties.png")


def fig4_discovery_pipeline():
    """Visual summary of the IL discovery pipeline."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Pipeline stages
    stages = [
        (1, 2, 2.2, 2.0, 'SMILES\nGeneration', '#E3F2FD', '#1976D2',
         '11 cations × 10 anions\n= 110 candidates'),
        (3.5, 2, 2.2, 2.0, 'STILT\nPrediction', '#E8F5E9', '#388E3C',
         '7 properties\n× 4 temperatures'),
        (6, 2, 2.2, 2.0, 'Multi-Objective\nRanking', '#FFF3E0', '#F57C00',
         'Eq. (1): miscibility\n+ mixing + stability'),
        (8.5, 2, 2.2, 2.0, 'COSMO Surface\nGeneration', '#F3E5F5', '#7B1FA2',
         'DFT geometry →\nESP point cloud'),
        (11, 2, 2.2, 2.0, 'Experimental\nValidation', '#FFEBEE', '#D32F2F',
         'Top 3 → synthesis\n→ characterization'),
    ]

    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    for x, y, w, h, title, fc, ec, subtitle in stages:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.65, title, ha='center', va='center',
                fontsize=8, fontweight='bold', color=ec)
        ax.text(x + w/2, y + h*0.25, subtitle, ha='center', va='center',
                fontsize=6, color='#555555')

    # Arrows
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + stages[i][2]
        x2 = stages[i+1][0]
        y = stages[i][1] + stages[i][3] / 2
        arrow = FancyArrowPatch((x1, y), (x2, y), arrowstyle='->', color='#666666',
                                 lw=2, mutation_scale=15)
        ax.add_patch(arrow)

    ax.set_title('AI-Guided Ionic Liquid Discovery Pipeline', fontsize=12,
                 fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(OUT / "discovery_pipeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved discovery_pipeline.png")


def main():
    print("=== Generating Publication Figures ===\n")
    fig1_candidate_panel()
    fig2_sigma_profiles()
    fig3_property_comparison()
    fig4_discovery_pipeline()
    print("\nAll figures generated!")


if __name__ == "__main__":
    main()
