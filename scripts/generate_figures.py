"""Generate publication-quality figures for the dataset report.

Figures:
  1. Model progression bar chart (R² improvement across phases)
  2. Per-property heatmap (all models × 7 properties)
  3. Structure vs temperature property analysis
  4. Architecture search comparison
  5. Point cloud visualization samples
  6. Pipeline diagram data flow
  7. Radar chart of best model per-property performance
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = Path("paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]
TARGET_LABELS = [r"$\gamma_1$", r"$\gamma_2$", r"$G^E$", r"$H^E$",
                 r"$G_{mix}$", r"$H_{vap}$", r"$P$"]


def load_results():
    """Load all result JSON files."""
    results = {}
    files = {
        "Baseline GNN": "results/gnn_results.json",
        "Phase 1\n(+Surface Desc)": "results/gnn_surface_results.json",
        "Phase 2\n(Transfer)": "results/transfer_results.json",
        "Phase 3\n(PointCloud)": "results/pointcloud_results.json",
        "Hard Ensemble": "results/ensemble_phase23_results.json",
        "DGCNN": "results/dgcnn_results.json",
        "Contrastive": "results/contrastive_results.json",
        "Strategies A-E": "results/strategies_abcde_results.json",
    }
    for name, path in files.items():
        try:
            with open(path) as f:
                data = json.load(f)
            # Handle ensemble format
            if "hard_ensemble" in data:
                results[name] = data["hard_ensemble"]
            elif "test_metrics" in data:
                results[name] = data["test_metrics"]
            else:
                results[name] = data
        except Exception:
            pass

    # Try to load hybrid results
    try:
        with open("results/hybrid_final_results.json") as f:
            hybrid = json.load(f)
        if "final_hybrid" in hybrid:
            fh = hybrid["final_hybrid"]
            # Convert to standard format
            metrics = {"avg_r2": fh["avg_r2"]}
            for prop, r2 in fh["per_property_r2"].items():
                metrics[f"{prop}_r2"] = r2
            results["Final Hybrid"] = metrics
        if "cv_ensemble_v2" in hybrid:
            results["CV Ensemble v2"] = hybrid["cv_ensemble_v2"].get("test_metrics", {})
    except Exception:
        pass

    return results


# ══════════════════════════════════════════════════════════════════════
# Figure 1: Model Progression (the main story)
# ══════════════════════════════════════════════════════════════════════

def fig1_model_progression(results):
    """Bar chart showing R² improvement across the 3-phase journey."""
    phases = [
        ("Baseline\nGNN", "Baseline GNN", "#95a5a6"),
        ("Phase 1\n(+Surface)", "Phase 1\n(+Surface Desc)", "#e74c3c"),
        ("Phase 2\n(Transfer)", "Phase 2\n(Transfer)", "#f39c12"),
        ("Phase 3\n(PointCloud)", "Phase 3\n(PointCloud)", "#27ae60"),
        ("Hard\nEnsemble", "Hard Ensemble", "#2980b9"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(phases))
    r2_vals = []
    colors = []
    for label, key, color in phases:
        r2 = results.get(key, {}).get("avg_r2", 0)
        r2_vals.append(r2)
        colors.append(color)

    bars = ax.bar(x, r2_vals, color=colors, width=0.6, edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add improvement arrows
    for i in range(1, len(r2_vals)):
        if r2_vals[i] > r2_vals[i-1]:
            delta = r2_vals[i] - r2_vals[i-1]
            ax.annotate(f'+{delta:.3f}',
                       xy=(i, r2_vals[i] + 0.04),
                       fontsize=9, ha='center', color='#27ae60', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in phases])
    ax.set_ylabel('Average $R^2$ (7 properties)')
    ax.set_title('Model Performance Progression Across 3 Phases')
    ax.set_ylim(0, max(r2_vals) + 0.12)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(OUTPUT_DIR / "model_progression_phases.png")
    plt.close()
    print("  Saved: model_progression_phases.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 2: Per-Property Heatmap
# ══════════════════════════════════════════════════════════════════════

def fig2_property_heatmap(results):
    """Heatmap of R² values: models × properties."""
    models = [
        "Baseline GNN", "Phase 1\n(+Surface Desc)", "Phase 2\n(Transfer)",
        "Phase 3\n(PointCloud)", "DGCNN", "Contrastive", "Hard Ensemble",
    ]

    model_labels = [
        "Baseline GNN", "Phase 1 (+Surface)", "Phase 2 (Transfer)",
        "Phase 3 (PointCloud)", "DGCNN", "Contrastive GNN", "Hard Ensemble",
    ]

    data = np.zeros((len(models), len(TARGET_COLUMNS)))
    for i, model in enumerate(models):
        for j, prop in enumerate(TARGET_COLUMNS):
            data[i, j] = results.get(model, {}).get(f"{prop}_r2", np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=1.0)

    ax.set_xticks(range(len(TARGET_COLUMNS)))
    ax.set_xticklabels(TARGET_LABELS, fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(model_labels, fontsize=10)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(TARGET_COLUMNS)):
            val = data[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.3 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=9, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('$R^2$')
    ax.set_title('Per-Property $R^2$ Across Model Variants')

    plt.savefig(OUTPUT_DIR / "property_heatmap.png")
    plt.close()
    print("  Saved: property_heatmap.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 3: Structure vs Temperature Property Split
# ══════════════════════════════════════════════════════════════════════

def fig3_struct_vs_temp(results):
    """Grouped bar chart showing Phase 2 vs Phase 3 strength split."""
    props = TARGET_COLUMNS
    labels = TARGET_LABELS

    p2 = [results.get("Phase 2\n(Transfer)", {}).get(f"{p}_r2", 0) for p in props]
    p3 = [results.get("Phase 3\n(PointCloud)", {}).get(f"{p}_r2", 0) for p in props]
    ens = [results.get("Hard Ensemble", {}).get(f"{p}_r2", 0) for p in props]

    x = np.arange(len(props))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width, p2, width, label='Phase 2 (Transfer GNN)', color='#f39c12', alpha=0.85)
    bars2 = ax.bar(x, p3, width, label='Phase 3 (PointCloud+GNN)', color='#27ae60', alpha=0.85)
    bars3 = ax.bar(x + width, ens, width, label='Hard Ensemble (best of both)', color='#2980b9', alpha=0.85)

    # Annotate which model wins
    for i in range(len(props)):
        winner = "P2" if p2[i] > p3[i] else "P3"
        y = max(p2[i], p3[i], ens[i]) + 0.02
        if i >= 5:  # H_vap, P
            ax.annotate('Temp-driven', xy=(x[i], y), fontsize=8,
                       ha='center', color='#e74c3c', fontstyle='italic')
        else:
            ax.annotate('Structure-driven', xy=(x[i], y), fontsize=8,
                       ha='center', color='#2c3e50', fontstyle='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('$R^2$')
    ax.set_title('Structure-Dependent vs Temperature-Dependent Properties')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Add vertical separator
    ax.axvline(x=4.5, color='gray', linestyle='--', alpha=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(OUTPUT_DIR / "struct_vs_temp.png")
    plt.close()
    print("  Saved: struct_vs_temp.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 4: Architecture Search (all 13+ models)
# ══════════════════════════════════════════════════════════════════════

def fig4_architecture_search(results):
    """Horizontal bar chart of all model variants sorted by R²."""
    model_r2 = []
    for name, metrics in results.items():
        r2 = metrics.get("avg_r2", None)
        if r2 is not None and isinstance(r2, (int, float)):
            model_r2.append((name.replace('\n', ' '), r2))

    # Sort by R²
    model_r2.sort(key=lambda x: x[1])

    names = [m[0] for m in model_r2]
    values = [m[1] for m in model_r2]
    colors = ['#27ae60' if v > 0.5 else '#f39c12' if v > 0 else '#e74c3c' for v in values]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.45)))
    bars = ax.barh(range(len(names)), values, color=colors, height=0.7, edgecolor='white')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Average $R^2$')
    ax.set_title('Comprehensive Architecture Search: 13 Model Variants')
    ax.axvline(x=0, color='black', linewidth=0.8)

    # Add value labels
    for bar, val in zip(bars, values):
        x_pos = val + 0.01 if val >= 0 else val - 0.06
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(OUTPUT_DIR / "architecture_search.png")
    plt.close()
    print("  Saved: architecture_search.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 5: Radar Chart of Best Model
# ══════════════════════════════════════════════════════════════════════

def fig5_radar_chart(results):
    """Radar chart comparing top 3 models across all 7 properties."""
    models = [
        ("Baseline GNN", "Baseline GNN", '#95a5a6'),
        ("Phase 3\n(PointCloud)", "Phase 3 (PointCloud)", '#27ae60'),
        ("Hard Ensemble", "Hard Ensemble", '#2980b9'),
    ]

    angles = np.linspace(0, 2 * np.pi, len(TARGET_COLUMNS), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for key, label, color in models:
        values = [max(0, results.get(key, {}).get(f"{p}_r2", 0)) for p in TARGET_COLUMNS]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color, markersize=5)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(TARGET_LABELS, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('Per-Property $R^2$ Comparison', y=1.08, fontsize=13)

    plt.savefig(OUTPUT_DIR / "radar_comparison.png")
    plt.close()
    print("  Saved: radar_comparison.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 6: Point Cloud vs 2D Image comparison
# ══════════════════════════════════════════════════════════════════════

def fig6_representation_comparison():
    """Side-by-side: 2D COSMO image vs 3D point cloud for same molecule."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: 2D COSMO image (load existing)
    cosmo_img_path = Path("data/pipeline/cosmo_images/ABXNZH_cosmo.png")
    if cosmo_img_path.exists():
        img = plt.imread(str(cosmo_img_path))
        axes[0].imshow(img)
        axes[0].set_title('(a) 2D COSMO Rendering\n$R^2 = -0.07$ (172M params)', fontsize=11)
    else:
        axes[0].text(0.5, 0.5, '2D COSMO\nImage', transform=axes[0].transAxes,
                    ha='center', va='center', fontsize=14)
        axes[0].set_title('(a) 2D COSMO Rendering\n$R^2 = -0.07$', fontsize=11)
    axes[0].axis('off')

    # Panel B: 3D point cloud scatter
    pc_path = Path("data/pipeline/point_clouds")
    idx_path = pc_path / "index.csv"
    if idx_path.exists():
        import pandas as pd
        idx_df = pd.read_csv(idx_path)
        if len(idx_df) > 0:
            first_file = idx_df.iloc[0]["filename"]
            data = np.load(pc_path / first_file)
            pts = data["points"]  # (1024, 7)

            # 3D scatter colored by ESP
            ax3d = fig.add_subplot(132, projection='3d')
            sc = ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                            c=pts[:, 6], cmap='RdBu_r', s=3, alpha=0.8)
            ax3d.set_title('(b) 3D Point Cloud (1024 pts)\n$R^2 = 0.655$ (500K params)', fontsize=11)
            ax3d.set_xlabel('x')
            ax3d.set_ylabel('y')
            ax3d.set_zlabel('z')
            ax3d.view_init(elev=20, azim=45)
            plt.colorbar(sc, ax=ax3d, shrink=0.6, label='ESP')
            axes[1].axis('off')  # Hide the original 2D axis
    else:
        axes[1].text(0.5, 0.5, '3D Point\nCloud', transform=axes[1].transAxes,
                    ha='center', va='center', fontsize=14)
        axes[1].set_title('(b) 3D Point Cloud\n$R^2 = 0.655$', fontsize=11)
        axes[1].axis('off')

    # Panel C: Improvement summary
    improvements = {
        '$\\gamma_1$': (0.593, 0.887),
        '$\\gamma_2$': (0.672, 0.845),
        '$G^E$': (0.506, 0.696),
        '$H^E$': (0.520, 0.695),
        '$G_{mix}$': (0.506, 0.669),
    }
    props = list(improvements.keys())
    baseline = [improvements[p][0] for p in props]
    improved = [improvements[p][1] for p in props]

    y_pos = np.arange(len(props))
    axes[2].barh(y_pos - 0.15, baseline, 0.3, label='Baseline GNN', color='#95a5a6')
    axes[2].barh(y_pos + 0.15, improved, 0.3, label='PointCloud+GNN', color='#27ae60')
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(props)
    axes[2].set_xlabel('$R^2$')
    axes[2].set_title('(c) Improvement on\nStructure-Dependent Properties', fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].set_xlim(0, 1.0)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "representation_comparison.png")
    plt.close()
    print("  Saved: representation_comparison.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 7: Innovation Pipeline Diagram
# ══════════════════════════════════════════════════════════════════════

def fig7_innovation_pipeline():
    """Visual diagram of the 3-phase discovery pipeline."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Phase boxes
    phases = [
        (1, 2.5, 3.5, 3, "Phase 1\nSurface Descriptors",
         "20 mesh features\nas tabular input",
         "$R^2$: 0.437 → 0.433\nSignal confirmed for\n$H_{vap}$, $P$",
         '#e74c3c', '#fce4e4'),
        (5.25, 2.5, 3.5, 3, "Phase 2\nTransfer Learning",
         "Pre-train on 5,622\nILThermo samples",
         "$R^2$: 0.437 → 0.549\nGNN learns general\nmolecular repr.",
         '#f39c12', '#fef5e7'),
        (9.5, 2.5, 3.5, 3, "Phase 3\nPoint Cloud Encoder",
         "PointNet on COSMO\nisosurface mesh",
         "$R^2$: 0.437 → 0.655\n300× fewer params\nthan 2D vision",
         '#27ae60', '#e8f8f5'),
    ]

    for x, y, w, h, title, desc, result, border_color, bg_color in phases:
        rect = mpatches.FancyBboxPatch((x, y-h/2), w, h, boxstyle="round,pad=0.1",
                                        facecolor=bg_color, edgecolor=border_color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + 0.9, title, ha='center', va='center',
               fontsize=11, fontweight='bold', color=border_color)
        ax.text(x + w/2, y + 0.15, desc, ha='center', va='center', fontsize=9)
        ax.text(x + w/2, y - 0.75, result, ha='center', va='center', fontsize=8,
               fontstyle='italic')

    # Arrows
    for x1, x2 in [(4.5, 5.25), (8.75, 9.5)]:
        ax.annotate('', xy=(x2, 2.5), xytext=(x1, 2.5),
                   arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

    # Title
    ax.text(7, 4.8, 'Systematic 3-Phase Surface Representation Discovery',
           ha='center', fontsize=14, fontweight='bold')

    # Result box
    rect_result = mpatches.FancyBboxPatch((4, 0), 6, 0.8, boxstyle="round,pad=0.1",
                                           facecolor='#2980b9', edgecolor='#2980b9', linewidth=2, alpha=0.15)
    ax.add_patch(rect_result)
    ax.text(7, 0.4, 'Ensemble: $R^2 = 0.681$ (Phase 2 for $H_{vap}$/$P$ + Phase 3 for rest)',
           ha='center', fontsize=11, fontweight='bold', color='#2980b9')

    plt.savefig(OUTPUT_DIR / "innovation_pipeline.png")
    plt.close()
    print("  Saved: innovation_pipeline.png")


# ══════════════════════════════════════════════════════════════════════

def main():
    print("Generating publication figures...\n")

    results = load_results()
    print(f"Loaded results for {len(results)} models\n")

    fig1_model_progression(results)
    fig2_property_heatmap(results)
    fig3_struct_vs_temp(results)
    fig4_architecture_search(results)
    fig5_radar_chart(results)
    fig6_representation_comparison()
    fig7_innovation_pipeline()

    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
