"""Generate all figures for the STILT paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
from pathlib import Path

TARGET = ['gamma1', 'gamma2', 'G_E', 'H_E', 'G_mix', 'H_vap', 'P']
TARGET_LABELS = [r'$\gamma_1$', r'$\gamma_2$', r'$G^E$', r'$H^E$', r'$G_{mix}$', r'$H_{vap}$', r'$P$']

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)


def load_results():
    models = {}
    specs = [
        ("COSMO-SAC", "results/cosmo_sac_results.json", "test_metrics"),
        ("Tabular", "results/tabular_improved_results.json", None),
        ("GNN", "results/gnn_v2_results.json", None),
        ("PointCloud", "results/pointcloud_results.json", None),
        ("MoE Fix6", "results/moe_fix6_results.json", "metrics"),
        ("Chemprop", "results/chemprop_results.json", "test_metrics"),
        ("CP+AtomSurf", "results/chemprop_atom_surface_results.json", "CP_AS"),
        ("STILT", "results/chemprop_tuned_results.json", "STILT"),
        ("Ens Top-2", "results/ensemble_all_models_results.json", "ENS"),
    ]
    for name, path, key in specs:
        try:
            data = json.load(open(path))
            if key == "CP_AS": m = data.get("chemprop_as_feat", {}).get("metrics", {})
            elif key == "STILT": m = data.get("C: full mask, 48x OS", {}).get("metrics", {})
            elif key == "ENS": m = data.get("top2_average", {}).get("metrics", {})
            elif key: m = data.get(key, {})
            else:
                for k in ['metrics', 'test_metrics', 'single_model']:
                    if k in data: m = data[k]; break
                else: m = data
            r2 = {}
            for p in TARGET:
                v = m.get(f"{p}_r2")
                r2[p] = float(v) if v is not None and not np.isnan(float(v)) else None
            r2["avg"] = m.get("avg_r2")
            if r2["avg"]: r2["avg"] = float(r2["avg"])
            models[name] = r2
        except Exception as e:
            print(f"  Skip {name}: {e}")
    return models


def fig1_bar_comparison(models):
    """Bar chart comparing all models per property."""
    fig, ax = plt.subplots(figsize=(14, 6))

    paper_models = ["Tabular", "GNN", "Chemprop", "PointCloud", "MoE Fix6",
                     "CP+AtomSurf", "STILT", "Ens Top-2"]
    colors = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12',
              '#9b59b6', '#1abc9c', '#e67e22']

    x = np.arange(len(TARGET))
    width = 0.1
    n = len(paper_models)

    for i, (name, color) in enumerate(zip(paper_models, colors)):
        if name not in models: continue
        vals = [models[name].get(p, 0) or 0 for p in TARGET]
        offset = (i - n/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=name, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_LABELS, fontsize=12)
    ax.set_ylabel(r'$R^2$', fontsize=13)
    ax.set_title('Per-Property Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "model_comparison_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved model_comparison_bar.png")


def fig2_radar(models):
    """Radar chart for top models."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    top_models = ["Chemprop", "PointCloud", "MoE Fix6", "STILT", "Ens Top-2"]
    colors = ['#e74c3c', '#2ecc71', '#f39c12', '#1abc9c', '#e67e22']
    styles = ['--', '-.', ':', '-', '-']

    angles = np.linspace(0, 2 * np.pi, len(TARGET), endpoint=False).tolist()
    angles += angles[:1]

    for name, color, style in zip(top_models, colors, styles):
        if name not in models: continue
        vals = [max(models[name].get(p, 0) or 0, 0) for p in TARGET]
        vals += vals[:1]
        lw = 3 if name == "STILT" else 1.5
        ax.plot(angles, vals, style, linewidth=lw, label=name, color=color)
        ax.fill(angles, vals, alpha=0.05, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(TARGET_LABELS, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=8)
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0), fontsize=9)
    ax.set_title('Top Models: Per-Property R²', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUT / "radar_top_models.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved radar_top_models.png")


def fig3_progression(models):
    """Model progression showing R² improvement."""
    fig, ax = plt.subplots(figsize=(12, 5))

    progression = [
        ("COSMO-SAC\n(physics)", -0.772, '#95a5a6'),
        ("Tabular\n(baseline)", 0.339, '#bdc3c7'),
        ("GNN v2\n(graph)", 0.578, '#3498db'),
        ("PointCloud\n(3D surface)", 0.655, '#2ecc71'),
        ("MoE Fix6\n(ILThermo)", 0.650, '#f39c12'),
        ("CP+AtomSurf\n(per-atom)", 0.721, '#9b59b6'),
        ("Chemprop\n(D-MPNN)", 0.770, '#e74c3c'),
        ("STILT\n(ours)", 0.773, '#1abc9c'),
        ("Ens Top-2\n(7 models)", 0.795, '#e67e22'),
    ]

    names = [p[0] for p in progression]
    vals = [p[1] for p in progression]
    colors = [p[2] for p in progression]

    bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)

    for i, (v, bar) in enumerate(zip(vals, bars)):
        if v > 0:
            ax.text(i, v + 0.015, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
        else:
            ax.text(i, 0.02, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

    # Highlight STILT
    bars[7].set_edgecolor('#006400')
    bars[7].set_linewidth(3)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8, rotation=0)
    ax.set_ylabel(r'Average $R^2$', fontsize=12)
    ax.set_title('Model Progression: From Physics Baseline to STILT', fontsize=13, fontweight='bold')
    ax.axhline(y=0.770, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(len(names)-0.5, 0.775, 'Chemprop baseline', fontsize=7, color='red', ha='right')
    ax.set_ylim(-0.1, 0.85)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "model_progression.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved model_progression.png")


def fig4_stilt_vs_chemprop(models):
    """Head-to-head STILT vs Chemprop."""
    fig, ax = plt.subplots(figsize=(10, 5))

    stilt = [models["STILT"].get(p, 0) or 0 for p in TARGET]
    chemp = [models["Chemprop"].get(p, 0) or 0 for p in TARGET]
    delta = [s - c for s, c in zip(stilt, chemp)]

    x = np.arange(len(TARGET))
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in delta]

    bars = ax.bar(x, delta, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)

    for i, (d, bar) in enumerate(zip(delta, bars)):
        sign = "+" if d > 0 else ""
        y = d + 0.002 if d > 0 else d - 0.005
        ax.text(i, y, f'{sign}{d:.3f}', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_LABELS, fontsize=12)
    ax.set_ylabel(r'$\Delta R^2$ (STILT $-$ Chemprop)', fontsize=12)
    ax.set_title('STILT vs Chemprop: Per-Property Improvement', fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylim(-0.04, 0.04)
    ax.grid(axis='y', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    ax.legend([Patch(color='#2ecc71'), Patch(color='#e74c3c')],
              ['STILT wins', 'Chemprop wins'], fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(OUT / "stilt_vs_chemprop.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved stilt_vs_chemprop.png")


def fig5_transfer_learning_ablation():
    """Ablation: oversampling ratio and gamma1 masking."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Oversampling ratio effect (full mask)
    ratios = [0, 10, 24, 48]
    avgs = [0.770, 0.753, 0.768, 0.773]  # base, B, v3b, C
    g1s = [0.828, 0.808, 0.824, 0.823]

    ax1.plot(ratios, avgs, 'o-', color='#1abc9c', linewidth=2, markersize=8, label='Avg R²')
    ax1.plot(ratios, g1s, 's--', color='#3498db', linewidth=2, markersize=8, label=r'$\gamma_1$ R²')
    ax1.axhline(y=0.770, color='red', linestyle=':', alpha=0.5)
    ax1.text(48, 0.772, 'Chemprop', fontsize=8, color='red')
    ax1.set_xlabel('Oversampling Ratio', fontsize=11)
    ax1.set_ylabel(r'$R^2$', fontsize=11)
    ax1.set_title('(a) Effect of Oversampling Ratio\n(full gamma1 mask)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_ylim(0.74, 0.84)
    ax1.grid(alpha=0.3)

    # Panel B: Gamma1 masking fraction (24x OS)
    fracs = [0, 50, 100]
    avgs2 = [0.743, 0.726, 0.768]  # v2, A, v3b
    g1s2 = [0.374, 0.201, 0.824]

    ax2.plot(fracs, avgs2, 'o-', color='#1abc9c', linewidth=2, markersize=8, label='Avg R²')
    ax2.plot(fracs, g1s2, 's--', color='#3498db', linewidth=2, markersize=8, label=r'$\gamma_1$ R²')
    ax2.axhline(y=0.770, color='red', linestyle=':', alpha=0.5)
    ax2.set_xlabel('ILThermo Gamma1 Mask %', fontsize=11)
    ax2.set_ylabel(r'$R^2$', fontsize=11)
    ax2.set_title('(b) Effect of Gamma1 Masking\n(24x oversampling)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_ylim(0.1, 0.9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "transfer_ablation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved transfer_ablation.png")


def main():
    print("Generating paper figures...")
    models = load_results()
    print(f"  Loaded {len(models)} models")

    fig1_bar_comparison(models)
    fig2_radar(models)
    fig3_progression(models)
    fig4_stilt_vs_chemprop(models)
    fig5_transfer_learning_ablation()

    print("\nAll figures generated!")


if __name__ == "__main__":
    main()
