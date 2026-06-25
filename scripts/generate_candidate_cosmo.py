"""Generate COSMO-style visualization for top IL candidate [MMIM][OAc].

Quick pipeline:
1. SMILES → 3D geometry (RDKit ETKDG + MMFF94)
2. Compute molecular surface (van der Waals)
3. Compute ESP on surface (Gasteiger charges → Coulomb potential)
4. Render 3D surface with ESP coloring
5. Save publication-quality figure

This is an approximation of full DFT COSMO surface (which requires NWChem).
For the paper, this shows the *type* of surface the PointCloud model processes.
"""

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from scipy.spatial import ConvexHull


def generate_surface_points(mol, n_points=5000, probe_radius=1.4):
    """Generate molecular surface points using van der Waals radii."""
    conf = mol.GetConformer()
    vdw_radii = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 16: 1.80, 17: 1.75, 35: 1.85}

    atom_positions = []
    atom_radii = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        atom_positions.append([pos.x, pos.y, pos.z])
        elem = mol.GetAtomWithIdx(i).GetAtomicNum()
        atom_radii.append(vdw_radii.get(elem, 1.70) + probe_radius * 0.3)

    atom_positions = np.array(atom_positions)
    atom_radii = np.array(atom_radii)

    # Generate points on each atom's sphere, keep exposed ones
    surface_points = []
    surface_atoms = []

    for i, (center, radius) in enumerate(zip(atom_positions, atom_radii)):
        # Random points on sphere
        n = max(100, int(n_points * radius**2 / sum(r**2 for r in atom_radii)))
        phi = np.random.uniform(0, 2 * np.pi, n)
        costheta = np.random.uniform(-1, 1, n)
        theta = np.arccos(costheta)

        x = center[0] + radius * np.sin(theta) * np.cos(phi)
        y = center[1] + radius * np.sin(theta) * np.sin(phi)
        z = center[2] + radius * np.sin(theta) * np.sin(phi)
        pts = np.column_stack([x, y, z])

        # Keep points not inside other atoms
        for j, (other_center, other_radius) in enumerate(zip(atom_positions, atom_radii)):
            if i == j: continue
            dists = np.linalg.norm(pts - other_center, axis=1)
            mask = dists > other_radius * 0.9
            pts = pts[mask]
            if len(pts) == 0: break

        if len(pts) > 0:
            surface_points.append(pts)
            surface_atoms.extend([i] * len(pts))

    if not surface_points:
        return np.zeros((0, 3)), np.array([])

    return np.concatenate(surface_points), np.array(surface_atoms)


def compute_esp(mol, surface_points):
    """Compute electrostatic potential at surface points using Gasteiger charges."""
    AllChem.ComputeGasteigerCharges(mol)
    conf = mol.GetConformer()

    charges = []
    positions = []
    for i in range(mol.GetNumAtoms()):
        charge = float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge'))
        if np.isnan(charge): charge = 0.0
        charges.append(charge)
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])

    charges = np.array(charges)
    positions = np.array(positions)

    # Coulomb potential at each surface point
    esp = np.zeros(len(surface_points))
    for i, pt in enumerate(surface_points):
        dists = np.linalg.norm(positions - pt, axis=1)
        dists = np.clip(dists, 0.5, None)  # avoid singularity
        esp[i] = np.sum(charges / dists)

    return esp


def render_cosmo_surface(mol, title, filename, n_points=3000):
    """Render COSMO-style ESP surface visualization."""
    surface_pts, surface_atoms = generate_surface_points(mol, n_points)
    if len(surface_pts) == 0:
        print(f"  No surface points generated for {title}")
        return

    esp = compute_esp(mol, surface_pts)

    # Normalize ESP for coloring
    vmax = max(abs(esp.min()), abs(esp.max()), 0.01)
    esp_norm = esp / vmax  # [-1, 1]

    # Color: blue (negative) → white (neutral) → red (positive)
    cmap = plt.cm.RdBu_r

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Center
    center = surface_pts.mean(axis=0)
    surface_pts_c = surface_pts - center

    colors = cmap((esp_norm + 1) / 2)
    ax.scatter(surface_pts_c[:, 0], surface_pts_c[:, 1], surface_pts_c[:, 2],
               c=colors, s=3, alpha=0.8)

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'COSMO-style ESP Surface\n{title}', fontsize=13, fontweight='bold')

    # Equal aspect ratio
    max_range = np.abs(surface_pts_c).max()
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Electrostatic Potential (a.u.)', fontsize=10)

    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

    # Also render second angle
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(surface_pts_c[:, 0], surface_pts_c[:, 1], surface_pts_c[:, 2],
                c=colors, s=3, alpha=0.8)
    ax2.set_xlabel('X (Å)'); ax2.set_ylabel('Y (Å)'); ax2.set_zlabel('Z (Å)')
    ax2.set_title(f'COSMO-style ESP Surface (side view)\n{title}', fontsize=13, fontweight='bold')
    ax2.set_xlim(-max_range, max_range)
    ax2.set_ylim(-max_range, max_range)
    ax2.set_zlim(-max_range, max_range)
    cbar2 = plt.colorbar(sm, ax=ax2, shrink=0.6, pad=0.1)
    cbar2.set_label('Electrostatic Potential (a.u.)', fontsize=10)
    ax2.view_init(elev=10, azim=135)
    plt.tight_layout()
    fn2 = filename.replace('.png', '_side.png')
    plt.savefig(fn2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fn2}")


def main():
    print("=== Generating COSMO-style Surfaces for Top IL Candidates ===\n")

    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        ("C[n+]1ccn(C)c1.CC(=O)[O-]", "[MMIM][OAc]", "Rank #1: 1,3-Dimethylimidazolium Acetate"),
        ("CN(C)C(=[NH2+])N(C)C.CC(=O)[O-]", "[TMG][OAc]", "Rank #3: Tetramethylguanidinium Acetate"),
        ("C[n+]1ccn(C)c1.CC(=O)CCC(=O)[O-]", "[MMIM][Lev]", "Rank #5: 1,3-Dimethylimidazolium Levulinate"),
    ]

    for smi, short, title in candidates:
        print(f"\n  Processing {short}: {smi}")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  ERROR: Invalid SMILES")
            continue

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except:
            pass

        fn = short.replace('[', '').replace(']', '').replace(' ', '_').lower()
        render_cosmo_surface(mol, f"{title}\n({smi})",
                              str(out_dir / f"cosmo_{fn}.png"))

    # Also render a training IL for comparison
    print(f"\n  Processing [BMIM][OAc] (training set reference)")
    smi_ref = "CCCCn1cc[n+](C)c1.CC(=O)[O-]"
    mol_ref = Chem.MolFromSmiles(smi_ref)
    mol_ref = Chem.AddHs(mol_ref)
    AllChem.EmbedMolecule(mol_ref, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol_ref, maxIters=500)
    render_cosmo_surface(mol_ref, "Reference: [BMIM][OAc] (training set)\n" + smi_ref,
                          str(out_dir / "cosmo_bmim_oac_ref.png"))

    print("\nDone!")


if __name__ == "__main__":
    main()
