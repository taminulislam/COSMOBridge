"""Step 4b: Extract numerical surface descriptors from COSMO-style isosurfaces.

Computes geometry, curvature, and ESP statistics from the marching-cubes mesh
for each unique ionic liquid. These descriptors are saved as a CSV that can be
merged into the training data as additional tabular features (Phase 1).

Input:  data/pipeline/geometry_status.csv  OR  data/processed/il_data_raw.csv
Output: data/pipeline/surface_descriptors.csv

Descriptors per IL (20 features):
  Geometry (4):  surface_area, volume, sphericity, aspect_ratio
  Curvature (6): curv_mean, curv_std, curv_skew, gcurv_mean, gcurv_std, gcurv_skew
  ESP (10):      esp_mean, esp_std, esp_min, esp_max, esp_skew, esp_kurtosis,
                 esp_pos_frac, esp_neg_frac, esp_charge_segregation, esp_range
"""

import csv
import numpy as np
from pathlib import Path
from scipy.stats import skew, kurtosis

from rdkit import Chem
from rdkit.Chem import AllChem
from skimage import measure

# ── Shared constants/functions from step4 ────────────────────────────────────

VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
    'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'P': 1.80, 'B': 1.92, 'I': 1.98,
}


def prepare_molecule(smiles):
    """Parse SMILES, embed 3D, optimize, compute charges."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if res != 0:
        p = AllChem.ETKDGv3()
        p.useRandomCoords = True
        res = AllChem.EmbedMolecule(mol, p)
        if res != 0:
            return None
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    AllChem.ComputeGasteigerCharges(mol)
    return mol


def get_mol_data(mol):
    """Extract positions, charges, radii."""
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    pos = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z] for i in range(n)])
    charges = []
    for a in mol.GetAtoms():
        try:
            c = float(a.GetProp('_GasteigerCharge'))
        except (KeyError, ValueError):
            c = 0.0
        if np.isnan(c):
            c = 0.0
        charges.append(c)
    radii = np.array([VDW_RADII.get(mol.GetAtomWithIdx(i).GetSymbol(), 1.7) for i in range(n)])
    return pos, np.array(charges), radii


def separate_ions(mol, pos, separation=2.5):
    """Shift cation and anion apart for clearer surfaces."""
    frags = Chem.GetMolFrags(mol, asMols=False)
    if len(frags) >= 2:
        cen0 = pos[list(frags[0])].mean(axis=0)
        cen1 = pos[list(frags[1])].mean(axis=0)
        direction = cen1 - cen0
        norm = np.linalg.norm(direction)
        if norm > 0.1:
            direction /= norm
        else:
            direction = np.array([1.0, 0.0, 0.0])
        offset = direction * separation / 2
        for ai in frags[0]:
            pos[ai] -= offset
        for ai in frags[1]:
            pos[ai] += offset
    return pos


def build_isosurface(pos, radii, grid_res=0.20, iso_level=0.18, pad=3.5,
                     sigma_factor=0.60):
    """Build molecular isosurface using Gaussian density + marching cubes."""
    n = len(pos)
    xr = [pos[:, 0].min() - pad, pos[:, 0].max() + pad]
    yr = [pos[:, 1].min() - pad, pos[:, 1].max() + pad]
    zr = [pos[:, 2].min() - pad, pos[:, 2].max() + pad]

    x = np.arange(xr[0], xr[1], grid_res)
    y = np.arange(yr[0], yr[1], grid_res)
    z = np.arange(zr[0], zr[1], grid_res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    density = np.zeros_like(X)
    for i in range(n):
        sigma = radii[i] * sigma_factor
        dist2 = (X - pos[i, 0])**2 + (Y - pos[i, 1])**2 + (Z - pos[i, 2])**2
        density += np.exp(-dist2 / (2 * sigma**2))

    try:
        verts, faces, normals, _ = measure.marching_cubes(
            density, level=iso_level, spacing=(grid_res, grid_res, grid_res)
        )
    except Exception:
        return None, None, None

    verts += np.array([xr[0], yr[0], zr[0]])
    return verts, faces, normals


def compute_surface_esp(verts, pos, charges, radii, mol=None, sigma=0.5):
    """Fragment-aware ESP interpolation at surface vertices."""
    from scipy.spatial import cKDTree
    n = len(pos)
    tree = cKDTree(pos)

    frags = None
    if mol is not None:
        try:
            frags = Chem.GetMolFrags(mol, asMols=False)
        except Exception:
            pass

    if frags is not None and len(frags) > 1:
        frag_centroids = [pos[list(f)].mean(axis=0) for f in frags]
        esp = np.zeros(len(verts))
        for vi in range(len(verts)):
            frag_dists = [np.linalg.norm(verts[vi] - fc) for fc in frag_centroids]
            nearest_frag = np.argmin(frag_dists)
            frag_atoms = list(frags[nearest_frag])
            frag_pos = pos[frag_atoms]
            frag_charges = charges[frag_atoms]
            dists_f = np.linalg.norm(frag_pos - verts[vi], axis=1)
            w = np.exp(-dists_f**2 / (2 * sigma**2))
            w /= w.sum() + 1e-10
            esp[vi] = np.sum(frag_charges * w)
    else:
        k = min(6, n)
        dists, idxs = tree.query(verts, k=k)
        if k == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        w = np.exp(-dists**2 / (2 * sigma**2))
        w /= w.sum(axis=1, keepdims=True) + 1e-10
        esp = np.sum(charges[idxs] * w, axis=1)

    return esp


# ── Descriptor computation ───────────────────────────────────────────────────

def triangle_areas(verts, faces):
    """Compute area of each triangle face."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def mesh_volume(verts, faces):
    """Compute signed volume of a closed triangulated mesh."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    # Signed volume of tetrahedra formed with origin
    return np.abs(np.sum(
        v0[:, 0] * (v1[:, 1] * v2[:, 2] - v1[:, 2] * v2[:, 1]) +
        v0[:, 1] * (v1[:, 2] * v2[:, 0] - v1[:, 0] * v2[:, 2]) +
        v0[:, 2] * (v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    )) / 6.0


def vertex_curvatures(verts, faces, normals):
    """Estimate mean and Gaussian curvature per vertex using discrete geometry.

    Uses the angle-deficit method for Gaussian curvature and the
    normal-divergence method for mean curvature.
    """
    n_verts = len(verts)
    # Accumulate angle around each vertex and mixed area
    angle_sum = np.zeros(n_verts)
    mixed_area = np.zeros(n_verts)
    mean_curv = np.zeros(n_verts)

    for face in faces:
        i0, i1, i2 = face
        v = verts[[i0, i1, i2]]
        # Edge vectors for each vertex in the triangle
        for local_idx, (a, b, c) in enumerate([(0, 1, 2), (1, 2, 0), (2, 0, 1)]):
            va = v[b] - v[a]
            vb = v[c] - v[a]
            cos_angle = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            vid = face[a]
            angle_sum[vid] += angle

            # Voronoi mixed area contribution
            tri_area = 0.5 * np.linalg.norm(np.cross(va, vb))
            mixed_area[vid] += tri_area / 3.0

            # Mean curvature via Laplace-Beltrami (cotangent weight)
            cot_angle = cos_angle / (np.sin(angle) + 1e-12)
            edge_b = v[c] - v[b]
            vid_b = face[b]
            vid_c = face[c]
            edge_bc = verts[vid_c] - verts[vid_b]
            mean_curv[vid_b] += cot_angle * np.dot(edge_bc, normals[vid_b])
            mean_curv[vid_c] -= cot_angle * np.dot(edge_bc, normals[vid_c])

    # Gaussian curvature: 2*pi - angle_sum, divided by area
    safe_area = np.maximum(mixed_area, 1e-10)
    gauss_curv = (2 * np.pi - angle_sum) / safe_area
    mean_curv = mean_curv / (2 * safe_area)

    # Clip extreme values (numerical artifacts at mesh boundaries)
    gauss_curv = np.clip(gauss_curv, -50, 50)
    mean_curv = np.clip(mean_curv, -50, 50)

    return mean_curv, gauss_curv


def compute_descriptors(smiles, il_name):
    """Compute all surface descriptors for one ionic liquid.

    Returns a dict of descriptor name -> value, or None on failure.
    """
    mol = prepare_molecule(smiles)
    if mol is None:
        print(f"  SKIP {il_name}: molecule preparation failed")
        return None

    pos, charges, radii = get_mol_data(mol)
    pos = separate_ions(mol, pos, separation=2.5)

    verts, faces, normals = build_isosurface(pos, radii, grid_res=0.20)
    if verts is None or len(verts) < 10:
        print(f"  SKIP {il_name}: isosurface failed")
        return None

    esp = compute_surface_esp(verts, pos, charges, radii, mol=mol)

    # ── Geometry descriptors ──
    areas = triangle_areas(verts, faces)
    surface_area = areas.sum()
    volume = mesh_volume(verts, faces)
    # Sphericity: ratio of surface area of equivalent sphere to actual surface area
    equiv_radius = (3 * volume / (4 * np.pi)) ** (1.0 / 3.0)
    sphere_area = 4 * np.pi * equiv_radius ** 2
    sphericity = sphere_area / (surface_area + 1e-10)
    # Aspect ratio from PCA of vertex positions
    centered = verts - verts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    aspect_ratio = eigvals[0] / (eigvals[2] + 1e-10)

    # ── Curvature descriptors ──
    mean_curv, gauss_curv = vertex_curvatures(verts, faces, normals)
    curv_mean = np.mean(mean_curv)
    curv_std = np.std(mean_curv)
    curv_skew = float(skew(mean_curv))
    gcurv_mean = np.mean(gauss_curv)
    gcurv_std = np.std(gauss_curv)
    gcurv_skew = float(skew(gauss_curv))

    # ── ESP descriptors ──
    esp_mean = np.mean(esp)
    esp_std = np.std(esp)
    esp_min = np.min(esp)
    esp_max = np.max(esp)
    esp_skew_val = float(skew(esp))
    esp_kurt = float(kurtosis(esp))
    esp_pos_frac = np.mean(esp > 0)
    esp_neg_frac = np.mean(esp < 0)
    # Charge segregation: weighted separation between positive and negative regions
    pos_mask = esp > 0
    neg_mask = esp < 0
    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
        pos_centroid = verts[pos_mask].mean(axis=0)
        neg_centroid = verts[neg_mask].mean(axis=0)
        esp_charge_seg = np.linalg.norm(pos_centroid - neg_centroid)
    else:
        esp_charge_seg = 0.0
    esp_range = esp_max - esp_min

    return {
        "il_short_name": il_name,
        "surface_area": round(surface_area, 4),
        "volume": round(volume, 4),
        "sphericity": round(sphericity, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "curv_mean": round(curv_mean, 6),
        "curv_std": round(curv_std, 6),
        "curv_skew": round(curv_skew, 6),
        "gcurv_mean": round(gcurv_mean, 6),
        "gcurv_std": round(gcurv_std, 6),
        "gcurv_skew": round(gcurv_skew, 6),
        "esp_mean": round(esp_mean, 6),
        "esp_std": round(esp_std, 6),
        "esp_min": round(esp_min, 6),
        "esp_max": round(esp_max, 6),
        "esp_skew": round(esp_skew_val, 6),
        "esp_kurtosis": round(esp_kurt, 6),
        "esp_pos_frac": round(esp_pos_frac, 6),
        "esp_neg_frac": round(esp_neg_frac, 6),
        "esp_charge_segregation": round(esp_charge_seg, 4),
        "esp_range": round(esp_range, 6),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import pandas as pd

    base_dir = Path(__file__).resolve().parent.parent.parent

    # Load unique ILs with SMILES
    raw_csv = base_dir / "data" / "processed" / "il_data_raw.csv"
    if not raw_csv.exists():
        print(f"ERROR: {raw_csv} not found. Run preprocessing first.")
        return

    df = pd.read_csv(raw_csv)
    unique_ils = df.drop_duplicates(subset=["il_short_name"])[["il_short_name", "smiles"]].sort_values("il_short_name")
    print(f"Computing surface descriptors for {len(unique_ils)} unique ILs...\n")

    results = []
    for _, row in unique_ils.iterrows():
        il_name = row["il_short_name"]
        smiles = row["smiles"]
        print(f"  Processing {il_name}...")
        desc = compute_descriptors(smiles, il_name)
        if desc is not None:
            results.append(desc)
            print(f"    OK: SA={desc['surface_area']:.1f}, V={desc['volume']:.1f}, "
                  f"spher={desc['sphericity']:.3f}, ESP_range={desc['esp_range']:.4f}")
        else:
            print(f"    FAILED")

    # Save
    output_path = base_dir / "data" / "pipeline" / "surface_descriptors.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys()) if results else []
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)}/{len(unique_ils)} descriptors to {output_path}")

    # Print summary statistics
    if results:
        desc_keys = [k for k in results[0].keys() if k != "il_short_name"]
        print(f"\nDescriptor summary ({len(desc_keys)} features):")
        for k in desc_keys:
            vals = [r[k] for r in results]
            print(f"  {k:30s}  mean={np.mean(vals):10.4f}  std={np.std(vals):10.4f}  "
                  f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]")


if __name__ == "__main__":
    main()
