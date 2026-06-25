"""Step 4: Render COSMO-style molecular isosurface images with ESP coloring.

Uses marching cubes to extract a smooth molecular isosurface from a Gaussian
electron density field, then maps Gasteiger partial charges as a continuous
ESP color gradient — producing images visually similar to DFT COSMO sigma surfaces.

Input:  data/pipeline/geometry_status.csv (with SMILES)
Output: data/pipeline/cosmo_images/{compound_id}_cosmo.png
        data/pipeline/cosmo_images/{compound_id}_ep.png
        data/pipeline/cosmo_images/{compound_id}_frames/*.png
"""

import csv
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from skimage import measure
from scipy.spatial import cKDTree

from rdkit import Chem
from rdkit.Chem import AllChem


DFT_SURFACE_DIR = Path("data/pipeline/dft_surface")
GEOMETRY_DIR = Path("data/pipeline/geometries")


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].strip())
    atoms, coords = [], []
    for i in range(n):
        parts = lines[2 + i].split()
        atoms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, np.asarray(coords)


def load_dft_points(compound_id, dft_dir=DFT_SURFACE_DIR):
    """Return (M,4) array of (x,y,z,esp) for the _pair DFT surface.

    Returns None if the file does not exist. The step2 XYZ geometries
    and the Psi4 DFT cavity share the same coordinate frame, so the
    returned points can be used for nearest-neighbor ESP lookup onto
    any isosurface built from those XYZ positions.
    """
    npz = dft_dir / f"{compound_id}_pair.npz"
    if not npz.exists():
        return None
    data = np.load(npz, allow_pickle=True)
    surface = data["surface"]  # (M, 7)
    return np.column_stack([surface[:, :3], surface[:, 6]])  # x,y,z,esp


def esp_from_dft(verts, dft_points):
    """Nearest-neighbor ESP lookup for each vertex."""
    from scipy.spatial import cKDTree
    tree = cKDTree(dft_points[:, :3])
    _, idx = tree.query(verts, k=1)
    return dft_points[idx, 3]

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


def build_isosurface(pos, radii, grid_res=0.25, iso_level=0.18, pad=3.5,
                     sigma_factor=0.60):
    """Build molecular isosurface using Gaussian density + marching cubes.

    Uses wide Gaussians (sigma_factor=0.60) and low iso-level (0.18) to
    produce a smooth, gap-free molecular envelope similar to DFT COSMO surfaces.
    """
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
    """Fragment-aware ESP interpolation at surface vertices.

    Each vertex is assigned to the nearest ionic fragment (cation or anion)
    and its ESP is computed only from atoms within that fragment. This
    prevents dilution of strongly charged ions (e.g., Cl-) by nearby
    neutral atoms from the counter-ion.

    Falls back to simple nearest-neighbor weighting if fragment info
    is not available.
    """
    n = len(pos)
    tree = cKDTree(pos)

    # Determine fragment membership
    frags = None
    if mol is not None:
        try:
            frags = Chem.GetMolFrags(mol, asMols=False)
        except Exception:
            pass

    if frags is not None and len(frags) > 1:
        # Fragment-aware: assign each vertex to nearest fragment centroid,
        # then interpolate ESP using only atoms in that fragment
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
        # Single fragment: standard distance-weighted interpolation
        k = min(6, n)
        dists, idxs = tree.query(verts, k=k)
        if k == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        w = np.exp(-dists**2 / (2 * sigma**2))
        w /= w.sum(axis=1, keepdims=True) + 1e-10
        esp = np.sum(charges[idxs] * w, axis=1)

    return esp


def rotation_matrix(elev_deg, azim_deg):
    """Build rotation matrix."""
    elev, azim = np.radians(elev_deg), np.radians(azim_deg)
    Ry = np.array([[np.cos(azim), 0, np.sin(azim)], [0, 1, 0], [-np.sin(azim), 0, np.cos(azim)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(elev), -np.sin(elev)], [0, np.sin(elev), np.cos(elev)]])
    return Ry @ Rx


def esp_to_rgb(t, vmax):
    """Standard COSMO sigma surface colormap.

    Red (strongly negative, H-bond donor, e.g. Cl-)
    → Yellow (weakly negative)
    → Green (neutral)
    → Cyan (weakly positive)
    → Blue (strongly positive, H-bond acceptor)

    Uses asymmetric vmax: negative charges (anions) typically have
    larger magnitude than positive (cations), so we use the actual
    range rather than symmetric clipping.
    """
    t = np.clip(t / max(vmax, 0.01), -1, 1)
    if t < -0.6:
        # Deep red (strongly negative — anion surface)
        f = (t + 1.0) / 0.4  # 0 at t=-1, 1 at t=-0.6
        return (230, int(20 * f), int(10 * f))
    elif t < -0.2:
        # Red → Yellow transition
        f = (t + 0.6) / 0.4  # 0 at t=-0.6, 1 at t=-0.2
        return (230, int(20 + 200 * f), int(10 + 20 * f))
    elif t < 0.2:
        # Yellow → Green transition (neutral zone)
        f = (t + 0.2) / 0.4  # 0 at t=-0.2, 1 at t=0.2
        return (int(230 - 200 * f), int(220 - 10 * f), int(30 + 60 * f))
    elif t < 0.6:
        # Green → Cyan transition
        f = (t - 0.2) / 0.4  # 0 at t=0.2, 1 at t=0.6
        return (int(30 - 20 * f), int(210 - 20 * f), int(90 + 140 * f))
    else:
        # Cyan → Blue (strongly positive)
        f = (t - 0.6) / 0.4  # 0 at t=0.6, 1 at t=1.0
        return (int(10), int(190 - 150 * f), int(230 + 25 * f))


def render_isosurface(verts, faces, normals, esp, R, img_size=512, alpha=1.0):
    """Rasterize triangulated isosurface with ESP coloring and Phong lighting."""
    center = verts.mean(axis=0)
    verts_r = (verts - center) @ R.T
    normals_r = normals @ R.T

    margin = 50
    eff = img_size - 2 * margin
    rng = max(verts_r[:, 0].ptp(), verts_r[:, 1].ptp(), 0.01)
    scale = eff / rng
    cx = img_size / 2 - (verts_r[:, 0].min() + verts_r[:, 0].max()) / 2 * scale
    cy = img_size / 2 - (verts_r[:, 1].min() + verts_r[:, 1].max()) / 2 * scale

    img_arr = np.full((img_size, img_size, 3), 255, dtype=np.uint8)
    zbuf = np.full((img_size, img_size), -np.inf)

    light = np.array([0.3, 0.5, 1.0])
    light /= np.linalg.norm(light)
    half_v = light + np.array([0, 0, 1])
    half_v /= np.linalg.norm(half_v)
    vmax = max(abs(esp).max(), 0.12)

    # Sort faces back to front
    face_depths = np.mean(verts_r[faces, 2], axis=1)
    sorted_faces = faces[np.argsort(face_depths)]

    for face in sorted_faces:
        v0, v1, v2 = face
        px_f = np.array([verts_r[v, 0] * scale + cx for v in face])
        py_f = np.array([verts_r[v, 1] * scale + cy for v in face])
        pz_f = np.array([verts_r[v, 2] for v in face])

        fn = (normals_r[v0] + normals_r[v1] + normals_r[v2]) / 3
        fn_len = np.linalg.norm(fn)
        if fn_len < 1e-8:
            continue
        fn /= fn_len
        if fn[2] < -0.05:
            continue

        diff = max(0, np.dot(fn, light))
        spec = max(0, np.dot(fn, half_v)) ** 35
        shade = 0.30 + 0.70 * diff

        xmin_f = max(0, int(min(px_f)) - 1)
        xmax_f = min(img_size - 1, int(max(px_f)) + 1)
        ymin_f = max(0, int(min(py_f)) - 1)
        ymax_f = min(img_size - 1, int(max(py_f)) + 1)

        for yi in range(ymin_f, ymax_f + 1):
            for xi in range(xmin_f, xmax_f + 1):
                # Barycentric coordinates
                d00 = (px_f[1] - px_f[0])**2 + (py_f[1] - py_f[0])**2
                d01 = (px_f[1] - px_f[0]) * (px_f[2] - px_f[0]) + (py_f[1] - py_f[0]) * (py_f[2] - py_f[0])
                d11 = (px_f[2] - px_f[0])**2 + (py_f[2] - py_f[0])**2
                d20 = (xi - px_f[0]) * (px_f[1] - px_f[0]) + (yi - py_f[0]) * (py_f[1] - py_f[0])
                d21 = (xi - px_f[0]) * (px_f[2] - px_f[0]) + (yi - py_f[0]) * (py_f[2] - py_f[0])
                denom = d00 * d11 - d01 * d01
                if abs(denom) < 1e-10:
                    continue
                u = (d11 * d20 - d01 * d21) / denom
                vb = (d00 * d21 - d01 * d20) / denom
                if u >= 0 and vb >= 0 and u + vb <= 1:
                    zi = (1 - u - vb) * pz_f[0] + u * pz_f[1] + vb * pz_f[2]
                    if zi > zbuf[yi, xi]:
                        zbuf[yi, xi] = zi
                        px_esp = (1 - u - vb) * esp[v0] + u * esp[v1] + vb * esp[v2]
                        bc = esp_to_rgb(px_esp, vmax)
                        rr = min(255, int(bc[0] * shade + 200 * spec * 0.25))
                        gg = min(255, int(bc[1] * shade + 200 * spec * 0.25))
                        bb = min(255, int(bc[2] * shade + 200 * spec * 0.25))
                        if alpha < 1.0:
                            rr = int(rr * alpha + 255 * (1 - alpha))
                            gg = int(gg * alpha + 255 * (1 - alpha))
                            bb = int(bb * alpha + 255 * (1 - alpha))
                        img_arr[yi, xi] = [rr, gg, bb]

    img = Image.fromarray(img_arr)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    return img


def render_ep_image(verts, faces, normals, esp, mol, pos, R, img_size=512):
    """Render EP-style: translucent surface + ball-and-stick wireframe."""
    img = render_isosurface(verts, faces, normals, esp, R, img_size, alpha=0.55)
    draw = ImageDraw.Draw(img)

    center = verts.mean(axis=0)
    verts_r = (verts - center) @ R.T
    margin = 50
    eff = img_size - 2 * margin
    rng = max(verts_r[:, 0].ptp(), verts_r[:, 1].ptp(), 0.01)
    scale = eff / rng
    cx = img_size / 2 - (verts_r[:, 0].min() + verts_r[:, 0].max()) / 2 * scale
    cy = img_size / 2 - (verts_r[:, 1].min() + verts_r[:, 1].max()) / 2 * scale

    atoms_r = (pos - center) @ R.T
    ax = (atoms_r[:, 0] * scale + cx).astype(int)
    ay = (atoms_r[:, 1] * scale + cy).astype(int)

    elem_colors = {
        'C': (80, 80, 80), 'N': (30, 30, 200), 'O': (200, 30, 30),
        'S': (200, 200, 30), 'F': (30, 200, 30), 'Cl': (30, 200, 30),
        'Br': (150, 50, 0), 'P': (200, 100, 0), 'H': (200, 200, 200),
    }

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        draw.line([(ax[i], ay[i]), (ax[j], ay[j])], fill=(100, 100, 100), width=1)

    n = mol.GetNumAtoms()
    for i in range(n):
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        if sym == 'H':
            continue
        color = elem_colors.get(sym, (150, 150, 150))
        draw.ellipse([ax[i] - 3, ay[i] - 3, ax[i] + 3, ay[i] + 3],
                      fill=color, outline=(60, 60, 60))

    return img


def separate_ions(mol, pos, separation=2.5):
    """Push cation and anion fragments apart for clearer COSMO visualization."""
    frags = Chem.GetMolFrags(mol, asMols=False)
    if len(frags) < 2:
        return pos.copy()

    pos = pos.copy()
    centroids = [pos[list(f)].mean(axis=0) for f in frags]
    direction = centroids[1] - centroids[0]
    dist = np.linalg.norm(direction)
    direction = direction / dist if dist > 0.1 else np.array([1.0, 0.0, 0.0])

    for fi, frag in enumerate(frags):
        offset = direction * separation * (1 if fi == 1 else -1) * 0.5
        for ai in frag:
            pos[ai] += offset

    return pos


def render_molecule(smiles, compound_id, output_dir, img_size=512,
                    grid_res=0.20, n_views=36, frames_only=False,
                    dft_dir=DFT_SURFACE_DIR, geom_dir=GEOMETRY_DIR):
    """Render full COSMO image set for one molecule.

    If a DFT surface + step2 XYZ geometry is available for the compound,
    uses the DFT-optimized positions for the isosurface shape and
    nearest-neighbor lookup into the Psi4 cavity for ESP coloring. Falls
    back to the RDKit + Gasteiger path otherwise.
    """
    dft_points = load_dft_points(compound_id, dft_dir)
    xyz_path = geom_dir / f"{compound_id}_pair.xyz"
    use_dft = dft_points is not None and xyz_path.exists()

    if use_dft:
        atoms, pos = read_xyz(xyz_path)
        radii = np.array([VDW_RADII.get(a, 1.7) for a in atoms])
        verts, faces, normals = build_isosurface(pos, radii, grid_res=grid_res)
        if verts is None or len(verts) < 10:
            return False
        esp = esp_from_dft(verts, dft_points)
        mol = None  # atom order differs from RDKit; skip EP wireframe view
    else:
        mol = prepare_molecule(smiles)
        if mol is None:
            return False
        pos, charges, radii = get_mol_data(mol)
        pos = separate_ions(mol, pos, separation=2.5)
        verts, faces, normals = build_isosurface(pos, radii, grid_res=grid_res)
        if verts is None or len(verts) < 10:
            return False
        esp = compute_surface_esp(verts, pos, charges, radii, mol=mol)

    if not frames_only:
        if mol is not None:
            frags = Chem.GetMolFrags(mol, asMols=False)
            if len(frags) >= 2:
                an_cen = pos[list(frags[1])].mean(axis=0)
                cat_cen = pos[list(frags[0])].mean(axis=0)
                v_dir = an_cen - cat_cen
                auto_azim = np.degrees(np.arctan2(v_dir[0], v_dir[2])) + 30
            else:
                auto_azim = 35
        else:
            auto_azim = 35

        R = rotation_matrix(20, auto_azim)
        cosmo = render_isosurface(verts, faces, normals, esp, R, img_size)
        cosmo.save(output_dir / f"{compound_id}_cosmo.png", quality=95)

        if mol is not None:
            ep = render_ep_image(verts, faces, normals, esp, mol, pos, R, img_size)
            ep.save(output_dir / f"{compound_id}_ep.png", quality=95)

    # Rotation frames (reuse fine-grid isosurface — no holes/artifacts)
    frames_dir = output_dir / f"{compound_id}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for fi in range(n_views):
        angle = fi * (360.0 / n_views)
        R_f = rotation_matrix(15, angle)
        frame = render_isosurface(verts, faces, normals, esp, R_f, img_size)
        frame.save(frames_dir / f"frame_{fi:03d}.png", quality=85)

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--frames-only", action="store_true",
                        help="Re-render only rotation frames (skip COSMO/EP main images)")
    args = parser.parse_args()

    geom_status_csv = Path("data/pipeline/geometry_status.csv")
    compounds_csv = Path("data/pipeline/ilthermo_compounds.csv")
    output_dir = Path("data/pipeline/cosmo_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    if geom_status_csv.exists() and geom_status_csv.stat().st_size > 200:
        source_csv = geom_status_csv
    elif compounds_csv.exists():
        source_csv = compounds_csv
    else:
        print("ERROR: No compound data found.")
        return

    compounds = []
    with open(source_csv) as f:
        for row in csv.DictReader(f):
            if row.get("smiles"):
                compounds.append(row)

    # Subset for parallel execution
    end = args.end if args.end is not None else len(compounds)
    compounds = compounds[args.start:end]
    print(f"Rendering compounds [{args.start}:{end}] ({len(compounds)} compounds)...", flush=True)

    success = 0
    failed = 0
    for i, comp in enumerate(compounds):
        cid = comp["compound_id"]
        smiles = comp["smiles"]

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(compounds)}] {cid} ({success} ok, {failed} fail)", flush=True)

        # Skip logic: if --frames-only, always re-render frames
        if not args.frames_only:
            cosmo_file = output_dir / f"{cid}_cosmo.png"
            if cosmo_file.exists() and cosmo_file.stat().st_size > 60000:
                frames_dir = output_dir / f"{cid}_frames"
                if frames_dir.exists() and len(list(frames_dir.glob("frame_*.png"))) >= 36:
                    success += 1
                    continue

        try:
            ok = render_molecule(smiles, cid, output_dir, img_size=512,
                                 grid_res=0.20, n_views=36,
                                 frames_only=args.frames_only)
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"    ERROR {cid}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"COSMO Isosurface Rendering Summary")
    print(f"{'='*60}")
    print(f"  Total: {len(compounds)}, Success: {success}, Failed: {failed}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
