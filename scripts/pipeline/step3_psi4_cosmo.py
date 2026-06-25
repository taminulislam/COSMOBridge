"""Step 3: DFT surface-charge calculation using Psi4 with PCM solvation.

Replaces the earlier NWChem-based implementation. Runs B3LYP/def2-SVP with
an implicit solvation cavity (epsilon = 15.0, matching the NWChem COSMO
setup) and samples the electrostatic potential on a solvent-accessible
surface tessellation. The output .npz file is in the same 7-column format
(x, y, z, nx, ny, nz, sigma) consumed by step4c_extract_point_clouds.py.

Input:  data/pipeline/geometries/{compound_id}.xyz
Output: data/pipeline/dft_surface/{compound_id}.npz

Charge and multiplicity are inferred from compound_id suffix:
  _cation -> +1, _anion -> -1, _pair -> 0 (all closed-shell singlets)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import psi4

# Van der Waals radii in angstroms; used to seed the SAS tessellation.
VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "F": 1.47, "Cl": 1.75, "Br": 1.85, "P": 1.80, "B": 1.92, "I": 1.98,
}
ATOMIC_NUMBER = {
    "H": 1, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "K": 19, "Ca": 20, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33,
    "Se": 34, "Br": 35, "Sn": 50, "Sb": 51, "Te": 52, "I": 53,
}
SOLVENT_PROBE = 1.385  # angstrom, same as NWChem COSMO rsolv=1.385


def multiplicity_for(atoms, charge):
    """Closed-shell singlet if even electrons, doublet if odd."""
    nelec = sum(ATOMIC_NUMBER[a] for a in atoms) - charge
    return 1 if nelec % 2 == 0 else 2


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].strip())
    atoms = []
    coords = np.zeros((n, 3))
    for i in range(n):
        parts = lines[2 + i].split()
        atoms.append(parts[0])
        coords[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
    return atoms, coords


def charge_from_id(compound_id):
    if compound_id.endswith("_cation"):
        return 1
    if compound_id.endswith("_anion"):
        return -1
    return 0


def fibonacci_sphere(n):
    """Uniform points on a unit sphere (golden-angle spiral)."""
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


def build_sas_cavity(atoms, coords, points_per_atom=110):
    """Solvent-accessible surface: atom-centered spheres at (vdW + probe),
    then remove points that fall inside any neighbouring sphere."""
    sphere = fibonacci_sphere(points_per_atom)
    all_pts, all_norm = [], []
    radii = np.array([VDW_RADII.get(a, 1.7) + SOLVENT_PROBE for a in atoms])
    for i, (c, r) in enumerate(zip(coords, radii)):
        pts = c + sphere * r
        d2 = ((pts[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
        d2[:, i] = np.inf
        mask = np.all(d2 > (radii[None, :] - 1e-6) ** 2, axis=1)
        if mask.any():
            all_pts.append(pts[mask])
            all_norm.append(sphere[mask])
    return np.concatenate(all_pts), np.concatenate(all_norm)


def run_psi4(atoms, coords, charge, cavity_points):
    """Run B3LYP/def2-SVP with PCM and return ESP at cavity points (a.u.)."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.set_output_file("psi4_scratch.out", False)
    import os
    mem_gb = int(os.environ.get("PSI4_MEM_GB", "8"))
    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    psi4.set_memory(f"{mem_gb} GB")
    psi4.set_num_threads(n_threads)

    mult = multiplicity_for(atoms, charge)
    geom_lines = ["", f"{charge} {mult}"]
    for a, xyz in zip(atoms, coords):
        geom_lines.append(f"{a} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}")
    geom_lines.append("symmetry c1")
    geom_lines.append("no_reorient")
    geom_lines.append("no_com")
    mol = psi4.geometry("\n".join(geom_lines) + "\n")

    psi4.set_options({
        "basis": "def2-svp",
        "scf_type": "df",
        "df_basis_scf": "def2-universal-jkfit",
        "reference": "rks" if mult == 1 else "uks",
        "e_convergence": 1e-6,
        "d_convergence": 1e-5,
        "maxiter": 200,
        "pcm": True,
        "pcm_scf_type": "total",
    })
    psi4.pcm_helper("""
        Units = Angstrom
        Medium {
          SolverType = CPCM
          Solvent = Explicit
          ProbeRadius = 1.385
          Green<inside> {
            Type = Vacuum
          }
          Green<outside> {
            Type = UniformDielectric
            Eps = 15.0
            EpsDyn = 15.0
          }
        }
        Cavity {
          Type = GePol
          Area = 0.3
          RadiiSet = UFF
          Mode = Implicit
          Scaling = True
        }
    """)

    _, wfn = psi4.energy("b3lyp", return_wfn=True)

    grid_file = Path("grid.dat")
    with open(grid_file, "w") as f:
        for p in cavity_points:
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
    psi4.oeprop(wfn, "GRID_ESP", title="sas")
    esp = np.loadtxt("grid_esp.dat")
    grid_file.unlink(missing_ok=True)
    Path("grid_esp.dat").unlink(missing_ok=True)
    psi4.core.clean()
    return esp


def process(compound_id, xyz_path, out_path):
    atoms, coords = read_xyz(xyz_path)
    charge = charge_from_id(compound_id)
    pts, normals = build_sas_cavity(atoms, coords)
    esp = run_psi4(atoms, coords, charge, pts)
    surface = np.concatenate([pts, normals, esp[:, None]], axis=1).astype(np.float32)
    np.savez_compressed(out_path, surface=surface, atoms=np.array(atoms),
                        coords=coords.astype(np.float32), charge=charge)
    return surface.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compound-id", required=True)
    ap.add_argument("--geom-dir", default="data/pipeline/geometries")
    ap.add_argument("--out-dir", default="data/pipeline/dft_surface")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz = Path(args.geom_dir) / f"{args.compound_id}.xyz"
    out = out_dir / f"{args.compound_id}.npz"
    if out.exists():
        print(f"SKIP {args.compound_id}")
        return 0
    n = process(args.compound_id, xyz, out)
    print(f"OK {args.compound_id} ({n} surface points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
