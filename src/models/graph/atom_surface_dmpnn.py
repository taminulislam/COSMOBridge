"""Shared-Atom Surface Encoding: COSMO surface patches as atom features in D-MPNN.

Novel approach: Instead of encoding the whole COSMO surface as one global vector
(PointNet → 256D), we assign surface patches to individual atoms and compute
per-atom local surface descriptors. These are concatenated with standard atom
features and fed into D-MPNN, inheriting its parameter-sharing inductive bias.

Each atom gets 10 local surface features:
  - n_surface_points (normalized)
  - local_esp_mean, local_esp_std, local_esp_min, local_esp_max
  - local_esp_pos_frac, local_esp_neg_frac
  - local_normal_variance (curvature proxy)
  - local_area_fraction
  - local_esp_range

This adds only 10 features per atom (vs 256D global PointNet vector),
keeping the model lightweight while giving each atom rich surface information.
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist
from rdkit import Chem
from rdkit.Chem import AllChem

N_SURFACE_FEATURES = 10


def compute_atom_surface_features(smiles, point_cloud):
    """Compute per-atom surface descriptors from COSMO point cloud.

    Args:
        smiles: SMILES string
        point_cloud: (N, 7) array — xyz + normals + ESP

    Returns:
        atom_surface_feats: (num_atoms, 10) per-atom surface features
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    success = AllChem.EmbedMolecule(mol, randomSeed=42)
    if success == -1:
        AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except:
        pass
    # Keep Hs to match graph builder's atom count
    num_atoms = mol.GetNumAtoms()
    conf = mol.GetConformer()
    atom_pos = np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y,
                           conf.GetAtomPosition(i).z] for i in range(num_atoms)])

    pc_xyz = point_cloud[:, :3]
    pc_normals = point_cloud[:, 3:6]
    pc_esp = point_cloud[:, 6]

    # Align coordinate systems: center and scale
    atom_centered = atom_pos - atom_pos.mean(axis=0)
    pc_centered = pc_xyz - pc_xyz.mean(axis=0)

    atom_scale = max(np.abs(atom_centered).max(), 1e-6)
    pc_scale = max(np.abs(pc_centered).max(), 1e-6)
    atom_scaled = atom_centered * (pc_scale / atom_scale)

    # Assign each surface point to nearest atom
    dists = cdist(atom_scaled, pc_centered)
    nearest_atom = dists.argmin(axis=0)

    # Compute per-atom features
    total_pts = len(pc_esp)
    features = np.zeros((num_atoms, N_SURFACE_FEATURES))

    for i in range(num_atoms):
        mask = nearest_atom == i
        n_pts = mask.sum()

        if n_pts == 0:
            # No surface points assigned — leave as zeros
            continue

        local_esp = pc_esp[mask]
        local_norms = pc_normals[mask]

        features[i, 0] = n_pts / total_pts                    # area fraction
        features[i, 1] = local_esp.mean()                     # ESP mean
        features[i, 2] = local_esp.std()                      # ESP std
        features[i, 3] = local_esp.min()                      # ESP min
        features[i, 4] = local_esp.max()                      # ESP max
        features[i, 5] = (local_esp > 0).mean()               # ESP positive fraction
        features[i, 6] = (local_esp < 0).mean()               # ESP negative fraction
        features[i, 7] = local_norms.var(axis=0).sum()        # normal variance (curvature)
        features[i, 8] = local_esp.max() - local_esp.min()    # ESP range
        features[i, 9] = np.abs(local_esp).mean()             # absolute ESP mean

    return features
