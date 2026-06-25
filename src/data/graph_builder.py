"""Convert SMILES strings to molecular graph representations for GNN input.

Each molecule is represented as a graph where:
- Nodes = atoms with feature vectors (element, charge, hybridization, etc.)
- Edges = bonds with feature vectors (bond type, conjugation, ring membership)
"""

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("Warning: RDKit not installed. Graph features will not be available.")

# ── Atom feature definitions ───────────────────────────────────────────────

ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "H", "P", "Other"]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
] if HAS_RDKIT else []


def one_hot(value, choices):
    """One-hot encode a value given a list of possible choices."""
    encoding = [0] * (len(choices) + 1)  # +1 for unknown
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1  # unknown
    return encoding


def get_atom_features(atom) -> list:
    """Extract feature vector for a single atom.

    Features (total dim = 9 + hybridization + extras):
      - Element type (one-hot, 11 dim)
      - Degree (1 dim)
      - Formal charge (1 dim)
      - Num Hs (1 dim)
      - Is aromatic (1 dim)
      - Hybridization (one-hot, 6 dim)
      - Is in ring (1 dim)
    Total: 22 features
    """
    features = []

    # Element type one-hot
    symbol = atom.GetSymbol()
    features.extend(one_hot(symbol, ATOM_TYPES))  # 11

    # Degree
    features.append(atom.GetDegree())  # 1

    # Formal charge
    features.append(atom.GetFormalCharge())  # 1

    # Number of Hs
    features.append(atom.GetTotalNumHs())  # 1

    # Aromaticity
    features.append(int(atom.GetIsAromatic()))  # 1

    # Hybridization one-hot
    features.extend(one_hot(atom.GetHybridization(), HYBRIDIZATIONS))  # 6

    # In ring
    features.append(int(atom.IsInRing()))  # 1

    return features  # total = 22


def get_bond_features(bond) -> list:
    """Extract feature vector for a single bond.

    Features:
      - Bond type (one-hot: single, double, triple, aromatic) (5 dim)
      - Is conjugated (1 dim)
      - Is in ring (1 dim)
    Total: 7 features
    """
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]

    features = []
    features.extend(one_hot(bond.GetBondType(), bond_types))  # 5
    features.append(int(bond.GetIsConjugated()))  # 1
    features.append(int(bond.IsInRing()))  # 1

    return features  # total = 7


ATOM_FEATURE_DIM = 22
BOND_FEATURE_DIM = 7


def smiles_to_graph(smiles: str) -> dict:
    """Convert a SMILES string to a molecular graph.

    Returns dict with:
      - atom_features: np.ndarray (num_atoms, ATOM_FEATURE_DIM)
      - bond_features: np.ndarray (num_bonds*2, BOND_FEATURE_DIM) (bidirectional)
      - edge_index: np.ndarray (2, num_bonds*2) (COO format, bidirectional)
      - num_atoms: int
      - num_bonds: int
    """
    if not HAS_RDKIT:
        raise ImportError("RDKit is required for SMILES-to-graph conversion.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Add hydrogens for complete representation
    mol = Chem.AddHs(mol)

    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    atom_features = np.array(atom_features, dtype=np.float32)

    # Bond features and edge index (bidirectional)
    edge_index = []
    bond_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = get_bond_features(bond)

        # Add both directions
        edge_index.append([i, j])
        edge_index.append([j, i])
        bond_features.append(bf)
        bond_features.append(bf)

    if len(edge_index) > 0:
        edge_index = np.array(edge_index, dtype=np.int64).T  # (2, num_edges)
        bond_features = np.array(bond_features, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        bond_features = np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32)

    return {
        "atom_features": atom_features,
        "bond_features": bond_features,
        "edge_index": edge_index,
        "num_atoms": len(atom_features),
        "num_bonds": len(bond_features) // 2,
    }


def smiles_to_separate_graphs(cation_smiles: str, anion_smiles: str) -> dict:
    """Convert cation and anion SMILES to separate graphs.

    Returns dict with cation_graph and anion_graph, each as from smiles_to_graph().
    """
    cation_graph = smiles_to_graph(cation_smiles)
    anion_graph = smiles_to_graph(anion_smiles)
    return {"cation": cation_graph, "anion": anion_graph}


def randomize_smiles(smiles: str) -> str:
    """Generate a random but valid SMILES representation of the same molecule.

    Uses RDKit's random SMILES generation for data augmentation.
    Returns the original SMILES if randomization fails.
    """
    if not HAS_RDKIT:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, doRandom=True)
    except Exception:
        return smiles


def smiles_to_combined_graph(il_smiles: str) -> dict:
    """Convert full IL SMILES (cation.anion) to a single combined graph.

    The cation and anion are in the same graph but as disconnected components.
    """
    return smiles_to_graph(il_smiles)


def precompute_graphs(df, output_path: str = None) -> dict:
    """Precompute molecular graphs for all unique ILs in the DataFrame.

    Returns a dict mapping il_short_name -> graph_data.
    """
    import pickle

    graphs = {}
    unique_ils = df.drop_duplicates(subset=["il_short_name"])

    for _, row in unique_ils.iterrows():
        name = row["il_short_name"]
        try:
            # Combined graph (cation + anion as disconnected components)
            combined = smiles_to_combined_graph(row["smiles"])
            # Separate graphs
            separate = smiles_to_separate_graphs(
                row["cation_smiles"], row["anion_smiles"]
            )
            graphs[name] = {
                "combined": combined,
                "cation": separate["cation"],
                "anion": separate["anion"],
            }
        except Exception as e:
            print(f"Warning: Failed to build graph for {name}: {e}")
            graphs[name] = None

    if output_path:
        with open(output_path, "wb") as f:
            pickle.dump(graphs, f)
        print(f"Saved {len(graphs)} molecular graphs to {output_path}")

    return graphs


if __name__ == "__main__":
    # Test with a sample IL
    test_smiles = "C=CC[n+]1ccn(C)c1.[Cl-]"
    print(f"Testing SMILES: {test_smiles}")
    graph = smiles_to_graph(test_smiles)
    print(f"  Atoms: {graph['num_atoms']}, Bonds: {graph['num_bonds']}")
    print(f"  Atom feature shape: {graph['atom_features'].shape}")
    print(f"  Edge index shape: {graph['edge_index'].shape}")
    print(f"  Bond feature shape: {graph['bond_features'].shape}")

    # Test separate cation/anion
    cation = "C=CC[n+]1ccn(C)c1"
    anion = "[Cl-]"
    sep = smiles_to_separate_graphs(cation, anion)
    print(f"\nCation atoms: {sep['cation']['num_atoms']}, Anion atoms: {sep['anion']['num_atoms']}")
