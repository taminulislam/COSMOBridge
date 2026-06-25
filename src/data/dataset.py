"""PyTorch Dataset classes for multimodal ionic liquid data.

Provides:
  - ILTabularDataset: tabular features only
  - ILImageDataset: images only (COSMO + EP)
  - ILMultimodalDataset: all modalities (tabular + images + graphs)
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.preprocessing import TARGET_COLUMNS, FEATURE_COLUMNS
from src.data.image_loader import load_image, get_train_transforms, get_eval_transforms
from src.data.graph_builder import smiles_to_combined_graph, randomize_smiles, smiles_to_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM


class ILTabularDataset(Dataset):
    """Dataset for tabular features only (temperature, composition, IL identity)."""

    def __init__(self, csv_path: str, num_ils: int = 28, num_cations: int = 9, num_anions: int = 7):
        self.df = pd.read_csv(csv_path)
        self.num_ils = num_ils
        self.num_cations = num_cations
        self.num_anions = num_anions

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Numerical features
        features = torch.tensor(
            [row[col] for col in FEATURE_COLUMNS],
            dtype=torch.float32,
        )

        # Categorical indices
        il_idx = torch.tensor(int(row["il_idx"]), dtype=torch.long)
        cation_idx = torch.tensor(int(row["cation_idx"]), dtype=torch.long)
        anion_idx = torch.tensor(int(row["anion_idx"]), dtype=torch.long)

        # Targets
        targets = torch.tensor(
            [row[col] for col in TARGET_COLUMNS],
            dtype=torch.float32,
        )

        return {
            "features": features,
            "il_idx": il_idx,
            "cation_idx": cation_idx,
            "anion_idx": anion_idx,
            "targets": targets,
        }


class ILMultimodalDataset(Dataset):
    """Full multimodal dataset: tabular + images + molecular graphs.

    Each sample provides:
      - Tabular: numerical features + categorical indices
      - Images: COSMO surface tensor + EP surface tensor
      - Graph: atom features, edge index, bond features (precomputed)
      - Targets: 7 thermodynamic properties
    """

    def __init__(
        self,
        csv_path: str,
        graph_cache_path: str = None,
        image_transform=None,
        config: dict = None,
        is_train: bool = True,
        smiles_augment: bool = False,
    ):
        self.df = pd.read_csv(csv_path)
        self.config = config or {}
        self.is_train = is_train
        self.smiles_augment = smiles_augment and is_train

        # Image transforms
        img_size = self.config.get("image", {}).get("size", 224)
        if image_transform:
            self.transform = image_transform
        elif is_train:
            self.transform = get_train_transforms(img_size, config)
        else:
            self.transform = get_eval_transforms(img_size, config)

        # Load precomputed graphs
        self.graphs = {}
        if graph_cache_path and Path(graph_cache_path).exists():
            with open(graph_cache_path, "rb") as f:
                self.graphs = pickle.load(f)
        else:
            # Build graphs on the fly from SMILES
            self._precompute_graphs()

    def _precompute_graphs(self):
        """Build molecular graphs for all unique ILs."""
        unique = self.df.drop_duplicates(subset=["il_short_name"])
        for _, row in unique.iterrows():
            name = row["il_short_name"]
            try:
                self.graphs[name] = {
                    "combined": smiles_to_combined_graph(row["smiles"]),
                }
            except Exception as e:
                print(f"Warning: Graph build failed for {name}: {e}")
                self.graphs[name] = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Tabular features ──
        features = torch.tensor(
            [row[col] for col in FEATURE_COLUMNS],
            dtype=torch.float32,
        )
        il_idx = torch.tensor(int(row["il_idx"]), dtype=torch.long)
        cation_idx = torch.tensor(int(row["cation_idx"]), dtype=torch.long)
        anion_idx = torch.tensor(int(row["anion_idx"]), dtype=torch.long)

        # ── Images ──
        cosmo_path = row.get("cosmo_image_path")
        ep_path = row.get("ep_image_path")

        cosmo_img = load_image(
            cosmo_path if pd.notna(cosmo_path) else None,
            self.transform,
        )
        ep_img = load_image(
            ep_path if pd.notna(ep_path) else None,
            self.transform,
        )

        # ── Graph ──
        il_name = row["il_short_name"]

        # SMILES augmentation: rebuild graph from a randomized SMILES during training
        if self.smiles_augment and "smiles" in row.index:
            try:
                rand_smiles = randomize_smiles(row["smiles"])
                g = smiles_to_graph(rand_smiles)
                atom_features = torch.tensor(g["atom_features"], dtype=torch.float32)
                edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
                bond_features = torch.tensor(g["bond_features"], dtype=torch.float32)
            except Exception:
                # Fall back to cached graph
                graph_data = self.graphs.get(il_name)
                if graph_data and graph_data.get("combined"):
                    g = graph_data["combined"]
                    atom_features = torch.tensor(g["atom_features"], dtype=torch.float32)
                    edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
                    bond_features = torch.tensor(g["bond_features"], dtype=torch.float32)
                else:
                    atom_features = torch.zeros(1, ATOM_FEATURE_DIM)
                    edge_index = torch.zeros(2, 0, dtype=torch.long)
                    bond_features = torch.zeros(0, BOND_FEATURE_DIM)
        else:
            graph_data = self.graphs.get(il_name)
            if graph_data and graph_data.get("combined"):
                g = graph_data["combined"]
                atom_features = torch.tensor(g["atom_features"], dtype=torch.float32)
                edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
                bond_features = torch.tensor(g["bond_features"], dtype=torch.float32)
            else:
                atom_features = torch.zeros(1, ATOM_FEATURE_DIM)
                edge_index = torch.zeros(2, 0, dtype=torch.long)
                bond_features = torch.zeros(0, BOND_FEATURE_DIM)

        # ── Targets ──
        targets = torch.tensor(
            [row[col] for col in TARGET_COLUMNS],
            dtype=torch.float32,
        )

        return {
            "features": features,
            "il_idx": il_idx,
            "cation_idx": cation_idx,
            "anion_idx": anion_idx,
            "cosmo_image": cosmo_img,
            "ep_image": ep_img,
            "atom_features": atom_features,
            "edge_index": edge_index,
            "bond_features": bond_features,
            "num_atoms": atom_features.shape[0],
            "targets": targets,
            "il_name": il_name,
        }


def collate_multimodal(batch: list) -> dict:
    """Custom collate function for multimodal batches.

    Handles variable-size molecular graphs by padding and creating batch indices.
    """
    # Stack fixed-size tensors
    features = torch.stack([b["features"] for b in batch])
    il_idx = torch.stack([b["il_idx"] for b in batch])
    cation_idx = torch.stack([b["cation_idx"] for b in batch])
    anion_idx = torch.stack([b["anion_idx"] for b in batch])
    cosmo_images = torch.stack([b["cosmo_image"] for b in batch])
    ep_images = torch.stack([b["ep_image"] for b in batch])
    targets = torch.stack([b["targets"] for b in batch])

    # Concatenate variable-size graphs with batch index
    atom_features_list = []
    edge_index_list = []
    bond_features_list = []
    graph_batch = []  # maps each atom to its sample index

    atom_offset = 0
    for i, b in enumerate(batch):
        n_atoms = b["atom_features"].shape[0]

        atom_features_list.append(b["atom_features"])
        graph_batch.extend([i] * n_atoms)

        # Offset edge indices
        ei = b["edge_index"].clone()
        if ei.shape[1] > 0:
            ei += atom_offset
        edge_index_list.append(ei)
        bond_features_list.append(b["bond_features"])

        atom_offset += n_atoms

    atom_features = torch.cat(atom_features_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros(2, 0, dtype=torch.long)
    bond_features = torch.cat(bond_features_list, dim=0) if bond_features_list else torch.zeros(0, BOND_FEATURE_DIM)
    graph_batch = torch.tensor(graph_batch, dtype=torch.long)

    return {
        "features": features,
        "il_idx": il_idx,
        "cation_idx": cation_idx,
        "anion_idx": anion_idx,
        "cosmo_image": cosmo_images,
        "ep_image": ep_images,
        "atom_features": atom_features,
        "edge_index": edge_index,
        "bond_features": bond_features,
        "graph_batch": graph_batch,
        "targets": targets,
    }
