"""Contrastive GNN pre-training on ILThermo molecular graphs.

Self-supervised learning: ILs sharing the same cation or anion form
positive pairs. NT-Xent (InfoNCE) loss learns molecular representations
without needing property labels.

Stage 1: Contrastive pre-train on 143 unique ILThermo SMILES
Stage 2: Fine-tune on 223 original samples (all 7 targets)
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import TARGET_COLUMNS, FEATURE_COLUMNS
from src.data.graph_builder import smiles_to_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.graph.gnn import MolecularGNN
from src.data.dataset import ILMultimodalDataset, collate_multimodal
from src.training.trainer import Trainer


class ContrastiveGraphDataset(Dataset):
    """Dataset that yields pairs of molecular graphs for contrastive learning."""

    def __init__(self, smiles_list):
        self.smiles_list = smiles_list
        self.graphs = {}
        failed = 0
        for smi in smiles_list:
            try:
                self.graphs[smi] = smiles_to_graph(smi)
            except Exception:
                self.graphs[smi] = None
                failed += 1
        self.valid_smiles = [s for s in smiles_list if self.graphs[s] is not None]
        print(f"  Contrastive dataset: {len(self.valid_smiles)} valid graphs ({failed} failed)")

    def __len__(self):
        return len(self.valid_smiles)

    def __getitem__(self, idx):
        smi = self.valid_smiles[idx]
        g = self.graphs[smi]
        return {
            "atom_features": torch.tensor(g["atom_features"], dtype=torch.float32),
            "edge_index": torch.tensor(g["edge_index"], dtype=torch.long),
            "bond_features": torch.tensor(g["bond_features"], dtype=torch.float32),
            "num_atoms": len(g["atom_features"]),
            "smiles": smi,
        }


def collate_contrastive(batch):
    """Collate graphs into a batch."""
    atom_features_list = []
    edge_index_list = []
    bond_features_list = []
    graph_batch = []
    atom_offset = 0
    smiles_list = []

    for i, b in enumerate(batch):
        n = b["atom_features"].shape[0]
        atom_features_list.append(b["atom_features"])
        graph_batch.extend([i] * n)
        ei = b["edge_index"].clone()
        if ei.shape[1] > 0:
            ei += atom_offset
        edge_index_list.append(ei)
        bond_features_list.append(b["bond_features"])
        atom_offset += n
        smiles_list.append(b["smiles"])

    return {
        "atom_features": torch.cat(atom_features_list),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "bond_features": torch.cat(bond_features_list),
        "batch": torch.tensor(graph_batch, dtype=torch.long),
        "smiles": smiles_list,
    }


def build_similarity_matrix(smiles_list):
    """Build positive pair matrix: 1 if same cation or anion, 0 otherwise."""
    n = len(smiles_list)
    # Split each SMILES into fragments (cation.anion)
    fragments = []
    for smi in smiles_list:
        parts = smi.split(".")
        fragments.append(set(parts))

    sim = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            # Positive if they share any fragment (cation or anion)
            if fragments[i] & fragments[j]:
                sim[i, j] = 1.0
                sim[j, i] = 1.0
    return sim


class NTXentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, positive_mask):
        """
        Args:
            features: (B, D) L2-normalized embeddings
            positive_mask: (B, B) binary mask of positive pairs
        """
        sim = torch.matmul(features, features.T) / self.temperature
        # Mask out self-similarity
        mask_self = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        sim = sim * mask_self.float() - 1e9 * (~mask_self).float()

        # For each anchor, positive pairs are in positive_mask
        # Use InfoNCE: -log(exp(sim_pos) / sum(exp(sim_all)))
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        exp_sim = torch.exp(sim) * mask_self.float()
        denom = exp_sim.sum(dim=1, keepdim=True)

        pos_sim = sim * positive_mask
        pos_exp = torch.exp(pos_sim) * positive_mask

        # Average log probability of positive pairs
        has_pos = positive_mask.sum(dim=1) > 0
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        log_prob = torch.log(pos_exp.sum(dim=1) / denom.squeeze() + 1e-10)
        loss = -log_prob[has_pos].mean()
        return loss


def contrastive_pretrain(model, dataset, device, num_epochs=50, lr=1e-3):
    """Pre-train GNN with contrastive loss."""
    projector = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    ).to(device)

    criterion = NTXentLoss(temperature=0.1).to(device)
    params = list(model.parameters()) + list(projector.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_contrastive)

    print(f"\n{'='*60}")
    print(f"CONTRASTIVE PRE-TRAINING ({len(dataset)} graphs)")
    print(f"{'='*60}")

    best_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        projector.train()
        total_loss = 0
        n = 0

        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Get graph representations
            features = model.get_features(
                batch["atom_features"], batch["edge_index"],
                batch["bond_features"], batch["batch"]
            )

            # Project to contrastive space
            z = projector(features)
            z = F.normalize(z, dim=1)

            # Build positive pair mask
            pos_mask = build_similarity_matrix(batch["smiles"]).to(device)

            loss = criterion(z, pos_mask)
            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n += 1

        scheduler.step()
        avg_loss = total_loss / max(n, 1)

        if avg_loss < best_loss and avg_loss > 0:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/contrastive/pretrained.pt")

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{num_epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")

    # Load best
    ckpt_path = Path("checkpoints/contrastive/pretrained.pt")
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"  Contrastive pre-training complete. Best loss: {best_loss:.4f}")
    return model


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    Path("checkpoints/contrastive").mkdir(parents=True, exist_ok=True)

    # Build GNN
    mc = config.get("model", {}).get("graph", {})
    model = MolecularGNN(
        atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
        hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
        dropout=0.3, pooling="mean", num_targets=7,
        aux_feature_dim=len(FEATURE_COLUMNS),
    )
    model.to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Stage 1: Contrastive pre-training on all unique SMILES
    all_smiles = set()
    for csv_path in ["data/augmented/ilthermo_data.csv", "data/processed/il_data_raw.csv"]:
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            all_smiles.update(df["smiles"].unique())
    all_smiles = sorted(all_smiles)
    print(f"Total unique SMILES for contrastive: {len(all_smiles)}")

    dataset = ContrastiveGraphDataset(all_smiles)
    model = contrastive_pretrain(model, dataset, device, num_epochs=50, lr=1e-3)

    # Stage 2: Fine-tune on original dataset
    print(f"\n{'='*60}")
    print("FINE-TUNING on original dataset")
    print(f"{'='*60}")

    config["training"]["num_epochs"] = 200
    config["training"]["learning_rate"] = 5e-5
    config["training"]["early_stopping_patience"] = 25
    config["experiment"]["checkpoint_dir"] = "checkpoints/contrastive/finetune"

    splits_dir = Path("data/processed/splits")
    graph_cache = "data/processed/graphs.pkl"
    graph_path = graph_cache if Path(graph_cache).exists() else None

    train_ds = ILMultimodalDataset(str(splits_dir / "train.csv"), graph_path, is_train=True, config=config)
    val_ds = ILMultimodalDataset(str(splits_dir / "val.csv"), graph_path, is_train=False, config=config)
    test_ds = ILMultimodalDataset(str(splits_dir / "test.csv"), graph_path, is_train=False, config=config)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_multimodal)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_multimodal)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_multimodal)

    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                      test_loader=test_loader, config=config, device=device)
    history = trainer.train(verbose=True)

    results = {"model": "contrastive_gnn", "best_val_loss": trainer.best_val_loss,
               "epochs_trained": len(history["train_loss"])}
    if "test_metrics" in history:
        results["test_metrics"] = {k: float(v) if not (isinstance(v, float) and np.isnan(v)) else None
                                   for k, v in history["test_metrics"].items()}

    with open("results/contrastive_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/contrastive_results.json")


if __name__ == "__main__":
    main()
