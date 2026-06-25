"""Shared-Atom Surface D-MPNN: per-atom COSMO features in Chemprop-style D-MPNN.

Novel architecture: COSMO surface patches assigned to atoms → augmented atom
features → D-MPNN with parameter sharing across atoms.

Combines:
- Chemprop's efficient D-MPNN (parameter sharing, directed messages)
- Our COSMO surface information (per-atom ESP, curvature, area)
- Thermodynamic features concatenated after graph encoding

Atom features: 22 (standard) + 10 (local surface) = 32D per atom
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
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.data.graph_builder import smiles_to_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.graph.dmpnn import DirectedMPNN
from src.models.graph.atom_surface_dmpnn import compute_atom_surface_features, N_SURFACE_FEATURES
from src.training.metrics import compute_metrics, format_metrics

AUGMENTED_ATOM_DIM = ATOM_FEATURE_DIM + N_SURFACE_FEATURES  # 22 + 10 = 32


class AtomSurfaceDataset(Dataset):
    """Dataset that augments atom features with per-atom COSMO surface descriptors."""

    def __init__(self, csv_path, pc_dir, is_train=True, n_points=1024):
        self.df = pd.read_csv(csv_path)
        self.pc_dir = Path(pc_dir)
        self.is_train = is_train
        self.n_points = n_points

        # Load point cloud index
        idx_path = self.pc_dir / "index.csv"
        self.pc_index = {}
        if idx_path.exists():
            idx_df = pd.read_csv(idx_path)
            self.pc_index = dict(zip(idx_df["smiles"], idx_df["filename"]))

        # Pre-build graphs with surface-augmented atom features
        self.graphs = {}
        self.surface_feats = {}
        n_with_surface = 0

        for smi in self.df["smiles"].unique():
            try:
                g = smiles_to_graph(smi)
                self.graphs[smi] = g
            except:
                self.graphs[smi] = None
                continue

            # Compute per-atom surface features
            fn = self.pc_index.get(smi)
            if fn and (self.pc_dir / fn).exists():
                pc = np.load(self.pc_dir / fn)["points"]
                surf_feats = compute_atom_surface_features(smi, pc)
                if surf_feats is not None and surf_feats.shape[0] == g["atom_features"].shape[0]:
                    self.surface_feats[smi] = surf_feats
                    n_with_surface += 1
                else:
                    # Atom count mismatch — use zeros
                    self.surface_feats[smi] = np.zeros((g["atom_features"].shape[0], N_SURFACE_FEATURES))
            else:
                self.surface_feats[smi] = np.zeros((g["atom_features"].shape[0], N_SURFACE_FEATURES))

        print(f"  {csv_path}: {len(self.df)} rows, "
              f"{sum(1 for g in self.graphs.values() if g is not None)} graphs, "
              f"{n_with_surface} with surface features")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smi = row["smiles"]

        features = torch.tensor([row[c] for c in FEATURE_COLUMNS], dtype=torch.float32)

        g = self.graphs.get(smi)
        if g is not None:
            af = np.array(g["atom_features"], dtype=np.float32)
            # Augment atom features with surface descriptors
            sf = self.surface_feats.get(smi, np.zeros((af.shape[0], N_SURFACE_FEATURES)))
            augmented_af = np.concatenate([af, sf], axis=1)  # (num_atoms, 32)

            atom_features = torch.tensor(augmented_af, dtype=torch.float32)
            edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
            bond_features = torch.tensor(g["bond_features"], dtype=torch.float32)
        else:
            atom_features = torch.zeros(1, AUGMENTED_ATOM_DIM)
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            bond_features = torch.zeros(0, BOND_FEATURE_DIM)

        targets = torch.tensor([row[c] for c in TARGET_COLUMNS], dtype=torch.float32)

        return {
            "features": features,
            "atom_features": atom_features,
            "edge_index": edge_index,
            "bond_features": bond_features,
            "num_atoms": atom_features.shape[0],
            "targets": targets,
        }


def collate_atom_surface(batch):
    feats = torch.stack([b["features"] for b in batch])
    targets = torch.stack([b["targets"] for b in batch])

    af_list, ei_list, bf_list, gb = [], [], [], []
    offset = 0
    for i, b in enumerate(batch):
        n = b["atom_features"].shape[0]
        af_list.append(b["atom_features"])
        gb.extend([i] * n)
        ei = b["edge_index"].clone()
        if ei.shape[1] > 0:
            ei += offset
        ei_list.append(ei)
        bf_list.append(b["bond_features"])
        offset += n

    return {
        "features": feats,
        "targets": targets,
        "atom_features": torch.cat(af_list),
        "edge_index": torch.cat(ei_list, dim=1),
        "bond_features": torch.cat(bf_list),
        "batch": torch.tensor(gb, dtype=torch.long),
    }


class AtomSurfaceDMPNN(nn.Module):
    """D-MPNN with per-atom COSMO surface features.

    Atom features (22D standard + 10D surface = 32D) → D-MPNN → graph repr
    → concat with thermo features → FFN → 7 targets
    """

    def __init__(self, feature_dim=25, hidden=300, depth=3, dropout=0.2):
        super().__init__()
        self.dmpnn = DirectedMPNN(
            atom_feature_dim=AUGMENTED_ATOM_DIM,  # 32D instead of 22D
            bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=hidden, num_layers=depth, dropout=dropout, num_targets=0)

        # FFN: graph repr + thermo features → predictions
        ffn_input = hidden + feature_dim
        self.ffn = nn.Sequential(
            nn.Linear(ffn_input, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 7),
        )

    def forward(self, features, atom_features, edge_index, bond_features,
                batch, **kwargs):
        graph_repr = self.dmpnn.get_features(atom_features, edge_index,
                                              bond_features, batch)
        combined = torch.cat([graph_repr, features], dim=-1)
        return self.ffn(combined)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            preds = model(features=batch["features"],
                          atom_features=batch["atom_features"],
                          edge_index=batch["edge_index"],
                          bond_features=batch["bond_features"],
                          batch=batch["batch"])
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    pc_dir = "data/pipeline/point_clouds"
    orig_splits = Path("data/processed/splits")

    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Shared-Atom Surface D-MPNN")
    print(f"Atom features: {ATOM_FEATURE_DIM} standard + {N_SURFACE_FEATURES} surface = {AUGMENTED_ATOM_DIM}D")
    print(f"{'='*60}")

    print("\nLoading datasets...")
    train_ds = AtomSurfaceDataset(str(orig_splits / "train.csv"), pc_dir, is_train=True)
    val_ds = AtomSurfaceDataset(str(orig_splits / "val.csv"), pc_dir, is_train=False)
    test_ds = AtomSurfaceDataset(str(orig_splits / "test.csv"), pc_dir, is_train=False)

    train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_atom_surface)
    val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_atom_surface)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_atom_surface)

    model = AtomSurfaceDMPNN(feature_dim=len(FEATURE_COLUMNS), hidden=300, depth=3, dropout=0.2)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,} (comparable to Chemprop ~300K)")

    ckpt_dir = Path("checkpoints/atom_surface_dmpnn")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=60, eta_min=1e-6)

    best_loss, no_improve = float("inf"), 0
    for epoch in range(200):
        model.train()
        tl, n = 0, 0
        for batch in train_ldr:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            optimizer.zero_grad()
            preds = model(features=batch["features"],
                          atom_features=batch["atom_features"],
                          edge_index=batch["edge_index"],
                          bond_features=batch["bond_features"],
                          batch=batch["batch"])
            loss = ((preds - batch["targets"])**2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item(); n += 1
        scheduler.step()

        model.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for batch in val_ldr:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                preds = model(features=batch["features"],
                              atom_features=batch["atom_features"],
                              edge_index=batch["edge_index"],
                              bond_features=batch["bond_features"],
                              batch=batch["batch"])
                vl += ((preds - batch["targets"])**2).mean().item()
                vn += 1
        avg_val = vl / max(vn, 1)
        if avg_val < best_loss:
            best_loss = avg_val; no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_improve += 1
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/200 | Train: {tl/max(n,1):.4f} | "
                  f"Val: {avg_val:.4f} | Best: {best_loss:.4f} | Pat: {no_improve}/25")
        if no_improve >= 25:
            print(f"  Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True))

    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")

    preds, targets = evaluate(model, test_ldr, device)
    metrics = compute_metrics(preds, targets)
    print(format_metrics(metrics, "Atom-Surface D-MPNN"))

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    prev = {}
    for name, path, key in [
        ("Chemprop", "results/chemprop_results.json", "test_metrics"),
        ("PointCloud", "results/pointcloud_results.json", None),
        ("MoE Fix6", "results/moe_fix6_results.json", "metrics"),
        ("MoE+DMPNN", "results/enhanced_models_results.json", "EMOE"),
    ]:
        try:
            data = json.load(open(path))
            if key == "EMOE":
                m = data["enhanced_moe"]["metrics"]
            elif key:
                m = data[key]
            else:
                for k in ['metrics', 'test_metrics']:
                    if k in data: m = data[k]; break
            prev[name] = m
        except:
            pass

    header = "  {:<12s}".format("Property")
    for name in prev:
        header += " {:>11s}".format(name)
    header += " {:>14s}".format("AtomSurf-DMPNN")
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<12s}".format(p)
        for name in prev:
            line += " {:11.4f}".format(prev[name].get(key, float('nan')))
        line += " {:14.4f}".format(metrics[key])
        print(line)

    line = "  {:<12s}".format("AVERAGE")
    for name in prev:
        line += " {:11.4f}".format(prev[name].get('avg_r2', float('nan')))
    line += " {:14.4f}".format(metrics['avg_r2'])
    print(line)

    print(f"\n  Params: {n_params:,} (Chemprop: ~300K, PointCloud: 3.7M)")

    # Save
    results = {
        "model": "atom_surface_dmpnn",
        "description": "D-MPNN with per-atom COSMO surface features (22+10=32D atom features). "
                       "Novel: surface patches assigned to atoms via spatial nearest-neighbor, "
                       "inheriting D-MPNN's parameter-sharing inductive bias.",
        "n_params": n_params,
        "atom_feature_dim": AUGMENTED_ATOM_DIM,
        "surface_features": N_SURFACE_FEATURES,
        "test_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                         for k, v in metrics.items()},
    }
    with open("results/atom_surface_dmpnn_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/atom_surface_dmpnn_results.json")


if __name__ == "__main__":
    main()
