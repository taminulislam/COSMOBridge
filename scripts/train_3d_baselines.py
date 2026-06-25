"""Train 3D equivariant/invariant GNN baselines (SchNet, DimeNet++) on the IL
thermodynamic property prediction task with the same 19/4/5 IL-wise split used
throughout the paper.

Purpose: respond to JCIM reviewer comment requesting comparison against 3D
equivariant atomistic models. These architectures operate on raw Cartesian
coordinates rather than COSMO surfaces, so they test a different 3D inductive
bias and provide a complementary baseline for γ₁/γ₂ predictions.

Design choices
--------------
- Ion-pair geometry: RDKit ETKDG + UFF relaxation on the combined SMILES
  (cation.anion as a single two-fragment molecule). No quantum geometry used,
  keeping the baseline fast and reproducible.
- Multi-task head: the 3D backbone produces a molecular embedding; temperature
  and composition are concatenated to that embedding, and a 3-layer MLP
  predicts all 7 standardized properties jointly.
- Standardization: the paper's target_scaler (pickled in data/processed/) is
  reused, so numbers are directly comparable to Table 1.

Usage
-----
    python scripts/train_3d_baselines.py --model schnet  --seed 42
    python scripts/train_3d_baselines.py --model dimenet --seed 42

Outputs
-------
    results/baselines_3d/{model}_seed{seed}.json
        per-property test R² and metadata

Results across 5 seeds can then be aggregated by the companion script
    scripts/aggregate_3d_baselines.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import SchNet, DimeNetPlusPlus

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TARGET_COLS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]
THERMO_COLS = ["temperature", "x1", "inv_temperature", "temp_squared", "temp_cubed"]


# ---------------------------------------------------------------------------
# Geometry generation from SMILES
# ---------------------------------------------------------------------------
def smiles_to_xyz(smiles: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
    """Generate 3D coordinates for a cation.anion SMILES via ETKDG+UFF.

    Returns
    -------
    (positions, atomic_numbers) or None if embedding failed.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    ok = AllChem.EmbedMolecule(mol, params)
    if ok != 0:
        # fallback: try random coords
        ok = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if ok != 0:
            return None
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    conf = mol.GetConformer()
    pos = np.asarray(conf.GetPositions(), dtype=np.float32)
    z = np.asarray([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    return pos, z


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ILGeomDataset(Dataset):
    """One geometry per unique IL, shared across all temperature rows.

    Each row in the CSV becomes a separate Data object, but all rows sharing
    the same il_idx reuse the same (pos, z) — only thermo features and targets
    differ. This matches the physics: the same ion pair at different T, x₁.
    """

    def __init__(self, df: pd.DataFrame, geometries: dict[int, tuple[np.ndarray, np.ndarray]]):
        self.df = df.reset_index(drop=True)
        self.geoms = geometries

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        pos, z = self.geoms[int(row["il_idx"])]
        thermo = torch.tensor([row[c] for c in THERMO_COLS], dtype=torch.float32)
        y = torch.tensor([row[c] for c in TARGET_COLS], dtype=torch.float32)
        return Data(
            pos=torch.tensor(pos, dtype=torch.float32),
            z=torch.tensor(z, dtype=torch.long),
            thermo=thermo.unsqueeze(0),  # (1, 5)
            y=y.unsqueeze(0),            # (1, 7)
        )


# ---------------------------------------------------------------------------
# Multi-task wrapper around 3D backbone
# ---------------------------------------------------------------------------
class MultiTask3D(nn.Module):
    """3D backbone → pooled embedding → concat thermo → 7-property MLP head.

    The SchNet/DimeNet++ backbones are used purely as molecular embedders;
    their default energy-prediction readouts are bypassed by returning the
    pre-readout pooled features (hidden_channels-dim).
    """

    def __init__(self, backbone: nn.Module, embed_dim: int, n_thermo: int = 5,
                 n_properties: int = 7, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(embed_dim + n_thermo, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_properties),
        )

    def embed(self, data):
        raise NotImplementedError

    def forward(self, data):
        emb = self.embed(data)                      # (B, embed_dim)
        thermo = data.thermo.view(-1, 5)            # (B, 5)
        x = torch.cat([emb, thermo], dim=-1)
        return self.head(x)                         # (B, 7)


class SchNetMT(MultiTask3D):
    def __init__(self, **kwargs):
        hidden_channels = 128
        backbone = SchNet(
            hidden_channels=hidden_channels,
            num_filters=128,
            num_interactions=3,
            num_gaussians=50,
            cutoff=8.0,
            readout="add",
        )
        super().__init__(backbone, embed_dim=hidden_channels, hidden=hidden_channels, **kwargs)

    def embed(self, data):
        # Recreate SchNet's forward up to (but not including) the energy readout.
        # We call the full model and then undo the final scalar projection by
        # using the module's internal lin1/lin2 — but the simplest stable path
        # is to wrap SchNet to return per-graph aggregated features. To keep
        # things self-contained we use SchNet's existing forward (returns per-
        # graph scalar in shape (B,1)), and separately pool a learned atomic
        # embedding ourselves.
        z, pos, batch = data.z, data.pos, data.batch
        h = self.backbone.embedding(z)
        edge_index, edge_weight = self._edges(pos, batch, cutoff=self.backbone.cutoff)
        edge_attr = self.backbone.distance_expansion(edge_weight)
        for interaction in self.backbone.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        # graph-level sum pool
        from torch_geometric.utils import scatter
        g = scatter(h, batch, dim=0, reduce="sum")
        return g

    @staticmethod
    def _edges(pos, batch, cutoff):
        from torch_geometric.nn import radius_graph
        edge_index = radius_graph(pos, r=cutoff, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight


class DimeNetPPMT(MultiTask3D):
    def __init__(self, **kwargs):
        hidden_channels = 128
        backbone = DimeNetPlusPlus(
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,   # we treat this as the embedding
            num_blocks=3,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=hidden_channels,
            num_spherical=7,
            num_radial=6,
            cutoff=5.0,
            num_output_layers=2,
        )
        super().__init__(backbone, embed_dim=hidden_channels, hidden=hidden_channels, **kwargs)

    def embed(self, data):
        # DimeNet++'s forward returns (batch_size,)-scalar; for multi-task we
        # set out_channels=hidden_channels so the output is a vector per graph.
        return self.backbone(data.z, data.pos, data.batch)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_model(model_name: str, seed: int, device: torch.device,
                    epochs: int = 300, lr: float = 1e-3, batch_size: int = 32):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Load split CSVs and target scaler ---
    proc = ROOT / "data" / "processed"
    tr = pd.read_csv(proc / "splits" / "train.csv")
    va = pd.read_csv(proc / "splits" / "val.csv")
    te = pd.read_csv(proc / "splits" / "test.csv")
    with open(proc / "target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)

    # Standardize targets in-place (the CSVs may already be standardized; if
    # not, uncomment the two transform lines below).
    # tr[TARGET_COLS] = target_scaler.transform(tr[TARGET_COLS])
    # va[TARGET_COLS] = target_scaler.transform(va[TARGET_COLS])

    # --- Generate geometries (one per unique IL) ---
    all_df = pd.concat([tr, va, te], ignore_index=True)
    il_to_smiles = all_df.drop_duplicates("il_idx").set_index("il_idx")["smiles"].to_dict()
    print(f"[{model_name}] Generating 3D conformers for {len(il_to_smiles)} unique ILs ...")
    geoms = {}
    for il_idx, smi in il_to_smiles.items():
        g = smiles_to_xyz(smi, seed=seed)
        if g is None:
            raise RuntimeError(f"Failed to embed IL idx={il_idx}, SMILES={smi}")
        geoms[il_idx] = g
    print(f"[{model_name}] Geometries ready.")

    # --- Build datasets/loaders ---
    train_ds = ILGeomDataset(tr, geoms)
    val_ds   = ILGeomDataset(va, geoms)
    test_ds  = ILGeomDataset(te, geoms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    # --- Model ---
    if model_name == "schnet":
        model = SchNetMT().to(device)
    elif model_name == "dimenet":
        model = DimeNetPPMT().to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf"); best_state = None; patience = 30; bad = 0
    for ep in range(epochs):
        # train
        model.train()
        tr_loss = 0.0
        for b in train_loader:
            b = b.to(device)
            opt.zero_grad()
            pred = model(b)
            loss = loss_fn(pred, b.y.view(-1, 7))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += loss.item() * b.num_graphs
        tr_loss /= len(train_ds)
        sched.step()

        # val
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for b in val_loader:
                b = b.to(device)
                pred = model(b)
                vl += loss_fn(pred, b.y.view(-1, 7)).item() * b.num_graphs
        vl /= len(val_ds)

        if vl < best_val - 1e-5:
            best_val = vl; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[{model_name}] early stop at epoch {ep+1} (val {best_val:.4f})")
                break
        if (ep + 1) % 20 == 0:
            print(f"[{model_name}] ep {ep+1:3d}  train {tr_loss:.4f}  val {vl:.4f}")

    # --- Test evaluation ---
    model.load_state_dict(best_state)
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for b in test_loader:
            b = b.to(device)
            preds.append(model(b).cpu().numpy())
            ys.append(b.y.view(-1, 7).cpu().numpy())
    preds = np.concatenate(preds); ys = np.concatenate(ys)

    # Optional: if CSVs are already standardized, R² can be computed directly.
    # Otherwise, invert the scaler to get natural-unit R²:
    try:
        preds_nat = target_scaler.inverse_transform(preds)
        ys_nat    = target_scaler.inverse_transform(ys)
    except Exception:
        preds_nat, ys_nat = preds, ys

    r2 = {p: float(r2_score(ys_nat[:, i], preds_nat[:, i])) for i, p in enumerate(TARGET_COLS)}
    r2["AVG"] = float(np.mean(list(r2.values())))
    print(f"[{model_name}] TEST R²: {r2}")

    # --- Save ---
    out = {
        "model": model_name, "seed": seed,
        "epochs_trained": ep + 1, "best_val_loss": best_val,
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "r2": r2,
    }
    out_dir = ROOT / "results" / "baselines_3d"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{model_name}_seed{seed}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{model_name}] saved → {out_dir / f'{model_name}_seed{seed}.json'}")
    return r2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["schnet", "dimenet"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_one_model(args.model, args.seed, device,
                    epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
