"""MoE-A + Fix3: Domain-conditional gamma1 prediction.

Adds a source indicator (0=original, 1=ILThermo) as input.
The model learns that gamma1 has different semantics per source.
At test time, source=0 (original) is used.
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
from src.data.graph_builder import smiles_to_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.moe import ExpertHead, PropertyConditionedGating
from src.models.fusion.multimodal_pointcloud import PointCloudFusion
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_moe import evaluate_single

TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


class DomainAwareMergedDataset(Dataset):
    """Merged dataset that includes source indicator."""

    def __init__(self, csv_path, pc_dir, feature_columns, is_train=True, n_points=1024):
        self.df = pd.read_csv(csv_path)
        self.pc_dir = Path(pc_dir)
        self.feature_columns = feature_columns
        self.is_train = is_train
        self.n_points = n_points

        # Point cloud index
        idx_path = self.pc_dir / "index.csv"
        self.pc_index = {}
        if idx_path.exists():
            idx_df = pd.read_csv(idx_path)
            self.pc_index = dict(zip(idx_df["smiles"], idx_df["filename"]))

        # Graphs
        self.graphs = {}
        for smi in self.df["smiles"].unique():
            try:
                self.graphs[smi] = smiles_to_graph(smi)
            except:
                self.graphs[smi] = None

    def __len__(self):
        return len(self.df)

    def _load_pc(self, smiles):
        fn = self.pc_index.get(smiles)
        if fn:
            path = self.pc_dir / fn
            if path.exists():
                pts = np.load(path)["points"]
                if len(pts) >= self.n_points:
                    idx = np.random.choice(len(pts), self.n_points, replace=False) if self.is_train else np.arange(self.n_points)
                    pts = pts[idx]
                else:
                    extra = np.random.choice(len(pts), self.n_points - len(pts), replace=True)
                    pts = np.concatenate([pts, pts[extra]])
                if self.is_train:
                    angle = np.random.uniform(0, 2*np.pi)
                    c, s = np.cos(angle), np.sin(angle)
                    R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
                    pts[:,:3] = pts[:,:3] @ R.T
                    pts[:,3:6] = pts[:,3:6] @ R.T
                return torch.tensor(pts, dtype=torch.float32)
        return torch.zeros(self.n_points, 7)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smi = row["smiles"]

        features = torch.tensor([row[c] for c in self.feature_columns], dtype=torch.float32)
        pc = self._load_pc(smi)

        g = self.graphs.get(smi)
        if g is not None:
            af = torch.tensor(g["atom_features"], dtype=torch.float32)
            ei = torch.tensor(g["edge_index"], dtype=torch.long)
            bf = torch.tensor(g["bond_features"], dtype=torch.float32)
        else:
            af = torch.zeros(1, ATOM_FEATURE_DIM)
            ei = torch.zeros(2, 0, dtype=torch.long)
            bf = torch.zeros(0, BOND_FEATURE_DIM)

        targets = torch.tensor(
            [row[c] if pd.notna(row.get(c)) else float("nan") for c in TARGET_COLUMNS],
            dtype=torch.float32)

        # Source indicator: 0=original, 1=ilthermo
        source = torch.tensor(0 if row.get("source", "original") == "original" else 1,
                              dtype=torch.long)

        return {"point_cloud": pc, "features": features,
                "atom_features": af, "edge_index": ei, "bond_features": bf,
                "num_atoms": af.shape[0], "targets": targets, "source": source}


def collate_domain(batch):
    pcs = torch.stack([b["point_cloud"] for b in batch])
    feats = torch.stack([b["features"] for b in batch])
    targets = torch.stack([b["targets"] for b in batch])
    sources = torch.stack([b["source"] for b in batch])

    af_list, ei_list, bf_list, gb = [], [], [], []
    offset = 0
    for i, b in enumerate(batch):
        n = b["atom_features"].shape[0]
        af_list.append(b["atom_features"])
        gb.extend([i] * n)
        ei = b["edge_index"].clone()
        if ei.shape[1] > 0: ei += offset
        ei_list.append(ei)
        bf_list.append(b["bond_features"])
        offset += n

    return {
        "point_cloud": pcs, "features": feats, "targets": targets, "source": sources,
        "atom_features": torch.cat(af_list),
        "edge_index": torch.cat(ei_list, dim=1),
        "bond_features": torch.cat(bf_list),
        "batch": torch.tensor(gb, dtype=torch.long),
    }


class DomainConditionalMoE(nn.Module):
    """MoE-A with domain-conditional gamma1 prediction.

    A domain embedding is added to the fused representation before
    the expert heads. The model learns source-specific adjustments
    for gamma1 while sharing the backbone across domains.
    """

    def __init__(self, feature_dim, fused_dim=256, dropout=0.3, pretrained_gnn_path=None):
        super().__init__()
        self.pointnet = PointNetEncoder(in_channels=7, feature_dim=256, dropout=dropout)
        self.gnn = MolecularGNN(
            atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
            dropout=dropout, pooling="mean", num_targets=0)
        if pretrained_gnn_path and Path(pretrained_gnn_path).exists():
            ckpt = torch.load(pretrained_gnn_path, map_location="cpu", weights_only=True)
            gnn_state = {k: v for k, v in ckpt.items()
                         if any(k.startswith(p) for p in ["atom_projection", "convs", "batch_norms", "pool"])}
            if gnn_state:
                self.gnn.load_state_dict(gnn_state, strict=False)

        self.fusion = PointCloudFusion(
            pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
            fused_dim=fused_dim, num_heads=8, dropout=dropout)

        # Domain embedding (2 sources: original, ilthermo)
        self.domain_embed = nn.Embedding(2, fused_dim)
        # Gated domain conditioning
        self.domain_gate = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.Sigmoid(),
        )

        self.experts = nn.ModuleList([
            ExpertHead(fused_dim, hidden_dim=128, num_targets=7, dropout=dropout)
            for _ in range(4)])
        self.gating = PropertyConditionedGating(
            input_dim=fused_dim, num_experts=4, num_properties=7, hidden_dim=64)

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, source=None, **kwargs):
        pc_feat = self.pointnet(point_cloud)
        graph_feat = self.gnn.get_features(atom_features, edge_index, bond_features, batch)
        fused = self.fusion(pc_feat, graph_feat, features)

        # Domain conditioning
        if source is not None:
            domain_emb = self.domain_embed(source)
            gate = self.domain_gate(torch.cat([fused, domain_emb], dim=1))
            fused = fused * gate  # Soft domain modulation

        expert_preds = torch.stack([e(fused) for e in self.experts], dim=2)
        gate_weights, lb_loss = self.gating(fused)
        predictions = (expert_preds * gate_weights).sum(dim=2)

        return predictions, {"load_balance_loss": lb_loss, "gate_weights": gate_weights.detach()}


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    merged_dir = Path("data/merged_v3")
    meta = json.load(open(merged_dir / "metadata.json"))
    merged_features = meta["feature_columns"]
    pc_dir = "data/pipeline/point_clouds"

    # ── Train ──
    print(f"{'='*60}")
    print("MoE-A + Fix3: Domain-Conditional Gamma1")
    print(f"{'='*60}")

    train_ds = DomainAwareMergedDataset(str(merged_dir / "splits/train.csv"), pc_dir, merged_features, is_train=True)
    val_ds = DomainAwareMergedDataset(str(merged_dir / "splits/val.csv"), pc_dir, merged_features, is_train=False)
    test_ds = DomainAwareMergedDataset(str(merged_dir / "splits/test.csv"), pc_dir, merged_features, is_train=False)

    train_ldr = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_domain)
    val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_domain)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_domain)

    model = DomainConditionalMoE(
        feature_dim=len(merged_features),
        pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
    model.to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_dir = Path("checkpoints/moe_fix3")
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
            preds, aux = model(point_cloud=batch["point_cloud"], features=batch["features"],
                               atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                               bond_features=batch["bond_features"], batch=batch["batch"],
                               source=batch["source"])
            mask = ~torch.isnan(batch["targets"])
            safe = batch["targets"].clone(); safe[~mask] = 0.0
            loss = ((preds-safe)**2 * mask.float()).sum() / mask.float().sum().clamp(min=1) + aux["load_balance_loss"]
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
                preds, _ = model(point_cloud=batch["point_cloud"], features=batch["features"],
                                 atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                                 bond_features=batch["bond_features"], batch=batch["batch"],
                                 source=batch["source"])
                mask = ~torch.isnan(batch["targets"])
                safe = batch["targets"].clone(); safe[~mask] = 0.0
                vl += ((preds-safe)**2*mask.float()).sum().item()/mask.float().sum().clamp(min=1).item()
                vn += 1
        avg_val = vl/max(vn,1)
        if avg_val < best_loss:
            best_loss = avg_val; no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_improve += 1
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/200 | Train: {tl/max(n,1):.4f} | Val: {avg_val:.4f} | Best: {best_loss:.4f} | Pat: {no_improve}/25")
        if no_improve >= 25:
            print(f"  Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True))

    # ── Evaluate with source=0 (original domain) ──
    print(f"\n{'='*60}")
    print("EVALUATION (source=0, original domain)")
    print(f"{'='*60}")

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_ldr:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            # Force source=0 at test time
            source_orig = torch.zeros_like(batch["source"])
            preds, aux = model(point_cloud=batch["point_cloud"], features=batch["features"],
                               atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                               bond_features=batch["bond_features"], batch=batch["batch"],
                               source=source_orig)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    metrics = compute_metrics(preds, targets)

    print(format_metrics(metrics, "MoE-A Fix3 (domain-conditional)"))

    # Compare
    print(f"\n  Per-property:")
    for p in TARGET_COLUMNS:
        print(f"    {p:15s} R² = {metrics[f'{p}_r2']:.4f}")
    print(f"    {'AVERAGE':15s} R² = {metrics['avg_r2']:.4f}")

    with open("results/moe_fix3_results.json", "w") as f:
        json.dump({"fix": "domain_conditional_gamma1",
                   "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                              for k, v in metrics.items()},
                   "gate_weights": aux["gate_weights"].cpu().numpy().mean(axis=0).tolist()}, f, indent=2)
    print("Saved: results/moe_fix3_results.json")


if __name__ == "__main__":
    main()
