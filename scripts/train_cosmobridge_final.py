"""COSMOBridge Final: True single model with embedded Chemprop D-MPNN.

Uses chemprop.models.MoleculeModel as a frozen sub-module for graph encoding.
PointNet for COSMO surface. GBH fusion + direct path + per-property gates.
3-stage training: fusion → direct → gates.
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
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.fusion.gated_bilinear_hyper_v2 import GatedBilinearHyperFusionV2
from src.models.fusion.multimodal_pointcloud import MultimodalPointCloudModel
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud

# Chemprop imports
from chemprop.utils import load_checkpoint as load_chemprop


class COSMOBridgeFinal(nn.Module):
    """COSMOBridge with embedded Chemprop D-MPNN.

    Single model, single forward pass:
      SMILES → Chemprop MPN (frozen) → 300D graph fingerprint
      COSMO point cloud → PointNet (frozen) → 256D surface fingerprint
      Thermo features → 25D

      Path A: GBH Fusion(graph × surface) → fused_head → 7 preds
      Path B: Direct FFN(graph + thermo) → direct_head → 7 preds
      Per-property gate: α_p blends paths

    3-stage training prevents gradient interference.
    """

    def __init__(self, chemprop_model, fused_dim=256, rank=32, hyper_hidden=64,
                 thermo_dim=25, n_properties=7, dropout=0.3):
        super().__init__()

        # Frozen encoder 1: Chemprop D-MPNN (embedded, not external)
        self.chemprop = chemprop_model
        for p in self.chemprop.parameters():
            p.requires_grad = False

        # Frozen encoder 2: PointNet
        self.pointnet = PointNetEncoder(in_channels=7, feature_dim=256, dropout=dropout)

        # Path A: GBH Bilinear Fusion
        self.graph_proj = nn.Linear(300, fused_dim)
        self.surface_proj = nn.Linear(256, fused_dim)
        self.fusion = GatedBilinearHyperFusionV2(
            pointcloud_dim=fused_dim, graph_dim=fused_dim, tabular_dim=thermo_dim,
            fused_dim=fused_dim, rank=rank, thermo_dim=5,
            hyper_hidden=hyper_hidden, dropout=dropout)
        self.fused_head = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, n_properties))

        # Path B: Direct graph FFN
        self.direct_head = nn.Sequential(
            nn.Linear(300 + thermo_dim, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, n_properties))

        # Per-property routing gates
        self.gate_logits = nn.Parameter(torch.tensor([
            2.0, 2.0, -2.0, -2.0, -2.0, 0.0, 1.5]))

    def _get_chemprop_fingerprint(self, smiles_batch, features_batch):
        """Get Chemprop MPN fingerprint (300D, no FFN readout)."""
        with torch.no_grad():
            # Chemprop's fingerprint method handles SMILES → MolGraph internally
            fp = self.chemprop.fingerprint(
                smiles_batch, features_batch=features_batch,
                fingerprint_type="MPN")
        return fp[:, :300]  # MPN output only (exclude concatenated features)

    def forward(self, point_cloud, features, smiles_batch, features_np_batch, **kwargs):
        """
        Args:
            point_cloud: (B, 1024, 7) — COSMO surface
            features: (B, 25) — normalized thermo features (tensor)
            smiles_batch: list of [smiles] — for Chemprop
            features_np_batch: list of np.array — for Chemprop
        """
        # Encoder 1: Chemprop graph fingerprint
        h_graph = self._get_chemprop_fingerprint(smiles_batch, features_np_batch)
        h_graph = h_graph.to(features.device)

        # Encoder 2: PointNet surface features
        with torch.no_grad():
            h_surface = self.pointnet(point_cloud)

        # Path A: Bilinear fusion
        g = self.graph_proj(h_graph)
        s = self.surface_proj(h_surface)
        h_fused = self.fusion(s, g, features)
        preds_fused = self.fused_head(h_fused)

        # Path B: Direct graph FFN
        h_direct_in = torch.cat([h_graph, features], dim=-1)
        preds_direct = self.direct_head(h_direct_in)

        # Per-property routing
        alpha = torch.sigmoid(self.gate_logits)
        preds = alpha.unsqueeze(0) * preds_fused + (1 - alpha.unsqueeze(0)) * preds_direct

        return preds, {"gate_values": alpha.detach()}


class COSMOBridgeDataset(Dataset):
    """Dataset that provides both tensor data and SMILES for Chemprop."""

    def __init__(self, csv_path, pc_dir, is_train=True):
        self.inner = PointCloudMultimodalDataset(csv_path, pc_dir, is_train=is_train)
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        item = self.inner[idx]
        row = self.df.iloc[idx]
        item["smiles"] = row["smiles"]
        # Raw thermo features for Chemprop (it does its own normalization)
        item["features_np"] = np.array([row[c] for c in FEATURE_COLUMNS[:5]], dtype=np.float32)
        return item


def collate_cosmobridge(batch):
    """Custom collate that handles both tensors and SMILES."""
    base = collate_pointcloud(batch)
    base["smiles_batch"] = [[b["smiles"]] for b in batch]
    base["features_np_batch"] = [b["features_np"] for b in batch]
    return base


import numpy as np


def evaluate(model, loader, device):
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            preds, _ = model(
                point_cloud=batch["point_cloud"], features=batch["features"],
                smiles_batch=batch["smiles_batch"],
                features_np_batch=batch["features_np_batch"])
            all_p.append(preds.cpu().numpy())
            all_t.append(batch["targets"].cpu().numpy())
    return np.concatenate(all_p), np.concatenate(all_t)


def train_stage(model, train_ldr, val_ldr, device, ckpt_path,
                lr=1e-3, epochs=200, patience=30, name=""):
    trainable = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in trainable)
    print(f"  {name}: {n:,} trainable, lr={lr}")
    if n == 0:
        print(f"  Skip (no trainable params)"); return

    optimizer = AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)

    best, no_imp = float("inf"), 0
    for ep in range(epochs):
        model.train()
        tl, cnt = 0, 0
        for batch in train_ldr:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            optimizer.zero_grad()
            preds, _ = model(
                point_cloud=batch["point_cloud"], features=batch["features"],
                smiles_batch=batch["smiles_batch"],
                features_np_batch=batch["features_np_batch"])
            loss = ((preds - batch["targets"])**2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            tl += loss.item(); cnt += 1
        scheduler.step()

        model.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for batch in val_ldr:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor): batch[k] = v.to(device)
                preds, _ = model(
                    point_cloud=batch["point_cloud"], features=batch["features"],
                    smiles_batch=batch["smiles_batch"],
                    features_np_batch=batch["features_np_batch"])
                vl += ((preds - batch["targets"])**2).mean().item(); vn += 1
        avg = vl / max(vn, 1)
        if avg < best: best = avg; no_imp = 0; torch.save(model.state_dict(), ckpt_path)
        else: no_imp += 1
        if ep % 30 == 0:
            g = torch.sigmoid(model.gate_logits).detach().cpu().numpy()
            print(f"    Ep {ep:3d} | T:{tl/max(cnt,1):.4f} V:{avg:.4f} B:{best:.4f} "
                  f"P:{no_imp}/{patience} | [{' '.join(f'{x:.2f}' for x in g)}]")
        if no_imp >= patience: print(f"    Early stop ep {ep}"); break
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    pc_dir = "data/pipeline/point_clouds"
    orig_splits = Path("data/processed/splits")

    # Data
    train_ds = COSMOBridgeDataset(str(orig_splits / "train.csv"), pc_dir, is_train=True)
    val_ds = COSMOBridgeDataset(str(orig_splits / "val.csv"), pc_dir, is_train=False)
    test_ds = COSMOBridgeDataset(str(orig_splits / "test.csv"), pc_dir, is_train=False)
    train_ldr = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_cosmobridge)
    val_ldr = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_cosmobridge)
    test_ldr = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_cosmobridge)

    # Build model
    print(f"\n{'='*60}")
    print("COSMOBridge Final: Embedded Chemprop + PointNet + GBH")
    print(f"{'='*60}")

    # Load Chemprop as sub-module
    chemprop_model = load_chemprop("checkpoints/chemprop/fold_0/model_0/model.pt")
    print(f"  Chemprop loaded: {sum(p.numel() for p in chemprop_model.parameters()):,} params")

    model = COSMOBridgeFinal(chemprop_model, fused_dim=256, rank=32,
                               hyper_hidden=64, thermo_dim=len(FEATURE_COLUMNS))
    model.to(device)

    # Load pre-trained PointNet
    pc_ckpt = torch.load("checkpoints/pointcloud/best_model.pt", map_location=device, weights_only=False)
    pc_sd = pc_ckpt["model_state_dict"] if "model_state_dict" in pc_ckpt else pc_ckpt
    my_sd = model.state_dict()
    loaded = 0
    for k, v in pc_sd.items():
        if k.startswith("pointnet.") and k in my_sd and my_sd[k].shape == v.shape:
            my_sd[k] = v; loaded += 1
    model.load_state_dict(my_sd)
    print(f"  PointNet loaded: {loaded} weight tensors")

    # Freeze PointNet
    for p in model.pointnet.parameters(): p.requires_grad = False

    n_total = sum(p.numel() for p in model.parameters())
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {n_total:,}, Frozen: {n_frozen:,}, Trainable: {n_train:,}")

    ckpt_dir = Path("checkpoints/cosmobridge_final")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ═══ STAGE 1: Train fusion path ═══
    print(f"\n{'='*60}")
    print("STAGE 1: Fusion path (graph_proj + surface_proj + fusion + fused_head)")
    print(f"{'='*60}")
    for p in model.direct_head.parameters(): p.requires_grad = False
    model.gate_logits.requires_grad = False

    train_stage(model, train_ldr, val_ldr, device, ckpt_dir / "s1.pt",
                lr=1e-3, epochs=200, patience=30, name="Stage 1")

    p1, t1 = evaluate(model, test_ldr, device)
    m1 = compute_metrics(p1, t1)
    print(f"  Stage 1: gamma1={m1['gamma1_r2']:.4f} gamma2={m1['gamma2_r2']:.4f} avg={m1['avg_r2']:.4f}")

    # ═══ STAGE 2: Train direct path ═══
    print(f"\n{'='*60}")
    print("STAGE 2: Direct path (direct_head only)")
    print(f"{'='*60}")
    for p in model.graph_proj.parameters(): p.requires_grad = False
    for p in model.surface_proj.parameters(): p.requires_grad = False
    for p in model.fusion.parameters(): p.requires_grad = False
    for p in model.fused_head.parameters(): p.requires_grad = False
    for p in model.direct_head.parameters(): p.requires_grad = True

    train_stage(model, train_ldr, val_ldr, device, ckpt_dir / "s2.pt",
                lr=1e-3, epochs=200, patience=30, name="Stage 2")

    p2, t2 = evaluate(model, test_ldr, device)
    m2 = compute_metrics(p2, t2)
    print(f"  Stage 2: G_E={m2['G_E_r2']:.4f} H_E={m2['H_E_r2']:.4f} avg={m2['avg_r2']:.4f}")

    # ═══ STAGE 3: Train gates ═══
    print(f"\n{'='*60}")
    print("STAGE 3: Gates only (7 params)")
    print(f"{'='*60}")
    for p in model.direct_head.parameters(): p.requires_grad = False
    model.gate_logits.requires_grad = True

    train_stage(model, train_ldr, val_ldr, device, ckpt_dir / "s3.pt",
                lr=0.1, epochs=100, patience=30, name="Stage 3")

    # ═══ FINAL EVAL ═══
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    pf, tf = evaluate(model, test_ldr, device)
    metrics = compute_metrics(pf, tf)
    print(format_metrics(metrics, "COSMOBridge Final"))

    gates = torch.sigmoid(model.gate_logits).detach().cpu().numpy()
    print(f"\n  Gates:")
    for i, p in enumerate(TARGET_COLUMNS):
        path = "FUSION" if gates[i] > 0.6 else ("DIRECT" if gates[i] < 0.4 else "MIXED")
        print(f"    {p:15s}: α={gates[i]:.3f} → {path}")

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    prev = {}
    for name, path, key in [
        ("Chemprop", "results/chemprop_results.json", "test_metrics"),
        ("3-Model Router", "results/per_property_router_results.json", "metrics"),
        ("CP-GBH Hybrid", "results/chemprop_gbh_hybrid_results.json", "metrics"),
    ]:
        try:
            data = json.load(open(path))
            prev[name] = data.get(key, {})
        except: pass

    header = f"  {'Property':<15s}"
    for n in prev: header += f" {n[:14]:>14s}"
    header += f" {'Final':>14s}"
    print(header)
    print("  " + "-" * len(header))
    for p in TARGET_COLUMNS:
        line = f"  {p:<15s}"
        for n in prev: line += f" {prev[n].get(f'{p}_r2', 0):14.4f}"
        line += f" {metrics[f'{p}_r2']:14.4f}"
        print(line)
    line = f"  {'AVERAGE':<15s}"
    for n in prev: line += f" {prev[n].get('avg_r2', 0):14.4f}"
    line += f" {metrics['avg_r2']:14.4f}"
    print(line)

    # Save
    torch.save(model.state_dict(), ckpt_dir / "final.pt")
    results = {
        "model": "COSMOBridge_Final",
        "description": "True single model: embedded Chemprop D-MPNN (frozen) + PointNet (frozen) "
                       "+ GBH bilinear fusion + direct FFN + per-property gates. 3-stage training.",
        "n_total_params": n_total,
        "gate_values": {p: float(g) for p, g in zip(TARGET_COLUMNS, gates)},
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items()},
    }
    with open("results/cosmobridge_final_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/cosmobridge_final_results.json")


if __name__ == "__main__":
    main()
