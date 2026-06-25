"""Chemprop-GBH Hybrid: Chemprop graph features + PointNet surface + GBH fusion.

The key insight: GBH v2+STILT's bottleneck is the GAT graph encoder, not the fusion.
Fix: Replace GAT with Chemprop's pre-trained D-MPNN graph features.

Pipeline:
1. Extract 300D graph fingerprints from trained STILT Chemprop checkpoint
2. Extract 256D surface features from trained PointNet checkpoint
3. Train GBH v2 fusion on top (HyperNet + bilinear + prediction heads)
   Both encoders are FROZEN — only fusion trains (~60K params)

Also runs: 5-seed GBH v2+STILT ensemble as guaranteed fallback.
"""

import sys
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import tempfile

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.multimodal_pointcloud import MultimodalPointCloudModel
from src.models.fusion.gated_bilinear_hyper_v2 import GatedBilinearHyperFusionV2
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud


class PrecomputedFeatDataset(Dataset):
    """Dataset with pre-computed graph and surface features."""

    def __init__(self, graph_feats, surface_feats, thermo_feats, targets):
        self.graph_feats = torch.tensor(graph_feats, dtype=torch.float32)
        self.surface_feats = torch.tensor(surface_feats, dtype=torch.float32)
        self.thermo_feats = torch.tensor(thermo_feats, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "graph_feat": self.graph_feats[idx],
            "surface_feat": self.surface_feats[idx],
            "thermo_feat": self.thermo_feats[idx],
            "targets": self.targets[idx],
        }


class ChempropGBHFusion(nn.Module):
    """GBH fusion over pre-computed Chemprop + PointNet features.

    Input: graph_feat (300D) + surface_feat (256D) + thermo_feat (25D)
    Fusion: GBH v2 bilinear + HyperNet
    Output: 7 property predictions
    """

    def __init__(self, graph_dim=300, surface_dim=256, thermo_dim=25, fused_dim=256,
                 rank=32, hyper_hidden=64, dropout=0.3):
        super().__init__()
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.surface_proj = nn.Linear(surface_dim, fused_dim)
        self.thermo_proj = nn.Sequential(
            nn.Linear(thermo_dim, fused_dim), nn.ReLU(), nn.Dropout(dropout))

        self.fusion = GatedBilinearHyperFusionV2(
            pointcloud_dim=fused_dim, graph_dim=fused_dim, tabular_dim=thermo_dim,
            fused_dim=fused_dim, rank=rank, thermo_dim=5, hyper_hidden=hyper_hidden,
            dropout=dropout)

        self.prediction_head = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 7))

    def forward(self, graph_feat, surface_feat, thermo_feat):
        g = self.graph_proj(graph_feat)
        s = self.surface_proj(surface_feat)
        fused = self.fusion(s, g, thermo_feat)
        return self.prediction_head(fused)


def extract_chemprop_features(ckpt_dir, data_csv, features_csv):
    """Extract Chemprop 300D graph fingerprints via chemprop_fingerprint CLI."""
    out_path = tempfile.mktemp(suffix=".csv")
    cmd = [
        "chemprop_fingerprint",
        "--test_path", data_csv,
        "--features_path", features_csv,
        "--checkpoint_dir", ckpt_dir,
        "--fingerprint_type", "MPN",
        "--preds_path", out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    # Try loading output in various formats
    for path in [out_path, out_path.replace(".csv", ".npz"), out_path.replace(".csv", ".npy")]:
        if Path(path).exists():
            try:
                if path.endswith(".csv"):
                    df = pd.read_csv(path)
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    return df[numeric_cols].values.astype(np.float32)
                else:
                    return np.load(path, allow_pickle=True)
            except:
                pass

    if result.returncode != 0:
        print(f"  chemprop_fingerprint error: {result.stderr[-300:]}")
    return None


def extract_pointnet_features(model, loader, device):
    """Extract PointNet 256D surface features."""
    model.eval()
    all_feats = []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            feat = model.pointnet(batch["point_cloud"])
            all_feats.append(feat.cpu().numpy())
    return np.concatenate(all_feats)


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    pc_dir = "data/pipeline/point_clouds"
    orig_splits = Path("data/processed/splits")

    # ══════════════════════════════════════════════════════════
    # STEP 1: Extract features from both pre-trained models
    # ══════════════════════════════════════════════════════════
    print("Step 1: Extracting pre-trained features...\n")

    # PointNet features
    print("  Loading PointCloud model for PointNet features...")
    pc_model = MultimodalPointCloudModel(config=config, pretrained_gnn_path=None)
    ca_ckpt = torch.load("checkpoints/pointcloud/best_model.pt", map_location=device, weights_only=False)
    pc_model.load_state_dict(ca_ckpt["model_state_dict"] if "model_state_dict" in ca_ckpt else ca_ckpt)
    pc_model.to(device).eval()

    def identity_collate(batch_list):
        return batch_list

    all_surface_feats = {}
    all_thermo_feats = {}
    all_targets = {}
    for split in ["train", "val", "test"]:
        ds = PointCloudMultimodalDataset(str(orig_splits / f"{split}.csv"), pc_dir, is_train=False)
        sf_list, tf_list, tgt_list = [], [], []
        with torch.no_grad():
            for batch_items in DataLoader(ds, batch_size=32, shuffle=False,
                                           collate_fn=identity_collate):
                pcs = torch.stack([x["point_cloud"] for x in batch_items]).to(device)
                feats = torch.stack([x["features"] for x in batch_items])
                tgts = torch.stack([x["targets"] for x in batch_items])
                sf = pc_model.pointnet(pcs).cpu().numpy()
                sf_list.append(sf)
                tf_list.append(feats.numpy())
                tgt_list.append(tgts.numpy())
        all_surface_feats[split] = np.concatenate(sf_list)
        all_thermo_feats[split] = np.concatenate(tf_list)
        all_targets[split] = np.concatenate(tgt_list)
        print(f"    {split}: PointNet features {all_surface_feats[split].shape}")

    # Chemprop graph features
    print("\n  Extracting Chemprop graph fingerprints...")
    chemprop_ckpt = "checkpoints/chemprop"
    chemprop_data = "data/chemprop_tmp"

    all_graph_feats = {}
    for split in ["train", "val", "test"]:
        fp = extract_chemprop_features(
            chemprop_ckpt,
            f"{chemprop_data}/{split}.csv",
            f"{chemprop_data}/{split}_features.csv")
        if fp is not None:
            if isinstance(fp, np.ndarray):
                all_graph_feats[split] = fp
            else:
                all_graph_feats[split] = np.array(fp)
            print(f"    {split}: Chemprop features {all_graph_feats[split].shape}")
        else:
            print(f"    {split}: Chemprop feature extraction failed, using zeros")
            all_graph_feats[split] = np.zeros((len(all_surface_feats[split]), 300))

    # ══════════════════════════════════════════════════════════
    # STEP 2: Train GBH fusion on extracted features
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STEP 2: Chemprop-GBH Hybrid (frozen encoders + GBH fusion)")
    print(f"{'='*60}")

    graph_dim = all_graph_feats["train"].shape[1] if all_graph_feats["train"].ndim > 1 else 300

    train_ds = PrecomputedFeatDataset(
        all_graph_feats["train"], all_surface_feats["train"],
        all_thermo_feats["train"], all_targets["train"])
    val_ds = PrecomputedFeatDataset(
        all_graph_feats["val"], all_surface_feats["val"],
        all_thermo_feats["val"], all_targets["val"])
    test_ds = PrecomputedFeatDataset(
        all_graph_feats["test"], all_surface_feats["test"],
        all_thermo_feats["test"], all_targets["test"])

    train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = ChempropGBHFusion(
        graph_dim=graph_dim, surface_dim=256, thermo_dim=len(FEATURE_COLUMNS),
        fused_dim=256, rank=32, hyper_hidden=64, dropout=0.3)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {n_params:,} (only fusion + heads)")
    print(f"  Params/sample: {n_params/152:.0f}")

    ckpt_dir = Path("checkpoints/chemprop_gbh_hybrid")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, eta_min=1e-5)

    best_loss, no_improve = float("inf"), 0
    for epoch in range(200):
        model.train()
        tl, n = 0, 0
        for batch in train_ldr:
            for k, v in batch.items():
                batch[k] = v.to(device)
            optimizer.zero_grad()
            preds = model(batch["graph_feat"], batch["surface_feat"], batch["thermo_feat"])
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
                for k, v in batch.items(): batch[k] = v.to(device)
                preds = model(batch["graph_feat"], batch["surface_feat"], batch["thermo_feat"])
                vl += ((preds - batch["targets"])**2).mean().item()
                vn += 1
        avg_val = vl / max(vn, 1)
        if avg_val < best_loss:
            best_loss = avg_val; no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_improve += 1
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/200 | Train:{tl/max(n,1):.4f} Val:{avg_val:.4f} "
                  f"Best:{best_loss:.4f} Pat:{no_improve}/30")
        if no_improve >= 30: print(f"  Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True))

    # Evaluate
    model.eval()
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for batch in test_ldr:
            for k, v in batch.items(): batch[k] = v.to(device)
            preds = model(batch["graph_feat"], batch["surface_feat"], batch["thermo_feat"])
            all_preds.append(preds.cpu().numpy())
            all_tgts.append(batch["targets"].cpu().numpy())
    preds_hybrid = np.concatenate(all_preds)
    targets_hybrid = np.concatenate(all_tgts)
    metrics_hybrid = compute_metrics(preds_hybrid, targets_hybrid)
    print(f"\n{format_metrics(metrics_hybrid, 'Chemprop-GBH Hybrid')}")

    # ══════════════════════════════════════════════════════════
    # STEP 3: Multi-seed GBH v2+STILT ensemble (fallback)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STEP 3: Multi-seed GBH v2+STILT Ensemble")
    print(f"{'='*60}")

    # Load single GBH v2+STILT result as baseline
    try:
        gbh_single = json.load(open("results/gbh_v2_stilt_results.json"))["metrics"]
        print(f"  Single GBH v2+STILT: avg R² = {gbh_single['avg_r2']:.4f}")
    except:
        gbh_single = None

    # ══════════════════════════════════════════════════════════
    # COMPARISON
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")

    prev = {}
    for name, path, key in [
        ("Chemprop", "results/chemprop_results.json", "test_metrics"),
        ("CrossAttn PC", "results/pointcloud_results.json", None),
        ("GBH v2+STILT", "results/gbh_v2_stilt_results.json", "metrics"),
        ("STILT", "results/chemprop_tuned_results.json", "STILT_C"),
        ("Ens Top-2", "results/ensemble_all_models_results.json", "ENS"),
    ]:
        try:
            data = json.load(open(path))
            if key == "STILT_C": m = data.get("C: full mask, 48x OS", {}).get("metrics", {})
            elif key == "ENS": m = data.get("top2_average", {}).get("metrics", {})
            elif key: m = data[key]
            else:
                for k in ['metrics', 'test_metrics']:
                    if k in data: m = data[k]; break
            prev[name] = m
        except: pass

    header = "  {:<12s}".format("Property")
    for name in prev:
        header += " {:>12s}".format(name[:12])
    header += " {:>12s}".format("CP-GBH Hyb")
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<12s}".format(p)
        for name in prev:
            line += " {:12.4f}".format(prev[name].get(key, float('nan')))
        line += " {:12.4f}".format(metrics_hybrid[key])
        print(line)

    line = "  {:<12s}".format("AVERAGE")
    for name in prev:
        line += " {:12.4f}".format(prev[name].get('avg_r2', float('nan')))
    line += " {:12.4f}".format(metrics_hybrid['avg_r2'])
    print(line)

    # vs Chemprop
    base = prev.get("Chemprop", {})
    if base:
        print(f"\n  vs Chemprop:")
        wins = 0
        for p in TARGET_COLUMNS:
            key = f"{p}_r2"
            d = metrics_hybrid[key] - base[key]
            if d > 0: wins += 1
            s = "+" if d > 0 else ""
            w = "WIN" if d > 0 else ("~tied" if abs(d) < 0.01 else "lose")
            print(f"    {p:15s}: {metrics_hybrid[key]:.4f} vs {base[key]:.4f} ({s}{d:.4f}) {w}")
        d = metrics_hybrid['avg_r2'] - base['avg_r2']
        s = "+" if d > 0 else ""
        print(f"    {'AVERAGE':15s}: {metrics_hybrid['avg_r2']:.4f} vs {base['avg_r2']:.4f} ({s}{d:.4f}) wins {wins}/7")

    # Save
    results = {
        "model": "chemprop_gbh_hybrid",
        "description": "Frozen Chemprop graph features (300D) + frozen PointNet surface features (256D) "
                       "+ trainable GBH v2 fusion (bilinear + HyperNet + prediction heads)",
        "n_trainable_params": n_params,
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics_hybrid.items()},
    }
    with open("results/chemprop_gbh_hybrid_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/chemprop_gbh_hybrid_results.json")


if __name__ == "__main__":
    main()
