"""Per-Property Router: Route each property to its best model.

Uses actual predictions (not just R² scores) from 3 models:
- COSMOBridge: gamma1, gamma2, P (best activity coefficients + vapor pressure)
- STILT: G_E, G_mix, H_vap (best bulk thermodynamic properties)
- Chemprop: H_E (best excess enthalpy)

This is a principled ensemble where the routing is determined by
validation performance, not learned from test data.
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
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.models.fusion.multimodal_pointcloud import MultimodalPointCloudModel
from src.models.fusion.gated_bilinear_hyper_v2 import GatedBilinearHyperFusionV2
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud


def identity_collate(batch_list):
    return batch_list


def get_cosmobridge_preds(device, orig_splits, pc_dir):
    """Get COSMOBridge predictions on test set."""
    config = load_config("configs/default.yaml")

    # Load PointNet
    pc_model = MultimodalPointCloudModel(config=config, pretrained_gnn_path=None)
    ca_ckpt = torch.load("checkpoints/pointcloud/best_model.pt", map_location=device, weights_only=False)
    pc_model.load_state_dict(ca_ckpt["model_state_dict"] if "model_state_dict" in ca_ckpt else ca_ckpt)
    pc_model.to(device).eval()

    # Extract PointNet features
    ds = PointCloudMultimodalDataset(str(orig_splits / "test.csv"), pc_dir, is_train=False)
    sf_list, tf_list, tgt_list = [], [], []
    with torch.no_grad():
        for batch_items in DataLoader(ds, batch_size=32, shuffle=False, collate_fn=identity_collate):
            pcs = torch.stack([x["point_cloud"] for x in batch_items]).to(device)
            feats = torch.stack([x["features"] for x in batch_items])
            tgts = torch.stack([x["targets"] for x in batch_items])
            sf = pc_model.pointnet(pcs).cpu().numpy()
            sf_list.append(sf)
            tf_list.append(feats.numpy())
            tgt_list.append(tgts.numpy())
    surface_feats = np.concatenate(sf_list)
    thermo_feats = np.concatenate(tf_list)
    targets = np.concatenate(tgt_list)

    # Extract Chemprop features
    out_path = tempfile.mktemp(suffix=".csv")
    cmd = ["chemprop_fingerprint",
           "--test_path", "data/chemprop_tmp/test.csv",
           "--features_path", "data/chemprop_tmp/test_features.csv",
           "--checkpoint_dir", "checkpoints/chemprop",
           "--fingerprint_type", "MPN",
           "--preds_path", out_path]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    graph_feats = pd.read_csv(out_path).select_dtypes(include=[np.number]).values.astype(np.float32)

    # Load COSMOBridge fusion
    from scripts.train_chemprop_gbh_hybrid import ChempropGBHFusion
    model = ChempropGBHFusion(graph_dim=graph_feats.shape[1], surface_dim=256,
                               thermo_dim=len(FEATURE_COLUMNS), fused_dim=256,
                               rank=32, hyper_hidden=64, dropout=0.3)
    model.load_state_dict(torch.load("checkpoints/chemprop_gbh_hybrid/best.pt",
                                      map_location=device, weights_only=True))
    model.to(device).eval()

    # Predict
    with torch.no_grad():
        g = torch.tensor(graph_feats, dtype=torch.float32).to(device)
        s = torch.tensor(surface_feats, dtype=torch.float32).to(device)
        t = torch.tensor(thermo_feats, dtype=torch.float32).to(device)
        preds = model(g, s, t).cpu().numpy()

    return preds, targets


def get_chemprop_preds():
    """Get Chemprop predictions on test set."""
    out_path = tempfile.mktemp(suffix=".csv")
    cmd = ["chemprop_predict",
           "--test_path", "data/chemprop_tmp/test.csv",
           "--features_path", "data/chemprop_tmp/test_features.csv",
           "--checkpoint_dir", "checkpoints/chemprop",
           "--preds_path", out_path]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return pd.read_csv(out_path)[TARGET_COLUMNS].values


def get_stilt_preds():
    """Get STILT predictions on test set."""
    out_path = tempfile.mktemp(suffix=".csv")
    # STILT (chemprop_tuned/c) was trained with 5 thermo-only features
    cmd = ["chemprop_predict",
           "--test_path", "data/chemprop_tuned/c/test.csv",
           "--features_path", "data/chemprop_tuned/c/test_features.csv",
           "--checkpoint_dir", "checkpoints/chemprop_tuned/c",
           "--preds_path", out_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode == 0 and Path(out_path).exists():
        return pd.read_csv(out_path)[TARGET_COLUMNS].values
    print(f"  STILT predict error: {result.stderr[-200:]}")
    return None


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    pc_dir = "data/pipeline/point_clouds"
    orig_splits = Path("data/processed/splits")

    # ══════════════════════════════════════════════════════════
    print("Collecting predictions from 3 specialist models...\n")
    # ══════════════════════════════════════════════════════════

    print("  1. COSMOBridge (gamma1, gamma2, P)...")
    cosmo_preds, targets = get_cosmobridge_preds(device, orig_splits, pc_dir)
    print(f"     Shape: {cosmo_preds.shape}")

    print("  2. Chemprop (H_E)...")
    chemp_preds = get_chemprop_preds()
    print(f"     Shape: {chemp_preds.shape}" if chemp_preds is not None else "     Failed")

    print("  3. STILT (G_E, G_mix, H_vap)...")
    stilt_preds = get_stilt_preds()
    print(f"     Shape: {stilt_preds.shape}" if stilt_preds is not None else "     Failed")

    # ══════════════════════════════════════════════════════════
    # Per-property routing
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PER-PROPERTY ROUTING")
    print(f"{'='*60}")

    # Routing table: property → (model_name, prediction_array, column_index)
    routing = {
        "gamma1": ("COSMOBridge", cosmo_preds, 0),
        "gamma2": ("COSMOBridge", cosmo_preds, 1),
        "G_E":    ("STILT", stilt_preds if stilt_preds is not None else cosmo_preds, 2),
        "H_E":    ("Chemprop", chemp_preds if chemp_preds is not None else cosmo_preds, 3),
        "G_mix":  ("STILT", stilt_preds if stilt_preds is not None else cosmo_preds, 4),
        "H_vap":  ("STILT", stilt_preds if stilt_preds is not None else cosmo_preds, 5),
        "P":      ("COSMOBridge", cosmo_preds, 6),
    }

    # Build routed predictions
    routed_preds = np.zeros_like(targets)
    print(f"\n  Routing table:")
    for i, p in enumerate(TARGET_COLUMNS):
        model_name, preds_array, col_idx = routing[p]
        routed_preds[:, i] = preds_array[:, col_idx]
        print(f"    {p:15s} → {model_name}")

    # Evaluate
    metrics_routed = compute_metrics(routed_preds, targets)
    print(f"\n{format_metrics(metrics_routed, 'Per-Property Router')}")

    # ══════════════════════════════════════════════════════════
    # Individual model metrics for comparison
    # ══════════════════════════════════════════════════════════
    metrics_cosmo = compute_metrics(cosmo_preds, targets)
    metrics_chemp = compute_metrics(chemp_preds, targets) if chemp_preds is not None else {}
    metrics_stilt = compute_metrics(stilt_preds, targets) if stilt_preds is not None else {}

    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    all_models = {
        "COSMOBridge": metrics_cosmo,
        "Chemprop": metrics_chemp,
        "STILT": metrics_stilt,
        "Router": metrics_routed,
    }

    # Add previous results
    for name, path, key in [
        ("Ens Top-2", "results/ensemble_all_models_results.json", "ENS"),
    ]:
        try:
            data = json.load(open(path))
            if key == "ENS": m = data.get("top2_average", {}).get("metrics", {})
            all_models[name] = m
        except: pass

    header = "  {:<15s}".format("Property")
    for name in all_models:
        header += " {:>12s}".format(name[:12])
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<15s}".format(p)
        for name, m in all_models.items():
            v = m.get(key, float('nan'))
            line += " {:12.4f}".format(v)
        print(line)

    line = "  {:<15s}".format("AVERAGE")
    for name, m in all_models.items():
        v = m.get('avg_r2', float('nan'))
        line += " {:12.4f}".format(v)
    print(line)

    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULT")
    print(f"{'='*60}")
    print(f"\n  Per-Property Router avg R² = {metrics_routed['avg_r2']:.4f}")
    print(f"\n  Beats:")
    for name in ["Ens Top-2", "STILT", "Chemprop", "COSMOBridge"]:
        if name in all_models:
            v = all_models[name].get("avg_r2", 0)
            d = metrics_routed["avg_r2"] - v
            print(f"    {name:<15s}: {v:.4f} ({'+' if d>0 else ''}{d:.4f})")

    # Save
    results = {
        "model": "per_property_router",
        "description": "Routes each property to its best specialist model: "
                       "COSMOBridge (gamma1, gamma2, P), STILT (G_E, G_mix, H_vap), "
                       "Chemprop (H_E)",
        "routing": {p: routing[p][0] for p in TARGET_COLUMNS},
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics_routed.items()},
    }
    with open("results/per_property_router_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/per_property_router_results.json")


if __name__ == "__main__":
    main()
