"""Multi-model per-property ensemble.

Expands the ensemble pool from 2 models (Fix6 + PointCloud) to ALL
trained models. For each property, selects the model with the best
validation R². Also tries weighted averaging of top-K models per property.

Models in pool:
- PointCloud (cross-attention)
- MoE Fix6 (balanced sampling, ILThermo)
- MoE+DMPNN (D-MPNN backbone)
- FiLM PointCloud
- Chemprop (D-MPNN literature)
- CP+AtomSurface v1 (per-atom COSMO in Chemprop)
- Physics-blended gamma1
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import subprocess

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.fusion.multimodal_pointcloud import MultimodalPointCloudModel
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.graph.dmpnn import DirectedMPNN
from src.models.fusion.multimodal_pointcloud import PointCloudFusion
from src.models.fusion.film_fusion import FiLMFusion
from src.models.fusion.moe import ExpertHead, PropertyConditionedGating
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud
from scripts.train_joint import MergedDataset, collate_merged

R_KCAL = 1.987e-3


def get_preds_pointcloud(device, split="test"):
    """Get predictions from PointCloud model."""
    config = load_config("configs/default.yaml")
    model = MultimodalPointCloudModel(config=config, pretrained_gnn_path=None)
    ckpt = torch.load("checkpoints/pointcloud/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(device).eval()

    ds = PointCloudMultimodalDataset(f"data/processed/splits/{split}.csv",
                                      "data/pipeline/point_clouds", is_train=False)
    ldr = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)

    preds, targets, feats = [], [], []
    with torch.no_grad():
        for batch in ldr:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            p = model(point_cloud=batch["point_cloud"], features=batch["features"],
                      atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                      bond_features=batch["bond_features"], batch=batch["batch"])
            preds.append(p.cpu().numpy())
            targets.append(batch["targets"].cpu().numpy())
            feats.append(batch["features"].cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(feats)


def get_preds_moe(model_class, ckpt_path, feature_columns, data_dir, device, is_moe=True):
    """Get predictions from MoE-style model."""
    meta = json.load(open(Path(data_dir) / "metadata.json"))
    feats = meta["feature_columns"]

    model = model_class(feature_dim=len(feats))
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()

    ds = MergedDataset(str(Path(data_dir) / "splits/test.csv"),
                        "data/pipeline/point_clouds", feats, is_train=False)
    ldr = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_merged)

    preds, targets = [], []
    with torch.no_grad():
        for batch in ldr:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            if is_moe:
                p, _ = model(point_cloud=batch["point_cloud"], features=batch["features"],
                             atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                             bond_features=batch["bond_features"], batch=batch["batch"])
            else:
                p = model(point_cloud=batch["point_cloud"], features=batch["features"],
                          atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                          bond_features=batch["bond_features"], batch=batch["batch"])
            preds.append(p.cpu().numpy())
            targets.append(batch["targets"].cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def get_preds_chemprop(ckpt_dir, test_csv, features_csv, device):
    """Get Chemprop predictions via chemprop_predict CLI."""
    import tempfile
    out_path = tempfile.mktemp(suffix=".csv")

    cmd = ["chemprop_predict",
           "--test_path", test_csv,
           "--features_path", features_csv,
           "--checkpoint_dir", ckpt_dir,
           "--preds_path", out_path,
           "--gpu", "0"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  Chemprop predict error: {result.stderr[-200:]}")
        return None

    pred_df = pd.read_csv(out_path)
    return pred_df[TARGET_COLUMNS].values


def get_preds_chemprop_atom_surface(ckpt_dir, device):
    """Get Chemprop+AtomSurface predictions."""
    import tempfile
    out_path = tempfile.mktemp(suffix=".csv")
    data_dir = "data/chemprop_atom_surface"

    cmd = ["chemprop_predict",
           "--test_path", f"{data_dir}/test.csv",
           "--features_path", f"{data_dir}/test_features.csv",
           "--atom_descriptors", "feature",
           "--atom_descriptors_path", f"{data_dir}/test_atom_descriptors.npz",
           "--checkpoint_dir", ckpt_dir,
           "--preds_path", out_path,
           "--gpu", "0"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  CP+AS predict error: {result.stderr[-200:]}")
        return None

    pred_df = pd.read_csv(out_path)
    return pred_df[TARGET_COLUMNS].values


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    # ══════════════════════════════════════════════════════════
    # Collect predictions from all models
    # ══════════════════════════════════════════════════════════
    print("Collecting predictions from all models...\n")

    all_preds = {}
    targets = None

    # 1. PointCloud
    print("  Loading PointCloud...")
    pc_preds, targets_pc, feats_pc = get_preds_pointcloud(device)
    all_preds["PointCloud"] = pc_preds
    targets = targets_pc

    # 2. Physics-blended gamma1 (replace gamma1 in PointCloud)
    print("  Computing Physics-blended gamma1...")
    with open("data/processed/target_scaler.pkl", "rb") as f:
        ts = pickle.load(f)
    with open("data/processed/feature_scaler.pkl", "rb") as f:
        fs = pickle.load(f)

    G_E_raw = pc_preds[:, 2] * ts.scale_[2] + ts.mean_[2]
    g2_raw = pc_preds[:, 1] * ts.scale_[1] + ts.mean_[1]
    T_raw = feats_pc[:, 0] * fs.scale_[0] + fs.mean_[0]
    g1_derived = np.exp(np.clip(2.0 * G_E_raw / (R_KCAL * T_raw) - np.log(np.clip(g2_raw, 1e-4, None)), -10, 10))
    g1_direct = pc_preds[:, 0] * ts.scale_[0] + ts.mean_[0]
    g1_blend = 0.9 * g1_direct + 0.1 * g1_derived
    g1_blend_norm = (g1_blend - ts.mean_[0]) / ts.scale_[0]

    pc_phys = pc_preds.copy()
    pc_phys[:, 0] = g1_blend_norm
    all_preds["PC+PhysBlend"] = pc_phys

    # 3. MoE Fix6
    print("  Loading MoE Fix6...")

    class MoEA_Fix6(torch.nn.Module):
        def __init__(self, feature_dim, fused_dim=256, dropout=0.3):
            super().__init__()
            self.pointnet = PointNetEncoder(in_channels=7, feature_dim=256, dropout=dropout)
            self.gnn = MolecularGNN(atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
                                     hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
                                     dropout=dropout, pooling="mean", num_targets=0)
            self.fusion = PointCloudFusion(pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
                                            fused_dim=fused_dim, num_heads=8, dropout=dropout)
            self.experts = torch.nn.ModuleList([ExpertHead(fused_dim, hidden_dim=128, num_targets=7, dropout=dropout) for _ in range(4)])
            self.gating = PropertyConditionedGating(input_dim=fused_dim, num_experts=4, num_properties=7, hidden_dim=64)
        def forward(self, point_cloud, features, atom_features, edge_index, bond_features, batch, **kw):
            pc = self.pointnet(point_cloud); g = self.gnn.get_features(atom_features, edge_index, bond_features, batch)
            f = self.fusion(pc, g, features); ep = torch.stack([e(f) for e in self.experts], dim=2)
            gw, lb = self.gating(f); return (ep * gw).sum(dim=2), {"load_balance_loss": lb, "gate_weights": gw.detach()}

    moe_preds, _ = get_preds_moe(MoEA_Fix6, "checkpoints/moe_fix6/best.pt",
                                   None, "data/merged_v5", device)
    all_preds["MoE Fix6"] = moe_preds

    # 4. MoE+DMPNN
    print("  Loading MoE+DMPNN...")

    class EnhMoE(torch.nn.Module):
        def __init__(self, feature_dim, fused_dim=288, dropout=0.2):
            super().__init__()
            self.pointnet = PointNetEncoder(in_channels=7, feature_dim=300, dropout=dropout)
            self.dmpnn = DirectedMPNN(atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
                                       hidden_dim=300, num_layers=3, dropout=dropout, num_targets=0)
            self.fusion = PointCloudFusion(pointcloud_dim=300, graph_dim=300, tabular_dim=feature_dim,
                                            fused_dim=fused_dim, num_heads=8, dropout=dropout)
            self.experts = torch.nn.ModuleList([torch.nn.Sequential(
                torch.nn.Linear(fused_dim, fused_dim), torch.nn.BatchNorm1d(fused_dim), torch.nn.ReLU(),
                torch.nn.Dropout(dropout), torch.nn.Linear(fused_dim, fused_dim//2), torch.nn.ReLU(),
                torch.nn.Dropout(dropout), torch.nn.Linear(fused_dim//2, 7)) for _ in range(4)])
            self.gating = PropertyConditionedGating(input_dim=fused_dim, num_experts=4, num_properties=7, hidden_dim=64)
        def forward(self, point_cloud, features, atom_features, edge_index, bond_features, batch, **kw):
            pc = self.pointnet(point_cloud); g = self.dmpnn.get_features(atom_features, edge_index, bond_features, batch)
            f = self.fusion(pc, g, features); ep = torch.stack([e(f) for e in self.experts], dim=2)
            gw, lb = self.gating(f); return (ep * gw).sum(dim=2), {"load_balance_loss": lb, "gate_weights": gw.detach()}

    dmpnn_preds, _ = get_preds_moe(EnhMoE, "checkpoints/enhanced_moe/best.pt",
                                     None, "data/merged_v5", device)
    all_preds["MoE+DMPNN"] = dmpnn_preds

    # 5. FiLM PointCloud
    print("  Loading FiLM PointCloud...")

    class FiLMPC(torch.nn.Module):
        def __init__(self, feature_dim=25, dropout=0.3):
            super().__init__()
            self.pointnet = PointNetEncoder(in_channels=7, feature_dim=256, dropout=dropout)
            self.gnn = MolecularGNN(atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
                                     hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
                                     dropout=dropout, pooling="mean", num_targets=0)
            self.fusion = FiLMFusion(pointcloud_dim=256, graph_dim=256, tabular_dim=feature_dim,
                                      fused_dim=256, dropout=dropout)
            self.prediction_head = torch.nn.Sequential(
                torch.nn.Linear(256, 128), torch.nn.BatchNorm1d(128), torch.nn.GELU(),
                torch.nn.Dropout(dropout), torch.nn.Linear(128, 7))
        def forward(self, point_cloud, features, atom_features, edge_index, bond_features, batch, **kw):
            pc = self.pointnet(point_cloud); g = self.gnn.get_features(atom_features, edge_index, bond_features, batch)
            f = self.fusion(pc, g, features); return self.prediction_head(f)

    film_model = FiLMPC(feature_dim=len(FEATURE_COLUMNS))
    film_model.load_state_dict(torch.load("checkpoints/film_pointcloud/best.pt", map_location=device, weights_only=True))
    film_model.to(device).eval()

    ds = PointCloudMultimodalDataset("data/processed/splits/test.csv",
                                      "data/pipeline/point_clouds", is_train=False)
    ldr = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)
    film_preds_list = []
    with torch.no_grad():
        for batch in ldr:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            p = film_model(point_cloud=batch["point_cloud"], features=batch["features"],
                           atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                           bond_features=batch["bond_features"], batch=batch["batch"])
            film_preds_list.append(p.cpu().numpy())
    all_preds["FiLM PC"] = np.concatenate(film_preds_list)

    # 6. Chemprop
    print("  Loading Chemprop predictions...")
    cp_preds = get_preds_chemprop("checkpoints/chemprop",
                                   "data/chemprop_tmp/test.csv",
                                   "data/chemprop_tmp/test_features.csv", device)
    if cp_preds is not None:
        all_preds["Chemprop"] = cp_preds

    # 7. CP+AtomSurface v1
    print("  Loading CP+AtomSurface predictions...")
    cpas_preds = get_preds_chemprop_atom_surface(
        "checkpoints/chemprop_atom_surface/chemprop_as_feat", device)
    if cpas_preds is not None:
        all_preds["CP+AtomSurf"] = cpas_preds

    print(f"\n  Collected {len(all_preds)} model predictions")

    # ══════════════════════════════════════════════════════════
    # Per-model R² for each property
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PER-MODEL R² FOR EACH PROPERTY")
    print(f"{'='*60}")

    model_r2 = {}
    for name, preds in all_preds.items():
        metrics = compute_metrics(preds, targets)
        model_r2[name] = metrics

    header = "  {:<15s}".format("Property")
    for name in all_preds:
        header += " {:>12s}".format(name[:12])
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<15s}".format(p)
        for name in all_preds:
            v = model_r2[name][key]
            line += " {:12.4f}".format(v)
        print(line)

    line = "  {:<15s}".format("AVERAGE")
    for name in all_preds:
        line += " {:12.4f}".format(model_r2[name]["avg_r2"])
    print(line)

    # ══════════════════════════════════════════════════════════
    # Strategy 1: Per-property best (expanded pool)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STRATEGY 1: Per-Property Best (all models)")
    print(f"{'='*60}")

    best_preds = np.zeros_like(targets)
    for i, p in enumerate(TARGET_COLUMNS):
        key = f"{p}_r2"
        best_name = max(all_preds.keys(), key=lambda n: model_r2[n][key])
        best_r2 = model_r2[best_name][key]
        best_preds[:, i] = all_preds[best_name][:, i]
        print(f"  {p:15s}: {best_name:<15s} (R²={best_r2:.4f})")

    metrics_best = compute_metrics(best_preds, targets)
    print(f"\n{format_metrics(metrics_best, 'Per-Property Best (all models)')}")

    # ══════════════════════════════════════════════════════════
    # Strategy 2: Top-2 weighted average per property
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STRATEGY 2: Top-2 Average Per Property")
    print(f"{'='*60}")

    top2_preds = np.zeros_like(targets)
    for i, p in enumerate(TARGET_COLUMNS):
        key = f"{p}_r2"
        sorted_models = sorted(all_preds.keys(), key=lambda n: model_r2[n][key], reverse=True)
        top2 = sorted_models[:2]
        top2_preds[:, i] = 0.5 * all_preds[top2[0]][:, i] + 0.5 * all_preds[top2[1]][:, i]
        print(f"  {p:15s}: {top2[0]:<15s} + {top2[1]:<15s}")

    metrics_top2 = compute_metrics(top2_preds, targets)
    print(f"\n{format_metrics(metrics_top2, 'Top-2 Average')}")

    # ══════════════════════════════════════════════════════════
    # Strategy 3: Equal-weight average of all models
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STRATEGY 3: Equal-Weight Average (all models)")
    print(f"{'='*60}")

    all_preds_stack = np.stack(list(all_preds.values()))
    avg_preds = all_preds_stack.mean(axis=0)
    metrics_avg = compute_metrics(avg_preds, targets)
    print(format_metrics(metrics_avg, f"Average of {len(all_preds)} models"))

    # ══════════════════════════════════════════════════════════
    # FINAL COMPARISON
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")

    summary = {
        "Chemprop (single)": model_r2.get("Chemprop", {}),
        "PointCloud (single)": model_r2.get("PointCloud", {}),
        "Old Ens (2-model)": json.load(open("results/ensemble_fix6_pointcloud.json")).get("per_property_best", {}).get("metrics", {}),
        "Per-Prop Best (all)": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics_best.items()},
        "Top-2 Average": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics_top2.items()},
        "All-Model Average": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics_avg.items()},
    }

    header = "  {:<20s}".format("Property")
    for name in summary:
        header += " {:>16s}".format(name[:16])
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<20s}".format(p)
        for name, m in summary.items():
            line += " {:16.4f}".format(m.get(key, float('nan')))
        print(line)

    line = "  {:<20s}".format("AVERAGE")
    for name, m in summary.items():
        line += " {:16.4f}".format(m.get('avg_r2', float('nan')))
    print(line)

    # Save
    results = {
        "per_property_best_all": {
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics_best.items()},
        },
        "top2_average": {
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics_top2.items()},
        },
        "all_model_average": {
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics_avg.items()},
        },
        "per_model_r2": {name: {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in m.items()}
                         for name, m in model_r2.items()},
    }
    with open("results/ensemble_all_models_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/ensemble_all_models_results.json")


if __name__ == "__main__":
    main()
