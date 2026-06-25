"""Staged Dual-Path: Freeze pre-trained paths, train only gates + heads.

Why the original dual-path failed: 1.4M params trained jointly, cross-attention
dominated gradients, gates never learned, bilinear path was ignored.

Fix: Load pre-trained frozen encoders from our TWO BEST models:
  Path A: CrossAttn PointCloud checkpoint (gamma1=0.887)
  Path B: GBH v2+STILT checkpoint (gamma2=0.852, avg=0.729)

Then train ONLY:
  - 7 per-property gate parameters (which path to use)
  - Per-property prediction heads (~1K params)

Total trainable: ~1K params on 10,950 samples = 0.1 params/sample
This is the ideal regime — pure routing optimization, no gradient competition.

Stage 1: Load both frozen backbones, extract fused representations
Stage 2: Train gates + heads to optimally route each property
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
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.multimodal_pointcloud import PointCloudFusion, MultimodalPointCloudModel
from src.models.fusion.gated_bilinear_hyper_v2 import GatedBilinearHyperFusionV2
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud
from scripts.train_joint import MergedDataset, collate_merged


class StagedDualPath(nn.Module):
    """Staged dual-path: frozen pre-trained encoders + trainable routing.

    Path A: CrossAttn PointCloud (frozen, original feature space) → preds_A
    Path B: GBH v2+STILT (frozen, merged_v5 feature space) → preds_B

    Feature normalization: Path B receives features re-normalized from
    original space → merged_v5 space via stored scaler statistics.

    Trainable:
      - 7 gate logits α_p (per property)
    """

    def __init__(self, path_a_model, path_b_model,
                 orig_feat_mean, orig_feat_scale,
                 merged_feat_mean, merged_feat_scale,
                 orig_target_mean, orig_target_scale,
                 merged_target_mean, merged_target_scale):
        super().__init__()

        self.path_a = path_a_model
        self.path_b = path_b_model
        for p in self.path_a.parameters():
            p.requires_grad = False
        for p in self.path_b.parameters():
            p.requires_grad = False

        # Feature re-normalization buffers (orig space → merged space)
        self.register_buffer("orig_feat_mean", orig_feat_mean)
        self.register_buffer("orig_feat_scale", orig_feat_scale)
        self.register_buffer("merged_feat_mean", merged_feat_mean)
        self.register_buffer("merged_feat_scale", merged_feat_scale)

        # Target re-normalization buffers (merged predictions → orig space)
        self.register_buffer("orig_target_mean", orig_target_mean)
        self.register_buffer("orig_target_scale", orig_target_scale)
        self.register_buffer("merged_target_mean", merged_target_mean)
        self.register_buffer("merged_target_scale", merged_target_scale)

        self.gate_logits = nn.Parameter(torch.tensor([
            1.5,   # gamma1 → path_a
            -1.0,  # gamma2 → path_b
            -0.5,  # G_E → slight path_b
            -0.5,  # H_E → slight path_b
            -0.5,  # G_mix → slight path_b
            -1.5,  # H_vap → path_b
            -1.5,  # P → path_b
        ]))

    def _renorm_features(self, features):
        """Convert features from original normalization to merged_v5 normalization."""
        # Inverse original: raw = features * orig_scale + orig_mean
        raw = features * self.orig_feat_scale + self.orig_feat_mean
        # Forward merged: norm = (raw - merged_mean) / merged_scale
        return (raw - self.merged_feat_mean) / self.merged_feat_scale

    def _renorm_targets(self, preds_merged):
        """Convert predictions from merged_v5 target space to original target space."""
        # Inverse merged: raw = preds * merged_scale + merged_mean
        raw = preds_merged * self.merged_target_scale + self.merged_target_mean
        # Forward original: norm = (raw - orig_mean) / orig_scale
        return (raw - self.orig_target_mean) / self.orig_target_scale

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        with torch.no_grad():
            # Path A: original feature space
            preds_a = self.path_a(
                point_cloud=point_cloud, features=features,
                atom_features=atom_features, edge_index=edge_index,
                bond_features=bond_features, batch=batch)
            if isinstance(preds_a, tuple):
                preds_a = preds_a[0]

            # Path B: re-normalize features to merged_v5 space
            features_merged = self._renorm_features(features)
            out_b = self.path_b(
                point_cloud=point_cloud, features=features_merged,
                atom_features=atom_features, edge_index=edge_index,
                bond_features=bond_features, batch=batch)
            if isinstance(out_b, tuple):
                preds_b = out_b[0]
            else:
                preds_b = out_b

            # Re-normalize Path B predictions to original target space
            preds_b = self._renorm_targets(preds_b)

        # Per-property gated combination (both now in original target space)
        alpha = torch.sigmoid(self.gate_logits)
        predictions = alpha.unsqueeze(0) * preds_a + (1 - alpha.unsqueeze(0)) * preds_b

        return predictions, {"gate_values": alpha.detach()}


class StagedDualPathV2(nn.Module):
    """V2: Extract fused representations (not final predictions) from both paths,
    then route through trainable heads.

    This is more expressive than V1 because the heads can learn non-linear
    combinations of both paths' internal representations.
    """

    def __init__(self):
        super().__init__()

        # Path A components (will be loaded from checkpoint)
        self.pointnet_a = PointNetEncoder(in_channels=7, feature_dim=256, dropout=0.3)
        self.gnn_a = MolecularGNN(
            atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
            dropout=0.3, pooling="mean", num_targets=0)
        self.fusion_a = PointCloudFusion(
            pointcloud_dim=256, graph_dim=256, tabular_dim=25,
            fused_dim=256, num_heads=8, dropout=0.3)

        # Path B components (will be loaded from checkpoint)
        self.pointnet_b = PointNetEncoder(in_channels=7, feature_dim=256, dropout=0.3)
        self.gnn_b = MolecularGNN(
            atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
            dropout=0.3, pooling="mean", num_targets=0)
        self.fusion_b = GatedBilinearHyperFusionV2(
            pointcloud_dim=256, graph_dim=256, tabular_dim=25,
            fused_dim=256, rank=32, thermo_dim=5, hyper_hidden=64, dropout=0.3)

        # Per-property gate
        self.gate_logits = nn.Parameter(torch.tensor([
            1.5, -1.0, -0.5, -0.5, -0.5, -1.5, -1.5
        ]))

        # Trainable heads operating on gated 256D representation
        self.shared_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.prop_heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(7)])

    def load_pretrained(self, crossattn_ckpt, gbh_ckpt, device):
        """Load pre-trained weights from both checkpoints."""
        # CrossAttn PointCloud checkpoint
        ca_state = torch.load(crossattn_ckpt, map_location=device, weights_only=False)
        if "model_state_dict" in ca_state:
            ca_state = ca_state["model_state_dict"]

        # Map CrossAttn weights to path_a
        mapped = {}
        for k, v in ca_state.items():
            if k.startswith("pointnet."):
                mapped[k.replace("pointnet.", "pointnet_a.")] = v
            elif k.startswith("gnn."):
                mapped[k.replace("gnn.", "gnn_a.")] = v
            elif k.startswith("fusion."):
                mapped[k.replace("fusion.", "fusion_a.")] = v

        # GBH v2+STILT checkpoint
        gbh_state = torch.load(gbh_ckpt, map_location=device, weights_only=True)
        for k, v in gbh_state.items():
            if k.startswith("pointnet."):
                mapped[k.replace("pointnet.", "pointnet_b.")] = v
            elif k.startswith("gnn."):
                mapped[k.replace("gnn.", "gnn_b.")] = v
            elif k.startswith("fusion."):
                mapped[k.replace("fusion.", "fusion_b.")] = v

        missing, unexpected = self.load_state_dict(mapped, strict=False)
        print(f"  Loaded {len(mapped)} params from checkpoints")
        print(f"  Missing (trainable heads): {len(missing)}")

        # Freeze all loaded params
        for name, p in self.named_parameters():
            if name.startswith(("pointnet_", "gnn_", "fusion_")):
                p.requires_grad = False

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        # Path A: CrossAttn fusion
        with torch.no_grad():
            pc_a = self.pointnet_a(point_cloud)
            g_a = self.gnn_a.get_features(atom_features, edge_index, bond_features, batch)
            h_a = self.fusion_a(pc_a, g_a, features)  # (B, 256)

        # Path B: GBH bilinear fusion
        with torch.no_grad():
            pc_b = self.pointnet_b(point_cloud)
            g_b = self.gnn_b.get_features(atom_features, edge_index, bond_features, batch)
            h_b = self.fusion_b(pc_b, g_b, features)  # (B, 256)

        # Per-property gated mixing of fused representations
        alpha = torch.sigmoid(self.gate_logits)  # (7,)

        # Predict each property from its gated representation
        preds = []
        for p in range(7):
            h_p = alpha[p] * h_a + (1 - alpha[p]) * h_b  # (B, 256)
            h_p = self.shared_head(h_p)
            preds.append(self.prop_heads[p](h_p))

        predictions = torch.cat(preds, dim=1)  # (B, 7)
        return predictions, {"gate_values": alpha.detach()}


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            preds, _ = model(point_cloud=batch["point_cloud"], features=batch["features"],
                             atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                             bond_features=batch["bond_features"], batch=batch["batch"])
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

    # Use original data for training (both paths already learned from STILT data)
    print("Loading original test data...")
    test_ds = PointCloudMultimodalDataset(str(orig_splits / "test.csv"), pc_dir, is_train=False)
    val_ds = PointCloudMultimodalDataset(str(orig_splits / "val.csv"), pc_dir, is_train=False)
    train_ds = PointCloudMultimodalDataset(str(orig_splits / "train.csv"), pc_dir, is_train=True)

    train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_pointcloud)
    val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)
    test_ldr = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)

    # ══════════════════════════════════════════════════════════
    # METHOD 1: Prediction-level gating (simplest)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("METHOD 1: Prediction-Level Gating (frozen paths, train gates only)")
    print(f"{'='*60}")

    # Load Path A: CrossAttn PointCloud
    path_a = MultimodalPointCloudModel(config=config, pretrained_gnn_path=None)
    ca_ckpt = torch.load("checkpoints/pointcloud/best_model.pt", map_location=device, weights_only=False)
    path_a.load_state_dict(ca_ckpt["model_state_dict"] if "model_state_dict" in ca_ckpt else ca_ckpt)
    path_a.to(device).eval()
    print(f"  Path A (CrossAttn): loaded, gamma1=0.887")

    # Load Path B: GBH v2+STILT
    # Need to reconstruct the GBH v2 model with original feature dim
    class GBHv2PC_Orig(nn.Module):
        def __init__(self):
            super().__init__()
            self.pointnet = PointNetEncoder(in_channels=7, feature_dim=256, dropout=0.3)
            self.gnn = MolecularGNN(atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
                                     hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
                                     dropout=0.3, pooling="mean", num_targets=0)
            self.fusion = GatedBilinearHyperFusionV2(
                pointcloud_dim=256, graph_dim=256, tabular_dim=25,
                fused_dim=256, rank=32, thermo_dim=5, hyper_hidden=64, dropout=0.3)
            self.prediction_head = nn.Sequential(
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(),
                nn.Dropout(0.3), nn.Linear(128, 7))
        def forward(self, point_cloud, features, atom_features, edge_index, bond_features, batch, **kw):
            pc = self.pointnet(point_cloud); g = self.gnn.get_features(atom_features, edge_index, bond_features, batch)
            f = self.fusion(pc, g, features); return self.prediction_head(f)

    # GBH v2+STILT was trained with merged_v5 features (25D) but we need original features (25D)
    # Both use the same FEATURE_COLUMNS, so we can load directly
    path_b = GBHv2PC_Orig()
    try:
        gbh_state = torch.load("checkpoints/gbh_v2_stilt/best.pt", map_location=device, weights_only=True)
        path_b.load_state_dict(gbh_state)
        print(f"  Path B (GBH v2+STILT): loaded, avg=0.729")
    except Exception as e:
        print(f"  Path B loading error: {e}")
        print(f"  Trying with strict=False...")
        path_b.load_state_dict(gbh_state, strict=False)
    path_b.to(device).eval()

    # Load scalers for feature/target re-normalization
    import pickle
    with open("data/processed/feature_scaler.pkl", "rb") as f:
        orig_fs = pickle.load(f)
    with open("data/processed/target_scaler.pkl", "rb") as f:
        orig_ts = pickle.load(f)
    with open("data/merged_v5/feature_scaler.pkl", "rb") as f:
        merged_fs = pickle.load(f)
    with open("data/merged_v5/target_scalers.pkl", "rb") as f:
        merged_ts_dict = pickle.load(f)

    merged_ts_mean = torch.tensor([merged_ts_dict[c].mean_[0] for c in TARGET_COLUMNS], dtype=torch.float32)
    merged_ts_scale = torch.tensor([merged_ts_dict[c].scale_[0] for c in TARGET_COLUMNS], dtype=torch.float32)

    # Build staged model with scaler info (use only first 25 dims of merged scaler)
    n_feat = len(FEATURE_COLUMNS)  # 25
    model1 = StagedDualPath(
        path_a, path_b,
        orig_feat_mean=torch.tensor(orig_fs.mean_[:n_feat], dtype=torch.float32),
        orig_feat_scale=torch.tensor(orig_fs.scale_[:n_feat], dtype=torch.float32),
        merged_feat_mean=torch.tensor(merged_fs.mean_[:n_feat], dtype=torch.float32),
        merged_feat_scale=torch.tensor(merged_fs.scale_[:n_feat], dtype=torch.float32),
        orig_target_mean=torch.tensor(orig_ts.mean_, dtype=torch.float32),
        orig_target_scale=torch.tensor(orig_ts.scale_, dtype=torch.float32),
        merged_target_mean=merged_ts_mean,
        merged_target_scale=merged_ts_scale,
    )
    model1.to(device)

    n_trainable = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model1.parameters())
    print(f"  Total params: {n_total:,}, Trainable: {n_trainable:,}")
    print(f"  Initial gates: {torch.sigmoid(model1.gate_logits).detach().cpu().numpy()}")

    # Train ONLY gates (7 params)
    ckpt_dir = Path("checkpoints/staged_dual_path")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW([model1.gate_logits], lr=0.1, weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)

    best_loss, no_improve = float("inf"), 0
    for epoch in range(100):
        model1.train()
        tl, n = 0, 0
        for batch in train_ldr:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            optimizer.zero_grad()
            preds, _ = model1(point_cloud=batch["point_cloud"], features=batch["features"],
                              atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                              bond_features=batch["bond_features"], batch=batch["batch"])
            loss = ((preds - batch["targets"])**2).mean()
            loss.backward()
            optimizer.step()
            tl += loss.item(); n += 1
        scheduler.step()

        model1.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for batch in val_ldr:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor): batch[k] = v.to(device)
                preds, _ = model1(point_cloud=batch["point_cloud"], features=batch["features"],
                                  atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                                  bond_features=batch["bond_features"], batch=batch["batch"])
                vl += ((preds - batch["targets"])**2).mean().item()
                vn += 1
        avg_val = vl / max(vn, 1)
        if avg_val < best_loss:
            best_loss = avg_val; no_improve = 0
            torch.save(model1.gate_logits.data, ckpt_dir / "best_gates_v1.pt")
        else:
            no_improve += 1
        if epoch % 20 == 0:
            gates = torch.sigmoid(model1.gate_logits).detach().cpu().numpy()
            g_str = " ".join(f"{g:.2f}" for g in gates)
            print(f"  Epoch {epoch:3d} | Train:{tl/max(n,1):.4f} Val:{avg_val:.4f} "
                  f"Best:{best_loss:.4f} Pat:{no_improve}/30 | Gates:[{g_str}]")
        if no_improve >= 30: print(f"  Early stopping at epoch {epoch}"); break

    model1.gate_logits.data = torch.load(ckpt_dir / "best_gates_v1.pt", map_location=device)

    # Evaluate
    preds1, targets1 = evaluate(model1, test_ldr, device)
    metrics1 = compute_metrics(preds1, targets1)
    print(f"\n{format_metrics(metrics1, 'Staged Dual-Path v1 (prediction gating)')}")

    gates1 = torch.sigmoid(model1.gate_logits).detach().cpu().numpy()
    print(f"\n  Learned gates (α: 1=CrossAttn, 0=GBH):")
    for i, p in enumerate(TARGET_COLUMNS):
        path = "CrossAttn" if gates1[i] > 0.6 else ("GBH" if gates1[i] < 0.4 else "mixed")
        bar = "█" * int(gates1[i] * 20) + "░" * (20 - int(gates1[i] * 20))
        print(f"    {p:15s}: α={gates1[i]:.3f} [{bar}] → {path}")

    # ══════════════════════════════════════════════════════════
    # COMPARISON
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    prev = {}
    for name, path, key in [
        ("Chemprop", "results/chemprop_results.json", "test_metrics"),
        ("CrossAttn PC", "results/pointcloud_results.json", None),
        ("GBH v2+STILT", "results/gbh_v2_stilt_results.json", "metrics"),
        ("GBH v3", "results/gbh_v3_results.json", "metrics"),
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
    header += " {:>12s}".format("Staged v1")
    print(header)
    print("  " + "-" * len(header))

    for p in TARGET_COLUMNS:
        key = f"{p}_r2"
        line = "  {:<12s}".format(p)
        for name in prev:
            line += " {:12.4f}".format(prev[name].get(key, float('nan')))
        line += " {:12.4f}".format(metrics1[key])
        print(line)

    line = "  {:<12s}".format("AVERAGE")
    for name in prev:
        line += " {:12.4f}".format(prev[name].get('avg_r2', float('nan')))
    line += " {:12.4f}".format(metrics1['avg_r2'])
    print(line)

    # vs Chemprop
    if "Chemprop" in prev:
        base = prev["Chemprop"]
        print(f"\n  vs Chemprop:")
        wins = 0
        for p in TARGET_COLUMNS:
            key = f"{p}_r2"
            d = metrics1[key] - base[key]
            if d > 0: wins += 1
            s = "+" if d > 0 else ""
            w = "WIN" if d > 0 else ("~tied" if abs(d) < 0.01 else "lose")
            print(f"    {p:15s}: {metrics1[key]:.4f} vs {base[key]:.4f} ({s}{d:.4f}) {w}")
        d = metrics1['avg_r2'] - base['avg_r2']
        s = "+" if d > 0 else ""
        print(f"    {'AVERAGE':15s}: {metrics1['avg_r2']:.4f} vs {base['avg_r2']:.4f} ({s}{d:.4f}) wins {wins}/7")

    # Save
    results = {
        "model": "staged_dual_path",
        "description": "Frozen CrossAttn PointCloud + frozen GBH v2+STILT, "
                       "trainable per-property gates (7 params) combining predictions",
        "n_trainable": n_trainable,
        "gate_values": {p: float(g) for p, g in zip(TARGET_COLUMNS, gates1)},
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics1.items()},
    }
    with open("results/staged_dual_path_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/staged_dual_path_results.json")


if __name__ == "__main__":
    main()
