"""MoE-A + Fix2: Freeze-and-Finetune gamma1 head.

Stage 1: Train MoE-A on merged_v3 (all 7 targets)
Stage 2: Freeze all, unfreeze gamma1 output weights, fine-tune on original 223 samples
"""

import sys
import json
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.moe import ExpertHead, PropertyConditionedGating
from src.models.fusion.multimodal_pointcloud import PointCloudFusion
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_joint import MergedDataset, collate_merged
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud
from scripts.train_moe import evaluate_single

TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


class MoEA(nn.Module):
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
        self.experts = nn.ModuleList([
            ExpertHead(fused_dim, hidden_dim=128, num_targets=7, dropout=dropout)
            for _ in range(4)])
        self.gating = PropertyConditionedGating(
            input_dim=fused_dim, num_experts=4, num_properties=7, hidden_dim=64)

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        pc_feat = self.pointnet(point_cloud)
        graph_feat = self.gnn.get_features(atom_features, edge_index, bond_features, batch)
        fused = self.fusion(pc_feat, graph_feat, features)
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

    # ── Stage 1: Train on merged_v3 ──
    print(f"\n{'='*60}")
    print("STAGE 1: Train MoE-A on merged_v3")
    print(f"{'='*60}")

    train_m = MergedDataset(str(merged_dir / "splits/train.csv"), pc_dir, merged_features, is_train=True)
    val_m = MergedDataset(str(merged_dir / "splits/val.csv"), pc_dir, merged_features, is_train=False)
    test_m = MergedDataset(str(merged_dir / "splits/test.csv"), pc_dir, merged_features, is_train=False)

    train_ldr_m = DataLoader(train_m, batch_size=64, shuffle=True, collate_fn=collate_merged)
    val_ldr_m = DataLoader(val_m, batch_size=32, shuffle=False, collate_fn=collate_merged)
    test_ldr_m = DataLoader(test_m, batch_size=32, shuffle=False, collate_fn=collate_merged)

    model = MoEA(feature_dim=len(merged_features),
                 pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
    model.to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_dir = Path("checkpoints/moe_fix2")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=60, eta_min=1e-6)

    # Masked MSE for merged data
    best_loss, no_improve = float("inf"), 0
    for epoch in range(200):
        model.train()
        tl, n = 0, 0
        for batch in train_ldr_m:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            optimizer.zero_grad()
            preds, aux = model(point_cloud=batch["point_cloud"], features=batch["features"],
                               atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                               bond_features=batch["bond_features"], batch=batch["batch"])
            mask = ~torch.isnan(batch["targets"])
            safe = batch["targets"].clone(); safe[~mask] = 0.0
            diff2 = (preds - safe)**2 * mask.float()
            loss = diff2.sum() / mask.float().sum().clamp(min=1) + aux["load_balance_loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item(); n += 1
        scheduler.step()

        model.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for batch in val_ldr_m:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                preds, _ = model(point_cloud=batch["point_cloud"], features=batch["features"],
                                 atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                                 bond_features=batch["bond_features"], batch=batch["batch"])
                mask = ~torch.isnan(batch["targets"])
                safe = batch["targets"].clone(); safe[~mask] = 0.0
                vl += ((preds-safe)**2 * mask.float()).sum().item() / mask.float().sum().clamp(min=1).item()
                vn += 1
        avg_val = vl / max(vn, 1)
        if avg_val < best_loss:
            best_loss = avg_val; no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "stage1.pt")
        else:
            no_improve += 1
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/200 | Train: {tl/max(n,1):.4f} | Val: {avg_val:.4f} | Best: {best_loss:.4f} | Pat: {no_improve}/25")
        if no_improve >= 25:
            print(f"  Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(ckpt_dir / "stage1.pt", map_location=device, weights_only=True))

    # Evaluate before fine-tune
    preds_before, targets_before, _ = evaluate_single(model, test_ldr_m, device)
    m_before = compute_metrics(preds_before, targets_before)
    print(f"\n  Before FT: avg R²={m_before['avg_r2']:.4f}, gamma1={m_before['gamma1_r2']:.4f}")

    # ── Stage 2: Freeze all, fine-tune gamma1 on original ──
    print(f"\n{'='*60}")
    print("STAGE 2: Fine-tune gamma1 on original data")
    print(f"{'='*60}")

    orig_splits = Path("data/processed/splits")
    train_o = PointCloudMultimodalDataset(str(orig_splits / "train.csv"), pc_dir, is_train=True)
    val_o = PointCloudMultimodalDataset(str(orig_splits / "val.csv"), pc_dir, is_train=False)
    test_o = PointCloudMultimodalDataset(str(orig_splits / "test.csv"), pc_dir, is_train=False)
    train_ldr_o = DataLoader(train_o, batch_size=32, shuffle=True, collate_fn=collate_pointcloud)
    test_ldr_o = DataLoader(test_o, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze expert last layers + gating property embeddings
    for expert in model.experts:
        for p in expert.net[-1].parameters():
            p.requires_grad = True
    model.gating.property_embed.weight.requires_grad = True

    n_ft = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Fine-tuning {n_ft:,} params")

    optimizer2 = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=30, eta_min=1e-6)

    best_g1 = float("inf")
    for epoch in range(30):
        model.train()
        tl, n = 0, 0
        for batch in train_ldr_o:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            optimizer2.zero_grad()
            preds, _ = model(point_cloud=batch["point_cloud"], features=batch["features"],
                             atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                             bond_features=batch["bond_features"], batch=batch["batch"])
            loss = ((preds[:, 0] - batch["targets"][:, 0])**2).mean()
            loss.backward()
            optimizer2.step()
            tl += loss.item(); n += 1
        scheduler2.step()
        avg = tl / max(n, 1)
        if avg < best_g1:
            best_g1 = avg
            torch.save(model.state_dict(), ckpt_dir / "stage2.pt")
        if epoch % 10 == 0:
            print(f"  FT Epoch {epoch:3d}/30 | gamma1 loss: {avg:.4f}")

    model.load_state_dict(torch.load(ckpt_dir / "stage2.pt", map_location=device, weights_only=True))
    for p in model.parameters():
        p.requires_grad = True

    # ── Final evaluation ──
    print(f"\n{'='*60}")
    print("FINAL EVALUATION (Fix 2: Freeze-and-Finetune)")
    print(f"{'='*60}")

    # Gamma1 from original test, others from merged test
    preds_o, targets_o, _ = evaluate_single(model, test_ldr_o, device)
    preds_m, targets_m, gw = evaluate_single(model, test_ldr_m, device)
    m_orig = compute_metrics(preds_o, targets_o)
    m_merged = compute_metrics(preds_m, targets_m)

    print(f"\n  gamma1 (original test): R² = {m_orig['gamma1_r2']:.4f}")
    print(f"  Other props (merged test):")
    for p in TARGET_COLUMNS[1:]:
        print(f"    {p:15s} R² = {m_merged[f'{p}_r2']:.4f}")

    # Hybrid: gamma1 from orig, rest from merged
    hybrid = {"gamma1_r2": m_orig["gamma1_r2"]}
    for p in TARGET_COLUMNS[1:]:
        hybrid[f"{p}_r2"] = m_merged[f"{p}_r2"]
    hybrid["avg_r2"] = np.mean([hybrid[f"{p}_r2"] for p in TARGET_COLUMNS])
    print(f"\n  HYBRID AVG R²: {hybrid['avg_r2']:.4f}")

    with open("results/moe_fix2_results.json", "w") as f:
        json.dump({"fix": "freeze_finetune_gamma1", "hybrid": hybrid,
                   "before_ft": {k: float(v) for k, v in m_before.items()},
                   "after_ft_orig": {k: float(v) for k, v in m_orig.items()},
                   "after_ft_merged": {k: float(v) for k, v in m_merged.items()},
                   "gate_weights": gw.mean(axis=0).tolist()}, f, indent=2)
    print("Saved: results/moe_fix2_results.json")


if __name__ == "__main__":
    main()
