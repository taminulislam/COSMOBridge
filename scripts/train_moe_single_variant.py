"""Train a single MoE fusion variant. Called with --variant A/B/C/D."""

import sys
import json
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device, set_seed
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_joint import MergedDataset, collate_merged
from scripts.train_moe import train_moe, evaluate_single
from scripts.train_moe_ablation import MoEWithFusionVariant, build_fusion_variant

TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]

VARIANT_NAMES = {
    "A": "Cross-Attention",
    "B": "Physics Bottleneck",
    "C": "Hierarchical Multi-Scale",
    "D": "Gated Residual",
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=["A", "B", "C", "D"])
    args = parser.parse_args()

    vid = args.variant
    vname = VARIANT_NAMES[vid]

    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)

    print(f"{'='*60}")
    print(f"MoE VARIANT {vid}: {vname}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    merged_dir = Path("data/merged")
    meta = json.load(open(merged_dir / "metadata.json"))
    feature_columns = meta["feature_columns"]
    n_features = len(feature_columns)

    pc_dir = "data/pipeline/point_clouds"
    splits = merged_dir / "splits"

    train_ds = MergedDataset(str(splits / "train.csv"), pc_dir, feature_columns, is_train=True)
    val_ds = MergedDataset(str(splits / "val.csv"), pc_dir, feature_columns, is_train=False)
    test_ds = MergedDataset(str(splits / "test.csv"), pc_dir, feature_columns, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_merged)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_merged)

    fusion = build_fusion_variant(vid, n_features)
    model = MoEWithFusionVariant(
        feature_dim=n_features, fusion_module=fusion, num_experts=4,
        fused_dim=256, dropout=0.3,
        pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    print(f"Total params: {n_params:,}  |  Fusion params: {fusion_params:,}")

    model, snapshots = train_moe(
        model, train_loader, val_loader, device,
        ckpt_dir=f"checkpoints/moe_ablation_{vid}",
        num_epochs=200, lr=1e-4, patience=25)

    preds, targets, gate_weights = evaluate_single(model, test_loader, device)
    metrics = compute_metrics(preds, targets)

    print(f"\n{format_metrics(metrics, f'MoE-{vid} ({vname})')}")

    # Save
    result = {
        "variant": vid,
        "name": vname,
        "total_params": n_params,
        "fusion_params": fusion_params,
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items()},
        "gate_weights_mean": gate_weights.mean(axis=0).tolist(),
    }
    with open(f"results/moe_ablation_{vid}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: results/moe_ablation_{vid}.json")


if __name__ == "__main__":
    main()
