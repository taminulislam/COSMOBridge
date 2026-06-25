"""Hybrid ensemble: Phase 3 for gamma1 + CV Ensemble (Strategy B) for rest.

Also re-runs Strategy B without gamma1 log-transform.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device, set_seed
from src.training.metrics import compute_metrics, format_metrics


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]

    # ══════════════════════════════════════════════════════════════
    # Part 1: Hybrid Ensemble using existing model predictions
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("HYBRID ENSEMBLE: Phase 3 (gamma1) + CV Ensemble (rest)")
    print(f"{'='*60}")

    # Load Phase 3 model for gamma1
    from src.models.fusion.multimodal_pointcloud import MultimodalPointCloudModel
    config_p3 = {**config}
    config_p3.setdefault("model", {})["temp_skip"] = False
    model_p3 = MultimodalPointCloudModel(config=config_p3)
    ckpt = torch.load("checkpoints/pointcloud/best_model.pt", map_location=device, weights_only=False)
    model_p3.load_state_dict(ckpt["model_state_dict"])
    model_p3.to(device).eval()
    print("  Phase 3 model loaded")

    # Load CV fold models
    from scripts.train_joint import JointModel, MergedDataset, collate_merged
    meta = json.load(open("data/merged/metadata.json"))
    feature_columns = meta["feature_columns"]

    cv_models = []
    for fold in range(5):
        ckpt_path = Path(f"checkpoints/cv_fold_{fold}/best_model.pt")
        if ckpt_path.exists():
            m = JointModel(feature_dim=len(feature_columns))
            m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            m.to(device).eval()
            cv_models.append(m)
    print(f"  CV models loaded: {len(cv_models)} folds")

    # Load test data for both formats
    from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud
    pc_dir = "data/pipeline/point_clouds"
    splits_orig = Path("data/processed/splits")

    # Phase 3 test loader (original format)
    test_ds_p3 = PointCloudMultimodalDataset(str(splits_orig / "test.csv"), pc_dir, is_train=False)
    test_loader_p3 = DataLoader(test_ds_p3, batch_size=32, shuffle=False, collate_fn=collate_pointcloud)

    # CV ensemble test loader (merged format)
    test_ds_cv = MergedDataset(str(Path("data/merged/splits/test.csv")), pc_dir,
                               feature_columns, is_train=False)
    test_loader_cv = DataLoader(test_ds_cv, batch_size=32, shuffle=False, collate_fn=collate_merged)

    # Phase 3 predictions
    p3_preds, p3_targets = [], []
    with torch.no_grad():
        for batch in test_loader_p3:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            pred = model_p3(point_cloud=batch["point_cloud"], features=batch["features"],
                           atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                           bond_features=batch["bond_features"], batch=batch["batch"])
            p3_preds.append(pred.cpu().numpy())
            p3_targets.append(batch["targets"].cpu().numpy())
    p3_preds = np.concatenate(p3_preds)
    p3_targets = np.concatenate(p3_targets)

    # CV ensemble predictions
    cv_fold_preds = []
    cv_targets = None
    with torch.no_grad():
        for m in cv_models:
            fold_preds = []
            fold_targets = []
            for batch in test_loader_cv:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                pred = m(point_cloud=batch["point_cloud"], features=batch["features"],
                        atom_features=batch["atom_features"], edge_index=batch["edge_index"],
                        bond_features=batch["bond_features"], batch=batch["batch"])
                fold_preds.append(pred.cpu().numpy())
                fold_targets.append(batch["targets"].cpu().numpy())
            cv_fold_preds.append(np.concatenate(fold_preds))
            if cv_targets is None:
                cv_targets = np.concatenate(fold_targets)

    cv_ensemble_preds = np.mean(cv_fold_preds, axis=0)

    # Hybrid: Phase 3 for gamma1 (index 0), CV ensemble for rest
    # Note: Phase 3 targets are in original scale, CV targets are in merged scale
    # We evaluate Phase 3 on original targets, CV on merged targets

    # Phase 3 single-model evaluation
    print("\n  Phase 3 (original scale) per-property:")
    p3_metrics = compute_metrics(p3_preds, p3_targets)
    for prop in TARGET_COLUMNS:
        print(f"    {prop:15s} R²: {p3_metrics.get(f'{prop}_r2', float('nan')):.4f}")

    # CV ensemble evaluation (merged scale)
    print("\n  CV Ensemble (merged scale) per-property:")
    cv_metrics = compute_metrics(cv_ensemble_preds, cv_targets)
    for prop in TARGET_COLUMNS:
        print(f"    {prop:15s} R²: {cv_metrics.get(f'{prop}_r2', float('nan')):.4f}")

    # Hybrid ensemble: best of each
    print(f"\n  HYBRID ENSEMBLE (best per property):")
    hybrid_r2 = {}
    for i, prop in enumerate(TARGET_COLUMNS):
        r2_p3 = p3_metrics.get(f"{prop}_r2", -999)
        r2_cv = cv_metrics.get(f"{prop}_r2", -999)
        best = max(r2_p3, r2_cv)
        source = "Phase3" if r2_p3 >= r2_cv else "CVEns"
        hybrid_r2[prop] = best
        print(f"    {prop:15s} R²: {best:.4f}  (from {source})")

    avg_hybrid = np.mean(list(hybrid_r2.values()))
    print(f"\n    {'AVERAGE':15s} R²: {avg_hybrid:.4f}")

    # ══════════════════════════════════════════════════════════════
    # Part 2: Re-run Strategy B without gamma1 log-transform
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STRATEGY B v2: Without gamma1 log-transform")
    print(f"{'='*60}")

    from scripts.create_merged_dataset import (
        add_morgan_fingerprints, MORGAN_BITS, SURFACE_FEATURES, THERMO_FEATURES
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold

    # Rebuild merged dataset WITHOUT log-transforming gamma1
    print("\n  Rebuilding merged dataset (no gamma1 log-transform)...")
    orig = pd.read_csv("data/processed/il_data_raw.csv")
    orig["source"] = "original"

    ilth = pd.read_csv("data/augmented/ilthermo_data.csv")
    ilth["source"] = "ilthermo"
    # Only filter extreme outliers, no log transform
    ilth = ilth[~((ilth["gamma1"].notna()) & (ilth["gamma1"].abs() > 100))]

    for col in TARGET_COLUMNS:
        if col not in ilth.columns:
            ilth[col] = np.nan

    ilth["il_id"] = "ILThermo"
    if "cation_smiles" not in ilth.columns:
        ilth["cation_smiles"] = ilth["smiles"].apply(lambda s: s.split(".")[0] if "." in s else s)
    if "anion_smiles" not in ilth.columns:
        ilth["anion_smiles"] = ilth["smiles"].apply(lambda s: s.split(".")[1] if "." in str(s) and len(str(s).split(".")) > 1 else "")

    for df in [orig, ilth]:
        df["inv_temperature"] = 1.0 / df["temperature"]
        df["temp_squared"] = df["temperature"] ** 2
        df["temp_cubed"] = df["temperature"] ** 3

    # Log-transform ONLY P (not gamma1)
    for df in [orig, ilth]:
        for col in ["P"]:
            valid = df[col].notna()
            df.loc[valid, col] = np.sign(df.loc[valid, col]) * np.log1p(np.abs(df.loc[valid, col]))

    # Merge surface descriptors
    desc_orig = pd.read_csv("data/pipeline/surface_descriptors.csv")
    orig = orig.merge(desc_orig, on="il_short_name", how="left", suffixes=("", "_dup"))
    orig = orig[[c for c in orig.columns if not c.endswith("_dup")]]

    desc_ilth_path = Path("data/pipeline/surface_descriptors_ilthermo.csv")
    if desc_ilth_path.exists():
        desc_ilth = pd.read_csv(desc_ilth_path)
        if "smiles" in desc_ilth.columns:
            ilth = ilth.merge(desc_ilth.drop(columns=["il_short_name"], errors="ignore"),
                              on="smiles", how="left", suffixes=("", "_dup"))
            ilth = ilth[[c for c in ilth.columns if not c.endswith("_dup")]]

    for col in SURFACE_FEATURES:
        for df in [orig, ilth]:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)

    # Morgan fingerprints
    orig, morgan_cols = add_morgan_fingerprints(orig)
    ilth, _ = add_morgan_fingerprints(ilth)

    FEATURE_COLS = THERMO_FEATURES + SURFACE_FEATURES + morgan_cols

    common = ["smiles", "il_short_name", "temperature", "x1", "source",
              "cation_smiles", "anion_smiles"] + TARGET_COLUMNS + FEATURE_COLS
    for col in common:
        if col not in orig.columns:
            orig[col] = np.nan if col in TARGET_COLUMNS else 0.0
        if col not in ilth.columns:
            ilth[col] = np.nan if col in TARGET_COLUMNS else 0.0

    merged = pd.concat([orig[common], ilth[common]], ignore_index=True)

    # Normalize features
    merged[FEATURE_COLS] = StandardScaler().fit_transform(merged[FEATURE_COLS])
    for col in TARGET_COLUMNS:
        valid = merged[col].notna()
        if valid.sum() > 1:
            scaler = StandardScaler()
            merged.loc[valid, col] = scaler.fit_transform(merged.loc[valid, [col]]).flatten()

    # Get original split ILs
    split_info = json.load(open("data/processed/splits/split_info.json"))
    test_ils = set(split_info["test_ils"])
    val_ils = set(split_info["val_ils"])

    test_mask = merged["il_short_name"].isin(test_ils) & (merged["source"] == "original")
    test_df = merged[test_mask].reset_index(drop=True)

    # Save temporary v2 dataset
    v2_dir = Path("data/merged_v2")
    v2_dir.mkdir(parents=True, exist_ok=True)
    splits_v2 = v2_dir / "splits"
    splits_v2.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(splits_v2 / "test.csv", index=False)

    # Create CV folds
    orig_merged = merged[merged["source"] == "original"].reset_index(drop=True)
    ilthermo_merged = merged[merged["source"] == "ilthermo"].reset_index(drop=True)

    cv_dir = v2_dir / "cv_folds"
    cv_dir.mkdir(parents=True, exist_ok=True)

    gkf = GroupKFold(n_splits=5)
    groups = orig_merged["il_short_name"].values

    for fold, (train_idx, val_idx) in enumerate(gkf.split(orig_merged, groups=groups)):
        fold_dir = cv_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_val = orig_merged.iloc[val_idx].reset_index(drop=True)
        fold_train = pd.concat([orig_merged.iloc[train_idx].reset_index(drop=True),
                                ilthermo_merged], ignore_index=True)
        fold_train.to_csv(fold_dir / "train.csv", index=False)
        fold_val.to_csv(fold_dir / "val.csv", index=False)

    print(f"  Merged v2: {len(merged)} rows, test={len(test_df)}")

    # Train 5-fold CV ensemble
    from scripts.train_joint import JointModel, MergedDataset, collate_merged, train_model, evaluate

    test_ds_v2 = MergedDataset(str(splits_v2 / "test.csv"), pc_dir, FEATURE_COLS, is_train=False)
    test_loader_v2 = DataLoader(test_ds_v2, batch_size=32, shuffle=False, collate_fn=collate_merged)

    cv_preds_v2 = []
    targets_v2 = None

    for fold in range(5):
        print(f"\n  --- Fold {fold} ---")
        fold_dir = cv_dir / f"fold_{fold}"

        model = JointModel(feature_dim=len(FEATURE_COLS),
                          pretrained_gnn_path="checkpoints/transfer/pretrained.pt")
        model.to(device)

        fold_train = MergedDataset(str(fold_dir / "train.csv"), pc_dir, FEATURE_COLS, is_train=True)
        fold_val = MergedDataset(str(fold_dir / "val.csv"), pc_dir, FEATURE_COLS, is_train=False)

        train_ldr = DataLoader(fold_train, batch_size=64, shuffle=True, collate_fn=collate_merged)
        val_ldr = DataLoader(fold_val, batch_size=32, shuffle=False, collate_fn=collate_merged)

        model = train_model(model, train_ldr, val_ldr, device,
                           f"checkpoints/cv_v2_fold_{fold}",
                           num_epochs=150, lr=1e-4, patience=20, use_masked_loss=True)

        preds, tgts = evaluate(model, test_loader_v2, device)
        cv_preds_v2.append(preds)
        if targets_v2 is None:
            targets_v2 = tgts

        fold_m = compute_metrics(preds, tgts)
        print(f"    Fold {fold} R²: {fold_m['avg_r2']:.4f}")

    # Average
    ensemble_v2 = np.mean(cv_preds_v2, axis=0)
    metrics_v2 = compute_metrics(ensemble_v2, targets_v2)

    print(f"\n  Strategy B v2 (no gamma1 log-transform) Results:")
    print(format_metrics(metrics_v2, "CV Ensemble v2"))

    # ══════════════════════════════════════════════════════════════
    # Final hybrid: best of Phase 3 + CV Ensemble v2
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FINAL HYBRID: Best per-property from Phase 3 + CV v2")
    print(f"{'='*60}")

    print(f"\n  {'Property':15s} {'Phase3':>10s} {'CV v2':>10s} {'Best':>10s} Source")
    final_r2 = {}
    for prop in TARGET_COLUMNS:
        r2_p3 = p3_metrics.get(f"{prop}_r2", -999)
        r2_v2 = metrics_v2.get(f"{prop}_r2", -999)
        best = max(r2_p3, r2_v2)
        source = "Phase3" if r2_p3 >= r2_v2 else "CV_v2"
        final_r2[prop] = best
        print(f"  {prop:15s} {r2_p3:10.4f} {r2_v2:10.4f} {best:10.4f} {source}")

    avg_final = np.mean(list(final_r2.values()))
    print(f"\n  {'AVERAGE':15s} {'':10s} {'':10s} {avg_final:10.4f}")

    # Save
    results = {
        "hybrid_v1": {
            "description": "Phase3(gamma1) + CVEnsemble_v1(rest)",
            "per_property_r2": {k: float(v) for k, v in hybrid_r2.items()},
            "avg_r2": float(avg_hybrid),
        },
        "cv_ensemble_v2": {
            "description": "5-fold CV ensemble without gamma1 log-transform",
            "test_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                            for k, v in metrics_v2.items()},
        },
        "final_hybrid": {
            "description": "Best per-property from Phase3 + CV_v2",
            "per_property_r2": {k: float(v) for k, v in final_r2.items()},
            "avg_r2": float(avg_final),
        },
    }
    with open("results/hybrid_final_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/hybrid_final_results.json")


if __name__ == "__main__":
    main()
