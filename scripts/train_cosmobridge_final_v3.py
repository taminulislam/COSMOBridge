"""COSMOBridge Final v3: Use Chemprop's OWN readout as direct path.

The key insight: don't train a new FFN on frozen fingerprints.
Use Chemprop's jointly-trained MPN+FFN as-is for the direct path.

Architecture:
  Path A: CP-GBH Fusion (frozen, gamma1=0.908, gamma2=0.936)
  Path B: Chemprop's full model (frozen, G_E=0.748, H_E=0.731)
  Gate: 7 learnable params (the ONLY trainable thing)

Both paths are frozen, pre-trained, proven. Only 7 gates are optimized.
This is the 3-Model Router collapsed into a single nn.Module.
"""

import sys
import json
import subprocess
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from src.utils.config import load_config, get_device, set_seed
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.models.fusion.multimodal_pointcloud import MultimodalPointCloudModel
from src.training.metrics import compute_metrics, format_metrics
from scripts.train_pointcloud import PointCloudMultimodalDataset, collate_pointcloud
from scripts.train_chemprop_gbh_hybrid import ChempropGBHFusion

from chemprop.utils import load_checkpoint as load_chemprop


class COSMOBridgeV3(nn.Module):
    """COSMOBridge v3: Frozen fusion + Frozen Chemprop readout + Trainable gates.

    Path A: GBH fusion of (Chemprop MPN fingerprint × PointNet surface) → 7 preds
            Pre-loaded from CP-GBH Hybrid (gamma1=0.908, gamma2=0.936)

    Path B: Chemprop's full forward pass (MPN → FFN) → 7 preds
            Chemprop's own jointly-trained readout (G_E=0.748, H_E=0.731)

    Gate: 7 per-property routing weights (ONLY trainable params)
    """

    def __init__(self, chemprop_model, fusion_model):
        super().__init__()

        # Path B: Full Chemprop (MPN + readout FFN), frozen
        self.chemprop = chemprop_model
        for p in self.chemprop.parameters():
            p.requires_grad = False

        # Path A: CP-GBH fusion, frozen
        self.fusion = fusion_model
        for p in self.fusion.parameters():
            p.requires_grad = False

        # Per-property gates (ONLY trainable: 7 params)
        self.gate_logits = nn.Parameter(torch.tensor([
            2.0,    # gamma1 → fusion (0.908 > 0.828)
            2.0,    # gamma2 → fusion (0.936 > 0.858)
            -2.0,   # G_E → chemprop (0.748 > 0.580)
            -2.0,   # H_E → chemprop (0.731 > 0.636)
            -2.0,   # G_mix → chemprop (0.725 > 0.578)
            -1.0,   # H_vap → slight chemprop (0.675 > 0.608)
            1.0,    # P → slight fusion (0.829 > 0.826)
        ]))

    def forward(self, graph_feat, surface_feat, thermo_feat,
                smiles_batch=None, features_np_batch=None):
        """
        Args:
            graph_feat: (B, 300) — cached Chemprop MPN fingerprint
            surface_feat: (B, 256) — cached PointNet features
            thermo_feat: (B, 25) — thermo features tensor
            smiles_batch: list of [smiles] — for Chemprop full forward
            features_np_batch: list of np.array — for Chemprop
        """
        with torch.no_grad():
            # Path A: GBH fusion (pre-loaded, frozen)
            preds_fused = self.fusion(graph_feat, surface_feat, thermo_feat)

            # Path B: Chemprop's full model (MPN → readout)
            if smiles_batch is not None:
                preds_chemprop = self.chemprop(smiles_batch, features_np_batch)
            else:
                preds_chemprop = torch.zeros_like(preds_fused)

        # Per-property gated combination
        alpha = torch.sigmoid(self.gate_logits)
        predictions = alpha.unsqueeze(0) * preds_fused + (1 - alpha.unsqueeze(0)) * preds_chemprop

        return predictions, {
            "gate_values": alpha.detach(),
            "preds_fused": preds_fused.detach(),
            "preds_chemprop": preds_chemprop.detach(),
        }


class CachedSmilesDataset(Dataset):
    """Dataset with cached encoder features + SMILES for Chemprop."""

    def __init__(self, graph_feats, surface_feats, thermo_feats, targets,
                 smiles_list, thermo_raw_list):
        self.g = torch.tensor(graph_feats, dtype=torch.float32)
        self.s = torch.tensor(surface_feats, dtype=torch.float32)
        self.t = torch.tensor(thermo_feats, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
        self.smiles = smiles_list
        self.thermo_raw = thermo_raw_list

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return (self.g[i], self.s[i], self.t[i], self.y[i],
                self.smiles[i], self.thermo_raw[i])


def collate_cached(batch):
    g = torch.stack([b[0] for b in batch])
    s = torch.stack([b[1] for b in batch])
    t = torch.stack([b[2] for b in batch])
    y = torch.stack([b[3] for b in batch])
    smiles = [[b[4]] for b in batch]
    thermo_raw = [b[5] for b in batch]
    return g, s, t, y, smiles, thermo_raw


def identity_collate(b): return b


def extract_features(device):
    pc_dir = "data/pipeline/point_clouds"
    orig_splits = Path("data/processed/splits")
    config = load_config("configs/default.yaml")

    pc_model = MultimodalPointCloudModel(config=config, pretrained_gnn_path=None)
    ckpt = torch.load("checkpoints/pointcloud/best_model.pt", map_location=device, weights_only=False)
    pc_model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    pc_model.to(device).eval()

    data = {}
    for split in ["train", "val", "test"]:
        ds = PointCloudMultimodalDataset(str(orig_splits / f"{split}.csv"), pc_dir, is_train=False)
        df = pd.read_csv(orig_splits / f"{split}.csv")

        sf, tf, tgt = [], [], []
        with torch.no_grad():
            for items in DataLoader(ds, batch_size=32, shuffle=False, collate_fn=identity_collate):
                pcs = torch.stack([x["point_cloud"] for x in items]).to(device)
                sf.append(pc_model.pointnet(pcs).cpu().numpy())
                tf.append(torch.stack([x["features"] for x in items]).numpy())
                tgt.append(torch.stack([x["targets"] for x in items]).numpy())

        # Chemprop fingerprints
        out = tempfile.mktemp(suffix=".csv")
        subprocess.run(["chemprop_fingerprint",
                         "--test_path", f"data/chemprop_tmp/{split}.csv",
                         "--features_path", f"data/chemprop_tmp/{split}_features.csv",
                         "--checkpoint_dir", "checkpoints/chemprop",
                         "--fingerprint_type", "MPN", "--preds_path", out],
                        capture_output=True, text=True, timeout=120)
        gf = pd.read_csv(out).select_dtypes(include=[np.number]).values.astype(np.float32)

        # SMILES and raw thermo for Chemprop full forward
        smiles_list = df["smiles"].tolist()
        thermo_raw = df[FEATURE_COLUMNS[:5]].values.astype(np.float32)

        data[split] = {
            "g": gf, "s": np.concatenate(sf), "t": np.concatenate(tf),
            "y": np.concatenate(tgt), "smiles": smiles_list,
            "thermo_raw": thermo_raw,
        }
        print(f"  {split}: graph={gf.shape} surface={data[split]['s'].shape} smiles={len(smiles_list)}")
    return data


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    # Extract features
    print("Step 1: Cache features...\n")
    data = extract_features(device)
    graph_dim = data["train"]["g"].shape[1]

    # Build datasets
    datasets = {}
    for split in ["train", "val", "test"]:
        d = data[split]
        datasets[split] = CachedSmilesDataset(
            d["g"], d["s"], d["t"], d["y"], d["smiles"],
            [d["thermo_raw"][i] for i in range(len(d["smiles"]))])

    train_ldr = DataLoader(datasets["train"], batch_size=32, shuffle=True, collate_fn=collate_cached)
    val_ldr = DataLoader(datasets["val"], batch_size=32, shuffle=False, collate_fn=collate_cached)
    test_ldr = DataLoader(datasets["test"], batch_size=32, shuffle=False, collate_fn=collate_cached)

    # Load both frozen paths
    print(f"\n{'='*60}")
    print("Step 2: Load frozen paths")
    print(f"{'='*60}")

    # Path A: CP-GBH Fusion
    fusion_model = ChempropGBHFusion(graph_dim=graph_dim, surface_dim=256,
                                       thermo_dim=len(FEATURE_COLUMNS), fused_dim=256,
                                       rank=32, hyper_hidden=64, dropout=0.3)
    fusion_model.load_state_dict(torch.load("checkpoints/chemprop_gbh_hybrid/best.pt",
                                              map_location=device, weights_only=True))
    fusion_model.to(device).eval()
    print(f"  Path A (CP-GBH Fusion): loaded, {sum(p.numel() for p in fusion_model.parameters()):,} params")

    # Path B: Full Chemprop
    chemprop_model = load_chemprop("checkpoints/chemprop/fold_0/model_0/model.pt")
    chemprop_model.to(device).eval()
    print(f"  Path B (Full Chemprop): loaded, {sum(p.numel() for p in chemprop_model.parameters()):,} params")

    # Build COSMOBridge v3
    model = COSMOBridgeV3(chemprop_model, fusion_model)
    model.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_trainable} (only gates)")

    # Train ONLY gates
    print(f"\n{'='*60}")
    print("Step 3: Train 7 gates")
    print(f"{'='*60}")

    optimizer = AdamW([model.gate_logits], lr=0.1)
    best, no_imp = float("inf"), 0
    for ep in range(200):
        model.train()
        tl, n = 0, 0
        for g, s, t, y, smiles, thermo_raw in train_ldr:
            g, s, t, y = g.to(device), s.to(device), t.to(device), y.to(device)
            optimizer.zero_grad()
            preds, _ = model(g, s, t, smiles_batch=smiles, features_np_batch=thermo_raw)
            loss = ((preds - y)**2).mean()
            loss.backward()
            optimizer.step()
            tl += loss.item(); n += 1

        model.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for g, s, t, y, smiles, thermo_raw in val_ldr:
                g, s, t, y = g.to(device), s.to(device), t.to(device), y.to(device)
                preds, _ = model(g, s, t, smiles_batch=smiles, features_np_batch=thermo_raw)
                vl += ((preds - y)**2).mean().item(); vn += 1
        avg = vl / max(vn, 1)
        if avg < best: best = avg; no_imp = 0; best_gates = model.gate_logits.data.clone()
        else: no_imp += 1
        if ep % 20 == 0:
            gates = torch.sigmoid(model.gate_logits).detach().cpu().numpy()
            print(f"  Ep {ep:3d} | T:{tl/max(n,1):.4f} V:{avg:.4f} B:{best:.4f} P:{no_imp}/40 "
                  f"| [{' '.join(f'{x:.2f}' for x in gates)}]")
        if no_imp >= 40: print(f"  Early stop ep {ep}"); break
    model.gate_logits.data = best_gates

    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")

    model.eval()
    all_p, all_t = [], []
    all_fused, all_chemp = [], []
    with torch.no_grad():
        for g, s, t, y, smiles, thermo_raw in test_ldr:
            g, s, t, y = g.to(device), s.to(device), t.to(device), y.to(device)
            preds, aux = model(g, s, t, smiles_batch=smiles, features_np_batch=thermo_raw)
            all_p.append(preds.cpu().numpy()); all_t.append(y.cpu().numpy())
            all_fused.append(aux["preds_fused"].cpu().numpy())
            all_chemp.append(aux["preds_chemprop"].cpu().numpy())

    preds = np.concatenate(all_p); targets = np.concatenate(all_t)
    pf = np.concatenate(all_fused); pc = np.concatenate(all_chemp)

    metrics = compute_metrics(preds, targets)
    mf = compute_metrics(pf, targets)
    mc = compute_metrics(pc, targets)

    print(format_metrics(metrics, "COSMOBridge v3"))

    gates = torch.sigmoid(model.gate_logits).detach().cpu().numpy()
    print(f"\n  Routing (α: 1=fusion, 0=chemprop):")
    for i, p in enumerate(TARGET_COLUMNS):
        path = "FUSION" if gates[i] > 0.6 else ("CHEMPROP" if gates[i] < 0.4 else "MIXED")
        print(f"    {p:15s}: α={gates[i]:.3f} {path:>8s}  "
              f"fusion={mf[f'{p}_r2']:.3f}  chemprop={mc[f'{p}_r2']:.3f}  → {metrics[f'{p}_r2']:.3f}")

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    prev = {}
    for name, path, key in [
        ("Chemprop", "results/chemprop_results.json", "test_metrics"),
        ("STILT", "results/chemprop_tuned_results.json", "STILT_C"),
        ("3-Model Router", "results/per_property_router_results.json", "metrics"),
        ("CP-GBH Hybrid", "results/chemprop_gbh_hybrid_results.json", "metrics"),
    ]:
        try:
            d = json.load(open(path))
            if key == "STILT_C": m = d.get("C: full mask, 48x OS", {}).get("metrics", {})
            elif key: m = d[key]
            prev[name] = m
        except: pass

    header = f"  {'Property':<15s}"
    for nm in prev: header += f" {nm[:14]:>14s}"
    header += f" {'COSMOBridge v3':>14s}"
    print(header); print("  " + "-" * len(header))
    for p in TARGET_COLUMNS:
        line = f"  {p:<15s}"
        for nm in prev: line += f" {prev[nm].get(f'{p}_r2', 0):14.4f}"
        line += f" {metrics[f'{p}_r2']:14.4f}"
        print(line)
    line = f"  {'AVERAGE':<15s}"
    for nm in prev: line += f" {prev[nm].get('avg_r2', 0):14.4f}"
    line += f" {metrics['avg_r2']:14.4f}"
    print(line)

    base = prev.get("Chemprop", {})
    wins = sum(1 for p in TARGET_COLUMNS if metrics[f"{p}_r2"] > base.get(f"{p}_r2", 0))
    d = metrics['avg_r2'] - base.get('avg_r2', 0)
    print(f"\n  vs Chemprop: {metrics['avg_r2']:.4f} vs {base.get('avg_r2',0):.4f} "
          f"({'+' if d>0 else ''}{d:.4f}) wins {wins}/7")

    results = {
        "model": "COSMOBridge_v3",
        "description": "Frozen CP-GBH fusion (gamma1=0.908) + frozen Chemprop full readout (G_E=0.748) "
                       "+ 7 trainable gates. True single nn.Module, both paths proven.",
        "n_trainable": 7,
        "gate_values": {p: float(g) for p, g in zip(TARGET_COLUMNS, gates)},
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items()},
    }
    with open("results/cosmobridge_v3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/cosmobridge_v3_results.json")


if __name__ == "__main__":
    main()
