"""Reviewer revision (COSMOBridge, Nice-9): t-SNE of the GBH fusion-path embeddings,
colored by the fusion-path predicted ln(gamma1), to show surface-aware organization.

We hook the 256-D output of the GatedBilinearHyperFusionV2 module (the fused embedding
fed to the prediction head) for every row across train+val+test (223 rows). The surface
input comes from the cached, row-identical PointNet feature bank (the raw clouds are 0600);
this is the SAME path used by revision_savepreds.py that reproduces the published 0.801.

Output:
  results/cosmobridge_fusion_tsne.npz  (emb2d, gamma1_pred, split_id, perplexity)
  paper/jcim/figures/fusion_tsne_gamma1.png

Run via jobs/revision_tsne_fusion.sh (GPU node; chemprop_fingerprint runs a D-MPNN forward).
"""
import sys, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
from src.utils.config import load_config, get_device
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from scripts.train_chemprop_gbh_hybrid import ChempropGBHFusion
import scripts.train_pointcloud as tp
from scripts.train_pointcloud import PointCloudMultimodalDataset
from scripts.cosmobridge_multiseed import identity_collate

PC_DIR = "data/pipeline/point_clouds_dft_apr11"
tp.PointCloudMultimodalDataset._load_point_cloud = lambda self, smiles: torch.zeros(1)


def extract_features():
    """Identical construction to revision_savepreds.extract_features."""
    orig = Path("data/processed/splits")
    data = {}
    for split in ["train", "val", "test"]:
        ds = PointCloudMultimodalDataset(str(orig / f"{split}.csv"), PC_DIR, is_train=False)
        df = pd.read_csv(orig / f"{split}.csv")
        tf, tgt = [], []
        for items in DataLoader(ds, batch_size=32, shuffle=False, collate_fn=identity_collate):
            tf.append(torch.stack([x["features"] for x in items]).numpy())
            tgt.append(torch.stack([x["targets"] for x in items]).numpy())
        s = np.load(f"data/chemprop_cosmo/{split}_pointnet_feats.npy").astype(np.float32)
        out = tempfile.mktemp(suffix=".csv")
        subprocess.run(["chemprop_fingerprint", "--test_path", f"data/chemprop_tmp/{split}.csv",
                        "--features_path", f"data/chemprop_tmp/{split}_features.csv",
                        "--checkpoint_dir", "checkpoints/chemprop", "--fingerprint_type", "MPN",
                        "--preds_path", out], capture_output=True, text=True, timeout=300)
        gf = pd.read_csv(out).select_dtypes(include=[np.number]).values.astype(np.float32)
        t = np.concatenate(tf); y = np.concatenate(tgt)
        assert s.shape[0] == t.shape[0] == len(df), f"row mismatch {split}: s={s.shape} t={t.shape} df={len(df)}"
        data[split] = {"g": gf, "s": s, "t": t, "y": y}
    return data


def main():
    device = get_device(load_config("configs/default.yaml"))
    print(f"Device: {device}")
    data = extract_features()
    graph_dim = data["train"]["g"].shape[1]
    print(f"graph_dim={graph_dim}, surface_dim={data['train']['s'].shape[1]}")

    fusion_model = ChempropGBHFusion(graph_dim=graph_dim, surface_dim=256,
                                     thermo_dim=len(FEATURE_COLUMNS), fused_dim=256,
                                     rank=32, hyper_hidden=64, dropout=0.3)
    fusion_model.load_state_dict(torch.load("checkpoints/chemprop_gbh_hybrid/best.pt",
                                            map_location=device, weights_only=True))
    fusion_model.to(device).eval()

    # hook the 256-D fused embedding (output of the GBH fusion module)
    captured = {}
    def hook(_m, _inp, out):
        captured["emb"] = out.detach().cpu().numpy()
    h = fusion_model.fusion.register_forward_hook(hook)

    embs, g1_pred, split_id = [], [], []
    for si, split in enumerate(["train", "val", "test"]):
        g = torch.tensor(data[split]["g"], dtype=torch.float32, device=device)
        s = torch.tensor(data[split]["s"], dtype=torch.float32, device=device)
        t = torch.tensor(data[split]["t"], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = fusion_model(g, s, t).cpu().numpy()   # (N,7), triggers hook
        embs.append(captured["emb"])
        g1_pred.append(pred[:, 0])                        # gamma1 = column 0
        split_id.append(np.full(len(pred), si))
        print(f"  {split}: emb={captured['emb'].shape} g1[min/max]={pred[:,0].min():.2f}/{pred[:,0].max():.2f}")
    h.remove()

    X = np.concatenate(embs).astype(np.float32)
    g1 = np.concatenate(g1_pred).astype(np.float32)
    sid = np.concatenate(split_id).astype(np.int32)
    print(f"Total rows: {X.shape[0]}, embedding dim {X.shape[1]}")
    assert TARGET_COLUMNS[0].lower().startswith("g") or "gamma" in TARGET_COLUMNS[0].lower(), \
        f"col0 is {TARGET_COLUMNS[0]} (expected gamma1)"

    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    perp = 30
    emb2d = TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto",
                 random_state=42).fit_transform(Xs)

    np.savez(project_root / "results" / "cosmobridge_fusion_tsne.npz",
             emb2d=emb2d, gamma1_pred=g1, split_id=sid, perplexity=perp)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sc = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=g1, cmap="viridis",
                    s=42, edgecolors="k", linewidths=0.4, alpha=0.9)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"Predicted $\ln\,\gamma_1^\infty$", fontsize=11)
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("GBH fusion-path embeddings", fontsize=12)
    fig.tight_layout()
    figpath = project_root / "paper" / "jcim" / "figures" / "fusion_tsne_gamma1.png"
    fig.savefig(figpath, dpi=300, bbox_inches="tight")
    print(f"Saved {figpath}")

    # quick quantitative check: does the 2D embedding organize by predicted gamma1?
    from scipy.stats import spearmanr
    r1 = spearmanr(emb2d[:, 0], g1).correlation
    r2 = spearmanr(emb2d[:, 1], g1).correlation
    print(f"Spearman(g1, tSNE1)={r1:.3f}  Spearman(g1, tSNE2)={r2:.3f}")


if __name__ == "__main__":
    main()
