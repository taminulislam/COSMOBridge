"""Phase #1 Step F.2: Extract V-JEPA CLS embeddings for the expanded
train/val/test splits (4637/32/39 rows), one embedding per SMILES.

Produces both Gasteiger and DFT V-JEPA variants:
    cosmobridge_v5/data/cached_image_features_{split}_expanded.npz         (Gasteiger)
    cosmobridge_v5/data/cached_image_features_{split}_expanded_dft.npz     (DFT)

Each file contains `vit_feat` shape (N, 192) in row order matching
`cosmobridge_v4/data/cached_{split}_expanded.npz`.

For SMILES without COSMO frames (not rendered), we fall back to the
zero vector — those rows will contribute only non-image features.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V5_ROOT = PROJECT_ROOT / "cosmobridge_v5"
sys.path.insert(0, str(V5_ROOT))
from models.multiview_vit import PatchEmbedding, ViTBlock  # noqa: E402

MERGED = PROJECT_ROOT / "data/merged_v4"
DFT_CKPT = V5_ROOT / "checkpoints/vjepa/vit_pretrained_vjepa.pt"
GAST_CKPT = V5_ROOT / "checkpoints/vjepa_gasteiger_apr10/vit_pretrained_vjepa.pt"
FRAMES_DIR = V5_ROOT / "data/cosmo_images"
OUT_DIR = V5_ROOT / "data"
N_FRAMES = 36


class ViTTinyEncoder(nn.Module):
    def __init__(self, embed_dim=192, img_size=224, patch_size=16,
                 n_layers=6, n_heads=3, mlp_ratio=4, dropout=0.1,
                 stochastic_depth=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, n_layers)]
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, n_heads, mlp_ratio, dropout, dpr[i])
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        patches = self.patch_embed(x)
        B = patches.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = self.pos_dropout(
            torch.cat([cls, patches], dim=1) + self.pos_embed
        )
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens[:, 0])


def smi_hash(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


def run_encoder(ckpt_path, label):
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ViTTinyEncoder().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("encoder_state_dict", ckpt)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    print(f"[{label}] loaded from {ckpt_path.parent.name}/{ckpt_path.name}")

    smiles_to_emb = {}

    for split in ("train", "val", "test"):
        df = pd.read_csv(MERGED / f"splits/{split}.csv")
        smiles_list = df["smiles"].astype(str).tolist()
        n = len(smiles_list)
        out = np.zeros((n, 192), dtype=np.float32)
        hit = miss = 0

        with torch.no_grad():
            for i, s in enumerate(smiles_list):
                if s in smiles_to_emb:
                    out[i] = smiles_to_emb[s]
                    hit += 1
                    continue
                h = smi_hash(s)
                d = FRAMES_DIR / f"{h}_frames"
                if not d.exists() or len(list(d.glob("frame_*.png"))) < 4:
                    miss += 1
                    continue
                frames = sorted(d.glob("frame_*.png"))[:N_FRAMES]
                imgs = torch.stack([tfm(Image.open(p).convert("RGB")) for p in frames]).to(device)
                cls = encoder(imgs).mean(dim=0).cpu().numpy().astype(np.float32)
                smiles_to_emb[s] = cls
                out[i] = cls
                hit += 1

        suffix = "_dft" if label == "dft" else ""
        out_path = OUT_DIR / f"cached_image_features_{split}_expanded{suffix}.npz"
        np.savez(out_path, vit_feat=out)
        print(f"  [{label}] {split}: {n} rows, {hit} hit ({len(smiles_to_emb)} unique), {miss} miss -> {out_path.name}")


def main():
    run_encoder(GAST_CKPT, "gasteiger")
    run_encoder(DFT_CKPT, "dft")


if __name__ == "__main__":
    main()
