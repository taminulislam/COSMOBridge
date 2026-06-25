"""Cache per-frame V-JEPA CLS embeddings for Ideas β (frame-level training) and ζ (TTA).

For both DFT and Gasteiger encoders, save a (N, 36, 192) array per split
(instead of the mean-pooled (N, 192) used elsewhere). Output files:
    cosmobridge_v5/data/cached_image_features_{split}_dft_perframe.npz
    cosmobridge_v5/data/cached_image_features_{split}_gasteiger_perframe.npz
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V5_ROOT = PROJECT_ROOT / "cosmobridge_v5"
sys.path.insert(0, str(V5_ROOT))
from models.multiview_vit import PatchEmbedding, ViTBlock  # noqa: E402


CACHED_DIR = PROJECT_ROOT / "cosmobridge_v4" / "data"
DFT_CKPT = V5_ROOT / "checkpoints" / "vjepa" / "vit_pretrained_vjepa.pt"
GAST_CKPT = V5_ROOT / "checkpoints" / "vjepa_gasteiger_apr10" / "vit_pretrained_vjepa.pt"
FRAMES_DIR = V5_ROOT / "data" / "cosmo_images"
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
    print(f"[{label}] encoder loaded from {ckpt_path.parent.name}/{ckpt_path.name}")

    # Compute per-frame features once per unique SMILES
    smiles_to_feats = {}

    for split in ("train", "val", "test"):
        c = np.load(CACHED_DIR / f"cached_{split}.npz", allow_pickle=True)
        smiles_list = [str(s) for s in c["smiles"]]
        n = len(smiles_list)
        out = np.zeros((n, N_FRAMES, 192), dtype=np.float32)

        for i, s in enumerate(smiles_list):
            if s not in smiles_to_feats:
                h = smi_hash(s)
                d = FRAMES_DIR / f"{h}_frames"
                if not d.exists():
                    smiles_to_feats[s] = np.zeros((N_FRAMES, 192), dtype=np.float32)
                    continue
                frames = sorted(d.glob("frame_*.png"))[:N_FRAMES]
                if len(frames) < N_FRAMES:
                    # Pad by repeating the last frame
                    while len(frames) < N_FRAMES:
                        frames.append(frames[-1])
                imgs = torch.stack([tfm(Image.open(p).convert("RGB")) for p in frames]).to(device)
                with torch.no_grad():
                    cls_feats = encoder(imgs).cpu().numpy().astype(np.float32)  # (36, 192)
                smiles_to_feats[s] = cls_feats
            out[i] = smiles_to_feats[s]

        out_path = OUT_DIR / f"cached_image_features_{split}_{label}_perframe.npz"
        np.savez(out_path, vit_feat=out)
        print(f"  [{label}] {split}: {n} samples ({N_FRAMES} frames each), saved to {out_path.name}")


def main():
    run_encoder(DFT_CKPT, "dft")
    run_encoder(GAST_CKPT, "gasteiger")
    print("\nDone.")


if __name__ == "__main__":
    main()
