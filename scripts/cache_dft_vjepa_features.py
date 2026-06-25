"""Extract DFT V-JEPA CLS features per sample and save to cached_image_features_{train,val,test}_dft.npz.

These files mirror the old (Gasteiger) `cached_image_features_*.npz`
files that the PerPropHead recipe in `slurm_combined_sigma.sh` reads.
Each file contains a `vit_feat` array of shape (N, 192) in the same
row order as `cosmobridge_v4/data/cached_{split}.npz`.
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
VJEPA_CKPT = V5_ROOT / "checkpoints" / "vjepa" / "vit_pretrained_vjepa.pt"
FRAMES_DIR_V5 = V5_ROOT / "data" / "cosmo_images"
FRAMES_DIR_PIPELINE = PROJECT_ROOT / "data" / "pipeline" / "cosmo_images"
OUT_DIR = V5_ROOT / "data"


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


def find_frames(smiles):
    h = smi_hash(str(smiles))
    d = FRAMES_DIR_V5 / f"{h}_frames"
    if d.exists():
        frames = sorted(d.glob("frame_*.png"))
        if len(frames) >= 4:
            return frames
    return None


def main():
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder = ViTTinyEncoder().to(device)
    ckpt = torch.load(VJEPA_CKPT, map_location=device, weights_only=False)
    state = ckpt.get("encoder_state_dict", ckpt)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    print(f"V-JEPA encoder loaded from {VJEPA_CKPT.name}")

    # Cache a single embedding per unique SMILES; then index by sample row.
    smiles_to_emb = {}

    for split in ("train", "val", "test"):
        c = np.load(CACHED_DIR / f"cached_{split}.npz", allow_pickle=True)
        smiles_list = [str(s) for s in c["smiles"]]
        n = len(smiles_list)
        out = np.zeros((n, 192), dtype=np.float32)
        miss = 0

        with torch.no_grad():
            for i, s in enumerate(smiles_list):
                if s in smiles_to_emb:
                    out[i] = smiles_to_emb[s]
                    continue
                frames = find_frames(s)
                if frames is None:
                    miss += 1
                    continue
                imgs = torch.stack([tfm(Image.open(p).convert("RGB")) for p in frames]).to(device)
                cls = encoder(imgs).mean(dim=0).cpu().numpy().astype(np.float32)
                smiles_to_emb[s] = cls
                out[i] = cls

        out_path = OUT_DIR / f"cached_image_features_{split}_dft.npz"
        np.savez(out_path, vit_feat=out)
        print(f"{split}: {n} samples, {miss} miss, saved to {out_path.name}")

    print(f"\n{len(smiles_to_emb)} unique compounds embedded.")


if __name__ == "__main__":
    main()
