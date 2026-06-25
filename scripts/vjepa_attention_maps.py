"""Idea 8: Attention map visualization for the DFT-pretrained V-JEPA encoder.

For a handful of representative ILs, extract the CLS-to-patch attention
weights from the last ViTBlock, average across heads, and overlay them
as a heatmap on the original rotation frame. Produces publication-quality
figures showing what the self-supervised encoder attends to on COSMO
surfaces.

Output: cosmobridge_v5/results/attention_maps/{compound_id}_frame_{i}.png
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V5_ROOT = PROJECT_ROOT / "cosmobridge_v5"
sys.path.insert(0, str(V5_ROOT))
from models.multiview_vit import PatchEmbedding, ViTBlock  # noqa: E402

VJEPA_CKPT = V5_ROOT / "checkpoints" / "vjepa" / "vit_pretrained_vjepa.pt"
FRAMES_DIR = V5_ROOT / "data" / "cosmo_images"
OUT_DIR = V5_ROOT / "results" / "attention_maps"

PATCH_SIZE = 16
IMG_SIZE = 224
N_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 196


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


def smi_hash(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


def capture_cls_attention(encoder, img_tensor):
    """Hook the last block's attention and return CLS→patch weights."""
    captured = {}

    def hook(module, inputs, outputs):
        # The MultiheadAttention __call__ is invoked as
        # self.attn(normed, normed, normed)
        # but our ViTBlock.forward discards need_weights kwarg. Re-run here.
        q, k, v = inputs
        attn_out, attn_w = module(q, k, v, need_weights=True, average_attn_weights=True)
        captured["weights"] = attn_w.detach()
        return attn_out, attn_w

    # Replace forward on last block's attn temporarily
    last_block = encoder.blocks[-1]
    orig_forward = last_block.attn.forward

    def patched(q, k, v, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True
        out, w = orig_forward(q, k, v, **kwargs)
        captured["weights"] = w.detach()
        return out, w

    last_block.attn.forward = patched
    try:
        with torch.no_grad():
            patches = encoder.patch_embed(img_tensor)
            B = patches.shape[0]
            cls = encoder.cls_token.expand(B, -1, -1)
            tokens = encoder.pos_dropout(
                torch.cat([cls, patches], dim=1) + encoder.pos_embed
            )
            for blk in encoder.blocks:
                tokens = blk(tokens)
    finally:
        last_block.attn.forward = orig_forward

    w = captured["weights"]  # (B, Q, K) where Q=K=1+N_patches
    cls_to_patches = w[:, 0, 1:]  # (B, N_patches)
    return cls_to_patches


def heatmap_overlay(img_pil, attn_vec, side=14):
    """Overlay (side, side) attention heatmap on the image."""
    import numpy as np
    attn = attn_vec.reshape(side, side).cpu().numpy()
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    # Upsample to image size via PIL resize
    heat = Image.fromarray((attn * 255).astype(np.uint8))
    heat = heat.resize(img_pil.size, Image.BILINEAR)
    heat_rgb = np.asarray(heat, dtype=np.float32) / 255.0
    colormap = np.zeros((*heat_rgb.shape, 3), dtype=np.uint8)
    colormap[..., 0] = (heat_rgb * 255).astype(np.uint8)
    colormap[..., 1] = ((1 - heat_rgb) * 64).astype(np.uint8)
    colormap[..., 2] = ((1 - heat_rgb) * 64).astype(np.uint8)
    base = np.asarray(img_pil.convert("RGB"), dtype=np.float32)
    blend = (0.55 * base + 0.45 * colormap).astype(np.uint8)
    return Image.fromarray(blend)


def main():
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ViTTinyEncoder().to(device)
    ckpt = torch.load(VJEPA_CKPT, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt.get("encoder_state_dict", ckpt), strict=True)
    encoder.eval()
    print(f"Encoder loaded ({ckpt.get('epoch', '?')})")

    # Pick 4 representative ILs — diverse anion families
    targets = [
        "CCN1C=C[N+](=C1)C.C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F",  # EMIM-NTf2
        "CCCCN1C=C[N+](=C1)C.[Cl-]",                                    # BMIM-Cl
        "C[N+](C)(C)CCO.C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F",      # Chol-NTf2
        "CCN1C=C[N+](=C1)C.CS(=O)(=O)[O-]",                             # EMIM-MeSO3
    ]
    names = ["EMIM-NTf2", "BMIM-Cl", "Chol-NTf2", "EMIM-MeSO3"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for smi, name in zip(targets, names):
        h = smi_hash(smi)
        frames_dir = FRAMES_DIR / f"{h}_frames"
        if not frames_dir.exists():
            print(f"SKIP {name}: {frames_dir.name} missing")
            continue
        frames = sorted(frames_dir.glob("frame_*.png"))[:6]  # 6 views each
        for i, fp in enumerate(frames):
            raw = Image.open(fp).convert("RGB")
            x = tfm(raw).unsqueeze(0).to(device)
            attn = capture_cls_attention(encoder, x)[0]  # (196,)
            overlay = heatmap_overlay(raw, attn)
            # Annotate
            draw = ImageDraw.Draw(overlay)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            draw.text((8, 8), f"{name}  view {i}", fill=(255, 255, 255), font=font)
            out_path = OUT_DIR / f"{name.replace('-', '_')}_view{i:02d}.png"
            overlay.save(out_path)
        print(f"  {name}: {len(frames)} attention overlays saved")

    print(f"\nAll overlays saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
