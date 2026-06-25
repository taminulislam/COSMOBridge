"""Extract DINOv2 (ViT-S/14) CLS embeddings per compound.

DINOv2 is trained on natural images, not COSMO surfaces — it brings a
different inductive bias than our self-supervised V-JEPA encoders.
Output: (N, 384) per split, mean-pooled across 36 rotation frames.

Saved to:
    cosmobridge_v5/data/cached_image_features_{split}_dinov2.npz
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V5_ROOT = PROJECT_ROOT / "cosmobridge_v5"
CACHED_DIR = PROJECT_ROOT / "cosmobridge_v4" / "data"
FRAMES_DIR = V5_ROOT / "data" / "cosmo_images"
OUT_DIR = V5_ROOT / "data"
N_FRAMES = 36


def smi_hash(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


def main():
    from torchvision import transforms
    # DINOv2 uses ImageNet normalization by default in timm
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = timm.create_model(
        "vit_small_patch14_dinov2",
        pretrained=True,
        num_classes=0,
        img_size=224,
    ).to(device)
    model.eval()
    print(f"DINOv2 ViT-S/14 loaded: {sum(p.numel() for p in model.parameters()):,} params, "
          f"num_features={model.num_features}")

    smiles_to_feat = {}

    for split in ("train", "val", "test"):
        c = np.load(CACHED_DIR / f"cached_{split}.npz", allow_pickle=True)
        smiles_list = [str(s) for s in c["smiles"]]
        n = len(smiles_list)
        out = np.zeros((n, model.num_features), dtype=np.float32)
        miss = 0

        with torch.no_grad():
            for i, s in enumerate(smiles_list):
                if s in smiles_to_feat:
                    out[i] = smiles_to_feat[s]
                    continue
                h = smi_hash(s)
                d = FRAMES_DIR / f"{h}_frames"
                if not d.exists():
                    miss += 1
                    continue
                frames = sorted(d.glob("frame_*.png"))[:N_FRAMES]
                imgs = torch.stack([tfm(Image.open(p).convert("RGB")) for p in frames]).to(device)
                feats = model(imgs).mean(dim=0).cpu().numpy().astype(np.float32)
                smiles_to_feat[s] = feats
                out[i] = feats

        out_path = OUT_DIR / f"cached_image_features_{split}_dinov2.npz"
        np.savez(out_path, vit_feat=out)
        print(f"  {split}: {n} samples, {miss} miss, saved to {out_path.name}")

    print(f"\n{len(smiles_to_feat)} unique compounds embedded.")


if __name__ == "__main__":
    main()
