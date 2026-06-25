"""Extract ChemBERTa CLS embeddings per unique SMILES.

ChemBERTa (DeepChem/ChemBERTa-77M-MLM) is a RoBERTa-style transformer
pretrained on SMILES strings via masked language modelling on the
ZINC/PubChem corpora. It outputs 384-D contextualized embeddings that
encode *molecular-string* chemistry — a complementary modality to the
image encoders (V-JEPA, DINOv2) and to the graph encoder (chemprop).

Output: (N, 384) per split, index-aligned with the v4 cached npz.
Saved to:
    cosmobridge_v5/data/cached_image_features_{split}_chemberta.npz
(kept in the same folder and file pattern so perprop_residual.py can
load it via the existing cache-loading helpers.)
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V5_ROOT = PROJECT_ROOT / "cosmobridge_v5"
CACHED_DIR = PROJECT_ROOT / "cosmobridge_v4" / "data"
OUT_DIR = V5_ROOT / "data"

MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"


def main():
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModel.from_pretrained(MODEL_NAME).to(device)
    mdl.eval()
    print(f"ChemBERTa loaded: {sum(p.numel() for p in mdl.parameters()):,} params")

    smiles_to_feat = {}

    for split in ("train", "val", "test"):
        c = np.load(CACHED_DIR / f"cached_{split}.npz", allow_pickle=True)
        smiles_list = [str(s) for s in c["smiles"]]
        n = len(smiles_list)
        out = np.zeros((n, 384), dtype=np.float32)

        with torch.no_grad():
            for i, s in enumerate(smiles_list):
                if s not in smiles_to_feat:
                    enc = tok(s, return_tensors="pt", padding=True,
                              truncation=True, max_length=256).to(device)
                    out_bert = mdl(**enc)
                    # Average of token embeddings, excluding special tokens
                    hidden = out_bert.last_hidden_state[0]  # (L, 384)
                    # Skip [CLS] and [SEP] tokens, mean-pool the middle
                    if hidden.shape[0] > 2:
                        mean_emb = hidden[1:-1].mean(dim=0)
                    else:
                        mean_emb = hidden[0]
                    smiles_to_feat[s] = mean_emb.cpu().numpy().astype(np.float32)
                out[i] = smiles_to_feat[s]

        out_path = OUT_DIR / f"cached_image_features_{split}_chemberta.npz"
        np.savez(out_path, vit_feat=out)
        print(f"  {split}: {n} samples, saved to {out_path.name}")

    print(f"\n{len(smiles_to_feat)} unique compounds embedded.")


if __name__ == "__main__":
    main()
