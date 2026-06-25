"""Generate novel ionic liquid structures using trained generative models.

Usage:
    python scripts/generate.py --model vae --checkpoint checkpoints/vae_best.pt --n 50
    python scripts/generate.py --model gan --checkpoint checkpoints/gan_best.pt --n 50
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd

from src.models.generative.vae import SMILESVAE, smiles_to_indices
from src.models.generative.gan import MolecularGAN
from src.utils.config import load_config, get_device


def generate_with_vae(checkpoint_path: str, n_samples: int, temperature: float, device):
    """Generate molecules using VAE."""
    model = SMILESVAE()
    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    smiles_list = model.generate(n_samples, temperature=temperature, device=device)
    return smiles_list


def generate_with_gan(checkpoint_path: str, n_samples: int, temperature: float, device):
    """Generate molecules using GAN."""
    model = MolecularGAN()
    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    smiles_list = model.generate(n_samples, temperature=temperature, device=device)
    return smiles_list


def analyze_generated(smiles_list: list, training_smiles: list = None):
    """Analyze generated molecules for validity, novelty, and diversity."""
    from src.models.generative.vae import SMILESVAE

    # Validity
    validation = SMILESVAE.validate_smiles(smiles_list)
    print(f"\nGeneration Results:")
    print(f"  Total generated: {validation['total']}")
    print(f"  Valid: {len(validation['valid'])} ({validation['validity_rate']*100:.1f}%)")
    print(f"  Invalid: {len(validation['invalid'])}")

    valid = validation["valid"]

    if valid:
        # Uniqueness
        unique = set(valid)
        print(f"  Unique: {len(unique)} ({len(unique)/len(valid)*100:.1f}%)")

        # Novelty (vs training set)
        if training_smiles:
            training_set = set(training_smiles)
            novel = [s for s in unique if s not in training_set]
            print(f"  Novel: {len(novel)} ({len(novel)/len(unique)*100:.1f}%)")

        # Print some examples
        print(f"\n  Sample valid SMILES:")
        for s in list(unique)[:10]:
            print(f"    {s}")

    return validation


def main():
    parser = argparse.ArgumentParser(description="Generate novel ionic liquids")
    parser.add_argument("--model", type=str, default="vae", choices=["vae", "gan"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vae_best.pt")
    parser.add_argument("--n", type=int, default=50, help="Number of molecules to generate")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config)

    print(f"Generating {args.n} molecules with {args.model} (T={args.temperature})")

    if args.model == "vae":
        smiles_list = generate_with_vae(args.checkpoint, args.n, args.temperature, device)
    else:
        smiles_list = generate_with_gan(args.checkpoint, args.n, args.temperature, device)

    # Load training SMILES for novelty check
    try:
        train_df = pd.read_csv("data/processed/il_data_raw.csv")
        training_smiles = train_df["smiles"].unique().tolist()
    except FileNotFoundError:
        training_smiles = None

    validation = analyze_generated(smiles_list, training_smiles)

    # Save results
    if args.output:
        df = pd.DataFrame({
            "smiles": smiles_list,
            "valid": [s in validation["valid"] for s in smiles_list],
        })
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
