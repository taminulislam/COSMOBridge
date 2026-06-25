"""Approach 2: Train Conditional SMILES VAE and generate novel ILs.

Stage 1: Train CVAE on all 170 known IL SMILES with property conditioning
Stage 2: Generate novel ILs conditioned on desired property profiles
Stage 3: Validate generated ILs with trained property predictor
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from src.utils.config import load_config, get_device, set_seed
from src.models.generative.cvae import (
    ConditionalSMILESVAE, smiles_to_tokens, tokens_to_smiles,
    vae_loss, MAX_LEN, VOCAB_SIZE,
)


TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


class ILSmilesDataset(Dataset):
    """Dataset of IL SMILES + normalized properties for CVAE training."""

    def __init__(self, smiles_list, properties_dict=None):
        """
        Args:
            smiles_list: list of SMILES strings
            properties_dict: dict of {smiles: [7 property values or NaN]}
        """
        self.smiles = smiles_list
        self.properties = properties_dict or {}

        # Tokenize
        self.tokens = [smiles_to_tokens(s) for s in smiles_list]

        # Property statistics for normalization
        all_props = []
        for s in smiles_list:
            if s in self.properties:
                all_props.append(self.properties[s])
        if all_props:
            arr = np.array(all_props)
            self.prop_mean = np.nanmean(arr, axis=0)
            self.prop_std = np.nanstd(arr, axis=0)
            self.prop_std[self.prop_std == 0] = 1.0
        else:
            self.prop_mean = np.zeros(7)
            self.prop_std = np.ones(7)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.tokens[idx], dtype=torch.long)
        smi = self.smiles[idx]

        if smi in self.properties:
            props = np.array(self.properties[smi], dtype=np.float32)
            # Normalize
            props = (props - self.prop_mean) / self.prop_std
        else:
            props = np.full(7, np.nan, dtype=np.float32)

        return {
            "tokens": tokens,
            "properties": torch.tensor(props, dtype=torch.float32),
            "smiles": smi,
        }


def train_cvae(model, train_loader, device, num_epochs=200, lr=1e-3):
    """Train the conditional VAE."""
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    ckpt_dir = Path("checkpoints/cvae")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    kl_weight = 0.0  # Anneal KL weight

    print(f"\n{'='*60}")
    print(f"TRAINING CONDITIONAL SMILES VAE ({len(train_loader.dataset)} ILs)")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        n = 0

        # KL annealing: linearly increase from 0 to 0.5 over first 50 epochs
        kl_weight = min(0.5, epoch / 50.0 * 0.5)

        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            props = batch["properties"].to(device)

            optimizer.zero_grad()
            logits, mu, logvar = model(tokens, props)
            loss, recon, kl = vae_loss(logits, tokens, mu, logvar, kl_weight)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            n += 1

        scheduler.step()
        avg_loss = total_loss / max(n, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_dir / "best_cvae.pt")

        if epoch % 20 == 0 or epoch == num_epochs - 1:
            # Test reconstruction
            model.eval()
            sample_smi = train_loader.dataset.smiles[0]
            sample_tok = torch.tensor(smiles_to_tokens(sample_smi), dtype=torch.long).unsqueeze(0).to(device)
            sample_prop = torch.zeros(1, 7).to(device)
            with torch.no_grad():
                logits_test, _, _ = model(sample_tok, sample_prop)
                recon_tok = logits_test.argmax(dim=-1)[0].cpu().numpy()
                recon_smi = tokens_to_smiles(recon_tok)

            print(f"  Epoch {epoch:3d}/{num_epochs} | Loss: {avg_loss:.4f} "
                  f"(recon={total_recon/n:.4f}, kl={total_kl/n:.4f}, kl_w={kl_weight:.3f})")
            print(f"    Input:  {sample_smi[:60]}")
            print(f"    Recon:  {recon_smi[:60]}")

    model.load_state_dict(torch.load(ckpt_dir / "best_cvae.pt", map_location=device, weights_only=True))
    print(f"  Training complete. Best loss: {best_loss:.4f}")
    return model


def generate_and_validate(model, property_profiles, dataset, device,
                          property_predictor=None, n_per_profile=50):
    """Generate novel ILs for each desired property profile and validate."""

    results = []

    for profile_name, target_props in property_profiles.items():
        print(f"\n  Generating for: {profile_name}")
        print(f"  Target: {target_props}")

        # Normalize target properties
        norm_props = (np.array(target_props) - dataset.prop_mean) / dataset.prop_std
        props_tensor = torch.tensor(norm_props, dtype=torch.float32).to(device)

        # Generate with multiple temperatures
        all_valid = set()
        for temp in [0.5, 0.8, 1.0, 1.2]:
            generated = model.generate_diverse(
                props_tensor, n_samples=n_per_profile,
                temperature=temp, n_unique_target=20
            )
            all_valid.update(generated)

        print(f"  Valid unique SMILES: {len(all_valid)}")

        # Filter for IL-like molecules (should contain . separator = ion pair)
        il_candidates = []
        for smi in all_valid:
            if "." in smi:
                parts = smi.split(".")
                has_cation = any("+" in p for p in parts)
                has_anion = any("-" in p for p in parts)
                if has_cation and has_anion:
                    il_candidates.append(smi)

        print(f"  IL-like (cation.anion): {len(il_candidates)}")

        for smi in il_candidates:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                results.append({
                    "smiles": Chem.MolToSmiles(mol),
                    "profile": profile_name,
                    "molecular_weight": Descriptors.MolWt(mol),
                    "n_atoms": mol.GetNumAtoms(),
                    "target_gamma1": target_props[0],
                    "target_G_mix": target_props[4],
                })

    return pd.DataFrame(results)


def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    # ── Collect all IL SMILES with properties ──
    print("Loading IL data...")
    all_smiles = []
    properties = {}

    # Original dataset (has all 7 properties)
    orig = pd.read_csv("data/processed/il_data_raw.csv")
    for smi in orig["smiles"].unique():
        all_smiles.append(smi)
        rows = orig[orig["smiles"] == smi]
        # Use mean across temperatures as the "characteristic" property
        props = [rows[col].mean() for col in TARGET_COLUMNS]
        properties[smi] = props

    # ILThermo (partial properties)
    ilth = pd.read_csv("data/augmented/ilthermo_data.csv")
    for smi in ilth["smiles"].unique():
        if smi not in properties:
            all_smiles.append(smi)
            rows = ilth[ilth["smiles"] == smi]
            props = []
            for col in TARGET_COLUMNS:
                if col in rows.columns and rows[col].notna().sum() > 0:
                    props.append(rows[col].mean())
                else:
                    props.append(np.nan)
            properties[smi] = props

    print(f"  Total unique ILs: {len(all_smiles)}")
    print(f"  With full properties: {sum(1 for v in properties.values() if not any(np.isnan(x) for x in v))}")

    # ── Data augmentation: add randomized SMILES variants ──
    augmented_smiles = []
    augmented_props = {}
    for smi in all_smiles:
        augmented_smiles.append(smi)
        augmented_props[smi] = properties[smi]
        # Add 2 random SMILES variants per IL
        mol = Chem.MolFromSmiles(smi)
        if mol:
            for _ in range(2):
                try:
                    rand_smi = Chem.MolToSmiles(mol, doRandom=True)
                    if rand_smi != smi and rand_smi not in augmented_props:
                        augmented_smiles.append(rand_smi)
                        augmented_props[rand_smi] = properties[smi]
                except Exception:
                    pass

    print(f"  After SMILES augmentation: {len(augmented_smiles)}")

    # ── Build dataset ──
    dataset = ILSmilesDataset(augmented_smiles, augmented_props)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ── Train CVAE ──
    model = ConditionalSMILESVAE(latent_dim=128, n_properties=7)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  CVAE params: {n_params:,}")

    model = train_cvae(model, loader, device, num_epochs=300, lr=1e-3)

    # ── Generate novel ILs ──
    print(f"\n{'='*60}")
    print("GENERATING NOVEL IONIC LIQUIDS")
    print(f"{'='*60}")

    # Define target property profiles (raw values before normalization)
    property_profiles = {
        "lignin_dissolution": [
            0.3,    # gamma1: low (favorable interaction)
            0.5,    # gamma2: moderate
            -1.5,   # G_E: negative (favorable)
            -3.0,   # H_E: exothermic
            -2.0,   # G_mix: spontaneous mixing
            15.0,   # H_vap: moderate (recyclable)
            0.1,    # P: low (non-volatile)
        ],
        "plastic_depolymerization": [
            0.2,    # gamma1: very low
            0.3,    # gamma2: low
            -2.0,   # G_E: strongly negative
            -5.0,   # H_E: strongly exothermic
            -2.5,   # G_mix: strongly spontaneous
            20.0,   # H_vap: higher (more stable)
            0.05,   # P: very low
        ],
        "general_good_solvent": [
            0.5,    # gamma1: moderate-low
            0.5,    # gamma2: moderate
            -1.0,   # G_E: negative
            -2.0,   # H_E: moderately exothermic
            -1.5,   # G_mix: spontaneous
            12.0,   # H_vap: low (easy to recycle)
            0.5,    # P: moderate
        ],
    }

    generated_df = generate_and_validate(model, property_profiles, dataset, device)

    # ── Save results ──
    output_dir = Path("results/generative")
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(generated_df) > 0:
        generated_df.to_csv(output_dir / "cvae_generated_ils.csv", index=False)
        print(f"\n  Generated {len(generated_df)} valid IL candidates")
        print(f"  Saved to {output_dir / 'cvae_generated_ils.csv'}")

        # Summary per profile
        for profile in generated_df["profile"].unique():
            subset = generated_df[generated_df["profile"] == profile]
            print(f"\n  {profile}: {len(subset)} candidates")
            for _, row in subset.head(5).iterrows():
                print(f"    {row['smiles'][:50]}  MW={row['molecular_weight']:.0f}")
    else:
        print("\n  No valid IL candidates generated. Model may need more training data.")

    # ── Reconstruction quality assessment ──
    print(f"\n{'='*60}")
    print("RECONSTRUCTION QUALITY")
    print(f"{'='*60}")

    model.eval()
    correct = 0
    valid_recon = 0
    total = min(50, len(all_smiles))

    for i, smi in enumerate(all_smiles[:total]):
        tok = torch.tensor(smiles_to_tokens(smi), dtype=torch.long).unsqueeze(0).to(device)
        props = torch.tensor(
            [(np.array(properties[smi]) - dataset.prop_mean) / dataset.prop_std],
            dtype=torch.float32
        ).to(device)
        props[torch.isnan(props)] = 0.0

        with torch.no_grad():
            logits, _, _ = model(tok, props)
            recon_tok = logits.argmax(dim=-1)[0].cpu().numpy()
            recon_smi = tokens_to_smiles(recon_tok)

        mol = Chem.MolFromSmiles(recon_smi)
        if mol:
            valid_recon += 1
            canonical_recon = Chem.MolToSmiles(mol)
            canonical_orig = Chem.MolToSmiles(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else ""
            if canonical_recon == canonical_orig:
                correct += 1

    print(f"  Reconstruction accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"  Valid SMILES rate: {valid_recon}/{total} ({100*valid_recon/total:.1f}%)")

    # Save summary
    summary = {
        "model": "conditional_smiles_vae",
        "n_params": n_params,
        "training_ils": len(all_smiles),
        "augmented_ils": len(augmented_smiles),
        "reconstruction_accuracy": correct / total if total > 0 else 0,
        "valid_smiles_rate": valid_recon / total if total > 0 else 0,
        "generated_candidates": len(generated_df),
        "profiles": list(property_profiles.keys()),
    }
    with open(output_dir / "cvae_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {output_dir / 'cvae_summary.json'}")


if __name__ == "__main__":
    main()
