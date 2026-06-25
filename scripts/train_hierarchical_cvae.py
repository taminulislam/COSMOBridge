"""Train Hierarchical CVAE and generate novel ionic liquids.

Stage 1: Build SELFIES vocabularies for cations and anions separately
Stage 2: Train Hierarchical CVAE with property conditioning
Stage 3: Generate novel ILs for target applications
Stage 4: Validate with trained property predictor
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

import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors

from src.utils.config import load_config, get_device, set_seed
from src.models.generative.hierarchical_cvae import (
    HierarchicalCVAE, build_selfies_vocab, selfies_to_tokens,
    tokens_to_selfies, selfies_to_smiles, hierarchical_vae_loss,
)


TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


# ── Dataset ──────────────────────────────────────────────────────────────────

class IonPairDataset(Dataset):
    """Dataset of separated cation/anion SELFIES + properties."""

    def __init__(self, cation_selfies, anion_selfies, properties,
                 cat_token_to_idx, an_token_to_idx, max_len=80):
        self.cat_selfies = cation_selfies
        self.an_selfies = anion_selfies
        self.properties = properties
        self.cat_t2i = cat_token_to_idx
        self.an_t2i = an_token_to_idx
        self.max_len = max_len

        # Normalize properties
        arr = np.array(properties)
        self.prop_mean = np.nanmean(arr, axis=0)
        self.prop_std = np.nanstd(arr, axis=0)
        self.prop_std[self.prop_std == 0] = 1.0

    def __len__(self):
        return len(self.cat_selfies)

    def __getitem__(self, idx):
        cat_tok = selfies_to_tokens(self.cat_selfies[idx], self.cat_t2i, self.max_len)
        an_tok = selfies_to_tokens(self.an_selfies[idx], self.an_t2i, self.max_len)

        props = np.array(self.properties[idx], dtype=np.float32)
        props = (props - self.prop_mean) / self.prop_std

        return {
            "cat_tokens": torch.tensor(cat_tok, dtype=torch.long),
            "an_tokens": torch.tensor(an_tok, dtype=torch.long),
            "properties": torch.tensor(props, dtype=torch.float32),
        }


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(model, loader, device, num_epochs=500, lr=5e-4):
    """Train hierarchical CVAE with KL annealing."""
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    ckpt_dir = Path("checkpoints/hierarchical_cvae")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"TRAINING HIERARCHICAL CVAE ({len(loader.dataset)} ion pairs)")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        model.train()
        total_losses = {"total": 0, "cat_recon": 0, "an_recon": 0,
                        "kl": 0, "prop_loss": 0}
        n = 0

        kl_weight = min(0.5, epoch / 100.0 * 0.5)
        prop_weight = min(1.0, epoch / 50.0)

        for batch in loader:
            cat_tok = batch["cat_tokens"].to(device)
            an_tok = batch["an_tokens"].to(device)
            props = batch["properties"].to(device)

            optimizer.zero_grad()
            outputs = model(cat_tok, an_tok, props)
            losses = hierarchical_vae_loss(outputs, cat_tok, an_tok, props,
                                           kl_weight=kl_weight, prop_weight=prop_weight)

            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in total_losses:
                total_losses[k] += losses.get(k, torch.tensor(0.0)).item()
            n += 1

        scheduler.step()
        avg = {k: v / max(n, 1) for k, v in total_losses.items()}

        if avg["total"] < best_loss:
            best_loss = avg["total"]
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch:3d}/{num_epochs} | Loss: {avg['total']:.4f} "
                  f"(cat={avg['cat_recon']:.3f} an={avg['an_recon']:.3f} "
                  f"kl={avg['kl']:.3f} prop={avg['prop_loss']:.3f}) "
                  f"kl_w={kl_weight:.2f}")

    model.load_state_dict(torch.load(ckpt_dir / "best_model.pt",
                                      map_location=device, weights_only=True))
    print(f"  Training complete. Best loss: {best_loss:.4f}")
    return model


# ── Generation + Validation ──────────────────────────────────────────────────

def generate_novel_ils(model, cat_idx_to_token, an_idx_to_token, dataset,
                       property_profiles, device, n_per_profile=200):
    """Generate and validate novel ILs for each property profile."""

    all_results = []

    for profile_name, raw_props in property_profiles.items():
        print(f"\n  Generating for: {profile_name}")

        # Normalize using dataset stats
        norm_props = (np.array(raw_props) - dataset.prop_mean) / dataset.prop_std
        props_tensor = torch.tensor(norm_props, dtype=torch.float32).to(device)

        valid_ils = []
        seen = set()

        for temp in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]:
            cat_tokens, an_tokens, compat, pred_props = model.generate(
                props_tensor, n_samples=n_per_profile, temperature=temp)

            for i in range(len(cat_tokens)):
                cat_sel = tokens_to_selfies(cat_tokens[i], cat_idx_to_token)
                an_sel = tokens_to_selfies(an_tokens[i], an_idx_to_token)

                cat_smi = selfies_to_smiles(cat_sel)
                an_smi = selfies_to_smiles(an_sel)

                if cat_smi and an_smi:
                    # Verify it's an actual cation-anion pair
                    if "+" in cat_smi and "-" in an_smi:
                        il_smi = f"{cat_smi}.{an_smi}"
                        mol = Chem.MolFromSmiles(il_smi)
                        if mol and il_smi not in seen:
                            seen.add(il_smi)
                            canonical = Chem.MolToSmiles(mol)
                            valid_ils.append({
                                "il_smiles": canonical,
                                "cation_smiles": cat_smi,
                                "anion_smiles": an_smi,
                                "profile": profile_name,
                                "compatibility": float(compat[i]),
                                "temperature": temp,
                                "mw": Descriptors.MolWt(mol),
                                "n_atoms": mol.GetNumAtoms(),
                            })

        print(f"  Valid unique ILs: {len(valid_ils)}")
        all_results.extend(valid_ils)

    return pd.DataFrame(all_results)


def validate_with_predictor(generated_df, device):
    """Validate generated ILs using the trained property predictor."""
    from src.data.preprocessing import FEATURE_COLUMNS
    from src.data.graph_builder import smiles_to_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
    from src.models.graph.gnn import MolecularGNN

    # Load Phase 2 GNN
    model = MolecularGNN(
        atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
        hidden_dim=256, num_layers=4, conv_type="GAT", heads=4,
        dropout=0.3, pooling="mean", num_targets=7,
        aux_feature_dim=len(FEATURE_COLUMNS),
    )
    ckpt = Path("checkpoints/transfer/finetune/best_model.pt")
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()

    feat_template = torch.zeros(len(FEATURE_COLUMNS)).to(device)

    predictions = []
    for _, row in generated_df.iterrows():
        smi = row["il_smiles"]
        try:
            g = smiles_to_graph(smi)
            af = torch.tensor(g["atom_features"], dtype=torch.float32).to(device)
            ei = torch.tensor(g["edge_index"], dtype=torch.long).to(device)
            bf = torch.tensor(g["bond_features"], dtype=torch.float32).to(device)
            batch = torch.zeros(af.shape[0], dtype=torch.long).to(device)

            with torch.no_grad():
                pred = model(atom_features=af, edge_index=ei, bond_features=bf,
                            batch=batch, features=feat_template.unsqueeze(0))
            pred = pred.cpu().numpy().flatten()
            predictions.append({f"{col}_pred": float(pred[i])
                               for i, col in enumerate(TARGET_COLUMNS)})
        except Exception:
            predictions.append({f"{col}_pred": np.nan for col in TARGET_COLUMNS})

    pred_df = pd.DataFrame(predictions)
    return pd.concat([generated_df.reset_index(drop=True), pred_df], axis=1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    config = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device(config)
    print(f"Device: {device}")

    # ── Collect all IL data ──
    print("Loading IL data...")
    all_cation_smiles = []
    all_anion_smiles = []
    all_properties = []

    for csv_path in ["data/processed/il_data_raw.csv", "data/augmented/ilthermo_data.csv"]:
        if not Path(csv_path).exists():
            continue
        df = pd.read_csv(csv_path)
        for smi in df["smiles"].unique():
            parts = smi.split(".")
            if len(parts) != 2:
                continue
            cat, an = None, None
            for p in parts:
                if "+" in p:
                    cat = p
                elif "-" in p:
                    an = p
            if not cat or not an:
                continue

            rows = df[df["smiles"] == smi]
            props = []
            for col in TARGET_COLUMNS:
                if col in rows.columns and rows[col].notna().sum() > 0:
                    props.append(float(rows[col].mean()))
                else:
                    props.append(np.nan)

            all_cation_smiles.append(cat)
            all_anion_smiles.append(an)
            all_properties.append(props)

    print(f"  Ion pairs: {len(all_cation_smiles)}")
    print(f"  Unique cations: {len(set(all_cation_smiles))}")
    print(f"  Unique anions: {len(set(all_anion_smiles))}")

    # ── Convert to SELFIES ──
    print("\nConverting to SELFIES...")
    cat_selfies = []
    an_selfies = []
    valid_props = []
    failed = 0

    for i in range(len(all_cation_smiles)):
        try:
            cs = sf.encoder(all_cation_smiles[i])
            ans = sf.encoder(all_anion_smiles[i])
            if cs and ans:
                cat_selfies.append(cs)
                an_selfies.append(ans)
                valid_props.append(all_properties[i])
            else:
                failed += 1
        except Exception:
            failed += 1

    print(f"  Valid SELFIES pairs: {len(cat_selfies)} ({failed} failed)")

    # Add SMILES augmentation (randomized SMILES → SELFIES)
    aug_cat, aug_an, aug_props = [], [], []
    for i in range(len(all_cation_smiles)):
        for _ in range(3):
            try:
                mol_c = Chem.MolFromSmiles(all_cation_smiles[i])
                mol_a = Chem.MolFromSmiles(all_anion_smiles[i])
                if mol_c and mol_a:
                    rand_c = Chem.MolToSmiles(mol_c, doRandom=True)
                    rand_a = Chem.MolToSmiles(mol_a, doRandom=True)
                    sel_c = sf.encoder(rand_c)
                    sel_a = sf.encoder(rand_a)
                    if sel_c and sel_a:
                        aug_cat.append(sel_c)
                        aug_an.append(sel_a)
                        aug_props.append(all_properties[i])
            except Exception:
                pass

    cat_selfies.extend(aug_cat)
    an_selfies.extend(aug_an)
    valid_props.extend(aug_props)
    print(f"  After augmentation: {len(cat_selfies)} pairs")

    # ── Build vocabularies ──
    print("\nBuilding vocabularies...")
    cat_vocab, cat_t2i, cat_i2t = build_selfies_vocab(all_cation_smiles)
    an_vocab, an_t2i, an_i2t = build_selfies_vocab(all_anion_smiles)
    print(f"  Cation vocab: {len(cat_vocab)} tokens")
    print(f"  Anion vocab: {len(an_vocab)} tokens")

    # ── Build dataset ──
    dataset = IonPairDataset(cat_selfies, an_selfies, valid_props,
                             cat_t2i, an_t2i, max_len=80)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ── Train ──
    model = HierarchicalCVAE(
        cation_vocab_size=len(cat_vocab),
        anion_vocab_size=len(an_vocab),
        latent_dim=64,
        n_properties=7,
        max_len=80,
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    model = train_model(model, loader, device, num_epochs=500, lr=5e-4)

    # ── Reconstruction test ──
    print(f"\n{'='*60}")
    print("RECONSTRUCTION TEST")
    print(f"{'='*60}")

    model.eval()
    correct_cat, correct_an, valid_cat, valid_an = 0, 0, 0, 0
    n_test = min(50, len(cat_selfies))

    for i in range(n_test):
        cat_tok = torch.tensor(selfies_to_tokens(cat_selfies[i], cat_t2i, 80),
                               dtype=torch.long).unsqueeze(0).to(device)
        an_tok = torch.tensor(selfies_to_tokens(an_selfies[i], an_t2i, 80),
                              dtype=torch.long).unsqueeze(0).to(device)
        props = torch.tensor(
            [(np.array(valid_props[i]) - dataset.prop_mean) / dataset.prop_std],
            dtype=torch.float32).to(device)
        props[torch.isnan(props)] = 0.0

        with torch.no_grad():
            outputs = model(cat_tok, an_tok, props)

        # Check cation reconstruction
        recon_cat = tokens_to_selfies(outputs["cat_logits"].argmax(dim=-1)[0].cpu().numpy(), cat_i2t)
        cat_smi = selfies_to_smiles(recon_cat)
        if cat_smi:
            valid_cat += 1
            orig_cat = selfies_to_smiles(cat_selfies[i])
            if cat_smi == orig_cat:
                correct_cat += 1

        # Check anion reconstruction
        recon_an = tokens_to_selfies(outputs["an_logits"].argmax(dim=-1)[0].cpu().numpy(), an_i2t)
        an_smi = selfies_to_smiles(recon_an)
        if an_smi:
            valid_an += 1
            orig_an = selfies_to_smiles(an_selfies[i])
            if an_smi == orig_an:
                correct_an += 1

    print(f"  Cation: {correct_cat}/{n_test} exact ({100*valid_cat/n_test:.0f}% valid)")
    print(f"  Anion:  {correct_an}/{n_test} exact ({100*valid_an/n_test:.0f}% valid)")

    # ── Generate novel ILs ──
    print(f"\n{'='*60}")
    print("GENERATING NOVEL IONIC LIQUIDS")
    print(f"{'='*60}")

    property_profiles = {
        "lignin_dissolution": [0.3, 0.5, -1.5, -3.0, -2.0, 15.0, 0.1],
        "plastic_depolymerization": [0.2, 0.3, -2.0, -5.0, -2.5, 20.0, 0.05],
        "general_good_solvent": [0.5, 0.5, -1.0, -2.0, -1.5, 12.0, 0.5],
    }

    generated_df = generate_novel_ils(
        model, cat_i2t, an_i2t, dataset,
        property_profiles, device, n_per_profile=200)

    if len(generated_df) > 0:
        # Validate with property predictor
        print(f"\n  Validating {len(generated_df)} candidates with property predictor...")
        generated_df = validate_with_predictor(generated_df, device)

        # Save
        output_dir = Path("results/generative")
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_df.to_csv(output_dir / "hierarchical_cvae_generated.csv", index=False)

        print(f"\n  Generated {len(generated_df)} valid IL candidates")

        # Show top candidates per profile
        for profile in generated_df["profile"].unique():
            subset = generated_df[generated_df["profile"] == profile]
            print(f"\n  {profile}: {len(subset)} candidates")
            # Sort by compatibility score
            subset = subset.sort_values("compatibility", ascending=False)
            for j, (_, row) in enumerate(subset.head(5).iterrows()):
                print(f"    #{j+1}: {row['il_smiles'][:55]}")
                print(f"         MW={row['mw']:.0f}  compat={row['compatibility']:.3f}", end="")
                if not np.isnan(row.get("gamma1_pred", np.nan)):
                    print(f"  g1={row['gamma1_pred']:.3f} G_mix={row.get('G_mix_pred', np.nan):.3f}", end="")
                print()

        # Check novelty
        known = set()
        for csv_path in ["data/processed/il_data_raw.csv", "data/augmented/ilthermo_data.csv"]:
            if Path(csv_path).exists():
                known.update(pd.read_csv(csv_path)["smiles"].unique())
        novel = generated_df[~generated_df["il_smiles"].isin(known)]
        print(f"\n  Novel (never seen in training): {len(novel)}/{len(generated_df)}")
    else:
        print("\n  No valid candidates generated.")

    # Save summary
    output_dir = Path("results/generative")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "hierarchical_cvae",
        "n_params": n_params,
        "training_pairs": len(cat_selfies),
        "cation_vocab": len(cat_vocab),
        "anion_vocab": len(an_vocab),
        "cation_recon_accuracy": correct_cat / n_test,
        "anion_recon_accuracy": correct_an / n_test,
        "cation_valid_rate": valid_cat / n_test,
        "anion_valid_rate": valid_an / n_test,
        "generated_candidates": len(generated_df) if len(generated_df) > 0 else 0,
    }
    with open(output_dir / "hierarchical_cvae_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {output_dir / 'hierarchical_cvae_summary.json'}")


if __name__ == "__main__":
    main()
