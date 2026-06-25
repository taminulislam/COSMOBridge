"""Hierarchical Conditional VAE for Ionic Liquid Generation.

Key improvements over basic CVAE:
1. Separate cation/anion encoders+decoders (compositional structure)
2. SELFIES encoding (100% valid molecules by construction)
3. Joint latent space with cation-anion compatibility modeling
4. Property predictor reward signal during training

Architecture:
  Cation Encoder: SELFIES tokens → GRU → μ_cat, σ_cat
  Anion Encoder:  SELFIES tokens → GRU → μ_an, σ_an
  Joint Layer:    [z_cat, z_an] → compatibility score + property prediction
  Cation Decoder: z_cat + properties → GRU → SELFIES tokens
  Anion Decoder:  z_an + properties → GRU → SELFIES tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import selfies as sf
from rdkit import Chem


# ── SELFIES Vocabulary ───────────────────────────────────────────────────────

def build_selfies_vocab(smiles_list):
    """Build SELFIES vocabulary from a list of SMILES."""
    all_tokens = set()
    all_tokens.add('[PAD]')
    all_tokens.add('[START]')
    all_tokens.add('[END]')

    for smi in smiles_list:
        try:
            sel = sf.encoder(smi)
            if sel:
                tokens = list(sf.split_selfies(sel))
                all_tokens.update(tokens)
        except Exception:
            pass

    vocab = sorted(all_tokens)
    token_to_idx = {t: i for i, t in enumerate(vocab)}
    idx_to_token = {i: t for i, t in enumerate(vocab)}
    return vocab, token_to_idx, idx_to_token


def selfies_to_tokens(selfies_str, token_to_idx, max_len=80):
    """Convert SELFIES string to token indices."""
    tokens = [token_to_idx.get('[START]', 1)]
    if selfies_str:
        for tok in sf.split_selfies(selfies_str):
            idx = token_to_idx.get(tok, 0)
            tokens.append(idx)
    tokens.append(token_to_idx.get('[END]', 2))

    if len(tokens) < max_len:
        pad_idx = token_to_idx.get('[PAD]', 0)
        tokens += [pad_idx] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len - 1] + [token_to_idx.get('[END]', 2)]
    return tokens


def tokens_to_selfies(token_ids, idx_to_token):
    """Convert token indices back to SELFIES string."""
    tokens = []
    for idx in token_ids:
        tok = idx_to_token.get(idx, '')
        if tok == '[END]':
            break
        if tok in ('[PAD]', '[START]'):
            continue
        tokens.append(tok)
    return ''.join(tokens)


def selfies_to_smiles(selfies_str):
    """Convert SELFIES to canonical SMILES."""
    try:
        smi = sf.decoder(selfies_str)
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return None


# ── Ion Encoder ──────────────────────────────────────────────────────────────

class IonEncoder(nn.Module):
    """Encode ion SELFIES tokens into latent distribution."""

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256,
                 latent_dim=64, n_properties=7):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=2,
                          batch_first=True, bidirectional=True)

        self.prop_proj = nn.Sequential(
            nn.Linear(n_properties, 32),
            nn.ReLU(),
        )

        combined_dim = hidden_dim * 2 + 32
        self.fc_mu = nn.Linear(combined_dim, latent_dim)
        self.fc_logvar = nn.Linear(combined_dim, latent_dim)

    def forward(self, tokens, properties):
        x = self.embedding(tokens)
        _, h = self.rnn(x)
        h = torch.cat([h[-2], h[-1]], dim=1)

        props = properties.clone()
        props[torch.isnan(props)] = 0.0
        prop_feat = self.prop_proj(props)

        combined = torch.cat([h, prop_feat], dim=1)
        return self.fc_mu(combined), self.fc_logvar(combined)


# ── Ion Decoder ──────────────────────────────────────────────────────────────

class IonDecoder(nn.Module):
    """Decode latent vector + properties into ion SELFIES tokens."""

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256,
                 latent_dim=64, n_properties=7, max_len=80):
        super().__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.prop_proj = nn.Sequential(
            nn.Linear(n_properties, 32),
            nn.ReLU(),
        )

        # Initial hidden state from z + properties
        self.init_hidden = nn.Sequential(
            nn.Linear(latent_dim + 32, hidden_dim * 2),
            nn.Tanh(),
        )

        self.rnn = nn.GRU(embed_dim + latent_dim + 32, hidden_dim,
                          num_layers=2, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, properties, target_tokens=None):
        B = z.shape[0]

        props = properties.clone()
        props[torch.isnan(props)] = 0.0
        prop_feat = self.prop_proj(props)

        h0 = self.init_hidden(torch.cat([z, prop_feat], dim=1))
        h0 = h0.view(B, 2, self.hidden_dim).permute(1, 0, 2).contiguous()

        context = torch.cat([z, prop_feat], dim=1).unsqueeze(1).expand(-1, self.max_len, -1)

        if target_tokens is not None:
            x = self.embedding(target_tokens)
            x = torch.cat([x, context], dim=2)
            out, _ = self.rnn(x, h0)
            logits = self.output(out)
        else:
            logits = []
            start_idx = 1  # [START] token
            token = torch.full((B, 1), start_idx, dtype=torch.long, device=z.device)
            h = h0
            for t in range(self.max_len):
                x = self.embedding(token)
                x = torch.cat([x, context[:, t:t+1, :]], dim=2)
                out, h = self.rnn(x, h)
                logit = self.output(out)
                logits.append(logit)
                token = logit.argmax(dim=-1)
            logits = torch.cat(logits, dim=1)

        return logits


# ── Joint Compatibility Layer ────────────────────────────────────────────────

class JointCompatibilityLayer(nn.Module):
    """Models cation-anion compatibility in the joint latent space.

    Takes concatenated [z_cat, z_an] and predicts:
    1. Compatibility score (should the pair form a stable IL?)
    2. Property predictions from the joint latent representation
    """

    def __init__(self, cation_latent=64, anion_latent=64, n_properties=7):
        super().__init__()
        joint_dim = cation_latent + anion_latent

        self.compatibility = nn.Sequential(
            nn.Linear(joint_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.property_pred = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_properties),
        )

    def forward(self, z_cat, z_an):
        joint = torch.cat([z_cat, z_an], dim=1)
        compat = self.compatibility(joint)
        props = self.property_pred(joint)
        return compat, props


# ── Hierarchical CVAE ────────────────────────────────────────────────────────

class HierarchicalCVAE(nn.Module):
    """Hierarchical Conditional VAE for ionic liquid generation.

    Separate cation/anion VAEs with joint compatibility modeling
    and property-conditioned generation using SELFIES encoding.
    """

    def __init__(self, cation_vocab_size, anion_vocab_size,
                 latent_dim=64, n_properties=7, max_len=80):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len

        # Cation VAE
        self.cat_encoder = IonEncoder(cation_vocab_size, latent_dim=latent_dim,
                                       n_properties=n_properties)
        self.cat_decoder = IonDecoder(cation_vocab_size, latent_dim=latent_dim,
                                       n_properties=n_properties, max_len=max_len)

        # Anion VAE
        self.an_encoder = IonEncoder(anion_vocab_size, latent_dim=latent_dim,
                                      n_properties=n_properties)
        self.an_decoder = IonDecoder(anion_vocab_size, latent_dim=latent_dim,
                                      n_properties=n_properties, max_len=max_len)

        # Joint compatibility
        self.joint = JointCompatibilityLayer(latent_dim, latent_dim, n_properties)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, cat_tokens, an_tokens, properties):
        """Training forward pass."""
        # Encode
        cat_mu, cat_logvar = self.cat_encoder(cat_tokens, properties)
        an_mu, an_logvar = self.an_encoder(an_tokens, properties)

        # Sample
        z_cat = self.reparameterize(cat_mu, cat_logvar)
        z_an = self.reparameterize(an_mu, an_logvar)

        # Decode (teacher forcing)
        cat_logits = self.cat_decoder(z_cat, properties, target_tokens=cat_tokens)
        an_logits = self.an_decoder(z_an, properties, target_tokens=an_tokens)

        # Joint compatibility + property prediction
        compat, pred_props = self.joint(z_cat, z_an)

        return {
            "cat_logits": cat_logits,
            "an_logits": an_logits,
            "cat_mu": cat_mu, "cat_logvar": cat_logvar,
            "an_mu": an_mu, "an_logvar": an_logvar,
            "compatibility": compat,
            "predicted_properties": pred_props,
        }

    def generate(self, properties, n_samples=10, temperature=1.0):
        """Generate novel IL SMILES conditioned on desired properties.

        Args:
            properties: (n_properties,) tensor
            n_samples: number of ILs to generate
            temperature: sampling diversity

        Returns:
            list of (cation_smiles, anion_smiles, il_smiles) tuples
        """
        self.eval()
        device = next(self.parameters()).device

        if properties.dim() == 1:
            properties = properties.unsqueeze(0).expand(n_samples, -1)

        z_cat = torch.randn(n_samples, self.latent_dim, device=device) * temperature
        z_an = torch.randn(n_samples, self.latent_dim, device=device) * temperature

        with torch.no_grad():
            cat_logits = self.cat_decoder(z_cat, properties)
            an_logits = self.an_decoder(z_an, properties)
            compat, pred_props = self.joint(z_cat, z_an)

        cat_tokens = cat_logits.argmax(dim=-1).cpu().numpy()
        an_tokens = an_logits.argmax(dim=-1).cpu().numpy()
        compat_scores = compat.cpu().numpy().flatten()

        return cat_tokens, an_tokens, compat_scores, pred_props.cpu().numpy()


# ── Loss Function ────────────────────────────────────────────────────────────

def hierarchical_vae_loss(outputs, cat_targets, an_targets, properties,
                          kl_weight=0.1, prop_weight=0.5):
    """Combined loss: reconstruction + KL + property prediction + compatibility."""

    # Cation reconstruction
    B, L, V = outputs["cat_logits"].shape
    cat_recon = F.cross_entropy(
        outputs["cat_logits"].reshape(B * L, V),
        cat_targets.reshape(B * L),
        ignore_index=0, reduction='mean')

    # Anion reconstruction
    B2, L2, V2 = outputs["an_logits"].shape
    an_recon = F.cross_entropy(
        outputs["an_logits"].reshape(B2 * L2, V2),
        an_targets.reshape(B2 * L2),
        ignore_index=0, reduction='mean')

    # KL divergence (cation + anion)
    kl_cat = -0.5 * torch.mean(1 + outputs["cat_logvar"] - outputs["cat_mu"].pow(2) - outputs["cat_logvar"].exp())
    kl_an = -0.5 * torch.mean(1 + outputs["an_logvar"] - outputs["an_mu"].pow(2) - outputs["an_logvar"].exp())

    # Property prediction loss (only on non-NaN targets)
    pred_props = outputs["predicted_properties"]
    mask = ~torch.isnan(properties)
    if mask.sum() > 0:
        safe_props = properties.clone()
        safe_props[~mask] = 0.0
        prop_loss = ((pred_props - safe_props) ** 2 * mask.float()).sum() / mask.float().sum().clamp(min=1)
    else:
        prop_loss = torch.tensor(0.0, device=pred_props.device)

    # Compatibility loss (all known pairs are compatible = 1)
    compat_loss = F.binary_cross_entropy(outputs["compatibility"],
                                          torch.ones_like(outputs["compatibility"]))

    total = (cat_recon + an_recon +
             kl_weight * (kl_cat + kl_an) +
             prop_weight * prop_loss +
             0.1 * compat_loss)

    return {
        "total": total,
        "cat_recon": cat_recon,
        "an_recon": an_recon,
        "kl": kl_cat + kl_an,
        "prop_loss": prop_loss,
        "compat_loss": compat_loss,
    }
