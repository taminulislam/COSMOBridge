"""Conditional Variational Autoencoder for Ionic Liquid SMILES Generation.

Encodes IL SMILES into a continuous latent space conditioned on desired
thermodynamic properties. Decoding samples from the conditioned latent
space produces novel IL SMILES with target property profiles.

Architecture:
  Encoder: SMILES → token embeddings → GRU → μ, σ (conditioned on properties)
  Decoder: z + properties → GRU → SMILES tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# SMILES vocabulary
SMILES_CHARS = [
    'PAD', 'START', 'END',
    'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'P', 'p',
    'F', 'I',
    'H',
    '(', ')', '[', ']',
    '=', '#', '+', '-',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    '.', '/',  '\\', '@',
    'l',  # for Cl
    'r',  # for Br
    'e',  # for Se, Fe
    'i',  # for Si
    'b',  # for aromatic B
]

CHAR_TO_IDX = {c: i for i, c in enumerate(SMILES_CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(SMILES_CHARS)}
VOCAB_SIZE = len(SMILES_CHARS)
MAX_LEN = 150  # Max SMILES length


def smiles_to_tokens(smiles, max_len=MAX_LEN):
    """Convert SMILES string to token indices."""
    tokens = [CHAR_TO_IDX.get('START', 1)]
    for ch in smiles:
        idx = CHAR_TO_IDX.get(ch, 0)  # PAD for unknown
        tokens.append(idx)
    tokens.append(CHAR_TO_IDX.get('END', 2))
    # Pad or truncate
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len-1] + [CHAR_TO_IDX.get('END', 2)]
    return tokens


def tokens_to_smiles(tokens):
    """Convert token indices back to SMILES string."""
    chars = []
    for idx in tokens:
        if idx == CHAR_TO_IDX.get('END', 2):
            break
        if idx == CHAR_TO_IDX.get('START', 1) or idx == 0:
            continue
        ch = IDX_TO_CHAR.get(idx, '')
        chars.append(ch)
    return ''.join(chars)


class ConditionalEncoder(nn.Module):
    """Encode SMILES tokens + property conditions into latent distribution."""

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64, hidden_dim=256,
                 latent_dim=128, n_properties=7):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=2,
                          batch_first=True, bidirectional=True)

        # Property conditioning
        self.prop_proj = nn.Sequential(
            nn.Linear(n_properties, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # Latent projections
        combined_dim = hidden_dim * 2 + 64  # bidirectional + properties
        self.fc_mu = nn.Linear(combined_dim, latent_dim)
        self.fc_logvar = nn.Linear(combined_dim, latent_dim)

    def forward(self, tokens, properties):
        """
        Args:
            tokens: (B, max_len) token indices
            properties: (B, n_properties) target properties (can have NaN)
        Returns:
            mu, logvar: (B, latent_dim) each
        """
        x = self.embedding(tokens)  # (B, L, embed)
        _, h = self.rnn(x)  # h: (2*num_layers, B, hidden)
        # Use last layer's forward + backward hidden states
        h = torch.cat([h[-2], h[-1]], dim=1)  # (B, hidden*2)

        # Property conditioning (replace NaN with 0)
        props = properties.clone()
        props[torch.isnan(props)] = 0.0
        prop_feat = self.prop_proj(props)  # (B, 64)

        combined = torch.cat([h, prop_feat], dim=1)
        return self.fc_mu(combined), self.fc_logvar(combined)


class ConditionalDecoder(nn.Module):
    """Decode latent vector + properties into SMILES tokens."""

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64, hidden_dim=256,
                 latent_dim=128, n_properties=7, max_len=MAX_LEN):
        super().__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Condition on z + properties
        self.cond_proj = nn.Sequential(
            nn.Linear(latent_dim + 64, hidden_dim * 2),
            nn.ReLU(),
        )
        self.prop_proj = nn.Sequential(
            nn.Linear(n_properties, 64),
            nn.ReLU(),
        )

        self.rnn = nn.GRU(embed_dim + latent_dim + 64, hidden_dim,
                          num_layers=2, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, properties, target_tokens=None):
        """
        Args:
            z: (B, latent_dim)
            properties: (B, n_properties)
            target_tokens: (B, max_len) for teacher forcing (training)
        Returns:
            logits: (B, max_len, vocab_size)
        """
        B = z.shape[0]

        props = properties.clone()
        props[torch.isnan(props)] = 0.0
        prop_feat = self.prop_proj(props)  # (B, 64)

        # Initial hidden state from z + properties
        h0 = self.cond_proj(torch.cat([z, prop_feat], dim=1))
        h0 = h0.view(B, 2, self.hidden_dim).permute(1, 0, 2).contiguous()

        # Context vector repeated for each timestep
        context = torch.cat([z, prop_feat], dim=1).unsqueeze(1)  # (B, 1, latent+64)
        context = context.expand(-1, self.max_len, -1)

        if target_tokens is not None:
            # Teacher forcing
            x = self.embedding(target_tokens)  # (B, L, embed)
            x = torch.cat([x, context], dim=2)  # (B, L, embed+latent+64)
            out, _ = self.rnn(x, h0)
            logits = self.output(out)
        else:
            # Autoregressive generation
            logits = []
            token = torch.ones(B, 1, dtype=torch.long, device=z.device)  # START
            h = h0
            for t in range(self.max_len):
                x = self.embedding(token)
                x = torch.cat([x, context[:, t:t+1, :]], dim=2)
                out, h = self.rnn(x, h)
                logit = self.output(out)
                logits.append(logit)
                token = logit.argmax(dim=-1)  # Greedy
            logits = torch.cat(logits, dim=1)

        return logits


class ConditionalSMILESVAE(nn.Module):
    """Conditional VAE for generating IL SMILES conditioned on properties.

    Training: Encode known ILs with their properties → reconstruct SMILES
    Generation: Sample z ~ N(0,1), condition on desired properties → decode SMILES
    """

    def __init__(self, latent_dim=128, n_properties=7):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConditionalEncoder(latent_dim=latent_dim, n_properties=n_properties)
        self.decoder = ConditionalDecoder(latent_dim=latent_dim, n_properties=n_properties)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, tokens, properties):
        """Training forward pass with teacher forcing."""
        mu, logvar = self.encoder(tokens, properties)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z, properties, target_tokens=tokens)
        return logits, mu, logvar

    def generate(self, properties, n_samples=1, temperature=1.0):
        """Generate SMILES conditioned on desired properties.

        Args:
            properties: (n_properties,) or (B, n_properties)
            n_samples: number of samples to generate
            temperature: sampling temperature (higher = more diverse)

        Returns:
            list of SMILES strings
        """
        self.eval()
        device = next(self.parameters()).device

        if properties.dim() == 1:
            properties = properties.unsqueeze(0).expand(n_samples, -1)

        z = torch.randn(n_samples, self.latent_dim, device=device) * temperature

        with torch.no_grad():
            logits = self.decoder(z, properties)

        tokens = logits.argmax(dim=-1).cpu().numpy()
        return [tokens_to_smiles(t) for t in tokens]

    def generate_diverse(self, properties, n_samples=100, temperature=1.0,
                         n_unique_target=20):
        """Generate diverse valid SMILES by sampling multiple times."""
        from rdkit import Chem
        valid_smiles = set()
        attempts = 0
        max_attempts = n_samples * 5

        while len(valid_smiles) < n_unique_target and attempts < max_attempts:
            batch = min(n_samples, max_attempts - attempts)
            generated = self.generate(properties, n_samples=batch, temperature=temperature)
            for smi in generated:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    canonical = Chem.MolToSmiles(mol)
                    valid_smiles.add(canonical)
            attempts += batch

        return sorted(valid_smiles)


def vae_loss(logits, target_tokens, mu, logvar, kl_weight=0.1):
    """VAE ELBO loss = reconstruction + KL divergence."""
    # Reconstruction loss (cross-entropy)
    B, L, V = logits.shape
    recon = F.cross_entropy(
        logits.reshape(B * L, V),
        target_tokens.reshape(B * L),
        ignore_index=0,  # Ignore PAD
        reduction='mean',
    )

    # KL divergence
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon + kl_weight * kl, recon, kl
