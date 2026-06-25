"""Variational Autoencoder for generating novel ionic liquid SMILES.

Encodes SMILES strings into a continuous latent space and decodes
to generate new, chemically valid molecular structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# ── SMILES Vocabulary ──────────────────────────────────────────────────────

# Characters commonly found in ionic liquid SMILES
SMILES_CHARS = [
    "PAD", "SOS", "EOS",
    "C", "c", "N", "n", "O", "o", "S", "s", "F", "P",
    "H", "B", "I",
    "[", "]", "(", ")", "=", "#", "+", "-",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    ".", "/", "\\", "@",
    "l",  # for Cl, Br
    "r",  # for Br
]

CHAR_TO_IDX = {c: i for i, c in enumerate(SMILES_CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(SMILES_CHARS)}
VOCAB_SIZE = len(SMILES_CHARS)
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
MAX_SEQ_LEN = 120


def smiles_to_indices(smiles: str, max_len: int = MAX_SEQ_LEN) -> list:
    """Convert SMILES string to list of character indices."""
    indices = [SOS_IDX]
    for c in smiles:
        idx = CHAR_TO_IDX.get(c, CHAR_TO_IDX.get("C"))  # default to C for unknown
        indices.append(idx)
    indices.append(EOS_IDX)
    # Pad
    while len(indices) < max_len:
        indices.append(PAD_IDX)
    return indices[:max_len]


def indices_to_smiles(indices: list) -> str:
    """Convert list of character indices back to SMILES string."""
    chars = []
    for idx in indices:
        if idx == EOS_IDX:
            break
        if idx in (PAD_IDX, SOS_IDX):
            continue
        chars.append(IDX_TO_CHAR.get(idx, ""))
    return "".join(chars)


# ── VAE Model ──────────────────────────────────────────────────────────────

class SMILESEncoder(nn.Module):
    """LSTM-based encoder for SMILES strings."""

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        """Encode SMILES indices to latent distribution parameters.

        Args:
            x: (B, seq_len) long tensor of SMILES indices

        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        embedded = self.embedding(x)  # (B, seq_len, embed_dim)
        _, (h, _) = self.lstm(embedded)  # h: (2, B, hidden_dim)
        h = torch.cat([h[0], h[1]], dim=-1)  # (B, hidden_dim*2)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class SMILESDecoder(nn.Module):
    """LSTM-based decoder for SMILES generation."""

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z: torch.Tensor, target: torch.Tensor = None, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
        """Decode latent vector to SMILES character logits.

        Args:
            z: (B, latent_dim)
            target: (B, seq_len) teacher forcing input (optional)
            max_len: maximum generation length

        Returns:
            logits: (B, seq_len, vocab_size)
        """
        B = z.shape[0]
        h0 = self.latent_to_hidden(z).unsqueeze(0)  # (1, B, hidden_dim)
        c0 = torch.zeros_like(h0)

        if target is not None:
            # Teacher forcing
            embedded = self.embedding(target)
            output, _ = self.lstm(embedded, (h0, c0))
            logits = self.output_proj(output)
        else:
            # Autoregressive generation
            input_idx = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=z.device)
            hidden = (h0, c0)
            logits_list = []

            for _ in range(max_len):
                embedded = self.embedding(input_idx)
                output, hidden = self.lstm(embedded, hidden)
                step_logits = self.output_proj(output)
                logits_list.append(step_logits)

                # Greedy selection for next input
                input_idx = step_logits.argmax(dim=-1)

            logits = torch.cat(logits_list, dim=1)

        return logits


class SMILESVAE(nn.Module):
    """Variational Autoencoder for SMILES generation.

    Can be conditioned on desired properties for targeted generation.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_properties: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = SMILESEncoder(vocab_size, embed_dim, hidden_dim, latent_dim)
        self.decoder = SMILESDecoder(vocab_size, embed_dim, hidden_dim, latent_dim + num_properties)

        # Optional property predictor from latent space
        if num_properties > 0:
            self.property_predictor = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_properties),
            )
        else:
            self.property_predictor = None

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, properties: torch.Tensor = None) -> dict:
        """Forward pass through VAE.

        Args:
            x: (B, seq_len) SMILES indices
            properties: (B, num_properties) optional conditioning

        Returns:
            dict with logits, mu, logvar, and optionally predicted properties
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        # Condition on properties if provided
        if properties is not None:
            z_cond = torch.cat([z, properties], dim=-1)
        else:
            z_cond = z

        # Decode (teacher forcing with input shifted right)
        decoder_input = x[:, :-1]  # remove last token
        logits = self.decoder(z_cond, target=decoder_input)

        result = {"logits": logits, "mu": mu, "logvar": logvar}

        if self.property_predictor is not None:
            result["predicted_properties"] = self.property_predictor(z)

        return result

    def generate(self, n_samples: int = 10, temperature: float = 1.0,
                 properties: torch.Tensor = None, device: torch.device = None) -> list:
        """Generate novel SMILES strings.

        Args:
            n_samples: number of molecules to generate
            temperature: sampling temperature (higher = more diverse)
            properties: (n_samples, num_properties) optional conditioning
            device: torch device

        Returns:
            list of generated SMILES strings
        """
        device = device or next(self.parameters()).device
        self.eval()

        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device) * temperature

            if properties is not None:
                z_cond = torch.cat([z, properties.to(device)], dim=-1)
            else:
                z_cond = z

            logits = self.decoder(z_cond, max_len=MAX_SEQ_LEN)
            indices = logits.argmax(dim=-1).cpu().numpy()

        generated = []
        for seq in indices:
            smiles = indices_to_smiles(seq.tolist())
            generated.append(smiles)

        return generated

    @staticmethod
    def validate_smiles(smiles_list: list) -> dict:
        """Check chemical validity of generated SMILES."""
        if not HAS_RDKIT:
            return {"valid": [], "invalid": [], "validity_rate": 0.0}

        valid = []
        invalid = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid.append(smi)
            else:
                invalid.append(smi)

        return {
            "valid": valid,
            "invalid": invalid,
            "validity_rate": len(valid) / max(len(smiles_list), 1),
            "total": len(smiles_list),
        }


def vae_loss(logits, targets, mu, logvar, kl_weight=0.01):
    """Compute VAE loss: reconstruction + KL divergence.

    Args:
        logits: (B, seq_len, vocab_size) predicted character logits
        targets: (B, seq_len) ground truth indices
        mu: (B, latent_dim) mean of latent distribution
        logvar: (B, latent_dim) log variance of latent distribution
        kl_weight: weight for KL divergence term

    Returns:
        dict with total, recon, and kl losses
    """
    # Reconstruction loss (cross-entropy)
    B, T, V = logits.shape
    recon_loss = F.cross_entropy(
        logits.reshape(-1, V),
        targets[:, :T].reshape(-1),
        ignore_index=PAD_IDX,
    )

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B

    total = recon_loss + kl_weight * kl_loss

    return {"total": total, "recon": recon_loss, "kl": kl_loss}
