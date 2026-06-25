"""Generative Adversarial Network for ionic liquid molecular design.

Uses a SMILES-based generator and discriminator for creating
novel, chemically valid ionic liquid structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.generative.vae import (
    VOCAB_SIZE, MAX_SEQ_LEN, PAD_IDX, SOS_IDX, EOS_IDX,
    smiles_to_indices, indices_to_smiles,
)


class Generator(nn.Module):
    """LSTM-based generator that produces SMILES sequences from noise."""

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        embed_dim: int = 64,
        vocab_size: int = VOCAB_SIZE,
        max_len: int = MAX_SEQ_LEN,
        num_properties: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.vocab_size = vocab_size

        input_dim = latent_dim + num_properties
        self.latent_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.latent_to_cell = nn.Linear(input_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z: torch.Tensor, temperature: float = 1.0) -> tuple:
        """Generate SMILES sequences from latent vectors.

        Args:
            z: (B, latent_dim) or (B, latent_dim + num_properties)
            temperature: sampling temperature

        Returns:
            logits: (B, max_len, vocab_size) - soft outputs for training
            sequences: (B, max_len) - hard token indices
        """
        B = z.shape[0]
        h0 = self.latent_to_hidden(z).unsqueeze(0).repeat(2, 1, 1)  # (2, B, hidden)
        c0 = self.latent_to_cell(z).unsqueeze(0).repeat(2, 1, 1)

        input_idx = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=z.device)
        hidden = (h0.contiguous(), c0.contiguous())

        logits_list = []
        indices_list = []

        for t in range(self.max_len):
            embedded = self.embedding(input_idx)  # (B, 1, embed)
            output, hidden = self.lstm(embedded, hidden)
            step_logits = self.output_proj(output.squeeze(1))  # (B, vocab)

            # Gumbel-softmax for differentiable sampling during training
            if self.training:
                soft = F.gumbel_softmax(step_logits / temperature, tau=1.0, hard=False)
                logits_list.append(soft.unsqueeze(1))
                input_idx = step_logits.argmax(dim=-1, keepdim=True)
            else:
                logits_list.append(step_logits.unsqueeze(1))
                if temperature > 0:
                    probs = F.softmax(step_logits / temperature, dim=-1)
                    input_idx = torch.multinomial(probs, 1)
                else:
                    input_idx = step_logits.argmax(dim=-1, keepdim=True)

            indices_list.append(input_idx)

        logits = torch.cat(logits_list, dim=1)  # (B, max_len, vocab)
        sequences = torch.cat(indices_list, dim=1).squeeze(-1)  # (B, max_len)

        return logits, sequences


class Discriminator(nn.Module):
    """CNN-based discriminator that classifies SMILES as real or generated."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        max_len: int = MAX_SEQ_LEN,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        # 1D CNN for sequence classification
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=k, padding=k // 2)
            for k in [3, 5, 7]
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify sequence as real or fake.

        Args:
            x: (B, seq_len) long tensor of SMILES indices
                OR (B, seq_len, vocab_size) soft logits from generator

        Returns:
            (B, 1) real/fake logits
        """
        if x.dim() == 2:
            # Hard indices
            embedded = self.embedding(x)  # (B, seq_len, embed)
        else:
            # Soft logits from generator (B, seq_len, vocab)
            embedded = x @ self.embedding.weight  # (B, seq_len, embed)

        embedded = embedded.transpose(1, 2)  # (B, embed, seq_len)

        conv_outputs = []
        for conv in self.convs:
            out = F.leaky_relu(conv(embedded), 0.2)
            out = F.adaptive_max_pool1d(out, 1).squeeze(-1)  # (B, hidden)
            conv_outputs.append(out)

        features = torch.cat(conv_outputs, dim=-1)
        return self.classifier(features)


class MolecularGAN(nn.Module):
    """GAN for molecular SMILES generation.

    Includes optional property conditioning for targeted generation.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        embed_dim: int = 64,
        vocab_size: int = VOCAB_SIZE,
        max_len: int = MAX_SEQ_LEN,
        num_properties: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_properties = num_properties

        self.generator = Generator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            max_len=max_len,
            num_properties=num_properties,
        )
        self.discriminator = Discriminator(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim // 2,
            max_len=max_len,
        )

    def generate(self, n_samples: int, temperature: float = 1.0,
                 properties: torch.Tensor = None, device: torch.device = None) -> list:
        """Generate novel SMILES strings."""
        device = device or next(self.parameters()).device
        self.eval()

        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device)
            if properties is not None:
                z = torch.cat([z, properties.to(device)], dim=-1)

            _, sequences = self.generator(z, temperature=temperature)
            indices = sequences.cpu().numpy()

        return [indices_to_smiles(seq.tolist()) for seq in indices]


def train_gan_step(
    gan: MolecularGAN,
    real_smiles_indices: torch.Tensor,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """One training step for the GAN.

    Returns dict with generator and discriminator losses.
    """
    B = real_smiles_indices.shape[0]
    real = real_smiles_indices.to(device)

    # ── Train Discriminator ──
    d_optimizer.zero_grad()

    # Real samples
    d_real = gan.discriminator(real)
    d_loss_real = F.binary_cross_entropy_with_logits(
        d_real, torch.ones(B, 1, device=device)
    )

    # Fake samples
    z = torch.randn(B, gan.latent_dim, device=device)
    fake_logits, fake_seq = gan.generator(z)
    d_fake = gan.discriminator(fake_logits.detach())
    d_loss_fake = F.binary_cross_entropy_with_logits(
        d_fake, torch.zeros(B, 1, device=device)
    )

    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    d_optimizer.step()

    # ── Train Generator ──
    g_optimizer.zero_grad()

    z = torch.randn(B, gan.latent_dim, device=device)
    fake_logits, _ = gan.generator(z)
    d_fake = gan.discriminator(fake_logits)
    g_loss = F.binary_cross_entropy_with_logits(
        d_fake, torch.ones(B, 1, device=device)
    )

    g_loss.backward()
    g_optimizer.step()

    return {
        "g_loss": g_loss.item(),
        "d_loss": d_loss.item(),
        "d_real": d_real.mean().item(),
        "d_fake": d_fake.mean().item(),
    }
