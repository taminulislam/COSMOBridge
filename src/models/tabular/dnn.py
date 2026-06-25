"""Deep Neural Network for predicting ionic liquid thermodynamic properties from tabular data.

This module implements a configurable DNN that combines learnable categorical
embeddings (for ionic liquid identity, cation type, and anion type) with
continuous numerical features (temperature, mole fraction, and engineered
thermodynamic features) to predict multiple thermodynamic target properties
simultaneously. Uses residual/skip connections for better gradient flow
with small datasets.
"""

import torch
import torch.nn as nn
from typing import List


class ResidualBlock(nn.Module):
    """A single residual block: Linear -> BN -> ReLU -> Dropout + skip connection.

    If input and output dimensions differ, a linear projection is used for the skip.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class TabularDNN(nn.Module):
    """Tabular Deep Neural Network with categorical embeddings and residual connections
    for IL property prediction.

    The model learns separate embedding vectors for each ionic liquid, cation,
    and anion. These are concatenated with numerical features (temperature, x1,
    and engineered thermodynamic features like 1/T, T², T³) and passed through
    residual blocks with BatchNorm, ReLU, and Dropout.

    Args:
        num_ils: Number of unique ionic liquids.
        num_cations: Number of unique cation types.
        num_anions: Number of unique anion types.
        feature_dim: Dimensionality of continuous input features.
        il_embed_dim: Embedding dimension for ionic liquid IDs.
        cation_embed_dim: Embedding dimension for cation types.
        anion_embed_dim: Embedding dimension for anion types.
        hidden_dims: List of hidden layer sizes for the MLP.
        dropout: Dropout probability applied after each hidden layer.
        num_targets: Number of output targets to predict.
    """

    def __init__(
        self,
        num_ils: int = 28,
        num_cations: int = 9,
        num_anions: int = 7,
        feature_dim: int = 5,
        il_embed_dim: int = 64,
        cation_embed_dim: int = 32,
        anion_embed_dim: int = 32,
        hidden_dims: List[int] = None,
        dropout: float = 0.4,
        num_targets: int = 7,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Categorical embeddings (larger for better IL discrimination)
        self.il_embedding = nn.Embedding(num_ils, il_embed_dim)
        self.cation_embedding = nn.Embedding(num_cations, cation_embed_dim)
        self.anion_embedding = nn.Embedding(num_anions, anion_embed_dim)

        # Input dimension after concatenation
        concat_dim = feature_dim + il_embed_dim + cation_embed_dim + anion_embed_dim

        # Build residual backbone
        blocks = []
        in_dim = concat_dim
        for h_dim in hidden_dims:
            blocks.append(ResidualBlock(in_dim, h_dim, dropout=dropout))
            in_dim = h_dim

        self.backbone = nn.Sequential(*blocks)

        # Final prediction head
        self.head = nn.Linear(hidden_dims[-1], num_targets)

        self._repr_dim = hidden_dims[-1]

    def _concat_inputs(
        self,
        features: torch.Tensor,
        il_idx: torch.Tensor,
        cation_idx: torch.Tensor,
        anion_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Embed categorical inputs and concatenate with numerical features.

        Args:
            features: Continuous features of shape (B, feature_dim).
            il_idx: Ionic liquid indices of shape (B,).
            cation_idx: Cation type indices of shape (B,).
            anion_idx: Anion type indices of shape (B,).

        Returns:
            Concatenated tensor of shape (B, concat_dim).
        """
        il_emb = self.il_embedding(il_idx)
        cat_emb = self.cation_embedding(cation_idx)
        an_emb = self.anion_embedding(anion_idx)
        return torch.cat([features, il_emb, cat_emb, an_emb], dim=-1)

    def get_features(
        self,
        features: torch.Tensor,
        il_idx: torch.Tensor,
        cation_idx: torch.Tensor,
        anion_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return the learned representation before the final prediction head.

        This is useful for downstream fusion models that combine representations
        from multiple modalities.

        Args:
            features: Continuous features of shape (B, feature_dim).
            il_idx: Ionic liquid indices of shape (B,).
            cation_idx: Cation type indices of shape (B,).
            anion_idx: Anion type indices of shape (B,).

        Returns:
            Representation tensor of shape (B, hidden_dims[-1]).
        """
        x = self._concat_inputs(features, il_idx, cation_idx, anion_idx)
        return self.backbone(x)

    def forward(
        self,
        features: torch.Tensor,
        il_idx: torch.Tensor,
        cation_idx: torch.Tensor,
        anion_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the full network.

        Args:
            features: Continuous features of shape (B, feature_dim).
            il_idx: Ionic liquid indices of shape (B,).
            cation_idx: Cation type indices of shape (B,).
            anion_idx: Anion type indices of shape (B,).

        Returns:
            Predictions of shape (B, num_targets).
        """
        h = self.get_features(features, il_idx, cation_idx, anion_idx)
        return self.head(h)

    @property
    def repr_dim(self) -> int:
        """Dimensionality of the representation returned by ``get_features``."""
        return self._repr_dim
