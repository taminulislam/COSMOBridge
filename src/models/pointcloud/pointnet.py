"""PointNet encoder for COSMO surface point clouds.

Encodes molecular surface point clouds (vertices + normals + ESP)
into fixed-dimensional feature vectors for property prediction.

Input:  (B, N, 7) — N points with [x, y, z, nx, ny, nz, esp]
Output: (B, feature_dim) — global surface representation

Architecture follows PointNet (Qi et al., 2017) with:
  - Shared MLPs (per-point feature extraction)
  - Max pooling for permutation invariance
  - Optional T-Net for input alignment (disabled by default for small molecules)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMLP(nn.Module):
    """1D convolution acting as a shared MLP across points."""

    def __init__(self, in_channels, out_channels, bn=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels) if bn else nn.Identity()

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class PointNetEncoder(nn.Module):
    """PointNet encoder for molecular surface point clouds.

    Parameters
    ----------
    in_channels : int
        Input feature dimension per point (default 7: xyz + normals + esp).
    feature_dim : int
        Output feature dimension (global representation).
    hidden_dims : list
        Hidden dimensions for shared MLPs.
    dropout : float
        Dropout probability on the global feature.
    """

    def __init__(
        self,
        in_channels: int = 7,
        feature_dim: int = 256,
        hidden_dims: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        # Per-point feature extraction (shared MLPs)
        layers = []
        in_dim = in_channels
        for h_dim in hidden_dims:
            layers.append(SharedMLP(in_dim, h_dim))
            in_dim = h_dim
        self.point_features = nn.Sequential(*layers)

        # Global feature projection
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self._feature_dim = feature_dim

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor of shape (B, N, in_channels)
            Batch of point clouds.

        Returns
        -------
        Tensor of shape (B, feature_dim)
            Global surface representation.
        """
        # (B, N, C) -> (B, C, N) for Conv1d
        x = x.transpose(1, 2)

        # Per-point features
        x = self.point_features(x)  # (B, hidden[-1], N)

        # Symmetric aggregation (max pool over points)
        x = x.max(dim=2)[0]  # (B, hidden[-1])

        # Project to feature_dim
        x = self.global_proj(x)  # (B, feature_dim)

        return x

    @property
    def feature_dim(self):
        return self._feature_dim


class PointNetClassifier(nn.Module):
    """PointNet with prediction head for standalone evaluation.

    Parameters
    ----------
    in_channels : int
        Input channels per point.
    feature_dim : int
        Encoder output dimension.
    num_targets : int
        Number of prediction targets.
    aux_feature_dim : int
        Dimension of auxiliary features (temperature etc.) concatenated
        with the global point cloud feature before prediction.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 7,
        feature_dim: int = 256,
        num_targets: int = 7,
        aux_feature_dim: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = PointNetEncoder(
            in_channels=in_channels,
            feature_dim=feature_dim,
            dropout=dropout,
        )

        self.aux_feature_dim = aux_feature_dim
        pred_input = feature_dim + aux_feature_dim

        self.prediction_head = nn.Sequential(
            nn.Linear(pred_input, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_targets),
        )

    def forward(self, point_cloud, features=None, **kwargs):
        """
        Parameters
        ----------
        point_cloud : Tensor (B, N, 7)
        features : Tensor (B, aux_feature_dim), optional

        Returns
        -------
        Tensor (B, num_targets)
        """
        x = self.encoder(point_cloud)
        if features is not None and self.aux_feature_dim > 0:
            x = torch.cat([x, features], dim=-1)
        return self.prediction_head(x)

    def get_features(self, point_cloud):
        """Return encoder features before prediction head."""
        return self.encoder(point_cloud)
