"""DGCNN (Dynamic Graph CNN) encoder for COSMO surface point clouds.

Uses EdgeConv layers that capture local surface topology by constructing
k-nearest neighbor graphs dynamically in feature space.

Reference: Wang et al., "Dynamic Graph CNN for Learning on Point Clouds", 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    """Compute k nearest neighbors for each point.

    Args:
        x: (B, C, N) point features
        k: number of neighbors

    Returns:
        (B, N, k) indices of k nearest neighbors
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    """Construct edge features for EdgeConv.

    For each point, concatenates [xi, xj - xi] for each neighbor j.

    Args:
        x: (B, C, N)
        k: number of neighbors
        idx: optional pre-computed neighbor indices

    Returns:
        (B, 2*C, N, k) edge features
    """
    B, C, N = x.shape
    if idx is None:
        idx = knn(x, k=k)
    device = x.device

    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(B * N, -1)[idx, :].view(B, N, k, C)
    x = x.view(B, N, 1, C).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature  # (B, 2*C, N, k)


class EdgeConv(nn.Module):
    """EdgeConv layer: shared MLP on edge features + max aggregation."""

    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, N) point features
        Returns:
            (B, out_channels, N) updated features
        """
        edge_feat = get_graph_feature(x, k=self.k)  # (B, 2*C, N, k)
        x = self.conv(edge_feat)  # (B, out, N, k)
        x = x.max(dim=-1)[0]  # (B, out, N)
        return x


class DGCNNEncoder(nn.Module):
    """DGCNN encoder for molecular surface point clouds.

    Parameters
    ----------
    in_channels : int
        Input feature dimension per point (default 7).
    feature_dim : int
        Output global feature dimension.
    k : int
        Number of nearest neighbors for EdgeConv.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 7,
        feature_dim: int = 256,
        k: int = 20,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.k = k

        self.edge_conv1 = EdgeConv(in_channels, 64, k)
        self.edge_conv2 = EdgeConv(64, 64, k)
        self.edge_conv3 = EdgeConv(64, 128, k)
        self.edge_conv4 = EdgeConv(128, 256, k)

        # Aggregate all layer outputs
        self.conv_agg = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
        )

        self.global_proj = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )

        self._feature_dim = feature_dim

    def forward(self, x):
        """
        Args:
            x: (B, N, in_channels) point cloud
        Returns:
            (B, feature_dim) global feature
        """
        x = x.transpose(1, 2)  # (B, C, N)

        x1 = self.edge_conv1(x)   # (B, 64, N)
        x2 = self.edge_conv2(x1)  # (B, 64, N)
        x3 = self.edge_conv3(x2)  # (B, 128, N)
        x4 = self.edge_conv4(x3)  # (B, 256, N)

        # Concatenate all layer outputs
        x = torch.cat([x1, x2, x3, x4], dim=1)  # (B, 512, N)
        x = self.conv_agg(x)  # (B, 512, N)

        # Global max + mean pooling
        x_max = x.max(dim=2)[0]  # (B, 512)

        x = self.global_proj(x_max)
        return x

    @property
    def feature_dim(self):
        return self._feature_dim
