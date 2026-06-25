"""Gated Bilinear HyperNetwork Fusion v2.

Improvements over v1:
1. Low-rank bilinear: h_s^T (U V^T) h_g with rank-R factorization
   instead of element-wise h_s ⊙ h_g. Captures cross-dimension interactions.
2. Residual paths: direct skip connections from each modality to output,
   preventing information loss through the bilinear bottleneck.
3. Deeper HyperNetwork: 5→64→64→64→outputs with residual connection.

Fusion equation:
    # Low-rank bilinear interaction
    h_bilinear = (U h_surface) ⊙ (V h_graph)     # rank-R bilinear, R<<D

    # HyperNetwork generates T-dependent parameters
    W, gate, bias = HyperNet(T, x₁)

    # T-dependent gated fusion with residual paths
    h_fused = gate ⊙ (W ⊙ h_bilinear) + (1-gate) ⊙ h_thermo + bias
    z = LayerNorm(h_fused + α·h_surface_proj + β·h_graph_proj)  # residual

Total fusion params: ~40K (vs 863K cross-attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperNetworkV2(nn.Module):
    """Deeper HyperNetwork with residual connection.

    3-layer MLP (5→64→64→64→outputs) with skip connection.
    Generates per-sample fusion weights conditioned on thermodynamic state.
    """

    def __init__(self, thermo_dim: int = 5, fused_dim: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.fused_dim = fused_dim

        # First block
        self.fc1 = nn.Linear(thermo_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Residual skip
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Output heads
        self.head_W = nn.Linear(hidden_dim, fused_dim)
        self.head_gate = nn.Linear(hidden_dim, fused_dim)
        self.head_bias = nn.Linear(hidden_dim, fused_dim)

        # Initialize near-identity
        nn.init.zeros_(self.head_W.weight)
        nn.init.ones_(self.head_W.bias)
        nn.init.zeros_(self.head_gate.weight)
        nn.init.zeros_(self.head_gate.bias)  # sigmoid(0) = 0.5
        nn.init.zeros_(self.head_bias.weight)
        nn.init.zeros_(self.head_bias.bias)

    def forward(self, thermo_features):
        h = F.gelu(self.fc1(thermo_features))
        h_res = F.gelu(self.fc2(h))
        h = F.gelu(self.fc3(h_res) + h)  # residual

        W = torch.sigmoid(self.head_W(h))
        gate = torch.sigmoid(self.head_gate(h))
        bias = self.head_bias(h)
        return W, gate, bias


class LowRankBilinear(nn.Module):
    """Low-rank bilinear interaction: h_a^T (U V^T) h_b.

    Factorizes the full bilinear matrix (D×D) into U (D×R) and V (D×R),
    reducing parameters from D² to 2DR.

    Computed as: (U h_a) ⊙ (V h_b), which equals h_a^T (U^T diag V^T) h_b
    generalized to rank-R.

    Parameters
    ----------
    dim_a : int
        First input dimension.
    dim_b : int
        Second input dimension.
    rank : int
        Rank of the factorization. Lower = fewer params, less expressive.
    out_dim : int
        Output dimension.
    """

    def __init__(self, dim_a: int, dim_b: int, rank: int = 32, out_dim: int = 256):
        super().__init__()
        self.U = nn.Linear(dim_a, rank, bias=False)
        self.V = nn.Linear(dim_b, rank, bias=False)
        self.proj = nn.Linear(rank, out_dim)

    def forward(self, h_a, h_b):
        """
        Args:
            h_a: (B, dim_a)
            h_b: (B, dim_b)
        Returns:
            (B, out_dim) — bilinear interaction features
        """
        return self.proj(self.U(h_a) * self.V(h_b))


class GatedBilinearHyperFusionV2(nn.Module):
    """Gated Bilinear HyperNetwork Fusion v2.

    Key improvements:
    1. Low-rank bilinear (rank R) replaces element-wise product
    2. Residual skip connections from each modality
    3. Deeper HyperNetwork with residual blocks

    Parameters
    ----------
    pointcloud_dim : int
        PointNet output dimension.
    graph_dim : int
        GNN output dimension.
    tabular_dim : int
        Full tabular feature dimension.
    fused_dim : int
        Output dimension.
    rank : int
        Bilinear factorization rank.
    thermo_dim : int
        Number of thermo features for HyperNet (first N features of tabular).
    hyper_hidden : int
        HyperNetwork hidden dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        pointcloud_dim: int = 256,
        graph_dim: int = 256,
        tabular_dim: int = 25,
        fused_dim: int = 256,
        rank: int = 32,
        thermo_dim: int = 5,
        hyper_hidden: int = 64,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.fused_dim = fused_dim
        self.thermo_dim = thermo_dim

        # Modality projections
        self.pc_proj = nn.Linear(pointcloud_dim, fused_dim)
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.tabular_proj = nn.Sequential(
            nn.Linear(tabular_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Low-rank bilinear interaction
        self.bilinear = LowRankBilinear(
            dim_a=fused_dim, dim_b=fused_dim,
            rank=rank, out_dim=fused_dim)

        # HyperNetwork v2 (deeper, with residual)
        self.hypernet = HyperNetworkV2(
            thermo_dim=thermo_dim,
            fused_dim=fused_dim,
            hidden_dim=hyper_hidden)

        # Residual path: learnable weights for skip connections
        self.residual_alpha = nn.Parameter(torch.tensor(0.1))  # surface residual
        self.residual_beta = nn.Parameter(torch.tensor(0.1))   # graph residual

        # Output
        self.layer_norm = nn.LayerNorm(fused_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, pc_feat, graph_feat, tabular_feat):
        """
        Args:
            pc_feat: (B, pointcloud_dim)
            graph_feat: (B, graph_dim)
            tabular_feat: (B, tabular_dim)
        Returns:
            (B, fused_dim)
        """
        # Project
        h_surface = self.pc_proj(pc_feat)
        h_graph = self.graph_proj(graph_feat)
        h_thermo = self.tabular_proj(tabular_feat)

        # Low-rank bilinear: captures cross-dimension surface×graph interactions
        h_bilinear = self.bilinear(h_surface, h_graph)

        # HyperNetwork: T-dependent fusion parameters
        thermo_input = tabular_feat[:, :self.thermo_dim]
        W, gate, bias = self.hypernet(thermo_input)

        # Gated fusion with T-dependent weights
        h_weighted = W * h_bilinear
        h_gated = gate * h_weighted + (1 - gate) * h_thermo + bias

        # Residual skip connections from individual modalities
        h_fused = h_gated + self.residual_alpha * h_surface + self.residual_beta * h_graph

        # Normalize and project
        h_fused = self.layer_norm(h_fused)
        h_fused = self.output_proj(h_fused)

        return h_fused
