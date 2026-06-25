"""Feature-wise Linear Modulation (FiLM) for multimodal fusion.

Replaces cross-attention with lightweight multiplicative conditioning:
  h_out = γ(condition) * h_input + β(condition)

FiLM is parameter-efficient (~33K vs 863K for cross-attention) while
still modeling inter-modality interactions through learned scale/shift.

Physically motivated: thermodynamic state (T, x₁) should SCALE molecular
property predictions — FiLM's multiplicative modulation captures this naturally.

Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018.
"""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Generate scale (γ) and shift (β) from a conditioning vector.

    output = γ * input + β

    Parameters
    ----------
    cond_dim : int
        Dimension of the conditioning input.
    feat_dim : int
        Dimension of the features to modulate.
    """

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(cond_dim, feat_dim)
        self.beta_net = nn.Linear(cond_dim, feat_dim)

        # Initialize gamma close to 1, beta close to 0 (identity at start)
        nn.init.ones_(self.gamma_net.bias)
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.zeros_(self.beta_net.bias)
        nn.init.zeros_(self.beta_net.weight)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, feat_dim) — features to modulate
            condition: (B, cond_dim) — conditioning signal

        Returns:
            (B, feat_dim) — modulated features
        """
        gamma = self.gamma_net(condition)  # (B, feat_dim)
        beta = self.beta_net(condition)    # (B, feat_dim)
        return gamma * features + beta


class FiLMFusion(nn.Module):
    """Multimodal fusion using FiLM conditioning.

    Three modalities are fused through mutual FiLM conditioning:
    1. Graph features modulate surface features (molecular structure → surface)
    2. Surface features modulate graph features (surface electrostatics → graph)
    3. Thermodynamic features modulate both (T, x₁ → scale/shift)

    Then all are concatenated and projected through a fusion MLP.

    Parameters
    ----------
    pointcloud_dim : int
        PointNet output dimension.
    graph_dim : int
        GNN/D-MPNN output dimension.
    tabular_dim : int
        Thermodynamic feature dimension.
    fused_dim : int
        Output fused representation dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        pointcloud_dim: int = 256,
        graph_dim: int = 256,
        tabular_dim: int = 25,
        fused_dim: int = 256,
        dropout: float = 0.3,
        **kwargs,  # Accept and ignore extra args like num_heads
    ):
        super().__init__()

        # Project to common dimension
        self.pc_proj = nn.Linear(pointcloud_dim, fused_dim)
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.tabular_proj = nn.Sequential(
            nn.Linear(tabular_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # FiLM: graph conditions surface
        self.film_graph_to_pc = FiLMLayer(fused_dim, fused_dim)
        # FiLM: surface conditions graph
        self.film_pc_to_graph = FiLMLayer(fused_dim, fused_dim)
        # FiLM: thermo conditions surface
        self.film_thermo_to_pc = FiLMLayer(fused_dim, fused_dim)
        # FiLM: thermo conditions graph
        self.film_thermo_to_graph = FiLMLayer(fused_dim, fused_dim)

        # Layer norms
        self.ln_pc = nn.LayerNorm(fused_dim)
        self.ln_graph = nn.LayerNorm(fused_dim)
        self.ln_tabular = nn.LayerNorm(fused_dim)

        # Learnable modality weights
        self.modality_weights = nn.Parameter(torch.ones(3) / 3)

        # Final fusion MLP (same interface as PointCloudFusion)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim * 3, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, pc_feat, graph_feat, tabular_feat):
        """
        Args:
            pc_feat: (B, pointcloud_dim) — from PointNet
            graph_feat: (B, graph_dim) — from GNN/D-MPNN
            tabular_feat: (B, tabular_dim) — thermodynamic features

        Returns:
            (B, fused_dim) — fused representation
        """
        # Project to common space
        pc = self.pc_proj(pc_feat)
        g = self.graph_proj(graph_feat)
        t = self.tabular_proj(tabular_feat)

        # FiLM conditioning (mutual + thermo)
        pc_modulated = self.film_graph_to_pc(pc, g)        # graph informs surface
        pc_modulated = self.film_thermo_to_pc(pc_modulated, t)  # thermo scales surface
        g_modulated = self.film_pc_to_graph(g, pc)          # surface informs graph
        g_modulated = self.film_thermo_to_graph(g_modulated, t)  # thermo scales graph

        # Residual + LayerNorm
        pc = self.ln_pc(pc + self.dropout(pc_modulated - pc))  # residual from modulation
        g = self.ln_graph(g + self.dropout(g_modulated - g))
        t = self.ln_tabular(t)

        # Weighted combination
        weights = torch.softmax(self.modality_weights, dim=0)
        concat = torch.cat([pc * weights[0], g * weights[1], t * weights[2]], dim=-1)

        return self.fusion_mlp(concat)
