"""Dual-Path Fusion: Cross-Attention + Low-Rank Bilinear in parallel.

Combines the strengths of two fusion mechanisms:
- Cross-attention (863K params): best for gamma1 (0.887) — captures complex
  bidirectional surface↔graph interactions
- Low-rank bilinear (25K params): best for gamma2 (0.916) — captures
  cross-dimensional surface×graph product for site-specific interactions

A learned per-property gate routes each property to its optimal path:
  h_p = α_p · h_crossattn + (1 - α_p) · h_bilinear + h_thermo

Expected: gamma1 α≈1 (cross-attention), gamma2 α≈0 (bilinear),
other properties learn their optimal mix.

Total added params: ~25K (bilinear) + 7 (gates) on top of cross-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.fusion.cross_attention import CrossAttention


class LowRankBilinear(nn.Module):
    """Low-rank bilinear: (U·h_a) ⊙ (V·h_b) → proj → out_dim."""

    def __init__(self, dim_a, dim_b, rank=32, out_dim=256):
        super().__init__()
        self.U = nn.Linear(dim_a, rank, bias=False)
        self.V = nn.Linear(dim_b, rank, bias=False)
        self.proj = nn.Linear(rank, out_dim)

    def forward(self, h_a, h_b):
        return self.proj(self.U(h_a) * self.V(h_b))


class DualPathFusion(nn.Module):
    """Dual-path fusion with per-property adaptive routing.

    Path A: Cross-attention (surface ↔ graph, bidirectional)
    Path B: Low-rank bilinear (surface × graph, cross-dimensional)
    Gate: Learned per-property weight α_p ∈ [0,1]

    Parameters
    ----------
    pointcloud_dim : int
        PointNet output dimension.
    graph_dim : int
        GNN output dimension.
    tabular_dim : int
        Tabular feature dimension.
    fused_dim : int
        Output fused dimension.
    bilinear_rank : int
        Rank of bilinear factorization.
    num_heads : int
        Cross-attention heads.
    n_properties : int
        Number of target properties (for per-property gates).
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        pointcloud_dim: int = 256,
        graph_dim: int = 256,
        tabular_dim: int = 25,
        fused_dim: int = 256,
        bilinear_rank: int = 32,
        num_heads: int = 8,
        n_properties: int = 7,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.fused_dim = fused_dim

        # Projections to common space
        self.pc_proj = nn.Linear(pointcloud_dim, fused_dim)
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.tabular_proj = nn.Sequential(
            nn.Linear(tabular_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # PATH A: Cross-attention (bidirectional)
        self.pc_graph_attn = CrossAttention(fused_dim, fused_dim, num_heads, dropout)
        self.graph_pc_attn = CrossAttention(fused_dim, fused_dim, num_heads, dropout)
        self.ln_pc = nn.LayerNorm(fused_dim)
        self.ln_graph = nn.LayerNorm(fused_dim)
        self.crossattn_mlp = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # PATH B: Low-rank bilinear
        self.bilinear = LowRankBilinear(
            fused_dim, fused_dim, rank=bilinear_rank, out_dim=fused_dim)
        self.bilinear_norm = nn.LayerNorm(fused_dim)

        # Per-property adaptive gate: α_p for each of 7 properties
        # Initialize: gamma1 → cross-attn (α=1), gamma2 → bilinear (α=0)
        self.gate_logits = nn.Parameter(torch.zeros(n_properties))
        with torch.no_grad():
            self.gate_logits[0] = 1.0   # gamma1 → cross-attention
            self.gate_logits[1] = -1.0  # gamma2 → bilinear

        # Tabular integration
        self.ln_tabular = nn.LayerNorm(fused_dim)

        # Learnable path-tabular mixing
        self.output_mlp = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, pc_feat, graph_feat, tabular_feat):
        """
        Returns:
            fused: (B, fused_dim) — for standard single-head prediction
            fused_per_prop: (B, fused_dim, 7) — for per-property prediction
            gate_values: (7,) — learned routing weights
        """
        # Project
        pc = self.pc_proj(pc_feat)
        g = self.graph_proj(graph_feat)
        t = self.tabular_proj(tabular_feat)
        t = self.ln_tabular(t)

        # PATH A: Cross-attention
        pc_attended = self.pc_graph_attn(pc, g)
        g_attended = self.graph_pc_attn(g, pc)
        pc_ca = self.ln_pc(pc + self.dropout(pc_attended))
        g_ca = self.ln_graph(g + self.dropout(g_attended))
        h_crossattn = self.crossattn_mlp(torch.cat([pc_ca, g_ca], dim=-1))

        # PATH B: Low-rank bilinear
        h_bilinear = self.bilinear(pc, g)
        h_bilinear = self.bilinear_norm(h_bilinear)

        # Per-property routing
        alpha = torch.sigmoid(self.gate_logits)  # (7,)

        # For per-property prediction: (B, D, 7)
        h_ca_exp = h_crossattn.unsqueeze(2)   # (B, D, 1)
        h_bi_exp = h_bilinear.unsqueeze(2)    # (B, D, 1)
        alpha_exp = alpha.view(1, 1, -1)       # (1, 1, 7)
        h_mixed = alpha_exp * h_ca_exp + (1 - alpha_exp) * h_bi_exp  # (B, D, 7)

        # Default fused output (average across properties for compatibility)
        h_fused_avg = (h_crossattn + h_bilinear) / 2
        h_out = self.output_mlp(torch.cat([h_fused_avg, t], dim=-1))

        return h_out, h_mixed, alpha.detach()
