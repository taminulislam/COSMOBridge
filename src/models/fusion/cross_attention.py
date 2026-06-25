"""Cross-attention mechanism for multimodal feature fusion."""

import torch
import torch.nn as nn
import math


class CrossAttention(nn.Module):
    """Cross-attention between two modalities.

    Query from modality A attends to key/value from modality B,
    producing attention-weighted features.
    """

    def __init__(self, dim_q: int, dim_kv: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_q // num_heads
        assert dim_q % num_heads == 0, "dim_q must be divisible by num_heads"

        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_kv, dim_q)
        self.v_proj = nn.Linear(dim_kv, dim_q)
        self.out_proj = nn.Linear(dim_q, dim_q)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, dim_q) or (B, seq_q, dim_q)
            key_value: (B, dim_kv) or (B, seq_kv, dim_kv)

        Returns:
            (B, dim_q) or (B, seq_q, dim_q)
        """
        # Handle 2D inputs by adding sequence dimension
        squeeze = False
        if query.dim() == 2:
            query = query.unsqueeze(1)  # (B, 1, dim_q)
            squeeze = True
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)  # (B, 1, dim_kv)

        B, seq_q, _ = query.shape
        _, seq_kv, _ = key_value.shape

        # Project
        Q = self.q_proj(query).view(B, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value).view(B, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value).view(B, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, heads, seq_q, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, seq_q, -1)
        out = self.out_proj(out)

        if squeeze:
            out = out.squeeze(1)

        return out


class MultimodalFusion(nn.Module):
    """Fuse features from multiple modalities using cross-attention.

    Supports: vision (COSMO + EP), graph, and tabular features.
    """

    def __init__(
        self,
        vision_dim: int = 512,
        graph_dim: int = 256,
        tabular_dim: int = 64,
        fused_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Project all modalities to the same dimension
        self.vision_proj = nn.Linear(vision_dim, fused_dim)
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.tabular_proj = nn.Linear(tabular_dim, fused_dim)

        # Cross-attention: vision <-> graph
        self.vision_graph_attn = CrossAttention(fused_dim, fused_dim, num_heads, dropout)
        self.graph_vision_attn = CrossAttention(fused_dim, fused_dim, num_heads, dropout)

        # Layer norms
        self.ln_vision = nn.LayerNorm(fused_dim)
        self.ln_graph = nn.LayerNorm(fused_dim)
        self.ln_tabular = nn.LayerNorm(fused_dim)

        # Learnable modality weights
        self.modality_weights = nn.Parameter(torch.ones(3) / 3)

        # Final fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim * 3, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        vision_features: torch.Tensor,
        graph_features: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse multimodal features.

        Args:
            vision_features: (B, vision_dim) - from dual image encoder
            graph_features: (B, graph_dim) - from GNN
            tabular_features: (B, tabular_dim) - from tabular DNN

        Returns:
            (B, fused_dim) - fused feature vector
        """
        # Project to common space
        v = self.vision_proj(vision_features)
        g = self.graph_proj(graph_features)
        t = self.tabular_proj(tabular_features)

        # Cross-attention between vision and graph
        v_attended = self.vision_graph_attn(v, g)
        g_attended = self.graph_vision_attn(g, v)

        # Residual + LayerNorm
        v = self.ln_vision(v + self.dropout(v_attended))
        g = self.ln_graph(g + self.dropout(g_attended))
        t = self.ln_tabular(t)

        # Weighted combination
        weights = torch.softmax(self.modality_weights, dim=0)
        v_weighted = v * weights[0]
        g_weighted = g * weights[1]
        t_weighted = t * weights[2]

        # Concatenate and fuse
        concat = torch.cat([v_weighted, g_weighted, t_weighted], dim=-1)
        fused = self.fusion_mlp(concat)

        return fused
