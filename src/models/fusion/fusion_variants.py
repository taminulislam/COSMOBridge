"""Fusion module variants for MoE ablation study.

All modules have the same interface:
  Input:  pc_feat (B, D_pc), graph_feat (B, D_graph), tabular_feat (B, D_tab)
  Output: fused (B, fused_dim)

Variant A: CrossAttentionFusion (baseline, in multimodal_pointcloud.py)
Variant B: PhysicsInformedBottleneckFusion
Variant C: HierarchicalMultiScaleFusion
Variant D: GatedResidualFusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ══════════════════════════════════════════════════════════════════════════════
# Variant B: Physics-Informed Bottleneck Fusion
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsBottleneckFusion(nn.Module):
    """Physics-informed bottleneck fusion.

    Learnable bottleneck tokens represent physical interaction concepts
    (solvation, electrostatics, steric, H-bonding, dispersion).
    Each modality attends to bottlenecks, bottlenecks aggregate
    cross-modal information, then attend back to produce fused features.

    This is inspired by Perceiver IO but with domain-specific bottleneck
    semantics for molecular property prediction.
    """

    def __init__(self, pointcloud_dim=256, graph_dim=256, tabular_dim=25,
                 fused_dim=256, n_bottlenecks=6, num_heads=4, dropout=0.3):
        super().__init__()
        self.fused_dim = fused_dim
        self.n_bottlenecks = n_bottlenecks

        # Project all modalities to common dimension
        self.pc_proj = nn.Linear(pointcloud_dim, fused_dim)
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.tab_proj = nn.Sequential(
            nn.Linear(tabular_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Learnable bottleneck tokens (physical interaction concepts)
        # These learn to represent: solvation, electrostatics, steric,
        # H-bonding, dispersion, and a general interaction token
        self.bottleneck_tokens = nn.Parameter(
            torch.randn(n_bottlenecks, fused_dim) * 0.02)

        # Cross-attention: modalities → bottlenecks (encode)
        self.encode_attn = nn.MultiheadAttention(
            fused_dim, num_heads, dropout=dropout, batch_first=True)
        self.encode_norm = nn.LayerNorm(fused_dim)

        # Self-attention among bottlenecks (integrate)
        self.bottleneck_self_attn = nn.MultiheadAttention(
            fused_dim, num_heads, dropout=dropout, batch_first=True)
        self.bottleneck_norm = nn.LayerNorm(fused_dim)

        # Cross-attention: bottlenecks → output (decode)
        self.decode_query = nn.Parameter(torch.randn(1, fused_dim) * 0.02)
        self.decode_attn = nn.MultiheadAttention(
            fused_dim, num_heads, dropout=dropout, batch_first=True)
        self.decode_norm = nn.LayerNorm(fused_dim)

        # Output projection
        self.output_mlp = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, pc_feat, graph_feat, tabular_feat):
        B = pc_feat.shape[0]

        # Project modalities
        pc = self.pc_proj(pc_feat).unsqueeze(1)       # (B, 1, D)
        graph = self.graph_proj(graph_feat).unsqueeze(1)  # (B, 1, D)
        tab = self.tab_proj(tabular_feat).unsqueeze(1)    # (B, 1, D)

        # Stack modalities as sequence: (B, 3, D)
        modality_seq = torch.cat([pc, graph, tab], dim=1)

        # Expand bottleneck tokens for batch: (B, K, D)
        bottlenecks = self.bottleneck_tokens.unsqueeze(0).expand(B, -1, -1)

        # Encode: bottlenecks attend to modalities
        bn_encoded, _ = self.encode_attn(
            query=bottlenecks, key=modality_seq, value=modality_seq)
        bottlenecks = self.encode_norm(bottlenecks + bn_encoded)

        # Self-attention among bottlenecks
        bn_self, _ = self.bottleneck_self_attn(
            query=bottlenecks, key=bottlenecks, value=bottlenecks)
        bottlenecks = self.bottleneck_norm(bottlenecks + bn_self)

        # Decode: query token attends to enriched bottlenecks
        query = self.decode_query.unsqueeze(0).expand(B, -1, -1)  # (B, 1, D)
        decoded, self._attn_weights = self.decode_attn(
            query=query, key=bottlenecks, value=bottlenecks)
        output = self.decode_norm(query + decoded).squeeze(1)  # (B, D)

        return self.output_mlp(output)

    def get_bottleneck_attention(self):
        """Return last attention weights for interpretability.

        Returns (1, n_bottlenecks) showing which bottleneck was most important.
        """
        if hasattr(self, '_attn_weights') and self._attn_weights is not None:
            return self._attn_weights.detach().mean(dim=0)  # Average over heads
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Variant C: Hierarchical Multi-Scale Fusion
# ══════════════════════════════════════════════════════════════════════════════

class HierarchicalMultiScaleFusion(nn.Module):
    """Hierarchical multi-scale fusion at atom, fragment, and molecule levels.

    Scale 1 (Local):    Graph node features attend to surface point features
    Scale 2 (Fragment): Cation/anion-specific pooling and cross-attention
    Scale 3 (Global):   Molecule-level pooled features + tabular

    The three scales are combined with learned scale weights.
    """

    def __init__(self, pointcloud_dim=256, graph_dim=256, tabular_dim=25,
                 fused_dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        self.fused_dim = fused_dim

        # Projections
        self.pc_proj = nn.Linear(pointcloud_dim, fused_dim)
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.tab_proj = nn.Sequential(
            nn.Linear(tabular_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Scale 1: Local cross-attention (simplified — operates on global features
        # as proxy for atom-level since we have pooled representations)
        self.local_attn = nn.MultiheadAttention(
            fused_dim, num_heads, dropout=dropout, batch_first=True)
        self.local_norm = nn.LayerNorm(fused_dim)
        self.local_mlp = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), nn.GELU(), nn.Dropout(dropout))

        # Scale 2: Fragment-level (cation-anion decomposition)
        # Learns separate cation and anion projections from the fused surface/graph
        self.cation_proj = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim), nn.ReLU(), nn.Dropout(dropout))
        self.anion_proj = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim), nn.ReLU(), nn.Dropout(dropout))
        self.fragment_attn = nn.MultiheadAttention(
            fused_dim, num_heads, dropout=dropout, batch_first=True)
        self.fragment_norm = nn.LayerNorm(fused_dim)

        # Scale 3: Global molecule-level
        self.global_mlp = nn.Sequential(
            nn.Linear(fused_dim * 3, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Scale combination: learned weights
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)

        # Final output
        self.output_mlp = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, pc_feat, graph_feat, tabular_feat):
        B = pc_feat.shape[0]

        pc = self.pc_proj(pc_feat)
        graph = self.graph_proj(graph_feat)
        tab = self.tab_proj(tabular_feat)

        # ── Scale 1: Local (surface ↔ graph cross-attention) ──
        # Treat as sequence of 2 tokens
        local_seq = torch.stack([pc, graph], dim=1)  # (B, 2, D)
        local_out, _ = self.local_attn(local_seq, local_seq, local_seq)
        local_out = self.local_norm(local_seq + local_out)
        local_feat = self.local_mlp(local_out.mean(dim=1))  # (B, D)

        # ── Scale 2: Fragment-level (cation-anion decomposition) ──
        # The surface and graph each encode both ions; learn to decompose
        combined = torch.cat([pc, graph], dim=1)  # (B, 2*D)
        cation_feat = self.cation_proj(combined).unsqueeze(1)  # (B, 1, D)
        anion_feat = self.anion_proj(combined).unsqueeze(1)    # (B, 1, D)
        frag_seq = torch.cat([cation_feat, anion_feat], dim=1)  # (B, 2, D)
        frag_out, _ = self.fragment_attn(frag_seq, frag_seq, frag_seq)
        frag_feat = self.fragment_norm(frag_seq + frag_out).mean(dim=1)  # (B, D)

        # ── Scale 3: Global (all modalities) ──
        global_feat = self.global_mlp(torch.cat([pc, graph, tab], dim=1))  # (B, D)

        # ── Combine scales with learned weights ──
        weights = F.softmax(self.scale_weights, dim=0)
        fused = weights[0] * local_feat + weights[1] * frag_feat + weights[2] * global_feat

        return self.output_mlp(fused)


# ══════════════════════════════════════════════════════════════════════════════
# Variant D: Gated Residual Fusion
# ══════════════════════════════════════════════════════════════════════════════

class GatedResidualFusion(nn.Module):
    """Gated residual fusion with modality-specific gates.

    Each modality gets a learned gate that controls how much it contributes
    to the fused representation. Gates are conditioned on ALL modalities
    so each gate "sees" what others offer. Residual connections preserve
    individual modality information.

    Simpler than attention but effective — tests whether complexity helps.
    """

    def __init__(self, pointcloud_dim=256, graph_dim=256, tabular_dim=25,
                 fused_dim=256, dropout=0.3):
        super().__init__()
        self.fused_dim = fused_dim

        # Project all to common dimension
        self.pc_proj = nn.Linear(pointcloud_dim, fused_dim)
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.tab_proj = nn.Sequential(
            nn.Linear(tabular_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gate networks: each gate sees concatenation of all modalities
        gate_input = fused_dim * 3
        self.gate_pc = nn.Sequential(
            nn.Linear(gate_input, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )
        self.gate_graph = nn.Sequential(
            nn.Linear(gate_input, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )
        self.gate_tab = nn.Sequential(
            nn.Linear(gate_input, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )

        # Residual transform per modality
        self.residual_pc = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), nn.LayerNorm(fused_dim), nn.GELU())
        self.residual_graph = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), nn.LayerNorm(fused_dim), nn.GELU())
        self.residual_tab = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), nn.LayerNorm(fused_dim), nn.GELU())

        # Final fusion
        self.output_mlp = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, pc_feat, graph_feat, tabular_feat):
        # Project
        pc = self.pc_proj(pc_feat)
        graph = self.graph_proj(graph_feat)
        tab = self.tab_proj(tabular_feat)

        # Concatenate for gate conditioning
        concat = torch.cat([pc, graph, tab], dim=1)  # (B, 3*D)

        # Compute gates
        g_pc = self.gate_pc(concat)       # (B, D) values in [0, 1]
        g_graph = self.gate_graph(concat)
        g_tab = self.gate_tab(concat)

        # Gated features with residual
        pc_gated = g_pc * self.residual_pc(pc) + (1 - g_pc) * pc
        graph_gated = g_graph * self.residual_graph(graph) + (1 - g_graph) * graph
        tab_gated = g_tab * self.residual_tab(tab) + (1 - g_tab) * tab

        # Sum fusion (not concat — keeps dimension constant)
        fused = pc_gated + graph_gated + tab_gated

        return self.output_mlp(fused)

    def get_gate_values(self, pc_feat, graph_feat, tabular_feat):
        """Return gate activations for interpretability."""
        with torch.no_grad():
            pc = self.pc_proj(pc_feat)
            graph = self.graph_proj(graph_feat)
            tab = self.tab_proj(tabular_feat)
            concat = torch.cat([pc, graph, tab], dim=1)
            return {
                "surface_gate": self.gate_pc(concat).mean(dim=1).cpu(),
                "graph_gate": self.gate_graph(concat).mean(dim=1).cpu(),
                "tabular_gate": self.gate_tab(concat).mean(dim=1).cpu(),
            }
