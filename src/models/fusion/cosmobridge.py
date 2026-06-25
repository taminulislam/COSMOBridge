"""COSMOBridge: Bridging 2D Molecular Graphs and 3D COSMO Surfaces
via Temperature-Conditioned Bilinear Fusion with Property-Adaptive Routing.

Single-model architecture that internally routes each property to its
optimal prediction path:

  Path A (Fusion): GBH bilinear of graph × surface features
    → Best for surface-dependent properties (γ₁, γ₂, P)

  Path B (Direct): Graph-only FFN bypassing fusion
    → Best for bulk thermodynamic properties (G_E, H_E, G_mix)

  Per-property gate α_p learns the optimal mix:
    pred_p = α_p · head_fused(h_fused) + (1-α_p) · head_direct(h_direct)

Architecture:
  Chemprop D-MPNN (frozen, 300D) ──┐
                                    ├→ GBH Bilinear Fusion → h_fused ──┐
  PointNet COSMO (frozen, 256D) ──┘                                    │
                                                                        ├→ α_p gate → pred_p
  Chemprop D-MPNN (frozen, 300D) → Graph FFN → h_direct ──────────────┘

  Thermo features (25D) → used by both paths

Total trainable: ~505K (fusion ~471K + direct FFN ~34K + 7 gates)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.fusion.gated_bilinear_hyper_v2 import GatedBilinearHyperFusionV2


class COSMOBridge(nn.Module):
    """COSMOBridge with built-in per-property routing.

    Parameters
    ----------
    graph_dim : int
        Chemprop graph fingerprint dimension (300D).
    surface_dim : int
        PointNet surface feature dimension (256D).
    thermo_dim : int
        Thermodynamic feature dimension (25D).
    fused_dim : int
        Internal fused representation dimension.
    rank : int
        Low-rank bilinear factorization rank.
    hyper_hidden : int
        HyperNetwork hidden dimension.
    n_properties : int
        Number of target properties.
    dropout : float
        Dropout probability.
    """

    def __init__(self, graph_dim=300, surface_dim=256, thermo_dim=25,
                 fused_dim=256, rank=32, hyper_hidden=64, n_properties=7,
                 dropout=0.3):
        super().__init__()

        # ── Path A: GBH Bilinear Fusion (graph × surface) ──
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.surface_proj = nn.Linear(surface_dim, fused_dim)

        self.fusion = GatedBilinearHyperFusionV2(
            pointcloud_dim=fused_dim, graph_dim=fused_dim, tabular_dim=thermo_dim,
            fused_dim=fused_dim, rank=rank, thermo_dim=5, hyper_hidden=hyper_hidden,
            dropout=dropout)

        self.fused_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_properties),
        )

        # ── Path B: Direct Graph FFN (bypass fusion) ──
        self.direct_head = nn.Sequential(
            nn.Linear(graph_dim + thermo_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_properties),
        )

        # ── Per-Property Gate ──
        # Initialize: γ₁,γ₂,P → fusion (α≈1); G_E,H_E,G_mix → direct (α≈0); H_vap → mixed
        self.gate_logits = nn.Parameter(torch.tensor([
            2.0,    # gamma1 → fusion (sigmoid(2) = 0.88)
            2.0,    # gamma2 → fusion
            -2.0,   # G_E → direct (sigmoid(-2) = 0.12)
            -2.0,   # H_E → direct
            -2.0,   # G_mix → direct
            0.0,    # H_vap → mixed (sigmoid(0) = 0.5)
            1.5,    # P → fusion (sigmoid(1.5) = 0.82)
        ]))

    def forward(self, graph_feat, surface_feat, thermo_feat):
        """
        Args:
            graph_feat: (B, graph_dim) — frozen Chemprop fingerprint
            surface_feat: (B, surface_dim) — frozen PointNet features
            thermo_feat: (B, thermo_dim) — thermodynamic features

        Returns:
            predictions: (B, n_properties)
            aux: dict with gate values for analysis
        """
        # Path A: Bilinear fusion of projected graph × surface features
        g_proj = self.graph_proj(graph_feat)
        s_proj = self.surface_proj(surface_feat)
        h_fused = self.fusion(s_proj, g_proj, thermo_feat)
        preds_fused = self.fused_head(h_fused)  # (B, 7)

        # Path B: Direct graph + thermo → FFN (no surface)
        h_direct = torch.cat([graph_feat, thermo_feat], dim=-1)
        preds_direct = self.direct_head(h_direct)  # (B, 7)

        # Per-property gated combination
        alpha = torch.sigmoid(self.gate_logits)  # (7,)
        predictions = alpha.unsqueeze(0) * preds_fused + (1 - alpha.unsqueeze(0)) * preds_direct

        return predictions, {"gate_values": alpha.detach(),
                              "preds_fused": preds_fused.detach(),
                              "preds_direct": preds_direct.detach()}
