"""COSMOBridge v4 Triple-Path (I3): Fusion + Chemprop + Atom-Surface D-MPNN.

Adds a third frozen path (Chemprop with per-atom COSMO features) to the
per-molecule router. Path C specializes in γ₂ (R²=0.892 in Section 3.3).

Per-property softmax routing over 3 paths:
    ŷ_p = α^A_p(x) · pred^A_p + α^B_p(x) · pred^B_p + α^C_p(x) · pred^C_p
    where (α^A, α^B, α^C) = softmax(router(x)) per property.
"""

import torch
import torch.nn as nn


# Initial per-property path preferences (7 properties × 3 paths)
# Path A (Fusion, CP-GBH): γ₁=0.908, γ₂=0.936 (BEST for activity coefficients)
# Path B (Chemprop): γ₁=0.828, γ₂=0.858 (weak for γ) but strong for excess energies
# Path C (AtomSurf): γ₁=0.826, γ₂=0.892 (γ₂ specialist but weaker than Fusion overall)
#
# Tuned: route γ₁/γ₂ HEAVILY to Fusion (Path A) since it is the best individual
# predictor for both. Allow small AtomSurf contribution for γ₂ (ensemble diversity).
V3_TRIPLE_INIT = torch.tensor([
    #   A     B     C      property                 softmax
    [ 2.5, -0.5,  0.5],   # gamma1: Fusion dom      [0.81, 0.04, 0.15]
    [ 2.0, -0.5,  1.0],   # gamma2: Fusion+AS blend [0.69, 0.06, 0.25]
    [ 0.3,  1.5, -0.2],   # G_E: Chemprop dom       [0.22, 0.65, 0.13]
    [ 0.3,  1.5, -0.2],   # H_E: Chemprop dom       [0.22, 0.65, 0.13]
    [ 0.4,  1.5, -0.4],   # G_mix: Chemprop dom     [0.24, 0.66, 0.10]
    [ 0.5,  1.0,  0.0],   # H_vap: mixed            [0.31, 0.50, 0.19]
    [ 1.5,  0.8, -0.2],   # P: Fusion dom           [0.64, 0.26, 0.10]
])  # shape (7, 3) — will be flattened to (21,) for router final bias


class TriplePathRouter(nn.Module):
    """MLP that predicts per-property path-softmax weights from molecular features."""

    def __init__(self, graph_dim=300, surface_dim=256, thermo_dim=25,
                 hidden=64, n_properties=7, n_paths=3, dropout=0.3):
        super().__init__()
        self.n_properties = n_properties
        self.n_paths = n_paths
        input_dim = graph_dim + surface_dim + thermo_dim
        output_dim = n_properties * n_paths  # 21

        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, output_dim),
        )

        # Initialize final layer to zero weight + triple-init bias
        init_bias = V3_TRIPLE_INIT.flatten()  # (21,)
        with torch.no_grad():
            self.router[-1].weight.zero_()
            self.router[-1].bias.copy_(init_bias)

    def forward(self, graph_fp, surface_fp, thermo_feat):
        """Returns logits of shape (B, 7, 3) for per-property softmax."""
        x = torch.cat([graph_fp, surface_fp, thermo_feat], dim=-1)
        logits = self.router(x)  # (B, 21)
        logits = logits.view(-1, self.n_properties, self.n_paths)  # (B, 7, 3)
        return logits

    def init_logits(self):
        return V3_TRIPLE_INIT.clone()


class COSMOBridgeV4Triple(nn.Module):
    """Three-path router combining Path A (Fusion), B (Chemprop), C (Atom-Surface)."""

    def __init__(self, graph_dim=300, surface_dim=256, thermo_dim=25,
                 hidden=64, n_properties=7, dropout=0.3):
        super().__init__()
        self.router = TriplePathRouter(
            graph_dim=graph_dim, surface_dim=surface_dim,
            thermo_dim=thermo_dim, hidden=hidden,
            n_properties=n_properties, n_paths=3, dropout=dropout,
        )

    def forward(self, graph_fp, surface_fp, thermo_feat,
                preds_fusion, preds_chemprop, preds_atom_surface):
        """
        Args:
            graph_fp: (B, graph_dim)
            surface_fp: (B, surface_dim)
            thermo_feat: (B, thermo_dim)
            preds_fusion, preds_chemprop, preds_atom_surface: (B, 7) each

        Returns:
            predictions: (B, 7)
            aux: dict with routing weights
        """
        logits = self.router(graph_fp, surface_fp, thermo_feat)  # (B, 7, 3)
        weights = torch.softmax(logits, dim=-1)  # (B, 7, 3)

        # Stack path predictions: (B, 7, 3)
        path_preds = torch.stack([preds_fusion, preds_chemprop, preds_atom_surface], dim=-1)

        # Weighted sum over paths
        predictions = (weights * path_preds).sum(dim=-1)  # (B, 7)

        return predictions, {"weights": weights.detach(), "logits": logits.detach()}

    def anchor_loss(self, logits):
        """MSE between predicted logits and v3 triple initialization."""
        init = self.router.init_logits().to(logits.device)  # (7, 3)
        return ((logits - init.unsqueeze(0)) ** 2).mean()
