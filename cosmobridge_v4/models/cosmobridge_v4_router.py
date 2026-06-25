"""COSMOBridge v4: Per-Molecule Routing (I1).

Replaces v3's 7 scalar sigmoid gates with a tiny MoE router that predicts
per-sample, per-property routing weights α_p(x) from molecular features.

Router: [chemprop_fp (300D), surface_fp (256D), thermo_feat (25D)] = 581D
        → Linear(581, 64) → GELU → Dropout(0.3) → LayerNorm → Linear(64, 7)
        → sigmoid

Prediction: ŷ_p = α_p(x) · ŷ^fusion_p + (1 - α_p(x)) · ŷ^chemprop_p

Anchor regularization ties the router output toward v3's learned scalar
logits at the start of training, preventing the router from drifting
far on 223 training samples.
"""

import torch
import torch.nn as nn


# v3's best-seed learned gate logits (approximate)
V3_INIT_LOGITS = torch.tensor([0.36, 0.39, 0.36, 0.42, 0.45, 0.37, 0.69])
# Convert sigmoid outputs back to logits
V3_INIT_LOGITS = torch.log(V3_INIT_LOGITS / (1 - V3_INIT_LOGITS))


class PerMoleculeRouter(nn.Module):
    """Tiny MLP that predicts per-property routing weights from molecular features."""

    def __init__(self, graph_dim=300, surface_dim=256, thermo_dim=25,
                 hidden=64, n_properties=7, dropout=0.3):
        super().__init__()
        self.n_properties = n_properties
        input_dim = graph_dim + surface_dim + thermo_dim

        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n_properties),
        )

        # Initialize final layer to zero weight + v3 bias → starts as v3 identity
        with torch.no_grad():
            self.router[-1].weight.zero_()
            self.router[-1].bias.copy_(V3_INIT_LOGITS)

    def forward(self, graph_fp, surface_fp, thermo_feat):
        """Returns per-sample, per-property alpha values in [0, 1]."""
        x = torch.cat([graph_fp, surface_fp, thermo_feat], dim=-1)
        logits = self.router(x)  # (B, 7)
        return logits  # caller applies sigmoid

    def init_logits(self):
        """Return the v3 initialization logits for anchor loss."""
        return V3_INIT_LOGITS.clone()


class COSMOBridgeV4Router(nn.Module):
    """v4 wrapper: uses frozen v3 paths (passed as pre-computed predictions)
    + a per-molecule router that learns per-sample α_p(x).

    The 'fusion' and 'chemprop' path predictions are expected to be computed
    outside this module (via the frozen models) and passed in. This matches
    the efficient multi-seed training pattern.
    """

    def __init__(self, graph_dim=300, surface_dim=256, thermo_dim=25,
                 hidden=64, n_properties=7, dropout=0.3):
        super().__init__()
        self.router = PerMoleculeRouter(
            graph_dim=graph_dim, surface_dim=surface_dim,
            thermo_dim=thermo_dim, hidden=hidden,
            n_properties=n_properties, dropout=dropout,
        )

    def forward(self, graph_fp, surface_fp, thermo_feat, preds_fusion, preds_chemprop):
        """
        Args:
            graph_fp: (B, graph_dim) frozen Chemprop fingerprint
            surface_fp: (B, surface_dim) frozen PointNet features
            thermo_feat: (B, thermo_dim) thermodynamic features
            preds_fusion: (B, 7) precomputed predictions from Path A (frozen)
            preds_chemprop: (B, 7) precomputed predictions from Path B (frozen)

        Returns:
            predictions: (B, 7)
            aux: dict with routing weights
        """
        logits = self.router(graph_fp, surface_fp, thermo_feat)  # (B, 7)
        alpha = torch.sigmoid(logits)

        predictions = alpha * preds_fusion + (1.0 - alpha) * preds_chemprop

        return predictions, {"alpha": alpha.detach(), "logits": logits.detach()}

    def anchor_loss(self, logits):
        """MSE between predicted logits and v3 initialization.

        Call during training to keep the router from drifting too far.
        Decayed over epochs via caller.
        """
        init = self.router.init_logits().to(logits.device)  # (7,)
        return ((logits - init.unsqueeze(0)) ** 2).mean()
