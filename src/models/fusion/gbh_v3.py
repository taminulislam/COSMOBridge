"""GBH v3: Property-Adaptive HyperNetwork Fusion with D-MPNN.

Four targeted fixes to close the gap with Chemprop:

Fix 1: D-MPNN backbone (replaces GAT-GNN)
  - Directed message passing, bond-level states, sum pooling
  - Proven to improve G_E, H_E, G_mix (MoE+DMPNN showed +0.022 avg)

Fix 2: Compact architecture (~400K total, close to Chemprop's 300K)
  - PointNet: 256D → 128D (halved)
  - D-MPNN: 300D → 200D (reduced)
  - Fusion: rank 16 (was 32), projections via bottleneck
  - Total: ~400K vs 737K (GBH v2) vs 300K (Chemprop)

Fix 3: Property-adaptive fusion
  - Learned per-property gate α_p decides how much to use fusion vs graph-only
  - For gamma1: α≈1 (use surface fusion heavily)
  - For G_E: α≈0 (rely on graph backbone)
  - This lets the model bypass fusion for properties that don't need it

Fix 4: Physics auxiliary loss
  - G_E = RT(x1 ln γ1 + x2 ln γ2) enforced as soft constraint
  - Propagates gradient from well-predicted γ1/γ2 into G_E
  - Weight ramped from 0→0.1 over training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactHyperNet(nn.Module):
    """Lightweight HyperNetwork (~3K params)."""

    def __init__(self, thermo_dim=5, fused_dim=128, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(thermo_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head_W = nn.Linear(hidden, fused_dim)
        self.head_gate = nn.Linear(hidden, fused_dim)
        self.head_bias = nn.Linear(hidden, fused_dim)

        nn.init.zeros_(self.head_W.weight); nn.init.ones_(self.head_W.bias)
        nn.init.zeros_(self.head_gate.weight); nn.init.zeros_(self.head_gate.bias)
        nn.init.zeros_(self.head_bias.weight); nn.init.zeros_(self.head_bias.bias)

    def forward(self, thermo):
        h = self.net(thermo)
        return torch.sigmoid(self.head_W(h)), torch.sigmoid(self.head_gate(h)), self.head_bias(h)


class CompactBilinear(nn.Module):
    """Low-rank bilinear with bottleneck projections."""

    def __init__(self, dim_a, dim_b, rank=16, out_dim=128):
        super().__init__()
        self.U = nn.Linear(dim_a, rank, bias=False)
        self.V = nn.Linear(dim_b, rank, bias=False)
        self.proj = nn.Linear(rank, out_dim)

    def forward(self, h_a, h_b):
        return self.proj(self.U(h_a) * self.V(h_b))


class PropertyAdaptiveGate(nn.Module):
    """Learns per-property mixing weight: α_p * h_fused + (1-α_p) * h_graph.

    Properties that benefit from surface fusion (gamma1, gamma2) get high α.
    Properties that prefer graph-only (G_E, H_E) get low α.
    """

    def __init__(self, fused_dim, n_properties=7):
        super().__init__()
        # One gate per property, initialized to 0.5
        self.gate_logits = nn.Parameter(torch.zeros(n_properties))

    def forward(self, h_fused, h_graph):
        """
        Args:
            h_fused: (B, fused_dim) — fusion output
            h_graph: (B, fused_dim) — graph-only output
        Returns:
            (B, fused_dim, 7) — per-property mixed representations
        """
        alpha = torch.sigmoid(self.gate_logits)  # (7,)
        # Mix: alpha * fused + (1-alpha) * graph, per property
        # Expand for broadcasting: (B, D, 1) * (1, 1, 7)
        h_fused_exp = h_fused.unsqueeze(2)   # (B, D, 1)
        h_graph_exp = h_graph.unsqueeze(2)   # (B, D, 1)
        alpha_exp = alpha.view(1, 1, -1)      # (1, 1, 7)
        return alpha_exp * h_fused_exp + (1 - alpha_exp) * h_graph_exp  # (B, D, 7)


class GBHv3(nn.Module):
    """GBH v3: Compact Property-Adaptive HyperNetwork Fusion.

    PointNet(128D) + D-MPNN(200D) → Low-rank bilinear(rank=16) + HyperNet
    → Property-adaptive gate → Per-property prediction heads
    """

    def __init__(self, feature_dim=25, thermo_dim=5, dropout=0.25):
        super().__init__()
        from src.models.pointcloud.pointnet import PointNetEncoder
        from src.models.graph.dmpnn import DirectedMPNN

        pc_dim = 128
        graph_dim = 200
        fused_dim = 128

        # Compact encoders
        self.pointnet = PointNetEncoder(in_channels=7, feature_dim=pc_dim, dropout=dropout)
        self.dmpnn = DirectedMPNN(
            atom_feature_dim=22, bond_feature_dim=7,
            hidden_dim=graph_dim, num_layers=3, dropout=dropout, num_targets=0)

        # Bottleneck projections to fused_dim
        self.pc_proj = nn.Linear(pc_dim, fused_dim)
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.tabular_proj = nn.Sequential(
            nn.Linear(feature_dim, fused_dim), nn.ReLU(), nn.Dropout(dropout))

        # Compact bilinear + HyperNet
        self.bilinear = CompactBilinear(fused_dim, fused_dim, rank=16, out_dim=fused_dim)
        self.hypernet = CompactHyperNet(thermo_dim=thermo_dim, fused_dim=fused_dim, hidden=32)

        # Residual weights
        self.res_alpha = nn.Parameter(torch.tensor(0.1))
        self.res_beta = nn.Parameter(torch.tensor(0.1))

        # Fusion output
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.fusion_out = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), nn.GELU(), nn.Dropout(dropout))

        # Property-adaptive gate
        self.prop_gate = PropertyAdaptiveGate(fused_dim, n_properties=7)

        # Per-property prediction (shared base + lightweight per-prop heads)
        self.shared_head = nn.Sequential(
            nn.Linear(fused_dim, 64), nn.GELU(), nn.Dropout(dropout))
        self.prop_heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(7)])

        self.thermo_dim = thermo_dim

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        # Encode
        h_pc = self.pc_proj(self.pointnet(point_cloud))
        h_graph_raw = self.dmpnn.get_features(atom_features, edge_index, bond_features, batch)
        h_graph = self.graph_proj(h_graph_raw)
        h_thermo = self.tabular_proj(features)

        # Low-rank bilinear interaction
        h_bilinear = self.bilinear(h_pc, h_graph)

        # HyperNet T-dependent parameters
        thermo_input = features[:, :self.thermo_dim]
        W, gate, bias = self.hypernet(thermo_input)

        # Gated fusion
        h_weighted = W * h_bilinear
        h_fused = gate * h_weighted + (1 - gate) * h_thermo + bias
        h_fused = h_fused + self.res_alpha * h_pc + self.res_beta * h_graph
        h_fused = self.fusion_norm(h_fused)
        h_fused = self.fusion_out(h_fused)

        # Property-adaptive: mix fused and graph-only per property
        h_mixed = self.prop_gate(h_fused, h_graph)  # (B, D, 7)

        # Predict each property from its adapted representation
        preds = []
        for p in range(7):
            h_p = h_mixed[:, :, p]  # (B, D)
            h_p = self.shared_head(h_p)
            preds.append(self.prop_heads[p](h_p))
        predictions = torch.cat(preds, dim=1)  # (B, 7)

        # Return gate values for analysis
        gate_values = torch.sigmoid(self.prop_gate.gate_logits).detach()
        return predictions, {"gate_values": gate_values}
