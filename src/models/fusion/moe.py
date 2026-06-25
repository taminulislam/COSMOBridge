"""Property-Conditioned Mixture of Experts for IL Property Prediction.

A single model that learns which internal experts handle which properties
via a learned gating network. Replaces external ensemble routing with
end-to-end trainable expert specialization.

Architecture:
  Shared Backbone: PointNet (surface) + GNN (graph) + cross-attention fusion
  Expert Heads: K specialized MLPs (surface, mixing, thermo, generalist)
  Gating Network: property-conditioned soft routing over experts
  Output: weighted combination of expert predictions per property

Innovation:
  - MoE applied to multimodal molecular property prediction (novel)
  - Property-conditioned gating (gate depends on WHICH property, not just input)
  - Interpretable: gating weights reveal property-expert specialization
  - Snapshot ensemble at inference for variance reduction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.cross_attention import CrossAttention
from src.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM


class ExpertHead(nn.Module):
    """A single expert MLP that predicts all 7 properties."""

    def __init__(self, input_dim, hidden_dim=128, num_targets=7, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets),
        )

    def forward(self, x):
        return self.net(x)


class PropertyConditionedGating(nn.Module):
    """Gating network that produces per-property weights over experts.

    For each of the 7 properties, outputs a soft weight distribution
    over K experts. The gating is conditioned on:
    1. The fused molecular representation (input-dependent)
    2. A learned property embedding (property-dependent)

    This allows the gate to learn, e.g., "for H_vap, rely on expert 3 (thermo)
    but for gamma1, rely on expert 1 (surface)".
    """

    def __init__(self, input_dim, num_experts=4, num_properties=7, hidden_dim=64):
        super().__init__()
        self.num_experts = num_experts
        self.num_properties = num_properties

        # Learned property embeddings
        self.property_embed = nn.Embedding(num_properties, hidden_dim)

        # Gate MLP: [molecular_repr, property_embed] → expert weights
        self.gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

        # Load balancing loss coefficient
        self.load_balance_coeff = 0.01

    def forward(self, x):
        """
        Args:
            x: (B, input_dim) fused molecular representation

        Returns:
            weights: (B, num_properties, num_experts) soft routing weights
            load_balance_loss: scalar encouraging uniform expert utilization
        """
        B = x.shape[0]
        prop_ids = torch.arange(self.num_properties, device=x.device)
        prop_emb = self.property_embed(prop_ids)  # (P, hidden)

        # Expand: (B, 1, input) + (1, P, hidden) → (B, P, input+hidden)
        x_expanded = x.unsqueeze(1).expand(-1, self.num_properties, -1)
        prop_expanded = prop_emb.unsqueeze(0).expand(B, -1, -1)
        combined = torch.cat([x_expanded, prop_expanded], dim=-1)  # (B, P, input+hidden)

        # Compute gate logits
        gate_logits = self.gate(combined)  # (B, P, K)
        weights = F.softmax(gate_logits, dim=-1)  # (B, P, K)

        # Load balancing loss: encourage all experts to be used
        avg_weights = weights.mean(dim=(0, 1))  # (K,)
        uniform = torch.ones_like(avg_weights) / self.num_experts
        load_balance = self.load_balance_coeff * F.kl_div(
            avg_weights.log(), uniform, reduction='sum')

        return weights, load_balance


class SharedBackbone(nn.Module):
    """Shared encoder: PointNet + GNN + cross-attention fusion."""

    def __init__(self, feature_dim, pc_feat_dim=256, graph_hidden=256,
                 fused_dim=256, dropout=0.3):
        super().__init__()

        # PointNet
        self.pointnet = PointNetEncoder(
            in_channels=7, feature_dim=pc_feat_dim, dropout=dropout)

        # GNN
        self.gnn = MolecularGNN(
            atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=graph_hidden, num_layers=4, conv_type="GAT", heads=4,
            dropout=dropout, pooling="mean", num_targets=0)

        # Cross-attention: surface ↔ graph
        self.pc_graph_attn = CrossAttention(fused_dim, fused_dim, num_heads=8, dropout=dropout)
        self.graph_pc_attn = CrossAttention(fused_dim, fused_dim, num_heads=8, dropout=dropout)

        # Projections to common space
        self.pc_proj = nn.Linear(pc_feat_dim, fused_dim)
        self.graph_proj = nn.Linear(graph_hidden, fused_dim)
        self.feat_proj = nn.Sequential(
            nn.Linear(feature_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.ln_pc = nn.LayerNorm(fused_dim)
        self.ln_graph = nn.LayerNorm(fused_dim)

        # Final fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim * 3, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_dim = fused_dim

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch):
        # Encode
        pc_feat = self.pc_proj(self.pointnet(point_cloud))
        graph_feat = self.graph_proj(
            self.gnn.get_features(atom_features, edge_index, bond_features, batch))
        tab_feat = self.feat_proj(features)

        # Cross-attention
        pc_attended = self.pc_graph_attn(pc_feat, graph_feat)
        graph_attended = self.graph_pc_attn(graph_feat, pc_feat)
        pc_feat = self.ln_pc(pc_feat + pc_attended)
        graph_feat = self.ln_graph(graph_feat + graph_attended)

        # Fuse all three
        concat = torch.cat([pc_feat, graph_feat, tab_feat], dim=-1)
        return self.fusion_mlp(concat)


class MixtureOfExpertsModel(nn.Module):
    """Property-Conditioned Mixture of Experts for IL property prediction.

    A single model that internally routes different properties to
    specialized expert heads via a learned gating mechanism.

    Parameters
    ----------
    feature_dim : int
        Dimension of tabular features (thermo + surface + Morgan FP).
    num_experts : int
        Number of expert heads.
    num_targets : int
        Number of target properties.
    fused_dim : int
        Dimension of shared backbone output.
    pretrained_gnn_path : str, optional
        Path to pre-trained GNN weights.
    """

    def __init__(self, feature_dim, num_experts=4, num_targets=7,
                 fused_dim=256, dropout=0.3, pretrained_gnn_path=None):
        super().__init__()
        self.num_experts = num_experts
        self.num_targets = num_targets

        # Shared backbone
        self.backbone = SharedBackbone(
            feature_dim=feature_dim, fused_dim=fused_dim, dropout=dropout)

        # Load pre-trained GNN
        if pretrained_gnn_path:
            self._load_pretrained_gnn(pretrained_gnn_path)

        # Expert heads
        self.experts = nn.ModuleList([
            ExpertHead(fused_dim, hidden_dim=128, num_targets=num_targets, dropout=dropout)
            for _ in range(num_experts)
        ])

        # Property-conditioned gating
        self.gating = PropertyConditionedGating(
            input_dim=fused_dim, num_experts=num_experts,
            num_properties=num_targets, hidden_dim=64)

    def _load_pretrained_gnn(self, path):
        """Load pre-trained GNN weights into backbone."""
        import os
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found")
            return
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        gnn_state = {}
        for k, v in ckpt.items():
            for prefix in ["atom_projection", "convs", "batch_norms", "pool"]:
                if k.startswith(prefix):
                    gnn_state[k] = v
        if gnn_state:
            missing, _ = self.backbone.gnn.load_state_dict(gnn_state, strict=False)
            print(f"  Loaded pre-trained GNN: {len(gnn_state)} params")

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        """
        Returns:
            predictions: (B, num_targets)
            aux_losses: dict with load_balance_loss and gating_weights
        """
        # Shared representation
        h = self.backbone(point_cloud, features, atom_features, edge_index,
                          bond_features, batch)  # (B, fused_dim)

        # Expert predictions: each expert predicts all 7 properties
        expert_preds = torch.stack(
            [expert(h) for expert in self.experts], dim=2)  # (B, P, K)

        # Gating: per-property weights over experts
        gate_weights, load_balance_loss = self.gating(h)  # (B, P, K)

        # Weighted combination
        predictions = (expert_preds * gate_weights).sum(dim=2)  # (B, P)

        return predictions, {
            "load_balance_loss": load_balance_loss,
            "gate_weights": gate_weights.detach(),
        }

    def predict(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        """Inference-only forward (returns just predictions)."""
        preds, _ = self.forward(point_cloud, features, atom_features,
                                edge_index, bond_features, batch)
        return preds

    def get_expert_assignments(self, point_cloud, features, atom_features,
                                edge_index, bond_features, batch):
        """Get interpretable expert-property assignments.

        Returns:
            (num_properties,) indices of dominant expert per property
            (num_properties, num_experts) mean gating weights
        """
        self.eval()
        with torch.no_grad():
            h = self.backbone(point_cloud, features, atom_features,
                              edge_index, bond_features, batch)
            gate_weights, _ = self.gating(h)
            mean_weights = gate_weights.mean(dim=0)  # (P, K)
            assignments = mean_weights.argmax(dim=1)  # (P,)
        return assignments.cpu(), mean_weights.cpu()
