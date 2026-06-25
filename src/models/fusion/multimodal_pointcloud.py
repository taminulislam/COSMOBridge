"""Multimodal model: PointNet (COSMO surface) + GNN (molecular graph) + Tabular.

Replaces the 2D vision stream with a 3D point cloud encoder that operates
directly on the COSMO isosurface mesh (vertices + normals + ESP).
"""

import torch
import torch.nn as nn

from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.graph.gnn import MolecularGNN
from src.models.fusion.cross_attention import CrossAttention
from src.data.preprocessing import FEATURE_COLUMNS


class PointCloudFusion(nn.Module):
    """Cross-attention fusion for PointNet + GNN + Tabular features."""

    def __init__(
        self,
        pointcloud_dim: int = 256,
        graph_dim: int = 256,
        tabular_dim: int = 25,
        fused_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Project all modalities to common dimension
        self.pc_proj = nn.Linear(pointcloud_dim, fused_dim)
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.tabular_proj = nn.Sequential(
            nn.Linear(tabular_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Cross-attention: pointcloud <-> graph
        self.pc_graph_attn = CrossAttention(fused_dim, fused_dim, num_heads, dropout)
        self.graph_pc_attn = CrossAttention(fused_dim, fused_dim, num_heads, dropout)

        # Layer norms
        self.ln_pc = nn.LayerNorm(fused_dim)
        self.ln_graph = nn.LayerNorm(fused_dim)
        self.ln_tabular = nn.LayerNorm(fused_dim)

        # Learnable modality weights
        self.modality_weights = nn.Parameter(torch.ones(3) / 3)

        # Final fusion
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
            pc_feat: (B, pointcloud_dim)
            graph_feat: (B, graph_dim)
            tabular_feat: (B, tabular_dim)
        Returns:
            (B, fused_dim)
        """
        pc = self.pc_proj(pc_feat)
        g = self.graph_proj(graph_feat)
        t = self.tabular_proj(tabular_feat)

        # Cross-attention between point cloud and graph
        pc_attended = self.pc_graph_attn(pc, g)
        g_attended = self.graph_pc_attn(g, pc)

        pc = self.ln_pc(pc + self.dropout(pc_attended))
        g = self.ln_graph(g + self.dropout(g_attended))
        t = self.ln_tabular(t)

        # Weighted combination
        weights = torch.softmax(self.modality_weights, dim=0)
        concat = torch.cat([pc * weights[0], g * weights[1], t * weights[2]], dim=-1)

        return self.fusion_mlp(concat)


class MultimodalPointCloudModel(nn.Module):
    """Multimodal model: PointNet + GNN + Tabular with cross-attention fusion.

    Parameters
    ----------
    config : dict
        Model configuration.
    pretrained_gnn_path : str, optional
        Path to pre-trained GNN checkpoint (from Phase 2 transfer learning).
    """

    def __init__(self, config: dict = None, pretrained_gnn_path: str = None):
        super().__init__()
        config = config or {}
        mc = config.get("model", {})

        # ── PointNet encoder ──
        pc_config = mc.get("pointcloud", {})
        pc_feat_dim = pc_config.get("feature_dim", 256)
        self.pointnet = PointNetEncoder(
            in_channels=pc_config.get("in_channels", 7),
            feature_dim=pc_feat_dim,
            hidden_dims=pc_config.get("hidden_dims", [64, 128, 256]),
            dropout=pc_config.get("dropout", 0.3),
        )

        # ── GNN encoder ──
        gc = mc.get("graph", {})
        graph_hidden = gc.get("hidden_dim", 256)
        self.gnn = MolecularGNN(
            atom_feature_dim=gc.get("atom_feature_dim", 22),
            bond_feature_dim=gc.get("bond_feature_dim", 7),
            hidden_dim=graph_hidden,
            num_layers=gc.get("num_layers", 4),
            conv_type=gc.get("conv_type", "GAT"),
            heads=gc.get("heads", 4),
            dropout=gc.get("dropout", 0.3),
            pooling=gc.get("pooling", "mean"),
            num_targets=0,
        )

        # Load pre-trained GNN weights if available
        if pretrained_gnn_path:
            self._load_pretrained_gnn(pretrained_gnn_path)

        # ── Fusion ──
        fc = mc.get("fusion", {})
        fused_dim = fc.get("fused_dim", 256)
        tabular_dim = len(FEATURE_COLUMNS)

        self.fusion = PointCloudFusion(
            pointcloud_dim=pc_feat_dim,
            graph_dim=graph_hidden,
            tabular_dim=tabular_dim,
            fused_dim=fused_dim,
            num_heads=fc.get("num_attention_heads", 8),
            dropout=fc.get("dropout", 0.3),
        )

        # ── Temperature skip connection (optional) ──
        self.thermo_dim = 5
        self.temp_skip = mc.get("temp_skip", False)

        # ── Prediction head ──
        pc2 = mc.get("prediction", {})
        num_targets = pc2.get("num_targets", 7)
        dropout = pc2.get("dropout", 0.3)

        pred_input_dim = fused_dim + (self.thermo_dim if self.temp_skip else 0)
        self.prediction_head = nn.Sequential(
            nn.Linear(pred_input_dim, fused_dim // 2),
            nn.BatchNorm1d(fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_targets),
        )

    def _load_pretrained_gnn(self, path):
        """Load pre-trained GNN weights, matching only GNN layers."""
        import os
        if not os.path.exists(path):
            print(f"  WARNING: Pre-trained GNN not found at {path}")
            return

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        # The checkpoint is a full model state_dict; filter to GNN keys
        gnn_state = {}
        for k, v in checkpoint.items():
            # Match keys that belong to GNN layers
            if any(k.startswith(prefix) for prefix in [
                "atom_projection", "convs", "batch_norms", "pool"
            ]):
                gnn_state[k] = v

        if gnn_state:
            missing, unexpected = self.gnn.load_state_dict(gnn_state, strict=False)
            print(f"  Loaded pre-trained GNN: {len(gnn_state)} params, "
                  f"{len(missing)} missing, {len(unexpected)} unexpected")
        else:
            print(f"  WARNING: No matching GNN keys in checkpoint")

    def forward(
        self,
        point_cloud,
        features,
        atom_features, edge_index, bond_features, batch,
        **kwargs,
    ):
        """
        Parameters
        ----------
        point_cloud : Tensor (B, N, 7)
        features : Tensor (B, feature_dim) — thermo + surface descriptors
        atom_features, edge_index, bond_features, batch : graph data
        """
        # Point cloud features
        pc_feat = self.pointnet(point_cloud)

        # Graph features
        graph_feat = self.gnn.get_features(atom_features, edge_index, bond_features, batch)

        # Fusion (tabular features fed directly — no separate encoder needed)
        fused = self.fusion(pc_feat, graph_feat, features)

        # Temperature skip: inject thermo features directly into prediction head
        if self.temp_skip:
            thermo_feat = features[:, :self.thermo_dim]
            fused = torch.cat([fused, thermo_feat], dim=-1)

        return self.prediction_head(fused)
