"""Multimodal model combining vision, graph, and tabular encoders with fusion."""

import torch
import torch.nn as nn

from src.models.fusion.cross_attention import MultimodalFusion


class MultimodalILModel(nn.Module):
    """Full multimodal model for ionic liquid property prediction.

    Combines:
      - DualImageEncoder (COSMO + EP images) -> vision features
      - MolecularGNN (molecular graphs from SMILES) -> graph features
      - TabularDNN (temperature, composition, IL identity) -> tabular features
      - MultimodalFusion (cross-attention) -> fused features
      - Prediction head -> 7 thermodynamic properties
    """

    def __init__(self, config: dict = None):
        super().__init__()
        config = config or {}
        mc = config.get("model", {})

        # ── Vision encoder ──
        vc = mc.get("vision", {})
        from src.models.vision.vit import DualImageEncoder
        freeze_layers = vc.get("freeze_layers", 0)
        self.vision_encoder = DualImageEncoder(
            backbone=vc.get("backbone", "resnet34"),
            pretrained=vc.get("pretrained", True),
            shared_encoder=False,
            dropout=vc.get("dropout", 0.3),
            num_targets=0,  # we use our own prediction head
        )
        # Determine vision feature dim
        if "resnet" in vc.get("backbone", "resnet34"):
            vision_feat_dim = 512 * 2  # two encoders concatenated
        else:
            vision_feat_dim = 768 * 2

        # Freeze vision layers if configured
        if freeze_layers > 0:
            for encoder in [self.vision_encoder.encoder_cosmo, self.vision_encoder.encoder_ep]:
                encoder._freeze(freeze_layers)

        # ── Graph encoder ──
        gc = mc.get("graph", {})
        from src.models.graph.gnn import MolecularGNN
        self.graph_encoder = MolecularGNN(
            atom_feature_dim=gc.get("atom_feature_dim", 22),
            bond_feature_dim=gc.get("bond_feature_dim", 7),
            hidden_dim=gc.get("hidden_dim", 256),
            num_layers=gc.get("num_layers", 4),
            conv_type=gc.get("conv_type", "GAT"),
            heads=gc.get("heads", 4),
            dropout=gc.get("dropout", 0.3),
            pooling=gc.get("pooling", "mean"),
            num_targets=0,  # we use our own prediction head
        )
        graph_feat_dim = gc.get("hidden_dim", 256)

        # ── Tabular encoder ──
        tc = mc.get("tabular", {})
        from src.models.tabular.dnn import TabularDNN
        from src.data.preprocessing import FEATURE_COLUMNS
        self.tabular_encoder = TabularDNN(
            num_ils=28,
            num_cations=9,
            num_anions=7,
            feature_dim=len(FEATURE_COLUMNS),
            il_embed_dim=tc.get("il_embed_dim", 64),
            cation_embed_dim=tc.get("cation_embed_dim", 32),
            anion_embed_dim=tc.get("anion_embed_dim", 32),
            hidden_dims=tc.get("hidden_dims", [128, 64, 32]),
            dropout=tc.get("dropout", 0.4),
            num_targets=0,  # we use our own prediction head
        )
        tabular_feat_dim = tc.get("hidden_dims", [128, 64, 32])[-1]

        # ── Fusion ──
        fc = mc.get("fusion", {})
        fused_dim = fc.get("fused_dim", 512)
        self.fusion = MultimodalFusion(
            vision_dim=vision_feat_dim,
            graph_dim=graph_feat_dim,
            tabular_dim=tabular_feat_dim,
            fused_dim=fused_dim,
            num_heads=fc.get("num_attention_heads", 8),
            dropout=fc.get("dropout", 0.3),
        )

        # ── Prediction head ──
        pc = mc.get("prediction", {})
        pred_hidden = pc.get("hidden_dims", [256, 128])
        num_targets = pc.get("num_targets", 7)
        dropout = pc.get("dropout", 0.3)

        layers = []
        in_dim = fused_dim
        for h_dim in pred_hidden:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_targets))
        self.prediction_head = nn.Sequential(*layers)

    def get_features(
        self,
        features, il_idx, cation_idx, anion_idx,
        cosmo_image, ep_image,
        atom_features, edge_index, bond_features, graph_batch,
        **kwargs,
    ):
        """Extract fused multimodal features before prediction."""
        # Vision features
        vision_feat = self.vision_encoder.get_features(cosmo_image, ep_image)

        # Graph features
        graph_feat = self.graph_encoder.get_features(
            atom_features, edge_index, bond_features, graph_batch
        )

        # Tabular features
        tabular_feat = self.tabular_encoder.get_features(
            features, il_idx, cation_idx, anion_idx
        )

        # Fusion
        fused = self.fusion(vision_feat, graph_feat, tabular_feat)
        return fused

    def forward(
        self,
        features, il_idx, cation_idx, anion_idx,
        cosmo_image, ep_image,
        atom_features, edge_index, bond_features, graph_batch,
        **kwargs,
    ):
        """Full forward pass: encoders -> fusion -> prediction."""
        fused = self.get_features(
            features, il_idx, cation_idx, anion_idx,
            cosmo_image, ep_image,
            atom_features, edge_index, bond_features, graph_batch,
        )
        return self.prediction_head(fused)
