"""
Vision models (ViT and CNN) for molecular image analysis.

Provides encoders that extract feature vectors from molecular visualizations
(e.g., COSMO surface maps, electrostatic potential images) for downstream
ionic liquid property prediction.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.models import (
    ViT_B_16_Weights,
    ResNet34_Weights,
)


class MolecularVisionEncoder(nn.Module):
    """Encodes a single molecular image into a feature vector and optional
    prediction head.

    Parameters
    ----------
    backbone : str
        Either ``"vit_small_patch16_224"`` or ``"resnet34"``.
    pretrained : bool
        Load ImageNet-pretrained weights.
    feature_dim : int
        Dimensionality of the backbone feature vector (768 for ViT, 512 for
        ResNet34).  Used to size the projection head.
    freeze_layers : int
        Number of initial layers/blocks to freeze for transfer learning.
    dropout : float
        Dropout probability before the projection head.
    num_targets : int
        Number of regression/classification targets.
    """

    # Backbone name -> (factory, pretrained feature dim)
    _BACKBONES = {
        "vit_small_patch16_224": ("vit", 768),
        "resnet34": ("resnet", 512),
    }

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        feature_dim: int = 512,
        freeze_layers: int = 0,
        dropout: float = 0.3,
        num_targets: int = 7,
    ):
        super().__init__()
        if backbone not in self._BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                f"Choose from {list(self._BACKBONES.keys())}."
            )

        family, self._backbone_dim = self._BACKBONES[backbone]
        self.backbone_name = backbone

        if family == "vit":
            self.encoder = self._build_vit(pretrained)
        else:
            self.encoder = self._build_resnet(pretrained)

        if freeze_layers > 0:
            self._freeze(freeze_layers)

        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Linear(self._backbone_dim, num_targets)

    # ------------------------------------------------------------------
    # Backbone builders
    # ------------------------------------------------------------------
    def _build_vit(self, pretrained: bool) -> nn.Module:
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.vit_b_16(weights=weights)
        # Replace classification head with identity to expose raw features.
        model.heads = nn.Identity()
        return model

    def _build_resnet(self, pretrained: bool) -> nn.Module:
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.resnet34(weights=weights)
        # Remove the final fully-connected layer.
        model.fc = nn.Identity()
        return model

    # ------------------------------------------------------------------
    # Freezing helpers
    # ------------------------------------------------------------------
    def _freeze(self, n_layers: int) -> None:
        """Freeze the first *n_layers* layers/blocks of the backbone.

        For ResNet this corresponds to the children (conv1, bn1, relu,
        maxpool, layer1, layer2, ...).  For ViT this corresponds to the
        first *n_layers* transformer encoder blocks.
        """
        if self.backbone_name == "resnet34":
            children = list(self.encoder.children())
            for child in children[:n_layers]:
                for param in child.parameters():
                    param.requires_grad = False
        else:
            # ViT: freeze the first n encoder blocks
            # torchvision ViT stores blocks in encoder.layers
            blocks = list(self.encoder.encoder.layers.children())
            for block in blocks[:n_layers]:
                for param in block.parameters():
                    param.requires_grad = False

    # ------------------------------------------------------------------
    # Forward API
    # ------------------------------------------------------------------
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the backbone feature vector.

        Parameters
        ----------
        x : Tensor
            Image batch of shape ``(B, 3, 224, 224)``.

        Returns
        -------
        Tensor
            Feature vector of shape ``(B, backbone_dim)`` where
            *backbone_dim* is 768 for ViT and 512 for ResNet34.
        """
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: features -> dropout -> projection.

        Parameters
        ----------
        x : Tensor
            Image batch of shape ``(B, 3, 224, 224)``.

        Returns
        -------
        Tensor
            Predictions of shape ``(B, num_targets)``.
        """
        features = self.get_features(x)
        features = self.dropout(features)
        return self.projection(features)


class DualImageEncoder(nn.Module):
    """Encodes two molecular images (e.g., COSMO surface + electrostatic
    potential) and fuses their representations for property prediction.

    Parameters
    ----------
    backbone : str
        Backbone architecture (see :class:`MolecularVisionEncoder`).
    pretrained : bool
        Load ImageNet-pretrained weights.
    shared_encoder : bool
        If ``True``, a single encoder is used for both images.  Otherwise
        two independent encoders are created.
    feature_dim : int
        Per-image backbone feature dimensionality.
    dropout : float
        Dropout probability applied to the concatenated features.
    num_targets : int
        Number of prediction targets.
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        shared_encoder: bool = False,
        feature_dim: int = 512,
        dropout: float = 0.3,
        num_targets: int = 7,
    ):
        super().__init__()
        self.shared_encoder = shared_encoder

        # Determine the true backbone feature dim from the backbone name.
        _, backbone_dim = MolecularVisionEncoder._BACKBONES[backbone]

        # Build encoder(s) -- we set num_targets=1 as a dummy; we won't use
        # the single-image projection head.
        self.encoder_cosmo = MolecularVisionEncoder(
            backbone=backbone,
            pretrained=pretrained,
            feature_dim=feature_dim,
            freeze_layers=0,
            dropout=0.0,
            num_targets=1,
        )

        if shared_encoder:
            self.encoder_ep = self.encoder_cosmo
        else:
            self.encoder_ep = MolecularVisionEncoder(
                backbone=backbone,
                pretrained=pretrained,
                feature_dim=feature_dim,
                freeze_layers=0,
                dropout=0.0,
                num_targets=1,
            )

        # Fusion head: concatenated features -> prediction
        fused_dim = backbone_dim * 2
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim // 2, num_targets),
        )

    def get_features(
        self, cosmo_image: torch.Tensor, ep_image: torch.Tensor
    ) -> torch.Tensor:
        """Return fused features from both images.

        Parameters
        ----------
        cosmo_image : Tensor
            COSMO surface image batch ``(B, 3, 224, 224)``.
        ep_image : Tensor
            Electrostatic potential image batch ``(B, 3, 224, 224)``.

        Returns
        -------
        Tensor
            Concatenated feature vector ``(B, 2 * backbone_dim)``.
        """
        feat_cosmo = self.encoder_cosmo.get_features(cosmo_image)
        feat_ep = self.encoder_ep.get_features(ep_image)
        return torch.cat([feat_cosmo, feat_ep], dim=1)

    def forward(
        self, cosmo_image: torch.Tensor, ep_image: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Full forward: dual features -> dropout -> projection.

        Parameters
        ----------
        cosmo_image : Tensor
            COSMO surface image batch ``(B, 3, 224, 224)``.
        ep_image : Tensor
            Electrostatic potential image batch ``(B, 3, 224, 224)``.

        Returns
        -------
        Tensor
            Predictions of shape ``(B, num_targets)``.
        """
        fused = self.get_features(cosmo_image, ep_image)
        fused = self.dropout(fused)
        return self.projection(fused)
