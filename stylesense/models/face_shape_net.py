"""EfficientNet image backbone fused with landmark-geometry features (PMP 10.1)."""

from __future__ import annotations

import numpy as np
import timm
import torch
from torch import nn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def backbone_output_dim(model: nn.Module) -> int:
    """Width of a timm model's pooled output with num_classes=0.

    Some models (e.g. MobileNetV3) have a head projection, so ``num_features`` (960) differs from the actual
    output (1280); ``head_hidden_size`` gives the right value when present.
    """
    return int(getattr(model, "head_hidden_size", None) or model.num_features)


class FaceShapeNet(nn.Module):
    def __init__(
        self,
        n_classes: int = 5,
        n_geom: int = 7,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
        geom_hidden: int = 32,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        d = backbone_output_dim(self.backbone)
        self.geom = nn.Sequential(nn.Linear(n_geom, geom_hidden), nn.LayerNorm(geom_hidden), nn.GELU())
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d + geom_hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, image: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.backbone(image), self.geom(geometry)], dim=1))

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable
