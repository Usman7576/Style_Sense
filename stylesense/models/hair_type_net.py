"""Shared MobileNetV3 backbone with one classification head per hair/beard attribute (PMP 10.3)."""

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn

from stylesense.models.face_shape_net import backbone_output_dim


class HairTypeNet(nn.Module):
    def __init__(
        self,
        heads: dict[str, list[str]],
        backbone: str = "mobilenetv3_large_100",
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        d = backbone_output_dim(self.backbone)
        self.heads = nn.ModuleDict(
            {name: nn.Sequential(nn.Dropout(dropout), nn.Linear(d, len(cls))) for name, cls in heads.items()}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.backbone(x)
        return {name: head(z) for name, head in self.heads.items()}


def multi_head_loss(
    logits: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], label_smoothing: float = 0.0
) -> torch.Tensor:
    """Mean cross-entropy over heads that have at least one label in the batch (missing labels are -1)."""
    losses = []
    for name, out in logits.items():
        y = targets[name]
        if (y >= 0).any():
            losses.append(F.cross_entropy(out, y, ignore_index=-1, label_smoothing=label_smoothing))
    if not losses:
        return next(iter(logits.values())).sum() * 0.0
    return torch.stack(losses).mean()
