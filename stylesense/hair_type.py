"""Hair and beard attributes (FR-6): curl pattern, density, hairline and beard coverage.

Used as a feasibility filter by the recommender (e.g. patchy beard growth down-ranks a full beard). Training labels
are partial - each source labels only some heads - so missing labels are encoded as -1 and ignored by the loss.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

HEADS: dict[str, list[str]] = {
    "curl": ["straight", "wavy", "curly", "coily"],
    "density": ["normal", "thinning"],
    "hairline": ["normal", "receding", "bald"],
    "beard": ["none", "patchy", "full"],
}
MISSING = -1


def encode_labels(row: Mapping[str, object], heads: Mapping[str, list[str]] = HEADS) -> dict[str, int]:
    """Map a row of string labels to class indices; empty / unknown values become MISSING (-1)."""
    out = {}
    for head, classes in heads.items():
        v = row.get(head)
        v = v.strip().lower() if isinstance(v, str) else None
        out[head] = classes.index(v) if v in classes else MISSING
    return out


# Figaro-1k: 1,050 images, 150 per class, numbered in class order. Verify against the downloaded release.
FIGARO_CLASSES = ["straight", "wavy", "curly", "kinky", "braids", "dreadlocks", "short-men"]
FIGARO_TO_CURL = {"straight": "straight", "wavy": "wavy", "curly": "curly", "kinky": "coily"}


def figaro_class(frame_number: int) -> str:
    if not 1 <= frame_number <= 1050:
        raise ValueError(f"Figaro-1k frame number out of range: {frame_number}")
    return FIGARO_CLASSES[(frame_number - 1) // 150]


def celeba_labels(attrs: Mapping[str, int]) -> dict[str, str | None]:
    """Weak labels from CelebA binary attributes (+1 / -1). Only heads CelebA can inform are filled."""

    def on(name: str) -> bool:
        return attrs.get(name, -1) == 1

    curl = None
    if on("Straight_Hair") != on("Wavy_Hair"):
        curl = "straight" if on("Straight_Hair") else "wavy"
    hairline = "bald" if on("Bald") else "receding" if on("Receding_Hairline") else "normal"
    if on("Wearing_Hat"):
        hairline = None
    beard = "none" if on("No_Beard") and not (on("Goatee") or on("Mustache") or on("5_o_Clock_Shadow")) else None
    return {"curl": curl, "density": None, "hairline": hairline, "beard": beard}


class HairTypePredictor:
    """Inference wrapper: 512 aligned crop -> {head: {class: prob}}."""

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        import torch

        from stylesense.models.hair_type_net import HairTypeNet

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.heads = ckpt["heads"]
        self.image_size = ckpt["image_size"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HairTypeNet(self.heads, pretrained=False, **ckpt["model_kwargs"])
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

    def predict(self, aligned: np.ndarray) -> dict[str, dict[str, float]]:
        import cv2
        import torch

        from stylesense.models.face_shape_net import IMAGENET_MEAN, IMAGENET_STD

        img = cv2.resize(aligned, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img = (img.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(img.transpose(2, 0, 1))[None].to(self.device)
        with torch.no_grad():
            logits = self.model(x)
        return {
            h: dict(zip(self.heads[h], torch.softmax(v, 1)[0].cpu().tolist(), strict=True)) for h, v in logits.items()
        }
