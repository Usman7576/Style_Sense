"""Image loading helpers. Images are handled as RGB uint8 numpy arrays throughout the project."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_rgb(path: str | Path, max_side: int | None = 1600) -> np.ndarray:
    """Load an image as RGB uint8, applying EXIF rotation and optionally capping the longest side."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max_side and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.asarray(im).copy()


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Validate and normalise an input array to HxWx3 uint8 RGB."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"expected an HxWx3 RGB image, got shape {image.shape}")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)
