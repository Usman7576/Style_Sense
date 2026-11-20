"""Face-shape classification (FR-4): learned classes, diamond rule, rule-based baseline and inference wrapper.

Learned classes follow the Kaggle face-shape dataset (heart, oblong, oval, round, square). That dataset has no
diamond class, so diamond is derived from landmark geometry (cheekbones wider than both forehead and jaw) and mixed
into the probability distribution.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np

from stylesense.metrics import macro_f1
from stylesense.utils.config import load_config

CLASSES = ["heart", "oblong", "oval", "round", "square"]
ALL_SHAPES = [*CLASSES, "diamond"]
# Scale-invariant geometric features fed to the model (face_width_px is excluded: it depends on image size).
GEOMETRY_FEATURES = [
    "length_to_width",
    "jaw_to_cheek",
    "forehead_to_cheek",
    "jaw_to_forehead",
    "lower_third_ratio",
    "chin_angle_deg",
    "jaw_angle_deg",
]


def normalize_label(label: str) -> str | None:
    key = label.strip().lower().replace(" ", "_")
    return key if key in ALL_SHAPES else None


def diamond_probability(features: Mapping[str, float], cfg: Mapping | None = None) -> float:
    """Soft probability that a face is diamond-shaped from forehead/jaw-to-cheekbone ratios."""
    cfg = cfg or load_config("face_shape")["diamond"]
    margin = min(
        cfg["max_forehead_to_cheek"] - features["forehead_to_cheek"],
        cfg["max_jaw_to_cheek"] - features["jaw_to_cheek"],
    )
    return 1.0 / (1.0 + math.exp(-margin / cfg["softness"]))


def add_diamond(probs: Mapping[str, float], features: Mapping[str, float], cfg: Mapping | None = None) -> dict:
    """Mix the diamond rule into a 5-class distribution, returning a 6-class distribution that sums to 1."""
    p_d = diamond_probability(features, cfg)
    out = {k: float(v) * (1.0 - p_d) for k, v in probs.items()}
    out["diamond"] = p_d
    return out


class RuleClassifier:
    """Interpretable threshold baseline on landmark ratios. ``fit`` grid-searches thresholds for macro-F1."""

    GRID = {
        "oblong_min_lw": [1.25, 1.30, 1.35, 1.40],
        "round_max_lw": [1.05, 1.10, 1.15, 1.20],
        "heart_min_forehead": [0.86, 0.90, 0.94],
        "heart_max_jaw": [0.78, 0.82, 0.86],
        "square_min_jaw": [0.86, 0.90, 0.94],
    }

    def __init__(self, **thresholds: float):
        self.t = {k: v[len(v) // 2] for k, v in self.GRID.items()} | thresholds

    def predict_one(self, f: Mapping[str, float]) -> str:
        t = self.t
        if f["length_to_width"] >= t["oblong_min_lw"]:
            return "oblong"
        if f["forehead_to_cheek"] >= t["heart_min_forehead"] and f["jaw_to_cheek"] <= t["heart_max_jaw"]:
            return "heart"
        if f["jaw_to_cheek"] >= t["square_min_jaw"]:
            return "square"
        if f["length_to_width"] <= t["round_max_lw"]:
            return "round"
        return "oval"

    def predict(self, rows: Sequence[Mapping[str, float]]) -> list[str]:
        return [self.predict_one(r) for r in rows]

    def fit(self, rows: Sequence[Mapping[str, float]], labels: Sequence[str]) -> RuleClassifier:
        y = np.array([CLASSES.index(lbl) for lbl in labels])
        keys = list(self.GRID)
        best, best_score = self.t, -1.0
        for combo in itertools.product(*(self.GRID[k] for k in keys)):
            self.t = dict(zip(keys, combo, strict=True))
            pred = np.array([CLASSES.index(p) for p in self.predict(rows)])
            score = macro_f1(y, pred, len(CLASSES))
            if score > best_score:
                best, best_score = dict(self.t), score
        self.t = best
        return self


def crop_for_classifier(aligned: np.ndarray, box: Sequence[float], size: int) -> np.ndarray:
    """Cut the face region out of the 512 aligned crop and resize to the classifier input size."""
    h, w = aligned.shape[:2]
    x0, y0, x1, y1 = (int(round(v * s)) for v, s in zip(box, (w, h, w, h), strict=True))
    return cv2.resize(aligned[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


class FaceShapePredictor:
    """Inference wrapper: aligned crop + geometric features -> 6-class probability distribution."""

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        import torch

        from stylesense.models.face_shape_net import FaceShapeNet

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = ckpt["config"]
        self.classes = ckpt["classes"]
        self.geom_mean = np.asarray(ckpt["geom_mean"], np.float32)
        self.geom_std = np.asarray(ckpt["geom_std"], np.float32)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FaceShapeNet(n_classes=len(self.classes), pretrained=False, **self.cfg["model_kwargs"])
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

    def predict(self, aligned: np.ndarray, features: Mapping[str, float], with_diamond: bool = True) -> dict:
        import torch

        from stylesense.models.face_shape_net import IMAGENET_MEAN, IMAGENET_STD

        img = crop_for_classifier(aligned, self.cfg["crop_box"], self.cfg["image_size"]).astype(np.float32) / 255
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(img.transpose(2, 0, 1))[None].to(self.device)
        g = (np.array([features[k] for k in GEOMETRY_FEATURES], np.float32) - self.geom_mean) / self.geom_std
        with torch.no_grad():
            p = torch.softmax(self.model(x, torch.from_numpy(g)[None].to(self.device)), 1)[0].cpu().numpy()
        probs = dict(zip(self.classes, p.tolist(), strict=True))
        return add_diamond(probs, features, self.cfg.get("diamond")) if with_diamond else probs
