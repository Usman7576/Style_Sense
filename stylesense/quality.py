"""Photo quality checks (FR-2): reject photos that are blurry, too dark/bright, too small, multi-face or turned."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from stylesense.landmarks import FaceDetection
from stylesense.utils.config import load_config
from stylesense.utils.image import ensure_rgb_uint8

MESSAGES = {
    "no_face": "No face was found. Please use a clear, front-facing photo.",
    "multiple_faces": "More than one face is visible. Please use a photo with only you in it.",
    "face_too_small": "Your face is too small in the photo. Move closer to the camera.",
    "too_blurry": "The photo is too blurry. Hold the camera steady and make sure your face is in focus.",
    "too_dark": "The photo is too dark. Face a window or a light source.",
    "too_bright": "The photo is over-exposed. Avoid strong direct light on your face.",
    "head_turned": "Your head is turned to the side. Please look straight at the camera.",
    "head_tilted_vertical": "Your head is tilted up or down. Keep your chin level.",
    "head_tilted_sideways": "Your head is tilted sideways. Keep your head upright.",
}

# A second face smaller than this fraction of the main face (e.g. someone in the background) is tolerated.
SECONDARY_FACE_AREA_FRAC = 0.30


@dataclass
class QualityConfig:
    blur_min_laplacian_var: float = 45.0
    brightness_min: float = 60.0
    brightness_max: float = 235.0
    min_face_px: int = 160
    min_face_frac: float = 0.20
    max_yaw_deg: float = 20.0
    max_pitch_deg: float = 22.0
    max_roll_deg: float = 15.0

    @classmethod
    def load(cls, name: str = "quality") -> QualityConfig:
        return cls(**load_config(name))


@dataclass
class QualityReport:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def messages(self) -> list[str]:
        return [MESSAGES[r] for r in self.reasons]

    def to_dict(self) -> dict:
        return {**asdict(self), "messages": self.messages}


def blur_score(image_rgb: np.ndarray, width: int = 256, target_std: float = 50.0) -> float:
    """Variance of the Laplacian (higher = sharper), measured at a fixed width and contrast.

    The grey image is rescaled to a fixed standard deviation first, so a dark or low-contrast but sharp photo is
    not mistaken for a blurry one (darkness is checked separately by ``brightness``).
    """
    gray = cv2.cvtColor(ensure_rgb_uint8(image_rgb), cv2.COLOR_RGB2GRAY).astype(np.float64)
    h, w = gray.shape
    if w != width:
        gray = cv2.resize(gray, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    std = gray.std()
    if std > 1e-6:
        gray = (gray - gray.mean()) * (target_std / std)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(image_rgb: np.ndarray) -> float:
    """Mean of the HSV value channel (0-255)."""
    hsv = cv2.cvtColor(ensure_rgb_uint8(image_rgb), cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 2].mean())


def face_crop(image_rgb: np.ndarray, det: FaceDetection, margin: float = 0.0) -> np.ndarray:
    x0, y0, x1, y1 = det.bbox
    if margin:
        mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
        h, w = image_rgb.shape[:2]
        x0, y0, x1, y1 = max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)
    return image_rgb[y0:y1, x0:x1]


def check_quality(
    image_rgb: np.ndarray, detections: list[FaceDetection], cfg: QualityConfig | None = None
) -> QualityReport:
    """Evaluate a photo against the quality thresholds. ``detections`` come from FaceAnalyzer.detect."""
    cfg = cfg or QualityConfig.load()
    image_rgb = ensure_rgb_uint8(image_rgb)
    if not detections:
        return QualityReport(ok=False, reasons=["no_face"])

    reasons: list[str] = []
    main = detections[0]
    others = [d for d in detections[1:] if d.area >= SECONDARY_FACE_AREA_FRAC * main.area]
    if others:
        reasons.append("multiple_faces")

    h = image_rgb.shape[0]
    metrics: dict[str, float] = {"face_px": float(main.height), "face_frac": main.height / h}
    if main.height < cfg.min_face_px or main.height / h < cfg.min_face_frac:
        reasons.append("face_too_small")

    crop = face_crop(image_rgb, main)
    if crop.size:
        metrics["blur"] = blur_score(crop)
        metrics["brightness"] = brightness(crop)
        if metrics["blur"] < cfg.blur_min_laplacian_var:
            reasons.append("too_blurry")
        if metrics["brightness"] < cfg.brightness_min:
            reasons.append("too_dark")
        elif metrics["brightness"] > cfg.brightness_max:
            reasons.append("too_bright")

    if main.pose is not None:
        metrics.update(yaw=main.pose.yaw, pitch=main.pose.pitch, roll=main.pose.roll)
        if abs(main.pose.yaw) > cfg.max_yaw_deg:
            reasons.append("head_turned")
        if abs(main.pose.pitch) > cfg.max_pitch_deg:
            reasons.append("head_tilted_vertical")
        if abs(main.pose.roll) > cfg.max_roll_deg:
            reasons.append("head_tilted_sideways")

    return QualityReport(ok=not reasons, reasons=reasons, metrics=metrics)
