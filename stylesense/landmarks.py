"""Face landmarks, head pose, alignment and geometric face features (FR-3).

Uses the MediaPipe Face Landmarker task (478 landmarks = 468 face mesh points + 10 iris points). The legacy
MediaPipe Face Mesh solution is deprecated and is not used.

Coordinate convention: landmark arrays are (N, 3) float32 in *pixel* units of the source image; z uses the same
scale as x (MediaPipe normalises z by image width).
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from stylesense.utils.config import load_config, repo_path
from stylesense.utils.image import ensure_rgb_uint8

NUM_LANDMARKS = 478

# Landmark indices on the MediaPipe 478-point topology.
IDX = {
    "forehead_top": 10,  # upper forehead on the mesh (not the true hairline)
    "chin": 152,
    "cheek_left": 234,
    "cheek_right": 454,
    "jaw_left": 172,
    "jaw_right": 397,
    "forehead_left": 54,
    "forehead_right": 284,
    "nose_tip": 1,
    "subnasale": 2,
    "mouth_left": 61,
    "mouth_right": 291,
    "iris_a": 468,
    "iris_b": 473,
}

# Canonical 5-point template for alignment, as fractions of the output crop size. Chosen so the face occupies
# roughly the middle half of the crop, leaving room for hair above and a beard below.
ALIGN_TEMPLATE = np.array(
    [
        [0.385, 0.46],  # eye on the image-left
        [0.615, 0.46],  # eye on the image-right
        [0.500, 0.57],  # nose tip
        [0.410, 0.67],  # mouth corner image-left
        [0.590, 0.67],  # mouth corner image-right
    ],
    dtype=np.float32,
)

FEATURE_NAMES = [
    "length_to_width",
    "jaw_to_cheek",
    "forehead_to_cheek",
    "jaw_to_forehead",
    "lower_third_ratio",
    "chin_angle_deg",
    "jaw_angle_deg",
    "face_width_px",
]


@dataclass
class HeadPose:
    yaw: float
    pitch: float
    roll: float


@dataclass
class FaceDetection:
    landmarks: np.ndarray  # (478, 3) pixel coordinates
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (pixels, clipped to image)
    pose: HeadPose | None = None
    extra: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


def pose_from_matrix(matrix: np.ndarray) -> HeadPose:
    """Euler angles (degrees) from MediaPipe's 4x4 facial transformation matrix.

    Only magnitudes matter for quality gating; signs follow a right-handed camera frame.
    """
    r = np.asarray(matrix, dtype=np.float64)[:3, :3]
    # Remove any uniform scale so r is a pure rotation.
    r = r / np.cbrt(max(abs(np.linalg.det(r)), 1e-12))
    sy = math.hypot(r[0, 0], r[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:  # gimbal lock - not expected for usable portraits
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    return HeadPose(*(math.degrees(a) for a in (yaw, pitch, roll)))


def _bbox(points: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0 = np.floor(points[:, :2].min(0)).astype(int)
    x1, y1 = np.ceil(points[:, :2].max(0)).astype(int)
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def ensure_model(model_path: str | Path, url: str) -> Path:
    """Download the Face Landmarker model to ``model_path`` if it is missing."""
    path = repo_path(model_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)  # official Google storage URL from configs/landmarks.yaml
    return path


class FaceAnalyzer:
    """Thin wrapper around the MediaPipe Face Landmarker task (image mode)."""

    def __init__(self, config: dict | None = None, download: bool = True):
        from mediapipe.tasks.python import BaseOptions, vision  # imported lazily: heavy import

        self.cfg = config or load_config("landmarks")
        model_path = repo_path(self.cfg["model_path"])
        if not model_path.exists():
            if not download:
                raise FileNotFoundError(f"Face Landmarker model not found at {model_path}")
            ensure_model(model_path, self.cfg["model_url"])
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=int(self.cfg.get("num_faces", 2)),
            min_face_detection_confidence=float(self.cfg.get("min_detection_confidence", 0.5)),
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceAnalyzer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def detect(self, image_rgb: np.ndarray) -> list[FaceDetection]:
        """Detect faces; returns detections sorted by face area, largest first."""
        import mediapipe as mp

        image_rgb = ensure_rgb_uint8(image_rgb)
        h, w = image_rgb.shape[:2]
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb))
        detections = []
        for i, lms in enumerate(result.face_landmarks):
            pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lms], dtype=np.float32)
            pose = None
            if result.facial_transformation_matrixes and i < len(result.facial_transformation_matrixes):
                pose = pose_from_matrix(np.array(result.facial_transformation_matrixes[i]))
            detections.append(FaceDetection(landmarks=pts, bbox=_bbox(pts, w, h), pose=pose))
        return sorted(detections, key=lambda d: d.area, reverse=True)


def _eye_mouth_points(landmarks: np.ndarray) -> np.ndarray:
    """Five 2D points ordered to match ALIGN_TEMPLATE (left/right taken in image space)."""
    eyes = landmarks[[IDX["iris_a"], IDX["iris_b"]], :2]
    eyes = eyes[np.argsort(eyes[:, 0])]
    mouth = landmarks[[IDX["mouth_left"], IDX["mouth_right"]], :2]
    mouth = mouth[np.argsort(mouth[:, 0])]
    nose = landmarks[IDX["nose_tip"], :2][None]
    return np.vstack([eyes, nose, mouth]).astype(np.float32)


def align_face(
    image_rgb: np.ndarray, landmarks: np.ndarray, size: int = 512
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Similarity-align a face to the canonical template.

    Returns (aligned_crop HxWx3, 2x3 affine matrix source->crop, landmarks mapped into crop coordinates).
    """
    src = _eye_mouth_points(landmarks)
    dst = ALIGN_TEMPLATE * size
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("could not estimate alignment transform")
    crop = cv2.warpAffine(
        ensure_rgb_uint8(image_rgb), matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    pts2 = np.hstack([landmarks[:, :2], np.ones((len(landmarks), 1), np.float32)]) @ matrix.T
    scale = math.hypot(matrix[0, 0], matrix[0, 1])
    mapped = np.hstack([pts2, landmarks[:, 2:3] * scale]).astype(np.float32)
    return crop, matrix.astype(np.float32), mapped


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _angle(at: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    v1, v2 = p - at, q - at
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def face_features(landmarks: np.ndarray) -> dict[str, float]:
    """Geometric face-shape features from 478 landmarks (pixel units; scale-invariant except face_width_px)."""
    p = {k: landmarks[i] for k, i in IDX.items()}
    length = _dist(p["forehead_top"], p["chin"])
    cheek = _dist(p["cheek_left"], p["cheek_right"])
    jaw = _dist(p["jaw_left"], p["jaw_right"])
    forehead = _dist(p["forehead_left"], p["forehead_right"])
    jaw_angles = [
        _angle(p["jaw_left"], p["cheek_left"], p["chin"]),
        _angle(p["jaw_right"], p["cheek_right"], p["chin"]),
    ]
    return {
        "length_to_width": length / cheek,
        "jaw_to_cheek": jaw / cheek,
        "forehead_to_cheek": forehead / cheek,
        "jaw_to_forehead": jaw / forehead,
        "lower_third_ratio": _dist(p["subnasale"], p["chin"]) / length,
        "chin_angle_deg": _angle(p["chin"], p["jaw_left"], p["jaw_right"]),
        "jaw_angle_deg": float(np.mean(jaw_angles)),
        "face_width_px": cheek,
    }
