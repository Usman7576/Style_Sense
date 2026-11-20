import cv2
import numpy as np
import pytest

from stylesense.landmarks import FaceDetection, HeadPose
from stylesense.quality import MESSAGES, QualityConfig, blur_score, brightness, check_quality


def _textured(seed: int = 0, size: int = 400) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, (size // 8, size // 8, 3), dtype=np.uint8)
    img = cv2.resize(base, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.add(img, rng.integers(0, 25, img.shape, dtype=np.uint8))


FRONTAL = HeadPose(0.0, 5.0, 0.0)


def _fake_detection(bbox=(50, 50, 350, 350), pose=FRONTAL) -> FaceDetection:
    return FaceDetection(landmarks=np.zeros((478, 3), np.float32), bbox=bbox, pose=pose)


CFG = QualityConfig()


def test_blur_score_drops_when_blurred():
    img = _textured()
    assert blur_score(cv2.GaussianBlur(img, (0, 0), 4)) < blur_score(img) / 5


def test_blur_score_is_contrast_invariant():
    img = _textured()
    dark = (img * 0.2).astype(np.uint8)
    assert blur_score(dark) == pytest.approx(blur_score(img), rel=0.25)


def test_brightness_range():
    assert brightness(np.zeros((10, 10, 3), np.uint8)) == 0
    assert brightness(np.full((10, 10, 3), 255, np.uint8)) == 255


def test_no_face_rejected():
    rep = check_quality(_textured(), [], CFG)
    assert not rep.ok and rep.reasons == ["no_face"]


def test_good_photo_passes():
    rep = check_quality(_textured(), [_fake_detection()], CFG)
    assert rep.ok, rep.reasons


@pytest.mark.parametrize(
    "transform, reason",
    [
        (lambda im: cv2.GaussianBlur(im, (0, 0), 6), "too_blurry"),
        (lambda im: (im * 0.2).astype(np.uint8), "too_dark"),
        (lambda im: np.full_like(im, 250), "too_bright"),
    ],
)
def test_image_level_rejections(transform, reason):
    rep = check_quality(transform(_textured()), [_fake_detection()], CFG)
    assert reason in rep.reasons


def test_small_face_rejected():
    rep = check_quality(_textured(), [_fake_detection(bbox=(10, 10, 60, 60))], CFG)
    assert "face_too_small" in rep.reasons


def test_pose_rejections():
    rep = check_quality(_textured(), [_fake_detection(pose=HeadPose(35.0, 30.0, 25.0))], CFG)
    assert {"head_turned", "head_tilted_vertical", "head_tilted_sideways"} <= set(rep.reasons)


def test_multiple_faces_rejected_but_small_background_face_tolerated():
    main = _fake_detection()
    big_second = _fake_detection(bbox=(0, 0, 250, 250))
    tiny_second = _fake_detection(bbox=(0, 0, 40, 40))
    assert "multiple_faces" in check_quality(_textured(), [main, big_second], CFG).reasons
    assert "multiple_faces" not in check_quality(_textured(), [main, tiny_second], CFG).reasons


def test_every_reason_has_a_message():
    rep = check_quality(_textured(), [_fake_detection(bbox=(10, 10, 60, 60), pose=HeadPose(40, 40, 40))], CFG)
    assert all(r in MESSAGES for r in rep.reasons)
    assert len(rep.to_dict()["messages"]) == len(rep.reasons)
