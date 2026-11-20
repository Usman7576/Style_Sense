import cv2
import numpy as np
import pytest

from stylesense.landmarks import (
    ALIGN_TEMPLATE,
    FEATURE_NAMES,
    NUM_LANDMARKS,
    align_face,
    face_features,
    pose_from_matrix,
)
from stylesense.quality import check_quality


def test_pose_identity_matrix_is_zero():
    pose = pose_from_matrix(np.eye(4))
    assert abs(pose.yaw) < 1e-6 and abs(pose.pitch) < 1e-6 and abs(pose.roll) < 1e-6


def test_pose_roll_recovered_and_scale_ignored():
    a = np.radians(25)
    m = np.eye(4)
    m[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    m[:3, :3] *= 3.0  # uniform scale must not affect angles
    assert pose_from_matrix(m).roll == pytest.approx(25, abs=1e-6)


@pytest.mark.needs_model
def test_detect_portrait(analyzer, portrait):
    dets = analyzer.detect(portrait)
    assert len(dets) == 1
    d = dets[0]
    assert d.landmarks.shape == (NUM_LANDMARKS, 3)
    assert d.pose is not None and abs(d.pose.yaw) < 15


@pytest.mark.needs_model
def test_portrait_passes_quality(analyzer, portrait, portrait_cfg):
    rep = check_quality(portrait, analyzer.detect(portrait), portrait_cfg)
    assert rep.ok, rep.to_dict()


@pytest.mark.needs_model
def test_blurred_portrait_rejected(analyzer, portrait, portrait_cfg):
    blurred = cv2.GaussianBlur(portrait, (0, 0), 3)
    assert "too_blurry" in check_quality(blurred, analyzer.detect(blurred), portrait_cfg).reasons


@pytest.mark.needs_model
def test_alignment_maps_keypoints_to_template(analyzer, portrait):
    d = analyzer.detect(portrait)[0]
    crop, matrix, mapped = align_face(portrait, d.landmarks, size=512)
    assert crop.shape == (512, 512, 3) and matrix.shape == (2, 3) and mapped.shape == (NUM_LANDMARKS, 3)
    eyes = np.sort(mapped[[468, 473], 0])
    assert eyes == pytest.approx(ALIGN_TEMPLATE[:2, 0] * 512, abs=12)


@pytest.mark.needs_model
def test_features_plausible_and_scale_invariant(analyzer, portrait):
    double = cv2.resize(portrait, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    f1 = face_features(analyzer.detect(double)[0].landmarks)
    f2 = face_features(analyzer.detect(portrait)[0].landmarks)
    assert list(f1) == FEATURE_NAMES
    assert 1.0 < f1["length_to_width"] < 2.0
    assert 0.5 < f1["jaw_to_cheek"] < 1.0
    assert f1["length_to_width"] == pytest.approx(f2["length_to_width"], rel=0.03)
    assert f1["face_width_px"] == pytest.approx(2 * f2["face_width_px"], rel=0.05)


@pytest.mark.needs_model
def test_rotated_photo_flagged(analyzer, portrait, portrait_cfg):
    rot = cv2.warpAffine(portrait, cv2.getRotationMatrix2D((256, 256), 30, 1.0), (512, 512))
    rep = check_quality(rot, analyzer.detect(rot), portrait_cfg)
    assert "head_tilted_sideways" in rep.reasons
