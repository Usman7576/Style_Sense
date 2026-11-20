import numpy as np
import pytest

from stylesense.face_shape import (
    ALL_SHAPES,
    CLASSES,
    RuleClassifier,
    add_diamond,
    crop_for_classifier,
    diamond_probability,
    normalize_label,
)

CFG = {"max_forehead_to_cheek": 0.80, "max_jaw_to_cheek": 0.78, "softness": 0.02}


def test_normalize_label():
    assert normalize_label(" Oval ") == "oval" and normalize_label("Diamond") == "diamond"
    assert normalize_label("triangle") is None


def test_diamond_probability_direction():
    narrow = {"forehead_to_cheek": 0.70, "jaw_to_cheek": 0.68}
    wide = {"forehead_to_cheek": 0.92, "jaw_to_cheek": 0.90}
    assert diamond_probability(narrow, CFG) > 0.95 and diamond_probability(wide, CFG) < 0.01


def test_add_diamond_sums_to_one():
    probs = dict(zip(CLASSES, [0.1, 0.2, 0.4, 0.2, 0.1], strict=True))
    out = add_diamond(probs, {"forehead_to_cheek": 0.80, "jaw_to_cheek": 0.78}, CFG)
    assert list(out) == ALL_SHAPES and sum(out.values()) == pytest.approx(1.0)
    assert out["diamond"] == pytest.approx(0.5)


def _synthetic(n_per=60, seed=0):
    rng = np.random.default_rng(seed)
    proto = {
        "oblong": (1.45, 0.85, 0.88),
        "round": (1.00, 0.88, 0.88),
        "heart": (1.20, 0.74, 0.97),
        "square": (1.20, 0.97, 0.90),
        "oval": (1.22, 0.84, 0.84),
    }
    rows, labels = [], []
    for lbl, (lw, jaw, fh) in proto.items():
        for _ in range(n_per):
            rows.append(
                {
                    "length_to_width": lw + rng.normal(0, 0.02),
                    "jaw_to_cheek": jaw + rng.normal(0, 0.01),
                    "forehead_to_cheek": fh + rng.normal(0, 0.01),
                }
            )
            labels.append(lbl)
    return rows, labels


def test_rule_classifier_fit_improves_accuracy():
    rows, labels = _synthetic()
    base = np.mean(np.array(RuleClassifier().predict(rows)) == np.array(labels))
    fitted = RuleClassifier().fit(rows, labels)
    acc = np.mean(np.array(fitted.predict(rows)) == np.array(labels))
    assert acc >= base and acc > 0.85


def test_crop_for_classifier_shape():
    out = crop_for_classifier(np.zeros((512, 512, 3), np.uint8), [0.12, 0.16, 0.88, 0.92], 224)
    assert out.shape == (224, 224, 3)


def test_face_shape_net_forward():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from stylesense.models.face_shape_net import FaceShapeNet

    net = FaceShapeNet(pretrained=False).eval()
    with torch.no_grad():
        out = net(torch.randn(2, 3, 224, 224), torch.randn(2, 7))
    assert out.shape == (2, 5)
    net.set_backbone_trainable(False)
    assert not any(p.requires_grad for p in net.backbone.parameters())
