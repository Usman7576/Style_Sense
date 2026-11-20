import numpy as np
import pytest

from stylesense.metrics import (
    accuracy,
    bootstrap_ci,
    classification_report,
    confusion_matrix,
    macro_f1,
    mst_band,
    per_group,
)


def test_mst_band():
    assert [mst_band(m) for m in (1, 3, 4, 6, 7, 10)] == ["1-3", "1-3", "4-6", "4-6", "7-10", "7-10"]
    assert mst_band(None) is None and mst_band(float("nan")) is None
    with pytest.raises(ValueError):
        mst_band(11)


def test_confusion_and_f1():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    assert confusion_matrix(y_true, y_pred, 3).tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert accuracy(y_true, y_pred) == pytest.approx(4 / 6)
    # per-class F1: 0.5, 0.8, 0.6667
    assert macro_f1(y_true, y_pred, 3) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_macro_f1_ignores_absent_classes():
    assert macro_f1([0, 0, 1], [0, 0, 1], n_classes=5) == 1.0


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    p = np.where(rng.random(300) < 0.8, y, (y + 1) % 3)
    lo, hi = bootstrap_ci(accuracy, y, p, n_boot=300)
    assert lo <= accuracy(y, p) <= hi and hi - lo < 0.15


def test_per_group_gap_respects_min_count():
    y = np.array([0] * 40 + [0] * 40 + [0] * 5)
    p = np.array([0] * 40 + [0] * 20 + [1] * 20 + [1] * 5)
    g = ["a"] * 40 + ["b"] * 40 + ["c"] * 5
    out = per_group(accuracy, y, p, g, min_count=20)
    assert out["groups"]["a"]["value"] == 1.0 and out["groups"]["b"]["value"] == 0.5
    assert out["groups"]["c"]["n"] == 5
    assert out["gap"] == pytest.approx(0.5)  # group c (n=5) excluded from the gap


def test_classification_report_shape():
    rep = classification_report([0, 1, 2, 1], [0, 1, 1, 1], ["a", "b", "c"], groups=["x", "x", "y", "y"], n_boot=50)
    assert rep["n"] == 4 and set(rep["per_class"]) == {"a", "b", "c"}
    assert len(rep["confusion_matrix"]) == 3 and "per_group_macro_f1" in rep
