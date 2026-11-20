"""Evaluation metrics with confidence intervals and per-group (fairness) breakdowns.

Pure numpy so it runs anywhere (CI, Kaggle, local CPU). Used by every training and evaluation script.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

MST_BANDS = ("1-3", "4-6", "7-10")


def mst_band(mst: int | float | None) -> str | None:
    """Map a Monk Skin Tone value (1-10) to a reporting band, or None if missing."""
    if mst is None or (isinstance(mst, float) and np.isnan(mst)):
        return None
    m = int(mst)
    if not 1 <= m <= 10:
        raise ValueError(f"MST must be 1-10, got {mst}")
    return "1-3" if m <= 3 else "4-6" if m <= 6 else "7-10"


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def per_class_prf(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int | None = None) -> float:
    """Macro-F1 over classes present in y_true (absent classes would otherwise count as 0)."""
    y_true = np.asarray(y_true)
    if not len(y_true):
        return float("nan")
    n = n_classes or int(max(y_true.max(), np.max(y_pred)) + 1)
    _, _, f1 = per_class_prf(confusion_matrix(y_true, y_pred, n))
    present = np.bincount(y_true, minlength=n) > 0
    return float(f1[present].mean())


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: Sequence,
    y_pred: Sequence,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for any metric(y_true, y_pred)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.array([metric(y_true[idx], y_pred[idx]) for idx in rng.integers(0, n, (n_boot, n))])
    stats = stats[~np.isnan(stats)]
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def per_group(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: Sequence,
    y_pred: Sequence,
    groups: Sequence,
    min_count: int = 20,
) -> dict:
    """Metric per group plus the worst-group gap (best minus worst among groups with >= min_count samples).

    Groups below ``min_count`` are reported with their count but excluded from the gap, per the evaluation plan.
    """
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups, dtype=object)
    out: dict = {"groups": {}}
    for g in sorted({x for x in groups if x is not None}, key=str):
        m = groups == g
        out["groups"][str(g)] = {"n": int(m.sum()), "value": metric(y_true[m], y_pred[m])}
    eligible = [v["value"] for v in out["groups"].values() if v["n"] >= min_count and not np.isnan(v["value"])]
    out["gap"] = float(max(eligible) - min(eligible)) if len(eligible) >= 2 else None
    return out


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    groups: Sequence | None = None,
    n_boot: int = 1000,
) -> dict:
    """Full report: accuracy and macro-F1 with 95% CIs, per-class P/R/F1, confusion matrix, per-group breakdown."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    k = len(class_names)
    cm = confusion_matrix(y_true, y_pred, k)
    p, r, f = per_class_prf(cm)

    def mf1(a, b):
        return macro_f1(a, b, k)

    report = {
        "n": int(len(y_true)),
        "accuracy": accuracy(y_true, y_pred),
        "accuracy_ci95": bootstrap_ci(accuracy, y_true, y_pred, n_boot),
        "macro_f1": mf1(y_true, y_pred),
        "macro_f1_ci95": bootstrap_ci(mf1, y_true, y_pred, n_boot),
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(cm[i].sum())}
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }
    if groups is not None:
        report["per_group_accuracy"] = per_group(accuracy, y_true, y_pred, groups)
        report["per_group_macro_f1"] = per_group(mf1, y_true, y_pred, groups)
    return report
