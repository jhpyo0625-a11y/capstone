"""Per-class metrics, fail-recall @ threshold (spec §6.4).

Core functions used by the trainer for early stopping; Phase 4 adds the
fail-recall vs false-reject curve, threshold selection, and per-run breakdown.

Convention: `probs` is (N, n_classes) softmax output in the order of
config data.classes; `fail_indices` are the defect class indices (Dent, Loose).
An image is predicted "fail" when P(fail) = sum of defect probs >= threshold.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def fail_scores(probs: np.ndarray, fail_indices: list[int]) -> np.ndarray:
    return probs[:, fail_indices].sum(axis=1)


def fail_recall(y_true: np.ndarray, probs: np.ndarray, fail_indices: list[int], threshold: float = 0.5) -> float:
    """Fraction of true defects (any fail class) that are predicted fail."""
    is_fail = np.isin(y_true, fail_indices)
    if not is_fail.any():
        return float("nan")
    predicted_fail = fail_scores(probs, fail_indices) >= threshold
    return float(predicted_fail[is_fail].mean())


def false_reject_rate(y_true: np.ndarray, probs: np.ndarray, fail_indices: list[int], threshold: float = 0.5) -> float:
    """Fraction of true Pass images that are predicted fail."""
    is_pass = ~np.isin(y_true, fail_indices)
    if not is_pass.any():
        return float("nan")
    predicted_fail = fail_scores(probs, fail_indices) >= threshold
    return float(predicted_fail[is_pass].mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[float]:
    out = []
    for c in range(n_classes):
        mask = y_true == c
        out.append(float((y_pred[mask] == c).mean()) if mask.any() else float("nan"))
    return out
