"""Fitting and evaluating Degradation Analyzer thresholds on control videos with known conditions."""

from collections.abc import Sequence

import numpy as np

from resilens.config.schema import LevelThresholds

THRESHOLD_QUANTILES = np.linspace(0, 1, 101)


def predict_levels(scores: np.ndarray, thresholds: LevelThresholds) -> np.ndarray:
    """0 = absent, 1 = level 1, 2 = level 2."""
    return (scores >= thresholds.low).astype(int) + (scores >= thresholds.high).astype(int)


def balanced_accuracy(true: np.ndarray, predicted: np.ndarray) -> float:
    """Mean recall over the classes present in `true`: every class counts equally, whatever its size."""
    return float(np.mean([(predicted[true == label] == label).mean() for label in np.unique(true)]))


def confusion_matrix(true: Sequence, predicted: Sequence, rows: Sequence, columns: Sequence) -> np.ndarray:
    """Counts of samples with true label rows[i] and predicted label columns[j]."""
    matrix = np.zeros((len(rows), len(columns)), dtype=int)
    row_index = {label: i for i, label in enumerate(rows)}
    column_index = {label: j for j, label in enumerate(columns)}
    for t, p in zip(true, predicted, strict=True):
        matrix[row_index[t], column_index[p]] += 1
    return matrix


def centre_in_gap(scores: np.ndarray, threshold: float) -> float:
    """Move a threshold to the middle of the gap between the scores around it; predictions do not change."""
    below, above = scores[scores < threshold], scores[scores >= threshold]
    if not below.size or not above.size:
        return threshold
    return float((below.max() + above.min()) / 2)


def fit_thresholds(scores: np.ndarray, labels: np.ndarray) -> LevelThresholds:
    """Brute-force the (low, high) pair with the best balanced accuracy over score quantiles, then center it."""
    options = np.unique(np.quantile(scores, THRESHOLD_QUANTILES)).tolist()
    if len(options) < 2:
        raise ValueError("Cannot fit thresholds: all scores are equal")
    best_accuracy, best = -1.0, LevelThresholds(low=options[0], high=options[1])
    for i, low in enumerate(options[:-1]):
        for high in options[i + 1 :]:
            thresholds = LevelThresholds(low=low, high=high)
            accuracy = balanced_accuracy(labels, predict_levels(scores, thresholds))
            if accuracy > best_accuracy:
                best_accuracy, best = accuracy, thresholds
    return LevelThresholds(low=centre_in_gap(scores, best.low), high=centre_in_gap(scores, best.high))


def format_matrix(matrix: np.ndarray, rows: Sequence[str], columns: Sequence[str]) -> str:
    """Plain-text table with row and column labels."""
    widths = [max(6, len(str(c)) + 2) for c in columns]
    first = max(len(str(r)) for r in rows) + 2
    lines = [" " * first + "".join(f"{c:>{w}}" for c, w in zip(columns, widths, strict=True))]
    for label, row in zip(rows, matrix, strict=True):
        lines.append(f"{label:<{first}}" + "".join(f"{v:>{w}}" for v, w in zip(row, widths, strict=True)))
    return "\n".join(lines)
