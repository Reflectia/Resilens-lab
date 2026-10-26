import numpy as np
import pytest

from resilens.analysis.calibration import balanced_accuracy, confusion_matrix, fit_thresholds, predict_levels
from resilens.config.schema import LevelThresholds

# Class sizes of one control sequence for one type: 7 clips without it, 1 per level
LABELS = np.array([0] * 56 + [1] * 8 + [2] * 8)


def test_always_absent_scores_chance_level() -> None:
    always_absent = np.zeros_like(LABELS)

    assert (always_absent == LABELS).mean() == pytest.approx(56 / 72)
    assert balanced_accuracy(LABELS, always_absent) == pytest.approx(1 / 3)


def test_perfect_prediction() -> None:
    assert balanced_accuracy(LABELS, LABELS) == 1.0


def test_predict_levels() -> None:
    levels = predict_levels(np.array([0.0, 1.0, 1.5, 2.0, 3.0]), LevelThresholds(low=1, high=2))

    assert levels.tolist() == [0, 1, 1, 2, 2]


def test_confusion_matrix_counts() -> None:
    matrix = confusion_matrix(["a", "a", "b"], ["a", "b", "b"], rows=["a", "b"], columns=["a", "b", "x"])

    assert matrix.tolist() == [[1, 1, 0], [0, 1, 0]]


def test_fit_separates_levels() -> None:
    rng = np.random.default_rng(0)
    scores = LABELS + rng.uniform(0, 0.5, LABELS.size)  # absent ~0, level 1 ~1, level 2 ~2

    thresholds = fit_thresholds(scores, LABELS)

    assert balanced_accuracy(LABELS, predict_levels(scores, thresholds)) == 1.0


def test_fit_centres_thresholds_in_gaps() -> None:
    scores = np.array([0.0] * 56 + [10.0] * 8 + [20.0] * 8)

    thresholds = fit_thresholds(scores, LABELS)

    assert (thresholds.low, thresholds.high) == (5.0, 15.0)


def test_fit_does_not_sacrifice_small_classes() -> None:
    scores = np.concatenate([np.linspace(0, 1, 56), np.linspace(0.8, 1.0, 8), np.full(8, 5.0)])

    predicted = predict_levels(scores, fit_thresholds(scores, LABELS))

    assert (predicted[LABELS == 1] == 1).mean() >= 0.5


def test_fit_rejects_constant_scores() -> None:
    with pytest.raises(ValueError, match="all scores are equal"):
        fit_thresholds(np.ones(10), np.zeros(10, dtype=int))
