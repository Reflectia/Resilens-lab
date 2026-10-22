"""Leave-one-sequence-out evaluation of a global threshold pair per candidate.

Classes for degradation type T: 0 = not T (clean and other types), 1 = T level 1, 2 = T level 2.
Thresholds are fitted on three sequences and applied to the fourth, for each sequence in turn.

Columns:
  presence_acc     accuracy of "T present or not" on the held-out sequence (mean over folds)
  level_acc        accuracy of the 3-class decision (mean over folds)
  worst_seq_level  level accuracy on the hardest held-out sequence
"""

from pathlib import Path

import numpy as np
from candidates import CANDIDATES

OUT = Path(__file__).parent / "_out"
THRESHOLD_QUANTILES = np.linspace(0, 1, 101)


def predict(scores: np.ndarray, low: float, high: float) -> np.ndarray:
    return (scores >= low).astype(int) + (scores >= high).astype(int)


def fit_thresholds(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Brute-force the (low, high) pair with the best accuracy over score quantiles."""
    options = np.unique(np.quantile(scores, THRESHOLD_QUANTILES))
    best_accuracy, best_pair = -1.0, (options[0], options[0])
    for low in options:
        for high in options[options >= low]:
            accuracy = (predict(scores, low, high) == labels).mean()
            if accuracy > best_accuracy:
                best_accuracy, best_pair = accuracy, (low, high)
    return best_pair


def main() -> None:
    data = np.load(OUT / "frames.npz")
    distortion, levels, sequences = data["distortion"], data["level"], data["sequence"]

    print(f"{'type':<12}{'candidate':<24}{'presence_acc':>14}{'level_acc':>11}{'worst_seq_level':>17}")
    previous_type = None
    for candidate in CANDIDATES:
        scores = np.load(OUT / f"scores_{candidate.degradation}_{candidate.name}.npy")
        labels = np.where(distortion == candidate.degradation, levels, 0)

        presence_accs, level_accs = [], []
        for held_out in np.unique(sequences):
            train, test = sequences != held_out, sequences == held_out
            low, high = fit_thresholds(scores[train], labels[train])
            predicted = predict(scores[test], low, high)
            level_accs.append((predicted == labels[test]).mean())
            presence_accs.append(((predicted > 0) == (labels[test] > 0)).mean())

        if previous_type not in (None, candidate.degradation):
            print()
        previous_type = candidate.degradation
        print(
            f"{candidate.degradation:<12}{candidate.name:<24}"
            f"{np.mean(presence_accs):>14.3f}{np.mean(level_accs):>11.3f}{min(level_accs):>17.3f}"
        )


if __name__ == "__main__":
    main()
