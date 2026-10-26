"""Leave-one-sequence-out evaluation of a global threshold pair per candidate.

Classes for degradation type T: 0 = not T (clean and other types), 1 = T level 1, 2 = T level 2.
Thresholds are fitted on three sequences (maximising balanced accuracy) and applied to the fourth,
for each sequence in turn. Class 0 is 7 of 9 clips, so plain accuracy is shown only for comparison:
always answering "absent" already scores 0.78 on it, but only 0.33 on balanced accuracy.

Columns (means over held-out sequences):
  balanced     balanced accuracy of the 3-class decision
  plain        plain accuracy of the 3-class decision
  detect       share of T frames (any level) predicted as T
  false_clean  share of clean frames predicted as T
  worst_seq    balanced accuracy on the hardest held-out sequence
For the best candidate of each type, a confusion matrix summed over folds shows where errors come from;
its rows split "not T" by the actual distortion, so cross-triggers from other degradations are visible.
"""

from pathlib import Path

import numpy as np
from candidates import CANDIDATES

from resilens.analysis.calibration import (
    balanced_accuracy,
    confusion_matrix,
    fit_thresholds,
    format_matrix,
    predict_levels,
)

OUT = Path(__file__).parent / "_out"
PREDICTED = ["absent", "L1", "L2"]


def main() -> None:
    data = np.load(OUT / "frames.npz")
    distortion, levels, sequences = data["distortion"], data["level"], data["sequence"]
    clean = distortion == "none"
    clip_names = np.array([d if d == "none" else f"{d}_l{lv}" for d, lv in zip(distortion, levels, strict=True)])

    print(f"{'type':<12}{'candidate':<24}{'balanced':>10}{'plain':>8}{'detect':>8}{'false_clean':>13}{'worst_seq':>11}")
    best: dict[str, tuple[float, str, np.ndarray]] = {}
    previous_type = None
    for candidate in CANDIDATES:
        scores = np.load(OUT / f"scores_{candidate.degradation}_{candidate.name}.npy")
        labels = np.where(distortion == candidate.degradation, levels, 0)

        predicted = np.zeros_like(labels)
        balanced, plain, detect, false_clean = [], [], [], []
        for held_out in np.unique(sequences):
            train, test = sequences != held_out, sequences == held_out
            fold = predict_levels(scores[test], fit_thresholds(scores[train], labels[train]))
            predicted[test] = fold
            balanced.append(balanced_accuracy(labels[test], fold))
            plain.append((fold == labels[test]).mean())
            detect.append((fold[labels[test] > 0] > 0).mean())
            false_clean.append((fold[clean[test]] > 0).mean())

        if previous_type not in (None, candidate.degradation):
            print()
        previous_type = candidate.degradation
        print(
            f"{candidate.degradation:<12}{candidate.name:<24}{np.mean(balanced):>10.3f}{np.mean(plain):>8.3f}"
            f"{np.mean(detect):>8.3f}{np.mean(false_clean):>13.3f}{min(balanced):>11.3f}"
        )
        if candidate.degradation not in best or np.mean(balanced) > best[candidate.degradation][0]:
            best[candidate.degradation] = (float(np.mean(balanced)), candidate.name, predicted)

    rows = list(dict.fromkeys(sorted(clip_names, key=lambda name: (name != "none", name))))
    for degradation, (_, name, predicted) in best.items():
        print(f"\n{degradation} / {name}: true clip (rows) vs predicted level (columns), summed over folds")
        matrix = confusion_matrix(clip_names, [PREDICTED[p] for p in predicted], rows, PREDICTED)
        print(format_matrix(matrix, rows, PREDICTED))


if __name__ == "__main__":
    main()
