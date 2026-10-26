"""Score every candidate on the cached control frames and print a comparison table.

Columns, for degradation type T:
  auc_vs_rest   ROC-AUC of "T (any level)" vs everything else, clean included
  auc_l2_vs_l1  ROC-AUC of level 2 vs level 1 of T
  auc_l1_vs_cl  ROC-AUC of level 1 vs clean
  mono          share of sequences where mean score satisfies clean < level 1 < level 2
  ms            median time per frame
Scores are saved to _out/scores_<type>_<name>.npy for eval_thresholds.py.
"""

import time
from pathlib import Path

import numpy as np
from candidates import CANDIDATES, Candidate

OUT = Path(__file__).parent / "_out"


def roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Probability that a random positive scores higher than a random negative (ties count half)."""
    diff = positive[:, None] - negative[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def score_all(candidate: Candidate, frames: np.ndarray) -> tuple[np.ndarray, float]:
    """Return per-sample scores and the median time per call in ms."""
    scores, durations = [], []
    for sample in frames:
        arg = sample if candidate.needs_pair else sample[0]
        start = time.perf_counter()
        scores.append(candidate.func(arg))
        durations.append((time.perf_counter() - start) * 1000)
    return np.array(scores), float(np.median(durations))


def is_monotonic_per_sequence(scores, sequences, in_type, levels, clean) -> float:
    hits = []
    for sequence in np.unique(sequences):
        in_seq = sequences == sequence
        clean_mean = scores[in_seq & clean].mean()
        level1_mean = scores[in_seq & in_type & (levels == 1)].mean()
        level2_mean = scores[in_seq & in_type & (levels == 2)].mean()
        hits.append(clean_mean < level1_mean < level2_mean)
    return float(np.mean(hits))


def main() -> None:
    data = np.load(OUT / "frames.npz")
    frames, distortion, levels, sequences = data["frames"], data["distortion"], data["level"], data["sequence"]
    clean = distortion == "none"

    print(
        f"{'type':<12}{'candidate':<24}{'auc_vs_rest':>12}{'auc_l2_vs_l1':>14}{'auc_l1_vs_cl':>14}{'mono':>7}{'ms':>8}"
    )
    previous_type = None
    for candidate in CANDIDATES:
        scores, ms = score_all(candidate, frames)
        np.save(OUT / f"scores_{candidate.degradation}_{candidate.name}.npy", scores)

        in_type = distortion == candidate.degradation
        level1, level2 = in_type & (levels == 1), in_type & (levels == 2)
        if previous_type not in (None, candidate.degradation):
            print()
        previous_type = candidate.degradation
        print(
            f"{candidate.degradation:<12}{candidate.name:<24}"
            f"{roc_auc(scores[in_type], scores[~in_type]):>12.3f}"
            f"{roc_auc(scores[level2], scores[level1]):>14.3f}"
            f"{roc_auc(scores[level1], scores[clean]):>14.3f}"
            f"{is_monotonic_per_sequence(scores, sequences, in_type, levels, clean):>7.2f}"
            f"{ms:>8.2f}"
        )


if __name__ == "__main__":
    main()
