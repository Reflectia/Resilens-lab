"""Calibrate and evaluate the Degradation Analyzer end to end on the cached control frames.

1. Leave-one-sequence-out: fit thresholds for each type on three sequences, run the Analyzer decision
   (dominant type, level, UNKNOWN on low confidence) on the fourth. Reported for several confidence
   thresholds, with a confusion matrix of clip conditions vs Analyzer output at the chosen one.
2. Fit the final thresholds on all sequences and write them to configs/analyzer/thresholds.yaml (--write).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from resilens.analysis.analyzer import compute_scores, decide
from resilens.analysis.calibration import balanced_accuracy, confusion_matrix, fit_thresholds, format_matrix
from resilens.analysis.heuristics import to_gray
from resilens.config.loader import ANALYZER_CONFIG
from resilens.config.schema import AnalyzerConfig
from resilens.core.contracts import DETECTABLE_DEGRADATIONS, DegradationType

OUT = Path(__file__).parent / "_out"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIDENCE_SWEEP = [0.0, 0.25, 0.5, 0.75]


def label(degradation: str, level: int) -> str:
    return degradation if degradation in (DegradationType.NONE, DegradationType.UNKNOWN) else f"{degradation}_l{level}"


def score_frames(frames: np.ndarray) -> tuple[dict[DegradationType, np.ndarray], float]:
    """Scores of every sample (second frame of each pair) and the median time per frame in ms."""
    scores: dict[DegradationType, list[float]] = {d: [] for d in DETECTABLE_DEGRADATIONS}
    durations = []
    for previous, image in frames:
        started = time.perf_counter()
        frame_scores = compute_scores(image, to_gray(image), to_gray(previous))
        durations.append((time.perf_counter() - started) * 1000)
        for degradation, score in frame_scores.items():
            scores[degradation].append(score)
    return {d: np.array(s) for d, s in scores.items()}, float(np.median(durations))


def fit_config(scores, distortion, levels, mask, confidence_threshold: float) -> AnalyzerConfig:
    thresholds = {
        d: fit_thresholds(scores[d][mask], np.where(distortion[mask] == d, levels[mask], 0))
        for d in DETECTABLE_DEGRADATIONS
    }
    return AnalyzerConfig(confidence_threshold=confidence_threshold, thresholds=thresholds)


def predict(scores, config: AnalyzerConfig, indices: np.ndarray) -> list[str]:
    estimates = [decide({d: float(scores[d][i]) for d in DETECTABLE_DEGRADATIONS}, config) for i in indices]
    return [label(e.type, e.level) for e in estimates]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confidence", type=float, default=0.5, help="confidence threshold to report and write")
    parser.add_argument("--write", action="store_true", help=f"write thresholds to {ANALYZER_CONFIG}")
    args = parser.parse_args()

    data = np.load(OUT / "frames.npz")
    distortion, levels, sequences = data["distortion"], data["level"], data["sequence"]
    truth = np.array([label(d, lv) for d, lv in zip(distortion, levels, strict=True)])
    clean = distortion == DegradationType.NONE
    scores, ms = score_frames(data["frames"])
    print(f"{len(truth)} samples, {len(np.unique(sequences))} sequences; Analyzer indicators: {ms:.1f} ms/frame\n")

    print(f"{'confidence':>10}{'coverage':>10}{'balanced':>10}{'covered_acc':>13}{'false_clean':>13}{'worst_seq':>11}")
    reported: np.ndarray | None = None
    for threshold in CONFIDENCE_SWEEP + ([args.confidence] if args.confidence not in CONFIDENCE_SWEEP else []):
        predicted = np.empty_like(truth, dtype=object)
        per_sequence = []
        for held_out in np.unique(sequences):
            train, test = sequences != held_out, sequences == held_out
            config = fit_config(scores, distortion, levels, train, threshold)
            predicted[test] = predict(scores, config, np.flatnonzero(test))
            per_sequence.append(balanced_accuracy(truth[test], predicted[test]))
        covered = predicted != DegradationType.UNKNOWN
        degraded = (predicted != DegradationType.NONE) & covered
        print(
            f"{threshold:>10.2f}{covered.mean():>10.3f}{balanced_accuracy(truth, predicted):>10.3f}"
            f"{(predicted[covered] == truth[covered]).mean():>13.3f}{degraded[clean].mean():>13.3f}"
            f"{min(per_sequence):>11.3f}"
        )
        if threshold == args.confidence:
            reported = predicted

    rows = sorted(set(truth), key=lambda name: (name != DegradationType.NONE, name))
    columns = [*rows, DegradationType.UNKNOWN.value]
    print(f"\nTrue clip condition (rows) vs Analyzer output (columns), confidence {args.confidence}, summed over folds")
    print(format_matrix(confusion_matrix(truth, reported, rows, columns), rows, columns))

    if args.write:
        config = fit_config(scores, distortion, levels, np.ones_like(clean), args.confidence)
        header = (
            "# Generated by benchmarks/degradation/calibrate.py - do not edit by hand.\n"
            f"# Fitted on {len(truth)} samples from sequences {', '.join(np.unique(sequences))}.\n"
        )
        body = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
        path = REPO_ROOT / ANALYZER_CONFIG
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + body, encoding="utf-8")
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
