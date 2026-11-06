"""Tracking quality via TrackEval: HOTA, IDF1 and ID switches reported separately."""

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from trackeval.metrics import CLEAR, HOTA, Identity

from resilens.core.contracts import FrameContext


def _mot_row(frame_index: int, object_id: int, bbox: tuple[float, float, float, float], score: float) -> str:
    x1, y1, x2, y2 = bbox
    return f"{frame_index + 1},{object_id},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{score:.3f},-1,-1,-1\n"


def export_mot_format(frames: list[FrameContext], output: Path) -> None:
    """Write tracks in MOTChallenge format expected by TrackEval."""
    rows = (_mot_row(ctx.index, t.track_id, t.bbox, t.score) for ctx in frames for t in ctx.tracks)
    output.write_text("".join(rows), encoding="utf-8")


def export_mot_detections(frames: list[FrameContext], output: Path) -> None:
    """Write detections in MOTChallenge detection format (det.txt): as tracks, with id -1."""
    rows = (_mot_row(ctx.index, -1, d.bbox, d.score) for ctx in frames for d in ctx.detections)
    output.write_text("".join(rows), encoding="utf-8")


def tracking_sequence(tracks_file: Path, annotations: Path, frame_count: int | None = None) -> dict:
    all_gt = _read_rows(annotations, ground_truth=True)
    tracks = _read_rows(tracks_file, ground_truth=False)
    gt = [row[:6] for row in all_gt if row[6] != 0 and row[7] == 1]
    if not gt:
        raise ValueError("No person ground truth available for tracking evaluation")
    max_frame = max([row[0] for row in all_gt] + [row[0] for row in tracks])
    if frame_count is not None and max_frame > frame_count:
        raise ValueError("MOT annotation frames do not match video frames")
    # Follow TrackEval MOT20 preprocessing: match all GT and remove predictions matched to distractors.
    ignored = set()
    for frame in range(1, max_frame + 1):
        g = [row for row in all_gt if row[0] == frame]
        t = [row for row in tracks if row[0] == frame]
        if not g or not t:
            continue
        overlaps = _overlaps(g, t)
        scores = overlaps.copy()
        scores[scores < 0.5 - np.finfo(float).eps] = 0
        gi, ti = linear_sum_assignment(-scores)
        ignored.update(
            (frame, t[j][1])
            for i, j in zip(gi, ti, strict=True)
            if scores[i, j] > np.finfo(float).eps and g[i][7] in (2, 6, 7, 8, 12)
        )
    tracks = [row for row in tracks if (row[0], row[1]) not in ignored]
    gt_ids = {value: i for i, value in enumerate(sorted({row[1] for row in gt}))}
    track_ids = {value: i for i, value in enumerate(sorted({row[1] for row in tracks}))}
    data = {
        "num_gt_ids": len(gt_ids),
        "num_tracker_ids": len(track_ids),
        "num_gt_dets": len(gt),
        "num_tracker_dets": len(tracks),
        "num_timesteps": max_frame,
        "gt_ids": [],
        "tracker_ids": [],
        "similarity_scores": [],
    }
    for frame in range(1, max_frame + 1):
        g = [row for row in gt if row[0] == frame]
        t = [row for row in tracks if row[0] == frame]
        data["gt_ids"].append(np.array([gt_ids[row[1]] for row in g], dtype=int))
        data["tracker_ids"].append(np.array([track_ids[row[1]] for row in t], dtype=int))
        data["similarity_scores"].append(_overlaps(g, t))
    return data


def _overlaps(g: list[tuple], t: list[tuple]) -> np.ndarray:
    gb = np.array([r[2:6] for r in g], dtype=float).reshape(-1, 4)
    tb = np.array([r[2:6] for r in t], dtype=float).reshape(-1, 4)
    low = np.maximum(gb[:, None, :2], tb[None, :, :2])
    high = np.minimum((gb[:, :2] + gb[:, 2:])[:, None, :], (tb[:, :2] + tb[:, 2:])[None, :, :])
    intersection = np.prod(np.maximum(high - low, 0), axis=2)
    union = np.prod(gb[:, 2:], axis=1)[:, None] + np.prod(tb[:, 2:], axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def compute_tracking_metrics(tracks_file: Path, annotations: Path, frame_count: int | None = None) -> dict[str, float]:
    data = tracking_sequence(tracks_file, annotations, frame_count)
    return tracking_metrics_from_sequences([data])


def tracking_metrics_from_sequences(sequences: list[dict]) -> dict[str, float]:
    metrics = [HOTA(), Identity({"PRINT_CONFIG": False}), CLEAR({"PRINT_CONFIG": False})]
    results = []
    for metric in metrics:
        values = {str(i): metric.eval_sequence(sequence) for i, sequence in enumerate(sequences)}
        results.append(metric.combine_sequences(values))
    hota, identity, clear = results
    return {"hota": float(np.mean(hota["HOTA"])), "idf1": float(identity["IDF1"]), "id_switches": float(clear["IDSW"])}


def _read_rows(path: Path, ground_truth: bool) -> list[tuple]:
    rows = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        if len(parts) < 9:
            raise ValueError(f"Invalid MOT annotation: {path}")
        frame, identity = int(parts[0]), int(parts[1])
        box = tuple(float(value) for value in parts[2:6])
        if frame < 1 or identity < 1 or box[2] <= 0 or box[3] <= 0 or (frame, identity) in seen:
            raise ValueError(f"Invalid MOT row in {path}")
        seen.add((frame, identity))
        rows.append(
            (frame, identity, *box, float(parts[6]), int(float(parts[7]))) if ground_truth else (frame, identity, *box)
        )
    return rows
