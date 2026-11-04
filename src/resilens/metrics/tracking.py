"""Tracking quality via TrackEval: HOTA, IDF1 and ID switches reported separately."""

from pathlib import Path

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


def compute_tracking_metrics(tracks_file: Path, annotations: Path) -> dict[str, float]:
    pass
