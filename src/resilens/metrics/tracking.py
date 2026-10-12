"""Tracking quality via TrackEval: HOTA, IDF1 and ID switches reported separately."""

from pathlib import Path

from resilens.core.contracts import FrameContext


def export_mot_format(frames: list[FrameContext], output: Path) -> None:
    """Write tracks in MOTChallenge format expected by TrackEval."""
    pass


def compute_tracking_metrics(tracks_file: Path, annotations: Path) -> dict[str, float]:
    pass
