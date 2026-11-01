from pathlib import Path

import numpy as np

from resilens.core.contracts import Detection, FrameContext, Track
from resilens.metrics.tracking import export_mot_detections, export_mot_format


def frames() -> list[FrameContext]:
    first = FrameContext(index=0, timestamp_s=0.0, image=np.zeros((1, 1, 3), np.uint8))
    first.detections = [Detection(bbox=(10.0, 20.0, 50.0, 100.0), score=0.87654, class_id=0)]
    first.tracks = [Track(track_id=3, bbox=(10.0, 20.0, 50.0, 100.0), score=1.0, class_id=0)]
    return [first, FrameContext(index=1, timestamp_s=0.04, image=np.zeros((1, 1, 3), np.uint8))]


def test_tracks_in_mot_format(tmp_path: Path) -> None:
    export_mot_format(frames(), tmp_path / "tracks.txt")

    assert (tmp_path / "tracks.txt").read_text() == "1,3,10.0,20.0,40.0,80.0,1.000,-1,-1,-1\n"


def test_detections_in_mot_det_format(tmp_path: Path) -> None:
    export_mot_detections(frames(), tmp_path / "detections.txt")

    assert (tmp_path / "detections.txt").read_text() == "1,-1,10.0,20.0,40.0,80.0,0.877,-1,-1,-1\n"
