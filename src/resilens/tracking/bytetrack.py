import numpy as np

from resilens.core.contracts import Detection, Track
from resilens.tracking.base import Tracker


class ByteTrackTracker(Tracker):
    name = "bytetrack"

    def __init__(
        self, track_thresh: float = 0.5, match_thresh: float = 0.8, track_buffer: int = 30, frame_rate: float = 30.0
    ) -> None:
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate

    def setup(self) -> None:
        pass

    def update(self, detections: list[Detection], frame_index: int, image: np.ndarray) -> list[Track]:
        pass
