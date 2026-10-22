import logging
import time
from collections.abc import Iterator

import cv2

from resilens.config.schema import ReplayMode
from resilens.core.contracts import FrameContext, StageTiming, VideoInfo
from resilens.io.validation import VideoValidationError

logger = logging.getLogger(__name__)


class VideoReplay:
    """Yields frames of a validated video in order, each with its index and timestamp.

    Timestamps are index / fps rather than decoder positions, so repeated runs
    on the same file give identical values. The time spent reading and decoding
    each frame is recorded in ctx.timings as stage "replay".
    """

    stage_name = "replay"

    def __init__(self, video: VideoInfo, mode: ReplayMode = ReplayMode.OFFLINE) -> None:
        self.video = video
        self.mode = mode
        self._frames_read = 0
        self._frames_dropped = 0
        self._ended_early = False

    def __iter__(self) -> Iterator[FrameContext]:
        if self.mode is not ReplayMode.OFFLINE:
            raise NotImplementedError(f"Replay mode '{self.mode}' is not implemented yet")

        cap = cv2.VideoCapture(str(self.video.path))
        if not cap.isOpened():
            raise VideoValidationError(f"Cannot open video: {self.video.path}")

        self._frames_read = 0
        self._ended_early = False
        try:
            while True:
                started = time.perf_counter()
                ok, frame = cap.read()
                read_ms = (time.perf_counter() - started) * 1000
                if not ok or frame is None:
                    self._ended_early = self._frames_read < self.video.frame_count
                    break
                yield FrameContext(
                    index=self._frames_read,
                    timestamp_s=self._frames_read / self.video.fps,
                    image=frame,
                    timings=[StageTiming(self.stage_name, read_ms)],
                )
                self._frames_read += 1
        finally:
            cap.release()

        if self._ended_early:
            logger.warning(
                "Video %s ended early: read %d of %d frames (file may be truncated or corrupted)",
                self.video.path,
                self._frames_read,
                self.video.frame_count,
            )

    @property
    def frames_read(self) -> int:
        """Frames yielded by the last iteration."""
        return self._frames_read

    @property
    def ended_early(self) -> bool:
        """True if the file ran out of frames before the frame count it reports."""
        return self._ended_early

    @property
    def frames_dropped(self) -> int:
        """Frames skipped in REALTIME mode because the pipeline was late (always 0 in OFFLINE)."""
        return self._frames_dropped
