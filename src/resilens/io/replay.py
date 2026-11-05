import logging
import time
from collections.abc import Callable, Iterator

import cv2

from resilens.config.schema import ReplayMode
from resilens.core.contracts import FrameContext, StageTiming, VideoInfo
from resilens.io.validation import VIDEO_BACKEND, VideoValidationError

logger = logging.getLogger(__name__)


class VideoReplay:
    """Yields frames of a validated video in order, each with its index and timestamp.

    Timestamps are index / fps rather than decoder positions, so repeated runs
    on the same file give identical values. The time spent reading and decoding
    each frame is recorded in ctx.timings as stage "replay".

    OFFLINE yields every frame and waits for the consumer. REALTIME plays the video
    like a camera at its own fps, starting when the first frame is requested: a frame
    is never yielded before its time, and frames whose time passed while the consumer
    was busy are dropped, so indices of the yielded frames can have gaps.
    """

    stage_name = "replay"

    def __init__(
        self,
        video: VideoInfo,
        mode: ReplayMode = ReplayMode.OFFLINE,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.video = video
        self.mode = mode
        self._clock = clock
        self._sleep = sleep
        self._frames_read = 0
        self._frames_dropped = 0
        self._ended_early = False

    def __iter__(self) -> Iterator[FrameContext]:
        cap = cv2.VideoCapture(str(self.video.path), VIDEO_BACKEND)
        if not cap.isOpened():
            raise VideoValidationError(f"Cannot open video: {self.video.path}")

        self._frames_read = 0
        self._frames_dropped = 0
        self._ended_early = False
        position = 0  # index of the next frame in the file
        started_at: float | None = None
        try:
            while True:
                if self.mode is ReplayMode.REALTIME:
                    if started_at is None:
                        started_at = self._clock()
                    for _ in range(self._frames_late(position, started_at)):
                        if not cap.grab():
                            break  # the video ended; the read below fails too
                        position += 1
                        self._frames_dropped += 1

                started = time.perf_counter()
                ok, frame = cap.read()
                read_ms = (time.perf_counter() - started) * 1000
                if not ok or frame is None:
                    self._ended_early = position < self.video.frame_count
                    break
                yield FrameContext(
                    index=position,
                    timestamp_s=position / self.video.fps,
                    image=frame,
                    timings=[StageTiming(self.stage_name, read_ms)],
                )
                self._frames_read += 1
                position += 1
        finally:
            cap.release()

        if self._ended_early:
            logger.warning(
                "Video %s ended early: read %d of %d frames (file may be truncated or corrupted)",
                self.video.path,
                position,
                self.video.frame_count,
            )

    def _frames_late(self, position: int, started_at: float) -> int:
        """Frames before the one the camera shows now that were not read yet, starting at `position`.

        If the frame at `position` is not due yet, waits until it is and returns 0 (no wait past the last frame).
        """
        elapsed_s = self._clock() - started_at
        due = int(elapsed_s * self.video.fps)  # frame the camera shows now
        if due <= position:
            if position < self.video.frame_count:
                self._sleep(max(position / self.video.fps - elapsed_s, 0.0))
            return 0
        return due - position

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
