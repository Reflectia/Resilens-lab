from collections.abc import Iterator

from resilens.config.schema import ReplayMode
from resilens.core.contracts import FrameContext, VideoInfo


class VideoReplay:
    def __init__(self, video: VideoInfo, mode: ReplayMode = ReplayMode.OFFLINE) -> None:
        self.video = video
        self.mode = mode

    def __iter__(self) -> Iterator[FrameContext]:
        pass

    @property
    def dropped_frames(self) -> int:
        """Frames skipped in REALTIME mode because the pipeline was late."""
        pass
