from pathlib import Path

from resilens.core.contracts import VideoInfo

SUPPORTED_EXTENSIONS = {".mp4", ".avi"}


class VideoValidationError(Exception):
    pass


class VideoValidator:
    def __init__(self, max_duration_s: float | None = None, max_resolution: tuple[int, int] | None = None) -> None:
        self.max_duration_s = max_duration_s
        self.max_resolution = max_resolution

    def validate(self, path: Path) -> VideoInfo:
        pass
