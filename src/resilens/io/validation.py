from pathlib import Path

import cv2

from resilens.core.contracts import VideoInfo

SUPPORTED_EXTENSIONS = {".mp4", ".avi"}


class VideoValidationError(Exception):
    """Video cannot be processed."""


class VideoValidator:
    def __init__(self, max_duration_s: float | None = None, max_resolution: tuple[int, int] | None = None) -> None:
        """Limits are optional: None means no limit (values are agreed after baseline measurements).

        max_resolution is a tuple - (width, height).
        """
        self.max_duration_s = max_duration_s
        self.max_resolution = max_resolution

    def validate(self, path: Path) -> VideoInfo:
        """Check extension, readability, duration and resolution before processing."""
        path = Path(path)
        if not path.is_file():
            raise VideoValidationError(f"Video file not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise VideoValidationError(f"Unsupported video format '{path.suffix}'. Supported: {supported}")

        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise VideoValidationError(f"Cannot open video (file is corrupted or codec is not supported): {path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            ok, frame = cap.read()
            if not ok or frame is None:
                raise VideoValidationError(f"Cannot read frames from video (file is corrupted): {path}")
            if fps <= 0:
                raise VideoValidationError(f"Video has no valid frame rate: {path}")
            if frame_count <= 0:
                raise VideoValidationError(f"Cannot determine the number of frames in video: {path}")
        finally:
            cap.release()

        duration_s = frame_count / fps
        if self.max_duration_s is not None and duration_s > self.max_duration_s:
            raise VideoValidationError(f"Video is too long: {duration_s:.1f} s, maximum is {self.max_duration_s:.1f} s")
        if self.max_resolution is not None:
            max_width, max_height = self.max_resolution
            if width > max_width or height > max_height:
                raise VideoValidationError(
                    f"Video resolution {width}x{height} exceeds the maximum {max_width}x{max_height}"
                )

        return VideoInfo(path=path, fps=fps, frame_count=frame_count, width=width, height=height, duration_s=duration_s)
