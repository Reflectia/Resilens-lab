from contextlib import AbstractContextManager

from resilens.core.contracts import FrameContext


class ResourceSampler:
    """Peak RAM / CPU, GPU / VRAM when a GPU is present."""

    def start(self) -> None:
        pass

    def stop(self) -> dict[str, float]:
        """Return peak values collected since start()."""
        pass


class Telemetry:
    def __init__(self, warmup_frames: int = 10) -> None:
        self.warmup_frames = warmup_frames
        self.resources = ResourceSampler()

    def stage_timer(self, ctx: FrameContext, stage: str) -> AbstractContextManager[None]:
        pass

    def record_frame(self, ctx: FrameContext) -> None:
        """Store end-to-end latency and per-stage timings of a finished frame."""
        pass

    def summary(self) -> dict:
        """Latency summary plus peak resource usage."""
        pass
