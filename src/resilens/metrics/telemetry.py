from contextlib import AbstractContextManager
from threading import Event, Thread

import psutil

from resilens.core.contracts import FrameContext


class ResourceSampler:
    """Observe peak process RAM while a run is executing."""

    def __init__(self, interval_s: float = 0.01) -> None:
        self.interval_s = interval_s
        self._process = psutil.Process()
        self._stopping = Event()
        self._thread: Thread | None = None
        self._peak_ram_mb = 0.0

    def start(self) -> None:
        self._peak_ram_mb = 0.0
        self._stopping.clear()
        self._sample()
        self._thread = Thread(target=self._poll, name="resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        """Return peak values collected since start()."""
        self._stopping.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._sample()
        return {"peak_ram_mb": self._peak_ram_mb}

    def _sample(self) -> None:
        self._peak_ram_mb = max(self._peak_ram_mb, self._process.memory_info().rss / 2**20)

    def _poll(self) -> None:
        while not self._stopping.wait(self.interval_s):
            self._sample()


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
