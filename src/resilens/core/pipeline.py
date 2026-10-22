from collections.abc import Iterable

from resilens.config.schema import RunConfig
from resilens.core.contracts import FrameContext
from resilens.core.stage import Stage
from resilens.metrics.telemetry import Telemetry


class Pipeline:
    def __init__(self, stages: list[Stage], telemetry: Telemetry) -> None:
        self.stages = stages
        self.telemetry = telemetry

    @classmethod
    def from_config(cls, config: RunConfig) -> "Pipeline":
        """Build stages for the selected mode (A–E) and resource profile."""
        pass

    def process_frame(self, ctx: FrameContext) -> FrameContext:
        """Run all stages on one frame, recording per-stage timing."""
        pass

    def run(self, frames: Iterable[FrameContext]) -> list[FrameContext]:
        """Setup stages, process every frame, teardown stages."""
        pass
