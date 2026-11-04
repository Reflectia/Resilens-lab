import inspect
from collections.abc import Iterable

import numpy as np

# Importing the stage modules registers their classes under the names used in configs/modes/*.yaml
import resilens.detection.yolox  # noqa: F401
import resilens.enhancement.bypass  # noqa: F401
import resilens.enhancement.tanet  # noqa: F401
import resilens.tracking.flow_tracker  # noqa: F401
from resilens.analysis.analyzer import DegradationAnalyzer
from resilens.config.schema import RunConfig
from resilens.core.contracts import FrameContext
from resilens.core.registry import registry
from resilens.core.stage import Stage
from resilens.metrics.telemetry import Telemetry

NO_PIXELS = np.empty((0, 0, 3), dtype=np.uint8)  # replaces frames already processed, see Pipeline.run


def create_stage(name: str, params: dict, device: str) -> Stage:
    """Stage from the registry; the profile's device is passed to stages that take one, unless the mode sets it."""
    stage_class = registry.get(name)
    if "device" in inspect.signature(stage_class).parameters:
        params = {"device": device, **params}
    return stage_class(**params)


class Pipeline:
    def __init__(self, stages: list[Stage], telemetry: Telemetry) -> None:
        self.stages = stages
        self.telemetry = telemetry

    @classmethod
    def from_config(cls, config: RunConfig) -> "Pipeline":
        """Build stages for the selected mode (A–E) and resource profile.

        Order: analyzer (if used) -> enhancer -> detector -> tracker. Stage weights load in run(), not here.
        """
        mode, device = config.mode, config.profile.device.value
        if mode.use_policy:
            raise NotImplementedError(f"Mode {mode.name} needs the Policy Engine, which is not implemented yet (T-10)")

        stages: list[Stage] = [DegradationAnalyzer()] if mode.use_analyzer else []
        stages += [
            create_stage(mode.enhancer, mode.enhancer_params, device),
            create_stage(mode.detector, mode.detector_params, device),
            create_stage(mode.tracker, mode.tracker_params, device),
        ]
        return cls(stages, Telemetry(config.warmup_frames))

    def process_frame(self, ctx: FrameContext) -> FrameContext:
        """Run all stages on one frame, recording per-stage timing and the frame's latency in telemetry."""
        for stage in self.stages:
            with self.telemetry.stage_timer(ctx, stage.name):
                ctx = stage.process(ctx)
        self.telemetry.record_frame(ctx)
        return ctx

    def run(self, frames: Iterable[FrameContext]) -> list[FrameContext]:
        """Setup stages, process every frame, teardown stages.

        Returned frames keep their results (degradation, detections, tracks, timings) but not their pixels, so
        memory does not grow with the video length.
        """
        processed: list[FrameContext] = []
        try:
            for stage in self.stages:
                stage.setup()
            for ctx in frames:
                ctx = self.process_frame(ctx)
                ctx.image, ctx.enhanced = NO_PIXELS, None
                processed.append(ctx)
        finally:
            for stage in self.stages:
                stage.teardown()
        return processed
