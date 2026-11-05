import inspect
from collections.abc import Iterable

import numpy as np

# Importing the stage modules registers their classes under the names used in configs/modes/*.yaml
import resilens.detection.yolox  # noqa: F401
import resilens.enhancement.bypass  # noqa: F401
import resilens.enhancement.tanet  # noqa: F401
import resilens.tracking.flow_tracker  # noqa: F401
from resilens.analysis.analyzer import DegradationAnalyzer
from resilens.config.schema import Device, PolicyBudget, RunConfig
from resilens.core.contracts import FrameContext
from resilens.core.registry import registry
from resilens.core.stage import Stage
from resilens.metrics.telemetry import Telemetry
from resilens.policy.engine import PolicyEngine
from resilens.policy.rules import RuleTable

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

        Order: analyzer (if used) -> policy (if used) -> enhancer -> detector -> tracker. Stage weights load in
        run(), not here. The policy decides on the analyzer's estimate and the telemetry's budget state.
        """
        mode, profile = config.mode, config.profile
        if mode.use_policy and not mode.use_analyzer:
            raise ValueError(f"Mode {mode.name} uses the Policy Engine, which needs the analyzer (use_analyzer: true)")

        rules = RuleTable.load() if mode.use_policy else None
        budget = rules.config.budget if rules else PolicyBudget()
        telemetry = Telemetry(
            config.warmup_frames, budget.latency_window, budget.latency_metric, gpu=profile.device is Device.CUDA
        )
        stages: list[Stage] = [DegradationAnalyzer()] if mode.use_analyzer else []
        if rules:
            stages.append(PolicyEngine(rules, profile, telemetry.budget_state))
        stages += [
            create_stage(mode.enhancer, mode.enhancer_params, profile.device.value),
            create_stage(mode.detector, mode.detector_params, profile.device.value),
            create_stage(mode.tracker, mode.tracker_params, profile.device.value),
        ]
        return cls(stages, telemetry)

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
