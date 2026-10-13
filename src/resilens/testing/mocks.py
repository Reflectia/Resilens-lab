"""Mock stages with fixed outputs"""

from collections.abc import Iterator

import numpy as np

from resilens.core.contracts import (
    Action,
    DegradationEstimate,
    DegradationType,
    Detection,
    FrameContext,
    PolicyDecision,
    Track,
)
from resilens.core.stage import Stage


def mock_frames(count: int = 5, width: int = 64, height: int = 48, fps: float = 25.0) -> Iterator[FrameContext]:
    """Black frames with timestamps, in place of VideoReplay."""
    for i in range(count):
        yield FrameContext(index=i, timestamp_s=i / fps, image=np.zeros((height, width, 3), dtype=np.uint8))


class MockAnalyzer(Stage):
    name = "mock_analyzer"

    def process(self, ctx: FrameContext) -> FrameContext:
        ctx.degradation = DegradationEstimate(DegradationType.FOG, level=1, confidence=1.0)
        return ctx


class MockPolicy(Stage):
    name = "mock_policy"

    def process(self, ctx: FrameContext) -> FrameContext:
        ctx.decision = PolicyDecision(Action.BYPASS, reason="mock")
        return ctx


class MockEnhancer(Stage):
    name = "mock_enhancer"

    def process(self, ctx: FrameContext) -> FrameContext:
        ctx.enhanced = ctx.image.copy()
        return ctx


class MockDetector(Stage):
    name = "mock_detector"

    def process(self, ctx: FrameContext) -> FrameContext:
        ctx.detections = [Detection(bbox=(10.0, 10.0, 30.0, 40.0), score=0.9, class_id=0)]
        return ctx


class MockTracker(Stage):
    name = "mock_tracker"

    def process(self, ctx: FrameContext) -> FrameContext:
        ctx.tracks = [Track(track_id=1, bbox=d.bbox, score=d.score, class_id=d.class_id) for d in ctx.detections]
        return ctx
