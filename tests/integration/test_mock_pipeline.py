"""Mock stages chained together follow the FrameContext contract."""

import pytest

from resilens.testing.mocks import (
    MockAnalyzer,
    MockDetector,
    MockEnhancer,
    MockPolicy,
    MockTracker,
    mock_frames,
)


@pytest.mark.integration
def test_mock_stages_fill_frame_context() -> None:
    stages = [MockAnalyzer(), MockPolicy(), MockEnhancer(), MockDetector(), MockTracker()]
    for ctx in mock_frames(count=3):
        for stage in stages:
            ctx = stage.process(ctx)
        assert ctx.degradation is not None
        assert ctx.decision is not None
        assert ctx.enhanced is not None
        assert ctx.tracks[0].track_id == 1
