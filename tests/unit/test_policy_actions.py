"""How the enhancer and the detector carry out the Policy Engine's decisions."""

import numpy as np
import pytest

from resilens.core.contracts import Action, Detection, FrameContext, PolicyDecision
from resilens.detection.base import Detector
from resilens.enhancement.base import Enhancer, fit_into

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


class InvertingEnhancer(Enhancer):
    def __init__(self) -> None:
        self.calls = 0

    def enhance(self, image: np.ndarray) -> np.ndarray:
        self.calls += 1
        return 255 - image


class FixedDetector(Detector):
    """One box at (8, 6, 16, 12) in the pixels of the image it gets."""

    def __init__(self) -> None:
        self.image_shapes: list[tuple[int, ...]] = []

    def detect(self, image: np.ndarray) -> list[Detection]:
        self.image_shapes.append(image.shape)
        return [Detection(bbox=(8.0, 6.0, 16.0, 12.0), score=0.9, class_id=0)]


def frame_with(decision: PolicyDecision | None) -> FrameContext:
    return FrameContext(index=0, timestamp_s=0.0, image=FRAME, decision=decision)


@pytest.mark.parametrize(
    ("decision", "enhanced"),
    [
        (None, True),
        (PolicyDecision(Action.ENHANCE, "", enhance=True), True),
        (PolicyDecision(Action.SKIP, "", enhance=True), True),
        (PolicyDecision(Action.SKIP, "", enhance=False), False),
        (PolicyDecision(Action.BYPASS, ""), False),
    ],
    ids=["no_policy", "enhance", "skip_nth_frame", "skip_other_frame", "bypass"],
)
def test_enhancer_runs_only_when_decided(decision: PolicyDecision | None, enhanced: bool) -> None:
    enhancer = InvertingEnhancer()

    ctx = enhancer.process(frame_with(decision))

    assert enhancer.calls == int(enhanced)
    assert (ctx.enhanced is not None) == enhanced
    if enhanced:
        assert ctx.working_image.mean() == 255


def test_downscale_fits_frame_into_profile_resolution_without_enhancing() -> None:
    enhancer = InvertingEnhancer()

    ctx = enhancer.process(frame_with(PolicyDecision(Action.DOWNSCALE, "", resize_to=(32, 32))))

    assert enhancer.calls == 0
    assert ctx.working_image.shape == (24, 32, 3)


@pytest.mark.parametrize(
    ("size", "shape"),
    [((32, 24), (24, 32, 3)), ((40, 100), (30, 40, 3)), ((100, 24), (24, 32, 3)), ((640, 480), (48, 64, 3))],
    ids=["exact_half", "width_limits", "height_limits", "never_upscales"],
)
def test_fit_into_keeps_aspect_ratio(size: tuple[int, int], shape: tuple[int, ...]) -> None:
    assert fit_into(FRAME, size).shape == shape


def test_detections_on_downscaled_frame_are_in_original_pixels() -> None:
    detector = FixedDetector()
    ctx = frame_with(None)
    ctx.enhanced = fit_into(FRAME, (32, 24))

    detector.process(ctx)

    assert detector.image_shapes == [(24, 32, 3)]
    assert ctx.detections[0].bbox == (16.0, 12.0, 32.0, 24.0)


def test_detections_on_full_size_frame_are_unchanged() -> None:
    ctx = FixedDetector().process(frame_with(None))

    assert ctx.detections[0].bbox == (8.0, 6.0, 16.0, 12.0)
