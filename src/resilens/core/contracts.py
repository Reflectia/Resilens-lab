from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np


class DegradationType(StrEnum):
    FOG = "fog"
    RAIN = "rain"
    LOW_LIGHT = "low_light"
    MOTION_BLUR = "motion_blur"
    NONE = "none"
    UNKNOWN = "unknown"


# Types the Degradation Analyzer can report besides NONE and UNKNOWN
DETECTABLE_DEGRADATIONS = (
    DegradationType.FOG,
    DegradationType.RAIN,
    DegradationType.LOW_LIGHT,
    DegradationType.MOTION_BLUR,
)


class Action(StrEnum):
    ENHANCE = "enhance"
    BYPASS = "bypass"
    DOWNSCALE = "downscale"
    SKIP = "skip"
    ENHANCE_ROI = "enhance_roi"  # Optional to implement


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float


@dataclass(frozen=True)
class DegradationEstimate:
    type: DegradationType
    level: int  # 0 = none, 1..2 = intensity levels used in scenarios
    confidence: float


@dataclass(frozen=True)
class BudgetState:
    """Latency and resource usage the policy sees before a frame; None = not measured yet."""

    latency_ms: float | None = None  # moving latency of the previous frames
    ram_mb: float | None = None
    cpu_percent: float | None = None
    gpu_memory_mb: float | None = None


@dataclass(frozen=True)
class PolicyDecision:
    action: Action  # final action, after the budget correction
    reason: str
    base_action: Action | None = None  # action from the rule table
    downgraded: bool = False  # the budget correction lowered the action
    enhance: bool = False  # the enhancer runs on this frame (enhance, and every N-th frame of skip)
    resize_to: tuple[int, int] | None = None  # (width, height) to fit the frame into, for downscale
    budget: BudgetState = field(default_factory=BudgetState)


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    score: float
    class_id: int


@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    score: float
    class_id: int


@dataclass(frozen=True)
class StageTiming:
    stage: str
    duration_ms: float


@dataclass
class FrameContext:
    index: int
    timestamp_s: float
    image: np.ndarray
    enhanced: np.ndarray | None = None
    degradation: DegradationEstimate | None = None
    decision: PolicyDecision | None = None
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    timings: list[StageTiming] = field(default_factory=list)

    @property
    def working_image(self) -> np.ndarray:
        """Image the next stage should consume: enhanced if available, otherwise original."""
        return self.enhanced if self.enhanced is not None else self.image
