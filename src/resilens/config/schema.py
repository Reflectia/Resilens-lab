from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from resilens.core.contracts import DETECTABLE_DEGRADATIONS, DegradationType


class ReplayMode(StrEnum):
    OFFLINE = "offline"
    REALTIME = "realtime"


class Device(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class ProfileConfig(BaseModel):
    name: str
    device: Device = Device.CPU
    cpu_threads: int | None = None
    max_resolution: tuple[int, int] | None = None
    frame_budget_ms: float | None = None
    memory_limit_mb: int | None = None


class ModeConfig(BaseModel):
    name: str
    description: str = ""
    use_analyzer: bool = False
    use_policy: bool = False
    enhancer: str = "bypass"
    enhancer_params: dict = Field(default_factory=dict)
    detector: str = "yolox_nano"
    detector_params: dict = Field(default_factory=dict)
    tracker: str = "bytetrack"
    tracker_params: dict = Field(default_factory=dict)


class LevelThresholds(BaseModel):
    """Indicator score at which a degradation counts as level 1 (low) and level 2 (high)."""

    low: float
    high: float

    @model_validator(mode="after")
    def _check_order(self) -> "LevelThresholds":
        if self.high <= self.low:
            raise ValueError(f"high ({self.high}) must be greater than low ({self.low})")
        return self


class AnalyzerConfig(BaseModel):
    """Degradation Analyzer thresholds, fitted on control videos."""

    confidence_threshold: float = Field(0.5, ge=0, le=1)
    thresholds: dict[DegradationType, LevelThresholds]

    @model_validator(mode="after")
    def _check_types(self) -> "AnalyzerConfig":
        missing = set(DETECTABLE_DEGRADATIONS) - set(self.thresholds)
        extra = set(self.thresholds) - set(DETECTABLE_DEGRADATIONS)
        if missing or extra:
            raise ValueError(f"thresholds: missing {sorted(missing)}, unexpected {sorted(extra)}")
        return self


class RunConfig(BaseModel):
    mode: ModeConfig
    profile: ProfileConfig
    source: Path
    annotations: Path | None = None  # ground truth, only for control videos
    replay_mode: ReplayMode = ReplayMode.OFFLINE
    seed: int = 0
    warmup_frames: int = 10
    repeats: int = 1
    save_annotated_video: bool = False
