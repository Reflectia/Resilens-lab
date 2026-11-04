from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from resilens.core.contracts import DETECTABLE_DEGRADATIONS, Action, DegradationType


class ReplayMode(StrEnum):
    OFFLINE = "offline"
    REALTIME = "realtime"


class Device(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class ProfileConfig(BaseModel):
    """Resource profile. Limits left at None are not checked by the Policy Engine."""

    name: str
    device: Device = Device.CPU
    cpu_threads: int | None = None
    max_resolution: tuple[int, int] | None = None  # (width, height); larger videos are rejected
    downscale_resolution: tuple[int, int] | None = None  # (width, height) the policy's downscale action fits into
    frame_budget_ms: float | None = None
    memory_limit_mb: int | None = None
    cpu_limit_percent: float | None = None  # process CPU as psutil reports it: 100 = one fully used core
    gpu_memory_limit_mb: int | None = None


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


BASE_ACTIONS = (Action.ENHANCE, Action.SKIP, Action.BYPASS)  # downscale only results from the budget correction
POLICY_ACTIONS = (*BASE_ACTIONS, Action.DOWNSCALE)


class PolicyRule(BaseModel):
    """Base action per resource profile for some degradation types at one level (1 = low, 2 = high)."""

    types: list[DegradationType] = Field(min_length=1)
    level: int = Field(ge=1, le=2)
    actions: dict[str, Action]

    @model_validator(mode="after")
    def _check(self) -> "PolicyRule":
        undetectable = [t for t in self.types if t not in DETECTABLE_DEGRADATIONS]
        if undetectable:
            raise ValueError(f"types {undetectable} cannot have rules: they always get default_action")
        not_base = {profile: a for profile, a in self.actions.items() if a not in BASE_ACTIONS}
        if not_base:
            raise ValueError(f"actions {not_base} are not allowed in rules, use one of {[*BASE_ACTIONS]}")
        return self


class PolicyBudget(BaseModel):
    """When the budget correction lowers the action by one step."""

    latency_window: int = Field(30, ge=1)  # frames
    latency_metric: Literal["mean", "p95"] = "mean"
    near_limit_ratio: float = Field(0.9, gt=0, le=1)  # share of a RAM/CPU/GPU limit that counts as "near"


class PolicyConfig(BaseModel):
    """Policy Engine rule table (task T-07) and its parameters."""

    skip_every_n: int = Field(3, ge=1)  # skip: the enhancer runs on every N-th frame
    downgrade_order: list[Action] = Field(default_factory=lambda: list(POLICY_ACTIONS))
    default_action: Action = Action.BYPASS  # for unknown, none and types or levels without a rule
    budget: PolicyBudget = Field(default_factory=PolicyBudget)
    rules: list[PolicyRule]

    @model_validator(mode="after")
    def _check(self) -> "PolicyConfig":
        if sorted(self.downgrade_order) != sorted(POLICY_ACTIONS):
            raise ValueError(f"downgrade_order must list each of {[*POLICY_ACTIONS]} once")
        if self.default_action not in BASE_ACTIONS:
            raise ValueError(f"default_action must be one of {[*BASE_ACTIONS]}")
        seen: set[tuple[DegradationType, int]] = set()
        for rule in self.rules:
            for degradation in rule.types:
                if (degradation, rule.level) in seen:
                    raise ValueError(f"more than one rule for {degradation} level {rule.level}")
                seen.add((degradation, rule.level))
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
