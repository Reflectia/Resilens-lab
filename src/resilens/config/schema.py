from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


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
