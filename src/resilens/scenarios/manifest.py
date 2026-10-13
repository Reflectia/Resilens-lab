"""Known degradation type and level for each clip"""

from dataclasses import dataclass
from pathlib import Path

from resilens.core.contracts import DegradationType


@dataclass(frozen=True)
class ScenarioEntry:
    scenario_id: str
    video: Path
    annotations: Path | None
    degradation: DegradationType
    level: int
    source_clip: str
    seed: int


class ScenarioManifest:
    def __init__(self, entries: list[ScenarioEntry]) -> None:
        self.entries = entries

    @classmethod
    def load(cls, path: Path) -> "ScenarioManifest":
        pass

    def save(self, path: Path) -> None:
        pass

    def get(self, scenario_id: str) -> ScenarioEntry:
        pass
