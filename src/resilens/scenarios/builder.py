from pathlib import Path

from resilens.core.contracts import DegradationType
from resilens.scenarios.manifest import ScenarioManifest


class ScenarioBuilder:
    def __init__(self, source_dir: Path, output_dir: Path, seed: int = 0) -> None:
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.seed = seed

    def build(self, degradations: list[DegradationType], levels: list[int]) -> ScenarioManifest:
        """Apply each degradation/level to each annotated clip and write the manifest."""
        pass
