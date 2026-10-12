from pathlib import Path

from resilens.core.contracts import Action, DegradationType


class RuleTable:
    @classmethod
    def load(cls, path: Path) -> "RuleTable":
        pass

    def lookup(self, degradation: DegradationType, level: int, profile: str) -> Action:
        pass
