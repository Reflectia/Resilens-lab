from pathlib import Path

import pytest
import yaml

from resilens.config.loader import load_analyzer_config
from resilens.config.schema import ModeConfig, ProfileConfig


@pytest.mark.parametrize("mode", ["A", "B", "C", "D", "E"])
def test_mode_presets_are_valid(configs_dir: Path, mode: str) -> None:
    data = yaml.safe_load((configs_dir / "modes" / f"{mode}.yaml").read_text(encoding="utf-8"))
    assert ModeConfig(**data).name == mode


@pytest.mark.parametrize("profile", ["minimal", "standard", "performance"])
def test_profiles_are_valid(configs_dir: Path, profile: str) -> None:
    path = configs_dir / "profiles" / f"{profile}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert ProfileConfig(**data).name == profile


def test_analyzer_thresholds_are_valid(configs_dir: Path) -> None:
    config = load_analyzer_config(configs_dir / "analyzer" / "thresholds.yaml")
    assert 0 <= config.confidence_threshold <= 1
