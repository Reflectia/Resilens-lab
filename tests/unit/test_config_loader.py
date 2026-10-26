from pathlib import Path

import pytest

from resilens.config.loader import (
    ConfigNotFoundError,
    available_modes,
    available_profiles,
    build_run_config,
    load_mode,
    load_profile,
)


def test_available_presets() -> None:
    assert available_modes() == ["A", "B", "C", "D", "E"]
    assert available_profiles() == ["minimal", "performance", "standard"]


def test_load_by_name() -> None:
    assert load_mode("D").use_analyzer
    assert load_profile("standard").name == "standard"


def test_load_by_path(configs_dir: Path) -> None:
    assert load_mode(configs_dir / "modes" / "B.yaml").name == "B"


@pytest.mark.parametrize("name", ["Z", "../profiles/minimal"])
def test_unknown_mode_lists_available(name: str) -> None:
    with pytest.raises(ConfigNotFoundError, match="Available: A, B, C, D, E"):
        load_mode(name)


def test_build_run_config() -> None:
    config = build_run_config("A", "minimal", Path("clip.mp4"), seed=7)

    assert (config.mode.name, config.profile.name, config.source, config.seed) == ("A", "minimal", Path("clip.mp4"), 7)
