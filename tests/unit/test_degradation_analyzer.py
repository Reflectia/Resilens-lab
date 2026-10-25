from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from resilens.analysis import analyzer as analyzer_module
from resilens.analysis.analyzer import DegradationAnalyzer, decide
from resilens.config.schema import AnalyzerConfig, LevelThresholds
from resilens.core.contracts import DegradationType, FrameContext
from resilens.scenarios.manifest import ScenarioEntry

FOG, RAIN, LOW_LIGHT, BLUR = (
    DegradationType.FOG,
    DegradationType.RAIN,
    DegradationType.LOW_LIGHT,
    DegradationType.MOTION_BLUR,
)


@pytest.fixture
def config() -> AnalyzerConfig:
    """Thresholds on very different scales, as real indicators have."""
    return AnalyzerConfig(
        confidence_threshold=0.5,
        thresholds={
            FOG: LevelThresholds(low=100, high=200),
            RAIN: LevelThresholds(low=0.01, high=0.02),
            LOW_LIGHT: LevelThresholds(low=150, high=200),
            BLUR: LevelThresholds(low=-0.3, high=-0.2),
        },
    )


def scores(fog=0.0, rain=0.0, low_light=0.0, blur=-1.0) -> dict[DegradationType, float]:
    return {FOG: fog, RAIN: rain, LOW_LIGHT: low_light, BLUR: blur}


def frame(index: int) -> FrameContext:
    return FrameContext(index=index, timestamp_s=index / 25, image=np.zeros((48, 64, 3), dtype=np.uint8))


def test_clean_frame(config) -> None:
    estimate = decide(scores(), config)

    assert (estimate.type, estimate.level, estimate.confidence) == (DegradationType.NONE, 0, 1.0)


@pytest.mark.parametrize(("fog", "level"), [(160, 1), (260, 2)])
def test_level_from_thresholds(config, fog: float, level: int) -> None:
    estimate = decide(scores(fog=fog), config)

    assert (estimate.type, estimate.level) == (FOG, level)


def test_dominant_type_compares_normalised_scores(config) -> None:
    estimate = decide(scores(fog=210, rain=0.04), config)

    assert estimate.type == RAIN
    assert estimate.level == 2


def test_close_to_threshold_is_unknown(config) -> None:
    estimate = decide(scores(fog=105), config)  # 0.05 above the level 1 threshold

    assert estimate.type == DegradationType.UNKNOWN
    assert estimate.level == 0
    assert estimate.confidence < config.confidence_threshold


def test_two_equally_strong_types_are_unknown(config) -> None:
    estimate = decide(scores(fog=200, low_light=200), config)

    assert estimate.type == DegradationType.UNKNOWN


def test_works_without_rain_score(config) -> None:
    frame_scores = scores(low_light=220)
    del frame_scores[RAIN]

    assert decide(frame_scores, config).type == LOW_LIGHT


def test_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="greater than low"):
        LevelThresholds(low=2, high=1)


def test_config_must_cover_every_type() -> None:
    with pytest.raises(ValidationError, match="missing"):
        AnalyzerConfig(thresholds={FOG: LevelThresholds(low=0, high=1)})


def test_scenario_metadata_wins_over_heuristics() -> None:
    scenario = ScenarioEntry("fog_l2", Path("fog_l2.mp4"), None, FOG, level=2, source_clip="MOT20-01", seed=0)
    analyzer = DegradationAnalyzer(scenario=scenario)

    ctx = analyzer.process(frame(0))

    assert (ctx.degradation.type, ctx.degradation.level, ctx.degradation.confidence) == (FOG, 2, 1.0)


def test_rain_needs_consecutive_frames(config, monkeypatch) -> None:
    monkeypatch.setattr(analyzer_module, "rain_score", lambda previous, gray: 1.0)
    analyzer = DegradationAnalyzer(config)
    analyzer.setup()

    is_rain = [analyzer.process(frame(i)).degradation.type == RAIN for i in (0, 1, 2, 5)]

    # no previous frame, consecutive, consecutive, gap after skipped frames
    assert is_rain == [False, True, True, False]
