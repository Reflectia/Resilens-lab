import numpy as np

from resilens.analysis.heuristics import fog_score, low_light_score, motion_blur_score, prepare, rain_score, to_gray
from resilens.config.loader import load_analyzer_config
from resilens.config.schema import AnalyzerConfig
from resilens.core.contracts import DegradationEstimate, DegradationType, FrameContext
from resilens.core.stage import Stage
from resilens.scenarios.manifest import ScenarioEntry

# Distance in normalised score units (1.0 = span between the level 1 and level 2 thresholds)
# from the nearest competing decision at which confidence reaches 1
CONFIDENCE_MARGIN = 0.5


def compute_scores(
    image: np.ndarray, gray: np.ndarray, previous_gray: np.ndarray | None
) -> dict[DegradationType, float]:
    """Indicator scores of a frame prepared by heuristics.prepare(); rain needs the previous frame."""
    scores = {
        DegradationType.FOG: fog_score(image),
        DegradationType.LOW_LIGHT: low_light_score(gray),
        DegradationType.MOTION_BLUR: motion_blur_score(gray),
    }
    if previous_gray is not None:
        scores[DegradationType.RAIN] = rain_score(previous_gray, gray)
    return scores


def normalise(scores: dict[DegradationType, float], config: AnalyzerConfig) -> dict[DegradationType, float]:
    """Map each score to a common scale: 0 = level 1 threshold, 1 = level 2 threshold."""
    return {
        degradation: (score - config.thresholds[degradation].low)
        / (config.thresholds[degradation].high - config.thresholds[degradation].low)
        for degradation, score in scores.items()
    }


def decide(scores: dict[DegradationType, float], config: AnalyzerConfig) -> DegradationEstimate:
    """Pick the dominant degradation and its level, or UNKNOWN when the decision is too close to call."""
    ranked = sorted(normalise(scores, config).items(), key=lambda item: item[1], reverse=True)
    best_type, best = ranked[0]
    if best < 0:
        confidence = min(-best / CONFIDENCE_MARGIN, 1.0)
        estimate = DegradationEstimate(DegradationType.NONE, level=0, confidence=confidence)
    else:
        runner_up = max(ranked[1][1], 0.0) if len(ranked) > 1 else 0.0
        confidence = min((best - runner_up) / CONFIDENCE_MARGIN, 1.0)
        estimate = DegradationEstimate(best_type, level=2 if best >= 1 else 1, confidence=confidence)

    if confidence < config.confidence_threshold:
        return DegradationEstimate(DegradationType.UNKNOWN, level=0, confidence=confidence)
    return estimate


class DegradationAnalyzer(Stage):
    """Uses scenario metadata for control videos, heuristics for user videos.

    Low confidence -> UNKNOWN; several degradations -> the dominant one. Frames must arrive in order:
    the rain indicator compares each frame with the previous one and is skipped after a gap.
    """

    name = "analyzer"

    def __init__(self, config: AnalyzerConfig | None = None, scenario: ScenarioEntry | None = None) -> None:
        self.scenario = scenario
        self.config = config if config is not None or scenario is not None else load_analyzer_config()
        self._previous_gray: np.ndarray | None = None
        self._previous_index: int | None = None

    def setup(self) -> None:
        self._previous_gray = None
        self._previous_index = None

    def estimate(self, ctx: FrameContext) -> DegradationEstimate:
        if self.scenario is not None:
            return DegradationEstimate(self.scenario.degradation, self.scenario.level, confidence=1.0)

        image = prepare(ctx.image)
        gray = to_gray(image)
        consecutive = self._previous_index is not None and ctx.index == self._previous_index + 1
        scores = compute_scores(image, gray, self._previous_gray if consecutive else None)
        self._previous_gray, self._previous_index = gray, ctx.index
        return decide(scores, self.config)

    def process(self, ctx: FrameContext) -> FrameContext:
        ctx.degradation = self.estimate(ctx)
        return ctx
