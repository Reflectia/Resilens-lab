from resilens.core.contracts import DegradationEstimate, FrameContext
from resilens.core.stage import Stage
from resilens.scenarios.manifest import ScenarioEntry


class DegradationAnalyzer(Stage):
    """Uses scenario metadata for control videos, heuristics for user videos.

    Low confidence -> UNKNOWN; several degradations -> the dominant one.
    """

    name = "analyzer"

    def __init__(self, scenario: ScenarioEntry | None = None, confidence_threshold: float = 0.5) -> None:
        self.scenario = scenario
        self.confidence_threshold = confidence_threshold

    def estimate(self, ctx: FrameContext) -> DegradationEstimate:
        pass

    def process(self, ctx: FrameContext) -> FrameContext:
        pass
