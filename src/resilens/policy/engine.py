from resilens.core.contracts import DegradationEstimate, FrameContext, PolicyDecision
from resilens.core.stage import Stage
from resilens.policy.rules import RuleTable


class PolicyEngine(Stage):
    """UNKNOWN -> bypass. If moving-average latency exceeds the budget, the action is downgraded"""

    name = "policy"

    def __init__(self, rules: RuleTable, profile: str, frame_budget_ms: float | None) -> None:
        self.rules = rules
        self.profile = profile
        self.frame_budget_ms = frame_budget_ms

    def decide(self, estimate: DegradationEstimate, recent_latency_ms: float) -> PolicyDecision:
        pass

    def process(self, ctx: FrameContext) -> FrameContext:
        pass
