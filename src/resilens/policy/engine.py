from collections.abc import Callable

from resilens.config.schema import ProfileConfig
from resilens.core.contracts import (
    Action,
    BudgetState,
    DegradationEstimate,
    DegradationType,
    FrameContext,
    PolicyDecision,
)
from resilens.core.stage import Stage
from resilens.policy.rules import RuleTable

UNKNOWN = DegradationEstimate(DegradationType.UNKNOWN, level=0, confidence=0.0)


class PolicyEngine(Stage):
    """Chooses the action for each frame from the rule table and the profile's budget.

    UNKNOWN -> bypass. If the moving latency exceeds the profile's frame budget or RAM/CPU/GPU get near the
    profile's limits, the action is lowered one step (enhance -> skip -> bypass -> downscale). The decision
    only describes what to do; the enhancer applies it.
    """

    name = "policy"

    def __init__(
        self, rules: RuleTable, profile: ProfileConfig, budget_source: Callable[[], BudgetState] = BudgetState
    ) -> None:
        if profile.name not in rules.profiles():
            raise ValueError(f"Policy rules have no actions for profile '{profile.name}'")
        self.rules = rules
        self.profile = profile
        self.budget_source = budget_source
        self._skipped_frames = 0

    def setup(self) -> None:
        self._skipped_frames = 0

    def over_budget(self, budget: BudgetState) -> list[str]:
        """Why the budget correction applies; empty if it does not. Limits set to None are not checked."""
        profile, ratio = self.profile, self.rules.config.budget.near_limit_ratio
        reasons = []
        if None not in (profile.frame_budget_ms, budget.latency_ms) and budget.latency_ms > profile.frame_budget_ms:
            reasons.append(f"latency {budget.latency_ms:.0f} ms > budget {profile.frame_budget_ms:.0f} ms")
        for resource, value, limit, unit in (
            ("RAM", budget.ram_mb, profile.memory_limit_mb, "MB"),
            ("CPU", budget.cpu_percent, profile.cpu_limit_percent, "%"),
            ("GPU memory", budget.gpu_memory_mb, profile.gpu_memory_limit_mb, "MB"),
        ):
            if None not in (value, limit) and value >= ratio * limit:
                reasons.append(f"{resource} {value:.0f} {unit} >= {ratio:.0%} of limit {limit:.0f} {unit}")
        return reasons

    def decide(self, estimate: DegradationEstimate | None, budget: BudgetState) -> PolicyDecision:
        estimate = estimate or UNKNOWN
        base = self.rules.lookup(estimate.type, estimate.level, self.profile.name)
        reason = f"{estimate.type} level {estimate.level} on {self.profile.name}: {base}"

        action, reasons = base, self.over_budget(budget)
        if reasons:
            action = self.rules.downgrade(base)
            reason += f"; {', '.join(reasons)} -> {action}"

        enhance = action is Action.ENHANCE
        if action is Action.SKIP:
            enhance = self._skipped_frames % self.rules.config.skip_every_n == 0
            self._skipped_frames += 1
        resize_to = self.profile.downscale_resolution if action is Action.DOWNSCALE else None
        if action is Action.DOWNSCALE and resize_to is None:
            reason += " (profile has no downscale_resolution: frame unchanged)"

        return PolicyDecision(
            action=action,
            reason=reason,
            base_action=base,
            downgraded=action is not base,
            enhance=enhance,
            resize_to=resize_to,
            budget=budget,
        )

    def process(self, ctx: FrameContext) -> FrameContext:
        ctx.decision = self.decide(ctx.degradation, self.budget_source())
        return ctx
