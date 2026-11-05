"""Per-frame policy log and run summary of the Policy Engine's decisions."""

import csv
from pathlib import Path

from resilens.config.schema import POLICY_ACTIONS
from resilens.core.contracts import FrameContext

LOG_COLUMNS = (
    "frame",
    "degradation",
    "level",
    "confidence",
    "profile",
    "base_action",
    "action",
    "downgraded",
    "enhanced",
    "latency_ms",
    "ram_mb",
    "cpu_percent",
    "gpu_memory_mb",
    "reason",
)


def _number(value: float | None, digits: int = 1) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def export_policy_log(frames: list[FrameContext], profile: str, output: Path) -> None:
    """One CSV row per frame with a decision; frame numbers are 1-based as in the MOT files."""
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(LOG_COLUMNS)
        for ctx in frames:
            decision = ctx.decision
            if decision is None:
                continue
            estimate, budget = ctx.degradation, decision.budget
            writer.writerow(
                [
                    ctx.index + 1,
                    estimate.type if estimate else "",
                    estimate.level if estimate else "",
                    _number(estimate.confidence if estimate else None, 2),
                    profile,
                    decision.base_action or "",
                    decision.action,
                    int(decision.downgraded),
                    int(decision.enhance),
                    _number(budget.latency_ms),
                    _number(budget.ram_mb),
                    _number(budget.cpu_percent),
                    _number(budget.gpu_memory_mb),
                    decision.reason,
                ]
            )


def policy_summary(frames: list[FrameContext]) -> dict[str, float]:
    """Share of frames per final action, downgraded by the budget correction and actually enhanced."""
    decisions = [ctx.decision for ctx in frames if ctx.decision is not None]
    if not decisions:
        return {}
    summary = {
        f"policy_{action}_share": sum(d.action is action for d in decisions) / len(decisions)
        for action in POLICY_ACTIONS
    }
    summary["policy_downgraded_share"] = sum(d.downgraded for d in decisions) / len(decisions)
    summary["policy_enhanced_share"] = sum(d.enhance for d in decisions) / len(decisions)
    return summary
