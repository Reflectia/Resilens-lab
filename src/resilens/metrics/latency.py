"""Latency statistics: FPS, p50 / p95 and the share of frames within the frame budget."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LatencySummary:
    fps: float
    p50_ms: float
    p95_ms: float
    in_budget_ratio: float | None


def summarize_latencies(
    latencies_ms: list[float], frame_budget_ms: float | None, warmup_frames: int = 0
) -> LatencySummary:
    pass
