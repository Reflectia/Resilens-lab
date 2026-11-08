"""Latency statistics: FPS, p50 / p95 and the share of frames within the frame budget."""

from dataclasses import dataclass

import numpy as np

from resilens.core.contracts import FrameContext


@dataclass(frozen=True)
class LatencySummary:
    fps: float  # frames per second the pipeline could sustain: 1000 / mean latency
    p50_ms: float
    p95_ms: float
    in_budget_ratio: float | None  # share of frames with latency <= frame budget; None without a budget


def after_warmup[T](values: list[T], warmup_frames: int) -> list[T]:
    """Values without the first `warmup_frames`; all of them if the run is shorter than the warm-up."""
    return values[warmup_frames:] or values


def summarize_latencies(
    latencies_ms: list[float], frame_budget_ms: float | None, warmup_frames: int = 0
) -> LatencySummary:
    if not latencies_ms:
        raise ValueError("No frame latencies to summarize")
    measured = np.array(after_warmup(latencies_ms, warmup_frames))
    p50, p95 = np.percentile(measured, [50, 95])
    return LatencySummary(
        fps=float(1000 / measured.mean()) if measured.mean() > 0 else float("inf"),
        p50_ms=float(p50),
        p95_ms=float(p95),
        in_budget_ratio=None if frame_budget_ms is None else float(np.mean(measured <= frame_budget_ms)),
    )


def frame_latency_ms(ctx: FrameContext) -> float:
    """End-to-end latency of a processed frame: all its stage timings, frame reading included."""
    return sum(timing.duration_ms for timing in ctx.timings)


def latency_metrics(frames: list[FrameContext], frame_budget_ms: float | None, warmup_frames: int) -> dict[str, float]:
    """Flat run metrics: end-to-end latency_* and stage_<name>_p50_ms / _p95_ms, all after warm-up."""
    summary = summarize_latencies([frame_latency_ms(ctx) for ctx in frames], frame_budget_ms, warmup_frames)
    metrics = {"latency_fps": summary.fps, "latency_p50_ms": summary.p50_ms, "latency_p95_ms": summary.p95_ms}
    if summary.in_budget_ratio is not None:
        metrics["latency_in_budget_ratio"] = summary.in_budget_ratio

    stage_ms: dict[str, list[float]] = {}
    for ctx in after_warmup(frames, warmup_frames):
        for timing in ctx.timings:
            stage_ms.setdefault(timing.stage, []).append(timing.duration_ms)
    for stage, durations in stage_ms.items():
        p50, p95 = np.percentile(durations, [50, 95])
        metrics[f"stage_{stage}_p50_ms"] = float(p50)
        metrics[f"stage_{stage}_p95_ms"] = float(p95)
    return metrics
