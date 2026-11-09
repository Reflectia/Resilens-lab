"""Measure latency and resource use of the processing modes on each resource profile (T-20, NFR-01/02/05).

For every scenario x mode x profile the pipeline runs `--repeats` times, one run after another, offline and
without annotations: only speed and resources are measured here, quality is the quality benchmark's job (T-12).
The first `warmup_frames` (10) frames of every run are left out of the latency statistics; `--warmup-runs`
discards whole runs before the measured ones (cold caches, first model load).

Each measured run is one row of the CSV; the console shows the median over repeats and how much the repeats
differ, so a noisy measurement is easy to spot.

    python benchmarks/profiles/measure.py --manifest data/manifest.json --modes A B D --profiles minimal standard \
        --scenario control_videos/MOT20-01/clean.mp4 --scenario control_videos/MOT20-01/fog_l2.mp4 --frames 150
"""

import argparse
import csv
import itertools
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil
import torch

from resilens.config.loader import build_run_config
from resilens.config.schema import Device
from resilens.core.pipeline import Pipeline
from resilens.io.replay import VideoReplay
from resilens.io.validation import VideoValidator
from resilens.metrics.latency import latency_metrics
from resilens.metrics.telemetry import ResourceSampler
from resilens.policy.log import policy_summary

OUT = Path(__file__).parent / "_out"
SUMMARY_KEYS = ("latency_p50_ms", "latency_p95_ms", "latency_fps", "peak_ram_mb", "peak_cpu_percent")


def measure_run(scenario: Path, mode: str, profile: str, frames: int | None) -> dict[str, float]:
    """One offline run of the pipeline; latency, resources and (mode D) policy action shares."""
    config = build_run_config(mode, profile, scenario)
    video = VideoValidator(max_resolution=config.profile.max_resolution).validate(scenario)
    pipeline = Pipeline.from_config(config)
    sampler = ResourceSampler(gpu=config.profile.device is Device.CUDA)
    replay = VideoReplay(video, config.replay_mode)
    sampler.start()
    started = time.perf_counter()
    try:
        processed = pipeline.run(itertools.islice(replay, frames) if frames else replay)
    finally:
        elapsed_s = time.perf_counter() - started
        resources = sampler.stop()
    return {
        "frames": float(len(processed)),
        "wall_s": elapsed_s,
        **latency_metrics(processed, config.profile.frame_budget_ms, config.warmup_frames),
        **resources,
        **policy_summary(processed),
    }


def environment() -> dict:
    """What the numbers depend on, saved next to them (NFR-04)."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def spread(values: list[float]) -> float:
    """(max - min) / median: how much the repeats of one measurement differ."""
    median = float(np.median(values))
    return (max(values) - min(values)) / median if median else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure latency and resources per mode and profile.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenario", action="append", help="video path from the manifest; default: all")
    parser.add_argument("--modes", nargs="+", default=["A", "B", "D"])
    parser.add_argument("--profiles", nargs="+", default=["minimal", "standard"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--frames", type=int, help="process only the first N frames of each video")
    parser.add_argument("--output", type=Path, default=OUT / "profiles.csv")
    args = parser.parse_args()

    entries = json.loads(args.manifest.read_text(encoding="utf-8"))["test_cases"]
    scenarios = [e["file"] for e in entries if not args.scenario or e["file"] in args.scenario]
    missing = set(args.scenario or []) - set(scenarios)
    if missing:
        raise SystemExit(f"Scenarios not in the manifest: {sorted(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")
    rows: list[dict] = []
    for scenario, profile, mode in itertools.product(scenarios, args.profiles, args.modes):
        video = args.manifest.parent / scenario
        for _ in range(args.warmup_runs):
            measure_run(video, mode, profile, args.frames)
        runs = [measure_run(video, mode, profile, args.frames) for _ in range(args.repeats)]
        rows += [
            {"scenario": scenario, "profile": profile, "mode": mode, "repeat": i + 1, **r} for i, r in enumerate(runs)
        ]
        summary = ", ".join(
            f"{key} {np.median([r[key] for r in runs]):.0f} (+/-{spread([r[key] for r in runs]):.0%})"
            for key in SUMMARY_KEYS
        )
        print(f"{scenario} {profile} {mode}: {summary}", flush=True)

    columns = list(dict.fromkeys(key for row in rows for key in row))
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} runs -> {args.output}")


if __name__ == "__main__":
    main()
