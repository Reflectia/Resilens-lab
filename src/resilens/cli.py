import argparse
import json
import sys
from pathlib import Path

import numpy as np
import psutil

from resilens.config.config import RUNS_DIR
from resilens.io.replay import VideoReplay
from resilens.io.validation import VideoValidationError, VideoValidator
from resilens.reports.generator import ReportGenerator
from resilens.runs.models import RunNotFoundError, RunStatus
from resilens.runs.store import RunStore

FRAME_TABLE_HEADER = f"{'frame':>7} {'time_s':>9} {'read_ms':>9} {'avg_ms':>9} {'fps':>8} {'ram_mb':>9}"


def cmd_inspect(args: argparse.Namespace) -> int:
    """Validate a video and replay all its frames offline; --frames prints per-frame statistics."""
    validator = VideoValidator(max_duration_s=args.max_duration, max_resolution=args.max_resolution)
    try:
        info = validator.validate(args.video)
    except VideoValidationError as error:
        print(f"Invalid video: {error}", file=sys.stderr)
        return 1

    print(f"{info.path}: {info.width}x{info.height}, {info.fps:.2f} FPS, {info.frame_count} frames")
    replay = VideoReplay(info)
    process = psutil.Process()
    read_times_ms: list[float] = []
    peak_ram_mb = 0.0
    last_timestamp_s = 0.0
    if args.frames:
        print(FRAME_TABLE_HEADER)
    for frame in replay:
        last_timestamp_s = frame.timestamp_s
        read_times_ms.append(sum(t.duration_ms for t in frame.timings if t.stage == VideoReplay.stage_name))
        if args.frames:
            ram_mb = process.memory_info().rss / 2**20
            peak_ram_mb = max(peak_ram_mb, ram_mb)
            avg_ms = sum(read_times_ms) / len(read_times_ms)
            fps = 1000 / avg_ms if avg_ms > 0 else float("inf")
            print(
                f"{frame.index:>7} {frame.timestamp_s:>9.3f} {read_times_ms[-1]:>9.2f} "
                f"{avg_ms:>9.2f} {fps:>8.1f} {ram_mb:>9.1f}"
            )
    print(f"Replayed {replay.frames_read} frames, last timestamp {last_timestamp_s:.3f} s of {info.duration_s:.2f} s")
    if args.frames and read_times_ms:
        p50, p95 = np.percentile(read_times_ms, [50, 95])
        print(
            f"Read time per frame: avg {np.mean(read_times_ms):.2f} ms, p50 {p50:.2f} ms, "
            f"p95 {p95:.2f} ms, max {max(read_times_ms):.2f} ms; peak RAM {peak_ram_mb:.1f} MB"
        )
    if replay.ended_early:
        print(f"Warning: video ended early ({replay.frames_read} of {info.frame_count} frames)", file=sys.stderr)
    return 0


def cmd_run(args: argparse.Namespace) -> None:
    pass


def cmd_benchmark(args: argparse.Namespace) -> None:
    pass


def cmd_report(args: argparse.Namespace) -> int:
    """Print completed run reports already saved in the run store."""
    store = RunStore(RUNS_DIR)
    reports = ReportGenerator(store)
    for run_id in args.run_ids:
        try:
            record = store.get(run_id)
            if record.status != RunStatus.COMPLETED:
                raise ValueError(f"Run '{run_id}' is {record.status}; no completed report is available")
            print(json.dumps(reports.load_report(run_id), indent=2))
        except (RunNotFoundError, FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            print(f"Cannot show report for '{run_id}': {error}", file=sys.stderr)
            return 1
    return 0


def cmd_build_scenarios(args: argparse.Namespace) -> None:
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="resilens")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="validate a video and replay its frames")
    inspect.add_argument("video", type=Path)
    inspect.add_argument("--max-duration", type=float, help="maximum duration in seconds")
    inspect.add_argument("--max-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    inspect.add_argument(
        "--frames", action="store_true", help="print read time, running average, FPS and RAM for every frame"
    )
    inspect.set_defaults(func=cmd_inspect)

    run = sub.add_parser("run", help="process one video")
    run.add_argument("video", type=Path)
    run.add_argument("--mode", default="A", choices=["A", "B", "C", "D", "E"])
    run.add_argument("--profile", default="minimal")
    run.add_argument("--annotations", type=Path)
    run.add_argument("--replay", default="offline", choices=["offline", "realtime"])
    run.set_defaults(func=cmd_run)

    bench = sub.add_parser("benchmark", help="compare modes on scenarios")
    bench.add_argument("--modes", nargs="+", default=["A", "B", "C", "D", "E"])
    bench.add_argument("--profile", default="minimal")
    bench.add_argument("--repeats", type=int, default=3)
    bench.set_defaults(func=cmd_benchmark)

    report = sub.add_parser("report", help="build reports")
    report.add_argument("run_ids", nargs="+")
    report.set_defaults(func=cmd_report)

    scen = sub.add_parser("build-scenarios", help="generate degraded control clips")
    scen.add_argument("source_dir", type=Path)
    scen.add_argument("--seed", type=int, default=0)
    scen.set_defaults(func=cmd_build_scenarios)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    exit_code = args.func(args)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
