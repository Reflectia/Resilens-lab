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
from resilens.runs.store import RunStore, _write_json

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


def cmd_run(args: argparse.Namespace) -> int:
    from resilens.config.loader import build_run_config
    from resilens.runs.manager import RunManager

    manager = RunManager(RunStore(RUNS_DIR), recover_interrupted=False)
    config = build_run_config(
        args.mode,
        args.profile,
        args.video,
        annotations=args.annotations,
        tracking_annotations=args.tracking_annotations,
        replay_mode=args.replay,
    )
    run_id = manager.submit(config)
    manager.execute(run_id)
    record = manager.store.get(run_id)
    print(json.dumps({"run_id": run_id, "status": record.status, "error": record.error}))
    return 0 if record.status == RunStatus.COMPLETED else 1


def _verify_benchmark_resume(manager, existing: dict, entry: dict, args: argparse.Namespace) -> None:
    run_id = existing["run_id"]
    record = manager.store.get(run_id)
    config = manager.store.load_config(run_id)
    if record.status != RunStatus.COMPLETED or config.profile.name != args.profile:
        raise ValueError(f"Saved benchmark run is incompatible: {run_id}")
    current_source = (args.manifest.parent / entry["file"]).resolve()
    if config.source.resolve() != current_source:
        raise ValueError(f"Saved benchmark source differs: {run_id}")
    if (
        config.annotations.resolve() != (args.manifest.parent / entry["gt_coco"]).resolve()
        or config.tracking_annotations.resolve() != (args.manifest.parent / entry["gt_mot"]).resolve()
    ):
        raise ValueError(f"Saved benchmark annotations differ: {run_id}")
    manager.store.verify_source(run_id)
    manager.store.verify_annotations(run_id)


def cmd_benchmark(args: argparse.Namespace) -> int:
    from resilens.config.loader import build_run_config
    from resilens.runs.manager import RunManager

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scenarios = manifest["test_cases"]
    if args.scenario:
        scenarios = [entry for entry in scenarios if entry["file"] in args.scenario]
        if len(scenarios) != len(set(args.scenario)):
            raise ValueError("Some requested scenarios were not found")
    manager = RunManager(RunStore(RUNS_DIR), recover_interrupted=False)
    results = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else []
    for entry in scenarios:
        for mode in args.modes:
            existing = next(
                (
                    r
                    for r in results
                    if r["scenario"] == entry["file"] and r["mode"] == mode and r["status"] == "completed"
                ),
                None,
            )
            if existing:
                _verify_benchmark_resume(manager, existing, entry, args)
                continue
            results = [r for r in results if not (r["scenario"] == entry["file"] and r["mode"] == mode)]
            config = build_run_config(
                mode,
                args.profile,
                args.manifest.parent / entry["file"],
                annotations=args.manifest.parent / entry["gt_coco"],
                tracking_annotations=args.manifest.parent / entry["gt_mot"],
            )
            run_id = manager.submit(config)
            manager.execute(run_id)
            record = manager.store.get(run_id)
            result = {
                "scenario": entry["file"],
                "mode": mode,
                "run_id": run_id,
                "status": record.status,
                "error": record.error,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            _write_json(args.output, results)
    return 1 if any(result["status"] != RunStatus.COMPLETED for result in results) else 0


def cmd_compare(args: argparse.Namespace) -> int:
    from resilens.metrics.aggregate import aggregate_runs

    store = RunStore(RUNS_DIR)
    try:
        run_ids = list(args.run_ids)
        excluded = []
        if args.benchmark:
            entries = json.loads(args.benchmark.read_text(encoding="utf-8"))
            excluded.extend(
                {"run_id": entry["run_id"], "reason": entry["status"]}
                for entry in entries
                if entry["status"] != "completed"
            )
            run_ids.extend(entry["run_id"] for entry in entries if entry["status"] == "completed")
        if not run_ids:
            raise ValueError("No runs provided")
        records = [store.get(run_id) for run_id in run_ids]
        by_video = {}
        for record in records:
            if record.status != RunStatus.COMPLETED:
                excluded.append({"run_id": record.run_id, "reason": record.status.value})
                continue
            fingerprint = json.loads((store.run_dir(record.run_id) / "fingerprint.json").read_text())["source_sha256"]
            if record.mode not in ("A", "B"):
                raise ValueError("Only A and B runs can be compared")
            group = by_video.setdefault(fingerprint, {})
            if record.mode in group:
                raise ValueError("Duplicate mode for video")
            group[record.mode] = record.run_id
        for video_hash, pair in list(by_video.items()):
            if set(pair) != {"A", "B"}:
                excluded.extend(
                    {"run_id": run_id, "reason": "video has no completed A/B pair"} for run_id in pair.values()
                )
                del by_video[video_hash]
        if not by_video:
            raise ValueError("No completed A/B video pairs available")
        if not args.manifest:
            raise ValueError("A manifest is required for grouping by distortion and level")
        entries = json.loads(args.manifest.read_text(encoding="utf-8"))["test_cases"]
        manifest = {str((args.manifest.parent / e["file"]).resolve()): e for e in entries}
        scenarios = []
        groups = {}
        for pair in by_video.values():
            source = str(store.load_config(pair["A"]).source.resolve())
            entry = manifest.get(source)
            if entry is None:
                raise ValueError(f"Video is absent from manifest: {source}")
            from resilens.runs.fingerprint import file_sha256

            expected = {
                "annotations_sha256": file_sha256(args.manifest.parent / entry["gt_coco"]),
                "tracking_annotations_sha256": file_sha256(args.manifest.parent / entry["gt_mot"]),
            }
            for run_id in pair.values():
                fingerprint = json.loads((store.run_dir(run_id) / "fingerprint.json").read_text(encoding="utf-8"))
                if any(fingerprint.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"Run annotations do not match manifest: {run_id}")
            item = {"scenario": entry["file"]}
            for mode, run_id in pair.items():
                metrics = store.load_metrics(run_id)
                if not {"map_50_95", "hota", "idf1"} <= metrics.keys():
                    raise ValueError("Runs must have quality metrics")
                keys = (
                    "map_50_95",
                    "precision",
                    "recall",
                    "hota",
                    "idf1",
                    "id_switches",
                    "detection_confidence_threshold",
                )
                item[mode] = {"run_id": run_id, "metrics": {key: metrics[key] for key in keys}}
            scenarios.append(item)
            key = (entry["distortion"], entry["level"])
            groups.setdefault(key, []).append(pair)

        def summarize(pairs):
            return {mode: aggregate_runs(store, [pair[mode] for pair in pairs]) for mode in ("A", "B")}

        report = {
            "excluded": excluded,
            "scenarios": scenarios,
            "groups": [
                {"distortion": kind, "level": level, "metrics": summarize(pairs)}
                for (kind, level), pairs in groups.items()
            ],
            "overall": summarize(list(by_video.values())),
            "notes": [
                "mAP uses detections with confidence >= 0.3",
                "Tracking uses full person ground truth, including score-zero tracks.",
                "FlowTracker cannot add later objects or recover lost targets.",
            ],
        }
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        return 0
    except (OSError, KeyError, ValueError) as error:
        print(f"Cannot compare runs: {error}", file=sys.stderr)
        return 1


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
    run.add_argument("--tracking-annotations", type=Path)
    run.add_argument("--replay", default="offline", choices=["offline", "realtime"])
    run.set_defaults(func=cmd_run)

    bench = sub.add_parser("benchmark", help="compare modes on scenarios")
    bench.add_argument("--modes", nargs="+", choices=["A", "B"], default=["A", "B"])
    bench.add_argument("--profile", default="minimal")
    bench.add_argument("--manifest", type=Path, required=True)
    bench.add_argument("--scenario", action="append", help="video path from manifest; repeat to select multiple")
    bench.add_argument("--output", type=Path, required=True)
    bench.set_defaults(func=cmd_benchmark)

    report = sub.add_parser("report", help="build reports")
    report.add_argument("run_ids", nargs="+")
    report.set_defaults(func=cmd_report)

    compare = sub.add_parser("compare", help="compare completed A and B runs")
    compare.add_argument("run_ids", nargs="*")
    compare.add_argument("--benchmark", type=Path, help="benchmark JSON containing completed run IDs")
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--manifest", type=Path, help="manifest used for degradation grouping")
    compare.set_defaults(func=cmd_compare)

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
