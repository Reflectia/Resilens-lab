"""Compare completed benchmark runs independently of the command-line interface."""

import json
from pathlib import Path

import numpy as np

from resilens.metrics.aggregate import aggregate_runs
from resilens.runs.fingerprint import file_sha256
from resilens.runs.models import RunStatus
from resilens.runs.store import RunStore

REFERENCE_MODE = "A"  # every mode is compared with mode A (NFR-05)
QUALITY_KEYS = ("map_50_95", "precision", "recall", "hota", "idf1", "id_switches", "detection_confidence_threshold")
# speed and resources of one run; groups report the mean over their runs, peaks the maximum
SPEED_KEYS = ("latency_fps", "latency_p50_ms", "latency_p95_ms", "latency_in_budget_ratio", "mean_cpu_percent")
PEAK_KEYS = ("peak_ram_mb", "peak_cpu_percent", "peak_gpu_memory_mb")


def speed_summary(metrics: list[dict[str, float]]) -> dict[str, float]:
    """Mean latency figures and maximum peaks over runs; keys missing in older runs are left out."""
    summary = {key: float(np.mean([m[key] for m in metrics])) for key in SPEED_KEYS if all(key in m for m in metrics)}
    summary |= {key: float(max(m[key] for m in metrics)) for key in PEAK_KEYS if all(key in m for m in metrics)}
    return summary


def compare_runs(store: RunStore, run_ids: list[str], benchmark_path: Path | None, manifest_path: Path | None) -> dict:
    run_ids = list(run_ids)
    excluded = []
    if benchmark_path:
        entries = json.loads(benchmark_path.read_text(encoding="utf-8"))
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
        group = by_video.setdefault(fingerprint, {})
        if record.mode in group:
            raise ValueError("Duplicate mode for video")
        group[record.mode] = record.run_id
    modes = sorted({mode for runs in by_video.values() for mode in runs})
    if REFERENCE_MODE not in modes or len(modes) < 2:
        raise ValueError(f"Comparison needs mode {REFERENCE_MODE} (the reference) and at least one other mode")
    for video_hash, runs in list(by_video.items()):
        if set(runs) != set(modes):
            reason = f"video has no completed run for every compared mode ({', '.join(modes)})"
            excluded.extend({"run_id": run_id, "reason": reason} for run_id in runs.values())
            del by_video[video_hash]
    if not by_video:
        raise ValueError(f"No video has completed runs for every compared mode ({', '.join(modes)})")
    if not manifest_path:
        raise ValueError("A manifest is required for grouping by distortion and level")
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))["test_cases"]
    manifest = {str((manifest_path.parent / e["file"]).resolve()): e for e in entries}
    scenarios = []
    groups = {}
    for pair in by_video.values():
        source = str(store.load_config(pair[REFERENCE_MODE]).source.resolve())
        entry = manifest.get(source)
        if entry is None:
            raise ValueError(f"Video is absent from manifest: {source}")
        expected = {
            "annotations_sha256": file_sha256(manifest_path.parent / entry["gt_coco"]),
            "tracking_annotations_sha256": file_sha256(manifest_path.parent / entry["gt_mot"]),
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
            item[mode] = {
                "run_id": run_id,
                "metrics": {key: metrics[key] for key in QUALITY_KEYS},
                "speed": speed_summary([metrics]),
            }
        scenarios.append(item)
        key = (entry["distortion"], entry["level"])
        groups.setdefault(key, []).append(pair)

    def summarize(pairs):
        return {mode: aggregate_runs(store, [pair[mode] for pair in pairs]) for mode in modes}

    def summarize_speed(pairs):
        return {mode: speed_summary([store.load_metrics(pair[mode]) for pair in pairs]) for mode in modes}

    report = {
        "modes": modes,
        "excluded": excluded,
        "scenarios": scenarios,
        "groups": [
            {"distortion": kind, "level": level, "metrics": summarize(pairs), "speed": summarize_speed(pairs)}
            for (kind, level), pairs in groups.items()
        ],
        "overall": summarize(list(by_video.values())),
        "overall_speed": summarize_speed(list(by_video.values())),
        "notes": [
            "mAP uses detections with confidence >= 0.3",
            "Tracking uses full person ground truth, including score-zero tracks.",
            "FlowTracker cannot add later objects or recover lost targets.",
        ],
    }
    return report
