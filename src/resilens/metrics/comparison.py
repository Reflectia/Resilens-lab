"""Compare completed benchmark runs independently of the command-line interface."""

import json
from pathlib import Path

from resilens.metrics.aggregate import aggregate_runs
from resilens.runs.models import RunStatus
from resilens.runs.store import RunStore


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
        if record.mode not in ("A", "B"):
            raise ValueError("Only A and B runs can be compared")
        group = by_video.setdefault(fingerprint, {})
        if record.mode in group:
            raise ValueError("Duplicate mode for video")
        group[record.mode] = record.run_id
    for video_hash, pair in list(by_video.items()):
        if set(pair) != {"A", "B"}:
            excluded.extend({"run_id": run_id, "reason": "video has no completed A/B pair"} for run_id in pair.values())
            del by_video[video_hash]
    if not by_video:
        raise ValueError("No completed A/B video pairs available")
    if not manifest_path:
        raise ValueError("A manifest is required for grouping by distortion and level")
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))["test_cases"]
    manifest = {str((manifest_path.parent / e["file"]).resolve()): e for e in entries}
    scenarios = []
    groups = {}
    for pair in by_video.values():
        source = str(store.load_config(pair["A"]).source.resolve())
        entry = manifest.get(source)
        if entry is None:
            raise ValueError(f"Video is absent from manifest: {source}")
        from resilens.runs.fingerprint import file_sha256

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
            {"distortion": kind, "level": level, "metrics": summarize(pairs)} for (kind, level), pairs in groups.items()
        ],
        "overall": summarize(list(by_video.values())),
        "notes": [
            "mAP uses detections with confidence >= 0.3",
            "Tracking uses full person ground truth, including score-zero tracks.",
            "FlowTracker cannot add later objects or recover lost targets.",
        ],
    }
    return report
