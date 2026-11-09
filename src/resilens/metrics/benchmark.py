"""Run benchmark scenarios independently of command-line argument parsing."""

import json
from collections.abc import Callable
from pathlib import Path

from resilens.config.loader import build_run_config
from resilens.runs.manager import RunManager
from resilens.runs.models import RunStatus
from resilens.runs.store import RunStore, _write_json


def _verify_benchmark_resume(
    manager: RunManager, existing: dict, entry: dict, manifest_path: Path, profile: str
) -> None:
    run_id = existing["run_id"]
    record = manager.store.get(run_id)
    config = manager.store.load_config(run_id)
    if record.status != RunStatus.COMPLETED or config.profile.name != profile:
        raise ValueError(f"Saved benchmark run is incompatible: {run_id}")
    current_source = (manifest_path.parent / entry["file"]).resolve()
    if config.source.resolve() != current_source:
        raise ValueError(f"Saved benchmark source differs: {run_id}")
    if (
        config.annotations.resolve() != (manifest_path.parent / entry["gt_coco"]).resolve()
        or config.tracking_annotations.resolve() != (manifest_path.parent / entry["gt_mot"]).resolve()
    ):
        raise ValueError(f"Saved benchmark annotations differ: {run_id}")
    manager.store.verify_source(run_id)
    manager.store.verify_annotations(run_id)


INTERRUPTED = "interrupted: the benchmark stopped before this run finished"


def _fail_interrupted_runs(store: RunStore, results: list[dict]) -> None:
    """Runs this benchmark had started when its process died (crash, out of memory) stay queued or running
    forever; mark them failed so they do not look active. Runs of the API are not in `results`."""
    for result in results:
        if result["status"] in (RunStatus.QUEUED, RunStatus.RUNNING):
            if store.get(result["run_id"]).status in (RunStatus.QUEUED, RunStatus.RUNNING):
                store.update_status(result["run_id"], RunStatus.FAILED, error=INTERRUPTED)
            result.update(status=RunStatus.FAILED.value, error=INTERRUPTED)


def run_benchmark(
    store: RunStore,
    manifest_path: Path,
    profile: str,
    modes: list[str],
    selected_scenarios: list[str] | None,
    output_path: Path,
    on_result: Callable[[dict], None],
) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenarios = manifest["test_cases"]
    if selected_scenarios:
        scenarios = [entry for entry in scenarios if entry["file"] in selected_scenarios]
        if len(scenarios) != len(set(selected_scenarios)):
            raise ValueError("Some requested scenarios were not found")
    manager = RunManager(store, recover_interrupted=False)
    results = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else []
    _fail_interrupted_runs(store, results)
    for entry in scenarios:
        for mode in modes:
            existing = next(
                (
                    r
                    for r in results
                    if r["scenario"] == entry["file"] and r["mode"] == mode and r["status"] == "completed"
                ),
                None,
            )
            if existing:
                _verify_benchmark_resume(manager, existing, entry, manifest_path, profile)
                continue
            results = [r for r in results if not (r["scenario"] == entry["file"] and r["mode"] == mode)]
            config = build_run_config(
                mode,
                profile,
                manifest_path.parent / entry["file"],
                annotations=manifest_path.parent / entry["gt_coco"],
                tracking_annotations=manifest_path.parent / entry["gt_mot"],
            )
            run_id = manager.submit(config)
            result = {"scenario": entry["file"], "mode": mode, "run_id": run_id, "status": "running", "error": None}
            results.append(result)
            _write_json(output_path, results)  # saved before it runs, so a crash leaves a trace to clean up
            manager.execute(run_id)
            record = manager.store.get(run_id)
            result.update(status=record.status, error=record.error)
            on_result(result)
            _write_json(output_path, results)
    return 1 if any(result["status"] != RunStatus.COMPLETED for result in results) else 0
