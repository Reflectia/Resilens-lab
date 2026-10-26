"""HTML/JSON report of a run and comparison report of modes A–E."""

from pathlib import Path

from resilens.runs.store import RunStore


class ReportGenerator:
    def __init__(self, store: RunStore) -> None:
        self.store = store

    def run_report(self, run_id: str) -> Path:
        record = self.store.get(run_id)
        metrics = self.store.load_metrics(run_id)
        required = {"processing_time_s", "fps", "peak_ram_mb", "processed_frames"}
        if not required <= metrics.keys():
            raise ValueError(f"Run {run_id} has incomplete metrics")
        return self.store.save_report(
            run_id,
            {"run_id": run_id, "mode": record.mode, "profile": record.profile, "metrics": metrics},
        )

    def load_report(self, run_id: str) -> dict:
        return self.store.load_report(run_id)

    def comparison_report(self, run_ids: list[str], output: Path) -> Path:
        """Side-by-side metrics of several runs plus the Pareto plot."""
        raise NotImplementedError("Run comparisons are not implemented yet")

    def pareto_plot(self, run_ids: list[str], output: Path) -> Path:
        """Scatter of HOTA vs p95 latency with the Pareto front highlighted."""
        raise NotImplementedError("Run comparisons are not implemented yet")
