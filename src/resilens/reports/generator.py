"""HTML/JSON report of a run and comparison report of modes A–E."""

from pathlib import Path

from resilens.runs.store import RunStore


class ReportGenerator:
    def __init__(self, store: RunStore) -> None:
        self.store = store

    def run_report(self, run_id: str) -> Path:
        pass

    def comparison_report(self, run_ids: list[str], output: Path) -> Path:
        """Side-by-side metrics of several runs plus the Pareto plot."""
        pass

    def pareto_plot(self, run_ids: list[str], output: Path) -> Path:
        """Scatter of HOTA vs p95 latency with the Pareto front highlighted."""
        pass
