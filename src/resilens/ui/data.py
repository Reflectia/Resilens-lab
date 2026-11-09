"""Turning API responses into tables for the Web UI."""

import io
from dataclasses import dataclass

import pandas as pd

MOT_COLUMNS = ["frame", "id", "x", "y", "w", "h", "score"]
POLICY_ACTIONS = ["enhance", "skip", "bypass", "downscale"]

# Fixed colour per mode and per action, so a colour always means the same thing (validated categorical palette)
MODE_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "E": "#e87ba4"}
ACTION_COLORS = dict(zip(POLICY_ACTIONS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], strict=True))
DETECTIONS_COLOR, TRACKS_COLOR = "#2a78d6", "#eb6834"  # the same colours as the boxes in the annotated video


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    fmt: str  # format spec, "%" for shares
    better: str | None = None  # "higher", "lower" or None when there is no better direction

    def format(self, value: float) -> str:
        return f"{value:.0%}" if self.fmt == "%" else f"{value:{self.fmt}}"


METRIC_GROUPS: dict[str, list[Metric]] = {
    "Performance": [
        Metric("fps", "FPS", ".2f", "higher"),
        Metric("processing_time_s", "Processing time, s", ".1f", "lower"),
        Metric("peak_ram_mb", "Peak RAM, MB", ".0f", "lower"),
        Metric("processed_frames", "Processed frames", ".0f"),
        Metric("frames_dropped", "Dropped frames", ".0f", "lower"),
    ],
    "Pipeline output": [
        Metric("detections_per_frame", "Detections per frame", ".1f"),
        Metric("tracks_per_frame", "Tracks per frame", ".1f"),
    ],
    "Detection quality": [
        Metric("map_50_95", "mAP@[.5:.95]", ".3f", "higher"),
        Metric("precision", "Precision", ".3f", "higher"),
        Metric("recall", "Recall", ".3f", "higher"),
        Metric("detection_confidence_threshold", "Evaluation confidence threshold", ".2f"),
    ],
    "Tracking quality": [
        Metric("hota", "HOTA", ".3f", "higher"),
        Metric("idf1", "IDF1", ".3f", "higher"),
        Metric("id_switches", "ID switches", ".0f", "lower"),
    ],
    "Policy Engine": [
        *(Metric(f"policy_{action}_share", f"Frames: {action}", "%") for action in POLICY_ACTIONS),
        Metric("policy_downgraded_share", "Frames downgraded by the budget", "%"),
        Metric("policy_enhanced_share", "Frames enhanced", "%"),
    ],
}
KNOWN_METRICS = {metric.key: metric for metrics in METRIC_GROUPS.values() for metric in metrics}
QUALITY_KEYS = {m.key for m in METRIC_GROUPS["Detection quality"] + METRIC_GROUPS["Tracking quality"]}


def metric_info(key: str) -> Metric:
    """Known metric, or a generic one so metrics added later still show up."""
    return KNOWN_METRICS.get(key) or Metric(key, key.replace("_", " ").capitalize(), ".3g")


def group_metrics(metrics: dict[str, float]) -> dict[str, list[tuple[Metric, float]]]:
    """Metrics of a report by group, in display order; unknown keys go to "Other". Empty groups are left out."""
    groups = {
        title: [(metric, metrics[metric.key]) for metric in group if metric.key in metrics]
        for title, group in METRIC_GROUPS.items()
    }
    groups["Other"] = [(metric_info(key), value) for key, value in metrics.items() if key not in KNOWN_METRICS]
    return {title: rows for title, rows in groups.items() if rows}


def run_label(run: dict) -> str:
    return f"{run['mode']} · {run['profile']} · {run.get('source_name', '?')} · {run['run_id'][:6]}"


def default_comparison(runs: list[dict]) -> list[str]:
    """Runs (newest first) to compare when the page opens: the newest run of each mode on the video of the newest
    run, so the comparison is of one video; if that video has a single mode, the two newest runs."""
    if not runs:
        return []
    video = runs[0]["source_name"]
    newest: dict[str, str] = {}
    for run in runs:
        if run["source_name"] == video:
            newest.setdefault(run["mode"], run["run_id"])
    if len(newest) < 2:
        return [run["run_id"] for run in runs[:2]]
    return [newest[mode] for mode in sorted(newest)]


def runs_table(runs: list[dict], status_icons: dict[str, str]) -> pd.DataFrame:
    """Runs list for display. Times are parsed as UTC: while every run is still active, finished_at is all null,
    and a column without values would otherwise come out tz-naive and not subtract from created_at."""
    table = pd.DataFrame(runs)
    table["created_at"] = pd.to_datetime(table["created_at"], utc=True)
    finished = pd.to_datetime(table["finished_at"], utc=True)
    table["duration_s"] = (finished - table["created_at"]).dt.total_seconds()  # includes the time spent queued
    table["status"] = table["status"].map(lambda status: f"{status_icons.get(status, '')} {status}")
    columns = ["run_id", "status", "mode", "profile", "source_name", "created_at", "duration_s", "error"]
    return table[columns]


def parse_mot(content: bytes) -> pd.DataFrame:
    """MOTChallenge rows (frame is 1-based); the -1 world-coordinate columns are dropped."""
    if not content.strip():
        return pd.DataFrame(columns=MOT_COLUMNS)
    table = pd.read_csv(io.BytesIO(content), header=None)
    return table.iloc[:, : len(MOT_COLUMNS)].set_axis(MOT_COLUMNS, axis=1)


def per_frame_counts(
    detections: pd.DataFrame, tracks: pd.DataFrame, frame_count: int, processed: list[int] | None = None
) -> pd.DataFrame:
    """Detections and tracks per frame, with zeros for processed frames that have none. In realtime replay pass the
    processed frames: dropped frames have no results, and counting them as zero would look like a detector failure."""
    frames = (
        pd.Index(processed, name="frame") if processed is not None else pd.RangeIndex(1, frame_count + 1, name="frame")
    )
    return pd.DataFrame(
        {
            "detections": detections.groupby("frame").size().reindex(frames, fill_value=0),
            "tracks": tracks.groupby("frame").size().reindex(frames, fill_value=0),
        }
    ).reset_index()


def track_summary(tracks: pd.DataFrame) -> pd.DataFrame:
    """One row per track ID: first and last frame, frames present, mean box size."""
    if tracks.empty:
        return pd.DataFrame(columns=["id", "first_frame", "last_frame", "frames", "mean_width", "mean_height"])
    return (
        tracks.groupby("id")
        .agg(
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
            frames=("frame", "size"),
            mean_width=("w", "mean"),
            mean_height=("h", "mean"),
        )
        .reset_index()
    )


def parse_policy_log(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content), keep_default_na=False)


def comparison_table(runs: list[dict], reports: list[dict]) -> pd.DataFrame:
    """Metrics (rows) of several runs (columns, labelled by run_label), known metrics first."""
    table = pd.DataFrame({run_label(run): report["metrics"] for run, report in zip(runs, reports, strict=True)})
    order = [key for key in KNOWN_METRICS if key in table.index] + sorted(set(table.index) - set(KNOWN_METRICS))
    return table.loc[order]


def relative_change(table: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Change of every run against the reference column, in percent; NaN where the reference is 0 or missing."""
    base = table[reference].where(table[reference] != 0)
    return table.sub(base, axis=0).div(base.abs(), axis=0) * 100
