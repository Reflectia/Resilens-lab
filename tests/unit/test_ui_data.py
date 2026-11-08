import math

import pandas as pd

from resilens.ui.data import (
    comparison_table,
    default_comparison,
    group_metrics,
    metric_info,
    parse_mot,
    parse_policy_log,
    per_frame_counts,
    relative_change,
    run_label,
    runs_table,
    track_summary,
)

TRACKS = b"1,1,10.0,20.0,30.0,40.0,0.900,-1,-1,-1\n1,2,0,0,10,10,0.5,-1,-1,-1\n3,1,12,20,30,42,0.8,-1,-1,-1\n"


def run(mode: str, run_id: str) -> dict:
    return {"run_id": run_id, "mode": mode, "profile": "minimal", "source_name": "fog_l2.mp4"}


def test_metrics_are_grouped_in_display_order_with_unknown_keys_last() -> None:
    groups = group_metrics({"tracks_per_frame": 3.0, "fps": 4.5, "hota": 0.41, "new_metric": 1.0})

    assert list(groups) == ["Performance", "Pipeline output", "Tracking quality", "Other"]
    assert [(m.label, m.format(v)) for m, v in groups["Tracking quality"]] == [("HOTA", "0.410")]
    assert groups["Other"][0][0].label == "New metric"


def test_shares_are_formatted_as_percent() -> None:
    assert metric_info("policy_enhance_share").format(0.375) == "38%"
    assert metric_info("fps").better == "higher" and metric_info("id_switches").better == "lower"


def test_parse_mot_and_per_frame_counts_fill_missing_frames() -> None:
    tracks = parse_mot(TRACKS)
    detections = parse_mot(b"")

    counts = per_frame_counts(detections, tracks, frame_count=4)

    assert list(tracks.columns) == ["frame", "id", "x", "y", "w", "h", "score"]
    assert counts.to_dict("list") == {"frame": [1, 2, 3, 4], "detections": [0, 0, 0, 0], "tracks": [2, 0, 1, 0]}


def test_track_summary() -> None:
    summary = track_summary(parse_mot(TRACKS))

    first = summary.iloc[0]
    assert (first["id"], first["first_frame"], first["last_frame"], first["frames"]) == (1, 1, 3, 2)
    assert first["mean_height"] == 41.0
    assert track_summary(parse_mot(b"")).empty


def test_policy_log_keeps_empty_cells_as_text() -> None:
    log = parse_policy_log(b"frame,degradation,action,latency_ms\n1,,bypass,\n2,fog,enhance,120.5\n")

    assert log["degradation"].tolist() == ["", "fog"]


def test_comparison_table_and_change_against_reference() -> None:
    runs = [run("A", "a" * 32), run("D", "d" * 32)]
    reports = [{"metrics": {"fps": 4.0, "custom": 0.0}}, {"metrics": {"fps": 5.0, "custom": 1.0, "hota": 0.5}}]

    table = comparison_table(runs, reports)
    change = relative_change(table, run_label(runs[0]))

    assert list(table.index) == ["fps", "hota", "custom"]
    assert list(table.columns) == ["A · minimal · fog_l2.mp4 · aaaaaa", "D · minimal · fog_l2.mp4 · dddddd"]
    assert change.loc["fps"].tolist() == [0.0, 25.0]
    assert math.isnan(change.at["custom", table.columns[1]])  # reference value 0
    assert pd.isna(table.at["hota", table.columns[0]])


def test_runs_table_while_no_run_has_finished() -> None:
    queued = run("A", "a" * 32) | {
        "status": "queued",
        "created_at": "2026-11-08T19:21:06Z",
        "finished_at": None,
        "error": None,
    }

    table = runs_table([queued], {"queued": "Q"})

    assert table.at[0, "status"] == "Q queued"
    assert pd.isna(table.at[0, "duration_s"])


def test_runs_table_duration() -> None:
    done = run("A", "a" * 32) | {
        "status": "completed",
        "created_at": "2026-11-08T19:21:06Z",
        "finished_at": "2026-11-08T19:21:18.5Z",
        "error": None,
    }

    assert runs_table([done], {})["duration_s"].tolist() == [12.5]


def test_per_frame_counts_in_realtime_keep_only_processed_frames() -> None:
    counts = per_frame_counts(parse_mot(b""), parse_mot(TRACKS), frame_count=10, processed=[1, 7])

    assert counts.to_dict("list") == {"frame": [1, 7], "detections": [0, 0], "tracks": [2, 0]}


def test_default_comparison_is_one_video_with_the_newest_run_per_mode() -> None:
    runs = [  # newest first
        run("D", "d2") | {"source_name": "fog.mp4"},
        run("B", "b1") | {"source_name": "rain.mp4"},
        run("D", "d1") | {"source_name": "fog.mp4"},
        run("A", "a1") | {"source_name": "fog.mp4"},
    ]

    assert default_comparison(runs) == ["a1", "d2"]
    assert default_comparison(runs[:2]) == ["d2", "b1"]  # one mode on fog.mp4: the two newest runs
    assert default_comparison([]) == []
