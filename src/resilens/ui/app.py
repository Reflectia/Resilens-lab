"""Streamlit Web UI (FR-09): upload a video, start a run, watch it with bounding boxes, compare runs.

The UI talks to the REST API only (RESILENS_API_URL); it never reads runs/ directly.
"""

import json

import altair as alt
import pandas as pd
import streamlit as st

from resilens.ui import data
from resilens.ui.api_client import ApiClient, ApiError

ACTIVE_STATUSES = {"queued", "running"}
STATUS_ICONS = {"queued": "⏳", "running": "⚙️", "completed": "✅", "failed": "❌"}
REPLAY_HELP = {
    "offline": "Every frame is processed. Use it to measure quality.",
    "realtime": "Frames arrive at the video's own fps; the ones the pipeline is too slow for are dropped.",
}


def get_client() -> ApiClient:
    # Keyed by the class: when Streamlit reloads the edited api_client module, a client cached from the old module
    # would raise the old ApiError class, which the except clauses here no longer catch
    return _cached_client(id(ApiClient))


@st.cache_resource
def _cached_client(class_id: int) -> ApiClient:
    return ApiClient()


# Results of a completed run never change, so they are cached per run
@st.cache_data(show_spinner=False, max_entries=64)
def cached_report(run_id: str) -> dict:
    return get_client().get_report(run_id)


@st.cache_data(show_spinner=False, max_entries=128)
def cached_file(run_id: str, name: str) -> bytes:
    return get_client().get_file(run_id, name)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_video(run_id: str) -> bytes:
    return get_client().get_video(run_id)


def stage_chain(mode: dict) -> str:
    stages = ["Analyzer"] if mode["use_analyzer"] else []
    stages += ["Policy Engine"] if mode["use_policy"] else []
    stages += [mode["enhancer"], mode["detector"], mode["tracker"]]
    return " → ".join(stages)


def profile_limits(profile: dict) -> pd.DataFrame:
    rows = {
        "Device": profile["device"],
        "Max resolution": profile["max_resolution"],
        "Downscale to": profile["downscale_resolution"],
        "Frame budget, ms": profile["frame_budget_ms"],
        "Memory limit, MB": profile["memory_limit_mb"],
        "CPU limit, %": profile["cpu_limit_percent"],
        "GPU memory limit, MB": profile["gpu_memory_limit_mb"],
        "CPU threads": profile["cpu_threads"],
    }

    def show(value) -> str:
        if value is None:
            return "no limit"
        return "×".join(map(str, value)) if isinstance(value, list) else str(value)

    return pd.DataFrame({"Setting": rows.keys(), "Value": [show(v) for v in rows.values()]})


# --------------------------------------------------------------------------- new run
def render_run_form(client: ApiClient) -> None:
    modes = {mode["name"]: mode for mode in client.list_modes()}
    order = {"minimal": 0, "standard": 1, "performance": 2}  # least to most resources; unknown profiles last
    listed = sorted(
        client.list_profiles(), key=lambda profile: (order.get(profile["name"], len(order)), profile["name"])
    )
    profiles = {profile["name"]: profile for profile in listed}

    video = st.file_uploader("Video", type=["mp4", "avi"], help="MP4 or AVI. The API checks that it can be decoded.")
    mode_column, profile_column = st.columns(2)
    with mode_column:
        mode = st.selectbox(
            "Mode", list(modes), format_func=lambda name: f"{name} — {modes[name]['description'] or name}"
        )
        st.caption(f"Stages: {stage_chain(modes[mode])}")
        with st.expander("Stage parameters"):
            st.json({key: modes[mode][key] for key in ("enhancer_params", "detector_params", "tracker_params")})
    with profile_column:
        profile = st.selectbox("Resource profile", list(profiles), index=list(profiles).index("minimal"))
        st.dataframe(profile_limits(profiles[profile]), hide_index=True, width="stretch")
    replay = st.radio(
        "Replay", list(REPLAY_HELP), horizontal=True, captions=list(REPLAY_HELP.values()), key="replay_mode"
    )

    if st.button("Start run", type="primary", disabled=video is None):
        with st.spinner("Uploading and checking the video…"):
            try:
                created = client.create_run(mode, profile, video.name, video.getvalue(), replay)
            except ApiError as error:
                st.error(f"The run was not started: {error}")
                return
        st.session_state["selected_run"] = created["run_id"]
        st.success(f"Run {created['run_id'][:6]} is queued. Its results open in the **Runs** tab.")

    if run_id := st.session_state.get("selected_run"):
        render_progress(client, run_id)


@st.fragment(run_every=2)
def render_progress(client: ApiClient, run_id: str) -> None:
    """Status of the run started from this page, polled until it finishes."""
    run = client.get_run(run_id)
    status, seen = run["status"], f"status_{run_id}"
    if st.session_state.get(seen) in ACTIVE_STATUSES and status not in ACTIVE_STATUSES:
        st.session_state[seen] = status
        st.rerun()  # the whole page, so the Runs and Comparison tabs pick up the finished run
    st.session_state[seen] = status
    st.write(f"{STATUS_ICONS[status]} Run **{data.run_label(run)}** is **{status}**.")
    if status in ACTIVE_STATUSES:
        st.caption("The API processes one run at a time; this status updates every 2 seconds.")
    elif status == "failed":
        st.error(run["error"])


# --------------------------------------------------------------------------- runs
def render_runs(client: ApiClient) -> None:
    runs = client.list_runs()
    if not runs:
        st.info("No runs yet. Start one in the **New run** tab.")
        return
    st.dataframe(
        data.runs_table(runs, STATUS_ICONS),
        hide_index=True,
        width="stretch",
        column_config={
            "run_id": "Run ID",
            "status": "Status",
            "mode": "Mode",
            "profile": "Profile",
            "source_name": "Video",
            "created_at": st.column_config.DatetimeColumn("Started", format="YYYY-MM-DD HH:mm:ss"),
            "duration_s": st.column_config.NumberColumn(
                "Duration, s", format="%.1f", help="From submission to the end of the run, including time in the queue"
            ),
            "error": "Error",
        },
    )
    by_id = {run["run_id"]: run for run in runs}
    selected = st.session_state.get("selected_run")
    run_id = st.selectbox(
        "Run details",
        list(by_id),
        index=list(by_id).index(selected) if selected in by_id else 0,
        format_func=lambda run_id: f"{STATUS_ICONS[by_id[run_id]['status']]} {data.run_label(by_id[run_id])}",
    )
    render_run_detail(client, by_id[run_id])


def render_run_detail(client: ApiClient, run: dict) -> None:
    st.subheader(data.run_label(run))
    if run["status"] in ACTIVE_STATUSES:
        render_progress(client, run["run_id"])
        st.button("Refresh", key="refresh_detail")
        return
    if run["status"] == "failed":
        st.error(f"The run failed: {run['error']}")
        render_config(run["run_id"], client.list_files(run["run_id"]))
        return

    run_id = run["run_id"]
    metrics = cached_report(run_id)["metrics"]
    files = client.list_files(run_id)
    render_metrics(metrics)
    render_video(run_id)

    detections = data.parse_mot(cached_file(run_id, "detections.txt")) if "detections.txt" in files else None
    tracks = data.parse_mot(cached_file(run_id, "tracks.txt")) if "tracks.txt" in files else None
    if detections is not None and tracks is not None:
        frame_count = int(metrics.get("processed_frames", 0) + metrics.get("frames_dropped", 0))
        frame_count = max(frame_count, int(detections["frame"].max() if len(detections) else 0))
        processed = None
        if metrics.get("frames_dropped", 0) > 0:  # realtime: the policy log lists the processed frames, if there is one
            log = data.parse_policy_log(cached_file(run_id, "policy_log.csv")) if "policy_log.csv" in files else None
            frames = log["frame"] if log is not None else pd.concat([detections["frame"], tracks["frame"]])
            processed = sorted(int(frame) for frame in set(frames))
        render_frame_counts(data.per_frame_counts(detections, tracks, frame_count, processed), frame_count)
        summary = data.track_summary(tracks)
        with st.expander(f"Tracks: {len(summary)} IDs"):
            st.dataframe(summary, hide_index=True, width="stretch")
    if "policy_log.csv" in files:
        render_policy_log(data.parse_policy_log(cached_file(run_id, "policy_log.csv")))
    render_config(run_id, files)


def render_metrics(metrics: dict[str, float]) -> None:
    groups = data.group_metrics(metrics)
    for title, rows in groups.items():
        st.markdown(f"**{title}**")
        columns = st.columns(min(len(rows), 5))
        for index, (metric, value) in enumerate(rows):
            columns[index % len(columns)].metric(metric.label, metric.format(value))
    if not data.QUALITY_KEYS & metrics.keys():
        st.caption(
            "No detection or tracking quality metrics: this run had no ground-truth annotations. "
            "Annotated runs are started with `resilens benchmark` (CLI) and show up here too."
        )


def render_video(run_id: str) -> None:
    st.markdown("**Video with bounding boxes**")
    try:
        with st.spinner("Rendering the annotated video (only on the first view)…"):
            video = cached_video(run_id)
    except ApiError as error:
        st.warning(f"No annotated video: {error}")
        return
    st.video(video, format="video/webm")
    st.caption("Orange: tracks with their ID · thin blue: detections · banner: frame, counts and policy decision.")


def render_frame_counts(counts: pd.DataFrame, frame_count: int) -> None:
    st.markdown("**Detections and tracks per frame**")
    dropped = frame_count - len(counts)
    if dropped > 0:
        st.caption(
            f"Realtime replay: {len(counts)} processed frames shown; the {dropped} dropped frames have no results."
        )
    long = counts.melt("frame", var_name="series", value_name="count")
    hover = alt.selection_point(fields=["frame"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(long).encode(
        x=alt.X("frame:Q", title="Frame"),
        y=alt.Y("count:Q", title="Objects"),
        color=alt.Color(
            "series:N",
            title=None,
            scale=alt.Scale(domain=["detections", "tracks"], range=[data.DETECTIONS_COLOR, data.TRACKS_COLOR]),
            legend=alt.Legend(orient="top"),
        ),
    )
    frames = alt.Scale(domain=[1, max(frame_count, 1)], nice=False)
    base = base.encode(x=alt.X("frame:Q", title="Frame", scale=frames))
    # with dropped frames a line would join results across the gaps, so every processed frame is a dot instead
    lines = base.mark_line(strokeWidth=2) if dropped <= 0 else base.mark_point(size=60, filled=True)
    points = base.mark_point(size=60, filled=True).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    rule = (
        alt.Chart(counts)
        .mark_rule(color="gray", opacity=0.5)
        .encode(
            x="frame:Q",
            opacity=alt.condition(hover, alt.value(0.5), alt.value(0)),
            tooltip=[alt.Tooltip("frame:Q", title="Frame"), "detections:Q", "tracks:Q"],
        )
        .add_params(hover)
    )
    st.altair_chart((lines + points + rule).properties(height=240), width="stretch")


def render_policy_log(log: pd.DataFrame) -> None:
    st.markdown("**Policy Engine decisions**")
    # same frame range and y-axis width in both charts, so a decision sits right above its latency
    frames = alt.Scale(domain=[int(log["frame"].min()), int(log["frame"].max())], nice=False)
    y_axis = {"minExtent": 90, "maxExtent": 90}
    actions = (
        alt.Chart(log)
        .mark_tick(thickness=2, size=14)
        .encode(
            x=alt.X("frame:Q", title="Frame", scale=frames),
            y=alt.Y("action:N", title=None, sort=data.POLICY_ACTIONS, axis=alt.Axis(**y_axis)),
            color=alt.Color(
                "action:N",
                title="Action",
                scale=alt.Scale(domain=data.POLICY_ACTIONS, range=list(data.ACTION_COLORS.values())),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=["frame", "degradation", "level", "confidence", "base_action", "action", "downgraded", "reason"],
        )
    )
    st.altair_chart(actions.properties(height=alt.Step(18)), width="stretch")
    if log["latency_ms"].astype(str).str.len().gt(0).any():
        latency = log[log["latency_ms"] != ""].astype({"latency_ms": float})
        line = (
            alt.Chart(latency)
            .mark_line(strokeWidth=2, color=data.DETECTIONS_COLOR)
            .encode(
                x=alt.X("frame:Q", title="Frame", scale=frames),
                y=alt.Y("latency_ms:Q", title=None, axis=alt.Axis(**y_axis)),
                tooltip=["frame", "latency_ms", "ram_mb", "cpu_percent"],
            )
        )
        st.caption("Moving latency, ms (the window the Policy Engine compares with the frame budget)")
        st.altair_chart(line.properties(height=180), width="stretch")
    degradations = log.groupby(["degradation", "level"]).size().rename("frames").reset_index()
    left, right = st.columns(2)
    left.caption("Detected degradation")
    left.dataframe(degradations, hide_index=True, width="stretch")
    right.caption("Final action")
    right.dataframe(log["action"].value_counts().rename("frames"), width="stretch")
    with st.expander(f"Policy log: {len(log)} frames"):
        st.dataframe(log, hide_index=True, width="stretch")


def render_config(run_id: str, files: list[str]) -> None:
    if "config.json" in files:
        with st.expander("Configuration snapshot"):
            st.json(json.loads(cached_file(run_id, "config.json")))
    if "fingerprint.json" in files:
        with st.expander("Input fingerprint"):
            st.json(json.loads(cached_file(run_id, "fingerprint.json")))
    if files:
        st.markdown("**Download**")
        columns = st.columns(len(files))
        for column, name in zip(columns, files, strict=True):
            column.download_button(name, cached_file(run_id, name), file_name=name, key=f"download_{run_id}_{name}")


# --------------------------------------------------------------------------- comparison
def render_comparison(client: ApiClient) -> None:
    runs = [run for run in client.list_runs() if run["status"] == "completed"]
    if len(runs) < 2:
        st.info("Comparison needs at least two completed runs.")
        return
    by_id = {run["run_id"]: run for run in runs}
    selected = st.multiselect(
        "Runs to compare",
        list(by_id),
        default=data.default_comparison(runs),
        format_func=lambda run_id: data.run_label(by_id[run_id]),
    )
    if len(selected) < 2:
        st.info("Select at least two runs.")
        return
    chosen = [by_id[run_id] for run_id in selected]
    videos = sorted({run["source_name"] for run in chosen})
    if len(videos) > 1:
        st.warning(f"The selected runs use different videos ({', '.join(videos)}): their metrics are not comparable.")
    table = data.comparison_table(chosen, [cached_report(run_id) for run_id in selected])
    labels = list(table.columns)
    mode_a = next((label for label, run in zip(labels, chosen, strict=True) if run["mode"] == "A"), labels[0])
    reference = st.selectbox("Reference run (mode A is the baseline)", labels, index=labels.index(mode_a))

    render_comparison_table(table, reference)
    render_metric_bars(table, chosen)
    render_tradeoff(table, chosen)

    if st.toggle("Show annotated videos side by side"):
        columns = st.columns(min(len(chosen), 3))
        for index, run in enumerate(chosen):
            with columns[index % len(columns)]:
                st.caption(data.run_label(run))
                try:
                    st.video(cached_video(run["run_id"]), format="video/webm")
                except ApiError as error:
                    st.warning(str(error))


def render_comparison_table(table: pd.DataFrame, reference: str) -> None:
    change = data.relative_change(table, reference)

    def cell(key: str, column: str) -> str:
        value = table.at[key, column]
        if pd.isna(value):
            return "—"
        text = data.metric_info(key).format(value)
        delta = change.at[key, column]
        if column != reference and pd.notna(delta) and abs(delta) >= 0.5:
            text += f" ({delta:+.0f}%)"
        return text

    shown = pd.DataFrame(
        {"Metric": [data.metric_info(key).label for key in table.index]}
        | {column: [cell(key, column) for key in table.index] for column in table.columns}
    )
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        height=35 * (len(shown) + 1) + 3,  # every metric visible without scrolling inside the table
        column_config={"Metric": st.column_config.TextColumn(width="medium", pinned=True)},
    )
    st.caption(f"In brackets: change against the reference run {reference}.")


def render_metric_bars(table: pd.DataFrame, runs: list[dict]) -> None:
    keys = list(table.index)
    key = st.selectbox("Metric", keys, format_func=lambda k: data.metric_info(k).label)
    metric = data.metric_info(key)
    values = pd.DataFrame(
        {"run": table.columns, "mode": [run["mode"] for run in runs], "value": table.loc[key].to_numpy()}
    ).dropna()
    bars = (
        alt.Chart(values)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("value:Q", title=metric.label),
            y=alt.Y("run:N", title=None, sort=None, scale=alt.Scale(paddingInner=0.3), axis=alt.Axis(labelLimit=320)),
            color=alt.Color(
                "mode:N",
                title="Mode",
                scale=alt.Scale(domain=list(data.MODE_COLORS), range=list(data.MODE_COLORS.values())),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=["run", "mode", alt.Tooltip("value:Q", title=metric.label, format=".3f")],
        )
    )
    labels = bars.mark_text(align="left", dx=4).encode(text=alt.Text("value:Q", format=".3g"), color=alt.value("gray"))
    if metric.better:
        st.caption(f"{metric.better.capitalize()} is better.")
    st.altair_chart((bars + labels).properties(height=alt.Step(36)), width="stretch")  # 36 px per run


def render_tradeoff(table: pd.DataFrame, runs: list[dict]) -> None:
    """Quality against speed: the runs towards the top right are the better trade-offs."""
    quality = next((key for key in ("hota", "map_50_95") if key in table.index and table.loc[key].notna().all()), None)
    if quality is None or "fps" not in table.index:
        st.caption("The quality-vs-speed chart needs HOTA or mAP for every selected run (annotated benchmark runs).")
        return
    points = pd.DataFrame(
        {
            "run": table.columns,
            "mode": [run["mode"] for run in runs],
            "fps": table.loc["fps"].to_numpy(),
            "quality": table.loc[quality].to_numpy(),
        }
    )
    label = data.metric_info(quality).label
    st.markdown(f"**{label} vs FPS**")
    base = alt.Chart(points).encode(
        x=alt.X("fps:Q", title="FPS"),
        y=alt.Y("quality:Q", title=label),
        tooltip=[
            "run",
            "mode",
            alt.Tooltip("fps:Q", format=".2f"),
            alt.Tooltip("quality:Q", title=label, format=".3f"),
        ],
    )
    dots = base.mark_point(size=120, filled=True, stroke="white", strokeWidth=2).encode(
        color=alt.Color(
            "mode:N",
            title="Mode",
            scale=alt.Scale(domain=list(data.MODE_COLORS), range=list(data.MODE_COLORS.values())),
        )
    )
    names = base.mark_text(dx=10, align="left", color="gray").encode(text="mode:N")
    st.altair_chart((dots + names).properties(height=300), width="stretch")


# --------------------------------------------------------------------------- main page
def main() -> None:
    st.set_page_config(page_title="Resilens", page_icon="🎥", layout="wide")
    st.title("Resilens v2.0")
    client = get_client()
    with st.sidebar:
        st.markdown("Detection and tracking in degraded video (fog, rain, low light, motion blur).")
        st.caption(f"API: {client.base_url}")
        if st.button("Refresh"):
            st.rerun()

    new_tab, runs_tab, compare_tab = st.tabs(["New run", "Runs", "Comparison"])
    for tab, render in ((new_tab, render_run_form), (runs_tab, render_runs), (compare_tab, render_comparison)):
        with tab:
            try:
                render(client)
            except ApiError as error:
                st.error(str(error))


if __name__ == "__main__":
    main()
