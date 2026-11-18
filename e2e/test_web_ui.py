import json
import re
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import Locator, Page, expect

from resilens.api.schemas import RunInfo, RunReport

RUN_TIMEOUT_MS = 300_000


def open_form(page: Page, url: str) -> Locator:
    page.goto(url)
    form = page.get_by_role("tabpanel", name="New run")
    mode = form.get_by_role("combobox", name="Mode", exact=True)
    mode.click()
    page.get_by_role("option", name=re.compile(r"^A —")).click()
    profile = form.get_by_role("combobox", name="Resource profile")
    profile.click()
    page.get_by_role("option", name="minimal", exact=True).click()
    return form


def upload(form: Locator, label: str, path: Path) -> None:
    uploader = form.get_by_role("region", name=label, exact=True)
    uploader.locator("input[type=file]").set_input_files(path)
    expect(uploader.get_by_role("button", name=f"Remove {path.name}", exact=True)).to_be_visible(timeout=15_000)


def start_run(form: Locator) -> None:
    form.get_by_role("button", name="Start run").click()


def read_run(form: Locator, api: httpx.Client, status: str) -> RunInfo:
    progress = form.get_by_text(re.compile(r"Run .+ is (completed|failed)\."))
    expect(progress).to_be_visible(timeout=RUN_TIMEOUT_MS)
    runs = api.get("/runs").raise_for_status().json()
    assert len(runs) == 1
    run_id = runs[0]["run_id"]
    run = RunInfo.model_validate(api.get(f"/runs/{run_id}").raise_for_status().json())
    assert (run.run_id, run.mode, run.profile, run.status) == (run_id, "A", "minimal", status), run.error
    expect(progress).to_contain_text(f"{run_id[:6]} is {status}.")
    return run


def open_results(page: Page, run: RunInfo) -> Locator:
    page.get_by_role("tab", name="Runs", exact=True).click()
    results = page.get_by_role("tabpanel", name="Runs")
    heading = f"{run.mode} · {run.profile} · {run.source_name} · {run.run_id[:6]}"
    expect(results.get_by_role("heading", name=heading, exact=True)).to_be_visible()

    expect(results.get_by_text(f"Run ID: {run.run_id}", exact=True)).to_be_visible()
    return results


def read_report(api: httpx.Client, run_id: str) -> RunReport:
    report = RunReport.model_validate(api.get(f"/runs/{run_id}/report").raise_for_status().json())
    assert report.run_id == run_id
    return report


def expect_metric(results: Locator, label: str, value: str) -> None:
    card = results.get_by_test_id("stMetric").filter(has=results.page.get_by_text(label, exact=True))
    expect(card.get_by_test_id("stMetricValue")).to_have_text(value)


def test_mode_a_upload_displays_the_completed_run(
    page: Page, ui_url: str, api: httpx.Client, control_video: Path
) -> None:
    form = open_form(page, ui_url)
    upload(form, "Video", control_video)
    start_run(form)

    run = read_run(form, api, "completed")
    assert run.source_name == control_video.name
    report = read_report(api, run.run_id)
    assert report.metrics["processed_frames"] > 0

    results = open_results(page, run)
    expect_metric(results, "Processed frames", f"{report.metrics['processed_frames']:.0f}")
    expect_metric(results, "FPS", f"{report.metrics['fps']:.2f}")
    expect(results.get_by_text("Video with bounding boxes", exact=True)).to_be_visible()
    video = results.locator("video")
    expect(video).to_be_visible(timeout=RUN_TIMEOUT_MS)
    video.evaluate("video => video.load()")
    page.wait_for_function("video => video.readyState >= 2", arg=video.element_handle(), timeout=30_000)


def test_invalid_video_shows_an_error_without_creating_a_run(
    page: Page, ui_url: str, api: httpx.Client, tmp_path: Path
) -> None:
    broken_video = tmp_path / "broken.mp4"
    broken_video.write_bytes(b"not a video")
    form = open_form(page, ui_url)
    upload(form, "Video", broken_video)
    start_run(form)

    expect(form.get_by_text(re.compile(r"The run was not started: Invalid video"))).to_be_visible()
    assert api.get("/runs").raise_for_status().json() == []


@pytest.mark.parametrize("api_server", ["offline"], indirect=True)
def test_unreachable_api_shows_a_connection_error(page: Page, ui_url: str) -> None:
    page.goto(ui_url)
    form = page.get_by_role("tabpanel", name="New run")

    expect(form.get_by_text(re.compile(r"Cannot reach the Resilens API"))).to_be_visible(timeout=40_000)


@pytest.mark.parametrize("api_server", ["missing_weights"], indirect=True)
def test_failed_run_shows_the_pipeline_error(page: Page, ui_url: str, api: httpx.Client, control_video: Path) -> None:
    form = open_form(page, ui_url)
    upload(form, "Video", control_video)
    start_run(form)

    run = read_run(form, api, "failed")
    assert run.error is not None and "missing-yolox.pth" in run.error
    results = open_results(page, run)
    expect(results.get_by_text(re.compile(r"The run failed: .*missing-yolox\.pth"))).to_be_visible()


def test_coco_and_mot_upload_displays_quality_metrics(
    page: Page, ui_url: str, api: httpx.Client, control_video: Path, annotations: tuple[Path, Path]
) -> None:
    form = open_form(page, ui_url)
    upload(form, "Video", control_video)
    coco, mot = annotations
    upload(form, "COCO annotations", coco)
    upload(form, "MOT annotations", mot)
    start_run(form)

    run = read_run(form, api, "completed")
    report = read_report(api, run.run_id)
    results = open_results(page, run)
    expect(results.get_by_text("Detection quality", exact=True)).to_be_visible()
    expect(results.get_by_text("Tracking quality", exact=True)).to_be_visible()
    for key, label in (
        ("map_50_95", "mAP@[.5:.95]"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("hota", "HOTA"),
        ("idf1", "IDF1"),
    ):
        expect_metric(results, label, f"{report.metrics[key]:.3f}")
    expect_metric(results, "ID switches", f"{report.metrics['id_switches']:.0f}")


def test_one_annotation_file_blocks_run_submission(
    page: Page, ui_url: str, api: httpx.Client, control_video: Path, annotations: tuple[Path, Path]
) -> None:
    form = open_form(page, ui_url)
    upload(form, "Video", control_video)
    start_button = form.get_by_role("button", name="Start run")
    expect(start_button).to_be_enabled()
    coco, _ = annotations
    upload(form, "COCO annotations", coco)

    expect(form.get_by_text(re.compile(r"Add the MOT annotations too"))).to_be_visible()
    expect(start_button).to_be_disabled()
    assert api.get("/runs").raise_for_status().json() == []


def test_mismatched_annotations_show_an_error_without_creating_a_run(
    page: Page, ui_url: str, api: httpx.Client, control_video: Path, annotations: tuple[Path, Path], tmp_path: Path
) -> None:
    coco, mot = annotations
    wrong_coco = tmp_path / "different-resolution.json"
    ground_truth = json.loads(coco.read_text(encoding="utf-8"))
    for image in ground_truth["images"]:
        image["width"] += 2
    wrong_coco.write_text(json.dumps(ground_truth), encoding="utf-8")

    form = open_form(page, ui_url)
    upload(form, "Video", control_video)
    upload(form, "COCO annotations", wrong_coco)
    upload(form, "MOT annotations", mot)
    start_run(form)

    expect(
        form.get_by_text(
            re.compile(r"Invalid annotations: COCO annotations do not match video frame count or dimensions")
        )
    ).to_be_visible()
    assert api.get("/runs").raise_for_status().json() == []
