from pathlib import Path

import cv2
import numpy as np
import pytest

from resilens.reports.annotated_video import (
    DETECTION_COLOR,
    TRACK_COLOR,
    draw_frame,
    read_mot,
    read_policy_log,
    render_annotated_video,
)

POLICY_LOG = (
    "frame,degradation,level,confidence,profile,base_action,action,downgraded,enhanced,"
    "latency_ms,ram_mb,cpu_percent,gpu_memory_mb,reason\n"
    "1,fog,2,0.91,minimal,enhance,enhance,0,1,,512.0,80.0,,rule\n"
    "2,,,,minimal,,bypass,1,0,120.5,512.0,80.0,,budget\n"
)


def test_read_mot_groups_rows_by_frame(tmp_path: Path) -> None:
    path = tmp_path / "tracks.txt"
    path.write_text("1,1,10.0,20.0,30.0,40.0,0.900,-1,-1,-1\n1,2,0,0,5,5,0.5,-1,-1,-1\n3,1,11,21,30,40,0.8,-1,-1,-1\n")

    rows = read_mot(path)

    assert rows[1] == [(1, (10.0, 20.0, 30.0, 40.0), 0.9), (2, (0.0, 0.0, 5.0, 5.0), 0.5)]
    assert rows[3][0][0] == 1 and 2 not in rows
    assert read_mot(tmp_path / "missing.txt") == {}


def test_read_policy_log_captions(tmp_path: Path) -> None:
    path = tmp_path / "policy_log.csv"
    path.write_text(POLICY_LOG)

    assert read_policy_log(path) == {1: "fog L2 -> enhance", 2: "unknown L- -> bypass (downgraded)"}
    assert read_policy_log(tmp_path / "missing.csv") == {}


def test_draw_frame_scales_boxes_and_uses_colours() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    draw_frame(image, 0.5, 1, 1, [(-1, (100, 100, 100, 60), 0.9)], [(7, (200, 20, 100, 120), 0.9)])

    assert tuple(image[80, 75]) == DETECTION_COLOR  # left edge of the detection, (50, 50)-(100, 80) after scaling
    assert tuple(image[60, 100]) == TRACK_COLOR  # left edge of the track, (100, 10)-(150, 70)


def test_render_writes_a_playable_webm(tmp_path: Path, make_video) -> None:
    source = make_video("clip.mp4", frames=6, width=64, height=48)
    detections, tracks = tmp_path / "detections.txt", tmp_path / "tracks.txt"
    detections.write_text("1,-1,5,5,20,20,0.9,-1,-1,-1\n")
    tracks.write_text("1,1,5,5,20,20,0.9,-1,-1,-1\n6,1,8,5,20,20,0.9,-1,-1,-1\n")

    output = render_annotated_video(source, tmp_path / "annotated.webm", detections, tracks, tmp_path / "none.csv")

    capture = cv2.VideoCapture(str(output))
    frames = []
    while (frame := capture.read())[0]:
        frames.append(frame[1])
    capture.release()
    assert len(frames) == 6
    assert frames[0].shape == (48, 64, 3)
    assert not list(tmp_path.glob(".annotated*"))  # temporary file replaced


def test_render_downscales_wide_videos(tmp_path: Path, make_video) -> None:
    source = make_video("wide.mp4", frames=2, width=320, height=240)

    output = render_annotated_video(source, tmp_path / "a.webm", tmp_path / "d.txt", tmp_path / "t.txt", max_width=160)

    capture = cv2.VideoCapture(str(output))
    assert (capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (160, 120)
    capture.release()


def test_missing_source_fails_without_output(tmp_path: Path) -> None:
    output = tmp_path / "annotated.webm"

    with pytest.raises(FileNotFoundError, match="Cannot open source video"):
        render_annotated_video(tmp_path / "gone.mp4", output, tmp_path / "d.txt", tmp_path / "t.txt")
    assert not output.exists()
