from pathlib import Path

import cv2
import numpy as np

from resilens.core.contracts import Detection, FrameContext
from resilens.tracking.flow_tracker import (
    MAX_MISSES,
    FlowTracker,
    load_targets,
    mot_row,
    move_tracks,
    save_targets,
    track_box,
)

RNG = np.random.default_rng(0)
BACKGROUND = cv2.GaussianBlur(RNG.integers(0, 256, (240, 320), dtype=np.uint8), (9, 9), 0)
OBJECT = cv2.GaussianBlur(RNG.integers(0, 256, (80, 40), dtype=np.uint8), (3, 3), 0)
FLAT = np.full((240, 320), 128, dtype=np.uint8)
START = (40.0, 80.0, 40.0, 80.0)  # x, y, w, h of the object on frame 0
SPEED = 3  # px per frame to the right


def gray_frame(index: int) -> np.ndarray:
    frame = BACKGROUND.copy()
    x = 40 + SPEED * index
    frame[80:160, x : x + 40] = OBJECT
    return frame


def color_frame(index: int) -> np.ndarray:
    return cv2.cvtColor(gray_frame(index), cv2.COLOR_GRAY2BGR)


def test_box_follows_moving_object() -> None:
    box = START
    for i in range(1, 21):
        box = track_box(gray_frame(i - 1), gray_frame(i), box)

    assert abs(box[0] - (40 + SPEED * 20)) < 3
    assert abs(box[1] - 80) < 3


def test_flow_fails_on_flat_image() -> None:
    assert track_box(FLAT, FLAT, START) is None


def test_target_waits_then_is_dropped_when_flow_keeps_failing() -> None:
    tracks, misses = {1: START}, {}

    for _ in range(MAX_MISSES):
        tracks = move_tracks(FLAT, FLAT, tracks, misses)
        assert tracks == {1: START}  # waits in place

    assert move_tracks(FLAT, FLAT, tracks, misses) == {}


def test_target_leaving_the_frame_is_dropped() -> None:
    assert move_tracks(gray_frame(0), gray_frame(0), {1: (300.0, 80.0, 40.0, 80.0)}, {}) == {}


def test_tracker_returns_corner_boxes_with_ids_in_target_order() -> None:
    tracker = FlowTracker([START, (200.0, 20.0, 40.0, 80.0)])

    tracks = tracker.update([], 0, color_frame(0))

    assert [t.track_id for t in tracks] == [1, 2]
    assert tracks[0].bbox == (40.0, 80.0, 80.0, 160.0)


def test_tracker_follows_object_and_scores_waiting_boxes_zero() -> None:
    tracker = FlowTracker([START])
    for i in range(11):
        tracks = tracker.update([], i, color_frame(i))
    assert abs(tracks[0].bbox[0] - (40 + SPEED * 10)) < 3
    assert tracks[0].score == 1.0

    waiting = tracker.update([], 11, cv2.cvtColor(FLAT, cv2.COLOR_GRAY2BGR))

    assert waiting[0].score == 0.0


def test_tracker_stage_uses_enhanced_frame() -> None:
    tracker = FlowTracker([START])
    tracker.process(FrameContext(index=0, timestamp_s=0.0, image=color_frame(0)))

    ctx = FrameContext(index=1, timestamp_s=0.04, image=color_frame(0), enhanced=color_frame(1))

    assert abs(tracker.process(ctx).tracks[0].bbox[0] - (40 + SPEED)) < 2


def test_setup_restarts_tracking() -> None:
    tracker = FlowTracker([START])
    tracker.update([], 0, color_frame(0))
    tracker.update([], 1, color_frame(5))

    tracker.setup()

    assert tracker.update([], 0, color_frame(0))[0].bbox == (40.0, 80.0, 80.0, 160.0)


def test_targets_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "targets.json"

    save_targets(path, Path("clip.mp4"), [START, (1.26, 2.0, 3.0, 4.0)])

    assert load_targets(path) == [START, (1.3, 2.0, 3.0, 4.0)]


def test_mot_row_uses_one_based_frames_and_xywh() -> None:
    track = FlowTracker([START]).update([], 0, color_frame(0))[0]

    assert mot_row(0, track) == "1,1,40.0,80.0,40.0,80.0,1,-1,-1,-1"


def detection(box: tuple[float, float, float, float], score: float) -> Detection:
    x, y, w, h = box
    return Detection(bbox=(x, y, x + w, y + h), score=score, class_id=0)


def test_without_targets_tracker_starts_from_first_frame_with_detections() -> None:
    tracker = FlowTracker()
    other = (200.0, 20.0, 40.0, 80.0)

    assert tracker.update([], 0, color_frame(0)) == []  # nothing detected yet
    tracks = tracker.update([detection(other, 0.5), detection(START, 0.9)], 1, color_frame(1))

    assert [(t.track_id, t.bbox) for t in tracks] == [(1, (40.0, 80.0, 80.0, 160.0)), (2, (200.0, 20.0, 240.0, 100.0))]


def test_tracker_follows_detected_object_and_ignores_later_detections() -> None:
    tracker = FlowTracker()
    tracker.update([detection(START, 0.9)], 0, color_frame(0))
    for i in range(1, 11):
        tracks = tracker.update([detection((250.0, 10.0, 40.0, 80.0), 0.9)], i, color_frame(i))

    assert [t.track_id for t in tracks] == [1]
    assert abs(tracks[0].bbox[0] - (40 + SPEED * 10)) < 3


def test_hand_selected_targets_ignore_detections() -> None:
    tracks = FlowTracker([START]).update([detection((200.0, 20.0, 40.0, 80.0), 0.9)], 0, color_frame(0))

    assert [t.bbox for t in tracks] == [(40.0, 80.0, 80.0, 160.0)]


def test_tracker_keeps_original_pixels_when_policy_downscales_some_frames() -> None:
    tracker = FlowTracker([START])
    for i in range(11):
        full = color_frame(i)
        ctx = FrameContext(index=i, timestamp_s=i / 25, image=full)
        if i % 2:  # every other frame arrives at half size, as after the policy's downscale
            ctx.enhanced = cv2.resize(full, (160, 120), interpolation=cv2.INTER_AREA)
        tracks = tracker.process(ctx).tracks

    assert abs(tracks[0].bbox[0] - (40 + SPEED * 10)) < 3
    assert abs(tracks[0].bbox[1] - 80) < 3
    assert tracks[0].bbox[2] - tracks[0].bbox[0] == 40.0
