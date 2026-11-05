from dataclasses import replace

import numpy as np
import pytest

from resilens.config.schema import ReplayMode
from resilens.io.replay import VideoReplay
from resilens.io.validation import VideoValidationError, VideoValidator


@pytest.fixture
def video_info(make_video):
    return VideoValidator().validate(make_video(frames=12, width=64, height=48, fps=10.0))


def test_yields_every_frame_in_order(video_info) -> None:
    frames = list(VideoReplay(video_info))

    assert [f.index for f in frames] == list(range(12))
    brightness = [float(f.image.mean()) for f in frames]
    assert brightness == sorted(brightness)  # synthetic frames get brighter frame by frame


def test_timestamps_follow_fps(video_info) -> None:
    timestamps = [f.timestamp_s for f in VideoReplay(video_info)]

    assert timestamps == pytest.approx([i / 10.0 for i in range(12)])


def test_frames_are_bgr_images_of_video_size(video_info) -> None:
    frame = next(iter(VideoReplay(video_info)))

    assert frame.image.shape == (48, 64, 3)
    assert frame.image.dtype == np.uint8
    assert frame.enhanced is None
    assert frame.detections == []


def test_each_frame_records_replay_timing(video_info) -> None:
    for frame in VideoReplay(video_info):
        assert len(frame.timings) == 1
        assert frame.timings[0].stage == VideoReplay.stage_name
        assert frame.timings[0].duration_ms >= 0


def test_counters_after_full_replay(video_info) -> None:
    replay = VideoReplay(video_info)
    list(replay)

    assert replay.frames_read == 12
    assert not replay.ended_early
    assert replay.frames_dropped == 0


def test_can_replay_twice(video_info) -> None:
    replay = VideoReplay(video_info)

    assert len(list(replay)) == len(list(replay)) == 12


def test_stops_gracefully_when_file_has_fewer_frames_than_reported(video_info, caplog) -> None:
    replay = VideoReplay(replace(video_info, frame_count=20))

    frames = list(replay)

    assert len(frames) == 12
    assert replay.ended_early
    assert "ended early" in caplog.text


def test_not_ended_early_before_iterating(video_info) -> None:
    assert not VideoReplay(video_info).ended_early


def test_not_ended_early_when_consumer_stops(video_info, caplog) -> None:
    replay = VideoReplay(video_info)

    for frame in replay:
        if frame.index == 4:
            break

    assert replay.frames_read == 4
    assert not replay.ended_early
    assert "ended early" not in caplog.text


def test_file_removed_after_validation(video_info) -> None:
    video_info.path.unlink()

    with pytest.raises(VideoValidationError, match="Cannot open"):
        list(VideoReplay(video_info))


class FakeClock:
    """Time that moves only when the replay sleeps or the consumer "processes" a frame."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def realtime_replay(video_info, clock: FakeClock) -> VideoReplay:
    return VideoReplay(video_info, ReplayMode.REALTIME, clock=clock, sleep=clock.sleep)


def consume(replay: VideoReplay, clock: FakeClock, processing_s: float) -> list[int]:
    indices = []
    for frame in replay:
        indices.append(frame.index)
        clock.now += processing_s
    return indices


def test_realtime_fast_consumer_gets_every_frame_at_video_speed(video_info) -> None:
    clock = FakeClock()
    replay = realtime_replay(video_info, clock)

    assert consume(replay, clock, processing_s=0.0) == list(range(12))
    assert replay.frames_dropped == 0
    assert clock.now == pytest.approx(1.1)  # frame 11 is due at 11 / 10 fps


def test_realtime_drops_frames_that_passed_while_consumer_was_busy(video_info) -> None:
    clock = FakeClock()
    replay = realtime_replay(video_info, clock)

    indices = consume(replay, clock, processing_s=0.25)  # 2.5 frame periods per frame

    assert indices == [0, 2, 5, 7, 10]
    assert replay.frames_read == 5
    assert replay.frames_dropped == 7
    assert not replay.ended_early


def test_realtime_timestamps_follow_original_frame_index(video_info) -> None:
    clock = FakeClock()
    frames = []
    for frame in realtime_replay(video_info, clock):
        frames.append(frame)
        clock.now += 0.25

    assert [f.timestamp_s for f in frames] == pytest.approx([i / 10.0 for i in (0, 2, 5, 7, 10)])


def test_realtime_clock_starts_at_first_requested_frame(video_info) -> None:
    clock = FakeClock()
    clock.now = 100.0  # time before iteration must not count as lateness

    assert consume(realtime_replay(video_info, clock), clock, processing_s=0.0) == list(range(12))


def test_realtime_counters_reset_on_second_replay(video_info) -> None:
    clock = FakeClock()
    replay = realtime_replay(video_info, clock)
    consume(replay, clock, processing_s=0.25)

    assert consume(replay, clock, processing_s=0.0) == list(range(12))
    assert replay.frames_dropped == 0


def test_offline_ignores_consumer_speed(video_info) -> None:
    clock = FakeClock()
    replay = VideoReplay(video_info, ReplayMode.OFFLINE, clock=clock, sleep=clock.sleep)

    assert consume(replay, clock, processing_s=0.25) == list(range(12))
    assert replay.frames_dropped == 0
