import os
from pathlib import Path

import pytest

from resilens.io.validation import VideoValidationError, VideoValidator


@pytest.mark.parametrize("name", ["clip.avi", "clip.mp4", "CLIP.AVI"])
def test_valid_video_returns_info(make_video, name: str) -> None:
    path = make_video(name, frames=20, width=64, height=48, fps=10.0)

    info = VideoValidator().validate(path)

    assert info.path == path
    assert info.frame_count == 20
    assert info.fps == pytest.approx(10.0)
    assert (info.width, info.height) == (64, 48)
    assert info.duration_s == pytest.approx(2.0)


def test_accepts_string_path(make_video) -> None:
    path = make_video()

    assert VideoValidator().validate(str(path)).path == path


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(VideoValidationError, match="not found"):
        VideoValidator().validate(tmp_path / "missing.mp4")


def test_directory_is_not_a_video(tmp_path: Path) -> None:
    folder = tmp_path / "folder.mp4"
    folder.mkdir()

    with pytest.raises(VideoValidationError, match="not found"):
        VideoValidator().validate(folder)


def test_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "clip.mkv"
    path.write_bytes(b"not checked")

    with pytest.raises(VideoValidationError, match="Unsupported video format"):
        VideoValidator().validate(path)


@pytest.mark.parametrize(
    "name, content",
    [
        ("empty.avi", b""),
        ("random.mp4", os.urandom(4096)),
        ("text.avi", b"this is not a video"),
    ],
)
def test_corrupted_file_raises_validation_error(tmp_path: Path, name: str, content: bytes) -> None:
    path = tmp_path / name
    path.write_bytes(content)

    with pytest.raises(VideoValidationError, match="corrupted"):
        VideoValidator().validate(path)


@pytest.mark.parametrize("name", ["clip.avi", "clip.mp4"])
def test_truncated_file_raises_validation_error(make_video, name: str) -> None:
    path = make_video(name, frames=20)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])

    with pytest.raises(VideoValidationError, match="corrupted"):
        VideoValidator().validate(path)


def test_duration_within_limit(make_video) -> None:
    path = make_video(frames=20, fps=10.0)  # 2 s

    assert VideoValidator(max_duration_s=2.0).validate(path).duration_s == pytest.approx(2.0)


def test_duration_over_limit(make_video) -> None:
    path = make_video(frames=20, fps=10.0)  # 2 s

    with pytest.raises(VideoValidationError, match="too long"):
        VideoValidator(max_duration_s=1.5).validate(path)


def test_resolution_within_limit(make_video) -> None:
    path = make_video(width=64, height=48)

    assert VideoValidator(max_resolution=(64, 48)).validate(path).width == 64


@pytest.mark.parametrize("max_resolution", [(32, 48), (64, 32)])
def test_resolution_over_limit(make_video, max_resolution: tuple[int, int]) -> None:
    path = make_video(width=64, height=48)

    with pytest.raises(VideoValidationError, match="exceeds the maximum"):
        VideoValidator(max_resolution=max_resolution).validate(path)
