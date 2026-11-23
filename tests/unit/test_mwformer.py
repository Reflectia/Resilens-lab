import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from resilens.cli import main
from resilens.core.contracts import FrameContext
from resilens.core.pipeline import create_stage
from resilens.enhancement.mwformer_cache import MWFormerCacheEnhancer
from resilens.enhancement.mwformer_offline import MWFormerOfflineRunner, frame_key, model_size


def tiny_weights(tmp_path: Path) -> tuple[Path, Path]:
    style, backbone = tmp_path / "style.pth", tmp_path / "backbone.pth"
    style.write_bytes(b"style")
    backbone.write_bytes(b"backbone")
    return style, backbone


@pytest.fixture
def runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MWFormerOfflineRunner:
    """A runner whose model is replaced by a brightness inversion, so no weights are needed."""
    style, backbone = tiny_weights(tmp_path)
    runner = MWFormerOfflineRunner(style, backbone, device="cpu")
    monkeypatch.setattr(runner, "restore", lambda image: 255 - image)
    return runner


@pytest.mark.parametrize(
    ("size", "expected"),
    [((1920, 1080), (1024, 576)), ((1080, 1920), (576, 1024)), ((64, 48), (64, 48)), ((100, 50), (112, 64))],
)
def test_model_size_is_at_most_1024_and_multiple_of_16(size: tuple[int, int], expected: tuple[int, int]) -> None:
    assert model_size(*size) == expected


def test_frame_key_depends_on_pixels_and_shape() -> None:
    frame = np.zeros((4, 6, 3), np.uint8)

    assert frame_key(frame) == frame_key(frame.copy())
    assert frame_key(frame) != frame_key(frame + 1)
    assert frame_key(frame) != frame_key(np.zeros((6, 4, 3), np.uint8))


def test_runner_caches_every_frame_and_records_weights(
    runner: MWFormerOfflineRunner, make_video: Callable[..., Path], tmp_path: Path
) -> None:
    video = make_video(frames=5)
    cache = tmp_path / "cache"

    assert runner.run(video, cache) == 5

    assert len(list(cache.glob("*.png"))) == 5
    index = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    assert index["videos"][video.name] == {"frames": 5}
    assert set(index["weights"]) == {"backbone_sha256", "style_filter_sha256"}


def test_runner_skips_frames_already_cached(
    runner: MWFormerOfflineRunner, make_video: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video, cache = make_video(frames=3), tmp_path / "cache"
    runner.run(video, cache)
    monkeypatch.setattr(runner, "restore", lambda image: pytest.fail("restored a cached frame"))

    assert runner.run(video, cache) == 3


def test_cache_enhancer_returns_the_restored_frames(
    runner: MWFormerOfflineRunner, make_video: Callable[..., Path], tmp_path: Path
) -> None:
    video, cache = make_video(frames=4), tmp_path / "cache"
    runner.run(video, cache)
    enhancer = create_stage("mwformer_cache", {"cache_dir": cache}, "cpu")
    enhancer.setup()
    capture = cv2.VideoCapture(str(video))
    for index in range(4):
        _, frame = capture.read()
        ctx = enhancer.process(FrameContext(index=index, timestamp_s=index / 10, image=frame))
        np.testing.assert_array_equal(ctx.enhanced, 255 - frame)


def test_cache_enhancer_names_the_missing_cache_and_frame(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist; build it first"):
        MWFormerCacheEnhancer(tmp_path / "none").setup()
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="Frame not in the MWFormer cache"):
        MWFormerCacheEnhancer(tmp_path / "empty").enhance(np.zeros((8, 8, 3), np.uint8))


def test_cli_reports_missing_weights(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = ["--style-weights", str(tmp_path / "a"), "--backbone-weights", str(tmp_path / "b")]

    with pytest.raises(SystemExit):
        main(["mwformer-cache", "clip.mp4", *missing])

    assert "Weights not found" in capsys.readouterr().err


@pytest.mark.real_weights
def test_mwformer_restores_a_frame_at_its_size(request: pytest.FixtureRequest) -> None:
    weights = Path(request.config.getoption("--weights-dir")) / "mwformer"
    runner = MWFormerOfflineRunner(weights / "stylefilter_original.pth", weights / "backbone_original.pth", "cpu")
    frame = np.random.default_rng(0).integers(0, 255, (96, 160, 3), dtype=np.uint8)

    restored = runner.restore(frame)

    assert restored.shape == frame.shape and restored.dtype == np.uint8
