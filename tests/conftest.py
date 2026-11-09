from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from resilens.runs.store import RunStore
from resilens.testing.runs import StubRunManager

REPO_ROOT = Path(__file__).resolve().parents[1]
FOURCC_BY_EXTENSION = {".avi": "MJPG", ".mp4": "mp4v"}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-real-weights", action="store_true", help="run local tests using trained model weights")
    parser.addoption("--test-video", default=None, help="control video for trained-weight integration tests")
    parser.addoption("--weights-dir", default="weights", help="directory containing YOLOX and TANet weights")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--run-real-weights"):
        skip = pytest.mark.skip(reason="requires --run-real-weights")
        for item in items:
            if "real_weights" in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def configs_dir() -> Path:
    return REPO_ROOT / "configs"


@pytest.fixture
def make_video(tmp_path: Path) -> Callable[..., Path]:
    """Write a small synthetic video; frame i is filled with brightness i * 10."""

    def _make(name: str = "clip.avi", frames: int = 10, width: int = 64, height: int = 48, fps: float = 10.0) -> Path:
        path = tmp_path / name
        fourcc = cv2.VideoWriter_fourcc(*FOURCC_BY_EXTENSION[path.suffix.lower()])
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        assert writer.isOpened(), f"OpenCV cannot write {path.suffix} files in this environment"
        for i in range(frames):
            writer.write(np.full((height, width, 3), min(i * 10, 255), dtype=np.uint8))
        writer.release()
        return path

    return _make


@pytest.fixture
def run_store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "runs")


@pytest.fixture
def run_manager(run_store: RunStore) -> StubRunManager:
    return StubRunManager(run_store)
