import csv

import pytest
import torch

from resilens.config.loader import build_run_config
from resilens.detection.yolox import YOLOXNanoDetector
from resilens.enhancement.tanet import TANetEnhancer
from resilens.runs.manager import RunManager
from resilens.runs.models import RunStatus
from resilens.runs.store import RunStore


class FixedYOLOX(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pixels: list[float] = []

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.pixels.append(image[0, 0, 0, 0].item())
        prediction = image.new_zeros((1, 1, 85))
        prediction[0, 0, :5] = image.new_tensor([208, 156, 130, 130, 0.9])
        prediction[0, 0, 5] = 1.0
        return prediction


class FixedTANet(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.full_like(image, 0.25)


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["A", "B"])
def test_mode_smoke_through_run_manager(mode, tmp_path, make_video, monkeypatch) -> None:
    detector_model = FixedYOLOX()
    tanet_calls = []

    def setup_detector(detector):
        detector.model = detector_model

    def setup_tanet(enhancer):
        tanet_calls.append(True)
        enhancer.model = FixedTANet()

    monkeypatch.setattr(YOLOXNanoDetector, "setup", setup_detector)
    monkeypatch.setattr(TANetEnhancer, "setup", setup_tanet)

    video = make_video(frames=3)
    config = build_run_config(mode, "minimal", video)
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store)
    run_id = manager.submit(config)
    manager.execute(run_id)

    record = store.get(run_id)
    assert record.status == RunStatus.COMPLETED, record.error
    metrics = store.load_metrics(run_id)
    report = store.load_report(run_id)
    assert metrics["processed_frames"] == 3
    assert metrics["processing_time_s"] > 0 and metrics["fps"] > 0
    assert metrics["peak_ram_mb"] > 0
    assert report["run_id"] == run_id and report["mode"] == mode
    assert report["metrics"] == metrics

    run_dir = store.run_dir(run_id)
    for filename in ("detections.txt", "tracks.txt"):
        with (run_dir / filename).open(encoding="utf-8", newline="") as file:
            rows = list(csv.reader(file))
        assert [int(row[0]) for row in rows] == [1, 2, 3]
        assert all(len(row) == 10 and float(row[4]) > 0 and float(row[5]) > 0 for row in rows)
        assert [int(row[1]) for row in rows] == ([-1] * 3 if filename == "detections.txt" else [1] * 3)

    assert len(detector_model.pixels) == 3
    if mode == "A":
        assert not tanet_calls
        assert detector_model.pixels[0] < 5
    else:
        assert len(tanet_calls) == 1
        assert detector_model.pixels == pytest.approx([191] * 3, abs=1)
