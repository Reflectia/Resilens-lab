from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from resilens.core.contracts import Detection, FrameContext
from resilens.detection.base import Detector
from resilens.detection.yolox import PAD_VALUE, YOLOXNanoDetector, decode_detections, letterbox
from resilens.detection.yolox_model import yolox_nano

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "weights" / "yolox_nano_coco.pth"
CONTROL_DIR = REPO_ROOT / "benchmarks" / "degradation"


def prediction(cx: float, cy: float, w: float, h: float, objectness: float, class_id: int, class_score: float = 1.0):
    """One raw YOLOX output row: box centre and size in input pixels, objectness, 80 class probabilities."""
    row = torch.zeros(85)
    row[:5] = torch.tensor([cx, cy, w, h, objectness])
    row[5 + class_id] = class_score
    return row


def decode(*rows: torch.Tensor, ratio: float = 1.0, image_size=(416, 416), **kwargs) -> list[Detection]:
    params = {"conf_threshold": 0.3, "nms_threshold": 0.45} | kwargs
    return decode_detections(torch.stack(rows), ratio, image_size, **params)


class FixedOutputModel(torch.nn.Module):
    """Stands in for YOLOX: records the input and returns fixed predictions."""

    def __init__(self, *rows: torch.Tensor) -> None:
        super().__init__()
        self.output = torch.stack(rows).unsqueeze(0)
        self.inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs.append(x)
        return self.output


def test_letterbox_keeps_aspect_ratio_in_top_left_corner() -> None:
    image = np.full((200, 400, 3), 7, dtype=np.uint8)

    padded, ratio = letterbox(image, (416, 416))

    assert padded.shape == (416, 416, 3)
    assert ratio == pytest.approx(1.04)
    assert (padded[:208, :416] == 7).all()
    assert (padded[208:] == PAD_VALUE).all()


def test_score_is_objectness_times_class_probability() -> None:
    (detection,) = decode(prediction(100, 100, 40, 80, objectness=0.8, class_id=0, class_score=0.5))

    assert detection.score == pytest.approx(0.4)
    assert detection.class_id == 0
    assert detection.bbox == pytest.approx((80, 60, 120, 140))  # x1, y1, x2, y2


def test_low_scores_are_dropped() -> None:
    assert decode(prediction(100, 100, 40, 80, objectness=0.5, class_id=0, class_score=0.5)) == []


def test_nms_keeps_best_box_per_class() -> None:
    detections = decode(
        prediction(100, 100, 40, 80, objectness=0.6, class_id=0),
        prediction(102, 101, 40, 80, objectness=0.9, class_id=0),  # same person, higher score
        prediction(101, 100, 40, 80, objectness=0.7, class_id=2),  # car at the same place is not suppressed
        prediction(300, 300, 40, 80, objectness=0.5, class_id=0),
    )

    assert [(d.class_id, round(d.score, 2)) for d in detections] == [(0, 0.9), (2, 0.7), (0, 0.5)]


def test_class_filter() -> None:
    detections = decode(
        prediction(100, 100, 40, 80, objectness=0.9, class_id=0),
        prediction(300, 300, 40, 80, objectness=0.9, class_id=2),
        class_ids=[0],
    )

    assert [d.class_id for d in detections] == [0]


def test_boxes_are_mapped_back_to_original_size_and_clipped() -> None:
    (inside, at_edge) = decode(
        prediction(104, 52, 52, 26, objectness=0.9, class_id=0),
        prediction(410, 200, 40, 40, objectness=0.8, class_id=0),
        ratio=0.52,
        image_size=(400, 800),
    )

    assert inside.bbox == pytest.approx((150, 75, 250, 125))
    assert at_edge.bbox[2] == 800  # 430 / 0.52 = 827 is clipped to the frame width


def test_model_outputs_one_prediction_per_grid_cell() -> None:
    model = yolox_nano().eval()

    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 416, 416))

    assert output.shape == (1, 52 * 52 + 26 * 26 + 13 * 13, 85)
    assert ((output[..., 4:] >= 0) & (output[..., 4:] <= 1)).all()  # objectness and class probabilities


@pytest.mark.parametrize("official_format", [True, False], ids=["checkpoint", "state_dict"])
def test_setup_loads_official_checkpoint_and_plain_state_dict(tmp_path: Path, official_format: bool) -> None:
    path = tmp_path / "yolox_nano.pth"
    state = yolox_nano().state_dict()
    torch.save({"start_epoch": 300, "model": state} if official_format else state, path)
    detector = YOLOXNanoDetector(path)

    detector.setup()

    assert not detector.model.training


def test_setup_rejects_weights_of_another_model(tmp_path: Path) -> None:
    path = tmp_path / "other.pth"
    torch.save({"model": {"conv.weight": torch.zeros(1)}}, path)

    with pytest.raises(RuntimeError, match="state_dict"):
        YOLOXNanoDetector(path).setup()


def test_detect_before_setup_fails() -> None:
    with pytest.raises(RuntimeError, match="setup"):
        YOLOXNanoDetector(Path("missing.pth")).detect(np.zeros((10, 10, 3), dtype=np.uint8))


def test_detect_feeds_letterboxed_bgr_frame_and_returns_frame_coordinates() -> None:
    detector = YOLOXNanoDetector(Path("unused.pth"), input_size=[416, 416])
    detector.model = FixedOutputModel(prediction(208, 104, 104, 52, objectness=0.9, class_id=0))
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    image[..., 0] = 255  # blue channel of a BGR frame

    (detection,) = detector.detect(image)

    x = detector.model.inputs[0]
    assert x.shape == (1, 3, 416, 416)
    assert x[0, :, 0, 0].tolist() == [255, 0, 0]  # no colour swap and no normalisation
    assert detection.bbox == pytest.approx((150, 75, 250, 125))


class RecordingDetector(Detector):
    def __init__(self) -> None:
        self.images: list[np.ndarray] = []

    def detect(self, image: np.ndarray) -> list[Detection]:
        self.images.append(image)
        return [Detection(bbox=(0, 0, 1, 1), score=1.0, class_id=0)]


def test_stage_detects_on_enhanced_frame() -> None:
    detector = RecordingDetector()
    image, enhanced = np.zeros((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8)

    ctx = detector.process(FrameContext(index=0, timestamp_s=0.0, image=image, enhanced=enhanced))

    assert detector.images[0] is enhanced
    assert ctx.detections == [Detection(bbox=(0, 0, 1, 1), score=1.0, class_id=0)]


@pytest.mark.integration
@pytest.mark.skipif(not WEIGHTS.is_file(), reason="needs weights/yolox_nano_coco.pth (see README)")
@pytest.mark.skipif(not (CONTROL_DIR / "manifest.json").is_file(), reason="needs the control videos (see benchmarks)")
def test_finds_pedestrians_on_control_video() -> None:
    capture = cv2.VideoCapture(str(CONTROL_DIR / "control_videos" / "MOT20-01" / "clean.mp4"))
    ok, frame = capture.read()
    capture.release()
    assert ok
    detector = YOLOXNanoDetector(WEIGHTS, class_ids=[0])
    detector.setup()

    detections = detector.detect(frame)

    assert len(detections) >= 10
    assert all(d.class_id == 0 and d.score >= 0.3 for d in detections)
