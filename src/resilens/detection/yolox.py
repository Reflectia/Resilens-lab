from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

from resilens.core.contracts import Detection
from resilens.core.registry import registry
from resilens.detection.base import Detector
from resilens.detection.yolox_model import yolox_nano

PAD_VALUE = 114  # grey padding YOLOX was trained with


def load_state_dict(weights_path: Path) -> dict[str, torch.Tensor]:
    """Official YOLOX checkpoints keep the weights under "model" next to optimizer state."""
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    return checkpoint.get("model", checkpoint)


def letterbox(image: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Resize keeping the aspect ratio into the top-left corner of a padded (height, width) canvas."""
    height, width = image.shape[:2]
    ratio = min(input_size[0] / height, input_size[1] / width)
    resized = cv2.resize(image, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_LINEAR)
    padded = np.full((*input_size, 3), PAD_VALUE, dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return padded, ratio


def decode_detections(
    output: torch.Tensor,
    ratio: float,
    image_size: tuple[int, int],
    conf_threshold: float,
    nms_threshold: float,
    class_ids: list[int] | None = None,
) -> list[Detection]:
    """Turn raw predictions (anchors, 5 + classes) into detections in original image pixels, best first.

    Score = objectness x best class probability, as in YOLOX postprocess; NMS runs per class.
    """
    class_scores, classes = output[:, 5:].max(dim=1)
    scores = output[:, 4] * class_scores
    keep = scores >= conf_threshold
    if class_ids is not None:
        keep &= torch.isin(classes, torch.tensor(class_ids, device=classes.device))
    centers, sizes = output[keep, :2], output[keep, 2:4]
    scores, classes = scores[keep], classes[keep]

    height, width = image_size
    boxes = torch.cat([centers - sizes / 2, centers + sizes / 2], dim=1) / ratio
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)

    kept = batched_nms(boxes, scores, classes, nms_threshold)
    return [
        Detection(bbox=tuple(boxes[i].tolist()), score=float(scores[i]), class_id=int(classes[i]))
        for i in kept.tolist()
    ]


@registry.register("yolox_nano")
class YOLOXNanoDetector(Detector):
    """YOLOX-Nano trained on COCO (80 classes; 0 = person). Frames are BGR uint8, as read by OpenCV."""

    name = "yolox_nano"

    def __init__(
        self,
        weights_path: Path,
        device: str = "cpu",
        class_ids: list[int] | None = None,
        conf_threshold: float = 0.3,
        nms_threshold: float = 0.45,
        input_size: tuple[int, int] = (416, 416),
    ) -> None:
        self.weights_path = weights_path
        self.device = device
        self.class_ids = class_ids
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = tuple(input_size)  # (height, width); YAML gives a list
        self.model: torch.nn.Module | None = None

    def setup(self) -> None:
        """Load YOLOX-Nano weights onto the device."""
        model = yolox_nano()
        model.load_state_dict(load_state_dict(Path(self.weights_path)))
        self.model = model.to(self.device).eval()

    def detect(self, image: np.ndarray) -> list[Detection]:
        if self.model is None:
            raise RuntimeError("YOLOXNanoDetector.setup() must be called before detect()")
        padded, ratio = letterbox(image, self.input_size)
        x = torch.from_numpy(padded).permute(2, 0, 1).float().unsqueeze(0).to(self.device)  # 0-255, no normalisation
        with torch.inference_mode():
            output = self.model(x)[0]
        return decode_detections(
            output, ratio, image.shape[:2], self.conf_threshold, self.nms_threshold, self.class_ids
        )
