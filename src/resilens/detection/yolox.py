from pathlib import Path

import numpy as np

from resilens.core.contracts import Detection
from resilens.detection.base import Detector


class YOLOXNanoDetector(Detector):
    name = "yolox_nano"

    def __init__(
        self,
        weights_path: Path,
        device: str = "cpu",
        class_ids: list[int] | None = None,
        conf_threshold: float = 0.3,
        input_size: tuple[int, int] = (416, 416),
    ) -> None:
        self.weights_path = weights_path
        self.device = device
        self.class_ids = class_ids
        self.conf_threshold = conf_threshold
        self.input_size = input_size

    def setup(self) -> None:
        pass

    def detect(self, image: np.ndarray) -> list[Detection]:
        pass
