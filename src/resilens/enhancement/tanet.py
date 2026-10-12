from pathlib import Path

import numpy as np

from resilens.enhancement.base import Enhancer


class TANetEnhancer(Enhancer):
    name = "tanet"

    def __init__(self, weights_path: Path, device: str = "cpu", scale: float = 1.0) -> None:
        self.weights_path = weights_path
        self.device = device
        self.scale = scale

    def setup(self) -> None:
        """Load TANet weights onto the device."""
        pass

    def enhance(self, image: np.ndarray) -> np.ndarray:
        pass
