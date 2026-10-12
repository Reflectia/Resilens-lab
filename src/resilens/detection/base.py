from abc import abstractmethod

import numpy as np

from resilens.core.contracts import Detection, FrameContext
from resilens.core.stage import Stage


class Detector(Stage):
    name = "detector"

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        pass

    def process(self, ctx: FrameContext) -> FrameContext:
        """Run detect() on ctx.working_image and store ctx.detections."""
        pass
