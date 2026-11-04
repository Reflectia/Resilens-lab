from abc import abstractmethod
from dataclasses import replace

import numpy as np

from resilens.core.contracts import Detection, FrameContext
from resilens.core.stage import Stage


class Detector(Stage):
    name = "detector"

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        pass

    def process(self, ctx: FrameContext) -> FrameContext:
        """Run detect() on ctx.working_image and store ctx.detections in original frame pixels.

        The working image is smaller than the frame after the policy's downscale, so boxes are scaled back.
        """
        image = ctx.working_image
        detections = self.detect(image)
        if image.shape[:2] != ctx.image.shape[:2]:
            sx, sy = ctx.image.shape[1] / image.shape[1], ctx.image.shape[0] / image.shape[0]
            detections = [
                replace(d, bbox=(d.bbox[0] * sx, d.bbox[1] * sy, d.bbox[2] * sx, d.bbox[3] * sy)) for d in detections
            ]
        ctx.detections = detections
        return ctx
