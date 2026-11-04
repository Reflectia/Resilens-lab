from abc import abstractmethod

import cv2
import numpy as np

from resilens.core.contracts import FrameContext
from resilens.core.stage import Stage


def fit_into(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The image scaled down, keeping its aspect ratio, to fit into (width, height); never scaled up."""
    height, width = image.shape[:2]
    scale = min(size[0] / width, size[1] / height)
    if scale >= 1:
        return image
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


class Enhancer(Stage):
    name = "enhancer"

    @abstractmethod
    def enhance(self, image: np.ndarray) -> np.ndarray:
        pass

    def process(self, ctx: FrameContext) -> FrameContext:
        """Apply the policy decision; without one (modes without policy) every frame is enhanced.

        enhance and every N-th frame of skip: ctx.enhanced = enhance(image). downscale: ctx.enhanced = the frame
        fitted into the profile's downscale resolution. bypass and the other skip frames: ctx.enhanced stays None.
        """
        decision = ctx.decision
        if decision is None or decision.enhance:
            ctx.enhanced = self.enhance(ctx.image)
        elif decision.resize_to is not None:
            ctx.enhanced = fit_into(ctx.image, decision.resize_to)
        return ctx
