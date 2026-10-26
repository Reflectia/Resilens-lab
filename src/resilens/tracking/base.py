from abc import abstractmethod

import numpy as np

from resilens.core.contracts import Detection, FrameContext, Track
from resilens.core.stage import Stage


class Tracker(Stage):
    name = "tracker"

    @abstractmethod
    def update(self, detections: list[Detection], frame_index: int, image: np.ndarray) -> list[Track]:
        """Advance all tracks by one frame. `image` is for trackers that look at pixels (e.g. optical flow)."""
        pass

    def process(self, ctx: FrameContext) -> FrameContext:
        """Track on the enhanced frame when there is one, so the tracker benefits from enhancement."""
        ctx.tracks = self.update(ctx.detections, ctx.index, ctx.working_image)
        return ctx
