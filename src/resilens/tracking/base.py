from abc import abstractmethod

from resilens.core.contracts import Detection, FrameContext, Track
from resilens.core.stage import Stage


class Tracker(Stage):
    name = "tracker"

    @abstractmethod
    def update(self, detections: list[Detection], frame_index: int) -> list[Track]:
        pass

    def process(self, ctx: FrameContext) -> FrameContext:
        pass
