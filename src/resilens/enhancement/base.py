from abc import abstractmethod

import numpy as np

from resilens.core.contracts import FrameContext
from resilens.core.stage import Stage


class Enhancer(Stage):
    name = "enhancer"

    @abstractmethod
    def enhance(self, image: np.ndarray) -> np.ndarray:
        pass

    def process(self, ctx: FrameContext) -> FrameContext:
        """Make decision (enhance / downscale / skip / bypass) and set ctx.enhanced."""
        pass
