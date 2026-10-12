import numpy as np

from resilens.enhancement.base import Enhancer


class BypassEnhancer(Enhancer):
    name = "bypass"

    def enhance(self, image: np.ndarray) -> np.ndarray:
        pass
