import numpy as np

from resilens.core.registry import registry
from resilens.enhancement.base import Enhancer


@registry.register("bypass")
class BypassEnhancer(Enhancer):
    name = "bypass"

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Mode A and the policy's bypass action: the frame goes on unchanged."""
        return image
