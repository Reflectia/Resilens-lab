from pathlib import Path

import numpy as np

from resilens.core.contracts import FrameContext
from resilens.enhancement.base import Enhancer


class MWFormerCacheEnhancer(Enhancer):
    name = "mwformer_cache"

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def enhance(self, image: np.ndarray) -> np.ndarray:
        pass

    def process(self, ctx: FrameContext) -> FrameContext:
        """Load the cached frame"""
        pass
