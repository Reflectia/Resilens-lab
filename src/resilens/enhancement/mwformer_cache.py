from pathlib import Path

import cv2
import numpy as np

from resilens.core.registry import registry
from resilens.enhancement.base import Enhancer
from resilens.enhancement.mwformer_offline import frame_key

BUILD_HINT = "resilens mwformer-cache VIDEO --style-weights STYLE --backbone-weights BACKBONE"


@registry.register("mwformer_cache")
class MWFormerCacheEnhancer(Enhancer):
    """Mode C: serves frames that MWFormerOfflineRunner restored in advance (`resilens mwformer-cache`)."""

    name = "mwformer_cache"

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    def setup(self) -> None:
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f"MWFormer cache {self.cache_dir} does not exist; build it first: {BUILD_HINT}")

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """The cached restoration of this frame, found by the frame's pixels."""
        path = self.cache_dir / f"{frame_key(image)}.png"
        restored = cv2.imread(str(path), cv2.IMREAD_COLOR) if path.is_file() else None
        if restored is None:
            raise FileNotFoundError(
                f"Frame not in the MWFormer cache {self.cache_dir}; build the cache for this exact video: {BUILD_HINT}"
            )
        return restored
