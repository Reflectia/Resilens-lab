from pathlib import Path


class MWFormerOfflineRunner:
    def __init__(self, style_weights: Path, backbone_weights: Path, device: str = "cuda") -> None:
        self.style_weights = style_weights
        self.backbone_weights = backbone_weights
        self.device = device

    def run(self, video: Path, cache_dir: Path) -> None:
        """Restore every frame of the clip and save it to cache_dir for MWFormerCacheEnhancer."""
        pass
