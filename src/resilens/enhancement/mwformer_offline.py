"""Offline MWFormer baseline (mode C): restore every frame of a clip once and keep the result in a cache.

MWFormer is too heavy for the realtime pipeline, so mode C does not run it there. `MWFormerOfflineRunner` cleans a
clip in advance; `MWFormerCacheEnhancer` then serves the cleaned frames during the run. The model code in
`mwformer_model/` is taken from github.com/Reflectia/Weather-removal-mwformer (inference only).
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch

from resilens.enhancement.mwformer_model.EncDec import Network_top
from resilens.enhancement.mwformer_model.style_filter64 import StyleFilter_Top

MAX_SIDE = 1024  # MWFormer restores at most this many pixels along the longer side
SIZE_MULTIPLE = 16
CACHE_INDEX = "index.json"


def frame_key(image: np.ndarray) -> str:
    """Cache key of a frame: SHA-256 of its pixels, so a cache serves any run that reads the same decoded frames."""
    digest = hashlib.sha256(np.ascontiguousarray(image).data)
    digest.update(str(image.shape).encode())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state_dict(weights_path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    return {key.removeprefix("module."): value for key, value in state.items()}


def model_size(width: int, height: int) -> tuple[int, int]:
    """Size MWFormer works at: longer side at most 1024, both sides rounded up to a multiple of 16."""
    scale = min(1.0, MAX_SIDE / max(width, height))
    rounded_up = [SIZE_MULTIPLE * -(-round(side * scale) // SIZE_MULTIPLE) for side in (width, height)]
    return rounded_up[0], rounded_up[1]


class MWFormerOfflineRunner:
    def __init__(self, style_weights: Path, backbone_weights: Path, device: str = "cuda") -> None:
        self.style_weights = Path(style_weights)
        self.backbone_weights = Path(backbone_weights)
        self.device = device
        self._backbone: Network_top | None = None
        self._style_filter: StyleFilter_Top | None = None

    def setup(self) -> None:
        backbone, style_filter = Network_top(), StyleFilter_Top()
        backbone.load_state_dict(load_state_dict(self.backbone_weights))
        style_filter.load_state_dict(load_state_dict(self.style_weights))
        self._backbone = backbone.to(self.device).eval()
        self._style_filter = style_filter.to(self.device).eval()

    def restore(self, image: np.ndarray) -> np.ndarray:
        """One BGR frame restored by MWFormer, at the original size."""
        if self._backbone is None or self._style_filter is None:
            self.setup()
        assert self._backbone is not None and self._style_filter is not None
        height, width = image.shape[:2]
        size = model_size(width, height)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_LANCZOS4)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255 * 2 - 1  # [-1, 1]
        with torch.inference_mode():
            y = self._backbone(x, self._style_filter(x)).clamp(0, 1)
        out = (y[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out if out.shape[:2] == (height, width) else cv2.resize(out, (width, height))

    def run(self, video: Path, cache_dir: Path, progress: Callable[[int, int], None] | None = None) -> int:
        """Restore every frame of the clip and save it to cache_dir for MWFormerCacheEnhancer; returns the count.

        Frames already in the cache are skipped, so an interrupted run can be repeated.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        index_path = cache_dir / CACHE_INDEX
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
        index["weights"] = {
            "backbone_sha256": file_sha256(self.backbone_weights),
            "style_filter_sha256": file_sha256(self.style_weights),
        }
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise FileNotFoundError(f"Cannot open video {video}")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                target = cache_dir / f"{frame_key(frame)}.png"
                if not target.exists():
                    cv2.imwrite(str(target), self.restore(frame), [cv2.IMWRITE_PNG_COMPRESSION, 1])
                count += 1
                if progress:
                    progress(count, total)
        finally:
            capture.release()
        index.setdefault("videos", {})[Path(video).name] = {"frames": count}
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        return count
