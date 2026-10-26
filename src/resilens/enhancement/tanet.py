from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from resilens.enhancement.base import Enhancer
from resilens.enhancement.tanet_model import TANet

SIZE_MULTIPLE = 4


def load_state_dict(weights_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state", checkpoint)
    return {key.removeprefix("module."): value for key, value in state.items()}


class TANetEnhancer(Enhancer):
    name = "tanet"

    def __init__(self, weights_path: Path, device: str = "cpu", scale: float = 1.0) -> None:
        self.weights_path = weights_path
        self.device = device
        self.scale = scale
        self.model: TANet | None = None

    def setup(self) -> None:
        """Load TANet weights onto the device."""
        model = TANet()
        model.load_state_dict(load_state_dict(Path(self.weights_path)))
        self.model = model.to(self.device).eval()

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TANetEnhancer.setup() must be called before enhance()")
        height, width = image.shape[:2]
        small = image if self.scale == 1.0 else cv2.resize(image, None, fx=self.scale, fy=self.scale)

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - 0.5
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        h, w = x.shape[2:]
        x = F.pad(x, (0, -w % SIZE_MULTIPLE, 0, -h % SIZE_MULTIPLE), mode="reflect")
        with torch.inference_mode():
            y = self.model(x)[:, :, :h, :w].clamp(-0.5, 0.5) + 0.5

        restored = (y[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
        return restored if restored.shape[:2] == (height, width) else cv2.resize(restored, (width, height))
