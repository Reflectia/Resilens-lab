from pathlib import Path

import numpy as np
import pytest
import torch

from resilens.core.contracts import FrameContext
from resilens.enhancement.tanet import TANetEnhancer
from resilens.enhancement.tanet_model import TANet

RNG = np.random.default_rng(0)
IMAGE = RNG.integers(0, 256, (37, 50, 3), dtype=np.uint8)


def identity_model() -> TANet:
    torch.manual_seed(0)
    model = TANet()
    torch.nn.init.zeros_(model.conv_2.weight)
    torch.nn.init.zeros_(model.conv_2.bias)
    return model


def save_training_checkpoint(path: Path, model: TANet) -> Path:
    state = {f"module.{key}": value for key, value in model.state_dict().items()}
    torch.save({"model_state": state, "epoch": 30}, path)
    return path


def make_enhancer(weights_path: Path, **kwargs) -> TANetEnhancer:
    enhancer = TANetEnhancer(weights_path, **kwargs)
    enhancer.setup()
    return enhancer


def test_loads_training_checkpoint_and_keeps_colours(tmp_path: Path) -> None:
    enhancer = make_enhancer(save_training_checkpoint(tmp_path / "tanet.pth", identity_model()))

    restored = enhancer.enhance(IMAGE)

    assert restored.dtype == np.uint8
    assert np.abs(restored.astype(int) - IMAGE).max() <= 1


def test_loads_plain_state_dict(tmp_path: Path) -> None:
    path = tmp_path / "tanet.pth"
    torch.save(identity_model().state_dict(), path)

    assert np.abs(make_enhancer(path).enhance(IMAGE).astype(int) - IMAGE).max() <= 1


def test_scale_keeps_original_frame_size(tmp_path: Path) -> None:
    torch.manual_seed(1)
    enhancer = make_enhancer(save_training_checkpoint(tmp_path / "tanet.pth", TANet()), scale=0.5)

    assert enhancer.enhance(IMAGE).shape == IMAGE.shape


def test_enhance_before_setup_fails() -> None:
    with pytest.raises(RuntimeError, match="setup"):
        TANetEnhancer(Path("missing.pth")).enhance(IMAGE)


def test_stage_sets_enhanced_frame(tmp_path: Path) -> None:
    enhancer = make_enhancer(save_training_checkpoint(tmp_path / "tanet.pth", identity_model()))

    ctx = enhancer.process(FrameContext(index=0, timestamp_s=0.0, image=IMAGE))

    assert ctx.enhanced is not None and ctx.working_image is ctx.enhanced
    assert ctx.enhanced.shape == IMAGE.shape
