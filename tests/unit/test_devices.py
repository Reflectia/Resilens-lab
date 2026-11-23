"""Whether a profile's device exists on this machine (D-10): GPU profiles need a CUDA build of PyTorch and a GPU."""

import pytest
import torch

from resilens.config.devices import check_profile_device, device_name, device_unavailable_reason
from resilens.config.loader import load_profile
from resilens.config.schema import Device


@pytest.mark.parametrize("profile", ["minimal", "standard"])
def test_cpu_profiles_are_always_available(no_cuda, profile: str) -> None:
    device = load_profile(profile).device

    assert device_unavailable_reason(device) is None
    assert device_name(device) is None
    check_profile_device(load_profile(profile))


def test_performance_profile_runs_on_cuda() -> None:
    assert load_profile("performance").device is Device.CUDA


def test_cuda_without_a_cuda_build_of_pytorch(no_cuda) -> None:
    assert "no CUDA support" in device_unavailable_reason(Device.CUDA)
    assert device_name(Device.CUDA) is None
    with pytest.raises(RuntimeError, match="Profile 'performance' needs a CUDA device: this PyTorch build"):
        check_profile_device(load_profile("performance"))


def test_cuda_build_without_a_visible_gpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert "sees no GPU" in device_unavailable_reason(Device.CUDA)


def test_cuda_with_a_gpu(fake_cuda) -> None:
    assert device_unavailable_reason(Device.CUDA) is None
    assert device_name(Device.CUDA) == fake_cuda
    check_profile_device(load_profile("performance"))
