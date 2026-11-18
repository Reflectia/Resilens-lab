"""Whether the device a resource profile asks for exists on this machine (D-10).

The API, the Run Manager and the profile measurements check this before a run starts, so a GPU profile on a
machine without CUDA fails at once with a reason instead of deep inside model loading.
"""

import torch

from resilens.config.schema import Device, ProfileConfig


def device_unavailable_reason(device: Device) -> str | None:
    """Why `device` cannot be used on this machine, or None when it can."""
    if device is not Device.CUDA:
        return None
    if torch.version.cuda is None:
        return "this PyTorch build has no CUDA support (install a CUDA build of PyTorch, see docs/PROFILES.md)"
    if not torch.cuda.is_available():
        return "PyTorch has CUDA support but sees no GPU (check the NVIDIA driver with nvidia-smi)"
    return None


def device_name(device: Device) -> str | None:
    """Name of the GPU a CUDA profile runs on; None for CPU profiles and when there is no GPU."""
    if device is Device.CUDA and device_unavailable_reason(device) is None:
        return torch.cuda.get_device_name(0)
    return None


def check_profile_device(profile: ProfileConfig) -> None:
    """Raise RuntimeError when the profile's device is not available here."""
    reason = device_unavailable_reason(profile.device)
    if reason is not None:
        raise RuntimeError(f"Profile '{profile.name}' needs a {profile.device.value.upper()} device: {reason}")
