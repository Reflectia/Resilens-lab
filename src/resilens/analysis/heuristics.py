import numpy as np


def dark_channel_score(image: np.ndarray) -> float:
    """Fog indicator."""
    pass


def brightness_score(image: np.ndarray) -> float:
    """Low-light indicator."""
    pass


def laplacian_variance(image: np.ndarray) -> float:
    """Motion-blur indicator (low variance = blurry)."""
    pass


def rain_streak_score(image: np.ndarray) -> float:
    """Rain indicator based on oriented streaks."""
    pass
