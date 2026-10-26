"""Model-free degradation indicators chosen by the benchmark in benchmarks/degradation/.

Every score is HIGHER = MORE of that degradation. Pixel constants assume frames resized by prepare(),
so scores are comparable between videos of any resolution.
"""

from functools import lru_cache

import cv2
import numpy as np

ANALYSIS_SIZE = (480, 270)  # width, height; all constants below are tuned for it
DARK_CHANNEL_PATCH = 15  # px, patch size for the dark channel
TOP_HAT_KERNEL = 7  # px, removes details wider than this before looking for thin bright ones
THIN_BRIGHT_LEVEL = 25  # top-hat response above this counts as a thin bright detail
STREAK_LENGTH = 15  # px, minimal vertical length of a rain streak
STREAK_SHIFT = 5  # px, horizontal tolerance when matching a streak to the previous frame
HF_CUTOFF = 0.15  # normalised frequency separating low from high spatial frequencies


def prepare(image: np.ndarray) -> np.ndarray:
    """Resize a BGR frame to the analysis resolution."""
    return cv2.resize(image, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def dark_channel(image: np.ndarray) -> np.ndarray:
    """Per-pixel minimum over channels, eroded over a local patch."""
    patch = np.ones((DARK_CHANNEL_PATCH, DARK_CHANNEL_PATCH), np.uint8)
    blue, green, red = cv2.split(image)
    return cv2.erode(cv2.min(cv2.min(blue, green), red), patch)  # ~8x faster than image.min(axis=2)


def thin_bright_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask of thin bright details (white top-hat above a threshold)."""
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOP_HAT_KERNEL, TOP_HAT_KERNEL))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, ellipse)
    return (top_hat > THIN_BRIGHT_LEVEL).astype(np.uint8)


def vertical_streaks(mask: np.ndarray) -> np.ndarray:
    """Keep only mask pixels that belong to near-vertical lines."""
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((STREAK_LENGTH, 1), np.uint8))


@lru_cache(maxsize=4)
def high_frequency_mask(height: int, width: int) -> np.ndarray:
    """Bins of a real 2D FFT (rfft2 layout) whose normalised frequency is above HF_CUTOFF."""
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    return np.hypot(fy, fx) > HF_CUTOFF


def fog_score(image: np.ndarray) -> float:
    """5th percentile of the dark channel: fog lifts even the darkest patches."""
    return float(np.percentile(dark_channel(image), 5))


def low_light_score(gray: np.ndarray) -> float:
    """255 minus mean luma."""
    return 255.0 - float(gray.mean())


def motion_blur_score(gray: np.ndarray) -> float:
    """Negative share of spectral energy at high spatial frequencies: blur removes fine detail."""
    gray = gray.astype(np.float32)
    spectrum = np.abs(np.fft.rfft2(gray - gray.mean()))
    high = high_frequency_mask(*gray.shape)
    return -float(spectrum[high].sum() / (spectrum.sum() + 1e-6))


def rain_score(previous_gray: np.ndarray, gray: np.ndarray) -> float:
    """Share of pixels on vertical streaks that were absent in the previous frame.

    Rain streaks move between frames while static vertical edges (poles, door frames) stay,
    so only new streaks count. Needs two consecutive frames.
    """
    before, after = (vertical_streaks(thin_bright_mask(g)) for g in (previous_gray, gray))
    persisted = cv2.dilate(before, np.ones((1, STREAK_SHIFT), np.uint8))
    return float((after & (1 - persisted)).mean())
