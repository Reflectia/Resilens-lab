"""Candidate indicators for each degradation type.

Every indicator takes a BGR uint8 frame (temporal ones take a pair of consecutive frames)
and returns a float where HIGHER = MORE of that degradation.
"""

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

DARK_CHANNEL_PATCH = 15  # px, patch size for the dark channel
DARK_PIXEL_LEVEL = 50  # luma below this counts as a "dark" pixel
THIN_BRIGHT_LEVEL = 25  # top-hat response above this counts as a thin bright detail
STREAK_LENGTH = 15  # px, minimal vertical length of a rain streak
HF_CUTOFF = 0.15  # normalised radius separating low from high spatial frequencies


@dataclass(frozen=True)
class Candidate:
    degradation: str
    name: str
    func: Callable[[np.ndarray], float]
    needs_pair: bool = False  # True: func receives (2, H, W, 3) consecutive frames


def to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def dark_channel(frame: np.ndarray) -> np.ndarray:
    """Per-pixel minimum over channels, eroded over a local patch."""
    patch = np.ones((DARK_CHANNEL_PATCH, DARK_CHANNEL_PATCH), np.uint8)
    return cv2.erode(frame.min(axis=2), patch)


def thin_bright_mask(frame: np.ndarray) -> np.ndarray:
    """Binary mask of thin bright details (white top-hat above a threshold)."""
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    top_hat = cv2.morphologyEx(to_gray(frame), cv2.MORPH_TOPHAT, ellipse)
    return (top_hat > THIN_BRIGHT_LEVEL).astype(np.uint8)


def vertical_streaks(mask: np.ndarray) -> np.ndarray:
    """Keep only mask pixels that belong to near-vertical lines."""
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((STREAK_LENGTH, 1), np.uint8))


def sobel_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1)


# --------------------------------------------------------------------------- fog
def fog_dark_channel_mean(frame: np.ndarray) -> float:
    return float(dark_channel(frame).mean())


def fog_dark_channel_p5(frame: np.ndarray) -> float:
    """5th percentile of the dark channel: fog lifts even the darkest patches."""
    return float(np.percentile(dark_channel(frame), 5))


def fog_low_contrast(frame: np.ndarray) -> float:
    return -float(to_gray(frame).std())


def fog_low_saturation(frame: np.ndarray) -> float:
    return -float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1].mean())


# --------------------------------------------------------------------------- low light
def light_dark_mean(frame: np.ndarray) -> float:
    return 255.0 - float(to_gray(frame).mean())


def light_dark_median(frame: np.ndarray) -> float:
    return 255.0 - float(np.median(to_gray(frame)))


def light_dark_fraction(frame: np.ndarray) -> float:
    return float((to_gray(frame) < DARK_PIXEL_LEVEL).mean())


def light_dark_p95(frame: np.ndarray) -> float:
    """Even the bright tail is dim."""
    return 255.0 - float(np.percentile(to_gray(frame), 95))


# --------------------------------------------------------------------------- motion blur
def blur_laplacian_var(frame: np.ndarray) -> float:
    return -float(cv2.Laplacian(to_gray(frame), cv2.CV_64F).var())


def blur_laplacian_var_norm(frame: np.ndarray) -> float:
    """Laplacian variance divided by luma variance (less scene-dependent)."""
    gray = to_gray(frame)
    return -float(cv2.Laplacian(gray, cv2.CV_64F).var() / (gray.astype(np.float64).var() + 1e-6))


def blur_tenengrad(frame: np.ndarray) -> float:
    gx, gy = sobel_gradients(to_gray(frame))
    return -float((gx**2 + gy**2).mean())


def blur_tenengrad_norm(frame: np.ndarray) -> float:
    gray = to_gray(frame)
    gx, gy = sobel_gradients(gray)
    return -float((gx**2 + gy**2).mean() / (gray.astype(np.float64).var() + 1e-6))


def blur_high_freq_ratio(frame: np.ndarray) -> float:
    """Negative share of spectral energy above the low-frequency disc."""
    gray = to_gray(frame).astype(np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean())))
    h, w = gray.shape
    rows, cols = np.ogrid[:h, :w]
    radius = np.hypot((rows - h / 2) / h, (cols - w / 2) / w)
    return -float(spectrum[radius > HF_CUTOFF].sum() / (spectrum.sum() + 1e-6))


def blur_anisotropy(frame: np.ndarray) -> float:
    """|log(horizontal / vertical gradient energy)|: directional blur is lopsided."""
    gx, gy = sobel_gradients(to_gray(frame))
    return float(abs(np.log((np.abs(gx).mean() + 1e-6) / (np.abs(gy).mean() + 1e-6))))


# --------------------------------------------------------------------------- rain
def rain_vertical_tophat(frame: np.ndarray) -> float:
    """Density of thin bright details that are elongated vertically."""
    return float(vertical_streaks(thin_bright_mask(frame)).mean())


def rain_vertical_ratio(frame: np.ndarray) -> float:
    """Share of thin bright pixels that lie on vertical lines."""
    mask = thin_bright_mask(frame)
    if mask.sum() < 50:
        return 0.0
    return float(vertical_streaks(mask).sum() / mask.sum())


def rain_orientation_dominance(frame: np.ndarray) -> float:
    """Horizontal vs vertical gradient energy of thin details: vertical streaks favour horizontal gradients."""
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    top_hat = cv2.morphologyEx(to_gray(frame), cv2.MORPH_TOPHAT, ellipse).astype(np.float32)
    gx = np.abs(cv2.Sobel(top_hat, cv2.CV_32F, 1, 0)).mean()
    gy = np.abs(cv2.Sobel(top_hat, cv2.CV_32F, 0, 1)).mean()
    return float(gx / (gy + 1e-6))


def rain_vertical_line_response(frame: np.ndarray) -> float:
    """Thin vertical line detector: 1x15 mean minus a wider 7x15 mean, clipped at 0."""
    gray = to_gray(frame).astype(np.float32)
    return float(np.clip(cv2.blur(gray, (1, 15)) - cv2.blur(gray, (7, 15)), 0, None).mean())


def rain_temporal_streaks(pair: np.ndarray) -> float:
    """Vertical streaks in frame t+1 that were absent (even after a small horizontal dilation) in frame t."""
    before, after = (vertical_streaks(thin_bright_mask(frame)) for frame in pair)
    persisted = cv2.dilate(before, np.ones((1, 5), np.uint8))
    return float((after & (1 - persisted)).mean())


CANDIDATES = [
    Candidate("fog", "dark_channel_mean", fog_dark_channel_mean),
    Candidate("fog", "dark_channel_p5", fog_dark_channel_p5),
    Candidate("fog", "low_contrast", fog_low_contrast),
    Candidate("fog", "low_saturation", fog_low_saturation),
    Candidate("low_light", "dark_mean", light_dark_mean),
    Candidate("low_light", "dark_median", light_dark_median),
    Candidate("low_light", "dark_fraction", light_dark_fraction),
    Candidate("low_light", "dark_p95", light_dark_p95),
    Candidate("motion_blur", "laplacian_var", blur_laplacian_var),
    Candidate("motion_blur", "laplacian_var_norm", blur_laplacian_var_norm),
    Candidate("motion_blur", "tenengrad", blur_tenengrad),
    Candidate("motion_blur", "tenengrad_norm", blur_tenengrad_norm),
    Candidate("motion_blur", "high_freq_ratio", blur_high_freq_ratio),
    Candidate("motion_blur", "anisotropy", blur_anisotropy),
    Candidate("rain", "vertical_tophat", rain_vertical_tophat),
    Candidate("rain", "vertical_ratio", rain_vertical_ratio),
    Candidate("rain", "orientation_dominance", rain_orientation_dominance),
    Candidate("rain", "vertical_line_response", rain_vertical_line_response),
    Candidate("rain", "temporal_streaks", rain_temporal_streaks, needs_pair=True),
]
