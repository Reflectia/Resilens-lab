"""Candidate indicators for each degradation type.

Every indicator takes a BGR uint8 frame at the analysis resolution (temporal ones take a pair of
consecutive frames) and returns a float where HIGHER = MORE of that degradation.
"""

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from resilens.analysis.heuristics import (
    TOP_HAT_KERNEL,
    dark_channel,
    fog_score,
    low_light_score,
    motion_blur_score,
    rain_score,
    thin_bright_mask,
    to_gray,
    vertical_streaks,
)

DARK_PIXEL_LEVEL = 50  # luma below these counts as a "dark" pixel
MIN_THIN_BRIGHT_PIXELS = 50  # below this many thin bright pixels the vertical ratio is not meaningful


@dataclass(frozen=True)
class Candidate:
    degradation: str
    name: str
    func: Callable[[np.ndarray], float]
    needs_pair: bool = False  # True: func receives (2, H, W, 3) consecutive frames


def sobel_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1)


# --------------------------------------------------------------------------- fog
def fog_dark_channel_mean(frame: np.ndarray) -> float:
    return float(dark_channel(frame).mean())


def fog_low_contrast(frame: np.ndarray) -> float:
    return -float(to_gray(frame).std())


def fog_low_saturation(frame: np.ndarray) -> float:
    return -float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1].mean())


# --------------------------------------------------------------------------- low light
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


def blur_anisotropy(frame: np.ndarray) -> float:
    """|log(horizontal / vertical gradient energy)|: directional blur is lopsided."""
    gx, gy = sobel_gradients(to_gray(frame))
    return float(abs(np.log((np.abs(gx).mean() + 1e-6) / (np.abs(gy).mean() + 1e-6))))


# --------------------------------------------------------------------------- rain
def rain_vertical_tophat(frame: np.ndarray) -> float:
    """Density of thin bright details that are elongated vertically."""
    return float(vertical_streaks(thin_bright_mask(to_gray(frame))).mean())


def rain_vertical_ratio(frame: np.ndarray) -> float:
    """Share of thin bright pixels that lie on vertical lines."""
    mask = thin_bright_mask(to_gray(frame))
    if mask.sum() < MIN_THIN_BRIGHT_PIXELS:
        return 0.0
    return float(vertical_streaks(mask).sum() / mask.sum())


def rain_orientation_dominance(frame: np.ndarray) -> float:
    """Horizontal vs vertical gradient energy of thin details: vertical streaks favour horizontal gradients."""
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOP_HAT_KERNEL, TOP_HAT_KERNEL))
    top_hat = cv2.morphologyEx(to_gray(frame), cv2.MORPH_TOPHAT, ellipse).astype(np.float32)
    gx = np.abs(cv2.Sobel(top_hat, cv2.CV_32F, 1, 0)).mean()
    gy = np.abs(cv2.Sobel(top_hat, cv2.CV_32F, 0, 1)).mean()
    return float(gx / (gy + 1e-6))


def rain_vertical_line_response(frame: np.ndarray) -> float:
    """Thin vertical line detector: 1x15 mean minus a wider 7x15 mean, clipped at 0."""
    gray = to_gray(frame).astype(np.float32)
    return float(np.clip(cv2.blur(gray, (1, 15)) - cv2.blur(gray, (7, 15)), 0, None).mean())


CANDIDATES = [
    Candidate("fog", "dark_channel_mean", fog_dark_channel_mean),
    Candidate("fog", "dark_channel_p5", fog_score),
    Candidate("fog", "low_contrast", fog_low_contrast),
    Candidate("fog", "low_saturation", fog_low_saturation),
    Candidate("low_light", "dark_mean", lambda frame: low_light_score(to_gray(frame))),
    Candidate("low_light", "dark_median", light_dark_median),
    Candidate("low_light", "dark_fraction", light_dark_fraction),
    Candidate("low_light", "dark_p95", light_dark_p95),
    Candidate("motion_blur", "laplacian_var", blur_laplacian_var),
    Candidate("motion_blur", "laplacian_var_norm", blur_laplacian_var_norm),
    Candidate("motion_blur", "tenengrad", blur_tenengrad),
    Candidate("motion_blur", "tenengrad_norm", blur_tenengrad_norm),
    Candidate("motion_blur", "high_freq_ratio", lambda frame: motion_blur_score(to_gray(frame))),
    Candidate("motion_blur", "anisotropy", blur_anisotropy),
    Candidate("rain", "vertical_tophat", rain_vertical_tophat),
    Candidate("rain", "vertical_ratio", rain_vertical_ratio),
    Candidate("rain", "orientation_dominance", rain_orientation_dominance),
    Candidate("rain", "vertical_line_response", rain_vertical_line_response),
    Candidate("rain", "temporal_streaks", lambda pair: rain_score(to_gray(pair[0]), to_gray(pair[1])), needs_pair=True),
]
