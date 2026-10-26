import cv2
import numpy as np
import pytest

from resilens.analysis.heuristics import (
    ANALYSIS_SIZE,
    fog_score,
    low_light_score,
    motion_blur_score,
    prepare,
    rain_score,
    to_gray,
)


@pytest.fixture
def scene() -> np.ndarray:
    """Textured color frame at the analysis resolution: random blocks plus fine noise."""
    rng = np.random.default_rng(0)
    width, height = ANALYSIS_SIZE
    blocks = rng.integers(0, 256, (height // 15, width // 15, 3), dtype=np.uint8)
    image = cv2.resize(blocks, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.clip(image.astype(int) + rng.integers(-20, 21, image.shape), 0, 255).astype(np.uint8)


def add_vertical_streaks(gray: np.ndarray, columns: range) -> np.ndarray:
    streaked = gray.copy()
    for x in columns:
        streaked[40:120, x] = 255
    return streaked


def test_prepare_resizes_to_analysis_size() -> None:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    assert prepare(image).shape == (ANALYSIS_SIZE[1], ANALYSIS_SIZE[0], 3)


def test_fog_score_rises_with_haze(scene) -> None:
    fog = cv2.addWeighted(scene, 0.4, np.full_like(scene, 200), 0.6, 0)

    assert fog_score(fog) > fog_score(scene)


def test_low_light_score_rises_when_darker(scene) -> None:
    dark = (scene * 0.3).astype(np.uint8)

    assert low_light_score(to_gray(dark)) > low_light_score(to_gray(scene))


def test_motion_blur_score_rises_with_blur(scene) -> None:
    kernel = np.zeros((1, 15), np.float32)
    kernel[0, :] = 1 / 15
    blurred = cv2.filter2D(scene, -1, kernel)

    assert motion_blur_score(to_gray(blurred)) > motion_blur_score(to_gray(scene))


def test_rain_score_counts_new_streaks(scene) -> None:
    gray = to_gray(scene) // 2  # darker background so streaks stand out
    rainy = add_vertical_streaks(gray, range(20, 460, 20))

    assert rain_score(gray, gray) == 0
    assert rain_score(gray, rainy) > 0


def test_rain_score_ignores_static_vertical_lines(scene) -> None:
    gray = add_vertical_streaks(to_gray(scene) // 2, range(20, 460, 20))  # e.g. poles present in both frames

    assert rain_score(gray, gray) == 0
