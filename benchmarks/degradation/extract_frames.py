"""Sample frames from every control video and cache them (at the analysis resolution) to _out/frames.npz.

Each sample is a pair of consecutive frames (t, t+1) so temporal indicators can be tested too.
Frames are read in order with VideoReplay rather than by seeking, which is not frame-exact for H.264.
"""

import json
from pathlib import Path

import numpy as np

from resilens.analysis.heuristics import prepare
from resilens.io.replay import VideoReplay
from resilens.io.validation import VideoValidator

ROOT = Path(__file__).parent
OUT = ROOT / "_out"
N_SAMPLES = 8
EDGE_FRAMES = 10  # skip the first and last frames of each clip


def sample_pairs(path: Path) -> list[tuple[int, np.ndarray]]:
    """Return (index of the first frame, (2, H, W, 3) pair) for N_SAMPLES evenly spaced pairs."""
    info = VideoValidator().validate(path)
    if info.frame_count < 2 * EDGE_FRAMES + N_SAMPLES:
        raise ValueError(f"{path}: {info.frame_count} frames is too short to sample {N_SAMPLES} pairs")
    starts = set(np.linspace(EDGE_FRAMES, info.frame_count - EDGE_FRAMES, N_SAMPLES).astype(int).tolist())

    pairs, previous = [], None
    for frame in VideoReplay(info):
        current = prepare(frame.image)
        if frame.index - 1 in starts:
            pairs.append((frame.index - 1, np.stack([previous, current])))
            if len(pairs) == len(starts):
                break
        previous = current
    if len(pairs) < len(starts):
        raise ValueError(f"{path}: ended after {len(pairs)} of {len(starts)} sampled pairs")
    return pairs


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cases = json.loads((ROOT / "manifest.json").read_text())["test_cases"]
    frames, meta = [], []
    for case in cases:
        for index, pair in sample_pairs(ROOT / case["file"]):
            frames.append(pair)
            meta.append((case["sequence"], case["distortion"], case["level"], index))
        print("done", case["file"])
    np.savez(
        OUT / "frames.npz",
        frames=np.stack(frames),
        sequence=[m[0] for m in meta],
        distortion=[m[1] for m in meta],
        level=[m[2] for m in meta],
        index=[m[3] for m in meta],
    )


if __name__ == "__main__":
    main()
