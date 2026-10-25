"""Sample frames from every control video and cache them (downscaled) to _out/frames.npz.

Each sample is a pair of consecutive frames (t, t+1) so temporal candidates can be tested too.
"""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / "_out"
WIDTH, HEIGHT = 480, 270  # analysis resolution (fixed so all sequences stack; real Analyzer should also run downscaled)
N_SAMPLES = 8


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cases = json.loads((ROOT / "manifest.json").read_text())["test_cases"]
    frames, meta = [], []
    for case in cases:
        cap = cv2.VideoCapture(str(ROOT / case["file"]))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in np.linspace(10, total - 10, N_SAMPLES).astype(int):
            pair = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            for _ in range(2):
                ok, frame = cap.read()
                assert ok, (case["file"], idx)

                pair.append(cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
            frames.append(np.stack(pair))
            meta.append((case["sequence"], case["distortion"], case["level"], int(idx)))
        cap.release()
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
