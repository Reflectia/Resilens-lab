"""One video frame through TANet on CPU and on GPU: do both give the same picture?

    python benchmarks/checks/tanet_cpu_vs_gpu.py control_videos/MOT20-01/fog_l2.mp4 [weights/tanet.pth]

TANet runs on a quarter-size frame, as in modes B and D. PSNR between the two outputs above ~40 dB means the GPU
result is the same as the CPU one; far below that means they differ.
"""

import sys

import cv2
import numpy as np
import torch

from resilens.enhancement.tanet import TANetEnhancer

FRAME = 100


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255**2 / mse))


def enhance(frame: np.ndarray, weights: str, device: str) -> np.ndarray:
    enhancer = TANetEnhancer(weights, device=device, scale=0.25)
    enhancer.setup()
    return enhancer.enhance(frame)


def main() -> None:
    video = sys.argv[1]
    weights = sys.argv[2] if len(sys.argv) > 2 else "weights/tanet.pth"
    capture = cv2.VideoCapture(video)
    capture.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
    ok, frame = capture.read()
    if not ok:
        raise SystemExit(f"Cannot read frame {FRAME} of {video}")

    print(f"torch {torch.__version__}, opencv {cv2.__version__}, cuda available: {torch.cuda.is_available()}")
    cpu = enhance(frame, weights, "cpu")
    print(f"input vs CPU output: PSNR {psnr(frame, cpu):.1f} dB (low: TANet changed the frame)")
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA here, GPU part skipped")
    gpu = enhance(frame, weights, "cuda")
    print(f"input vs GPU output: PSNR {psnr(frame, gpu):.1f} dB")
    difference = int(np.abs(cpu.astype(int) - gpu.astype(int)).max())
    print(f"CPU vs GPU output: PSNR {psnr(cpu, gpu):.1f} dB, largest pixel difference {difference}")


if __name__ == "__main__":
    main()
