"""Video of a finished run: its tracks and detections drawn on the source frames, for viewing in a browser."""

import csv
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from resilens.io.validation import VIDEO_BACKEND

ANNOTATED_VIDEO = "annotated.webm"  # VP8 in WebM: plays in every browser and needs no extra codec library
DETECTION_COLOR = (214, 120, 42)  # BGR of #2a78d6, blue
TRACK_COLOR = (52, 104, 235)  # BGR of #eb6834, orange
TEXT_COLOR = (255, 255, 255)
BANNER_COLOR = (32, 32, 32)

Box = tuple[float, float, float, float]  # x, y, width, height in source pixels


def read_mot(path: Path) -> dict[int, list[tuple[int, Box, float]]]:
    """MOTChallenge rows grouped by 1-based frame: (id, box, score). A missing file has no rows."""
    rows: dict[int, list[tuple[int, Box, float]]] = defaultdict(list)
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        frame, object_id, x, y, w, h, score = line.split(",")[:7]
        rows[int(frame)].append((int(object_id), (float(x), float(y), float(w), float(h)), float(score)))
    return rows


def read_policy_log(path: Path) -> dict[int, str]:
    """Short caption of the Policy Engine decision per 1-based frame, e.g. 'fog L2 -> enhance'."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as file:
        return {
            int(row["frame"]): f"{row['degradation'] or 'unknown'} L{row['level'] or '-'} -> {row['action']}"
            + (" (downgraded)" if row["downgraded"] == "1" else "")
            for row in csv.DictReader(file)
        }


def _label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    x, y = origin
    y = max(y, height + baseline)
    cv2.rectangle(image, (x, y - height - baseline), (x + width + 4, y), color, cv2.FILLED)
    cv2.putText(image, text, (x + 2, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)


def draw_frame(
    image: np.ndarray,
    scale: float,
    frame_number: int,
    frame_count: int,
    detections: list[tuple[int, Box, float]],
    tracks: list[tuple[int, Box, float]],
    caption: str | None = None,
) -> np.ndarray:
    """Draw detections (thin, blue), tracks (thick, orange, with ID) and a banner with frame and policy caption."""

    def corners(box: Box) -> tuple[tuple[int, int], tuple[int, int]]:
        x, y, w, h = (round(v * scale) for v in box)
        return (x, y), (x + w, y + h)

    for _, box, _ in detections:
        cv2.rectangle(image, *corners(box), DETECTION_COLOR, 1)
    for track_id, box, _ in tracks:
        top_left, bottom_right = corners(box)
        cv2.rectangle(image, top_left, bottom_right, TRACK_COLOR, 2)
        _label(image, f"#{track_id}", (top_left[0], top_left[1] - 1), TRACK_COLOR)

    banner = f"frame {frame_number}/{frame_count}  det {len(detections)}  tracks {len(tracks)}"
    if caption:
        banner += f"  |  {caption}"
    _label(image, banner, (4, 20), BANNER_COLOR)
    return image


def render_annotated_video(
    source: Path,
    output: Path,
    detections_file: Path,
    tracks_file: Path,
    policy_log: Path | None = None,
    max_width: int = 960,
) -> Path:
    """Write the annotated video of a run; frames are downscaled to max_width to keep the file small."""
    capture = cv2.VideoCapture(str(source), VIDEO_BACKEND)
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open source video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    scale = min(1.0, max_width / width)
    size = (round(width * scale) // 2 * 2, round(height * scale) // 2 * 2)  # VP8 wants even dimensions

    detections, tracks = read_mot(detections_file), read_mot(tracks_file)
    captions = read_policy_log(policy_log) if policy_log else {}
    temporary = output.with_name(f".{output.stem}.{uuid4().hex}{output.suffix}")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"VP80"), fps, size)
    try:
        if not writer.isOpened():
            raise RuntimeError("OpenCV cannot write WebM (VP8) video in this environment")
        frame_number = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if frame.shape[1::-1] != size else frame
            writer.write(
                draw_frame(
                    frame,
                    scale,
                    frame_number,
                    frame_count,
                    detections.get(frame_number, []),
                    tracks.get(frame_number, []),
                    captions.get(frame_number),
                )
            )
        writer.release()
        if frame_number == 0:
            raise ValueError(f"Source video has no frames: {source}")
        temporary.replace(output)
    finally:
        writer.release()
        capture.release()
        temporary.unlink(missing_ok=True)
    return output
