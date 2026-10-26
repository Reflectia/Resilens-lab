"""Optical-flow tracker for targets selected by hand, with a command for demos and reproducible runs.

Tracking: every frame each box moves by the median Lucas-Kanade motion of corner points in its central part.
Points that do not come back to where they started when tracked backwards are ignored. If flow fails, the box
waits in place (score 0) for up to MAX_MISSES frames before it is dropped. Detections are not used.

In the pipeline, FlowTracker(targets) is a Tracker stage that follows ctx.working_image (the enhanced frame).

Command line:
    python -m resilens.tracking.flow_tracker VIDEO                          # select targets in a window
    python -m resilens.tracking.flow_tracker VIDEO --targets targets.json   # same targets again, no selection

Selection window: drag a box, Enter or Space adds it, Esc starts tracking.
Playback: Space pauses and resumes, Q or closing the window stops.

Outputs in runs/tracking/<folder>_<video name>/ (or --out-dir):
    targets.json  starting boxes; pass it with --targets to repeat the run on another clip or mode
    tracks.txt    MOT format: frame, id, x, y, w, h, score, -1, -1, -1 (score 1 = moved by flow, 0 = waiting)
    timings.csv   tracker time per frame in ms
    tracked.mp4   video with drawn boxes (only with --save-video)
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from resilens.core.contracts import Detection, Track
from resilens.tracking.base import Tracker

Box = tuple[float, float, float, float]  # x, y, width, height

MIN_POINTS = 5  # fewer reliable points than this means flow failed for the target
MAX_MISSES = 10  # frames a target may wait without usable flow before it is dropped
MAX_FB_ERROR = 1.0  # px; points that do not return closer than this when tracked backwards are ignored
MARGIN = 0.2  # share of the box width/height cut from each side: edges of a person box are mostly background
LK_PARAMS = {"winSize": (21, 21), "maxLevel": 3}
PERSON_CLASS_ID = 0  # COCO "person"

DISPLAY_WIDTH = 1280  # windows are scaled down to this width so 1080p video fits on a laptop screen
SELECT_WINDOW = "Drag a box, Enter = add, Esc = start tracking"
PLAY_WINDOW = "Tracking (Space = pause, Q = quit)"
MOVED_COLOR, WAITING_COLOR = (0, 255, 0), (0, 165, 255)  # green, orange (BGR)


# --------------------------------------------------------------------------- tracking
def find_points(gray: np.ndarray, box: Box) -> np.ndarray | None:
    """Corner points in the central part of the box, shape (N, 1, 2)."""
    x, y, w, h = box
    height, width = gray.shape
    x1, y1 = max(int(x + MARGIN * w), 0), max(int(y + MARGIN * h), 0)
    x2, y2 = min(int(x + (1 - MARGIN) * w), width), min(int(y + (1 - MARGIN) * h), height)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    corners = cv2.goodFeaturesToTrack(gray[y1:y2, x1:x2], maxCorners=50, qualityLevel=0.01, minDistance=3)
    return None if corners is None else corners + np.array([x1, y1], dtype=np.float32)


def track_box(previous_gray: np.ndarray, gray: np.ndarray, box: Box) -> Box | None:
    """Move the box by the median optical flow of its reliable points; None if flow failed."""
    points = find_points(previous_gray, box)
    if points is None or len(points) < MIN_POINTS:
        return None
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, points, None, **LK_PARAMS)
    back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, previous_gray, moved, None, **LK_PARAMS)
    error = np.linalg.norm(points - back, axis=2).ravel()
    reliable = (status.ravel() == 1) & (status_back.ravel() == 1) & (error < MAX_FB_ERROR)
    if reliable.sum() < MIN_POINTS:
        return None

    dx, dy = np.median(moved[reliable] - points[reliable], axis=0).ravel()
    x, y, w, h = box
    return (float(x + dx), float(y + dy), w, h)


def move_tracks(
    previous_gray: np.ndarray, gray: np.ndarray, tracks: dict[int, Box], misses: dict[int, int]
) -> dict[int, Box]:
    """Move all tracks to the new frame; `misses` counts frames without usable flow per track."""
    height, width = gray.shape
    updated: dict[int, Box] = {}
    for track_id, box in tracks.items():
        moved = track_box(previous_gray, gray, box)
        if moved is None:
            misses[track_id] = misses.get(track_id, 0) + 1
            moved = box  # keep the last box and try again on the next frame
        else:
            misses[track_id] = 0
        x, y, w, h = moved
        inside = 0 <= x + w / 2 < width and 0 <= y + h / 2 < height
        if inside and misses[track_id] <= MAX_MISSES:
            updated[track_id] = moved
    return updated


class FlowTracker(Tracker):
    """Tracker stage: starts on the first frame it gets, with target IDs 1..N in the order of `targets`."""

    name = "flow"

    def __init__(self, targets: list[Box]) -> None:
        self.targets = targets
        self.setup()

    def setup(self) -> None:
        self.tracks: dict[int, Box] = {}
        self.misses: dict[int, int] = {}
        self._previous_gray: np.ndarray | None = None

    def update(self, detections: list[Detection], frame_index: int, image: np.ndarray) -> list[Track]:
        """Detections are ignored: targets are followed by optical flow only."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self._previous_gray is None:
            self.tracks = {i + 1: box for i, box in enumerate(self.targets)}
        else:
            self.tracks = move_tracks(self._previous_gray, gray, self.tracks, self.misses)
        self._previous_gray = gray

        tracks: list[Track] = []
        for track_id, (x, y, w, h) in self.tracks.items():
            score = 1.0 if self.misses.get(track_id, 0) == 0 else 0.0
            tracks.append(Track(track_id, (x, y, x + w, y + h), score, PERSON_CLASS_ID))
        return tracks


# --------------------------------------------------------------------------- command line
def save_targets(path: Path, video: Path, boxes: list[Box]) -> None:
    targets = [{"id": i + 1, "box_xywh": [round(v, 1) for v in box]} for i, box in enumerate(boxes)]
    path.write_text(json.dumps({"video": str(video), "frame": 0, "targets": targets}, indent=2), encoding="utf-8")


def load_targets(path: Path) -> list[Box]:
    boxes: list[Box] = []
    for target in json.loads(path.read_text(encoding="utf-8"))["targets"]:
        x, y, w, h = target["box_xywh"]
        boxes.append((float(x), float(y), float(w), float(h)))
    return boxes


def fit_to_screen(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """The frame scaled down to DISPLAY_WIDTH (never up), and the scale factor."""
    scale = min(1.0, DISPLAY_WIDTH / frame.shape[1])
    return (cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame), scale


def select_targets(frame: np.ndarray) -> list[Box]:
    """Let the user draw target boxes on a screen-sized copy of the frame; boxes are in full-frame pixels."""
    shown, scale = fit_to_screen(frame)
    boxes = cv2.selectROIs(SELECT_WINDOW, shown, showCrosshair=False)
    cv2.destroyWindow(SELECT_WINDOW)
    return [(x / scale, y / scale, w / scale, h / scale) for x, y, w, h in boxes]


def draw(frame: np.ndarray, tracks: list[Track]) -> None:
    for track in tracks:
        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        color = MOVED_COLOR if track.score > 0 else WAITING_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, str(track.track_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def show(frame: np.ndarray, delay_ms: int) -> bool:
    """Display the frame; False when the user stops (Q or the window closed)."""
    cv2.imshow(PLAY_WINDOW, fit_to_screen(frame)[0])
    key = cv2.waitKey(max(delay_ms, 1)) & 0xFF
    if key == ord(" "):
        key = cv2.waitKey(0) & 0xFF  # paused until the next key press
    return key != ord("q") and cv2.getWindowProperty(PLAY_WINDOW, cv2.WND_PROP_VISIBLE) >= 1


def mot_row(frame_index: int, track: Track) -> str:
    x1, y1, x2, y2 = track.bbox
    box = f"{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f}"
    return f"{frame_index + 1},{track.track_id},{box},{track.score:.0f},-1,-1,-1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Track hand-selected targets with optical flow.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--targets", type=Path, help="targets.json from an earlier run; skips the selection window")
    parser.add_argument("--out-dir", type=Path, help="default: runs/tracking/<folder>_<video name>")
    parser.add_argument("--no-display", action="store_true", help="run without windows (needs --targets)")
    parser.add_argument("--save-video", action="store_true", help="also write the video with drawn boxes")
    args = parser.parse_args()
    if args.no_display and not args.targets:
        parser.error("--no-display needs --targets")

    capture = cv2.VideoCapture(str(args.video))
    ok, frame = capture.read()
    if not ok:
        raise SystemExit(f"Cannot read video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    boxes = load_targets(args.targets) if args.targets else select_targets(frame)
    if not boxes:
        raise SystemExit("No targets selected")
    out_dir = args.out_dir or Path("runs") / "tracking" / f"{args.video.parent.name}_{args.video.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_targets(out_dir / "targets.json", args.video, boxes)

    writer = None
    if args.save_video:
        size = (frame.shape[1], frame.shape[0])
        writer = cv2.VideoWriter(str(out_dir / "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    tracker = FlowTracker(boxes)
    rows: list[str] = []
    timings_ms: list[float] = []
    frame_index = 0
    while ok:
        started = time.perf_counter()
        tracks = tracker.update([], frame_index, frame)
        timings_ms.append((time.perf_counter() - started) * 1000)
        rows += [mot_row(frame_index, track) for track in tracks]

        if writer is not None or not args.no_display:
            draw(frame, tracks)
        if writer is not None:
            writer.write(frame)
        if not args.no_display and not show(frame, int(1000 / fps - timings_ms[-1])):
            break
        ok, frame = capture.read()
        frame_index += 1

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    (out_dir / "tracks.txt").write_text("".join(row + "\n" for row in rows), encoding="utf-8")
    timing_lines = [f"{i},{ms:.2f}" for i, ms in enumerate(timings_ms)]
    (out_dir / "timings.csv").write_text("frame,tracker_ms\n" + "\n".join(timing_lines) + "\n", encoding="utf-8")

    p50, p95 = np.percentile(timings_ms, [50, 95])
    print(f"{len(timings_ms)} frames, {len(tracker.tracks)} of {len(boxes)} targets still tracked at the end")
    print(f"Tracker time per frame: p50 {p50:.1f} ms, p95 {p95:.1f} ms. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
