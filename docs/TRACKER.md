# Optical-flow tracker (T-04)

`src/resilens/tracking/flow_tracker.py` follows targets chosen by hand on the first frame of a video. Every frame each box moves by the median optical flow (Lucas-Kanade) of corner points in its central part. When flow fails, the box waits in place for up to 10 frames before it is dropped. Detections are not used.

The same file provides:

- **`FlowTracker`**, a `Tracker` stage for the pipeline. It tracks on `ctx.working_image`, i.e. the enhanced frame when an enhancer ran.
- **A command** for demos and reproducible runs: `python -m resilens.tracking.flow_tracker`.

## Demonstration

```powershell
python -m resilens.tracking.flow_tracker path/to/video.mp4 --save-video
```

1. A window shows the first frame, scaled to fit the screen.
2. Drag a box around a target and press **Enter** (or **Space**) to add it. Repeat for more targets, then press **Esc** to start tracking.
3. The video plays at its normal speed:
   - **green** box: moved by optical flow in this frame;
   - **orange** box: flow failed, the box is waiting in place.
4. **Space** pauses and resumes. **Q** or closing the window stops.

`--save-video` also writes `tracked.mp4` with the drawn boxes, so you can show the result later without selecting the targets again.

To show the effect of a degradation, run the same targets on the degraded clip. The `targets.json` from the first run is the input:

```powershell
python -m resilens.tracking.flow_tracker <videos>/MOT20-01/clean.mp4 --save-video
python -m resilens.tracking.flow_tracker <videos>/MOT20-01/fog_l2.mp4 --targets runs/tracking/MOT20-01_clean/targets.json --save-video
```

## Benchmarking

Results are only comparable when **every run tracks exactly the same targets**. So select targets once per sequence, then always pass `--targets`.

**1. Create the targets once per sequence.** Select them on the clean clip (as in the demo):

```powershell
python -m resilens.tracking.flow_tracker <videos>/MOT20-01/clean.mp4
```

This writes `runs/tracking/MOT20-01_clean/targets.json`. Instead of selecting by hand, you can also write the file yourself, for example from ground-truth boxes on the first frame. Boxes are `x, y, width, height` in pixels:

```json
{"video": "MOT20-01", "frame": 0, "targets": [{"id": 1, "box_xywh": [199, 813, 140, 268]}]}
```

Keep the targets files that the benchmark uses under version control or next to the dataset, so runs can be repeated.

**2. Run every clip with the same targets, without windows:**

```powershell
$targets = "runs/tracking/MOT20-01_clean/targets.json"
foreach ($clip in "clean", "fog_l1", "fog_l2", "rain_l1", "rain_l2", "low_light_l1", "low_light_l2", "motion_blur_l1", "motion_blur_l2") {
    python -m resilens.tracking.flow_tracker "<videos>/MOT20-01/$clip.mp4" --targets $targets --no-display
}
```

Each run gets its own folder `runs/tracking/<sequence folder>_<clip>/`; use `--out-dir` to choose another one. Same video and same targets give identical `tracks.txt`.

**3. Track on enhanced frames (modes B–E).** The command reads the raw video, so it measures the tracker without enhancement (mode A). To track after an enhancer, use the stage in the pipeline:

```python
from pathlib import Path

from resilens.tracking.flow_tracker import FlowTracker, load_targets

tracker = FlowTracker(load_targets(Path("runs/tracking/MOT20-01_clean/targets.json")))
tracker.setup()                # before each video
ctx = tracker.process(ctx)     # every frame, after the enhancer; fills ctx.tracks
```

Mode configs (`configs/modes/*.yaml`) still name `bytetrack`. Wiring `FlowTracker` into `Pipeline.from_config` is part of the pipeline task.

### Outputs

| File | Content |
| --- | --- |
| `targets.json` | Starting boxes, IDs 1..N in this order. Pass the file back with `--targets`. |
| `tracks.txt` | MOT format: `frame, id, x, y, w, h, score, -1, -1, -1`, frames from 1. Score `1` means the box was moved by flow; `0` means it was waiting in place (a stale position). |
| `timings.csv` | `frame, tracker_ms`: time of the tracker alone, excluding video decoding and display. |
| `tracked.mp4` | Only with `--save-video`. |

### Evaluating the results (not implemented yet)

Notes for the benchmark task:

- **Use single-object metrics, not plain TrackEval.** The tracker follows only the selected targets, while MOT20 ground truth contains every person. Running TrackEval (HOTA, IDF1) on the full ground truth counts everyone else as missed, so HOTA comes out near 0 whatever the tracker does.
- **Match targets to ground truth first.** Pair each target with the GT ID whose box overlaps it most on the first frame. Then report per target:
  - the share of frames with IoU ≥ 0.5;
  - the mean IoU;
  - the frame where the target was lost.

  Alternatively, filter the ground truth to those IDs before running TrackEval.
- **Decide how to treat waiting boxes.** Rows with score `0` are stale positions; decide once whether they count, and treat every mode the same way.
- **Use the same frames in every run.** Every run processes the whole video, even when all targets are lost, so frame counts match between runs.

## Limitations

- The box size is fixed at selection time, so overlap with the ground truth drops when a person walks towards or away from the camera.
- A lost target is never found again.
- In crowds, flow can drift onto a person who walks in front of the target.
- Works noticeably worse in rain and low light. That is expected: the project measures how much enhancement helps.
