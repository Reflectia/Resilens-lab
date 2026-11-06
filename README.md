# Resilens v2.0

Resilens is a platform for reproducible experiments with object detection and multi-object tracking in video under adverse conditions (fog, rain, low light, motion blur). It compares processing modes A–E by tracking quality, latency and resource usage.

## Running locally

Requires Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Put model weights into `weights/` and videos and scenarios into `data/` (both folders are excluded from git).
TANet weights: download [TANnet.pth](https://drive.google.com/file/d/128zwUylKmnW1-octTVxCHRIbriOfb0Kf/view?usp=sharing) and save it as `weights/tanet.pth`.
YOLOX-Nano weights: download [yolox_nano.pth](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth) and save it as `weights/yolox_nano_coco.pth`.

### Quality benchmark (T-12)

For annotated control videos, provide both COCO detection annotations and MOT tracking annotations. Videos without
annotations still run, but have no detection or tracking quality metrics. Paths may point outside the repository.

```powershell
python -m resilens.cli benchmark --manifest "B:\Frogs Vision\data\manifest.json" --profile minimal --output runs\benchmark.json
```

The benchmark defaults to all manifest scenarios in offline mode for A and B. To select a scenario, repeat
`--scenario` with a video path from the manifest, for example `--scenario control_videos/MOT20-01/clean.mp4`.
Do not start or restart the API while a CLI benchmark is running: the API's existing recovery logic can mark an
active CLI run as interrupted.
The output records each run ID and status as the runs complete. Repeating the command with the same output resumes
completed scenarios after verifying their inputs. Use the benchmark output to generate a comparison:

```powershell
python -m resilens.cli compare --benchmark runs\benchmark.json --manifest "B:\Frogs Vision\data\manifest.json" --output runs\comparison.json
```

Alternatively, pass completed A/B run IDs directly. Each pair must use the same video contents. The comparison includes
per-scenario metrics, pooled overall metrics, and groups by distortion and level from the required manifest.
Detection metrics use COCO evaluation; tracking metrics combine sequences with TrackEval. The detection confidence
threshold is 0.3, so mAP50–95 describes only detections surviving that cutoff, not a full low-confidence PR curve.
COCO mAP uses its standard limit of 100 detections per image; precision and recall use IoU 0.5 and all retained
detections. Tracking uses the MOT20 distractor preprocessing and full marked person ground truth,
and includes score-zero waiting boxes. COCO and MOT person boxes must match. Annotation hashes are checked before
evaluation and aggregation; matching dimensions and annotation files alone cannot prove an arbitrary video belongs
to a ground-truth sequence, so select matching entries from the trusted manifest.
For the provided MOT20-03 videos, the original 1173-pixel frames were encoded at 1172 pixels wide. Comparing
frames 1, 51, 151 and 300 with the original MOT20 archive supports a one-pixel crop at the right edge; feature
matching of the degraded variants against the clean video supports the same alignment. The explicit benchmark flag
records this assumption per run and keeps original bounding-box coordinates. It must only be used for this verified
dataset; it does not resize images or alter the ground truth. Without the flag, the mismatch fails validation.
This is an alignment assumption supported by sampled frames, not proof of every frame's transformation. In the
provided MOT20-03 annotations, 35 boxes cross the new right edge; retaining their original coordinates may slightly
affect their IoU. Results should be interpreted with this one-pixel boundary limitation.
FlowTracker only initializes targets from the first detected frame; it cannot add later people or recover lost targets.
Consequently its full-ground-truth tracking scores reflect those limitations.

```bash
uvicorn resilens.api.app:app --reload    # REST API: http://127.0.0.1:8000/docs
streamlit run src/resilens/ui/app.py     # Web UI:   http://localhost:8501
resilens run path/to/video.mp4 --mode A --profile minimal   # CLI
pytest                                   # tests
ruff check .                             # linter
```

Or run everything in Docker: `docker compose up --build`.

## Runs, modes and the Policy Engine

Start a run through the REST API with a mode (A–E, `configs/modes/`), a resource profile (`configs/profiles/`) and how the video is fed:

```bash
curl -F mode=D -F profile=standard -F replay_mode=realtime -F "video=@path/to/fog_l2.mp4" http://127.0.0.1:8000/runs
curl http://127.0.0.1:8000/runs/<run_id>/report
```

- `replay_mode=offline` (default) processes every frame; `realtime` plays the video at its own fps and drops the frames the pipeline is too slow for (`frames_dropped` in the report).
- Mode D runs the Policy Engine: it picks enhance / skip / bypass / downscale for every frame from the degradation, the profile and the time and resource budget. See [docs/POLICY.md](docs/POLICY.md).
- Each run writes `detections.txt`, `tracks.txt` and, in mode D, `policy_log.csv` to `runs/<run_id>/`.

## Tracker

An optical-flow tracker follows targets you select by hand on the first frame:

```bash
# Demo: draw boxes around targets (Enter = add, Esc = start), watch the tracking, save the annotated video
python -m resilens.tracking.flow_tracker path/to/clean.mp4 --save-video

# Benchmark runs: reuse the same targets on every clip, without windows
python -m resilens.tracking.flow_tracker path/to/fog_l2.mp4 --targets runs/tracking/<folder>_clean/targets.json --no-display
```

Outputs go to `runs/tracking/<folder>_<video name>/`. See [docs/TRACKER.md](docs/TRACKER.md) for controls, the benchmark workflow, output formats and how to evaluate the results.
