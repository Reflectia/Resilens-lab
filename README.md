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

### Quality benchmark

The benchmark runs modes on annotated videos (`--modes`, default `A B`) and compares each of them with mode A, the reference. Each scenario needs matching COCO and MOT annotations.
From the project directory, run all scenarios in `data/manifest.json`:

```bash
python -m resilens.cli benchmark
```

To select one scenario or use a different manifest, pass the paths explicitly:

```bash
python -m resilens.cli benchmark --manifest /path/to/manifest.json --scenario control_videos/MOT20-01/clean.mp4
```

Progress is saved in `runs/benchmark.json`; each run's metrics and report are in `runs/<run_id>/`. Use `--output`
to choose another progress file. Running the same command again resumes completed scenarios after checking inputs.

Create a comparison of the completed runs (every video needs a run in each compared mode, mode A included):

```bash
python -m resilens.cli compare --benchmark runs/benchmark.json --manifest data/manifest.json --output runs/comparison.json
```

The comparison contains quality metrics and speed and resources (latency p50/p95, peak RAM/CPU/GPU) per scenario, by distortion and level, and overall. Detection evaluation uses a
confidence threshold of 0.3. FlowTracker cannot add new targets later or recover lost targets, which affects tracking
scores. Videos without annotations can be processed with `run`, but do not have quality metrics.
For annotated realtime runs, dropped frames count as frames without detections or tracks in quality metrics.

The provided MOT20-03 videos and annotations differ in width and must be aligned before evaluation. Do not start or
restart the API during a CLI benchmark: it can mark active runs as interrupted.

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
