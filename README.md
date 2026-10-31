# Resilens v2.0

Resilens is a platform for reproducible experiments with object detection and multi-object tracking in video under adverse conditions (fog, rain, low light, motion blur). It compares processing modes A–E by tracking quality, latency and resource usage.

## Running locally

Requires Python 3.13.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Put model weights into `weights/` and videos and scenarios into `data/` (both folders are excluded from git).
TANet weights: download [TANnet.pth](https://drive.google.com/file/d/128zwUylKmnW1-octTVxCHRIbriOfb0Kf/view?usp=sharing) and save it as `weights/tanet.pth`.

```powershell
uvicorn resilens.api.app:app --reload    # REST API: http://127.0.0.1:8000/docs
streamlit run src/resilens/ui/app.py     # Web UI:   http://localhost:8501
resilens run path/to/video.mp4 --mode A --profile minimal   # CLI
pytest                                   # tests
ruff check .                             # linter
```

Or run everything in Docker: `docker compose up --build`.

## Tracker

An optical-flow tracker follows targets you select by hand on the first frame:

```powershell
# Demo: draw boxes around targets (Enter = add, Esc = start), watch the tracking, save the annotated video
python -m resilens.tracking.flow_tracker path/to/clean.mp4 --save-video

# Benchmark runs: reuse the same targets on every clip, without windows
python -m resilens.tracking.flow_tracker path/to/fog_l2.mp4 --targets runs/tracking/<folder>_clean/targets.json --no-display
```

Outputs go to `runs/tracking/<folder>_<video name>/`. See [docs/TRACKER.md](docs/TRACKER.md) for controls, the benchmark workflow, output formats and how to evaluate the results.
