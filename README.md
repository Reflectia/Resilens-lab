# Resilens v2.0

Resilens is a platform for reproducible experiments with object detection and multi-object tracking in video under adverse conditions (fog, rain, low light, motion blur). It compares processing modes A–E by tracking quality, latency and resource usage.

## Running locally

Requires Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -c requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"
```

`-c requirements.lock` installs the exact versions the project is tested with (CI and the Docker image install the
same way), and the extra index provides the CPU build of PyTorch the lock pins. On a GPU machine use
`requirements-cu130.lock` instead, see [GPU](#gpu-performance-profile). How runs record the code version and the
environment, and how to update the locks: [Reproducibility](#reproducibility-code-version-and-environment).

Put model weights into `weights/` and videos and scenarios into `data/` (both folders are excluded from git).
TANet weights: download [TANnet.pth](https://drive.google.com/file/d/128zwUylKmnW1-octTVxCHRIbriOfb0Kf/view?usp=sharing) and save it as `weights/tanet.pth`.
YOLOX-Nano weights: download [yolox_nano.pth](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth) and save it as `weights/yolox_nano_coco.pth`.

### Local integration tests with trained weights

These opt-in tests run A and D on a real video with each configured profile; the performance profile requires CUDA.
They process the entire video and can take a long time on CPU. Missing assets or CUDA are reported as skips, not passes.
Run from the repository root (use paths appropriate to your machine):

```bash
python -m pytest -v -rs tests/integration/test_real_weights.py --run-real-weights --test-video /path/to/short-video.mp4 --weights-dir /path/to/weights
```

The weights directory must contain `yolox_nano_coco.pth` and `tanet.pth`. The tests use temporary run stores.
They force analyzer estimates for controlled policy paths and use a test rule for `enhance`; a separate test uses the
real analyzer and normal rules. The default test suite skips these tests, including in CI.

### Browser E2E tests

Run from the repository root with a short clip and matching COCO JSON and MOT TXT files.

```bash
python -m pip install -c requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev,e2e]"
python -m playwright install chromium
python -m pytest -v e2e --e2e-video "../data/control_videos/MOT20-01/clean.mp4" --e2e-coco "../data/ground_truth/MOT20-01_coco.json" --e2e-mot "../data/ground_truth/MOT20-01_gt.txt"
```

Seven local tests cover mode A results and errors, plus COCO/MOT upload and validation. They start the API and Streamlit
automatically and run on CPU. Weights default to `weights/yolox_nano_coco.pth` (`--e2e-weights PATH` overrides it).
CI excludes this suite.

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
scores. Videos without annotations can be processed with `run`, but do not have quality metrics. A single annotated video
can also be evaluated with `run --annotations <coco.json> --tracking-annotations <gt.txt>`, through the REST API or in
the Web UI (see below).
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

Or run everything in Docker: `GIT_COMMIT=$(git rev-parse HEAD) docker compose up --build`. The image installs from
`requirements.lock` and has no `.git`, so the commit is passed at build time; without `GIT_COMMIT` runs record the
commit as unknown.

### GPU (performance profile)

The `performance` profile runs TANet and YOLOX on an NVIDIA GPU and needs the CUDA build of PyTorch, pinned in
`requirements-cu130.lock` (CUDA 13.0, NVIDIA driver 580 or newer). On the GPU machine install from that lock instead:

```bash
pip install -c requirements-cu130.lock --extra-index-url https://download.pytorch.org/whl/cu130 -e ".[dev]"
```

Then check it as described in [docs/PROFILES.md](docs/PROFILES.md#gpu-check-of-the-performance-profile-d-10).
Without a GPU the profile is listed as unavailable: `GET /profiles` gives the reason, the Web UI does not start it and
`POST /runs` answers 422.

## Reproducibility: code version and environment

Every run records the code and the environment it was executed with, so its results can be reproduced and explained. 
They are saved in `runs/<run_id>/environment.json` before processing starts, so failed runs have them too,
and copied into the run report (`report.json`, `GET /runs/<run_id>/report`: fields `code` and `environment`):

- `code`: git commit and branch, and whether the code had uncommitted changes (`git_dirty`): edited tracked files, a
  hash of their diff, and new files in `src/` and `configs/` not committed yet. With uncommitted changes the commit
  alone is not the code that ran.
- `environment`: full Python version, OS, CPU, RAM, PyTorch, CUDA, cuDNN, NVIDIA driver and GPUs, and every installed
  package with its exact version. `lock_file` names the lock the environment should match (`requirements.lock`, or
  `requirements-cu130.lock` for a CUDA build of PyTorch), and `differs_from_lock` lists installed packages whose
  version differs from it: empty when the environment was installed from the lock.

The Web UI shows a summary on the run page under **Code and environment**; benchmark results name the commit of each
run, and `benchmarks/profiles/measure.py` writes the same information to `profiles.json`. Both are captured once per
process: the API records the code it was started with, so restart it after changing the code.

To reproduce a run, check out its commit, install from the lock it names, and start it again with its
`config.json`. The input hashes are in `fingerprint.json`.

The lock files pin every dependency, including transitive ones and the dev and e2e extras, for any OS. After changing
dependencies in `pyproject.toml`, update both locks and commit them together; CI fails while they do not match:

```bash
python scripts/lock_dependencies.py            # keeps locked versions that still fit pyproject.toml
python scripts/lock_dependencies.py --upgrade-package numpy   # upgrade one package on purpose
```

## Runs, modes and the Policy Engine

Start a run through the REST API with a mode (A–E, `configs/modes/`), a resource profile (`configs/profiles/`) and how the video is fed:

```bash
curl -F mode=D -F profile=standard -F replay_mode=realtime -F "video=@path/to/fog_l2.mp4" http://127.0.0.1:8000/runs
curl http://127.0.0.1:8000/runs/<run_id>/report
```

To evaluate the run, also send the video's ground truth: `-F "gt_coco=@MOT20-01_coco.json" -F "gt_mot=@MOT20-01_gt.txt"`.
Both files are required together. The COCO file must have one image per video frame (ids 1…N, the video's width and
height) and one `person` category; the MOT file is a MOTChallenge `gt.txt` with the same person boxes. Files that do not
match the video are rejected with 422 before the run is queued. The report then also has detection (mAP, precision,
recall) and tracking (HOTA, IDF1, ID switches) metrics.

- `replay_mode=offline` (default) processes every frame; `realtime` plays the video at its own fps and drops the frames the pipeline is too slow for (`frames_dropped` in the report).
- Mode D runs the Policy Engine: it picks enhance / skip / bypass / downscale for every frame from the degradation, the profile and the time and resource budget. See [docs/POLICY.md](docs/POLICY.md).
- Each run writes `detections.txt`, `tracks.txt` and, in mode D, `policy_log.csv` to `runs/<run_id>/`.
- Results for the Web UI: `GET /runs/{id}/files` lists the run's artifacts and `GET /runs/{id}/files/{name}` serves one;
  `GET /runs/{id}/video` returns the source video with tracks and detections drawn on it (WebM, rendered on the first
  request and kept as `runs/<run_id>/annotated.webm`); `GET /modes` and `GET /profiles` list the presets.

## Web UI

`streamlit run src/resilens/ui/app.py` (the API must be running; set `RESILENS_API_URL` if it is not on
`http://127.0.0.1:8000`). The UI works only through the REST API:

- **New run**: upload an MP4/AVI video, optionally its COCO (`.json`) and MOT (`.txt`) ground-truth annotations to get
  quality metrics, pick a mode (with its stages) and a resource profile (with its device and limits; a GPU profile
  shows the API host's GPU, or why it cannot run there), choose offline or realtime replay, start the run and follow
  its status.
- **Runs**: every run with status, video, duration and error; for a completed run its metrics (performance, pipeline
  output, detection and tracking quality when the run had annotations, Policy Engine shares), the annotated video,
  detections and tracks per frame, the track list, the policy decisions per frame (mode D), the config snapshot and
  downloads of all run files.
- **Comparison**: metrics of several runs side by side with the change against a reference run (mode A by default), a
  bar chart per metric and, for annotated runs, quality vs FPS.
- Resource profiles (`minimal`, `standard`, `performance`) set CPU threads, the frame budget and memory limits; [docs/PROFILES.md](docs/PROFILES.md) explains the numbers and how to measure them again with `benchmarks/profiles/measure.py`.

## Tracker

An optical-flow tracker follows targets you select by hand on the first frame:

```bash
# Demo: draw boxes around targets (Enter = add, Esc = start), watch the tracking, save the annotated video
python -m resilens.tracking.flow_tracker path/to/clean.mp4 --save-video

# Benchmark runs: reuse the same targets on every clip, without windows
python -m resilens.tracking.flow_tracker path/to/fog_l2.mp4 --targets runs/tracking/<folder>_clean/targets.json --no-display
```

Outputs go to `runs/tracking/<folder>_<video name>/`. See [docs/TRACKER.md](docs/TRACKER.md) for controls, the benchmark workflow, output formats and how to evaluate the results.
