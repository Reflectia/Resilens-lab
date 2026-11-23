# Resilens 2.0.0 — Release Notes

**Tag:** `v2.0.0` · **Commit:** `0acec56` · **Release date:** `23.11.2026` · **Repository:** https://github.com/Reflectia/Resilens-lab

## Overview

Resilens 2.0 is a platform for adaptive object detection and tracking in video shot under adverse conditions. Version 1.0 (the Frogs Vision project) was an image-enhancement subsystem in which the TANet or MWFormer model was chosen by hand. In version 2.0 the system analyzes the degradation itself, picks an action using the Policy Engine rules, and measures the effect on detection and tracking with metrics.

## What's new

- Pipeline: Analyzer → Policy Engine → Enhancer (TANet) → YOLOX-Nano → FlowTracker → Run Store.
- Modes A–D: A — no enhancement (baseline), B — TANet, C — MWFormer as an offline baseline (frames are restored in advance with the `mwformer-cache` command, see `docs/MWFORMER.md`), D — adaptive mode with the Policy Engine.
- For every frame the Policy Engine chooses `enhance`, `skip`, `bypass` or `downscale`, depending on the degradation, the profile and the time and resource budget; decisions are written to `policy_log.csv`.
- Video replay in offline mode (every frame) and realtime mode (frames the pipeline cannot keep up with are dropped; `frames_dropped` in the report).
- Resource profiles `minimal`, `standard` and `performance`. The `performance` profile runs on an NVIDIA GPU with the CUDA build of PyTorch (CUDA 13.0, driver 580 or newer); without a GPU it is unavailable.
- REST API and Streamlit Web UI: upload a video and COCO and MOT annotations (validated; mismatching files are rejected with 422), browse runs, view the annotated video, compare runs.
- Quality metrics: mAP, precision, recall (detection); HOTA, IDF1, ID switches (tracking). The `benchmark` and `compare` commands give a comparison with mode A, latency p50/p95 and peak RAM/CPU/GPU.
- Reproducibility: every run records the commit, branch, uncommitted-changes flag, environment and package versions in `environment.json` and `report.json`; dependencies are pinned in `requirements.lock` and `requirements-cu130.lock`.
- Stack: Python 3.13, PyTorch 2.x, Docker Compose, CI.

## Fixed compared with version 1.0

- **P-01.** Manual choice of TANet or MWFormer replaced by automatic degradation analysis and the Policy Engine.
- **P-02.** Low light and motion blur are now part of the control videos and the benchmark. No enhancement is applied to them: the Policy chooses `bypass`.
- **TD-01.** There is a standardized integration contract: a REST API with OpenAPI.
- **TD-02.** Added a dedicated API, Docker Compose and CI. There is no automatic deployment (CD).
- **TD-03.** Added automated tests (unit, integration, regression, smoke, e2e Web UI) and code review through pull requests.
- **TD-04.** Added metrics of the effect on detection and tracking (mAP, HOTA, IDF1) and a comparison of modes.

## Known issues

- **P-03.** Removing real snow remains a weak spot; out of scope for v2.0.
- **DEF-05.** FlowTracker cannot add new targets or recover lost ones, which lowers tracking scores. The planned ByteTrack is not connected.
- **DEF-02.** The MOT20-03 videos and their annotations differ in width. It is excluded from the release benchmark.
- **CPU and GPU.** Mode B results differ on CPU and GPU: mAP 0.123 vs 0.071 on fog\_l2 MOT20-01. Outcome of the check: \[enter the PSNR "CPU vs GPU" result\].
- **CPU performance.** TANet takes about 0.7–1 s per frame, so on the `minimal` and `standard` profiles the Policy mostly chooses `skip`.
- **Tests.** Integration tests with real weights and e2e Web UI tests are run locally; CI does not run them.

## Limitations

- Not included in version 2.0: a physical camera and robot, production deployment, parallel processing of several streams, 24-hour stability and recovery after failures, mode E (YOLOX fine-tuning).
- Mode C needs a MWFormer cache built in advance and does not run in real time.
- In realtime mode, dropped frames count in quality metrics as frames without detections and tracks.
- Model weights and videos are not stored in git: TANet is downloaded manually and saved as `weights/tanet.pth`, YOLOX-Nano as `weights/yolox_nano_coco.pth`.

## Installation and migration from v1.0

Detailed steps in `README.md` In short: `pip install -c requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"` (on a GPU machine use `requirements-cu130.lock` and the `cu130` index) or `GIT_COMMIT=$(git rev-parse HEAD) docker compose up --build`; put weights into `weights/` and videos and scenarios into `data/`; the profile is chosen when a run is started.

When moving from v1.0 the stack changes (Python 3.13 and PyTorch 2.x instead of Python 3.10 and PyTorch 1.13) and so does the interface (REST API and Streamlit instead of Gradio).

## Documentation

- Test Summary Report — `docs/TEST_SUMMARY_REPORT.md`
- Resource Profiles — `docs/PROFILES.md`
