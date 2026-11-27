# Resource profiles and their limits (T-20)

A resource profile (`configs/profiles/*.yaml`) describes the hardware a run may use and the limits the Policy Engine keeps to (NFR-01, NFR-02). This page explains where the numbers come from and how to measure them again.

## Profiles

| | minimal | standard | performance |
| --- | --- | --- | --- |
| Hardware | 2 CPUs, 4 GB RAM | 4 CPUs, 8 GB RAM | NVIDIA GPU with CUDA (measured in D-10, see below) |
| `cpu_threads` (PyTorch and OpenCV) | 2 | 4 | all cores |
| Docker limits | `docker-compose.minimal.yml` | `docker-compose.standard.yml` | — |
| `frame_budget_ms` | 300 | 200 | 220 |
| `memory_limit_mb` | 4096 | 8192 | `null` |
| `gpu_memory_limit_mb` | `null` | `null` | 4096 |
| `cpu_limit_percent` | `null` | `null` | `null` |
| `downscale_resolution` | 960×540 | 1280×720 | 1280×720 |

- **`frame_budget_ms`.** On standard, 200 ms is about 1.1 × the p95 latency of mode A (no enhancement): the policy may spend time on TANet only while the moving latency stays near the speed of the pipeline without enhancement. On minimal, a budget derived the same way (~200 ms) would switch TANet off completely, so it is 300 ms: enough for `skip` on two threads (other stages ~160 ms + TANet ~980 ms / 11 ≈ 250 ms on average).
- **`memory_limit_mb`** is the Docker memory limit. Measured peaks are ~0.5–0.75 GB, so this is a safety net that does not fire in normal runs.
- **`cpu_limit_percent` stays `null`.** A run always uses all its `cpu_threads` (peak ~250 % with 2 threads, ~450–530 % with 4), so a CPU limit would lower the action on every frame. CPU is limited by `cpu_threads` and Docker instead.
- **`skip_every_n: 11`** in `configs/policy/rules.yaml` follows from N ≥ TANet time / (budget − time of the other stages) = 661 / (200 − 137) ≈ 10.5 on standard (fog_l2, mode D). The other stages take longer in mode D than in mode A because the flow tracker follows more targets on enhanced frames.
- **`latency_metric: mean`.** With `skip`, the p95 latency is always a TANet frame, so p95 would report "over budget" even when the average load fits. The mean shows the real load.

## Measurements

MOT20-01, CPU only (no GPU on the measuring machine), one Windows laptop on mains power with nothing else running, local runs with `cpu_threads`, no Docker. 60 frames, of which the first 10 are warm-up. Median of 2 repeats after 1 warm-up run; repeats differ by 0–4 %. Latency is per frame, end to end; TANet / YOLOX / tracker are the median stage times.

| Clip | Profile | Mode | p50 / p95 | Peak RAM | Peak CPU | TANet | YOLOX | Tracker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | minimal | A | 163 / 168 ms | 506 MB | 251 % | — | 28 ms | 130 ms |
| clean | minimal | B | 1150 / 1269 ms | 720 MB | 225 % | 975 ms | 29 ms | 139 ms |
| clean | standard | A | 157 / 178 ms | 520 MB | 521 % | — | 32 ms | 119 ms |
| clean | standard | B | 911 / 1029 ms | 712 MB | 446 % | 754 ms | 29 ms | 122 ms |
| fog_l2 | minimal | A | 107 / 110 ms | 532 MB | 248 % | — | 28 ms | 75 ms |
| fog_l2 | minimal | B | 1139 / 1214 ms | 722 MB | 224 % | 982 ms | 29 ms | 122 ms |
| fog_l2 | standard | A | 94 / 103 ms | 531 MB | 533 % | — | 28 ms | 63 ms |
| fog_l2 | standard | B | 793 / 883 ms | 726 MB | 451 % | 661 ms | 28 ms | 102 ms |

- TANet is the main cost: ~0.65–1 s per frame. Mode B does not fit any of these budgets on CPU.
- Four threads instead of two make mode A only slightly faster (157 vs 163 ms on clean): most of the time goes to the flow tracker, which hardly uses more threads. The difference between the profiles is mostly in TANet.
- In mode A the flow tracker takes longer than YOLOX, so `downscale`, which speeds up the tracker, is a useful last step.
- **The measuring machine is not always stable.** An earlier session on the same day was up to 2× slower (mode A on clean, minimal: p95 335 ms instead of 168 ms; TANet on standard up to 1.6 s), most likely because of thermal throttling. The table above was measured in a quiet session and the noisy rows were measured again. Check the `+/-` spread in the console before trusting a new measurement.

### Mode D with these limits

100 frames, one run:

| Clip | Profile | Actions | TANet ran on | Frames within budget |
| --- | --- | --- | --- | --- |
| fog_l2 | standard | enhance 11 %, then skip 89 % (lowered by the budget) | 20 % of frames | 89 % |
| fog_l2 | minimal | skip 93 %, lowered to bypass on 7 % of frames | 9 % of frames | 91 % |
| clean | standard | bypass 100 % | 0 % | 100 % |
| clean | minimal | bypass 100 % | 0 % | 100 % |

The budget correction works as intended: on standard, fog moves from `enhance` to `skip` once TANet on every frame pushes the moving latency over 200 ms, and on minimal the rule table already gives `skip`. Clean video costs as much as mode A (minimal: p50 / p95 170 / 177 ms).

## GPU check of the performance profile (D-10)

The performance profile runs TANet and YOLOX on CUDA (`device: cuda`); the analyzer and the flow tracker stay on the
CPU. Its limits come from this check (see Results). Integration: Бучко Устим; measurement and check: Козло Богдан.

What the code does without a GPU: `GET /profiles` reports the profile as unavailable with the reason, the Web UI marks
it "(unavailable)" and does not start it, `POST /runs` answers 422, a CLI run fails before loading models, and
`measure.py` stops before the first run.

### 1. Set up the GPU machine

Install the project from the CUDA lock: it pins the same versions as `requirements.lock`, with the CUDA 13.0
build of PyTorch and the CUDA runtime it needs. It replaces a CPU build of PyTorch already in the environment. The
NVIDIA driver must be 580 or newer (`nvidia-smi` shows the version).

```bash
pip install -c requirements-cu130.lock --extra-index-url https://download.pytorch.org/whl/cu130 -e ".[dev]"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and the GPU name. `nvidia-smi` should show no other process using the GPU while you
measure (NFR-05: one experiment at a time). Put the weights into `weights/` and the MOT20 videos and
`data/manifest.json` into `data/`, as for the CPU measurements.

### 2. Measure

The same clips, frames and repeats as the CPU measurements above, with `standard` on the same machine as the
reference, so the GPU effect is not mixed with a different CPU:

```bash
python benchmarks/profiles/measure.py --manifest data/manifest.json   --scenario control_videos/MOT20-01/clean.mp4 --scenario control_videos/MOT20-01/fog_l2.mp4   --modes A B D --profiles standard performance --frames 60 --repeats 2 --warmup-runs 1
```

The script prints the GPU it runs on and writes, in `benchmarks/profiles/_out/`:
`profiles.csv` (one row per run), `profiles.json` (git commit and uncommitted changes, machine, GPU, driver, CUDA,
cuDNN, every package version and its difference from the lock; README, "Reproducibility") and
`profiles.md` (median per clip, profile and mode, in the layout of the table below). Stage times are measured with
CUDA synchronisation, so GPU work is counted in the stage that started it. Repeats that differ by more than ~10 %
(`+/-` in the console) should be measured again.

### 3. Set the limits in `configs/profiles/performance.yaml`

Proposed rules, the same as on the CPU profiles where they apply (agree the final values with TL):

- `frame_budget_ms`: on performance the rule table asks for `enhance` on fog and rain, so the budget should let TANet
  run on every frame: about 1.1 × the p95 latency of mode B. If mode B stays several times slower than mode A even on
  the GPU, use 1.1 × p95 of mode A, as on standard, and note it here.
- `gpu_memory_limit_mb`: the GPU's memory (`gpu_memory_gb` in `profiles.json`, in MB), or less if the GPU is shared.
  The measured peak GPU memory must be well below it; the policy lowers the action at 90 % of the limit.
- `memory_limit_mb`: there is no Docker limit for this profile; leave `null` unless the machine has a RAM budget. The
  peak RAM goes into the table either way.
- `cpu_threads`: `null` (all cores), unless the CPU part (the flow tracker) is to be limited.

Then check mode D with the new limits, as for the CPU profiles:

```bash
python benchmarks/profiles/measure.py --manifest data/manifest.json   --scenario control_videos/MOT20-01/fog_l2.mp4 --scenario control_videos/MOT20-01/clean.mp4   --modes D --profiles performance --frames 100 --repeats 1 --warmup-runs 1
```

### 4. Check the profile in the Web UI

Start the API and the UI on the GPU machine (README). On **New run** the profile list shows `performance · GPU` and
the profile table shows `Device: GPU: <name>`. Start mode B on a short clip with this profile; the completed run
shows *Peak GPU memory, MB* under Performance, and in mode D the policy log tooltip shows `gpu_memory_mb`.

### Results

Machine: NVIDIA GeForce GTX 1650 (4 GiB), CUDA 12.8, PyTorch 2.9.1+cu128, AMD Ryzen 5 5500U (6 cores, 12 threads), 16 GiB RAM, Windows 11, commit `5c0f8b9`.
These results predate the lock files: a new measurement from `requirements-cu130.lock` uses PyTorch 2.14.1+cu130.


| Clip   | Profile     | Mode | p50 / p95      | Peak RAM | Peak CPU | Peak GPU memory | TANet   | YOLOX | Tracker |
| ------ | ----------- | ---- | -------------- | -------- | -------- | --------------- | ------- | ----- | ------- |
| clean  | standard    | A    | 240 / 267 ms   | 767 MB   | 543 %    | —               | —       | 48 ms | 187 ms  |
| clean  | standard    | B    | 944 / 985 ms   | 1027 MB  | 463 %    | —               | 797 ms  | 39 ms | 105 ms  |
| clean  | performance | A    | 177 / 180 ms   | 1294 MB  | 929 %    | 30 MB           | —       | 35 ms | 138 ms  |
| clean  | performance | B    | 234 / 246 ms   | 1456 MB  | 749 %    | 182 MB          | 102 ms  | 22 ms | 106 ms  |
| fog_l2 | standard    | A    | 121 / 126 ms   | 1469 MB  | 526 %    | —               | —       | 40 ms | 77 ms   |
| fog_l2 | standard    | B    | 1346 / 1414 ms | 1505 MB  | 447 %    | —               | 1230 ms | 49 ms | 62 ms   |
| fog_l2 | performance | A    | 118 / 121 ms   | 1395 MB  | 918 %    | 182 MB          | —       | 35 ms | 79 ms   |
| fog_l2 | performance | B    | 189 / 197 ms   | 1464 MB  | 778 %    | 182 MB          | 102 ms  | 23 ms | 60 ms   |

Mode D with the chosen limits: not measured yet (the second command of step 3, on the GPU machine).

Chosen limits and why:

- `frame_budget_ms: 220`: 1.1 × p95 of mode B on fog_l2 (197 ms), so TANet can run on every foggy frame. Mode B is
  not several times slower than mode A on the GPU (197 vs 121 ms), so the mode B rule applies.
- `gpu_memory_limit_mb: 4096`: the memory of the GTX 1650; the measured peak (182 MB) is far below it.
- `memory_limit_mb` and `cpu_threads`: `null` (no Docker limit for this profile, all cores).

These limits are a starting point: mode D has not been run with them, and they fit this one GPU.

**Peak GPU memory is an upper bound.** It is read with `torch.cuda.max_memory_reserved()`, which includes memory the
allocator cached in earlier runs of the same process, so mode A shows 182 MB it did not use itself.

Web UI check: _to be confirmed by Козло Богдан (profile shown as `performance · GPU` with the device name, run completed, peak GPU memory shown)._

## Measuring again

```bash
python benchmarks/profiles/measure.py --manifest path/to/manifest.json \
  --scenario control_videos/MOT20-01/clean.mp4 --scenario control_videos/MOT20-01/fog_l2.mp4 \
  --modes A B D --profiles minimal standard --frames 60 --repeats 2 --warmup-runs 1
```

About 25 minutes on the measuring laptop. Each run is a row of `benchmarks/profiles/_out/profiles.csv`, with latency (end to end and per stage), peak RAM / CPU / GPU and policy action shares. `profiles.json` records the machine, the libraries and the git commit. The console prints the median over repeats and how much the repeats differ (`+/-`).

To emulate the profile hardware, run inside Docker with the profile's limits (the override mounts `benchmarks/`; the videos go to `data/`):

```bash
docker compose -f docker-compose.yml -f docker-compose.minimal.yml run --rm api \
  python benchmarks/profiles/measure.py --manifest data/manifest.json --profiles minimal
```

Runs through `POST /runs` and `benchmark` report the same figures in their metrics (`latency_p50_ms`, `latency_p95_ms`, `latency_in_budget_ratio`, `stage_<name>_p95_ms`, `peak_ram_mb`, `peak_cpu_percent`, `mean_cpu_percent`, `peak_gpu_memory_mb`), and `compare` puts them next to the quality metrics.

## Open points

- The **performance** limits fit the GTX 1650 above; mode D with these limits is not measured yet, and on another GPU the profile should be measured again (step 2).
- **Measure on a stable machine** (or in Docker) before using the limits for the final comparison (T-18).
- Whether the one-step budget correction is enough (see `docs/POLICY.md`) is still with TL/PM. With these limits `skip` fits the budget on both profiles, so in these runs one step was enough.
