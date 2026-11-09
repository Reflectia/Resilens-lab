# Resource profiles and their limits (T-20)

A resource profile (`configs/profiles/*.yaml`) describes the hardware a run may use and the limits the Policy Engine keeps to (NFR-01, NFR-02). This page explains where the numbers come from and how to measure them again.

## Profiles

| | minimal | standard | performance |
| --- | --- | --- | --- |
| Hardware | 2 CPUs, 4 GB RAM | 4 CPUs, 8 GB RAM | GPU (not measured yet) |
| `cpu_threads` (PyTorch and OpenCV) | 2 | 4 | all cores |
| Docker limits | `docker-compose.minimal.yml` | `docker-compose.standard.yml` | — |
| `frame_budget_ms` | 300 | 200 | `null` |
| `memory_limit_mb` | 4096 | 8192 | `null` |
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

- The **performance** profile needs a machine with a GPU; its limits stay `null` until then.
- **Measure on a stable machine** (or in Docker) before using the limits for the final comparison (T-18).
- Whether the one-step budget correction is enough (see `docs/POLICY.md`) is still with TL/PM. With these limits `skip` fits the budget on both profiles, so in these runs one step was enough.
