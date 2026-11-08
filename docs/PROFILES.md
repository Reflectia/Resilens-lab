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

- **`frame_budget_ms`** is about 1.2 × the p95 latency of mode A (no enhancement) on that profile. The policy may spend time on TANet only while the moving latency stays near the speed of the pipeline without enhancement.
- **`memory_limit_mb`** is the Docker memory limit. Measured peaks are ~0.5–0.7 GB, so this is a safety net that does not fire in normal runs.
- **`cpu_limit_percent` stays `null`.** A run always uses all its `cpu_threads` (peak ~240 % with 2 threads, ~500 % with 4), so a CPU limit would lower the action on every frame. CPU is limited by `cpu_threads` and Docker instead.
- **`skip_every_n: 7`** in `configs/policy/rules.yaml` follows from N ≥ TANet time / (budget − time of the other stages) = 650 / (200 − 100) on standard.
- **`latency_metric: mean`.** With `skip`, the p95 latency is always a TANet frame, so p95 would report "over budget" even when the average load fits. The mean shows the real load.

## Measurements

MOT20-01, CPU only (no GPU on the measuring machine), one Windows laptop on mains power, local runs with `cpu_threads`, no Docker. 60 frames, of which the first 10 are warm-up. Median of 2 repeats after 1 warm-up run. Latency is per frame, end to end; TANet / YOLOX / tracker are the median stage times.

| Clip | Profile | Mode | p50 / p95 | Peak RAM | Peak CPU | TANet | YOLOX | Tracker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fog_l2 | minimal | A | 228 / 250 ms | 529 MB | 223 % | — | 90 ms | 132 ms |
| fog_l2 | minimal | B | 2710 / 3395 ms | 717 MB | 239 % | 2463 ms | 73 ms | 178 ms |
| fog_l2 | standard | A | 91 / 97 ms | 528 MB | 485 % | — | 25 ms | 61 ms |
| fog_l2 | standard | B | 770 / 947 ms | 720 MB | 451 % | 635 ms | 28 ms | 102 ms |
| clean | minimal | A | 293 / 335 ms | 507 MB | 233 % | — | 80 ms | 210 ms |
| clean | minimal | B | 3450 / 3661 ms | 700 MB | 235 % | 3125 ms | 84 ms | 242 ms |
| clean | standard | A | 170 / 192 ms | 520 MB | 517 % | — | 47 ms | 120 ms |
| clean | standard | B | 1804 / 1908 ms | 721 MB | 451 % | 1617 ms | 56 ms | 131 ms |

- TANet is the main cost: from ~0.6 s to ~3 s per frame. Mode B does not fit any of these budgets on CPU.
- In mode A the flow tracker takes longer than YOLOX, so `downscale`, which speeds up the tracker, is a useful last step.
- **The measuring machine is not stable.** The two sessions of the same day differ by up to 2.5× (TANet on standard: 635 ms in the first, 1617 ms in the second), with no other load running. This is most likely thermal throttling. Within a session, repeats differ by 0–14 %. Treat the limits as provisional until they are measured on a stable machine, ideally in Docker with the profile limits.

### Mode D with these limits

100 frames, one run, during the slower second session:

| Clip | Profile | Actions | TANet ran on | Frames within budget |
| --- | --- | --- | --- | --- |
| fog_l2 | standard | enhance 11 %, then skip 89 % (lowered by the budget) | 24 % of frames | 74 % |
| fog_l2 | minimal | skip lowered to bypass on 81 % of frames | 3 % of frames | 36 % |
| clean | standard | bypass 99 % | 0 % | 70 % |
| clean | minimal | bypass 84 %, lowered to downscale 16 % | 0 % | 34 % |

The budget correction works as intended: on standard, fog moves from `enhance` to `skip`, and on minimal TANet is practically switched off, as risk R-02 expects. On minimal even mode A was over its 300 ms budget in this slower session (p95 335 ms), so the policy also downscaled some clean frames.

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
- Whether the one-step budget correction is enough (see `docs/POLICY.md`) is still with TL/PM. On minimal it already turns TANet off; on standard the latency stays above the budget part of the time.
