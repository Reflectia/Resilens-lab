# Policy Engine (T-10)

Mode D (`use_analyzer: true`, `use_policy: true`) chooses an action for every frame:

```
Replay -> Analyzer -> Policy -> Enhancer (TANet) -> Detector -> Tracker
```

1. The Analyzer estimates the degradation type and level (1 = low, 2 = high), or `unknown` / `none`.
2. The Policy looks up the base action in the rule table for that type, level and the run's resource profile.
3. **Budget correction:** the action is lowered one step along `enhance -> skip -> bypass -> downscale` when the moving latency of the previous frames is over the profile's `frame_budget_ms`, or RAM / CPU / GPU memory reach `near_limit_ratio` of the profile's limit. Several problems at once still lower it by one step only.
4. The Enhancer carries out the decision.

| Action | What happens |
| --- | --- |
| `enhance` | TANet on the frame |
| `skip` | TANet on every N-th frame with this action (`skip_every_n`), the others go on unchanged |
| `bypass` | the frame goes on unchanged |
| `downscale` | no TANet; the frame is fitted into the profile's `downscale_resolution` before the detector |

After `downscale` the detector's boxes are mapped back to original pixels, and the flow tracker follows targets across frames of different sizes, so `detections.txt` and `tracks.txt` always use original video coordinates.

## Configuration

- **Rule table:** [`configs/policy/rules.yaml`](../configs/policy/rules.yaml), version 0.1 from T-07. It also sets `skip_every_n`, the downgrade order and the budget parameters (latency window, `mean` or `p95`, `near_limit_ratio`). The values marked provisional are to be confirmed by TL/TA.
- **Profiles:** [`configs/profiles/*.yaml`](../configs/profiles/). The Policy uses:
  - `frame_budget_ms`;
  - `memory_limit_mb`;
  - `cpu_limit_percent` (process CPU as `psutil` reports it, 100 = one fully used core);
  - `gpu_memory_limit_mb`;
  - `downscale_resolution`.

  A limit set to `null` is not checked. All limits are `null` until the baseline measurements of mode A, so the budget correction does not fire yet.
- The first `warmup_frames` (10) frames do not count towards the moving latency.

## Running

```powershell
curl.exe -F mode=D -F profile=standard -F replay_mode=realtime -F "video=@path/to/fog_l2.mp4" http://127.0.0.1:8000/runs
```

`replay_mode=realtime` plays the video at its own fps and drops frames the pipeline is too slow for (`frames_dropped` in the report). `offline` (default) processes every frame.

## Outputs

`runs/<run_id>/policy_log.csv` has one row per processed frame:

| Column | Meaning |
| --- | --- |
| `frame` | 1-based, as in the MOT files |
| `degradation`, `level`, `confidence` | Analyzer estimate |
| `profile` | resource profile of the run |
| `base_action`, `action` | from the rule table / after the budget correction |
| `downgraded` | 1 if the budget correction applied |
| `enhanced` | 1 if TANet ran on the frame |
| `latency_ms`, `ram_mb`, `cpu_percent`, `gpu_memory_mb` | what the policy saw before the frame; empty = not measured |
| `reason` | readable explanation, e.g. `fog level 2 on standard: enhance; latency 342 ms > budget 150 ms -> skip` |

The report adds `policy_<action>_share` for each action, `policy_downgraded_share` and `policy_enhanced_share`.
