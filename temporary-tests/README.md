# Degradation indicator benchmark

Compares classical (model-free) CV indicators for the Degradation Analyzer (FR-03) on the control videos
listed in `manifest.json`: 4 MOT20 sequences x {clean, fog, rain, low light, motion blur} x levels 1-2.

## Files

| File | Purpose |
| --- | --- |
| `extract_frames.py` | Samples 8 pairs of consecutive frames per video, resizes to 480x270, caches to `_out/frames.npz` |
| `candidates.py` | All candidate indicators (`image -> float`, higher = more degradation) and the `CANDIDATES` registry |
| `compare.py` | ROC-AUC, per-sequence monotonicity and time per frame for every candidate |
| `eval_thresholds.py` | Leave-one-sequence-out accuracy of a global two-threshold decision (absent / level 1 / level 2) |

## Data (not in git)

`control_videos/`, `ground_truth/` and `manifest.json` are too large or are dataset files, so they are git-ignored
and must be downloaded separately and placed in this folder before running anything:

```
temporary-tests/
  manifest.json
  control_videos/<sequence>/{clean,fog_l1,fog_l2,rain_l1,rain_l2,low_light_l1,low_light_l2,motion_blur_l1,motion_blur_l2}.mp4
  ground_truth/<sequence>_gt.txt, <sequence>_coco.json
```

Sequences: MOT20-01, MOT20-02, MOT20-03, MOT20-05.

## Run

Run from this folder.

```
pip install -r requirements.txt
python extract_frames.py
python compare.py
python eval_thresholds.py
```

## Results

Held-out accuracy of the 3-class decision (not T / level 1 / level 2), thresholds fitted on 3 sequences and tested on the 4th.

| Type | Best indicator | Level accuracy | ms/frame |
| --- | --- | --- | --- |
| fog | `low_saturation`, `dark_channel_p5` | 1.00 | 0.3 / 2.7 |
| rain | `temporal_streaks` | 0.997 | 1.4 |
| rain (single frame) | `orientation_dominance` | 0.98 | 0.7 |
| low light | `dark_mean` | 0.93 | 0.2 |
| motion blur | `high_freq_ratio` | 0.99 | 11 |
| motion blur (fast) | `laplacian_var_norm` | 0.86 | 2.7 |

Full tables: output of `compare.py` and `eval_thresholds.py`.

## Methods

### Fog
- `dark_channel_mean` - mean of the 15x15 dark channel; fog lifts it.
- `dark_channel_p5` - 5th percentile of the dark channel; fog lifts even the darkest patches.
- `low_contrast` - negative std of luma.
- `low_saturation` - negative mean HSV saturation; fog washes out colour.

### Low light
- `dark_mean` - 255 minus mean luma.
- `dark_median` - 255 minus median luma.
- `dark_fraction` - share of pixels with luma below 50.
- `dark_p95` - 255 minus the 95th luma percentile.

### Motion blur
- `laplacian_var` - negative Laplacian variance.
- `laplacian_var_norm` - Laplacian variance divided by luma variance.
- `tenengrad` / `tenengrad_norm` - negative Sobel gradient energy, plain / normalised by luma variance.
- `high_freq_ratio` - negative share of FFT energy outside the low-frequency disc.
- `anisotropy` - |log(horizontal / vertical gradient energy)|.

### Rain
- `vertical_tophat` - density of thin bright details that survive a vertical line opening.
- `vertical_ratio` - share of thin bright pixels lying on vertical lines.
- `orientation_dominance` - horizontal / vertical gradient energy of the top-hat image.
- `vertical_line_response` - thin vertical line detector (1x15 mean minus 7x15 mean).
- `temporal_streaks` - vertical streaks in frame t+1 absent in frame t (takes a frame pair).

## How to apply

1. Resize the frame to 480x270 (BGR uint8) and call the indicator; `temporal_streaks` takes `(prev, next)` stacked as `(2, H, W, 3)`.
2. Compare the score with two thresholds fitted by `fit_thresholds` in `eval_thresholds.py`:
   below `low` = not present, `low..high` = level 1, `>= high` = level 2.
3. Run one indicator per type and pick the dominant type from the normalised scores.

## Tests to run next

- Run the winners on real footage (rain, fog, night, handheld blur) and refit thresholds.
- Cross-trigger matrix: every indicator against every other degradation, per level.
- Combined degradations (e.g. fog + low light, rain + blur): check dominant-type selection.
- Scene-type robustness: indoor / outdoor, crowded / empty, bright / dark clean scenes (false positives on clean frames).
- Temporal stability: per-frame score variance over a clip, effect of EMA / median smoothing and frame skipping.
- Resolution sweep (240p-720p) for accuracy vs time, especially `high_freq_ratio`.
- Aggregation: normalisation to [0, 1], confidence formula, UNKNOWN threshold sweep with a confusion matrix.
- Compression sensitivity: re-encode the clips at lower bitrates and check the indicators still separate.
- Latency benchmark of the full indicator set on the target CPU profile.
