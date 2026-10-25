# Degradation indicator benchmark and Analyzer calibration

Compares classical (model-free) CV indicators for the Degradation Analyzer (FR-03, T-09) on the control videos
listed in `manifest.json`: 4 MOT20 sequences x {clean, fog, rain, low light, motion blur} x levels 1-2.
The winning indicators live in `src/resilens/analysis/heuristics.py`; `calibrate.py` fits their thresholds
and writes `configs/analyzer/thresholds.yaml`, which the Analyzer loads.

## Files

| File | Purpose |
| --- | --- |
| `extract_frames.py` | Samples 8 pairs of consecutive frames per video, resizes to 480x270, caches to `_out/frames.npz` |
| `candidates.py` | All candidate indicators (`image -> float`, higher = more degradation) and the `CANDIDATES` registry |
| `compare.py` | ROC-AUC, per-sequence monotonicity and time per frame for every candidate |
| `eval_thresholds.py` | Leave-one-sequence-out balanced accuracy, detection and false-alarm rates of each candidate; confusion matrix of the best one per type |
| `calibrate.py` | Leave-one-sequence-out evaluation of the full Analyzer decision (type, level, `unknown`); `--write` saves the final thresholds |

## Data (not in git)

`control_videos/`, `ground_truth/` and `manifest.json` are dataset files, so they are git-ignored. Download them
from the control data folder linked on the project's Notion page and place them in this folder:

```
benchmarks/degradation/
  manifest.json
  control_videos/<sequence>/{clean,fog_l1,fog_l2,rain_l1,rain_l2,low_light_l1,low_light_l2,motion_blur_l1,motion_blur_l2}.mp4
  ground_truth/<sequence>_gt.txt, <sequence>_coco.json
```

Sequences: MOT20-01, MOT20-02, MOT20-03, MOT20-05.

## Run

Install the project first (`pip install -e ".[dev]"` from the repo root), then run from this folder:

```
python extract_frames.py
python compare.py
python eval_thresholds.py
python calibrate.py --write      # refit configs/analyzer/thresholds.yaml
```

Re-run `calibrate.py --write` whenever an indicator in `heuristics.py`, the control videos or the scenario
generator change: the thresholds are only valid for the exact indicator code and data they were fitted on.

## How results are measured

For each type T a held-out sequence has 72 samples: 56 without T (clean + the 3 other types x 2 levels),
8 at level 1 and 8 at level 2. Plain accuracy is dominated by the 56: answering "absent" every time scores 0.78.
Thresholds are therefore fitted and reported with **balanced accuracy** (mean recall over classes; the same
always-absent answer scores 0.33). Plain accuracy is printed next to it only for comparison.

Alongside it:
- `detect` - share of T frames predicted as T: misses mean the Policy bypasses enhancement when it should not.
- `false_clean` - share of clean frames predicted as T: false alarms run TANet for nothing and cost latency.
- `worst_seq` - balanced accuracy on the hardest held-out sequence; shows scene dependence a mean hides.
- Confusion matrix with "absent" split by the actual clip condition, so cross-triggers between types are visible.

With 8 samples per class per fold, one wrong frame moves a recall by 12.5%; read differences of a few percent
as noise. Counts are in the confusion matrices.

## Analyzer decision

1. Resize the frame to 480x270 and compute one score per type: `fog_score` (dark channel p5), `low_light_score`
   (255 - mean luma), `motion_blur_score` (high-frequency FFT energy share), `rain_score` (new vertical streaks
   vs the previous frame; skipped on the first frame and after a gap).
2. Normalise each score by its thresholds: 0 = level 1 threshold, 1 = level 2 threshold. This puts indicators
   with different units on one scale.
3. The highest normalised score above 0 is the dominant type; its level is 2 if it is at least 1, else 1.
   No score above 0 means `none`.
4. Confidence grows with the distance from the nearest competing decision (the level 1 threshold, or the
   runner-up type), reaching 1 at 0.5 normalised units. Below `confidence_threshold` the result is `unknown`,
   which the Policy maps to `bypass`.
5. Control videos with a scenario entry skip all of this and use the manifest's type and level.

`calibrate.py` prints coverage (share not `unknown`), balanced accuracy and false alarms for several confidence
thresholds, so the value in the config can be chosen from measurements.

## Results

Control videos: 4 MOT20 sequences x 9 clips x 300 frames, 8 sampled pairs per clip (288 samples). All numbers
are leave-one-sequence-out (thresholds fitted on 3 sequences, tested on the 4th), unless marked otherwise.

### Indicators (`eval_thresholds.py`)

| Type | Indicator | Balanced | Plain | Detect | False alarm on clean | Worst sequence | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fog | `dark_channel_p5` (Analyzer) | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.9 |
| fog | `low_saturation` | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.2 |
| low light | `dark_mean` (Analyzer) | 0.829 | 0.934 | 1.000 | 0.000 | 0.649 | 0.1 |
| motion blur | `high_freq_ratio` (Analyzer) | 0.979 | 0.993 | 1.000 | 0.000 | 0.917 | 4.2 |
| motion blur | `laplacian_var_norm` | 0.568 | 0.701 | 0.891 | 0.438 | 0.417 | 2.1 |
| rain | `temporal_streaks` (Analyzer) | 0.958 | 0.986 | 1.000 | 0.000 | 0.875 | 1.3 |
| rain | `vertical_line_response` | 0.978 | 0.969 | 0.984 | 0.250 | 0.952 | 1.3 |

Plain accuracy overstated weaker indicators: `dark_mean` drops from 0.93 to 0.83 and `laplacian_var_norm`
(the earlier "fast" blur option) from 0.86 to 0.57 with a 44% false-alarm rate. For rain the Analyzer keeps
`temporal_streaks` over `vertical_line_response`: slightly lower balanced accuracy, but no false alarms on clean
frames (a false alarm makes the Policy run TANet for nothing).

Low light detects presence perfectly but confuses level 1 and 2: absolute brightness depends on the scene.
MOT20-05's clean frames score 204 (255 - mean luma) against 142 for MOT20-01/02, so its level 1 lands where
other scenes' level 2 is.

### Analyzer decision (`calibrate.py`)

9 classes (none + 4 types x 2 levels); `unknown` counts as wrong in balanced accuracy.

| Confidence threshold | Coverage (not `unknown`) | Balanced | Accuracy when not `unknown` | False alarm on clean | Worst sequence |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 1.000 | 0.865 | 0.865 | 0.094 | 0.847 |
| **0.25 (config)** | 0.972 | 0.847 | 0.871 | 0.000 | 0.806 |
| 0.50 | 0.792 | 0.667 | 0.842 | 0.000 | 0.583 |
| 0.75 | 0.719 | 0.594 | 0.826 | 0.000 | 0.458 |

0.25 removes all false alarms on clean frames for 3% `unknown`; higher thresholds mostly turn correct clean
frames into `unknown` without making the remaining answers more accurate. Remaining errors at 0.25 are almost
all low-light level confusions (32 of 64 low-light samples) and rain level 1 / level 2 (4 of 64).

Thresholds are centred in the gap between the training scores around them. Without centring the search left
level 1 thresholds at the very edge of the level 1 scores, so those frames were always borderline.

### Full stage on every frame (in-sample)

`DegradationAnalyzer` over all 10,800 frames of the 36 clips with the committed thresholds (fitted on these
same sequences, so optimistic): 94.5% correct, decision changes between consecutive frames 3.4%, 8.2 ms/frame
p50 and 10.7 ms p95 including the resize from 1080p. The first frame of every rain clip is `none` because the
rain indicator needs a previous frame. MOT20-05 low light level 1 is reported as level 2 on all 300 frames.

## Known limitations

- All control clips use synthetic degradations, so indicators may partly learn the generator rather than real
  weather (risk R-04). In particular, the rain indicators look for exactly vertical streaks (real rain is often
  slanted) and `anisotropy` assumes one blur direction. Thresholds must be rechecked on real footage.
- Absolute-brightness indicators (`low_light_score`) read a clean night scene as low light.
- `rain_score` assumes a mostly static camera: camera motion or crowds create new vertical edges between frames.
- Snow is out of scope for v2.0; a snowy video may trigger the fog or rain indicators instead of `unknown`.

## Tests to run next

- Real footage per type (rain, fog, night, handheld blur), plus one snowy clip to check it ends up `unknown`.
- Combined degradations (e.g. fog + low light, rain + blur): check dominant-type selection.
- Scene-type robustness: indoor / outdoor, crowded / empty, bright / dark clean scenes.
- Temporal stability: per-frame decision flicker over a clip, effect of EMA / median smoothing.
- Resolution sweep (240p-720p) for accuracy vs time.
- Compression sensitivity: re-encode the clips at lower bitrates and check the indicators still separate.
- Latency of the full Analyzer on the target CPU profile.

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
- `high_freq_ratio` - negative share of FFT energy above a normalised frequency of 0.15.
- `anisotropy` - |log(horizontal / vertical gradient energy)|.

### Rain
- `vertical_tophat` - density of thin bright details that survive a vertical line opening.
- `vertical_ratio` - share of thin bright pixels lying on vertical lines.
- `orientation_dominance` - horizontal / vertical gradient energy of the top-hat image.
- `vertical_line_response` - thin vertical line detector (1x15 mean minus 7x15 mean).
- `temporal_streaks` - vertical streaks in frame t+1 absent in frame t (takes a frame pair).
