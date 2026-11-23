# Mode C: MWFormer offline baseline

Mode C compares Resilens with MWFormer, a heavy all-in-one weather-removal transformer. MWFormer is **not** applied in
realtime: every frame of a clip is restored in advance and cached, and mode C only reads the cached frames, then runs
the same YOLOX-Nano detector and optical-flow tracker as the other modes. It is a quality reference, not a candidate
for the realtime pipeline (ADR-06, NFR-05), so its latency numbers are not comparable with modes A, B and D: they
contain only reading the cache. Mark it separately on the Pareto plot.

## Weights

MWFormer needs two files; neither is committed (`weights/*` is git-ignored). The PM's Drive folder
`Checkpoints/MWFormer/` has them in `backbone/` and `stylefilter/`, one file per training variant:

| Variant | Trained on |
| --- | --- |
| `original` | the original MWFormer data |
| `ar` | artificial fog and rain data |
| `ws` | WeatherStream |
| `ar_ws` | both |

Mode C uses `original` unless stated otherwise. Download the files without an extension and put them here:

```bash
mkdir -p weights/mwformer
mv original_backbone     weights/mwformer/backbone_original.pth
mv original_style_filter weights/mwformer/stylefilter_original.pth
sha256sum weights/mwformer/*.pth   # record the hashes next to the results
```

## Build the cache and run

```bash
python -m resilens.cli mwformer-cache data/scenarios/control_videos/MOT20-01/fog_l2.mp4 \
    --style-weights weights/mwformer/stylefilter_original.pth \
    --backbone-weights weights/mwformer/backbone_original.pth \
    --device cuda            # cpu works too: about 2.5 s per 1080p frame
python -m resilens.cli run data/scenarios/control_videos/MOT20-01/fog_l2.mp4 --mode C --profile minimal
```

Frames are cached in `data/mwformer_cache/` (git-ignored) as lossless PNG, keyed by the SHA-256 of the decoded frame,
so one cache serves several videos and a rerun skips frames already restored. `index.json` there lists the videos and
the SHA-256 of both weight files. A run on a video without a cache stops with an error that names the missing cache;
it never silently falls back to unrestored frames. The cache of one 300-frame 1080p clip is about 1 GB.

The Run Store fingerprint of a run does not record the weights, so keep the cache's `index.json` with the results.

## Preprocessing

As in the authors' `test.py`: pixels scaled to [-1, 1], the longer side limited to 1024 and both sides rounded up to a
multiple of 16 (Lanczos), the output clamped to [0, 1] and resized back to the original size. The style filter output
is the feature vector the backbone is conditioned on.

## Code origin

`src/resilens/enhancement/mwformer_model/` is the inference part of
[Reflectia/Weather-removal-mwformer](https://github.com/Reflectia/Weather-removal-mwformer) (`EncDec.py`,
`style_filter64.py`, `base_networks.py`). The repository has no licence file, so ask the authors before publishing the
code outside the project. Changes: absolute imports, `timm.layers` instead of the deprecated `timm.models.layers`, no
`pdb`. The folder is excluded from ruff.

## Checks

- The restored frame equals the authors' pipeline (PIL Lanczos, `test.py` sizing) bit for bit on a 1080p frame
  (`tests/unit/test_mwformer.py::test_mwformer_restores_a_frame_at_its_size` needs the weights; the sizing rules are
  tested without them). Pillow's Lanczos filters before shrinking and OpenCV's does not; with OpenCV the output
  differed by up to 47 gray levels, so the resize uses Pillow.
- The network fails on inputs smaller than 48 pixels per side, so the runner works at 64 pixels at least.
- The output is deterministic and has the input's size for odd and portrait sizes.

## First quality check of the `original` variant

PSNR against the clean clip, `fog_l2`, 1080p, two frames (50 and 150) per clip, CPU:

| Clip | Fogged | TANet | MWFormer `original` |
| --- | --- | --- | --- |
| MOT20-01 | 8.9 / 8.9 dB | 19.1 / 16.0 dB | 19.0 / 17.8 dB |
| MOT20-02 | 9.0 / 9.1 dB | 16.3 / 20.4 dB | 10.4 / 13.1 dB |
| MOT20-03 | 7.8 / 7.5 dB | 10.6 / 10.3 dB | 8.3 / 8.2 dB |
| MOT20-05 | 6.1 / 6.3 dB | 8.4 / 10.0 dB | 6.0 / 6.7 dB |

On MOT20-01 MWFormer is on par with TANet; on the other clips it removes much less fog. Eight frames are a smoke
test, not a result: compare the variants (`ar`, `ar_ws`) on whole clips before drawing conclusions for LR5.
