# Resilens 2.0 Test Summary Report

- **Status:** Complete
- **Release candidate:** `9dbf188` (`2.0.0`)

## Test environments

| Check                          | Environment                              | Test data                                             |
| ------------------------------ | ---------------------------------------- | ----------------------------------------------------- |
| CI                             | Ubuntu, Python 3.13.18, CPU-only PyTorch | Small test videos and model test doubles where needed |
| Integration with model weights | Local Windows                            | TANet, YOLOX-Nano, 300-frame control video            |
| Web UI E2E                     | Local Windows, Chromium                  | YOLOX-Nano, 24-frame MOT20-01 clip                    |
| GPU benchmark                  | GeForce GTX 1650 (4 GiB), CUDA           | MOT20-01 and MOT20-02, offline replay                 |
| D-10 profile check             | Windows 11, GTX 1650, CUDA 12.8          | Short clean and fog clips                             |

## Test results

The table summarizes [CI on the release candidate][ci] and local checks completed on earlier development revisions.

| Check                            | Result                                              | Evidence              |
| -------------------------------- | --------------------------------------------------- | --------------------- |
| Default CI suite                 | 448 passed, 16 skipped, 0 failed                    | [CI run][ci]          |
| Pipeline smoke tests             | 2 passed in modes A and B, included in the CI total | [CI run][ci]          |
| Integration with trained weights | 14 passed in modes A and D in Lab 4                 | [PR #18][integration] |
| Web UI end-to-end tests          | 7 passed locally in T-23                            | [PR #21][e2e]         |
| Mode C and A–D comparison        | A, B, C, and D ran on a 20-frame clip               | [PR #24][mode-c]      |
| GPU `performance` profile        | Modes A and B measured on clean and fog clips       | [D-10 results][gpu]   |

### What the tests cover

- **CI and regression tests.** The default suite checks video input, run management, policy rules, reports, and quality metrics. Regression tests cover dropped frames counting as missed detections and rejection of mismatched annotations. Lint and formatting passed on the CI-tested revision `7f33adb`.
- **Pipeline smoke tests.** Modes A and B process a three-frame clip with model test doubles. The tests check completion, the run ID and metrics in the report, and the saved detections and tracks. They confirm that B applies TANet before detection and that A processes the original frames.
- **Integration with trained weights.** The tests run modes A and D with TANet and YOLOX-Nano across the three resource profiles. They check D's bypass and enhancement actions, the policy log, and the real analyzer in a separate test. They also check failures caused by missing weights or an invalid video. These 14 tests are skipped by default in CI because they require local test assets.
- **Web UI end-to-end tests.** The browser tests cover a completed mode A run, video playback and results that match the API, quality metrics from COCO and MOT annotations, and clear errors for invalid video, an unavailable API, missing weights, or incomplete or mismatched annotations. The browser suite runs separately from the default CI job.
- **Mode C tests.** CI checks the MWFormer cache workflow, including frame lookup, reuse of cached frames, and clear errors for missing cache entries or weights. The test that loads the real MWFormer weights is skipped by default. [PR #24][mode-c] records a separate 20-frame A–D benchmark and comparison with mode C enabled.
- **GPU profile check.** On `5c0f8b9`, mode B on the fog clip had p95 latency of **197 ms** with GPU `performance` and **1,414 ms** with CPU `standard` on the same machine. These are measurements of the two configured profiles.

## Success criteria

| Criterion              | Target                           | Available result                             | Assessment                 |
| ---------------------- | -------------------------------- | -------------------------------------------- | -------------------------- |
| Detection and tracking | Compare with mode A              | [Benchmark results](benchmark-report.md)     | Final threshold to confirm |
| Speed and resource use | Measure across profiles          | CPU and GPU measurements are available       | Pending final review       |
| Mode C workflow        | Run mode C in the A–D comparison | A, B, C, and D completed on a 20-frame clip. | Passed on the test clip    |

## Defects and remaining risks

**Documented status:** 5 tracked · 3 fixed · 2 closed · 0 open among DEF-01 through DEF-05.

| ID     | Issue                                              | Status | Verification or action                               |
| ------ | -------------------------------------------------- | ------ | ---------------------------------------------------- |
| DEF-01 | Dropped frames affected quality scores.            | Fixed  | Lab 4 regression test passed.                        |
| DEF-02 | MOT20-03 annotations have the wrong dimensions.    | Closed | The clip was excluded from the benchmark.            |
| DEF-03 | A planned early stop could look like a read error. | Fixed  | Retested in Lab 4.                                   |
| DEF-04 | A configuration change lowered detection coverage. | Fixed  | Retested in Lab 4.                                   |
| DEF-05 | Low detection and tracking scores.                 | Closed | [Benchmark evidence](benchmark-report.md)             |

[ci]: https://github.com/Reflectia/Resilens-lab/actions/runs/45290983882
[release-pr]: https://github.com/Reflectia/Resilens-lab/pull/27
[integration]: https://github.com/Reflectia/Resilens-lab/pull/18
[e2e]: https://github.com/Reflectia/Resilens-lab/pull/21
[mode-c]: https://github.com/Reflectia/Resilens-lab/pull/24
[gpu]: https://github.com/Reflectia/Resilens-lab/blob/e6f8150def86d7e5a7b47ec4f7ffab6810ca0701/docs/PROFILES.md
