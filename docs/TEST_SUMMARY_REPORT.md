# Resilens 2.0 Test Summary Report

**Status:** Draft  
**Development baseline:** `main` at `333d619` (`2.0.0.dev0`)  
**Final release tag and commit:** TBD

## Test environments

| Check              | Environment                              | Test data                                  |
| ------------------ | ---------------------------------------- | ------------------------------------------ |
| CI                 | Ubuntu, Python 3.13.18, CPU-only PyTorch | Model test doubles where needed            |
| Integration        | Local Windows                            | TANet, YOLOX-Nano, 300-frame control video |
| Web UI E2E         | Local Windows, Chromium                  | YOLOX-Nano, 24-frame MOT20-01 clip         |
| GPU benchmark      | GeForce GTX 1650 (4 GiB), CUDA           | Offline replay                             |
| D-10 profile check | Windows 11, GTX 1650, CUDA 12.8          | Short clean and fog clips                  |

## Test results

These results come from the development baseline and earlier runs. The results for the final release commit are pending.

| Check                          | Status           | Key result                                              | Evidence                                |
| ------------------------------ | ---------------- | ------------------------------------------------------- | --------------------------------------- |
| Default CI suite               | Passed           | 399 passed, 15 skipped, 0 failed                        | [CI run][ci]                            |
| Integration with model weights | Passed in Lab 4  | 14 passed in modes A and D                              | [PR #18][integration]                   |
| Pipeline smoke tests           | Passed           | 2 passed in modes A and B                               | [CI run][ci]                            |
| Web UI E2E                     | Passed locally   | 7 passed, including upload, results, and error handling | [PR #21][e2e]                           |
| Benchmark for modes A–D        | Partial          | 54 runs across 18 scenarios in A, B, and D              | [Benchmark report](benchmark-report.md) |
| GPU `performance` profile      | Measured in D-10 | Modes A and B measured on clean and fog clips           | [D-10 results][gpu]                     |


The D-10 profile check used commit `5c0f8b9`. For mode B on the fog clip, p95 latency was **197 ms** with the GPU `performance` profile and **1,414 ms** with the CPU `standard` profile on the same machine.

## Success criteria

| Criterion              | Target                  | Available result                                            | Assessment           |
| ---------------------- | ----------------------- | ----------------------------------------------------------- | -------------------- |
| Detection and tracking | Compare with mode A     | Mode D improved mAP, HOTA, and IDF1                         | Threshold to confirm |
| Speed and resource use | Measure across profiles | [CPU profiles](PROFILES.md) and [GPU profile][gpu] measured | Pending final review |
| Benchmark coverage     | Cover modes A–D         | A, B, and D measured. C is pending.                         | Partly met           |


## Defects and remaining risks

**Last documented status:** 5 tracked · 3 fixed · 1 open · 1 closed by exclusion. Final retest results are pending.

| ID     | Issue                                                   | Status              | Verification or action                                           |
| ------ | ------------------------------------------------------- | ------------------- | ---------------------------------------------------------------- |
| DEF-01 | Dropped frames affected quality scores.                 | Fixed               | Lab 4 regression test passed.                                    |
| DEF-02 | MOT20-03 annotations have the wrong dimensions.         | Closed by exclusion | Clip removed from the benchmark.                                 |
| DEF-03 | A planned early stop could look like a read error.      | Fixed               | Retested in Lab 4.                                               |
| DEF-04 | A configuration change lowered detection coverage.      | Fixed               | Retested in Lab 4.                                               |
| DEF-05 | Low scores and limited tracking of new or lost targets. | Open                | The team still needs to decide whether to accept the limitation. |

The main remaining risks are limited tracking quality and incomplete benchmark coverage. Confirm mode C's release scope and record whether the team accepts the tracking limitation.

[ci]: https://github.com/Reflectia/Resilens-lab/actions/runs/45062215598
[integration]: https://github.com/Reflectia/Resilens-lab/pull/18
[e2e]: https://github.com/Reflectia/Resilens-lab/pull/21
[gpu]: https://github.com/Reflectia/Resilens-lab/blob/e6f8150def86d7e5a7b47ec4f7ffab6810ca0701/docs/PROFILES.md
