ID: ISSUE-GH-1548
Title: **The `cuda` lane built its ten-SM fat binary at `$(nproc)` on a hosted runner and the runner died under it, and no job declared `timeout-minutes`.** Run [32447481128](https://github.com/mudler/vllm.cpp/actions/runs/32447481128) died at object 512 of 787, about 35 minutes in, with `the runner has received a shutdown signal` and exit 143. Not a timeout: no `timeout-minutes` appeared anywhere in `.github/workflows/containers.yml`, so the six-hour default applied. `scripts/build-linux-accelerator-release.sh:24` sets ten device architectures, so each `.cu` is compiled ten times and one compiler process holds many times the resident set of a `cpu` or `vulkan` translation unit. MITIGATED, NOT DIAGNOSED: memory exhaustion is the leading hypothesis and it is NOT proven, because GitHub infrastructure reclamation produces the same message and the same exit code and the available logs cannot separate them. The change removes the one cause this repository controls. Parallelism is now LANE-AWARE, so the `cpu` and `vulkan` lanes are not slowed: they keep `$(nproc)` and only `cuda` takes 2. The value is measured, not guessed. `.github/workflows/ci.yml:801` already builds the SAME ten-SM fat gencode set on a hosted runner at `--parallel 2` and is green, and the 512-of-787 data point puts a halved build near two hours, which answers `.agents/specs/container-images.md:200-204` and its concern that two jobs would not finish inside a hosted runner's limits. `timeout-minutes: 300` goes on both building jobs for a separate reason: under the six-hour default a hang, a reclaimed runner and an exhausted one all report the same exit 143, so the next failure is undiagnosable. The budget LABELS it rather than policing it, and is loose because no arm64 container leg has ever built. `scripts/check-container-workflow.py` gates the SHAPE and not the number, so retuning the cap needs no checker edit, and prints the resolved cap and budget in its OK line. Spec [fix-ci-container-publish.md](../../specs/fix-ci-container-publish.md)
Row: ENG-RELEASE-CONTAINERS
State: UNKNOWN
Kind: bug
GitHub: 1548
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:545`

### Frozen archive evidence

> | [#1548](https://github.com/mudler/vllm.cpp/issues/1548) | `ENG-RELEASE-CONTAINERS` | **The `cuda` lane built its ten-SM fat binary at `$(nproc)` on a hosted runner and the runner died under it, and no job declared `timeout-minutes`.** Run [32447481128](https://github.com/mudler/vllm.cpp/actions/runs/32447481128) died at object 512 of 787, about 35 minutes in, with `the runner has received a shutdown signal` and exit 143. Not a timeout: no `timeout-minutes` appeared anywhere in `.github/workflows/containers.yml`, so the six-hour default applied. `scripts/build-linux-accelerator-release.sh:24` sets ten device architectures, so each `.cu` is compiled ten times and one compiler process holds many times the resident set of a `cpu` or `vulkan` translation unit. MITIGATED, NOT DIAGNOSED: memory exhaustion is the leading hypothesis and it is NOT proven, because GitHub infrastructure reclamation produces the same message and the same exit code and the available logs cannot separate them. The change removes the one cause this repository controls. Parallelism is now LANE-AWARE, so the `cpu` and `vulkan` lanes are not slowed: they keep `$(nproc)` and only `cuda` takes 2. The value is measured, not guessed. `.github/workflows/ci.yml:801` already builds the SAME ten-SM fat gencode set on a hosted runner at `--parallel 2` and is green, and the 512-of-787 data point puts a halved build near two hours, which answers `.agents/specs/container-images.md:200-204` and its concern that two jobs would not finish inside a hosted runner's limits. `timeout-minutes: 300` goes on both building jobs for a separate reason: under the six-hour default a hang, a reclaimed runner and an exhausted one all report the same exit 143, so the next failure is undiagnosable. The budget LABELS it rather than policing it, and is loose because no arm64 container leg has ever built. `scripts/check-container-workflow.py` gates the SHAPE and not the number, so retuning the cap needs no checker edit, and prints the resolved cap and budget in its OK line. Spec [fix-ci-container-publish.md](../../specs/fix-ci-container-publish.md) | bug |

## Resolution

-
