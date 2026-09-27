ID: ISSUE-GH-1660
Title: **A DFlash2 lease needs `cuda-libraries-dev-13-0` and `python3-dev`, and `RC_LEASE_ID` does not exist on this fleet.** Measured on `dgx:gpu0` 2026-08-22 over leases `52ac5673` and `a03f34e4`. DFlash2's `compute_candidates` -> `_topk` -> `flashinfer.topk` JIT-compiles `topk.cu`, which includes `<curand.h>`; the `cuda-toolkit-13-0` metapackage does NOT install that header, and leg B died on it INSIDE `profile_run` **after a 12-minute model load**, presenting as a model failure rather than a missing header. Leg C installed `cuda-libraries-dev-13-0` and got past it. `python3-dev` is likewise required or Triton driver JIT compilation fails, and that failure surfaces as `Model architectures ['Qwen3_5ForConditionalGeneration'] failed to be inspected` -- which names the model, not the toolchain. Separately, the leased worker carries `RC_DEVICE`, `RC_JOB_ID` and `RC_TOKEN` and no `RC_LEASE_ID`, so the gate's `--lease-id` default read empty and the `## Owed` O26 recipe run VERBATIM refused with `no lease id. Claim the device with rc run/rc hold first; never ssh to a fleet box`. The refusal is correct on a missing lease; the defect is that the committed procedure could not satisfy its own gate. FIXED IN FLOW: the default takes `RC_JOB_ID` first and falls back to `RC_LEASE_ID` -- the measured variable first, the documented-but-absent one for a controller that does export it -- both packages are recorded in [`environment.md`](../../environment.md) beside the existing note that no CUDA toolkit is preinstalled, and the O26 recipe now names them. Owned by [`dflash2-spec-decode.md`](../../specs/dflash2-spec-decode.md) `## Owed` O28
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1660
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:598`

### Frozen archive evidence

> | [#1660](https://github.com/mudler/vllm.cpp/issues/1660) | `SPEC-DFLASH2` | **A DFlash2 lease needs `cuda-libraries-dev-13-0` and `python3-dev`, and `RC_LEASE_ID` does not exist on this fleet.** Measured on `dgx:gpu0` 2026-08-22 over leases `52ac5673` and `a03f34e4`. DFlash2's `compute_candidates` -> `_topk` -> `flashinfer.topk` JIT-compiles `topk.cu`, which includes `<curand.h>`; the `cuda-toolkit-13-0` metapackage does NOT install that header, and leg B died on it INSIDE `profile_run` **after a 12-minute model load**, presenting as a model failure rather than a missing header. Leg C installed `cuda-libraries-dev-13-0` and got past it. `python3-dev` is likewise required or Triton driver JIT compilation fails, and that failure surfaces as `Model architectures ['Qwen3_5ForConditionalGeneration'] failed to be inspected` -- which names the model, not the toolchain. Separately, the leased worker carries `RC_DEVICE`, `RC_JOB_ID` and `RC_TOKEN` and no `RC_LEASE_ID`, so the gate's `--lease-id` default read empty and the `## Owed` O26 recipe run VERBATIM refused with `no lease id. Claim the device with rc run/rc hold first; never ssh to a fleet box`. The refusal is correct on a missing lease; the defect is that the committed procedure could not satisfy its own gate. FIXED IN FLOW: the default takes `RC_JOB_ID` first and falls back to `RC_LEASE_ID` -- the measured variable first, the documented-but-absent one for a controller that does export it -- both packages are recorded in [`environment.md`](../../environment.md) beside the existing note that no CUDA toolkit is preinstalled, and the O26 recipe now names them. Owned by [`dflash2-spec-decode.md`](../../specs/dflash2-spec-decode.md) `## Owed` O28 | bug |

## Resolution

-
