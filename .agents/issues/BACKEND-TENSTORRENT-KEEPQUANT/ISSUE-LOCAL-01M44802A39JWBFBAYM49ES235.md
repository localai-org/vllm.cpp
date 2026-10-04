ID: ISSUE-LOCAL-01M44802A39JWBFBAYM49ES235
Title: int8-dot keep-quant capture: eager host activation staging did not persist as the slot device shadow
Row: BACKEND-TENSTORRENT-KEEPQUANT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-04
Updated: 2026-10-04
Closed: 2026-10-04

## Problem

MatmulBTQuantInt8DotKernel's eager host staging (ttnn::Tensor::from_span over the bf16/f32 master) built the activation tensor per call and discarded it, so the capture pass re-reached the staging branch and refused by name ('bf16 activation staging during trace capture'). Both int8-dot 50 MiB trace-capture legs (F32-out, BF16-out dispatch) fatalled on it. The break is pre-existing at the pre-flip HEAD (bf69008cd~1): the F32-out leg skipped under the old default-off lever, so the flip exposed it on the default configuration. One such mid-capture throw also leaks tt_capture_active() (no EndCapture on the exception path), which poisoned the 17 following capture-based cases in-suite (word-shadow miss, UploadRowsBf16 refused inside trace, nested TraceBeginCapture). Diagnosis: VT_TT_TRACE_DEBUG act-serve miss probe shows slot=1 hasdev=0 devcur=0 hostcur=1 on both the eager and the capture call.

## Resolution

FIXED (2026-10-04, commit 13b51cd54, row/int8dot-default-flip): MatmulBTQuantInt8DotKernel's eager host staging persists as the slot's device shadow (the words-shadow discipline applied to the activation). Evidence 2026-10-04: both 50 MiB int8-dot capture legs green isolated; full TT suite 102/103, the only failure the pre-recorded owed RAC 2261 flake (/tmp/flip-suite3.log). The tt_capture_active() leak on a mid-capture throw is unreachable for these legs after the fix and is noted as robustness debt in the row spec.
