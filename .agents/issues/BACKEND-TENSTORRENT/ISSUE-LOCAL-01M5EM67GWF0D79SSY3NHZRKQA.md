ID: ISSUE-LOCAL-01M5EM67GWF0D79SSY3NHZRKQA
Title: the installed tt-metal evidence substrate carries an unrecorded rms_norm precision edit, so the pinned rms numbers are not reproducible from the recorded pin
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-21
Updated: 2026-10-21
Closed: -

## Problem

The P150 evidence substrate /tmp/umdtrial-install runs ttnn::rms_norm with approx_mode=false and fp32_acc=true, because the build tree /tmp/tt-metal-umdtrial carries two uncommitted working-tree edits (rmsnorm.cpp: approx_mode false, fp32_acc true; layernorm_op_multi_core.cpp: the 1/W scaler tile as Float32 when fp32_dest_acc_en is set). Neither edit is in the recorded pin source tree ~/Sources/tt/tt-metal-pin (rmsnorm.cpp there still reads approx_mode=true, fp32_acc=false) and neither is in the recorded patch series ~/Sources/tt/tt-metal-patches/0001..0004. The ttnn library was re-installed on 2026-10-10 17:36 with the edits, so every P150 measurement taken since then, including the rms_norm deviation record, rests on a substrate the pin does not describe. MEASURED 2026-10-21 (P150, docs/bench-evidence/tt-rms-norm-compute-config-20261021.md): the same captured rows read 0.980947 on this substrate when the rms call runs with fp32_acc=false and 1.007197 when it runs with fp32_acc=true and approx_mode=false; the default decides because the vllm.cpp rms path passes no compute configuration. FIX: record the two edits in the pin (a fifth patch file plus the pin commit) or restore the pin source to the installed substrate's state, and state in the pin record which rms_norm compute configuration the evidence substrate runs. Until then the P150 numerics are reproducible only from the installed tree, not from the recorded pin.

## Resolution

-
