ID: ISSUE-LOCAL-01M54CKBVMKZEGP8ZWM3G0CCT6
Title: kolibri1 P150: tt-metal fused rms_norm shows an input-distribution-dependent ~0.8% low bias; pre-scale falsified, fix is in the fused kernel round/reduce path
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-17
Updated: 2026-10-17
Closed: -

## Problem

Diagnostic (VT_TT_NORM_DUMP_DIR capture + VT_TT_RMSNORM_MICRO_INPUT replay in the SCRATCH dbg rmsnorm micro): a REAL in-model norm row reproduces the bias standalone (tt/host-double sum ratio 0.99176, max_abs 0.0625) where the random row reads 0.9997 — input-distribution dependence confirmed, program provenance excluded (same call site). Scaling probes x0.25/x1/x4: ratio UNCHANGED (0.995/0.992) — the bias is scale-invariant, so the classic pre-scale-by-max fix is falsified. Elementwise: 74% of elements match the host-double oracle exactly; the differing ~26% are net-low 1-3 ULP artifacts. The composed f32-primitives arm (VT_TT_RMSNORM_COMPOSED=1) scores 0.99956 / max_abs 0.0039 on the same row — the fused ttnn::rms_norm kernel's internal round/reduce composition is the defect. BLOCKER: routing the model through the composed arm throws binary_ng TT_THROW (binary_ng_device_operation.cpp:224) on model tensor geometry — the existing documented blocker; fixing the fused layernorm kernel's round path is tt-metal substrate work beyond the rmsnorm kernel.

## Resolution

-
