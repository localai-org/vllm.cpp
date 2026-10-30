ID: ISSUE-LOCAL-01M54CKBVMKZEGP8ZWM3G0CCT6
Title: kolibri1 P150: tt-metal fused rms_norm shows an input-distribution-dependent ~0.8% low bias; pre-scale falsified, fix is in the fused kernel round/reduce path
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-17
Updated: 2026-10-21
Closed: 2026-10-21

## Problem

Diagnostic (VT_TT_NORM_DUMP_DIR capture + VT_TT_RMSNORM_MICRO_INPUT replay in the SCRATCH dbg rmsnorm micro): a REAL in-model norm row reproduces the bias standalone (tt/host-double sum ratio 0.99176, max_abs 0.0625) where the random row reads 0.9997 — input-distribution dependence confirmed, program provenance excluded (same call site). Scaling probes x0.25/x1/x4: ratio UNCHANGED (0.995/0.992) — the bias is scale-invariant, so the classic pre-scale-by-max fix is falsified. Elementwise: 74% of elements match the host-double oracle exactly; the differing ~26% are net-low 1-3 ULP artifacts. The composed f32-primitives arm (VT_TT_RMSNORM_COMPOSED=1) scores 0.99956 / max_abs 0.0039 on the same row — the fused ttnn::rms_norm kernel's internal round/reduce composition is the defect. BLOCKER: routing the model through the composed arm throws binary_ng TT_THROW (binary_ng_device_operation.cpp:224) on model tensor geometry — the existing documented blocker; fixing the fused layernorm kernel's round path is tt-metal substrate work beyond the rmsnorm kernel.

## Resolution

2026-10-21: DUPLICATE, closed onto the surviving record
`ISSUE-LOCAL-01M578VP02HPVBYSMRP32GWZGF` ("kolibri1 P150: tt-metal rms_norm
shows a 0.7% input-dependent deviation on real in-model rows -- kernel
precision, not call-site config"), the same row, opened the next day.

Both records name one substrate defect: the pinned tt-metal fused rms_norm
kernel's round and reduce composition on kolibri's real in-model activation
distribution. The surviving record reproduces this record's numbers with the
standalone micro, carries the same composed-arm blocker
(`VT_TT_RMSNORM_COMPOSED=1` is refused on the model's tensor geometry), and
points at the same fix area, the fused kernel's round path in `layernorm.cpp`.
It also owns the upstream re-check this record predates (tt-metal head
2500c1d8f63, 2026-10-19: not improved, and regressed on the largest-magnitude
captured row), and the row's other records already cite it as the canonical
rms_norm-deviation record.

The evidence unique to this record is preserved here so nothing is lost. The
pre-scale fix is falsified: scaling probes x0.25/x1/x4 leave the tt/host-double
sum ratio unchanged (0.995/0.992), so the bias is scale-invariant and the
classic pre-scale-by-max fix does not apply. Elementwise, 74% of the elements
match the host-double oracle exactly and the differing ~26% are net-low 1-3
ULP artifacts. The composed f32-primitives arm scores 0.99956 with max_abs
0.0039 on the same row, against the fused kernel's 0.99176 with max_abs
0.0625, which is why the defect is named to the fused kernel's round and
reduce composition rather than to the accumulation.

