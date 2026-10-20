# C4 and mixed-work attribution on Intel Arc Pro B70

This is the bounded attribution requested by work package 1 of the current Pro
plan. The reviewed baseline is `e69198adc06401dac1d26411bc03fb0ebd1b1e7a`,
request-stable auto alignment, long-Q1 mode2, SiLU1 and mixed-prefill0. Original
C4-4K/16K prompts, seeds and MTP3 sampling are retained. Each worker stops after
three observed steady B4/Q4 target graph replays; requests are deliberately
cancelled afterward. These profiles are not completed serving requests or a new
native/Python comparison.

## Steady decode

GPU busy below is the union of observed PTI execution intervals, not occupancy.
Kernel families are sums of their own intervals; enclosing host/device spans
are not added to these totals. Medians are from three actual graph replays per
workload, at cycles 6/7/8 and 19/20/21.

| Metric, ms per recorded cycle | C4-4K | C4-16K |
| --- | ---: | ---: |
| GPU interval union | 64.451 | 82.363 |
| EXL3 transforms/GEMM | 36.769 | 36.766 |
| Full attention kernels | 6.094 | 17.414 |
| GDN/convolution | 5.885 | 5.964 |
| Standalone MLP SiLU | 2.622 | 2.632 |
| Norm | 1.700 | 1.828 |
| Sampling/rejection | 2.347 | 9.257 |
| 64 complete target-MLP GPU chains | 23.834 | 23.911 |
| Three complete draft-MLP GPU chains | 1.031 | 1.020 |

The complete MLP chain is source-constrained HadIn -> gate/up GEMM -> HadOut ->
SiLU -> HadIn -> down GEMM -> HadOut, including intervening GPU gaps. It is an
enclosing measurement, not another disjoint kernel family. The first 64 chains
are target MLPs, followed by three draft MLPs. EXL3 remains the largest steady
kernel cost. The measured attention cost and sampling/rejection bucket are
higher in this 16K trace; that observation is not a proved general causal model
for every context or stochastic trajectory.

There are 67 SiLU kernels per steady cycle, 65 generic and two table calls.
Standalone SiLU accounts for 4.07%/3.20% of the union. This justified the narrow
coverage experiment, whose complete operator/MLP and serving results are
reported in [the separate SiLU result](EXL3_XPU_CONCURRENCY_SILU_RESULT_2026-10-20.md).
Graph replay is already active; it is not a missing optimization.

Two generic Split-K attention calls remain per steady cycle: the two pure B4/Q1
draft calls. Target Q4 verification and the first draft Q4 call are separate
routes. Each eligible call has one partial and one cooperative-reduction
kernel, verified against the recorded route events and PTI names.

| Existing eligible Q1 cost | C4-4K | C4-16K |
| --- | ---: | ---: |
| Two calls' GPU kernels, median ms | 1.918 | 6.155 |
| Median per-cycle fraction of GPU interval union | 3.017% | 7.474% |

That is the entire old eligible kernel cost, an upper limit on recoverable GPU
work. It is not a predicted serving speedup, and the new complete call still
pays validation, dispatch, scratch/reduction and queue-completion costs. This
route cannot be credited with the entire approximately24% native/Python C4
throughput gap.

## Cold admission and warmup

The cold subset is all non-steady steps through admission and graph warmup:
cycles 0-5 at 4K and 0-18 at 16K. It includes mixed packets and early partial decode;
it is not isolated client TTFT or a single request's prefill time. The table
reports raw kernel-family durations as a fraction of the GPU interval union in
that subset. Copies/fills and nested host waits remain separate.

| Kernel family, percent of cold-subset GPU interval union | C4-4K | C4-16K |
| --- | ---: | ---: |
| EXL3 transforms/reconstruction/preparation and W8A8 GEMM | 39.07% | 43.67% |
| GDN/convolution | 40.09% | 29.44% |
| Full attention | 3.64% | 12.54% |
| Norm/preamble | 5.29% | 5.66% |
| Dense FP16 oneDNN GEMM | 0.24% | 0.26% |

The earlier generic-name grouping left much cold work in an unknown bucket.
The corrected classification resolves GDN ChunkScratch helpers and EXL3 typed
preparation from current source. Shared `gemm_kernel` names are assigned using
the source's paired oneDNN submission markers and recorded stream kinds, not
by assuming that every GEMM belongs to EXL3. There are 15,484/65,807 EXL3 W8A8
calls and 288/912 dense FP16 calls, with exactly one GEMM appended between each
marker pair on the same submission thread and ordered correlation IDs. No
GEMM remains unassigned.

The initial GPU-timestamp-enclosure assertion fails for two 16K marker pairs
(maximum overlap 9.526 microseconds). That failed analysis and raw timestamps
are retained. The final membership proof uses actual append/correlation order;
it records the GPU timestamp overlaps rather than silently dropping kernels.
Durations remain the individual PTI kernel durations. No extra GPU trace was
collected to obtain a preferred attribution.

Host target/draft, staged-copy, validation and queue waits are retained in raw
cycles. A copy wait can include completion of earlier GPU work; it is not
independent compute. Instrumented median host steps are 313.520/321.521 ms,
including profiler and queue-barrier overhead. Their difference from GPU
interval union must not be presented as production CPU overhead. Capture, warmup, allocation
and queue gaps belong to the diagnostic lifecycle, not to a new scored latency
claim.

## Mixed MTP0 cause and snapshot disposition

The separate slow MTP0 packet has offsets `[0,1,4096]`: one decode query plus
4,095 prefill queries, not uniform B2/Q1. The dominant measured cause is the
whole-batch loss of oneDNN attention when a single-query row is present. The
scoped opt-in correction, same-input numerical control and actual bounded
completion are recorded in [the mixed-prefill result](EXL3_XPU_MIXED_PREFILL_RESULT_2026-10-20.md).
The original FAIL/partial run remains unchanged. Default mixed-prefill remains
off; its fresh default liveness limitation is not relabelled as passed.

There are no snapshot publications in the 4K diagnostic. The 16K diagnostic
publishes one owner at each of cycles 3/7/11/15, never multiple owners in one
step. Each owner has 96 array copies; that is not 96 completion boundaries.
Work package 4A snapshot batching is therefore skipped for this measured scope.
The existing explicit force policy for repeated short prefixes remains a
separate documented deployment choice, not a benchmark-specific dispatch rule.

## Identity and retained evidence

Profile library SHA-256: `62e0ecebfe6a09aa014bcb127995d80e7a41135b1917f1e50d01a4d9b91852b8`.

Diagnostic executable SHA-256: `ea9c5d692062694814ea0609a0d3fcd94ca7bd4fc07a435c65f5ba18d543bd0b`.

The library matches the retained reviewed auto/2 library. Build/source identity,
commands, receipts, original fixtures, per-cycle host/device records and raw PTI
captures are retained under `phase-next-pro-c4-profile-v1`. Its `analysis.json`,
`cold-kernel-analysis.json`, `cold-attribution.json` and`q1-fraction.json` keep
the detailed definitions, counts and arithmetic. Mixed-work controls remain in
`phase-next-pro-attribution-v1`, `phase-next-pro-mixed-control-v1` and
`phase-next-pro-mixed-candidate-v1`. This cost account establishes a bounded
optimization order; it does not reopen broad EXL3, norm or tile campaigns.
