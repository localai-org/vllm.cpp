# B70 preparation capacity result, 15 October 2026

Wave 1A keeps a bounded, on-demand 72 MiB preparation policy shared by the
backend lease and model-private direct dispatch. The previous two independent
64 MiB checks excluded M3841–4096/K17408/group1 after 256-row padding.
The required prefix is 71,319,552 bytes, including FP32 row scales.
The public scratch layout, arithmetic, validation and completion fences remain
unchanged. Remaining-budget arithmetic now rejects exhausted accounting without
unsigned wraparound. The internal `VT_XPU_W8A8_PREPARE_MAX_MIB=64` control is
retained; only 64/72 are accepted, once per process.

## Fixed workload and result

Base: `bef524f0cb9df20089da2ed9cebce9773deecba4`, with the Wave 1A patch.
The rebuilt unchanged baseline binary matched the historical SHA-256.
Both engines used C1/P4096/O256, greedy/ignore-EOS, prefix caching off,
FP8 KV, batch budget4096, context limit262144, two CPUs and24 GiB/no swap.
Native used page1600/180 blocks. Installed Python retained its own documented
page/cache negotiation. Checkpoint, compact draft vocabulary, image pins,
oneAPI2026.1.1, native oneDNN3.13.0 and Python oneDNN3.13.4 were unchanged.

Three fresh-process control/candidate blocks per mode counterbalanced order;
each used the original primer/four-request protocol, retaining the last-three
median. No profiler or native tracing ran in scored workers. The unaffected
decode-duration tolerance was declared as3% before confirmation.

| Mode / engine | Prefill s | Decode tok/s | Total s |
|---|---:|---:|---:|
| MTP0, 64 MiB control | 2.0222 | 25.365 | 12.0744 |
| MTP0, 72 MiB direct | 1.8297 | 25.351 | 11.8823 |
| MTP0, contemporary Python | 1.4765 | 29.624 | 10.0802 |
| MTP3, 64 MiB control | 2.0795 | 57.821 | 6.4914 |
| MTP3, 72 MiB direct | 1.8818 | 57.858 | 6.2892 |
| MTP3, contemporary Python | 1.4879 | 67.532 | 5.2680 |

Native entries are medians of independent block medians. Median **paired block**
changes are: prefill−9.46%/−9.51%, total−1.50%/−2.94%, and decode duration
+0.08%/+0.14% for MTP0/MTP3. Different aggregation orders need not agree exactly.
Decode is unchanged within observed drift; no decode improvement is claimed.
Rates use255 divided by post-first-token decode duration. Three blocks provide
limited uncertainty evidence, not a population confidence interval.

The candidate still takes23.9%/26.5% more prefill time and has14.4%/14.3% lower
decode throughput than contemporary Python. Total time remains17.9%/19.4% higher.
This is a successful prefill repair, not strict or near Python parity.

## Detecting tests, routes and memory

The new lease test failed on the old code with7 failed assertions:3839/3840
fit, while3841/4095/4096 were refused. After repair:

- Host policy/layout tests:5 cases,887 assertions.
- GPU workspace growth, readers, exceptions and retirement:2 cases,305 assertions.
- Real down-projection complete-call tests:129/3839/3840/3841/4095/4096,
  122 assertions each, exact INT8/scales/padding, FP16 intermediates and outputs.
- Last-row/group failure contracts:8720 assertions atM4096/K17408 and1552
  on the existing two-group fixture. NaN/Inf, both overflow boundaries,
  invalid maps/scales, aliases and failure-then-valid reuse remain protected.
- Forced budget refusal:17 assertions; unsupported145-block producer fallback:
  16 assertions. Both pass.
- Fresh-process complete-operator controls/copy/direct/legacy: exact outputs;
  direct median about3.04 ms versus5.85 ms, approximately48% faster.

The registered external `test_xpu_exl3_preparation_capacity` passes with the
dedicated `VT_B70_EXL3_W8A8_DOWN_FIXTURE`. Optional absence returns77;
required absence and wrong geometry fail. Filtered-out tests are not passes.

Separate fresh P4096/O64 diagnostics show64/65 prefill `ValidateRows` calls
becoming zero in MTP0/MTP3. Direct preparation has no prepared-copy events;
copy controls have512/522. All64 raw emitted IDs match among the three native
variants in each mode. All68 scored serving texts match, and all60 native
responses pass same-engine public-token/usage comparison. HTTP comparisons
do not establish cross-engine raw-ID or full internal-state equality.

The full-model **tracked** device peak increases8,355,840 bytes (7.96875 MiB),
because the preparation high-water rises from62,963,712 to71,319,552 bytes.
The main shared pool stays290,455,616 bytes. Copy preparation increases the
tracked peak48 MiB instead. Graph storage remains8192 bytes in these diagnostics.
These counters exclude untracked driver allocations and do not qualify arbitrary
context/concurrency memory use. A real driver allocation failure was not forced;
the existing error/retirement path is preserved and budget refusal was exercised.

## Decision and evidence

Keep Wave1A. Do not implement conditional Wave1B tiling: the measured extra
tracked memory is small and the direct route fits its real budget. There is
no evidence justifying additional tiled transaction/reconstruction complexity.
The next independent optimization is speculative GDN; numerical/state contracts
remain unchanged.

External evidence directory: `pro-optimization-2026-10-15/wave1a` within the local
recovery artifact root. It contains immutable source snapshots, binary, exact
commands, raw samples, telemetry, fixture provenance, failed initial attempts,
diagnostics, `wave1-summary.json` and `wave1-decision.json`. No automatic mutation
review, broad suite, driver/library change or production restart was performed.
