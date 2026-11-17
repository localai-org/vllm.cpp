# B70 bounded C1 Xe2 decode,15 October2026

Wave3 keeps32-split Xe2 decode for the existing F16 Q/output,24Q/4KV heads,
D256,unit-scaled E4M3 cache and the pinned B70/compiler/driver admission.
Pages1600/1664 are supported, with a host length bound of at most three pages
(4800/4992). Other forms retain the existing fallback. This does not qualify the
configured262144-token maximum. No library, dtype, model, depth or workload changes.

## Route, scratch and graph contract

`VT_XPU_XE2_LONG_DECODE` defaults1 after qualification;0 retains the old control.
`VT_XPU_XE2_DECODE_SPLITS` accepts16/32,default32. The one-split short route uses
a stable960-key launch bound. The long route rounds to a physical page and uses
the existing16MiB stable attention reservation, with disjoint F16 partial outputs
and F32 sums/maxima. No new per-token host length readback or allocation is added.
Fresh metadata checks enforce the captured bucket before any output write.
The model graph key includes route flags, splits and cache-specific launch bound.

CPU routing/planner tests pass1 case /82 assertions; final host graph-policy tests
pass3 cases /106 assertions. The GPU canary fails its selection assertion on the
old route, then passes the existing matrix tolerances on the candidate.
Bounded tests cover960/961,1599/1600/1601,4095/4096,4800/4801 and4992/4993,
16/32splits,planar/interleaved storage,permuted active pages,invalid unused table
entries and zero/NaN-poisoned inactive capacity. They also check nonunit-scale and
strided-table fallback. Graphs read fresh metadata, reject stale buckets and bad
offsets/pages without modifying output, and share stable scratch across queues.
These two focused tests pass1506 assertions. Initial test compile/setup failures
are retained; missing includes and duplicate doctest capture names were fixed.

## Actual operator replay

A separate eager driver captures the first actual target-only C1 attention after
P4096,length4097. Active encoded KV pages are copied losslessly and remapped into
a bounded replay fixture; the original physical strides/page IDs are recorded.
The packed control reproduces the captured model output byte-for-byte.

Against the same native reference operator,relative L2 is1.34623e-5 for the old
Split-K route,3.10521e-4 for16splits,and3.07799e-4 for32splits. The latter maxima
are0.0009765625 at reference peak1.919921875. All pass the unchanged matrix gate:
relative L2<=0.001 and maximum<=3e-6+0.002*reference_peak. These are tensor errors,
not answer-error rates; the faster route is not bit-exact to the old route.

Balanced isolated trials give complete-operator medians0.576067ms control,
0.255706ms split16,and0.1854505ms split32. Profiled forward-plus-reduction regions
are546.666/223.907/153.958us. These diagnostics exclude setup/restoration/downloads
and are not serving scores. Inputs remain byte-identical.

## Trace-free serving confirmation

Three counterbalanced fresh-process blocks per mode preserve C1/P4096/O256,
greedy/ignore-EOS/logprobs0,prefix-off,FP8 KV,page1600/180blocks,and2CPU/24GiB/no
swap. W8A8 cap72 and the approved subgroup recurrence are fixed in both arms.
Scores contain no profiler. Values below are medians of block medians.

| Mode | Route | Prefill s | Decode tok/s | Total s |
|---|---|---:|---:|---:|
| MTP0 | old Split-K | 1.833354 | 25.2760 | 11.916602 |
| MTP0 | Xe2 split32 | 1.834384 | 27.3103 | 11.171521 |
| MTP3 | old Split-K | 1.890117 | 62.7502 | 5.953856 |
| MTP3 | Xe2 split32 | 1.887923 | 63.3255 | 5.914603 |

Median paired changes: MTP0 decode throughput+7.77%,decode duration-7.21%,total
-6.04%,prefill+0.06%; MTP3 throughput+0.82%,total-0.52%,prefill+0.10%.
MTP0 throughput gains range6.77–8.12% across blocks; MTP3 gains0.57–0.92%.
Unaffected metrics meet the predeclared3% duration non-regression tolerance.
All48 native responses pass the frozen same-engine token/text/usage baseline
and direct paired comparisons. This is bounded workload evidence,not cross-engine
raw-ID parity or general long-generation quality. No contemporary Python score
is inferred from these native-only blocks; final comparison remains separate.

Frozen source/binaries,commands,raw replies,traces,input manifests and metrics
are under `pro-optimization-2026-10-15/wave3` in the local recovery artifact root.
The default-adoption build uses the same qualified route selected explicitly1
in scored workers. Production stays stopped. Wave4 starts with a separate PTI
reprofile of this effective configuration; no generic Split-K sweep is needed.


## Fresh Python comparison after Wave3

Python was rerun sequentially in the same pinned installed image,with the same
P4096/O256 protocol and no profiler. Native uses three block medians; Python
has one fresh four-request worker per mode,so uncertainty is asymmetric.

| Mode | Native/Python prefill s | Native extra prefill time | Native/Python decode tok/s | Native lower throughput | Native/Python total s |
|---|---:|---:|---:|---:|---:|
| MTP0 | 1.834/1.462 | +25.46% | 27.31/29.90 | -8.65% | 11.172/9.995 |
| MTP3 | 1.888/1.475 | +28.04% | 63.33/68.64 | -7.74% | 5.915/5.193 |

All four decoded texts per mode and raw prompt IDs agree between engines.
Cross-engine raw output IDs or full logits/state equality are not established.
Decode rates include the actual target/draft acceptance behavior in MTP3.
This remains a measured remaining gap,not fixed-workload parity.
