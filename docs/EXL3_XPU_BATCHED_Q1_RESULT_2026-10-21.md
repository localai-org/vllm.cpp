# Batched pure-Q1 attention on B70: qualified bounded default

The candidate adds exact B2-B4 one-query-per-request admission to the existing
bounded Xe2 decode donor. `VT_XPU_XE2_BATCHED_Q1` now defaults to `1`; explicit `0` rolls back
to the original batched fallback. Original B1 behavior remains. Q4, ragged and mixed packets retain
their existing routes. The experiment uses current request-stable auto/long2
with SiLU 1 and mixed-prefill 0; each new mechanism is isolated.

## Route and storage contract

Both total queries and every existing host offset must prove pure Q1:
`[0,1,...,B]`. Total queries equal to B is insufficient for valid ragged packets
such as `[0,0,2]` or `[0,2,2]`. The existing device metadata check independently
validates fresh offsets, lengths, pages and the bounded launch on eager calls
and graph replays. No new device-to-host length readback is introduced.

Existing F16 query/output, E4M3 KV, head/GQA, layout, scale, causal, device,
compiler, driver and page guards remain. Batched table rows must be contiguous.
The long modes 0/1/2 and splits 16/32 keep their existing meanings. The same
bounded policy admits at most 128 physical pages per request, for pages 1600/1664;
outside that range, auto retains the previous fallback.

Scratch scales its disjoint F16 partials and F32 sum/maxima regions by B. The
per-request size is checked before multiplying by batch against the actual
fixed 16-MiB allocation. The real allocator enforces that budget, prevents growth
and retains queue/graph lifetime ownership. There is no new per-token allocation
or synchronization in the contiguous admitted route. Existing public alias and
strided-output materialization retain their own established behavior.

The mode and bounded launch enter graph identity. Route logging distinguishes
`xe2_decode_batch` and records actual requests and queries.

## Real-input complete-call screen

Transparent same-queue copies capture three actual C4 replay frames per context.
The final pure B4/Q1 call has live lengths `[4100,4110,4103,4073]` at 4K and
`[16423,16420,16349,16396]` at 16K. Queries and outputs are finite. The final frame
retains all live FP8 page bytes, relocated through an explicit bijection that
preserves sharing and original page IDs; unused table cells are invalid.
The generic flag0 call exactly reproduces the originally consumed F16 output
after relocation. Each context's operator case passes 58 assertions.

The complete public-call screen includes metadata checking, dispatch,
scratch/reduction and queue completion. File I/O, buffer setup and comparison
are outside timing. One frozen binary runs both modes, with two warmups and
seven samples per arm, one A/A pair and three alternating A/B pairs. Profilers
and traces are off. Operator containers have two CPU cores and 12 GiB RAM;
ordinary serving measurements retain the original uncapped CPU/RAM envelope.
Absolute operator times are not substituted for scored request latency.

| Input | Pair | Flag0 median, microseconds | Flag1 median, microseconds | Complete-call change |
| --- | ---: | ---: | ---: | ---: |
| c4-4k | 1 | 961.116 | 168.199 | -82.500% |
| c4-4k | 2 | 1009.165 | 164.829 | -83.667% |
| c4-4k | 3 | 998.445 | 162.529 | -83.722% |
| c4-16k | 1 | 3026.625 | 528.537 | -82.537% |
| c4-16k | 2 | 4468.348 | 523.798 | -88.278% |
| c4-16k | 3 | 2982.426 | 400.427 | -86.574% |

All raw seven-sample vectors, including outliers, remain unchanged. A/A is
variable in this short operator test: reference medians 2,150.369 -> 1,838.642 us
at 4K and 5,666.812 -> 2,993.155 us at 16K. No extra samples were collected to
improve the sign. Every paired candidate call is substantially cheaper even
against the fastest reference samples. The allowed relative L2 maxima are
0.000320212/0.000333713, with maximum absolute difference 0.015625; all outputs
are finite and pass the unchanged relative-L2<=0.001 and
max-absolute<=3e-6+0.002*reference-peak matrix contract.

The old eligible GPU cost is only 3.017%/7.474% of the recorded 4K/16K GPU interval
union, from two pure-Q1 draft calls. Target Q4 and first-draft Q4 are separate.
These fractions are the entire old cost, an upper bound on recoverable GPU work,
not a claim of 84-87% higher model throughput. See the [attribution result](EXL3_XPU_C4_ATTRIBUTION_RESULT_2026-10-21.md).

## Detecting focused checks

- RED: exactly three missing route-admission assertions for B2/B3/B4 fail before
  the donor extension; old fallback outputs still match the reference.
- Final same-process GPU guard set: four cases / 1,880 assertions pass. It covers
  heterogeneous queries/lengths/page order, both supported cache layouts, zero
  versus NaN neighbor/tail poison, B4->B3->B2->B1, replacement of a cancelled
  owner's pages, aliasing, unchanged inputs/metadata/KV, invalid live metadata,
  cross-queue scratch clobbering and graph destruction with work pending.
- Actual last-capacity computation and one-page overflow are exercised at both
  page sizes and splits 16/32 with a constant-value oracle and poisoned unused
  neighbors. The full 16-MiB reservation stays fixed.
- Default/off, missing host hints, ragged offsets, BF16/F32, changed scales,
  strided/padded tables and unsupported query layout preserve fallback/refusal.
  Existing strided-output materialization preserves guards. Q4 remains on its
  qualified packed route; mixed Q1+prefill does not enter the new route.
- Host policy: one case / 36 assertions pass. Graph policy: three cases / 250
  assertions pass, including page/capacity/mode transitions and B1 parameters.
- Existing B1 numeric/layout/poison retention: one case / 1,426 assertions pass.
  No full test suite was run.

The first lifecycle-test compile fails because two doctest capture macros share
one source line. That log remains; the corrected build and final focused tests
pass. This is distinct from a product numerical or lifecycle failure.

## Actual engine lifecycle

The current linked engine completes one bounded MTP3 lifecycle with the original
four 4K prompts, short unequal caps, replacement requests and a 16K insertion.
Six uncancelled requests finish with 32/64/96/64/48/48 tokens; request 3 is actually
aborted after 54 tokens and does not return. There are 43 completed steps, within
the declared 400-step/300s+15stop limits; total wall time including startup is
46.200 s. Cumulative-prefix, active-owner and finish-reason checks pass.

The actual new-route admissions are B2: 4, B3: 6 and B4: 62. All six completed
greedy token sequences exactly match the retained auto/2 lifecycle reference.
This is scoped functional evidence, not a universal quality or serving-speed
claim. The known strict closing-fence failure and separate incomplete default
64K MTP0/C2 liveness remain explicit. The earlier opt-in mixed-prefill bounded
completion is preserved separately; it is not this test.

## Final short-context and NaN-scratch checks

After all 16 scored workers finished, test-only checks explicitly filled the
entire shared 16-MiB reservation with NaNs in both F16/F32 views and exercised
B2/B3/B4 at contexts 1/960/961/4096, including one-split launches. The intermediate
four-case set passed 1,875 assertions without changing the product library or
serving executable. After changing the default literal to 1, detecting tests of
the actual unset/default route brought the final set to 1,880 passing assertions.
Policy 1 / 36 and graph 3 / 250 also passed, including the unset/default bounded launch.
No compile or second GPU workload overlapped scored workers.

## Serving disposition: promoted

All 16 predeclared workers completed: two C4-4K A/A controls, three alternating
A/B pairs per C4 context and one C1-4K sentinel pair. Both arms used the same
frozen executable, auto/2, SiLU 1, mixed-prefill 0, original stochastic HTTP
payloads and 1,024-token caps. Profilers/traces were off; original child flags,
raw metrics, outputs, resource receipts and sample order are retained.

| Case | Pair | Flag0 aggregate decode tok/s | Flag1 aggregate decode tok/s | Decode change | Client TTFT change | Client E2E change | Control accepted/proposed | Candidate accepted/proposed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C4-4K | 1 | 140.838 | 144.066 | +2.291% | +0.140% | -1.205% | 2575/4563 | 2561/4608 |
| C4-4K | 2 | 139.987 | 143.875 | +2.777% | -0.205% | -2.002% | 2566/4599 | 2565/4602 |
| C4-4K | 3 | 132.785 | 137.178 | +3.308% | +0.033% | -2.350% | 2496/4809 | 2496/4809 |
| C4-16K | 1 | 125.516 | 133.803 | +6.603% | +0.056% | -2.655% | 2603/4488 | 2594/4512 |
| C4-16K | 2 | 125.515 | 133.139 | +6.074% | +0.013% | -1.960% | 2603/4488 | 2615/4437 |
| C4-16K | 3 | 125.518 | 134.209 | +6.925% | -0.086% | -2.654% | 2603/4488 | 2594/4512 |

Decode is aggregate generated-counter throughput in the fully overlapping C4
window. TTFT/E2E are client latencies, not summed GPU compute durations. All
workers completed; cold committed prefill was 16,292/65,460 tokens for 4K/16K,
with zero cache hits and zero preemptions. Accepted/proposed counts above cover
the full wave; overlap-window counts are separately retained in the raw report.

The new C4-4K A/A pair differed by -3.515% decode, +0.733% TTFT, +1.142% E2E.
A prior comparable auto/2 A/A pair differed by -4.835% decode. These are observed
pairs, not confidence bounds. The 4K median +2.777% decode is a positive trend
below observed A/A variation, not an independently noise-qualified gain.
Every 16K gain exceeds both observed control differences and agrees with the
complete-call result. The 16K medians are +6.603% decode, +0.013% TTFT and -2.654%
E2E. The C1 sentinel was effectively unchanged: -0.001% client post-first-token
decode, -0.099% TTFT and -0.008% E2E, with exact output hashes.

This supports bounded B2-B4 Q1 promotion on the intended concurrent long-context
workload, not statistical significance or a universal speedup. Stochastic C4
output hashes and MTP trajectories can differ, including between A/A controls.
Numerical-contract checks and exact ownership/lifetime checks remain separate.
The already accepted auto/2 baseline costs are not attributed to this patch.

The promoted combination has completed its [fresh full native/Python comparison](benchmarks/exl3-batched-q1-2026-10-21/README.md):
20 scenarios / 70 measured waves / 124 requests plus three isolated 64K requests
per engine, all 254 measured requests with 1,024-token length finishes and zero
preemptions. Compiled auto/long2/batched-Q1=1, SiLU1/mixed-prefill0 are used with
all five policy overrides unset. The earlier auto/2 matrix remains historical.
The report discloses its one offline counter-window boundary correction and
retains original samples/driver fields; the paired Q1 C4 screen is unchanged.
Its C1 sentinel uses the declared client-timing metric; one auxiliary counter
view is unqualified and is not used as performance evidence.

## Exact identity and raw evidence

Serving-screen executable SHA-256: `0c821133e1f6e4a5548938b7ee4972473d7aeab462e552139abf7abfc59d56b8`.

Product library SHA-256: `86707cd80f8b998d6c146e9a38797bf21fd46c5536a7bcf89e46f8581f5f8a35`.

The source is base `ec0875cc503231f4c76f98a0880d442cf5a2716a` plus the eight exact
source/test files in `phase-next-pro-q1-candidate-v1/final-focused-identities.json`
and `source-final`. Frozen test binaries are bound there too. The earlier real-
input operator binary and its exact source are frozen separately in
`operator-identity.json` and `source-operator-v1`; its product route is the same.

Raw evidence remains in `phase-next-pro-q1-operands-v1`,
`phase-next-pro-q1-candidate-v1`, `phase-next-pro-q1-model-lifecycle-v1` and
`phase-next-pro-q1-serving-v1`. Failures, original sample vectors and earlier
identities are preserved; no partial worker is relabelled as passed.

Promoted-default executable SHA-256:
`c87739a6986b13de1240213ba0ed12075cea05b007ec6cc79c754e5042ed9f07`.
Its exact eight source/test hashes and the focused test binaries are in
`phase-next-pro-q1-promoted-v1/identity.json`; qualification and raw evidence links
are in `qualification.json`. These files bind product source revision
`5fcad5dcab817a7462b5c884243c9b168d4b2843` by content equivalence. The build used
the recorded previous commit plus a working delta; it is not claimed to be a
clean-HEAD build. The only post-screen production change is default 0->1;
additional NaN/short-context/default checks are test-only.
