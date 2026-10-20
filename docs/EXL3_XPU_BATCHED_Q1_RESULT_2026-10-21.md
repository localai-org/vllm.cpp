# Batched pure-Q1 attention on B70: qualification in progress

The candidate adds exact B2-B4 one-query-per-request admission to the existing
bounded Xe2 decode donor. `VT_XPU_XE2_BATCHED_Q1=1` opts in; the current compiled
default is `0`. Original B1 behavior remains. Q4, ragged and mixed packets retain
their existing routes. The experiment uses current request-stable auto/long2
with SiLU1 and mixed-prefill0; each new mechanism is isolated.

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
- Final same-process GPU guard set: four cases / 1,741 assertions pass. It covers
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
- Host policy: one case / 36 assertions pass. Graph policy: three cases / 249
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

## Remaining focused checks before a promotion decision

Cross-queue and cross-route scratch clobbering pass. Explicit NaN filling of
the whole shared reservation and newly batched one-split contexts at or below
960 tokens are still pending. These focused test additions will run after the
frozen serving workers finish; no compilation or second GPU workload will be
introduced into the scored runs. They are part of the existing correctness
contract, not a replacement for the original C4 serving evidence.

## Serving disposition: pending

The predeclared 16-worker original-fixture screen is running: two C4-4K A/A
workers, three alternating A/B pairs at C4-4K and C4-16K, and one C1-4K sentinel
pair. Both arms use the same frozen executable, auto/2, SiLU1, mixed-prefill0,
original stochastic HTTP payloads and 1,024-token caps. All profilers/traces are
off. Actual child flags, cold computed work, cache hits, preemptions, overlap
counter/window, TTFT/E2E, accepted/proposed drafts, outputs and host cgroups are
retained. A completed operator/lifecycle PASS alone does not promote the route.
No new full native/Python matrix has been started.

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
