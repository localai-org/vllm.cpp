# Codex implementation plan: B70 / Qwen3.8-27B EXL3 performance parity

**Prepared:** 15 October 2026  
**Scope:** native `vllm.cpp`, Intel B70 device 0, Qwen3.8-27B EXL3 4.00 bpw, compared with the existing installed Python vLLM reference.  
**Evidence base:** `B70_EXL3_DEEP_PERFORMANCE_REVIEW_bef524f0c_2026-10-15.zip`, native source head `bef524f0cb9df20089da2ed9cebce9773deecba4`.  
**Execution status:** this is an implementation and hardware-validation plan, not a claim that any optimization has already been implemented or benchmarked. Offline source/evidence checks are recorded separately in `B70_SOURCE_AUDIT_EN.md` and `B70_CAPSULE_VERIFICATION.json`.

## 1. Mission and first decision

Close the measured C++/Python gap on the existing hardware without changing model weights, activation precision, recurrent-state precision, KV precision, speculative depth, accepted-state semantics, or the benchmark workload. Start with the smallest repair that reaches the measured slow path. After every successful repair, remeasure and reprioritize the remaining gap.

**Recommended sequence:** establish a current control; fix the W8A8 preparation-capacity cliff using the existing direct-preparation architecture; optimize speculative GDN; qualify long-context C1 Xe2 attention; then select the largest remaining producer/fusion or large-M matmul opportunity. Investigate row-chunked preparation as a bounded alternative, not as a prerequisite for shipping a simpler successful fix.

The 64 MiB finding is a **private preparation-capacity limit that triggers a checked fallback**, not evidence of an out-of-bounds memory write or a limit on total prefill memory. There are **two separate 64 MiB admission checks**, so changing the backend allocator alone is incomplete. The existing model-private direct-preparation route is already enabled by default when eligible; extend its eligibility rather than introducing a duplicate architecture. [E3–E5]

## 2. Evidence and performance contract

### 2.1 Reproduce this comparison, not a nearby benchmark

| Property | Required control |
|---|---|
| Model | Existing Qwen3.8-27B EXL3 4.00 bpw checkpoint; revision `113cf7ab958054860e43fb7f3063b1af19171095` |
| GPU | B70, device 0 only; one inference worker at a time |
| Request | C1 raw completion, exactly 4,096 prompt IDs and 256 emitted tokens |
| Sampling | Greedy, `ignore_eos=true`, `logprobs=0`; preserve the recorded HTTP request semantics |
| Arithmetic | FP16 activations, FP32 recurrent state, FP8 E4M3 KV |
| Modes | Target-only/MTP0 and MTP3; unchanged compact draft-vocabulary bytes |
| Serving | Prefix caching off; batch budget 4,096; configured context limit 262,144 |
| Native KV | Physical page 1,600; 180 blocks in the recorded control |
| Reference KV | Effective page 1,600; recorded reference block counts differ by mode; retain and report them |
| Container resources | 2 CPUs, 24 GiB host-memory limit, no swap, as in the receipts |
| Score | Trace-free HTTP latency and actual emitted-token throughput |

Resolve paths and permissions from the live checkout and its recorded configuration. Paths embedded in the archive identify the historical machine; they are not permission to operate on an unrelated host. The 24 GiB container limit is not the GPU's VRAM budget. Do not change either budget merely to make a test pass. [E1–E2]

### 2.2 Historical score to reproduce

| Mode | Native prefill | Python prefill | Native post-first-token rate | Python rate | Native total | Python total |
|---|---:|---:|---:|---:|---:|---:|
| Target-only | 1.984 s | 1.462 s | 25.72 tok/s | 29.91 tok/s | 11.898 s | 9.992 s |
| MTP3 | 2.055 s | 1.472 s | 58.76 tok/s | 68.67 tok/s | 6.394 s | 5.190 s |

These are medians of the last three of four requests following the recorded primer, not high-confidence population estimates. All 16 recorded decoded texts match. Cross-engine raw output-ID, logit, or internal-state equality was not established by those HTTP results. The independent offline audit reproduced these medians and prompt/text comparisons. [E1–E2, E11]

Use `255 / post_first_token_decode_seconds` for the existing decode-rate metric. Do not substitute 256, proposed tokens, accepted draft tokens alone, or SSE chunk count. Preserve client total, server total, TTFT, prefill and decode as distinct metrics; their medians need not add exactly.

### 2.3 What the kernel evidence does and does not establish

| Opportunity | Measured evidence | Interpretation |
|---|---|---|
| W8A8 preparation | 64/65 separate row-validation calls; 237.657/240.792 ms in separate native prefill probes | Remove the capacity-induced fallback while preserving validation |
| Speculative GDN | 139.46 versus 16.14 µs/call; 3,888 versus 3,936 calls | Largest demonstrated MTP decode kernel disparity |
| Target-only attention | 251.43 versus 102.68 µs per forward-plus-reduction operation | Different long-context routes, not just launch overhead |
| Residual normalization | Native short-row kernel contributes about 225.5 ms in the approximate MTP decode window | Secondary opportunity; compare complete producer/consumer regions |
| EXL3 q4 GEMV/DPAS | Mean differences about 0.5%/1.9% | Do not rewrite these cores first |

The offline audit independently recomputed the selected kernel-family counts and sums from the complete raw PTI captures. Instrumented sums are prioritization evidence, **not additive predictions of trace-free latency savings**. The near-match for small-M GEMV/DPAS does not prove that large-M W8A8/oneDNN execution is already optimal. [E1, E6–E9, E11]

## 3. Operating procedure for every implementation wave

Read the live `AGENTS.md`, current ownership, local issue/spec, numerical contracts, and benchmarking guide before editing. The uploaded integration spec excluded performance redesign; this request starts a separately recorded optimization scope rather than silently rewriting historical integration claims. Follow the current repository's spec-before-code, issue, review and hardware-ownership requirements. Do not infer authority to push, merge, restart production, update drivers, or use other GPUs.

For each wave, use this sequence:

1. **Reconcile and specify.** Check whether the change already exists in current HEAD. Write the concrete hypothesis, affected dispatch, baseline, numerical contract, memory/lifetime contract and stop conditions before implementation.
2. **Add a detecting test.** Record the failing reachability or correctness regression at the old head. A timing assertion is not a substitute for proving that the intended path is selected.
3. **Implement one variable.** Keep a selectable old route and a narrowly admitted candidate. Avoid mixing arithmetic changes, allocation changes and scheduler changes in the same experiment.
4. **Prove correctness and reachability.** Run the focused CPU/GPU tests, real-input replay and route probe. Treat missing external fixtures, skipped GPU execution or exit 77 as unavailable, not PASS.
5. **Measure.** Use identical restored inputs/states for microbenchmarks, then trace-free serving. Keep profiling in a separate process/run.
6. **Keep or reject.** Save all results, including slower or numerically failing candidates. Revert rejected product changes. A fresh reviewer should mutate the key guarantee to prove the new test detects its violation, following repository policy.

Use new attempt directories; never overwrite archived receipts. Save immutable source/binary hashes, image digests, compiler flags, environment, exact argv, selected kernels, numerical results, memory counters, metrics and raw samples. Run GPU workers sequentially. Do not leave experiments running after their bounded test completes.

## 4. Wave 0 — establish a trustworthy current control

### Actions

Check current HEAD against the uploaded head and inspect the relevant diff. Preserve the measured binary and build a current baseline in the existing pinned builder. Confirm Release, `-O3`, `-DNDEBUG`, XPU, oneDNN and Xe2 GDN settings; inspect per-translation-unit floating-point flags rather than guessing them from an unrelated extension.

Keep the installed performance reference fixed: recorded Python vLLM 0.31.0, Torch 2.14.0+xpu, oneDNN 3.13.4; native oneDNN 3.13.0 and oneAPI 2026.1.1. Record actual live identities if any have changed and label that as a new baseline. Do not upgrade either stack during this campaign. The captured GDN reference source is useful implementation guidance, but it is not a proven byte-for-byte build of the installed wheel. [E1–E2]

Run the exact four-arm comparison first. Reuse `scripts/mm/measure_native_exl3_text_regression.py` and the recorded container commands, adapting only resolved paths, ports, binary mount and attempt names. Its `--baseline` comparison is appropriate for **same-engine native control versus native candidate**; do not force cross-engine public token-string spellings to act as raw-ID parity.

Record selected routes in a separate diagnostic request: direct preparation, checked preparation versus fallback, typed speculative GDN, attention mode, graph usage, actual verifier cycles, active row counts and draft proposals. Preserve already-enabled optimizations including model-map paths, typed SLM, FP16 SiLU tables and the existing verification route.

### Exit gate

A current baseline exists with the fixed workload, passing behavioral checks, no competing GPU process, no profiler in scored runs, and all configuration differences explained. If performance differs materially from the historical result, diagnose identity/route/environment changes before attributing a new candidate's improvement to its code.

## 5. Wave 1A — remove the W8A8 capacity cliff with the existing direct path

### 5.1 Exact mechanism and files

Inspect these specific sites at the uploaded head:

- `src/vt/xpu/xpu_backend.cpp:980–984`: `WithExl3W8A8Preparation` refuses requests above 64 MiB.
- `src/vt/xpu/xpu_exl3_w8a8.cpp:190–218`: the **model-private direct route independently requires** `plan.intermediate_offset <= 64 MiB`.
- `src/vt/xpu/xpu_exl3_w8a8.cpp:298–358`: checked preparation, validation, publication and legacy fallback.
- `src/vt/exl3_grouped.cpp:27–60`: `PlanExl3W8A8`, 256-row padding and 64-byte region alignment.
- `src/vt/xpu/xpu_common.h`: preparation-pool contract and shared declarations.

For the measured down-projection, `K=17408` and one scale group:

```text
padded_M = round_up(M, 256)
qbytes   = groups * padded_M * K
sbytes   = groups * padded_M * sizeof(float)

M=4096:
qbytes = 71,303,168 bytes
sbytes =     16,384 bytes
required = 71,319,552 bytes = 68.015625 MiB
64 MiB  = 67,108,864 bytes
shortfall = 4,210,688 bytes = 4.015625 MiB
```

**68 MiB is still too small.** Also, `M=3841` already pads to 4096 and encounters this cliff; it is not exclusive to a prompt of exactly 4096 tokens. At this shape, M=3840 needs 63.7646484375 MiB. K has 136 Hadamard blocks, within the existing checked producer's 144-block support. The obstruction is capacity, not unsupported K. [E3–E5, E11]

### 5.2 Implementation

Centralize the preparation policy so the backend lease and the direct-route predicate cannot disagree. Compute the aligned required layout once with checked size arithmetic and use that result at both sites. Keep the policy separate from the actual allocation size and the overall device-memory budget.

**First candidate: a 72 MiB maximum, allocated on demand.** This is an experiment default, not an instruction to reserve 72 MiB per layer or preallocate it unconditionally. The measured shape requests approximately 68.016 MiB; allocation must still fit the accounted device budget. An exact shape-derived limit is also valid, but a 72 MiB experimental cap avoids another near-boundary mistake while remaining bounded.

For A/B testing, a small internal cap override may be added, but it must be explicitly documented as new—not confused with existing flags. Validate its value, keep a hard upper bound, log the effective policy outside the hot path, and retain a reproducible 64 MiB control. Prefer one shared accessor over independent environment parsing in allocation and dispatch.

Preserve the **existing model-private direct route**: the private checked INT8 activation/scales buffer is consumed directly by oneDNN, and the main W8A8 workspace omits its duplicate activation region. Do not accidentally route the candidate through copy-based public preparation instead. Preserve the caller-owned public overload's observable workspace and failure behavior.

Do not remove `ValidateRows` as a function. On eligible calls, replace its separate work through the already checked producer; on unsupported K, insufficient budget or ineligible layouts, retain the safe fallback. Keep finite-input, finite `suh`, finite `svh`, FP16 product-overflow and transformed-row-overflow checks and the current error priority.

### 5.3 Ownership and memory requirements

Keep the current workspace-before-preparation lock order. The preparation lease must remain live through validation, all oneDNN consumers and output Hadamard. Keep eager-only admission and completion/exception retirement. This wave does not introduce asynchronous leases or graph capture of these buffers.

Preserve accounting for allocation growth, graph bytes and peak/live device memory. Distinguish policy refusal from real device allocation failure. Test that each has a safe, documented outcome; never fall back after partially publishing output. Guard capacity arithmetic and remaining-budget arithmetic against overflow/underflow.

Measure memory in **fresh worker processes**. Both pools retain their high-water allocations, so toggling routes inside one process can falsely hide the compact route's memory benefit. A larger preparation cap can reduce duplicate live storage via the direct route, but the final whole-model high-water change depends on other projections. Do not predict it from the cap alone.

### 5.4 Minimal A/B matrix

| Arm | Preparation cap | `VT_XPU_W8A8_PREPARE` | `VT_XPU_W8A8_DIRECT_PREPARE` | Purpose |
|---|---:|---:|---:|---|
| B0 | 64 MiB | 1 | 1 | Current production-style control |
| A-copy | 72 MiB | 1 | 0 | Isolate checked preparation without direct consumption |
| A-direct | 72 MiB | 1 | 1 | Intended candidate: checked preparation plus compact direct consumption |
| Legacy diagnostic | Existing policy | 0 | 0 | Explicit separate-validation/quantization numerical control |

Keep the existing 1024-column model-panel setting unchanged. A matrix label with `N=1024` can denote one panel, not the full projection width. Obtain full N and source-group mapping from the real descriptor/checkpoint.

### 5.5 Tests and acceptance

Extend `test_xpu_exl3_direct_prepare.cpp`, `test_xpu_exl3_w8a8.cpp`, `test_xpu_w8a8_workspaces.cpp`, and planner tests. The current direct-preparation complete-call test admits only M=129/896/1600; it is not proof for the measured M4096 down-projection. Add a real M4096/K17408 fixture/replay and route assertion. Capture only the required local fixture, with provenance, rather than downloading a new model.

Test M=3839, 3840, 3841, 4095 and 4096, plus one smaller existing case. Test one multi-group layout and a still-unsupported K route. Verify exact INT8 preparation, FP32 row-scale bytes, padding, public workspace where observable, oneDNN FP16 intermediate and final output against the existing qualified route/oracle. Retain zero-row scale=1 and all-zero padded activation.

Negative tests must place bad values in the **last active row and last scale group**, not just element zero. Cover NaN, positive/negative infinity, product overflow around the established 65520 FP16 rounding boundary, transform overflow with finite products, invalid output scale, invalid map, aliases, and failure followed by a valid call. Verify public output/scratch/panel publication semantics remain unchanged, including documented status-word exceptions.

Exercise growth, same-queue reuse, cross-queue reuse, exceptions, queue destruction, output retirement and forced memory pressure. Mutate only the backend cap, then only the direct predicate, to confirm the reachability test detects either incomplete repair.

**Accept when:** the target shape demonstrably selects checked direct preparation; the capacity-induced 64/65 separate row validations disappear from the corresponding qualified calls; complete operator time improves; numerical and ownership tests pass; device memory stays bounded; and trace-free prefill/total latency improves without a reproducible regression in the other mode. Do not demand that the entire 238–241 ms diagnostic sum disappear from serving latency.

## 6. Wave 1B — evaluate a genuinely bounded row-chunk alternative

Perform the design/accounting comparison after A-direct. Build a chunked prototype only when the cap increase has a material memory cost, the direct route cannot be admitted under the real budget, or evidence suggests tiling can improve complete execution. Do not spend the entire campaign rebuilding this path if A-direct is already the best tradeoff.

### 6.1 Candidate tile sizes

For one group at K17408 with current 256-row padding, the largest preparation tile within 64 MiB is **3840 rows**. The full case can use 3840+256 rows. A balanced 2048+2048 split uses 34.0078125 MiB of activation/scales per tile and is a second bounded candidate. Calculate sizes from the actual group count and aligned layout; do not hardcode these values for all projections.

Split **M/rows**, not K. Splitting K changes full-row maxima/scales and may split Hadamard transforms or alter quantization semantics. Preserve source-group indices, row offsets, output strides, padded rows and the full original large-M arithmetic route.

For an M3841 request, a 3840-row tile leaves one real row. Do **not** send that tail through the normal small-M GEMV/DPAS dispatcher merely because it is short. It must retain the qualified W8A8 transform/rounding semantics, with appropriate tile padding.

### 6.2 Choose an explicit transaction model

A loop that prepares chunk 0, copies it into caller-visible scratch/output, and only then discovers an invalid value in the final chunk is incorrect. It violates the existing no-publication-on-validation-failure contract.

Evaluate two honest designs:

**Public-layout-preserving two-pass design.** Complete whole-input validation before writing published activation/scales, then prepare chunks into their final destination. This can bound private staging, but retains a full public activation buffer and extra work. Count both retained storage and recomputation; do not advertise it as a one-pass replacement. It only wins if a better validation organization outweighs that cost.

**Model-private tiled consumption.** For each row tile, run checked preparation, consume it through oneDNN into a private FP16 Y buffer, then reuse the tile only after all its readers retire. Keep all externally visible output unchanged until every required tile and scale validation succeeds; only then run the final output stage/publication. Private partially computed Y is discarded or safely overwritten on failure. Keep the public caller-owned operator on its unchanged transactional path unless its full guarantees are separately implemented.

The private design avoids materializing full-M INT8 activation but requires a clear error/status protocol and full accounting for private Y, weight panels, oneDNN scratch and any extra output staging. It may repeat weight reconstruction or launch more matmuls. Compare row-outer and panel-outer loop ordering on paper before choosing one; avoid reconstructing every panel again without measuring the cost. No double-buffering until the single-buffer lease is correct and its bottleneck is demonstrated.

### 6.3 Acceptance and decision

Use A-direct and B0 as controls. Test last-tile validation failure, zero/tiny tail, padding poisoning, multi-group maps, repeated reuse and memory pressure. Verify the oneDNN implementation selected for tiled M does not silently change qualified FP16 results.

Report complete-operator latency, whole-prefill latency, preparation high-water bytes, total peak/live device bytes, reconstruction/quantization/GEMM counts and actual bytes copied. Select the simpler faster route when memory differences are negligible. Retain a chunked fallback only if it is correct and measurably useful. A lower preparation number with larger total retained memory is not a memory optimization.

## 7. Wave 2 — reduce speculative GDN recurrence cost

### 7.1 Establish the exact boundary before porting

Start in `src/vt/xpu/xpu_gdn.cpp:534–738`. The measured route is already `gdn_spec_decode_slm_typed`, with FP32 state, 32 KiB SLM, 64 value rows per workgroup and serial ascending-K arithmetic. Do not present enabling SLM or typed loads as a new fix. [E6]

The captured reference `reference-xpu-source/csrc/xpu/gdn_attn/gated_delta_rule.hpp:294–528` distributes keys across a subgroup and keeps a small state slice in registers. It also computes Q/K normalization and gating from raw producer inputs. These are different operator boundaries.

Write a contract table for the native seam:

| Input/output | Preserve |
|---|---|
| Q/K | Existing normalized FP16 inputs; no second normalization |
| V | Existing FP16 producer output |
| `g` | Existing FP32 log-decay input; recurrence computes `exp(g)` |
| `beta` | Existing derived FP32 beta; no second sigmoid |
| Recurrent state | FP32 with existing slot/head/value/key layout |
| Initial state | Slot selected by the actual accepted-token metadata |
| Each token | Required FP16 output and FP32 rollback snapshot |
| Mapping | Actual compact token range, output remap, valid/null state slots and accepted-prefix ownership |

Do not import the reference's defensive handling of invalid accepted counts if it conflicts with the native API's rejection contract. Distinguish rollback capacity from actual active token count.

### 7.2 Real-input replay and numerical gate

Capture one actual verifier invocation after the fixed prefill: normalized Q/K/V, g/beta, initial full FP32 state, offsets, slot IDs, accepted counts, output ownership and expected snapshots. Save before execution. Restore the identical input and entire relevant state before each candidate timing trial; do not let repeated timing iterations advance recurrence.

Use two oracles: the unchanged native operator for exact existing guarantees, and the approved pinned arithmetic/reference fixture for correctness beyond native-to-native agreement. The installed Python wheel is the performance control, not automatic permission to redefine numerical correctness.

Existing speculative SLM tests assert exact output and snapshot bytes. A subgroup reduction changes floating-point association. **Do not weaken these tests to make a port pass.** Record maximum/relative/RMS/ULP error, first mismatch, affected snapshot and continuation behavior, but treat those as diagnostics unless a pre-existing approved contract permits them. An arithmetic-changing route must remain experimental if it fails the current contract.

### 7.3 Implement two bounded paths, in order

**Candidate G1: data movement and arithmetic-order preservation.** Reduce avoidable state/Q/K traffic, redundant per-value producer work and synchronization while preserving the established multiply/add boundaries and ascending-K accumulation. Inspect generated register/scratch/SLM usage. Test an explicitly bounded value tile or state-loading arrangement against the typed-SLM control. Do not replace SLM with an unbounded private array and assume it stays in registers.

**Candidate G2: recurrence-only subgroup-distributed keys.** Adapt the reference dataflow to the existing normalized-input API rather than porting its fused raw-input interface. For the measured 128-key case, the reference's 32-lane/four-key organization is a concrete starting point. Keep a small FP32 state slice per lane, coalesced loads/stores, subgroup dot reductions, and every required per-token snapshot. Start with the captured reference workgroup/value tiling; vary at most one or two configurations based on register/spill evidence.

G2 requires its numerical contract to pass before it can replace G1/default. If exactness blocks a tree-reduction route, keep it as a rejected or opt-in experiment and continue the campaign with G1 and attention work; do not silently globally enable fast math or change persistent state to FP16.

After recurrence-only correctness/performance is established, consider a separate fused producer variant that combines normalization/gating with recurrence. That variant must consume the correct raw FP32/FP16 producer boundary and independently reproduce all required rounding. Do not combine this fusion with the initial recurrence rewrite.

Intel's SYCL guidance identifies register pressure, spills, subgroup geometry and SLM occupancy as interacting factors; inspect actual generated code instead of treating lower SLM or higher occupancy alone as success. [W2–W3]

### 7.4 State/lifetime tests

Extend the existing P5/P7, R07 and MTP tests in `tests/vt/test_xpu_gdn.cpp`, plus `tests/vllm/models/test_xpu_exl3_mtp.cpp` and the model routing tests as needed. Test the real C1/MTP3 shape first, then the existing narrow C4/variable-active-length contract cases—not a broad serving-concurrency campaign.

Cover each supported accepted prefix, rejection at each draft position, correction/bonus continuation, all-accepted verification, nonzero initial state, null slots, permuted slots, every intermediate snapshot, untouched slots, invalid offsets/counts/slot IDs, aliases, fresh metadata on graph replay, and graph retirement. Verify invalid metadata fails before protected state/output changes.

Replay several consecutive cycles using each candidate's own produced state. Compare a short target-only and MTP3 continuation with the same emitted-ID sequence and approved state expectations. A plausible first output with diverging later state is a failure.

### 7.5 Performance and exit gate

Measure isolated recurrence, complete GDN region, complete target verifier cycle and trace-free MTP3 serving. Normalize by actual calls/cycles and report proposed, accepted and emitted tokens. The original 81 versus 82 verifier cycles must not be mistaken for identical work.

The direction is toward the reference's approximately 16 µs recurrence observation; a 2× improvement is a useful engineering milestone, not a correctness threshold or promised outcome. Accept only a reproducible complete-region and serving improvement with unchanged required behavior. Reject local wins that add packing/copies or reduce draft acceptance enough to lose end-to-end performance.

## 8. Wave 3 — qualify a bounded long-context C1 Xe2 attention route

### 8.1 Extend policy, not just the length guard

Read `xpu_attention.cpp`, `xpu_attention_decode_xe2.cpp`, `xpu_attention_split.cpp`, `include/vt/paged_attn_route.h`, and `BuildXpuAttentionGraphPolicy` in `src/vllm/model_executor/models/qwen3_5.cpp:274–310`. Also inspect the already vendored `third_party/b70_xe2_verify/csrc/xpu/attn/xe_2/paged_decode.hpp`. [E7]

The C1 Xe2 wrapper currently admits only the short one-split bound of 960 tokens. At 4096+ it falls back to generic Split-K. The existing donor infrastructure includes split-KV and reduction support, but the short wrapper sets `num_kv_splits=1` and aliases temporary output to final output. Removing the guard without constructing correct multi-split storage and policy is not a fix.

Implement a separate bounded long-C1 policy first for the measured FP16 query/FP8 E4M3 KV shape: one query, 24 query heads, 4 KV heads, head dimension 256, supported 1600/1664 page layouts, existing scale/causal/stride/device/stack restrictions. Keep the short route and generic fallback intact.

Derive a small split-count candidate set from the installed/reference policy and actual active length. Allocate correctly aligned partial outputs, maxima and sums within the accounted attention workspace. Respect the existing 16 MiB budget or separately specify and qualify any change. Check all size products and tails. Keep partial buffers disjoint when split count exceeds one.

Use stable softmax combination of partial maxima/sums/outputs, including empty/masked split handling. Read dynamic device sequence lengths and fresh block tables; do not process reserved future pages as active tokens. Do not add per-token host readback to determine the split.

### 8.2 Graph invalidation is part of this implementation

The current graph policy tracks the short-route bound, 4096 split-policy threshold, environment and active-page count. Extend it for **every new host-baked launch choice**: route identity, split count, tile/length bucket, scratch layout and relevant address/residency bindings.

Choose a stable qualified upper bound inside a bucket and let kernels read current device lengths. Retire and safely recapture when the bucket/split/page policy changes. Do not recapture on every token merely because the exact sequence length changes. Conversely, do not keep replaying a graph whose baked capacity is no longer sufficient.

Initial performance qualification should cover the actual P4096/O256 active-length interval. That does not grant admission for the entire configured 262,144-token maximum. Outside the bounded route, preserve the generic path.

### 8.3 Focused tests

Extend `test_paged_attn_route.cpp`, `test_xpu_attention_fast.cpp`, and the existing model graph-policy tests. Compare complete attention output against the established qualified oracle with existing tolerances.

Test the relevant sides of 960/961, 4095/4096, page boundaries 4800/4801 for page1600 and 4992/4993 for page1664, plus every newly introduced split/tile transition. Include final partial tiles, permuted active pages, extra inactive block-table entries, poisoned inactive capacity, invalid page IDs, stale metadata, disallowed scale/layout fallback and graph replay across a transition. Mutate graph retirement to prove the boundary test detects stale launch policy.

Measure forward plus reduction plus packing/unpacking/validation, not the forward kernel alone. Benchmark target-only serving first, then MTP3 as the control. Target verification under MTP3 already uses Xe2; the target-only gap must not be counted as an equal MTP3 opportunity. Some matching C1 draft calls may change too; report those separately.

### 8.4 Bounded fallback experiment

If the long Xe2 route is numerically or structurally unsuitable, perform a small evidence-led generic Split-K experiment with existing `VT_XPU_ATTN_SPLIT_SPAN`, `VT_XPU_ATTN_SPLIT_MAX_PARTS` and reduction controls. Verify current active-page capping before changing it; it is already implemented. Limit configurations to the actual 4K shape and one nearby boundary case. Keep a complete-operation and graph-correctness gate; avoid a broad blind tile sweep.

## 9. Wave 4 — optimize the remaining measured producer region

Reprofile after Waves 1–3. Rebuild the residual gap table before choosing this wave. Candidate order below is provisional and must follow the new measurements.

### 9.1 Residual/RMSNorm

Inspect `src/vt/xpu/xpu_norm.cpp`, especially `Gemma5120ShortRowsKernel`. Typed FP16 pointers are already present. The implementation preserves a specific virtual-lane reduction tree and FP32 rounding boundaries. The model's use of this named kernel is not permission to replace it with a generic RMSNorm formula.

First compare the complete native and reference regions at real row counts, including residual add, normalization, any fused linear operations and materialization. Optimize synchronization, load reuse or lane mapping while preserving the same arithmetic graph. A later fusion may remove a memory round trip while explicitly retaining its rounding value.

Preserve the FP32 residual sum for normalization, the separate narrowed FP16 residual output, `1 + weight` semantics, reciprocal/epsilon placement, reduction order, alias behavior and graph replay. Test rows 1 and 4 and representative existing tails. A 3 µs reference fused kernel is not automatically equivalent to one 20 µs native kernel.

### 9.2 Activation, Q/K/RoPE and output transforms

Inspect SiLU/multiply, Q/K normalization/RoPE/gating, EXL3 input/output Hadamard and GDN post-convolution producers. Existing lookup tables and fused routes may already be selected. Prove the actual route and removable memory/launch cost before adding another variant.

Consider producer-consumer fusion that eliminates full temporary tensors, redundant casts or duplicate loads while preserving FP16/FP32 boundaries. Keep a round-to-FP16 step logically present even when its value stays in a register. Verify exact table semantics, gate ordering, output layout and downstream consumption on real inputs.

### 9.3 Large-M W8A8 and oneDNN

The separate native prefill probe reports about 753 ms in `onednn_exl3_w8a8_stream`; that is a significant region but not a matched proof of a library defect. Once preparation is fixed, compare real per-panel M/N/K/group shapes, primitive selection, output scaling and scratch against the installed reference.

A bounded next experiment may adjust the existing 128/1024/2048 panel policy, primitive reuse or redundant reconstruction only when source-group boundaries and memory limits permit it. Include input preparation, all panels, output Hadamard and lifetime fences in timing. Do not cache expanded copies of all model weights to buy speed without accounting for the resulting VRAM/context-capacity loss. Do not upgrade oneDNN or change global precision as part of a panel experiment.

## 10. Conditional backlog — pursue only after a concrete remaining bottleneck

| Approach | Evidence required before implementation | Main constraint |
|---|---|---|
| W8A8 asynchronous leases / bounded per-queue slots | Material exposed prefill host serialization after Wave 1 | Event-owned lifetime through every consumer; bounded slots and safe exceptions |
| Eliminate redundant already-completed waits | A specific duplicate wait with no intervening work | Prove the earlier fence covers every protected consumer; no blanket removal |
| Reduce graph-time allocation or metadata churn | Exposed critical-path cost, not inclusive waits on GPU work | Preserve live metadata validation, allocation generation and graph retirement |
| Improve draft logits/argmax or compact-vocabulary handling | Remaining meaningful MTP draft/acceptance cost | Preserve exact mapping, selected tokens and actual emitted-token accounting |
| Prefill convolution/GDN optimization | New post-repair per-shape evidence | Preserve FP32 convolution and approved FP16 producer boundaries |
| Enable another text-prefill attention backend | Full region evidence that existing prefill attention matters | Existing Xe2 verification/vision support does not qualify text-prefill automatically |
| Cache immutable scale/map/reconstruction metadata | Redundant repeated work with immutable owner identity | Versioned ownership; do not cache validation of dynamic activations |
| Queue overlap/double buffering | Demonstrated idle/overlap opportunity | No races in shared pools or state; actual latency must improve |
| Wider prompt/concurrency/MTP-depth tuning | Initial parity achieved and a new workload contract | Report as separate workload optimization, not fixed-workload parity |
| Compiler/driver/library comparison | Residual reproducible kernel issue and separate approval | New experiment pins; do not erase the original baseline |

The recorded approximately 48 ms staged-H2D host wait in MTP3 largely overlaps pending target GPU work; the actual copy-event sum is about 0.046 ms/cycle. Do not treat it as a 48 ms PCIe-transfer problem. The approximately 99% device-busy windows also do not prove bandwidth saturation or that any particular kernel is efficient. [E1, E9]

## 11. Hardware test and benchmark execution

### 11.1 Build and focused tests

Execute commands inside the resolved pinned build/test environment. Discover available tests before constructing a filtered run. The following is a template using existing target names; `XPU_BUILD` and artifact variables must point to the actual resolved environment.

```bash
set -euo pipefail
: "${XPU_BUILD:?Set the resolved XPU build directory}"
: "${EVIDENCE_DIR:?Set a new, non-overwriting evidence directory}"
mkdir -p "$EVIDENCE_DIR"

ctest --test-dir "$XPU_BUILD" -N > "$EVIDENCE_DIR/ctest-inventory.txt"

cmake --build "$XPU_BUILD" --parallel 2 --target \
  vllm-server test_xpu_exl3_w8a8 test_xpu_exl3_direct_prepare \
  test_xpu_w8a8_workspaces test_xpu_gdn test_xpu_attention_fast \
  test_paged_attn_route test_xpu_qwen_exl3_fp16 test_xpu_exl3_mtp
```

Run only the relevant wave's tests first. After focused green, run the repository-required impacted regression gate. The existing fixture contracts include `VT_B70_EXL3_W8A8_FIXTURE` and `VT_B70_EXL3_PANEL_ROWS`; extend the tests deliberately for new real row counts rather than setting unsupported values. Use `EXL3_REQUIRE_ARTIFACTS=1` where supported and explicitly audit skip results everywhere else. Memory-pressure tests need their existing isolated process/budget environment and must not share a warmed high-water pool.

Do not claim GPU kernel tests have run because the CPU-only planner tests pass. Do not call an empty CTest filter a successful gate.

### 11.2 Same-engine serving comparison

Launch control and candidate sequentially using the preserved server recipe. For each mode, the existing client can compare a new native result with its native baseline:

```bash
: "${SERVER_URL:?Set the current test server URL}"
: "${SPEC_DEPTH:?Set 0 or 3}"
: "${NATIVE_BASELINE_JSON:?Set the matching native control result}"
: "${CANDIDATE_JSON:?Set a new output path}"

python3 scripts/mm/measure_native_exl3_text_regression.py \
  --url "$SERVER_URL" --model Qwen3.8-27B \
  --spec-depth "$SPEC_DEPTH" \
  --baseline "$NATIVE_BASELINE_JSON" \
  --output "$CANDIDATE_JSON"
```

The existing client marks a greater-than-5% duration increase as a diagnostic trigger, not a universal parity test. Add an explicit campaign comparator rather than assuming that `status=PASS` means the performance goal was reached.

### 11.3 Sampling and acceptance

For an initial repair, preserve the historical primer/four-request protocol and retain every raw sample. For confirming a winner, repeat at least three balanced baseline/candidate blocks, using fresh workers for memory comparisons. Re-run both Python modes in the same measurement campaign. Counterbalance order to expose thermal/frequency drift. Log clocks, power/temperature and CPU throttling without changing their settings.

Use medians and dispersion. Report uncertainty from independent run blocks; do not manufacture narrow confidence intervals by treating dependent tokens or repeated samples in the same state trajectory as independent. A small difference inside observed noise is inconclusive; collect another bounded block rather than declaring a win.

Predeclare a non-regression tolerance for confirmation, for example 3% on unaffected metrics with supporting uncertainty. This is an engineering tolerance, not a claim that 3% slower equals identical speed.

For each mode define:

```text
latency_ratio    = native_candidate_latency / contemporaneous_python_latency
throughput_ratio = native_candidate_decode_rate / contemporaneous_python_decode_rate
```

The objective is latency ratios no greater than 1 and throughput ratios at least 1 for prefill/decode/total on the fixed workload. Label results within a 3% band as **near parity**, not strict parity. Report per-mode and per-phase results; a win in MTP3 total does not hide a target-only or prefill regression.

### 11.4 Profiling discipline

Scored runs must have PTI, Torch profiling, native device/host tracing and legacy graph-tracing debug controls disabled. Use the recorded graph-coverage controls only in separate diagnostics when needed by the installed stack. First verify capture completeness with replayed-kernel counts and valid timestamps.

The initial native trace missed graph replay; the complete native trace exposed 384,640 target-only kernels instead of 11,830. Python MTP3 wall time was heavily perturbed by detailed tracing. Neither instrumented wall time nor apparent idle gaps in those captures should rank serving engines. [E1, E9]

Never add nested graph/queue spans to their constituent kernels, or inclusive host waits to GPU time. Keep approximate PTI prefill boundaries distinct from exact native step attribution. Device-wide ComputeBasic occupancy/stall averages are not per-kernel attribution. Use actual register/spill/SLM information and matched kernel timestamps to explain a candidate, then return to trace-free scoring.

## 12. Completion report and decision rules

For every wave produce a short report containing the hypothesis, immutable head, touched files, exact tests, selected path, numerical/state result, memory result, before/after complete-operator timings, four-arm serving comparison, rejected variants and remaining limitations.

Maintain this campaign table with actual measurements, leaving missing values as `NOT RUN`:

| Head / variant | MTP0 prefill | MTP0 decode tok/s | MTP0 total | MTP3 prefill | MTP3 decode tok/s | MTP3 total | Peak device bytes | Correctness / state | Route reached |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Current control | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Wave 1 selected | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Wave 2 selected | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Wave 3 selected | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Final candidate | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

Reject a candidate on any unapproved numerical change, accepted-state error, stale graph policy, partial publication after validation failure, unbounded retained memory or repeatable serving regression. Keep the prior successful head and proceed to the next justified hypothesis. A profiler-only improvement is not sufficient.

After initial C1 parity or a demonstrably strong repair, add only the smallest nearby cases needed to prove the new route's scope: preparation padding boundary, one attention page/split transition, repeated prompt/state reuse, and the already affected model/lifecycle regressions. Existing vision correctness tests may need rerunning when shared code changes; restarting vision implementation or demanding a new broad vision-performance matrix is not part of this campaign.

The campaign ends with a truthful scope statement: strict fixed-workload parity, near parity, or quantified remaining gaps. Do not generalize P4096/C1 results to the full context limit, arbitrary prompts or concurrent serving without separate evidence.

## 13. Evidence index

Paths below are relative to the original extracted capsule unless stated otherwise. Line numbers refer to the uploaded head and must be re-resolved on current HEAD.

**[E1]** `REPORT_EN.md`, sections 1–8; `SUMMARY.json`, keys `baseline`, `kernel_comparisons`, `prefill_validation`, `native_spans`, `hardware`, `limits`.

**[E2]** `evidence/pr-main-performance-2026-10-15/`: `comparison-contract.json`; successful native MTP0/MTP3 v1 and Python MTP0 v3/MTP3 v1 result files; matching command/exit receipts; checkpoint/image/build identity files. Exact performance-reference identities are in the capsule report and these receipts.

**[E3]** `source/vllm.cpp/src/vt/xpu/xpu_backend.cpp:930–984`; `source/vllm.cpp/src/vt/xpu/xpu_common.h`: bounded eager workspace ownership and preparation cap.

**[E4]** `source/vllm.cpp/src/vt/xpu/xpu_exl3_w8a8.cpp:168–225, 275–358, 371–401`: direct admission, compact layout, validation, preparation, publication and consumers.

**[E5]** `source/vllm.cpp/src/vt/exl3_grouped.cpp:27–60`; `include/vt/exl3_grouped.h`; direct-preparation, W8A8 and workspace tests under `tests/vt/`.

**[E6]** `source/vllm.cpp/src/vt/xpu/xpu_gdn.cpp:534–738`; `tests/vt/test_xpu_gdn.cpp:1260–1460`; captured reference `evidence/pr-performance-diagnosis-2026-10-15/reference-xpu-source/csrc/xpu/gdn_attn/gated_delta_rule.hpp:294–528`.

**[E7]** `source/vllm.cpp/include/vt/paged_attn_route.h:28–54`; `src/vt/xpu/xpu_attention_decode_xe2.cpp:19–95`; `xpu_attention_split.cpp:11–95`; `src/vllm/model_executor/models/qwen3_5.cpp:274–310`; vendored `third_party/b70_xe2_verify/csrc/xpu/attn/xe_2/paged_decode.hpp`.

**[E8]** `source/vllm.cpp/src/vt/xpu/xpu_norm.cpp:7–108, 209–225` and existing producer tests; summary/PTI kernel attribution for normalization.

**[E9]** Successful complete PTI directories `native-mtp0-flatpti-v1`, `native-mtp3-flatpti-v1`, `python-mtp0-flatpti-v2`, `python-mtp3-flatpti-v2`; separate native `*-spans-v1/result.json` and hardware metric captures under `evidence/pr-performance-diagnosis-2026-10-15/`.

**[E10]** `source/vllm.cpp/AGENTS.md`; `.agents/specs/backend-xpu-exl3-serving.md`; `scripts/mm/measure_native_exl3_text_regression.py`; `tests/CMakeLists.txt`. Live repository instructions take precedence over this archived snapshot.

**[E11]** Companion `B70_CAPSULE_VERIFICATION.json` and `verify_b70_capsule.py`: read-only manifest verification, raw selected-family PTI recount, baseline-median recomputation, prompt/text comparison and byte/padding calculations performed for this review. The four supplied Python analytics unit tests were also rerun and passed. These are not hardware tests.

**External context, checked 15 October 2026; not replacements for installed pins:**

**[W1]** Intel PTI SDK documentation, repository `intel/pti-gpu`, `sdk/README.md`, and Intel PTI API documentation. The current README was not sufficient to independently verify the legacy graph environment switches; their successful measured use is supported by the archived receipts, not inferred from current web documentation.

**[W2]** Intel oneAPI GPU Optimization Guide, *Registers and Performance*, 2025.2 edition: register pressure, non-register promotion and spills can affect subgroup execution and memory traffic.

**[W3]** Intel oneAPI GPU Optimization Guide, *Thread Mapping and GPU Occupancy*, 2025.2 edition: workgroup/subgroup geometry and SLM allocation interact; maximum occupancy is not a universal latency objective.

**[W4]** Upstream `vllm-project/vllm-xpu-kernels`, `csrc/xpu/gdn_attn/gated_delta_rule.hpp`: subgroup-distributed speculative recurrence dataflow. Use the captured pinned source for the implementation comparison and preserve its license notices.
