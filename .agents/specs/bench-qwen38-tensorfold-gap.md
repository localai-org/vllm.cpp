# Close the Qwen3.8 Flash Next speed gap against TensorFold

| Field | Value |
|---|---|
| Issue | `ISSUE-LOCAL-01M3PWSWVEQ5J1GABVEPFX9HQK` |
| Ownership | `BENCH-QWEN38-TENSORFOLD-GAP` owns `ISSUE-LOCAL-01M3PWSWVEQ5J1GABVEPFX9HQK` |
| Subject | vllm.cpp `qwen4_exp` GGUF path, initially `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S; TensorFold `qwen4_exp` CUDA path on `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` |
| Host | `dgx:gpu0`, one GB10 with 128 GB unified memory, through an `rc` lease |
| Branch | `row/BENCH-QWEN38-TENSORFOLD-GAP`, base `d15b1cc095694df69d8014ea6597582882e1bc1c` |
| Integration | One pull request. The spec commit precedes implementation commits. |
| Status | `BLOCKED_MISSING_ARTIFACTS`; the 2026-09-29 leased discovery produced no benchmark number |

## Scope

Measure TensorFold and vllm.cpp on the same DGX Spark, identify the work that
explains the gap, and port the useful mechanisms into vllm.cpp. Optimize the
existing supported GGUF arm first. Do not make MLX checkpoint support a hidden
prerequisite. If the checkpoint format or quantization explains a material part
of the measured gap, record that result and scope a separate arm.

The campaign covers:

- a pinned TensorFold comparator and a pinned MiaAI deployment recipe;
- one harness and one corpus for both servers;
- correctness checks before performance ratios;
- separate prefill, serial decode, drafted decode, and concurrency results;
- profiles that attribute time before code changes;
- incremental QSA state, native MTP, long-context selection, PLE staging,
  prefill geometry, and verified copy drafting when measurements justify each;
- telemetry needed to explain a result rather than report only a rate.

The campaign does not copy TensorFold code. TensorFold is Apache-2.0, but the
implementation is evidence and design input. New product code follows this
repository's APIs, cache ownership, scheduler, and test conventions.

## Success criteria

The campaign succeeds only when all of these statements are true:

1. Both engines have reproducible runs on one leased `dgx:gpu0` with pinned
   revisions, binary or image identity, artifact identity, flags, clock windows,
   and raw results committed.
2. Each published ratio names artifact, quantization, KV dtype, context window,
   draft mode, modality, concurrency, prompt tokens, output tokens, and timing
   domain. Unlike artifacts are labeled rather than called matched.
3. A profile identifies the dominant prefill and decode costs in vllm.cpp.
4. Every optimization lands behind a correctness gate and has a runtime-selected
   same-binary A/B result before acceptance. An adjacent-commit result is
   exploratory evidence only and remains pending. A change without a measured
   win is removed or retained only as an explicitly justified prerequisite.
5. The final record reports both engines' absolute results and the mechanisms
   that improve vllm.cpp. It reports a TensorFold/vllm.cpp fraction only after a
   common artifact, quantization, token sequence, KV dtype, and draft policy
   exist. Otherwise it uses `PROFILE_COMPARISON` and makes no cross-engine ratio.

There is no invented speed floor before the baseline runs. After W1, W2 records
a numerical target from the measured gap. The target cannot be lower than:

- no regression beyond 3% in short prefill, long prefill, serial decode, or
  concurrency throughput; and
- at least 10% improvement in one declared bottleneck before this campaign may
  claim a performance win.

The developer's direction is stronger than this minimum: bring the applicable
TensorFold mechanisms here and close as much of the measured gap as possible.

## Oracles

### Primary and executable behavior oracles

Pinned vLLM remains the primary behavior oracle wherever it implements
`Qwen4ExpForConditionalGeneration`. On this fleet, the active vLLM pin cannot
run Qwen4-Exp, so it is a named `BLOCKED` mandatory denominator, not a silently
replaced one. The lane-pinned Transformers fixtures remain the executable
component oracle. The runnable `llama-cpp-qwen4exp` pin is the same-GGUF
end-to-end comparator, subject to its recorded fidelity limitation; W1 must run
its existing decode proof and stop performance acceptance if the declared
component and token gates are not established. Existing Qwen4-Exp fixtures and
parity records continue to define architecture behavior. TensorFold does not
overrule vLLM or Transformers on logits, state transitions, cache semantics, or
multimodal behavior.

### TensorFold implementation comparator

TensorFold remains an external implementation and performance comparator. It
is not a secondary correctness oracle and therefore does not enter the
`AGENTS.md` secondary-oracle registry. W0 uses the dedicated `.agents/comparators/tensorfold.md` record because the
existing oracle registry admits only correctness oracles. The comparator checker
pins both repositories and requires the explicit `not-an-oracle` classification. No policy wording may imply that TensorFold supplies
correct output.

Pin these two independent objects:

- `ashhart/TensorFold` commit
  `191188075bca56a7c71074a79375eb4c1cb22e1c`, tag `v0.3.6.3`;
- `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold` commit
  `856bb6be4b58ce6a6727e6d071fb1c52f3f80e6e`.

The first object owns the CUDA family implementation. The second owns the eight
runtime patches, image recipe, flags, and publisher claims. The comparator record
must name both because neither object alone reconstructs the reported engine.

TensorFold may answer:

- how its QSA compressor state, sparse block selection, PLE staging, and MTP
  verification are organized;
- whether its pinned build runs on GB10;
- its throughput under this campaign's harness;
- whether a vllm.cpp change narrows a measured performance gap.

TensorFold may not answer:

- exact behavior when pinned vLLM implements the path;
- a matched-artifact ratio against GGUF, because its default model is MLX 4-bit;
- historical byte-identity claims for which the recipe commits no outputs or
  hashes;
- the published concurrency table unless this campaign reproduces it.

## Baseline facts, not results

The MiaAI recipe reports, but this campaign has not verified:

- 62.4 token/s for one prose request;
- 119.3 aggregate token/s at concurrency five;
- about 2.2 to 2.5k prefill token/s from 8k through 131k tokens;
- int8 KV, five 262,144-token stream pools, six MTP drafts at confidence 0.60;
- a 29.8 GiB PLE table on SSD.

Its checked-in benchmark does not generate the README concurrency table, does
not generate the exact 8k-to-128k prefill table, and commits no raw logs. These
figures are publisher claims until W1 records local evidence.

## Design

### D1 — One canonical corpus, one client harness, two adapters

The canonical corpus stores UTF-8 request payloads, not an undefined token-ID
transport. It pins tokenizer repository revision, tokenizer-file hashes, chat
template, special-token policy, and the canonical token-ID fingerprint produced
before either server starts. Each adapter sends the same text/messages and
records the server's consumed token IDs when the endpoint exposes them. Without
consumed IDs, it records the server tokenizer identity and input-token count.
Equal counts alone do not prove a matched token sequence, so those legs are
`PROFILE_COMPARISON` unless both token-ID fingerprints are available and equal.
Engine adapters may translate only capabilities and telemetry. They may not
change prompts, requested output length, warmups, concurrency, sampling, or
`ignore_eos`.

Every run stores:

- canonical and consumed token-sequence fingerprints, tokenizer identity, and
  prompt token count;
- generated token count excluding prompt tokens;
- wall TTFT, TPOT, ITL, E2E latency, request rate, input/output/total token
  throughput, each with mean, P50, P90, and P99 where the sample count permits;
- steady-state decode rate and peak host/device/unified memory;
- server-reported prefill and decode timers when available, labeled as a
  different timing domain;
- cached-token count;
- draft rounds, proposed tokens, accepted tokens, and acceptance rate;
- per-request errors and full deterministic outputs for correctness legs.

The harness refuses a performance ratio when prompt-token counts differ. It may
still store two absolute results with `NOT_COMPARABLE_TOKENIZATION`.

### D2 — Four workload families

1. **Serial decode:** 1k input, 256 output, concurrency one, greedy,
   `temperature=0`, seed 0, `ignore_eos=true`, drafts off.
2. **Drafted decode:** the same corpus, parameters, and dimensions with each
   engine's native verified draft path enabled. Record draft depth and acceptance.
3. **Prefill ladder:** 1k, 8k, 32k, 64k, 128k input, 1 output, concurrency one,
   seed 0 and `ignore_eos=true`.
4. **Serving ladder:** 1k input, 128 output, concurrency 1, 2, 4, and 5,
   greedy, seed 0, and `ignore_eos=true`. Each timed repetition is closed-loop
   and contains at least `5 * concurrency` requests in at least five waves.

Use at least one code prompt and one prose prompt for decode. Add a repeated-text
corpus only for copy-draft evaluation; never mix it into the ordinary decode
median. Before interpreting deltas, measure five repeated baseline legs to set a
noise band. Then use order-alternated paired A/B legs and require a majority of
at least five valid pairs outside that band. Long 128k legs use the same rule;
if five valid pairs are infeasible, record absolute observations only and make
no comparative claim.

### D3 — Honest artifact matrix

The first matrix has two labeled arms:

| Arm | Artifact | Format | Purpose |
|---|---|---|---|
| vllm.cpp | staged `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S, exact existing pin and sha256 | GGUF IQ1_S | supported product baseline |
| TensorFold | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`, revision and file hashes captured before run | MLX affine 4-bit, group 32 | published TensorFold profile |

This is an engine-profile comparison, not a matched-quantization comparison.
No cross-engine ratio, speedup, fraction, winner, or residual gap is valid from
these two arms. Record model bytes, resident bytes, KV dtype, and tensor
placement. If a common artifact becomes runnable in both engines, add a separate
matched arm; do not rewrite the first matrix.

A third required row records pinned production vLLM as `BLOCKED` on this fleet,
with the active oracle's exact reason. It remains the mandatory production
reference even though it cannot produce a number. The same-GGUF llama.cpp arm
is the executable end-to-end correctness comparator, not a replacement for
production vLLM.

### D4 — QSA compressed state is incremental

The first implementation candidate is an append-only compressed-key side cache
per request and QSA layer. A completed compression block is pooled, normalized,
rotated, and stored exactly once when `(position + 1) % compress_ratio == 0`.
Decode scores cached complete blocks plus the current incomplete tail. Prefix
reuse, fork, rollback, eviction, and request teardown apply the same ownership
operations to this side cache as to its corresponding KV sequence.

Required invariants:

- the cached path is numerically equivalent to recomputing all complete blocks;
- one new decode token causes zero compressor writes except at a block boundary;
- a block-boundary token causes exactly one new state per QSA layer;
- rollback removes states beyond the restored token position;
- no request can observe another request's QSA state;
- cache bytes appear in admission and memory accounting.

Do not implement this as a process-global map keyed by request address.

### D5 — Native MTP is a verified speculative path

Implement the checkpoint's native MTP head through the existing speculative
scheduler seams only after W1 identifies a vllm.cpp-loadable artifact that
contains the MTP tensors. The current UD-IQ1_S GGUF manifest contains no
MTP-named tensors, so W1 must inspect candidate GGUF/safetensors artifacts,
record tensor names and hashes, and either define a separate conversion/load
work item or mark W3 `BLOCKED_NO_MTP_WEIGHTS`. MLX support is not smuggled into
this campaign. The target behavior, once weights exist, is exact verification:
drafted tokens are accepted only when the main model confirms them under the
selected sampling policy. Serial decode remains available as the correctness
and raw-throughput control.

Separate these numbers:

- main-model rounds per generated token;
- drafted tokens per round;
- accepted tokens per round;
- acceptance rate;
- user-visible generated token/s.

Start with draft depths 1, 2, 4, and 6. Select a default only from the measured
throughput curve and memory impact. TensorFold's six drafts and 0.60 confidence
are inputs to the sweep, not inherited defaults.

### D6 — Long-context QSA selection is profiled before replacement

At 64k and 128k context, capture kernel duration, registers, spills, occupancy,
and temporary bytes for score and top-k selection. Port the tiled selection
shape only if selection is a material bottleneck or spills. Preserve deterministic
tie handling: equal cutoff scores select lower block IDs first.

The candidate algorithm may use bounded tiles and radix or histogram selection,
but the spec does not prescribe TensorFold's four-pass implementation. The gate
is selected-block equivalence and measured improvement.

### D7 — PLE movement follows placement

First record where PLE weights reside and how much time each request spends in
lookup, transfer, and compute. Three outcomes are possible:

- **device resident:** optimize gather or layout; SSD work is irrelevant;
- **host resident:** overlap pinned-host staging with GPU work and deduplicate
  rows within a chunk;
- **SSD resident:** add bounded read-ahead, persistent native workers, duplicate
  elimination, and adjacent-read coalescing.

Read-ahead has a one-chunk bound. It must cancel on request failure and must not
change lookup ordering. Any SSD result records device model, filesystem, mount
options, cache state, and whether pages were warm or cold.

### D8 — Prefill rows are a measured memory trade

Expose no user default until the baseline profiles 1,024, 2,048, and 4,096-row
chunks. Admission must include all chunk-dependent buffers. A winning chunk size
must improve the declared prefill ladder without reducing the configured
context or concurrency pool.

### D9 — Copy drafts are optional and isolated

A prompt-copy drafter may propose the sequence following a prior match of the
last eight tokens. The main model verifies every proposal. It is off by default
until the repeated-text corpus shows a win and ordinary code/prose shows no
regression beyond 3%. This work follows native MTP and does not block it.

## Source anchors and local seams

The implementation wave must cite exact lines again at its own pinned SHAs. The
initial symbol anchors are:

- QSA selection: TensorFold
  `src/tensorfold/families/qwen4_exp/cuda/attention.py:qsa_select` and `_select`;
  local `src/vllm/model_executor/models/qwen4_exp_qsa_block.cpp` at the
  `Qwen4ExpQsaCompress` call, with CUDA implementation in
  `src/vt/cuda/cuda_qwen4_exp_qsa.cu:Qwen4ExpQsaCompressKernelCuda`.
- QSA sequence ownership: TensorFold
  `src/tensorfold/families/qwen4_exp/cuda/kvcache.py` and `decode.py:State`;
  local `src/vllm/model_executor/models/qwen4_exp_qsa_block.h` cache views and
  the existing `qwen4_exp` KV lifecycle tests.
- MTP drafting and exact verification: TensorFold
  `src/tensorfold/families/qwen4_exp/cuda/decode.py:mtp_forward`, `draft`, and
  `decode`; weight presence is checked by
  `src/tensorfold/families/qwen4_exp/__init__.py:has_mtp`.
- PLE host movement: TensorFold
  `src/tensorfold/families/qwen4_exp/host_table.py:ReadAhead`; recipe patches
  `0002-flash-next-ssd-read-ahead.patch` and
  `0003-flash-next-ssd-native-reader.patch`; local
  `src/vllm/model_executor/models/qwen4_exp_ple.cpp` and
  `src/vt/cuda/cuda_qwen4_exp_ple.cu`.
- Prefill geometry: TensorFold
  `src/tensorfold/families/qwen4_exp/cuda/decode.py:Engine.__init__` and
  `prefill`; recipe patch `0006-flash-next-prefill-rows.patch`.
- Copy drafting: recipe patch `0007-flash-next-copy-drafts.patch` and the
  TensorFold engine stream lifecycle in `src/tensorfold/engine/lane_family.py`.

Before each port, read the complete executing chain around these symbols and map
it to the local scheduler/cache owner. A symbol name in this spec is not a claim
that memory layout or lifecycle matches.

## Work breakdown

### W0 — Record and harness

- add the dedicated TensorFold comparator pin/scope record without admitting it
  as a secondary correctness oracle;
- create `BENCH-QWEN38-TENSORFOLD-GAP` and atomically adopt its issue;
- add the shared endpoint harness and unit tests for corpus identity, token-ID
  fingerprints, refusal verdicts, metrics, and raw-result serialization;
- add launch manifests for each server without embedding secrets or host paths.

Exit: the harness runs against a fake server, the comparator record has
independent review, its pin checker passes, and ownership is internally
consistent.

### W1 — Reproduce and profile

- acquire one `dgx:gpu0` lease for each uncontended measurement series;
- build or pull TensorFold from the pinned objects and assert runtime identity;
- build vllm.cpp at the campaign SHA and record the binary hash;
- capture deterministic serial outputs and run the pinned Transformers component
  fixtures plus the existing llama.cpp same-GGUF decode proof before timing;
- inspect the declared GGUF and candidate source artifacts for MTP tensors and
  record whether W3 is implementable;
- run all four workload families and record production vLLM as a named blocked
  denominator;
- profile representative 1k decode, 32k prefill, and 128k decode/prefill legs on
  both runnable engines with the same profiler and explicit phase ranges;
- commit raw results, `docs/benchmarks/<benchmark-id>.md`, its
  `docs/BENCHMARKS.md` index row, and `.agents/benchmark-record.md` entries.

Exit: the absolute profile results are reproducible, executable correctness
gates hold, and the profiles rank costs. If either engine cannot load or a
correctness gate fails, stop and record the exact blocker instead of optimizing
from the publisher's table.

### W2 — Incremental QSA state

Implement D4 if W1 shows repeated compressor work or context-growing QSA cost.
Run CPU/reference fixtures, CUDA differential tests, rollback/reuse tests, and
the complete benchmark subset.

Exit: exact selected tokens and bounded state growth, plus a measured result.

### W3 — Native MTP

Implement D5 only if W1 establishes loadable MTP weights and their mapping.
Otherwise commit the `BLOCKED_NO_MTP_WEIGHTS` evidence and scope conversion or
artifact support as a separate issue. When unblocked, add model-weight loading,
draft-state lifecycle, scheduler integration, exact verification, metrics, and
serial fallback. Sweep depth and confidence rather than copying defaults.

Exit: deterministic drafted output equals serial output for greedy fixtures;
sampled tests use fixed RNG state and the existing speculative equivalence
contract. Report acceptance and throughput together.

### W4 — QSA long-context selection

Implement D6 only when W1 or W2 profiles justify it. Keep the old selector as an
A/B arm until parity and speed are recorded.

### W5 — PLE and prefill movement

Implement the applicable branch of D7 and the chunk sweep in D8. Do not add an
SSD subsystem to accelerate a path that does not use SSD.

### W6 — Copy drafts and final synthesis

Evaluate D9, retain it only on evidence, rerun the full matrix at final head, and
publish the vllm.cpp A/B decomposition: raw kernels, cache/state, speculation,
and memory cost. Publish a cross-engine residual gap only if W1 added a valid
matched-artifact arm; otherwise publish absolute profile results side by side.

## Correctness gates

Before a performance number can support a change:

1. Existing Qwen4-Exp unit, fixture, GGUF loader, and CUDA tests pass.
2. The pinned Transformers component fixtures and the same-GGUF llama.cpp
   proof establish the declared executable gate. The blocked production-vLLM
   denominator stays visible. A known fidelity blocker is not permission to
   accept a performance change.
3. Optimized and control paths agree on QSA compressed states, selected block
   IDs, expanded token IDs, and attention output within the existing component
   tolerances.
4. Drafted greedy output is token-identical to serial greedy output.
5. Prefix reuse, fork, rollback, cancellation, eviction, and two concurrent
   requests pass under sanitizer or equivalent lifetime instrumentation where
   available.
6. Provider statistics show no new CPU reference-tier fallback in a claimed
   CUDA-native path.

## Performance gates

Use the repository benchmark protocol:

- no competing GPU process;
- one declared power and clock policy for both arms;
- use `tools/bench/gpu_clock_state.py` for one busy window per arm on the same
  boot; retain at least 30 observed busy samples per window, require busy
  majority and no throttle reason, require at most 5% within-window spread, and
  require arm mean and median clocks within 1%; reject the pair otherwise;
- tear down each engine, verify no serving process remains, and restore declared
  free unified memory and page-cache state before the next engine arm;
- randomize or alternate arm order;
- warm compilation and model state before timed runs, while keeping prompt-cache
  hits at zero;
- report median, mean, P50/P90/P99 where valid, spread, peak memory, and every
  raw repetition; apply the calibrated noise band and paired-majority rule;
- use generated tokens, not requested maximum tokens, as the decode numerator;
- same client harness for both endpoints;
- separate internal timers from client wall time.

For an implementation A/B within vllm.cpp, use one binary with a narrow runtime
switch. A two-binary or adjacent-commit run may locate a candidate, but it cannot
support acceptance and remains `PENDING` until a same-binary control reproduces
it. For exploratory two-binary runs, record both commit SHAs and run
`scripts/ab-arms-differ.py` (or its current repository successor) with both the
required source control and behavioral control. A hash or source assertion by
itself does not prove different arms.

## Evidence

Each evidence directory contains:

- repository SHAs and dirty-tree assertions;
- TensorFold image digest, TensorFold SHA, recipe SHA, CUDA/runtime versions;
- model repository revision, file list, sizes, and sha256 values;
- vllm.cpp binary sha256 and build configuration;
- complete launch commands with host-specific secrets removed;
- GPU identity, memory, clocks, power, temperature, and throttle reasons;
- storage identity and cache state for PLE I/O runs;
- raw harness JSON, unfiltered server logs, profiler exports, and summary tables;
- correctness outputs or hashes and the policy that judges them;
- explicit `COMPARABLE`, `PROFILE_COMPARISON`, or refusal verdict per table.

## Risks

- **Artifact mismatch dominates.** Mitigation: label the first run as a profile
  comparison and add a common-artifact arm only when real support exists.
- **MTP hides raw decode cost.** Mitigation: drafts-off and drafts-on are separate
  workload families.
- **Unified-memory pressure changes placement during a run.** Mitigation: record
  residency, faults, and free memory; do not compare legs with different pools.
- **SSD page cache fabricates PLE gains.** Mitigation: report warm and cold state
  separately and never mix them in one median.
- **QSA side state breaks reuse or rollback.** Mitigation: make it sequence-owned
  and gate every lifecycle operation before timing.
- **A fused optimization changes tie order.** Mitigation: compare selected IDs,
  not only final logits.
- **Publisher figures become accidental acceptance criteria.** Mitigation: local
  evidence defines the baseline and target after W1.

## Stop conditions

Stop the current wave and record evidence when:

- the pinned TensorFold objects do not build or run on GB10;
- an artifact hash or revision cannot be pinned;
- prompt token counts differ for a purported matched workload;
- clocks or thermal state invalidate the pair;
- correctness fails before timing;
- the profile does not show the cost that a proposed optimization removes;
- an optimization gains less than noise or regresses another declared workload
  beyond 3%;
- memory accounting reduces context or concurrency and the result was presented
  as a pure speed gain;
- a GPU hang, OOM, or board reset occurs. Do not continue the same series after a
  board fault.

## Git integration

One pull request carries this campaign. This is the repository default selected
after the developer said to proceed. The committed spec is the first commit.
Implementation and evidence follow as separately reviewable commits. A fresh
review is required before landing. No merge or push authority is inferred from
approval to implement.

## Outcome

Task 4 / W1 stopped at its declared artifact prerequisite on 2026-09-29.
A fresh successful lease executed the committed bounded discovery script on
`dgx:gpu0`. Sanitized evidence retains only its SHA-256 lease fingerprint and
raw marker receipt. The two documented staging paths were absent, and bounded
`/workspace` searches to depth 3 found zero matches for the committed
TensorFold/MiaAI/Vontra and MLX-MTP/Flash-Next-MTP patterns. This is not a claim
of exhaustive host-wide absence. Required artifacts/source were not found in
that retained scope, so the runner prerequisites could not be populated and
measurement did not start.

The outcome is `BLOCKED_MISSING_ARTIFACTS`, not a failed benchmark. No server,
correctness gate, timed ladder, clock window, memory series, or profile ran; no
number or ratio is admissible. Production vLLM remains the named denominator
and is `NOT_RUN`: blocked discovery did not stage or inspect a runnable
production-vLLM denominator.
TensorFold publisher figures remain unverified. The independent GGUF artifact
verdict is `BLOCKED_NO_MTP_WEIGHTS`: the committed 1,224-entry real-header
manifest has only trunk blocks 0 through 47 and no name matching `mtp`, `nextn`,
`draft`, `eh_proj`, `enorm`, or `hnorm`.

Task 5/W2 is `NO_PORT_DECISION`: no incremental-QSA product edit is justified.
Task 6/W3 is `BLOCKED_NO_MTP_WEIGHTS`. W4-W5 are `SKIPPED_NO_PROFILE`.
Task 7/W6 completed synthesis as
`SYNTHESIS_COMPLETED_NO_PRODUCT_OPTIMIZATION`, publishing these outcomes without
a product optimization. Evidence:
`.agents/evidence/bench-qwen38-tensorfold-gap/20260929T180547Z/`; public record:
`docs/benchmarks/qwen38-tensorfold-gap.md`.

## Now

`BLOCKED_MISSING_ARTIFACTS`. Resume W1 only after the runner prerequisites are
available in the documented/configured staging scope. Re-run
executable correctness before any timing; do not reuse this blocker as profile
evidence.

## Owed

- W0 is delivered: comparator pin, ownership, endpoint harness, and launch/capture
  runner are committed and focused tests pass.
- W1 is blocked until required artifacts/source are available in the retained
  documented/configured staging scope. Its 2026-09-29 attempt owes no number; on resume it
  still owes correctness, ladders, clock windows, memory capture, and profiles.
- Task 5 / W2 records `NO_PORT_DECISION`; no QSA product edit is justified
  without the profile premise.
- Task 6 / W3 is `BLOCKED_NO_MTP_WEIGHTS` for the selected GGUF.
- W4-W5 are `SKIPPED_NO_PROFILE`.
- Task 7 / W6 completed synthesis as
  `SYNTHESIS_COMPLETED_NO_PRODUCT_OPTIMIZATION`: the public and agent benchmark
  records carry the outcomes without a product optimization or cross-engine ratio.
