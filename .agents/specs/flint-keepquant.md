# Flint TT keep-quant device arm — resident quantized experts on the P150

Row: MODEL-TEXT-FLINT (canonical; the roadmap does not yet carry a dedicated
flint-TT sub-row — the same roadmap gap the flint-tenstorrent and flint-tt-w2
waves noted).

Issue: ISSUE-LOCAL-01M5WCPZC4TWR9GZVJHH73E1XE.

## Scope

The Flint TT device arm today (`src/vllm/model_executor/models/flint_tt.cpp`)
expands every GGUF k-quant to bf16 at load
(`DequantGgufRowToBf16`, flint_gguf_weights.cpp) and then cannot resident the
routed-expert tier: 47 MoE layers x 256 experts x (3x2048x512 + 512x2048) bf16
≈ 70.5 GiB against a 32 GiB P150, so the tier STREAMS from host bytes per layer
(37.7 MB/layer measured) with a 3.606 GiB resident slice and a named budget
refusal above it. The keep-quant machinery proven on Qwen3.8 is not wired into
the Flint arm at all. This row wires it:

- **W-a, the loader arm.** `LoadFlintFromGguf` gains a keep-quant arm that
  keeps each admitted matmul/expert weight's VERBATIM GGUF block bytes in
  `OwnedTensor` with the block `vt::DType`, shape `[N, K]`, `nk == true` —
  the GGUF disk order, which is exactly `MatmulBTQuant`'s orientation, so no
  transpose. Admission per tensor is `KeepQuantDType(ggml_type)` AND
  `DeviceKeepQuantSupported(dt, kTENSTORRENT)` (gguf_keep_quant.cpp:162-263,
  the TT set: Q4_K/Q5_K/Q6_K/Q8_0/IQ3_XXS/IQ2_XXS/IQ2_S/Q3_K/IQ3_S/IQ4_XS/
  IQ2_XS/Q2_K/IQ1_S/IQ1_M). A weight tensor carrying an inadmissible block
  encoding REFUSES BY NAME (the encoding name, the tensor name, the loader
  arm) — it never silently expands, and IQ2_S is already in the TT set
  (enc_sel 6), so the APEX I-Mini IQ2_S towers admit. Norms, sinks, and the
  embedding table keep today's bf16/F32 path (the TT gather of quant rows is
  not registered: `DeviceQuantGatherSupported(kTENSTORRENT)` is false). The
  router, if a container quantizes it, follows the same admission rule.
- **The budget math, re-derived.** `PlanFlintTTStaging` accounts each
  weight's RAW bytes (`OwnedTensor::bytes.size()`) instead of `numel * 2`
  (identical for bf16 — a pure generalization). For flint-Q4_K_M the expert
  tier drops from ≈70.5 GiB bf16 to ≈19-20 GiB of Q4_K blocks and the whole
  model becomes RESIDENT inside the 32 GiB budget; the streamed tier byte
  count goes to zero and the plan reports it.
- **W-b, the dispatch arm.** `flint_tt.cpp` stages a quant weight as raw
  block bytes (`rows * vt::RowSizeBytes(dt, K)`) and routes its matmul
  through `vt::MatmulBT` (b = `[N, K]`, nk), whose `IsBlockQuant` dispatch
  reaches `kMatmulBTQuant` on TENSTORRENT — the same kernel set Qwen3.8
  reaches (IQ encodings UNCONDITIONAL int8-dot; k-quants grouped
  {Q4_K,Q5_K,Q6_K,Q8_0} under `VT_TT_KEEPQUANT_INT8DOT`). bf16 tensors
  (embed, norms, sinks, any expanded weight) keep today's `vt::Matmul` path
  unchanged.
- **W-c, the correctness gate.** The W0 harness recipe on a quantized
  container end to end (flint-Q4_K_M): device-vs-CPU distances against the
  CPU reference loaded from the SAME container. The CPU dequantizes; the
  device keeps blocks — the comparison is function-level near-tie in the
  house band, NOT per-op bit-exactness (the int8-dot activation quantization
  is intentionally coarser than the CPU bf16 path; the Qwen3.8 arm ships the
  same polarity and passed the same kind of gate).
- **W-d, the memory evidence.** Resident bytes before/after on the same
  container: BEFORE 3.606 GiB resident + 70.5 GiB streamed; AFTER the
  measured quantized resident total (the row's reason to exist).
- **W-e, the kolibri discipline check.** If shared keepquant code changed,
  run kolibri's device token gate
  (`VT_KOLIBRI1_TT_B2II_MODEL=/mnt/models/Aleph-Alpha/Kolibri-1
  test_kolibri1_tt_b2ii -tc="*device token gate*"`); its collapsed 3/141
  level is the known baseline and is keepquant-insensitive.
- **W-f (stretch).** TT greedy chains vs the llama-cpp-flint oracle on the
  quantized container, near-tie band.

NOT in scope: engine integration, batching, KV-cache paging (the W0 arm's
one-contiguous-sequence contract stands), performance claims beyond the
resident-byte accounting, any new kernel.

## Upstream anchors

- vLLM implements nothing for this architecture; the oracle is the pinned
  llama-cpp-flint revision (`.agents/oracles/llama-cpp-flint*`). This wave
  gates device-vs-CPU on the SAME quantized weights, not against the oracle
  (W-f is the stretch exception).
- The keep-quant admission set: `src/vllm/model_executor/model_loader/
  gguf_keep_quant.cpp:162-263` (`DeviceKeepQuantSupported`, kTENSTORRENT case)
  — consumed, never copied: Flint admits through the SAME predicate so a
  kernel widening flows to Flint without a loader edit.
- The TT dispatch: `src/vt/tenstorrent/tenstorrent_keepquant.cpp` (~:890-960,
  the kMatmulBTQuant entry and its admitted-encoding VT_CHECK), kernels in
  `kernels/keepquant_kernel_code.h` (enc_sel 0-13).
- The Qwen3.8 wiring this mirrors: loader `OwnGgufQuantBlocks`/
  `OwnGgufKeptSlice` (qwen3_5_gguf_weights.cpp:101-141, 617-640), dispatch
  `vt::MatmulBT`'s `IsBlockQuant` hand-off (vt/ops.cpp:181-196).
- Prior wave: `.agents/specs/flint-tenstorrent.md` (W0 arm this builds on).

## Design

One loader arm + one dispatch generalization; no new forward.

1. `flint_gguf_weights.cpp`: `OwnProjectionKeepQuant(g, name, in, out)` —
   verify geometry, map ggml type through `KeepQuantDType`, refuse-by-name if
   `!DeviceKeepQuantSupported(dt, kTENSTORRENT)`, else own the verbatim
   `[N, K]` block bytes (`OwnedTensor{dtype=dt, shape={out, in}, nk=true}`,
   `bytes = RowSizeBytes(dt, in) * out`). Applied to q/k/v/o, dense
   gate/up/down, shared-expert projections, and — sliced per expert ROW-RANGE
   out of the verbatim stacked tower, `[E*out, in]` rows `e*out..(e+1)*out-1`
   (whole rows, so block-aligned by construction) — the routed towers.
   Selected by `VT_FLINT_KEEP_QUANT` (default ON when `kMatmulBTQuant` is
   registered on kTENSTORRENT and `VT_CPU_REF` is unset — the same polarity
   GgufQuantComputeAvailable gives Qwen3.8; `=0` restores the bf16 arm, the
   same-binary A/B).
2. `flint_tt.cpp`: `StageWeight` handles both dtypes — bf16 as today; block
   dtype allocates `rows * RowSizeBytes(dt, K)` bytes and builds the tensor
   via `MakeTensor` with the block dtype. `MatmulWeight(out, act, w)` picks
   `vt::MatmulBT` when `IsBlockQuant(w.dtype)` (f32/bf16 out as before),
   else `vt::Matmul`. Norms, sinks, embed, and the per-token staging
   (`StageBf16`) are untouched.
3. `PlanFlintTTStaging` accounts `bytes.size()` per OwnedTensor (bf16 tensors
   unchanged numerically). The budget refusal message names the encoding
   census of what did not fit.

## Risks

- Per-expert row-slicing of a stacked quant tower assumes whole rows per
  expert — true because K (in-features) is the block-aligned dim and experts
  tile the OUT dimension. A container with a transposed tower is refused by
  the geometry check.
- The int8-dot kernel's activation quantization makes the device path
  coarser than the CPU bf16 path BY DESIGN; the W-c band states this and is
  not per-op bit-exactness.
- 256*3 per-layer DBuf stagements for 47 layers ≈ 36k device blocks at build;
  memory comes from the pool, and if the pool churns, the towers can be
  staged once per layer as three whole-tower buffers with expert row offsets
  (the fallback this row falls back to before widening scope).

## Tests (red-first)

- `tests/vllm/models/test_flint_keepquant.cpp`: RED first — build a minimal
  in-memory flint GGUF with a Q4_K projection and assert the keep-quant arm
  KEEPS blocks (dtype, `[N,K]`, byte length) — this REDS before the arm
  exists (the loader has no keep-quant arm; the bf16 loader cannot produce a
  quant-typed tensor). Then: an inadmissible encoding (Q4_0, no TT arm)
  refuses BY NAME; the plan accounting reports raw bytes and the budget
  refusal still fires under a shrunken budget; `VT_FLINT_KEEP_QUANT=0`
  restores the bf16 arm.
- Device gate (P150, flock-guarded): the W0 harness recipe on
  /mnt/models/flint-GGUF/flint-Q4_K_M.gguf end to end, device-vs-CPU logits
  distances recorded in docs/bench-evidence/flint-keepquant-<date>/.

## Gates

1. Loader arm: admission census on the container (admitted vs refused-by-name
   counts per encoding) recorded in the evidence report.
2. W-c distances inside the stated near-tie band, CPU on the SAME quantized
   weights as denominator.
3. W-d before/after resident bytes on the same container and budget.
4. Full CPU test suite for flint arms unchanged (bf16 loader is default-on
   for CPU-ref and `=0`); kolibri device token gate at its 3/141 baseline if
   shared keepquant code changed.

## Evidence

docs/bench-evidence/flint-keepquant-<date>/report.md: admission table,
before/after resident bytes, distances + band justification, recipe/env.

## Stop conditions

- NEEDS_CONTEXT if the Qwen3.8 keepquant path cannot express Flint's stacked
  expert towers (it can: kStackedExpertWeight is a supported role and whole
  rows slice block-cleanly — recorded here so the check is done).
- Device unavailable or container missing: the CPU-half lands with the device
  gate recorded PENDING, never faked.

## Owed

- THE DEVICE-FORWARD DISPATCH GATE: the loader arm and the staging plan are
  gated (census + budget + refusal, 4/4 on device), but the flint_tt dispatch
  arm's forward with RESIDENT block weights has not yet run end to end — the
  W0-harness leg with VT_FLINT_KEEP_QUANT=1 (distances vs the CPU reference
  on the same quantized weights, near-tie band) is the next increment.
- The remaining containers (UD-Q4_K_XL, UD-IQ1_M, the 7 APEX files): same
  census + gate per container; IQ2_S admission verified in the capability
  set check.
- The kolibri no-regress leg if shared keepquant code changes further.
