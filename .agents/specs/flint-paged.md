# Flint — CPU paged-engine serving arm (`MODEL-TEXT-FLINT`)

Row: `MODEL-TEXT-FLINT`
State: WAVE
Issue: ISSUE-LOCAL-01M5W2QQBJ32HRB29G1KVQJS77

## Scope

Bring the CPU Flint arm from the single-sequence reference forward to the
shared paged engine: continuous batching through the LLMEngine/runner, a
synthetic-gated paged forward arm, a real-weights paged gate against the
reference forward, the first honest CPU throughput numbers for
flint-Q4_K_M, and the serving docs. CPU only; the P150 device belongs to
another session.

## Upstream anchors

- vLLM implements no flint; the reference for numerics stays the CPU
  reference forward (`FlintModel::Forward`, spec
  `.agents/specs/flint-cpu-gguf.md`) and, for serving behavior, vLLM's
  paged engine conventions already mirrored by the shared seams
  (`dense_attn::AttnBlock`, `vt::PagedAttention`, `v1` KV cache groups).
- llama.cpp PR #30928 handles sinks in its paged path by folding the
  per-head sink into the softmax denominator; our equivalent is
  `PagedAttentionArgs::attn_sink`, implemented on CPU
  (`src/vt/cpu/cpu_paged_attn.cpp:207`).

## What already exists (verified at base e4918e77)

- `MakeFlintKVCache` (flint.cpp:196) already emits the two KV cache groups:
  `FullAttentionSpec` for the full layers, `SlidingWindowSpec`
  (window 513) for the sliding layers, derived from `layer_types`, which
  the GGUF loader builds from `flint.attention.sliding_window_pattern`
  (flint_gguf_weights.cpp:320). W-a is therefore verification + the red
  gate, not new allocation code.
- Per-head sinks, window masking, and the CPU paged attention kernel.
- `dense_attn::StepInputs`/`BuildStepInputs`, `ResidentWeight`, `KvSlice`,
  `WriteKvCache`, `AttnQkNormRope` (per-head QK RMSNorm + NeoX RoPE),
  `MoeRouterTopK` (sigmoid + renorm), `MoeSiluMul`, `MoeCombine`,
  `SharedExpertGate` — the mimo_v2.cpp hybrid pattern is the template.

## Design

W-a (paged KV): the hybrid split above; a focused check that the two
groups carry the right layer membership, window, and dims.

W-b (paged arm): `FlintModel::ForwardPaged` over vt ops, reached through
`ForwardFlintForCausalLM` when `input.attn_kv` is non-empty:

- embed → per-layer: input RMSNorm (add+norm) → q/k/v `MatmulBT` →
  `AttnQkNormRope` on sliding layers only (full layers: QK norm without
  RoPE) → `WriteKvCache` → `vt::PagedAttention` with
  `attn_sink` on EVERY layer and `AttentionWindow{W-1, 0}` on sliding
  layers only → o_proj → add → post RMSNorm (add+norm) → dense SiLU MLP
  or sigmoid top-k renormalized MoE + ungated shared expert → final norm
  → untied lm_head into f32 logits.
- The reference forward stays byte-identical; it remains the oracle.

W-c (real-weights gate): flint-Q4_K_M through the full paged engine,
batch > 1 with continuous arrival; greedy chains compared against the
reference forward's chains within the house near-tie band.

W-d (bench): `vllm-bench` on flint-Q4_K_M, decode + prefill tok/s,
single prompt + a small concurrency sweep, recorded under
`docs/bench-evidence/` and the benchmark record.

W-e (docs): `docs/USAGE.md` Flint serving commands + container pins;
`docs/FEATURES.md` row.

## Risks

- QK-norm + RoPE numerics must match the reference's f32 head stream;
  the synthetic gate asserts the sinks softmax and the window boundary
  token-exactly against hand-computed probes.
- The full-attention layers carry NO RoPE — a wrong preamble silently
  degrades; the position-independence probe (test 4 of the reference
  gate) is ported to the paged arm.

## Tests

- `test_flint_paged.cpp`: KV group shape/membership; red-first paged-arm
  synthetic gate (single-batch paged arm vs reference forward chains;
  window boundary; sink probe; MoE renorm).
- Real gate: reference-vs-paged greedy chains on flint-Q4_K_M.
- Bench evidence files.

## Gates

Focused ctest sets (`flint`, `engine`, `runner`, `paged`), commit style +
trailers over base, `check-agent-record`, preflight.

## Stop conditions

NEEDS_CONTEXT if the shared paged attention cannot express per-head
sinks or the hybrid window split — checked false at base; sinks and
windows both exist on CPU.

## Findings

W-b (2026-10-26/27, worktree /tmp/flint-paged): the paged arm LANDED and
the synthetic gate runs green and GATES: the hybrid KV groups, a mixed
2-request batch through `ModelRegistry::Forward` with continuous decode,
window masking, sinks, and the house 2.5-nat near-tie band against the
reference forward (request 0 replays the reference chain token-for-token;
the one request-1 divergence sits inside the band).

The gate caught THREE real defects:

1. The per-head QK RMSNorm ran IN-PLACE over the head view and corrupted
   the normed stream (q/k sums off ~5%); the arm now writes qn/kn.
2. THE SHARED KERNEL (ISSUE-LOCAL-01M5YMB0XZ8723N0688YHMKP3N, fixed in
   the same flow): `cpu_paged_attn.cpp` seeded the softmax denominator at
   exp(0) with the running max seeded at the sink, in a two-pass form
   that never rescales the seed once a score exceeds the sink — the sink
   term silently became exp(m-m)=1 instead of exp(sink-m). MiMoV2's big
   learned sinks always stay the max, so every earlier gate passed;
   Flint's small sinks exposed it. Both lanes now seed exp(sink - m)
   (bit-identical when the sink IS the max). Measured red-first: L0
   attention output -0.128 vs hand-computed -1.139 over identical bf16
   streams; exact after the fix.
3. The paged arm passed a LIVE residual buffer to the fused
   `vt::RmsNorm(residual=...)` on BOTH norms, so the second call added
   the residual again (residual += x) and every MLP input was
   rmsnorm(2x). The reference keeps ONE residual stream (x itself), so
   the arm now runs plain norms + explicit adds.

Contract alignment (the dead agent's ~3-nat "unexplained residual" was
defects 2+3, not conditioning): the arm mirrors the reference's rounding
exactly at every stage the dump can see — projections round to bf16
before the head norm (MatmulBf16's own final store), the head norm runs
f32-in/f32-out with one bf16 rounding after (the FLINT_REF_BF16_HEAD
control), silu-mul consumes the same bf16 g/u. FLINT_CPU_DUMP now dumps
every paged layer ([FLINT-PAGED] L<l> <stage>) for like-for-like stage
sums against the reference.

## Owed

- CUDA/ROCm device arms: refused by name in the registry switch (existing
  behavior).
- The llama.cpp token gate for the paged arm is NOT owed: the reference
  forward is the oracle for paged batching per row direction.
