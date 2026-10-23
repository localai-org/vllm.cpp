# Flint — CPU + GGUF port spec (`MODEL-TEXT-FLINT`)

Row: `MODEL-TEXT-FLINT`
State: SPIKE
Issue: ISSUE-LOCAL-01M5HZ7K4N8TQ9W2E5R3V1X0 (filed with this spec)

## Scope

Port RecursiveSI/flint (`FlintForCausalLM`, published 2026-10-21) to vllm.cpp:
the CPU safetensors arm and the GGUF arm. The Tenstorrent arm is a separate
later row. The quantized artifacts are already on this box and sha256-verified
(`unsloth/flint-GGUF` @ `6c2910f97fb36b5cb22bbdc8f014ab42656b0d14`:
UD-IQ1_M 11.35 GB `dd45136a...`, UD-Q4_K_XL 25.05 GB `63c2ecc4...`).

## Upstream anchors

- **llama.cpp PR #30928** (the model author's own; head
  `1b44e30775dcb2f93952116b05f89e5bd797fb00`, MERGEABLE, CI green on
  ubuntu/windows/webgpu/vulkan/cuda/metal/model-naming/Lint): the only
  llama.cpp that converts, loads and runs flint. The GGUF tensor set, the
  `flint` pre-type regex (id 62), and `src/models/flint.cpp` are the
  reference. Our oracle record pins this head.
- **transformers `models/flint/` on main** (unreleased; absent from 5.20.2 and
  every local install): the numerics reference.
- **vLLM: does not implement flint** (0 code hits). mistral.rs: oc's
  in-progress implementation is a cross-check, not an oracle.

## The model (from the GGUF header, the transformers reference and PR #30928)

49 layers, hidden 2048, 32 heads / 4 KV heads, head_dim 128, vocab 150144,
rms eps 1e-6, rope 10000, context 262144. Hybrid attention `FFSSSF` repeating
(14 full, 35 sliding), sliding_window 513, **no RoPE on the full-attention
layers** (`attention.rope_pattern`). MoE: 256 experts / 6 per token, expert
ff 512, **shared expert 1024 (`expert_shared_count` = 2), always added,
ungated**; layers 0-1 dense (`leading_dense_block_count` 2); routing = fp32
linear → **sigmoid** → top-6 → renorm (`expert_weights_norm` true);
**per-head attention sinks** (`softmax_offset`, one scalar per head) folded
into the softmax denominator; attention scores in fp32 ("QK-normed logits
reach tens"); q/k RMSNorm over the head dim.

## Design

1. The oracle record first: `.agents/oracles/llama-cpp-flint.md` pinning
   `1b44e30...` with a gateability measurement (build that head, run the
   UD-IQ1_M GGUF, capture tokens), plus the registry row in AGENTS.md.
2. The model TU `src/vllm/model_executor/models/flint.cpp|h` +
   `flint_weights.cpp|h` (safetensors arm), mirroring the numerics above.
   The sink softmax: `out = Σ exp(S_j)·V_j / (Σ exp(S_j) + exp(sink))` —
   no ttnn sink parameter exists, so the denominator is folded.
3. The GGUF loader row: one `GgufArchArm` entry (`flint`) + a weights TU
   mapping the PR #30928 tensor names (`self_attn.softmax_offset`,
   `ffn_{gate,up,down}_shexp`, `ffn_exp_probs_b`) and the sliding/rope
   patterns; the `flint` pre-type regex (PR #30928's
   `LLAMA_VOCAB_PRE_TYPE_FLINT`) ported into the tokenizer dispatch as a
   named pattern (never aliased).
4. Gates: the token gate against the reference implementation's own captured
   tokens (the llama.cpp run from step 1), plus the GGUF-vs-safetensors
   equivalence on the same prompts.

## Risks

- The sink softmax has no ttnn/kernel precedent in this tree; the folded
  denominator must be proven exact, not assumed.
- The only reference is an UNMERGED llama.cpp PR + transformers main: both
  pins are moving targets and must be recorded with their SHAs.
- `transformers` cannot adjudicate a near-tie (the recorded 0.125-ULP
  limitation); token-exact gates come from the llama.cpp floor.

## Owed

- The device arm (Tenstorrent), a separate row.
- The imatrix question: any container we quantize differs from the published
  Hob-forge files (whose imatrix was never published) — a quantizer
  difference, recorded as such, never a quant-type difference.
- The REAL-WEIGHTS token gate: RecursiveSI/flint's safetensors are on no box
  here and the download grant does not cover Flint, so the gate ran on the
  synthetic config only. Owed by this row's next wave.
- The flint GGUF arm: the loader row gates the published Hob-forge
  containers, and the token gate's floor for them is owed (llama.cpp aborts
  on their `expert_gating_func = 5` and mistral.rs's `build_kolibri1`
  requires a `kolibri1.feed_forward_length` KV they do not carry) — the
  candidate floor is a metadata-patched derived copy plus mistral.rs parity.
- The paged/engine path: the first wave is the reference HostLogits forward
  only; the paged engine and device routing are the later wave's.