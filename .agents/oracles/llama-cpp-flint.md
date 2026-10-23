# Oracle: llama.cpp PR #30928 head, the model author's own Flint implementation

A scoped, second llama.cpp record. It exists for one reason: RecursiveSI's
`FlintForCausalLM` (published 21 October 2026) is carried by NO llama.cpp
release, and the row `MODEL-TEXT-FLINT` needs a runnable reference and a
quant-matched denominator on the artifact users actually download (the
`unsloth/flint-GGUF` containers). It does not replace the
[`llama-cpp`](llama-cpp.md) oracle, it does not outrank vLLM, and it is never a
mirror source.

## Why this record pins a pull-request head, not a release

No llama.cpp release converts, loads or runs `flint`: the architecture lands in
the unmerged model-author PR
[ggml-org/llama.cpp#30928](https://github.com/ggml-org/llama.cpp/pull/30928)
("models: Add Flint model"), and this record pins its head
`1b44e30775dcb2f93952116b05f89e5bd797fb00`. The precedent is
[`llama-cpp-qwen4exp`](llama-cpp-qwen4exp.md) and
[`llama-cpp-glm5next`](llama-cpp-glm5next.md), which pin PR heads for exactly
this reason: the architecture exists only on the PR. The contrast is
[`llama-cpp-dsv4vision`](llama-cpp-dsv4vision.md), whose support is merged and
released and therefore pins release `b10766` instead. An unmerged head is a
weaker pin than a release — the PR can be rebased or closed — so this record
must be retired the moment a llama.cpp release carries `flint`, and the
[`llama-cpp`](llama-cpp.md) pin advances then.

The head was MERGEABLE at pin time with CI green on
ubuntu/windows/webgpu/vulkan/cuda/metal/model-naming/Lint (vulkan-amd and rocm
pending), read from the PR's own check state on 23 October 2026.

## Scope, and what this oracle may not do

Use it ONLY for the `flint` architecture: the GGUF conversion in the PR's
converter entry, the graph in `src/models/flint.cpp`, the arch and
hyper-parameter registration (`LLM_ARCH_FLINT`, `src/llama-arch.cpp`,
`src/llama-model.cpp`), the `flint` pre-tokenizer type
(`LLAMA_VOCAB_PRE_TYPE_FLINT`, `src/llama-vocab.cpp`), the sigmoid-gated MoE
router (`expert_gating_func` sigmoid with `expert_weights_norm` renormalisation
and the optional `ffn_exp_probs_b`), the ungated shared expert
(`ffn_{gate,up,down}_shexp`, `expert_shared_count`), the per-head attention
sinks (`attn_sinks`, one scalar per query head), and the CPU and GGUF k-quant
speed and memory numbers those produce on a Flint checkpoint.

The architecture this scope names: 49 layers, hidden 2048, 32 query heads
against 4 KV heads at head_dim 128, vocab 150144, RMS eps 1e-6, hybrid
`FFSSSF` sliding/full attention (sliding window 513, `is_swa_impl` array) with
NO RoPE on the full-attention layers (`rope_pattern` array), 256 routed experts
top-6 with layers 0-1 dense (`n_layer_dense_lead` 2), and standard pre-norm
residual (NOT Kolibri-1's sandwich).

For every other path, including CPU and GGUF k-quant floors generally, the
oracle is [`llama-cpp`](llama-cpp.md) at its own pin. Where vLLM implements the
behaviour at all, vLLM wins; llama.cpp's structure is not vLLM's, and a
difference in structure is never a reason to diverge. The model author's own
runtime wins where the two disagree on model semantics.

## Gateability

`gateable = yes`. The flag says one thing: this head BUILT and RAN the Flint
model and produced tokens. The evidence is
[the Flint llama.cpp oracle gate](../../docs/bench-evidence/flint-llamacpp-oracle-gate-20261023.md),
which records the clone at the exact SHA, the CPU-only build flags, the loaded
container with its sha256, the prompt, and the generated tokens.

**Gateable is not measured.** The flag says the oracle runs. It says nothing
about a throughput, a latency or a memory floor; the evidence run was a
correctness gate on a busy host and no rate in it is quotable. The floors are
the row's own later gates.

```oracle-pin
id = llama-cpp-flint
role = secondary
upstream = https://github.com/ggml-org/llama.cpp
scope = the flint architecture only: its GGUF conversion, its graph, its arch and hyper-parameter registration, its sigmoid-gated MoE router, its shared expert, its per-head attention sinks, and the CPU and GGUF k-quant floor those produce on a Flint checkpoint
pin = 1b44e30775dcb2f93952116b05f89e5bd797fb00
pin_label = PR #30928 head (unmerged; the model author's own Flint PR)
pinned_on = 2026-10-23
gateable = yes
evidence = docs/bench-evidence/flint-llamacpp-oracle-gate-20261023.md
```
