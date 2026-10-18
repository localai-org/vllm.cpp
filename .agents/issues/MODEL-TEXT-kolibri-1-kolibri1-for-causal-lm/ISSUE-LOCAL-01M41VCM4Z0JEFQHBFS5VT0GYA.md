ID: ISSUE-LOCAL-01M41VCM4Z0JEFQHBFS5VT0GYA
Title: Port Kolibri-1 (kolibri1, Kolibri1ForCausalLM) to the CPU path
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-03
Updated: 2026-10-08
Closed: -

## Problem

The model registry has no kolibri1 support. Aleph-Alpha/Kolibri-1 (Kolibri1ForCausalLM, model_type kolibri1) is downloaded at /mnt/models/Aleph-Alpha/Kolibri-1 and the primary oracle plugin aleph-alpha-inference (pinned 049a6a7bd240) implements it on vLLM. This row adds the CPU-only port: config parse + validation, two-group KV cache (full + sliding-window 513), 32-shard FP8-block safetensors weight loader, sigmoid-logit-add MoE router (384 experts, top-6, shared expert, norm_topk_prob=false), hybrid SWA/full attention with qk-norm and RNoPE, sandwich norms, and tokenizer integration, token-exact against a reference. Spec: .agents/specs/kolibri-1-cpu.md

## Resolution

- 2026-10-03 (W1): config parse, registry, KV-cache spec, fp8-block loader
  land on row/kolibri-cpu.
- 2026-10-03 (W2): the CPU hybrid forward lands, gated against an in-test
  scalar transcription of the plugin's math (max abs logits gap 0.014).
- 2026-10-05 (W3): the row's token gate runs against the REAL checkpoint —
  bf16 reference dequantized (spot-check 1.05e-4 vs direct fp8->f32 decode,
  docs/bench-evidence/kolibri1-goldens-20261004.md), golden run over 8
  prompts (greedy, 32 tokens, fingerprints committed at
  tests/vllm/models/kolibri1_goldens.json), and the gate replays them:
  141/145 argmax positions match, 4 flips all near-ties inside the
  measured 2.19 bf16 envelope, no hard flips. Still OPEN: the oracle
  gateability measurement (GPU) and the quantized/backend arms.
- 2026-10-08 (R7): the tokenizer engine accepts the Kolibri-1 split regex
  (recognition-only: the \p{N}{1} spelling of the classic Qwen2 pattern,
  mapped onto SplitPattern::kQwen2Classic). The W1 tokenizer test now
  asserts our Encode reproduces the HF reference ids from
  kolibri1_goldens.json on all 8 golden prompts, gating the no-BOS encode
  contract on a real load. Closed as
  ISSUE-LOCAL-01M4D7DFJRQSCSRC03CHZW3EQN; W3 re-run green at 900/900 and
  decode_bench at anchor 109726 on row/kolibri-r7.
