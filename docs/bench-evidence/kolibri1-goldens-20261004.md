# Kolibri-1 W3 goldens — the token gate evidence

Date: 2026-10-04/05. Row: MODEL-TEXT-kolibri-1 (branch `row/kolibri-cpu`).
Host: 128-core CPU box, 255 GB RAM, no GPU. Spec risk R4 (the transformers
golden run) and the row's token gate. Gate log: `/tmp/w3-gate-final.log`
(2026-10-05 run, `test_kolibri1_w3`).

## What this is

The W3 gate compares OUR CPU forward (wave 2's `kolibri1_forward.cpp`, fed
by wave 1's fp8-block loader on the REAL checkpoint) against a
transformers-class golden run of the same model. The golden side is a
pure-torch transcription of the pinned plugin's math
(`aleph-alpha-inference/kolibri1.py` @ pin `049a6a7bd240`, Apache-2.0),
because `transformers` has no native `Kolibri1ForCausalLM` and the plugin
file imports vLLM throughout, so it cannot run under a plain torch stack.
The transcription is the same one the W2 scalar reference verified line by
line (`tests/vllm/models/test_kolibri1_w2.cpp`, max abs logits gap 0.014
at tiny scale).

## The bf16 reference (dequant verification)

`scripts/gen-kolibri1-bf16-reference.py` streamed the 32 fp8 shards through
the W1 loader math (F8_E4M3 x F32 `weight_scale_inv` per 128x128 block, f32
arithmetic) and wrote bf16 shards with the same layout and a new index,
stripping `quantization_config` (the dequantized weights ARE the reference).

- 116303 tensors: 26657 fp8 dequantized, 184 bf16 copied byte-identical,
  26657 f32 scale grids consumed.
- Spot check against a direct fp8->f32 decode (independent manual e4m3
  decode table in numpy) on three tensors across shards
  (`layers.0.self_attn.q_proj`, `layers.30.mlp.experts.200.down_proj`,
  `layers.49.mlp.shared_experts.gate_proj`): max diff 1.05e-4 — pure bf16
  rounding of the f32 product, no math error.
- Byte-identical passthrough verified for `model.layers.0.mlp.gate.weight`,
  `model.embed_tokens.weight`, `model.norm.weight`.
- Output: `/mnt/models/Aleph-Alpha/Kolibri-1-bf16`, 149 GB, 32 shards.

## The golden run

`scripts/gen-kolibri1-goldens.py`: greedy decode, temperature 0, seed 0,
32 tokens, full recompute per step (no KV cache — numerically identical for
this fixed math). 8 prompts: 4 synthetic word-pattern (bench generator
style), 4 natural, two German. Fingerprints (input ids, generated chain,
final-step top-8 logits + sums) committed at
`tests/vllm/models/kolibri1_goldens.json`. Wall time 2542 s (~5.3
min/prompt, oneDNN bf16 GEMMs). Chains are repetitive (base-LM greedy on
short prompts) — that is the reference behavior, and the gate replays them
token by token.

## The gate measurement

2026-10-05 run, wall 12991 s including the ~3.5 min checkpoint load; each
prompt is 32 greedy prefills plus one teacher-forced fingerprint prefill.

- ARGMAX CHAIN: 141/145 compared positions match the golden greedy decode.
  4 flips, ALL at near-ties: gaps 0.0093 ('Ein gutes Buch...', step 7),
  0.262 ('Wissen ist Macht...', step 1), 0.289 ('alpha beta...', step 0),
  0.382 ('Translation to German...', step 5). No flip exceeded the measured
  noise envelope. 4 of 8 prompts replay token-identically end to end
  ('der Mond...', 'The capital of Australia is', 'x1 = 3, x2 = 7...',
  'counting: one two...'); diverged chains stop comparing at the first
  flip, because later positions would compare different contexts.
- FINAL-STEP FINGERPRINT (teacher-forced on the golden chain, fixed ids on
  both sides): worst top-8 logit diff 2.186. This measured envelope on
  token-identical chains IS the bf16-noise scale of the 50-layer forward
  and is what the near-tie band (2.5) is derived from. The a-priori 0.35
  band (a naive W2 extrapolation) was wrong by an order of magnitude.
- Cross-check that the flips are rounding, not math: an fp32-activation
  rerun of prompt 1 step 0 on the torch side keeps the golden ordering
  (121598 over 116026) with a 0.305 margin — the two candidates are a
  genuine knife-edge pair (~0.3 apart on both sides), and the C++ orders
  the pair the other way within rounding noise.

## Tolerance statement

Fixed-ids top-8 logit values agree within the measured envelope 2.19
absolute (band 2.5); greedy argmax chains are token-identical except where
the golden's own top-1/top-2 margin is of the same order as the rounding
noise (0.009-0.38 observed), where the flip is adjudicated a near-tie. A
flip outside the envelope is a hard failure: the gate CHECKs
`hard_flips == 0` and `worst_topk <= kLogitBand`.

## Red-first captures from development

1. Iterating the safetensors index's weight_map opened one fd per TENSOR
   (116303 opens) and died with EMFILE at the 1024 fd limit — the fix
   opens one fd per DISTINCT shard (32).
2. An inverted sliding/full group split failed the two-group KV resolution
   at layer 4 (`KV cache not found for layer 4`).
3. Semantics: the fingerprint context is the forward that PREDICTED the
   32nd token — input + generated[0..30]. Using the full 32-token chain is
   one position later and measured topk diffs up to 11.95.

## What remains owed

- The tokenizer engine gap (R7): the gate feeds HF token ids directly; the
  no-BOS encode contract still needs the engine's regex set extended.
- The aleph-alpha-inference oracle's gateability measurement (GPU lease).
- GGUF arms, CUDA arm, Tenstorrent arm — later rows.
- A deeper separation of rounding from systematic difference at depth
  (fp32-activation reference over the full model) if a future lever needs
  tighter than the 2.19 envelope.
