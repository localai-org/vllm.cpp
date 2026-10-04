# Kolibri-1 W3 goldens — the token gate evidence

Date: 2026-10-04. Row: MODEL-TEXT-kolibri-1 (branch `row/kolibri-cpu`).
Host: 128-core CPU box, 255 GB RAM, no GPU. Spec risk R4 (the transformers
golden run) and the row's token gate.

## What this is

The W3 gate compares OUR CPU forward (wave 2's `kolibri1_forward.cpp`, fed
by wave 1's fp8-block loader on the REAL checkpoint) against a
transformers-class golden run of the same model. The golden side is a
pure-torch transcription of the pinned plugin's math
(`aleph-alpha-inference/kolibri1.py` @ pin `049a6a7bd240`, Apache-2.0),
because `transformers` has no native `Kolibri1ForCausalLM` and the plugin
file imports vLLM throughout, so it cannot run under a plain torch stack.
The transcription is the same one the W2 scalar reference verified line by
line (`tests/vllm/models/test_kolibri1_w2.cpp`, max abs logits gap 0.014).

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
`tests/vllm/models/kolibri1_goldens.json`.

- Wall time: 2542 s (~5.3 min/prompt) on oneDNN bf16 GEMMs.
- The chains are repetitive (base-LM greedy on short prompts): e.g.
  "alpha beta gamma delta epsilon zeta" repeats 121598; "counting: one two
  three four five six seven" repeats 12843. This is the reference behavior,
  not a defect — the gate replays them token by token.

## The gate

`tests/vllm/models/test_kolibri1_w3.cpp` loads the real fp8 checkpoint
through `LoadKolibri1Weights`, decodes each prompt with the HF token ids
from the golden file fed DIRECTLY as input ids (the engine cannot yet
tokenize the kolibri1 pre-tokenizer regex — the W1 recorded gap), and runs
33 prefills per prompt (32 decode steps + 1 final fingerprint step).
Tolerance discipline, stated up front in the test header: both sides carry
bf16 activations and bf16-rounded weights, but the accumulation orders
differ (oneDNN GEMM vs torch GEMM), so the band is kLogitBand = 0.35 (the
W2 measured bf16-noise gap was 0.014 on the tiny model; the 50-layer/37-token
chains are longer). The TOKEN bar is strict: identical argmax for all
8x32 = 256 positions, with near-tie adjudication (a flip is reported and
tolerated only when its top-1/top-2 gap is under the band).

Result: see the gate run log `/tmp/w3-gate.log` (ARGMAX CHAIN and
FINAL-STEP FINGERPRINT MESSAGE lines) and the test's own output; the
verdict lines are quoted into the row spec's ## Now.

## What remains owed

- The tokenizer engine gap (R7): the gate feeds HF token ids directly; the
  no-BOS encode contract still needs the engine's regex set extended.
- The aleph-alpha-inference oracle's gateability measurement (GPU lease).
- GGUF arms, CUDA arm, Tenstorrent arm — later rows.
