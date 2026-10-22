ID: ISSUE-LOCAL-01M5HHGT86M8D1FDKMA20EX950
Title: kolibri1 CPU: the reference keeps the residual stream in f32 where the model dtype is bf16 - a candidate lever for the remaining gap to llama.cpp
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-22
Updated: 2026-10-22
Closed: 2026-10-22

## Problem

Measured on 2026-10-21 with the peer's layer dumps (mistral.rs) against our own: the relative rms of the difference at layer 0 attn_out_pre_norm is 1.66e-03 between llama.cpp and the peer CPU path (the closest pair), 3.01e-03 between ours and llama.cpp, 3.54e-03 between ours and the peer CPU path, and 1.18e-02 between the device and every CPU implementation. The peer reports that the reference implementations keep the ACTIVATIONS in f32 where the model dtype is bf16, and that mistral.rs does not - about 0.9 of a bf16 ulp at layer 0, and the whole of the gap between mistral.rs and llama.cpp. If that is right, the activations are the thing to match in our CPU control arm, not the norm (whose scale is already f32 on every side, including the peer's candle fix). That would make the residual stream's dtype the leading candidate for our 3.01e-03 distance from llama.cpp against the CPU pair's 1.66e-03. This is a HYPOTHESIS with a measurement attached, not a diagnosis: our forward's residual stream is bf16 today, and no A/B has been run. The owed measurement is a red-first A/B on the CPU: the same prompt, the same weights, the residual stream in bf16 against the residual stream in f32, scored against the committed per-position rows the same instrument already uses (the three-way capture), reporting the layer-0 attn_out_pre_norm rel rms and the top-1 agreement against the primary oracle for both. If the f32 stream moves the layer-0 rel rms toward the pair's 1.66e-03, the lever is real and it is ours to take; if it does not, the lever is closed and the record says so with the numbers.

## Resolution

## Resolution

MEASURED 2026-10-22 (branch `row/KOLIBRI1-F32-RESIDUAL`, evidence
`docs/bench-evidence/kolibri1-cpu-f32-residual-20261022.md`): the lever is
REAL. A red-first A/B on the CPU (same prompt-0 ids, same weights, one
6-token teacher-forced prefill, the residual buffer's dtype the only
difference; the baseline arm reproduces the row's recorded dump byte for
byte, sha256 `7e572ef4…`):

- Layer 0, the quantity and position the issue named (`attn_out_pre_norm`,
  position 0): the arms are BIT-IDENTICAL — 1.6569e-03 against llama.cpp in
  both — because the layer-0 residual IS the embedding and bf16 is exact in
  f32. In the attention quantities, layer 0 cannot read this lever.
- Layer 0, the residual stream itself (`l_out`): the arms differ, and the
  lever CLOSES 25 % of the distance to llama.cpp (2.0016e-03 → 1.5006e-03 at
  position 0, 1.4344e-03 → 1.0981e-03 at position 5; confirmed against a
  second llama.cpp file, `/tmp/opencode/llamacpp-lout.tsv`).
- At depth the lever is an order of magnitude: over layers 12-49 the mean
  rel rms against llama.cpp falls 1.3566e-01 → 1.9704e-02 (`attn_out_pre_norm`)
  and 1.5981e-01 → 4.2759e-02 (`l_out`).
- Not monotone everywhere: at position 5, shallow layers 3-7, the f32 arm is
  briefly worse against llama.cpp, and layer-0 `ffn_post_norm` is mixed.

The lever stays behind `VT_KOLIBRI1_F32_RESIDUAL` (default off). Taking it is
a PRODUCT decision, not a mirror decision: vLLM resolves one model dtype and
every layer inherits it, so the bf16 residual mirrors the primary oracle, and
the f32 residual moves us toward llama.cpp (a secondary oracle) and away from
vLLM's width. No product code lands with this measurement; the decision is
recorded in the spec's owed section with the token-gate wave it would need.
 2026-10-22 (`row/KOLIBRI1-F32-RESIDUAL`, records + a gated measurement arm;
  `VT_KOLIBRI1_F32_RESIDUAL=1`): the lever is REAL and LARGE, and it is a
  PRODUCT decision, not a mirror decision.

  Measured (both arms of our forward on the same released FP8 checkpoint, the
  baseline arm reproducing the row's recorded dump BYTE FOR BYTE, sha256
  `7e572ef4...`): the layer-0 residual IS the embedding and bf16 values are
  exact in f32, so the two arms are BIT-IDENTICAL at layer 0 and no layer-0
  criterion can read this lever. From layer 1 the f32 residual moves us
  toward llama.cpp (layer 1: 3.1794e-03 -> 2.6284e-03), and from layer 11 the
  gap opens in the bf16 arm and not in the f32 one (layer 12: 3.4822e-02 ->
  5.4126e-03). Over layers 12-49 the mean relative rms against llama.cpp
  falls **1.3566e-01 -> 1.9704e-02** (median 1.3229e-01 -> 1.6361e-02), and
  the residual stream's own `l_out` falls 1.5981e-01 -> 4.2759e-02. The
  lever is ONE line (`DBuf res(d, DType::kBF16, ...)` -> `kF32`): `vt::RmsNorm`
  already supports an f32 residual.

  THE POLARITY: vLLM resolves ONE model dtype and every layer inherits it, and
  the primary oracle keeps the residual in the model dtype. The f32 residual
  moves us toward llama.cpp and AWAY from the primary oracle's width. Taking
  it is a product decision with a token-gate wave behind it (the W3
  fingerprints, the decode-bench anchor, the spec-probe anchor), and the
  throughput cost of a doubled residual buffer is not measured. NO PRODUCT
  CODE LANDS WITH THIS MEASUREMENT; the arm is gated behind
  `VT_KOLIBRI1_F32_RESIDUAL` and marked as a scratch measurement instrument.

  ALSO CORRECTED: the peer's three-way table had the sides' names swapped —
  its "1.66e-03" is llama.cpp vs **vllm.cpp (us)**, its "3.01e-03" is
  mistral.rs vs llama.cpp, and its "3.54e-03" is mistral.rs vs us. Our
  distance to llama.cpp was ALREADY the CPU cluster's closest pair, so the
  brief's layer-0 criterion ("move toward the pair's 1.66e-03") was addressed
  to the wrong side and could not have detected the lever. Evidence:
  `docs/bench-evidence/kolibri1-cpu-f32-residual-20261022.md`; spec:
  `.agents/specs/kolibri-1-cpu.md`.


