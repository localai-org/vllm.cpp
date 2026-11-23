# SPEC — `NEW-CANDIDATE-PORTS` W2e: olala keep-quant decode (q6_k / q5_k_m / q8_0-safe2)

- Owning row: `NEW-CANDIDATE-PORTS`. W2e is a wave of the W2 spec
  (`.agents/specs/new-candidate-ports-olala-w2.md`); this file scopes the
  quantized arms only. Base: PR #3585 head `ccf038670` (the W2d bf16 token
  gate EXACT vs `llama-cpp-olala`).

## Scope

The three quantized olala GGUF arms stop refusing by name and decode on the
production load path with a token gate vs the pinned `llama-cpp-olala` oracle
(`OVHai-LLM/llama.cpp` branch `olala-master` @ `bd20ca9b`, pin in
`.agents/oracles/llama-cpp-olala.md`):

- `Olala-v1.0-mini-7A1B-q6_k.gguf` — sha256
  `58a2b658158290687a431705c3c256c1331580ad4e8e99db55d03a8131d223c8`
- `Olala-v1.0-mini-7A1B-q5_k_m.gguf` — sha256
  `ed353d2632c406be9ca2022b35e4c13049d6f8f7390b1c1bb61d6a7d6b971860`
- `Olala-v1.0-mini-7A1B-q8_0-safe2.gguf` — sha256
  `0bb4560fa6800b41d16a15507ad1843ca17c0849ce0cc989bca337dcf55136d7`

Out of scope: new ops, device arms, the `olala-transformers` oracle run
(gateable=no, owed separately).

## Tensor/type census (W2e walks, this branch)

`docs/bench-evidence/new-candidate-ports-olala-w2e-20261122/olala-tensor-walk-{q6_k,q5_k_m,q8_0-safe2}.txt`
(989 tensors each, per-tensor sha256; the q6_k walk cross-checks against the
W1 walk of 20261120 — types, shapes, nbytes and hashes agree).

| arm | Q6_K | Q5_K | Q8_0 | BF16 | F32 |
|---|---|---|---|---|---|
| q6_k | 303 | — | 36 | 36 | 614 |
| q5_k_m | 37 | 266 | 36 | 36 | 614 |
| q8_0-safe2 | 36 | — | 231 | 108 | 614 |

Quantized are the big matmul weights (attn_output/attn_q/attn_gate/wa/wb/
lambda/shift, ssm_in/ssm_in_dyn, ffn up/down experts and shared, latent_down,
token_embd, output). Always F32: router (ffn_gate_inp), exp_probs_b, all geo_*
and ssm_* scalars/biases/norms. Always BF16: ffn_latent_up (36; q8_0-safe2
additionally keeps attn_output and ffn_down_shexp in BF16). Per-layer expert
up (`ffn_up_exps`) is Q8_0 in every arm.

## Design (choice recorded per the repo precedent survey)

Repo precedent for a host-op model is **keep-quant: the ggml blocks stay in
their file encoding and every matmul dequantizes on the fly** (the keep-quant
residency family: `GgufResidency::kKeepQuant`, `gguf_keep_quant.h`; the
qwen4/qwen3_5 loaders copy blocks verbatim and consume them quantized). The
olala forward is all-host f32 (`WAt`/`W3At` funnel every weight read,
`olala_ops.cpp:37-62`), so the minimal complete keep-quant arm here is:

1. **Loader** (`OwnScalar`, `olala_gguf_weights.cpp`): block-quant tensors are
   no longer refused. Map ggml type id -> `vt::DType` with
   `BlockDTypeFromGgmlTypeId`, size the owned buffer with `RowSizeBytes` per
   row (never `vt::SizeOf`, which is a VT_CHECK failure on block dtypes by
   design), and copy the raw blocks verbatim. Supported encodings: Q8_0(8),
   Q5_K(13), Q6_K(14) — exactly the set the three artifacts carry. Any other
   block encoding still refuses by name.
2. **Decode** (`olala_ops.cpp`): `WAt`/`W3At` gain a quant arm —
   dequantize the CONTAINING ROW to f32 with `DequantGgufRowToF32` (ported
   from llama.cpp @ 237ad9b9, already in-tree and sha-audited) and cache it in
   a 64-slot direct-mapped thread_local row cache (multiplicative row hash; a base-only key collides the ops' same-tensor row pairs, see ISSUE-LOCAL-01M82WJVXS9WFJ4WPRQ4BWX29H) keyed on
   (tensor bytes base, row). Element access is then the same f32 read the
   bf16 arm does. The cache is a performance device only: a miss dequantizes,
   a hit is free, and eviction is always correct.
3. **Embed/head gather** (`OlalaGenerateHost`): the hand-rolled bf16 byte
   reads in `embed_at` and the lm-head argmax loop route through the same
   per-row dequant helper. On the bf16 arm the helper reads the identical
   bytes through the identical `Bf16ToF32`, so the bf16 token gate is
   unchanged.

Compute stays f32 with double accumulation exactly as the bf16 arm does; the
quant arm changes only WHERE the weight values come from (blocks, decoded at
read time) not HOW they are consumed. The norm/router/scalar handling is
unchanged from the bf16 arm (F32 passthrough).

## Upstream anchors

- vLLM implements no `olala` arm — no primary oracle path to mirror; the port
  mirrors the fork's quant plumbing behavior:
  `src/llama-model-loader.cpp` (llama.cpp @ bd20ca9b, olala-master) keeps k-quant
  tensors in their ggml type; `ggml-cpu` vec_dot quantized matmuls dequantize
  activations, not weights. Our host-op forward dequantizes per row instead of
  running vec_dot — a documented adaptation, recorded because the row-dequant
  values are bit-identical to `dequantize_row_q6_K`/`_q5_K`/`_q8_0`, which the
  fork itself calls.
- In-tree: `gguf_dequant.{h,cpp}` (block format byte-for-byte from llama.cpp
  @ 237ad9b9), `gguf_keep_quant.h` (residency vocabulary),
  `vt::BlockDTypeFromGgmlTypeId`/`RowSizeBytes` (dtype.h).

## Gates / tests

- Loader: the refusal test (if any) is inverted — q6_k/q5_k_m/q8_0 arms now
  LOAD; an unsupported block encoding (e.g. IQ2_S) still refuses by name.
  Synthetic-fixture unit tests first (red before), mirroring the existing
  fixture style in `tests/support/*_gguf_fixture.h`.
- Token gate per arm: `OlalaGenerateHost` on prompt
  `[151644, 872, 198, 15588, 220, 151645]`, n_gen=6, vs fork goldens generated
  at the same pin (harness `llama-fork-golden-harness.cpp`). Bar: EXACT or a
  documented near-tie per the ratified #2839 standard. Quant arms change
  logits vs bf16 by construction — each arm is gated against ITS OWN fork
  golden on the identical artifact sha.
- bf16 arm regression: the existing W2d EXACT gate stays EXACT.
- Suite green in Release AND bounds-checked Debug (`/tmp/build-w2e-rel`,
  `/tmp/build-w2e-dbg`), preflight --staged before commits, record rot 0.
- doctest discipline: no compound `&&` in CHECK (compute bool first), separate
  CAPTUREs, no `M_PI`, `OwnedBytes .data()` for const reads.

## Risks / decisions

- Numerics differ from the fork's vec_dot path (it quantizes activations to
  Q8_K; we dequantize weights to f32 and accumulate f32). First token flips on
  near-ties are documented, not hidden; the gate bar follows #2839.
- The row cache is thread_local and single-slot-per-row; ops are documented
  single-threaded host code. A wrong cache key would corrupt decode — covered
  by the per-arm token gates.
- q8_0-safe2 carries BF16 attn_output/down_shexp — the mixed load must keep
  per-tensor dtypes, not promote the whole layer.

## Stop conditions

- An arm whose fork golden cannot be produced at the pinned build reports the
  arm PENDING, not passing.
- A first-token flip with no near-tie evidence (both sides' logits within
  tie distance) opens a bug issue in the same flow.

## Owed

- `olala-transformers` oracle measurement (gateable=no) — unchanged, separate.
- Prompt-level tokenizer parity for arbitrary prompts (the gate prompt ids are
  pinned; tokenizer port is not in W2e).

## Git integration

One PR per wave (row claim answer recorded 2026-11-20); W2e rides the same
`row/NEW-CANDIDATE-PORTS` stack as W2c/W2d (branch `row/NEW-CANDIDATE-PORTS-w2e`).

## Now

2026-11-22: W2e in progress on `/tmp/olala-w2e`. Spec committed before code.
