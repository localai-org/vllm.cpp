# Muse Glimmer parity: a vision reference run and the GGUF text path against llama.cpp

**Row:** `MODEL-MM-muse-glimmer-muse-glimmer-for-conditional-generation`
(`SPIKE`, unchanged by this spec).
**Issue:** `ISSUE-LOCAL-01M3S0ZTRRVQWCZ35W33AJJNN8`.
**Date:** 2026-09-30. **Base:** `b45a94273`.
**Branch:** `feat/muse-glimmer-parity` (developer direction for this campaign:
one branch per item, local commits only, no pull request).

## Now

`SPIKE`, unchanged. Scope 1-3 are done as the amended `## Gates` define them
(see `## Evidence`). The secondary
llama.cpp gate passes, but the row's lifecycle stays where it is: its advance
needs the bf16 gate against the vLLM pin (owed) and the `docs/STATUS.md` /
`.agents/NOW.md` rows the matrix row already names.

## Scope

1. **Record correction.** The row, `docs/models/muse-glimmer.md` and
   `docs/FEATURES.md` say the pinned vLLM oracle cannot load `muse_glimmer`.
   That was true at `555967922`. The parity pin is now `a7c23ac96d`, which
   registers `MuseGlimmerForConditionalGeneration` and ships
   `vllm/model_executor/models/muse_glimmer.py`. The text is corrected; the
   oracle's gateability on this fleet stays unmeasured.
2. **Vision: the first reference run.** The perception encoder, the adapter and
   `vision_projection` run on the RELEASED tensors
   (`meta-models/Muse-Glimmer-30B` @ `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`)
   against a torch transcription of the pinned vLLM formulas, stage by stage, in
   f32 and in the production bf16.
3. **Text: the GGUF k-quant against llama.cpp.** Greedy decode of
   `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`
   (`meta-models/Muse-Glimmer-30B-GGUF` @ `70bf1b61ac09f91b24d39038091b41c582bc5d7a`)
   through our loader and forward, compared token by token with the registered
   `llama-cpp` oracle at its pin `b10451` on the same file, CPU on both sides.
   Every differing cell is classified the way the tree's paged-engine gates
   classify it (`test_olmo3_paged_engine.cpp`): by the ORACLE's teacher-forced
   gap on our prefix (its top logit minus its logit for our token), inside the
   ratified 500 mnat band a near-tie, outside it a forward divergence. Our own
   margin is not computed. (Amended 2026-09-30 after review: the first draft
   promised margins on both sides, which the tree's gate form does not use.)

Out of scope: the bf16 token gate against pinned vLLM (a ~60 GB checkpoint and
a `dgx:gpu0` lease with an oracle build, owed), the image processor (no C++
port exists; the tower is fed the reference's pixels), video, and speed.

## Upstream chain

The authoritative vLLM parity pin is `a7c23ac96d` (`.agents/upstream-sync.md`,
`vllm_commit`, advanced by `4f11dfc10`); several records on `main` still name
`e126687a9a`. Both register `MuseGlimmerForConditionalGeneration` and ship
`vllm/model_executor/models/muse_glimmer.py`. Its vision and text semantics are
unchanged from vllm#51655 head `075d645af`, which the existing port and
`scripts/mm/muse_glimmer_vision_ref.py` cite: `075d645af..e126687a9a` is line
wrapping, the multimodal processor's call signature, `keep_on_cpu` metadata and
`get_mm_mapping`, and `e126687a9a..a7c23ac96d` is type annotations and the
processor's `_get_hf_mm_inputs` plumbing (both diffs read 2026-09-30).
llama.cpp `b10451` (`10bf611`), `LLM_ARCH_MUSE_GLIMMER`, for the GGUF text path.

## Our baseline

At `b45a94273`: the vision tower is gated only on synthetic LCG weights
(`test_muse_glimmer_vision`); no real tensor has passed through it. The GGUF
text path generates `" Paris. The capital of France is Paris. ..."` and agrees
with llama.cpp on the first token only (`docs/models/muse-glimmer.md`).

## Port map

| Ours | Upstream (`muse_glimmer.py` @ `a7c23ac96d`) |
|---|---|
| `MuseGlimmerVisionForward` | `MuseGlimmerVisionEncoder.forward` |
| `MuseGlimmerVisionAdapterForward` | `MuseGlimmerVisionAdapter.forward` |
| `MuseGlimmerEncodePixelGroups` (projection) | `_encode_pixel_groups` |

## Dependencies

- The staged tensors: `model.vision_*` from both shards, cut by HTTP range
  onto the NAS (`/mnt/nas_share/rc/muse-glimmer-30b/`), not committed.
- The GGUF file on the same NAS directory, and a CPU build of llama.cpp `b10451`.
- Existing: `muse_glimmer_vision.cpp`, `muse_glimmer_gguf_weights.cpp`,
  `muse_glimmer.cpp`, the GGUF tokenizer.

## Tests to port

None new from upstream: vLLM's `tests/models/multimodal` has no Muse Glimmer
numeric test at the pin (only `tests/transformers_utils/test_muse_glimmer_config.py`
and the tool/reasoning parser tests, already ported by the row).

## Gates

- Vision (env-gated, real tensors): `ln_pre`, block 0, tower output, adapter
  and the soft tokens (projection plus `perception_emb_norm`), relative max
  error: `ln_pre` and block 0 under 1e-4, tower, adapter and soft tokens under
  1e-3 in f32. `patchify` and the positional table are gated on the synthetic
  fixture only; the real run checks them through `ln_pre`. The bf16 arm is
  gated against the f32 truth, not against the reference's bf16 arm: its row
  cosines (mean within 1e-3, worst within 0.02 of the reference bf16 arm's)
  and its relative max error (at most twice the reference bf16 arm's, because
  cosine cannot see a scale error).
- Text (env-gated, real file): 16 prompts x 32 greedy tokens through the paged
  engine; our ids equal the committed anchor exactly, and every cell that
  differs from llama.cpp's greedy has an oracle teacher-forced gap within
  500 mnats. The exact-match count is reported next to the pass.

## Work breakdown

1. This spec, the issue and the record correction.
2. The vision reference run: extend the reference script with a real-weights
   mode and add the env-gated `test_muse_glimmer_vision_real`.
3. The GGUF comparison: a llama.cpp driver that writes the oracle's greedy ids
   and its teacher-forced gap on our prefix, our side through the production
   GGUF load, and the paged-engine gate.
4. Records: the model page, FEATURES, the matrix row, `## Outcome`.

## Risks/decisions

- A transcription is not the runtime. The vision gate establishes agreement
  with the pinned formulas on real tensors, not image-to-text correctness.
- A Q4_K_M comparison against llama.cpp can only show quantization-matched
  agreement. A near-tie cell passes the gate (the tree's ratified band) but is
  reported separately from an exact match, never folded into it; a defect
  found is fixed in this branch only if it is inside this item.
- Staging: the 16.76 GB GGUF and ~3.7 GB of vision tensors live on the NAS,
  because the development host has under 10 GB free.

## Evidence

All on CPU, 2026-09-30.

- **Vision** (`test_muse_glimmer_vision_real`, env
  `MUSE_GLIMMER_VISION_REAL_DIR` / `MUSE_GLIMMER_VISION_REAL_GOLDEN`; goldens
  from `scripts/mm/muse_glimmer_vision_ref.py --real-dir`): a 588x644 image,
  483 soft-token rows, weights through `LoadMuseGlimmerVisionTower` and soft
  tokens through `MuseGlimmerEncodePixelGroups`. f32 arm: `ln_pre` 1.22e-6,
  block 0 3.56e-6, tower 7.87e-5, adapter 4.58e-5, soft tokens 4.07e-5
  relative max error, row cosine 1.0. bf16 arm, gated against the f32 truth:
  ours worst row cosine 0.967, mean 0.99875, relative max error 0.0916; the
  reference's own bf16 arm worst 0.970, mean 0.99856, 0.0837. PASS. Mutations
  (restored): dropping the adapter's outer GELU, skipping
  `perception_emb_norm` (now red in the bf16 arm too, through the magnitude
  bound), and loading `ln_post.bias` from the `ln_pre.bias` slot each turn it
  red.
- **Text** (`test_muse_glimmer_gguf_paged_engine`, env `VLLM_MUSE_GGUF_PARITY`;
  golden `tests/parity/goldens/muse_glimmer_30b_q4km/`, file sha256
  `4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e`): 16
  prompts x 32 greedy tokens through `LoadedEngine::FromModelDir`, against the
  llama.cpp `b10451` oracle on the same file. 10/16 token-exact; the other 6
  stay inside the 500 mnat band at every cell (max teacher-forced gap 93 mnats,
  prompt 15 token 8); zero forward-divergent cells. Anchor: our ids equal the
  committed `our_ids.npy`. Gate run: "16/16 prompts PASS (token-exact 10/16; near-tie band only 6/16; max gap 0.093 nats @ prompt[15] tok=8; 0 forward-divergent)", 45 min on a shared 20-core host.
- **Record correction**: the model page, FEATURES and both matrix rows now
  qualify every "beyond pin" and "no oracle" statement with the pin it held
  for (`555967922`) and state that `a7c23ac96d` registers the model.

## Stop conditions

- The vision tensors or the GGUF cannot be staged.
- A divergence needs a runtime feature this item cannot finish; it is then
  reported with its size and left owed.

## Owed

- The bf16 token gate against pinned vLLM `a7c23ac96d` on a GPU lease (see
  [Tracked](#tracked)).

## Tracked

- `ISSUE-LOCAL-01M3S0ZTRRVQWCZ35W33AJJNN8`, the row-owned issue under
  `.agents/issues/MODEL-MM-muse-glimmer-muse-glimmer-for-conditional-generation/`,
  tracks the bf16 token gate above.
