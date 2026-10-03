# Nemotron Nano VL / Omni: the RADIO image path

**Row:** `MODEL-MM-nano-nemotron-vl-nemotron-h-nano-vl-v2` (`INVENTORIED` ->
`SPIKE` with this spec).
**Issue:** `ISSUE-LOCAL-01M3RY6G385D41W5SNF1C85RRS`.
**Date:** 2026-09-30. **Base:** `b45a94273`.
**Branch:** `feat/nemotron-omni-vl` (developer direction for this campaign:
one branch per item, local commits only, no pull request). The spec commit
precedes the implementation commits on the same branch.

## Now

`ACTIVE`. The image path is implemented and reachable: both architecture names
register, the BF16 checkpoint's language tower loads through the NemotronH
loader under `language_model.`, `encode_mm` runs the RADIO tower and `mlp1`,
`embed_mm` splices the rows, the paged NemotronH forward reads
`inputs_embeds`, and `mm_chat_nano_nemotron_vl.cpp` serves `image_url` parts.
Next: the end-to-end token gate on `dgx:gpu0` (see `## Owed`).

## Scope

In scope, image modality only:

1. The RADIO vision tower (`C-RADIOv4-H`, ViT-H/16, 32 blocks, width 1280,
   16 heads, MLP 5120, 4 CLS + 6 register tokens, CPE positional table
   128 x 128).
2. The projector `mlp1`: RMSNorm(5120, eps 1e-5) -> Linear(5120 -> 20480, no
   bias) -> ReLU squared -> Linear(20480 -> 2688, no bias), after the v2 pixel
   shuffle (downsample ratio 0.5).
3. The dynamic-resolution image processor (`DynamicResolutionImageTiler`):
   the per-image patch-grid budget, the antialiased bicubic resize, the
   `(x/255 - mean)/std` normalization, and the patchify.
4. The prompt expansion: each `<image>` becomes
   `<img>` + `<image>` x N + `</img>`, with the embedding rows on the `<image>`
   positions only.
5. The registration of `NemotronH_Nano_VL_V2` and
   `NemotronH_Nano_Omni_Reasoning_V3`, whose language tower is the existing
   `NemotronHForCausalLM` forward under the `language_model.` prefix.
   Upstream maps two more names to the same class (`registry.py:512-515` @
   `e126687a9a`): `NemotronH_Super_Omni_Reasoning_V3` and
   `NemotronH_Omni_Reasoning_V3`. They are not registered here (see
   `## Owed`).

Out of scope, and refused by name: audio (`sound_encoder`, `sound_projection`,
`<so_embedding>`), video (`video_embedder`, `<video>`, EVS pruning), and the
static-tile (non-dynamic) InternVL tiling arm, which the released Omni
checkpoints do not select (`min_num_patches` is in `vision_config.args`).

## Upstream chain

The authoritative vLLM parity pin is `a7c23ac96d` (`.agents/upstream-sync.md`,
`vllm_commit`, advanced by `4f11dfc10`). The port was read, and every
`file:line` in this spec and in the code is cited, at `e126687a9a`, the pin
several records on `main` still name. The full diff between the two revisions
of the files this port mirrors (`git diff e126687a9a a7c23ac96d --
vllm/model_executor/models/{radio,intern_vit,nano_nemotron_vl}.py
vllm/transformers_utils/{configs/radio,processors/nano_nemotron_vl}.py
vllm/ir/ops/layernorm.py`) was read on 2026-09-30. It touches type
annotations, docstrings, tuple construction, the LoRA / dummy-input plumbing,
the processor call path for audio in video, the audio input construction and
the static-tile image arm, plus about 25 type-narrowing asserts that hold for
every valid input: among them `nano_nemotron_vl.py:478` (dynamic-tiler
placeholder replacement), `:1158` and `:1164-1165` (dynamic arm), the
`multimodal_config is not None` asserts in `__init__` and `load_weights`, a
`torch.is_tensor` to `isinstance(..., torch.Tensor)` swap on the video path,
and `radio.py:551` (`assert self.img_size is not None`) in the hunk where
`_init_img_size` now expands an int `patch_size` itself instead of the caller
wrapping it in `to_2tuple` (`radio.py:577-594`, same tuple). In `nano_nemotron_vl.py`, cited at `a7c23ac96d`,
that is: `SupportsLoRA` on `NemotronH_Nano_VL_V2`, covering the language model
only, with the video indicator tokens embedded by the base weights; the
`video_embeds` input mode, now parsed and passed through; a
`kwargs.pop("truncation", None)` in `get_hf_processor`; the
`_get_hf_processor_text` to `_get_hf_mm_text` rename (`:374`) and the `audio`
to `audios` key rename in the new `_get_hf_mm_inputs`;
`_extract_audio_from_videos` copying the metadata list, starting from
`MultiModalDataItems()` and returning `new_mm_items` (`:679-697`); the
`use_audio_in_video` bypass in `apply()` storing `_apply_hf_processor` in
`mm_res` and passing `mm_res=` to `_maybe_apply_prompt_updates`, with the
`_postprocess_prompt` call removed (`:767-787`); the audio inputs built from
three named fields instead of `**kwargs` (`:1496-1507`); the static-tile arm
dropping `**kwargs` (`:1172-1177`); `image_embeds is not None` in place of the
walrus truthiness test in `_parse_and_validate_image_input`; and an
`isinstance` test in place of the `self.dynamic_resolution` flag to select the
dynamic arm (`:1529`). None of these touches the dynamic-resolution
pixel-image path this port implements: the renames, the audio-in-video path,
the audio fields and the LoRA embedding are audio and video, the walrus change
is the `image_embeds` arm, the static-tile arm is not ported, the dispatch
change selects the same arm for the same inputs, and the dynamic arm moved from
`**kwargs` to the same two named fields and gained type asserts that pass for
valid inputs. No image-path semantics moved.
The torch reference is v2.11.0, for the two `F.interpolate` calls the
processor and the tower make.

## Port map

| Ours | Upstream |
|---|---|
| RADIO patch generator, CPE table, CLS/register tokens | `vllm/model_executor/models/radio.py:109-421` |
| `Im2Patches`, `ViTPatchLinear` | `radio.py:424-461` |
| RADIO block (`norm1 -> attn -> +`, `norm2 -> mlp -> +`) | `radio.py:470-520`, `intern_vit.py:145-350` |
| per-image mask and `_extract_final` | `radio.py:579-604`, `:745-774` |
| RADIO config (ViT-H dims, eps 1e-6, gelu, qkv bias) | `vllm/transformers_utils/configs/radio.py:12-110` |
| `pixel_shuffle` v2, `pixel_shuffle_dynamic_res`, `extract_feature_dynamic` | `nano_nemotron_vl.py:1012-1058` |
| `mlp1` | `nano_nemotron_vl.py:955-976` |
| `get_vit_model_from_radio_config` | `nano_nemotron_vl.py:1568-1598` |
| `DynamicResolutionImageTiler` | `vllm/transformers_utils/processors/nano_nemotron_vl.py:252-570` |
| `_bicubic_resize_and_normalize` (antialiased bicubic) | `processors/nano_nemotron_vl.py:61-83` |
| `get_image_repl` | `processors/nano_nemotron_vl.py:1086-1098` |
| weight-name mapping (`language_model.backbone` -> `language_model.model`) | `nano_nemotron_vl.py:904-908`, `:1499-1566` |

The resize is torch `F.interpolate(mode="bicubic", align_corners=False,
antialias=True)`. Its kernel is `aten/src/ATen/native/cpu/UpSampleKernel.cpp`
(the `_upsample_bicubic2d_aa` path, `a = -0.5`, support scaled by the
downscale factor, weights normalized per output pixel, f32 throughout). It is
NOT Pillow's resize (`pil_resize.h`), which rounds through uint8 between the
two passes.

## Our baseline

At `b45a94273`: `NemotronHForCausalLM` is registered and forwards (paged and
host), but nothing registers either wrapper name, no RADIO tower exists, and
the NemotronH loader and forwards take neither a name prefix nor
`inputs_embeds`.

## Dependencies

- The NemotronH language tower (`.agents/specs/nemotron-h-model.md`), reused
  unchanged except for two additive seams: a loader name prefix and an
  `inputs_embeds` input on both forwards.
- The multimodal runner seam (`encode_mm` / `embed_mm`,
  ENG-MM-INPUT-PIPELINE P2) and the per-architecture chat seam registry.
- The shared ops `vt::MatmulBT`, `vt::LayerNorm`, `vt::GeluErf`,
  `vt::AttentionDenseFlash`, `vt::RmsNorm`, `vt::MoeRelu2`.

## Design

- `include/vllm/model_executor/models/radio.h` + `src/.../radio.cpp`: the tower,
  composed from `vt::MatmulBT`, `vt::Add`, `vt::LayerNorm`, `vt::GeluErf`,
  `vt::AttentionDenseFlash`, and the shared merged-QKV split, exactly as the
  Muse Glimmer and Qwen3-VL towers are. bf16 production dtype, f32 for the
  numeric gate.
- `include/vllm/model_executor/models/nano_nemotron_vl.h` + `.cpp`: the pixel
  shuffle and the projector.
- `include/vllm/multimodal/nano_nemotron_vl_processor.h` + `.cpp`: the tiler,
  the antialiased bicubic resize, normalization, patchify, and the placeholder
  expansion through `multimodal::MakeTokenTripleReplacement`.
- `src/.../nano_nemotron_vl_registry.cpp`: the registration, `encode_mm`,
  `embed_mm`, and the forward, which delegates to the NemotronH forward.

## Risks/decisions

Each risk below is silent, so each is a gate.

1. The CPE table is interpolated to a SQUARE `max(h, w)` grid and then cropped
   (`radio.py:401-410`). Interpolating straight to `(h, w)` keeps every shape.
2. The CLS and register tokens are prepended per image and stripped per image
   (`num_skip` = 10). Stripping 4 or 0 keeps the row count wrong only by a
   constant that a length check can miss when it is paired with a wrong grid.
3. Pixel shuffle v2 permutes `(0, 1, 3, 2, 4, 5)`. The v1 order transposes the
   image and keeps the shape and the multiset of values.
4. The resize is antialiased. A four-tap bicubic without support scaling
   differs on every downscale.
5. The vision output is cast to bf16 before the pixel shuffle
   (`nano_nemotron_vl.py:1055`), and the projector runs in bf16.

## Tests to port

From `tests/models/multimodal/test_nano_nemotron_vl.py` @ `e126687a9a`:

| Upstream case | Here |
|---|---|
| `test_nano_nemotron_vl_skips_multimodal_weights_in_text_only_mode` (:77-96) | ported to `test_nano_nemotron_vl_registry.cpp`: a `language_model_only` load of a checkpoint with NO `sound_config` and a `sound_encoder.encoder.weight` tensor succeeds, reads zero vision and `mlp1` tensors (`NanoNemotronVLVisionLoadOf`), and `encode_mm` refuses by name. Adapted: a real tiny checkpoint instead of mocked modules, and the "not inspected" assertion is the load's own vision count |
| `test_nano_nemotron_vl_loads_vision_weights_without_sound_encoder` (:99-121) | ported, same file: no `sound_config`, no sound tensor, the load succeeds |
| `test_nano_nemotron_vl_requires_sound_encoder_for_sound_weights` (:124-136) | ported, same file: a sound tensor without `sound_config` is refused |
| `test_extract_audio_from_videos_*` (:150-182) | not applicable: audio is refused by name |

`tests/models/multimodal/pooling/test_radio.py` compares against the HF remote
code on a GPU with a downloaded checkpoint and does not port to this harness as
written. Its assertion (the tower matches a reference on the real weights) is
carried by the REAL arm of `test_nano_nemotron_vl_vision.cpp`.

## Gates

- Stage gates against a torch transcription of the pinned vLLM formulas
  (`scripts/mm/nano_nemotron_vl_ref.py`), on the REAL checkpoint tensors
  (`vision_model.*`, `mlp1.*` of `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`
  at `e5e9932441de940c9a62185c870ea5bcd4cd24e2`): processor pixels, tower
  output, projector output. f32 arm with a tight bound, bf16 arm with the bf16
  envelope. The committed fixture holds reduced inputs and a digest, not the
  weights.
- Synthetic unit gates that need no checkpoint: CPE interpolation, pixel
  shuffle order, tiler geometry, placeholder expansion, config parse of the
  real released `config.json` (committed fixture).
- A reachability gate that enters through `ModelRegistry` (encode_mm and
  embed_mm through the registered architecture name).
- Mutation: each risk in §4 is mutated by the fresh reviewer.

## Work breakdown

1. Spec and records (this file).
2. Processor: tiler, antialiased resize, normalize, patchify, expansion.
3. RADIO tower and `mlp1`, gated per stage on synthetic and real weights.
4. Registration, loader prefix, `inputs_embeds` in the NemotronH forwards,
   `encode_mm` / `embed_mm`, chat seam; reachability gate through the runner.
5. Owed: the end-to-end token gate on `dgx:gpu0`.

## Stop conditions

- A language-tower format this tree cannot load (the NVFP4 Omni checkpoint's
  per-module scheme differs from Nemotron 3.5 Lightning's: FP8 `o_proj` and FP8
  shared experts, bf16 `lm_head`). If the loader cannot represent it, it is
  refused by name and recorded under `## Owed`; it is not approximated.
- The end-to-end token gate needs the pinned vLLM oracle and a GPU lease for a
  23 GB (NVFP4) or 62 GB (bf16) checkpoint. If that cannot run in this session,
  it stays owed.

## Evidence (2026-09-30, CPU, this worktree)

Reference: `scripts/mm/nano_nemotron_vl_ref.py`, torch `2.11.0+cu130`.
Real tensors: the `vision_model.*` and `mlp1.*` entries of shard 1 of
`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` @ `e5e99324`, cut by HTTP
range into one 1 627 048 984-byte file (sha256 `314151d13e98016226d534a707e176f4`
`9066e6fe3ad42dfb14f8501b86178383`), not committed.

| Gate | Result |
|---|---|
| `test_nano_nemotron_vl_vision` (synthetic, CI) | 12/12 cases: tiler params exact, resize f32 max abs 2e-5 bound, bf16 pixels <= 1% one-ulp, CPE square-then-crop 1e-6, tower and projector f32 1e-4 and bf16 3e-2, shuffle order (every output row, so the v1 transpose is distinguished), expansion, template prefix, both released configs |
| same, REAL arm (333x517 image, `max_model_len` 16384) | grid 26x42, 273 rows exact; pixels 344/838656 one-ulp; tower f32 rel 2.36e-5, bf16 4.71e-2; projector f32 4.40e-6, bf16 5.85e-3; end-to-end worst row cosine 0.99776 |
| `test_nano_nemotron_vl_registry` (CI) | tiny Omni checkpoint loads under the released name; stray tensor refused; image decode through `GPUModelRunner` == independent host reference (4 tokens) and != the text-only decode; a text-only load reads 0 vision/`mlp1` tensors and skips a `sound_encoder.*` tensor with no `sound_config`; audio refused |
| same, STRUCTURAL on the BF16 index | 6243/6243 language tensors claimed, 390/390 vision, 716 deferred by name, 0 unclaimed |
| same, STRUCTURAL on the NVFP4 index @ `16993199` | FAILS as recorded: 5946 language tensors unclaimed, 60 enumerated-but-not-shipped (`ISSUE-LOCAL-01M3S07X6Y4ADHHQ02FYMR4RXF`) |
| `test_nano_nemotron_vl_mm_chat` (CI, server build) | 6/6: seams for both names, one- and two-image prefix and expansion, text passthrough, refusal without `max_model_len`, and the tiler's text length taken with no special tokens (`add_special_tokens=False`, processors/nano_nemotron_vl.py:689-692) on a BOS-prepending tokenizer |
| `test_serve_nano_nemotron_vl_mm` (CI, server build) | 3/3: the tiny checkpoint as a model directory through the REAL `VllmServerMain` (seam wired, not UNAVAILABLE) and through `vllm_engine_load` + `vllm_chat` (an image request answered, 17 prompt tokens at the default max_model_len 128; at max_model_len 16 the tiler budget 16 - 11 - 4 = 1 shrinks the grid to 2x2 and the prompt is 14 tokens). Deleting either production `mm_ctx.max_model_len` assignment (`server_main.cpp`, `vllm_c.cpp`) reddens its case. The C-ABI case proves the ENGINE's value is carried: a constant `1 << 20` in `vllm_c.cpp` reddens the max_model_len 16 case. The server case proves only that the value is supplied, because the server exits on the unbindable port before it serves a request and the install announcement does not print the budget |
| no-regression | `test_nemotron_h_paged_forward` 13/13, `test_nemotron_h_scaffold` 14/14, `test_nemotron_h_loader` 4/4, `test_model_registry` 24/24 |

## Owed

- The end-to-end image token gate against pinned vLLM (`dgx:gpu0` lease,
  oracle run of `NemotronH_Nano_Omni_Reasoning_V3`). Tracked by
  `ISSUE-LOCAL-01M3RY6G385D41W5SNF1C85RRS`.
- Audio (`sound_encoder`) and video (`video_embedder`, EVS) arms. Refused by
  name. Tracked by the same issue.
- The GGUF arm of the language tower (inherited from `NemotronHForCausalLM`,
  `.agents/specs/nemotron-h-model.md` §5b W7).
- `ISSUE-LOCAL-01M3RZ6SFVDJRX1Z5GZG1CKJ7G`: `LoadHfConfig` cannot parse a
  config.json that carries the Python JSON literal `Infinity`, which the 12B
  `NemotronH_Nano_VL_V2` release does (`llm_config.time_step_limit`). Found by
  this row, not fixed by it (developer direction: file, do not fix, anything
  outside the item). The refusal test sanitizes the literal before parsing.
- `ISSUE-LOCAL-01M3S07X6Y4ADHHQ02FYMR4RXF`: the NVFP4 Omni language tower. The
  NemotronH loader does not resolve the ModelOpt scheme per module, so that
  checkpoint is refused at load (row-owned issue, not an `_owed` one).
- The static InternVL tiling arm (`NemotronH_Nano_VL_V2` 12B), refused by name
  at config parse. Tracked by `ISSUE-LOCAL-01M3RY6G385D41W5SNF1C85RRS`.
- `NemotronH_Super_Omni_Reasoning_V3` and `NemotronH_Omni_Reasoning_V3`
  (`registry.py:514-515` @ `e126687a9a`), the same upstream class. Not
  registered: no public checkpoint exists at the pin (upstream's
  `tests/models/registry.py:1217-1223` points both at the Nano Omni repository
  with `is_available_online=False`), so no config shows whether their language
  tower is one the NemotronH loader and forward accept. A load under either name
  is refused by the registry as an unknown architecture. Tracked by
  `ISSUE-LOCAL-01M3RY6G385D41W5SNF1C85RRS`.
