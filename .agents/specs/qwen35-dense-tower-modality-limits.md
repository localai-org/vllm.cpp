# The Qwen3.5 dense tower loads at zero modality limits — ISSUE-LOCAL-01M3QEHEDB23V1VF0VWWDR4DM9

L3 of [#607](https://github.com/mudler/vllm.cpp/issues/607) (tower skip) landed for
Qwen3-VL, MuseGlimmer, the `clip` projector and DeepSeek-V4. The Qwen3.5 **dense**
arm still read `model.visual.*` whenever the checkpoint index named it, so a
text-only serve (`--language-model-only`, sugar over `--limit-mm-per-prompt` with
every modality at 0) paid for the tower: 333 bf16 tensors, `host_copy` 7.621 ->
6.763 GiB at load on Qwen3.6-27B dense, same binary both arms.

Issue: [ISSUE-LOCAL-01M3QEHEDB23V1VF0VWWDR4DM9](../issues/ENG-MM-INPUT-PIPELINE/ISSUE-LOCAL-01M3QEHEDB23V1VF0VWWDR4DM9.md).
Owning row: `ENG-MM-INPUT-PIPELINE` ([engine-matrix.md](../engine-matrix.md)); the
row's contract is [multimodal-track.md](multimodal-track.md) §1.5, where L3 is the
tower skip.

## Premise, grounded

| Where (line anchors at this branch's base, `b45a94273`) | What |
|---|---|
| `include/vllm/model_executor/models/interfaces.h:66`, `src/vllm/model_executor/models/interfaces.cpp:9` | `SkipTowerForModalities(mm_config, modalities)` — the contract every other tower skip uses. |
| `src/vllm/model_executor/models/qwen3_vl.cpp:463` | Qwen3-VL skips `{"image", "video"}`. |
| `src/vllm/model_executor/models/muse_glimmer_weights.cpp:803` | MuseGlimmer skips the same pair. |
| `src/vllm/model_executor/models/deepseek_v4_mm.cpp:93,122` | DeepSeek-V4 skips `{"image"}`. |
| `include/vllm/model_executor/models/qwen3_5_dense.h:338` | `LoadQwen3_5Dense` took only `(shards, config, load_queue)` — no modality limits. |
| `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:1185` | `w.visual = LoadQwen3_5DenseVision(shards, config)` runs whenever the index names the tower. |
| `src/vllm/model_executor/models/qwen3_5_dense.cpp:128` | The production registry call, `LoadQwen3_5DenseModel` -> `LoadQwen3_5Dense(*source.safetensors, config, source.load_queue)`. |
| `tests/vllm/models/test_qwen3_5_dense_vision.cpp` | A 602-line suite that is in **no `CMakeLists.txt` target** on the base: it has never been built. |

Upstream anchor, from #607: `--language-model-only` sets every modality limit to
`0` (`vllm/config/multimodal.py:78,321-327` at the pin #607 read) and the model is
built with the tower uninitialised (`vllm/model_executor/models/interfaces.py:293`).
Re-reading those two anchors at the current pin (`a7c23ac96d`) is **owed**: this box
has no `VLLM_SOURCE` checkout, so nothing here re-derives them.

## Design

Route the engine's limits into the loader and reuse the existing contract rather
than inventing a second one:

- `LoadQwen3_5Dense` gains `const MultiModalConfig* mm_config = nullptr`, so every
  non-engine caller (tests, KEV) keeps the old behaviour.
- When `SkipTowerForModalities(mm_config, {"image", "video"})` holds, the loader
  leaves `model.visual.*` unread and sets `Qwen3_5DenseWeights::vision_skipped`.
- `Qwen3_5DenseLoadedModel::skipped_towers()` reports
  `kVisionTowerStageName`, which is what the server's NOT-loaded line reads.
- The production path passes `source.multimodal` at
  `qwen3_5_dense.cpp:128`.

`vision_skipped` is deliberately distinct from `has_visual == false`: a text-only
checkpoint differs from a multimodal checkpoint whose tower was skipped, and the
report must not claim a skip that did not happen.

## Tests

`tests/vllm/models/test_qwen3_5_dense_vision.cpp` gains
`qwen3_5_dense_loader_leaves_the_tower_unread_at_zero_limits`, and — because the
file had no target — `tests/CMakeLists.txt` gains the
`vllm_cpp_add_test(test_qwen3_5_dense_vision ...)` registration in the same
change. The case is its own negative control:

1. An incomplete-tower shard loaded with **no limits** fails BY NAME on
   `model.visual.` — the tower is reached.
2. The same shard loaded with `language_model_only = true` must NOT name
   `model.visual.` — the failure moves to the backbone, which is what proves the
   tensors were never read.
3. `limit_per_prompt = {{"image",0},{"video",0}}` — the other spelling upstream
   treats identically — must behave like (2).

What the gate does not prove: the GiB saving, and the production call site. The
server measurement below is the production evidence; a unit mutation of
`source.multimodal` is invisible to a test that calls the loader directly.

## Evidence

- Load-time byte saving on **Qwen3.6-27B dense**, `--language-model-only`,
  `VT_LOAD_STATS`, same binary both arms, 2026-09-27: `host_copy`
  **7.621 -> 6.763 GiB**, the server reporting `vision_tower` in its NOT-loaded
  line. Recorded in [docs/FEATURES.md](../../docs/FEATURES.md).
- The new case RED before the change (the zero-limit arm names `model.visual.`)
  and GREEN after; the rest of the suite unchanged.

## Gates

- `ctest --test-dir build -R test_qwen3_5_dense_vision` — the new case, plus the
  pre-existing 602-line suite that this change makes runnable for the first time.
- The loader's neighbours: `test_qwen3_5_dense_load_residency`,
  `test_qwen3_5_dense` and the `test_qwen3_5_moe_vision` sibling.
- The full `ctest --test-dir build`.
- `scripts/agent-preflight.sh --staged`.

## Owed

- Re-read the two upstream anchors at the current pin when a `VLLM_SOURCE`
  checkout is available.
- The test file and its target belong to the `MODEL-QWEN35-DENSE-VL-EXL3` row
  (`.agents/specs/tenstorrent-qwen35-dense-vl-exl3.md`). This change registers the
  target because the new case cannot run otherwise; that row still owes its own
  phases (the tower wired to a reachable VL entry point).
