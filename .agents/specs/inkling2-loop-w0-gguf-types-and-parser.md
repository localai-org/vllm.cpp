# SPEC — `NEW-CANDIDATE-PORTS` W0 (inkling2_loop): ggml Q4_1/Q5_1 types + the inkling2_loop parser and refusing loader row

- Owning row: `NEW-CANDIDATE-PORTS`. Issue: ISSUE-LOCAL-01M80WA9T38SWMMWYDJ9T938K8.
- Worktree `/tmp/vllm-inkling2-w0`, branch `row/NEW-CANDIDATE-PORTS-inkling2`, base `224564919`.
- Developer authorization: "Take all three" tasks in mk-C's CPU-parallel work split; this spec covers the W0 slice (tasks a and b). One PR for spec + implementation (developer-preferences.md, `NEW-CANDIDATE-PORTS` row claim answer 2026-11-20).

## Trigger / problem

The pinned artifact `mudler/Inkling-2-Mini-APEX-GGUF @ d85a58007995c3365eb2323b177937ce4d20b8f0`, mounted at `/mnt/models/inkling2-mini-apex-gguf/Inkling-2-Mini-APEX-Balanced.gguf` (20,684,174,784 bytes), carries `general.architecture = "inkling2_loop"`, 59 KVs and 704 tensors. Its full header/tensor walk is committed at `docs/bench-evidence/new-candidate-inkling2-20261121/apex-balanced-walk.txt`. Fourteen `blk.N.ffn_down_exps.weight` tensors use ggml type id 7 (Q5_1). Our reader's `FindGgmlTraits` (`src/vllm/model_executor/model_loader/gguf_reader.cpp:200-409`) jumps case 6 -> case 8 and never tabulates 7, so `GgufFile::Open` throws `gguf: tensor "..." has unknown ggml type id 7` at `gguf_reader.cpp:524`, refusing the whole 20 GB file over 14 tensors. The architecture also has no config builder or arch-table row, so `HfConfigFromGgufDispatch` (`src/vllm/entrypoints/model_loader.cpp:1316`) refuses it as unrecognized. The broader support path is tracked by ISSUE-LOCAL-01M7ZYAY4W7STQCBPSF6NQRF78; this row owns its concrete steps (1) and (2).

## Upstream anchors (oracle)

- Primary oracle: vLLM PR [#69028](https://github.com/vllm-project/vllm/pull/69028) "[Model] Add Inkling 2 Mini", merged 2026-11-21T16:19:34Z (`Inkling2LoopForConditionalGeneration`, `model_type: inkling2_loop`). Read with `gh pr diff 69028 --repo vllm-project/vllm`; PR body's Config section requires `model_type: inkling2_loop` and `architectures: [Inkling2LoopForConditionalGeneration]`.
  - `vllm/models/inkling2/configs.py` (PR diff): `INKLING2_LOOP_ARCHITECTURE = "Inkling2LoopForConditionalGeneration"`; `class Inkling2LoopMMConfig`, `model_type = "inkling2_loop"`; `Inkling2ModelConfig.loop_config` requires integer fields `{core_start, core_end, num_passes, gate_rank}` plus boolean `is_varloop` (`:108-122`), enforces `0 < core_start < core_end <= num_hidden_layers` (`:119-120`) and `gate_rank >= 1` (`:121-122`); `check_loop_options` requires at least 2 passes for a fixed loop (`:129-138`); `Inkling2MMConfig.__init__` requires looped checkpoints to use `model_type == "inkling2_loop"` (`:204-210`) and `architectures == [self.architecture]` (`:212-216`).
  - `vllm/model_executor/models/registry.py` (PR diff): `"Inkling2LoopForConditionalGeneration": ("vllm.models.inkling2", "Inkling2LoopForConditionalGeneration")`.
  - `vllm/transformers_utils/config.py` (PR diff): `inkling2_loop="Inkling2LoopMMConfig"`.
  - Upstream test `tests/models/inkling2/test_model.py::test_loop_serving_options_validation` (PR diff) is the applicable config test; its forward/GPU cases are port-row scope and NOT ported here.
- GGML wire geometry: llama.cpp `ggml-common.h` `block_q4_1 = f16 d + f16 m + QK4_1/2 qs` = 2+2+16 = 20 bytes, QK4_1 = 32; `block_q5_1 = f16 d + f16 m + u8 qh + QK5_1/2 qs` = 2+2+1+16 = 21 bytes, QK5_1 = 32. (Same table the existing cases 2, 6, 8 cite at `gguf_reader.cpp:214-225`.) `GGML_QUANT_SIZES` in `gguf-py/gguf/constants.py` agrees.

## Scope

IN:
1. Two rows in `FindGgmlTraits`: id 3 `{32, 20, "Q4_1"}` and id 7 `{32, 21, "Q5_1"}`, so the reader can parse the header (no dequantization — this row does not run the forward).
2. A CPU unit test over the real reader and the pinned artifact path: type-id geometry, `GgufFile::Open` succeeds, 704 tensors, arch `inkling2_loop`, 14 type-7 tensors, `token_embd.weight` shape; clean 77-style skip when the artifact is absent.
3. `Inkling2LoopHfConfigFromGguf` (header + TU), mapping the walk's standard fields by name, mirroring upstream field names and validations, carrying every unmapped KV verbatim via the shared `gguf_config::CarryUnmapped`.
4. ONE loader row `{"inkling2_loop", &vllm::Inkling2LoopHfConfigFromGguf}` in `kGgufArchArms` and one forward-boundary refusal in the GGUF branch, exactly the olala/hy_v4_flash pattern (`model_loader.cpp:1293-1294`, `:2913-2915`).
5. CPU tests for the parser (field mapping, loop_config, unmapped carry, wrong-arch refusal, forward-refusal text), registered in CMake with `SKIP_RETURN_CODE 77` via `vllm_cpp_add_test`.
6. Evidence under `docs/bench-evidence/new-candidate-inkling2-w0-20261122/` (flat), plus FEATURES/USAGE parse-status rows.

OUT: the looped forward and its kernels, the per-pass KV/conv caches, the message controllers, DSpark, the name-map census, dequantization kernels for Q4_1/Q5_1, tokenizer behavior beyond verbatim carry, any GPU leg, and the vLLM pin advance (see Owed).

## Design

- ggml ids 3 and 7 are added to the switch in `FindGgmlTraits`, each with a comment citing the block layout and the artifact that needs 7. No other reader behavior changes: the block-size divisibility and span bounds checks at `gguf_reader.cpp:528-572` apply unchanged.
- The parser header follows `olala_gguf_config.h`: `kInkling2LoopGgufArch = "inkling2_loop"`, `IsInkling2LoopGguf`, `Inkling2LoopHfConfigFromGguf`, `Inkling2LoopForwardRefusal`. The TU uses `src/vllm/model_executor/models/gguf_config_kv.h` helpers (`ReqInt`, `OptInt`, `ReqFloat`, `KvToJson`, `CarryUnmapped`, `HasTensor`).
- Field map (GGUF key -> HfConfig / raw): `block_count` -> num_hidden_layers; `embedding_length` -> hidden_size; `feed_forward_length` -> intermediate_size; `attention.head_count` (per-layer array, 24 entries of 24) -> num_attention_heads = first entry, full array carried in `raw["inkling2_loop_attention_head_count"]` (hy_v4_flash precedent); `attention.head_count_kv` (array of 8) -> num_key_value_heads = first entry; `attention.key_length` -> head_dim; `expert_count` -> num_experts; `expert_used_count` -> num_experts_per_tok; `expert_feed_forward_length` -> moe_intermediate_size; `context_length` -> max_position_embeddings; `attention.sliding_window` -> sliding_window; `attention.layer_norm_rms_epsilon` -> rms_norm_eps; `vocab_size` (also cross-checked against `token_embd.weight` shape[0]) -> vocab_size.
- `loop_config` is built in `c.raw["loop_config"]` with UPSTREAM field names: `core_start`=`loop.start` (6), `core_end`=`loop.end` (18), `num_passes`=`loop.count` (3), `gate_rank`=`loop.gate_rank` (64), `is_varloop`=`loop.varying` (false). Validation mirrors `configs.py:108-138` on the parsed integers: exact field set/type, `0 < core_start < core_end <= num_hidden_layers`, `gate_rank >= 1`, and at least 2 passes for a fixed loop. The upstream `attn_res_normalize_values` requirement (`:123-126`) is NOT enforced because the GGUF container has no such KV; recorded as a deviation. The nested HF `text_config` structure stays port-row scope; the flat HfConfig plus carried loop_config is what this build's loader consumes.
- `c.model_type = "inkling2_loop"`, `c.architectures = {"Inkling2LoopForConditionalGeneration"}` (mirroring the PR's Config section, not guessed).
- The refusal text names `Inkling2LoopHfConfigFromGguf`, row `NEW-CANDIDATE-PORTS`, and "Refused at the forward boundary — nothing ran.", exactly like `OlalaForwardRefusal` (`olala_gguf_config.cpp:27-32`) and `HyV4FlashForwardRefusal` (`hy_v4_flash_gguf_config.cpp:70-77`).

## Tests (TDD)

1. Red: `tests/vllm/models/test_inkling2_loop_gguf.cpp` with the type-id cases first. Before the table edit, `GgmlTraits(3)`, `GgmlTraits(7)` and `GgufFile::Open(APEX_PATH)` throw "unknown ggml type id". Capture the red to `docs/bench-evidence/new-candidate-inkling2-w0-20261122/red.txt`.
2. Minimum change: add cases 3 and 7 to `FindGgmlTraits`; focused green for the type cases.
3. Parser tests (same binary): field mapping, `loop_config` values + validation, unmapped carry (`sliding_window_pattern` bool array, `loop.*`, tokenizer.* all carried; mapped keys not carried), wrong-arch refusal (hermetic synthetic GGUF built with `tests/vllm/gguf_builder.h`), forward-refusal text. Port of upstream `test_loop_serving_options_validation` where applicable.
4. `std::exit(77)` on a missing artifact (CTest `SKIP_RETURN_CODE 77`, wired by `vllm_cpp_add_test` at `tests/CMakeLists.txt:38`) so CI without `/mnt/models` reports Skipped, never a false green.

## Gates / evidence

- Focused: `cmake -S . -B /tmp/opencode/vllm-inkling2-build -G Ninja` (CPU only) and `cmake --build /tmp/opencode/vllm-inkling2-build --target test_inkling2_loop_gguf -j4`; run `scripts/run-doctest-selected.sh build/tests/test_inkling2_loop_gguf` (or the direct binary) with the artifact mounted; log to `docs/bench-evidence/new-candidate-inkling2-w0-20261122/focused.log`.
- Full gate: `scripts/agent-preflight.sh --staged`; `python3 scripts/check-agent-record.py --report` (rot 0 before push).
- PR body: `python3 scripts/agent-pr-body.py --pr <N>` exit 0.

## Risks / decisions

- 20 GB mmap in a CPU test: read-only header parse; 14 type-7 spans are bounds-checked but never dequantized. The test asserts tensor metadata only.
- Type ids 3/7 are standard ggml, NOT fork extensions: no oracle record needed; the llama.cpp mainline `GGML_QUANT_SIZES` is the geometry source.
- `head_count`/`head_count_kv` are per-layer arrays: a scalar HfConfig cannot represent them, so first entry + verbatim carry (hy_v4_flash precedent, `hy_v4_flash_gguf_config.cpp:102-108`).
- Unknown ggml type id 43 exists in hy4 shard 2 (unrelated, already recorded); this row does not touch it.

## Stop conditions

Stop and report `NEEDS_CONTEXT` if the artifact is absent AND no skip can be demonstrated; `NEEDS_DECISION` if vLLM PR #69028's config naming conflicts with this build's HfConfig in a way the carry-verbatim pattern cannot hold. Do not stop at a correctable test failure. Do not advance the vLLM pin, touch other worktrees, or push anywhere but `mine`.

## Owed

- Advance the vLLM pin past PR #69028 (merged 2026-11-21) and re-run the inkling2_loop parity gate; out of scope here. The local pin in `.agents/upstream-sync.md` is unchanged.
- The looped forward, Q4_1/Q5_1 dequant kernels, name-map census, DSpark, and the full nested `text_config.loop_config` HF shape are port-row work.

## Now

ACTIVE — W0 implementation in `/tmp/vllm-inkling2-w0`; issue ISSUE-LOCAL-01M80WA9T38SWMMWYDJ9T938K8 open until the PR lands.

## Git integration

One PR carrying spec + implementation; push `row/NEW-CANDIDATE-PORTS-inkling2` to `mine`; open PR against `localai-org/vllm.cpp:main`; human developer merges.
