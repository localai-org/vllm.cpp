# SPEC: `MODEL-TEV1` Phase 7: stop on the tokenizer's EOS when the checkpoint names none

Row: `MODEL-TEV1`
Issue: `ISSUE-LOCAL-01M3RTGVTN34YQFBR1KZH117XA`

The issue says the fix changes the stop set of every model and needs its own
spec and review. This is that spec. The parent spec is [`tev1.md`](tev1.md).

## Now

`DONE` on CPU (2026-09-30). See `## Outcome`.

## The defect

`InputProcessor` (src/vllm/v1/engine/input_processor.cpp:39-83) takes the
primary EOS from the TOP-LEVEL `config.json` `eos_token_id`, else from
`tok::Tokenizer::EosId()`, which reads only the `tokenizer.json`
post_processor. It then unions the sibling `generation_config.json` ids.

Both Tev1 checkpoints nest their text config (`Qwen3_5ForConditionalGeneration`,
`text_config.eos_token_id` = 248044 `<|endoftext|>`), ship no
`generation_config.json`, and use a ByteLevel post_processor that carries no id.
So the engine has NO stop id at all. A chat reply at `max_tokens=8` is
`A<|im_end|>\n<|endoftext|>...` with `finish_reason: length`.

## Upstream (vLLM @ `5559679229`)

- The primary EOS is `tokenizer.eos_token_id` (`renderers/base.py:310-317`),
  which HF resolves from `tokenizer_config.json` `eos_token`: `<|im_end|>`, 248046.
- The secondary ids come from `ModelConfig.try_get_generation_config`
  (`config/model.py:1525`, via `transformers_utils/config.py:1066-1090`):
  `generation_config.json` when it exists, else
  `GenerationConfig.from_model_config(config)`, which reads the text config.
  On both Tev1 checkpoints `transformers` 5.3.0 gives `eos_token_id` 248044.
- `update_from_generation_config` (`sampling_params.py:629-655`) adds the
  secondary ids to `stop_token_ids` unless `ignore_eos`.

vLLM therefore stops Tev1 on 248046 (primary) and on 248044 (secondary).

## Design (the smallest change that reaches upstream's stop set here)

1. `HfConfig` gains `tokenizer_eos_token`: the `eos_token` of the sibling
   `tokenizer_config.json` (a string, or an object with `content`). It is read
   in the same place, and under the same condition, as the sibling
   `generation_config.json`.
2. When the sibling `generation_config.json` does NOT exist, and only then,
   `generation_config_eos_ids` is filled from the text config's `eos_token_id`
   (int or list). This is `from_model_config`. A present file keeps today's
   behavior exactly.
3. `InputProcessor` keeps today's primary resolution. Only when both of today's
   sources are empty does it resolve `tokenizer_eos_token` through the loaded
   tokenizer (it must encode to exactly one token) and use that id as the
   primary EOS.

The primary EOS is not moved to the tokenizer for a model that already resolves
one. That would mirror upstream more fully, but it would change the primary id,
and therefore `stop_reason` and `ignore_eos` behavior, on checkpoints that
work today. That difference stays a recorded deviation, and a test pins it.

## Who changes

Only a checkpoint with no top-level `eos_token_id`, no post_processor EOS, and
(for item 2) no `generation_config.json`. Every in-tree C++ test that builds an
`HfConfig` in memory reads no sibling file, so it cannot change. A model that
lists its ids keeps its stop set byte for byte.

## Tests

- Pin today's behavior first, before the change: a `config.json` with a
  top-level `eos_token_id` plus a `generation_config.json` and a
  `tokenizer_config.json` whose `eos_token` is a DIFFERENT id. The primary EOS
  and `stop_token_ids` must stay exactly the config's.
- Red then green: the Tev1 shape (nested `text_config.eos_token_id`, no
  generation config, tokenizer EOS `<|im_end|>`) gets primary 248046 and
  secondary 248044 in the synthetic vocab's ids.
- A present `generation_config.json` suppresses the text-config fallback.
- An `eos_token` that is not one token is ignored rather than guessed.
- Suites that must stay green: `test_input_processor`, `test_hf_config`,
  `test_capi`, `test_openai_api_server`, `test_loaded_engine_dense`,
  `test_llm_engine`, `test_tev1`, `test_nimble`.
- Real weights: both Tev1 checkpoints stop after the letter with
  `finish_reason: stop` and no `stop_token_ids` field.

## Stop conditions

- If any existing golden or strict gate changes, stop and report it; do not
  update the golden.

## Owed

- Moving the primary EOS to the tokenizer for every model (full upstream
  parity), with its own survey of affected checkpoints.
- `--tokenizer-config` (server) or `tokenizer_config_path` (C ABI) pointing
  elsewhere: the fallback reads the sibling file only. It matters only on the
  fallback path, where no other eos was found (fresh review of 53d58da46).

## Outcome

- Implemented as designed, with one change: the from_model_config ids live in
  their own `HfConfig::model_config_eos_ids` field instead of
  `generation_config_eos_ids`. Writing them into the existing field turned the
  existing strict test "absent generation_config.json leaves the list empty"
  (`test_hf_config`) red, and this spec's stop condition forbids changing an
  existing gate.
- Red then green: of the five new `test_input_processor` cases, the pinned
  "checkpoint that lists its ids" case and the "not one token" case passed
  before the change and after it; the Tev1-shape, object-form and
  generation-config-wins cases failed before (no primary eos) and pass after.
- No existing gate changed: `test_input_processor` 22/22, `test_hf_config`
  22/22, and the touched suites (`test_llm_engine`, `test_openai_*`,
  `test_runner`, `test_async_*`, `test_loaded_engine_dense`, `test_nimble`,
  `test_tev1`, `test_tev1_systemone`, `test_generation_config`,
  `test_qwen3_8_text_only`, `test_llama_*`) are green. Three suites fail on this
  branch for reasons that predate it: `test_capi` (GLiNER fixture,
  ISSUE-LOCAL-01M3RWMQD1RBNEGWZZRDSFTEPR), `test_model_loader_gguf` (a pinned
  architecture list that predates many registrations), and
  `test_bench_eos_chat_template` (vllm-bench has no `--no-ignore-eos`).
- The fresh review of 53d58da46 FAILED it on one finding: the from_model_config
  ids read the text config even when the outer config names an eos, so a nested
  checkpoint with a top-level eos and no generation_config.json gained the
  text config's id as a stop id. transformers lets the outer value win. Fixed
  in the repair commit, with a test that is red on 53d58da46 and green after.
- Real weights, CPU: both checkpoints (the 4B from its unmodified snapshot)
  answer the card's decision prompt with the letter alone, `finish_reason:
  stop`, `stop_reason: null` (the primary eos), with no `stop_token_ids`.
