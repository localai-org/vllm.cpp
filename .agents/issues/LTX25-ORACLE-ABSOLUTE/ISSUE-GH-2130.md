ID: ISSUE-GH-2130
Title: **`vllm_video_params.steps` ships in the ABI, the engine honours it, and no shipped client can set it, so every LTX-2.5 render silently runs the recipe default.** `include/vllm.h:1075` declares `int32_t steps`, `src/capi/vllm_c.cpp:1664` forwards it, and `src/vllm/multimodal/ltx2_video.cpp:4027` reads it — `int64_t steps = gen.steps > 0 ? gen.steps : recipe.num_inference_steps;` — with `allow_request_sigmas` true and `fixed_num_inference_steps` false (`include/vllm/model_executor/models/ltx2_pipeline.h:872-877`), so a request value is honoured rather than clamped. `examples/ltx2_gen/main.cpp:306-451` parses `--frames`, `--width`, `--height` and `--seed` and assigns `steps` nowhere, so `one_stage` at model version 2.5 always runs **30** (`ltx2_pipeline.cpp:1157` from `Ltx2Params24()` → `Ltx2Params23()`, where `:968` sets `num_inference_steps = 30`). WHAT IT COST: [#1864](https://github.com/mudler/vllm.cpp/issues/1864)'s reference render was taken at **8** steps (`tools/oracle/ltx2_oracle.py:88`), so [#1854](https://github.com/mudler/vllm.cpp/issues/1854)'s absolute comparison had a 3.75x denoise-budget confound on the one axis the CLI cannot reach — and it confounds in the direction that FLATTERS us, so a pass taken on it would be unearned. The reference is arm-matched on all four checkpoints, geometry, seed and prompt; this was the only unmatched axis. The `AGENTS.md` "Nothing lands dead" shape at the SEAM rather than in the engine: the capability is reachable through `include/vllm.h` and the thin ABI client that exposes every neighbouring field does not expose this one, and nothing detects it because the renders are correct, no refusal fires, and the only symptom is that every LTX-2.5 render in this tree has run one step count. FIXED IN FLOW: `--steps N` forwards to the existing ABI field and to nothing else; no engine code changes. Owner: row `LTX25-ORACLE-ABSOLUTE`, spec [`ltx25-oracle-absolute.md`](../../specs/ltx25-oracle-absolute.md)
Row: LTX25-ORACLE-ABSOLUTE
State: UNKNOWN
Kind: bug
GitHub: 2130
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:854`

### Frozen archive evidence

> | [#2130](https://github.com/mudler/vllm.cpp/issues/2130) | `LTX25-ORACLE-ABSOLUTE` | **`vllm_video_params.steps` ships in the ABI, the engine honours it, and no shipped client can set it, so every LTX-2.5 render silently runs the recipe default.** `include/vllm.h:1075` declares `int32_t steps`, `src/capi/vllm_c.cpp:1664` forwards it, and `src/vllm/multimodal/ltx2_video.cpp:4027` reads it — `int64_t steps = gen.steps > 0 ? gen.steps : recipe.num_inference_steps;` — with `allow_request_sigmas` true and `fixed_num_inference_steps` false (`include/vllm/model_executor/models/ltx2_pipeline.h:872-877`), so a request value is honoured rather than clamped. `examples/ltx2_gen/main.cpp:306-451` parses `--frames`, `--width`, `--height` and `--seed` and assigns `steps` nowhere, so `one_stage` at model version 2.5 always runs **30** (`ltx2_pipeline.cpp:1157` from `Ltx2Params24()` → `Ltx2Params23()`, where `:968` sets `num_inference_steps = 30`). WHAT IT COST: [#1864](https://github.com/mudler/vllm.cpp/issues/1864)'s reference render was taken at **8** steps (`tools/oracle/ltx2_oracle.py:88`), so [#1854](https://github.com/mudler/vllm.cpp/issues/1854)'s absolute comparison had a 3.75x denoise-budget confound on the one axis the CLI cannot reach — and it confounds in the direction that FLATTERS us, so a pass taken on it would be unearned. The reference is arm-matched on all four checkpoints, geometry, seed and prompt; this was the only unmatched axis. The `AGENTS.md` "Nothing lands dead" shape at the SEAM rather than in the engine: the capability is reachable through `include/vllm.h` and the thin ABI client that exposes every neighbouring field does not expose this one, and nothing detects it because the renders are correct, no refusal fires, and the only symptom is that every LTX-2.5 render in this tree has run one step count. FIXED IN FLOW: `--steps N` forwards to the existing ABI field and to nothing else; no engine code changes. Owner: row `LTX25-ORACLE-ABSOLUTE`, spec [`ltx25-oracle-absolute.md`](../../specs/ltx25-oracle-absolute.md) | bug |

## Resolution

-
