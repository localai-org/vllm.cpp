ID: ISSUE-GH-1150
Title: The sigma SHIFT is derived from the target latent on every arm, where six of upstream's seven `LTX2Scheduler.execute` call sites pass NO latent and take `default_number_of_tokens` = `MAX_SHIFT_ANCHOR` = 4096 (`schedulers.py:11,:29,:31`). `grep -rn '\.execute(' packages/ltx-pipelines/src/ltx_pipelines/` at `fd4ded7f` returns seven and that grep is the whole population: only `ti2vid_two_stages_hq.py:267` passes `latent=empty_latent`. This engine passes `target_tokens` at `src/vllm/multimodal/ltx2_video.cpp:3442-3443`, so it mirrors the exception and diverges from the rule. Correct today: `t2a_one_stage` (passes 0 at `src/vllm/model_executor/models/ltx2_t2a.cpp:178`) and `res2s_two_stage`. DIVERGENT: `one_stage` at four version keys (`ti2vid_one_stage.py:207`), `a2vid_two_stage` stage 1 (`a2vid_two_stage.py:226`) and `retake`'s non-distilled arm (`retake.py:287`). Recipes carrying explicit `sigmas` never reach the derivation and are unaffected. Not a rounding difference: at the recipe default geometry the target latent is 6144 tokens, giving `sigma_shift` 2.78 against upstream's 2.05, so every sigma moves. Invisible because the trajectory changes while the frame count, shapes, sample rate and errors do not, and our goldens were captured from this engine so they PIN it rather than detect it. `.agents/specs/ltx25-res2s-loop.md:80-88` saw the HQ/plain split and concluded the divergence was on the plain two-stage arm alone; that is right about HQ and wrong about the blast radius. Found by row `LTX25-TI2VID-RECIPE`, which added the seam — `Ltx2PhaseRecipe::schedule_tokens`, defaulted to today's behaviour so nothing moves — and set it on the one phase it ships. Not fixed in flow because flipping the other three re-samples five shipped, gated arms and rewrites their goldens, which needs its own spec and fresh review. Listed under `## Owed` in [`ltx25-ti2vid-recipe.md`](../../specs/ltx25-ti2vid-recipe.md)
Row: LTX25-TI2VID-RECIPE
State: UNKNOWN
Kind: bug
GitHub: 1150
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:355`

### Frozen archive evidence

> | [#1150](https://github.com/mudler/vllm.cpp/issues/1150) | `LTX25-TI2VID-RECIPE` | The sigma SHIFT is derived from the target latent on every arm, where six of upstream's seven `LTX2Scheduler.execute` call sites pass NO latent and take `default_number_of_tokens` = `MAX_SHIFT_ANCHOR` = 4096 (`schedulers.py:11,:29,:31`). `grep -rn '\.execute(' packages/ltx-pipelines/src/ltx_pipelines/` at `fd4ded7f` returns seven and that grep is the whole population: only `ti2vid_two_stages_hq.py:267` passes `latent=empty_latent`. This engine passes `target_tokens` at `src/vllm/multimodal/ltx2_video.cpp:3442-3443`, so it mirrors the exception and diverges from the rule. Correct today: `t2a_one_stage` (passes 0 at `src/vllm/model_executor/models/ltx2_t2a.cpp:178`) and `res2s_two_stage`. DIVERGENT: `one_stage` at four version keys (`ti2vid_one_stage.py:207`), `a2vid_two_stage` stage 1 (`a2vid_two_stage.py:226`) and `retake`'s non-distilled arm (`retake.py:287`). Recipes carrying explicit `sigmas` never reach the derivation and are unaffected. Not a rounding difference: at the recipe default geometry the target latent is 6144 tokens, giving `sigma_shift` 2.78 against upstream's 2.05, so every sigma moves. Invisible because the trajectory changes while the frame count, shapes, sample rate and errors do not, and our goldens were captured from this engine so they PIN it rather than detect it. `.agents/specs/ltx25-res2s-loop.md:80-88` saw the HQ/plain split and concluded the divergence was on the plain two-stage arm alone; that is right about HQ and wrong about the blast radius. Found by row `LTX25-TI2VID-RECIPE`, which added the seam — `Ltx2PhaseRecipe::schedule_tokens`, defaulted to today's behaviour so nothing moves — and set it on the one phase it ships. Not fixed in flow because flipping the other three re-samples five shipped, gated arms and rewrites their goldens, which needs its own spec and fresh review. Listed under `## Owed` in [`ltx25-ti2vid-recipe.md`](../../specs/ltx25-ti2vid-recipe.md) | bug |

## Resolution

-
