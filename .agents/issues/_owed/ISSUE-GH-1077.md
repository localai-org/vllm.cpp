ID: ISSUE-GH-1077
Title: `.env.example:37`, `.agents/environment.md:29` and `tests/vllm/multimodal/test_ltx2_video.cpp:2128-2132` each state that nothing in the tree reads `CHECKPOINT_ROOT`, and six gates read it: `tests/parity/test_minimax_music3_ar_real.cpp:162`, `_e2e_real.cpp:170`, `_llm_real.cpp:137`, `_quant_real.cpp:130,140`, `tests/vllm/models/test_ltx2_text_encoder.cpp:2299`, and `test_nemotron_h_loader.cpp:161` tells the reader to export it. No product code under `src/` or `include/` reads it, so the accurate statement is that the LIBRARY never reads it while several gates do. It costs more than tidiness: `test_ltx2_video.cpp` reasons FROM the claim when it chooses a separate `LTX2_CHECKPOINT_ROOT` ("this would be its first reader"), and that reasoning is void. Found while repairing [#1073](https://github.com/mudler/vllm.cpp/issues/1073) and NOT fixed there, because reversing a design decision needs its own review rather than a path substitution. Listed under `## Owed` in [`nas-mount-path.md`](../../specs/nas-mount-path.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1077
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:315`

### Frozen archive evidence

> | [#1077](https://github.com/mudler/vllm.cpp/issues/1077) | — | `.env.example:37`, `.agents/environment.md:29` and `tests/vllm/multimodal/test_ltx2_video.cpp:2128-2132` each state that nothing in the tree reads `CHECKPOINT_ROOT`, and six gates read it: `tests/parity/test_minimax_music3_ar_real.cpp:162`, `_e2e_real.cpp:170`, `_llm_real.cpp:137`, `_quant_real.cpp:130,140`, `tests/vllm/models/test_ltx2_text_encoder.cpp:2299`, and `test_nemotron_h_loader.cpp:161` tells the reader to export it. No product code under `src/` or `include/` reads it, so the accurate statement is that the LIBRARY never reads it while several gates do. It costs more than tidiness: `test_ltx2_video.cpp` reasons FROM the claim when it chooses a separate `LTX2_CHECKPOINT_ROOT` ("this would be its first reader"), and that reasoning is void. Found while repairing [#1073](https://github.com/mudler/vllm.cpp/issues/1073) and NOT fixed there, because reversing a design decision needs its own review rather than a path substitution. Listed under `## Owed` in [`nas-mount-path.md`](../../specs/nas-mount-path.md) | bug |

## Resolution

-
