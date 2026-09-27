ID: ISSUE-GH-1659
Title: **The speed harness asserts `--attention-backend` and never passes it to `LLM()`, so the declared denominator is unreachable by any path.** Measured on `dgx:gpu0` 2026-08-22 at `bed3feae6`. `capture()` built `LLM(...)` with no backend kwarg while `attention_backend_reasons` requires `resolved == declared`, and the arm log under a declared `TRITON_ATTN` reads `Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN','FLASHINFER','TRITON_ATTN','FLEX_ATTENTION']`. So repairing [#1658](https://github.com/mudler/vllm.cpp/issues/1658) ALONE would make the arm resolve `FLASH_ATTN`, compare it against the declared `TRITON_ATTN` and refuse on the mismatch: a working gate needs this, the probe entry and [#1657](https://github.com/mudler/vllm.cpp/issues/1657), not any one of them. This reproduces W6's recorded failure at a later head -- [#1456](https://github.com/mudler/vllm.cpp/issues/1456) measured vLLM's vendored flash-attention unable to target sm_12x, which is why `TRITON_ATTN` is the declared oracle backend on this box at all. FIXED IN FLOW: the declared backend is passed over the spellings `ATTENTION_BACKEND_KWARGS` names -- `attention_config` carrying a `backend` key first, because the measured read-back walk is `vllm_config.attention_config.backend`, then a bare `attention_backend`. The spelling is UNVERIFIED at the beyond-pin head, on the same footing `BACKEND_PROBES` was on: a wheel REJECTS a kwarg it does not declare with a `TypeError` raised while `EngineArgs` is built and therefore before anything loads, so trying both costs no lease time; a wheel that takes NEITHER is a loud refusal naming both; a spelling that is accepted and IGNORED is caught by the read-back, which is this refusal; and `--attention-backend-kwarg` pins the answer once known, with no code change. Owned by [`dflash2-spec-decode.md`](../../specs/dflash2-spec-decode.md) `## Owed` O28
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1659
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:597`

### Frozen archive evidence

> | [#1659](https://github.com/mudler/vllm.cpp/issues/1659) | `SPEC-DFLASH2` | **The speed harness asserts `--attention-backend` and never passes it to `LLM()`, so the declared denominator is unreachable by any path.** Measured on `dgx:gpu0` 2026-08-22 at `bed3feae6`. `capture()` built `LLM(...)` with no backend kwarg while `attention_backend_reasons` requires `resolved == declared`, and the arm log under a declared `TRITON_ATTN` reads `Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN','FLASHINFER','TRITON_ATTN','FLEX_ATTENTION']`. So repairing [#1658](https://github.com/mudler/vllm.cpp/issues/1658) ALONE would make the arm resolve `FLASH_ATTN`, compare it against the declared `TRITON_ATTN` and refuse on the mismatch: a working gate needs this, the probe entry and [#1657](https://github.com/mudler/vllm.cpp/issues/1657), not any one of them. This reproduces W6's recorded failure at a later head -- [#1456](https://github.com/mudler/vllm.cpp/issues/1456) measured vLLM's vendored flash-attention unable to target sm_12x, which is why `TRITON_ATTN` is the declared oracle backend on this box at all. FIXED IN FLOW: the declared backend is passed over the spellings `ATTENTION_BACKEND_KWARGS` names -- `attention_config` carrying a `backend` key first, because the measured read-back walk is `vllm_config.attention_config.backend`, then a bare `attention_backend`. The spelling is UNVERIFIED at the beyond-pin head, on the same footing `BACKEND_PROBES` was on: a wheel REJECTS a kwarg it does not declare with a `TypeError` raised while `EngineArgs` is built and therefore before anything loads, so trying both costs no lease time; a wheel that takes NEITHER is a loud refusal naming both; a spelling that is accepted and IGNORED is caught by the read-back, which is this refusal; and `--attention-backend-kwarg` pins the answer once known, with no code change. Owned by [`dflash2-spec-decode.md`](../../specs/dflash2-spec-decode.md) `## Owed` O28 | bug |

## Resolution

-
