ID: ISSUE-GH-1538
Title: **vllm#52816's head moved a THIRD time (`66e5414c` -> `3406ec1d`) while it is still open, and refactors `compute_candidates` into `LogitsProcessor.get_top_k_tokens`.** Measured 2026-08-21 by W6 from `raw.githubusercontent.com` at both heads: +11/-80 on `qwen3_dflash2.py`, +4/-16 on `dflash2/speculator.py`, +2/-5 on the base `speculator.py`. The big one is a RELOCATION rather than new math -- the padding mask, the id rebase, the TP all-gather and the scale-THEN-softcap order all survive in `logits_processor.py:241-286`, so `## Owed` O16's reading of the codebook-span question holds at BOTH heads. What does NOT survive is the explicit `UnquantizedEmbeddingMethod`/`UnquantizedLinearMethod` guard that `## Risks/decisions` D12 ports as `RefuseQuantizedDflash2LmHead`, which is deleted at `3406ec1d`; our guard's own reason (the GGUF arm dequantizes `output.weight` to bf16, and a GGUF target with a safetensors DFlash2 draft is admitted here) is independent of upstream's and stands. NOT reconciled in flow, deliberately: `## Gates` G2 fixes the gate head at `66e5414c` while the pull request is unmerged, and moving the port onto a third unmerged head during the gate would move the thing being measured. Owed under `## Owed` O21 of [the DFlash2 spec](../../specs/dflash2-spec-decode.md)
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: verification
GitHub: 1538
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:586`

### Frozen archive evidence

> | [#1538](https://github.com/mudler/vllm.cpp/issues/1538) | `SPEC-DFLASH2` | **vllm#52816's head moved a THIRD time (`66e5414c` -> `3406ec1d`) while it is still open, and refactors `compute_candidates` into `LogitsProcessor.get_top_k_tokens`.** Measured 2026-08-21 by W6 from `raw.githubusercontent.com` at both heads: +11/-80 on `qwen3_dflash2.py`, +4/-16 on `dflash2/speculator.py`, +2/-5 on the base `speculator.py`. The big one is a RELOCATION rather than new math -- the padding mask, the id rebase, the TP all-gather and the scale-THEN-softcap order all survive in `logits_processor.py:241-286`, so `## Owed` O16's reading of the codebook-span question holds at BOTH heads. What does NOT survive is the explicit `UnquantizedEmbeddingMethod`/`UnquantizedLinearMethod` guard that `## Risks/decisions` D12 ports as `RefuseQuantizedDflash2LmHead`, which is deleted at `3406ec1d`; our guard's own reason (the GGUF arm dequantizes `output.weight` to bf16, and a GGUF target with a safetensors DFlash2 draft is admitted here) is independent of upstream's and stands. NOT reconciled in flow, deliberately: `## Gates` G2 fixes the gate head at `66e5414c` while the pull request is unmerged, and moving the port onto a third unmerged head during the gate would move the thing being measured. Owed under `## Owed` O21 of [the DFlash2 spec](../../specs/dflash2-spec-decode.md) | verification |

## Resolution

-
