ID: ISSUE-GH-1793
Title: The GGUF Qwen3.5/3.6 MoE loader (`qwen3_5_gguf_weights.cpp:1082-1099` @ `5d638b67e`) keeps `in_proj_b`/`in_proj_a` split across three residency routes (kept-quant slice, bf16-expand with optional V-row reorder, `gdn_expand_nk` orientation), so packed GDN decode stays unreached on every GGUF MoE checkpoint after `GDN-MOE-PACKED-BA` closes the safetensors arm. Split out so the first unit carries one byte-exactness argument, not three. Listed under `## Owed` in [gdn-moe-packed-ba.md](../../specs/gdn-moe-packed-ba.md)
Row: -
State: UNKNOWN
Kind: gap
GitHub: 1793
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:666`

### Frozen archive evidence

> | [#1793](https://github.com/mudler/vllm.cpp/issues/1793) | — | The GGUF Qwen3.5/3.6 MoE loader (`qwen3_5_gguf_weights.cpp:1082-1099` @ `5d638b67e`) keeps `in_proj_b`/`in_proj_a` split across three residency routes (kept-quant slice, bf16-expand with optional V-row reorder, `gdn_expand_nk` orientation), so packed GDN decode stays unreached on every GGUF MoE checkpoint after `GDN-MOE-PACKED-BA` closes the safetensors arm. Split out so the first unit carries one byte-exactness argument, not three. Listed under `## Owed` in [gdn-moe-packed-ba.md](../../specs/gdn-moe-packed-ba.md) | gap |

## Resolution

-
