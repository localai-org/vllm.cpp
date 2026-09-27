ID: ISSUE-GH-2268
Title: **The published GGUF spells MLA geometry by llama.cpp's `attention.key_length` convention and carries no `attention.linear_head_count`, so the loader stops there next.** Measured 2026-08-29 on one tree and one binary, with the [#2243](https://github.com/mudler/vllm.cpp/issues/2243) / [#2177](https://github.com/mudler/vllm.cpp/issues/2177) array fix reverted and restored, driving `LoadedEngine::FromModelDir` at `/mnt/nas_share/rc/ckpt/GLM-5.3-Flash-UD-Q2_K_XL/` on `device = kCPU`, headers only: without the fix it stops at `key glm5next.attention.head_count_kv is not an integer`, with it at `attention.key_length_mla - attention.key_length is -256 but rope.dimension_count is 0` (`glm5_next_weights.cpp:435`). The file is not malformed. llama.cpp writes `key_length = kv_lora_rank + qk_rope_head_dim` and `key_length_mla = qk_nope_head_dim + qk_rope_head_dim` (`b10451:conversion/deepseek.py:345-348`), which for this model gives the artifact's own 512 / 512 / 256 / 256; `scripts/convert-glm5-next-gguf.py` writes `key_length = qk_nope_head_dim`, a different quantity under the same name. `glm5next.attention.linear_head_count`, a `ReqInt` in the builder, is in none of the file's 72 keys and llama.cpp spells it nowhere. Filed rather than fixed in that flow because it moves the WRITE side: putting the reader on llama.cpp's meaning without moving the converter would refuse our own output, and which spelling this project writes is a row-and-spec decision. O7 records that our converter has never been run, so no artifact of ours is invalidated by the move. Recorded under `## Owed` as O18 in [`specs/glm5-next-flash.md`](../../specs/glm5-next-flash.md)
Row: MODEL-MM-glm5-next-glm5-next-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 2268
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:877`

### Frozen archive evidence

> | [#2268](https://github.com/mudler/vllm.cpp/issues/2268) | `MODEL-MM-glm5-next-glm5-next-for-conditional-generation` | **The published GGUF spells MLA geometry by llama.cpp's `attention.key_length` convention and carries no `attention.linear_head_count`, so the loader stops there next.** Measured 2026-08-29 on one tree and one binary, with the [#2243](https://github.com/mudler/vllm.cpp/issues/2243) / [#2177](https://github.com/mudler/vllm.cpp/issues/2177) array fix reverted and restored, driving `LoadedEngine::FromModelDir` at `/mnt/nas_share/rc/ckpt/GLM-5.3-Flash-UD-Q2_K_XL/` on `device = kCPU`, headers only: without the fix it stops at `key glm5next.attention.head_count_kv is not an integer`, with it at `attention.key_length_mla - attention.key_length is -256 but rope.dimension_count is 0` (`glm5_next_weights.cpp:435`). The file is not malformed. llama.cpp writes `key_length = kv_lora_rank + qk_rope_head_dim` and `key_length_mla = qk_nope_head_dim + qk_rope_head_dim` (`b10451:conversion/deepseek.py:345-348`), which for this model gives the artifact's own 512 / 512 / 256 / 256; `scripts/convert-glm5-next-gguf.py` writes `key_length = qk_nope_head_dim`, a different quantity under the same name. `glm5next.attention.linear_head_count`, a `ReqInt` in the builder, is in none of the file's 72 keys and llama.cpp spells it nowhere. Filed rather than fixed in that flow because it moves the WRITE side: putting the reader on llama.cpp's meaning without moving the converter would refuse our own output, and which spelling this project writes is a row-and-spec decision. O7 records that our converter has never been run, so no artifact of ours is invalidated by the move. Recorded under `## Owed` as O18 in [`specs/glm5-next-flash.md`](../../specs/glm5-next-flash.md) | bug |

## Resolution

-
