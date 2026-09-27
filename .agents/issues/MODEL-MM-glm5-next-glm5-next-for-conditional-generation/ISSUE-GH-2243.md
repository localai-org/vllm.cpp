ID: ISSUE-GH-2243
Title: **`glm5next.attention.head_count_kv` is a per-layer `array[i32]` in the published artifact and `Glm5NextHfConfigFromGguf` reads it as a scalar.** Found while landing [#2240](https://github.com/mudler/vllm.cpp/issues/2240): with IQ2_XS and IQ4_XS decoded, the production loader gets past the type-17 refusal, opens all four shards, sizes all 1412 tensors, and stops instead at `glm5_next gguf: key glm5next.attention.head_count_kv is not an integer`. The artifact stores the layer schedule there — length 46, `0` on the 35 KDA layers and `1` on the 11 DSA/MLA layers — and `swiglu_clamp_exp`/`swiglu_clamp_shexp` are per-layer `array[f32]` of the same length directly behind it. Filed rather than fixed in that flow because it belongs to this row's config/loader wave and not to a dequant change; listed under `## Owed` as O18 in [`specs/glm5-next-flash.md`](../../specs/glm5-next-flash.md)
Row: MODEL-MM-glm5-next-glm5-next-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 2243
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:870`

### Frozen archive evidence

> | [#2243](https://github.com/mudler/vllm.cpp/issues/2243) | `MODEL-MM-glm5-next-glm5-next-for-conditional-generation` | **`glm5next.attention.head_count_kv` is a per-layer `array[i32]` in the published artifact and `Glm5NextHfConfigFromGguf` reads it as a scalar.** Found while landing [#2240](https://github.com/mudler/vllm.cpp/issues/2240): with IQ2_XS and IQ4_XS decoded, the production loader gets past the type-17 refusal, opens all four shards, sizes all 1412 tensors, and stops instead at `glm5_next gguf: key glm5next.attention.head_count_kv is not an integer`. The artifact stores the layer schedule there — length 46, `0` on the 35 KDA layers and `1` on the 11 DSA/MLA layers — and `swiglu_clamp_exp`/`swiglu_clamp_shexp` are per-layer `array[f32]` of the same length directly behind it. Filed rather than fixed in that flow because it belongs to this row's config/loader wave and not to a dequant change; listed under `## Owed` as O18 in [`specs/glm5-next-flash.md`](../../specs/glm5-next-flash.md) | bug |

## Resolution

-
