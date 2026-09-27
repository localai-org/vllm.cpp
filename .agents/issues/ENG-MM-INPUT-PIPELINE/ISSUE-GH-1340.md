ID: ISSUE-GH-1340
Title: `VT_FUSE_ATTN_PREAMBLE=0` on the Qwen3.6 MRoPE path silently applies 1-D RoPE instead of refusing. Our fused preamble is the ONLY MRoPE arm: `vt::AttnQkNormRopeGate` takes no positions at all, only a precomputed per-token `cos_sin` cache `[T, rotary_dim]`, which `src/vllm/model_executor/models/qwen3_5.cpp::BuildMropeCosSinHost` fills with the interleaved 3-section MRoPE axis selection for the M3-b image and M3d video paths. The eager arm in `::FullAttnBlockPaged` calls `vt::RopeNeox` on the 1-D positions vector, which that same file comments as "unused for rope under the fused MRoPE path", so with the toggle off the model computes plain NeoX RoPE where MRoPE is required: every shape agrees, nothing throws, the tokens are wrong. AGENTS.md requires an unimplemented arm to be REFUSED naming the missing part, and this one is substituted instead. Found while landing [#607](https://github.com/mudler/vllm.cpp/issues/607) wave L4, which is the same asymmetry read the other way: mirroring upstream's `text_only` conjunct (`vllm/model_executor/models/qwen3_next.py:324-331` at the pin) would route the multimodal configuration into this arm deliberately. NOT fixed in flow because its gate is a VL token-exactness run through `ModelRegistry::Forward` on a real checkpoint and the L4 wave had no GPU: both fleet devices were held and it carried no lease authority. Also under `## Owed` in [`multimodal-track.md`](../../specs/multimodal-track.md) §1.6
Row: ENG-MM-INPUT-PIPELINE
State: UNKNOWN
Kind: bug
GitHub: 1340
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:437`

### Frozen archive evidence

> | [#1340](https://github.com/mudler/vllm.cpp/issues/1340) | `ENG-MM-INPUT-PIPELINE` | `VT_FUSE_ATTN_PREAMBLE=0` on the Qwen3.6 MRoPE path silently applies 1-D RoPE instead of refusing. Our fused preamble is the ONLY MRoPE arm: `vt::AttnQkNormRopeGate` takes no positions at all, only a precomputed per-token `cos_sin` cache `[T, rotary_dim]`, which `src/vllm/model_executor/models/qwen3_5.cpp::BuildMropeCosSinHost` fills with the interleaved 3-section MRoPE axis selection for the M3-b image and M3d video paths. The eager arm in `::FullAttnBlockPaged` calls `vt::RopeNeox` on the 1-D positions vector, which that same file comments as "unused for rope under the fused MRoPE path", so with the toggle off the model computes plain NeoX RoPE where MRoPE is required: every shape agrees, nothing throws, the tokens are wrong. AGENTS.md requires an unimplemented arm to be REFUSED naming the missing part, and this one is substituted instead. Found while landing [#607](https://github.com/mudler/vllm.cpp/issues/607) wave L4, which is the same asymmetry read the other way: mirroring upstream's `text_only` conjunct (`vllm/model_executor/models/qwen3_next.py:324-331` at the pin) would route the multimodal configuration into this arm deliberately. NOT fixed in flow because its gate is a VL token-exactness run through `ModelRegistry::Forward` on a real checkpoint and the L4 wave had no GPU: both fleet devices were held and it carried no lease authority. Also under `## Owed` in [`multimodal-track.md`](../../specs/multimodal-track.md) §1.6 | bug |

## Resolution

-
