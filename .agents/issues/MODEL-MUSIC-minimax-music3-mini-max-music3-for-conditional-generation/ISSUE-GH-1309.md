ID: ISSUE-GH-1309
Title: MiniMax-Music3's 0.646 B RVQ depth decoder is **48.4 % of a run** on `thor:gpu0` and is the last large stage still on the host, so a 0.646 B model costs **6.3x** the 8.6 B language model beside it on the same box. §11.4's last owed device row: the vocoder closed in §13, the DiT in §14, and §14.5 blocked this one on a dtype rather than on the work. **The dtype is settled in [§19.2](../../specs/minimax-music3.md): bf16 storage, f32 accumulation.** The oracle settles it by declaring nothing — `MiniMaxMusic3RVQDepthDecoder` takes no `dtype` parameter and contains no `torch.float32` literal and no `.float()` call (`diffusers` @ `c6da9936`, `models/transformers/minimax_music3_rvq_depth_decoder.py:101-125`); its one cast is the **down**-cast `:51` `.to(query.dtype)`; and `tools/oracle/music3_oracle.py:91-95,103-108` resolves `rvq_depth_decoder: torch.bfloat16` under both policies. `vt::MatmulBT` already carries exactly that contract (`include/vt/ops.h:1281-1283`), so §14.5's objection — that an **f32** `vt::MatmulBT` would drop the bf16 rounding every gated number was taken with — is answered by using the **bf16** one. A finding falls out that no gate here can see: the host arm keeps bf16-exact weights in `std::vector<float>`, so every golden, token gate and WAV hash passes while the path moves twice the oracle's bytes ([§19.2a](../../specs/minimax-music3.md))
Row: MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation
State: UNKNOWN
Kind: feature
GitHub: 1309
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:452`

### Frozen archive evidence

> | [#1309](https://github.com/mudler/vllm.cpp/issues/1309) | `MUSIC3-DEPTH-DEVICE` | MiniMax-Music3's 0.646 B RVQ depth decoder is **48.4 % of a run** on `thor:gpu0` and is the last large stage still on the host, so a 0.646 B model costs **6.3x** the 8.6 B language model beside it on the same box. §11.4's last owed device row: the vocoder closed in §13, the DiT in §14, and §14.5 blocked this one on a dtype rather than on the work. **The dtype is settled in [§19.2](../../specs/minimax-music3.md): bf16 storage, f32 accumulation.** The oracle settles it by declaring nothing — `MiniMaxMusic3RVQDepthDecoder` takes no `dtype` parameter and contains no `torch.float32` literal and no `.float()` call (`diffusers` @ `c6da9936`, `models/transformers/minimax_music3_rvq_depth_decoder.py:101-125`); its one cast is the **down**-cast `:51` `.to(query.dtype)`; and `tools/oracle/music3_oracle.py:91-95,103-108` resolves `rvq_depth_decoder: torch.bfloat16` under both policies. `vt::MatmulBT` already carries exactly that contract (`include/vt/ops.h:1281-1283`), so §14.5's objection — that an **f32** `vt::MatmulBT` would drop the bf16 rounding every gated number was taken with — is answered by using the **bf16** one. A finding falls out that no gate here can see: the host arm keeps bf16-exact weights in `std::vector<float>`, so every golden, token gate and WAV hash passes while the path moves twice the oracle's bytes ([§19.2a](../../specs/minimax-music3.md)) | feature |

## Resolution

-
