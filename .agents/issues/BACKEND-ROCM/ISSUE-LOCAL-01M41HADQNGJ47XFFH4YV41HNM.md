ID: ISSUE-LOCAL-01M41HADQNGJ47XFFH4YV41HNM
Title: f32-query decode attention falls back to PagedAttnOnline on gfx1100
Row: BACKEND-ROCM
State: CLOSED
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-03
Updated: 2026-10-03
Closed: 2026-10-03

## Problem

Qwen3.8-27B-EXL3 runs attention as f32 query / f32 out over bf16 KV, which excludes the bf16 decode arms and falls to the generic per-context-token PagedAttnOnline: rocprofv3 220-token census shows 747ms total (212us/call x16 layers = ~3.4ms/token) at ~5 GB/s effective. The fused GQA decode arm PagedAttnDecodeGqaF32Q<6,8> exists for this dtype mix but is gated VT_ATTN_DECODE_GQA4 default OFF. Serving with it ON measures 26.24 vs 22.13 decode tok/s end-to-end with golden greedy parity.

## Resolution

Resolved by 25f3398ba: VT_ATTN_DECODE_GQA4 default ON for the f32-query/bf16-KV path. Same-leg A/B (12-prompt HumanEval, 128 tok): OFF 25.23 decode/20.08 whole-run -> ON 27.43 decode/23.15 whole-run (+8.7% decode). Golden parity holds.
