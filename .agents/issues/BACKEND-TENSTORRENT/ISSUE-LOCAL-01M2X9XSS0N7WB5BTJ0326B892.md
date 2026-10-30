ID: ISSUE-LOCAL-01M2X9XSS0N7WB5BTJ0326B892
Title: TT grouped keep-quant arm crashes at M>1 (MeshBuffer too small, infinite retry)
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-19
Updated: 2026-10-21
Closed: 2026-10-21

## Problem

Concurrency=2 (grouped keep-quant arm at M=2, leaving the f32-exact P==1 decode path) fatals in prefill with 'TT_FATAL: MeshBuffer must be large enough to hold the tensor (assert.hpp:104)' repeated 463x in a retry loop; no tokens produced. Repro: vllm-bench --model /mnt/models/mudler-qwen3.8-27B-APEX-gguf/Qwen3.8-27B-APEX-I-Nano.gguf --num-prompts 2 --input-len 128 --output-len 32 --concurrency 2 --temperature 0 on P150, main@1f46d44e1, env VT_TT_AFFINE_F32=1 VT_TT_NORM_PAD=1 VT_TT_PROGRAM_CACHE=1. Log: /tmp/bench_b3.log. Single-stream (M=1) is clean and token-exact vs the llama.cpp oracle.

## Resolution

- 2026-09-19 (PARTIAL): gdb backtraces (TT_ASSERT_ABORT=1, /tmp/m2_gdb.log,
  /tmp/m2_gdb4.log) located the decode-stage sites: TryPagedAttentionDeviceDecode
  built tt::tt_metal::view metadata views whose spec exceeds the buffer at B>1
  eager — the Q 4D padded view (padded dim1=1 < logical dim1=B,
  tenstorrent_ops.cpp:2787) and the sdpa-output [B,H*D] flatten through the
  free ttnn::reshape (padded=logical -> physical up to 16x the buffer). Each
  fatal was silently caught and the op fell back to the host path. Fixed:
  eager materializes through the free reshape's device program; capture keeps
  the B==1 view; B>1 capture refuses loudly. Red-first unit test
  "kPagedAttention pure-decode batched B=2 keeps the device path" (DeviceShadowExact
  requirement; on base: 1 MeshBuffer fatal + host fallback). Full TT suite
  524,523 assertions green; GDN/PA/capture focused runs green.
- REMAINING (moved to ISSUE-LOCAL-01M2XSWCP507W0M18EXHGR5TGS): the prefill step
  still fatals in RmsNormGatedKernel's gate reshape, then SigmoidGateBf16 OOMs —
  the M=2 e2e does not produce tokens yet. Issue stays OPEN.

- 2026-10-21 (CLOSED): FALSIFIED BY THE TREE. Both conditions this record
  names are gone. The decode-stage MeshBuffer fatal is fixed by `3fca8e541`
  ("fix(BACKEND-TENSTORRENT): the batched paged-attention decode MeshBuffer
  view fatal", 2026-09-19), the squash that carries PR #3230 into `main`:
  the batched eager path now materializes the Q 4D view and the sdpa-output
  flatten through the free reshape's device program, and the red-first case
  "kTENSTORRENT kPagedAttention pure-decode batched B=2 keeps the device
  path" pins it. That commit's body names this record and hands its
  REMAINING prefill lane to `ISSUE-LOCAL-01M2XSWCP507W0M18EXHGR5TGS`.
  That lane closed on 2026-09-20 with a P150 A/B on the same repro: base
  `main@1f46d44e1` enters the MeshBuffer retry loop with 247+ fatals, and the
  tip completes the run clean three times (output-len 1, 8, and 32) with 0
  fatals and 16/16/64 tokens. With the decode fix and the prefill fix both
  in `main`, the subject of this issue -- the concurrency-2 grouped
  keep-quant arm fataling on the MeshBuffer and producing no tokens -- no
  longer exists.

