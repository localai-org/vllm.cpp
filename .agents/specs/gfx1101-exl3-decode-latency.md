# gfx1101 EXL3 decode/prefill latency arms (BACKEND-ROCM)

Row: `BACKEND-ROCM`. Issue:
`.agents/issues/BACKEND-ROCM/ISSUE-LOCAL-01M3VQ4MPSG8NNZZ9B6R8AK28R.md`.
Sibling issue `.agents/issues/BACKEND-ROCM/ISSUE-LOCAL-01M3T5GBR5NS2B778YRKWK0ZG3.md`
(the hipBLASLt recon arm, same worktree, committed separately).

## Problem

Measured on gfx1101 (RX 7700 XT, `local/rocm-base:10.0.0` container, rocprofv3
kernel trace + end-to-end `vllm-server` timing) serving
`turboderp/Qwen3.5-9B-EXL3-4.00bpw`:

- Baseline (stock `vllm-cpp:rocm-gfx1101-exl3` image): 2022-token prompt +
  32-token greedy = 11.05 s; 256-token decode = 29.6 tok/s.
- Per-token kernel census (decode, ~1300 launches/token, ~22 ms busy):
  `PagedAttnOnline<float,bf16,float>` at 2.7 ms/call serving BOTH prefill and
  decode attention; `GdnScanCoopK` at 350 µs/call on decode; `Exl3GemvM1K4`
  12.8 ms/token (~370 GB/s, near the ~10.5 ms stream floor).
- Root causes:
  1. Qwen3.5's attention runs f32 query + f32 out over a bf16 KV cache. Every
     fast arm is gated on bf16 query (`bf16_decode_opt`) or fp8 KV, so the
     model hit the O(n·d) `PagedAttnOnline` fallback for prefill and decode.
  2. `GdnScanCoopK` reads each state row twice (dot pass + update pass) and
     uses one block per (hv, seq) — 32 blocks total at decode, latency-bound.
  3. `GcnArchNameHasSharedKAttentionWmma` admits gfx1100 + gfx12 only; gfx1101
     is the same gfx11 WMMA family and was excluded.
  4. `static_graph_requires_opt_in()` returns eager for every ROCm arch except
     gfx1100; gfx1101 therefore ran ~1300 eager launches/token. The
     `VLLM_CPP_ROCM_STATIC_GRAPH=1` escape hatch exists and is verified here.

## Design

All changes in `src/vt/rocm/` + `include/vt/rocm/`:

1. **f32-query/bf16-KV prefill arm** (`rocm_paged_attn.hip`):
   `PagedAttnPrefillSharedK<QG=2, EPL=8, BM=32, BN=32, __hip_bfloat16, float,
   float>` dispatched when `query/out==kF32 && kv==kBF16 && d==256 &&
   total_q>=64 && num_reqs==1` and `qg ∈ {2,4,8}` (grid.z = qg/2). This is the
   identical kernel instantiation and z-tiling the existing fp8 prefill arm
   already uses for Qwen3.5-4B (same file, `fp8_prefill_fast` block); only the
   KV element type differs. Env `VT_ATTN_PREFILL_F32Q_B16KV=0` opts out;
   default ON.
2. **qg=4 bf16-native prefill** (`rocm_paged_attn.hip`): an
   `qg==4 && d==256 && num_reqs==1 && total_q>=64` branch inside the existing
   `bf16_decode_opt` gate, same kernel (bf16 Q/KV/O). Unexercised by this
   checkpoint (it is f32-query) but covers bf16-native qg=4 models; same
   `SharedKAttentionWmmaHostOk()` admission.
3. **gfx1101 prefill-WMMA admission** (`rocm_attn_wmma_arch.h`):
   `GcnArchNameHasSharedKAttentionWmma` gains `prefix_ok(arch, "gfx1101")`.
   Decode admission (`GcnArchNameHasGemmaDecodeWmma`, `decode_ok` slot) is
   untouched — decode stays gfx1100-only.
4. **`GdnScanCoopFusedK`** (`rocm_gdn_scan.hip`): same warp-per-row mapping as
   `GdnScanCoopK`; the row is loaded once into `KD` per-lane registers
   (`kd ∈ {2,4}` for `dk ∈ {64,128}`), dotted, updated in registers, stored
   once — one state read + one write instead of two reads + one write.
   `gridDim.z` splits the dv rows across blocks (default 4 via
   `VT_GDN_SCAN_ZSPLIT`, 0 disables the arm). Dispatch: only when
   `VT_GDN_SCAN_COOP=1` (the existing opt-in) AND `qsl == nullptr` (per-seq
   single token, i.e. decode) AND `dk ∈ {64,128}`. Prefill keeps `GdnScanCoopK`
   (the z-split/zero-fill path handles `state_slot<0` identically).
5. **Diagnostic**: `VT_EXL3_GEMV_DUMP=1` prints resolved (m,k,n,bits,cb,cols)
   per `Exl3GemvTryLaunchRocm` launch (rocm_exl3_gemv.hip).

## Evidence (this worktree, gfx1101, Qwen3.5-9B-EXL3-4.00bpw)

- Prefill+decode A/B, 2022 in / 32 out, greedy: 11.05 s → 3.10 s (3.56x).
- Decode-only, 256 tokens: 29.6 → 40.4 tok/s.
- Trace deltas: `PagedAttnOnline<float,bf16,float>` 2368 ms → gone from
  prefill (8 `PagedAttnPrefillSharedK` calls, 91 µs total) and decode
  (DecodeGqaF32Q arm via `VT_ATTN_DECODE_GQA4=1`). `GdnScanCoopFusedK`
  11.4 µs/call vs `GdnScanCoopK` 349.7 µs/call.
- `VLLM_CPP_ROCM_STATIC_GRAPH=1` on gfx1101: `[DenseDecodeGraph] captured ...
  S=1 ... 62 total replays` — first captured-graph evidence on this arch.
- Greedy parity vs the stock image on 3 prompts: all three diverge only at
  phrase level at tokens 136/204/548 (synonym swaps; same character as the
  documented reordering drift on the existing opt-in arms). The fused GDN
  kernel preserves `GdnScanCoopK`'s reduction order, so it is bit-identical
  to the coop arm it replaces.

## Rejected levers (measured, not kept)

`VT_EXL3_GEMV_KSPLIT` 2/4/8 flat; `VT_EXL3_DOT_FIRST=1` 11.1 s vs 8.5 s;
`VT_ROCM_EXL3_WMMA=1` 11.1 s; `VT_ARGMAX_SPLIT=1` and `VT_ASYNC_EXECUTOR=1`
no measurable change; M1K4 prefetch ring PF 4→8 regressed (9.7 s vs 8.5 s) and
was reverted.

## Gates

- `test_ops_gdn`, `test_ops_paged_attn`, `test_ops_paged_attn_dtype`,
  `test_gdn_prefill_conv`, `test_paged_attn_route`, `test_exl3_rocm`,
  `test_exl3_rocm_gemv` — all pass on gfx1101 (68+15+3+10+4+7+6 cases,
  0 failures).
- Serving smoke on the real checkpoint (numbers above).
- The new arms are opt-in or dtype/shape-gated; the byte-exact fallbacks
  (`GdnScanK`, `PagedAttnOnline`) stay reachable via env or non-matching
  dtype.

## Risks

- Fused scan register array `rv[KD]` is small (≤4 floats/lane); dk outside
  {64,128} is not admitted.
- The f32-q SharedK arm reorders the online-softmax accumulation vs
  `PagedAttnOnline` — NMSE-equal, phrase-level divergence observed and
  consistent with the documented drift of the sibling opt-in arms.
- gfx1101 WMMA admission also enables the sharedk_wmma prefill arm for qg=2
  bf16 models on gfx1101 — intended coverage, same-family hardware.
- `VT_EXL3_GEMV_DUMP` writes one stderr line per launch; opt-in only.

## Owed / stop conditions

- `M1K4` at ~370 GB/s vs ~10.5 ms/token stream floor: ~2 ms/token of headroom
  left; instruction-level rework is a separate change.
- ~10 ms/token host-serial decode path (per-step metadata, argmax D2H,
  detokenize) even under the captured graph: separate structural work.
- Admitting gfx1101 into `static_graph_requires_opt_in()` (dropping the env)
  is a one-line platform change that should ride with gfx1101 captured-arm
  evidence — this spec's serving run IS that evidence; leaving the flip to
  review discretion.

## Outcome (2026-10-01)

Landed on `pp-exl3-recon-lt` (worktree vllm.cpp-pp), five commits:

- `8103ca546` spec + issue records
- `b690dea96` hipBLASLt recon GEMM (closes ISSUE-LOCAL-01M3T5GBR5NS2B778YRKWK0ZG3)
- `63e117e81` f32-query attention + fused GDN scan + gfx1101 WMMA admission
- `8a230b5c4` recon issue record close
- `b2028bd57` review repairs

Measured on gfx1101, `Qwen3.5-9B-EXL3-4.00bpw`, greedy: 11.05s → 3.10s
(2022+32 tok), 29.6 → 40.4 tok/s decode. Fresh-review loop: pass 1 found 2×P1
(gfx1101 host-admitted but compile-excluded WMMA stub; test_rocm_arch
contradiction), 2×P2 (capture-unsafe Lt sweep; ws_sizes desync), 3×P3 — all
repaired in `b2028bd57`. Pass 2: PASS; residual P3s recorded below.

Residuals owed: (a) gfx1101 takes the gfx12-style plain `mma_sync` semantics in
`AttentionMmaSync` rather than gfx1100's canonicalized path — correct
(oracle-verified) but numerically distinct; a dedicated gfx1101 parity gate
would decide whether to extend the gfx1100 gates. (b) `ws_sizes`/`cache`
pairing is unobservable by current tests. (c) `static_graph_requires_opt_in()`
still returns eager for gfx1101 by default — `VLLM_CPP_ROCM_STATIC_GRAPH=1`
verified working (62 replays); the arch flip wants a gfx1101 evidence row.
(d) M1K4 ~370 GB/s vs ~10.5ms/token stream floor; HadK+casts ~2.6ms/token;
~10ms/token host-serial step — next structural items.
