# the TT-DECODE-FUSION fused decode: measured (2026-09-30)

Branch `row/tt-decode-fusion` (worktree /tmp/vllm-region-capture-spec),
kernel commit `1d84f00a6`, pin `6449cf13f7b`, build2, Blackhole P150.

## 1. What landed

ONE full-grid `CreateKernelFromString` data-movement kernel
(`kKeepQuantDecodeFusedKernelSrc`, RISCV_0, DM_DEDICATED_NOC, O2, compiled
against `kernels/keepquant_kernel_code.h`) replaces the ~85-program eltwise
Q6_K decode chain per chunk. Host dispatch `DecodeKeepQuantWordsFusedQ6K`
keeps the int8dot house style: ONE `SetCommonRuntimeArgs` vector (7 words),
2 uniform self-cycled CBs (words row page + 1 KiB out block page),
in-kernel `row0/rowc` derivation with idle/partial tails, warm-first
workload cache keyed (rows x nb x grid) that refuses a capture-time miss by
name. The chain stays as the named fallback: `VT_TT_KEEPQUANT_FUSED=0`
forces it; an over-L1-budget shape declines by name (`[TT-KQ-FUSED]
decline:`) and falls through.

## 2. The red gate goes GREEN — the row's arbiter

Region-replay doctest (tests/vt/test_tenstorrent_backend.cpp:11527),
`/tmp/kq-focused.log` (2026-09-30):

| arm | region 1 (keepquant) record |
|---|---|
| pre-fusion (RED by design, `7a0f1ca3f`) | 2,965,504 B |
| fused (this row) | **32,768 B** |

32,768 B <= the 64 KiB gate: 8 chunks x (ONE fused launch at the 1,024-4 KB
packed floor) + the ~8 matmul-chain programs, exactly the spec's
"~8-16 KB + the matmul term" prediction. The whole-graph trace-fit leg
re-measures the 2,925,109,248 B demand below.

## 3. Bit-exactness, fused vs chain

`kTENSTORRENT fused keep-quant decode is bit-exact to the chain (Q6_K:
idle-core and partial-last-core tails)` — `/tmp/kqf-bitexact.log`,
2026-09-30: **1 case, 7/7 assertions, EXIT=0.** Byte-identity of the
grouped P=1 decode op output (decode -> typecast/to_layout -> broadcast
multiply -> sum, identical downstream ops on both arms, so decode bit
identity propagates) on rows=8 (idle cores), rows=250 and rows=251 (partial
last core), with planted zero-d and zero-scale blocks — the elements where
the chain repairs -0 by algebra and the kernel gets it from IEEE multiply
semantics must agree bit for bit, and do.

Per-enc-select: Q6_K is the only encoding the fused dispatch serves in
wave 1; every other enc-select arm takes the named decline into the proven
chain (so its golden is the chain itself, unchanged). The doctest runs both
arms of the SAME seam through the production op entry.

## 4. Full TT suite

`/tmp/kqf-suite.log` (2026-09-30, build2, full binary): **100 cases, 99
passed, 527,825/527,826 assertions**; the single failure is the
pre-recorded owed 2261-class RAC flake (`test_tenstorrent_backend.cpp:2261
CHECK( k_ok == total )`) — exactly the standing bar, and one case GREENER
than the pre-fusion record (the intentional red gate now passes). The
pre-existing teardown segfault after the run is unchanged.

## 5. The 27B money legs

- c1 INT8DOT=0 (`/tmp/kqf-c1-int8dot0.log`, 2026-09-30 18:21, fresh reset,
  `Qwen3.8-27B-Q4_K_M` 2x128/32, `--num-blocks 64 --max-num-batched-tokens
  64 --seed 0`): **BENCH_EXIT=1 — the whole-graph trace still does not fit;
  c1 does not serve; no TPOT table.** The capture died in
  `populate_mesh_buffer` (`mesh_trace.cpp:126`): `Trace buffer at address
  4,015,745,024 overlaps with DRAM activity during trace capture.
  Allocation high water mark: 4,228,372,992`. This is the fit wall's
  overlap-fatal shape, not the old single-allocation OOM — the decode fuse
  removed ~680 - ~64 programs per captured region (~2.9 MB -> 32,768 B at
  the vehicle shape, section 2), but the whole-graph demand (matmul chain +
  the rest of the 27B graph) still exceeds free DRAM, exactly the outcome
  the spec's `## Owed` named: the leg numbers decide the follow-up, and
  they point at the matmul-chain fuse + a re-measure of the whole-graph
  bound before any escalation beside tt-metal#57970.
- c1 INT8DOT=1 and c2: NOT run this window — the device lock carried
  multiple concurrent 27B legs (each up to 2.5 h), and the window closed
  after the INT8DOT=0 leg; both remain queued work for this row.

## Run recipe

`flock $HOME/gpu.lock`, `~/Sources/tt/luwen/target/release/reset` + 15 s,
`TT_METAL_HOME`/`TT_METAL_RUNTIME_ROOT` at
`~/Sources/tt/tt-metal-pin`, `LD_LIBRARY_PATH` pin-first
(`build_release_script/lib64` and `libexec/tt-metalium` BEFORE the
env-script's old paths — a stale-library mix exits 127 on symbol clash).
