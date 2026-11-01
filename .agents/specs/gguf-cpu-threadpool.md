# Leaf spec: vt CPU threadpool — parallel op dispatch for the CPU backend

**Row:** `QUANT-GGUF-CPU-THREADPOOL` (quantization-matrix.md, leaf of the
`QUANT-GGUF-COMPUTE` block) · **status:** W1-W3 implemented and correctness
gated; `GATING` on the W4 idle-host performance/RSS reproduction ·
**upstream pin:** llama.cpp local fork `237ad9b96` (b9892) ·
**parent evidence:** the B4 CPU decision measurement
([parity-ledger.md L290](../parity-ledger.md#L290)): our CPU path is
single-threaded scalar loops and trails same-file llama.cpp by 54–75× decode /
≈1,480× prefill / 2.65× peak RSS on Qwen3.5-2B-UD-Q8_K_XL. The threadpool is
the PREREQUISITE leaf: ~N_core× on every op before any kernel work, and the
chunked-GEMM worker loop that `QUANT-GGUF-CIQ-GEMM` reuses.

## Scope

- **In scope:** a `vt::cpu` threadpool (create/park/wake/barrier/atomic chunk
  counter) ported 1:1 from ggml's CPU threadpool; conversion of the CPU op
  kernels in `src/vt/cpu/cpu_ops.cpp` to chunked parallel execution, GEMM
  first (`kMatmul`, `kMatmulBT`), then the row/batch-parallel remainder
  (norms, activations, attention, GDN, MoE glue); thread-count config
  (`VLLM_CPP_CPU_THREADS`, default = hardware concurrency) wired through the
  CPU backend at [cpu_backend.cpp:11](../../src/vt/cpu/cpu_backend.cpp#L11).
- **Out of scope:** quantized kernels (leaf `QUANT-GGUF-CIQ-GEMM`), loader
  changes (leaf `QUANT-GGUF-KEEPQ-LOADER`), CUDA paths (untouched), OpenMP
  (ggml's `GGML_USE_OPENMP` branch is NOT ported — we port the native
  pthread/atomic pool, one code path), NUMA node distribution (single-node
  stub; see Risks/decisions).
- **Dispatch behavior:** every `vt::cpu` kernel keeps its exact per-element
  math and per-output sequential reduction order; parallelism only partitions
  OUTPUT elements across threads (each output element is produced by exactly
  one thread with the same instruction sequence as today), so results are
  bit-identical to single-thread by construction — same argument as ggml's
  mul_mat row chunking. `n_threads==1` short-circuits to the current code.

## Upstream chain

All anchors in the pinned local fork `/home/mudler/_git/llama.cpp` @
`237ad9b96` (the `QUANT-GGUF-COMPUTE` upstream pin; the B4 bench arm ran this
exact tree with 20 threads).

| Component | Upstream anchor | What it does |
|---|---|---|
| Threadpool state | `ggml/src/ggml-cpu/ggml-cpu.c:471-508` | `struct ggml_threadpool` (mutex/cond, atomics `n_graph`, cache-aligned `n_barrier`/`n_barrier_passed`/`current_chunk`, `stop`/`pause`/`abort`, workers, `n_threads`, prio, poll) + per-thread `struct ggml_compute_state` (thread handle, `cpumask`, `ith`) |
| Spin relax | `ggml-cpu.c:510-530` | `ggml_thread_cpu_relax` (`yield`/`_mm_pause`) per arch |
| Barrier | `ggml-cpu.c:566-603` | `ggml_barrier`: atomic fetch-add entry, last thread resets and bumps `n_barrier_passed`, spinners relax-wait; seq-cst fences |
| Chunk counter | `ggml-cpu.c:604-610` | `ggml_threadpool_chunk_set/add` — the shared work-stealing cursor used by mul_mat and flash-attn |
| Worker loop | `ggml-cpu.c:3024-3101` | `ggml_graph_compute_thread`: per-node loop, `params{ith,nth,wsize,wdata,threadpool}`, barrier per node, abort check |
| Park/wake | `ggml-cpu.c:3103-3235` | `ggml_graph_compute_thread_ready/_sync`, polling (`poll` rounds then cond-wait), `n_graph` epoch increment as the kick |
| Pool create/destroy | `ggml-cpu.c:3237-3311`, `:2682-2745` | `ggml_threadpool_new_impl` (workers spawned via `ggml_thread_create` at `:3292`, worker 0 = caller), `ggml_threadpool_free`, pause/resume |
| Entry point | `ggml-cpu.c:3314-3390` | `ggml_graph_compute`: attach graph/cplan, kick epoch, main thread computes as worker 0, disposable-pool fallback |
| GEMM chunking policy | `ggml-cpu.c:1245-1443` | `ggml_compute_forward_mul_mat`: `chunk_size` 16 (64 when `nr0==1\|\|nr1==1`), `nchunk0×nchunk1` grid, re-chunk to `nth` when `<nth*4` chunks or NUMA, atomic `current_chunk` work stealing at `:1352-1357` and `:1437-1442` |
| GEMM chunk worker | `ggml-cpu.c:1155-1243` | `ggml_compute_forward_mul_mat_one_chunk`: 16×16 block tiling inside a chunk, `vec_dot` per output element |
| Work-size plan | `ggml-cpu.c:2752-2980` | `ggml_graph_plan`: per-op `n_tasks` + `work_size` (+`CACHE_LINE_SIZE*n_threads` padding) — the scratch model our per-call scratch mirrors |
| Non-GEMM split example | `ggml-cpu/ops.cpp:9070-9126` | flash-attn row chunking: 4×-oversubscribed chunks + `ggml_threadpool_chunk_add` stealing; `:9042-9069` decode KV-split + barrier + partial reduce |
| NUMA hooks | `ggml-cpu.c:538-560,613-716,2123-2168` | node discovery, `ggml_is_numa()` (disables chunk oversubscription), thread affinity — see Risks/decisions for our stub |

Runtime-trace plan: none needed — CPU dispatch is fully static (no heuristic
libraries); the B4 `llama-bench` arm already traced the real execution (20
threads, 99–100% × 20 cores).

## Our baseline

- [src/vt/cpu/cpu_ops.cpp:1](../../src/vt/cpu/cpu_ops.cpp#L1) — **zero
  threading anywhere**: `MatmulKernel`/`MatmulBTKernel` (`:30-57`) are naive
  sequential triple loops; all 30 registered CPU kernels (`:999-1066`) are
  single-thread scalar f32 loops. B4 observed 99–100% of ONE core for the
  whole run.
- [src/vt/cpu/cpu_backend.cpp:11](../../src/vt/cpu/cpu_backend.cpp#L11) —
  `CpuBackend` has no queue state (`CreateQueue` returns a null-handle
  `Queue`), no thread config, nothing to park/join.
- [include/vt/ops.h:285](../../include/vt/ops.h#L285) — op registry
  (`RegisterOp`/`GetOp`) dispatches per `(OpId, DeviceType)`; kernels receive
  `Queue&` first — the natural place to hang the pool without touching any
  call site.
- Tests: [tests/vt/test_ops_matmul.cpp](../../tests/vt/test_ops_matmul.cpp),
  [tests/parity/test_op_parity.cpp:34](../../tests/parity/test_op_parity.cpp#L34)
  run all CPU ops single-threaded today; the same binaries are the
  determinism harness (run at multiple thread counts).
- Honest gap: measured B4 floor ([parity-ledger.md L290](../parity-ledger.md#L290)) —
  ours 0.229 t/s prefill / 0.171 t/s decode / 7.43 GiB RSS vs llama.cpp
  372.65 t/s pp512 / 9.19–12.75 t/s tg / 2.80 GiB on the same file and box.

## Port map

| Upstream (llama.cpp @ 237ad9b96) | Local | Notes / deviations |
|---|---|---|
| `ggml-cpu.c:471-530` threadpool + compute_state structs | new `src/vt/cpu/cpu_threadpool.h` | C++ std::atomic re-expression, same fields/alignment annotations; drop `cgraph/cplan` (we dispatch per OP, not per graph — the "graph" is one op call) |
| `ggml-cpu.c:566-610` barrier + chunk counter | new `src/vt/cpu/cpu_threadpool.cpp` | port 1:1 incl. seq-cst fence discipline and relax-spin |
| `ggml-cpu.c:3024-3311` worker loop, park/wake epoch, create/free | `cpu_threadpool.cpp` | per-op kick instead of per-graph: `ParallelFor(pool, n_chunks, fn(ith,nth))` wraps {kick epoch, worker 0 = caller, barrier}; polling default `poll` behavior kept |
| `ggml-cpu.c:1245-1443` mul_mat chunk policy | `cpu_ops.cpp` `MatmulKernel`/`MatmulBTKernel` | same constants: chunk 16 (64 for vector shapes), `nth*4` oversubscription test, atomic stealing; inner 16×16 tiling from `:1155-1243`; per-element K loop byte-identical to today |
| `ggml-cpu/ops.cpp:9070-9126` row-chunk pattern | `cpu_ops.cpp` non-GEMM kernels | rows = tokens (RmsNorm/SiluAndMul/glue), (head,query) pairs (`AttentionKernel`), sequences (`GdnPrefillKernel`), batch (`GdnDecodeKernel`/conv update), token rows (`MoeRouterTopK`/`MoeCombine`/embedding) |
| `ggml-cpu.c` `GGML_USE_OPENMP` branches | NOT ported | single native-pthread path; recorded deviation |
| `ggml-cpu.c:613-716,2123-2168` NUMA | NOT ported (stub `IsNuma()==false`) | GB10 and the x86 bench VM are single-node; revisit with a dual-socket target |
| n_threads selection (`ggml_threadpool_params_default`, bench `-t 20`) | env `VLLM_CPP_CPU_THREADS`, default `std::thread::hardware_concurrency()` | B4 oracle ran 20/20 threads; default mirrors that operating point |

Pool lifetime: one static pool per process created lazily on first CPU op
(mirrors ggml's persistent threadpool path, not the disposable per-graph
fallback), workers parked between ops via the `n_graph` epoch + cond-wait.

## Tests to port

| Upstream test | Local tier / file | Notes |
|---|---|---|
| `tests/test-barrier.cpp` (llama.cpp) | T-unit `tests/vt/test_cpu_threadpool.cpp` (new) | barrier stress over repeated parallel ops, mixed chunk counts; port the reach-all-threads assertion shape |
| `tests/test-thread-safety.cpp` (llama.cpp) | T-unit, same file, reduced | concurrent op submission from one queue is serialized by the pool; full multi-context server case stays with `SERVE-E2E-NIGHTLY` (checked in SKIPPED with reason if not runnable at land time) |
| (no upstream analogue) determinism A/B | T-unit + T-parity | run the existing op suites ([test_ops_matmul.cpp](../../tests/vt/test_ops_matmul.cpp), [test_op_parity.cpp:34](../../tests/parity/test_op_parity.cpp#L34)) at `VLLM_CPP_CPU_THREADS=1,3,20` — byte-identical outputs required (3 = non-divisor thread count to catch boundary bugs) |
| existing GGUF engine gates | T-e2e [tests/parity/test_qwen36_gguf_engine.cpp:143](../../tests/parity/test_qwen36_gguf_engine.cpp#L143) | greedy 16/16 must stay token-exact with threads enabled (default on) |

## Gates

1. **Determinism (hard):** all existing CPU ctest suites + op-parity goldens
   byte-identical at `VLLM_CPP_CPU_THREADS ∈ {1, 3, 20}`; greedy GGUF engine
   gates 16/16 token-exact vs their existing goldens. Command:
   `VLLM_CPP_CPU_THREADS=N ctest --test-dir build -L cpu` (and the gguf
   suite: `ctest -R gguf`).
2. **Perf floor (B4-anchored, same box/file/recipe as
   [parity-ledger.md L290](../parity-ledger.md#L290)):** same-binary A/B
   `VLLM_CPP_CPU_THREADS=1` vs `=20` on the 20-core x86 box, Qwen3.5-2B
   Q8_K_XL, `vllm-bench --num-prompts 1 --input-len 128 --output-len 32`,
   ≥2 runs within noise: decode TPOT **≥10×** the single-thread arm
   (0.171 t/s → ≥1.7 t/s; target ~N_core×), prefill **≥10×**. Record the
   honest remaining gap vs llama.cpp tg32 12.75 t/s / pp128 339.63 t/s in
   the ledger (this leaf does NOT close it — compute-in-quant does).
3. **No-regression:** CUDA gate suites untouched/green (CPU-only change);
   single-thread mode still available (`=1`) as the debug/fallback path.
4. **Memory:** peak RSS within +5% of the single-thread arm (pool + per-thread
   scratch only; no weight-path change in this leaf).

## Dependencies

- None on other leaves (this is the first claimable leaf). `QUANT-GGUF-CIQ-GEMM`
  depends on THIS row's `ParallelFor`/chunk loop.
- Hardware: any multi-core box; the gate A/B runs on the same 20-core x86 box
  as B4 (idle, no GPU lock needed); a GB10 20-Arm-core rerun belongs to
  `BACKEND-GATE-CPU-LLAMACPP`, not this leaf.
- Models/data: `Qwen3.5-2B-UD-Q8_K_XL.gguf` (the B4 file) — NOTE: running it
  from main needs the dense-arch loader path, currently only on the unmerged
  bench branch; see `QUANT-GGUF-KEEPQ-LOADER` W-row L1 (merge `7c91a42`) or
  run the A/B from that branch rebased on the leaf, as B4 did.
- Toolchain: none new (std::thread/atomics; keep `-ffp-contract=off` pinned —
  bit-identity across compiled loops depends on it, see cpu_ops.cpp:876-881).

## Work breakdown

| W | Row (claim-sized) | Content | Depends |
|---|---|---|---|
| W1 | pool core | `cpu_threadpool.{h,cpp}`: struct, barrier, chunk counter, park/wake epoch, create/free, `VLLM_CPP_CPU_THREADS`, `ParallelFor` | - |
| W2 | GEMM chunking | `MatmulKernel`/`MatmulBTKernel` → mul_mat chunk policy + 16×16 tile worker; determinism A/B on op tests | W1 |
| W3 | non-GEMM ops | row/batch chunking for RmsNorm family, SiluAndMul/MoeSiluMul, Attention, GdnPrefill/Decode, conv1d, MoE router/combine, embedding, glue/fused-chain | W1 |
| W4 | gate + ledger | full determinism matrix, B4-recipe A/B, ledger row, matrix/roadmap flip | W2, W3 |

W2 and W3 are parallel-claimable after W1; W4 closes the row.

## Risks/decisions

- **NUMA stub (recorded deviation):** `ggml_is_numa()` gates chunk
  oversubscription and thread affinity upstream; both dev targets are
  single-node, so we stub `false` and skip affinity. Revisit only with real
  dual-socket hardware — not a product call, an environment fact.
- **No OpenMP path:** one native pool (ggml's non-OpenMP branch) keeps the
  build dependency-free; upstream keeps both only for platform breadth we
  don't have yet.
- **Per-op kick vs per-graph:** ggml amortizes wake-ups across a whole cgraph;
  we kick per op. Parked-worker wake cost is bounded by the polling window
  (ggml default polls before cond-wait); if per-op overhead shows up in the
  gate A/B at small shapes, the recorded fallback is a min-work threshold
  (`n_chunks==1` → run inline), which preserves determinism.
- **Determinism is load-bearing** for the whole parity method (goldens,
  VT_CPU_REF oracle in the sibling leaves): any future kernel change that
  splits a REDUCTION across threads (not just outputs) must move that op's
  goldens first — forbidden inside this leaf.

## Gating handoff (2026-07-10 crash recovery)

W1-W3 are implemented. The full CPU suite passes serially at
`VLLM_CPP_CPU_THREADS={1,3,20}` (94/94 each); the dedicated upstream-derived
suite also passes under GCC ThreadSanitizer (8 executed cases / 19,595
assertions, one explicitly registered `SERVE-E2E-NIGHTLY` skip). The local
35B/27B checkpoint gates were absent and therefore did not execute model
assertions.

W4 was deliberately not measured on the recovered host: at 2026-07-10 22:39
UTC the 20-core x86 box had persistent unowned `llama-cpp-avx512` and
`depth-anything-cpp` inference processes consuming CPU, with load average
`7.97/16.15/9.81`. Benchmark protocol makes a contended run void; those
processes were not killed or perturbed. The row therefore remains `GATING`.

Exact rerun after an exclusive idle-host window is available (hold the host
lock for the whole series; run each ours arm at least three times, interleaved):

```sh
export MODEL=/home/mudler/_git/prep-buddy/models/Qwen3.5-2B-UD-Q8_K_XL.gguf
git rev-parse HEAD
cmake -S . -B build-cpu-gate -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_SERVER=OFF
cmake --build build-cpu-gate -j 20 --target vllm-bench

flock /tmp/vllm-cpp-cpu-bench.lock sh -c '
  uptime
  ps -eo pid,psr,pcpu,pmem,comm,args --sort=-pcpu | head -20
  for rep in 1 2 3; do
    /usr/bin/time -v env VLLM_CPP_CPU_THREADS=1 \
      ./build-cpu-gate/examples/vllm-bench --model "$MODEL" \
      --num-prompts 1 --input-len 128 --output-len 32 --concurrency 1 \
      --seed 0 --temperature 0
    /usr/bin/time -v env VLLM_CPP_CPU_THREADS=20 \
      ./build-cpu-gate/examples/vllm-bench --model "$MODEL" \
      --num-prompts 1 --input-len 128 --output-len 32 --concurrency 1 \
      --seed 0 --temperature 0
  done
'
```

## W4 RESULT — reproduction TAKEN 2026-07-22, PARTIAL (`CLAIM-BENCH-CPU-LLAMA-REMEASURE-1`)

The idle-host window finally happened, on **`dgx.casa` (GB10, aarch64, 20
cores)** rather than the chronically co-tenanted x86 box: load average 0.01
before the series, `nvidia-smi` reporting no compute apps, the whole series
under one `flock $HOME/gpu.lock`. Same-binary A/B at `1cb5f64`, B4 recipe,
`Qwen3.5-2B-UD-Q8_K_XL.gguf`. Full context:
[CPU floor re-measurement](cpu-llamacpp-floor-remeasure-2026-07-22.md).

| | 1 thread | 20 threads | speed-up | bar | verdict |
|---|---|---|---|---|---|
| Prefill (TTFT) | 306,623.09 ms (0.4175 t/s) | 24,594.52 ms (5.205 t/s) | **12.47×** | ≥ 10× | **PASS** |
| Decode (TPOT) | 3,629.84 ms (0.2755 t/s) | 450.68 ms (2.219 t/s) | **8.05×** | ≥ 10× | **MISS** |
| Peak RSS | 7,788,056 KB | 7,788,220 KB | **1.000×** | ≤ 1.05× | **PASS** |

20-thread reproduction across 3 reps: TTFT spread **0.36 %**, TPOT **0.59 %**,
peak RSS identical to 12 KB. The x86 arm gives a consistent 13.6× / 7.8× but is
`VOID` for binding (co-tenant load).

**The row stays `GATING` on the decode axis.** 8.05× against a 10× bar is a
real miss, not a rounding one, and it is not noise. The most likely cause is
this leaf's own recorded risk — the **per-op kick** at M=1 decode shapes, where
the chunk grid is small and wake-up cost is a larger fraction of the op. The
spec already names the fallback (`n_chunks == 1` → run inline); that is now a
concrete, measured work item rather than a hypothetical.

**What this row is NOT responsible for.** The threadpool is the entire reason
the recorded CPU floor moved (decode 54–75× → 11.6×, prefill ≈1,480× → 33.5×
against llama.cpp), and the current binary at 1 thread reproduces the original
B4 numbers on B4's own box within 4.8 %/5.5 % — so W1–W3 are doing exactly what
they claimed. The *remaining* 11.6×/33.5× gap is attributed to `kMatmul` being
95.4 % of wall time at 6–18 GFLOP/s, which belongs to `QUANT-GGUF-CIQ-GEMM`
**G4**, not here.

Acceptance remains: both prefill and decode throughput at
20 threads are at least 10x their same-binary 1-thread arms, peak RSS is no
more than 1.05x, and outputs remain token-exact. In the same idle window,
rebuild a clean detached llama.cpp worktree at `237ad9b96` (`Release`,
`GGML_CUDA=OFF`, `GGML_NATIVE=ON`) and refresh the native floor:

```sh
llama-bench -m "$MODEL" -p 512,128 -n 128,32 -t 20 -r 5 -ngl 0
/usr/bin/time -v llama-bench -m "$MODEL" -p 0 -n 32 -t 20 -r 3 -ngl 0
```

## W5 — the row split degenerates at decode (T=1), so split the output range

**Status:** measured 2026-11-01 on the Lumen-3.8-Flare serving host; issue
[ISSUE-LOCAL-01M69GMT8WCQZMDNMEF8HH96D3](../issues/QUANT-GGUF-CPU-THREADPOOL/ISSUE-LOCAL-01M69GMT8WCQZMDNMEF8HH96D3.md).

**What W4 named, now measured.** W4 attributed the remaining decode gap to "the
per-op kick at M=1 decode shapes" and left the inline fallback as a work item.
It is real, but the cause is the split axis, not the kick. Every kernel ported
in W1-W3 partitions **rows**, and `ParallelForRows` runs `nr == 1` inline by
design. At decode `T == 1`, so every elementwise kernel hands it one row and
the caller executes the whole row while the other 17 workers poll. A SIGPROF
caller-thread profile plus an env-gated per-op timer (`VT_POOL_GAP`, scratch)
over one 32-token greedy request, 18 threads, 6.7-7.0 s wall:

| kernel | calls | inline total | per call |
|---|---|---|---|
| `kGeluAndMul` | 3,276 | 1.212 s | 370 us |
| `kPagedAttention` (decode) | 1,596 | 0.303 s | 190 us |
| `kRopeNeox` | 1,330 | 0.223 s | 167 us |
| `kAdd` | 4,788 | 0.167 s | 35 us |
| `kRmsNorm` | 8,018 | 0.139 s | 17 us |
| `kQkvSplit` | 1,596 | 0.119 s | 74 us |

Inline bodies total **2.197 s**, 31 % of the request, and 99.8 % of the
inter-kick gap is caller CPU rather than descheduling. Rejection recorded: the
kernels themselves are at per-thread parity with llama.cpp (1.47 vs 1.50 s per
token at one thread), allocation is not the cost (36 ms of 6.7 s in the
caller), and thread count, `taskset` pinning and `VT_CPU_MATMUL_STEAL` all
moved less than the run-to-run spread.

**Design.** Partition the **output element range**, not rows, for kernels whose
outputs are independent of each other. `ParallelForRows` is already a generic
splitter over a contiguous range, so no new dispatch seam is needed; the change
is the unit each caller hands it. Fourth and later passes are unchanged.
Converted in this change: `kGeluAndMul`, `kRopeNeox` (over `token x head`),
`kQkvSplit`, `kAdd`. `kMulScalar` and `kSoftCap` already split flat;
`kRmsNorm` keeps the row split because its reduction order must not move.

**Bit-identity.** Each output element is still produced by exactly one thread
with the identical expression and the identical rounding, and no cross-element
reduction changes shape, so the result is bit-identical to the single-thread
arm by construction — the same argument W1-W3 make for row chunking. The gate
proves it rather than assuming it.

**Upstream anchors.** ggml partitions elementwise ops over output rows with
`ith`/`nth` and `dr = (nr + nth - 1)/nth`
(`ggml/src/ggml-cpu/ops.cpp:2129-2152` at the W1 pin `237ad9b96`; the same shape
at oracle pin b10451 `10bf611e`, `ops.cpp:2129-2152`). Our tensors are 2-D
`[T, D]`, so the row count that ggml reads as `ne1*ne2*ne3` collapses to `T`;
this change restores the intended granularity for that layout. The SIMD arm
ggml pairs with it (`ggml_vec_gelu_f32` in `ggml/src/ggml-cpu/vec.cpp`) is
**not** ported here and stays owed — this change is the parallel half of the
same gap, and it keeps the exact scalar math vLLM's `activation.py:418`
requires.

**Tests and gates.** Same-binary A/B on one host in one session, greedy 32-token
output byte-identical between the arms; `test_cpu_threadpool` for the splitter
contract; the existing CPU op tests for the converted kernels; `verify.py
--expect-cache on` against the serving deployment.

**Stop conditions.** Stop and return `NEEDS_DECISION` if a conversion cannot
preserve the per-element expression, or if any A/B run changes a single output
byte without an explained arithmetic reason.
