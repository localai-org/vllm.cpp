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

## W5 RESULT (2026-11-01) — output-range split landed and measured

**Same-binary A/B.** One arm per server start, identical flags, 18 threads,
`VT_GGUF_KEEP_QUANT=1`, greedy 32 tokens, two runs per arm and three
interleaved base/final pairs in one session at 1-minute load 9.8-12.4 on the
AVX2 host:

| arm | wall median | CPU median |
|---|---|---|
| before | 6.355 s | 114.08 s |
| after | 5.495 s | 98.78 s |

Wall **1.157x**, CPU **1.155x**. Every pair favours the after arm (1.224x,
1.153x, 1.167x), so the ordering is not an artifact of drift. An earlier
three-run bracket and two live captures at load 58-79 are VOID and are not
quoted: the same binary moved between 13 and 27 s there.

**Byte identity.** All twelve greedy runs returned the same 32 tokens. A
`verify.py --expect-cache on` capture of the live service, compared field by
field against the pre-change capture, differs in zero fields outside `id` and
`created`.

**Review and coverage.** Ten mutations were applied to a scratch copy of the
head: both chunk-seam carries, both chunk starts, the rope split extent, the
k-head index, the q/k boundary, and the add extent and seam. Nine went RED on
a shipped suite. The tenth, `hh < hq` changed to `hh < hk`, stayed green
because every shipped rope case is `T=1, Hq=Hk=1`; the pool determinism
battery also omitted all four converted kernels. Both gaps are closed.

A GQA case at `T=10, Hq=4, Hk=2, D=8, rot=6` compares all 320 q and 160 k
slots bit-exactly against a reference computed from `cos`/`sin` in the test,
and every kernel-running case now holds 32 guard words on each side of q and
of k that are snapshotted before the call and compared bit for bit afterwards.
The guards are load-bearing rather than decorative: the boundary mutant writes
16 words before k and the head-shift mutant 30 words after it, so without them
the run ended in heap corruption and a SIGABRT instead of an assertion. With
them both mutants fail as assertion failures and exit normally, 233 and 156 of
1,142 at four threads. The battery runs the four kernels at 1, 3 and 20
threads, and reverting the add split extent leaves 378 of 384 elements
unwritten. `test_ops_rope` 7 cases / 1,142 assertions, `test_cpu_threadpool`
9 cases / 19,635 assertions (19,636 with the pool size set through the
environment), and `test_ops_activation`, `test_ops_qkv_merge` and
`test_ops_layernorm` unchanged at 27/291120, 2/14 and 6/9263.

**Rejected levers, measured in the same session.** Twelve and eight threads,
`taskset` pinning, `VT_CPU_MATMUL_STEAL=1` and `VT_CPU_SPIN_ROUNDS=0` all moved
less than the run-to-run spread. Allocation is 36 ms of the request. One thread
runs these kernels at llama.cpp's rate (1.47 s against 1.50 s per token), so
the gap was parallelism, not arithmetic.

**What stays open.** `kPagedAttention` still runs its decode body inline at
about 0.30 s per 32-token request and `kRmsNorm` keeps the row split at about
0.14 s; each is a separate change with its own issue. The ggml SIMD arm
(`ggml_vec_gelu_f32`) stays owed as recorded above. The after arm is 5.495 s
against llama.cpp's 5.0-5.3 s for the same file on the same host, so the
remaining gap is small but not closed.

**Evidence.** `evidence/gsq-lumen-w5-20261101/` in the serving deployment
(`README.md`, `ab-calm-20261101.json`, `verify-before/`, `verify-after/`,
`verify-live2/`).

## W6 — paged attention splits tokens, so decode runs every q head inline

**Status:** implemented and measured 2026-11-01 on the Lumen-3.8-Flare serving
host; issue
[ISSUE-LOCAL-01M6A7PTZC5T44XE7FCG4F2JHQ](../issues/QUANT-GGUF-CPU-THREADPOOL/ISSUE-LOCAL-01M6A7PTZC5T44XE7FCG4F2JHQ.md).
The design below is the committed pre-implementation text; the measured result
is in `## W6 RESULT`.

**What W5 left measured.** `kPagedAttention` was second in the W5 inline table:
1,596 calls, 0.303 s of a 6.7-7.0 s 32-token greedy request, 190 us per call.
The kernel partitions the **query-token** axis only
([cpu_paged_attn.cpp:341](../../src/vt/cpu/cpu_paged_attn.cpp#L341)), so at
decode `total_q == 1`, `ParallelForRows` takes its `nr == 1` inline fallback,
and the caller runs all `hq` heads of the layer while the pool spins.

**Re-profiled on the W5 tree, 2026-11-01.** The same `VT_POOL_GAP` build,
linked from the deployment archive with the W5 `cpu_ops`/`cpu_layernorm`
objects, one 8-token warm-up plus two 32-token greedy requests (72 tokens),
18 threads, load 7.7: inline total 0.686 s. `PagedAttentionKernel` is now the
largest inline site by a wide margin at 2,898 calls and 0.4606 s (158.9 us per
call, 0.214 s per 32-token request, 4.0 % of the 5.27 s wall). The W5
conversions no longer appear in the inline table at all. Next after it:
`RmsNormKernel` 14,559 calls and 0.1833 s (0.081 s per request, 1.5 %).

**Design.** Split the flat `token x q-head` range: `nr = total_q * hq`, and map
index `i` inside a chunk to `t = i / hq`, `h = i % hq`. Every output element
`out[(t*hq + h)*d_v .. +d_v)` is already produced independently by exactly one
iteration of the existing head loop
([cpu_paged_attn.cpp:367-439](../../src/vt/cpu/cpu_paged_attn.cpp#L367)), and
the per-token values `r`, `p`, `seqlen`, `jmin`, `jmax` come from the
pre-computed `tok_*` arrays, so the chunk body can recompute them from `t`
without a barrier or a cross-chunk reduction. The scratch that today lives
above the token loop (`probs`, `acc`, `qrow`) moves inside the (t, h) body and
is reused across the pairs of one chunk: `probs` is re-`assign`ed per pair and
`acc` is re-zeroed, exactly as the token loop does today. `qrow` stays a
per-token widening and is cached on the last `t` a chunk visited, because flat
index order is monotone in `t`; a chunk that spans several heads of one token
still widens that token once.

**Bit-identity.** Each `(t, h)` pair keeps the identical expression, the
identical sequential pass order over its own keys and the identical rounding;
the change only moves which thread executes the pair. No cross-head or
cross-token reduction exists to change shape. Duplicate `qrow` widening across
chunks is read-only, so it cannot reorder arithmetic. The gate proves the claim
rather than assuming it.

**Upstream anchors.** ggml splits attention over the query head axis inside
`ggml_compute_forward_flash_attn_ext_f16`
(`ggml/src/ggml-cpu/ops.cpp:9131-9140`, the `for (q_head ...)` loop at oracle
pin b10451 `10bf611e`). At decode with `nek1 >= 512` it additionally splits the
KV axis into `(nek1 + nth - 1) / nth` chunks per thread and merges partial
softmax terms (`:9115-9130` in the same file); this change does **not** take
that axis, because a KV split changes the softmax reduction order and the
strict goldens require the sequential order. The extent idiom is the
`ith`/`nth` split W5 already uses (`ops.cpp:2129-2152`).

**Tests.** A decode-shaped `PagedAttention` case (`T=1`, `Hq=8`, `Hk=2`, `D=8`,
`block_size=4`, causal) must exist in the threadpool determinism battery and
agree byte for byte at 1, 3 and 20 pool threads; the existing 37-token prefill
case stays so the token loop keeps coverage. `test_ops_paged_attn` and
`test_ops_paged_attn_dtype` must stay green unchanged, because they already
hold decode, sliding-window, sink and block-spanning cases.

**Mutations the reviewer must apply.** `nr = total_q * hq` reduced by one (an
unwritten last output element); the shifted-store form of `h` (compute from
`(h+1)%hq` and store at the pair's `h`, or the reverse) — 1586 failures in
`test_ops_paged_attn` plus 171 in the dtype suite, with the threadpool battery
green because it is blind to a deterministic mapping error; the `qrow` cache
key never advanced (every token after the first reuses the first query row);
the per-pair `probs` scratch no longer sized — the `assign` is the only sizing,
so deleting it ends the run in SIGSEGV rather than an assertion. Shifting `h`
itself is a pure relabelling and fails nothing, and a sized-but-unzeroed
`probs` is green everywhere, because pass 1 writes every `probs[j-jmin]` that
pass 2 reads. A mutation must produce an observed failure, not merely a
different number.

**Gates.** Same-binary interleaved A/B on the serving host, 18 threads, greedy
32 tokens, at least three pairs; byte-identical greedy output between arms;
`verify.py --expect-cache on` against the live deployment; and a prefill check
that the first-token latency of a long prompt does not regress beyond the
run-to-run spread, because prefill is the shape where the flat split changes
chunk composition.

**Expected gain.** Inline cost 0.303 s per 32-token request, of which the eight
q heads per layer are the natural units; the pool can retire a call in about
one head plus one kick. The measured wall gain is recorded in the result
section, not assumed here.

**Stop conditions.** Stop and return `NEEDS_DECISION` if the split cannot keep
one thread per output element, if any A/B run changes a single output byte
without an explained arithmetic reason, or if prefill regresses beyond the
spread while the decode gain is inside it.

## W6 RESULT (2026-11-01) — the (token, q-head) split landed and measured

**Change.** `PagedAttentionKernel` hands `ParallelForRows` `total_q * hq`
instead of `total_q`, and maps a flat index `i` to `t = i / hq` and
`h = i % hq`. The softmax scratch (`probs`, `acc`) and the widened query row
(`qrow`) moved inside the `(t, h)` body; `qrow` is widened once per token and
cached on the last `t` a chunk visited. `diff -w` shows the per-pair body is
the previous head-loop iteration, dedented and with its loop indices named, so
the arithmetic, the order over keys and the rounding did not move.

**Profile that chose the lever.** A `VT_POOL_GAP` build of the W5 tree, linked
from the deployment archive with the W5 `cpu_ops` and `cpu_layernorm` objects,
one 8-token warm-up plus two 32-token greedy requests (72 tokens), 18 threads,
load 7.7: `PagedAttentionKernel` 2,898 calls and 0.4606 s (158.9 us per call,
0.214 s per 32-token request, 4.0 % of the 5.27 s wall). `RmsNormKernel` was
next at 14,559 calls and 0.1833 s. The W5 conversions do not appear in the
inline table any more.

**Same-binary A/B.** Both arms are linked from the same deployment archive
(`libvllm-g5.a`) with the same W5 objects and differ only in the
`cpu_paged_attn.cpp` object, because the served artifact cannot yet be built
from this branch (see Owed). 18 threads, `VT_GGUF_KEEP_QUANT=1`, greedy 32
tokens, two runs per arm, seven interleaved pairs in one session:

| bracket | before wall | after wall | wall ratio |
|---|---|---|---|
| load 12.9-17.6 (3 pairs) | 5.157 s | 4.902 s | 1.052x |
| load 7.2-13.6 (4 pairs) | 5.037 s | 4.818 s | 1.045x |
| pooled (7 pairs) | 5.078 s | 4.830 s | 1.051x |

CPU seconds move with wall, 91.28 s to 86.90 s (1.051x). Six of the seven
pairs favour the after arm (1.056, 1.093, 0.983, 1.093, 1.013, 1.039, 1.012);
the 0.983 pair ran at load 17.55 against 15.90 and is kept in the pooled number
rather than dropped. All fourteen greedy runs returned the same 32 tokens.

**Prefill does not regress.** One warm-up plus three 402-token prompts with
`max_tokens=1` per server, first prompt dropped as an allocation warm-up:
0.306 s and 0.304 s before against 0.246 s and 0.243 s after, about 1.25x
faster, because a 402-token prefill now divides its 3,216 pairs across the pool
instead of its 402 tokens.

**Review.** A fresh reviewer read the head and mutated a scratch copy. Every
mutation below was applied, run, and reverted byte for byte; the counts are
observed, not predicted.

| mutation | `test_ops_paged_attn` | `test_ops_paged_attn_dtype` | `test_cpu_threadpool` |
|---|---|---|---|
| `nr = total_q * hq - 1` (unwritten last output element) | 69 fail | 171 fail | 6 fail |
| shifted store: compute from `(h+1)%hq`, store at `h` | 1586 fail | 171 fail | green |
| `h` shifted throughout (`(h+1)%hq` both sides) | green | green | green |
| `qrow` cache key never advanced | 4 fail | 98 / 16 / 0 at 1 / 3 / 20 threads | 4 fail |
| `qoff = (h*total_q + t)*d_v` (pair order flipped) | 1652 fail | 147 fail | green |
| `probs` sizing (`assign`) deleted | SIGSEGV | SIGSEGV | SIGSEGV |
| `probs` sized but never written by `assign(n, NaN)` | green | green | green |

The shifted store is the important one: the head mapping is a deterministic
mis-map, so the threadpool battery cannot see it and the two head tests can.
Shifting `h` on both sides is a pure relabelling and fails nothing, which is
why the battery comment now delegates the head mapping to
`test_ops_paged_attn.cpp:463` and `test_ops_paged_attn_dtype.cpp:421`. Deleting
the `assign` is the only sizing in the body, so that mutant dies in a signal
rather than an assertion; the NaN-poison variant proves the stated reason for
the green case, that pass 1 writes every `probs[j-jmin]` that pass 2 reads.

Reachability was mutated too: removing the `RegisterOp` entry fails the suites,
while removing the `qwen3_5` call site stays green because the tests enter at
the op seam. The production call site is exercised by the e2e A/B above, not by
the unit suites.

The first review returned three findings, all in prose that claimed coverage
the battery does not provide: the decode comment, the `qrow` cache comment and
this section's mutation list. A fresh implementer corrected all three in
`0903be579`, and a second fresh reviewer reproduced every mutation above and
returned `PASS` with no residual finding.

**Gate.** Release build, `ninja -j 6`, one binary per suite:

| `VLLM_CPP_CPU_THREADS` | `test_ops_paged_attn` | `test_ops_paged_attn_dtype` | `test_cpu_threadpool` |
|---|---|---|---|
| unset | 15 / 1838 | 3 / 172 | 9 / 19650 |
| 1 | 15 / 1838 | 3 / 172 | 9 / 19651 |
| 3 | 15 / 1838 | 3 / 172 | 9 / 19651 |
| 20 | 15 / 1838 | 3 / 172 | 9 / 19651 |

One pre-existing skip runs in every threadpool configuration. The thread count
adds one assertion because the explicit-pool case asserts its configured size.
`test_cuda_quant_dot.cpp` fails to compile in this CUDA-off configuration on
main's own text (`-Werror=unused-parameter`), so the whole-tree `ninja` target
stops there; the three suites above were built by name and are current.

**Live deployment.** `bin/lumenengine-server` at sha256 `9661f2cc` replaces the
W5 binary, which is kept as `bin/lumenengine-server.before-w6-20261101`
(sha256 `0f8d6cff`). The server restarts from `./serve.sh gsq-rco-cpu`, answers
`/health` with 200, and holds 6.54 GB RSS with the same flags as before. Two
`verify.py --expect-cache on` captures on the new process report 32 hits of 115
queries and then 96; that is the same pair W5 recorded in the opposite order
(96 then 32), so the alternation is cache warmth, not the kernel. `chat.json`,
`completion-1.json`, `completion-2.json`, `models.json` and `health.txt` are
field-identical to the W5 `verify-final` capture outside `id` and `created`.
The rebuilt kernel object is byte-identical to the object compiled from this
commit's `cpu_paged_attn.cpp`, and the deployed binary's
`PagedAttentionKernel` disassembles identically to that rebuild.

**What stays open.** `kRmsNorm` keeps its row split and about 0.081 s per
32-token request, the largest remaining inline site. The pool is now at
4.830 s against llama.cpp's 5.0-5.3 s for the same file on the same host, but
an official same-session denominator is a separate measurement and the gap is
not declared closed.

**Evidence.** `evidence/gsq-lumen-w6-20261101/` in the serving deployment
(`README.md`, `ab/`, `prefill/`, `profile/inline-post-w5.txt`,
`binaries/cpu_paged_attn.sha256`, `verify-after/`, `verify-after2/`).


## Owed

- ISSUE-LOCAL-01M6A8SZSN03ZNFY8Y7HB6M7TZ: the served Lumen-3.8-Flare binary cannot
  be reproduced from main. The GGUF KV array budget (1e6 against this artifact's
  1,301,418 elements) and the `gemma4` GGUF architecture exist only in the
  deployment scratch tree, so W6 measures on a binary linked from the deployment
  archive and the fix needs its own row.
