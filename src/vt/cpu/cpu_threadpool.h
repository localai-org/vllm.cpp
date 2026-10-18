// Ported from llama.cpp (local fork) ggml/src/ggml-cpu/ggml-cpu.c @ 237ad9b96
// (b9892) — the QUANT-GGUF-COMPUTE upstream pin. 1:1 port of ggml's native
// CPU threadpool (spec: .agents/specs/gguf-cpu-threadpool.md, row
// QUANT-GGUF-CPU-THREADPOOL):
//   struct ggml_threadpool + ggml_compute_state   ggml-cpu.c:471-507
//   ggml_thread_cpu_relax                         ggml-cpu.c:510-529
//   ggml_barrier                                  ggml-cpu.c:566-602
//   ggml_threadpool_chunk_set/_add                ggml-cpu.c:604-610
//   worker compute loop                           ggml-cpu.c:3024-3097
//   ready/sync/poll/check_for_work (park/wake)    ggml-cpu.c:3101-3163
//   secondary thread                              ggml-cpu.c:3165-3200
//   kickoff (n_graph epoch)                       ggml-cpu.c:3202-3233
//   pool create / free                            ggml-cpu.c:3237-3308, :2682-2711
//   compute entry (worker 0 = caller)             ggml-cpu.c:3314-3389
//   flash-attn row-chunk pattern (ParallelForRows) ggml-cpu/ops.cpp:9070-9126
//
// Recorded deviations (spec § Port map / Risks):
//  - C++ re-expression: std::thread/std::atomic/std::mutex/std::condition_variable
//    instead of pthreads/C11 atomics; same fields, alignment and memory-order
//    discipline.
//  - Per-OP dispatch, not per-graph: the pool stores one work function
//    fn(ith, nth) instead of a cgraph/cplan; Run() = {kick epoch, caller
//    computes as worker 0, final barrier} (ggml-cpu.c:3372-3376 + :3090).
//  - GGML_USE_OPENMP branches NOT ported (native pool only).
//  - NUMA/affinity/priority NOT ported: IsNuma() stubbed false; upstream
//    defaults are prio=0 + all-zero cpumask (= inherited affinity, ggml.c:8136),
//    so dropping the fields matches upstream default behavior on our
//    single-node targets.
//  - abort/ec dropped (no per-op abort callback); C++ exceptions thrown inside
//    the work fn are captured (first wins) and rethrown on the Run() caller,
//    preserving the kernels' VT_CHECK throw semantics.
//  - Run() takes an internal dispatch mutex so concurrent submitters from
//    different host threads are serialized (ggml relies on a single dispatcher;
//    mirrors the test-thread-safety.cpp guarantee at op granularity).
//
// Thread count: env VLLM_CPP_CPU_THREADS, default std::thread::hardware_concurrency()
// (spec § env contract; the B4 oracle arm ran 20/20 threads).
//
// DETERMINISM CONTRACT (spec § Dispatch behavior): parallelism partitions
// OUTPUT elements only; every output element is produced by exactly one thread
// running the same instruction sequence as the single-thread code. No atomic
// accumulation into shared outputs, no reduction-order changes — results are
// bit-identical to n_threads==1 by construction.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

namespace vt::cpu {

// ggml-cpu.c:60 GGML_CACHE_LINE
inline constexpr size_t kCacheLine = 64;
// ggml-cpu.c:202-203: n_graph packs (epoch << 16) | n_threads. Upstream uses
// an atomic int because it advances once per graph. Our recorded per-OP
// adaptation advances far more often, so keep the same packed layout in a
// uint64_t and avoid signed-shift UB / practical epoch rollover on long runs.
inline constexpr uint64_t kNThreadsMask =
    0xffffu;  // GGML_THREADPOOL_N_THREADS_MASK
inline constexpr int kNThreadsBits =
    16;  // GGML_THREADPOOL_N_THREADS_BITS
// ggml.h:GGML_MAX_N_THREADS (512).
inline constexpr int kMaxThreads = 512;
// ggml.c:8139 ggml_threadpool_params_init: poll = 50 (hybrid polling enabled).
inline constexpr uint32_t kDefaultPoll = 50;

// NUMA stub (spec § Risks/decisions): ggml_is_numa() gates chunk
// oversubscription + affinity upstream (ggml-cpu.c:613-716); both dev targets
// are single-node, so we pin false. Revisit with dual-socket hardware.
inline constexpr bool IsNuma() { return false; }

class Threadpool;

// Per-thread state — struct ggml_compute_state, ggml-cpu.c:497-507
// (cpumask dropped: affinity not ported, see header note).
struct ComputeState {
  std::thread thrd;         // ggml_thread_t thrd
  uint64_t last_graph = 0;  // widened epoch for the per-OP adaptation
  bool pending = false;     // bool pending
  Threadpool* threadpool = nullptr;
  int ith = 0;
};

// struct ggml_threadpool, ggml-cpu.c:471-495, re-expressed per-op. Its three
// cache-line-aligned atomics intentionally introduce padding between hot
// synchronization fields; MSVC C4324 describes that required layout.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
class Threadpool {
 public:
  // ggml_threadpool_new_impl, ggml-cpu.c:3237-3308 (created unpaused;
  // workers 1..n-1 spawned, worker 0 = caller).
  explicit Threadpool(int n_threads, uint32_t poll = kDefaultPoll);
  // ggml_threadpool_free, ggml-cpu.c:2682-2711.
  ~Threadpool();
  Threadpool(const Threadpool&) = delete;
  Threadpool& operator=(const Threadpool&) = delete;

  int NThreads() const { return n_threads_; }

  // Per-op analogue of ggml_graph_compute (ggml-cpu.c:3314-3389): reset the
  // chunk counter (:3337), kick the n_graph epoch (:3373), run fn(0, nth) on
  // the caller as worker 0 (:3376), and return after the final barrier
  // (:3090). fn is invoked once per thread as fn(ith, nth); n_threads==1
  // short-circuits to an inline call (spec § Dispatch behavior).
  void Run(const std::function<void(int ith, int nth)>& fn);

  // ggml_barrier, ggml-cpu.c:566-602. Callable only from inside a Run fn.
  void Barrier();

  // ggml_threadpool_chunk_set/_add, ggml-cpu.c:604-610 — the shared
  // work-stealing cursor used by the mul_mat / row-chunk policies.
  void ChunkSet(int value) {
    current_chunk_.store(value, std::memory_order_relaxed);
  }
  int ChunkAdd(int value) {
    return current_chunk_.fetch_add(value, std::memory_order_relaxed);
  }

  // Process-wide pool, created lazily on the first parallel CPU op (mirrors
  // ggml's persistent-threadpool path, not the disposable per-graph fallback;
  // spec § Port map "Pool lifetime"). Thread count: VLLM_CPP_CPU_THREADS,
  // default hardware_concurrency.
  static Threadpool& Global();
  // Test hook: swap the pool the CPU kernels dispatch through (returns the
  // previous one; pass nullptr to restore Global()). Determinism A/B tests
  // instantiate pools of different sizes in one process.
  static Threadpool* SwapForTesting(Threadpool* tp);

 private:
  // ggml_graph_compute_thread, ggml-cpu.c:3024-3097 (per-node loop replaced
  // by the single stored work fn; final barrier :3090 kept).
  void ComputeThread(ComputeState& state);
  // ggml_graph_compute_thread_ready, ggml-cpu.c:3103-3118.
  bool ThreadReady(ComputeState& state);
  // ggml_graph_compute_thread_sync, ggml-cpu.c:3121-3129.
  void ThreadSync();
  // ggml_graph_compute_poll_for_work, ggml-cpu.c:3131-3144.
  bool PollForWork(ComputeState& state);
  // ggml_graph_compute_check_for_work, ggml-cpu.c:3146-3163.
  bool CheckForWork(ComputeState& state);
  // ggml_graph_compute_secondary_thread, ggml-cpu.c:3165-3200.
  void SecondaryThread(ComputeState& state);
  // ggml_graph_compute_kickoff, ggml-cpu.c:3202-3233.
  void Kickoff(int n_threads);

  std::mutex mutex_;                 // mutex for cond.var (ggml-cpu.c:472)
  std::condition_variable cond_;     // cond.var for waiting for new work (:473)

  const std::function<void(int, int)>* work_ = nullptr;  // cgraph/cplan analogue

  // synchronization primitives (ggml-cpu.c:478-487)
  std::atomic<uint64_t> n_graph_{0};  // work epoch + active-thread count
  alignas(kCacheLine) std::atomic<int> n_barrier_{0};
  alignas(kCacheLine) std::atomic<int> n_barrier_passed_{0};
  alignas(kCacheLine) std::atomic<int> current_chunk_{0};
  std::atomic<bool> stop_{false};    // stopping the threadpool altogether
  std::atomic<bool> pause_{false};   // pausing the threadpool

  // exception plumbing (deviation, see header note)
  std::atomic<bool> has_exc_{false};
  std::exception_ptr exc_;

  ComputeState* workers_ = nullptr;  // per thread state (ggml-cpu.c:489)
  int n_threads_ = 1;                // number of threads in the pool
  uint32_t poll_ = kDefaultPoll;     // polling level (0 - no polling)

  std::mutex run_mutex_;             // deviation: serialize concurrent Run()
};
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// The pool the CPU kernels dispatch through: the test-swapped pool if set,
// else the lazy global.
Threadpool& CurrentThreadpool();

// Row/batch-chunked parallel-for — 1:1 port of the flash-attn row chunking
// pattern (ggml-cpu/ops.cpp:9070-9126): 4x-oversubscribed chunks, per-thread
// re-chunk fallback when the grid is too small (or NUMA — stubbed), atomic
// work stealing via the pool chunk cursor. body(r0, r1) processes output rows
// [r0, r1); every row is visited exactly once by exactly one thread.
void ParallelForRows(Threadpool& tp, int64_t nr,
                     const std::function<void(int64_t r0, int64_t r1)>& body);

// SCRATCH INSTRUMENTATION (row/kolibri-gemv-overhead, not for landing):
// env-gated (VT_GEMV_PROF) cumulative pool-kick counters so a caller can split
// wall time inside a kick into (a) the kick+handoff dispatch cost and (b) the
// busy compute summed across all workers.
struct GemvProfSnap {
  uint64_t kicks = 0;
  uint64_t pool_ns = 0;  // caller wall time inside kicks
  uint64_t busy_ns = 0;  // summed body execution across all workers
};
GemvProfSnap GemvProfTake();
void GemvProfReset();

// SCRATCH: accessors for cpu_ops.cpp MatmulChunked busy accounting.
struct GemvProfState {
  std::atomic<uint64_t> kicks{0};
  std::atomic<uint64_t> pool_ns{0};
  std::atomic<uint64_t> busy_ns{0};
  bool on = false;
};
GemvProfState& GemvProfStateForOps();
uint64_t ProfNowNs();

}  // namespace vt::cpu
