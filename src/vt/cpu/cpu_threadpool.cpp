// Ported from llama.cpp (local fork) ggml/src/ggml-cpu/ggml-cpu.c @ 237ad9b96
// (b9892). See cpu_threadpool.h for the full upstream anchor map and the
// recorded deviations. Line references below are into that file unless noted.
#include "cpu_threadpool.h"

#include <algorithm>
#include <cstdlib>

#include "vt/dtype.h"  // VT_CHECK

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#else
#include <sched.h>
#endif

// Match ggml's GGML_TSAN_ENABLED fence workaround. GCC and Clang both warn
// that a standalone atomic_thread_fence is not modeled by ThreadSanitizer;
// the upstream TSAN build uses a no-op seq-cst RMW on the same synchronization
// atomic instead (ggml-cpu.c:594-600,3122-3128).
#if defined(__SANITIZE_THREAD__)
#define VT_CPU_THREAD_SANITIZER 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define VT_CPU_THREAD_SANITIZER 1
#endif
#endif
#ifndef VT_CPU_THREAD_SANITIZER
#define VT_CPU_THREAD_SANITIZER 0
#endif

#include <atomic>
#include <chrono>
#include <cstdlib>

namespace vt::cpu {
namespace {

// SCRATCH (VT_GEMV_PROF): state defined in cpu_threadpool.h.
inline uint64_t NowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}
// ggml_thread_cpu_relax, ggml-cpu.c:510-529 (arm yield / x86 _mm_pause /
// riscv pause / no-op fallback).
inline void CpuRelax() {
#if defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
  __asm__ volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(_M_X64)
  _mm_pause();
#else
  ;
#endif
}

// Hand the core back to the scheduler. Purely a scheduling hint: it reads and
// writes no synchronization state and orders no memory, so inserting it into a
// spin-wait changes only WHEN a waiter is on a CPU, never what any thread
// computes or observes (see kDefaultSpinRounds).
inline void YieldThread() {
#if defined(_WIN32)
  SwitchToThread();
#else
  sched_yield();
#endif
}

// Spin budget before a waiter yields its core. NOT ported: ggml's two
// spin-waits (ggml_barrier :587-589 and ggml_graph_compute_poll_for_work
// :3137-3139) relax without ever yielding, because upstream kicks the pool once
// per GRAPH — a spinner there is idle only across a node boundary. Our recorded
// per-OP adaptation (cpu_threadpool.h) kicks per operation, so a spinner is
// also idle across the caller's serial between-op work, and the number of
// waits per token is multiplied by the op count.
//
// A never-yielding spin-wait has a hard scheduler cliff at
// (runnable threads) > (available cores): the barrier's last arrival may be
// off-CPU while every other worker burns its core spinning, so the wait costs a
// full scheduler timeslice instead of a cache-line transfer. MEASURED on this
// tree with an empty op, 8 CPUs (taskset), 5000 dispatches: 8 pool threads =
// 1.84 us mean; 9 pool threads = 6004 us mean, p50 6000 us — exactly one CFS
// timeslice, a 3265x cliff for one extra thread. The default pool width is
// hardware_concurrency(), so any other runnable thread in the process — the
// async-scheduling thread, the API/CLI thread — puts a stock decode over that
// cliff.
//
// Yielding after a bounded spin removes the cliff (5516 us -> 5.93 us at
// 9 threads on 8 CPUs) and is free when the pool fits (1.63 -> 1.68 us at 8 on
// 8, inside the run-to-run spread). The budget is in relax rounds, so it is
// arch-dependent: x86 `pause` is tens of cycles, aarch64 `yield` is a hint that
// often retires in one. VT_CPU_SPIN_ROUNDS overrides it for a same-binary A/B;
// 0 disables yielding entirely and restores the upstream never-yield wait.
#if defined(__aarch64__)
inline constexpr long kDefaultSpinRounds = 4096;
#else
inline constexpr long kDefaultSpinRounds = 256;
#endif

long SpinRoundsFromEnv() {
  if (const char* e = std::getenv("VT_CPU_SPIN_ROUNDS")) {
    const long n = std::atol(e);
    if (n >= 0) {
      return n;
    }
  }
  return kDefaultSpinRounds;
}

// Read once at static-init time: this is a tuning constant, and both readers
// are inner spin loops that must not pay a magic-static guard per iteration.
const long g_spin_rounds = SpinRoundsFromEnv();

// Thread count selection (spec § env contract): VLLM_CPP_CPU_THREADS, default
// std::thread::hardware_concurrency(); clamped to [1, kMaxThreads]
// (GGML_MAX_N_THREADS analogue).
int ThreadsFromEnv() {
  int n = 0;
  if (const char* e = std::getenv("VLLM_CPP_CPU_THREADS")) {
    n = std::atoi(e);
  }
  if (n <= 0) {
    n = static_cast<int>(std::thread::hardware_concurrency());
  }
  return std::clamp(n, 1, kMaxThreads);
}

std::atomic<Threadpool*> g_test_pool{nullptr};

// Loud-failure guard: a kernel body must never dispatch another parallel op
// from inside a parallel region (the non-recursive run mutex / parked-worker
// protocol would deadlock, as in ggml where a node body never calls
// ggml_graph_compute). Tracked per thread; checked at Run() entry.
thread_local bool t_in_parallel_region = false;

struct ParallelRegionScope {
  ParallelRegionScope() { t_in_parallel_region = true; }
  ~ParallelRegionScope() { t_in_parallel_region = false; }
};

}  // namespace

// SCRATCH (VT_GEMV_PROF).
GemvProfState& GemvProf() {
  static GemvProfState s{.on = std::getenv("VT_GEMV_PROF") != nullptr};
  return s;
}
GemvProfState& GemvProfStateForOps() { return GemvProf(); }
uint64_t ProfNowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// ggml_threadpool_new_impl, ggml-cpu.c:3237-3308. Workers are allocated and
// initialized (:3261-3271), then threads 1..n-1 are spawned on the secondary
// loop (:3289-3294); worker 0 is the caller. Affinity/priority not ported
// (upstream defaults are inherit-affinity / normal priority, ggml.c:8136-8143).
Threadpool::Threadpool(int n_threads, uint32_t poll)
    : n_threads_(std::clamp(n_threads, 1, kMaxThreads)), poll_(poll) {
  workers_ = new ComputeState[static_cast<size_t>(n_threads_)];
  for (int j = 0; j < n_threads_; ++j) {
    workers_[j].threadpool = this;
    workers_[j].ith = j;
  }
  for (int j = 1; j < n_threads_; ++j) {
    workers_[j].thrd = std::thread([this, j] { SecondaryThread(workers_[j]); });
  }
}

// ggml_threadpool_free, ggml-cpu.c:2682-2711: set stop under the mutex,
// broadcast, join workers 1..n-1.
Threadpool::~Threadpool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_.store(true, std::memory_order_relaxed);
    pause_.store(false, std::memory_order_relaxed);
    cond_.notify_all();
  }
  for (int j = 1; j < n_threads_; ++j) {
    workers_[j].thrd.join();
  }
  delete[] workers_;
}

// ggml_barrier, ggml-cpu.c:566-602 (non-OpenMP branch): relaxed read of the
// passed-count, seq-cst fetch-add entry; the last thread resets n_barrier and
// bumps n_barrier_passed (seq-cst); spinners relax-wait then issue a full
// seq-cst fence on exit.
void Threadpool::Barrier() {
  const int n_threads = static_cast<int>(
      n_graph_.load(std::memory_order_relaxed) & kNThreadsMask);
  if (n_threads == 1) {
    return;
  }

  const int n_passed = n_barrier_passed_.load(std::memory_order_relaxed);

  // enter barrier (full seq-cst fence)
  const int n_barrier = n_barrier_.fetch_add(1, std::memory_order_seq_cst);

  if (n_barrier == n_threads - 1) {
    // last thread
    n_barrier_.store(0, std::memory_order_relaxed);
    // exit barrier (full seq-cst fence)
    n_barrier_passed_.fetch_add(1, std::memory_order_seq_cst);
    return;
  }

  // wait for other threads. Deviation from ggml-cpu.c:587-589: after a bounded
  // spin, hand the core back so the arrival we are waiting for can be
  // scheduled. The loop condition, the atomics, their memory orders and the
  // exit fence below are all untouched, so the set of observable states this
  // wait can exit in is unchanged — only how long it holds a CPU moves.
  long spins = 0;
  while (n_barrier_passed_.load(std::memory_order_relaxed) == n_passed) {
    CpuRelax();
    if (g_spin_rounds > 0 && ++spins >= g_spin_rounds) {
      spins = 0;
      YieldThread();
    }
  }

  // exit barrier (full seq-cst fence). TSAN does not model a standalone
  // fence, so mirror ggml's dummy seq-cst RMW in sanitizer builds.
#if VT_CPU_THREAD_SANITIZER
  n_barrier_passed_.fetch_add(0, std::memory_order_seq_cst);
#else
  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

// ggml_graph_compute_thread, ggml-cpu.c:3024-3097, per-op: nth comes from the
// n_graph mask (:3039), the stored work fn replaces the per-node loop
// (:3052-3082), and the final barrier (:3090) synchronizes completion.
// Deviation: exceptions from the fn are captured (first wins) for rethrow on
// the Run() caller.
void Threadpool::ComputeThread(ComputeState& state) {
  const int nth = static_cast<int>(
      n_graph_.load(std::memory_order_relaxed) & kNThreadsMask);
  const std::function<void(int, int)>* fn = work_;
  if (fn != nullptr && state.ith < nth) {
    ParallelRegionScope scope;
    try {
      (*fn)(state.ith, nth);
    } catch (...) {
      if (!has_exc_.exchange(true)) {
        exc_ = std::current_exception();
      }
    }
  }
  Barrier();
}

// ggml_graph_compute_thread_ready, ggml-cpu.c:3103-3118: exit polling/sleep on
// pending work, stop, or pause; a new n_graph epoch marks this thread pending
// iff its ith is below the epoch's active-thread count.
bool Threadpool::ThreadReady(ComputeState& state) {
  if (state.pending || stop_.load(std::memory_order_relaxed) ||
      pause_.load(std::memory_order_relaxed)) {
    return true;
  }

  // check for new graph/work
  const uint64_t n_graph = n_graph_.load(std::memory_order_relaxed);
  const int n_threads = static_cast<int>(n_graph & kNThreadsMask);
  if (n_graph != state.last_graph) {
    state.pending = state.ith < n_threads;
    state.last_graph = n_graph;
    return true;
  }

  return false;
}

// ggml_graph_compute_thread_sync, ggml-cpu.c:3121-3129: full seq-cst fence
// after a polling exit (the relaxed epoch read needs it before touching work).
void Threadpool::ThreadSync() {
#if VT_CPU_THREAD_SANITIZER
  n_graph_.fetch_add(0, std::memory_order_seq_cst);
#else
  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

// ggml_graph_compute_poll_for_work, ggml-cpu.c:3131-3144: 1024*128*poll relax
// rounds before falling back to the cond-var sleep.
bool Threadpool::PollForWork(ComputeState& state) {
  const uint64_t n_rounds = 1024UL * 128 * poll_;
  // Same deviation as Barrier(): a poller that never yields holds a core
  // through the caller's whole serial between-op window, which upstream does
  // not have (one kickoff per graph vs our per-op kickoff).
  long spins = 0;
  for (uint64_t i = 0; !ThreadReady(state) && i < n_rounds; ++i) {
    CpuRelax();
    if (g_spin_rounds > 0 && ++spins >= g_spin_rounds) {
      spins = 0;
      YieldThread();
    }
  }
  return state.pending;
}

// ggml_graph_compute_check_for_work, ggml-cpu.c:3146-3163: hybrid poll then
// cond-wait under the mutex.
bool Threadpool::CheckForWork(ComputeState& state) {
  if (PollForWork(state)) {
    ThreadSync();
    return state.pending;
  }

  {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!ThreadReady(state)) {
      // No new work. Wait for the signal.
      cond_.wait(lock);
    }
  }
  return state.pending;
}

// ggml_graph_compute_secondary_thread, ggml-cpu.c:3165-3200 (priority/affinity
// application dropped — not ported).
void Threadpool::SecondaryThread(ComputeState& state) {
  while (true) {
    // Check if we need to sleep
    while (pause_.load(std::memory_order_relaxed)) {
      std::unique_lock<std::mutex> lock(mutex_);
      if (pause_.load(std::memory_order_relaxed)) {
        cond_.wait(lock);
      }
    }

    // This needs to be checked for after the cond_wait
    if (stop_.load(std::memory_order_relaxed)) {
      break;
    }

    // Check if there is new work; the main thread is the only dispatcher.
    CheckForWork(state);
    if (state.pending) {
      state.pending = false;
      ComputeThread(state);
    }
  }
}

// ggml_graph_compute_kickoff, ggml-cpu.c:3202-3233: always take the mutex
// (workers do hybrid poll/wait), bump the epoch in the high bits and store the
// active-thread count in the mask (seq-cst store — the polling threads pair it
// with ThreadSync), then broadcast.
void Threadpool::Kickoff(int n_threads) {
  std::lock_guard<std::mutex> lock(mutex_);

  uint64_t n_graph = n_graph_.load(std::memory_order_relaxed) >> kNThreadsBits;
  n_graph = ((n_graph + 1) << kNThreadsBits) |
            (static_cast<uint64_t>(n_threads) & kNThreadsMask);

  n_graph_.store(n_graph, std::memory_order_seq_cst);

  if (pause_.load(std::memory_order_relaxed)) {
    // resume does cond broadcast (ggml_threadpool_resume_locked, :2721-2725)
    pause_.store(false, std::memory_order_relaxed);
  }
  cond_.notify_all();
}

// ggml_graph_compute, ggml-cpu.c:3314-3389 (persistent-pool branch), per-op:
// reset the chunk cursor (:3337), kick (:3373), caller computes as worker 0
// (:3376). The final barrier inside ComputeThread makes the return of this
// function the completion point for every output element.
void Threadpool::Run(const std::function<void(int, int)>& fn) {
  VT_CHECK(!t_in_parallel_region,
           "cpu_threadpool: nested parallel dispatch from inside a parallel region");
  // SCRATCH (VT_GEMV_PROF): wall+kick accounting for every pool kick.
  GemvProfState& gp = GemvProf();
  struct KickGuard {
    GemvProfState& gp;
    uint64_t t0;
    ~KickGuard() {
      if (gp.on) {
        gp.pool_ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
      }
    }
  } guard{gp, gp.on ? NowNs() : 0};
  if (gp.on) gp.kicks.fetch_add(1, std::memory_order_relaxed);
  // Deviation: serialize concurrent submitters (ggml assumes one dispatcher).
  std::lock_guard<std::mutex> run_lock(run_mutex_);

  work_ = &fn;
  current_chunk_.store(0, std::memory_order_relaxed);
  has_exc_.store(false, std::memory_order_relaxed);
  exc_ = nullptr;

  if (n_threads_ == 1) {
    // spec § Dispatch behavior: n_threads==1 short-circuits to the current
    // (inline) code path; keep the epoch mask coherent for Barrier().
    const uint64_t epoch =
        n_graph_.load(std::memory_order_relaxed) >> kNThreadsBits;
    n_graph_.store(((epoch + 1) << kNThreadsBits) | 1u,
                   std::memory_order_relaxed);
    {
      ParallelRegionScope scope;
      fn(0, 1);
    }
    work_ = nullptr;
    return;
  }

  Kickoff(n_threads_);

  // This is a work thread too (worker 0 = caller).
  ComputeThread(workers_[0]);

  work_ = nullptr;
  if (has_exc_.load(std::memory_order_relaxed)) {
    has_exc_.store(false, std::memory_order_relaxed);
    std::exception_ptr e = exc_;
    exc_ = nullptr;
    std::rethrow_exception(e);
  }
}

Threadpool& Threadpool::Global() {
  static Threadpool pool(ThreadsFromEnv());
  return pool;
}

Threadpool* Threadpool::SwapForTesting(Threadpool* tp) {
  return g_test_pool.exchange(tp);
}

Threadpool& CurrentThreadpool() {
  Threadpool* tp = g_test_pool.load(std::memory_order_acquire);
  return tp != nullptr ? *tp : Threadpool::Global();
}

// Flash-attn row chunking, ggml-cpu/ops.cpp:9070-9126: 4x chunks per thread
// (:9078-9081), per-thread re-chunk when the grid is smaller than nth or on
// NUMA (:9083-9085, IsNuma() stubbed), thread 0 seeds the steal cursor at nth
// then a barrier publishes it (:9087-9091), and each thread walks chunk ith
// first then steals via the atomic cursor (:9109-9122).
void ParallelForRows(Threadpool& tp, int64_t nr,
                     const std::function<void(int64_t, int64_t)>& body) {
  if (nr <= 0) {
    return;
  }
  // Min-work fallback (spec § Risks/decisions: "n_chunks==1 → run inline"):
  // a single output row has no partitionable work — the 4x-oversubscribed
  // grid degenerates to one chunk — so run it inline on the caller (the same
  // body over the same [0,1) range: bit-identical by construction, no kick).
  // n_threads==1 likewise short-circuits to the current single-thread code.
  if (nr == 1 || tp.NThreads() == 1) {
    body(0, nr);
    return;
  }
  tp.Run([&tp, nr, &body](int ith, int nth) {
    // 4x chunks per thread
    const int nth_scaled = nth * 4;
    const int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
    int64_t nchunk = (nr + chunk_size - 1) / chunk_size;

    if (nth == 1 || nchunk < nth || IsNuma()) {
      nchunk = nth;
    }

    if (ith == 0) {
      tp.ChunkSet(nth);
    }

    tp.Barrier();

    const int64_t dr = (nr + nchunk - 1) / nchunk;

    int64_t current_chunk = ith;

    while (current_chunk < nchunk) {
      const int64_t ir0 = dr * current_chunk;
      const int64_t ir1 = std::min(ir0 + dr, nr);

      if (ir0 < ir1) {
        body(ir0, ir1);
      }

      current_chunk = tp.ChunkAdd(1);
    }
  });
}

GemvProfSnap GemvProfTake() {
  GemvProfState& gp = GemvProf();
  GemvProfSnap s;
  s.kicks = gp.kicks.load(std::memory_order_relaxed);
  s.pool_ns = gp.pool_ns.load(std::memory_order_relaxed);
  s.busy_ns = gp.busy_ns.load(std::memory_order_relaxed);
  return s;
}
void GemvProfReset() {
  GemvProfState& gp = GemvProf();
  gp.kicks.store(0, std::memory_order_relaxed);
  gp.pool_ns.store(0, std::memory_order_relaxed);
  gp.busy_ns.store(0, std::memory_order_relaxed);
}

}  // namespace vt::cpu
