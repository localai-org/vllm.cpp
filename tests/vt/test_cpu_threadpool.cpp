// Ported from llama.cpp (local fork @ 237ad9b96) tests/test-barrier.cpp
// (barrier stress over repeated parallel ops — the reach-all-threads assertion
// shape, plus test_active's mixed-thread-count rounds re-expressed as mixed
// chunk counts) and tests/test-thread-safety.cpp (REDUCED: concurrent op
// submission from one process serialized by the pool; the full multi-context
// server case stays with SERVE-E2E-NIGHTLY — see the tracked SKIP below).
// The determinism A/B battery has no upstream analogue: it is spec gate 1 of
// .agents/specs/gguf-cpu-threadpool.md (byte-identical outputs across
// VLLM_CPP_CPU_THREADS 1/3/20; 3 = non-divisor thread count for boundary bugs).
#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "vt/cpu/cpu_threadpool.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::DType;
using vt::Device;
using vt::DeviceType;
using vt::Queue;
using vt::Tensor;
using vt::cpu::ParallelForRows;
using vt::cpu::Threadpool;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

// Deterministic LCG fill (same values every run/thread count).
void FillF32(std::vector<float>& v, uint32_t seed) {
  uint32_t s = seed * 2654435761u + 1u;
  for (float& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / 16777216.0f) * 2.0f - 1.0f;  // [-1, 1)
  }
}

std::vector<uint8_t> Bytes(const void* p, size_t n) {
  std::vector<uint8_t> b(n);
  std::memcpy(b.data(), p, n);
  return b;
}

// Slots still holding the pre-call sentinel. An output-element split must
// produce every element of its declared range, so a nonzero count means the
// kernel under-produced and left part of the output untouched.
int64_t Unwritten(const std::vector<float>& v) {
  int64_t n = 0;
  for (float x : v) n += std::isnan(x) ? 1 : 0;
  return n;
}

// Swap the kernels' dispatch pool for the scope of one battery run.
struct ScopedPool {
  explicit ScopedPool(Threadpool* tp) { prev = Threadpool::SwapForTesting(tp); }
  ~ScopedPool() { Threadpool::SwapForTesting(prev); }
  Threadpool* prev;
};

}  // namespace

// test-barrier.cpp test_barrier: lots of small parallel ops where the barriers
// (kick/park/chunk-steal) in between dominate; every thread must reach every
// round.
TEST_CASE("barrier stress: 1000 rounds of chunked parallel ops reach all threads") {
  constexpr int kThreads = 4;
  constexpr int kRounds = 1000;
  Threadpool pool(kThreads);
  REQUIRE(pool.NThreads() == kThreads);

  // One cache-padded slot per thread: no shared writes (determinism contract).
  std::vector<int64_t> rounds_seen(kThreads * 16, 0);
  std::vector<int64_t> chunks_done(kThreads * 16, 0);

  for (int r = 0; r < kRounds; ++r) {
    pool.Run([&](int ith, int nth) {
      REQUIRE(nth == kThreads);
      rounds_seen[static_cast<size_t>(ith) * 16] += 1;
      // Chunk-steal exercise with a mixed (per-round) chunk count.
      const int nchunk = nth + (r % 5);  // nth .. nth+4 chunks
      if (ith == 0) pool.ChunkSet(nth);
      pool.Barrier();
      int current = ith;
      while (current < nchunk) {
        chunks_done[static_cast<size_t>(ith) * 16] += 1;
        current = pool.ChunkAdd(1);
      }
    });
  }

  int64_t total_chunks = 0;
  for (int t = 0; t < kThreads; ++t) {
    CHECK(rounds_seen[static_cast<size_t>(t) * 16] == kRounds);  // reach-all-threads
    total_chunks += chunks_done[static_cast<size_t>(t) * 16];
  }
  // Every chunk of every round executed exactly once across the pool.
  int64_t expect = 0;
  for (int r = 0; r < kRounds; ++r) expect += kThreads + (r % 5);
  CHECK(total_chunks == expect);
}

TEST_CASE("ParallelForRows visits every row exactly once at mixed sizes") {
  for (int nth : {1, 3, 20}) {
    Threadpool pool(nth);
    for (int64_t nr : {int64_t{0}, int64_t{1}, int64_t{2}, int64_t{3}, int64_t{17},
                       int64_t{64}, int64_t{1000}, int64_t{4096}}) {
      std::vector<int> visits(static_cast<size_t>(nr), 0);
      ParallelForRows(pool, nr, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) visits[static_cast<size_t>(r)] += 1;
      });
      for (int64_t r = 0; r < nr; ++r) CHECK(visits[static_cast<size_t>(r)] == 1);
    }
  }
}

TEST_CASE("per-op epoch remains valid beyond the upstream signed 16-bit epoch") {
  // Upstream advances the packed epoch once per graph. Our adaptation advances
  // it once per op, so a signed 32-bit (epoch << 16) reaches UB after 32767
  // dispatches. Exercise that boundary explicitly; poll=0 keeps the stress
  // test bounded while retaining the real park/wake + barrier path.
  constexpr int kDispatches = 33000;
  Threadpool pool(2, 0);
  std::atomic<int64_t> visits{0};
  for (int i = 0; i < kDispatches; ++i) {
    pool.Run([&](int, int) { visits.fetch_add(1, std::memory_order_relaxed); });
  }
  CHECK(visits.load(std::memory_order_relaxed) == int64_t{kDispatches} * 2);
}

// test-thread-safety.cpp REDUCED: several host threads submit ops through the
// same pool concurrently; the pool serializes dispatch and every result is
// correct. (The full multi-model/multi-context server case is intentionally
// NOT ported here: it needs real model contexts and belongs to the
// SERVE-E2E-NIGHTLY suite — tracked skip per test-porting.md rule 6.)
TEST_CASE("concurrent op submission from multiple host threads is serialized") {
  constexpr int kSubmitters = 4;
  constexpr int kIters = 25;
  constexpr int64_t kM = 8, kK = 16, kN = 8;

  // Reference result (single-threaded submitter, whatever pool is global).
  std::vector<float> a(kM * kK), b(kK * kN), ref(kM * kN);
  FillF32(a, 7);
  FillF32(b, 11);
  {
    Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {kM, kK});
    Tensor tb = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {kK, kN});
    Tensor to = Tensor::Contiguous(ref.data(), DType::kF32, Cpu(), {kM, kN});
    Queue q = Q();
    vt::Matmul(q, to, ta, tb);
  }

  std::atomic<int> mismatches{0};
  std::vector<std::thread> submitters;
  submitters.reserve(kSubmitters);
  for (int s = 0; s < kSubmitters; ++s) {
    submitters.emplace_back([&]() {
      std::vector<float> out(kM * kN);
      Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {kM, kK});
      Tensor tb = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {kK, kN});
      Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {kM, kN});
      Queue q = Q();
      for (int i = 0; i < kIters; ++i) {
        std::fill(out.begin(), out.end(), -1.0f);
        vt::Matmul(q, to, ta, tb);
        if (std::memcmp(out.data(), ref.data(), out.size() * sizeof(float)) != 0) {
          mismatches.fetch_add(1);
        }
      }
    });
  }
  for (auto& t : submitters) t.join();
  CHECK(mismatches.load() == 0);
}

TEST_CASE("thread safety full multi-context server case is tracked nightly" *
          doctest::skip(true) *
          doctest::description("requires SERVE-E2E-NIGHTLY checkpoint provisioning")) {
  // Porting debt from llama.cpp tests/test-thread-safety.cpp:16-165. This leaf
  // ports the shared-pool concurrent-submit semantics above; the upstream
  // multi-model/multi-context real-server case needs checkpoint provisioning
  // and remains assigned to SERVE-E2E-NIGHTLY (test-porting.md rule 6).
}

TEST_CASE("thread count: constructor clamps; global pool respects the env contract") {
  Threadpool zero(0);
  CHECK(zero.NThreads() == 1);
  Threadpool big(1 << 20);
  CHECK(big.NThreads() == vt::cpu::kMaxThreads);
  // Global(): VLLM_CPP_CPU_THREADS if set, else hardware_concurrency, >= 1.
  CHECK(Threadpool::Global().NThreads() >= 1);
  if (const char* e = std::getenv("VLLM_CPP_CPU_THREADS")) {
    const int want = std::max(1, std::min(atoi(e), vt::cpu::kMaxThreads));
    CHECK(Threadpool::Global().NThreads() == want);
  }
}

TEST_CASE("worker exceptions rethrow on the submitting thread") {
  Threadpool pool(3);
  ScopedPool swap(&pool);
  // embedding id out of range fires VT_CHECK inside the parallel body.
  std::vector<float> table(4 * 2);
  FillF32(table, 3);
  std::vector<int32_t> ids = {0, 1, 99, 2, 3, 1, 0, 2};  // 99 out of range
  std::vector<float> out(ids.size() * 2);
  Tensor tt = Tensor::Contiguous(table.data(), DType::kF32, Cpu(), {4, 2});
  Tensor ti = Tensor::Contiguous(ids.data(), DType::kI32, Cpu(),
                                 {static_cast<int64_t>(ids.size())});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(),
                                 {static_cast<int64_t>(ids.size()), 2});
  Queue q = Q();
  CHECK_THROWS_AS(vt::Embedding(q, to, tt, ti), std::runtime_error);
  // The pool stays usable after the exception.
  ids[2] = 3;
  vt::Embedding(q, to, tt, ti);
  CHECK(out[2 * 2] == table[3 * 2]);
}

TEST_CASE("nested parallel dispatch from a worker body fails loudly") {
  Threadpool pool(2);
  CHECK_THROWS_AS(
      pool.Run([&](int, int) {
        pool.Run([](int, int) {});  // would deadlock without the guard
      }),
      std::runtime_error);
}

// ---------------------------------------------------------------------------
// Determinism battery (spec gate 1): representative kernels produce
// byte-identical outputs at n_threads 1 / 3 / 20. Odd, non-multiple-of-16
// shapes stress chunk boundaries; 3 is a non-divisor thread count.
// ---------------------------------------------------------------------------

namespace {

// Runs the battery through the given pool and returns every output buffer
// (including in/out state tensors) as raw bytes.
std::vector<std::vector<uint8_t>> RunBattery(Threadpool& pool) {
  ScopedPool swap(&pool);
  Queue q = Q();
  std::vector<std::vector<uint8_t>> outs;

  // Matmul f32 [37,129]@[129,53] — small grid → per-thread re-chunk path.
  {
    std::vector<float> a(37 * 129), b(129 * 53), out(37 * 53);
    FillF32(a, 1);
    FillF32(b, 2);
    Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {37, 129});
    Tensor tb = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {129, 53});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {37, 53});
    vt::Matmul(q, to, ta, tb);
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
  }
  // Matmul [48,64]@[64,512] — nchunk0*nchunk1 = 32*3 = 96 >= 20*4 → chunk grid
  // + atomic stealing path; bf16 output exercises the store rounding.
  {
    std::vector<float> a(48 * 64), b(64 * 512);
    std::vector<uint16_t> out(48 * 512);
    FillF32(a, 3);
    FillF32(b, 4);
    Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {48, 64});
    Tensor tb = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {64, 512});
    Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, Cpu(), {48, 512});
    vt::Matmul(q, to, ta, tb);
    outs.push_back(Bytes(out.data(), out.size() * sizeof(uint16_t)));
  }
  // Decode-shaped Matmul [1,129]@[129,517] — nr1==1 → chunk_size 64 branch.
  {
    std::vector<float> a(1 * 129), b(129 * 517), out(1 * 517);
    FillF32(a, 5);
    FillF32(b, 6);
    Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {1, 129});
    Tensor tb = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {129, 517});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 517});
    vt::Matmul(q, to, ta, tb);
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
  }
  // MatmulBT [37,129] @ [53,129]^T.
  {
    std::vector<float> a(37 * 129), b(53 * 129), out(37 * 53);
    FillF32(a, 7);
    FillF32(b, 8);
    Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {37, 129});
    Tensor tb = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {53, 129});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {37, 53});
    vt::MatmulBT(q, to, ta, tb);
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
  }
  // RmsNorm with residual (gemma) [37,129] — residual stream is also an output.
  {
    std::vector<float> x(37 * 129), w(129), res(37 * 129), out(37 * 129);
    FillF32(x, 9);
    FillF32(w, 10);
    FillF32(res, 11);
    Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {37, 129});
    Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {129});
    Tensor tr = Tensor::Contiguous(res.data(), DType::kF32, Cpu(), {37, 129});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {37, 129});
    vt::RmsNorm(q, to, tx, tw, vt::RmsNormArgs{1e-6f, true}, &tr);
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
    outs.push_back(Bytes(res.data(), res.size() * sizeof(float)));
  }
  // SiluAndMul [37,258].
  {
    std::vector<float> x(37 * 258), out(37 * 129);
    FillF32(x, 12);
    Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {37, 258});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {37, 129});
    vt::SiluAndMul(q, to, tx);
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
  }
  // Attention causal GQA t=33, Hq=4, Hk=2, D=16.
  {
    std::vector<float> qq(33 * 4 * 16), kk(33 * 2 * 16), vv(33 * 2 * 16),
        out(33 * 4 * 16);
    FillF32(qq, 13);
    FillF32(kk, 14);
    FillF32(vv, 15);
    Tensor tq = Tensor::Contiguous(qq.data(), DType::kF32, Cpu(), {33, 4, 16});
    Tensor tk = Tensor::Contiguous(kk.data(), DType::kF32, Cpu(), {33, 2, 16});
    Tensor tv = Tensor::Contiguous(vv.data(), DType::kF32, Cpu(), {33, 2, 16});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {33, 4, 16});
    vt::Attention(q, to, tq, tk, tv, vt::AttentionArgs{0.25f, true});
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
  }
  // GdnPrefill: 3 sequences (one empty), Hk=2, Hv=4, Dk=8, Dv=8 — state is an
  // in/out output too.
  {
    const int64_t total = 17;
    std::vector<float> qq(total * 2 * 8), kk(total * 2 * 8), vv(total * 4 * 8),
        g(total * 4), beta(total * 4), state(3 * 4 * 8 * 8, 0.0f),
        out(total * 4 * 8);
    FillF32(qq, 16);
    FillF32(kk, 17);
    FillF32(vv, 18);
    FillF32(g, 19);
    FillF32(beta, 20);
    std::vector<int32_t> qsl = {0, 5, 5, 17};
    Tensor tq = Tensor::Contiguous(qq.data(), DType::kF32, Cpu(), {total, 2, 8});
    Tensor tk = Tensor::Contiguous(kk.data(), DType::kF32, Cpu(), {total, 2, 8});
    Tensor tv = Tensor::Contiguous(vv.data(), DType::kF32, Cpu(), {total, 4, 8});
    Tensor tg = Tensor::Contiguous(g.data(), DType::kF32, Cpu(), {total, 4});
    Tensor tb = Tensor::Contiguous(beta.data(), DType::kF32, Cpu(), {total, 4});
    Tensor ts = Tensor::Contiguous(state.data(), DType::kF32, Cpu(), {3, 4, 8, 8});
    Tensor tqsl = Tensor::Contiguous(qsl.data(), DType::kI32, Cpu(), {4});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {total, 4, 8});
    vt::GdnArgs args;
    args.scale = 0.353553f;
    vt::GdnPrefill(q, to, tq, tk, tv, tg, tb, ts, tqsl, args);
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
    outs.push_back(Bytes(state.data(), state.size() * sizeof(float)));
  }
  // GdnPrefill SINGLE sequence, Hv=20 > nth: exercises the (seq,head) parallel
  // axis at c1 (n==1, so heads are the only partitionable dimension). Hv=20 with
  // Hk=4 (ratio 5) and non-multiple-of-16 head dims stresses chunk boundaries.
  {
    const int64_t total = 23, Hk = 4, Hv = 20, Dk = 8, Dv = 8;
    std::vector<float> qq(total * Hk * Dk), kk(total * Hk * Dk), vv(total * Hv * Dv),
        g(total * Hv), beta(total * Hv), state(1 * Hv * Dv * Dk, 0.0f), out(total * Hv * Dv);
    FillF32(qq, 27);
    FillF32(kk, 28);
    FillF32(vv, 29);
    FillF32(g, 30);
    FillF32(beta, 31);
    std::vector<int32_t> qsl = {0, total};
    Tensor tq = Tensor::Contiguous(qq.data(), DType::kF32, Cpu(), {total, Hk, Dk});
    Tensor tk = Tensor::Contiguous(kk.data(), DType::kF32, Cpu(), {total, Hk, Dk});
    Tensor tv = Tensor::Contiguous(vv.data(), DType::kF32, Cpu(), {total, Hv, Dv});
    Tensor tg = Tensor::Contiguous(g.data(), DType::kF32, Cpu(), {total, Hv});
    Tensor tb = Tensor::Contiguous(beta.data(), DType::kF32, Cpu(), {total, Hv});
    Tensor ts = Tensor::Contiguous(state.data(), DType::kF32, Cpu(), {1, Hv, Dv, Dk});
    Tensor tqsl = Tensor::Contiguous(qsl.data(), DType::kI32, Cpu(), {2});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {total, Hv, Dv});
    vt::GdnArgs args;
    args.scale = 0.353553f;
    vt::GdnPrefill(q, to, tq, tk, tv, tg, tb, ts, tqsl, args);
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
    outs.push_back(Bytes(state.data(), state.size() * sizeof(float)));
  }
  // GdnPrefill BF16, 2 sequences (one EMPTY), Hk=2 Hv=4 Dk=Dv=64, 130 tokens.
  // KERNEL-GDN-CHUNKED-MIRROR T6/G5/R4. The two GdnPrefill entries above are
  // f32, so under D0's dtype predicate NEITHER of them reaches the chunked arm
  // -- they prove thread byte-identity for the algorithm this row did not
  // change. This entry is bf16 and does reach it, at 130 tokens so the
  // cross-chunk state carry (3 chunks incl. a partial tail) is inside the work
  // item rather than skipped.
  //
  // The chunked arm keeps byte-identity the same way the sequential one does:
  // the parallel axis is the (sequence, value-head) product, every reduction
  // lives inside one work item, and the chunk recurrence stays sequential
  // within it. Parallelising ACROSS chunks would break this, which is why R4
  // names it as a constraint on the parallelisation rather than a property to
  // hope for.
  {
    const int64_t total = 130, Hk = 2, Hv = 4, Dk = 64, Dv = 64;
    std::vector<float> qf(total * Hk * Dk), kf(total * Hk * Dk), vf(total * Hv * Dv),
        g(total * Hv), beta(total * Hv), state(2 * Hv * Dv * Dk, 0.0f),
        outf(total * Hv * Dv);
    FillF32(qf, 41);
    FillF32(kf, 42);
    FillF32(vf, 43);
    FillF32(g, 44);
    FillF32(beta, 45);
    // q and k reach GdnPrefill L2-NORMALISED, on every production path and in
    // every committed golden (`q_k_prenormalized: true`): the Qwen GDN layer
    // normalises in fused_post_conv_prep, and upstream's own CPU chunked test
    // runs the kernel with use_qk_l2norm_in_kernel=True. It is a PRECONDITION of
    // the op, not decoration, and un-normalised inputs make this suite useless
    // rather than merely noisy.
    //
    // THE MECHANISM IS THE GATED DELTA RULE ITSELF, NOT THE CHUNKED ARM. An
    // earlier draft of this comment blamed the materialised `(I+A)^-1`, on the
    // strength of seeing a 2.8e7 DIFFERENCE between the arms and assuming the
    // sequential one was well behaved. It is not: measured over 12 seeds at
    // T=70, Dk=Dv=128 with un-normalised q/k, the SEQUENTIAL recurrence reaches
    // max|out| of 1e23 to 1e31 and the chunked arm is SMALLER in every one of
    // them. With |k|^2 ~ Dk, the update S <- S(I - beta k k^T) + beta v k^T has a
    // per-token factor of order (1 - beta|k|^2), so |S| grows geometrically for
    // both arms. Normalisation is what makes |k|^2 = 1 and the recurrence a
    // contraction. There is no asymmetry between the arms to attribute here.
    auto l2 = [](std::vector<float>& x, int64_t rows, int64_t d) {
      for (int64_t r = 0; r < rows; ++r) {
        double ss = 0.0;
        for (int64_t j = 0; j < d; ++j) ss += static_cast<double>(x[r * d + j]) * x[r * d + j];
        const float inv = 1.0f / std::sqrt(static_cast<float>(ss) + 1e-6f);
        for (int64_t j = 0; j < d; ++j) x[r * d + j] *= inv;
      }
    };
    l2(qf, total * Hk, Dk);
    l2(kf, total * Hk, Dk);
    for (auto& x : g) x = -std::abs(x) * 0.25f;   // g is a log-decay: <= 0
    for (auto& x : beta) x = 0.1f + 0.8f * std::abs(x - std::floor(x));
    std::vector<uint16_t> qb(qf.size()), kb(kf.size()), vb(vf.size()),
        ob(outf.size(), 0);
    for (size_t i = 0; i < qf.size(); ++i) qb[i] = vt::F32ToBF16(qf[i]);
    for (size_t i = 0; i < kf.size(); ++i) kb[i] = vt::F32ToBF16(kf[i]);
    for (size_t i = 0; i < vf.size(); ++i) vb[i] = vt::F32ToBF16(vf[i]);
    std::vector<int32_t> qsl = {0, 130, 130};
    Tensor tq = Tensor::Contiguous(qb.data(), DType::kBF16, Cpu(), {total, Hk, Dk});
    Tensor tk = Tensor::Contiguous(kb.data(), DType::kBF16, Cpu(), {total, Hk, Dk});
    Tensor tv = Tensor::Contiguous(vb.data(), DType::kBF16, Cpu(), {total, Hv, Dv});
    Tensor tg = Tensor::Contiguous(g.data(), DType::kF32, Cpu(), {total, Hv});
    Tensor tb = Tensor::Contiguous(beta.data(), DType::kF32, Cpu(), {total, Hv});
    Tensor ts = Tensor::Contiguous(state.data(), DType::kF32, Cpu(), {2, Hv, Dv, Dk});
    Tensor tqsl = Tensor::Contiguous(qsl.data(), DType::kI32, Cpu(), {3});
    Tensor to = Tensor::Contiguous(ob.data(), DType::kBF16, Cpu(), {total, Hv, Dv});
    vt::GdnArgs args;
    args.scale = 0.125f;
    vt::GdnPrefill(q, to, tq, tk, tv, tg, tb, ts, tqsl, args);
    outs.push_back(Bytes(ob.data(), ob.size() * sizeof(uint16_t)));
    outs.push_back(Bytes(state.data(), state.size() * sizeof(float)));
  }
  // PagedAttention: one prefill sequence of 37 tokens (c1 prefill shape) — the
  // query-token rows are the parallel axis. Causal GQA Hq=4, Hk=2, D=8,
  // block_size=4, spanning 10 blocks. K/V live in a contiguous NHD cache filled
  // deterministically; block_table maps j → its cache block.
  {
    const int64_t T = 37, Hq = 4, Hk = 2, D = 8, bs = 4;
    const int64_t nblocks = (T + bs - 1) / bs;  // 10
    std::vector<float> query(T * Hq * D), out(T * Hq * D);
    // NHD cache (nblocks, bs, Hk, D) — one blocks-axis slice (unbind stride
    // matches the contiguous single-slice case).
    std::vector<float> kc(nblocks * bs * Hk * D), vc(nblocks * bs * Hk * D);
    FillF32(query, 32);
    FillF32(kc, 33);
    FillF32(vc, 34);
    std::vector<int32_t> block_table(nblocks);
    for (int64_t b = 0; b < nblocks; ++b) block_table[static_cast<size_t>(b)] = static_cast<int32_t>(b);
    std::vector<int32_t> seq_lens = {static_cast<int32_t>(T)};
    std::vector<int32_t> qsl = {0, static_cast<int32_t>(T)};
    Tensor tq = Tensor::Contiguous(query.data(), DType::kF32, Cpu(), {T, Hq, D});
    Tensor tkc = Tensor::Contiguous(kc.data(), DType::kF32, Cpu(), {nblocks, bs, Hk, D});
    Tensor tvc = Tensor::Contiguous(vc.data(), DType::kF32, Cpu(), {nblocks, bs, Hk, D});
    Tensor tbt = Tensor::Contiguous(block_table.data(), DType::kI32, Cpu(), {1, nblocks});
    Tensor tsl = Tensor::Contiguous(seq_lens.data(), DType::kI32, Cpu(), {1});
    Tensor tqsl = Tensor::Contiguous(qsl.data(), DType::kI32, Cpu(), {2});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {T, Hq, D});
    vt::PagedAttentionArgs args;
    args.scale = 0.353553f;
    args.causal = true;
    vt::PagedAttention(q, to, tq, tkc, tvc, tbt, tsl, tqsl, args);
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
  }
  // MoeRouterTopK t=29, E=16, k=4 (renormalize).
  {
    std::vector<float> logits(29 * 16), weights(29 * 4);
    std::vector<int32_t> indices(29 * 4);
    FillF32(logits, 21);
    Tensor tl = Tensor::Contiguous(logits.data(), DType::kF32, Cpu(), {29, 16});
    Tensor tw = Tensor::Contiguous(weights.data(), DType::kF32, Cpu(), {29, 4});
    Tensor ti = Tensor::Contiguous(indices.data(), DType::kI32, Cpu(), {29, 4});
    vt::MoeRouterTopK(q, tw, ti, tl, vt::MoeRouterTopKArgs{4, true});
    outs.push_back(Bytes(weights.data(), weights.size() * sizeof(float)));
    outs.push_back(Bytes(indices.data(), indices.size() * sizeof(int32_t)));
  }
  // L2Norm [29,31].
  {
    std::vector<float> x(29 * 31), out(29 * 31);
    FillF32(x, 22);
    Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {29, 31});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {29, 31});
    vt::L2Norm(q, to, tx, vt::L2NormArgs{1e-6f});
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
  }
  // CausalConv1dFwd: 2 sequences, C=13, K=4 — conv_state is an output too.
  {
    const int64_t total = 11, c_dim = 13, k = 4;
    std::vector<float> x(total * c_dim), w(c_dim * k), bias(c_dim),
        state(2 * c_dim * (k - 1)), out(total * c_dim);
    FillF32(x, 23);
    FillF32(w, 24);
    FillF32(bias, 25);
    FillF32(state, 26);
    std::vector<int32_t> qsl = {0, 4, 11};
    std::vector<int32_t> his = {1, 0};
    Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {total, c_dim});
    Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {c_dim, k});
    Tensor tbias = Tensor::Contiguous(bias.data(), DType::kF32, Cpu(), {c_dim});
    Tensor ts = Tensor::Contiguous(state.data(), DType::kF32, Cpu(), {2, c_dim, k - 1});
    Tensor tqsl = Tensor::Contiguous(qsl.data(), DType::kI32, Cpu(), {3});
    Tensor this_ = Tensor::Contiguous(his.data(), DType::kI32, Cpu(), {2});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {total, c_dim});
    vt::CausalConv1dFwd(q, to, tx, tw, &tbias, ts, tqsl, this_,
                        vt::CausalConv1dArgs{true});
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
    outs.push_back(Bytes(state.data(), state.size() * sizeof(float)));
  }
  // W5 split these four kernels over the output element range instead of over
  // rows. Their per-element expressions do not depend on the partition, so a
  // partition mistake cannot change a value. The entries assert two things
  // instead: the output bytes agree at every thread count, and every element of
  // the declared output range is written (the sentinel count), which an extent
  // that falls back to the row count leaves short. The shapes below are
  // non-decode so the chunk grid is wider than the pool, and the odd inner dims
  // put chunk boundaries inside a row.
  // GeluAndMul [7,26] -> [7,13].
  {
    const float kSentinel = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> x(7 * 26), out(7 * 13, kSentinel);
    FillF32(x, 35);
    Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {7, 26});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {7, 13});
    vt::GeluAndMul(q, to, tx);
    CHECK(Unwritten(out) == 0);
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
  }
  // RopeNeox GQA T=10, Hq=4, Hk=2, D=8, rotary_dim=6, in place on q and k.
  {
    const int64_t T = 10, Hq = 4, Hk = 2, D = 8;
    std::vector<float> qq(T * Hq * D), kk(T * Hk * D);
    std::vector<int32_t> pos(T);
    FillF32(qq, 36);
    FillF32(kk, 37);
    for (int64_t i = 0; i < T; ++i) pos[static_cast<size_t>(i)] = static_cast<int32_t>(1 + 3 * i);
    Tensor tq = Tensor::Contiguous(qq.data(), DType::kF32, Cpu(), {T, Hq, D});
    Tensor tk = Tensor::Contiguous(kk.data(), DType::kF32, Cpu(), {T, Hk, D});
    Tensor tp = Tensor::Contiguous(pos.data(), DType::kI32, Cpu(), {T});
    vt::RopeNeox(q, tq, tk, tp, vt::RopeArgs{10000.0f, 6});
    outs.push_back(Bytes(qq.data(), qq.size() * sizeof(float)));
    outs.push_back(Bytes(kk.data(), kk.size() * sizeof(float)));
  }
  // QkvSplit T=10, q_dim=12, k_dim=5, v_dim=7. All three outputs are compared.
  {
    const float kSentinel = std::numeric_limits<float>::quiet_NaN();
    const int64_t T = 10, Q = 12, K = 5, V = 7;
    std::vector<float> qkv(T * (Q + K + V)), qo(T * Q, kSentinel), ko(T * K, kSentinel),
        vo(T * V, kSentinel);
    FillF32(qkv, 38);
    Tensor tqkv = Tensor::Contiguous(qkv.data(), DType::kF32, Cpu(), {T, Q + K + V});
    Tensor tq = Tensor::Contiguous(qo.data(), DType::kF32, Cpu(), {T, Q});
    Tensor tk = Tensor::Contiguous(ko.data(), DType::kF32, Cpu(), {T, K});
    Tensor tv = Tensor::Contiguous(vo.data(), DType::kF32, Cpu(), {T, V});
    vt::QkvSplit(q, tq, tk, tv, tqkv);
    CHECK(Unwritten(qo) == 0);
    CHECK(Unwritten(ko) == 0);
    CHECK(Unwritten(vo) == 0);
    outs.push_back(Bytes(qo.data(), qo.size() * sizeof(float)));
    outs.push_back(Bytes(ko.data(), ko.size() * sizeof(float)));
    outs.push_back(Bytes(vo.data(), vo.size() * sizeof(float)));
  }
  // Add [6,64] plus a rank-1 bias [64]. The broadcast index is inside the
  // split element expression.
  {
    const float kSentinel = std::numeric_limits<float>::quiet_NaN();
    const int64_t rows = 6, D = 64;
    std::vector<float> a(rows * D), bias(D), out(rows * D, kSentinel);
    FillF32(a, 39);
    FillF32(bias, 40);
    Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {rows, D});
    Tensor tb = Tensor::Contiguous(bias.data(), DType::kF32, Cpu(), {D});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {rows, D});
    vt::Add(q, to, ta, tb);
    CHECK(Unwritten(out) == 0);
    outs.push_back(Bytes(out.data(), out.size() * sizeof(float)));
  }
  return outs;
}

}  // namespace

TEST_CASE("determinism: byte-identical outputs at n_threads 1 / 3 / 20") {
  Threadpool p1(1), p3(3), p20(20);
  auto ref = RunBattery(p1);
  auto got3 = RunBattery(p3);
  auto got20 = RunBattery(p20);
  REQUIRE(ref.size() == got3.size());
  REQUIRE(ref.size() == got20.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    CHECK_MESSAGE(ref[i] == got3[i], "buffer " << i << " differs at n_threads=3");
    CHECK_MESSAGE(ref[i] == got20[i], "buffer " << i << " differs at n_threads=20");
  }
}

// PERF-CPU-BARRIER (issue #391, spec cpu-decode-barrier-and-attn-dispatch.md
// §2). No upstream analogue: it pins the ONE recorded deviation from ggml's
// two spin-waits (the bounded spin then YieldThread in Threadpool::Barrier and
// ::PollForWork).
//
// The defect this catches: a spin-wait that never yields turns every dispatch
// into a full scheduler timeslice as soon as the pool is wider than the cores
// available to it, because the arrival everyone is waiting for is off-CPU while
// every other worker burns a core spinning. Measured on the pristine wait,
// 8 CPUs, empty op: 8 threads 1.84 us, 9 threads 6004 us — a 3265x cliff for
// one extra thread. With the yield the same pair is 1.68 us / 5.93 us.
//
// The assertion is a RATIO between two pools measured back to back in the same
// process, so it needs no absolute timing budget, and external load inflates
// BOTH sides, which shrinks the ratio. The 100x threshold sits ~30x below the
// defect and ~28x above the fixed behaviour.
//
// Know the one-sided failure mode before trusting a green: a box that is
// ALREADY saturated drives both arms to the same timeslice-bound cost and the
// ratio collapses toward 1, so this test can pass on the defective code there.
// Verified both ways on 20 cores: on the pristine wait at 1-min load 116 it
// reported 1.28-3.00 and PASSED, while on a quiet box the same pristine code
// reports 1835/2079/3836 and fails every time (21 threads costing 2999-5996 us
// against the fixed 18-20 us). It can
// therefore be trusted to fail loudly on a quiet machine and never to fail
// spuriously on a busy one, which is the safe direction for CI. Re-derive the
// defect on an idle box, not under a parallel gate run.
TEST_CASE("oversubscribed dispatch does not cost a scheduler timeslice") {
  const int cores = static_cast<int>(std::thread::hardware_concurrency());
  if (cores < 4) {
    MESSAGE("skipped: needs >= 4 cores to oversubscribe meaningfully");
    return;
  }
  // Half the cores always fits; one MORE than the cores never does. The
  // over arm is deliberately cores+1, not a multiple: that is the regime a
  // stock run lands in (pool = hardware_concurrency, plus our own async-
  // scheduling and API threads). A pool that is 2x the cores cannot be made
  // cheap by any wait policy — every dispatch must wait for 2x as many threads
  // as there are cores to be scheduled at least once — so testing that regime
  // would assert something this lever cannot deliver.
  const int fits = cores / 2;
  const int over = cores + 1;

  auto median_dispatch_us = [](int n_threads, int reps) {
    Threadpool tp(n_threads);
    for (int i = 0; i < 64; ++i) {
      tp.Run([](int, int) {});  // warm the workers out of the cond-var park
    }
    std::vector<double> us;
    us.reserve(static_cast<size_t>(reps));
    for (int i = 0; i < reps; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      tp.Run([](int, int) {});
      us.push_back(std::chrono::duration<double, std::micro>(
                       std::chrono::steady_clock::now() - t0)
                       .count());
    }
    std::sort(us.begin(), us.end());
    return us[us.size() / 2];
  };

  const double fits_us = median_dispatch_us(fits, 2000);
  const double over_us = median_dispatch_us(over, 2000);
  const double ratio = over_us / fits_us;
  MESSAGE("empty-op dispatch: " << fits << " threads " << fits_us << " us, "
                                << over << " threads " << over_us
                                << " us, ratio " << ratio);
  // Floor the threshold so a 4-vCPU runner whose fits arm measures
  // sub-microsecond dispatch does not trip the ratio on scheduler jitter
  // alone. The defect this test guards produces a full scheduler timeslice
  // (3-6 ms on 20 cores), so a 100 us floor sits ~30x below the defect floor.
  constexpr double kOverUsFloor = 100.0;
  CHECK_MESSAGE(over_us < std::max(100.0 * fits_us, kOverUsFloor),
                "oversubscribed dispatch costs " << over_us
                    << " us: the waiter is spinning through a "
                       "scheduler timeslice instead of yielding its core");
}
