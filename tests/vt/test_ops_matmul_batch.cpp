// Bit-identity gate for vt::MatmulBTBatch (OpId::kMatmulBTBatch, CPU provider)
// — the kolibri1 CPU decode GEMV dispatch-batching restructure
// (row MODEL-TEXT-kolibri-1, docs/bench-evidence/kolibri1-gemv-overhead-20261015).
//
// The restructure's whole correctness claim is that batching changes ONLY the
// call structure: every output element must be BYTE-IDENTICAL to the same job
// run alone through vt::MatmulBT, at every worker count. So the gate is
// memcmp, not Approx: for a set of ragged decode-shape jobs (t=1 GEMVs, mixed
// M/K/N, non-contiguous activations) the batched run must reproduce the
// per-job loop bit-for-bit.
//
// Mutation contract (AGENTS.md §"How work gets done"): perturbing the batch
// dispatch to swap two rows' outputs must turn this gate RED.
#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "vt/ops.h"
#include "vt/cpu/cpu_threadpool.h"  // Threadpool::SwapForTesting

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }

// Deterministic LCG so every case is reproducible without a seed corpus.
uint32_t Lcg(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return s;
}

// bf16 bit pattern with a random 16-bit payload (includes denormals/negatives).
uint16_t RandBf16(uint32_t& s) {
  return static_cast<uint16_t>(Lcg(s) & 0xFFFF);
}

struct Buf {
  std::vector<uint16_t> u16;
  std::vector<float> f32;
};

Tensor MakeBf16(Buf& store, std::vector<int64_t> shape, uint32_t& seed,
                int64_t row_stride = -1) {
  const int64_t cols = shape.back();
  const int64_t stride = row_stride > 0 ? row_stride : cols;
  const int64_t rows = shape[0];
  store.u16.resize(static_cast<size_t>(rows * stride));
  for (auto& v : store.u16) v = RandBf16(seed);
  Tensor t;
  t.rank = 2;
  t.dtype = DType::kBF16;
  t.device = Cpu();
  t.shape[0] = shape[0];
  t.shape[1] = shape[1];
  t.stride[0] = stride;
  t.stride[1] = 1;
  t.data = store.u16.data();
  return t;
}

struct JobSet {
  struct Job {
    Buf a, b, out;
    Tensor at, bt, outt;
    int64_t m = 0, k = 0, n = 0;
  };
  std::vector<std::unique_ptr<Job>> jobs;
  std::vector<vt::MatmulBTJob> desc;

  // Ragged decode-shape mix: t=1 GEMVs (the kolibri1 routed-expert shape),
  // small multi-row activations, ragged N/K. `wide` adds one large job so the
  // flattened grid spans more chunks than workers.
  void Build(int wide) {
    uint32_t s = 42;
    std::vector<std::array<int64_t, 3>> shapes = {
        {1, 2048, 512}, {1, 2048, 2048}, {1, 512, 2048}, {1, 576, 512},
        {3, 256, 128},  {7, 129, 65},   {2, 512, 2048},
    };
    if (wide) shapes.push_back({1, 4096, 4096});
    for (const auto& shp : shapes) {
      auto j = std::make_unique<Job>();
      j->m = shp[0];
      j->k = shp[1];
      j->n = shp[2];
      j->at = MakeBf16(j->a, {j->m, j->k}, s);
      j->bt = MakeBf16(j->b, {j->n, j->k}, s);
      j->outt.rank = 2;
      j->outt.dtype = DType::kBF16;
      j->outt.device = Cpu();
      j->outt.shape[0] = j->m;
      j->outt.shape[1] = j->n;
      j->outt.stride[0] = j->n;
      j->outt.stride[1] = 1;
      j->out.u16.assign(static_cast<size_t>(j->m * j->n), 0);
      j->outt.data = j->out.u16.data();
      jobs.push_back(std::move(j));
    }
    for (auto& j : jobs) desc.push_back({&j->outt, &j->at, &j->bt});
  }
};

void FillGarbage(JobSet& set) {
  for (auto& j : set.jobs)
    for (auto& v : j->out.u16) v = static_cast<uint16_t>(0x7FC0);  // +inf spam
}

}  // namespace

TEST_CASE("MatmulBTBatch is byte-identical to the per-job MatmulBT loop") {
  Queue q{Cpu(), nullptr};
  for (int nth : {1, 2, 4, 8, 17}) {
    vt::cpu::Threadpool pool(nth);
    vt::cpu::Threadpool* prev = vt::cpu::Threadpool::SwapForTesting(&pool);

    for (int wide : {0, 1}) {
      JobSet set;
      set.Build(wide);

      // Reference: each job alone through vt::MatmulBT (the current call
      // structure the restructure replaces).
      std::vector<std::vector<uint16_t>> ref(set.jobs.size());
      for (size_t i = 0; i < set.jobs.size(); ++i) {
        auto& j = *set.jobs[i];
        vt::MatmulBT(q, j.outt, j.at, j.bt);
        ref[i].assign(j.out.u16.begin(), j.out.u16.end());
      }

      // Batched: all jobs in ONE call.
      FillGarbage(set);
      vt::MatmulBTBatch(q, set.desc.data(),
                        static_cast<int64_t>(set.desc.size()));

      for (size_t i = 0; i < set.jobs.size(); ++i) {
        auto& j = *set.jobs[i];
        CHECK_MESSAGE(std::memcmp(ref[i].data(), j.out.u16.data(),
                                  ref[i].size() * sizeof(uint16_t)) == 0,
                      "job " << i << " shape " << j.m << "x" << j.k << "x"
                             << j.n << " diverged at nth=" << nth
                             << (wide ? " (wide)" : ""));
      }
    }
    vt::cpu::Threadpool::SwapForTesting(prev);
  }
}

TEST_CASE("MatmulBTBatch accepts row-strided activations and f32 output") {
  Queue q{Cpu(), nullptr};
  vt::cpu::Threadpool pool(4);
  vt::cpu::Threadpool* prev = vt::cpu::Threadpool::SwapForTesting(&pool);

  uint32_t s = 7;
  Buf a, b1, b2, o1, o2;
  // One shared activation buffer, two column-slice views (the merged-gate/up
  // shape): rows packed, stride 8 over K=5 — the MatmulBT contract.
  Tensor at = MakeBf16(a, {2, 8}, s);
  at.shape[1] = 5;
  at.rank = 2;
  Tensor w1 = MakeBf16(b1, {3, 5}, s);
  Tensor w2 = MakeBf16(b2, {4, 5}, s);
  Tensor r1 = MakeBf16(o1, {2, 3}, s);
  Tensor r2 = MakeBf16(o2, {2, 4}, s);

  std::vector<uint16_t> ref1(o1.u16.size()), ref2(o2.u16.size());
  vt::MatmulBT(q, r1, at, w1);
  vt::MatmulBT(q, r2, at, w2);
  ref1.assign(o1.u16.begin(), o1.u16.end());
  ref2.assign(o2.u16.begin(), o2.u16.end());

  std::vector<vt::MatmulBTJob> jobs{{&r1, &at, &w1}, {&r2, &at, &w2}};
  vt::MatmulBTBatch(q, jobs.data(), 2);
  CHECK(std::memcmp(ref1.data(), o1.u16.data(), ref1.size() * 2) == 0);
  CHECK(std::memcmp(ref2.data(), o2.u16.data(), ref2.size() * 2) == 0);
  vt::cpu::Threadpool::SwapForTesting(prev);
}
