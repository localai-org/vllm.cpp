// Op-level GFLOP/s microbench for kMatmulBTQuant — QUANT-GGUF-CIQ-GEMM G6
// evidence tool (NOT a ctest gate). Times the quant GEMM at this model's
// prefill/decode shapes so the portable nrc==1 tier and the Arm i8mm mmla tier
// can be compared apples-to-apples, decoupled from the mixed-file Amdahl
// dilution an end-to-end run mixes in.
//
//   VT_CPU_QUANT_MMLA=0 env  -> portable tier (nrc==1)
//   (unset, i8mm host)       -> i8mm mmla tier (nrc==2) at even M,N
//
// GFLOP/s = 2*M*N*K / best-of-R seconds. Run the two arms as two processes
// (the tier is picked once per process from the env).
//
// ROCm, when the backend is registered, runs the same shapes on-device
// (KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA4, issue #2109): the "prefill" shapes below
// are already tile-aligned (M=128, N and K multiples of 16/256), so a Q6_K
// leg on gfx1200/gfx1201 exercises the new WMMA arm by default. Same
// same-process-per-arm rule applies: `VT_ROCM_QUANT_WMMA=0` forces the scalar
// arm, and the env is read once and cached, so comparing arms means running
// this binary twice, not toggling the variable mid-process.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/quant.h"
#include "vt/tensor.h"

#if defined(VLLM_CPP_HIP)
// Declared here rather than included, mirroring the convention
// tests/vt/test_backend_cross_device.cpp already uses for the same symbol:
// the ROCm kernels have no public header, and this counter exists purely as
// a reachability witness for the kernel launcher.
namespace vt::rocm {
uint64_t KQuantWmmaDispatchCount();
}  // namespace vt::rocm
#endif

namespace {

// RAII device buffer over the PUBLIC vt::Backend allocator — no internal
// header, so this works for any registered backend, not only CPU.
class DeviceBuffer {
 public:
  DeviceBuffer(vt::Backend& backend, size_t bytes) : backend_(backend) {
    ptr_ = backend_.Alloc(bytes);
  }
  ~DeviceBuffer() { backend_.Free(ptr_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void Upload(vt::Queue& q, const void* host, size_t bytes) {
    backend_.Copy(q, ptr_, host, bytes);
  }
  void Download(vt::Queue& q, void* host, size_t bytes) const {
    backend_.Copy(q, host, ptr_, bytes);
  }
  void* ptr() const { return ptr_; }

 private:
  vt::Backend& backend_;
  void* ptr_;
};

}  // namespace

namespace {

struct Shape {
  int64_t m, n, k;
  const char* name;
};

std::vector<uint8_t> RandomBlocks(vt::DType dt, int64_t nblocks, int d_off, int dmin_off,
                                  uint32_t seed) {
  const int64_t be = (dt == vt::DType::kQ4_0 || dt == vt::DType::kQ8_0) ? 32 : 256;
  const size_t block_bytes = vt::RowSizeBytes(dt, be);
  std::mt19937 rng(seed);
  std::vector<uint8_t> bytes(static_cast<size_t>(nblocks) * block_bytes);
  for (uint8_t& b : bytes) b = static_cast<uint8_t>(rng() & 0xFF);
  for (int64_t i = 0; i < nblocks; ++i) {
    uint8_t* blk = bytes.data() + static_cast<size_t>(i) * block_bytes;
    const float jitter = 1.0F + 0.05F * static_cast<float>(i % 7);
    auto put = [&](int off, float v) {
      const uint16_t h = vt::F32ToF16(v);
      std::memcpy(blk + off, &h, sizeof(h));
    };
    if (d_off >= 0) put(d_off, 0.0125F * jitter);
    if (dmin_off >= 0) put(dmin_off, 0.0075F * jitter);
  }
  return bytes;
}

void Bench(vt::DType dt, int d_off, int dmin_off, const char* type_name) {
  const int64_t be = (dt == vt::DType::kQ4_0 || dt == vt::DType::kQ8_0) ? 32 : 256;
  const Shape shapes[] = {
      {128, 3072, 2048, "prefill qkv"},
      {128, 12288, 2048, "prefill gate_up"},
      {128, 2048, 6144, "prefill down"},
      {1, 3072, 2048, "decode qkv"},
  };
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  for (const Shape& s : shapes) {
    if (s.k % be != 0) continue;
    const int64_t nblk = s.n * (s.k / be);
    std::vector<uint8_t> wq = RandomBlocks(dt, nblk, d_off, dmin_off, 0x6A11U);
    std::vector<float> a(static_cast<size_t>(s.m * s.k));
    std::mt19937 rng(1);
    for (float& x : a) x = 0.1F + 0.001F * static_cast<float>(rng() % 2000 - 1000);
    std::vector<float> out(static_cast<size_t>(s.m * s.n));

    vt::Tensor at = vt::Tensor::Contiguous(a.data(), vt::DType::kF32, q.device, {s.m, s.k});
    vt::Tensor bt = vt::Tensor::Contiguous(wq.data(), vt::DType::kF32, q.device, {s.n, s.k});
    bt.dtype = dt;
    vt::Tensor ot = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, q.device, {s.m, s.n});

    auto time_best = [&](vt::Tensor& b) {
      vt::MatmulBTQuant(q, ot, at, b);  // warm
      double best = 1e30;
      const int reps = s.m == 1 ? 20 : 6;
      for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        vt::MatmulBTQuant(q, ot, at, b);
        const auto t1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        if (sec < best) best = sec;
      }
      return 2.0 * static_cast<double>(s.m * s.n * s.k) / best / 1e9;
    };

    const double gflops = time_best(bt);
    std::printf("%-9s %-16s M=%-4lld N=%-6lld K=%-5lld  %8.2f GFLOP/s\n", type_name,
                s.name, (long long)s.m, (long long)s.n, (long long)s.k, gflops);

    // CIQ G7: repacked-weight arm for q8_0 (i8mm interleave). Repack a copy of
    // the weight, mark it, and time the repacked gemm/gemv against the same
    // shapes — the tier-0/mmla vs repacked op-level A/B.
    if (dt == vt::DType::kQ8_0 && vt::cpu::QuantRepackActive()) {
      std::vector<uint8_t> wrp = wq;
      vt::cpu::QuantRepackWeight(dt, wrp.data(), s.n, s.k);
      vt::Tensor brp =
          vt::Tensor::Contiguous(wrp.data(), vt::DType::kF32, q.device, {s.n, s.k});
      brp.dtype = dt;
      brp.repacked = true;
      const double g2 = time_best(brp);
      std::printf("%-9s %-16s M=%-4lld N=%-6lld K=%-5lld  %8.2f GFLOP/s\n",
                  "q8_0-rp", s.name, (long long)s.m, (long long)s.n,
                  (long long)s.k, g2);
    }
  }
}

// Same shapes and methodology as `Bench` above, over a device backend that
// needs its own allocations and an explicit synchronize per timed call (async
// kernel launches would otherwise make the wall-clock window meaningless).
// KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA4 (issue #2109): these shapes are already
// tile-aligned (M=128, N%16==0, K%256==0), so the Q6_K leg on gfx1200/gfx1201
// takes the new WMMA arm by default; `VT_ROCM_QUANT_WMMA=0` forces the scalar
// one for the A/B (see the file header note — two processes, not one).
//
// "prefill gate_up tail" (M=132, N=12288) deliberately un-aligns M by one
// tile (132 = 8*16 + 4) at this row's own widest production N: an
// independent review found the M/N-tail-fill launch sized its host-side
// grid to the FULL m*n domain rather than to the remainder it actually
// computes, so a shape with a large aligned corner and a wide N launches
// and address-computes close to as many warps as the whole GEMM just to
// fill in a thin remainder strip. This shape isolates that launch's own
// cost from the WMMA corner's, at production scale.
void BenchDevice(vt::Backend& backend, vt::Device device, vt::DType dt, int d_off,
                 int dmin_off, const char* type_name) {
  const int64_t be = (dt == vt::DType::kQ4_0 || dt == vt::DType::kQ8_0) ? 32 : 256;
  const Shape shapes[] = {
      {128, 3072, 2048, "prefill qkv"},
      {128, 12288, 2048, "prefill gate_up"},
      {128, 2048, 6144, "prefill down"},
      {132, 12288, 2048, "prefill gate_up tail"},
      {1, 3072, 2048, "decode qkv"},
  };
  vt::Queue q = backend.CreateQueue();
  for (const Shape& s : shapes) {
    if (s.k % be != 0) continue;
    const int64_t nblk = s.n * (s.k / be);
    std::vector<uint8_t> wq = RandomBlocks(dt, nblk, d_off, dmin_off, 0x6A11U);
    std::vector<float> a(static_cast<size_t>(s.m * s.k));
    std::mt19937 rng(1);
    for (float& x : a) x = 0.1F + 0.001F * static_cast<float>(rng() % 2000 - 1000);

    DeviceBuffer da(backend, a.size() * sizeof(float));
    DeviceBuffer dwq(backend, wq.size());
    DeviceBuffer dout(backend, static_cast<size_t>(s.m * s.n) * sizeof(float));
    da.Upload(q, a.data(), a.size() * sizeof(float));
    dwq.Upload(q, wq.data(), wq.size());

    vt::Tensor at = vt::Tensor::Contiguous(da.ptr(), vt::DType::kF32, device, {s.m, s.k});
    vt::Tensor bt = vt::Tensor::Contiguous(dwq.ptr(), vt::DType::kF32, device, {s.n, s.k});
    bt.dtype = dt;
    vt::Tensor ot = vt::Tensor::Contiguous(dout.ptr(), vt::DType::kF32, device, {s.m, s.n});

    vt::MatmulBTQuant(q, ot, at, bt);  // warm
    backend.Synchronize(q);
    double best = 1e30;
    const int reps = s.m == 1 ? 20 : 6;
    for (int r = 0; r < reps; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      vt::MatmulBTQuant(q, ot, at, bt);
      backend.Synchronize(q);
      const auto t1 = std::chrono::steady_clock::now();
      const double sec = std::chrono::duration<double>(t1 - t0).count();
      if (sec < best) best = sec;
    }
    const double gflops = 2.0 * static_cast<double>(s.m * s.n * s.k) / best / 1e9;
    std::printf("%-9s %-16s M=%-4lld N=%-6lld K=%-5lld  %8.2f GFLOP/s\n", type_name,
                s.name, (long long)s.m, (long long)s.n, (long long)s.k, gflops);
  }
  backend.DestroyQueue(q);
}

}  // namespace

int main() {
  std::printf("== kMatmulBTQuant op-level bench — mmla tier %s ==\n",
              vt::cpu::QuantMmlaActive() ? "ON (i8mm)" : "OFF (portable)");
  Bench(vt::DType::kQ8_0, 0, -1, "q8_0");
  Bench(vt::DType::kQ4_K, 0, 2, "q4_K");
  Bench(vt::DType::kQ6_K, 208, -1, "q6_K");

  if (vt::Backend* rocm = vt::TryGetBackend(vt::DeviceType::kROCM)) {
    const vt::Device device{vt::DeviceType::kROCM, 0};
#if defined(VLLM_CPP_HIP)
    const uint64_t wmma_before = vt::rocm::KQuantWmmaDispatchCount();
#endif
    std::printf("== kMatmulBTQuant op-level bench — ROCm ==\n");
    BenchDevice(*rocm, device, vt::DType::kQ8_0, 0, -1, "q8_0");
    BenchDevice(*rocm, device, vt::DType::kQ4_K, 0, 2, "q4_K");
    BenchDevice(*rocm, device, vt::DType::kQ6_K, 208, -1, "q6_K");
#if defined(VLLM_CPP_HIP)
    const uint64_t wmma_after = vt::rocm::KQuantWmmaDispatchCount();
    std::printf("Q6_K WMMA tile arm dispatches this run: %llu\n",
                (unsigned long long)(wmma_after - wmma_before));
#endif
  }
  return 0;
}
