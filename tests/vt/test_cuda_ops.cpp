// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// CUDA op kernels vs the CPU reference ops on random inputs (fixed seeds).
// Guarded like test_cuda_backend.cpp: skips cleanly when no GPU is present.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/op_provider.h"

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

Tensor MakeTensor(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

std::vector<float> RandomF32(size_t n, uint32_t seed, float lo = -2.0f, float hi = 2.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

// Packs an f32 master vector into the byte representation of dt (f32 or bf16).
std::vector<uint8_t> Pack(const std::vector<float>& f, DType dt) {
  std::vector<uint8_t> out(f.size() * vt::SizeOf(dt));
  if (dt == DType::kF32) {
    std::memcpy(out.data(), f.data(), out.size());
  } else {
    REQUIRE(dt == DType::kBF16);
    auto* p = reinterpret_cast<uint16_t*>(out.data());
    for (size_t i = 0; i < f.size(); ++i) p[i] = vt::F32ToBF16(f[i]);
  }
  return out;
}

std::vector<float> Unpack(const std::vector<uint8_t>& b, DType dt) {
  const size_t n = b.size() / vt::SizeOf(dt);
  std::vector<float> out(n);
  if (dt == DType::kF32) {
    std::memcpy(out.data(), b.data(), b.size());
  } else {
    REQUIRE(dt == DType::kBF16);
    const auto* p = reinterpret_cast<const uint16_t*>(b.data());
    for (size_t i = 0; i < n; ++i) out[i] = vt::BF16ToF32(p[i]);
  }
  return out;
}

void CheckClose(const std::vector<float>& got, const std::vector<float>& want, float atol,
                float rtol) {
  REQUIRE(got.size() == want.size());
  size_t bad = 0;
  size_t first_bad = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const float tol = atol + rtol * std::fabs(want[i]);
    if (!(std::fabs(got[i] - want[i]) <= tol)) {  // catches NaN too
      if (bad == 0) first_bad = i;
      ++bad;
    }
  }
  if (bad != 0) {
    CAPTURE(bad);
    CAPTURE(first_bad);
    CAPTURE(got[first_bad]);
    CAPTURE(want[first_bad]);
  }
  CHECK(bad == 0);
}

struct QueueGuard {
  Backend& b;
  Queue q;
  explicit QueueGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

// Device buffer + tensor view; uploads on construction when host data given.
class DeviceTensor {
 public:
  DeviceTensor(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
               const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = MakeTensor(p_, dt, Gpu(), shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
  void Download(Queue& q, void* dst) {
    b_.Copy(q, dst, p_, bytes_);
    b_.Synchronize(q);
  }

 private:
  Backend& b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  Tensor t_;
};

// Input/output dtype combos per the M0.6 plan, with comparison tolerances:
// f32-in/f32-out 1e-5; bf16-in/f32-out 2e-3. bf16 outputs (compared after
// BF16ToF32) get rtol 8e-3 >= one bf16 ulp (2^-7 ≈ 7.8e-3 relative): the GPU tree
// reduction and the CPU sequential sum legitimately differ by ~1e-6 in f32,
// which can flip the final bf16 rounding by one ulp on large rows.
struct Combo {
  DType in;
  DType out;
  float atol;
  float rtol;
};
constexpr Combo kCombos[] = {
    {DType::kF32, DType::kF32, 1e-5f, 1e-5f},
    {DType::kBF16, DType::kF32, 2e-3f, 2e-3f},
    {DType::kBF16, DType::kBF16, 4e-3f, 8e-3f},
};

void RunRmsNormCase(int64_t t, int64_t h, const Combo& c, bool gemma, bool fused,
                    uint32_t seed) {
  const auto xf = RandomF32(static_cast<size_t>(t * h), seed);
  const auto wf = RandomF32(static_cast<size_t>(h), seed + 1);
  const auto resf = RandomF32(static_cast<size_t>(t * h), seed + 2);
  const auto xb = Pack(xf, c.in);
  const auto wb = Pack(wf, c.in);
  const vt::RmsNormArgs args{1e-6f, gemma};

  // CPU reference.
  std::vector<uint8_t> out_cpu(static_cast<size_t>(t * h) * vt::SizeOf(c.out));
  std::vector<float> res_cpu = resf;
  Tensor tx = MakeTensor(const_cast<uint8_t*>(xb.data()), c.in, Cpu(), {t, h});
  Tensor tw = MakeTensor(const_cast<uint8_t*>(wb.data()), c.in, Cpu(), {h});
  Tensor to = MakeTensor(out_cpu.data(), c.out, Cpu(), {t, h});
  Tensor tr = MakeTensor(res_cpu.data(), DType::kF32, Cpu(), {t, h});
  Queue cq{Cpu(), nullptr};
  vt::RmsNorm(cq, to, tx, tw, args, fused ? &tr : nullptr);

  // CUDA.
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceTensor dx(gpu, gq.q, c.in, {t, h}, xb.data());
  DeviceTensor dw(gpu, gq.q, c.in, {h}, wb.data());
  DeviceTensor dout(gpu, gq.q, c.out, {t, h});
  DeviceTensor dres(gpu, gq.q, DType::kF32, {t, h}, resf.data());
  vt::RmsNorm(gq.q, dout.tensor(), dx.tensor(), dw.tensor(), args,
              fused ? &dres.tensor() : nullptr);
  std::vector<uint8_t> out_gpu(out_cpu.size());
  dout.Download(gq.q, out_gpu.data());

  CheckClose(Unpack(out_gpu, c.out), Unpack(out_cpu, c.out), c.atol, c.rtol);
  if (fused) {
    // The residual stream is f32 on both sides; the add order is elementwise
    // and identical, so it matches tightly regardless of x's dtype.
    std::vector<float> res_gpu(static_cast<size_t>(t * h));
    dres.Download(gq.q, res_gpu.data());
    CheckClose(res_gpu, res_cpu, 1e-6f, 1e-6f);
  }
}

}  // namespace

TEST_CASE("CUDA rmsnorm matches CPU across row sizes and dtypes") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  // Odd sizes (1, 127, 129) exercise the shared-memory reduction tail.
  const int64_t sizes[] = {1, 8, 127, 128, 129, 4096};
  uint32_t seed = 100;
  for (int64_t h : sizes) {
    for (const Combo& c : kCombos) {
      CAPTURE(h);
      CAPTURE(static_cast<int>(c.in));
      CAPTURE(static_cast<int>(c.out));
      RunRmsNormCase(3, h, c, /*gemma=*/false, /*fused=*/false, seed);
      seed += 10;
    }
  }
}

TEST_CASE("CUDA rmsnorm gemma variant matches CPU") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  uint32_t seed = 500;
  for (const Combo& c : kCombos) {
    CAPTURE(static_cast<int>(c.in));
    CAPTURE(static_cast<int>(c.out));
    RunRmsNormCase(4, 128, c, /*gemma=*/true, /*fused=*/false, seed);
    seed += 10;
  }
}

TEST_CASE("CUDA rmsnorm fused residual matches CPU and updates residual") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  uint32_t seed = 700;
  for (int64_t h : {127, 128}) {
    for (const Combo& c : kCombos) {
      CAPTURE(h);
      CAPTURE(static_cast<int>(c.in));
      CAPTURE(static_cast<int>(c.out));
      RunRmsNormCase(3, h, c, /*gemma=*/false, /*fused=*/true, seed);
      seed += 10;
    }
  }
}

// VT_RMSNORM_DECODE_FAST: the vectorized decode kernel (RmsNormRowFastKernel) is
// BIT-IDENTICAL to the shipped RmsNormRowKernel — the through-stack 235/235
// bit-reference that matches vLLM's production greedy stream. The 2026-07-17
// bit-safety rework replicates shipped's variance path byte-for-byte (kBlock=256
// strided Pass 1 + shared-memory tree + 1.0f/sqrtf; residual add
// bf16(f32(x)+f32(res))) and only vectorizes Pass 2 (normalize), which is
// element-independent. So at the real 27B decode shape (M x H=5120, bf16, gemma)
// BOTH the residual stream AND the output are BIT-EXACT (0-ulp) vs shipped. This
// bit-identity guarantees fast+cubin ≡ shipped+cubin on the razor near-tie
// (token 6 = 198); token-exactness vs the oracle STREAM is adjudicated on DGX by
// test_qwen27_paged_engine (235/235) + test_qwen36_paged_engine (315/315).
// Runs both arms with EXPLICIT env values ("1" fast / "0" rollback; the launcher
// reads getenv per call) so the comparison is default-independent. Skips w/o CUDA.
void RunRmsNormDecodeFastCase(int64_t t, int64_t h, bool gemma, uint32_t seed) {
  const auto xf = RandomF32(static_cast<size_t>(t * h), seed);
  const auto wf = RandomF32(static_cast<size_t>(h), seed + 1);
  const auto resf = RandomF32(static_cast<size_t>(t * h), seed + 2);
  const auto xb = Pack(xf, DType::kBF16);
  const auto wb = Pack(wf, DType::kBF16);
  const auto rb = Pack(resf, DType::kBF16);  // bf16 residual = the real decode stream
  const vt::RmsNormArgs args{1e-6f, gemma};

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceTensor dx(gpu, gq.q, DType::kBF16, {t, h}, xb.data());
  DeviceTensor dw(gpu, gq.q, DType::kBF16, {h}, wb.data());

  auto run = [&](bool fast, std::vector<uint8_t>& out_bytes, std::vector<uint8_t>& res_bytes) {
    ::setenv("VT_RMSNORM_DECODE_FAST", fast ? "1" : "0", 1);
    DeviceTensor dout(gpu, gq.q, DType::kBF16, {t, h});
    DeviceTensor dres(gpu, gq.q, DType::kBF16, {t, h}, rb.data());
    vt::RmsNorm(gq.q, dout.tensor(), dx.tensor(), dw.tensor(), args, &dres.tensor());
    out_bytes.resize(static_cast<size_t>(t * h) * vt::SizeOf(DType::kBF16));
    res_bytes.resize(out_bytes.size());
    dout.Download(gq.q, out_bytes.data());
    dres.Download(gq.q, res_bytes.data());
  };

  std::vector<uint8_t> out_ref, res_ref, out_fast, res_fast;
  run(/*fast=*/false, out_ref, res_ref);
  run(/*fast=*/true, out_fast, res_fast);
  ::unsetenv("VT_RMSNORM_DECODE_FAST");

  // Output: BIT-EXACT (0-ulp). The 2026-07-17 bit-safety rework makes
  // RmsNormRowFastKernel's variance reduction (kBlock-thread strided Pass 1 +
  // shared-memory tree) and inv (1.0f/sqrtf) byte-for-byte the shipped
  // RmsNormRowKernel's; Pass 2 (normalize) is only vectorized, which is
  // element-independent => identical bits. So fast==shipped exactly, guaranteeing
  // fast+cubin ≡ shipped+cubin on the 27B greedy near-tie (token 6 = 198).
  CheckClose(Unpack(out_fast, DType::kBF16), Unpack(out_ref, DType::kBF16), 0.0f, 0.0f);
  // Residual stream: the add is bf16(f32(x)+f32(res)) == the shipped ResRound on
  // both sides, so the updated residual is bit-identical.
  CheckClose(Unpack(res_fast, DType::kBF16), Unpack(res_ref, DType::kBF16), 0.0f, 0.0f);
}

TEST_CASE("CUDA rmsnorm decode-fast (VT_RMSNORM_DECODE_FAST) matches rollback kernel") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  // The real 27B decode shape is M x 5120 (all 129 input/post-attn/final norms);
  // gemma=true is the shipped RmsNormArgs for those launches. Sweep c1-c32.
  uint32_t seed = 900;
  for (int64_t m : {1, 2, 4, 8, 16, 32}) {
    for (bool gemma : {true, false}) {
      CAPTURE(m);
      CAPTURE(gemma);
      RunRmsNormDecodeFastCase(m, 5120, gemma, seed);
      seed += 10;
    }
  }
}

TEST_CASE("CUDA silu_and_mul matches CPU") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  uint32_t seed = 900;
  for (int64_t d : {173, 256}) {
    for (const Combo& c : kCombos) {
      CAPTURE(d);
      CAPTURE(static_cast<int>(c.in));
      CAPTURE(static_cast<int>(c.out));
      const int64_t t = 5;
      const auto xf = RandomF32(static_cast<size_t>(t * 2 * d), seed);
      const auto xb = Pack(xf, c.in);

      std::vector<uint8_t> out_cpu(static_cast<size_t>(t * d) * vt::SizeOf(c.out));
      Tensor tx = MakeTensor(const_cast<uint8_t*>(xb.data()), c.in, Cpu(), {t, 2 * d});
      Tensor to = MakeTensor(out_cpu.data(), c.out, Cpu(), {t, d});
      Queue cq{Cpu(), nullptr};
      vt::SiluAndMul(cq, to, tx);

      QueueGuard gq(gpu);
      DeviceTensor dx(gpu, gq.q, c.in, {t, 2 * d}, xb.data());
      DeviceTensor dout(gpu, gq.q, c.out, {t, d});
      vt::SiluAndMul(gq.q, dout.tensor(), dx.tensor());
      std::vector<uint8_t> out_gpu(out_cpu.size());
      dout.Download(gq.q, out_gpu.data());

      CheckClose(Unpack(out_gpu, c.out), Unpack(out_cpu, c.out), c.atol, c.rtol);
      seed += 10;
    }
  }
}

TEST_CASE("CUDA embedding matches CPU (i32 and i64 ids)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  const int64_t v = 64, h = 33, t = 17;
  uint32_t seed = 1200;
  for (DType id_dt : {DType::kI32, DType::kI64}) {
    for (const Combo& c : kCombos) {
      CAPTURE(static_cast<int>(id_dt));
      CAPTURE(static_cast<int>(c.in));
      CAPTURE(static_cast<int>(c.out));
      const auto tf = RandomF32(static_cast<size_t>(v * h), seed);
      const auto tb = Pack(tf, c.in);
      std::mt19937 rng(seed + 1);
      std::uniform_int_distribution<int64_t> dist(0, v - 1);
      std::vector<int32_t> ids32(static_cast<size_t>(t));
      std::vector<int64_t> ids64(static_cast<size_t>(t));
      for (int64_t i = 0; i < t; ++i) {
        ids64[static_cast<size_t>(i)] = dist(rng);
        ids32[static_cast<size_t>(i)] = static_cast<int32_t>(ids64[static_cast<size_t>(i)]);
      }
      const void* ids_host = id_dt == DType::kI32 ? static_cast<const void*>(ids32.data())
                                                  : static_cast<const void*>(ids64.data());

      std::vector<uint8_t> out_cpu(static_cast<size_t>(t * h) * vt::SizeOf(c.out));
      Tensor ttab = MakeTensor(const_cast<uint8_t*>(tb.data()), c.in, Cpu(), {v, h});
      Tensor tids = MakeTensor(const_cast<void*>(ids_host), id_dt, Cpu(), {t});
      Tensor to = MakeTensor(out_cpu.data(), c.out, Cpu(), {t, h});
      Queue cq{Cpu(), nullptr};
      vt::Embedding(cq, to, ttab, tids);

      QueueGuard gq(gpu);
      DeviceTensor dtab(gpu, gq.q, c.in, {v, h}, tb.data());
      DeviceTensor dids(gpu, gq.q, id_dt, {t}, ids_host);
      DeviceTensor dout(gpu, gq.q, c.out, {t, h});
      vt::Embedding(gq.q, dout.tensor(), dtab.tensor(), dids.tensor());
      std::vector<uint8_t> out_gpu(out_cpu.size());
      dout.Download(gq.q, out_gpu.data());

      CheckClose(Unpack(out_gpu, c.out), Unpack(out_cpu, c.out), c.atol, c.rtol);
      seed += 10;
    }
  }
}

TEST_CASE("CUDA embedding: out-of-range device id is reported with the id") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  const int64_t v = 8, h = 4;
  const auto tf = RandomF32(static_cast<size_t>(v * h), 42);

  // CONTRACT (changed deliberately, see cuda_ops.cu EmbeddingErrRing): the CUDA
  // embedding no longer synchronizes the stream to read its out-of-range flag,
  // because that made a once-per-step op a hard barrier between engine steps.
  // The report is now DEFERRED: an out-of-range id is raised no later than the
  // NEXT Embedding on the queue, carrying the same message and the same id. The
  // gather itself is unchanged — bad ids are still clamped in-kernel, so the
  // offending call never reads out of bounds.
  //
  // The Synchronize below is what makes this deterministic rather than a race
  // with the flag's device-to-host copy: after it the copy has certainly landed,
  // so the next Embedding MUST raise.
  for (int32_t bad : {int32_t{8}, int32_t{-3}}) {
    CAPTURE(bad);
    std::vector<int32_t> bad_ids = {1, bad, 2};
    const std::vector<int32_t> good_ids = {1, 0, 2};
    QueueGuard gq(gpu);
    DeviceTensor dtab(gpu, gq.q, DType::kF32, {v, h}, tf.data());
    DeviceTensor dbad(gpu, gq.q, DType::kI32, {3}, bad_ids.data());
    DeviceTensor dgood(gpu, gq.q, DType::kI32, {3}, good_ids.data());
    DeviceTensor dout(gpu, gq.q, DType::kF32, {3, h});

    // The offending call is allowed to raise here or to defer; either is the
    // contract, so only the id is asserted when it does raise.
    bool threw = false;
    std::string msg;
    try {
      vt::Embedding(gq.q, dout.tensor(), dtab.tensor(), dbad.tensor());
    } catch (const std::runtime_error& e) {
      threw = true;
      msg = e.what();
    }
    // The kernel clamps bad ids, so the stream stays healthy either way.
    CHECK_NOTHROW(gpu.Synchronize(gq.q));

    if (!threw) {
      try {
        vt::Embedding(gq.q, dout.tensor(), dtab.tensor(), dgood.tensor());
      } catch (const std::runtime_error& e) {
        threw = true;
        msg = e.what();
      }
    }
    CAPTURE(msg);
    CHECK(threw);
    CHECK(msg.find("embedding") != std::string::npos);
    CHECK(msg.find(std::to_string(bad)) != std::string::npos);
    // Once reported, the error is CONSUMED: a subsequent clean call is clean,
    // so a single bad id cannot poison every later step.
    CHECK_NOTHROW(vt::Embedding(gq.q, dout.tensor(), dtab.tensor(), dgood.tensor()));
    CHECK_NOTHROW(gpu.Synchronize(gq.q));
  }
}

TEST_CASE("CUDA matmul (cuBLASLt) matches CPU on odd sizes") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  // Matmul-specific tolerances: cuBLASLt may reduce over K in a different
  // order than the CPU triple loop (split-K, tensor-core tiles), so even the
  // all-f32 combo gets 1e-4 instead of the elementwise 1e-5. bf16 inputs are
  // identical bytes on both sides (products are exact in f32), so bf16-in
  // stays at 2e-3 and bf16-out at one output ulp, as in kCombos.
  const Combo combos[] = {
      {DType::kF32, DType::kF32, 1e-4f, 1e-4f},
      {DType::kBF16, DType::kF32, 2e-3f, 2e-3f},
      {DType::kBF16, DType::kBF16, 4e-3f, 8e-3f},
  };
  struct Dims {
    int64_t m, k, n;
  };
  // Odd shapes exercise tile tails; {1,257,1} is a pure K-reduction.
  const Dims dims[] = {{17, 31, 13}, {64, 128, 32}, {1, 257, 1}};
  uint32_t seed = 3000;
  for (const Dims& d : dims) {
    for (const Combo& c : combos) {
      CAPTURE(d.m);
      CAPTURE(d.k);
      CAPTURE(d.n);
      CAPTURE(static_cast<int>(c.in));
      CAPTURE(static_cast<int>(c.out));
      const auto af = RandomF32(static_cast<size_t>(d.m * d.k), seed);
      const auto bf = RandomF32(static_cast<size_t>(d.k * d.n), seed + 1);
      const auto ab = Pack(af, c.in);
      const auto bb = Pack(bf, c.in);

      // CPU reference on the same packed inputs.
      std::vector<uint8_t> out_cpu(static_cast<size_t>(d.m * d.n) * vt::SizeOf(c.out));
      Tensor ta = MakeTensor(const_cast<uint8_t*>(ab.data()), c.in, Cpu(), {d.m, d.k});
      Tensor tb = MakeTensor(const_cast<uint8_t*>(bb.data()), c.in, Cpu(), {d.k, d.n});
      Tensor to = MakeTensor(out_cpu.data(), c.out, Cpu(), {d.m, d.n});
      Queue cq{Cpu(), nullptr};
      vt::Matmul(cq, to, ta, tb);

      // CUDA.
      QueueGuard gq(gpu);
      DeviceTensor da(gpu, gq.q, c.in, {d.m, d.k}, ab.data());
      DeviceTensor db(gpu, gq.q, c.in, {d.k, d.n}, bb.data());
      DeviceTensor dout(gpu, gq.q, c.out, {d.m, d.n});
      vt::Matmul(gq.q, dout.tensor(), da.tensor(), db.tensor());
      std::vector<uint8_t> out_gpu(out_cpu.size());
      dout.Download(gq.q, out_gpu.data());

      CheckClose(Unpack(out_gpu, c.out), Unpack(out_cpu, c.out), c.atol, c.rtol);
      seed += 10;
    }
  }
}

TEST_CASE("CUDA matmul_bt (cuBLASLt TN) matches CPU on odd sizes") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  // Same tolerance rationale as the CUDA matmul case above: the TN cuBLASLt
  // algo reduces over K in its own order vs the CPU triple loop.
  const Combo combos[] = {
      {DType::kF32, DType::kF32, 1e-4f, 1e-4f},
      {DType::kBF16, DType::kF32, 2e-3f, 2e-3f},
      {DType::kBF16, DType::kBF16, 4e-3f, 8e-3f},
  };
  struct Dims {
    int64_t m, k, n;
  };
  const Dims dims[] = {{17, 32, 13}, {64, 128, 48}, {1, 256, 1}};
  uint32_t seed = 7000;
  for (const Dims& d : dims) {
    for (const Combo& c : combos) {
      CAPTURE(d.m);
      CAPTURE(d.k);
      CAPTURE(d.n);
      CAPTURE(static_cast<int>(c.in));
      CAPTURE(static_cast<int>(c.out));
      const auto af = RandomF32(static_cast<size_t>(d.m * d.k), seed);
      const auto bf = RandomF32(static_cast<size_t>(d.n * d.k), seed + 1);  // b [N,K]
      const auto ab = Pack(af, c.in);
      const auto bb = Pack(bf, c.in);

      // CPU reference on the same packed inputs (b in [N,K] orientation).
      std::vector<uint8_t> out_cpu(static_cast<size_t>(d.m * d.n) * vt::SizeOf(c.out));
      Tensor ta = MakeTensor(const_cast<uint8_t*>(ab.data()), c.in, Cpu(), {d.m, d.k});
      Tensor tb = MakeTensor(const_cast<uint8_t*>(bb.data()), c.in, Cpu(), {d.n, d.k});
      Tensor to = MakeTensor(out_cpu.data(), c.out, Cpu(), {d.m, d.n});
      Queue cq{Cpu(), nullptr};
      vt::MatmulBT(cq, to, ta, tb);

      // CUDA.
      QueueGuard gq(gpu);
      DeviceTensor da(gpu, gq.q, c.in, {d.m, d.k}, ab.data());
      DeviceTensor db(gpu, gq.q, c.in, {d.n, d.k}, bb.data());
      DeviceTensor dout(gpu, gq.q, c.out, {d.m, d.n});
      vt::MatmulBT(gq.q, dout.tensor(), da.tensor(), db.tensor());
      std::vector<uint8_t> out_gpu(out_cpu.size());
      dout.Download(gq.q, out_gpu.data());

      CheckClose(Unpack(out_gpu, c.out), Unpack(out_cpu, c.out), c.atol, c.rtol);
      seed += 10;
    }
  }
}

TEST_CASE("CUDA matmul: unsupported dtype combo (f16 inputs) throws naming it") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  // The op-level validation admits f16 inputs; the cuBLASLt kernel does not
  // implement them and must throw before touching the data (left unset).
  DeviceTensor da(gpu, gq.q, DType::kF16, {4, 8});
  DeviceTensor db(gpu, gq.q, DType::kF16, {8, 3});
  DeviceTensor dout(gpu, gq.q, DType::kF32, {4, 3});
  bool threw = false;
  try {
    vt::Matmul(gq.q, dout.tensor(), da.tensor(), db.tensor());
  } catch (const std::runtime_error& e) {
    threw = true;
    const std::string msg = e.what();
    CAPTURE(msg);
    CHECK(msg.find("matmul") != std::string::npos);
    CHECK(msg.find("f16") != std::string::npos);
  }
  CHECK(threw);
  CHECK_NOTHROW(gpu.Synchronize(gq.q));
}

TEST_CASE("CUDA rope_neox matches CPU (partial rotary, i32/i64 positions)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  const int64_t t = 7, hq = 4, hk = 2, d = 32;
  uint32_t seed = 2000;
  for (int rotary : {32, 16}) {
    for (DType dt : {DType::kF32, DType::kBF16}) {
      for (DType pos_dt : {DType::kI64, DType::kI32}) {
        CAPTURE(rotary);
        CAPTURE(static_cast<int>(dt));
        CAPTURE(static_cast<int>(pos_dt));
        // bf16 rtol allows one bf16 ulp, as in kCombos.
        const float atol = dt == DType::kF32 ? 1e-5f : 4e-3f;
        const float rtol = dt == DType::kF32 ? 1e-5f : 8e-3f;
        const auto qf = RandomF32(static_cast<size_t>(t * hq * d), seed);
        const auto kf = RandomF32(static_cast<size_t>(t * hk * d), seed + 1);
        const auto qb = Pack(qf, dt);
        const auto kb = Pack(kf, dt);
        std::mt19937 rng(seed + 2);
        std::uniform_int_distribution<int64_t> dist(0, 131072);
        std::vector<int64_t> pos64(static_cast<size_t>(t));
        std::vector<int32_t> pos32(static_cast<size_t>(t));
        for (int64_t i = 0; i < t; ++i) {
          pos64[static_cast<size_t>(i)] = dist(rng);
          pos32[static_cast<size_t>(i)] = static_cast<int32_t>(pos64[static_cast<size_t>(i)]);
        }
        const void* pos_host = pos_dt == DType::kI32 ? static_cast<const void*>(pos32.data())
                                                     : static_cast<const void*>(pos64.data());
        const vt::RopeArgs args{10000.0f, rotary};

        // CPU reference (in place on copies).
        std::vector<uint8_t> q_cpu = qb, k_cpu = kb;
        Tensor tq = MakeTensor(q_cpu.data(), dt, Cpu(), {t, hq, d});
        Tensor tk = MakeTensor(k_cpu.data(), dt, Cpu(), {t, hk, d});
        Tensor tp = MakeTensor(const_cast<void*>(pos_host), pos_dt, Cpu(), {t});
        Queue cq{Cpu(), nullptr};
        vt::RopeNeox(cq, tq, tk, tp, args);

        // CUDA (in place on device copies of the same packed inputs).
        QueueGuard gq(gpu);
        DeviceTensor dq(gpu, gq.q, dt, {t, hq, d}, qb.data());
        DeviceTensor dk(gpu, gq.q, dt, {t, hk, d}, kb.data());
        DeviceTensor dp(gpu, gq.q, pos_dt, {t}, pos_host);
        vt::RopeNeox(gq.q, dq.tensor(), dk.tensor(), dp.tensor(), args);
        std::vector<uint8_t> q_gpu(qb.size()), k_gpu(kb.size());
        dq.Download(gq.q, q_gpu.data());
        dk.Download(gq.q, k_gpu.data());

        CheckClose(Unpack(q_gpu, dt), Unpack(q_cpu, dt), atol, rtol);
        CheckClose(Unpack(k_gpu, dt), Unpack(k_cpu, dt), atol, rtol);
        seed += 10;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// greedy_rejection_sample (SPEC-REJECTION I3): CUDA == the CPU reference,
// BIT-EXACT on the accepted token ids, at the GATE MODELS' real vocab (248320 —
// both Qwen3.6 gate checkpoints, spec §1). Mirrors the greedy branch of
// vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:564-585,628 @ e24d1b24.
TEST_CASE("CUDA greedy_rejection_sample matches CPU bit-exactly at gate vocab (248320)") {
  if (!HasCuda()) return;
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);

  constexpr int64_t kVocab = 248320;  // the gate checkpoints' vocab_size
  // Four requests with DIFFERENT k_i: 1, 3, 0 (no drafts), 2.
  const std::vector<int32_t> ks = {1, 3, 0, 2};
  const int64_t num_reqs = static_cast<int64_t>(ks.size());
  std::vector<int32_t> cu_num_logits(static_cast<size_t>(num_reqs) + 1, 0);
  for (size_t r = 0; r < ks.size(); ++r) {
    cu_num_logits[r + 1] = cu_num_logits[r] + ks[r] + 1;
  }
  const int64_t num_logits = cu_num_logits.back();
  const int64_t width = 4;  // >= max k + 1

  std::vector<float> logits = RandomF32(static_cast<size_t>(num_logits * kVocab), 4242);
  // Make each row's argmax a known, well-separated token id.
  std::vector<int32_t> row_argmax(static_cast<size_t>(num_logits));
  for (int64_t row = 0; row < num_logits; ++row) {
    const int32_t tok = static_cast<int32_t>((row * 7919 + 13) % kVocab);
    row_argmax[static_cast<size_t>(row)] = tok;
    logits[static_cast<size_t>(row * kVocab + tok)] = 100.0f;
  }
  // draft_sampled: accept the first draft of each request, reject the second.
  std::vector<int32_t> draft_sampled(static_cast<size_t>(num_logits), 0);
  for (size_t r = 0; r < ks.size(); ++r) {
    const int32_t start = cu_num_logits[r];
    for (int32_t i = 0; i < ks[r]; ++i) {
      const int32_t target = row_argmax[static_cast<size_t>(start + i)];
      draft_sampled[static_cast<size_t>(start + i + 1)] =
          (i == 0) ? target : static_cast<int32_t>((target + 1) % kVocab);
    }
  }

  // CPU reference.
  Queue cq{Cpu(), nullptr};
  std::vector<int32_t> cpu_sampled(static_cast<size_t>(num_reqs * width), 0);
  std::vector<int32_t> cpu_num_sampled(static_cast<size_t>(num_reqs), 0);
  {
    Tensor tl = MakeTensor(logits.data(), DType::kF32, Cpu(), {num_logits, kVocab});
    Tensor td = MakeTensor(draft_sampled.data(), DType::kI32, Cpu(), {num_logits});
    Tensor tc = MakeTensor(cu_num_logits.data(), DType::kI32, Cpu(), {num_reqs + 1});
    Tensor ts = MakeTensor(cpu_sampled.data(), DType::kI32, Cpu(), {num_reqs, width});
    Tensor tn = MakeTensor(cpu_num_sampled.data(), DType::kI32, Cpu(), {num_reqs});
    // The per-row argmax scratch is the CALLER's buffer (SPEC-DFLASH2 A2-2,
    // #2802): the op returns with both CUDA kernels still queued, so nothing
    // inside the backend may own the buffer between them.
    std::vector<int32_t> cpu_argmax(static_cast<size_t>(num_logits), 0);
    Tensor ta = MakeTensor(cpu_argmax.data(), DType::kI32, Cpu(), {num_logits});
    vt::GreedyRejectionSample(cq, ts, tn, ta, tl, td, tc);
  }

  // CUDA.
  QueueGuard gq(gpu);
  DeviceTensor dl(gpu, gq.q, DType::kF32, {num_logits, kVocab}, logits.data());
  DeviceTensor dd(gpu, gq.q, DType::kI32, {num_logits}, draft_sampled.data());
  DeviceTensor dc(gpu, gq.q, DType::kI32, {num_reqs + 1}, cu_num_logits.data());
  DeviceTensor ds(gpu, gq.q, DType::kI32, {num_reqs, width});
  DeviceTensor dn(gpu, gq.q, DType::kI32, {num_reqs});
  DeviceTensor da(gpu, gq.q, DType::kI32, {num_logits});
  vt::GreedyRejectionSample(gq.q, ds.tensor(), dn.tensor(), da.tensor(), dl.tensor(),
                            dd.tensor(), dc.tensor());
  std::vector<int32_t> gpu_sampled(static_cast<size_t>(num_reqs * width));
  std::vector<int32_t> gpu_num_sampled(static_cast<size_t>(num_reqs));
  ds.Download(gq.q, gpu_sampled.data());
  dn.Download(gq.q, gpu_num_sampled.data());

  // BIT-EXACT on both outputs.
  CHECK(gpu_num_sampled == cpu_num_sampled);
  CHECK(gpu_sampled == cpu_sampled);

  // And the accept rule itself: first draft accepted, second rejected.
  //   k=1 -> all accepted -> num_sampled = 2 (draft + bonus)
  //   k=3 -> 1 accepted   -> num_sampled = 2 (draft + replacement)
  //   k=0 -> num_sampled = 1 (the plain greedy argmax — the non-spec reduction)
  //   k=2 -> 1 accepted   -> num_sampled = 2
  CHECK(cpu_num_sampled == std::vector<int32_t>{2, 2, 1, 2});
  // The k=0 row emits exactly the target argmax of its single logit row.
  CHECK(cpu_sampled[2 * static_cast<size_t>(width)] ==
        row_argmax[static_cast<size_t>(cu_num_logits[2])]);
}

// ── PermuteVHeads (T25) ──────────────────────────────────────────────────────
//
// The op had CPU and ROCm kernels and NO test anywhere, and the GPU arm the GDN
// column-permuted keep-quant path needs (`out_proj_tiled`, `qwen3_5.cpp`
// `GdnOutProjMatmul`) was missing on CUDA. This is the op's first coverage on any
// backend. The expected permutation is written out HERE from the specification
// rather than taken from either kernel, so the CPU arm is checked against the
// mapping and not against itself:
//
//   out[row, (r*num_k + k)*dv + h] = in[row, (k*rpk + r)*dv + h]
//
// The comparison is bit-exact on both arms: a gather re-indexes values, so an
// equality test has no reduction-order excuse and a tolerance would hide exactly
// the defect (a wrong index). On the pre-change tree the CUDA half refuses by
// name — `vt::GetOp(kPermuteVHeads, kCUDA)` throws for an unregistered op — so
// this case is RED there and GREEN after the kernel lands.
namespace {

// First index where `got` differs from `want`, or -1. Reported on failure so the
// message names the ELEMENT a wrong index mapping displaced, not just that the
// two vectors differ.
int64_t FirstMismatch(const std::vector<uint16_t>& got,
                      const std::vector<uint16_t>& want) {
  for (size_t i = 0; i < got.size() && i < want.size(); ++i) {
    if (got[i] != want[i]) return static_cast<int64_t>(i);
  }
  return got.size() == want.size() ? -1 : static_cast<int64_t>(got.size());
}

void CheckSameAs(const std::vector<uint16_t>& got,
                 const std::vector<uint16_t>& want, const char* arm) {
  const int64_t bad = FirstMismatch(got, want);
  CAPTURE(arm);
  CAPTURE(bad);
  CHECK(bad == -1);
}

}  // namespace

TEST_CASE("PermuteVHeads: CUDA reproduces the CPU gather bit-exactly") {
  const int64_t T = 3;
  const int64_t num_k = 2;
  const int64_t rpk = 3;
  const int64_t dv = 4;
  const int64_t value_dim = num_k * rpk * dv;  // 24
  const int64_t n = T * value_dim;

  // Distinct bf16-exact values, so a wrong index is visible in the VALUE and not
  // only in a bit count.
  std::vector<uint16_t> in(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    in[static_cast<size_t>(i)] = vt::F32ToBF16(static_cast<float>(i) + 0.5F);
  }
  std::vector<uint16_t> want(static_cast<size_t>(n), 0);
  for (int64_t row = 0; row < T; ++row) {
    for (int64_t t = 0; t < num_k * rpk; ++t) {
      const int64_t r = t / num_k;
      const int64_t k = t % num_k;
      const int64_t g = k * rpk + r;
      for (int64_t h = 0; h < dv; ++h) {
        want[static_cast<size_t>(row * value_dim + t * dv + h)] =
            in[static_cast<size_t>(row * value_dim + g * dv + h)];
      }
    }
  }

  // CPU.
  std::vector<uint16_t> cpu_out(static_cast<size_t>(n), 0);
  Tensor tin = MakeTensor(const_cast<uint16_t*>(in.data()), DType::kBF16, Cpu(),
                          {T, value_dim});
  Tensor tout = MakeTensor(cpu_out.data(), DType::kBF16, Cpu(), {T, value_dim});
  Queue cq{Cpu(), nullptr};
  vt::PermuteVHeads(cq, tout, tin, T, num_k, rpk, dv);
  CheckSameAs(cpu_out, want, "cpu");

  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; the CPU arm above is the whole gate");
    return;
  }

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);

  // "THE CUDA ARM RAN" IS NOT IMPLIED BY THE VALUES. A missing CUDA kernel falls
  // through to the lazy CPU reference tier on a host-addressable device and
  // produces the SAME rows — measured: with this op's `RegisterOp` deleted, the
  // comparison below still passes. These probes are what make the arm observable,
  // and they are the reason a reviewer's deletion mutation bites.
  CHECK(vt::OpRegistered(vt::OpId::kPermuteVHeads, vt::DeviceType::kCUDA));
  vt::EnableOpProviderCallStats(true);
  const unsigned long long ref_before = vt::GetReferenceTierHits();

  DeviceTensor din(gpu, gq.q, DType::kBF16, {T, value_dim}, in.data());
  DeviceTensor dout(gpu, gq.q, DType::kBF16, {T, value_dim});
  vt::PermuteVHeads(gq.q, dout.tensor(), din.tensor(), T, num_k, rpk, dv);
  std::vector<uint16_t> gpu_out(static_cast<size_t>(n), 0);
  dout.Download(gq.q, gpu_out.data());

  const vt::OpProviderStats stats =
      vt::GetOpProviderStats(vt::OpId::kPermuteVHeads, vt::DeviceType::kCUDA);
  CHECK(stats.last_selected != nullptr);
  const bool selected_native =
      stats.last_selected != nullptr &&
      std::strcmp(stats.last_selected, vt::kReferenceProviderName) != 0;
  CHECK(selected_native);
  CHECK(vt::GetReferenceTierHits() == ref_before);

  CheckSameAs(gpu_out, want, "cuda");
}
