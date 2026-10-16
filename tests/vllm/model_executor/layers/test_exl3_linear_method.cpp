// EXL3 on the shared linear seam — QUANT-EXL3 W1a (#2181).
//
// Two questions, the same two `test_linear_method.cpp` asks of NVFP4 and FP8:
// does the factory pick the scheme ONCE from the checkpoint's populated weights
// (`get_quant_method`), and does the bound method compute the right thing.
//
// The reference is `vt::Exl3DequantLinear` — the W1a CPU dequant that
// `MODEL-DSV4-EXL3` gates against transcribed constants — followed by a plain
// f32 matmul. That is the OTHER side of upstream's own identity: the runtime
// form transforms ACTIVATIONS (`exl3.py:183-214`) and the reconstruction form
// transforms WEIGHTS (`:227-237`), and they are equal only up to summation
// order. So this is a bounded gate, and the bound is the one
// `tests/vt/test_exl3_gemm.cpp` already states for exactly this comparison
// (2.0e-3 relative RMS), not a number discovered when the gate first ran.
//
// CPU reference cases; the explicit FP16 gate/up policy runs on XPU, where
// the public SwiGLU operator supports FP16 output.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vllm/model_executor/layers/quantization/exl3.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/op_provider.h"

// The shared EXL3 fixture, at tests/vt/exl3_fixture.h. The three other users
// live in tests/vt/ and write the plain `"exl3_fixture.h"`, which resolves
// relative to their own directory and CANNOT resolve from here; this suite
// needs the `vt/` prefix against the `tests/` include root
// (tests/CMakeLists.txt:23). That root is searched BEFORE `include/`, and no
// `include/vt/exl3_fixture.h` exists, so the prefix is unambiguous today —
// adding one would silently switch this file's fixture, which is the shadowing
// surface tests/CMakeLists.txt:20-22 warns about and the reason this is spelled
// out rather than left to look like a typo.
#include "vt/exl3_fixture.h"

namespace {

using exl3_test::Exl3Fixture;
using exl3_test::MakeFixture;
using exl3_test::Rng;
using exl3_test::UlpF16;
using vllm::OwnedTensor;
using vt::DType;
namespace layers = vllm::layers;

// The fixture's three arrays, wrapped as the OwnedTensors a loader would fill.
vllm::Exl3Weight WrapFixture(const Exl3Fixture& f) {
  vllm::Exl3Weight w;
  // EXPLICIT: the struct no longer defaults, because an implicit codebook is
  // what shipped a wrong decode. These fixtures are random bytes, so any
  // codebook is self-consistent; cb 1 is what the synthetic suites have always
  // used and `test_exl3_real_decode` is what gates the arithmetic.
  w.codebook = 1;
  const auto bytes_of = [](const std::vector<uint16_t>& v) {
    return vllm::OwnedBytes(std::vector<uint8_t>(
        reinterpret_cast<const uint8_t*>(v.data()),
        reinterpret_cast<const uint8_t*>(v.data()) + v.size() * 2));
  };
  // Byte width, which is the shape vt::Exl3Gemm reads; the checkpoint's own
  // I16 [k/16, n/16, 16*bits] is the same bytes.
  w.trellis.dtype = DType::kI8;
  w.trellis.rank = 3;
  w.trellis.shape[0] = f.k / 16;
  w.trellis.shape[1] = f.n / 16;
  w.trellis.shape[2] = 32 * f.bits;
  w.trellis.bytes = bytes_of(f.trellis);

  w.suh.dtype = DType::kF16;
  w.suh.rank = 1;
  w.suh.shape[0] = f.k;
  w.suh.bytes = bytes_of(f.suh);

  w.svh.dtype = DType::kF16;
  w.svh.rank = 1;
  w.svh.shape[0] = f.n;
  w.svh.bytes = bytes_of(f.svh);
  return w;
}

// y = x @ Exl3DequantLinear(trellis, suh, /*codebook=*/1, svh), the weight-side form.
std::vector<float> ReferenceApply(const Exl3Fixture& f, const std::vector<float>& x, int64_t m) {
  std::vector<float> w(static_cast<size_t>(f.k * f.n), 0.0f);
  vt::Exl3DequantLinear(f.trellis.data(), f.suh.data(), f.svh.data(), f.k, f.n, f.bits, /*codebook=*/1, w.data());
  std::vector<float> y(static_cast<size_t>(m * f.n), 0.0f);
  for (int64_t i = 0; i < m; ++i)
    for (int64_t kk = 0; kk < f.k; ++kk) {
      const float xv = x[static_cast<size_t>(i * f.k + kk)];
      if (xv == 0.0f) continue;
      for (int64_t j = 0; j < f.n; ++j)
        y[static_cast<size_t>(i * f.n + j)] += xv * w[static_cast<size_t>(kk * f.n + j)];
    }
  return y;
}

double RelRms(const std::vector<float>& got, const std::vector<float>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return den == 0.0 ? 0.0 : std::sqrt(num / den);
}

vt::Queue CpuQueue() { return vt::GetBackend(vt::DeviceType::kCPU).CreateQueue(); }

}  // namespace

TEST_CASE("exl3 linear method: the factory selects the scheme ONCE from the weights") {
  const Exl3Fixture f = MakeFixture(128, 128, 3, 0xA5A5u);
  const vllm::Exl3Weight w = WrapFixture(f);
  OwnedTensor bf16;  // EMPTY, as an EXL3 checkpoint leaves it

  auto quantized = layers::MakeLinearMethod(bf16, w);
  CHECK(std::string(quantized->Name()) == "exl3-trellis");

  // The other direction: a bf16 checkpoint has no EXL3 weight, and must NOT get
  // the trellis method. Without this case the factory could return the EXL3 arm
  // unconditionally and the case above would still pass.
  vllm::Exl3Weight none;
  CHECK(none.Empty());
  OwnedTensor dense;
  dense.dtype = DType::kBF16;
  dense.rank = 2;
  dense.shape[0] = 4;
  dense.shape[1] = 8;
  dense.bytes = vllm::OwnedBytes(std::vector<uint8_t>(4 * 8 * 2, 0));
  auto unquantized = layers::MakeLinearMethod(dense, none);
  CHECK(std::string(unquantized->Name()) == "bf16-unquantized");
}

TEST_CASE("exl3 linear method: bits come from the TENSOR, never from a config scalar") {
  // The stock layout this row targets carries a 3-bit body and a SIX-bit
  // lm_head under a `quantization_config.bits` of 3.0, measured on
  // `turboderp/Llama-3.2-1B-Instruct-exl3` @ 3.0bpw. Both widths resolve from
  // the same field here, and nothing consults a config.
  const Exl3Fixture three = MakeFixture(128, 128, 3, 0x3333u);
  const Exl3Fixture six = MakeFixture(128, 128, 6, 0x6666u);
  CHECK(WrapFixture(three).Bits() == 3);
  CHECK(WrapFixture(six).Bits() == 6);

  // And the width is LOAD-BEARING rather than decorative: decoding the 6-bit
  // tensor at 3 bits is not a small error, it is a different weight. This is
  // the mutation the gate exists for, spelled as an assertion so it cannot be
  // silently lost.
  std::vector<float> w6(128 * 128, 0.0f), w3(128 * 128, 0.0f);
  vt::Exl3DequantLinear(six.trellis.data(), six.suh.data(), six.svh.data(), 128, 128, 6, /*codebook=*/1, w6.data());
  vt::Exl3DequantLinear(six.trellis.data(), six.suh.data(), six.svh.data(), 128, 128, 3, /*codebook=*/1, w3.data());
  CHECK(RelRms(w3, w6) > 0.5);

  // A trellis whose last dim is not a multiple of 32 BYTES (16 i16 words on
  // disk) is not a width this format can express, and is refused rather than
  // rounded.
  vllm::Exl3Weight bad = WrapFixture(three);
  bad.trellis.shape[2] = 47;
  CHECK_THROWS(bad.Bits());
}

TEST_CASE("exl3 linear method: Apply agrees with the weight-side dequant within the bound") {
  vt::Queue q = CpuQueue();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};

  const int64_t m = 3, k = 256, n = 256;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0x51ED270Bu);
  const vllm::Exl3Weight w = WrapFixture(f);
  OwnedTensor bf16;

  Rng rng;
  rng.s = 0xB5297A4Du;
  std::vector<float> x(static_cast<size_t>(m * k));
  // Through fp16 first: the method stages the activation to fp16, so an f32
  // input that does not survive that round would charge the rounding to the
  // kernel. The reference then reads the SAME values.
  for (auto& v : x) v = vt::F16ToF32(vt::F32ToF16(rng.next(1.0f)));

  auto method = layers::MakeLinearMethod(bf16, w);
  REQUIRE(std::string(method->Name()) == "exl3-trellis");

  vllm::dense_attn::DBuf xb(d, DType::kF32, {m, k}, x.data());
  vllm::dense_attn::DBuf out = method->Apply(d, xb.t(), DType::kF32);
  std::vector<float> got(static_cast<size_t>(m * n));
  out.Download(d, got.data());

  const std::vector<float> ref = ReferenceApply(f, x, m);
  const double rel = RelRms(got, ref);
  MESSAGE("exl3 linear method vs weight-side dequant: rel_rms=", rel);
  CHECK(rel <= 2.0e-3);
  // Not vacuous: the reference has to be a real, non-degenerate answer, or a
  // method returning zeros would pass the line above.
  double mag = 0.0;
  for (float v : ref) mag += static_cast<double>(v) * v;
  REQUIRE(mag > 0.0);

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(q);
}

TEST_CASE("exl3 linear method: the OUT dtype is the caller's, not the kernel's") {
  vt::Queue q = CpuQueue();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};

  const int64_t m = 2, k = 128, n = 128;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0x0DDBA11u);
  const vllm::Exl3Weight w = WrapFixture(f);
  OwnedTensor bf16;
  auto method = layers::MakeLinearMethod(bf16, w);

  Rng rng;
  rng.s = 0x1234567u;
  std::vector<float> x(static_cast<size_t>(m * k));
  for (auto& v : x) v = vt::F16ToF32(vt::F32ToF16(rng.next(1.0f)));
  vllm::dense_attn::DBuf xb(d, DType::kF32, {m, k}, x.data());

  vllm::dense_attn::DBuf f32_out = method->Apply(d, xb.t(), DType::kF32);
  vllm::dense_attn::DBuf bf16_out = method->Apply(d, xb.t(), DType::kBF16);
  CHECK(f32_out.t().dtype == DType::kF32);
  CHECK(bf16_out.t().dtype == DType::kBF16);

  // The bf16 arm is the f32 answer rounded ONCE, not a different computation.
  std::vector<float> a(static_cast<size_t>(m * n));
  std::vector<uint16_t> bbits(static_cast<size_t>(m * n));
  f32_out.Download(d, a.data());
  bf16_out.Download(d, bbits.data());
  for (size_t i = 0; i < a.size(); ++i) CHECK(bbits[i] == vt::F32ToBF16(a[i]));

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(q);
}

TEST_CASE("exl3 linear method: a mismatched activation width REFUSES BY NAME") {
  vt::Queue q = CpuQueue();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};

  const Exl3Fixture f = MakeFixture(128, 128, 3, 0xBADu);
  const vllm::Exl3Weight w = WrapFixture(f);
  OwnedTensor bf16;
  auto method = layers::MakeLinearMethod(bf16, w);

  std::vector<float> x(2 * 64, 0.5f);
  vllm::dense_attn::DBuf xb(d, DType::kF32, {2, 64}, x.data());

  // BY NAME is the claim, so the message is the assertion. A bare CHECK_THROWS
  // here passes on `vt::CastF16`'s downstream "same element count" throw just as
  // happily, which would leave THIS refusal ungated while the case still read
  // green — and deleting the check in a scratch copy proved exactly that.
  std::string what;
  try {
    method->Apply(d, xb.t(), DType::kF32);
    FAIL("exl3 linear: a mismatched activation width did NOT throw at all");
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("refusal message: " << what);
  CHECK(what.find("exl3 linear") != std::string::npos);
  CHECK(what.find("the weight needs K=") != std::string::npos);
  CHECK(what.find("128") != std::string::npos);

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(q);
}

TEST_CASE("exl3 merge: packed K rows and independent transforms are byte exact") {
  for (int bits : {4, 6}) {
    CAPTURE(bits);
    auto a = WrapFixture(MakeFixture(256, 128, bits, 0x1111u));
    auto b = WrapFixture(MakeFixture(256, 256, bits, 0x2222u));
    auto c = WrapFixture(MakeFixture(256, 128, bits, 0x3333u));
    a.codebook = b.codebook = c.codebook = 2;
    auto group = vllm::MergeExl3Weights({&a, &b, &c}, "three_sources");
    CHECK(group.output_offsets == std::vector<int64_t>{0, 128, 384, 512});
    REQUIRE(group.trellis.bytes.size() == 16 * 32 * 32 * bits);
    REQUIRE(group.suh.bytes.size() == 3 * 256 * 2);
    REQUIRE(group.svh.bytes.size() == 512 * 2);
    REQUIRE(group.source_map.bytes.size() == 4 * 4);
    const std::vector<const vllm::Exl3Weight*> sources{&a, &b, &c};
    for (size_t s = 0; s < sources.size(); ++s) {
      const auto& w = *sources[s];
      const size_t source_row = static_cast<size_t>(w.OutFeatures() / 16) * 32 * bits;
      for (int kt = 0; kt < 16; ++kt)
        CHECK(std::memcmp(group.trellis.bytes.data() + kt * 32 * 32 * bits +
                              group.output_offsets[s] / 16 * 32 * bits,
                          w.trellis.bytes.data() + kt * source_row, source_row) == 0);
      CHECK(std::memcmp(group.suh.bytes.data() + s * 256 * 2, w.suh.bytes.data(), 256 * 2) == 0);
      CHECK(std::memcmp(group.svh.bytes.data() + group.output_offsets[s] * 2,
                        w.svh.bytes.data(), w.OutFeatures() * 2) == 0);
    }
    const int32_t map[] = {0, 1, 1, 2};
    CHECK(std::memcmp(group.source_map.bytes.data(), map, sizeof(map)) == 0);
    // The result has independent ownership, including across a move/source loss.
    const auto expected = group.trellis.bytes;
    auto moved = std::move(group);
    a = {}; b = {}; c = {};
    CHECK(moved.trellis.bytes == expected);
  }
  auto a = WrapFixture(MakeFixture(128, 128, 4, 1));
  auto b = WrapFixture(MakeFixture(128, 128, 4, 2));
  a.codebook = b.codebook = 2;
  CHECK_THROWS(vllm::MergeExl3Weights({}, "empty"));
  CHECK_THROWS(vllm::MergeExl3Weights({nullptr}, "null"));
  auto bad = b;
  bad.codebook = 1;
  CHECK_THROWS(vllm::MergeExl3Weights({&a, &bad}, "codebook"));
  bad = WrapFixture(MakeFixture(128, 128, 6, 2)); bad.codebook = 2;
  CHECK_THROWS(vllm::MergeExl3Weights({&a, &bad}, "bits"));
  bad = WrapFixture(MakeFixture(256, 128, 4, 2)); bad.codebook = 2;
  CHECK_THROWS(vllm::MergeExl3Weights({&a, &bad}, "K"));
  bad = b; bad.trellis.shape[1] = 4;
  CHECK_THROWS(vllm::MergeExl3Weights({&a, &bad}, "boundary"));
  bad = b; bad.trellis.bytes.resize(1);
  CHECK_THROWS(vllm::MergeExl3Weights({&a, &bad}, "span"));
  bad = b; bad.trellis.shape[0] = INT64_MAX;
  CHECK_THROWS(vllm::MergeExl3Weights({&bad}, "K overflow"));
  bad = b; bad.trellis.shape[1] = INT64_MAX;
  CHECK_THROWS(vllm::MergeExl3Weights({&a, &bad}, "N overflow"));
  bad = b; bad.suh.shape[0] = 256;
  CHECK_THROWS(vllm::MergeExl3Weights({&a, &bad}, "transform shape"));
}

#ifdef VLLM_CPP_XPU
TEST_CASE("exl3 gate up: explicit FP16 policy preserves both projection boundaries (XPU)") {
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kXPU);
  vt::Queue q = b.CreateQueue();
  {
    vllm::dense_attn::Dev d{b, q, DType::kF16};
    const Exl3Fixture gf = MakeFixture(128, 128, 4, 0x1234u);
    const Exl3Fixture uf = MakeFixture(128, 128, 4, 0x5678u);
    vllm::Exl3Weight gate = WrapFixture(gf), up = WrapFixture(uf);
    gate.codebook = up.codebook = 2;  // mul1, as in the B70 checkpoint
    OwnedTensor dense;
    vllm::Exl3GroupedWeight grouped;
    auto method = layers::MakeMlpGateUpMethod(dense, gate, up, 128, &grouped);
    vt::EnableOpProviderCallStats(true);
    const void* packed_resident = nullptr;
    const void* map_resident = nullptr;
    for (int64_t m : {1, 4}) {
      CAPTURE(m);
      Rng rng;
      std::vector<uint16_t> x(static_cast<size_t>(m * 128));
      for (auto& v : x) v = vt::F32ToF16(rng.next(0.05f));
      vllm::dense_attn::DBuf xb(d, DType::kF16, {m, 128}, x.data());
      const auto calls = vt::GetOpProviderStats(vt::OpId::kExl3GroupedLinear, vt::DeviceType::kXPU).selections;
      auto out = method->Apply(d, xb.t());
      CHECK(vt::GetOpProviderStats(vt::OpId::kExl3GroupedLinear, vt::DeviceType::kXPU).selections == calls + 1);
      REQUIRE(grouped.suh.shape[0] == 2);
      REQUIRE(grouped.trellis.d_dev != nullptr);
      REQUIRE(grouped.source_map.d_dev != nullptr);
      CHECK(grouped.trellis.host_released);
      CHECK(grouped.suh.host_released);
      CHECK(grouped.svh.host_released);
      CHECK(grouped.source_map.host_released);
      if (packed_resident) {
        CHECK(grouped.trellis.d_dev.get() == packed_resident);
        CHECK(grouped.source_map.d_dev.get() == map_resident);
      } else {
        CHECK(gate.trellis.d_dev == nullptr);
        CHECK(up.trellis.d_dev == nullptr);
        packed_resident = grouped.trellis.d_dev.get();
        map_resident = grouped.source_map.d_dev.get();
      }
      REQUIRE(out.t().dtype == DType::kF16);
      REQUIRE(out.t().shape[0] == m);
      REQUIRE(out.t().shape[1] == 128);

      // Public kernel outputs bypass the gate/up seam. This checks its dtype
      // propagation and SwiGLU rounding, not independent checkpoint parity.
      vllm::dense_attn::DBuf g(d, DType::kF16, {m, 128});
      vllm::dense_attn::DBuf u(d, DType::kF16, {m, 128});
      const auto plan = vt::PlanExl3SmallM(m, 128, 128, 4);
      vllm::dense_attn::DBuf had(d, DType::kF16, {1, 8, plan.padded_rows, 16});
      vllm::dense_attn::DBuf parts(d, DType::kF32, {plan.splits, m, 128});
      vllm::dense_attn::DBuf shard(d, DType::kI32, {1});
      shard.Zero(d);
      const auto gt = vllm::dense_attn::ResidentWeight(d, gate.trellis);
      const auto gsuh = vllm::dense_attn::Reshape(
          vllm::dense_attn::ResidentWeight(d, gate.suh), {1, 128});
      const auto gsvh = vllm::dense_attn::ResidentWeight(d, gate.svh);
      const auto ut = vllm::dense_attn::ResidentWeight(d, up.trellis);
      const auto usuh = vllm::dense_attn::Reshape(
          vllm::dense_attn::ResidentWeight(d, up.suh), {1, 128});
      const auto usvh = vllm::dense_attn::ResidentWeight(d, up.svh);
      vt::Exl3GroupedLinear(q, g.t(), xb.t(), gt, gsuh, gsvh, shard.t(),
                            had.t(), parts.t(), {4, 2});
      vt::Exl3GroupedLinear(q, u.t(), xb.t(), ut, usuh, usvh, shard.t(),
                            had.t(), parts.t(), {4, 2});
      std::vector<uint16_t> gb(x.size()), ub(x.size()), got(x.size());
      g.Download(d, gb.data());
      u.Download(d, ub.data());
      out.Download(d, got.data());
      bool nonzero = false;
      for (size_t i = 0; i < got.size(); ++i) {
        const float gv = vt::F16ToF32(gb[i]);
        const float silu = vt::F16ToF32(vt::F32ToF16(gv / (1.0f + std::exp(-gv))));
        const uint16_t expected = vt::F32ToF16(silu * vt::F16ToF32(ub[i]));
        CHECK(std::isfinite(vt::F16ToF32(got[i])));
        CHECK(got[i] == expected);
        nonzero |= vt::F16ToF32(got[i]) != 0.0f;
      }
      CHECK(nonzero);
      vllm::dense_attn::Dev legacy{b, q};
      auto bf16_out = method->Apply(legacy, xb.t());
      CHECK(bf16_out.t().dtype == DType::kBF16);
      b.Synchronize(q);
    }
  }
  b.DestroyQueue(q);
}
#endif

TEST_CASE("exl3 linear method: the f16 OUT arm is the kernel's own, and is executed") {
  // The f16 arm is the one `Exl3Gemm` writes natively, and it was the arm no
  // case asked for: gutting it in a scratch copy left the suite green, so the
  // header's "the output dtype is the caller's" paragraph was unpinned exactly
  // where the caller and the kernel agree.
  vt::Queue q = CpuQueue();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};

  const int64_t m = 2, k = 128, n = 128;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0xF16Au);
  const vllm::Exl3Weight w = WrapFixture(f);
  OwnedTensor bf16;
  auto method = layers::MakeLinearMethod(bf16, w);

  Rng rng;
  rng.s = 0x2468ACEu;
  std::vector<float> x(static_cast<size_t>(m * k));
  for (auto& v : x) v = vt::F16ToF32(vt::F32ToF16(rng.next(1.0f)));
  vllm::dense_attn::DBuf xb(d, DType::kF32, {m, k}, x.data());

  vllm::dense_attn::DBuf f16_out = method->Apply(d, xb.t(), DType::kF16);
  vllm::dense_attn::DBuf f32_out = method->Apply(d, xb.t(), DType::kF32);
  CHECK(f16_out.t().dtype == DType::kF16);

  std::vector<uint16_t> got(static_cast<size_t>(m * n));
  std::vector<float> ref(static_cast<size_t>(m * n));
  f16_out.Download(d, got.data());
  f32_out.Download(d, ref.data());

  // The f16 arm is the same answer at the kernel's own width. It is NOT
  // byte-equal to `F32ToF16(f32 arm)` in general -- the kernel's own f16 output
  // transform rounds inside `had_r_128` rather than after it -- so this is a
  // bounded agreement, and the bound is one f16 ulp of the value.
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d16 = static_cast<double>(vt::F16ToF32(got[i]));
    worst = std::max(worst, std::abs(d16 - static_cast<double>(ref[i])));
  }
  MESSAGE("f16 out arm vs f32 out arm: worst abs = ", worst);
  double mag = 0.0;
  for (float v : ref) mag = std::max(mag, static_cast<double>(std::abs(v)));
  REQUIRE(mag > 0.0);  // not vacuous: a zeroed arm would pass any bound below
  CHECK(worst <= 4.0 * UlpF16(static_cast<float>(mag)));

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(q);
}

TEST_CASE("exl3 linear method: an out dtype it cannot write REFUSES") {
  // Deleting this check in a scratch copy left the suite green, and what it
  // then does is worse than a wrong number: `Apply(d, x, kI8)` falls through
  // the f16 and f32 arms and silently returns a kBF16 buffer, so the caller
  // gets a different dtype than it asked for with no diagnostic anywhere.
  vt::Queue q = CpuQueue();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};

  const Exl3Fixture f = MakeFixture(128, 128, 3, 0x0D7Du);
  const vllm::Exl3Weight w = WrapFixture(f);
  OwnedTensor bf16;
  auto method = layers::MakeLinearMethod(bf16, w);

  std::vector<float> x(2 * 128, 0.25f);
  vllm::dense_attn::DBuf xb(d, DType::kF32, {2, 128}, x.data());

  std::string what;
  try {
    method->Apply(d, xb.t(), DType::kI8);
    FAIL("exl3 linear: an unwritable out dtype did NOT throw at all");
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("refusal message: " << what);
  CHECK(what.find("out_dtype") != std::string::npos);

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(q);
}
