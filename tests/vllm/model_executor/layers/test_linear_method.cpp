// LinearMethod / QuantizationConfig seam — scheme×device selection + bf16 apply.
//
// Ports vLLM's scheme-parameterized create_weights/apply coverage
// (tests/kernels/quantization/**) to the vt::-native seam (work row S4 of
// .agents/specs/accelerator-seam-audit.md): the scheme is chosen ONCE by the
// factory from the checkpoint's populated weights, and the bf16 UnquantizedLinear
// apply runs the exact vt::MatmulBT the inline model path did.
//
// CPU-only (no checkpoint), runs in CI. The NVFP4 numeric path is gated on dgx
// via the paged-engine model tests; here we assert the FACTORY SELECTS the right
// method per scheme (the S4 policy decision) and that the bf16 apply is correct.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include "vt/merged_gemm.h"
#include <string>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/schemes/nvfp4.h"
#include "vllm/model_executor/layers/quantization/fp8.h"
#include "vllm/model_executor/model_loader/mxfp4_dequant.h"
#include "vt/backend.h"
#include "vt/dtype.h"

#include <cmath>
#include <random>

namespace {

using vllm::Nvfp4Weight;
using vllm::OwnedTensor;
using vt::DType;
namespace layers = vllm::layers;

vllm::OwnedTensor MakeBf16(const std::vector<int64_t>& shape, uint32_t seed) {
  OwnedTensor o;
  o.dtype = DType::kBF16;
  o.nk = true;
  o.rank = static_cast<int>(shape.size());
  int64_t numel = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[static_cast<size_t>(i)];
    numel *= shape[static_cast<size_t>(i)];
  }
  o.bytes.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(o.bytes.data());
  uint32_t s = seed;
  for (int64_t i = 0; i < numel; ++i) {
    s = s * 1664525u + 1013904223u;
    const float v = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 0.2f;
    p[i] = vt::F32ToBF16(v);
  }
  return o;
}

// A minimal non-empty W4A16 NVFP4 weight (alpha == 0): enough for the factory to
// select the quantized scheme. Its numeric path is exercised on dgx, not here.
Nvfp4Weight MakeNvfp4W4A16(int64_t N, int64_t K) {
  Nvfp4Weight w;
  w.n = N;
  w.k = K;
  w.scale2 = 1.0f;
  w.packed.dtype = DType::kI8;
  w.packed.rank = 2;
  w.packed.shape[0] = N;
  w.packed.shape[1] = K / 2;
  w.packed.bytes.resize(static_cast<size_t>(N) * (K / 2), 0);
  w.scale.dtype = DType::kI8;
  w.scale.rank = 2;
  w.scale.shape[0] = N;
  w.scale.shape[1] = K / 16;
  w.scale.bytes.resize(static_cast<size_t>(N) * (K / 16), 0);
  return w;
}

#ifdef VT_MARLIN_NVFP4
// A random MXFP4 W4A16 weight: E2M1 packed [N,K/2] + E8M0 scale [N,K/32], group
// 32, no global, is_mxfp4=true — so the factory + Apply route the MXFP4 keep-quant
// path (Marlin on GPU via BuildMarlinDenseResident).
Nvfp4Weight MakeMxfp4W4A16(int64_t N, int64_t K, uint32_t seed) {
  Nvfp4Weight w;
  w.n = N;
  w.k = K;
  w.group_size = 32;
  w.is_mxfp4 = true;
  w.scale2 = 0.0f;
  w.packed.dtype = DType::kI8;
  w.packed.rank = 2;
  w.packed.shape[0] = N;
  w.packed.shape[1] = K / 2;
  w.packed.bytes.resize(static_cast<size_t>(N) * (K / 2));
  w.scale.dtype = DType::kI8;
  w.scale.rank = 2;
  w.scale.shape[0] = N;
  w.scale.shape[1] = K / 32;
  w.scale.bytes.resize(static_cast<size_t>(N) * (K / 32));
  std::mt19937 rng(seed);
  for (auto& b : w.packed.bytes) b = static_cast<uint8_t>(rng() & 0xFFu);
  for (auto& s : w.scale.bytes) s = static_cast<uint8_t>(118u + (rng() % 15u));
  return w;
}

#endif  // VT_MARLIN_NVFP4
}  // namespace

#ifdef VT_MARLIN_NVFP4
// The model-facing MXFP4 path END-TO-END: MakeLinearMethod(bf16-empty, mxfp4) ->
// Apply -> MatmulNvfp4W4A16D -> (GPU) MatmulNvfp4MarlinD -> BuildMarlinDenseResident
// -> MoeGroupedGemmNvfp4Marlin. This is the ONE link the op-level unit gate does NOT
// cover (it feeds MANUALLY-built residents), so it isolates a resident-builder bug
// from the kernel. Reference = the INDEPENDENT CPU dequant (DequantMxfp4ToF32 + f32
// matmul). Real Qwen3-8B projection shapes; M=1 (decode) AND M=8 (prefill).
TEST_CASE("linear_method: MXFP4 W4A16 Apply (Marlin BuildMarlinDenseResident) == CPU dequant ref") {
  vt::Backend* gpu = nullptr;
  try {
    gpu = &vt::GetBackend(vt::DeviceType::kCUDA);
  } catch (...) {
    MESSAGE("SKIP: no CUDA backend");
    return;
  }
  // Persist weights in a vector so each shape has a DISTINCT, stable address —
  // the resident cache (MarlinDenseResidentFor) is keyed by weight pointer, and a
  // loop-local reused stack slot would alias residents across shapes (a test
  // artifact, not a model bug: the model's weights are distinct persistent objects).
  const std::vector<std::pair<int64_t, int64_t>> shapes{{4096, 4096}, {12288, 4096}};
  std::vector<Nvfp4Weight> weights;
  for (auto KN : shapes) weights.push_back(MakeMxfp4W4A16(KN.second, KN.first, 2024));
  for (size_t si = 0; si < shapes.size(); ++si) {
    const int64_t K = shapes[si].first, N = shapes[si].second;
    CAPTURE(K);
    CAPTURE(N);
    Nvfp4Weight& w = weights[si];
    OwnedTensor bf16_empty;  // Empty() => factory selects the fp4 method

    std::vector<float> w_f32(static_cast<size_t>(N * K));
    vllm::DequantMxfp4ToF32(reinterpret_cast<const uint8_t*>(w.packed.bytes.data()),
                            reinterpret_cast<const uint8_t*>(w.scale.bytes.data()), N, K,
                            w_f32.data());

    for (int64_t M : {int64_t{1}, int64_t{8}}) {
      CAPTURE(M);
      vt::Queue q = gpu->CreateQueue();
      vllm::dense_attn::Dev d{*gpu, q};

      std::vector<uint16_t> act_bf16(static_cast<size_t>(M * K));
      std::mt19937 rng(7 + static_cast<uint32_t>(M));
      std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
      std::vector<float> act_r(static_cast<size_t>(M * K));
      for (size_t i = 0; i < act_bf16.size(); ++i) {
        act_bf16[i] = vt::F32ToBF16(dist(rng));
        act_r[i] = vt::BF16ToF32(act_bf16[i]);
      }
      std::vector<float> ref(static_cast<size_t>(M * N), 0.0f);
      for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
          float acc = 0.0f;
          for (int64_t k = 0; k < K; ++k)
            acc += act_r[static_cast<size_t>(m * K + k)] * w_f32[static_cast<size_t>(n * K + k)];
          ref[static_cast<size_t>(m * N + n)] = acc;
        }

      vllm::dense_attn::DBuf x(d, DType::kBF16, {M, K}, act_bf16.data());
      auto method = layers::MakeLinearMethod(bf16_empty, w);
      vllm::dense_attn::DBuf out = method->Apply(d, x.t(), DType::kBF16);
      std::vector<uint16_t> got_bf16(static_cast<size_t>(M * N));
      gpu->Copy(q, got_bf16.data(), out.t().data,
                got_bf16.size() * sizeof(uint16_t));
      gpu->Synchronize(q);
      double max_rel = 0.0, max_abs = 0.0;
      size_t bad = 0;
      for (size_t i = 0; i < got_bf16.size(); ++i) {
        const float g = vt::BF16ToF32(got_bf16[i]);
        const float a = std::fabs(g - ref[i]);
        const float tol = 2e-2f + 2e-2f * std::fabs(ref[i]);
        if (a > tol) ++bad;
        max_abs = std::max(max_abs, static_cast<double>(a));
        max_rel = std::max(max_rel, static_cast<double>(a / (std::fabs(ref[i]) + 1e-6f)));
      }
      MESSAGE("MXFP4 Apply K=" << K << " N=" << N << " M=" << M
              << " bad=" << bad << " max_abs=" << max_abs << " max_rel=" << max_rel);
      CHECK(bad == 0);
      gpu->DestroyQueue(q);
    }
  }
}

// STEP-2 (row QUANT-CT-MXFP4-MARLIN-STRUCT): the FUSED MXFP4 gate_up Marlin GEMM
// (one [2N,K] operand + SiluAndMul, vLLM's merged gate_up_proj structure) must be
// NUMERICALLY EQUIVALENT to the SPLIT path (two single-expert MXFP4 GEMMs + MoeSiluMul)
// it replaces on the classic-dense Qwen3-8B-MXFP4 decode. MXFP4 has NO cross-shard
// scale interaction (each group's E8M0 byte is passed through independently, no combined
// factor, no global), so the two paths compute the SAME math; they are NOT bit-identical
// because Marlin's fp32 split-K reduce groups the K-slices differently when the operand
// is [2N,K] vs [N,K] (a handful of last-bit-of-bf16 differences, ~0.1% of elements) —
// exactly like the NVFP4 fused path, which is gated token-exact vs the ORACLE (the #44
// smoke), not bit-vs-split. The authoritative model-level bar is that oracle smoke; here
// we assert the fused output stays within the project's proven MXFP4 tolerance of the
// split reference (which the test above proves matches the independent CPU dequant), and
// that the fused path ACTUALLY RAN. RED-first: before mxfp4 was wired into
// GateUpFusedMarlinD (hardcoded K/16 scale grid, mxfp4=false in the GEMM args, and
// GateUpFusedEligible excluding mxfp4) the fused call misread the group-32 E8M0 scales as
// group-16 fp8-e4m3 -> GROSSLY wrong (most elements far outside tol) and the eligibility
// REQUIRE failed. Real Qwen3-8B gate/up shape; M∈{1,8}.
TEST_CASE("linear_method: MXFP4 fused gate_up ~= split (numerically) + fused path ran") {
  vt::Backend* gpu = nullptr;
  try {
    gpu = &vt::GetBackend(vt::DeviceType::kCUDA);
  } catch (...) {
    MESSAGE("SKIP: no CUDA backend");
    return;
  }
  const int64_t N = 12288, K = 4096;  // gate/up: N=intermediate, K=hidden
  // Distinct persistent addresses (resident caches are keyed by weight pointer).
  std::vector<Nvfp4Weight> gate, up;
  gate.push_back(MakeMxfp4W4A16(N, K, 111));
  up.push_back(MakeMxfp4W4A16(N, K, 222));
  REQUIRE(vllm::dense_nvfp4::GateUpFusedEligible(gate[0], up[0]));
  for (int64_t M : {int64_t{1}, int64_t{8}}) {
    CAPTURE(M);
    vt::Queue q = gpu->CreateQueue();
    vllm::dense_attn::Dev d{*gpu, q};
    std::vector<uint16_t> act(static_cast<size_t>(M * K));
    std::mt19937 rng(7 + static_cast<uint32_t>(M));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& a : act) a = vt::F32ToBF16(dist(rng));
    vllm::dense_attn::DBuf x(d, DType::kBF16, {M, K}, act.data());

    // SPLIT reference (the established, byte-exact-vs-CPU-ref MXFP4 path).
    vllm::dense_attn::DBuf sg =
        vllm::dense_nvfp4::MatmulMxfp4W4A16D(d, x.t(), gate[0], DType::kBF16);
    vllm::dense_attn::DBuf su =
        vllm::dense_nvfp4::MatmulMxfp4W4A16D(d, x.t(), up[0], DType::kBF16);
    vllm::dense_attn::DBuf sref(d, DType::kBF16, {M, N});
    vt::MoeSiluMul(d.q, sref.t(), sg.t(), su.t());
    std::vector<uint16_t> split(static_cast<size_t>(M * N));
    gpu->Copy(q, split.data(), sref.t().data, split.size() * sizeof(uint16_t));
    gpu->Synchronize(q);

    // FUSED path under test (one [2N,K] Marlin GEMM + SiluAndMul).
    const uint64_t before = vllm::dense_nvfp4::GetW4A16Stats().fused_gate_up;
    vllm::dense_attn::DBuf fused =
        vllm::dense_nvfp4::GateUpFusedMarlinD(d, x.t(), gate[0], up[0]);
    const uint64_t after = vllm::dense_nvfp4::GetW4A16Stats().fused_gate_up;
    std::vector<uint16_t> fus(static_cast<size_t>(M * N));
    gpu->Copy(q, fus.data(), fused.t().data, fus.size() * sizeof(uint16_t));
    gpu->Synchronize(q);

    CHECK(after == before + 1);  // the fused path ACTUALLY RAN (positive signal)
    // Numerical-equivalence bar via the BIT-EXACT FRACTION. Same math => the vast
    // majority of the post-silu bf16 outputs are bit-identical; the only differences are
    // the elements where Marlin's fp32 split-K reduce grouped the K-slices differently
    // for the [2N,K] operand vs the two [N,K] operands (one bf16 ULP, which SiluAndMul's
    // nonlinearity can occasionally amplify on out-of-distribution RANDOM inputs — real
    // model activations are well-conditioned, and the #44 oracle smoke is token-exact).
    // A STRUCTURAL bug (wrong scale format/group, wrong operand layout) corrupts ~ALL
    // elements => bit-exact fraction collapses to ~0. So >=99% bit-identical cleanly
    // separates the correct fusion (measured ~99.95%) from any structural regression.
    size_t exact = 0;
    double max_abs = 0.0;
    for (size_t i = 0; i < fus.size(); ++i) {
      if (fus[i] == split[i]) ++exact;
      max_abs = std::max(
          max_abs, static_cast<double>(std::fabs(vt::BF16ToF32(fus[i]) -
                                                 vt::BF16ToF32(split[i]))));
    }
    const double frac = static_cast<double>(exact) / static_cast<double>(fus.size());
    MESSAGE("MXFP4 fused vs split M=" << M << " bitexact=" << exact << "/" << fus.size()
                                      << " (" << frac << ") max_abs=" << max_abs);
    CHECK(frac >= 0.99);  // same math (a structural bug collapses this to ~0)
    gpu->DestroyQueue(q);
  }
}
#endif  // VT_MARLIN_NVFP4

TEST_CASE("linear_method: factory selects bf16 vs nvfp4-w4a16 by weight presence") {
  OwnedTensor bf16 = MakeBf16({4, 16}, 1);
  Nvfp4Weight empty_fp4;      // Empty() == true
  Nvfp4Weight fp4 = MakeNvfp4W4A16(4, 16);
  REQUIRE(empty_fp4.Empty());
  REQUIRE_FALSE(fp4.Empty());

  // get_quant_method analogue: a bf16 checkpoint => UnquantizedLinearMethod.
  auto m_bf16 = layers::MakeLinearMethod(bf16, empty_fp4);
  CHECK(std::string(m_bf16->Name()) == "bf16-unquantized");

  // An NVFP4-packed checkpoint => the compressed-tensors W4A16 method, chosen
  // ONCE here (not by a per-call IsNvfp4() probe in the model forward).
  auto m_fp4 = layers::MakeLinearMethod(bf16, fp4);
  CHECK(std::string(m_fp4->Name()) == "compressed-tensors-nvfp4-w4a16");
}

TEST_CASE("linear_method: gate_up factory selects scheme by weight presence") {
  OwnedTensor gate_up = MakeBf16({2 * 16, 8}, 2);
  Nvfp4Weight empty;
  Nvfp4Weight gate = MakeNvfp4W4A16(16, 8);
  Nvfp4Weight up = MakeNvfp4W4A16(16, 8);

  auto g_bf16 = layers::MakeMlpGateUpMethod(gate_up, empty, empty, 16);
  CHECK(std::string(g_bf16->Name()) == "bf16-gate-up");

  auto g_fp4 = layers::MakeMlpGateUpMethod(gate_up, gate, up, 16);
  CHECK(std::string(g_fp4->Name()) == "compressed-tensors-nvfp4-w4a16-gate-up");
}

// Tier-A1 fold REUSE PROOF (arch-fusion-fold-plan-2026-07-30 §A1): the SHARED bf16
// gate-up MLP seam (UnquantizedMlpGateUpMethod::Apply) must produce a BYTE-IDENTICAL
// result to the standalone {ResidentWeight; MatmulBT[2I,H]; SiluAndMul} sequence the
// five folded arch MLP blocks (OLMo-2 / Granite / StableLM / qwen3_dflash /
// deepseek_v2) hand-rolled before the fold. Same ops, same order, same device ⇒ the
// comparison is EXACT (raw bf16 bytes), not Approx. This is the CPU composite-golden
// backing the two archs (dflash, deepseek_v2) whose checkpoints are not always on the
// gate box; the three with SACRED goldens (OLMo-2/Granite/StableLM) are additionally
// gated token-exact on dgx.
TEST_CASE("linear_method: fused gate-up seam == standalone MatmulBT+SiluAndMul (byte-exact)") {
  const int64_t M = 3, H = 8, I = 5;
  OwnedTensor gate_up = MakeBf16({2 * I, H}, 11);  // merged [2I, H] raw-NK
  OwnedTensor xw = MakeBf16({M, H}, 13);

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};
  vllm::dense_attn::DBuf x(d, DType::kBF16, {M, H}, xw.bytes.data());

  // (A) SHARED seam: pick the bf16 arm via the factory (no fp4 present), Apply.
  auto method = layers::MakeMlpGateUpMethod(gate_up, Nvfp4Weight{}, Nvfp4Weight{}, I);
  REQUIRE(std::string(method->Name()) == "bf16-gate-up");
  vllm::dense_attn::DBuf act_fused = method->Apply(d, x.t());

  // (B) STANDALONE reference: the exact op sequence the arch MLP blocks ran.
  vt::Tensor wgu = vllm::dense_attn::ResidentWeight(d, gate_up);  // [2I, H]
  vllm::dense_attn::DBuf gu(d, DType::kBF16, {M, 2 * I});
  vt::MatmulBT(d.q, gu.t(), x.t(), wgu);
  vllm::dense_attn::DBuf act_ref(d, DType::kBF16, {M, I});
  vt::SiluAndMul(d.q, act_ref.t(), gu.t());

  std::vector<uint16_t> got(static_cast<size_t>(M) * I);
  std::vector<uint16_t> ref(static_cast<size_t>(M) * I);
  act_fused.Download(d, got.data());
  act_ref.Download(d, ref.data());
  for (size_t i = 0; i < got.size(); ++i)
    CHECK(got[i] == ref[i]);  // BYTE-IDENTICAL — the fold changes nothing numerically
}

// FUSION-DENSE-MIGRATE fold REUSE PROOF (issue #299, specs/fusion-dense-migrate.md):
// the five dense SwiGLU archs folded 2026-08-10 (Command-R / GLM-4 / MiniCPM /
// MiniCPM3 / Phi-3) construct layers::UnquantizedMlpGateUpMethod DIRECTLY --
// `layers::UnquantizedMlpGateUpMethod(&w.gate_up_proj, I).Apply(d, x)` -- rather than
// through MakeMlpGateUpMethod, because none of their loaders ever populates *_fp4.
// The case above proves the FACTORY arm at one prefill-shaped M; this one proves the
// DIRECTLY-CONSTRUCTED arm at BOTH the decode shape (M == 1, the batch-1 GEMV regime
// every one of these archs actually decodes in) and a prefill shape (M > 1), against
// the exact {ResidentWeight; MatmulBT[2I,H]; SiluAndMul} sequence each of the five
// hand-rolled before the fold. Raw bf16 byte compare, not Approx: the fold is a
// ROUTING change and any numerical difference at all is a defect, not a tolerance.
TEST_CASE("linear_method: direct UnquantizedMlpGateUpMethod == standalone (byte-exact, decode+prefill)") {
  const int64_t H = 8, I = 5;
  OwnedTensor gate_up = MakeBf16({2 * I, H}, 11);  // merged [2I, H] raw-NK

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};

  for (int64_t M : {int64_t{1}, int64_t{4}}) {  // decode, then prefill
    CAPTURE(M);
    OwnedTensor xw = MakeBf16({M, H}, 17);
    vllm::dense_attn::DBuf x(d, DType::kBF16, {M, H}, xw.bytes.data());

    // (A) The folded call, verbatim as the five model TUs now spell it.
    vllm::dense_attn::DBuf act_fused =
        layers::UnquantizedMlpGateUpMethod(&gate_up, I).Apply(d, x.t());

    // (B) The pre-fold hand-rolled sequence, verbatim.
    vt::Tensor wgu = vllm::dense_attn::ResidentWeight(d, gate_up);  // [2I, H]
    vllm::dense_attn::DBuf gu(d, DType::kBF16, {M, 2 * I});
    vt::MatmulBT(d.q, gu.t(), x.t(), wgu);
    vllm::dense_attn::DBuf act_ref(d, DType::kBF16, {M, I});
    vt::SiluAndMul(d.q, act_ref.t(), gu.t());

    // The seam derives its row count from the activation, which is what makes the
    // fold safe: every call site passed a DBuf{T,H}, so x.shape[0] == T identically.
    REQUIRE(act_fused.t().shape[0] == M);
    REQUIRE(act_fused.t().shape[1] == I);

    std::vector<uint16_t> got(static_cast<size_t>(M) * I);
    std::vector<uint16_t> ref(static_cast<size_t>(M) * I);
    act_fused.Download(d, got.data());
    act_ref.Download(d, ref.data());
    for (size_t i = 0; i < got.size(); ++i)
      CHECK(got[i] == ref[i]);  // BYTE-IDENTICAL — routing changed, numerics did not
  }
}

// Tier-C1 fold REUSE PROOF (arch-fusion-fold-plan-2026-07-30 §C1): the SHARED bf16
// GeGLU gate-up MLP seam (UnquantizedMlpGateUpGeluMethod::Apply) must produce a
// BYTE-IDENTICAL result to the standalone {ResidentWeight; MatmulBT[2I,H]; GeluAndMul}
// sequence the four Gemma-family MLP blocks (Gemma-1/2/3/4) hand-rolled before the
// fold. Same merged [2I,H] operand, same single MatmulBT, same GeluAndMul(tanh)
// epilogue, same device ⇒ EXACT (raw bf16 bytes), not Approx. This is the GeGLU
// sibling of the SwiGLU byte-exact case above; the SACRED gates (Gemma-2 48/48,
// Gemma-4 32/32) prove the same shared method is token-exact end-to-end on the GPU.
TEST_CASE("linear_method: fused GeGLU gate-up seam == standalone MatmulBT+GeluAndMul (byte-exact)") {
  const int64_t M = 3, H = 8, I = 5;
  OwnedTensor gate_up = MakeBf16({2 * I, H}, 17);  // merged [2I, H] raw-NK
  OwnedTensor xw = MakeBf16({M, H}, 19);

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};
  vllm::dense_attn::DBuf x(d, DType::kBF16, {M, H}, xw.bytes.data());

  // (A) SHARED seam: the bf16 GeGLU arm (GeluAndMul(tanh) epilogue), Apply.
  layers::UnquantizedMlpGateUpGeluMethod method(&gate_up, I);
  REQUIRE(std::string(method.Name()) == "bf16-gate-up-gelu");
  vllm::dense_attn::DBuf act_fused = method.Apply(d, x.t());

  // (B) STANDALONE reference: the exact op sequence the Gemma MLP blocks ran.
  vt::Tensor wgu = vllm::dense_attn::ResidentWeight(d, gate_up);  // [2I, H]
  vllm::dense_attn::DBuf gu(d, DType::kBF16, {M, 2 * I});
  vt::MatmulBT(d.q, gu.t(), x.t(), wgu);
  vllm::dense_attn::DBuf act_ref(d, DType::kBF16, {M, I});
  vt::GeluAndMul(d.q, act_ref.t(), gu.t());

  std::vector<uint16_t> got(static_cast<size_t>(M) * I);
  std::vector<uint16_t> ref(static_cast<size_t>(M) * I);
  act_fused.Download(d, got.data());
  act_ref.Download(d, ref.data());
  for (size_t i = 0; i < got.size(); ++i)
    CHECK(got[i] == ref[i]);  // BYTE-IDENTICAL — the fold changes nothing numerically
}

TEST_CASE("linear_method: bf16 UnquantizedLinearMethod apply == reference MatmulBT") {
  const int64_t M = 2, K = 16, N = 4;
  OwnedTensor w = MakeBf16({N, K}, 7);  // raw-NK [N=out, K=in]
  OwnedTensor xw = MakeBf16({M, K}, 9);

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};

  vllm::dense_attn::DBuf x(d, DType::kBF16, {M, K}, xw.bytes.data());
  auto method = layers::MakeLinearMethod(w, Nvfp4Weight{});
  REQUIRE(std::string(method->Name()) == "bf16-unquantized");
  vllm::dense_attn::DBuf out = method->Apply(d, x.t(), DType::kF32);

  std::vector<float> got(static_cast<size_t>(M) * N);
  out.Download(d, got.data());

  const auto* wp = reinterpret_cast<const uint16_t*>(w.bytes.data());
  const auto* xp = reinterpret_cast<const uint16_t*>(xw.bytes.data());
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k)
        acc += vt::BF16ToF32(xp[m * K + k]) * vt::BF16ToF32(wp[n * K + k]);
      CHECK(got[static_cast<size_t>(m) * N + n] == doctest::Approx(acc).epsilon(0.02));
    }
  }
}

// ===========================================================================
// FP8 W8A8 (per-tensor) — the seam extracted from qwen3_5.cpp by #940.
//
// Same two questions this file already asks of NVFP4, asked of FP8: does the
// factory pick the scheme ONCE from the checkpoint (get_quant_method), and is
// the extracted compute REACHED rather than sitting dead beside the model. The
// numeric arm runs on dgx (the fp8 GEMM is CUDA-only, see below); what a host
// build can prove is selection and the inherited device refusal, and both of
// those go red if the guard or the wiring in dense_fp8_gemm.h moves.
namespace {

// A minimal non-empty per-tensor FP8 W8A8 weight in the shape `LoadFp8Raw`
// produces (qwen3_5_weights.cpp:423): raw e4m3fn [N,K] bytes, per-tensor
// weight_scale + input_scale, folded alpha. Powers of two, so nothing here
// rounds. Bytes stay below 0x7E — 0x7F/0xFF are NaN in e4m3fn.
vllm::Fp8Weight MakeFp8W8A8(int64_t N, int64_t K, uint32_t seed) {
  vllm::Fp8Weight w;
  w.n = N;
  w.k = K;
  w.weight_scale = 0.00390625F;
  w.input_scale = 0.0078125F;
  w.alpha = w.input_scale * w.weight_scale;
  w.packed.dtype = DType::kI8;
  w.packed.rank = 2;
  w.packed.shape[0] = N;
  w.packed.shape[1] = K;
  w.packed.bytes.resize(static_cast<size_t>(N * K));
  uint32_t s = seed;
  auto* bytes = reinterpret_cast<uint8_t*>(w.packed.bytes.data());
  for (int64_t i = 0; i < N * K; ++i) {
    s = s * 1664525u + 1013904223u;
    bytes[static_cast<size_t>(i)] = static_cast<uint8_t>((s >> 16) % 0x7EU);
  }
  return w;
}

}  // namespace

TEST_CASE("linear_method: factory selects bf16 vs fp8-w8a8 by weight presence") {
  OwnedTensor bf16 = MakeBf16({4, 16}, 3);
  vllm::Fp8Weight empty_fp8;  // Empty() == true
  vllm::Fp8Weight fp8 = MakeFp8W8A8(4, 16, 31);
  REQUIRE(empty_fp8.Empty());
  REQUIRE_FALSE(fp8.Empty());

  // get_quant_method analogue: a bf16 checkpoint => UnquantizedLinearMethod.
  auto m_bf16 = layers::MakeLinearMethod(bf16, empty_fp8);
  CHECK(std::string(m_bf16->Name()) == "bf16-unquantized");

  // An fp8-resident checkpoint => the per-tensor W8A8 method, chosen ONCE here.
  auto m_fp8 = layers::MakeLinearMethod(bf16, fp8);
  CHECK(std::string(m_fp8->Name()) == "fp8-w8a8-per-tensor");

  // The two overloads coexist: same call spelling, different weight type, and
  // the NVFP4 one is unaffected by the FP8 one being in scope.
  auto m_fp4 = layers::MakeLinearMethod(bf16, MakeNvfp4W4A16(4, 16));
  CHECK(std::string(m_fp4->Name()) == "compressed-tensors-nvfp4-w4a16");
}

// REACHABILITY of the extracted seam from the POLICY layer. Both arms of
// dense_fp8_gemm.h run their `kMatmulFp8CublasLt`-registered guard first, and
// that op is CUDA-only (tests/vt/test_ops_fp8_cpu.cpp:445-453 pins that), so on
// a host queue each must THROW that exact refusal. This is not a test of the
// refusal for its own sake: it is the assertion that `Fp8W8A8LinearMethod::
// Apply` / `ApplyPreQuantized` actually execute the shared template bodies.
// Deleting either VT_CHECK from dense_fp8_gemm.h, or pointing the method at
// something else, turns this case red.
TEST_CASE("linear_method: the fp8 w8a8 method reaches the shared seam in both arms") {
  const int64_t M = 2, K = 16, N = 4;
  OwnedTensor bf16 = MakeBf16({N, K}, 11);
  OwnedTensor xw = MakeBf16({M, K}, 13);
  vllm::Fp8Weight fp8 = MakeFp8W8A8(N, K, 17);

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};
  vllm::dense_attn::DBuf x(d, DType::kBF16, {M, K}, xw.bytes.data());

  auto method = layers::MakeLinearMethod(bf16, fp8);
  REQUIRE(std::string(method->Name()) == "fp8-w8a8-per-tensor");

  CHECK_THROWS_WITH_AS(method->Apply(d, x.t(), DType::kF32),
                       doctest::Contains("MatmulFp8CutlassD: the fp8 W8A8 path is CUDA-only"),
                       std::runtime_error);

  // The pre-quantized arm (the `QuantizedActivation` overload) reaches its own
  // entry point, not the plain one — the message names which.
  const auto* fp8_method =
      dynamic_cast<const layers::Fp8W8A8LinearMethod*>(method.get());
  REQUIRE(fp8_method != nullptr);
  vllm::dense_attn::DBuf a_fp8(d, DType::kI8, {M, K});
  CHECK_THROWS_WITH_AS(
      fp8_method->ApplyPreQuantized(d, a_fp8.t(), DType::kBF16),
      doctest::Contains("MatmulFp8CutlassPreQuantD: the fp8 W8A8 path is CUDA-only"),
      std::runtime_error);

  // The refusal is the OP TABLE's answer, not a hardcoded device test: the two
  // CPU reference arms #468/#842 registered DO resolve here, and the model-layer
  // predicate still names the cuBLASLt op. That is the residual gap, pinned.
  CHECK(vt::OpRegistered(vt::OpId::kQuantFp8Static, vt::DeviceType::kCPU));
  CHECK(vt::OpRegistered(vt::OpId::kMatmulFp8Cutlass, vt::DeviceType::kCPU));
  CHECK_FALSE(vt::OpRegistered(vt::OpId::kMatmulFp8CublasLt, vt::DeviceType::kCPU));
}

namespace {
OwnedTensor MakeQuantShard(DType dtype, int64_t rows, int64_t cols, int seed) {
  OwnedTensor w;
  w.dtype = dtype;
  w.nk = true;
  w.rank = 2;
  w.shape[0] = rows;
  w.shape[1] = cols;
  const auto row_bytes = vt::RowSizeBytes(dtype, cols);
  w.bytes.resize(static_cast<size_t>(rows * row_bytes));
  const int64_t block_bytes = vt::BlockBytes(dtype);
  for (int64_t i = 0; i < rows * cols / vt::BlockElems(dtype); ++i) {
    auto* p = reinterpret_cast<uint8_t*>(w.bytes.data()) + i * block_bytes;
    p[0] = 0x00;
    p[1] = 0x24;  // f16 scale 1/64, finite and nonzero.
    for (int64_t j = 2; j < block_bytes; ++j)
      p[j] = static_cast<uint8_t>((i * 13 + j * 7 + seed) % 127);
  }
  return w;
}
}

TEST_CASE("dense merged shards preserve heterogeneous quantized contractions") {
  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  auto& backend = vt::GetBackend(vt::DeviceType::kCPU);
  layers::Dev d{backend, queue};
  auto input = MakeBf16({3, 64}, 91);
  auto x = layers::ResidentWeight(d, input);
  std::vector<OwnedTensor> owners;
  owners.push_back(MakeQuantShard(DType::kQ4_0, 5, 64, 1));
  owners.push_back(MakeQuantShard(DType::kQ8_0, 3, 64, 2));
  owners.push_back(MakeQuantShard(DType::kQ4_0, 3, 64, 3));
  std::vector<vt::Tensor> weights;
  std::vector<layers::DBuf> output, expected;
  for (const auto& w : owners) {
    weights.push_back(layers::ResidentWeight(d, w));
    output.emplace_back(d, DType::kBF16, std::initializer_list<int64_t>{3, w.shape[0]});
    expected.emplace_back(d, DType::kBF16, std::initializer_list<int64_t>{3, w.shape[0]});
    vt::MatmulBT(queue, expected.back().t(), x, weights.back());
  }
  vt::Tensor products[] = {output[0].t(), output[1].t(), output[2].t()};
  vt::MergedGemmDense(queue, vt::kDenseQkvShards, products, x, weights.data(), 3);
  for (size_t i = 0; i < output.size(); ++i) {
    const auto n = static_cast<size_t>(3 * owners[i].shape[0]);
    CHECK(std::memcmp(output[i].t().data, expected[i].t().data, n * 2) == 0);
    CHECK(weights[i].dtype == owners[i].dtype);
  }
}

TEST_CASE("dense quantized gate up shards use the MLP method and shared epilogue") {
  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  auto& backend = vt::GetBackend(vt::DeviceType::kCPU);
  layers::Dev d{backend, queue};
  auto input = MakeBf16({2, 64}, 17);
  auto x = layers::ResidentWeight(d, input);
  std::vector<OwnedTensor> weights;
  weights.push_back(MakeQuantShard(DType::kQ4_0, 8, 64, 5));
  weights.push_back(MakeQuantShard(DType::kQ8_0, 8, 64, 6));
  layers::DenseShardMlpGateUpMethod method(&weights, 8);
  auto actual = method.Apply(d, x);
  auto wg = layers::ResidentWeight(d, weights[0]);
  auto wu = layers::ResidentWeight(d, weights[1]);
  layers::DBuf gate(d, DType::kBF16, {2, 8}), up(d, DType::kBF16, {2, 8});
  layers::DBuf expected(d, DType::kBF16, {2, 8});
  vt::MatmulBT(queue, gate.t(), x, wg);
  vt::MatmulBT(queue, up.t(), x, wu);
  vt::MoeSiluMul(queue, expected.t(), gate.t(), up.t());
  CHECK(std::memcmp(actual.t().data, expected.t().data, 32) == 0);
  CHECK(std::string(method.Name()) == "dense-shard-gate-up");
}

TEST_CASE("dense malformed descriptors refuse unsafe contracts") {
  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  auto& backend = vt::GetBackend(vt::DeviceType::kCPU);
  layers::Dev d{backend, queue};
  auto input = MakeBf16({1, 64}, 23);
  const auto x = layers::ResidentWeight(d, input);
  auto owner = MakeQuantShard(DType::kQ4_0, 8, 64, 7);
  const auto w = layers::ResidentWeight(d, owner);
  const vt::Tensor weights[] = {w, w, w};
  layers::DBuf a(d, DType::kBF16, {1, 8}), b(d, DType::kBF16, {1, 8});
  layers::DBuf c(d, DType::kBF16, {1, 8}), out(d, DType::kBF16, {1, 8});
  vt::Tensor products[] = {a.t(), b.t(), c.t()};
  const auto refuses = [&](const vt::MergedGemmGroup& desc, vt::Tensor* dst,
                           const vt::Tensor* src, int count, vt::Tensor* epilogue,
                           const char* diagnostic) {
    CHECK_THROWS_WITH_AS(vt::MergedGemmDense(queue, desc, dst, x, src, count, epilogue),
                         doctest::Contains(diagnostic), std::runtime_error);
  };
  const char* count_error = "MergedGemmDense: descriptor and shard count must match";
  refuses(vt::kDenseQkvShards, products, weights, 2, nullptr, count_error);
  auto single = vt::kDenseQkvShards;
  single.arity = 1;
  refuses(single, products, weights, 1, nullptr, count_error);
  // Zero contractions keep a removed array guard from dereferencing null.
  auto empty = vt::kDenseQkvShards;
  empty.arity = 0;
  refuses(empty, nullptr, weights, 0, nullptr, count_error);
  refuses(empty, products, nullptr, 0, nullptr, count_error);
  auto fused = vt::kDenseQkvShards;
  fused.fast_op = static_cast<int>(vt::OpId::kMoeGateUpSwiGLUGrouped);
  refuses(fused, products, weights, 3, nullptr,
          "MergedGemmDense: no dense-shard fused provider is registered");
  auto three_way_swiglu = vt::kDenseQkvShards;
  three_way_swiglu.epilogue = vt::MergedEpilogue::kSiluMulClamp;
  refuses(three_way_swiglu, products, weights, 3, &out.t(),
          "MergedGemmDense: unsupported dense epilogue");
  auto unknown = vt::kDenseQkvShards;
  unknown.epilogue = static_cast<vt::MergedEpilogue>(255);
  refuses(unknown, products, weights, 3, nullptr,
          "MergedGemmDense: unsupported dense epilogue");
  refuses(vt::kDenseQkvShards, products, weights, 3, &out.t(),
          "MergedGemmDense: SwiGLU alone requires an epilogue output");
}

TEST_CASE("dense SwiGLU requires its output before contraction") {
  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  // The descriptor rejects the missing output before touching tensor storage.
  vt::Tensor products[2]{};
  vt::Tensor weights[2]{};
  const vt::Tensor input{};
  CHECK_THROWS_WITH_AS(
      vt::MergedGemmDense(queue, vt::kDenseGateUpShards, products, input, weights, 2),
      doctest::Contains("MergedGemmDense: SwiGLU alone requires an epilogue output"),
      std::runtime_error);
}

TEST_CASE("dense null shard arrays are rejected before tensor access") {
  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vt::Tensor products[2]{};
  vt::Tensor weights[2]{};
  const vt::Tensor input{};
  auto plain = vt::kDenseQkvShards;
  plain.arity = 2;
  const auto expected = doctest::Contains("MergedGemmDense: descriptor and shard count must match");
  CHECK_THROWS_WITH_AS(vt::MergedGemmDense(queue, plain, nullptr, input, weights, 2),
                       expected, std::runtime_error);
  CHECK_THROWS_WITH_AS(vt::MergedGemmDense(queue, plain, products, input, nullptr, 2),
                       expected, std::runtime_error);
}
