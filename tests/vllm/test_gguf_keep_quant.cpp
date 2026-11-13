// GGUF keep-quantized loader — work rows L2 (quant residency) and L3 (routing
// policy + VT_CPU_REF oracle switch) of
// .agents/specs/gguf-keep-quant-loader.md.
//
// Upstream idea being gated (llama.cpp @ 237ad9b96): a weight KEEPS the
// ggml_type it has in the file (src/llama-model-loader.cpp:1047 create_tensor,
// :1385 load_data_for) and rows are whole blocks (ggml_row_size, ggml/src/
// ggml.c). Ported test shape follows llama.cpp tests/gguf-model-data.cpp's
// harness idea — round-trip resident blocks through the decoder and demand the
// SAME bytes the direct-from-file decode produces.
//
// THE GATE (spec gate 1) IS BYTE-IDENTITY, PROVEN PER ENCODING. Keeping a
// weight in its blocks is only safe if dequantizing the resident blocks is
// indistinguishable from today's load-time expansion — not "close", identical.
// Each of the six executable encodings gets its own TEST_CASE so a failure
// names the encoding that broke.
//
// Block bytes are generated PSEUDO-RANDOMLY rather than from an encoder: the
// residency property must hold for every bit pattern a file can contain, and
// only Q8_0/Q8_K have a `from_float` in this project anyway. The one
// constraint applied is that every f16 scale field decodes finite — all six
// block structs place their f16 fields at EVEN offsets and all six block sizes
// are even, so clearing bit 6 of every odd-indexed byte (the f16 exponent MSB)
// bounds every scale away from Inf/NaN without otherwise restricting the data.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <cmath>

#include "gguf_builder.h"
#include "capi/rocm_quant_gather_fixture.h"
#include "vllm/config/weight_residency.h"
#include "vllm/model_executor/device_placement.h"
#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#ifdef VLLM_CPP_HIP
#include "vt/rocm/rocm_runtime.h"
#endif
#include "support/test_env.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/op_provider.h"  // vt::OpRegistered — the gather gate is a registry query
#include "vt/quant.h"

using gguf_test::F32Kv;
using gguf_test::GgufModelBuilder;
using gguf_test::StrKv;
using gguf_test::TempFile;
using gguf_test::U32Kv;
using vllm::GgufLoadPolicy;
using vllm::GgufResidency;
using vllm::GgufTensorRole;
using vllm::KeepQuantDType;
using vllm::OwnGgufQuantBlocks;
using vllm::QuantRepackForDevice;
using vllm::RouteGgufTensor;

namespace {

// ggml type ids (ggml/include/ggml.h:390-432).
constexpr uint32_t kF32 = 0, kF16 = 1, kQ4_0 = 2, kQ5_0 = 6, kQ8_0 = 8,
                   kQ2_K = 10, kQ3_K = 11, kQ4_K = 12, kQ5_K = 13, kQ6_K = 14,
                   kQ8_K = 15, kIQ2_XXS = 16, kIQ2_XS = 17, kIQ3_XXS = 18,
                   kIQ1_S = 19, kIQ4_NL = 20, kIQ3_S = 21, kIQ2_S = 22,
                   kIQ4_XS = 23, kIQ1_M = 29, kBF16 = 30, kMXFP4 = 39,
                   kQ1_0 = 41, kIQ1_XXXS = 66;

// Every executable weight encoding, with a K that is a whole number of blocks.
struct Encoding {
  uint32_t ggml_type;
  const char* name;
  int64_t k;
};
const Encoding kEncodings[] = {
    {kQ4_0, "Q4_0", 64},  {kQ8_0, "Q8_0", 64},  {kQ3_K, "Q3_K", 256},
    {kQ4_K, "Q4_K", 256}, {kQ5_K, "Q5_K", 256}, {kQ6_K, "Q6_K", 256},
};

// xorshift32 — deterministic across platforms and compilers, so a failure is
// reproducible from the seed alone.
uint32_t NextRand(uint32_t* s) {
  uint32_t x = *s;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *s = x;
  return x;
}

// `nbytes` of block payload; see the header comment for the odd-byte mask.
std::string RandomBlockBytes(size_t nbytes, uint32_t seed) {
  uint32_t s = seed | 1U;
  std::string out(nbytes, '\0');
  for (size_t i = 0; i < nbytes; ++i) {
    uint8_t b = static_cast<uint8_t>(NextRand(&s) & 0xFF);
    if ((i % 2) == 1) b &= 0xBF;  // clear the f16 exponent MSB -> finite
    out[i] = static_cast<char>(b);
  }
  return out;
}

// A GGUF holding exactly one block-quantized tensor with torch shape
// [n, k] (ggml dims are the reverse).
std::string OneTensorGguf(uint32_t ggml_type, int64_t n, int64_t k,
                          const std::string& blocks) {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "test"));
  b.AddTensor("w", {static_cast<uint64_t>(k), static_cast<uint64_t>(n)},
              ggml_type, blocks);
  return b.Build();
}

size_t BlockBytesFor(uint32_t ggml_type, int64_t numel) {
  vt::DType dt = vt::DType::kF32;
  REQUIRE(KeepQuantDType(ggml_type, &dt));
  return vt::RowSizeBytes(dt, numel);
}

// A policy with keep-quant forced ON (what G4 made the default wherever the
// running device can execute kMatmulBTQuant). `expand_nk` is left OFF here so
// the pre-existing L2/L3 losslessness cases keep comparing against the
// historical Matmul-B expansion; the orientation switch has its own cases.
GgufLoadPolicy KeepQuantOn() {
  GgufLoadPolicy p;
  p.keep_quant = true;
  return p;
}

}  // namespace

// ===========================================================================
// L2 gate 1 — LOSSLESSNESS, one case per encoding.
// ===========================================================================

namespace {

void CheckLossless(const Encoding& enc, int64_t n) {
  const int64_t numel = n * enc.k;
  const size_t nbytes = BlockBytesFor(enc.ggml_type, numel);
  const std::string blocks = RandomBlockBytes(nbytes, enc.ggml_type * 7919U + 1);
  const TempFile f(OneTensorGguf(enc.ggml_type, n, enc.k, blocks));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::GgufTensorInfo& t = g.Get("w");
  REQUIRE(t.nbytes == nbytes);

  // Residency: the raw ggml blocks, copied verbatim, [N,K] with nk = true.
  const vllm::OwnedTensor res = OwnGgufQuantBlocks(t, n, enc.k);
  vt::DType dt = vt::DType::kF32;
  REQUIRE(KeepQuantDType(enc.ggml_type, &dt));
  CHECK(res.dtype == dt);
  CHECK(res.rank == 2);
  CHECK(res.shape[0] == n);
  CHECK(res.shape[1] == enc.k);
  CHECK(res.nk == true);  // GGUF [out,in] IS the MatmulBT orientation
  REQUIRE(res.bytes.size() == nbytes);
  // Byte-for-byte the file's blocks — residency copies, it does not transform.
  CHECK(std::memcmp(res.bytes.data(), t.data, nbytes) == 0);

  // THE GATE: dequantizing the resident blocks is byte-identical to today's
  // load-time expansion straight from the mapped file, in BOTH target dtypes.
  const std::vector<float> from_file =
      vllm::DequantGgufRowToF32(enc.ggml_type, t.data, numel);
  const std::vector<float> from_resident =
      vllm::DequantGgufRowToF32(enc.ggml_type, res.bytes.data(), numel);
  REQUIRE(from_file.size() == static_cast<size_t>(numel));
  REQUIRE(from_resident.size() == from_file.size());
  CHECK(std::memcmp(from_resident.data(), from_file.data(),
                    from_file.size() * sizeof(float)) == 0);

  const std::vector<uint16_t> bf_file =
      vllm::DequantGgufRowToBf16(enc.ggml_type, t.data, numel);
  const std::vector<uint16_t> bf_resident =
      vllm::DequantGgufRowToBf16(enc.ggml_type, res.bytes.data(), numel);
  CHECK(std::memcmp(bf_resident.data(), bf_file.data(),
                    bf_file.size() * sizeof(uint16_t)) == 0);

  // Sanity that the random bytes decoded to real numbers (the odd-byte mask):
  // a losslessness gate over all-NaN data would prove much less.
  for (float v : from_file) REQUIRE(std::isfinite(v));
}

}  // namespace

TEST_CASE("keep-quant residency is lossless: Q4_0") {
  CheckLossless(kEncodings[0], /*n=*/5);
}
TEST_CASE("keep-quant residency is lossless: Q8_0") {
  CheckLossless(kEncodings[1], /*n=*/5);
}
TEST_CASE("keep-quant residency is lossless: Q3_K") {
  CheckLossless(kEncodings[2], /*n=*/3);
}
TEST_CASE("keep-quant residency is lossless: Q4_K") {
  CheckLossless(kEncodings[3], /*n=*/3);
}
TEST_CASE("keep-quant residency is lossless: Q5_K") {
  CheckLossless(kEncodings[4], /*n=*/3);
}
TEST_CASE("keep-quant residency is lossless: Q6_K") {
  CheckLossless(kEncodings[5], /*n=*/3);
}

TEST_CASE("keep-quant expert split is lossless per expert") {
  // A stacked [E, out, in] expert tensor: each expert is a whole number of
  // ROWS, hence whole blocks, so the split must be a pure byte range.
  const int64_t e_count = 4, out_dim = 3, in_dim = 256;
  const int64_t numel = e_count * out_dim * in_dim;
  const size_t nbytes = BlockBytesFor(kQ4_K, numel);
  const std::string blocks = RandomBlockBytes(nbytes, 4242);

  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "test"));
  b.AddTensor("exps", {static_cast<uint64_t>(in_dim),
                       static_cast<uint64_t>(out_dim),
                       static_cast<uint64_t>(e_count)},
              kQ4_K, blocks);
  const TempFile f(b.Build());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::GgufTensorInfo& t = g.Get("exps");
  REQUIRE(t.shape.size() == 3);
  REQUIRE(t.shape[0] == e_count);

  const std::vector<float> whole =
      vllm::DequantGgufRowToF32(kQ4_K, t.data, numel);
  const int64_t per = out_dim * in_dim;
  for (int64_t e = 0; e < e_count; ++e) {
    CAPTURE(e);
    const vllm::OwnedTensor res =
        OwnGgufQuantBlocks(t, out_dim, in_dim, /*row_offset=*/e * out_dim);
    const std::vector<float> slice =
        vllm::DequantGgufRowToF32(kQ4_K, res.bytes.data(), per);
    CHECK(std::memcmp(slice.data(), whole.data() + e * per,
                      static_cast<size_t>(per) * sizeof(float)) == 0);
  }
}

TEST_CASE("keep-quant routing respects the RUNNING DEVICE's format set (review #523)") {
  // Registering kMatmulBTQuant flips keep-quant loader-wide via the boolean
  // GgufQuantComputeAvailable(), but a device's kernel set can be narrower
  // than the CPU admission list. The provider-specific case below pins ROCm's
  // complete set; with no CPU fallback tier on a discrete card, an unsupported
  // format that flipped to keep-quant would throw at FORWARD time with the
  // model fully resident. The loader must keep the pre-existing expand_bf16
  // residency for those formats instead.
  //
  // ENG-GGUF-RESIDENCY-RESOLVED-DEVICE: the device is now a PARAMETER, so these
  // arms run everywhere. They used to read `CurrentPlatform()` and return early
  // with a MESSAGE on any host that is not gfx1100 — which is every host this
  // suite has ever run on, so the ROCm format set was asserted nowhere. The
  // whole point of removing the probe is that a device's routing is checkable
  // from a host that is not that device.
  const std::vector<int64_t> shape = {4, 256};  // [out, in]: K = 256 elems, whole blocks
  const auto route = [&](uint32_t ty) {
    return RouteGgufTensor(/*keep_quant=*/true, /*keep_f16=*/true,
                           /*nvfp4_fp4=*/false, /*cpu_ref=*/false,
                           GgufTensorRole::kMatmulWeight, ty, shape,
                           vt::DeviceType::kROCM);
  };
  // The supported set keeps quant residency (ggml type ids per the constants
  // at the top of this file).
  CHECK(route(kQ4_0) == GgufResidency::kExpandBf16);  // unsupported -> expand
  CHECK(route(kQ8_0) == GgufResidency::kKeepQuant);
  CHECK(route(kQ4_K) == GgufResidency::kKeepQuant);
  CHECK(route(kQ5_K) == GgufResidency::kKeepQuant);
  CHECK(route(kQ6_K) == GgufResidency::kKeepQuant);
  CHECK(route(kQ2_K) == GgufResidency::kKeepQuant);
  // keep-f16 must be OFF on ROCm: MatmulBTKernelRocm accepts bf16/f32 only.
  // Asked of a ROCm-resolved policy, not of this process's own.
  const GgufLoadPolicy pol = GgufLoadPolicy::FromEnv(vt::DeviceType::kROCM);
  CHECK(!pol.keep_f16);
}

TEST_CASE("ROCm loader admits every format implemented by the quant-dot provider") {
  struct ProviderEncoding {
    uint32_t ggml_type;
    const char* name;
  };
  const ProviderEncoding implemented[] = {
      {kIQ2_XXS, "IQ2_XXS"}, {kIQ3_XXS, "IQ3_XXS"},
      {kQ2_K, "Q2_K"},       {kQ3_K, "Q3_K"},
      {kIQ2_S, "IQ2_S"},     {kIQ1_S, "IQ1_S"},
      {kIQ1_XXXS, "IQ1_XXXS"},
      // IQ4_XS joined the provider's WType table with
      // KERNEL-QUANT-CIQ-GEMM-ROCM-IQUANT (#1940, pull request #3029). This
      // case gates "every format the provider implements", so the entry moves
      // up from the expansion list below rather than being added beside it.
      {kIQ4_XS, "IQ4_XS"},
  };
  GgufLoadPolicy rocm = KeepQuantOn();
  rocm.device = vt::DeviceType::kROCM;
  for (const ProviderEncoding& encoding : implemented) {
    CAPTURE(encoding.name);
    vllm::GgufTensorInfo tensor;
    tensor.name = "blk.0.attn_q.weight";
    tensor.ggml_type = encoding.ggml_type;
    tensor.shape = {8, 256};
    CHECK(rocm.Route(tensor, GgufTensorRole::kMatmulWeight) ==
          GgufResidency::kKeepQuant);

    tensor.name = "blk.0.ffn_gate_exps.weight";
    tensor.shape = {2, 8, 256};
    CHECK(rocm.Route(tensor, GgufTensorRole::kStackedExpertWeight) ==
          GgufResidency::kKeepQuant);
  }

  // These formats have no arm in rocm_quant_dot.hip, so the provider itself
  // never admits them. IQ4_NL is deliberately NOT in this list: it used to be,
  // but QUANT-GGUF-IQ4_NL landed after this case did and keeps IQ4_NL on ROCm
  // through the GDN provider's own dedicated `IQ4NLGemmK` arm instead (Q8_0
  // activation, 32-element block, not this provider's Q8_K/256-element
  // shape). `rocm.Route` sees both providers, so it correctly answers
  // `kKeepQuant` for IQ4_NL now, and asserting `kExpandBf16` here went stale
  // the moment that row landed -- this loader-level case cannot see which
  // provider does the keeping, only whether the tensor keeps at all.
  for (uint32_t ggml_type : {kQ4_0, kIQ2_XS, kMXFP4}) {
    CAPTURE(ggml_type);
    vllm::GgufTensorInfo tensor;
    tensor.name = "blk.0.attn_q.weight";
    tensor.ggml_type = ggml_type;
    tensor.shape = {8, 256};
    CHECK(rocm.Route(tensor, GgufTensorRole::kMatmulWeight) ==
          GgufResidency::kExpandBf16);
  }
}

TEST_CASE("keep-quant routing on TENSTORRENT admits exactly the registered decodes (KEEPQUANT W3)") {
  // The P150 is discrete with no CPU fallback tier, so the same #523 shape as
  // ROCm applies: the TT arm may admit only what tenstorrent_ops.cpp has a
  // registered decode for. W1 registered the Q4_K decode, W2 its dot; W3
  // generalizes the decode chain to Q5_K/Q6_K/Q8_0 (the q4km vehicle's actual
  // histogram: token_embd Q6_K, attn_qkv/ssm_out Q5_K, ssm_alpha/ssm_beta
  // Q8_0) and the arm widens to exactly those four IN THE SAME CHANGE — the
  // routing test still reds any widening PAST the registered set: kQ4_0 has
  // no TT arm at all, and kQ2_K stays owed. QUANT-GGUF-IQ-TENSTORRENT
  // wave 1 adds kIQ3_XXS (the int8-dot arm's enc_sel 4) in the same change as
  // its kernel; wave 2 adds kIQ2_XXS/kIQ2_S (enc_sel 5/6); wave 3 adds kQ3_K
  // (enc_sel 7). Admitting an encoding
  // without its kernel throws at first forward with the model resident, the
  // exact failure this predicate exists to prevent. The
  // tenstorrent-gsq-keepquant row's wave 1 adds kIQ3_S (enc_sel 8) the same
  // way.
  const std::vector<int64_t> shape = {4, 256};  // [out, in]: whole blocks
  const auto route = [&](uint32_t ty) {
    return RouteGgufTensor(/*keep_quant=*/true, /*keep_f16=*/true,
                           /*nvfp4_fp4=*/false, /*cpu_ref=*/false,
                           GgufTensorRole::kMatmulWeight, ty, shape,
                           vt::DeviceType::kTENSTORRENT);
  };
  CHECK(route(kQ4_K) == GgufResidency::kKeepQuant);
  CHECK(route(kQ8_0) == GgufResidency::kKeepQuant);  // W3 decode set
  CHECK(route(kQ5_K) == GgufResidency::kKeepQuant);  // W3 decode set
  CHECK(route(kQ6_K) == GgufResidency::kKeepQuant);  // W3 decode set
  CHECK(route(kIQ3_XXS) == GgufResidency::kKeepQuant);  // IQ3_XXS int8-dot arm
  // QUANT-GGUF-IQ-TENSTORRENT wave 2: IQ2_XXS (enc_sel 5) and IQ2_S
  // (enc_sel 6) join the int8-dot set in the same change as their kernels.
  CHECK(route(kIQ2_XXS) == GgufResidency::kKeepQuant);
  CHECK(route(kIQ2_S) == GgufResidency::kKeepQuant);
  CHECK(route(kQ4_0) == GgufResidency::kExpandBf16);  // no TT arm at all
  // tenstorrent-gsq-keepquant wave 4: this pin — "kQ2_K stays owed" — flips
  // to kKeepQuant in the same change as the kq_vec_dot_q2_k_q8_K kernel
  // (enc_sel 11); it redded against the widened predicate before this line
  // moved.
  CHECK(route(kQ2_K) == GgufResidency::kKeepQuant);
  // QUANT-GGUF-IQ-TENSTORRENT wave 3: Q3_K (enc_sel 7, the min-term
  // K-quant) joins the int8-dot set in the same change as its kernel
  // kq_vec_dot_q3_k_q8_K — this check flipped FROM kExpandBf16 and redded
  // before the predicate widened.
  CHECK(route(kQ3_K) == GgufResidency::kKeepQuant);
  // tenstorrent-gsq-keepquant wave 1: IQ3_S (enc_sel 8, the largest census
  // gap at 97 of the 198 missing tensors) joins the int8-dot set in the
  // same change as its kernel kq_vec_dot_iq3_s_q8_K — this check flipped
  // FROM kExpandBf16 and redded before the predicate widened.
  CHECK(route(kIQ3_S) == GgufResidency::kKeepQuant);
  // tenstorrent-gsq-keepquant wave 2: IQ4_XS (enc_sel 9, 33 census tensors,
  // ssm_out + attn) joins the int8-dot set in the same change as its kernel
  // kq_vec_dot_iq4_xs_q8_K — this check flipped FROM kExpandBf16 and redded
  // before the predicate widened.
  CHECK(route(kIQ4_XS) == GgufResidency::kKeepQuant);
  // tenstorrent-gsq-keepquant wave 3: IQ2_XS (enc_sel 10, 32 census tensors,
  // ffn) joins the int8-dot set in the same change as its kernel
  // kq_vec_dot_iq2_xs_q8_K — this check flipped FROM kExpandBf16 and redded
  // before the predicate widened.
  CHECK(route(kIQ2_XS) == GgufResidency::kKeepQuant);
  // tenstorrent-gsq-keepquant wave 4: Q2_K (enc_sel 11, 28 census tensors,
  // ffn + embd) joins the int8-dot set in the same change as its kernel
  // kq_vec_dot_q2_k_q8_K — this check flipped FROM kExpandBf16 and redded
  // before the predicate widened.
  CHECK(route(kQ2_K) == GgufResidency::kKeepQuant);
  // tenstorrent-gsq-keepquant wave 5: IQ1_S (enc_sel 12, 4 census tensors)
  // and IQ1_M (enc_sel 13, 4) join the int8-dot set in the same change as
  // their kernels kq_vec_dot_iq1_s_q8_K / kq_vec_dot_iq1_m_q8_K — these
  // checks redded before the predicate widened (the pins are NEW here: no
  // prior row pinned the IQ1 pair, so the widened predicate alone reds
  // nothing; the device sweep carries the red).
  CHECK(route(kIQ1_S) == GgufResidency::kKeepQuant);
  CHECK(route(kIQ1_M) == GgufResidency::kKeepQuant);
  // TT-KEEPQUANT-Q8K-WEIGHTS: Q8_K joins the GEMM set — the weights arm the
  // UD-Q8_K_XL checkpoint needs. Q8_K has NO vec_dot anywhere in the tree
  // (cpu_quant_dot.cpp's selector returns nullptr), so it keeps blocks and
  // serves the DEQUANT-COMPOSITE arm (the tree-standard for Q8_K weights,
  // staged at 80 padded words per 292-B block_q8_K). This check redded
  // against the unchanged predicate (kExpandBf16) before the loader widened.
  CHECK(route(kQ8_K) == GgufResidency::kKeepQuant);
  // The loader boolean flips only when the op is registered, so a host with a
  // P150 resolves keep-quant on by default; without the card the default arm
  // stays false and the load is unchanged.
  if (vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT)) {
    const GgufLoadPolicy tt = GgufLoadPolicy::FromEnv(vt::DeviceType::kTENSTORRENT);
    CHECK(tt.keep_quant);
  } else {
    MESSAGE("kMatmulBTQuant not registered on this host; loader boolean unchecked");
  }
}

TEST_CASE("quant_repack is decided WITH the resolved device (#2406)") {
  // THE DEFECT. `quant_repack` selects the CIQ G7 load-time permutation of a
  // Q8_0 weight into the ARM i8mm `block_q8_0x4` interleave. Only the CPU
  // `MatmulBTKernel` reads that layout; `grep -c repacked
  // src/vt/cuda/cuda_quant_dot.cu` is 0. The flag was resolved from
  // `vt::cpu::QuantRepackActive()` alone — a pure HOST-ISA probe — so on an
  // aarch64 i8mm host with a GPU (dgx GB10 and Jetson Thor, both fleet devices)
  // a `--device cuda` load repacked the weight and staged it to the card. Its
  // sibling `elem_kn_repack` has carried `dev == kCPU` for exactly this reason
  // since it landed, and says so in its own comment.
  //
  // WHY THIS IS ASSERTED OVER A PREDICATE AND NOT OVER `FromEnv`.
  // `QuantRepackActive()` compiles to a literal `false` on every non-aarch64
  // target (`src/vt/cpu/cpu_quant_repack_arm.cpp`). An assertion that reached
  // the decision only through `FromEnv` would therefore read `quant_repack ==
  // false` on x86 CI whether the device term existed or not — a mute switch
  // wearing a gate's clothes, and one that would have passed on every day this
  // defect was live. Passing the ISA answer in is the same move
  // `RouteGgufTensor` made for `dev`, for the same stated reason: a decision
  // that takes its inputs is checkable from a host that is not the one it
  // decides for.
  //
  // THE DISCRIMINATING PAIR is the first two rows. They differ in `dev` alone.
  CHECK(QuantRepackForDevice(/*keep_quant=*/true, /*cpu_ref=*/false,
                             /*host_repack_active=*/true,
                             vt::DeviceType::kCPU) == true);
  CHECK(QuantRepackForDevice(/*keep_quant=*/true, /*cpu_ref=*/false,
                             /*host_repack_active=*/true,
                             vt::DeviceType::kCUDA) == false);
  // Every other accelerator answers the same way: the gate is "is this the CPU",
  // not "is this CUDA". A `dev != kCUDA` spelling passes the row above and fails
  // these two.
  CHECK(QuantRepackForDevice(true, false, true, vt::DeviceType::kROCM) == false);
  CHECK(QuantRepackForDevice(true, false, true, vt::DeviceType::kXPU) == false);

  // The three pre-existing terms are unchanged, and each still vetoes alone.
  CHECK(QuantRepackForDevice(/*keep_quant=*/false, false, true,
                             vt::DeviceType::kCPU) == false);
  CHECK(QuantRepackForDevice(true, /*cpu_ref=*/true, true,
                             vt::DeviceType::kCPU) == false);
  CHECK(QuantRepackForDevice(true, false, /*host_repack_active=*/false,
                             vt::DeviceType::kCPU) == false);

  // AND THE LOADER USES IT. The predicate above is the rule; this is the wire.
  //
  // THIS HALF IS VACUOUS ON x86 AND THAT IS SAID RATHER THAN HIDDEN.
  // `QuantRepackActive()` is a compile-time `false` off aarch64, so no value of
  // `dev` can make `FromEnv` answer `true` here and no x86 run can tell the
  // wired predicate from the expression it replaced. `VT_GGUF_KEEP_QUANT=1`
  // removes the OTHER two vetoes — without it `keep_quant` is already false for
  // a device with no quant GEMM registered, and this would assert nothing on
  // any host. With it, on an aarch64 i8mm box (dgx, thor, orin — every box
  // where the defect was reachable) the old expression answers `true` and this
  // case reds. The truth table above is what gates the rule everywhere else.
  ::setenv("VT_GGUF_KEEP_QUANT", "1", 1);
  CHECK(!GgufLoadPolicy::FromEnv(vt::DeviceType::kCUDA).quant_repack);
  CHECK(!GgufLoadPolicy::FromEnv(vt::DeviceType::kROCM).quant_repack);
  ::unsetenv("VT_GGUF_KEEP_QUANT");
  // On the CPU the flag is exactly what the host ISA says, which is the
  // property that makes this a device gate and not a disablement. On an i8mm
  // box this reads `true`, which is the same answer the loader gave before
  // #2406 — the gate must not have turned the transform off where it works.
  {
    const GgufLoadPolicy cpu = GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU);
    CHECK(cpu.quant_repack ==
          QuantRepackForDevice(cpu.keep_quant, cpu.cpu_ref,
                               vt::cpu::QuantRepackActive(),
                               vt::DeviceType::kCPU));
  }
}

TEST_CASE("keep-quant residency refuses ragged K and out-of-span slices") {
  const int64_t n = 2, k = 64;
  const size_t nbytes = BlockBytesFor(kQ8_0, n * k);
  const TempFile f(OneTensorGguf(kQ8_0, n, k, RandomBlockBytes(nbytes, 9)));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::GgufTensorInfo& t = g.Get("w");

  // K must be a whole number of blocks (ggml_row_size's precondition).
  CHECK_THROWS(OwnGgufQuantBlocks(t, n, /*k=*/33));
  // Rows beyond the validated tensor span must never be handed out.
  CHECK_THROWS(OwnGgufQuantBlocks(t, /*n=*/3, k));
  CHECK_THROWS(OwnGgufQuantBlocks(t, n, k, /*row_offset=*/1));
  // A ragged-K weight is not merely refused at residency: the POLICY routes it
  // to expansion, so the loader never reaches the throw.
  CHECK(RouteGgufTensor(true, false, false, false, GgufTensorRole::kMatmulWeight,
                        kQ8_0, {n, 33}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
}

// ===========================================================================
// L3 — the routing policy table.
// ===========================================================================

TEST_CASE("KeepQuantDType covers the executable encodings") {
  vt::DType dt = vt::DType::kF32;
  for (const Encoding& e : kEncodings) {
    CAPTURE(e.name);
    CHECK(KeepQuantDType(e.ggml_type, &dt));
    CHECK(vt::cpu::HasQuantDotKernel(dt));
  }
  // IQ2_S (22, Q8_K-activation) and MXFP4 (39, Q8_0-activation) are the UD-IQ2_M
  // routed-expert encodings; both gained a keep-quant vec_dot, so the loader must
  // keep them COMPRESSED (never expand-to-bf16 -> OOM).
  //
  // IQ2_XS (17) and IQ4_XS (23) join them with QUANT-GGUF-IQ-VECDOT (#2247).
  // #2245 gave them a DECODER only, which left them keep-capable on the gather
  // arm and expand-only on the GEMM arm; the two `vec_dot` rows this row adds
  // close that, and 53.33 GiB of the staged GLM-5.3-Flash UD-Q2_K_XL artifact
  // stops expanding to 369.00 GiB of bf16 because of it.
  for (uint32_t id : {kIQ2_S, kMXFP4, kIQ2_XS, kIQ4_XS}) {
    CAPTURE(id);
    CHECK(KeepQuantDType(id, &dt));
    CHECK(vt::cpu::HasQuantDotKernel(dt));
  }
  // Unquantized file types and the encodings that DECODE but have no
  // keep-quant serve are NOT keep-quant capable. Q8_K left this list with
  // TT-KEEPQUANT-Q8K-WEIGHTS (it serves the dequant composite — no vec_dot
  // appeared, but a GEMM arm is a GEMM arm), leaving it the FIRST list in
  // a while with no decoder member; Q1_0 (41) carries the OLDEST failure
  // shape — the reader tabulates it and nothing in this tree decodes it.
  for (uint32_t id : {kF32, kF16, kBF16, kQ1_0}) {
    CAPTURE(id);
    CHECK_FALSE(KeepQuantDType(id, &dt));
  }
  // Q8_K keeps blocks through the composite: admitted WITHOUT a vec_dot row,
  // which is exactly what makes it the composite arm and not a dot arm.
  CHECK(KeepQuantDType(kQ8_K, &dt));
  CHECK_FALSE(vt::cpu::HasQuantDotKernel(dt));
}

namespace {

// One row of a checkpoint census: an encoding, how many tensor records carry
// it, and whether the loader is expected to keep it in blocks or expand it.
// `keep_quant == false` is a real entry, not an omission: F32 is 838 of the
// 1702 records in both target checkpoints, it is served by the expansion path,
// and leaving it out is what let an incomplete list call itself total.
struct CensusRow {
  uint32_t ggml_type;
  const char* name;
  int tensors;
  bool keep_quant;
};

// `split.tensors.count` from the shard headers, the same number on both
// checkpoints. The list below must add up to exactly this.
constexpr int kQwen38DeclaredTensors = 1702;

// unsloth/Qwen3.8-2.4T-A95B-GGUF @ 567d3e6ac2, quant UD-IQ1_S, all 12 shards.
constexpr CensusRow kUdIq1sCensus[] = {
    {19, "iq1_s", 276, true},  // 96.92 % of params
    {13, "q5_k", 420, true},   {10, "q2_k", 3, true},
    {14, "q6_k", 162, true},   {12, "q4_k", 2, true},
    {8, "q8_0", 1, true},      {0, "f32", 838, false},
};

// The same census over UD-Q1_0, all 10 shards. Identical apart from the expert
// encoding: ggml 66 where UD-IQ1_S has ggml 19.
constexpr CensusRow kUdQ10Census[] = {
    {66, "iq1_xxxs", 276, true},  // 96.92 % of params
    {13, "q5_k", 420, true},      {10, "q2_k", 3, true},
    {14, "q6_k", 162, true},      {12, "q4_k", 2, true},
    {8, "q8_0", 1, true},         {0, "f32", 838, false},
};

template <size_t N>
void CheckCheckpointCensus(const CensusRow (&census)[N], const char* quant) {
  CAPTURE(std::string(quant));
  vt::DType dt = vt::DType::kF32;
  int counted = 0;
  for (const CensusRow& p : census) {
    CAPTURE(p.name);
    CAPTURE(p.ggml_type);
    CAPTURE(p.tensors);
    counted += p.tensors;
    // Assert the ROUTING both ways. A row expected to keep its blocks must
    // have a dot kernel, and a row expected to expand must NOT silently be
    // reclassified as keep-quant by a later change.
    if (p.keep_quant) {
      CHECK(KeepQuantDType(p.ggml_type, &dt));
      CHECK(vt::cpu::HasQuantDotKernel(dt));
    } else {
      CHECK_FALSE(KeepQuantDType(p.ggml_type, &dt));
    }
  }
  // The list accounts for every declared record, so "this checkpoint needs
  // nothing else" is a measurement rather than a claim.
  CAPTURE(counted);
  CHECK(counted == kQwen38DeclaredTensors);
}

}  // namespace

TEST_CASE("every encoding in the Qwen3.8-2.4T UD-IQ1_S checkpoint decodes") {
  // Census of unsloth/Qwen3.8-2.4T-A95B-GGUF @ 567d3e6ac2, quant UD-IQ1_S,
  // taken by parsing the tensor header of all 12 shards: 1702 tensor records
  // against the 1702 declared in `split.tensors.count`, so this list is TOTAL
  // rather than sampled. See the target-checkpoint census section of
  // .agents/specs/expert-streaming.md.
  //
  // This is the row's real precondition: ENG-EXPERT-STREAM streams routed
  // expert slices off NVMe, and a slice it can address but cannot DECODE moves
  // bytes for nothing. IQ1_S alone is 96.92 % of the parameters (the
  // ffn_{down,gate,up}_exps of all 92 non-MTP layers), so an unsupported
  // encoding here is a refusal to run the model, never a slow path.
  //
  // The list has to SUM to the declared record count, and this case asserts
  // that it does. An earlier revision claimed total coverage while enumerating
  // six of the seven encodings: F32 was left out and the counts summed to 864,
  // not 1702. A census that cannot say how many records it accounted for has
  // not reported a census, and a list that is short by 838 tensors while
  // calling itself TOTAL is the more misleading of the two states.
  CheckCheckpointCensus(kUdIq1sCensus, "UD-IQ1_S");
}

TEST_CASE("every encoding in the Qwen3.8-2.4T UD-Q1_0 checkpoint decodes") {
  // Same census method and the same revision as the UD-IQ1_S case above, over
  // all 10 shards: 1702 tensor records against the 1702 declared. UD-Q1_0 is
  // structurally IDENTICAL to UD-IQ1_S apart from the expert encoding, which is
  // ggml 66 (IQ1_XXXS, 1.1875 bpw) instead of ggml 19 (IQ1_S, 1.5625 bpw). The
  // other six encodings and their tensor counts match exactly.
  //
  // Type 66 is defined by NO upstream llama.cpp. It comes from the pinned fork
  // oracle `llama-cpp-unsloth` (.agents/oracles/llama-cpp-unsloth.md), which is
  // admitted for this encoding family alone and is `gateable = no` until #933
  // measures it.
  CheckCheckpointCensus(kUdQ10Census, "UD-Q1_0");
}

TEST_CASE("ROCm IQ3_XXS loader admission keeps matrix and expert blocks") {
  // Direct GEMM tests bypass DeviceKeepQuantSupported. Enter the loader route
  // so deleting IQ3_XXS admission cannot silently restore bf16 expansion.
  constexpr uint32_t iq3_xxs = 18;
  for (const auto role : {GgufTensorRole::kMatmulWeight,
                          GgufTensorRole::kStackedExpertWeight}) {
    CAPTURE(vllm::Name(role));
    std::vector<int64_t> shape = role == GgufTensorRole::kMatmulWeight
                                     ? std::vector<int64_t>{8, 256}
                                     : std::vector<int64_t>{2, 8, 256};
    auto route = [&](bool keep_quant, bool cpu_ref) {
      return RouteGgufTensor(keep_quant, /*keep_f16=*/false, /*nvfp4_fp4=*/false,
                             cpu_ref, role, iq3_xxs, shape, vt::DeviceType::kROCM);
    };
    CHECK(route(true, false) == GgufResidency::kKeepQuant);
    CHECK(route(false, false) == GgufResidency::kExpandBf16);
    CHECK(route(true, true) == GgufResidency::kExpandBf16);
    shape.back() = 255;
    CHECK(route(true, false) == GgufResidency::kExpandBf16);
  }
}

TEST_CASE("routing table is TOTAL: every role x every encoding is explicit") {
  // The expectation is written out LONGHAND here rather than derived from the
  // implementation, so this is a real cross-check and not a tautology.
  const GgufTensorRole all_roles[] = {
      GgufTensorRole::kMatmulWeight,      GgufTensorRole::kStackedExpertWeight,
      GgufTensorRole::kTransformedWeight, GgufTensorRole::kEmbeddingTable,
      GgufTensorRole::kConvWeight,        GgufTensorRole::kVector,
  };
  // Q5_0 (6) and IQ4_NL (20) join the list with the encodings MODEL-MM-QWEN4-EXP
  // W6a added (#1989 review F8): a case that calls itself TOTAL and omits the
  // two newest encodings is total over yesterday's surface. IQ2_XS (17) joins
  // it for the same reason with LOADER-GGUF-IQ (#2240).
  // IQ3_S (21) joins with QUANT-IQ3S (#2510); it was the one entry that is
  // GATHER-capable and GEMM-incapable until aa85e9484 gave it a dot kernel,
  // so no entry holds that role any more and the two terms below agree.
  const uint32_t all_types[] = {
      kF32,      kF16,      kBF16,    kQ4_0,     kQ5_0,    kQ8_0,
      kQ2_K,     kQ3_K,     kQ4_K,    kQ5_K,     kQ6_K,    kQ8_K,
      kIQ2_XXS,  kIQ2_XS,   kIQ3_XXS, kIQ1_S,    kIQ4_NL,  kIQ3_S,
      kIQ2_S,    kIQ4_XS,   kMXFP4,   kIQ1_XXXS,
  };

  // ENG-GGUF-RESIDENCY-RESOLVED-DEVICE: ONE device, named here, and both the
  // route and the expectation read it. It used to be
  // `platforms::CurrentPlatform()` on the expectation side and an implicit
  // probe inside `RouteGgufTensor` on the other, so this table's totals were
  // host-dependent and the two sides could only agree by both probing. They are
  // now the same value, and the counts below are the same on every host.
  //
  // F6: and it runs over EVERY device rather than one. With a single `kCPU` the
  // three device terms below (`rocm`, `device_capable`, `gather_device_capable`)
  // collapse to compile-time constants that still READ as device-derived, so
  // the ROCm narrowing and the gather gate were asserted by nobody. The totals
  // are already written per device, so the loop costs nothing and makes them
  // mean it.
  for (vt::DeviceType kRouteDev : {vt::DeviceType::kCPU, vt::DeviceType::kCUDA,
                                   vt::DeviceType::kROCM}) {
  CAPTURE(std::string(vt::DeviceTypeName(kRouteDev)));

  int kept = 0;
  int expanded = 0;
  for (GgufTensorRole role : all_roles) {
    for (uint32_t type : all_types) {
      // Shapes: correct rank for the role, plus wrong-rank and ragged-K probes.
      const std::vector<std::vector<int64_t>> shapes = {
          {8, 256}, {2, 8, 256}, {256}, {8, 255}, {2, 8, 255}, {},
      };
      for (const std::vector<int64_t>& shape : shapes) {
        CAPTURE(vllm::Name(role));
        CAPTURE(type);
        CAPTURE(shape.size());

        // --- the independent expectation ---
        // IQ2_S (256-elem, Q8_K-act) and MXFP4 (32-elem, Q8_0-act) are keep-quant
        // capable as of the UD-IQ2_M vehicle, so they route like the others.
        // The DEVICE axis (review #523): the running device's kernel set can be
        // narrower than the loader's CPU-derived list. The quant-dot provider
        // gives ROCm eight Q8_K-activation formats in addition to the five
        // its GDN provider already owns (Q8_0, IQ4_NL, Q4_K, Q5_K, Q6_K).
        // IQ4_XS is the eighth Q8_K one, added by
        // KERNEL-QUANT-CIQ-GEMM-ROCM-IQUANT (#1940) in pull request #3029.
        // QUANT-GGUF-IQ-VECDOT (#2247) put IQ2_XS and IQ4_XS in this list.
        // They were gather-only between #2245 and #2247 — decoder, no vec_dot —
        // and the `vec_dot` rows are what moved them onto the GEMM arm.
        // IQ3_S dotted with aa85e9484 (tenstorrent-gsq-keepquant wave 1),
        // which landed the CPU oracle `VecDotIQ3_SQ8_K`, so it LEFT the
        // gather-only set and joins `cpu_capable` here — it no longer keeps
        // the two GEMM predicates apart.
        const bool cpu_capable =
            type == kQ4_0 || type == kQ5_0 || type == kQ8_0 || type == kQ3_K ||
            type == kQ2_K || type == kQ4_K || type == kQ5_K ||
            type == kQ6_K || type == kIQ2_XXS || type == kIQ2_XS ||
            type == kIQ3_XXS || type == kIQ1_S || type == kIQ4_NL ||
            type == kIQ2_S || type == kIQ4_XS || type == kIQ3_S ||
            type == kMXFP4 || type == kIQ1_XXXS ||
            // TT-KEEPQUANT-Q8K-WEIGHTS: Q8_K joins the CPU GEMM arm through
            // the dequant composite (cpu_quant_gemm.cpp's to_float
            // fallthrough) — no vec_dot row appeared, the composite IS the
            // serve. It leaves the gather-only surplus it held alone since
            // IQ3_S dotted.
            type == kQ8_K;
        const bool rocm = kRouteDev == vt::DeviceType::kROCM;
        // QUANT-GGUF-IQ4_NL adds kIQ4_NL to the ROCm set. It is the one entry
        // here whose activation encoding is Q8_0 rather than Q8_K, and it is
        // served on BOTH device arms by `DotIQ4_NL` through `IQ4NLGemmK`
        // (single matrix) and `GroupedIQ4NLK` (expert towers). The grouped arm
        // is the load-bearing one: the shipped Qwen3.8-Flash-Next checkpoints
        // store all 48 `ffn_down_exps` in IQ4_NL, and an expert tower reaches
        // the GROUPED provider, so this row is what stops those experts from
        // expanding to bf16 on a ROCm box.
        const bool device_capable =
            (!rocm || type == kQ8_0 || type == kQ2_K || type == kQ3_K ||
             type == kQ4_K || type == kQ5_K || type == kQ6_K ||
             type == kIQ2_XXS || type == kIQ3_XXS || type == kIQ2_S ||
             type == kIQ1_S || type == kIQ1_XXXS || type == kIQ4_NL ||
             type == kIQ4_XS) &&
            // TT-KEEPQUANT-Q8K-WEIGHTS scopes the Q8_K weights arm to CPU +
            // TENSTORRENT: the CUDA provider has no Q8_K-weight serve of its
            // own, so its default admission refuses it (DeviceKeepQuantSupported's
            // default case names the exception).
            !(kRouteDev == vt::DeviceType::kCUDA && type == kQ8_K);
        const bool block_capable = cpu_capable && device_capable;
        const int64_t blk = (type == kQ4_0 || type == kQ5_0 || type == kQ8_0 ||
                             type == kMXFP4 || type == kIQ4_NL)
                                ? 32
                                : 256;
        // MODEL-MM-QWEN4-EXP W6a: the GATHER role is now keep-capable too, and
        // it asks a DIFFERENT question. Its admission is the row decoder, so
        // everything with a `to_float` keeps. Since #2247 the GEMM arm's
        // surplus was Q8_K ALONE (IQ2_XS and IQ4_XS left the moment they got a
        // dot kernel); TT-KEEPQUANT-Q8K-WEIGHTS moved Q8_K onto the GEMM arm
        // through the dequant composite, so the two predicates now agree
        // across the block census and the `|| type == kQ8_K` term below is
        // redundant — kept as the historical shape, the surplus now empty.
        //
        // The DEVICE term is a REGISTRY QUERY since KGATHER, and it is asked of
        // `OpRegistered` rather than of `DeviceQuantGatherSupported` on purpose.
        // The predicate under test IS that function, so re-deriving the
        // expectation from it would be a tautology; asking the registry one
        // level below it asserts the real obligation, which is that
        // `RouteGgufTensor`'s gather term follows what the running build
        // actually registered. It cannot be hand-enumerated like `cpu_capable`
        // above, because whether the CUDA registrar is linked is a property of
        // the build and not of the encoding.
        // IQ3_S dotted with aa85e9484, so it left the gather-only surplus and
        // Q8_K holds that role alone again.
        const bool gather_cpu_capable =
            cpu_capable || type == kQ8_K;
        // THE MERGED FORM, which was in neither branch. #2396 made the gather
        // build-dependent and asked the OP REGISTRY instead of hard-coding
        // "CPU only" — correct, and this row keeps it. But it asked about
        // `CurrentPlatform()`, the process's own accelerator, while the route
        // below is taken for `kRouteDev`. Under this row's device loop those
        // are different devices for two of the three legs, so the probe form
        // would compare a CUDA-registry answer against a ROCm route. The
        // registry question is right; the device it is asked ABOUT must be the
        // one being routed.
        const bool gather_device_capable =
            vt::OpRegistered(vt::OpId::kEmbeddingQuant, kRouteDev);
        bool expect_keep = false;
        if (block_capable) {
          if (role == GgufTensorRole::kMatmulWeight && shape.size() == 2) {
            expect_keep = shape[1] % blk == 0;
          } else if (role == GgufTensorRole::kStackedExpertWeight &&
                     shape.size() == 3) {
            expect_keep = shape[2] % blk == 0;
          }
        }
        if (gather_cpu_capable && gather_device_capable &&
            role == GgufTensorRole::kEmbeddingTable && shape.size() == 2) {
          expect_keep = shape[1] % blk == 0;
        }
        const GgufResidency expected = expect_keep
                                           ? GgufResidency::kKeepQuant
                                           : GgufResidency::kExpandBf16;

        // Counted from the ROUTING RESULT, not from `expect_keep`. Incrementing
        // from the test's own expectation made the totals below a restatement of
        // this loop's arithmetic, so `CHECK(kept == gemm_kept + gather_kept)`
        // could not fail for any behaviour of `RouteGgufTensor` -- it had never
        // fired. Counting `got` makes the totals an independent second reading
        // of the same routes. (Both branches found this independently.)
        const GgufResidency got =
            RouteGgufTensor(/*keep_quant=*/true, /*keep_f16=*/false,
                            /*nvfp4_fp4=*/false, /*cpu_ref=*/false, role, type,
                            shape, kRouteDev);
        CHECK(got == expected);
        (got == GgufResidency::kKeepQuant ? kept : expanded)++;

        // The master switch OFF expands everything, always.
        CHECK(RouteGgufTensor(/*keep_quant=*/false, /*keep_f16=*/false,
                              /*nvfp4_fp4=*/false, /*cpu_ref=*/false, role,
                              type, shape, kRouteDev) ==
              GgufResidency::kExpandBf16);
        // The VT_CPU_REF oracle wins over keep-quant, always.
        CHECK(RouteGgufTensor(/*keep_quant=*/true, /*keep_f16=*/false,
                              /*nvfp4_fp4=*/false, /*cpu_ref=*/true, role,
                              type, shape, kRouteDev) ==
              GgufResidency::kExpandBf16);
      }
    }
  }
  // Both outcomes are actually exercised (a table that never keeps anything
  // would pass every assertion above vacuously). The kept count is
  // device-dependent (review #523): 18 block-capable encodings x 2 keep-capable
  // GEMM roles where the device covers the CPU list; 13 x 2 on ROCm (the
  // 11 Q8_K-activation and base formats, plus IQ4_NL via QUANT-GGUF-IQ4_NL
  // and IQ4_XS via KERNEL-QUANT-CIQ-GEMM-ROCM-IQUANT #1940). The GATHER
  // role adds 19 more (the 18, plus Q8_K) on a device that REGISTERS the
  // block gather, and nothing on a device that does not. Written as named
  // terms rather than one number so a future change to any one of them says
  // which one moved. Both moves are now on record and they are mirror images:
  // LOADER-GGUF-IQ (#2240) moved the GATHER term 11 -> 13 and left GEMM at 20,
  // the shape of a decode-only port; QUANT-GGUF-IQ-VECDOT (#2247) moves the
  // GEMM term 20 -> 24 and leaves GATHER at 13, the shape of a dot-only port.
  // The gather total is UNCHANGED because those two encodings were already
  // gather-kept — which is the whole reason the two arms are separate
  // predicates.
  //
  // KGATHER made the gather term BUILD-DEPENDENT rather than CPU-only: CUDA
  // registers `kEmbeddingQuant` too, so a CUDA build with a device keeps those
  // 13. That term used to read `host == kCPU ? 13 : 0` and would have red 13
  // assertions on exactly the configuration the flip enables.
  //
  // Asked about `kRouteDev`, not about the host. #2396 wrote `host` because at
  // that point one device was routed and it was the process's own; under this
  // row's device loop the two come apart, and asking the host would compare a
  // CUDA-registry answer against a ROCm route on two legs out of three.
  //
  // QUANT-IQ3S (#2510) moves the GATHER term 13 -> 14 and leaves GEMM where it
  // was, the same decode-only shape #2240 had. That asymmetry IS the row's
  // per-tier result: IQ3_S stays compressed in a gather table and expands to
  // bf16 in a GEMM, on every device.
  //
  // QUANT-GGUF-IQ4_NL moves the ROCm GEMM term 22 -> 24 and leaves the CPU/CUDA
  // term at 34, because IQ4_NL was already in the CPU list and only the DEVICE
  // set was narrower. That is the shape of a device-arm port: one encoding, two
  // keep-capable GEMM roles, and no change to either gather term.
  // KERNEL-QUANT-CIQ-GEMM-ROCM-IQUANT (#1940) moves the ROCm GEMM term 24 -> 26
  // (IQ4_XS, same shape: one encoding, two keep-capable GEMM roles). IQ3_XXS was
  // already in ROCm's Q8_K-activation set, so it does not move this number.
  //
  // tenstorrent-gsq-keepquant wave 1 (aa85e9484) moved the CPU/CUDA GEMM term
  // 34 -> 36 and left the ROCm and gather terms where they were: IQ3_S gained
  // a CPU `vec_dot`, so it joined the GEMM arm (one encoding x two keep-capable
  // GEMM roles) on the devices that cover the CPU list.
  //
  // TT-KEEPQUANT-Q8K-WEIGHTS moves the CPU GEMM term 36 -> 38 (Q8_K, one
  // encoding x two keep-capable GEMM roles, through the dequant composite).
  // CUDA stays at 36: its default admission names the Q8_K exception and
  // refuses it — the CUDA provider has no Q8_K-weight serve of its own, and
  // widening its number would be an admitted-but-unimplemented encoding. The
  // gather terms are unchanged (Q8_K was already gather-kept; the composite
  // moves only GEMM roles).
  const int gemm_kept = kRouteDev == vt::DeviceType::kROCM  ? 26
                        : kRouteDev == vt::DeviceType::kCUDA ? 36
                                                             : 38;
  const int gather_kept =
      vt::OpRegistered(vt::OpId::kEmbeddingQuant, kRouteDev) ? 19 : 0;
  CHECK(kept == gemm_kept + gather_kept);
  CHECK(expanded == 22 * 36 - (gemm_kept + gather_kept));
  }
}

TEST_CASE("tensors that are value- or layout-rewritten NEVER keep quant") {
  // These are the routes that would silently CORRUPT a model: the (w-1) norm
  // rewrite, ssm_a = log(-x), the V-head reorders and the conv filter. Every
  // encoding, both keep-quant-capable ranks.
  //
  // `kEmbeddingTable` LEFT this set in MODEL-MM-QWEN4-EXP W6a. It was here for
  // a reason that has since been removed rather than for a correctness reason:
  // a gather is not a value rewrite, and the only thing stopping the table from
  // staying compressed was the absence of a dequantizing gather. That op now
  // exists (`vt::Embedding` over a block table), so the table's residency moved
  // to its own case below. The other four are here on their VALUE, and no op
  // can move them.
  for (const Encoding& e : kEncodings) {
    CAPTURE(e.name);
    for (GgufTensorRole role :
         {GgufTensorRole::kTransformedWeight,
          GgufTensorRole::kConvWeight, GgufTensorRole::kVector}) {
      CAPTURE(vllm::Name(role));
      CHECK(RouteGgufTensor(true, false, false, false, role, e.ggml_type,
                            {8, e.k}, vt::DeviceType::kCPU) ==
            GgufResidency::kExpandBf16);
      CHECK(RouteGgufTensor(true, false, false, false, role, e.ggml_type,
                            {2, 8, e.k}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
    }
  }
}

TEST_CASE("a quantized GATHER TABLE keeps its blocks, per encoding and per K") {
  // The residency this row exists for. `unsloth/Qwen3.8-Flash-Next-GGUF
  // UD-IQ1_S` ships `per_layer_token_embd.weight` as IQ4_NL [160, 320001536]:
  // 51.2 G parameters, 28.8 GB of blocks against 102.4 GB expanded to bf16 on a
  // box with ~119.6 GiB. Expanding it is not a slow load, it is the end of the
  // box, so this is a memory-safety decision and not a performance one.
  //
  // The eligibility rule is the GATHER's, not the GEMM's: the encoding needs a
  // ROW DECODER, and the row needs to be a whole number of blocks. It does NOT
  // need a `vec_dot`, because nothing dots a gather table.
  for (const Encoding& e : kEncodings) {
    CAPTURE(e.name);
    CHECK(RouteGgufTensor(true, false, false, false,
                          GgufTensorRole::kEmbeddingTable, e.ggml_type,
                          {8, e.k}, vt::DeviceType::kCPU) == GgufResidency::kKeepQuant);
    // A RAGGED row cannot be decoded block-wise, so it expands. This is the
    // same `ggml_row_size` precondition the GEMM arm obeys, and it is why the
    // 160-wide table in the shipped file is IQ4_NL (32) and not a K-quant (256).
    CHECK(RouteGgufTensor(true, false, false, false,
                          GgufTensorRole::kEmbeddingTable, e.ggml_type,
                          {8, e.k - 1}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
    // Rank is still part of the role's contract: a table is 2-D.
    CHECK(RouteGgufTensor(true, false, false, false,
                          GgufTensorRole::kEmbeddingTable, e.ggml_type,
                          {2, 8, e.k}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
    // And the oracle switch still wins over everything.
    CHECK(RouteGgufTensor(true, false, false, true,
                          GgufTensorRole::kEmbeddingTable, e.ggml_type,
                          {8, e.k}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
  }
  // IQ4_NL at the SHIPPED shape: 160 is 5 whole 32-element blocks.
  CHECK(RouteGgufTensor(true, false, false, false,
                        GgufTensorRole::kEmbeddingTable, 20u,
                        {320001536, 160}, vt::DeviceType::kCPU) == GgufResidency::kKeepQuant);
  // The case that separates "the reader knows this id" from "this build can
  // gather it" is Q1_0 (41): tabulated by the READER, decodable by nobody here.
  // It must expand, or the table would be kept as bytes nothing can read.
  CHECK(RouteGgufTensor(true, false, false, false,
                        GgufTensorRole::kEmbeddingTable, 41u,
                        {8, 128}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
  // IQ2_XS (17) and IQ4_XS (23) keep on BOTH arms as of QUANT-GGUF-IQ-VECDOT
  // (#2247). They kept on the gather arm from #2245, on a row decoder alone;
  // the GEMM arm additionally wanted a `vec_dot`, and now has one. This is the
  // assertion that reds if either kernel is unregistered, and it is the
  // residency half of the 325.58 GiB the two kernels buy on the staged
  // GLM-5.3-Flash artifact.
  for (uint32_t id : {17u, 23u}) {
    CAPTURE(id);
    CHECK(RouteGgufTensor(true, false, false, false,
                          GgufTensorRole::kEmbeddingTable, id,
                          {8, 256}, vt::DeviceType::kCPU) == GgufResidency::kKeepQuant);
    CHECK(RouteGgufTensor(true, false, false, false,
                          GgufTensorRole::kMatmulWeight, id,
                          {8, 256}, vt::DeviceType::kCPU) == GgufResidency::kKeepQuant);
    // The stacked-expert role is the one that actually carries them in the
    // artifact: all 82 IQ2_XS tensors are `blk.N.ffn_{gate,up}_exps.weight`.
    CHECK(RouteGgufTensor(true, false, false, false,
                          GgufTensorRole::kStackedExpertWeight, id,
                          {4, 8, 256}, vt::DeviceType::kCPU) == GgufResidency::kKeepQuant);
    // A ragged K still expands: 256-element super-blocks admit no partial row.
    CHECK(RouteGgufTensor(true, false, false, false,
                          GgufTensorRole::kMatmulWeight, id,
                          {8, 255}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
  }
}

TEST_CASE("the gather table's admission is the DECODER, not the vec_dot") {
  // The discriminating case, and the reason this is not the same predicate as
  // the GEMM arm's. If the table were admitted through `KeepQuantDType` it
  // would inherit `HasQuantDotKernel`, which asks for a `vec_dot` AND a
  // `from_float` on the activation encoding — neither of which a gather uses.
  // Every encoding below has a row decoder; that is the whole requirement.
  vt::DType dt = vt::DType::kF32;
  for (uint32_t type : {kQ4_0, kQ8_0, kQ3_K, kQ4_K, kQ5_K, kQ6_K, 20u, 17u,
                        23u}) {
    CAPTURE(type);
    REQUIRE(vllm::KeepQuantGatherDType(type, &dt));
    REQUIRE(vt::cpu::BlockToFloat(dt) != nullptr);
  }
  // Q8_K passed this line and, between #2247 and TT-KEEPQUANT-Q8K-WEIGHTS
  // (2026-11-13), FAILED the GEMM predicate — the pair of predicates was
  // distinct exactly while Q8_K was the one decoder with no vec_dot. It left
  // the same way IQ2_XS and IQ4_XS left #2247: it gained a GEMM arm, the
  // dequant-COMPOSITE one (no vec_dot appeared; the composite is the serve).
  // With every decoder now admitted on both arms the discriminator is
  // historical; it is re-grounded, not deleted, per its own contract.
  REQUIRE(vllm::KeepQuantGatherDType(kQ8_K, &dt));
  CHECK(vllm::KeepQuantDType(kQ8_K, &dt));
  CHECK_FALSE(vt::cpu::HasQuantDotKernel(dt));  // the composite arm, still
  // And the pair that MOVED: both predicates now admit them.
  for (uint32_t type : {kIQ2_XS, kIQ4_XS}) {
    CAPTURE(type);
    CHECK(vllm::KeepQuantGatherDType(type, &dt));
    CHECK(vllm::KeepQuantDType(type, &dt));
  }
  // Not a block encoding at all -> never a gather keep. Q1_0 (41) is a block
  // encoding the READER tabulates and vt does not know, so it fails here too.
  for (uint32_t type : {kF32, kF16, kBF16, kQ1_0}) {
    CAPTURE(type);
    CHECK_FALSE(vllm::KeepQuantGatherDType(type, &dt));
  }
}

TEST_CASE("Q8_K GGUF reader preserves block geometry and packed bytes") {
  const auto& format = rocm_gather_test::kFormats[8];
  const TempFile file(rocm_gather_test::BuildModel(format, false));
  const auto gguf = vllm::GgufFile::Open(file.path());
  const auto& traits = vllm::GgmlTraits(15);
  CHECK(traits.block_elems == 256);
  CHECK(traits.block_bytes == 292);
  const auto& tensor = gguf.Get("token_embd.weight");
  CHECK(tensor.ggml_type == 15);
  CHECK(tensor.shape == std::vector<int64_t>{128, 256});
  REQUIRE(tensor.nbytes == 128 * 292);
  const auto packed = rocm_gather_test::PackedTable(format);
  REQUIRE(packed.size() == tensor.nbytes);
  CHECK(std::memcmp(tensor.data, packed.data(), tensor.nbytes) == 0);
}

TEST_CASE("Q8_K embedding loader retains decoder-only blocks; the matrix arm "
          "now admits the composite") {
  // BACKEND-ROCM-QUANT-GATHER (#3093): the ordinary materializer must follow
  // the tensor role already resolved by the loader, including decoder-only Q8_K.
  // TT-KEEPQUANT-Q8K-WEIGHTS moved Q8_K onto the GEMM arm through the
  // dequant composite, so the matrix-refusal legs this case carried are gone:
  // a Q8_K matrix now keeps its blocks (the legs below assert the ADMISSION,
  // and the throw legs keep only the repack refusal, which is role-shaped and
  // not encoding-shaped).
  const auto& format = rocm_gather_test::kFormats[8];
  REQUIRE(format.dtype == vt::DType::kQ8_K);
  const TempFile file(rocm_gather_test::BuildModel(format, false));
  const auto gguf = vllm::GgufFile::Open(file.path());
  const auto config = vllm::HfConfigFromGguf(gguf);
  auto policy = KeepQuantOn();
  policy.device = vt::DeviceType::kCPU;
  const auto& packed = gguf.Get("token_embd.weight");
  CHECK(vllm::KeepQuantDType(packed.ggml_type, nullptr));
  CHECK(vllm::KeepQuantGatherDType(packed.ggml_type, nullptr));
  CHECK(vllm::RouteGgufTensor(true, false, false, false,
          vllm::GgufTensorRole::kMatmulWeight, packed.ggml_type, packed.shape,
          vt::DeviceType::kCPU) == vllm::GgufResidency::kKeepQuant);
  CHECK_NOTHROW(vllm::OwnGgufQuantBlocks(packed, 128, 256));
  for (const bool cuda_align : {false, true}) {
    CHECK_THROWS_WITH_AS(vllm::OwnGgufQuantBlocks(packed, 128, 256, 0, nullptr,
                         !cuda_align, cuda_align, true, vllm::GgufTensorRole::kEmbeddingTable),
                         doctest::Contains("cannot use a matrix repack"), std::runtime_error);
  }
  const auto weights = vllm::LoadQwen3_5DenseFromGguf(gguf, config, &policy);
  CHECK(weights.embed_tokens.dtype == vt::DType::kQ8_K);
  CHECK_FALSE(weights.embed_tokens.nk);
  REQUIRE(weights.embed_tokens.bytes.size() == packed.nbytes);
  CHECK(std::memcmp(weights.embed_tokens.bytes.data(), packed.data, packed.nbytes) == 0);
}

TEST_CASE("the gather's DEVICE gate is the OP TABLE, not a hand-kept device list") {
  // KGATHER. `DeviceQuantGatherSupported` is `OpRegistered(kEmbeddingQuant, dev)`
  // and names no device. That is not a style preference: the GGUF loader is the
  // device-agnostic layer, `scripts/check-device-leakage.py` refuses a device
  // enumerator in it, and a per-device set written here drifts from the kernels
  // it claims to describe. `GgufQuantComputeAvailable()` asks the same way for
  // the GEMM arm.
  //
  // The predicate therefore answers for the BUILD, which is the point: a CUDA
  // build links the cuda_ops registrar and a CPU-only build does not, and the
  // residency decision must follow what is actually linked rather than what the
  // source could in principle do.
  CHECK(vllm::DeviceQuantGatherSupported(vt::DeviceType::kCPU));
  CHECK(vllm::DeviceQuantGatherSupported(vt::DeviceType::kCPU) ==
        vt::OpRegistered(vt::OpId::kEmbeddingQuant, vt::DeviceType::kCPU));

  // Derived from the registry on BOTH sides rather than hardcoded to a build
  // flavour — the same shape the keep_quant default case below uses, and the
  // reason this case reads identically on a CPU-only and a CUDA build.
  for (vt::DeviceType d :
       {vt::DeviceType::kCUDA, vt::DeviceType::kMETAL, vt::DeviceType::kVULKAN,
        vt::DeviceType::kROCM, vt::DeviceType::kTENSTORRENT}) {
    CAPTURE(vt::DeviceTypeName(d));
    CHECK(vllm::DeviceQuantGatherSupported(d) ==
          vt::OpRegistered(vt::OpId::kEmbeddingQuant, d));
  }

  // The three that have no block-decoding gather in ANY build: each of their
  // `kEmbedding` kernels asserts a float table by name (e.g. tenstorrent_ops.cpp
  // "tenstorrent kEmbedding: float table, f32/bf16 out"), none registers
  // `kEmbeddingQuant`, and their arms are owed. A backend that registered the
  // quant id without writing the decoder would turn a clean load-time refusal
  // into a forward-time throw with the whole model resident — the #523 failure —
  // so this half stays an absolute assertion and not a registry echo.
  for (vt::DeviceType d :
       {vt::DeviceType::kMETAL, vt::DeviceType::kVULKAN,
        vt::DeviceType::kTENSTORRENT}) {
    CAPTURE(vt::DeviceTypeName(d));
    CHECK_FALSE(vllm::DeviceQuantGatherSupported(d));
  }
#ifdef VLLM_CPP_HIP
  CHECK(vllm::DeviceQuantGatherSupported(vt::DeviceType::kROCM));
#else
  CHECK_FALSE(vllm::DeviceQuantGatherSupported(vt::DeviceType::kROCM));
#endif
}

TEST_CASE("GgufLoadPolicy::FromEnv reads VT_CPU_REF and VT_GGUF_KEEP_QUANT") {
  ::unsetenv("VT_CPU_REF");
  ::unsetenv("VT_GGUF_KEEP_QUANT");
  // keep_f16 additionally requires an f16-capable MatmulBT on the running
  // device (review #523): the ROCm kernel accepts bf16/f32 only, so keep_f16
  // is OFF on ROCm regardless of expand_nk.
  // ENG-GGUF-RESIDENCY-RESOLVED-DEVICE: read off the device the policies below
  // are BUILT FOR, not off the host. The two used to be the same read, which is
  // why they could not disagree; now that the device is a parameter, an
  // expectation that probed the host would contradict a policy resolved for
  // another device.
  const bool f16_device_ok = vt::DeviceType::kCPU != vt::DeviceType::kROCM;
  const auto keep_f16_expected = [&](const GgufLoadPolicy& q) {
    return q.expand_nk && f16_device_ok;
  };
  {
    // PRODUCTION DEFAULT SINCE CIQ G4: keep-quant follows the running device's
    // ability to EXECUTE the quantized GEMM. The expectation is derived from
    // the op registry, not hardcoded to a build flavour, so this case states
    // the same rule on a CPU-only build (available -> ON) and on a CUDA build
    // (kMatmulBTQuant unregistered for kCUDA -> OFF, and the loader keeps
    // expanding to bf16 exactly as before).
    const GgufLoadPolicy p = GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU);
    CHECK(p.keep_quant == vllm::GgufQuantComputeAvailable(vt::DeviceType::kCPU));
    CHECK(p.expand_nk == vllm::GgufQuantComputeAvailable(vt::DeviceType::kCPU));
    // L7 (2026-07-23): keep-f16 is now DEFAULT ON wherever expand_nk holds — the
    // repack-source release + load-time prefault removed L6's two objections
    // (RSS-neutral, prefill regression), so it buys 1.05 GiB of peak RSS with
    // byte-identical tokens. Its llama.cpp denominators are SUPERSEDED (fork
    // 237ad9b96, re-take owed under #1003, see gguf_keep_quant.cpp).
    //
    // This CHECK pins the DEFAULT, not the trade behind it, and separating the
    // two matters here. An earlier pass recorded the default as independent of
    // the contaminated floor, because the L7 acceptance is a same-binary
    // ours-versus-ours A/B. That reads one row of a three-row table: prefill
    // (~10%, 224 -> 204 t/s) and decode (~1.4%) BOTH regress to buy that RSS, and
    // the recorded reason the prefill loss is acceptable is "comfortably above
    // the competitor floor", which IS the contaminated pp128. If #1003's re-take
    // puts stock above 204 t/s, that justification is gone and the default
    // becomes a live question for QUANT-GGUF-KEEPQ-LOADER. Should that happen,
    // THIS assertion is one of the things that has to change, so it is flagged
    // here rather than discovered when it goes red.
    CHECK(p.keep_f16 == (vllm::GgufQuantComputeAvailable(vt::DeviceType::kCPU) && f16_device_ok));
    CHECK_FALSE(p.cpu_ref);
  }
  ::setenv("VT_GGUF_KEEP_QUANT", "1", 1);
  CHECK(GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU).keep_quant);
  CHECK(GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU).expand_nk);
  // L7: keep-f16 defaults to expand_nk (true here, keep-quant is env-forced ON).
  // NB compare to expand_nk, NOT GgufQuantComputeAvailable(): with keep-quant
  // env-forced, expand_nk holds even on a CUDA build where the quant GEMM is
  // unregistered (GgufQuantComputeAvailable() is false there).
  CHECK(GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU).keep_f16 ==
        keep_f16_expected(GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU)));
  // The opt-out must work after the default flip.
  ::setenv("VT_GGUF_KEEP_F16", "0", 1);
  CHECK_FALSE(GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU).keep_f16);
  ::unsetenv("VT_GGUF_KEEP_F16");
  // The OPT-OUT the spec promised must survive the default flip.
  for (const char* off : {"0", "false", "off", ""}) {
    ::setenv("VT_GGUF_KEEP_QUANT", off, 1);
    CAPTURE(off);
    CHECK_FALSE(GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU).keep_quant);
    CHECK_FALSE(GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU).expand_nk);
    CHECK_FALSE(GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU).keep_f16);
  }
  // VT_GGUF_KEEP_F16=1 opts IN, but ONLY where expand_nk holds (CPU, not oracle);
  // it is inert with keep-quant off (nothing to keep) or under VT_CPU_REF.
  ::setenv("VT_GGUF_KEEP_QUANT", "1", 1);
  ::setenv("VT_GGUF_KEEP_F16", "1", 1);
  CHECK(GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU).keep_f16 ==
        keep_f16_expected(GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU)));
  for (const char* on : {"1", "true", "on"}) {
    ::setenv("VT_GGUF_KEEP_F16", on, 1);
    CAPTURE(on);
    CHECK(GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU).keep_f16 ==
        keep_f16_expected(GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU)));
  }
  ::unsetenv("VT_GGUF_KEEP_F16");
  ::setenv("VT_GGUF_KEEP_QUANT", "1", 1);
  ::setenv("VT_CPU_REF", "1", 1);
  {
    // The oracle switch: keep-quant requested, oracle wins — and it takes the
    // orientation and keep-f16 with it, so VT_CPU_REF=1 is the FULL historical
    // load.
    ::setenv("VT_GGUF_KEEP_F16", "1", 1);
    const GgufLoadPolicy p = GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU);
    CHECK(p.keep_quant);
    CHECK(p.cpu_ref);
    CHECK_FALSE(p.expand_nk);
    CHECK_FALSE(p.keep_f16);
    ::unsetenv("VT_GGUF_KEEP_F16");
    CHECK(p.Route(vllm::GgufTensorInfo{"w", {8, 256}, kQ4_K, nullptr, 0},
                  GgufTensorRole::kMatmulWeight) ==
          GgufResidency::kExpandBf16);
  }
  ::unsetenv("VT_CPU_REF");
  ::unsetenv("VT_GGUF_KEEP_QUANT");
}

// The default is only correct if it means "a block weight has a consumer". On
// this CPU test binary the quantized GEMM IS registered, so the availability
// probe must say so — otherwise the flip above would be vacuous.
TEST_CASE("GgufQuantComputeAvailable tracks the kMatmulBTQuant registration") {
  // Both sides name the SAME device, which is the point of the parameter: this
  // used to read the process's platform on the right and the same probe,
  // implicitly, on the left, so it could not fail.
  for (vt::DeviceType d : {vt::DeviceType::kCPU, vt::DeviceType::kCUDA,
                           vt::DeviceType::kROCM, vt::DeviceType::kMETAL}) {
    CAPTURE(std::string(vt::DeviceTypeName(d)));
    CHECK(vllm::GgufQuantComputeAvailable(d) ==
          vt::OpRegistered(vt::OpId::kMatmulBTQuant, d));
  }
  // The CPU kernel IS registered in this binary, so keep-quant is live for a
  // CPU-resolved load and the default flip above is not vacuous.
  CHECK(vllm::GgufQuantComputeAvailable(vt::DeviceType::kCPU));
}

// ===========================================================================
// L3 — the loader routes EVERY tensor, and keep-quant stays lossless there.
// ===========================================================================

namespace {

struct DenseDims {
  int64_t H = 64, vocab = 32, n_head = 2, n_head_kv = 1, head_dim = 32,
          I = 64, n_layer = 2;
};

std::string F32Bytes(int64_t n, float base) {
  std::string s;
  s.reserve(static_cast<size_t>(n) * 4);
  for (int64_t i = 0; i < n; ++i) {
    const float v = base + 0.25F * static_cast<float>(i % 7);
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    for (int k = 0; k < 4; ++k) {
      s.push_back(static_cast<char>((bits >> (8 * k)) & 0xFF));
    }
  }
  return s;
}

// Real Q8_0 blocks produced by the ported `quantize_row_q8_0_ref`, so the
// loader test runs over data a converter could actually have written.
std::string Q8_0Bytes(int64_t rows, int64_t k, uint32_t seed) {
  std::vector<float> src(static_cast<size_t>(rows * k));
  uint32_t s = seed | 1U;
  for (float& v : src) {
    v = static_cast<float>(static_cast<int32_t>(NextRand(&s) % 2001) - 1000) /
        128.0F;
  }
  const size_t nbytes = vt::RowSizeBytes(vt::DType::kQ8_0, rows * k);
  std::string out(nbytes, '\0');
  vt::cpu::FromFloatFn q = vt::cpu::BlockFromFloat(vt::DType::kQ8_0);
  REQUIRE(q != nullptr);
  // Row by row: ggml quantizes whole rows, and a row is whole blocks.
  const size_t row_bytes = vt::RowSizeBytes(vt::DType::kQ8_0, k);
  for (int64_t r = 0; r < rows; ++r) {
    q(src.data() + r * k, out.data() + static_cast<size_t>(r) * row_bytes, k);
  }
  return out;
}

void AddQ8_0(GgufModelBuilder& b, const std::string& name, int64_t out_dim,
             int64_t in_dim, uint32_t seed) {
  b.AddTensor(name,
              {static_cast<uint64_t>(in_dim), static_cast<uint64_t>(out_dim)},
              kQ8_0, Q8_0Bytes(out_dim, in_dim, seed));
}

void AddF32T(GgufModelBuilder& b, const std::string& name,
             const std::vector<int64_t>& torch_shape, float base) {
  std::vector<uint64_t> dims;
  for (auto it = torch_shape.rbegin(); it != torch_shape.rend(); ++it) {
    dims.push_back(static_cast<uint64_t>(*it));
  }
  int64_t n = 1;
  for (int64_t d : torch_shape) n *= d;
  b.AddTensor(name, dims, kF32, F32Bytes(n, base));
}

// A tiny DENSE (`qwen35`) GGUF whose GEMM weights are Q8_0 and whose norms /
// embedding stay F32 — the realistic mixed-type file the routing policy exists
// for. All layers are full attention (interval 1), so no GDN tensors appear.
// `tied` omits `output.weight`, which is how a tied-embedding GGUF says "the
// head IS token_embd" (llama.cpp TENSOR_DUPLICATED) — the L5 sharing case.
std::string BuildDenseQ8Gguf(const DenseDims& d, bool tied = false) {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "qwen35"));
  b.AddKv(U32Kv("qwen35.embedding_length", static_cast<uint32_t>(d.H)));
  b.AddKv(U32Kv("qwen35.block_count", static_cast<uint32_t>(d.n_layer)));
  b.AddKv(U32Kv("qwen35.attention.head_count", static_cast<uint32_t>(d.n_head)));
  b.AddKv(U32Kv("qwen35.attention.head_count_kv",
                static_cast<uint32_t>(d.n_head_kv)));
  b.AddKv(U32Kv("qwen35.attention.key_length",
                static_cast<uint32_t>(d.head_dim)));
  b.AddKv(U32Kv("qwen35.feed_forward_length", static_cast<uint32_t>(d.I)));
  b.AddKv(F32Kv("qwen35.attention.layer_norm_rms_epsilon", 1e-6F));
  b.AddKv(F32Kv("qwen35.rope.freq_base", 1000000.0F));
  b.AddKv(U32Kv("qwen35.full_attention_interval", 1));
  b.AddKv(U32Kv("qwen35.context_length", 4096));

  AddF32T(b, "token_embd.weight", {d.vocab, d.H}, 0.5F);
  AddF32T(b, "output_norm.weight", {d.H}, 1.5F);
  if (!tied) AddQ8_0(b, "output.weight", d.vocab, d.H, 11);
  for (int64_t il = 0; il < d.n_layer; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    AddF32T(b, p + "attn_norm.weight", {d.H}, 1.25F);
    AddF32T(b, p + "post_attention_norm.weight", {d.H}, 1.75F);
    AddQ8_0(b, p + "attn_q.weight", d.n_head * d.head_dim, d.H, 21 + il);
    AddQ8_0(b, p + "attn_k.weight", d.n_head_kv * d.head_dim, d.H, 31 + il);
    AddQ8_0(b, p + "attn_v.weight", d.n_head_kv * d.head_dim, d.H, 41 + il);
    AddQ8_0(b, p + "attn_output.weight", d.H, d.n_head * d.head_dim, 51 + il);
    AddF32T(b, p + "attn_q_norm.weight", {d.head_dim}, 1.5F);
    AddF32T(b, p + "attn_k_norm.weight", {d.head_dim}, 1.5F);
    AddQ8_0(b, p + "ffn_gate.weight", d.I, d.H, 61 + il);
    AddQ8_0(b, p + "ffn_up.weight", d.I, d.H, 71 + il);
    AddQ8_0(b, p + "ffn_down.weight", d.H, d.I, 81 + il);
  }
  return b.Build();
}

// The bf16 [K,N] tensor today's loader produces from a [N,K] GGUF weight:
// dequant, then transpose. Used to prove the keep-quant weight carries the
// SAME information as the expanded one.
std::vector<uint16_t> ExpandAndTranspose(uint32_t ggml_type,
                                         const uint8_t* blocks, int64_t n,
                                         int64_t k) {
  const std::vector<uint16_t> dq =
      vllm::DequantGgufRowToBf16(ggml_type, blocks, n * k);
  std::vector<uint16_t> out(dq.size());
  for (int64_t r = 0; r < n; ++r) {
    for (int64_t c = 0; c < k; ++c) out[c * n + r] = dq[r * k + c];
  }
  return out;
}

}  // namespace

TEST_CASE("loader routes EVERY tensor in the file (total coverage)") {
  const DenseDims d;
  const TempFile f(BuildDenseQ8Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  std::set<std::string> routed;
  std::set<std::string> kept;
  GgufLoadPolicy pol = KeepQuantOn();
  pol.audit = [&](const std::string& name, GgufTensorRole,
                  GgufResidency res) {
    routed.insert(name);
    if (res == GgufResidency::kKeepQuant) kept.insert(name);
  };
  const vllm::Qwen3_5DenseWeights w =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &pol);
  REQUIRE(w.layers.size() == static_cast<size_t>(d.n_layer));

  // TOTALITY: the audited set is exactly the file's tensor list. Nothing the
  // loader consumed skipped the policy, and the policy saw nothing spurious.
  std::set<std::string> in_file;
  for (const vllm::GgufTensorInfo& t : g.Tensors()) in_file.insert(t.name);
  CHECK(routed == in_file);

  // And the decisions are the intended ones: every Q8_0 GEMM weight kept, the
  // F32 norms / embedding expanded.
  std::set<std::string> expect_kept;
  expect_kept.insert("output.weight");
  for (int64_t il = 0; il < d.n_layer; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    for (const char* s : {"attn_q.weight", "attn_k.weight", "attn_v.weight",
                          "attn_output.weight", "ffn_gate.weight",
                          "ffn_up.weight", "ffn_down.weight"}) {
      expect_kept.insert(p + s);
    }
  }
  CHECK(kept == expect_kept);
  CHECK(kept.count("token_embd.weight") == 0);
  CHECK(kept.count("output_norm.weight") == 0);
  CHECK(kept.count("blk.0.attn_q_norm.weight") == 0);
}

TEST_CASE("loader keep-quant weights are lossless vs the bf16 expansion") {
  const DenseDims d;
  const TempFile f(BuildDenseQ8Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  const GgufLoadPolicy off;  // production default: expand everything
  GgufLoadPolicy on = KeepQuantOn();
  const vllm::Qwen3_5DenseWeights base =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &off);
  const vllm::Qwen3_5DenseWeights kept =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &on);

  // Pairs of (kept weight, expanded weight, source tensor).
  struct Pair {
    const vllm::OwnedTensor* q;
    const vllm::OwnedTensor* b;
    std::string name;
  };
  std::vector<Pair> pairs = {{&kept.lm_head, &base.lm_head, "output.weight"}};
  for (int64_t il = 0; il < d.n_layer; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const auto& kl = kept.layers[static_cast<size_t>(il)];
    const auto& bl = base.layers[static_cast<size_t>(il)];
    pairs.push_back({&kl.attn.q_proj, &bl.attn.q_proj, p + "attn_q.weight"});
    pairs.push_back({&kl.attn.k_proj, &bl.attn.k_proj, p + "attn_k.weight"});
    pairs.push_back({&kl.attn.v_proj, &bl.attn.v_proj, p + "attn_v.weight"});
    pairs.push_back(
        {&kl.attn.o_proj, &bl.attn.o_proj, p + "attn_output.weight"});
    pairs.push_back(
        {&kl.mlp.gate_proj, &bl.mlp.gate_proj, p + "ffn_gate.weight"});
    pairs.push_back({&kl.mlp.up_proj, &bl.mlp.up_proj, p + "ffn_up.weight"});
    pairs.push_back(
        {&kl.mlp.down_proj, &bl.mlp.down_proj, p + "ffn_down.weight"});
  }

  for (const Pair& pr : pairs) {
    CAPTURE(pr.name);
    const vllm::GgufTensorInfo& t = g.Get(pr.name);
    const int64_t n = t.shape[0];
    const int64_t k = t.shape[1];

    // Kept: block dtype, file orientation [N,K], nk = true, raw file bytes.
    CHECK(pr.q->dtype == vt::DType::kQ8_0);
    CHECK(pr.q->nk == true);
    CHECK(pr.q->shape[0] == n);
    CHECK(pr.q->shape[1] == k);
    REQUIRE(pr.q->bytes.size() == t.nbytes);
    CHECK(std::memcmp(pr.q->bytes.data(), t.data, t.nbytes) == 0);

    // Expanded: bf16, Matmul-B orientation [K,N] — today's exact behavior.
    CHECK(pr.b->dtype == vt::DType::kBF16);
    CHECK(pr.b->nk == false);
    CHECK(pr.b->shape[0] == k);
    CHECK(pr.b->shape[1] == n);

    // THE GATE: dequantizing the resident blocks (and applying the same
    // transpose) reproduces the expanded tensor BYTE for BYTE.
    const std::vector<uint16_t> rehydrated =
        ExpandAndTranspose(kQ8_0, pr.q->bytes.data(), n, k);
    REQUIRE(rehydrated.size() * sizeof(uint16_t) == pr.b->bytes.size());
    CHECK(std::memcmp(rehydrated.data(), pr.b->bytes.data(),
                      pr.b->bytes.size()) == 0);
  }

  // Tensors that must NOT keep quant are bit-identical between the two loads,
  // so enabling keep-quant cannot perturb them.
  CHECK(kept.embed_tokens.bytes == base.embed_tokens.bytes);
  CHECK(kept.final_norm.bytes == base.final_norm.bytes);
  for (int64_t il = 0; il < d.n_layer; ++il) {
    const auto& kl = kept.layers[static_cast<size_t>(il)];
    const auto& bl = base.layers[static_cast<size_t>(il)];
    CHECK(kl.input_layernorm.bytes == bl.input_layernorm.bytes);
    CHECK(kl.post_attention_layernorm.bytes ==
          bl.post_attention_layernorm.bytes);
    CHECK(kl.attn.q_norm.bytes == bl.attn.q_norm.bytes);
    CHECK(kl.attn.k_norm.bytes == bl.attn.k_norm.bytes);
  }
}

TEST_CASE("VT_CPU_REF forces the full dequant oracle path in the loader") {
  const DenseDims d;
  const TempFile f(BuildDenseQ8Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  const GgufLoadPolicy off;
  GgufLoadPolicy oracle = KeepQuantOn();
  oracle.cpu_ref = true;  // VT_CPU_REF=1
  std::set<std::string> kept;
  oracle.audit = [&](const std::string& n, GgufTensorRole, GgufResidency r) {
    if (r == GgufResidency::kKeepQuant) kept.insert(n);
  };

  const vllm::Qwen3_5DenseWeights base =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &off);
  const vllm::Qwen3_5DenseWeights ref =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &oracle);

  // Nothing stayed quantized...
  CHECK(kept.empty());
  // ...and every weight is BIT-IDENTICAL to the historical load. This is the
  // whole point of the oracle switch: the reference numerics stay reachable.
  CHECK(ref.embed_tokens.bytes == base.embed_tokens.bytes);
  CHECK(ref.final_norm.bytes == base.final_norm.bytes);
  CHECK(ref.lm_head.bytes == base.lm_head.bytes);
  CHECK(ref.lm_head.dtype == base.lm_head.dtype);
  REQUIRE(ref.layers.size() == base.layers.size());
  for (size_t il = 0; il < ref.layers.size(); ++il) {
    CAPTURE(il);
    const auto& r = ref.layers[il];
    const auto& b = base.layers[il];
    CHECK(r.input_layernorm.bytes == b.input_layernorm.bytes);
    CHECK(r.post_attention_layernorm.bytes == b.post_attention_layernorm.bytes);
    CHECK(r.attn.q_proj.bytes == b.attn.q_proj.bytes);
    CHECK(r.attn.k_proj.bytes == b.attn.k_proj.bytes);
    CHECK(r.attn.v_proj.bytes == b.attn.v_proj.bytes);
    CHECK(r.attn.o_proj.bytes == b.attn.o_proj.bytes);
    CHECK(r.attn.q_norm.bytes == b.attn.q_norm.bytes);
    CHECK(r.attn.k_norm.bytes == b.attn.k_norm.bytes);
    CHECK(r.mlp.gate_proj.bytes == b.mlp.gate_proj.bytes);
    CHECK(r.mlp.up_proj.bytes == b.mlp.up_proj.bytes);
    CHECK(r.mlp.down_proj.bytes == b.mlp.down_proj.bytes);
  }
}

// ===========================================================================
// L2/L3 — the MoE loader's STACKED-EXPERT keep-quant split.
// ===========================================================================

namespace {

struct MoeDims {
  int64_t H = 64, vocab = 32, n_head = 2, n_head_kv = 1, head_dim = 32,
          E = 3, used = 2, I = 64, Is = 64, n_layer = 1;
};

// A tiny `qwen35moe` GGUF: Q8_0 GEMM weights (including the STACKED expert
// tensors) and F32 norms/embedding. full_attention_interval = 1 makes every
// layer full-attention, so no GDN tensors are needed.
std::string BuildMoeQ8Gguf(const MoeDims& d) {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "qwen35moe"));
  b.AddKv(U32Kv("qwen35moe.embedding_length", static_cast<uint32_t>(d.H)));
  b.AddKv(U32Kv("qwen35moe.block_count", static_cast<uint32_t>(d.n_layer)));
  b.AddKv(U32Kv("qwen35moe.attention.head_count",
                static_cast<uint32_t>(d.n_head)));
  b.AddKv(U32Kv("qwen35moe.attention.head_count_kv",
                static_cast<uint32_t>(d.n_head_kv)));
  b.AddKv(U32Kv("qwen35moe.attention.key_length",
                static_cast<uint32_t>(d.head_dim)));
  b.AddKv(U32Kv("qwen35moe.expert_count", static_cast<uint32_t>(d.E)));
  b.AddKv(U32Kv("qwen35moe.expert_used_count", static_cast<uint32_t>(d.used)));
  b.AddKv(U32Kv("qwen35moe.expert_feed_forward_length",
                static_cast<uint32_t>(d.I)));
  b.AddKv(U32Kv("qwen35moe.expert_shared_feed_forward_length",
                static_cast<uint32_t>(d.Is)));
  b.AddKv(F32Kv("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6F));
  b.AddKv(F32Kv("qwen35moe.rope.freq_base", 1000000.0F));
  b.AddKv(U32Kv("qwen35moe.full_attention_interval", 1));
  b.AddKv(U32Kv("qwen35moe.context_length", 4096));

  AddF32T(b, "token_embd.weight", {d.vocab, d.H}, 0.5F);
  AddF32T(b, "output_norm.weight", {d.H}, 1.5F);
  AddQ8_0(b, "output.weight", d.vocab, d.H, 111);
  for (int64_t il = 0; il < d.n_layer; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    AddF32T(b, p + "attn_norm.weight", {d.H}, 1.25F);
    AddF32T(b, p + "post_attention_norm.weight", {d.H}, 1.75F);
    AddQ8_0(b, p + "attn_q.weight", d.n_head * d.head_dim, d.H, 121 + il);
    AddQ8_0(b, p + "attn_k.weight", d.n_head_kv * d.head_dim, d.H, 131 + il);
    AddQ8_0(b, p + "attn_v.weight", d.n_head_kv * d.head_dim, d.H, 141 + il);
    AddQ8_0(b, p + "attn_output.weight", d.H, d.n_head * d.head_dim, 151 + il);
    AddF32T(b, p + "attn_q_norm.weight", {d.head_dim}, 1.5F);
    AddF32T(b, p + "attn_k_norm.weight", {d.head_dim}, 1.5F);
    AddQ8_0(b, p + "ffn_gate_inp.weight", d.E, d.H, 161 + il);
    AddF32T(b, p + "ffn_gate_inp_shexp.weight", {d.H}, 0.75F);
    // Stacked experts: torch [E, out, in].
    b.AddTensor(p + "ffn_gate_exps.weight",
                {static_cast<uint64_t>(d.H), static_cast<uint64_t>(d.I),
                 static_cast<uint64_t>(d.E)},
                kQ8_0, Q8_0Bytes(d.E * d.I, d.H, 171 + il));
    b.AddTensor(p + "ffn_up_exps.weight",
                {static_cast<uint64_t>(d.H), static_cast<uint64_t>(d.I),
                 static_cast<uint64_t>(d.E)},
                kQ8_0, Q8_0Bytes(d.E * d.I, d.H, 181 + il));
    b.AddTensor(p + "ffn_down_exps.weight",
                {static_cast<uint64_t>(d.I), static_cast<uint64_t>(d.H),
                 static_cast<uint64_t>(d.E)},
                kQ8_0, Q8_0Bytes(d.E * d.H, d.I, 191 + il));
    AddQ8_0(b, p + "ffn_gate_shexp.weight", d.Is, d.H, 201 + il);
    AddQ8_0(b, p + "ffn_up_shexp.weight", d.Is, d.H, 211 + il);
    AddQ8_0(b, p + "ffn_down_shexp.weight", d.H, d.Is, 221 + il);
  }
  return b.Build();
}

}  // namespace

TEST_CASE("loader keep-quant experts load as a lossless stacked tower (A3)") {
  const MoeDims d;
  const TempFile f(BuildMoeQ8Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  const GgufLoadPolicy off;
  GgufLoadPolicy on = KeepQuantOn();
  std::set<std::string> routed;
  std::set<std::string> kept_names;
  on.audit = [&](const std::string& n, GgufTensorRole, GgufResidency r) {
    routed.insert(n);
    if (r == GgufResidency::kKeepQuant) kept_names.insert(n);
  };
  const vllm::Qwen3_5MoeWeights base = vllm::LoadQwen3_5MoeFromGguf(g, c, &off);
  const vllm::Qwen3_5MoeWeights kept = vllm::LoadQwen3_5MoeFromGguf(g, c, &on);

  // Totality on the MoE tensor list too.
  std::set<std::string> in_file;
  for (const vllm::GgufTensorInfo& t : g.Tensors()) in_file.insert(t.name);
  CHECK(routed == in_file);

  for (int64_t il = 0; il < d.n_layer; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const auto& kl = kept.layers[static_cast<size_t>(il)].moe;
    const auto& bl = base.layers[static_cast<size_t>(il)].moe;
    // A3: keep-quant no longer splits per-expert — the whole [E*out,in] tower loads
    // as ONE stacked block tensor (expert_*_kq); the per-expert vector is EMPTY.
    CHECK(kl.expert_gate.empty());
    CHECK(kl.expert_up.empty());
    CHECK(kl.expert_down.empty());
    struct Stack {
      const vllm::OwnedTensor* kq;              // A3 stacked [E*out,in] (kept)
      const std::vector<vllm::OwnedTensor>* b;  // base expand-bf16 per-expert
      std::string name;
    };
    const Stack stacks[] = {
        {&kl.expert_gate_kq, &bl.expert_gate, p + "ffn_gate_exps.weight"},
        {&kl.expert_up_kq, &bl.expert_up, p + "ffn_up_exps.weight"},
        {&kl.expert_down_kq, &bl.expert_down, p + "ffn_down_exps.weight"},
    };
    for (const Stack& s : stacks) {
      CAPTURE(s.name);
      CHECK(kept_names.count(s.name) == 1);
      const vllm::GgufTensorInfo& t = g.Get(s.name);
      const int64_t out_dim = t.shape[1];
      const int64_t in_dim = t.shape[2];
      REQUIRE(s.b->size() == static_cast<size_t>(d.E));
      const size_t per_bytes =
          vt::RowSizeBytes(vt::DType::kQ8_0, out_dim * in_dim);
      // A3: the whole tower is ONE stacked [E*out, in] Q8_0 block tensor, byte-for-byte
      // the file's tensor (a wrong offset / short load is caught by the memcmp below).
      CHECK(s.kq->dtype == vt::DType::kQ8_0);
      CHECK(s.kq->nk == true);
      CHECK(s.kq->shape[0] == d.E * out_dim);
      CHECK(s.kq->shape[1] == in_dim);
      REQUIRE(s.kq->bytes.size() == static_cast<size_t>(d.E) * per_bytes);
      CHECK(std::memcmp(s.kq->bytes.data(), t.data,
                        static_cast<size_t>(d.E) * per_bytes) == 0);
      for (int64_t e = 0; e < d.E; ++e) {
        CAPTURE(e);
        const vllm::OwnedTensor& bexp = (*s.b)[static_cast<size_t>(e)];
        // Each expert slice of the stacked tower dequants to the base bf16 expert
        // byte for byte (a wrong per-expert offset is caught here).
        const std::vector<uint16_t> rehydrated = ExpandAndTranspose(
            kQ8_0, s.kq->bytes.data() + static_cast<size_t>(e) * per_bytes, out_dim,
            in_dim);
        REQUIRE(rehydrated.size() * sizeof(uint16_t) == bexp.bytes.size());
        CHECK(std::memcmp(rehydrated.data(), bexp.bytes.data(),
                          bexp.bytes.size()) == 0);
      }
    }
    // The 1-D shared gate is a vector: never kept, and bit-identical.
    CHECK(kept_names.count(p + "ffn_gate_inp_shexp.weight") == 0);
    CHECK(kl.shared_gate.bytes == bl.shared_gate.bytes);
  }
}

// CIQ G4. The production default is no longer "expand everything": where the
// device can run the quantized GEMM, an env-driven load must equal a load
// under an explicitly-ON policy, block dtypes and orientation included.
TEST_CASE("production default is keep-quant wherever the quant GEMM exists") {
  ::unsetenv("VT_CPU_REF");
  ::unsetenv("VT_GGUF_KEEP_QUANT");
  const DenseDims d;
  const TempFile f(BuildDenseQ8Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  // The explicit policy must mirror FromEnv's FULL CPU default, otherwise the
  // byte comparison below is not apples-to-apples. On an i8mm box (dgx aarch64)
  // FromEnv turns quant_repack ON, which PERMUTES the q8_0 bytes, so an `expect`
  // that leaves it off would differ from the env load on the repacked weights
  // (this bit was invisible on x86 CI, where repack is off — L6 fix).
  GgufLoadPolicy expect;
  expect.keep_quant = vllm::GgufQuantComputeAvailable(vt::DeviceType::kCPU);
  expect.expand_nk = expect.keep_quant;
  // L7: keep_f16 defaults ON with the quant path, so `expect` must mirror it or
  // the byte comparison diverges on any F16 verbatim weight in the fixture.
  expect.keep_f16 = expect.expand_nk;
  expect.mmap_residency = expect.keep_quant;
  expect.share_tied_head = expect.keep_quant;
  expect.gdn_expand_nk = expect.keep_quant;
  // #2406: the mirror carries the DEVICE term too, or an i8mm host would
  // compare an env load (kCPU-resolved) against a policy that decided without
  // one. `kCPU` is the device this whole case resolves for.
  expect.quant_repack = vllm::QuantRepackForDevice(
      expect.keep_quant, /*cpu_ref=*/false, vt::cpu::QuantRepackActive(),
      vt::DeviceType::kCPU);

  const vllm::Qwen3_5DenseWeights from_env =
      vllm::LoadQwen3_5DenseFromGguf(g, c, /*policy=*/nullptr);
  const vllm::Qwen3_5DenseWeights from_policy =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &expect);

  const vt::DType want =
      expect.keep_quant ? vt::DType::kQ8_0 : vt::DType::kBF16;
  CHECK(from_env.lm_head.bytes == from_policy.lm_head.bytes);
  CHECK(from_env.lm_head.dtype == want);
  CHECK(from_env.lm_head.nk == expect.keep_quant);
  REQUIRE(from_env.layers.size() == from_policy.layers.size());
  for (size_t il = 0; il < from_env.layers.size(); ++il) {
    const auto& a = from_env.layers[il];
    const auto& b = from_policy.layers[il];
    CHECK(a.attn.q_proj.bytes == b.attn.q_proj.bytes);
    CHECK(a.mlp.down_proj.bytes == b.mlp.down_proj.bytes);
    CHECK(a.attn.q_proj.dtype == want);
    CHECK(a.attn.q_proj.nk == expect.keep_quant);
  }
}

// ===========================================================================
// G4 — orientation: an EXPANDED matmul weight keeps the file's [N, K] order.
// ===========================================================================

// The measured 1.3-3.0x lever. Its correctness argument is that the two CPU
// GEMM kernels differ ONLY in the weight offset, so this must be BIT-EXACT
// against the transposed load, element by element — not merely "close".
TEST_CASE("expand_nk is the untransposed view of the SAME expanded weight") {
  const DenseDims d;
  const TempFile f(BuildDenseQ8Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  // keep_quant OFF so every matmul weight EXPANDS; only the orientation moves.
  const GgufLoadPolicy transposed;
  GgufLoadPolicy raw;
  raw.expand_nk = true;

  const vllm::Qwen3_5DenseWeights t =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &transposed);
  const vllm::Qwen3_5DenseWeights r = vllm::LoadQwen3_5DenseFromGguf(g, c, &raw);

  auto same_weight = [](const vllm::OwnedTensor& kn,
                        const vllm::OwnedTensor& nk) {
    REQUIRE(kn.rank == 2);
    REQUIRE(nk.rank == 2);
    CHECK_FALSE(kn.nk);
    CHECK(nk.nk);
    CHECK(kn.dtype == vt::DType::kBF16);
    CHECK(nk.dtype == vt::DType::kBF16);
    // [K, N] vs [N, K]: the same matrix, transposed.
    REQUIRE(kn.shape[0] == nk.shape[1]);
    REQUIRE(kn.shape[1] == nk.shape[0]);
    const auto* a = reinterpret_cast<const uint16_t*>(kn.bytes.data());
    const auto* b = reinterpret_cast<const uint16_t*>(nk.bytes.data());
    const int64_t K = kn.shape[0], N = kn.shape[1];
    REQUIRE(kn.bytes.size() == nk.bytes.size());
    for (int64_t i = 0; i < K; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        // A single mismatched element fails loudly with its index.
        if (a[i * N + j] != b[j * K + i]) {
          CAPTURE(i);
          CAPTURE(j);
          REQUIRE(a[i * N + j] == b[j * K + i]);
        }
      }
    }
  };

  same_weight(t.lm_head, r.lm_head);
  REQUIRE(t.layers.size() == r.layers.size());
  for (size_t il = 0; il < t.layers.size(); ++il) {
    CAPTURE(il);
    same_weight(t.layers[il].attn.q_proj, r.layers[il].attn.q_proj);
    same_weight(t.layers[il].attn.o_proj, r.layers[il].attn.o_proj);
    same_weight(t.layers[il].mlp.gate_proj, r.layers[il].mlp.gate_proj);
    same_weight(t.layers[il].mlp.down_proj, r.layers[il].mlp.down_proj);
    // Tensors that are NOT matmul weights are untouched by the orientation
    // switch — it must not leak into the norm/embedding/conv paths.
    CHECK(t.layers[il].input_layernorm.bytes ==
          r.layers[il].input_layernorm.bytes);
  }
  CHECK(t.embed_tokens.bytes == r.embed_tokens.bytes);
  CHECK_FALSE(r.embed_tokens.nk);
  CHECK(t.final_norm.bytes == r.final_norm.bytes);
}

// ===========================================================================
// G4 — the ROUTING itself: vt::MatmulBT consumes a block weight.
// ===========================================================================

// The whole enablement in one assertion: a block-typed [N, K] weight handed to
// the SAME entry point every model matmul helper already calls must produce
// the quantized GEMM's answer, not throw. Before G4 this threw
// "matmul_bt: float inputs and f32/bf16 output required".
TEST_CASE("vt::MatmulBT routes a block-quantized weight to MatmulBTQuant") {
  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue q = backend.CreateQueue();

  const int64_t M = 3, N = 5, K = 64;
  std::vector<float> act(static_cast<size_t>(M) * K);
  for (size_t i = 0; i < act.size(); ++i) {
    act[i] = 0.25F * static_cast<float>((i % 17)) - 1.5F;
  }
  std::vector<float> w(static_cast<size_t>(N) * K);
  for (size_t i = 0; i < w.size(); ++i) {
    w[i] = 0.125F * static_cast<float>((i % 23)) - 1.0F;
  }
  // Quantize the weight rows to Q8_0 exactly as a GGUF file stores them.
  std::vector<uint8_t> blocks(vt::RowSizeBytes(vt::DType::kQ8_0, K) *
                              static_cast<size_t>(N));
  for (int64_t j = 0; j < N; ++j) {
    vt::cpu::QuantTraits(vt::DType::kQ8_0)
        .from_float(w.data() + j * K,
                    blocks.data() + static_cast<size_t>(j) *
                                        vt::RowSizeBytes(vt::DType::kQ8_0, K),
                    K);
  }

  vt::Tensor a = vt::Tensor::Contiguous(act.data(), vt::DType::kF32,
                                        q.device, {M, K});
  vt::Tensor b = vt::Tensor::Contiguous(blocks.data(), vt::DType::kQ8_0,
                                        q.device, {N, K});
  std::vector<float> got(static_cast<size_t>(M) * N, 0.0F);
  std::vector<float> want(static_cast<size_t>(M) * N, 0.0F);
  vt::Tensor og = vt::Tensor::Contiguous(got.data(), vt::DType::kF32,
                                         q.device, {M, N});
  vt::Tensor ow = vt::Tensor::Contiguous(want.data(), vt::DType::kF32,
                                         q.device, {M, N});

  vt::MatmulBT(q, og, a, b);        // the routed call site
  vt::MatmulBTQuant(q, ow, a, b);   // the op it must have reached
  backend.Synchronize(q);

  // Bit-identical: MatmulBT must DELEGATE, not approximate.
  CHECK(std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0);
  // ...and the answer is a real GEMM, not zeros.
  bool nonzero = false;
  for (float v : got) nonzero = nonzero || v != 0.0F;
  CHECK(nonzero);
}

// ===========================================================================
// L5 — mmap residency and tied-head sharing.
//
// Both replace a COPY with a VIEW, so both raise the same two questions: are
// the bytes identical to the copy arm, and can the view outlive what it views?
// The second is the real hazard the spec names, so it is gated by actually
// destroying the producer rather than by reasoning about it.
// ===========================================================================

namespace {

GgufLoadPolicy KeepQuantMmap() {
  GgufLoadPolicy p = KeepQuantOn();
  p.mmap_residency = true;
  return p;
}

// The L5 production CPU shape: keep-quant + untransposed expansion + both L5
// residency refinements, with keep_f16 held OFF so these cases exercise the bf16
// tied-head SHARE path specifically (VT_GGUF_KEEP_F16=0; the L7 default-on f16
// share is covered by the KeepF16On() cases below).
GgufLoadPolicy ProductionCpu() {
  GgufLoadPolicy p = KeepQuantOn();
  p.expand_nk = true;
  p.mmap_residency = true;
  p.share_tied_head = true;
  return p;
}

bool FileHasTensor(const vllm::GgufFile& g, const std::string& name) {
  for (const vllm::GgufTensorInfo& t : g.Tensors()) {
    if (t.name == name) return true;
  }
  return false;
}

}  // namespace

TEST_CASE("mmap residency is byte-identical to the copy arm, and borrows") {
  const DenseDims d;
  const TempFile f(BuildDenseQ8Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  GgufLoadPolicy copy = KeepQuantOn();    // L2 arm: blocks copied out
  GgufLoadPolicy mmap = KeepQuantMmap();  // L5 arm: blocks viewed in place
  const vllm::Qwen3_5DenseWeights wc =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &copy);
  const vllm::Qwen3_5DenseWeights wm =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &mmap);

  // The kept weights: same dtype/shape/orientation, same BYTES, different
  // residency — and the mmap arm's bytes are literally the file's own.
  const vllm::GgufTensorInfo& t = g.Get("output.weight");
  CHECK(wc.lm_head.bytes.borrowed() == false);
  CHECK(wm.lm_head.bytes.borrowed() == true);
  CHECK(wm.lm_head.dtype == wc.lm_head.dtype);
  CHECK(wm.lm_head.nk == wc.lm_head.nk);
  CHECK(wm.lm_head.shape[0] == wc.lm_head.shape[0]);
  CHECK(wm.lm_head.shape[1] == wc.lm_head.shape[1]);
  CHECK(wm.lm_head.bytes == wc.lm_head.bytes);
  CHECK(wm.lm_head.bytes.data() == t.data);  // IN PLACE, not a copy
  CHECK(wm.lm_head.View().data == const_cast<uint8_t*>(t.data));

  for (int64_t il = 0; il < d.n_layer; ++il) {
    CAPTURE(il);
    const auto& lc = wc.layers[static_cast<size_t>(il)];
    const auto& lm = wm.layers[static_cast<size_t>(il)];
    CHECK(lm.attn.q_proj.bytes.borrowed());
    CHECK(lm.attn.q_proj.bytes == lc.attn.q_proj.bytes);
    CHECK(lm.attn.o_proj.bytes == lc.attn.o_proj.bytes);
    CHECK(lm.mlp.down_proj.bytes == lc.mlp.down_proj.bytes);
    // An EXPANDED tensor owns its bytes in BOTH arms — mmap residency applies
    // to kept blocks only and must not leak into the expansion path.
    CHECK(lm.input_layernorm.bytes.borrowed() == false);
    CHECK(lm.input_layernorm.bytes == lc.input_layernorm.bytes);
  }
}

TEST_CASE("a borrowed weight OUTLIVES the GgufFile and the file itself") {
  const DenseDims d;
  std::vector<uint8_t> expected;
  std::unique_ptr<vllm::Qwen3_5DenseWeights> w;
  int64_t numel = 0;

  {
    // Both the reader and the on-disk file go away inside this scope. The
    // weights must stay readable: the mapping is refcounted (GgufMapping), and
    // a POSIX mapping survives unlink of its path.
    const TempFile f(BuildDenseQ8Gguf(d));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    const vllm::HfConfig c = vllm::HfConfigFromGguf(g);
    GgufLoadPolicy mmap = KeepQuantMmap();
    w = std::make_unique<vllm::Qwen3_5DenseWeights>(
        vllm::LoadQwen3_5DenseFromGguf(g, c, &mmap));
    const vllm::GgufTensorInfo& t = g.Get("output.weight");
    expected.assign(t.data, t.data + t.nbytes);
    numel = w->lm_head.shape[0] * w->lm_head.shape[1];
  }

  REQUIRE(w->lm_head.bytes.borrowed());
  REQUIRE(w->lm_head.bytes.size() == expected.size());
  // Reading through the view that would be dangling if it were not refcounted.
  CHECK(std::memcmp(w->lm_head.bytes.data(), expected.data(),
                    expected.size()) == 0);
  // ...and decoding it still gives the right answer, i.e. these are the file's
  // pages and not recycled memory that happens to compare equal.
  const std::vector<uint16_t> re =
      vllm::DequantGgufRowToBf16(kQ8_0, w->lm_head.bytes.data(), numel);
  const std::vector<uint16_t> ref =
      vllm::DequantGgufRowToBf16(kQ8_0, expected.data(), numel);
  CHECK(re == ref);

  w.reset();  // last reference: the mapping is unmapped exactly once, here
}

TEST_CASE("tied lm_head SHARES the embedding expansion instead of copying it") {
  const DenseDims d;
  const TempFile f(BuildDenseQ8Gguf(d, /*tied=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);
  REQUIRE_FALSE(FileHasTensor(g, "output.weight"));

  GgufLoadPolicy shared = ProductionCpu();
  GgufLoadPolicy unshared = ProductionCpu();
  unshared.share_tied_head = false;
  const vllm::Qwen3_5DenseWeights ws =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &shared);
  const vllm::Qwen3_5DenseWeights wu =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &unshared);

  // THE GATE: sharing changes WHERE the bytes live, never WHAT they are.
  CHECK(ws.embed_tokens.bytes == wu.embed_tokens.bytes);
  CHECK(ws.lm_head.bytes == wu.lm_head.bytes);
  CHECK(ws.lm_head.dtype == wu.lm_head.dtype);
  CHECK(ws.lm_head.nk == wu.lm_head.nk);
  CHECK(ws.lm_head.rank == wu.lm_head.rank);
  CHECK(ws.lm_head.shape[0] == wu.lm_head.shape[0]);
  CHECK(ws.lm_head.shape[1] == wu.lm_head.shape[1]);

  // ONE buffer in the shared arm, TWO in the unshared one. This is the whole
  // point of the row: a failure here means the second vocab matrix is still
  // being paid for.
  CHECK(ws.embed_tokens.bytes.data() == ws.lm_head.bytes.data());
  CHECK(wu.embed_tokens.bytes.data() != wu.lm_head.bytes.data());
  CHECK(ws.embed_tokens.bytes.borrowed());
  CHECK(ws.lm_head.bytes.borrowed());

  // The head is in the file's own [N = vocab, K = H] order, which is the ONLY
  // orientation in which the two byte images can coincide...
  CHECK(ws.lm_head.nk == true);
  CHECK(ws.lm_head.shape[0] == d.vocab);
  CHECK(ws.lm_head.shape[1] == d.H);
  // ...and the gather table keeps its own metadata over the same bytes.
  CHECK(ws.embed_tokens.nk == false);
  CHECK(ws.embed_tokens.shape[0] == d.vocab);
  CHECK(ws.embed_tokens.shape[1] == d.H);
}

TEST_CASE("a shared tied head is freed ONCE: either half may die first") {
  const DenseDims d;
  const TempFile f(BuildDenseQ8Gguf(d, /*tied=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);
  GgufLoadPolicy shared = ProductionCpu();

  // Drop the EMBEDDING first, then read the head.
  {
    vllm::Qwen3_5DenseWeights w = vllm::LoadQwen3_5DenseFromGguf(g, c, &shared);
    const vllm::OwnedBytes& head = w.lm_head.bytes;
    const std::vector<uint8_t> want(head.begin(), head.end());
    w.embed_tokens.bytes.Reset();
    REQUIRE(head.size() == want.size());
    CHECK(std::memcmp(head.data(), want.data(), want.size()) == 0);
  }
  // Drop the HEAD first, then read the embedding.
  {
    vllm::Qwen3_5DenseWeights w = vllm::LoadQwen3_5DenseFromGguf(g, c, &shared);
    const vllm::OwnedBytes& emb = w.embed_tokens.bytes;
    const std::vector<uint8_t> want(emb.begin(), emb.end());
    w.lm_head.bytes.Reset();
    REQUIRE(emb.size() == want.size());
    CHECK(std::memcmp(emb.data(), want.data(), want.size()) == 0);
  }
  // ReleaseHost() on a borrowed weight drops only the reference: it must not
  // madvise pages it does not own, and the other half must stay readable.
  {
    vllm::Qwen3_5DenseWeights w = vllm::LoadQwen3_5DenseFromGguf(g, c, &shared);
    const vllm::OwnedBytes& head = w.lm_head.bytes;
    const std::vector<uint8_t> want(head.begin(), head.end());
    w.embed_tokens.ReleaseHost();
    CHECK(w.embed_tokens.bytes.empty());
    REQUIRE(head.size() == want.size());
    CHECK(std::memcmp(head.data(), want.data(), want.size()) == 0);
  }
}

TEST_CASE("the oracle path shares NOTHING and borrows NOTHING") {
  const DenseDims d;
  const TempFile f(BuildDenseQ8Gguf(d, /*tied=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  // VT_CPU_REF forces every L5 refinement off (see FromEnv), so the load is the
  // historical two-buffer, transposed-head one, allocation for allocation.
  GgufLoadPolicy oracle = ProductionCpu();
  oracle.cpu_ref = true;
  oracle.expand_nk = false;
  oracle.mmap_residency = false;
  oracle.share_tied_head = false;
  const vllm::Qwen3_5DenseWeights wo =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &oracle);

  CHECK(wo.embed_tokens.bytes.borrowed() == false);
  CHECK(wo.lm_head.bytes.borrowed() == false);
  CHECK(wo.embed_tokens.bytes.data() != wo.lm_head.bytes.data());
  // Transposed to Matmul-B [K = H, N = vocab] — genuinely different bytes from
  // the gather table, which is precisely why this arm cannot share.
  CHECK(wo.lm_head.nk == false);
  CHECK(wo.lm_head.shape[0] == d.H);
  CHECK(wo.lm_head.shape[1] == d.vocab);
  CHECK(wo.lm_head.bytes != wo.embed_tokens.bytes);
}

TEST_CASE("FromEnv derives both L5 switches, and VT_CPU_REF overrides them") {
  ::unsetenv("VT_CPU_REF");
  ::unsetenv("VT_GGUF_MMAP");
  ::unsetenv("VT_GGUF_SHARE_TIED_HEAD");
  ::setenv("VT_GGUF_KEEP_QUANT", "1", 1);
  {
    const GgufLoadPolicy p = GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU);
    CHECK(p.keep_quant);
    CHECK(p.expand_nk);
    CHECK(p.mmap_residency);   // defaults ON with keep-quant
    CHECK(p.share_tied_head);  // defaults ON with expand_nk
  }
  // Each is independently opt-out-able, so the copy and duplicate arms stay
  // A/B-able against the production default.
  for (const char* off : {"0", "false", "off", ""}) {
    CAPTURE(off);
    ::setenv("VT_GGUF_MMAP", off, 1);
    ::setenv("VT_GGUF_SHARE_TIED_HEAD", off, 1);
    const GgufLoadPolicy p = GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU);
    CHECK(p.keep_quant);
    CHECK_FALSE(p.mmap_residency);
    CHECK_FALSE(p.share_tied_head);
  }
  // Turning keep-quant off takes both with it: there is nothing to borrow when
  // every weight expands, and the head is transposed again.
  ::unsetenv("VT_GGUF_MMAP");
  ::unsetenv("VT_GGUF_SHARE_TIED_HEAD");
  ::setenv("VT_GGUF_KEEP_QUANT", "0", 1);
  {
    const GgufLoadPolicy p = GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU);
    CHECK_FALSE(p.mmap_residency);
    CHECK_FALSE(p.share_tied_head);
  }
  // The oracle switch wins over both, even when they are asked for explicitly.
  ::setenv("VT_GGUF_KEEP_QUANT", "1", 1);
  ::setenv("VT_GGUF_MMAP", "1", 1);
  ::setenv("VT_GGUF_SHARE_TIED_HEAD", "1", 1);
  ::setenv("VT_CPU_REF", "1", 1);
  {
    const GgufLoadPolicy p = GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU);
    CHECK(p.cpu_ref);
    CHECK_FALSE(p.expand_nk);
    CHECK_FALSE(p.mmap_residency);
    CHECK_FALSE(p.share_tied_head);
  }
  ::unsetenv("VT_CPU_REF");
  ::unsetenv("VT_GGUF_MMAP");
  ::unsetenv("VT_GGUF_SHARE_TIED_HEAD");
  ::unsetenv("VT_GGUF_KEEP_QUANT");
}

TEST_CASE("OwnedBytes refuses to MUTATE a borrowed buffer") {
  auto holder = std::make_shared<const std::vector<uint8_t>>(
      std::vector<uint8_t>{1, 2, 3, 4});
  vllm::OwnedBytes b =
      vllm::OwnedBytes::Borrow(holder->data(), holder->size(), holder);
  REQUIRE(b.borrowed());
  REQUIRE(b.size() == 4);
  const vllm::OwnedBytes& rb = b;
  CHECK(rb.data()[0] == 1);  // a const read is fine

  // Every write path is a loud failure instead of a silent write through
  // read-only memory. That is what makes the borrowed residency safe to
  // introduce underneath ~300 existing `.bytes` call sites.
  CHECK_THROWS_AS(b.resize(8), std::runtime_error);
  CHECK_THROWS_AS(b.assign(size_t{2}, uint8_t{0}), std::runtime_error);
  CHECK_THROWS_AS(b.Share(), std::runtime_error);
  // Element access, const or not, is NOT rejected: constness does not
  // distinguish a read from a write, and this tree reads weights through
  // non-const handles everywhere. Only the STRUCTURAL mutations above — the
  // ones that would move the buffer out from under the owner — are refused.
  CHECK(b.data() == rb.data());

  // A borrow with no keep-alive is unrepresentable.
  CHECK_THROWS_AS(vllm::OwnedBytes::Borrow(holder->data(), 4, nullptr),
                  std::runtime_error);

  b.Reset();
  CHECK_FALSE(b.borrowed());
  CHECK(b.empty());
  b.resize(2);  // and it is reusable as an owned buffer afterwards
  CHECK(b.size() == 2);
}

TEST_CASE("OwnedBytes::Share hands the SAME bytes to a second viewer") {
  vllm::OwnedBytes a(std::vector<uint8_t>{9, 8, 7, 6, 5});
  const std::vector<uint8_t> want(a.begin(), a.end());
  REQUIRE_FALSE(a.borrowed());

  std::shared_ptr<const void> owner = a.Share();
  REQUIRE(a.borrowed());
  const vllm::OwnedBytes& ra = a;
  vllm::OwnedBytes b = vllm::OwnedBytes::Borrow(ra.data(), ra.size(), owner);
  const vllm::OwnedBytes& rb = b;

  CHECK(ra.data() == rb.data());  // one buffer, two viewers
  CHECK(a == b);
  REQUIRE(ra.size() == want.size());
  CHECK(std::memcmp(ra.data(), want.data(), want.size()) == 0);  // no copy

  owner.reset();
  a.Reset();  // the first viewer goes; the second still reads
  REQUIRE(rb.size() == want.size());
  CHECK(std::memcmp(rb.data(), want.data(), want.size()) == 0);
}

// ===========================================================================
// L6 — keep-f16 residency. An F16 file weight stays F16 (no bf16 re-expansion),
// resident in the file's own [N, K] order, consumed by the elementwise f16 GEMM.
// The residency is byte-lossless; the compute dtype changes bf16 -> f16, which
// the engine gate (test_qwen36_gguf_engine, same-file llama.cpp oracle) covers.
// These units gate the LOADER: routing, byte-identity, borrow/lifetime, sharing.
// ===========================================================================

namespace {

// F16 bit patterns for a torch [n, k] tensor, ggml dims reversed (ne0 = k).
std::string F16Bytes(int64_t n, float base) {
  std::string s;
  s.reserve(static_cast<size_t>(n) * 2);
  for (int64_t i = 0; i < n; ++i) {
    const float v = base + 0.25F * static_cast<float>(i % 7) -
                    0.125F * static_cast<float>(i % 3);
    const uint16_t h = vt::F32ToF16(v);
    s.push_back(static_cast<char>(h & 0xFF));
    s.push_back(static_cast<char>((h >> 8) & 0xFF));
  }
  return s;
}

void AddF16T(GgufModelBuilder& b, const std::string& name, int64_t out_dim,
             int64_t in_dim, float base) {
  b.AddTensor(name,
              {static_cast<uint64_t>(in_dim), static_cast<uint64_t>(out_dim)},
              kF16, F16Bytes(out_dim * in_dim, base));
}

// A tiny DENSE GGUF whose GEMM weights AND token_embd are F16 (norms stay F32) —
// the "56 f16 tensors" shape of the real 2B bench file, the case keep-f16 exists
// for. `tied` omits output.weight so the head IS the f16 token_embd.
std::string BuildDenseF16Gguf(const DenseDims& d, bool tied = false, bool gated = false) {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "qwen35"));
  b.AddKv(U32Kv("qwen35.embedding_length", static_cast<uint32_t>(d.H)));
  b.AddKv(U32Kv("qwen35.block_count", static_cast<uint32_t>(d.n_layer)));
  b.AddKv(U32Kv("qwen35.attention.head_count", static_cast<uint32_t>(d.n_head)));
  b.AddKv(U32Kv("qwen35.attention.head_count_kv",
                static_cast<uint32_t>(d.n_head_kv)));
  b.AddKv(U32Kv("qwen35.attention.key_length",
                static_cast<uint32_t>(d.head_dim)));
  b.AddKv(U32Kv("qwen35.feed_forward_length", static_cast<uint32_t>(d.I)));
  b.AddKv(F32Kv("qwen35.attention.layer_norm_rms_epsilon", 1e-6F));
  b.AddKv(F32Kv("qwen35.rope.freq_base", 1000000.0F));
  b.AddKv(U32Kv("qwen35.full_attention_interval", 1));
  b.AddKv(U32Kv("qwen35.context_length", 4096));

  AddF16T(b, "token_embd.weight", d.vocab, d.H, 0.5F);
  AddF32T(b, "output_norm.weight", {d.H}, 1.5F);
  if (!tied) AddF16T(b, "output.weight", d.vocab, d.H, 0.6F);
  for (int64_t il = 0; il < d.n_layer; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    AddF32T(b, p + "attn_norm.weight", {d.H}, 1.25F);
    AddF32T(b, p + "post_attention_norm.weight", {d.H}, 1.75F);
    AddF16T(b, p + "attn_q.weight", (gated ? 2 : 1) * d.n_head * d.head_dim, d.H, 0.11F);
    AddF16T(b, p + "attn_k.weight", d.n_head_kv * d.head_dim, d.H, 0.13F);
    AddF16T(b, p + "attn_v.weight", d.n_head_kv * d.head_dim, d.H, 0.17F);
    AddF16T(b, p + "attn_output.weight", d.H, d.n_head * d.head_dim, 0.19F);
    AddF32T(b, p + "attn_q_norm.weight", {d.head_dim}, 1.5F);
    AddF32T(b, p + "attn_k_norm.weight", {d.head_dim}, 1.5F);
    AddF16T(b, p + "ffn_gate.weight", d.I, d.H, 0.21F);
    AddF16T(b, p + "ffn_up.weight", d.I, d.H, 0.23F);
    AddF16T(b, p + "ffn_down.weight", d.H, d.I, 0.27F);
  }
  return b.Build();
}

// keep-f16 forced ON, copy residency (mmap off) — the A/B analogue of KeepQuantOn.
GgufLoadPolicy KeepF16On() {
  GgufLoadPolicy p;
  p.keep_quant = true;   // GgufQuantComputeAvailable() would set this on CPU
  p.expand_nk = true;
  p.keep_f16 = true;
  return p;
}

}  // namespace

TEST_CASE("keep-f16 routing: F16 matmul/expert/embed keep, others expand") {
  const uint32_t f16 = kF16;
  // The keep-eligible roles + right rank keep f16 ONLY when keep_f16 is on.
  CHECK(RouteGgufTensor(false, true, false, false, GgufTensorRole::kMatmulWeight,
                        f16, {8, 64}, vt::DeviceType::kCPU) == GgufResidency::kKeepF16);
  CHECK(RouteGgufTensor(false, true, false, false,
                        GgufTensorRole::kStackedExpertWeight, f16,
                        {2, 8, 64}, vt::DeviceType::kCPU) ==
        GgufResidency::kKeepF16);
  // The embedding table keeps f16 too (a gather widens f16 like bf16); this is
  // what lets a tied token_embd/lm_head be ONE resident f16 vocab matrix.
  CHECK(RouteGgufTensor(false, true, false, false, GgufTensorRole::kEmbeddingTable,
                        f16, {32, 64}, vt::DeviceType::kCPU) == GgufResidency::kKeepF16);
  // Value/layout-rewritten, conv and vector roles NEVER keep, even f16.
  for (GgufTensorRole role :
       {GgufTensorRole::kTransformedWeight, GgufTensorRole::kConvWeight,
        GgufTensorRole::kVector}) {
    CAPTURE(vllm::Name(role));
    CHECK(RouteGgufTensor(false, true, false, false, role, f16, {8, 64}, vt::DeviceType::kCPU) ==
          GgufResidency::kExpandBf16);
  }
  // Wrong rank never keeps.
  CHECK(RouteGgufTensor(false, true, false, false, GgufTensorRole::kMatmulWeight,
                        f16, {64}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
  // keep_f16 OFF -> expand; VT_CPU_REF -> expand; a non-f16 type -> no keep-f16.
  CHECK(RouteGgufTensor(false, false, false, false, GgufTensorRole::kMatmulWeight,
                        f16, {8, 64}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
  CHECK(RouteGgufTensor(true, true, false, true, GgufTensorRole::kMatmulWeight, f16,
                        {8, 64}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
  CHECK(RouteGgufTensor(false, true, false, false, GgufTensorRole::kMatmulWeight,
                        kQ8_0, {8, 64}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
  // keep-quant WINS over keep-f16 when both are on and the encoding is a block
  // type (they are mutually exclusive by encoding, so this only asserts order).
  CHECK(RouteGgufTensor(true, true, false, false, GgufTensorRole::kMatmulWeight,
                        kQ8_0, {8, 64}, vt::DeviceType::kCPU) == GgufResidency::kKeepQuant);
}

// `QUANT-GGUF-NVFP4` column C — the fp4 residency's routing contract.
TEST_CASE("nvfp4 routing: type 40 in a verbatim role gets the fp4 residency") {
  const uint32_t nvfp4 = 40;
  // The two verbatim-bytes roles, at their expected ranks, with K a whole
  // number of 64-element NVFP4 blocks.
  CHECK(RouteGgufTensor(false, false, /*nvfp4_fp4=*/true, false,
                        GgufTensorRole::kMatmulWeight, nvfp4,
                        {8, 128}, vt::DeviceType::kCPU) == GgufResidency::kNvfp4Fp4);
  CHECK(RouteGgufTensor(false, false, /*nvfp4_fp4=*/true, false,
                        GgufTensorRole::kStackedExpertWeight, nvfp4,
                        {2, 8, 128}, vt::DeviceType::kCPU) == GgufResidency::kNvfp4Fp4);
  // Independent of keep_quant / keep_f16: NVFP4 is neither a vt block dtype nor
  // F16, so those switches cannot produce (or suppress) this residency.
  CHECK(RouteGgufTensor(true, true, /*nvfp4_fp4=*/true, false,
                        GgufTensorRole::kMatmulWeight, nvfp4,
                        {8, 128}, vt::DeviceType::kCPU) == GgufResidency::kNvfp4Fp4);
  CHECK(RouteGgufTensor(true, true, /*nvfp4_fp4=*/false, false,
                        GgufTensorRole::kMatmulWeight, nvfp4,
                        {8, 128}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);

  // Roles whose value or layout is rewritten at load NEVER go fp4-resident —
  // this is what keeps the GDN V-head reorders and the (w-1) norm rewrite
  // correct on a file that stores them as NVFP4 (the 27B GGUF does).
  for (GgufTensorRole role :
       {GgufTensorRole::kTransformedWeight, GgufTensorRole::kEmbeddingTable,
        GgufTensorRole::kConvWeight, GgufTensorRole::kVector}) {
    CAPTURE(vllm::Name(role));
    CHECK(RouteGgufTensor(false, false, true, false, role, nvfp4, {8, 128}, vt::DeviceType::kCPU) ==
          GgufResidency::kExpandBf16);
    CHECK(RouteGgufTensor(false, false, true, false, role, nvfp4,
                          {2, 8, 128}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
  }
  // Wrong rank for the role never keeps.
  CHECK(RouteGgufTensor(false, false, true, false,
                        GgufTensorRole::kMatmulWeight, nvfp4,
                        {128}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
  CHECK(RouteGgufTensor(false, false, true, false,
                        GgufTensorRole::kStackedExpertWeight, nvfp4,
                        {8, 128}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
  // A ragged K cannot be repacked block-wise, so it expands. 128 keeps, 96 (a
  // multiple of 16 and 32 but NOT of the 64-element ggml block) does not.
  CHECK(RouteGgufTensor(false, false, true, false,
                        GgufTensorRole::kMatmulWeight, nvfp4,
                        {8, 96}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
  CHECK(RouteGgufTensor(false, false, true, false,
                        GgufTensorRole::kMatmulWeight, nvfp4,
                        {8, 0}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
  // The VT_CPU_REF oracle wins over the fp4 residency, like every other one.
  CHECK(RouteGgufTensor(false, false, true, /*cpu_ref=*/true,
                        GgufTensorRole::kMatmulWeight, nvfp4,
                        {8, 128}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
  // And the switch is NVFP4-specific: it does not divert any other encoding.
  for (const Encoding& e : kEncodings) {
    CAPTURE(e.name);
    CHECK(RouteGgufTensor(false, false, true, false,
                          GgufTensorRole::kMatmulWeight, e.ggml_type,
                          {8, e.k}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
  }
  CHECK(RouteGgufTensor(false, false, true, false,
                        GgufTensorRole::kMatmulWeight, kF16,
                        {8, 128}, vt::DeviceType::kCPU) == GgufResidency::kExpandBf16);
  CHECK(vllm::KeepNvfp4DType(40));
  CHECK_FALSE(vllm::KeepNvfp4DType(kQ8_0));
  CHECK_FALSE(vllm::KeepNvfp4DType(kF16));
  CHECK(std::string(vllm::Name(GgufResidency::kNvfp4Fp4)) == "nvfp4_fp4");
}

TEST_CASE("OwnGgufF16 keeps the file's f16 bytes verbatim, [N,K] nk=true") {
  const int64_t n = 5, k = 64;
  const TempFile f([&] {
    GgufModelBuilder b;
    b.AddKv(StrKv("general.architecture", "test"));
    b.AddTensor("w", {static_cast<uint64_t>(k), static_cast<uint64_t>(n)}, kF16,
                F16Bytes(n * k, 0.3F));
    return b.Build();
  }());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::GgufTensorInfo& t = g.Get("w");
  REQUIRE(t.nbytes == static_cast<size_t>(n) * k * 2);

  const vllm::OwnedTensor res = vllm::OwnGgufF16(t, n, k);
  CHECK(res.dtype == vt::DType::kF16);
  CHECK(res.rank == 2);
  CHECK(res.shape[0] == n);
  CHECK(res.shape[1] == k);
  CHECK(res.nk == true);
  REQUIRE(res.bytes.size() == t.nbytes);
  CHECK(std::memcmp(res.bytes.data(), t.data, t.nbytes) == 0);  // verbatim

  // nk=false builds a [vocab,H] gather table over the SAME bytes.
  const vllm::OwnedTensor emb = vllm::OwnGgufF16(t, n, k, 0, nullptr, false);
  CHECK(emb.nk == false);
  CHECK(std::memcmp(emb.bytes.data(), t.data, t.nbytes) == 0);

  // Guards: not f16, ragged span, bad slice.
  CHECK_THROWS(vllm::OwnGgufF16(t, n, k, /*row_offset=*/1));  // slice past span
  const std::string q = OneTensorGguf(kQ8_0, n, k, RandomBlockBytes(
      BlockBytesFor(kQ8_0, n * k), 7));
  const TempFile qf(q);
  const vllm::GgufFile qg = vllm::GgufFile::Open(qf.path());
  CHECK_THROWS(vllm::OwnGgufF16(qg.Get("w"), n, k));  // not f16
}

TEST_CASE("loader keeps F16 weights f16, byte-identical, and expands norms") {
  const DenseDims d;
  const TempFile f(BuildDenseF16Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  GgufLoadPolicy on = KeepF16On();
  const vllm::Qwen3_5DenseWeights w = vllm::LoadQwen3_5DenseFromGguf(g, c, &on);

  // The F16 GEMM weights STAY f16, in [N,K] nk=true, byte-identical to the file.
  const vllm::GgufTensorInfo& oh = g.Get("output.weight");
  CHECK(w.lm_head.dtype == vt::DType::kF16);
  CHECK(w.lm_head.nk == true);
  CHECK(std::memcmp(w.lm_head.bytes.data(), oh.data, oh.nbytes) == 0);
  for (int64_t il = 0; il < d.n_layer; ++il) {
    CAPTURE(il);
    const auto& la = w.layers[static_cast<size_t>(il)];
    for (const vllm::OwnedTensor* wt :
         {&la.attn.q_proj, &la.attn.o_proj, &la.mlp.gate_proj, &la.mlp.down_proj}) {
      CHECK(wt->dtype == vt::DType::kF16);
      CHECK(wt->nk == true);
    }
    const std::string p = "blk." + std::to_string(il) + ".";
    const vllm::GgufTensorInfo& gq = g.Get(p + "attn_q.weight");
    CHECK(std::memcmp(la.attn.q_proj.bytes.data(), gq.data, gq.nbytes) == 0);
    // Norms are F32 in the file and (w-1)-rewritten -> still bf16, never kept.
    CHECK(la.input_layernorm.dtype == vt::DType::kBF16);
  }
  // The embedding table (F16) is kept f16 as a [vocab,H] gather, nk=false.
  CHECK(w.embed_tokens.dtype == vt::DType::kF16);
  CHECK(w.embed_tokens.nk == false);
}

TEST_CASE("VT_CPU_REF / keep_f16=off reproduce the historical bf16 expansion") {
  const DenseDims d;
  const TempFile f(BuildDenseF16Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  // Oracle: everything expands to bf16, exactly as before L6.
  GgufLoadPolicy oracle;  // default-constructed = all-expand
  const vllm::Qwen3_5DenseWeights wo =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &oracle);
  CHECK(wo.lm_head.dtype == vt::DType::kBF16);
  CHECK(wo.embed_tokens.dtype == vt::DType::kBF16);
  CHECK(wo.layers[0].attn.q_proj.dtype == vt::DType::kBF16);

  // The kept-f16 weight carries the SAME information as the bf16 expansion, but
  // MORE precisely: dequantizing f16->bf16 from the resident f16 equals the
  // loader's bf16 expansion byte-for-byte (both round the file's f16 to bf16).
  GgufLoadPolicy on = KeepF16On();
  const vllm::Qwen3_5DenseWeights wk =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &on);
  const vllm::GgufTensorInfo& gq = g.Get("blk.0.attn_q.weight");
  const int64_t numel = gq.shape[0] * gq.shape[1];
  const std::vector<uint16_t> bf_from_f16 =
      vllm::DequantGgufRowToBf16(kF16, wk.layers[0].attn.q_proj.bytes.data(),
                                 numel);
  // wo.q_proj is the bf16 expansion in [N,K] (expand_nk off here -> [K,N]); to
  // compare information not layout, expand the SAME file tensor to bf16 [N,K].
  const std::vector<uint16_t> bf_from_file =
      vllm::DequantGgufRowToBf16(kF16, gq.data, numel);
  CHECK(bf_from_f16 == bf_from_file);
}

TEST_CASE("keep-f16 mmap residency BORROWS in place; copy arm OWNS") {
  const DenseDims d;
  const TempFile f(BuildDenseF16Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  GgufLoadPolicy copy = KeepF16On();       // mmap off
  GgufLoadPolicy mmap = KeepF16On();
  mmap.mmap_residency = true;
  const vllm::Qwen3_5DenseWeights wc = vllm::LoadQwen3_5DenseFromGguf(g, c, &copy);
  const vllm::Qwen3_5DenseWeights wm = vllm::LoadQwen3_5DenseFromGguf(g, c, &mmap);

  const vllm::GgufTensorInfo& oh = g.Get("output.weight");
  CHECK(wc.lm_head.bytes.borrowed() == false);
  CHECK(wm.lm_head.bytes.borrowed() == true);
  CHECK(wm.lm_head.bytes == wc.lm_head.bytes);      // same bytes
  CHECK(wm.lm_head.bytes.data() == oh.data);        // IN PLACE, the file's own
  CHECK(wm.lm_head.dtype == vt::DType::kF16);
}

TEST_CASE("L7 load-time prefault is byte-transparent on a borrowed F16 weight") {
  // The prefault (VT_GGUF_PREFAULT) only READS the borrowed pages to fault them
  // in at load, off the timed prefill path. It must change no byte and must leave
  // the weight borrowed in place. A/B the two env settings on the SAME file.
  //
  // ENG-RESIDENCY-CONFIG (#1110) MADE THIS CASE MEAN SOMETHING. Two problems, both
  // measured. The site cached its answer in a function-local static, so the second
  // `setenv` below could not affect anything and both arms ran identically — the
  // A/B was vacuous. And byte-transparency holds whether the prefault runs or not,
  // so with the site mutated to never consult its resolver at all this whole suite
  // stayed green (39/39), as did test_gguf_qwen36_loader (6/6) and
  // test_gguf_expert_span (11/11). The static is gone and the span counter below is
  // the observable: it is the only thing that separates "prefaulted" from
  // "skipped", because the operation reads pages and writes nothing.
  const DenseDims d;
  const TempFile f(BuildDenseF16Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);
  const vllm::GgufTensorInfo& oh = g.Get("output.weight");

  GgufLoadPolicy mmap = KeepF16On();
  mmap.mmap_residency = true;

  ::setenv("VT_GGUF_PREFAULT", "0", 1);
  vllm::ResetGgufPrefaultedSpanCountForTesting();
  const vllm::Qwen3_5DenseWeights woff =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &mmap);
  const uint64_t spans_off = vllm::GgufPrefaultedSpanCount();

  ::setenv("VT_GGUF_PREFAULT", "1", 1);
  vllm::ResetGgufPrefaultedSpanCountForTesting();
  const vllm::Qwen3_5DenseWeights won =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &mmap);
  const uint64_t spans_on = vllm::GgufPrefaultedSpanCount();
  ::unsetenv("VT_GGUF_PREFAULT");

  CHECK(won.lm_head.bytes.borrowed());        // still an in-place borrow
  CHECK(won.lm_head.bytes.data() == oh.data);
  CHECK(won.lm_head.bytes == woff.lm_head.bytes);  // byte-identical to no-prefault

  // THE ARMS ACTUALLY DIFFER. Without these two the case asserts only that the
  // prefault is harmless, which is equally true of a prefault that never ran.
  CHECK(spans_off == 0);
  CHECK(spans_on > 0);

  // And the CONFIG reaches the same site: `vllm_cpp.mmap.prefault` turns it off
  // with no environment variable in play, which is the whole point of #1110.
  vllm::ResetWeightResidencyConfigForTesting();
  vllm::WeightResidencyConfig cfg;
  cfg.prefault = false;
  vllm::SetWeightResidencyConfig(cfg);
  vllm::ResetGgufPrefaultedSpanCountForTesting();
  const vllm::Qwen3_5DenseWeights wcfg =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &mmap);
  CHECK(vllm::GgufPrefaultedSpanCount() == 0);
  CHECK(wcfg.lm_head.bytes == woff.lm_head.bytes);  // still byte-identical
  vllm::ResetWeightResidencyConfigForTesting();
}

TEST_CASE("LOAD-IO the prefault reports the BYTES it paged in, not just spans") {
  // WHY A SECOND COUNTER. The span count answers "did the prefault run". It
  // cannot answer the question a slow load poses -- how much of the wall time was
  // the artifact arriving -- because a span is not a size: one span may be 4 KiB
  // or 25 GiB. `GgufPrefaultedBytes()` is what turns the prefault's elapsed
  // seconds into a RATE, and a rate is the only form in which "the file was still
  // arriving" and "we were computing" can be told apart.
  //
  // THE ASSERTION IS AN EQUALITY, NOT A LOWER BOUND. `> 0` is satisfied by a
  // counter wired to the span count, to a constant, or to the last span only, and
  // all three are the defects this instrument would be used to hide. The expected
  // total is summed from the FILE's own tensor table, so it is independent of the
  // loader that reports it.
  const DenseDims d;
  const TempFile f(BuildDenseF16Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  GgufLoadPolicy mmap = KeepF16On();
  mmap.mmap_residency = true;

  ::setenv("VT_GGUF_PREFAULT", "0", 1);
  vllm::ResetGgufPrefaultedSpanCountForTesting();
  { const vllm::Qwen3_5DenseWeights woff = vllm::LoadQwen3_5DenseFromGguf(g, c, &mmap); (void)woff; }
  // OFF means zero in BOTH units. A bytes counter incremented outside the
  // resolver's guard would show up here and nowhere else.
  CHECK(vllm::GgufPrefaultedBytes() == 0);
  CHECK(vllm::GgufPrefaultSeconds() == 0.0);

  ::setenv("VT_GGUF_PREFAULT", "1", 1);
  vllm::ResetGgufPrefaultedSpanCountForTesting();
  const vllm::Qwen3_5DenseWeights won =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &mmap);
  const uint64_t spans = vllm::GgufPrefaultedSpanCount();
  const uint64_t bytes = vllm::GgufPrefaultedBytes();

  CHECK(spans > 0);
  CHECK(bytes > 0);
  // A span is not a size: the counter must not be the span count wearing a
  // different name. Every borrowed span in this fixture is far larger than the
  // number of spans, so equality here would mean the wrong quantity was wired.
  CHECK(bytes != spans);

  // The head really is borrowed in this fixture, and its whole span was paged in.
  // Read from the FILE's tensor table, so the expectation does not come from the
  // loader that reports the number.
  CHECK(won.lm_head.bytes.borrowed());
  const vllm::GgufTensorInfo& oh = g.Get("output.weight");
  CHECK(won.lm_head.bytes.data() == oh.data);
  CHECK(bytes >= static_cast<uint64_t>(oh.nbytes));

  // IT ACCUMULATES. A counter that is ASSIGNED rather than added -- reporting the
  // last span, or a constant -- passes every assertion above and fails this one:
  // a second load with no reset in between must add a second load's worth.
  const vllm::Qwen3_5DenseWeights wtwice =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &mmap);
  (void)wtwice;
  CHECK(vllm::GgufPrefaultedBytes() == 2 * bytes);
  CHECK(vllm::GgufPrefaultedSpanCount() == 2 * spans);
  ::unsetenv("VT_GGUF_PREFAULT");
}

TEST_CASE("a borrowed F16 weight OUTLIVES the GgufFile and the file") {
  const DenseDims d;
  std::vector<uint8_t> expected;
  std::unique_ptr<vllm::Qwen3_5DenseWeights> w;

  {
    const TempFile f(BuildDenseF16Gguf(d));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    const vllm::HfConfig c = vllm::HfConfigFromGguf(g);
    GgufLoadPolicy mmap = KeepF16On();
    mmap.mmap_residency = true;
    w = std::make_unique<vllm::Qwen3_5DenseWeights>(
        vllm::LoadQwen3_5DenseFromGguf(g, c, &mmap));
    const vllm::GgufTensorInfo& t = g.Get("output.weight");
    expected.assign(t.data, t.data + t.nbytes);
  }

  REQUIRE(w->lm_head.bytes.borrowed());
  REQUIRE(w->lm_head.bytes.size() == expected.size());
  CHECK(std::memcmp(w->lm_head.bytes.data(), expected.data(),
                    expected.size()) == 0);
  w.reset();  // last reference to the refcounted mapping
}

TEST_CASE("tied F16 head SHARES one f16 vocab matrix (copy AND mmap)") {
  const DenseDims d;
  const TempFile f(BuildDenseF16Gguf(d, /*tied=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);
  REQUIRE_FALSE(FileHasTensor(g, "output.weight"));

  for (bool use_mmap : {false, true}) {
    CAPTURE(use_mmap);
    GgufLoadPolicy shared = KeepF16On();
    shared.share_tied_head = true;
    shared.mmap_residency = use_mmap;
    const vllm::Qwen3_5DenseWeights ws =
        vllm::LoadQwen3_5DenseFromGguf(g, c, &shared);

    // ONE f16 buffer serves both: the gather table [vocab,H] nk=false and the
    // GEMM head [N=vocab,K=H] nk=true, over the same bytes.
    CHECK(ws.embed_tokens.dtype == vt::DType::kF16);
    CHECK(ws.lm_head.dtype == vt::DType::kF16);
    CHECK(ws.embed_tokens.bytes.data() == ws.lm_head.bytes.data());
    CHECK(ws.embed_tokens.bytes == ws.lm_head.bytes);
    CHECK(ws.embed_tokens.nk == false);
    CHECK(ws.lm_head.nk == true);
    CHECK(ws.lm_head.shape[0] == d.vocab);
    CHECK(ws.lm_head.shape[1] == d.H);
    CHECK(ws.embed_tokens.bytes.borrowed());
    CHECK(ws.lm_head.bytes.borrowed());

    // The share is freed exactly once, either half first.
    {
      vllm::Qwen3_5DenseWeights w = vllm::LoadQwen3_5DenseFromGguf(g, c, &shared);
      const vllm::OwnedBytes& head = w.lm_head.bytes;
      const std::vector<uint8_t> want(head.begin(), head.end());
      w.embed_tokens.bytes.Reset();
      REQUIRE(head.size() == want.size());
      CHECK(std::memcmp(head.data(), want.data(), want.size()) == 0);
    }
  }
}

// ENG-GGUF-RESIDENCY-RESOLVED-DEVICE ------------------------------------------
//
// `RouteGgufTensor` and `GgufLoadPolicy::FromEnv` used to read the device from
// `vllm::platforms::CurrentPlatform().device_type()`. That is the accelerator
// PROBE, and it answers `kCUDA` on every process where `platforms/cuda.cpp`'s
// `Registrar` saw a usable GPU — including a load the ENGINE resolved onto the
// CPU queue, because `LoadedEngine::ResolveExplicitDeviceType` returns `kCPU`
// for an explicit `--device cpu` "even on a CUDA-capable build/process".
//
// So the residency policy and the queue the forward runs on disagreed by
// construction, on every GGUF model. The device is now a PARAMETER with no
// default, which is what lets these cases run on a CPU-only host at all.
TEST_CASE("gguf residency: the routed device is a parameter, and it decides") {
  // The gather arm is the sharpest case: `DeviceQuantGatherSupported` answers
  // the OP TABLE (`OpId::kEmbeddingQuant`, #2396), and Metal registers no
  // gather in any build. Same tensor, same flags, same file bytes — two
  // devices, two residencies. Metal rather than CUDA on purpose: CUDA's answer
  // became build-dependent when #2396 landed, so naming it here would state a
  // fact this case does not own and would go red on a CUDA lane alone.
  const std::vector<int64_t> table = {320001536, 160};
  CHECK(vllm::RouteGgufTensor(/*keep_quant=*/true, /*keep_f16=*/false,
                              /*nvfp4_fp4=*/false, /*cpu_ref=*/false,
                              vllm::GgufTensorRole::kEmbeddingTable,
                              /*ggml_type=*/8u, table,
                              vt::DeviceType::kCPU) ==
        vllm::GgufResidency::kKeepQuant);
  // Metal, NOT CUDA, for the hard-coded contrast. This case owns one claim —
  // the DEVICE PARAMETER decides — and it must not also freeze WHICH devices
  // can gather, because #2396 is concurrently giving CUDA a block-decoding
  // gather. A `kCUDA` literal here would go red on that landing for a reason
  // this row has no opinion about, and worse, it would do so invisibly: these
  // cases sit at the end of the file and merge cleanly with #2396, and on a
  // CPU-only CI build the red would not appear at all. Metal has no gather arm
  // in any build, so it stays a valid contrast across both.
  CHECK(vllm::RouteGgufTensor(/*keep_quant=*/true, /*keep_f16=*/false,
                              /*nvfp4_fp4=*/false, /*cpu_ref=*/false,
                              vllm::GgufTensorRole::kEmbeddingTable,
                              /*ggml_type=*/8u, table,
                              vt::DeviceType::kMETAL) ==
        vllm::GgufResidency::kExpandBf16);

  // Every device, with the expectation DERIVED from the gate rather than from a
  // list written here. This is what keeps the claim ("the parameter decides")
  // separable from the fact ("only the CPU gathers today"): when a backend
  // gains a gather, this loop follows it and the case above still discriminates.
  for (vt::DeviceType d : {vt::DeviceType::kCPU, vt::DeviceType::kCUDA,
                           vt::DeviceType::kMETAL, vt::DeviceType::kVULKAN,
                           vt::DeviceType::kROCM}) {
    CAPTURE(std::string(vt::DeviceTypeName(d)));
    const vllm::GgufResidency want = vllm::DeviceQuantGatherSupported(d)
                                         ? vllm::GgufResidency::kKeepQuant
                                         : vllm::GgufResidency::kExpandBf16;
    CHECK(vllm::RouteGgufTensor(/*keep_quant=*/true, /*keep_f16=*/false,
                                /*nvfp4_fp4=*/false, /*cpu_ref=*/false,
                                vllm::GgufTensorRole::kEmbeddingTable,
                                /*ggml_type=*/8u, table, d) == want);
  }

  // The GEMM arm's device gate is per DTYPE, and ROCm's kernel set is narrower
  // than the CPU's: Q4_0 (ggml type 2) has no ROCm keep-quant `vec_dot`.
  const std::vector<int64_t> w = {8, 256};
  CHECK(vllm::RouteGgufTensor(true, false, false, false,
                              vllm::GgufTensorRole::kMatmulWeight,
                              /*ggml_type=*/2u, w, vt::DeviceType::kCPU) ==
        vllm::GgufResidency::kKeepQuant);
  CHECK(vllm::RouteGgufTensor(true, false, false, false,
                              vllm::GgufTensorRole::kMatmulWeight,
                              /*ggml_type=*/2u, w, vt::DeviceType::kROCM) ==
        vllm::GgufResidency::kExpandBf16);
}

TEST_CASE("gguf residency: the policy carries its device, and Route uses it") {
  // `FromEnv` resolves four flags from the device. `keep_f16` is the one that
  // is device-dependent WITHOUT depending on which ops this build registered,
  // so it is the flag that reads the same on a CPU-only and on a CUDA build:
  // `DeviceKeepF16Supported` is false for ROCm alone.
  vllm::GgufLoadPolicy cpu = vllm::GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU);
  vllm::GgufLoadPolicy rocm =
      vllm::GgufLoadPolicy::FromEnv(vt::DeviceType::kROCM);
  CHECK(cpu.device == vt::DeviceType::kCPU);
  CHECK(rocm.device == vt::DeviceType::kROCM);
  CHECK_FALSE(rocm.keep_f16);
  CHECK(cpu.keep_f16 == (cpu.expand_nk));

  // And `Route` carries the policy's device down to the decision, rather than
  // asking the platform a second question. Proven on the gather, whose answer
  // differs between the two devices: a `Route` that ignored `device` would
  // return the same residency for both policies on a CPU-only build.
  vllm::GgufTensorInfo t;
  t.name = "per_layer_token_embd.weight";
  t.ggml_type = 8u;             // Q8_0 — a block encoding with a row decoder
  t.shape = {320001536, 160};
  vllm::GgufLoadPolicy keep_cpu = cpu;
  keep_cpu.keep_quant = true;
  // Metal for the same reason as the case above: this asserts that `Route`
  // reads `policy.device`, not which devices gather.
  vllm::GgufLoadPolicy keep_metal =
      vllm::GgufLoadPolicy::FromEnv(vt::DeviceType::kMETAL);
  keep_metal.keep_quant = true;
  CHECK(keep_cpu.Route(t, vllm::GgufTensorRole::kEmbeddingTable) ==
        vllm::GgufResidency::kKeepQuant);
  CHECK(keep_metal.Route(t, vllm::GgufTensorRole::kEmbeddingTable) ==
        vllm::GgufResidency::kExpandBf16);
}

// ── BACKEND-ROCM-IQ-EXPERT-RESIDENCY (#2516) ────────────────────────────────
//
// A residency decision is a question about the device that will EXECUTE the
// tensor. Hybrid placement (#2023/#2314) made that device differ from the engine
// for exactly one role, and the loader kept asking the engine.
//
// The file this exists for: GLM-5.3 `UD-IQ1_S` on `strix:gpu0`. These cases
// need one encoding the CPU keeps and ROCm expands, so that a CPU placement
// and an unplaced ROCm engine give different answers for the same tensor. They
// used IQ1_S, then IQ4_XS, and each time the ROCm provider grew an arm for the
// chosen format and the inertness pins went red for the right reason.
// IQ2_XS is the current one: `VecDotIQ2_XSQ8_K` keeps it on the CPU (#2247)
// and no ROCm provider implements it (#1940 owns that gap). Pick the next
// format off `DeviceKeepQuantSupported`'s kROCM arm when this one lands too.
namespace {

// A routed-expert tower as the file stores it: [E, N, K] with a K that is a
// whole number of 256-element super-blocks, which is what `KeepQuantKDim` reads.
vllm::GgufTensorInfo IqTower(const std::string& name) {
  vllm::GgufTensorInfo t;
  t.name = name;
  t.ggml_type = kIQ2_XS;
  t.shape = {4, 2, 256};
  return t;
}

// The plan the loader installs: `cpu_moe` over `layers` layers on an engine that
// is NOT the CPU, built through the SAME resolver production uses so a pass here
// cannot come from a hand-made per-layer vector.
vllm::MoePlacementPlan CpuMoePlan(vt::DeviceType engine, int64_t layers,
                                  int64_t first_placed) {
  std::vector<vllm::PlacementOverride> ov;
  for (int64_t l = first_placed; l < layers; ++l)
    ov.push_back({vllm::LlmFfnExpsBlockRegex(l), "cpu"});
  return vllm::MoePlacementPlan::Resolve(
      vllm::DevicePlacement::FromOverrides(ov, engine), layers);
}

// RAII, because the plan is a process global and a case that leaked one would
// change the answer of every case that ran after it — including the inertness
// pins, which would then pass for the wrong reason.
struct ScopedPlan {
  explicit ScopedPlan(const vllm::MoePlacementPlan& p) {
    vllm::SetActiveMoePlacementPlan(p);
  }
  ~ScopedPlan() { vllm::ResetActiveMoePlacementPlanForTesting(); }
};

}  // namespace

TEST_CASE(
    "#2516: a routed-expert tower PLACED on the CPU keeps its blocks on a ROCm "
    "engine") {
  vllm::ResetActiveMoePlacementPlanForTesting();
  vllm::GgufLoadPolicy rocm;
  rocm.keep_quant = true;
  rocm.device = vt::DeviceType::kROCM;
  const vllm::GgufTensorInfo t = IqTower("blk.3.ffn_gate_exps.weight");

  // NO PLAN INSTALLED — the inertness pin, and the state every load in this
  // tree that configured no placement is in. No ROCm provider implements
  // IQ2_XS (#1940 owns that gap), so the tower expands.
  CHECK(rocm.Route(t, vllm::GgufTensorRole::kStackedExpertWeight) ==
        vllm::GgufResidency::kExpandBf16);

  // WITH the plan installed, the same tensor under the same policy keeps its
  // blocks, because the device that will run it is the CPU and
  // `vt::cpu::HasQuantDotKernel(kIQ2_XS)` is true (#2247).
  {
    ScopedPlan guard(CpuMoePlan(vt::DeviceType::kROCM, /*layers=*/8,
                                /*first_placed=*/0));
    CHECK(rocm.Route(t, vllm::GgufTensorRole::kStackedExpertWeight) ==
          vllm::GgufResidency::kKeepQuant);
  }

  // ...and the plan going away restores the old answer, so the override is the
  // plan's and not a latch.
  CHECK(rocm.Route(t, vllm::GgufTensorRole::kStackedExpertWeight) ==
        vllm::GgufResidency::kExpandBf16);
}

TEST_CASE("#2516: the override is PER LAYER, not per load") {
  // Layers 4..7 placed, 0..3 not. One assertion pair on one file under one
  // policy: a change that keyed the override on "a plan exists" rather than on
  // THIS tensor's layer passes the case above and fails this one.
  ScopedPlan guard(CpuMoePlan(vt::DeviceType::kROCM, /*layers=*/8,
                              /*first_placed=*/4));
  vllm::GgufLoadPolicy rocm;
  rocm.keep_quant = true;
  rocm.device = vt::DeviceType::kROCM;

  CHECK(rocm.Route(IqTower("blk.5.ffn_up_exps.weight"),
                   vllm::GgufTensorRole::kStackedExpertWeight) ==
        vllm::GgufResidency::kKeepQuant);
  CHECK(rocm.Route(IqTower("blk.1.ffn_up_exps.weight"),
                   vllm::GgufTensorRole::kStackedExpertWeight) ==
        vllm::GgufResidency::kExpandBf16);
}

TEST_CASE("#2516: only the STACKED-EXPERT role moves; the plan places nothing "
          "else") {
  // The plan moves a routed-expert BLOCK. An attention weight and the vocabulary
  // table of a placed layer's block stay on the engine and must keep answering
  // the engine's own capability — widening the override to every role would
  // keep a Q8_0 attention weight quantized for a device that will execute it.
  ScopedPlan guard(CpuMoePlan(vt::DeviceType::kROCM, /*layers=*/8,
                              /*first_placed=*/0));
  vllm::GgufLoadPolicy rocm;
  rocm.keep_quant = true;
  rocm.device = vt::DeviceType::kROCM;

  vllm::GgufTensorInfo w;
  w.name = "blk.3.attn_q.weight";
  w.ggml_type = kIQ2_XS;  // kept on the CPU, expanded on ROCm
  w.shape = {2, 256};
  CHECK(rocm.ComputeDeviceFor(w.name, vllm::GgufTensorRole::kMatmulWeight) ==
        vt::DeviceType::kROCM);
  CHECK(rocm.Route(w, vllm::GgufTensorRole::kMatmulWeight) ==
        vllm::GgufResidency::kExpandBf16);
  CHECK(rocm.ComputeDeviceFor("blk.3.ffn_gate_exps.weight",
                              vllm::GgufTensorRole::kStackedExpertWeight) ==
        vt::DeviceType::kCPU);
}

TEST_CASE("#2516: PeekRoute and Route resolve the device the same way") {
  // `GgufExpertTowersReachSlotLane` peeks the very tensors the loader routes.
  // Two spellings of the device term is how a bound and a forward come to
  // disagree about one file (#1378), so this pins that they cannot.
  ScopedPlan guard(CpuMoePlan(vt::DeviceType::kROCM, /*layers=*/8,
                              /*first_placed=*/4));
  vllm::GgufLoadPolicy rocm;
  rocm.keep_quant = true;
  rocm.device = vt::DeviceType::kROCM;
  for (const char* name : {"blk.5.ffn_gate_exps.weight",
                           "blk.1.ffn_gate_exps.weight",
                           "blk.5.attn_q.weight"}) {
    CAPTURE(name);
    const vllm::GgufTensorInfo t = IqTower(name);
    CHECK(vllm::PeekRoute(rocm, t,
                          vllm::GgufTensorRole::kStackedExpertWeight) ==
          rocm.Route(t, vllm::GgufTensorRole::kStackedExpertWeight));
  }
}

TEST_CASE("#2516: a plan that places NOTHING never overrides the engine") {
  // `cpu_moe` on a CPU engine resolves to a plan over every layer that places
  // nothing (`DevicePlacement::IsTrivial`). A default-constructed plan's engine
  // device is also `kCPU`. Either one, adopted unconditionally, would route a
  // ROCm load's experts as though they were on the host — which is the same
  // silent wrong answer in the opposite direction.
  ScopedPlan guard(CpuMoePlan(vt::DeviceType::kCPU, /*layers=*/8,
                              /*first_placed=*/0));
  CHECK_FALSE(vllm::ActiveMoePlacementPlan().PlacesAnything());
  vllm::GgufLoadPolicy rocm;
  rocm.keep_quant = true;
  rocm.device = vt::DeviceType::kROCM;
  CHECK(rocm.ComputeDeviceFor("blk.3.ffn_gate_exps.weight",
                              vllm::GgufTensorRole::kStackedExpertWeight) ==
        vt::DeviceType::kROCM);
  CHECK(rocm.Route(IqTower("blk.3.ffn_gate_exps.weight"),
                   vllm::GgufTensorRole::kStackedExpertWeight) ==
        vllm::GgufResidency::kExpandBf16);
}

namespace {
class ScopedF16Env {
 public:
  ScopedF16Env(const char* name, const char* value) : name_(name) {
    const char* old = std::getenv(name);
    had_value_ = old != nullptr;
    if (old != nullptr) old_value_ = old;
    vllm_test::SetEnv(name, value);
  }
  ~ScopedF16Env() {
    if (!had_value_) {
      vllm_test::UnsetEnv(name_);
    } else {
#if defined(_WIN32)
      vllm_test::SetEnv(name_, old_value_);
#else
      // Preserve an empty POSIX value as distinct from an absent variable.
      if (::setenv(name_, old_value_.c_str(), 1) != 0) std::terminate();
#endif
    }
  }
  ScopedF16Env(const ScopedF16Env&) = delete;
  ScopedF16Env& operator=(const ScopedF16Env&) = delete;
 private:
  const char* name_;
  bool had_value_ = false;
  std::string old_value_;
};
}  // namespace

// BACKEND-ROCM-F16-WEIGHTS (#3092): enter through the registered GGUF loader.
// A kernel-only F16 change cannot satisfy this admission test.
TEST_CASE("ROCm F16 production registry retains the embedding table") {
  ScopedF16Env cpu_ref("VT_CPU_REF", nullptr);
  ScopedF16Env keep_f16("VT_GGUF_KEEP_F16", nullptr);
  ScopedF16Env keep_quant("VT_GGUF_KEEP_QUANT", nullptr);
  for (bool tied : {false, true}) {
    CAPTURE(tied);
    const DenseDims d;
    const TempFile file(BuildDenseF16Gguf(d, tied));
    const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());
    auto config = vllm::HfConfigFromGguf(gguf);
    REQUIRE(config.torch_dtype == "bfloat16");
    for (const char* dtype : {"bfloat16", "bf16", ""}) {
      CAPTURE(std::string(dtype));
      config.torch_dtype = dtype;
      auto loaded = vllm::ModelRegistry::Load(
          config, vllm::ModelSource::FromGguf(gguf, vt::DeviceType::kROCM));
      const auto* embedding = loaded->shared_embed_tokens();
      REQUIRE(embedding != nullptr);
      REQUIRE_MESSAGE(embedding->dtype == vt::DType::kF16,
                      "the default ROCm registry must retain eligible F16 storage");
      REQUIRE(embedding->weight_value_dtype == vt::DType::kBF16);
      REQUIRE(embedding->bytes.size() == gguf.Get("token_embd.weight").nbytes);
      CHECK(std::memcmp(embedding->bytes.data(), gguf.Get("token_embd.weight").data,
                        embedding->bytes.size()) == 0);
    }
  }
}

TEST_CASE("ROCm F16 production registry refuses incompatible model values") {
  ScopedF16Env cpu_ref("VT_CPU_REF", nullptr);
  ScopedF16Env keep_f16("VT_GGUF_KEEP_F16", nullptr);
  ScopedF16Env keep_quant("VT_GGUF_KEEP_QUANT", nullptr);
  const DenseDims d;
  const TempFile file(BuildDenseF16Gguf(d));
  const auto gguf = vllm::GgufFile::Open(file.path());
  auto config = vllm::HfConfigFromGguf(gguf);
  for (const char* dtype : {"float16", "float32"}) {
    CAPTURE(std::string(dtype));
    config.torch_dtype = dtype;
    CHECK_THROWS_WITH(vllm::ModelRegistry::Load(
        config, vllm::ModelSource::FromGguf(gguf, vt::DeviceType::kROCM)),
        doctest::Contains("Qwen3.5 retained F16 GGUF: this forward resolves BF16 model values; "
                          "an unsupported model dtype cannot silently select BF16 semantics"));
  }
}

TEST_CASE("ROCm F16 admission requires explicit model values and eligible roles") {
  vllm_test::UnsetEnv("VT_CPU_REF");
  vllm_test::UnsetEnv("VT_GGUF_KEEP_F16");
  vllm_test::UnsetEnv("VT_GGUF_KEEP_QUANT");
  CHECK_FALSE(GgufLoadPolicy::FromEnv(vt::DeviceType::kROCM).keep_f16);
  for (vt::DType dtype : {vt::DType::kBF16, vt::DType::kF32}) {
    const auto policy = GgufLoadPolicy::FromEnv(vt::DeviceType::kROCM, dtype);
    CHECK(policy.keep_f16);
    CHECK(policy.weight_value_dtype == dtype);
    for (GgufTensorRole role : {GgufTensorRole::kMatmulWeight, GgufTensorRole::kEmbeddingTable})
      CHECK(RouteGgufTensor(false, true, false, false, role, kF16, {7, 13},
                            vt::DeviceType::kROCM, dtype) == GgufResidency::kKeepF16);
    CHECK(RouteGgufTensor(false, true, false, false, GgufTensorRole::kStackedExpertWeight,
                          kF16, {2, 7, 13}, vt::DeviceType::kROCM, dtype) ==
          GgufResidency::kExpandBf16);
    for (GgufTensorRole role : {GgufTensorRole::kTransformedWeight,
                               GgufTensorRole::kVector, GgufTensorRole::kConvWeight})
      CHECK(RouteGgufTensor(false, true, false, false, role, kF16, {7, 13},
                            vt::DeviceType::kROCM, dtype) == GgufResidency::kExpandBf16);
  }
  CHECK_THROWS(GgufLoadPolicy::FromEnv(vt::DeviceType::kROCM, vt::DType::kF16));
  CHECK_THROWS(GgufLoadPolicy::FromEnv(vt::DeviceType::kROCM, vt::DType::kI32));
  for (auto device : {vt::DeviceType::kCPU, vt::DeviceType::kCUDA}) {
    const auto old_policy = GgufLoadPolicy::FromEnv(device);
    const auto with_dtype = GgufLoadPolicy::FromEnv(device, vt::DType::kBF16);
    CHECK(old_policy.keep_f16 == with_dtype.keep_f16);
    CHECK(old_policy.expand_nk == with_dtype.expand_nk);
    CHECK_FALSE(with_dtype.weight_value_dtype.has_value());
  }
  for (const char* opt_out : {"VT_GGUF_KEEP_F16", "VT_CPU_REF"}) {
    vllm_test::SetEnv(opt_out, std::string(opt_out) == "VT_CPU_REF" ? "1" : "0");
    const auto policy = GgufLoadPolicy::FromEnv(vt::DeviceType::kROCM, vt::DType::kBF16);
    CHECK_FALSE(policy.keep_f16);
    const DenseDims d;
    const TempFile file(BuildDenseF16Gguf(d));
    const auto gguf = vllm::GgufFile::Open(file.path());
    const auto config = vllm::HfConfigFromGguf(gguf);
    auto loaded = vllm::ModelRegistry::Load(config,
        vllm::ModelSource::FromGguf(gguf, vt::DeviceType::kROCM));
    REQUIRE(loaded->shared_embed_tokens() != nullptr);
    CHECK(loaded->shared_embed_tokens()->dtype == vt::DType::kBF16);
    CHECK_FALSE(loaded->shared_embed_tokens()->weight_value_dtype.has_value());
    vllm_test::UnsetEnv(opt_out);
  }
}

TEST_CASE("ROCm retained F16 mapped and copied tied weights preserve bytes and values") {
  for (bool mapped : {false, true}) {
    for (bool tied : {false, true}) {
      CAPTURE(mapped); CAPTURE(tied);
      vllm_test::SetEnv("VT_GGUF_MMAP", mapped ? "1" : "0");
      const DenseDims d;
      const TempFile file(BuildDenseF16Gguf(d, tied));
      const auto gguf = vllm::GgufFile::Open(file.path());
      const auto config = vllm::HfConfigFromGguf(gguf);
      const auto policy = GgufLoadPolicy::FromEnv(vt::DeviceType::kROCM, vt::DType::kBF16);
      auto loaded = vllm::ModelRegistry::Load(config,
          vllm::ModelSource::FromGguf(gguf, vt::DeviceType::kROCM));
      REQUIRE(loaded->shared_embed_tokens() != nullptr);
      const auto& embedding = *loaded->shared_embed_tokens();
      CHECK(embedding.dtype == vt::DType::kF16);
      CHECK(embedding.weight_value_dtype == vt::DType::kBF16);
      CHECK(embedding.bytes.borrowed() == (mapped || tied));
      const auto weights = vllm::LoadQwen3_5DenseFromGguf(gguf, config, &policy);
      CHECK(weights.lm_head.dtype == vt::DType::kF16);
      CHECK(weights.lm_head.weight_value_dtype == vt::DType::kBF16);
      CHECK(weights.layers[0].mlp.gate_proj.weight_value_dtype == vt::DType::kBF16);
      if (tied) CHECK(weights.lm_head.bytes.data() == weights.embed_tokens.bytes.data());
      else CHECK(weights.lm_head.bytes.data() != weights.embed_tokens.bytes.data());
      const auto& source = gguf.Get(tied ? "token_embd.weight" : "output.weight");
      CHECK(weights.lm_head.bytes.size() == source.nbytes);
      CHECK(std::memcmp(weights.lm_head.bytes.data(), source.data, source.nbytes) == 0);
    }
  }
  vllm_test::UnsetEnv("VT_GGUF_MMAP");
}

#ifdef VLLM_CPP_HIP
#include "rocm_f16_native_observer.h"
namespace {
struct F16ForwardObservation {
  vt::MatmulFn nn = nullptr, bt = nullptr;
  vt::EmbeddingFn embedding = nullptr;
  struct Gemm {
    vt::Tensor output, weight;
    bool transposed;
  };
  std::vector<Gemm> gemms;
  struct Cast { vt::Tensor output, input; };
  std::vector<Cast> casts;
  int marked_gemms = 0, marked_embeddings = 0;
  static F16ForwardObservation* active;
  static void Nn(vt::Queue& q, vt::Tensor& out, const vt::Tensor& a, const vt::Tensor& b) {
    active->nn(q, out, a, b);
    Observe(out, b, false);
  }
  static void Bt(vt::Queue& q, vt::Tensor& out, const vt::Tensor& a, const vt::Tensor& b) {
    active->bt(q, out, a, b);
    Observe(out, b, true);
  }
  static void Observe(const vt::Tensor& out, const vt::Tensor& b, bool bt) {
    // Native F16 conversion recursively invokes a prepared GEMM. Record after
    // it returns so the completed outer call retains the model's weight marker.
    active->gemms.push_back({out, b, bt});
    if (!b.weight_value_dtype) return;
    CHECK(b.dtype == vt::DType::kF16);
    CHECK(b.weight_value_dtype == vt::DType::kBF16);
    ++active->marked_gemms;
  }
  static bool SameTensor(const vt::Tensor& a, const vt::Tensor& b) {
    if (a.data != b.data || a.device != b.device || a.dtype != b.dtype ||
        a.rank != b.rank || a.weight_value_dtype != b.weight_value_dtype) return false;
    for (int i = 0; i < a.rank; ++i) {
      if (a.shape[i] != b.shape[i] || a.stride[i] != b.stride[i]) return false;
    }
    return true;
  }
  int MarkedHeads(const vt::Tensor& logits) const {
    REQUIRE_FALSE(gemms.empty());
    // Earlier projections can share this head's width or a recycled output
    // address. Follow the actual BF16-to-F32 cast into returned logits (#3112).
    const auto& head = gemms.back();
    if (head.output.dtype == vt::DType::kBF16) {
      REQUIRE_FALSE(casts.empty());
      REQUIRE(SameTensor(casts.back().input, head.output));
      REQUIRE(SameTensor(casts.back().output, logits));
    } else {
      REQUIRE(SameTensor(head.output, logits));
    }
    int count = 0;
    for (const auto& call : gemms) {
      if (call.weight.weight_value_dtype && call.transposed == head.transposed &&
          SameTensor(call.weight, head.weight)) ++count;
    }
    return count;
  }
  static void Widen(vt::Queue& q, vt::Tensor& out, const vt::Tensor& in) {
    active->casts.push_back({out, in});
    rocm_f16_test::RealCastF32(q, out, in);
  }
  static void Gather(vt::Queue& q, vt::Tensor& out, const vt::Tensor& table,
                      const vt::Tensor& ids) {
    if (table.weight_value_dtype) {
      CHECK(table.dtype == vt::DType::kF16);
      CHECK(table.weight_value_dtype == vt::DType::kBF16);
      ++active->marked_embeddings;
    }
    active->embedding(q, out, table, ids);
  }
  F16ForwardObservation() {
    const auto device = vt::DeviceType::kROCM;
    nn = rocm_f16_test::RealNn;
    REQUIRE(vt::GetOp(vt::OpId::kMatmul, device) == reinterpret_cast<void*>(rocm_f16_test::WrapNn));
    bt = rocm_f16_test::RealBt;
    REQUIRE(vt::GetOp(vt::OpId::kMatmulBT, device) == reinterpret_cast<void*>(rocm_f16_test::WrapBt));
    embedding = rocm_f16_test::RealEmbedding;
    REQUIRE(vt::GetOp(vt::OpId::kEmbedding, device) == reinterpret_cast<void*>(rocm_f16_test::WrapEmbedding));
    REQUIRE(vt::GetOp(vt::OpId::kCastF32, device) == reinterpret_cast<void*>(rocm_f16_test::WrapCastF32));
    for (auto op : {vt::OpId::kMatmul, vt::OpId::kMatmulBT, vt::OpId::kEmbedding,
                    vt::OpId::kCastF32})
      REQUIRE(std::string(vt::GetOpProviderStats(op, device).last_selected) == vt::kNativeProviderName);
    active = this;
    rocm_f16_test::on_nn = Nn;
    rocm_f16_test::on_bt = Bt;
    rocm_f16_test::on_embedding = Gather;
    rocm_f16_test::on_cast_f32 = Widen;
  }
  ~F16ForwardObservation() {
    rocm_f16_test::on_nn = nullptr;
    rocm_f16_test::on_bt = nullptr;
    rocm_f16_test::on_embedding = nullptr;
    rocm_f16_test::on_cast_f32 = nullptr;
    active = nullptr;
  }
};
F16ForwardObservation* F16ForwardObservation::active = nullptr;

std::vector<float> F16RegistryForward(const vllm::GgufFile& gguf, bool keep) {
  if (keep) vllm_test::UnsetEnv("VT_GGUF_KEEP_F16");
  else vllm_test::SetEnv("VT_GGUF_KEEP_F16", "0");
  const auto config = vllm::HfConfigFromGguf(gguf);
  auto loaded = vllm::ModelRegistry::Load(config,
      vllm::ModelSource::FromGguf(gguf, vt::DeviceType::kROCM));
  auto& backend = vt::GetBackend(vt::DeviceType::kROCM);
  auto queue = backend.CreateQueue();
  struct Resources {
    vt::Backend& backend;
    vt::Queue& queue;
    std::vector<void*> buffers;
    ~Resources() {
      backend.Synchronize(queue);
      for (void* p : buffers) backend.Free(p);
      backend.DestroyQueue(queue);
    }
  } resources{backend, queue, {}};
  REQUIRE(queue.device.type == vt::DeviceType::kROCM);
  std::vector<vllm::PagedKvCache> caches;
  for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    const size_t bytes = static_cast<size_t>(2 * 2 * 16 * config.num_key_value_heads * config.head_dim) * 2;
    void* data = backend.Alloc(bytes);
    resources.buffers.push_back(data);
    backend.Memset(queue, data, 0, bytes);
    vllm::PagedKvCache cache;
    cache.data = data; cache.dtype = vt::DType::kBF16;
    cache.num_blocks = 2; cache.block_size = 16;
    cache.num_kv_heads = config.num_key_value_heads; cache.head_size = config.head_dim;
    caches.push_back(cache);
  }
  const std::vector<int32_t> ids{1, 2, 3}, positions{0, 1, 2}, logits_indices;
  vllm::v1::CommonAttentionMetadata attention;
  attention.num_reqs = 1; attention.num_actual_tokens = 3;
  attention.query_start_loc = {0, 3}; attention.query_start_loc_cpu = attention.query_start_loc;
  attention.seq_lens = {3}; attention.seq_lens_cpu = attention.seq_lens;
  attention.max_query_len = 3; attention.max_seq_len = 3;
  attention.block_table_num_cols = 2; attention.block_table_tensor = {0, 1};
  attention.slot_mapping = {0, 1, 2}; attention.causal = true;
  const vllm::v1::GDNAttentionMetadata gdn;
  std::vector<vllm::GdnStateCache> states;
  F16ForwardObservation observe;
  vllm::ModelForwardInput input{ids, positions, attention, gdn, caches, states,
                                config, queue, logits_indices};
  input.num_reqs = 1;
  const auto output = vllm::ModelRegistry::Forward(*loaded, input);
  backend.Synchronize(queue);
  REQUIRE(output.on_device());
  REQUIRE(output.device_tensor.dtype == vt::DType::kF32);
  REQUIRE(output.rows == 3);
  REQUIRE(output.vocab == config.vocab_size);
  const int marked_heads = observe.MarkedHeads(output.device_tensor);
  if (keep) {
    CHECK(observe.marked_embeddings == 1);
    CHECK(marked_heads == 1);
    CHECK(observe.marked_gemms >= 7 * config.num_hidden_layers + 1);
  } else {
    CHECK(observe.marked_embeddings == 0);
    CHECK(marked_heads == 0);
    CHECK(observe.marked_gemms == 0);
  }
  std::vector<float> result(static_cast<size_t>(3 * config.vocab_size));
  backend.Copy(queue, result.data(), output.device_tensor.data, result.size() * sizeof(float));
  backend.Synchronize(queue);
  return result;
}
}  // namespace

TEST_CASE("ROCm F16 production registered forward reaches retained embedding and projection weights") {
  REQUIRE_MESSAGE(vt::rocm::DeviceAvailable(), "this production execution gate requires a ROCm device");
  vllm_test::UnsetEnv("VT_CPU_REF");
  vllm_test::UnsetEnv("VT_GGUF_KEEP_QUANT");
  for (bool tied : {false, true}) {
    CAPTURE(tied);
    const DenseDims dims;
    const TempFile file(BuildDenseF16Gguf(dims, tied, /*gated=*/true));
    const auto gguf = vllm::GgufFile::Open(file.path());
    const auto retained = F16RegistryForward(gguf, true);
    const auto expanded = F16RegistryForward(gguf, false);
    REQUIRE(retained.size() == expanded.size());
    for (size_t i = 0; i < retained.size(); ++i) {
      CHECK(std::isfinite(retained[i]));
      CHECK(retained[i] == expanded[i]);
    }
  }
  vllm_test::UnsetEnv("VT_GGUF_KEEP_F16");
}
#endif

// W4d W4 (#3042, spec tenstorrent-27b-gdn-keepquant.md): the byte-level
// packed V-row reorder must be EXACTLY the element-level reorder seen
// through the row dequantizer — bit-for-bit, per row, for every encoding
// the GDN family carries. RED until ReorderVPackedForTest existed: the
// function did not, and the loader dequantized+reordered elements instead,
// which is what forced the 9.7 GiB of bf16 GDN projections onto the P150.
namespace {

std::vector<float> ReorderVRowsRef(const std::vector<float>& in,
                                   int64_t cols, int64_t row_off,
                                   int64_t num_k, int64_t num_v_per_k,
                                   int64_t head_rows) {
  const int64_t cs = head_rows * cols;
  std::vector<float> out = in;
  for (int64_t k = 0; k < num_k; ++k) {
    for (int64_t r = 0; r < num_v_per_k; ++r) {
      const int64_t g = k * num_v_per_k + r;
      const int64_t t = r * num_k + k;
      std::memcpy(out.data() + (row_off + g * head_rows) * cols,
                  in.data() + (row_off + t * head_rows) * cols,
                  static_cast<size_t>(cs) * sizeof(float));
    }
  }
  return out;
}

}  // namespace

TEST_CASE("packed V-row reorder equals the element-level reorder (W4d W4)") {
  // Geometry: K=512 (2 blocks/row, whole blocks per row), 15 weight rows =
  // row_off(3, non-V) + 6 heads x head_rows(2). q6_K row 420 B, q4_K 288 B.
  struct Enc {
    const char* name;
    uint32_t ggml_type;
    int64_t block_bytes;
  };
  const Enc encs[] = {{"q6_K", 14, 210}, {"q4_K", 12, 144}};
  const int64_t K = 512, row_off = 3, num_k = 2, rpk = 3, head_rows = 2;
  const int64_t rows = row_off + num_k * rpk * head_rows;  // 15

  std::mt19937 rng(20260912u);
  for (const Enc& e : encs) {
    const int64_t row_bytes = K / 256 * e.block_bytes;
    std::vector<uint8_t> packed(static_cast<size_t>(rows * row_bytes));
    for (size_t b = 0; b < packed.size(); b += e.block_bytes) {
      // pin the block's scale/min bits to valid f16 payloads; randomize the
      // quant elements (a random f16 scale can be inf/NaN and NaN != NaN).
      const uint16_t d = vt::F32ToF16(0.05f + 0.01f * static_cast<float>(b % 97));
      const uint16_t s = vt::F32ToF16(0.004f + 0.001f * static_cast<float>(b % 31));
      std::memcpy(packed.data() + b, &d, 2);
      if (e.block_bytes > 2) std::memcpy(packed.data() + b + 2, &s, 2);
      for (int64_t i = 4; i < e.block_bytes; ++i)
        packed[b + i] = static_cast<uint8_t>(rng() & 0xFF);
    }

    // candidate: reorder the packed bytes, then dequant per row
    std::vector<uint8_t> pb = packed;
    vllm::ReorderVPackedForTest(pb, row_bytes, row_off, num_k, rpk, head_rows);
    std::vector<float> fb(static_cast<size_t>(rows * K));
    for (int64_t r = 0; r < rows; ++r) {
      auto row = vllm::DequantGgufRowToF32(
          e.ggml_type, pb.data() + r * row_bytes, K);
      REQUIRE(row.size() == static_cast<size_t>(K));
      std::memcpy(fb.data() + r * K, row.data(), static_cast<size_t>(K) * 4);
    }

    // reference: dequant per row, then the independent element reorder
    std::vector<float> fa(static_cast<size_t>(rows * K));
    for (int64_t r = 0; r < rows; ++r) {
      auto row = vllm::DequantGgufRowToF32(
          e.ggml_type, packed.data() + r * row_bytes, K);
      std::memcpy(fa.data() + r * K, row.data(), static_cast<size_t>(K) * 4);
    }
    fa = ReorderVRowsRef(fa, K, row_off, num_k, rpk, head_rows);

    // NaN != NaN under float operator==, so compare raw bytes.
    CHECK(std::memcmp(fa.data(), fb.data(),
                      fa.size() * sizeof(float)) == 0);
  }
}

// ── BACKEND-TENSTORRENT: weight residency route ────────────────────────────
// spec .agents/specs/tenstorrent-bfp-weight-residency.md. VT_TT_WEIGHT_RESIDENCY=bfp8
// admits an UNQUANTIZED (ggml F32/BF16) GEMM/expert weight as kBfp8Device on a
// load the engine resolved onto kTENSTORRENT; every other role/encoding/device
// refuses by fall-through, and the DEFAULT OFF policy never elects the arm at
// all (the inertness pin below). The loader-side product of kBfp8Device is the
// bf16 expansion — admission decides only whether the Tenstorrent staging seam
// converts the weight to device-resident BFLOAT8_B.
TEST_CASE("BFP8 weight residency route: VT_TT_WEIGHT_RESIDENCY=bfp8 admits an "
          "unquantized GEMM weight on Tenstorrent; default OFF stays inert") {
  ::unsetenv("VT_TT_WEIGHT_RESIDENCY");
  ::unsetenv("VT_CPU_REF");
  vllm::GgufTensorInfo f32_w;
  f32_w.name = "blk.0.attn_q.weight";
  f32_w.ggml_type = 0;  // GGML_TYPE_F32
  f32_w.shape = {512, 512};

  // Parser: off (unset/empty/"off"/"0"), bfp8, bfp4 (reserved value parses;
  // the refusal to ACT on it is named where it is consumed).
  CHECK(vllm::ParseTtWeightResidency(nullptr) ==
        vllm::TtWeightResidency::kOff);
  CHECK(vllm::ParseTtWeightResidency("") == vllm::TtWeightResidency::kOff);
  CHECK(vllm::ParseTtWeightResidency("off") == vllm::TtWeightResidency::kOff);
  CHECK(vllm::ParseTtWeightResidency("0") == vllm::TtWeightResidency::kOff);
  CHECK(vllm::ParseTtWeightResidency("bfp8") == vllm::TtWeightResidency::kBfp8);
  CHECK(vllm::ParseTtWeightResidency("bfp4") == vllm::TtWeightResidency::kBfp4);

  // DEFAULT OFF: unset AND explicit "off" never elect the arm, on any device.
  for (const char* env_val : {(const char*)nullptr, "off"}) {
    if (env_val != nullptr) ::setenv("VT_TT_WEIGHT_RESIDENCY", env_val, 1);
    for (auto dev : {vt::DeviceType::kCPU, vt::DeviceType::kTENSTORRENT}) {
      const vllm::GgufLoadPolicy p = vllm::GgufLoadPolicy::FromEnv(dev);
      CHECK(p.tt_weight_residency == vllm::TtWeightResidency::kOff);
      CHECK(p.Route(f32_w, vllm::GgufTensorRole::kMatmulWeight) ==
            vllm::GgufResidency::kExpandBf16);
    }
  }
  ::unsetenv("VT_TT_WEIGHT_RESIDENCY");

  // Lever ON but the load resolved onto the CPU: refused (no consumer there).
  ::setenv("VT_TT_WEIGHT_RESIDENCY", "bfp8", 1);
  {
    const vllm::GgufLoadPolicy cpu = vllm::GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU);
    CHECK(cpu.tt_weight_residency == vllm::TtWeightResidency::kOff);
    CHECK(cpu.Route(f32_w, vllm::GgufTensorRole::kMatmulWeight) ==
          vllm::GgufResidency::kExpandBf16);
  }

  // Lever ON and the load resolved onto kTENSTORRENT: admitted, both verbatim
  // roles, both unquantized encodings.
  {
    const vllm::GgufLoadPolicy tt =
        vllm::GgufLoadPolicy::FromEnv(vt::DeviceType::kTENSTORRENT);
    CHECK(tt.tt_weight_residency == vllm::TtWeightResidency::kBfp8);
    CHECK(tt.Route(f32_w, vllm::GgufTensorRole::kMatmulWeight) ==
          vllm::GgufResidency::kBfp8Device);
    vllm::GgufTensorInfo bf16_w;
    bf16_w.name = "blk.3.ffn_up_exps.weight";
    bf16_w.ggml_type = 30;  // GGML_TYPE_BF16
    bf16_w.shape = {4, 512, 512};
    CHECK(tt.Route(bf16_w, vllm::GgufTensorRole::kStackedExpertWeight) ==
          vllm::GgufResidency::kBfp8Device);
    CHECK(Name(vllm::GgufResidency::kBfp8Device) == std::string("bfp8_device"));

    // REFUSALS BY FALL-THROUGH, each by its reason:
    // a value-rewritten tensor never keeps anything,
    CHECK(tt.Route(f32_w, vllm::GgufTensorRole::kTransformedWeight) ==
          vllm::GgufResidency::kExpandBf16);
    CHECK(tt.Route(f32_w, vllm::GgufTensorRole::kVector) ==
          vllm::GgufResidency::kExpandBf16);
    CHECK(tt.Route(f32_w, vllm::GgufTensorRole::kEmbeddingTable) ==
          vllm::GgufResidency::kExpandBf16);
    // a ragged K cannot pack BFP tiles,
    vllm::GgufTensorInfo ragged = f32_w;
    ragged.shape = {512, 33};
    CHECK(tt.Route(ragged, vllm::GgufTensorRole::kMatmulWeight) ==
          vllm::GgufResidency::kExpandBf16);
    // a block encoding stays keep-quant when that residency is available and
    // NEVER reaches the BFP arm,
    vllm::GgufTensorInfo q8 = f32_w;
    q8.ggml_type = 8;  // GGML_TYPE_Q8_0
    const vllm::GgufResidency q8_r =
        tt.Route(q8, vllm::GgufTensorRole::kMatmulWeight);
    CHECK(q8_r != vllm::GgufResidency::kBfp8Device);
    // and F16 stays keep-f16 / expands, never BFP (this wave: bf16/f32 only).
    vllm::GgufTensorInfo f16 = f32_w;
    f16.ggml_type = 1;
    CHECK(tt.Route(f16, vllm::GgufTensorRole::kMatmulWeight) !=
          vllm::GgufResidency::kBfp8Device);
  }

  // The RESERVED bfp4 value parses, rides the policy on a TT load, and the
  // route refuses it BY FALL-THROUGH (never kBfp8Device) until it is
  // implemented — the staging seam is what names it in its refusal message.
  ::setenv("VT_TT_WEIGHT_RESIDENCY", "bfp4", 1);
  {
    const vllm::GgufLoadPolicy tt =
        vllm::GgufLoadPolicy::FromEnv(vt::DeviceType::kTENSTORRENT);
    CHECK(tt.tt_weight_residency == vllm::TtWeightResidency::kBfp4);
    CHECK(tt.Route(f32_w, vllm::GgufTensorRole::kMatmulWeight) ==
          vllm::GgufResidency::kExpandBf16);
  }
  ::unsetenv("VT_TT_WEIGHT_RESIDENCY");

  // The oracle switch forces the arm off, like every other residency.
  ::setenv("VT_CPU_REF", "1", 1);
  {
    const vllm::GgufLoadPolicy oracle =
        vllm::GgufLoadPolicy::FromEnv(vt::DeviceType::kTENSTORRENT);
    CHECK(oracle.tt_weight_residency == vllm::TtWeightResidency::kOff);
    CHECK(oracle.Route(f32_w, vllm::GgufTensorRole::kMatmulWeight) ==
          vllm::GgufResidency::kExpandBf16);
  }
  ::unsetenv("VT_CPU_REF");
  ::unsetenv("VT_TT_WEIGHT_RESIDENCY");
}
