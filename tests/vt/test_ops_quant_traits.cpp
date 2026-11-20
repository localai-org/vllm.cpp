// QUANT-GGUF-CIQ-GEMM work row G1 — block dtypes, the CPU quant traits table,
// and the `kMatmulBTQuant` skeleton's generic-composite fallback.
//
// The load-bearing case in this file is the TRAIT CROSS-CHECK: vt's block
// geometry table (src/vt/dtype.cpp) and the GGUF reader's `GgmlTraits`
// (src/vllm/model_executor/model_loader/gguf_reader.cpp) are two INDEPENDENT
// ports of the same llama.cpp facts (@ 237ad9b96, ggml/include/ggml.h:390-432
// + ggml/src/ggml-common.h block structs). A silent disagreement between them
// would mis-stride every packed weight buffer downstream, so they are compared
// element-for-element here, together with the third independent statement of
// the same facts — the block struct arithmetic spelled out from ggml-common.h.
//
// Traits-table cases mirror llama.cpp `ggml/src/ggml-cpu/ggml-cpu.c:211-406`
// (`type_traits_cpu[]`): the `vec_dot_type` column is the dispatch fact G1
// owns (Q8_0 for the legacy 32-element types, Q8_K for the K-quants).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/quant.h"
#include "vt/tensor.h"

namespace {

struct BlockCase {
  vt::DType dtype;
  uint32_t ggml_type;
  int64_t block_elems;
  int64_t block_bytes;
  vt::DType vec_dot_type;
  const char* name;
};

// Expected values written out from the ggml-common.h struct definitions, NOT
// copied from either table under test:
//   q4_0  :213-218  f16 d + 32/2 qs                        = 2 + 16  = 18
//   q8_0  :242-245  f16 d + 32 qs                          = 2 + 32  = 34
//   q3_K  :305-310  256/8 hmask + 256/4 qs + 12 sc + f16 d = 32+64+12+2  = 110
//   q4_K  :317-327  2*f16 dm + 12 sc + 256/2 qs            = 4+12+128    = 144
//   q5_K  :334-345  2*f16 dm + 12 sc + 256/8 qh + 256/2 qs = 4+12+32+128 = 176
//   q6_K  :352-357  256/2 ql + 256/4 qh + 256/16 sc + f16 d= 128+64+16+2 = 210
//   q8_K  :361-365  f32 d + 256 qs + 256/16 i16 bsums      = 4+256+32    = 292
const BlockCase kBlockCases[] = {
    {vt::DType::kQ4_0, 2, 32, 2 + 16, vt::DType::kQ8_0, "q4_0"},
    {vt::DType::kQ8_0, 8, 32, 2 + 32, vt::DType::kQ8_0, "q8_0"},
    {vt::DType::kQ3_K, 11, 256, 32 + 64 + 12 + 2, vt::DType::kQ8_K, "q3_K"},
    {vt::DType::kQ4_K, 12, 256, 4 + 12 + 128, vt::DType::kQ8_K, "q4_K"},
    {vt::DType::kQ5_K, 13, 256, 4 + 12 + 32 + 128, vt::DType::kQ8_K, "q5_K"},
    {vt::DType::kQ6_K, 14, 256, 128 + 64 + 16 + 2, vt::DType::kQ8_K, "q6_K"},
    {vt::DType::kQ8_K, 15, 256, 4 + 256 + 32, vt::DType::kQ8_K, "q8_K"},
    // Q5_0 and IQ4_NL, added for MODEL-MM-QWEN4-EXP W6a, are the two 32-element
    // legacy-family encodings this table lacked. Both dot against Q8_0, not
    // Q8_K, because their block is 32 elements wide (ggml-cpu.c:259-264 and
    // :379-384 @ llama.cpp b10451). Sizes written out from ggml-common.h
    // @ b10451, NOT copied from either table under test:
    //   q5_0    :229-235  f16 d + 4 qh + 32/2 qs             = 2+4+16 = 22
    //   iq4_nl  :447-452  f16 d + 32/2 qs                    = 2+16    = 18
    {vt::DType::kQ5_0, 6, 32, 2 + 4 + 16, vt::DType::kQ8_0, "q5_0"},
    {vt::DType::kIQ4_NL, 20, 32, 2 + 16, vt::DType::kQ8_0, "iq4_nl"},
};

// Deterministic pseudo-random block bytes. Any bit pattern is a legal block for
// every one of these encodings (all fields are unconstrained integers/halfs),
// except that f16 scale fields must avoid inf/nan — bit patterns with all
// exponent bits set. Those are masked out below so the composite comparison is
// finite.
std::vector<uint8_t> RandomBlocks(const BlockCase& c, int64_t nblocks,
                                  uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> bytes(static_cast<size_t>(nblocks * c.block_bytes));
  for (uint8_t& b : bytes) b = static_cast<uint8_t>(rng() & 0xFF);
  // Overwrite every f16/f32 scale field with a small, exactly-representable
  // value so no block carries an inf/nan delta.
  for (int64_t i = 0; i < nblocks; ++i) {
    uint8_t* blk = bytes.data() + i * c.block_bytes;
    auto put_f16 = [&](int off, float v) {
      const uint16_t h = vt::F32ToF16(v);
      std::memcpy(blk + off, &h, sizeof(h));
    };
    switch (c.dtype) {
      case vt::DType::kQ4_0:
      case vt::DType::kQ8_0:
      // q5_0 and iq4_nl both open with the f16 delta, like q4_0/q8_0.
      case vt::DType::kQ5_0:
      case vt::DType::kIQ4_NL:
        put_f16(0, 0.0125F);
        break;
      case vt::DType::kQ3_K:
        put_f16(108, 0.0125F);
        break;
      case vt::DType::kQ4_K:
      case vt::DType::kQ5_K:
        put_f16(0, 0.0125F);
        put_f16(2, 0.0075F);
        break;
      case vt::DType::kQ6_K:
        put_f16(208, 0.0125F);
        break;
      case vt::DType::kQ8_K: {
        const float d = 0.0125F;
        std::memcpy(blk, &d, sizeof(d));
        break;
      }
      default:
        break;
    }
  }
  return bytes;
}

}  // namespace

// --- The cross-check ---------------------------------------------------------

TEST_CASE("vt block geometry agrees with the GGUF reader's GgmlTraits") {
  for (const BlockCase& c : kBlockCases) {
    CAPTURE(c.name);
    // vt's own table vs the values written out from ggml-common.h.
    CHECK(vt::IsBlockQuant(c.dtype));
    CHECK(vt::BlockElems(c.dtype) == c.block_elems);
    CHECK(vt::BlockBytes(c.dtype) == c.block_bytes);
    CHECK(vt::GgmlTypeId(c.dtype) == c.ggml_type);
    CHECK(std::string(vt::Name(c.dtype)) == c.name);

    // Q8_K (id 15) is a FILE type the reader carries like any other, so the
    // cross-check below runs for it too. Q8_K is the K-quants' ACTIVATION
    // encoding, and llama.cpp still declares its file geometry: a real traits
    // row with a nonzero `type_size` and deliberately with NO `.vec_dot` and NO
    // `.to_float` (ggml/src/ggml.c:879-884 at the pinned llama.cpp `10bf611e`,
    // tag `b10451`), whose row data the file loader validates as it reads the
    // tensor (src/llama-model-loader.cpp:1416 into
    // ggml/src/ggml-quants.c:5590-5598). This case asserted the OPPOSITE -- an
    // absence -- until #3097, and no test pins that absence any more.
    const vllm::GgmlTypeTraits& g = vllm::GgmlTraits(c.ggml_type);
    CHECK(g.block_elems == vt::BlockElems(c.dtype));
    CHECK(g.block_bytes == vt::BlockBytes(c.dtype));

    // ...and the id round-trips back to the same vt dtype.
    vt::DType back = vt::DType::kF32;
    REQUIRE(vt::BlockDTypeFromGgmlTypeId(c.ggml_type, &back));
    CHECK(back == c.dtype);

    if (c.dtype == vt::DType::kQ8_K) {
      // FILE GEOMETRY IS NOT KEEP-QUANT CAPABILITY, and neither is the absence
      // of a weight-side dot kernel any more: TT-KEEPQUANT-Q8K-WEIGHTS landed
      // kQ8_K INTO the GEMM keep-quant set as the named dequant-composite
      // exception (gguf_keep_quant.cpp:334-340 — it serves the tree-standard
      // Q8_K weight arm via ComposeChunk on CPU and the TT composite chain on
      // device, precisely because it has no vec_dot). So the loader's public
      // predicate answers TRUE with kept=kQ8_K; what stays absent is the
      // weight-side dot kernel itself.
      CHECK(vt::cpu::BlockVecDot(vt::DType::kQ8_K) == nullptr);
      CHECK_FALSE(vt::cpu::HasQuantDotKernel(vt::DType::kQ8_K));
      vt::DType kept = vt::DType::kF32;
      CHECK(vllm::KeepQuantDType(c.ggml_type, &kept));
      CHECK(kept == vt::DType::kQ8_K);
    }
  }
}

// The ~2-3-bit / fp4 codebook encodings the DeepSeek-V4-Flash GGUF checkpoints
// use for routed experts. They register geometry + a `to_float` decode AND a
// keep-quant `vec_dot`, so `HasQuantDotKernel` is TRUE and the loader keeps the
// blocks COMPRESSED (never expand-to-bf16). This is a DISTINCT contract from the
// executable weight types in kBlockCases, so it gets its own case. Note MXFP4
// dots against Q8_0 (32-element blocks), the rest against Q8_K.
TEST_CASE("IQ/MXFP4 keep-quant block dtypes (geometry + vec_dot)") {
  // DeepSeek-V4 W8 (CLAIM-DEEPSEEK-V4-W8): these codebook encodings — the
  // single-Spark UD-IQ2_XXS (IQ2_XXS gate/up + IQ3_XXS down) / UD-Q2_K_XL and
  // the UD-IQ2_M (IQ2_S gate/up + MXFP4 down) routed experts — carry a keep-quant
  // `vec_dot`, so they are HasQuantDotKernel-TRUE (the memory enabler).
  struct KeepQuantCase {
    vt::DType dtype;
    uint32_t ggml_type;
    int64_t block_elems;
    int64_t block_bytes;
    vt::DType vec_dot_type;
    const char* name;
  };
  // Sizes written out from ggml-common.h @ 237ad9b96:
  //   q2_K    :288-299  16 sc + 64 qs + f16 d + f16 dmin        = 16+64+2+2 = 84
  //   iq2_xxs :371-374  f16 d + 32 u16 qs                       = 2 + 64    = 66
  //   iq3_xxs :385-400  f16 d + 96 u8 qs                        = 2 + 96    = 98
  //   iq2_s   :386-392  f16 d + 64 qs + 8 qh + 8 scales         = 2+64+8+8  = 82
  //   mxfp4   :204-209  u8 e + 16 qs                            = 1 + 16    = 17
  //   iq1_s   :414-419  f16 d + 32 qs + 8 u16 qh                 = 2+32+16   = 50
  const KeepQuantCase cases[] = {
      {vt::DType::kQ2_K, 10, 256, 84, vt::DType::kQ8_K, "q2_K"},
      {vt::DType::kIQ2_XXS, 16, 256, 66, vt::DType::kQ8_K, "iq2_xxs"},
      {vt::DType::kIQ3_XXS, 18, 256, 98, vt::DType::kQ8_K, "iq3_xxs"},
      {vt::DType::kIQ2_S, 22, 256, 82, vt::DType::kQ8_K, "iq2_s"},
      {vt::DType::kMXFP4, 39, 32, 17, vt::DType::kQ8_0, "mxfp4"},
      {vt::DType::kIQ1_S, 19, 256, 50, vt::DType::kQ8_K, "iq1_s"},
      {vt::DType::kIQ1_XXXS, 66, 256, 38, vt::DType::kQ8_K, "iq1_xxxs"},
      // QUANT-GGUF-IQ-VECDOT (#2247). The two encodings that carry the staged
      // `unsloth/GLM-5.3-Flash-GGUF UD-Q2_K_XL` arm: 82 of its 1412 tensors are
      // IQ2_XS and 3 are IQ4_XS, against TWO that are Q2_K. Sizes written out
      // from llama.cpp @ b10451 ggml-common.h:
      //   iq2_xs :388-393  f16 d + 32 u16 qs + 8 scales          = 2+64+8   = 74
      //   iq4_xs :454-460  f16 d + u16 scales_h + 4 scales_l
      //                    + 128 qs                              = 2+2+4+128 = 136
      // IQ4_XS dots against Q8_K, NOT against the Q8_0 its codebook sibling
      // IQ4_NL uses (ggml-cpu.c:385-390 against :379-384): the codebook is
      // shared, the block geometry is not.
      {vt::DType::kIQ2_XS, 17, 256, 74, vt::DType::kQ8_K, "iq2_xs"},
      {vt::DType::kIQ4_XS, 23, 256, 136, vt::DType::kQ8_K, "iq4_xs"},
  };
  for (const KeepQuantCase& c : cases) {
    CAPTURE(c.name);
    // vt geometry.
    CHECK(vt::IsBlockQuant(c.dtype));
    CHECK(vt::BlockElems(c.dtype) == c.block_elems);
    CHECK(vt::BlockBytes(c.dtype) == c.block_bytes);
    CHECK(vt::GgmlTypeId(c.dtype) == c.ggml_type);
    CHECK(std::string(vt::Name(c.dtype)) == c.name);
    CHECK_THROWS(vt::SizeOf(c.dtype));
    CHECK(vt::RowSizeBytes(c.dtype, c.block_elems) ==
          static_cast<size_t>(c.block_bytes));

    // Reader GgmlTraits agreement (all are real FILE types, unlike Q8_K).
    const vllm::GgmlTypeTraits& g = vllm::GgmlTraits(c.ggml_type);
    CHECK(g.block_elems == c.block_elems);
    CHECK(g.block_bytes == c.block_bytes);
    vt::DType back = vt::DType::kF32;
    REQUIRE(vt::BlockDTypeFromGgmlTypeId(c.ggml_type, &back));
    CHECK(back == c.dtype);

    // Decodes (to_float present) AND keep-quant capable: a vec_dot row against
    // its activation encoding exists, so the loader keeps the blocks compressed.
    CHECK(vt::cpu::BlockToFloat(c.dtype) != nullptr);
    CHECK(vt::cpu::HasQuantDotKernel(c.dtype));
    const vt::cpu::QuantTypeTraits& t = vt::cpu::QuantTraits(c.dtype);
    CHECK(t.vec_dot != nullptr);
    CHECK(t.vec_dot_type == c.vec_dot_type);
    // The activation encoding it dots against must itself be encodable.
    CHECK(vt::cpu::BlockFromFloat(c.vec_dot_type) != nullptr);
    // No `from_float`: nothing quantizes an activation INTO these weight types.
    CHECK(vt::cpu::BlockFromFloat(c.dtype) == nullptr);
  }
}

// The DECODE-ONLY class — a block dtype with a `to_float` and no keep-quant
// `vec_dot`, which the GGUF loader can gather from but must EXPAND on a GEMM.
// It is worth its own case because it is the class that silently costs memory:
// nothing throws, tokens still match, and a routed-expert slab quietly lands in
// bf16.
//
// IQ2_XS (17) and IQ4_XS (23) were its only file-type members, put there by
// LOADER-GGUF-IQ (#2245) and taken out by QUANT-GGUF-IQ-VECDOT (#2247). It then
// held NO encoding a checkpoint could be stored in.
//
// QUANT-IQ3S (#2510) puts one back, DELIBERATELY and exactly one: IQ3_S (21)
// has a `to_float` and no `vec_dot`, so the loader gathers it compressed and
// EXPANDS it on the GEMM arm. That is stated here by NAME rather than left to a
// count, because the whole point of the class is that nothing throws when a
// member of it quietly costs memory. The `vec_dot` is owed jointly with the
// CUDA `WType::kIQ3_S` arm — landing the CPU half alone would send every CUDA
// IQ3_S GEMM down `IsCudaKeepQuantSupported`'s host fallback, which segfaults
// on a discrete card — and the day it lands, this case reds and the tier table
// in `.agents/specs/gguf-iq3s.md` has to move with it.
TEST_CASE("the decode-only class is Q8_K and IQ3_S: exactly one FILE type expands") {
  // Q8_K is the K-quants' ACTIVATION encoding. Upstream gives it no `vec_dot`
  // row at all (ggml-cpu.c:391-393 carries only a `from_float`), so it can
  // never leave this class the way the two IQ*_XS rows did.
  CHECK(vt::cpu::BlockToFloat(vt::DType::kQ8_K) != nullptr);
  CHECK(vt::cpu::BlockVecDot(vt::DType::kQ8_K) == nullptr);
  CHECK_FALSE(vt::cpu::HasQuantDotKernel(vt::DType::kQ8_K));
  // It is the one block dtype here that goes the OTHER way: it has a
  // `from_float`, because something does have to produce the activation.
  CHECK(vt::cpu::BlockFromFloat(vt::DType::kQ8_K) != nullptr);

  // Every OTHER block dtype this tree knows must now also dot. The population
  // is SWEPT out of `BlockDTypeFromGgmlTypeId` rather than hand-listed, so the
  // next decoder that lands without a kernel reds this case instead of slipping
  // in behind a list nobody updated.
  int swept = 0;
  for (uint32_t id = 0; id < 256; ++id) {
    vt::DType d = vt::DType::kF32;
    if (!vt::BlockDTypeFromGgmlTypeId(id, &d)) continue;
    if (d == vt::DType::kQ8_K) continue;
    // IQ3_S dotted with aa85e9484 (tenstorrent-gsq-keepquant wave 1), which
    // landed the CPU oracle `VecDotIQ3_SQ8_K` alongside the TT on-core decode,
    // so it now sweeps like every other file weight and is asserted explicitly
    // below as well.
    CAPTURE(id);
    CAPTURE(vt::Name(d));
    ++swept;
    REQUIRE(vt::cpu::BlockToFloat(d) != nullptr);
    CHECK(vt::cpu::BlockVecDot(d) != nullptr);
    CHECK(vt::cpu::HasQuantDotKernel(d));
    // Nothing quantizes an activation INTO a weight encoding. Q8_0 is the one
    // exemption and not an exception: it is a file weight type AND the
    // 32-element activation encoding, so it has to encode.
    if (d != vt::DType::kQ8_0) CHECK(vt::cpu::BlockFromFloat(d) == nullptr);
  }
  // The sweep found something. A `BlockDTypeFromGgmlTypeId` that started
  // refusing every id would otherwise pass the loop above vacuously.
  // IQ1_M (ggml id 29) joined with BOTH a decoder and a dot kernel, so the
  // sweep count moved 17 -> 18 when it landed.
  // 18 -> 19 when IQ3_S gained its dot kernel with aa85e9484
  // (tenstorrent-gsq-keepquant wave 1). TQ1_0 and TQ2_0 moved it
  // 19 -> 21 when their Q8_K dot kernels matched the Vulkan keep-quant path.
  CAPTURE(swept);
  CHECK(swept == 21);

  // The decode-only FILE member, named and asserted in BOTH directions. Sizes
  // written out from llama.cpp @ b10451 ggml-common.h:413-422, NOT copied from
  // either table under test:
  //   iq3_s :413-422  f16 d + 256/4 qs + 256/32 qh + 256/8 signs
  //                   + 256/64 scales                    = 2+64+8+32+4 = 110
  // IQ1_M (e71f6d6fd) moved the sweep count 17 -> 18; IQ3_S dotting moved it
  // 18 -> 19 with aa85e9484, so the old "decode-only" claim is INVERTED here:
  // the dot kernel is asserted present, not absent.
  CHECK(vt::cpu::BlockToFloat(vt::DType::kIQ3_S) != nullptr);
  CHECK(vt::cpu::BlockVecDot(vt::DType::kIQ3_S) != nullptr);
  CHECK(vt::cpu::HasQuantDotKernel(vt::DType::kIQ3_S));
  CHECK(vt::cpu::BlockFromFloat(vt::DType::kIQ3_S) == nullptr);
  CHECK(vt::BlockElems(vt::DType::kIQ3_S) == 256);
  CHECK(vt::BlockBytes(vt::DType::kIQ3_S) == 2 + 64 + 8 + 32 + 4);
  CHECK(vt::GgmlTypeId(vt::DType::kIQ3_S) == 21U);
  CHECK(std::string(vt::Name(vt::DType::kIQ3_S)) == "iq3_s");
  // The reader agrees, which is the half #2510 was actually about: without this
  // trait `GgufFile::Open` refused the whole 16.4 GB artifact.
  const vllm::GgmlTypeTraits& g_iq3s = vllm::GgmlTraits(21U);
  CHECK(g_iq3s.block_elems == 256);
  CHECK(g_iq3s.block_bytes == 2 + 64 + 8 + 32 + 4);

  // The pair that moved, named explicitly: the geometry and the reader
  // agreement are unchanged from #2245, only the dot arrived. Sizes written out
  // from llama.cpp @ b10451 ggml-common.h, NOT copied from either table under
  // test:
  //   iq2_xs :388-393  f16 d + 256/8 u16 qs + 256/32 scales   = 2+64+8    = 74
  //   iq4_xs :454-460  f16 d + u16 scales_h + 256/64 scales_l
  //                    + 256/2 qs                             = 2+2+4+128 = 136
  struct MovedCase {
    vt::DType dtype;
    uint32_t ggml_type;
    int64_t block_bytes;
  };
  for (const MovedCase& c : {MovedCase{vt::DType::kIQ2_XS, 17, 2 + 64 + 8},
                             MovedCase{vt::DType::kIQ4_XS, 23,
                                       2 + 2 + 4 + 128}}) {
    CAPTURE(c.ggml_type);
    CHECK(vt::BlockElems(c.dtype) == 256);
    CHECK(vt::BlockBytes(c.dtype) == c.block_bytes);
    const vllm::GgmlTypeTraits& g = vllm::GgmlTraits(c.ggml_type);
    CHECK(g.block_elems == 256);
    CHECK(g.block_bytes == c.block_bytes);
    CHECK(vt::cpu::HasQuantDotKernel(c.dtype));
    CHECK(vt::cpu::QuantTraits(c.dtype).vec_dot_type == vt::DType::kQ8_K);
  }
}

// TQ2_0/TQ1_0 are Vulkan-native ternary keep-quant encodings. They have block
// geometry + a to_float dequantizer (CPU reference oracle) + a QuantTraits row
// (vec_dot_type = kQ8_K), but NO CPU vec_dot — the keep-quant dot is
// Vulkan-only. HasQuantDotKernel is therefore FALSE, and the CPU
// MatmulBTQuantKernel takes the dequant-composite path. The GGUF reader does
// not yet carry ids 42/43, so they are NOT in kBlockCases (which cross-checks
// against GgmlTraits).
TEST_CASE("TQ2_0/TQ1_0 ternary block dtypes (geometry + dequant, no CPU vec_dot)") {
  struct TQCase {
    vt::DType dtype;
    uint32_t ggml_type;
    int64_t block_elems;
    int64_t block_bytes;
    vt::DType vec_dot_type;
    const char* name;
  };
  const TQCase cases[] = {
      {vt::DType::kTQ2_0, 42, 256, 66, vt::DType::kQ8_K, "tq2_0"},
      {vt::DType::kTQ1_0, 43, 256, 54, vt::DType::kQ8_K, "tq1_0"},
  };
  for (const TQCase& c : cases) {
    CAPTURE(c.name);
    CHECK(vt::IsBlockQuant(c.dtype));
    CHECK(vt::BlockElems(c.dtype) == c.block_elems);
    CHECK(vt::BlockBytes(c.dtype) == c.block_bytes);
    CHECK(vt::GgmlTypeId(c.dtype) == c.ggml_type);
    CHECK(std::string(vt::Name(c.dtype)) == c.name);
    CHECK_THROWS(vt::SizeOf(c.dtype));
    CHECK(vt::RowSizeBytes(c.dtype, c.block_elems) ==
          static_cast<size_t>(c.block_bytes));
    // The ggml type id round-trips back to the same vt dtype.
    vt::DType back = vt::DType::kF32;
    REQUIRE(vt::BlockDTypeFromGgmlTypeId(c.ggml_type, &back));
    CHECK(back == c.dtype);
    // Has a to_float dequantizer (CPU reference oracle for Vulkan tests).
    CHECK(vt::cpu::BlockToFloat(c.dtype) != nullptr);
    // Has a QuantTraits row with vec_dot_type = kQ8_K...
    const vt::cpu::QuantTypeTraits& t = vt::cpu::QuantTraits(c.dtype);
    CHECK(t.vec_dot_type == c.vec_dot_type);
    // Has a vec_dot against Q8_K activations (the CPU keep-quant tier that
    // reproduces the on-device Vulkan shader, which quantizes activations to Q8_K),
    // so HasQuantDotKernel is TRUE. Added with the TQ Vulkan shaders so the CPU
    // fallback is numerically consistent with the GPU and the keep-quant tests compare
    // against a Q8_K-quantized oracle at nmse<1e-6 (not the whole Q8_K
    // activation-quantization error of the old exact-f32 composite oracle).
    CHECK(t.vec_dot != nullptr);
    CHECK(vt::cpu::HasQuantDotKernel(c.dtype));
    // Nothing quantizes an activation INTO them (weight-only encodings).
    CHECK(vt::cpu::BlockFromFloat(c.dtype) == nullptr);
  }
}

TEST_CASE("elementwise dtypes are not block-quantized and reject block queries") {
  for (vt::DType d : {vt::DType::kF32, vt::DType::kF16, vt::DType::kBF16,
                      vt::DType::kI8, vt::DType::kI32, vt::DType::kI64}) {
    CHECK_FALSE(vt::IsBlockQuant(d));
    CHECK_THROWS(vt::BlockElems(d));
    CHECK_THROWS(vt::BlockBytes(d));
  }
  // The GGUF reader knows F32/F16/BF16 but they are not BLOCK dtypes, so the
  // id->block-dtype map must reject them rather than aliasing onto one.
  for (uint32_t id : {0U, 1U, 30U}) {
    vt::DType out = vt::DType::kF32;
    CHECK_FALSE(vt::BlockDTypeFromGgmlTypeId(id, &out));
  }
}

TEST_CASE("block dtypes are storage-only: SizeOf throws") {
  for (const BlockCase& c : kBlockCases) {
    CAPTURE(c.name);
    CHECK_THROWS(vt::SizeOf(c.dtype));
  }
}

TEST_CASE("RowSizeBytes mirrors ggml_row_size") {
  for (const BlockCase& c : kBlockCases) {
    CAPTURE(c.name);
    const int64_t be = c.block_elems;
    CHECK(vt::RowSizeBytes(c.dtype, be) == static_cast<size_t>(c.block_bytes));
    CHECK(vt::RowSizeBytes(c.dtype, 8 * be) ==
          static_cast<size_t>(8 * c.block_bytes));
    // ggml asserts rows are whole blocks.
    CHECK_THROWS(vt::RowSizeBytes(c.dtype, be + 1));
  }
  // Elementwise dtypes stay k * SizeOf so callers can use one helper.
  CHECK(vt::RowSizeBytes(vt::DType::kF32, 7) == 28);
  CHECK(vt::RowSizeBytes(vt::DType::kBF16, 7) == 14);
}

// --- The traits table --------------------------------------------------------

TEST_CASE("quant traits mirror type_traits_cpu (ggml-cpu.c:211-406)") {
  for (const BlockCase& c : kBlockCases) {
    CAPTURE(c.name);
    const vt::cpu::QuantTypeTraits& t = vt::cpu::QuantTraits(c.dtype);
    CHECK(t.vec_dot_type == c.vec_dot_type);
    // The generic tier pins nrows == 1; nrows == 2 arrives only with the i8mm
    // mmla kernels (G6) and their odd-shape boundary guards.
    CHECK(t.nrows == 1);
    // Every executable block type must decode.
    CHECK(t.to_float != nullptr);
    // G2/G3 populated the remaining columns: the six executable WEIGHT types
    // now have a vec_dot and can execute a quantized dot; Q8_K is the
    // ACTIVATION encoding, so it has a from_float but deliberately no vec_dot
    // (upstream gives it no type_traits_cpu row) and stays on the composite.
    // The full population contract is gated in test_ops_quant_dot.cpp.
    if (c.dtype == vt::DType::kQ8_K) {
      CHECK(t.vec_dot == nullptr);
      CHECK(t.from_float != nullptr);
      CHECK_FALSE(vt::cpu::HasQuantDotKernel(c.dtype));
    } else {
      CHECK(t.vec_dot != nullptr);
      CHECK(vt::cpu::HasQuantDotKernel(c.dtype));
    }
  }
  CHECK_THROWS(vt::cpu::QuantTraits(vt::DType::kF32));
}

TEST_CASE("BlockToFloat matches the GGUF loader's dequant byte-for-byte") {
  for (const BlockCase& c : kBlockCases) {
    CAPTURE(c.name);
    // Q8_K has no arm in the loader's expansion path: `DequantGgufRowToF32`
    // refuses it there, while the decode-only embedding gather takes it packed.
    if (c.dtype == vt::DType::kQ8_K) continue;
    constexpr int64_t kBlocks = 5;
    const std::vector<uint8_t> bytes = RandomBlocks(c, kBlocks, 1234U);
    const int64_t numel = kBlocks * c.block_elems;

    std::vector<float> direct(static_cast<size_t>(numel));
    vt::cpu::BlockToFloat(c.dtype)(bytes.data(), direct.data(), numel);

    const std::vector<float> loader =
        vllm::DequantGgufRowToF32(c.ggml_type, bytes.data(), numel);
    REQUIRE(loader.size() == direct.size());
    for (size_t i = 0; i < direct.size(); ++i) {
      // Byte-identical, not approximate: they are the same implementation.
      CHECK(loader[i] == direct[i]);
    }
  }
}

// --- The op skeleton ---------------------------------------------------------

namespace {

// Reference: dequantize the whole weight to f32, then the same f32 dot the
// composite performs. This is the `DequantGgufRowToF32 + MatmulBTKernel`
// oracle the spec's ported MUL_MAT cases use.
std::vector<float> ReferenceMatmul(const std::vector<float>& a,
                                   const std::vector<float>& w, int64_t m,
                                   int64_t k, int64_t n) {
  std::vector<float> out(static_cast<size_t>(m * n));
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float acc = 0.0F;
      for (int64_t p = 0; p < k; ++p) {
        acc += a[static_cast<size_t>(i * k + p)] *
               w[static_cast<size_t>(j * k + p)];
      }
      out[static_cast<size_t>(i * n + j)] = acc;
    }
  }
  return out;
}

}  // namespace

TEST_CASE("MatmulBTQuant generic-composite fallback == dequant-then-matmul") {
  // G2/G3 gave the six executable WEIGHT encodings a real vec_dot, so they now
  // take the quantized path (gated in test_ops_quant_dot.cpp against an
  // independent f64 reference). The generic composite this case covers remains
  // the fallback for any block dtype WITHOUT a vec_dot — today that is Q8_K,
  // the activation-only encoding — and it stays in the tree as the reference
  // the ported MUL_MAT NMSE cases measure the quantized path against.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const BlockCase& c = kBlockCases[6];  // q8_K
  REQUIRE(c.dtype == vt::DType::kQ8_K);
  REQUIRE_FALSE(vt::cpu::HasQuantDotKernel(c.dtype));

  // N weight rows of K elements each; M activation rows. K spans several
  // blocks so the per-row block walk is exercised.
  const int64_t k = 4 * c.block_elems;
  const int64_t n = 3;
  for (int64_t m : {int64_t{1}, int64_t{4}}) {
    CAPTURE(m);
    const std::vector<uint8_t> wq = RandomBlocks(c, n * (k / c.block_elems), 99U);

    std::vector<float> a(static_cast<size_t>(m * k));
    std::mt19937 rng(7U);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (float& v : a) v = dist(rng);

    // The reference decodes the SAME bytes through the traits `to_float`.
    // (The loader's DequantGgufRowToF32 has no Q8_K arm and refuses it there,
    // while the reader carries its file geometry — see the cross-check above.)
    std::vector<float> w(static_cast<size_t>(n * k));
    vt::cpu::BlockToFloat(c.dtype)(wq.data(), w.data(), n * k);
    const std::vector<float> expected = ReferenceMatmul(a, w, m, k, n);

    vt::Tensor at =
        vt::Tensor::Contiguous(a.data(), vt::DType::kF32, q.device, {m, k});
    vt::Tensor bt = vt::Tensor::Contiguous(const_cast<uint8_t*>(wq.data()),
                                           vt::DType::kF32, q.device, {n, k});
    bt.dtype = c.dtype;  // block dtype: strides are not meaningful
    std::vector<float> got(static_cast<size_t>(m * n));
    vt::Tensor ot =
        vt::Tensor::Contiguous(got.data(), vt::DType::kF32, q.device, {m, n});

    vt::MatmulBTQuant(q, ot, at, bt);
    for (size_t i = 0; i < got.size(); ++i) {
      CHECK(got[i] == doctest::Approx(expected[i]).epsilon(1e-6));
    }
  }
}

TEST_CASE("MatmulBTQuant rejects the shapes/dtypes ggml would assert on") {
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const int64_t k = 256, n = 2, m = 1;
  const std::vector<uint8_t> wq =
      RandomBlocks(kBlockCases[3] /*q4_K*/, n * (k / 256), 5U);
  std::vector<float> a(static_cast<size_t>(m * k), 0.5F);
  std::vector<float> got(static_cast<size_t>(m * n));

  vt::Tensor at =
      vt::Tensor::Contiguous(a.data(), vt::DType::kF32, q.device, {m, k});
  vt::Tensor ot =
      vt::Tensor::Contiguous(got.data(), vt::DType::kF32, q.device, {m, n});
  vt::Tensor bt = vt::Tensor::Contiguous(const_cast<uint8_t*>(wq.data()),
                                         vt::DType::kF32, q.device, {n, k});

  // An elementwise weight belongs on MatmulBT, not here.
  CHECK_THROWS(vt::MatmulBTQuant(q, ot, at, bt));

  bt.dtype = vt::DType::kQ4_K;
  // K that is not a whole number of blocks (ggml_row_size's assert).
  vt::Tensor a_bad = vt::Tensor::Contiguous(a.data(), vt::DType::kF32, q.device,
                                            {m, k - 1});
  vt::Tensor b_bad = bt;
  b_bad.shape[1] = k - 1;
  CHECK_THROWS(vt::MatmulBTQuant(q, ot, a_bad, b_bad));

  // Inner-dim mismatch.
  vt::Tensor a_k2 =
      vt::Tensor::Contiguous(a.data(), vt::DType::kF32, q.device, {m, 128});
  CHECK_THROWS(vt::MatmulBTQuant(q, ot, a_k2, bt));

  // The well-formed call still works (guards are not over-tight).
  CHECK_NOTHROW(vt::MatmulBTQuant(q, ot, at, bt));
}
