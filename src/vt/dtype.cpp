// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include "vt/dtype.h"

#include <cstring>

namespace vt {

namespace {

// Block geometry table for the ggml block-quantized encodings, ported from
// llama.cpp @ 237ad9b96 `ggml/src/ggml-common.h` (the block structs and their
// static_asserts) with the type ids from `ggml/include/ggml.h:390-432`:
//   Q4_0 (id 2)  ggml-common.h:213-218  f16 d + QK4_0/2 qs        = 2 + 16
//   Q8_0 (id 8)  ggml-common.h:242-245  f16 d + QK8_0 qs          = 2 + 32
//   Q3_K (id 11) ggml-common.h:305-310  QK_K/8 hmask + QK_K/4 qs
//                                       + 12 scales + f16 d       = 32+64+12+2
//   Q4_K (id 12) ggml-common.h:317-327  2*f16 dm + 12 scales
//                                       + QK_K/2 qs               = 4+12+128
//   Q5_K (id 13) ggml-common.h:334-345  2*f16 dm + 12 scales
//                                       + QK_K/8 qh + QK_K/2 qs   = 4+12+32+128
//   Q6_K (id 14) ggml-common.h:352-357  QK_K/2 ql + QK_K/4 qh
//                                       + QK_K/16 scales + f16 d  = 128+64+16+2
//   Q8_K (id 15) ggml-common.h:361-365  f32 d + QK_K qs
//                                       + QK_K/16 i16 bsums       = 4+256+32
// This table is INDEPENDENT of the GGUF reader's `GgmlTraits`
// (src/vllm/model_executor/model_loader/gguf_reader.cpp) on purpose: vt:: must
// not depend on the loader. `tests/vt/test_ops_quant_traits.cpp` cross-checks
// the two element-for-element, so a divergence in either port is caught.
struct BlockGeometry {
  int64_t block_elems;
  int64_t block_bytes;
  uint32_t ggml_type;
  const char* name;
};

const BlockGeometry* FindBlockGeometry(DType dtype) {
  switch (dtype) {
    case DType::kQ4_0: {
      static constexpr BlockGeometry g{32, 18, 2, "q4_0"};
      return &g;
    }
    case DType::kQ5_0: {
      // block_q5_0 (llama.cpp @ b10451 ggml/src/ggml-common.h:229-235):
      // f16 d + u8 qh[4] + u8 qs[QK5_0/2] = 2 + 4 + 16 = 22, QK5_0 = 32.
      // ggml type id 6 (ggml/include/ggml.h:396). The 5th bit of each quant
      // lives in the qh bitfield, split across the two halves of the block.
      static constexpr BlockGeometry g{32, 22, 6, "q5_0"};
      return &g;
    }
    case DType::kQ8_0: {
      static constexpr BlockGeometry g{32, 34, 8, "q8_0"};
      return &g;
    }
    case DType::kQ2_K: {
      // block_q2_K (ggml-common.h:288-299): u8 scales[16] + u8 qs[64]
      // + f16 d + f16 dmin = 16 + 64 + 2 + 2 = 84. ggml type id 10.
      static constexpr BlockGeometry g{256, 84, 10, "q2_K"};
      return &g;
    }
    case DType::kQ3_K: {
      static constexpr BlockGeometry g{256, 110, 11, "q3_K"};
      return &g;
    }
    case DType::kQ4_K: {
      static constexpr BlockGeometry g{256, 144, 12, "q4_K"};
      return &g;
    }
    case DType::kQ5_K: {
      static constexpr BlockGeometry g{256, 176, 13, "q5_K"};
      return &g;
    }
    case DType::kQ6_K: {
      static constexpr BlockGeometry g{256, 210, 14, "q6_K"};
      return &g;
    }
    case DType::kQ8_K: {
      static constexpr BlockGeometry g{256, 292, 15, "q8_K"};
      return &g;
    }
    case DType::kIQ2_XXS: {
      // block_iq2_xxs (ggml-common.h:371-374): f16 d + u16 qs[32]
      // = 2 + 64 = 66. ggml type id 16. Codebook (iq2xxs_grid) decode.
      static constexpr BlockGeometry g{256, 66, 16, "iq2_xxs"};
      return &g;
    }
    case DType::kIQ3_XXS: {
      // block_iq3_xxs (ggml-common.h:385-400): f16 d + u8 qs[3*QK_K/8]
      // = 2 + 96 = 98. ggml type id 18. Codebook (iq3xxs_grid) decode;
      // qs[0..63] grid indices, qs[64..95] the per-32 scale+sign u32s.
      static constexpr BlockGeometry g{256, 98, 18, "iq3_xxs"};
      return &g;
    }
    case DType::kIQ2_S: {
      // block_iq2_s (ggml-common.h:386-392): f16 d + u8 qs[QK_K/4]
      // + u8 qh[QK_K/32] + u8 scales[QK_K/32] = 2 + 64 + 8 + 8 = 82.
      // ggml type id 22. Codebook (iq2s_grid, 1024 entries) decode; qs holds
      // both the 8-bit grid-index low bytes (first 32) and the direct sign
      // bytes (last 32), qh supplies the 2 high index bits per lane.
      static constexpr BlockGeometry g{256, 82, 22, "iq2_s"};
      return &g;
    }
    case DType::kIQ1_S: {
      // block_iq1_s (ggml-common.h:414-419): f16 d + u8 qs[QK_K/8]
      // + u16 qh[QK_K/32] = 2 + 32 + 16 = 50, i.e. 1.5625 bpw. ggml type id 19.
      // Codebook (iq1s_grid, 2048 entries) decode on an 11-bit index; the grid
      // entries are packed ternary (-1/0/+1) bytes and qh carries the high index
      // bits, the per-32 scale and the delta sign all at once.
      static constexpr BlockGeometry g{256, 50, 19, "iq1_s"};
      return &g;
    }
    case DType::kIQ1_XXXS: {
      // block_iq1_xxxs, from the PINNED FORK oracle `llama-cpp-unsloth`
      // (ggml-common.h:478-483 @ 36fe8e1cc): f16 d + u8 qs[QK_K/8]
      // + u8 sc[QK_K/64] = 2 + 32 + 4 = 38, i.e. 1.1875 bpw. ggml type id 66,
      // which NO upstream llama.cpp defines. The grid holds 256 entries, so qs
      // is a whole 8-bit index with no high bits to splice; sc packs one nibble
      // per 32-element sub-block, bits 0-2 the scale and bit 3 the delta sign.
      static constexpr BlockGeometry g{256, 38, 66, "iq1_xxxs"};
      return &g;
    }
    case DType::kIQ1_M: {
      // block_iq1_m (llama.cpp @ b10451 ggml-common.h:429-437): QK_K/8 u8 qs
      // + QK_K/16 u8 qh + QK_K/32 u8 scales = 56, i.e. 1.75 bpw. ggml type
      // id 29 (ggml/include/ggml.h). NO f16 field anywhere in the block: the
      // super-block scale is an f16 assembled from the four TOP nibbles of
      // scales[0..3], and the same bytes carry the eight 3-bit sub-block
      // scales in their low 6 bits. Codebook (iq1s_grid, 2048 entries) decode
      // on an 11-bit index, like IQ1_S.
      static constexpr BlockGeometry g{256, 56, 29, "iq1_m"};
      return &g;
    }
    case DType::kIQ4_NL: {
      // block_iq4_nl (llama.cpp @ b10451 ggml/src/ggml-common.h:447-452):
      // f16 d + u8 qs[QK4_NL/2] = 2 + 16 = 18, QK4_NL = 32. ggml type id 20
      // (ggml/include/ggml.h:410). Q4_0's geometry exactly, but the nibble
      // indexes the 16-entry NON-LINEAR codebook `kvalues_iq4nl`
      // (ggml-common.h:1120) instead of being an affine `nibble - 8`, so the
      // two decoders are NOT interchangeable despite the identical layout.
      static constexpr BlockGeometry g{32, 18, 20, "iq4_nl"};
      return &g;
    }
    case DType::kIQ2_XS: {
      // block_iq2_xs (llama.cpp @ b10451 ggml/src/ggml-common.h:388-392):
      // f16 d + u16 qs[QK_K/8] + u8 scales[QK_K/32] = 2 + 64 + 8 = 74, i.e.
      // 2.3125 bpw. ggml type id 17 (ggml/include/ggml.h:407). Codebook
      // (iq2xs_grid, 512 entries) decode: each u16 of `qs` carries a 9-bit grid
      // index in its low bits and a 7-bit ksigns selector in its high bits.
      static constexpr BlockGeometry g{256, 74, 17, "iq2_xs"};
      return &g;
    }
    case DType::kIQ4_XS: {
      // block_iq4_xs (llama.cpp @ b10451 ggml/src/ggml-common.h:454-459):
      // f16 d + u16 scales_h + u8 scales_l[QK_K/64] + u8 qs[QK_K/2]
      // = 2 + 2 + 4 + 128 = 136. ggml type id 23 (ggml/include/ggml.h:413).
      // Same 16-entry `kValuesIq4nl` codebook as IQ4_NL, so the DELTA is the
      // super-block scale layout: a 6-bit `ls` spliced from a `scales_l` nibble
      // and a `scales_h` bit pair, biased by -32, per 32 elements.
      static constexpr BlockGeometry g{256, 136, 23, "iq4_xs"};
      return &g;
    }
    case DType::kIQ3_S: {
      // block_iq3_s (llama.cpp @ b10451 ggml/src/ggml-common.h:413-422):
      // f16 d + u8 qs[QK_K/4] + u8 qh[QK_K/32] + u8 signs[QK_K/8]
      // + u8 scales[QK_K/64] = 2 + 64 + 8 + 32 + 4 = 110, i.e. 3.4375 bpw.
      // ggml type id 21 (ggml/include/ggml.h:411). The 110 is the ORACLE's own
      // `sizeof(block_iq3_s)`, printed by the harness in
      // tests/vt/iq3s_golden_vectors.h, not a sum read off the struct.
      // Codebook (iq3s_grid, 512 entries) decode with the ninth index bit
      // spliced out of `qh` and DIRECT sign bytes; see include/vt/dtype.h for
      // why none of the IQ3_XXS decode transfers.
      static constexpr BlockGeometry g{256, 110, 21, "iq3_s"};
      return &g;
    }
    case DType::kMXFP4: {
      // block_mxfp4 (ggml-common.h:204-209): u8 e (E8M0 shared exponent)
      // + u8 qs[QK_MXFP4/2] = 1 + 16 = 17, QK_MXFP4 = 32. ggml type id 39.
      // OCP micro-scaling fp4: e2m1 nibbles times one power-of-two block scale.
      static constexpr BlockGeometry g{32, 17, 39, "mxfp4"};
      return &g;
    }
    case DType::kTQ2_0: {
      // block_tq2_0 (llama.cpp PR #25850): u8 qs[64] + f16 d = 64 + 2 = 66,
      // QK = 256. ggml type id 42 (killgate fork extension after Q1_0 = 41).
      // 2-bit ternary codes: element e = j*4 + l*32 + k (j in {0,32},
      // l in 0..3, k in 0..31) reads byte qs[j+k], value ((qs[j+k]>>(l*2))&3)-1.
      static constexpr BlockGeometry g{256, 66, 42, "tq2_0"};
      return &g;
    }
    case DType::kTQ1_0: {
      // block_tq1_0: u8 qs[48] + u8 qh[4] + f16 d = 48 + 4 + 2 = 54,
      // QK = 256. ggml type id 43. Packed base-3 trits: q = byte * pow3[l]
      // (uint8 wrap); xi = (q * 3) >> 8; w = (xi - 1) * d, giving {-1, 0, +1}.
      static constexpr BlockGeometry g{256, 54, 43, "tq1_0"};
      return &g;
    }
    case DType::kF32:
    case DType::kF16:
    case DType::kBF16:
    case DType::kI8:
    case DType::kI32:
    case DType::kI64:
      return nullptr;
  }
  return nullptr;
}


const BlockGeometry& RequireBlockGeometry(DType dtype) {
  const BlockGeometry* g = FindBlockGeometry(dtype);
  VT_CHECK(g != nullptr, std::string("dtype ") + Name(dtype) +
                             " is not block-quantized");
  return *g;
}

}  // namespace

bool IsBlockQuant(DType dtype) { return FindBlockGeometry(dtype) != nullptr; }

bool IsFloat(DType dtype) {
  return dtype == DType::kF32 || dtype == DType::kBF16 || dtype == DType::kF16;
}

int64_t BlockElems(DType dtype) { return RequireBlockGeometry(dtype).block_elems; }

int64_t BlockBytes(DType dtype) { return RequireBlockGeometry(dtype).block_bytes; }

uint32_t GgmlTypeId(DType dtype) { return RequireBlockGeometry(dtype).ggml_type; }

bool BlockDTypeFromGgmlTypeId(uint32_t ggml_type, DType* out) {
  static constexpr DType kBlockDTypes[] = {
      DType::kQ4_0, DType::kQ5_0,  DType::kQ8_0,     DType::kQ2_K, DType::kQ3_K,
      DType::kQ4_K, DType::kQ5_K,  DType::kQ6_K,     DType::kQ8_K,
      DType::kIQ2_XXS, DType::kIQ3_XXS, DType::kIQ2_S, DType::kMXFP4,
      DType::kIQ1_S, DType::kIQ1_M, DType::kIQ1_XXXS, DType::kIQ4_NL,
      DType::kIQ2_XS, DType::kIQ4_XS, DType::kIQ3_S,
      DType::kTQ2_0, DType::kTQ1_0};
  for (DType d : kBlockDTypes) {
    if (FindBlockGeometry(d)->ggml_type == ggml_type) {
      if (out != nullptr) *out = d;
      return true;
    }
  }
  return false;
}

size_t RowSizeBytes(DType dtype, int64_t k) {
  VT_CHECK(k >= 0, "RowSizeBytes: negative element count");
  const BlockGeometry* g = FindBlockGeometry(dtype);
  if (g == nullptr) return static_cast<size_t>(k) * SizeOf(dtype);
  // ggml_row_size asserts ne % blck_size == 0: a row is whole blocks.
  VT_CHECK(k % g->block_elems == 0,
           std::string("RowSizeBytes: ") + std::to_string(k) +
               " elements is not a whole number of " + g->name + " blocks");
  return static_cast<size_t>(k / g->block_elems) *
         static_cast<size_t>(g->block_bytes);
}

// The cold half of the now-inline `SizeOf` (include/vt/dtype.h). Both are
// [[noreturn]] and both keep the exact message the out-of-line SizeOf threw, so
// nothing that catches or matches on that text changes.
void ThrowBlockQuantHasNoElementSize(DType dtype) {
  VT_CHECK(false, std::string("SizeOf: block-quantized dtype ") + Name(dtype) +
                      " has no per-element size");
  throw std::runtime_error("unreachable");
}

void ThrowUnknownDType() {
  VT_CHECK(false, "unknown dtype");
  throw std::runtime_error("unreachable");
}

const char* Name(DType dtype) {
  switch (dtype) {
    case DType::kF32: return "f32";
    case DType::kF16: return "f16";
    case DType::kBF16: return "bf16";
    case DType::kI8: return "i8";
    case DType::kI32: return "i32";
    case DType::kI64: return "i64";
    case DType::kQ4_0: return "q4_0";
    case DType::kQ5_0: return "q5_0";
    case DType::kQ8_0: return "q8_0";
    case DType::kQ2_K: return "q2_K";
    case DType::kQ3_K: return "q3_K";
    case DType::kQ4_K: return "q4_K";
    case DType::kQ5_K: return "q5_K";
    case DType::kQ6_K: return "q6_K";
    case DType::kQ8_K: return "q8_K";
    case DType::kIQ2_XXS: return "iq2_xxs";
    case DType::kIQ3_XXS: return "iq3_xxs";
    case DType::kIQ2_S: return "iq2_s";
    case DType::kIQ1_S: return "iq1_s";
    case DType::kIQ1_XXXS: return "iq1_xxxs";
    case DType::kIQ1_M: return "iq1_m";
    case DType::kIQ4_NL: return "iq4_nl";
    case DType::kMXFP4: return "mxfp4";
    case DType::kIQ2_XS: return "iq2_xs";
    case DType::kIQ4_XS: return "iq4_xs";
    case DType::kIQ3_S: return "iq3_s";
    case DType::kTQ2_0: return "tq2_0";
    case DType::kTQ1_0: return "tq1_0";
  }
  return "?";
}

namespace {
uint32_t AsU32(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
}  // namespace

uint16_t F32ToF16(float f) {
  uint32_t u = AsU32(f);
  uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000);
  int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = u & 0x7FFFFF;
  if (((u >> 23) & 0xFF) == 0xFF) {  // inf/nan
    return static_cast<uint16_t>(sign | 0x7C00 | (mant ? 0x200 | (mant >> 13) : 0));
  }
  if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00);  // overflow → inf
  if (exp <= 0) {
    if (exp < -10) return sign;  // underflow → zero
    // subnormal
    mant |= 0x800000;
    uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t half = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1);
    uint32_t mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1))) ++half;  // round to nearest even
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = static_cast<uint32_t>(exp << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) ++half;  // may carry into exp: correct
  return static_cast<uint16_t>(sign | half);
}

uint16_t F32ToBF16(float f) {
  uint32_t u = AsU32(f);
  if ((u & 0x7F800000) == 0x7F800000 && (u & 0x7FFFFF)) {  // nan: keep quiet, truncate
    return static_cast<uint16_t>((u >> 16) | 0x0040);
  }
  uint32_t rounding = 0x7FFF + ((u >> 16) & 1);  // round to nearest even
  return static_cast<uint16_t>((u + rounding) >> 16);
}

}  // namespace vt
