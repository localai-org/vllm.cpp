// QUANT-GGUF-CIQ-GEMM work rows G2 (activation quant) and G3 (the six tier-0
// generic `vec_dot` kernels + their GEMM wiring).
//
// Ported from llama.cpp @ 237ad9b96:
//   tests/test-quantize-fns.cpp   — the RMSE/dot-product bounds at :17-28
//     (MAX_QUANTIZATION_TOTAL_ERROR 0.002, MAX_QUANTIZATION_REFERENCE_ERROR
//     0.0001, MAX_DOT_PRODUCT_ERROR 0.02, ..._LOWBIT 0.04), the synthetic data
//     generator at :35, `array_rmse` at :41, `dot_product_error` at :86, and
//     the 32*128 test size at :132. Upstream's thresholds are used as-is; none
//     is widened.
//   tests/test-backend-ops.cpp    — the MUL_MAT NMSE bound `max_nmse_err()`
//     = 5e-4 at :4277-4279, applied to `kMatmulBTQuant` at model-ish shapes.
//
// WHY THE PRIMARY GATE IS NOT AN UPSTREAM PORT.
// A `vec_dot` is exactly where a subtle block-decode slip yields
// plausible-but-wrong numbers, and checking one copy of the decode against
// another copy of the same logic would not catch it. So the load-bearing case
// here — "vec_dot agrees with an INDEPENDENT f64 reference" — dequantizes both
// operands through `BlockToFloat` (the loader-side `dequantize_row_*` decoders,
// a SEPARATE port that walks the block layout differently from the inline
// decode inside each vec_dot) and dots them in DOUBLE precision. The two agree
// exactly in exact arithmetic, so the check can be tight: the tolerance is
// relative to the dot's L1 magnitude, which makes it immune to the sign
// cancellation that would otherwise let a large error hide behind a small sum.
//
// SCOPE NOTE — what is honestly NOT covered. Upstream's `dot_product_error`
// quantizes BOTH operands with `from_float`. We port `from_float` only for the
// two activation encodings Q8_0 and Q8_K (that is all of G2's scope — nothing
// in this project ever quantizes an activation into a k-quant), so the full
// upstream round-trip is reproduced for Q8_0 and the k-quants are gated on the
// f64 reference above with random-bit-pattern blocks instead. Those random
// blocks are strictly HARDER on the decode than encoder output: they exercise
// bit patterns a real encoder never emits.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <iterator>

#include "iq1_golden_vectors.h"  // oracle-produced IQ1_S / IQ1_XXXS goldens
#include "iq1m_golden_vectors.h"  // oracle-produced IQ1_M goldens
#include "iq2xs_iq4xs_dot_golden.h"  // oracle-produced IQ2_XS / IQ4_XS dots
#include "iq2xs_iq4xs_golden_vectors.h"  // the same artifact bytes, decoded
#include "vt/cpu/cpu_threadpool.h"  // Threadpool::SwapForTesting (via -I src)
#include "vt/cpu/cpu_quant_iq_tables.h"  // kIq1sGrid provenance check
#if defined(__x86_64__) || defined(_M_X64)
#include "vt/cpu/cpu_isa_x86.h"
#endif
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/quant.h"
#include "vt/tensor.h"

#if defined(_WIN32)
#define VT_TEST_SETENV(name, value) _putenv_s((name), (value))
#define VT_TEST_UNSETENV(name) _putenv_s((name), "")
#else
#define VT_TEST_SETENV(name, value) setenv((name), (value), 1)
#define VT_TEST_UNSETENV(name) unsetenv((name))
#endif

namespace {

// --- upstream thresholds, test-quantize-fns.cpp:17-28 -----------------------
constexpr float kMaxQuantizationReferenceError = 0.0001F;
constexpr float kMaxQuantizationTotalError = 0.002F;
constexpr float kMaxDotProductError = 0.02F;
// test-backend-ops.cpp:4277-4279
constexpr double kMaxNmseErr = 5e-4;
// test-quantize-fns.cpp:132
constexpr size_t kTestSize = 32 * 128;

// test-quantize-fns.cpp:35 — the synthetic signal upstream measures on.
void GenerateData(float offset, size_t n, float* dst) {
  for (size_t i = 0; i < n; i++) {
    dst[i] = 0.1F + 2 * std::cos(static_cast<float>(i) + offset);
  }
}

// test-quantize-fns.cpp:41 — note the normalization is sqrt(sum)/n, NOT
// sqrt(sum/n). Kept verbatim because the thresholds above are calibrated to it.
float ArrayRmse(const float* a1, const float* a2, size_t n) {
  double sum = 0;
  for (size_t i = 0; i < n; i++) {
    double diff = static_cast<double>(a1[i]) - static_cast<double>(a2[i]);
    sum += diff * diff;
  }
  return std::sqrt(sum) / static_cast<float>(n);
}

// --- block field offsets, written out FRESH from ggml-common.h --------------
// Deliberately independent of src/vt/cpu/cpu_quant_blocks.h (the struct mirror
// the kernels use) so this file is a third statement of the same layout facts,
// in the same spirit as the G1 trait cross-check.
struct WeightCase {
  vt::DType dtype;
  int64_t block_elems;
  int64_t block_bytes;
  int d_off;      // ggml_half super-block scale, -1 when absent
  int dmin_off;   // ggml_half super-block min scale, -1 when absent
  int e8m0_off;   // u8 E8M0 (power-of-two) block scale, -1 when absent (MXFP4)
  const char* name;
  // NMSE ceiling for the MatmulBTQuant-vs-f32 case below. 0 means "use
  // kMaxNmseErr". This is per-CASE because that test does not bound decode at
  // all: it compares a Q8_K-quantized activation against an f32 reference, so
  // what it measures is ACTIVATION error, and how far that error travels
  // depends on the weight distribution the encoding produces. Decode itself is
  // bounded exactly, and shared across every case, by "vec_dot == independent
  // f64 dequantize-then-dot" and "MatmulBTQuant matches per-row vec_dot".
  double nmse_max = 0.0;
};

// Byte offsets are the running field sums of each ggml-common.h struct. The
// K-quants do NOT agree on where the delta sits — q4_K/q5_K lead with it while
// q3_K/q6_K trail it — which is exactly the kind of detail a port gets wrong,
// so each is spelled out:
//   q4_0 :213-218  d@0  qs@2                                    (18B)
//   q8_0 :242-245  d@0  qs@2                                    (34B)
//   q3_K :305-310  hmask@0 qs@32 sc@96 d@108                    (110B)
//   q4_K :317-327  d@0 dmin@2 sc@4 qs@16                        (144B)
//   q5_K :334-345  d@0 dmin@2 sc@4 qh@16 qs@48                  (176B)
//   q6_K :352-357  ql@0 qh@128 sc@192 d@208                     (210B)
// q2_K :288-299   scales@0 qs@16 d@80 dmin@82                    (84B)
// iq2_xxs :371-374 d@0 qs@2 (u16[32])                            (66B)
// iq3_xxs :385-400 d@0 qs@2 (u8[96]: 64 grid idx + 32 sc/sig)    (98B)
// iq2_s   :386-392 d@0 qs@2 (64: 32 grid idx + 32 signs) qh@66 sc@74 (82B)
// mxfp4   :204-209 e8m0@0 qs@1 (16: 32 e2m1 nibbles)             (17B)
const WeightCase kWeightCases[] = {
    {vt::DType::kQ4_0, 32, 18, 0, -1, -1, "q4_0"},
    {vt::DType::kQ8_0, 32, 34, 0, -1, -1, "q8_0"},
    {vt::DType::kQ2_K, 256, 84, 80, 82, -1, "q2_K"},
    {vt::DType::kQ3_K, 256, 110, 108, -1, -1, "q3_K"},
    {vt::DType::kQ4_K, 256, 144, 0, 2, -1, "q4_K"},
    {vt::DType::kQ5_K, 256, 176, 0, 2, -1, "q5_K"},
    {vt::DType::kQ6_K, 256, 210, 208, -1, -1, "q6_K"},
    // DeepSeek-V4 W8 keep-quant enablers — the codebook / fp4 encodings the
    // single-Spark routed experts use: UD-IQ2_XXS (IQ2_XXS gate/up, IQ3_XXS
    // down; Q2_K is the UD-Q2_K_XL sibling) and UD-IQ2_M (IQ2_S gate/up dotting
    // Q8_K, MXFP4 down dotting Q8_0 — the one 32-element / Q8_0-activation type
    // here besides q4_0/q8_0, so it also exercises that path for a codebook).
    {vt::DType::kIQ2_XXS, 256, 66, 0, -1, -1, "iq2_xxs"},
    {vt::DType::kIQ3_XXS, 256, 98, 0, -1, -1, "iq3_xxs"},
    {vt::DType::kIQ2_S, 256, 82, 0, -1, -1, "iq2_s"},
    // IQ1_S (1.5625 bpw) is 96.92 % of the parameters of the
    // Qwen3.8-2.4T-A95B UD-IQ1_S checkpoint ENG-EXPERT-STREAM targets, and
    // IQ1_XXXS (1.1875 bpw) is 96.92 % of the UD-Q1_0 one. Both need a ceiling
    // above upstream's 5e-4: their weights are ternary times a per-32 scale, so
    // a super-block spans a wider dynamic range than a 4-6 bit codebook does,
    // while Q8_K gives the activation ONE scale per 256 elements.
    //
    // 6e-4, set from the measurement rather than chosen for headroom. Every
    // shape in this case, all 12 per type, re-measured 16 August 2026 with the
    // ceiling forced to 1e-12 so doctest prints each captured value:
    //
    //   iq1_s     max 5.240e-4 at m=4  n=1, then 3.498e-4 at m=1 n=1
    //   iq1_xxxs  max 3.109e-4 at m=1  n=1, then 2.637e-5 at m=1 n=7
    //
    // The n=1 column is where both the signal and the noise live: the NMSE
    // denominator there is a single dot product, so it neither averages nor
    // cancels. Every n=7 and n=16 shape sits below 3e-5.
    //
    // An earlier revision set this to 2e-3, described as ~4x the residual. That
    // was a relaxation past the point where the statistic can discriminate.
    // Doubling kIq1sDelta to 0.25 moves iq1_s to 6.967e-4, which 6e-4 fails and
    // 2e-3 passes. Measure before widening a bound.
    //
    // Be precise about what this bound is worth, because the same mutation
    // moves iq1_xxxs the WRONG WAY: 3.109e-4 unmutated against 1.420e-4 with the
    // doubled delta, so NO ceiling catches that defect on this statistic. An
    // NMSE against a dequant-f32 reference cannot seal a decode parameter at
    // all, since both sides decode through the same `BlockToFloat`. What seals
    // it is the pair of golden-vector cases below, which compare against the
    // ORACLES bit for bit. This ceiling bounds quantization error, which is its
    // own job, and is kept tight enough to stay a second signal where it can be
    // one.
    {vt::DType::kIQ1_S, 256, 50, 0, -1, -1, "iq1_s", 6e-4},
    {vt::DType::kIQ1_XXXS, 256, 38, 0, -1, -1, "iq1_xxxs", 6e-4},
    // IQ1_M (1.75 bpw, ggml id 29) is a STANDARD llama.cpp encoding, but this
    // tree gained it for one artifact: 4 attention-gate tensors inside
    // `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`'s IQ3_XXS file, which `GgufFile::Open`
    // refused whole on the unknown id. Its block carries NO f16 field at all —
    // the super-block scale is spliced out of four nibbles of `scales` — so
    // d_off is -1 and RandomBlocks pins the packed scale directly.
    {vt::DType::kIQ1_M, 256, 56, -1, -1, -1, "iq1_m", 6e-4},
    {vt::DType::kMXFP4, 32, 17, -1, -1, 0, "mxfp4"},
    // MODEL-MM-QWEN4-EXP W6a. `unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S` stores
    // all 48 `ffn_down_exps` as IQ4_NL, so those GEMMs run `VecDotIQ4_NLQ8_0`;
    // Q5_0 arrives with it because llama.cpp's own `tensor_type_fallback`
    // (src/llama-quant.cpp:374-405 @ b10451) maps `Q4_K -> Q5_0` for the same
    // non-multiple-of-256 rows that force `IQ4_XS -> IQ4_NL`. Both were shipped
    // in #1989 with no numerical gate at all: review mutations R12 (swap
    // IQ4_NL's two nibble halves) and R13 (change Q5_0's upper-half `qh` shift)
    // each survived all eight suites, this one included, because these two rows
    // were missing. Both dot against Q8_0, which already has a `from_float`, so
    // the harness above reaches them unchanged — MXFP4 is the precedent.
    //   q5_0    :229-235  d@0  qh@2 (u8[4])  qs@6 (u8[16])       (22B)
    //   iq4_nl  :447-452  d@0  qs@2 (u8[16]: 32 codebook nibbles) (18B)
    {vt::DType::kQ5_0, 32, 22, 0, -1, -1, "q5_0"},
    {vt::DType::kIQ4_NL, 32, 18, 0, -1, -1, "iq4_nl"},
    // QUANT-GGUF-IQ-VECDOT. The staged `unsloth/GLM-5.3-Flash-GGUF UD-Q2_K_XL`
    // artifact stores 82 tensors as IQ2_XS and 3 as IQ4_XS, and #2245 gave both
    // a decoder but no `vec_dot` — which left `HasQuantDotKernel` false, so the
    // loader expanded every one of them to bf16. Both dot against Q8_K
    // (ggml-cpu.c:342-347, :385-390); IQ4_XS does NOT inherit IQ4_NL's Q8_0
    // pairing even though it reuses `kValuesIq4nl`, because its block is a
    // 256-element super-block rather than 32 elements.
    //   iq2_xs :388-393  d@0 qs@2 (u16[32]) scales@66 (u8[8])     (74B)
    //   iq4_xs :454-460  d@0 scales_h@2 (u16) scales_l@4 (u8[4]) qs@8 (128B)
    {vt::DType::kIQ2_XS, 256, 74, 0, -1, -1, "iq2_xs"},
    {vt::DType::kIQ4_XS, 256, 136, 0, -1, -1, "iq4_xs"},
};

// Random raw blocks: every quant/scale payload byte is arbitrary (all legal),
// with only the f16 delta fields pinned to finite, modest values so the dot
// stays in range. Random payloads are the point — they sweep bit patterns
// (including the 6-bit packed-scale corner cases and every hmask/qh bit) that
// an encoder would never produce.
std::vector<uint8_t> RandomBlocks(const WeightCase& c, int64_t nblocks,
                                  uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> bytes(static_cast<size_t>(nblocks * c.block_bytes));
  for (uint8_t& b : bytes) b = static_cast<uint8_t>(rng() & 0xFF);
  for (int64_t i = 0; i < nblocks; ++i) {
    uint8_t* blk = bytes.data() + i * c.block_bytes;
    auto put_f16 = [&](int off, float v) {
      const uint16_t h = vt::F32ToF16(v);
      std::memcpy(blk + off, &h, sizeof(h));
    };
    // Vary the deltas per block so a kernel that hoisted the scale out of the
    // block loop would be caught.
    const float jitter = 1.0F + 0.05F * static_cast<float>(i % 7);
    if (c.d_off >= 0) put_f16(c.d_off, 0.0125F * jitter);
    if (c.dmin_off >= 0) put_f16(c.dmin_off, 0.0075F * jitter);
    // MXFP4's block scale is a single E8M0 byte (exponent, value 2^(byte-128)),
    // NOT an f16. Random bytes would decode to wildly out-of-range 2^127 scales
    // (inf dots), so pin a small, per-block-varying exponent instead.
    if (c.e8m0_off >= 0) {
      bytes[static_cast<size_t>(i * c.block_bytes + c.e8m0_off)] =
          static_cast<uint8_t>(0x7E + (i % 5));  // 2^-2 .. 2^2
    }
    // IQ1_S carries its per-32 sub-block scale INSIDE qh (bits 12-14, decoding
    // to 2*ls+1), so uniformly random qh spreads neighbouring 32-groups across
    // a 15x scale range. That is not a decode question but an activation one:
    // Q8_K holds ONE scale per 256 elements, so a 15x spread within the block
    // inflates relative error (measured NMSE 3.8e-4 random vs 3.8e-5 here,
    // against a 5e-4 bound) for weights no encoder would emit. Same class of
    // fix as the MXFP4 exponent directly above, and for the same reason.
    //
    // The scale still VARIES per sub-block and per block, so a kernel that
    // dropped or hoisted it is still caught; only the pathological dynamic
    // range is removed. Every other IQ1_S bit stays random: the 11-bit grid
    // index, the delta sign (bit 15) and all of qs.
    if (c.dtype == vt::DType::kIQ1_S) {
      for (int ib = 0; ib < 8; ++ib) {
        uint16_t qh = 0;
        std::memcpy(&qh, blk + 34 + 2 * ib, sizeof(qh));
        const uint16_t ls = static_cast<uint16_t>(2 + ((i + ib) % 3));
        qh = static_cast<uint16_t>((qh & 0x8FFFU) | (ls << 12));
        std::memcpy(blk + 34 + 2 * ib, &qh, sizeof(qh));
      }
    }
    // IQ1_M splices its super-block scale from the u16* view of scales:
    // h = (sc[0]>>12) | ((sc[1]>>8)&0xf0) | ((sc[2]>>4)&0xf00) | (sc[3]&0xf000),
    // i.e. the nibbles live in scales[1] hi, scales[3] lo, scales[5] hi,
    // scales[7] hi. All other scales bits stay random (they are the eight
    // 3-bit sub-block scales), as do qs, qh and every delta sign.
    if (c.dtype == vt::DType::kIQ1_M) {
      const uint16_t h = static_cast<uint16_t>(0x2C00U + 13U * (i % 4));
      uint8_t* sc = blk + 48;
      sc[1] = static_cast<uint8_t>((sc[1] & 0x0FU) | ((h & 0xFU) << 4));
      sc[3] = static_cast<uint8_t>((sc[3] & 0xF0U) | ((h >> 4) & 0xFU));
      sc[5] = static_cast<uint8_t>((sc[5] & 0x0FU) | ((h >> 8) & 0xF0U));
      sc[7] = static_cast<uint8_t>((sc[7] & 0x0FU) | ((h >> 12) & 0xF0U));
    }
    // IQ1_XXXS packs the same scale, plus the delta sign, into one NIBBLE of
    // sc (bits 0-2 scale, bit 3 sign), two sub-blocks per byte. Same narrowing
    // and same reason as IQ1_S above; the sign bit stays random.
    if (c.dtype == vt::DType::kIQ1_XXXS) {
      for (int ib = 0; ib < 8; ++ib) {
        uint8_t& byte = blk[34 + ib / 2];
        const int shift = 4 * (ib & 1);
        const uint8_t ls = static_cast<uint8_t>(2 + ((i + ib) % 3));
        const uint8_t keep_sign = static_cast<uint8_t>((byte >> shift) & 0x8);
        byte = static_cast<uint8_t>((byte & ~(0xFU << shift)) |
                                    ((keep_sign | ls) << shift));
      }
    }
  }
  return bytes;
}

// The INDEPENDENT reference: decode both operands through the loader-side
// `to_float` decoders and accumulate in f64. Returns the dot and, separately,
// the L1 magnitude sum(|w*a|) used to set a cancellation-robust tolerance.
struct RefDot {
  double dot;
  double l1;
};

RefDot ReferenceDotF64(vt::DType wtype, const uint8_t* wq, vt::DType atype,
                       const uint8_t* aq, int64_t k) {
  std::vector<float> w(static_cast<size_t>(k));
  std::vector<float> a(static_cast<size_t>(k));
  vt::cpu::BlockToFloat(wtype)(wq, w.data(), k);
  vt::cpu::BlockToFloat(atype)(aq, a.data(), k);
  RefDot r{0.0, 0.0};
  for (int64_t i = 0; i < k; ++i) {
    // A non-finite REFERENCE means the synthetic blocks are malformed (a delta
    // field left as random bytes decoding to inf/nan), not that a kernel is
    // wrong. Failing here keeps that diagnosis unambiguous — it is how the
    // q6_K delta offset (208, not 0: q4_K/q5_K lead with the delta, q6_K
    // trails it) was caught while writing this file.
    REQUIRE(std::isfinite(w[static_cast<size_t>(i)]));
    REQUIRE(std::isfinite(a[static_cast<size_t>(i)]));
    const double t = static_cast<double>(w[static_cast<size_t>(i)]) *
                     static_cast<double>(a[static_cast<size_t>(i)]);
    r.dot += t;
    r.l1 += std::fabs(t);
  }
  return r;
}

// Quantize an f32 activation row into the encoding `wtype` dots against.
std::vector<uint8_t> QuantizeActivation(vt::DType wtype, const float* x,
                                        int64_t k) {
  const vt::DType at = vt::cpu::QuantTraits(wtype).vec_dot_type;
  std::vector<uint8_t> q(vt::RowSizeBytes(at, k));
  vt::cpu::QuantTraits(at).from_float(x, q.data(), k);
  return q;
}

float RunVecDot(vt::DType wtype, const uint8_t* wq, const uint8_t* aq,
                int64_t k) {
  float s = 0.0F;
  vt::cpu::QuantTraits(wtype).vec_dot(static_cast<int>(k), &s, 0, wq, 0, aq, 0,
                                      1);
  return s;
}

float RunVecDotFn(vt::cpu::VecDotFn fn, const uint8_t* wq,
                  const uint8_t* aq, int64_t k) {
  float s = 0.0F;
  fn(static_cast<int>(k), &s, 0, wq, 0, aq, 0, 1);
  return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// G2 + G3 — the traits table is populated
// ---------------------------------------------------------------------------

TEST_CASE("G2/G3 populate from_float and vec_dot (ggml-cpu.c:211-406)") {
  for (const WeightCase& c : kWeightCases) {
    CAPTURE(std::string(c.name));
    const vt::cpu::QuantTypeTraits& t = vt::cpu::QuantTraits(c.dtype);
    CHECK(t.to_float != nullptr);
    CHECK(t.vec_dot != nullptr);           // G3
    CHECK(t.nrows == 1);                   // nrows==2 is i8mm-only (G6)
    // The activation encoding it dispatches to must be able to encode.
    CHECK(vt::cpu::QuantTraits(t.vec_dot_type).from_float != nullptr);  // G2
    CHECK(vt::cpu::HasQuantDotKernel(c.dtype));
  }

  // from_float exists for exactly the two activation encodings.
  CHECK(vt::cpu::BlockFromFloat(vt::DType::kQ8_0) != nullptr);
  CHECK(vt::cpu::BlockFromFloat(vt::DType::kQ8_K) != nullptr);
  for (vt::DType d : {vt::DType::kQ4_0, vt::DType::kQ2_K, vt::DType::kQ3_K,
                      vt::DType::kQ4_K, vt::DType::kQ5_K, vt::DType::kQ6_K,
                      vt::DType::kIQ2_XXS, vt::DType::kIQ3_XXS,
                      vt::DType::kIQ2_S, vt::DType::kMXFP4,
                      // Upstream's row for IQ4_XS DOES carry a `from_float`
                      // (ggml-cpu.c:386, `quantize_row_iq4_xs`); porting it
                      // would be dead code here for the same reason the k-quant
                      // encoders are unported — nothing in this project ever
                      // quantizes an activation into a weight encoding.
                      vt::DType::kIQ2_XS, vt::DType::kIQ4_XS}) {
    CHECK(vt::cpu::BlockFromFloat(d) == nullptr);
  }

  // Q8_K is activation-only: upstream gives it no vec_dot row, so it must stay
  // on the generic dequant composite rather than pretending to have a kernel.
  CHECK(vt::cpu::BlockVecDot(vt::DType::kQ8_K) == nullptr);
  CHECK_FALSE(vt::cpu::HasQuantDotKernel(vt::DType::kQ8_K));
}

// ---------------------------------------------------------------------------
// G2 — activation quantization
// ---------------------------------------------------------------------------

TEST_CASE("G2 quantize_row_q8_0/q8_K round-trip within upstream RMSE bounds") {
  // Port of test-quantize-fns.cpp `total_quantization_error` (:52) on the
  // upstream synthetic signal at the upstream test size.
  std::vector<float> data(kTestSize);
  GenerateData(0.0F, data.size(), data.data());

  for (vt::DType d : {vt::DType::kQ8_0, vt::DType::kQ8_K}) {
    CAPTURE(vt::Name(d));
    std::vector<uint8_t> q(vt::RowSizeBytes(d, static_cast<int64_t>(kTestSize)));
    std::vector<float> back(kTestSize);

    vt::cpu::QuantTraits(d).from_float(data.data(), q.data(),
                                       static_cast<int64_t>(kTestSize));
    vt::cpu::BlockToFloat(d)(q.data(), back.data(),
                             static_cast<int64_t>(kTestSize));

    const float rmse = ArrayRmse(data.data(), back.data(), kTestSize);
    CAPTURE(rmse);
    CHECK(rmse < kMaxQuantizationTotalError);
  }
}

TEST_CASE("G2 from_float is the reference encoder (reference error == 0)") {
  // Upstream's `reference_quantization_error` (:62) compares the CPU-tier
  // `from_float` against the `from_float_ref` reference encoder; on the generic
  // tier they are literally the same function (quants.c:45,117 just call the
  // _ref form), so ours must agree to the BIT, not merely within 1e-4.
  std::vector<float> data(kTestSize);
  GenerateData(0.0F, data.size(), data.data());

  for (vt::DType d : {vt::DType::kQ8_0, vt::DType::kQ8_K}) {
    CAPTURE(vt::Name(d));
    const size_t bytes = vt::RowSizeBytes(d, static_cast<int64_t>(kTestSize));
    std::vector<uint8_t> q1(bytes);
    std::vector<uint8_t> q2(bytes);
    vt::cpu::QuantTraits(d).from_float(data.data(), q1.data(),
                                       static_cast<int64_t>(kTestSize));
    vt::cpu::QuantTraits(d).from_float(data.data(), q2.data(),
                                       static_cast<int64_t>(kTestSize));
    // Bit-exact and deterministic run to run (project rule: fixed reduction
    // order, no nondeterministic accumulation anywhere in the quant path).
    CHECK(std::memcmp(q1.data(), q2.data(), bytes) == 0);

    std::vector<float> b1(kTestSize);
    std::vector<float> b2(kTestSize);
    vt::cpu::BlockToFloat(d)(q1.data(), b1.data(),
                             static_cast<int64_t>(kTestSize));
    vt::cpu::BlockToFloat(d)(q2.data(), b2.data(),
                             static_cast<int64_t>(kTestSize));
    CHECK(ArrayRmse(b1.data(), b2.data(), kTestSize) <
          kMaxQuantizationReferenceError);
  }
}

namespace {

// Round half AWAY FROM ZERO — an independent statement of `roundf`'s rule
// (ggml-quants.c:260 uses roundf for Q8_0), written without calling roundf.
float RoundHalfAway(float v) {
  const float m = std::floor(std::fabs(v) + 0.5F);
  return v < 0 ? -m : m;
}

// Round half to EVEN — an independent statement of what ggml-quants.c:563's
// `nearest_int` magic-constant trick (fval + 12582912.f) actually computes,
// written via the FP rounding mode instead of the bit hack. Q8_K's quants
// depend on this exact tie rule.
int RoundHalfEven(float v) { return static_cast<int>(std::nearbyint(v)); }

}  // namespace

TEST_CASE("G2 from_float is BYTE-EXACT vs an independent reference encoder") {
  // The upstream RMSE/NMSE bounds are calibrated for 8-bit and are, measurably,
  // too loose to distinguish round-to-nearest from truncation: a mutant that
  // replaced `roundf(x0)` with a bare cast passed every statistical gate in
  // this file. The encoder's ROUNDING RULE therefore gets a byte-level gate of
  // its own, against a reference written from the upstream prose rather than
  // by calling the same library function.
  constexpr int64_t kK = 256 * 4;  // whole blocks for both encodings
  std::vector<float> data(static_cast<size_t>(kK));
  GenerateData(0.25F, data.size(), data.data());

  SUBCASE("q8_0 (ggml-quants.c:238)") {
    std::vector<uint8_t> got(vt::RowSizeBytes(vt::DType::kQ8_0, kK));
    vt::cpu::QuantTraits(vt::DType::kQ8_0)
        .from_float(data.data(), got.data(), kK);

    for (int64_t b = 0; b < kK / 32; ++b) {
      CAPTURE(b);
      const float* x = data.data() + b * 32;
      float amax = 0.0F;
      for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[j]));
      // NOTE the deliberate upstream asymmetry: the quants are derived from the
      // UNROUNDED d, while the STORED d is the f16 rounding of it.
      const float d = amax / 127.0F;
      const float id = d ? 1.0F / d : 0.0F;

      const uint8_t* blk = got.data() + b * 34;
      uint16_t d_stored = 0;
      std::memcpy(&d_stored, blk, sizeof(d_stored));
      CHECK(d_stored == vt::F32ToF16(d));
      const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
      for (int j = 0; j < 32; ++j) {
        CHECK(qs[j] == static_cast<int8_t>(RoundHalfAway(x[j] * id)));
      }
    }
  }

  SUBCASE("q8_K (ggml-quants.c:2696)") {
    std::vector<uint8_t> got(vt::RowSizeBytes(vt::DType::kQ8_K, kK));
    vt::cpu::QuantTraits(vt::DType::kQ8_K)
        .from_float(data.data(), got.data(), kK);

    for (int64_t b = 0; b < kK / 256; ++b) {
      CAPTURE(b);
      const float* x = data.data() + b * 256;
      float mx = 0;
      float amax = 0;
      for (int j = 0; j < 256; ++j) {
        const float ax = std::fabs(x[j]);
        if (ax > amax) {
          amax = ax;
          mx = x[j];
        }
      }
      REQUIRE(amax != 0.0F);
      // Keyed on the SIGNED extremum, so `iscale` is negative and the quants
      // are sign-flipped relative to the input — a detail that is easy to
      // "correct" into a silent divergence from every consumer vec_dot.
      const float iscale = -127.0F / mx;

      const uint8_t* blk = got.data() + b * 292;
      float d_stored = 0;
      std::memcpy(&d_stored, blk, sizeof(d_stored));
      CHECK(d_stored == 1.0F / iscale);
      const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 4);
      for (int j = 0; j < 256; ++j) {
        const int v = RoundHalfEven(iscale * x[j]);
        // Upstream's MIN(127, v) is reproduced, but note it is UNREACHABLE for
        // well-formed input: |iscale| is 127/amax and |x[j]| <= amax, so
        // |iscale*x[j]| <= 127 by construction. A mutant deleting the clamp is
        // therefore NOT caught by this file — correctly, since the two are
        // equivalent on every reachable input. It is kept for upstream
        // fidelity, not because a test can distinguish it.
        CHECK(qs[j] == static_cast<int8_t>(v < 127 ? v : 127));
      }
    }
  }
}

TEST_CASE("G2 Q8_K bsums equal the per-16 group sums (ggml-quants.c:2696)") {
  // The Q4_K/Q5_K vec_dots subtract the block minimum using `bsums` alone, so
  // a wrong bsum corrupts those dots while `qs` still looks perfect. Check the
  // invariant directly against the stored quants.
  constexpr int64_t kK = 256 * 5;
  std::vector<float> data(static_cast<size_t>(kK));
  GenerateData(0.5F, data.size(), data.data());

  std::vector<uint8_t> q(vt::RowSizeBytes(vt::DType::kQ8_K, kK));
  vt::cpu::QuantTraits(vt::DType::kQ8_K).from_float(data.data(), q.data(), kK);

  // block_q8_K (ggml-common.h:361-365): f32 d @0, i8 qs[256] @4, i16 bsums[16]
  // @260. Offsets restated here rather than taken from the kernel's header.
  for (int64_t b = 0; b < kK / 256; ++b) {
    const uint8_t* blk = q.data() + b * 292;
    const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 4);
    for (int g = 0; g < 16; ++g) {
      int sum = 0;
      for (int i = 0; i < 16; ++i) sum += qs[g * 16 + i];
      int16_t stored = 0;
      std::memcpy(&stored, blk + 260 + 2 * g, sizeof(stored));
      CAPTURE(b);
      CAPTURE(g);
      CHECK(stored == static_cast<int16_t>(sum));
    }
    // Upstream keys iscale on the SIGNED extremum (-127/max), so quants reach
    // -127 on the extreme element and the positive side is MIN(127, v)-clamped.
    for (int i = 0; i < 256; ++i) CHECK(qs[i] >= -127);
  }
}

TEST_CASE("G2 all-zero Q8_K block encodes to zero delta, quants and bsums") {
  // ggml-quants.c:2706-2711 — the `!amax` early-out. Upstream relies on calloc'd
  // scratch for bsums here; ours zeroes them explicitly, so assert it.
  std::vector<float> zeros(256, 0.0F);
  std::vector<uint8_t> q(vt::RowSizeBytes(vt::DType::kQ8_K, 256), 0xAB);
  vt::cpu::QuantTraits(vt::DType::kQ8_K).from_float(zeros.data(), q.data(), 256);

  float d = 1.0F;
  std::memcpy(&d, q.data(), sizeof(d));
  CHECK(d == 0.0F);
  for (int i = 0; i < 256; ++i) CHECK(q[4 + static_cast<size_t>(i)] == 0);
  for (int g = 0; g < 16; ++g) {
    int16_t stored = 1;
    std::memcpy(&stored, q.data() + 260 + 2 * g, sizeof(stored));
    CHECK(stored == 0);
  }
}

TEST_CASE("G2 activation quant rejects ragged K (fails loudly)") {
  std::vector<float> data(1024, 0.5F);
  std::vector<uint8_t> q(64 * 1024);
  // Q8_0 blocks are 32 elements, Q8_K blocks are 256: a partial trailing block
  // has no representation, so it must throw rather than write a short row.
  CHECK_THROWS(vt::cpu::QuantTraits(vt::DType::kQ8_0)
                   .from_float(data.data(), q.data(), 33));
  CHECK_THROWS(vt::cpu::QuantTraits(vt::DType::kQ8_K)
                   .from_float(data.data(), q.data(), 257));
  CHECK_THROWS(vt::cpu::QuantTraits(vt::DType::kQ8_K)
                   .from_float(data.data(), q.data(), 255));
  // ...and the exact block multiples do not.
  CHECK_NOTHROW(vt::cpu::QuantTraits(vt::DType::kQ8_0)
                    .from_float(data.data(), q.data(), 32));
  CHECK_NOTHROW(vt::cpu::QuantTraits(vt::DType::kQ8_K)
                    .from_float(data.data(), q.data(), 256));
}

TEST_CASE("G2 scratch sizing mirrors ggml_row_size / graph_plan wdata") {
  // ggml-cpu.c:1313-1349 lays src1 out one ggml_row_size(vec_dot_type, k) row
  // at a time with no padding; ggml-cpu.c:2752-2980 sizes wdata as rows*that.
  CHECK(vt::cpu::QuantActRowBytes(vt::DType::kQ4_K, 256) == 292);   // one q8_K
  CHECK(vt::cpu::QuantActRowBytes(vt::DType::kQ6_K, 512) == 584);   // two
  CHECK(vt::cpu::QuantActRowBytes(vt::DType::kQ4_0, 32) == 34);     // one q8_0
  CHECK(vt::cpu::QuantActRowBytes(vt::DType::kQ8_0, 96) == 102);    // three
  CHECK(vt::cpu::QuantActScratchBytes(vt::DType::kQ4_K, 4, 512) == 4 * 584);
  CHECK(vt::cpu::QuantActScratchBytes(vt::DType::kQ8_0, 0, 32) == 0);
  // Ragged K has no valid scratch layout.
  CHECK_THROWS(vt::cpu::QuantActRowBytes(vt::DType::kQ4_K, 255));
  CHECK_THROWS(vt::cpu::QuantActRowBytes(vt::DType::kQ4_0, 33));
}

// ---------------------------------------------------------------------------
// G3 — vec_dot vs the independent f64 reference  (THE primary correctness gate)
// ---------------------------------------------------------------------------

TEST_CASE("G3 vec_dot == independent f64 dequantize-then-dot, all six types") {
  for (const WeightCase& c : kWeightCases) {
    // Block-boundary and ragged-K coverage: a SINGLE-block row, an even
    // multiple, ODD multiples (3, 5, 7 blocks — these catch a kernel that
    // assumed pairs or an unrolled-by-2 block loop), and a large row.
    for (int64_t nblocks : {int64_t{1}, int64_t{2}, int64_t{3}, int64_t{5},
                            int64_t{7}, int64_t{16}}) {
      const int64_t k = nblocks * c.block_elems;
      CAPTURE(std::string(c.name));
      CAPTURE(nblocks);
      CAPTURE(k);

      const std::vector<uint8_t> wq =
          RandomBlocks(c, nblocks, 0xC1A0U + static_cast<uint32_t>(nblocks));

      // Activation: a real f32 signal pushed through G2's from_float, so this
      // case also gates the G2 -> G3 handoff (the round-trip that feeds the
      // dot) and not just the weight decode.
      std::vector<float> act(static_cast<size_t>(k));
      GenerateData(1.0F, act.size(), act.data());
      const std::vector<uint8_t> aq = QuantizeActivation(c.dtype, act.data(), k);

      const float got = RunVecDot(c.dtype, wq.data(), aq.data(), k);
      const vt::DType at = vt::cpu::QuantTraits(c.dtype).vec_dot_type;
      const RefDot ref = ReferenceDotF64(c.dtype, wq.data(), at, aq.data(), k);

      CAPTURE(got);
      CAPTURE(ref.dot);
      CAPTURE(ref.l1);
      CHECK(std::isfinite(got));
      // Tolerance relative to the L1 magnitude, not to |dot|: with random
      // signs the dot can sit near zero, and a |dot|-relative bound would then
      // be unsatisfiable while an absolute bound would be vacuous. f32
      // accumulation over <=16 blocks costs ~1e-6 relative; 1e-5 is a tight
      // ceiling that still catches any real decode error (those are O(1)).
      CHECK(std::fabs(static_cast<double>(got) - ref.dot) <= 1e-5 * ref.l1);
    }
  }
}

// ---------------------------------------------------------------------------
// G5 — the x86 AVX2 K-quant tier  (selection + numeric A/B)
//
// The G3 case above runs through BlockVecDot, which selects the wide kernel on
// this host. This case proves three separate things that a single selected
// call cannot: `auto` chooses the AVX2 arm, `VT_CPU_QUANT_AVX2=portable`
// really returns the portable kernel (the defeat seam), and both arms stay
// inside the same f64-reference tolerance. Ported from the `__AVX2__` arms of
// llama.cpp b10451 arch/x86/quants.c:2018, :2406 and :3364.
// ---------------------------------------------------------------------------
TEST_CASE("G5 AVX2 K-quant tier is selectable and matches the portable tier") {
#if defined(__x86_64__) || defined(_M_X64)
  if (!vt::cpu::DetectX86IsaCaps().avx2) {
    MESSAGE("host has no AVX2; only the portable tier exists here");
    return;
  }

  const char* previous = std::getenv("VT_CPU_QUANT_AVX2");
  const bool had_previous = previous != nullptr;
  const std::string saved = had_previous ? previous : "";
  struct Restore {
    bool had;
    std::string value;
    ~Restore() {
      if (had) {
        VT_TEST_SETENV("VT_CPU_QUANT_AVX2", value.c_str());
      } else {
        VT_TEST_UNSETENV("VT_CPU_QUANT_AVX2");
      }
    }
  } restore{had_previous, saved};

  // A G5-local list keeps IQ3_S out of `kWeightCases`. The MatmulBTQuant NMSE
  // case there runs at m=1,n=1, where IQ3_S's dot can cancel near zero and its
  // unmutated NMSE is already ~1.10, so adding the row would fail that case.
  // The q4_K and q6_K entries copy `kWeightCases` unchanged. IQ3_S's block is
  // `d@0 qs@2 qh@66 signs@74 scales@106` (110 bytes).
  const WeightCase kG5WeightCases[] = {
      {vt::DType::kQ4_K, 256, 144, 0, 2, -1, "q4_K"},
      {vt::DType::kQ6_K, 256, 210, 208, -1, -1, "q6_K"},
      {vt::DType::kIQ3_S, 256, 110, 0, -1, -1, "iq3_s"},
  };
  for (const WeightCase& c : kG5WeightCases) {
    CAPTURE(std::string(c.name));

    VT_TEST_UNSETENV("VT_CPU_QUANT_AVX2");
    const vt::cpu::VecDotFn wide =
        vt::cpu::SelectQuantKVecDot(c.dtype, nullptr);
    REQUIRE(wide != nullptr);
    const vt::cpu::VecDotFn portable = vt::cpu::QuantKPortableVecDot(c.dtype);
    REQUIRE(portable != nullptr);
    CHECK(reinterpret_cast<uintptr_t>(wide) !=
          reinterpret_cast<uintptr_t>(portable));

    // The PRODUCTION dispatch must route to the wide kernel. Deleting the
    // SelectKQuantVecDot arm in BlockVecDot has to turn this case red; a test
    // that only called the selector would stay green with the tier unreached.
    CHECK(reinterpret_cast<uintptr_t>(vt::cpu::BlockVecDot(c.dtype)) ==
          reinterpret_cast<uintptr_t>(wide));

    VT_TEST_SETENV("VT_CPU_QUANT_AVX2", "portable");
    CHECK(vt::cpu::SelectQuantKVecDot(c.dtype, nullptr) == nullptr);
    CHECK(reinterpret_cast<uintptr_t>(vt::cpu::BlockVecDot(c.dtype)) ==
          reinterpret_cast<uintptr_t>(portable));

    VT_TEST_SETENV("VT_CPU_QUANT_AVX2", "avx2");
    CHECK(vt::cpu::SelectQuantKVecDot(c.dtype, nullptr) == wide);

    VT_TEST_UNSETENV("VT_CPU_QUANT_AVX2");
    for (int64_t nblocks : {int64_t{1}, int64_t{3}, int64_t{5}}) {
      const int64_t k = nblocks * c.block_elems;
      CAPTURE(nblocks);
      const std::vector<uint8_t> wq = RandomBlocks(
          c, nblocks, 0x5A00U + static_cast<uint32_t>(nblocks));
      std::vector<float> act(static_cast<size_t>(k));
      GenerateData(1.0F, act.size(), act.data());
      const std::vector<uint8_t> aq =
          QuantizeActivation(c.dtype, act.data(), k);
      const vt::DType at = vt::cpu::QuantTraits(c.dtype).vec_dot_type;
      const RefDot ref = ReferenceDotF64(c.dtype, wq.data(), at, aq.data(), k);

      float got_wide = 0.0F;
      float got_portable = 0.0F;
      wide(static_cast<int>(k), &got_wide, 0, wq.data(), 0, aq.data(), 0, 1);
      portable(static_cast<int>(k), &got_portable, 0, wq.data(), 0, aq.data(),
               0, 1);

      CAPTURE(got_wide);
      CAPTURE(got_portable);
      CAPTURE(ref.dot);
      CAPTURE(ref.l1);
      CHECK(std::isfinite(got_wide));
      CHECK(std::fabs(static_cast<double>(got_wide) - ref.dot) <=
            1e-5 * ref.l1);
      CHECK(std::fabs(static_cast<double>(got_portable) - ref.dot) <=
            1e-5 * ref.l1);
      CHECK(std::fabs(static_cast<double>(got_wide) - got_portable) <=
            1e-5 * ref.l1);
    }
  }

  VT_TEST_SETENV("VT_CPU_QUANT_AVX2", "bogus");
  CHECK_THROWS(vt::cpu::SelectQuantKVecDot(vt::DType::kQ4_K, nullptr));
#else
  MESSAGE("not x86-64; the AVX2 tier does not exist in this build");
#endif
}

TEST_CASE("kIq1sGrid is the PINNED upstream table, not a look-alike") {
  // The dot-vs-dequantize check above cannot catch a wrong IQ1_S codebook:
  // VecDotIQ1_SQ8_K and DequantIQ1_S read the SAME kIq1sGrid, so a corrupted
  // table moves both sides together and they still agree. Consistency is not
  // correctness (the same trap the shared-helper gates hit), so the table is
  // pinned to its provenance instead: these are the bytes of
  // llama.cpp @ 237ad9b96 `ggml/src/ggml-common.h:1124 iq1s_grid`, which is
  // where it was extracted from mechanically rather than transcribed.
  //
  // Re-derive upstream-side with:
  //   git show 237ad9b96:ggml/src/ggml-common.h |
  //     python3 -c '...FNV-1a 64 over each entry little-endian...'
  CHECK(std::size(vt::cpu::kIq1sGrid) == 2048);  // NGRID_IQ1S

  uint64_t h = 0xcbf29ce484222325ULL;  // FNV-1a 64 offset basis
  for (uint64_t v : vt::cpu::kIq1sGrid) {
    for (int b = 0; b < 8; ++b) {  // little-endian, as ggml stores it
      h ^= static_cast<uint8_t>(v >> (8 * b));
      h *= 0x100000001b3ULL;
    }
  }
  CHECK(h == 0x6703ed863501ae2eULL);

  // Every entry packs 8 TERNARY lanes, each exactly -1, 0 or +1: the codebook
  // carries the sign itself, which is why IQ1_S needs no sign array (unlike
  // IQ2_XXS's ksigns or IQ2_S's direct sign bytes). This is an independent
  // structural check the digest alone would not explain, and it is deliberately
  // stated as ternary: an earlier draft of this file asserted +/-1 and the
  // table disproved it (the 16384 lanes are 6649 zeros, 4860 +1, 4875 -1).
  int lanes[3] = {0, 0, 0};
  for (uint64_t v : vt::cpu::kIq1sGrid) {
    for (int b = 0; b < 8; ++b) {
      const int8_t lane = static_cast<int8_t>(v >> (8 * b));
      REQUIRE((lane == -1 || lane == 0 || lane == 1));
      ++lanes[lane + 1];
    }
  }
  // Pin the census too, so a table swapped for another ternary codebook of the
  // same size still fails here rather than only in the digest.
  CHECK(lanes[0] == 4875);  // -1
  CHECK(lanes[1] == 6649);  //  0
  CHECK(lanes[2] == 4860);  // +1
}

TEST_CASE("kIq1xxxsGrid is the PINNED FORK table, not a look-alike") {
  // Same argument as the IQ1_S seal above, and it binds harder here. This table
  // does not come from an upstream release but from a BRANCH,
  // unslothai/llama.cpp @ iq1-narrow (36fe8e1cc), pinned in
  // .agents/oracles/llama-cpp-unsloth.md. A branch can be rebased or amended
  // under its own name, so a digest over the bytes we actually ported is the
  // only thing that makes "the pin" mean something a year from now.
  CHECK(std::size(vt::cpu::kIq1xxxsGrid) == 256);  // NGRID_IQ1XXXS

  uint64_t h = 0xcbf29ce484222325ULL;
  for (uint64_t v : vt::cpu::kIq1xxxsGrid) {
    for (int b = 0; b < 8; ++b) {
      h ^= static_cast<uint8_t>(v >> (8 * b));
      h *= 0x100000001b3ULL;
    }
  }
  CHECK(h == 0x24421301ff77509cULL);

  int lanes[3] = {0, 0, 0};
  for (uint64_t v : vt::cpu::kIq1xxxsGrid) {
    for (int b = 0; b < 8; ++b) {
      const int8_t lane = static_cast<int8_t>(v >> (8 * b));
      REQUIRE((lane == -1 || lane == 0 || lane == 1));
      ++lanes[lane + 1];
    }
  }
  // Far sparser than IQ1_S, which is what the smaller bit budget buys: 1243 of
  // the 2048 lanes are zero here, against 6649 of 16384 for IQ1_S.
  CHECK(lanes[0] == 408);   // -1
  CHECK(lanes[1] == 1243);  //  0
  CHECK(lanes[2] == 397);   // +1
}

TEST_CASE("kIq3sGrid is the pinned 512-entry table, not kIq3xxsGrid") {
  // Two u32 codebooks, both read four bytes at a time, one 256 rows and one
  // 512. A decoder pointed at the sibling indexes in range for the first 256
  // rows and returns plausible magnitudes, so nothing but a digest over the
  // bytes we actually ported separates them. IQ3_S carried no `vec_dot` until
  // the TT GSQ-RCO keep-quant wave 1 added the CPU oracle `VecDotIQ3_SQ8_K`
  // (cpu_quant_dot.cpp, quants.c:1094) beside this table seal; the DOT
  // evidence is the device sweep in tests/vt/test_tenstorrent_backend.cpp,
  // which runs the on-core decode against that oracle bit-exactly, and the
  // decode golden in tests/vt/iq3s_golden_vectors.h.
  CHECK(std::size(vt::cpu::kIq3sGrid) == 512);
  CHECK(std::size(vt::cpu::kIq3xxsGrid) == 256);

  uint64_t h = 0xcbf29ce484222325ULL;
  for (uint32_t v : vt::cpu::kIq3sGrid) {
    for (int b = 0; b < 4; ++b) {
      h ^= static_cast<uint8_t>(v >> (8 * b));
      h *= 0x100000001b3ULL;
    }
  }
  CHECK(h == 0xfa37020c25b44829ULL);

  // The grid alphabet is a second, independent handle. Every lane byte of
  // iq3s_grid is an ODD value in 1..15 -- a different alphabet from
  // iq3xxs_grid's {0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e}, so a
  // look-alike table of the same size still fails this.
  int counts[16] = {0};
  for (uint32_t v : vt::cpu::kIq3sGrid) {
    for (int b = 0; b < 4; ++b) {
      const uint8_t lane = static_cast<uint8_t>(v >> (8 * b));
      REQUIRE(lane < 16);
      REQUIRE((lane % 2) == 1);
      ++counts[lane];
    }
  }
  CHECK(counts[1] == 436);
  CHECK(counts[3] == 344);
  CHECK(counts[5] == 327);
  CHECK(counts[7] == 271);
  CHECK(counts[9] == 223);
  CHECK(counts[11] == 185);
  CHECK(counts[13] == 112);
  CHECK(counts[15] == 150);
  // The first two entries pin the table's ORIGIN as well as its content:
  // ggml-common.h:1053 begins 0x01010101, 0x01010103.
  CHECK(vt::cpu::kIq3sGrid[0] == 0x01010101u);
  CHECK(vt::cpu::kIq3sGrid[1] == 0x01010103u);
  CHECK(vt::cpu::kIq3sGrid[511] == 0x0f0f0101u);
}

TEST_CASE("kIq2xsGrid is the pinned 512-entry table, not a sibling grid") {
  // The IQ2 family carries THREE codebooks with the identical 8-byte-per-entry
  // shape — iq2xxs_grid (256), iq2xs_grid (512) and iq2s_grid (1024) — and this
  // tree now holds all three. A decoder that reaches for the wrong one still
  // runs, still indexes in range for the first 256 entries, and still produces
  // plausible magnitudes, so nothing but a digest over the bytes we actually
  // ported separates them.
  CHECK(std::size(vt::cpu::kIq2xsGrid) == 512);

  uint64_t h = 0xcbf29ce484222325ULL;
  for (uint64_t v : vt::cpu::kIq2xsGrid) {
    for (int b = 0; b < 8; ++b) {
      h ^= static_cast<uint8_t>(v >> (8 * b));
      h *= 0x100000001b3ULL;
    }
  }
  CHECK(h == 0xc9b1ee61e79909bdULL);

  // The grid alphabet is a second, independent handle: every lane byte is one
  // of three magnitudes, and the counts are a shape a look-alike table of the
  // same size still fails. (They differ from iq2s_grid's, which is 1024 rows.)
  int n8 = 0, n25 = 0, n43 = 0;
  for (uint64_t v : vt::cpu::kIq2xsGrid) {
    for (int b = 0; b < 8; ++b) {
      const uint8_t lane = static_cast<uint8_t>(v >> (8 * b));
      REQUIRE((lane == 8 || lane == 25 || lane == 43));
      if (lane == 8) ++n8;
      else if (lane == 25) ++n25;
      else ++n43;
    }
  }
  CHECK(n8 == 2114);
  CHECK(n25 == 1142);
  CHECK(n43 == 840);
  // The first two entries pin the table's ORIGIN as well as its content:
  // ggml-common.h:628 begins 0x0808080808080808, 0x080808080808082b.
  CHECK(vt::cpu::kIq2xsGrid[0] == 0x0808080808080808ULL);
  CHECK(vt::cpu::kIq2xsGrid[1] == 0x080808080808082bULL);
  CHECK(vt::cpu::kIq2xsGrid[511] == 0x2b2b2b2b2b2b2b2bULL);
}

// ---------------------------------------------------------------------------
// QUANT-GGUF-IQ-VECDOT — the two keep-quant `vec_dot` rows the staged
// GLM-5.3-Flash UD-Q2_K_XL artifact needs, gated against the ORACLE'S OWN
// KERNELS on that artifact's OWN BYTES.
//
// Why these cases exist beside the battery above. Every other check on a
// `vec_dot` in this file is a consistency check: it compares the kernel against
// `BlockToFloat` (a second port of the same layout) or against `MatmulBTQuant`
// (the same kernel, driven differently). Those cannot see a decode parameter
// that BOTH ports read the same wrong way, and they cannot see a reduction-order
// difference at all, because the f64 reference deliberately has none. Only the
// pinned oracle's own f32 accumulation can, so it is compared BIT for BIT.
// ---------------------------------------------------------------------------

namespace {

// The activation signal the oracle harness quantized with its own
// `quantize_row_q8_K_generic`. Every value is an integer in [-1024, 1023]
// divided by 64, so it is exact in binary32 and the sequence is identical on
// any compiler — which is what lets the goldens carry the RESULTING Q8_K bytes
// instead of 1024 floats.
void MakeDotActivation(int n, uint32_t seed, float* x) {
  uint32_t s = seed;
  for (int i = 0; i < n; ++i) {
    s = s * 1664525U + 1013904223U;
    const int32_t v = static_cast<int32_t>((s >> 16) & 0x7ffU) - 1024;
    x[i] = static_cast<float>(v) / 64.0F;
  }
}

uint32_t FloatBits(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

// One oracle-gated case, shared by both types: the shapes differ only in the
// block stride and in which golden arrays are read.
void CheckOracleDot(vt::DType dtype, const uint8_t* weights, size_t wbytes,
                    const uint8_t* act, uint32_t seed, uint32_t expected_total,
                    const uint32_t (&expected_per_block)[4]) {
  constexpr int kN = 1024;   // 4 super-blocks
  constexpr int kBlocks = 4;
  constexpr size_t kQ8KBytes = 292;

  // The pairing itself is a claim: both rows dot against Q8_K (ggml-cpu.c:342
  // and :385), NOT against the Q8_0 that IQ4_NL's 32-element block forces.
  REQUIRE(vt::cpu::QuantTraits(dtype).vec_dot_type == vt::DType::kQ8_K);
  REQUIRE(vt::cpu::BlockVecDot(dtype) != nullptr);
  REQUIRE(vt::cpu::HasQuantDotKernel(dtype));

  // The activation bytes the oracle dotted are reproduced by OUR encoder, so
  // the golden below is a statement about the dot alone and not about a
  // divergent activation.
  std::vector<float> x(kN);
  MakeDotActivation(kN, seed, x.data());
  std::vector<uint8_t> ours(kBlocks * kQ8KBytes);
  vt::cpu::QuantTraits(vt::DType::kQ8_K)
      .from_float(x.data(), ours.data(), kN);
  CHECK(std::memcmp(ours.data(), act, ours.size()) == 0);

  const size_t block_bytes = wbytes / kBlocks;
  float s = 0.0F;
  vt::cpu::QuantTraits(dtype).vec_dot(kN, &s, 0, weights, 0, act, 0, 1);
  CAPTURE(s);
  CHECK(FloatBits(s) == expected_total);

  for (int b = 0; b < kBlocks; ++b) {
    CAPTURE(b);
    float sb = 0.0F;
    vt::cpu::QuantTraits(dtype).vec_dot(256, &sb, 0,
                                        weights + b * block_bytes, 0,
                                        act + b * kQ8KBytes, 0, 1);
    CAPTURE(sb);
    CHECK(FloatBits(sb) == expected_per_block[b]);
  }

  // A second, structurally independent statement of the same product: decode
  // both operands through `BlockToFloat` (the loader-side decoders, gated
  // byte-for-byte against the oracle by iq2xs_iq4xs_golden_vectors.h) and dot
  // them in f64. This one cannot see reduction order — that is the golden's
  // job — but it does catch a kernel that consumes the wrong field entirely.
  std::vector<float> w(kN);
  std::vector<float> a(kN);
  vt::cpu::BlockToFloat(dtype)(weights, w.data(), kN);
  vt::cpu::BlockToFloat(vt::DType::kQ8_K)(act, a.data(), kN);
  double ref = 0.0;
  double l1 = 0.0;
  for (int i = 0; i < kN; ++i) {
    ref += static_cast<double>(w[i]) * static_cast<double>(a[i]);
    l1 += std::fabs(static_cast<double>(w[i]) * static_cast<double>(a[i]));
  }
  CAPTURE(ref);
  CAPTURE(l1);
  CHECK(std::fabs(static_cast<double>(s) - ref) < 1e-5 * l1);
}

}  // namespace

TEST_CASE("IQ2_XS vec_dot is the oracle's own kernel, bit for bit") {
  // Weights: `blk.3.ffn_gate_exps.weight` of the staged GLM-5.3-Flash
  // UD-Q2_K_XL artifact — the exact tensor `LoadedEngine::FromModelDir` used to
  // stop on. Expected value: llama.cpp b10451
  // ggml/src/ggml-cpu/quants.c:948 `ggml_vec_dot_iq2_xs_q8_K_generic`.
  CheckOracleDot(vt::DType::kIQ2_XS, vllm_test::kIq2xsGoldenBlocks,
                 std::size(vllm_test::kIq2xsGoldenBlocks),
                 vllm_test::kIq2xsDotActQ8K, 0x2247U,
                 vllm_test::kIq2xsDotExpectedBits,
                 vllm_test::kIq2xsDotPerBlockBits);
}

TEST_CASE("IQ4_XS vec_dot is the oracle's own kernel, bit for bit") {
  // Weights: `blk.11.ffn_down_exps.weight` of the same artifact. Expected
  // value: quants.c:1283 `ggml_vec_dot_iq4_xs_q8_K_generic`.
  CheckOracleDot(vt::DType::kIQ4_XS, vllm_test::kIq4xsGoldenBlocks,
                 std::size(vllm_test::kIq4xsGoldenBlocks),
                 vllm_test::kIq4xsDotActQ8K, 0x4247U,
                 vllm_test::kIq4xsDotExpectedBits,
                 vllm_test::kIq4xsDotPerBlockBits);
}

TEST_CASE("the IQ2_XS dot consumes the SEALED 512-entry grid") {
  // The seal above proves `kIq2xsGrid` holds the pinned bytes. It says nothing
  // about which table the KERNEL reads, and iq2xxs_grid / iq2s_grid have the
  // identical 8-byte-per-entry shape, so a kernel pointed at a sibling still
  // indexes in range and still returns a plausible magnitude. What ties the two
  // together is that the oracle golden above moves when the kernel's grid is
  // swapped: this case states the coupling, and the review mutation that swaps
  // `kIq2xsGrid` for `kIq2xxsGrid` in `VecDotIQ2_XSQ8_K` is what proves it.
  //
  // The three tables are DIFFERENT over their shared first 256 rows, which is
  // the only region a swapped index could reach; if they agreed there, no dot
  // over these blocks could tell them apart.
  int differ = 0;
  for (int i = 0; i < 256; ++i) {
    if (vt::cpu::kIq2xxsGrid[i] != vt::cpu::kIq2xsGrid[i]) ++differ;
    if (vt::cpu::kIq2sGrid[i] != vt::cpu::kIq2xsGrid[i]) ++differ;
  }
  CHECK(differ > 400);

  // And the blocks actually dotted use indices ABOVE 255 as well, so a
  // 256-entry sibling cannot even be addressed without wrapping.
  int above_255 = 0;
  for (size_t b = 0; b < std::size(vllm_test::kIq2xsGoldenBlocks); b += 74) {
    for (int l = 0; l < 32; ++l) {
      uint16_t q = 0;
      std::memcpy(&q, vllm_test::kIq2xsGoldenBlocks + b + 2 + 2 * l, 2);
      if ((q & 511) > 255) ++above_255;
    }
  }
  CHECK(above_255 > 32);
}

TEST_CASE("kIq1sDelta is upstream IQ1S_DELTA, not a value this tree chose") {
  // `ggml/src/ggml-common.h:1121` at the pinned 237ad9b96 is
  // `#define IQ1S_DELTA 0.125f`. The FORK reuses that same macro for IQ1_XXXS
  // (`dequantize_row_iq1_xxxs` and `ggml_vec_dot_iq1_xxxs_q8_K` both spell
  // IQ1S_DELTA), and upstream's IQ1_M reuses it too as IQ1M_DELTA
  // (ggml-common.h:1133 at b10451, the same 0.125f), so one constant serves
  // all THREE encodings here and one wrong value corrupts all of them at once.
  //
  // Sealed by value for exactly the reason the two grids are sealed by digest:
  // `DequantIQ1_S`, `DequantIQ1_XXXS`, `VecDotIQ1_SQ8_K` and
  // `VecDotIQ1_XXXSQ8_K` all read THIS definition, so every one of them agrees
  // with every other whatever it holds. Consistency is not correctness. The
  // grid seal stopped one table short of this constant, and a doubled delta
  // survived the whole suite until the golden vectors below were added.
  CHECK(vt::cpu::kIq1sDelta == 0.125F);
}

// Compare a decode against the oracle-produced goldens, BIT for BIT. Both sides
// evaluate the same f32 expression `dl * (grid[j] + delta)`, so equality is
// exact and any difference means a decode PARAMETER diverged, not that rounding
// moved. Asserting per element (rather than on a digest or a mismatch count)
// makes the assertion total say how many values were actually examined.
void CheckAgainstOracle(vt::DType dtype, const uint8_t* blocks,
                        const uint32_t* golden_bits, size_t n) {
  std::vector<float> got(n);
  vt::cpu::BlockToFloat(dtype)(blocks, got.data(), static_cast<int64_t>(n));
  for (size_t i = 0; i < n; ++i) {
    CAPTURE(i);
    uint32_t bits = 0;
    std::memcpy(&bits, &got[i], sizeof(bits));
    CHECK(bits == golden_bits[i]);
  }
}

TEST_CASE("IQ1_S decodes REAL checkpoint bytes as the PINNED ORACLE does") {
  // The first independent reference this encoding has. Everything else in this
  // file decodes the weight with `vt::cpu::BlockToFloat`, the function under
  // test, so it is independent only in the summation: a wrong scale field, a
  // wrong delta magnitude or a flipped delta sign moves the reference and the
  // kernel together and they still agree.
  //
  // Here the expected values come from ggml-org/llama.cpp @ 237ad9b96 running
  // its OWN `ggml_get_type_traits(GGML_TYPE_IQ1_S)->to_float`, and the inputs
  // are real `blk.0.ffn_gate_exps.weight` bytes from the UD-IQ1_S checkpoint.
  // Provenance and the reproduction recipe are in iq1_golden_vectors.h.
  //
  // This seals the DEQUANT arm. The vec_dot arm is tied to it by the G3
  // "vec_dot matches f64 dequantize-then-dot" case above, so a defect injected
  // into either one alone now fails somewhere, and a defect injected into both
  // fails here.
  CHECK(std::size(vllm_test::kIq1sGoldenBlocks) == 4 * 50);  // 4 blocks
  CHECK(std::size(vllm_test::kIq1sGoldenBits) == 4 * 256);
  CheckAgainstOracle(vt::DType::kIQ1_S, vllm_test::kIq1sGoldenBlocks,
                     vllm_test::kIq1sGoldenBits,
                     std::size(vllm_test::kIq1sGoldenBits));
}

TEST_CASE("IQ1_XXXS decodes REAL checkpoint bytes as the PINNED FORK does") {
  // Same argument as the IQ1_S case above. The oracle is the fork,
  // unslothai/llama.cpp @ 36fe8e1cc, because no upstream llama.cpp defines ggml
  // type 66 at all; the inputs are real UD-Q1_0 bytes.
  //
  // What this does and does not establish is the same as the spec's
  // 1179648-weight run, of which these four blocks are the committed slice: it
  // removes transcription error from OUR C++ and proves the block layout is
  // read correctly. It does NOT make the fork gateable, because the fork's own
  // decode is still unvalidated against anything but itself (#933).
  CHECK(std::size(vllm_test::kIq1xxxsGoldenBlocks) == 4 * 38);  // 4 blocks
  CHECK(std::size(vllm_test::kIq1xxxsGoldenBits) == 4 * 256);
  CheckAgainstOracle(vt::DType::kIQ1_XXXS, vllm_test::kIq1xxxsGoldenBlocks,
                     vllm_test::kIq1xxxsGoldenBits,
                     std::size(vllm_test::kIq1xxxsGoldenBits));
}

TEST_CASE("IQ1_M decodes synthetic blocks as the PINNED ORACLE does") {
  // The oracle is ggml-org/llama.cpp @ b10451, the pinned upstream build. The
  // inputs are synthetic (the GSQ-RCO artifact was still downloading when the
  // fixture was written; real checkpoint bytes are owed), but they exercise
  // every field the decode reads: both delta polarities, all eight 3-bit
  // sub-block scales per block, and arbitrary 11-bit grid indices. Provenance
  // and the reproduction recipe are in iq1m_golden_vectors.h.
  //
  // This seals the DEQUANT arm. The vec_dot arm is tied to it by the G3
  // "vec_dot matches f64 dequantize-then-dot" case above, so a defect injected
  // into either one alone now fails somewhere.
  CHECK(std::size(vllm_test::kIq1mGoldenBlocks) == 4 * 56);  // 4 blocks
  CHECK(std::size(vllm_test::kIq1mGoldenBits) == 4 * 256);
  CheckAgainstOracle(vt::DType::kIQ1_M, vllm_test::kIq1mGoldenBlocks,
                     vllm_test::kIq1mGoldenBits,
                     std::size(vllm_test::kIq1mGoldenBits));
}

TEST_CASE("G3 vec_dot is bit-exact run to run (fixed reduction order)") {
  for (const WeightCase& c : kWeightCases) {
    CAPTURE(std::string(c.name));
    const int64_t k = 5 * c.block_elems;
    const std::vector<uint8_t> wq = RandomBlocks(c, 5, 4242U);
    std::vector<float> act(static_cast<size_t>(k));
    GenerateData(2.0F, act.size(), act.data());
    const std::vector<uint8_t> aq = QuantizeActivation(c.dtype, act.data(), k);

    const float first = RunVecDot(c.dtype, wq.data(), aq.data(), k);
    for (int rep = 0; rep < 4; ++rep) {
      // Bit equality, not Approx: the kernels fix the reduction order, so any
      // drift would mean nondeterministic accumulation crept in.
      CHECK(RunVecDot(c.dtype, wq.data(), aq.data(), k) == first);
    }
  }
}

TEST_CASE("G3 vec_dot rejects ragged n and the unported nrc==2 mmla mode") {
  for (const WeightCase& c : kWeightCases) {
    CAPTURE(std::string(c.name));
    const int64_t k = 2 * c.block_elems;
    const std::vector<uint8_t> wq = RandomBlocks(c, 2, 7U);
    std::vector<float> act(static_cast<size_t>(k), 0.25F);
    const std::vector<uint8_t> aq = QuantizeActivation(c.dtype, act.data(), k);
    const vt::cpu::VecDotFn f = vt::cpu::QuantTraits(c.dtype).vec_dot;
    float s = 0.0F;

    // n must be a whole number of weight blocks.
    CHECK_THROWS(f(static_cast<int>(c.block_elems) + 1, &s, 0, wq.data(), 0,
                   aq.data(), 0, 1));
    // nrc==2 is the i8mm mmla contract (G6). Accepting it here would silently
    // read a second row that the generic kernel never dots.
    CHECK_THROWS(f(static_cast<int>(k), &s, 0, wq.data(), 0, aq.data(), 0, 2));
    CHECK_NOTHROW(f(static_cast<int>(k), &s, 0, wq.data(), 0, aq.data(), 0, 1));
  }
}

TEST_CASE("G3 dot_product_error within upstream bound (test-quantize-fns:86)") {
  // The literal upstream case, reproducible for Q8_0 because it is the one
  // type where we have BOTH the encoder (G2) and the vec_dot (G3): quantize
  // both operands, dot them, and compare against the full-precision dot of the
  // original data, normalized by test_size exactly as upstream does.
  std::vector<float> d1(kTestSize);
  std::vector<float> d2(kTestSize);
  GenerateData(0.0F, d1.size(), d1.data());
  GenerateData(1.0F, d2.size(), d2.data());

  const int64_t k = static_cast<int64_t>(kTestSize);
  std::vector<uint8_t> q1(vt::RowSizeBytes(vt::DType::kQ8_0, k));
  vt::cpu::QuantTraits(vt::DType::kQ8_0).from_float(d1.data(), q1.data(), k);
  const std::vector<uint8_t> q2 =
      QuantizeActivation(vt::DType::kQ8_0, d2.data(), k);

  const float got = RunVecDot(vt::DType::kQ8_0, q1.data(), q2.data(), k);
  double dot_ref = 0;
  for (size_t i = 0; i < kTestSize; i++) dot_ref += d1[i] * d2[i];

  const float err =
      std::fabs(got - static_cast<float>(dot_ref)) / static_cast<float>(kTestSize);
  CAPTURE(got);
  CAPTURE(dot_ref);
  CAPTURE(err);
  CHECK(err < kMaxDotProductError);
}

TEST_CASE("KERNEL-CPU-A76-Q8-DOT explicit SDOT and assembly match portable") {
  const vt::cpu::VecDotFn sdot = vt::cpu::QuantQ8SdotVecDot();
  const vt::cpu::VecDotFn assembly = vt::cpu::QuantQ8A76AsmVecDot();
  CHECK((sdot == nullptr) == (assembly == nullptr));
  if (sdot == nullptr) {
    CHECK_FALSE(vt::cpu::QuantQ8SdotActive());
    CHECK_FALSE(vt::cpu::QuantQ8A76AsmActive());
    return;
  }
  CHECK(vt::cpu::QuantQ8SdotActive());

  // The TRUE portable reference. QuantTraits(kQ8_0).vec_dot is the SELECTED
  // kernel — on a real A76 that is the assembly tier, and referencing it here
  // made every byte-equality CHECK below a self-comparison.
  const vt::cpu::VecDotFn portable = vt::cpu::QuantQ8PortableVecDot();
  for (int blocks : {1, 2, 3, 5, 64}) {
    CAPTURE(blocks);
    const int64_t k = 32 * blocks;
    std::vector<uint8_t> wq =
        RandomBlocks(kWeightCases[1], blocks, 0xA760U + blocks);
    std::vector<float> act(static_cast<size_t>(k));
    GenerateData(0.75F, act.size(), act.data());
    std::vector<uint8_t> aq =
        QuantizeActivation(vt::DType::kQ8_0, act.data(), k);

    // Offset both buffers by one byte: Q8 blocks are 34 bytes and therefore
    // alternate natural alignment in real rows. All three tiers must accept
    // an unaligned block base without reading beyond the final block.
    std::vector<uint8_t> wu(wq.size() + 2, 0xA5);
    std::vector<uint8_t> au(aq.size() + 2, 0x5A);
    std::memcpy(wu.data() + 1, wq.data(), wq.size());
    std::memcpy(au.data() + 1, aq.data(), aq.size());
    const float ref = RunVecDotFn(portable, wu.data() + 1, au.data() + 1, k);
    CHECK(RunVecDotFn(sdot, wu.data() + 1, au.data() + 1, k) == ref);
    CHECK(RunVecDotFn(assembly, wu.data() + 1, au.data() + 1, k) == ref);
  }

  auto check_edge_blocks = [&](uint16_t wd, uint16_t ad, bool zero_payload) {
    constexpr int blocks = 2;
    constexpr int block_bytes = 34;
    std::vector<uint8_t> wq(blocks * block_bytes);
    std::vector<uint8_t> aq(blocks * block_bytes);
    for (int ib = 0; ib < blocks; ++ib) {
      uint8_t* wb = wq.data() + ib * block_bytes;
      uint8_t* ab = aq.data() + ib * block_bytes;
      std::memcpy(wb, &wd, sizeof(wd));
      std::memcpy(ab, &ad, sizeof(ad));
      for (int j = 0; j < 32; ++j) {
        wb[2 + j] = zero_payload
                        ? 0
                        : static_cast<uint8_t>((j & 1) != 0 ? 127 : -128);
        ab[2 + j] = zero_payload
                        ? 0
                        : static_cast<uint8_t>((j & 2) != 0 ? -128 : 127);
      }
    }
    const float ref = RunVecDotFn(portable, wq.data(), aq.data(), 64);
    CHECK(RunVecDotFn(sdot, wq.data(), aq.data(), 64) == ref);
    CHECK(RunVecDotFn(assembly, wq.data(), aq.data(), 64) == ref);
  };
  check_edge_blocks(vt::F32ToF16(1.0F), vt::F32ToF16(1.0F), true);
  check_edge_blocks(/*maximum finite f16=*/0x7BFFU,
                    /*minimum normal negative f16=*/0x8400U, false);

  std::vector<uint8_t> one(34, 0);
  float out = 0.0F;
  CHECK_THROWS(sdot(33, &out, 0, one.data(), 0, one.data(), 0, 1));
  CHECK_THROWS(assembly(32, &out, 0, one.data(), 0, one.data(), 0, 2));
}

TEST_CASE(
    "KERNEL-CPU-A76-Q8-DOT portable seam pins the quants.c:400 order on every "
    "platform") {
  // The A76 case above can only execute its byte-equality CHECKs on a DotProd
  // core (the sdot/assembly getters are null elsewhere). This case pins the
  // reference arm ITSELF everywhere: QuantQ8PortableVecDot must be the exact
  // per-block accumulation order of the portable kernel — an independent
  // scalar transcription here, compared byte-equal — so a perturbation of the
  // portable order (or a seam regression back to the SELECTED kernel wired to
  // a different tier) is RED on x86 too, not only on a physical A76.
  const vt::cpu::VecDotFn portable = vt::cpu::QuantQ8PortableVecDot();
  REQUIRE(portable != nullptr);
  for (int blocks : {1, 3, 64}) {
    CAPTURE(blocks);
    const int64_t k = 32 * blocks;
    const std::vector<uint8_t> wq =
        RandomBlocks(kWeightCases[1], blocks, 0x9700U + blocks);
    std::vector<float> act(static_cast<size_t>(k));
    GenerateData(0.25F, act.size(), act.data());
    const std::vector<uint8_t> aq =
        QuantizeActivation(vt::DType::kQ8_0, act.data(), k);

    // Two references sharing the SAME per-block order, differing only in
    // whether the final multiply-add is fused: the portable TU may legally
    // compile `sumf += sumi * dx * dy` either way (-ffp-contract), and pinning
    // one form would break on the other compiler regime. An accumulation-ORDER
    // mutation moves BOTH candidates, so the pin holds in both regimes.
    constexpr int kBlockBytes = 34;  // f16 scale + 32 int8 payload
    float ref_mul = 0.0F;
    float ref_fma = 0.0F;
    for (int ib = 0; ib < blocks; ++ib) {
      const uint8_t* xb = wq.data() + static_cast<size_t>(ib) * kBlockBytes;
      const uint8_t* yb = aq.data() + static_cast<size_t>(ib) * kBlockBytes;
      uint16_t xd = 0;
      uint16_t yd = 0;
      std::memcpy(&xd, xb, sizeof(xd));
      std::memcpy(&yd, yb, sizeof(yd));
      int sumi = 0;
      for (int j = 0; j < 32; ++j) {
        sumi += static_cast<int>(static_cast<int8_t>(xb[2 + j])) *
                static_cast<int>(static_cast<int8_t>(yb[2 + j]));
      }
      // quants.c:400 order: the two f16 scales multiply FIRST, then sumi.
      const float d = vt::F16ToF32(xd) * vt::F16ToF32(yd);
      // volatile pins the separately-rounded product so THIS TU cannot itself
      // be contracted into the fma form.
      volatile float prod = static_cast<float>(sumi) * d;
      ref_mul += prod;
      ref_fma = std::fmaf(static_cast<float>(sumi), d, ref_fma);
    }
    const float got = RunVecDotFn(portable, wq.data(), aq.data(), k);
    CAPTURE(got);
    CAPTURE(ref_mul);
    CAPTURE(ref_fma);
    CHECK((got == ref_mul || got == ref_fma));
  }
}
// ---------------------------------------------------------------------------
// G3 — the GEMM wiring (kMatmulBTQuant), ported MUL_MAT cases
// ---------------------------------------------------------------------------

namespace {

// Build the [N,K] block weight tensor + [M,K] f32 activations and run the op.
struct GemmFixture {
  std::vector<uint8_t> wq;
  std::vector<float> a;
  std::vector<float> out;
};

GemmFixture RunGemm(const WeightCase& c, int64_t m, int64_t k, int64_t n,
                    uint32_t seed, int64_t* out_n = nullptr) {
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  GemmFixture f;
  f.wq = RandomBlocks(c, n * (k / c.block_elems), seed);
  f.a.resize(static_cast<size_t>(m * k));
  GenerateData(1.0F, f.a.size(), f.a.data());
  f.out.assign(static_cast<size_t>(m * n), 0.0F);

  vt::Tensor at =
      vt::Tensor::Contiguous(f.a.data(), vt::DType::kF32, q.device, {m, k});
  vt::Tensor bt = vt::Tensor::Contiguous(f.wq.data(), vt::DType::kF32, q.device,
                                         {n, k});
  bt.dtype = c.dtype;  // block dtype: elementwise strides are not meaningful
  vt::Tensor ot =
      vt::Tensor::Contiguous(f.out.data(), vt::DType::kF32, q.device, {m, n});
  vt::MatmulBTQuant(q, ot, at, bt);
  if (out_n != nullptr) *out_n = n;
  return f;
}

}  // namespace

TEST_CASE("G3 MatmulBTQuant NMSE <= 5e-4 vs dequant-f32 (test-backend-ops)") {
  // Port of the MUL_MAT cases: the quantized GEMM is measured against the
  // full-precision composite (dequantize the weight, dot in f32 against the
  // UNQUANTIZED activation), which is exactly the quantization error upstream
  // bounds at 5e-4. Shapes span decode (M=1) through prefill (M=512) at
  // model-ish K/N, plus the odd N that catches a chunking assumption.
  for (const WeightCase& c : kWeightCases) {
    const int64_t k = 8 * c.block_elems;
    for (int64_t m : {int64_t{1}, int64_t{4}, int64_t{32}, int64_t{512}}) {
      for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{16}}) {
        CAPTURE(std::string(c.name));
        CAPTURE(m);
        CAPTURE(k);
        CAPTURE(n);
        const GemmFixture f = RunGemm(c, m, k, n, 0x5EEDU);

        // Independent reference: the loader-side decoder + an f64 dot.
        std::vector<float> w(static_cast<size_t>(n * k));
        vt::cpu::BlockToFloat(c.dtype)(f.wq.data(), w.data(), n * k);

        double num = 0;
        double den = 0;
        for (int64_t i = 0; i < m; ++i) {
          for (int64_t j = 0; j < n; ++j) {
            double ref = 0;
            for (int64_t p = 0; p < k; ++p) {
              ref += static_cast<double>(f.a[static_cast<size_t>(i * k + p)]) *
                     static_cast<double>(w[static_cast<size_t>(j * k + p)]);
            }
            const double got = f.out[static_cast<size_t>(i * n + j)];
            num += (got - ref) * (got - ref);
            den += ref * ref;
          }
        }
        const double nmse = den > 0 ? num / den : num;
        CAPTURE(nmse);
        const double nmse_ceiling =
            c.nmse_max > 0 ? c.nmse_max : kMaxNmseErr;
        CAPTURE(nmse_ceiling);
        CHECK(nmse <= nmse_ceiling);
      }
    }
  }
}

TEST_CASE("G3 MatmulBTQuant is bit-exact run to run and across thread counts") {
  // The project rule: output rows are partitioned, each output keeps its own
  // sequential K reduction, so thread count must not perturb a single bit.
  for (const WeightCase& c : kWeightCases) {
    CAPTURE(std::string(c.name));
    const int64_t k = 4 * c.block_elems;
    const int64_t m = 3;
    const int64_t n = 9;

    const GemmFixture base = RunGemm(c, m, k, n, 0xBEEFU);
    for (int threads : {1, 2, 4}) {
      CAPTURE(threads);
      vt::cpu::Threadpool tp(threads);
      vt::cpu::Threadpool* prev = vt::cpu::Threadpool::SwapForTesting(&tp);
      const GemmFixture again = RunGemm(c, m, k, n, 0xBEEFU);
      vt::cpu::Threadpool::SwapForTesting(prev);

      REQUIRE(again.out.size() == base.out.size());
      CHECK(std::memcmp(again.out.data(), base.out.data(),
                        base.out.size() * sizeof(float)) == 0);
    }
  }
}

TEST_CASE("G3 MatmulBTQuant fails loudly on ragged K") {
  // K must be a whole number of blocks for BOTH the weight encoding and the
  // activation encoding it dots against (256 for the K-quants). A partial
  // block would mis-stride the scratch, so it must throw, never round down.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  for (const WeightCase& c : kWeightCases) {
    CAPTURE(std::string(c.name));
    const int64_t k = c.block_elems + 1;  // ragged
    const int64_t m = 1;
    const int64_t n = 1;
    std::vector<uint8_t> wq(static_cast<size_t>(4 * c.block_bytes), 0);
    std::vector<float> a(static_cast<size_t>(m * k), 0.25F);
    std::vector<float> out(static_cast<size_t>(m * n), 0.0F);

    vt::Tensor at =
        vt::Tensor::Contiguous(a.data(), vt::DType::kF32, q.device, {m, k});
    vt::Tensor bt =
        vt::Tensor::Contiguous(wq.data(), vt::DType::kF32, q.device, {n, k});
    bt.dtype = c.dtype;
    vt::Tensor ot =
        vt::Tensor::Contiguous(out.data(), vt::DType::kF32, q.device, {m, n});
    CHECK_THROWS(vt::MatmulBTQuant(q, ot, at, bt));
  }
}

// ---------------------------------------------------------------------------
// G6 — Arm i8mm (mmla) tier cross-check
//
// The mmla `nrc==2` kernels (cpu_quant_dot_arm.cpp) only exist on i8mm-capable
// aarch64; on every other host QuantMmlaVecDot is null and these cases skip
// after asserting that coherently. Where the tier IS live, the mmla GEMM output
// is cross-checked against the PORTABLE scalar vec_dot (a genuinely different
// tier): the ratified bar is NMSE <= 5e-4, and Q8_0/Q4_0 — whose only float step
// is a block-by-block non-fused vmlaq_f32 MAC in the scalar kernel's order — are
// additionally asserted BIT-identical to portable. Cross-thread bit-identity on
// the mmla path is the determinism gate.
// ---------------------------------------------------------------------------

namespace {

const WeightCase kMmlaCases[] = {
    {vt::DType::kQ4_0, 32, 18, 0, -1, -1, "q4_0"},
    {vt::DType::kQ8_0, 32, 34, 0, -1, -1, "q8_0"},
    {vt::DType::kQ4_K, 256, 144, 0, 2, -1, "q4_K"},
    {vt::DType::kQ6_K, 256, 210, 208, -1, -1, "q6_K"},
};

// Q8_0/Q4_0's only float op is the block-by-block vmlaq_f32 MAC, in the same
// order as the scalar kernel and non-fused under -ffp-contract=off, so it is
// expected bit-identical to the portable tier. The K-quants add a vpaddq/vmull
// bias reduction that reassociates and are gated at NMSE only.
bool MmlaBitExactTier(vt::DType d) {
  return d == vt::DType::kQ4_0 || d == vt::DType::kQ8_0;
}

}  // namespace

TEST_CASE("G6 mmla tier availability is coherent") {
  // Non-null iff the tier is live AND the dtype has an upstream mmla path.
  for (vt::DType d : {vt::DType::kQ4_0, vt::DType::kQ8_0, vt::DType::kQ4_K,
                      vt::DType::kQ6_K}) {
    CAPTURE(vt::Name(d));
    CHECK((vt::cpu::QuantMmlaVecDot(d) != nullptr) == vt::cpu::QuantMmlaActive());
  }
  // q3_K/q5_K have no upstream mmla path: always null, even when the tier is live.
  CHECK(vt::cpu::QuantMmlaVecDot(vt::DType::kQ3_K) == nullptr);
  CHECK(vt::cpu::QuantMmlaVecDot(vt::DType::kQ5_K) == nullptr);
  if (!vt::cpu::QuantMmlaActive()) {
    MESSAGE("i8mm mmla tier not live on this host; G6 numeric checks skipped");
  }
}

TEST_CASE("G6 mmla GEMM agrees with portable per-element vec_dot") {
  if (!vt::cpu::QuantMmlaActive()) return;  // portable-only host
  // Even M and N so the GEMM engages the 2x2 mmla tile for EVERY output element.
  for (const WeightCase& c : kMmlaCases) {
    const int64_t k = 6 * c.block_elems;
    for (int64_t m : {int64_t{2}, int64_t{4}, int64_t{8}, int64_t{128}}) {
      for (int64_t n : {int64_t{2}, int64_t{16}, int64_t{48}}) {
        CAPTURE(std::string(c.name));
        CAPTURE(m);
        CAPTURE(k);
        CAPTURE(n);
        const GemmFixture f = RunGemm(c, m, k, n, 0x6A11U);
        const size_t w_row = static_cast<size_t>(vt::RowSizeBytes(c.dtype, k));

        double num = 0;
        double den = 0;
        int64_t exact = 0;
        int64_t total = 0;
        for (int64_t i = 0; i < m; ++i) {
          const std::vector<uint8_t> aq =
              QuantizeActivation(c.dtype, f.a.data() + i * k, k);
          for (int64_t j = 0; j < n; ++j) {
            const float ref = RunVecDot(
                c.dtype, f.wq.data() + static_cast<size_t>(j) * w_row, aq.data(), k);
            const float got = f.out[static_cast<size_t>(i * n + j)];
            num += static_cast<double>(got - ref) * static_cast<double>(got - ref);
            den += static_cast<double>(ref) * static_cast<double>(ref);
            if (got == ref) ++exact;
            ++total;
          }
        }
        const double nmse = den > 0 ? num / den : num;
        CAPTURE(nmse);
        CAPTURE(exact);
        CAPTURE(total);
        const double nmse_ceiling =
            c.nmse_max > 0 ? c.nmse_max : kMaxNmseErr;
        CAPTURE(nmse_ceiling);
        CHECK(nmse <= nmse_ceiling);
        if (MmlaBitExactTier(c.dtype)) CHECK(exact == total);
      }
    }
  }
}

TEST_CASE("G6 mmla GEMM is bit-identical across thread counts") {
  if (!vt::cpu::QuantMmlaActive()) return;
  for (const WeightCase& c : kMmlaCases) {
    CAPTURE(std::string(c.name));
    const int64_t k = 4 * c.block_elems;
    const int64_t m = 8;   // even -> mmla path
    const int64_t n = 24;  // even -> mmla path
    const GemmFixture base = RunGemm(c, m, k, n, 0x711EU);
    for (int threads : {1, 2, 4, 20}) {
      CAPTURE(threads);
      vt::cpu::Threadpool tp(threads);
      vt::cpu::Threadpool* prev = vt::cpu::Threadpool::SwapForTesting(&tp);
      const GemmFixture again = RunGemm(c, m, k, n, 0x711EU);
      vt::cpu::Threadpool::SwapForTesting(prev);
      REQUIRE(again.out.size() == base.out.size());
      CHECK(std::memcmp(again.out.data(), base.out.data(),
                        base.out.size() * sizeof(float)) == 0);
    }
  }
}

TEST_CASE("G3 MatmulBTQuant matches per-row vec_dot exactly (no GEMM drift)") {
  // The GEMM must be nothing but "quantize src1 once, then one vec_dot per
  // output" (ggml-cpu.c:1313-1443). Comparing the op's output BIT-EXACTLY
  // against a hand-driven vec_dot per element proves the wiring adds no
  // reordering, no fused accumulation and no stride slip.
  for (const WeightCase& c : kWeightCases) {
    CAPTURE(std::string(c.name));
    const int64_t k = 3 * c.block_elems;
    const int64_t m = 2;
    const int64_t n = 5;
    const GemmFixture f = RunGemm(c, m, k, n, 0xF00DU);

    const size_t w_row = static_cast<size_t>(vt::RowSizeBytes(c.dtype, k));
    for (int64_t i = 0; i < m; ++i) {
      const std::vector<uint8_t> aq =
          QuantizeActivation(c.dtype, f.a.data() + i * k, k);
      for (int64_t j = 0; j < n; ++j) {
        const float expect =
            RunVecDot(c.dtype, f.wq.data() + static_cast<size_t>(j) * w_row,
                      aq.data(), k);
        CHECK(f.out[static_cast<size_t>(i * n + j)] == expect);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// GROUPED keep-quant GEMM (kMatmulBTQuantGrouped) — the ACTIVATION/OUTPUT DTYPE
// contract.
//
// `vt::MatmulBTQuantGrouped` accepts ANY float activation (`IsFloat`) and an
// f32/bf16 output (`IsOutFloat`) — ops.cpp:220-221 — and the CUDA provider
// honours all three activation dtypes (cuda_quant_dot.cu:1868-1871). Every
// pre-existing caller and test happened to pass f32/f32, so the CPU provider's
// f32-only row addressing was never exercised; qwen3_5's grouped MoE
// (`KqGrouped`, bf16 activations) was the first bf16 caller and produced
// garbage on CPU.
//
// The contract, stated once per dtype: grouped over the stacked [E*N,K] tower
// is BIT-IDENTICAL to `vt::MatmulBTQuant` on the per-expert [N,K] row-slice,
// for the SAME activation bytes. Any dtype the op ACCEPTS must satisfy it.
namespace {

// The stacked tower + P (token,expert) pairs the grouped op consumes.
struct GroupedFixture {
  std::vector<uint8_t> tower;  // [E*N, K] block-quant
  std::vector<float> a_f32;    // [P, K] activation, f32 master copy
  std::vector<int32_t> eids;   // [P]
};

GroupedFixture MakeGrouped(const WeightCase& c, int64_t E, int64_t N, int64_t K,
                           int64_t P, uint32_t seed) {
  GroupedFixture f;
  f.tower = RandomBlocks(c, E * N * (K / c.block_elems), seed);
  f.a_f32.resize(static_cast<size_t>(P * K));
  GenerateData(1.0F, f.a_f32.size(), f.a_f32.data());
  f.eids.resize(static_cast<size_t>(P));
  for (int64_t p = 0; p < P; ++p)
    f.eids[static_cast<size_t>(p)] = static_cast<int32_t>((p * 3 + 1) % E);
  return f;
}

// One expert's [N,K] row-slice of the stacked tower, as kMatmulBTQuant sees it.
vt::Tensor ExpertSlice(const GroupedFixture& f, const WeightCase& c, int64_t e,
                       int64_t N, int64_t K, vt::Device dev) {
  vt::Tensor w{};
  w.data = const_cast<uint8_t*>(f.tower.data()) +
           static_cast<size_t>(e) * N * vt::RowSizeBytes(c.dtype, K);
  w.dtype = c.dtype;
  w.device = dev;
  w.rank = 2;
  w.shape[0] = N;
  w.shape[1] = K;
  w.stride[0] = K;
  w.stride[1] = 1;
  return w;
}

}  // namespace

TEST_CASE("grouped keep-quant GEMM == per-expert slice for EVERY accepted "
          "activation dtype (f32/f16/bf16)") {
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const int64_t E = 4;
  const int64_t N = 6;
  const int64_t P = 5;

  for (const WeightCase& c : kWeightCases) {
    CAPTURE(std::string(c.name));
    const int64_t K = 2 * c.block_elems;
    const GroupedFixture f = MakeGrouped(c, E, N, K, P, 0xA5A5U);

    for (vt::DType adt : {vt::DType::kF32, vt::DType::kF16, vt::DType::kBF16}) {
      CAPTURE(static_cast<int>(adt));

      // The SAME activation values in the dtype under test. f16/bf16 round the
      // f32 master, so both arms below read the identical rounded bytes — any
      // difference is the op's row addressing, never the rounding.
      std::vector<float> a32(f.a_f32.size());
      std::vector<uint16_t> a16(f.a_f32.size());
      for (size_t i = 0; i < f.a_f32.size(); ++i) {
        if (adt == vt::DType::kF32) {
          a32[i] = f.a_f32[i];
        } else if (adt == vt::DType::kF16) {
          a16[i] = vt::F32ToF16(f.a_f32[i]);
        } else {
          a16[i] = vt::F32ToBF16(f.a_f32[i]);
        }
      }
      void* adata = adt == vt::DType::kF32 ? static_cast<void*>(a32.data())
                                           : static_cast<void*>(a16.data());

      // (A) ONE grouped launch over the stacked tower.
      std::vector<float> og(static_cast<size_t>(P * N), 0.0F);
      {
        std::vector<int32_t> ids = f.eids;
        vt::Tensor at = vt::Tensor::Contiguous(adata, adt, q.device, {P, K});
        vt::Tensor ot =
            vt::Tensor::Contiguous(og.data(), vt::DType::kF32, q.device, {P, N});
        vt::Tensor eid =
            vt::Tensor::Contiguous(ids.data(), vt::DType::kI32, q.device, {P});
        vt::Tensor wt = vt::Tensor::Contiguous(const_cast<uint8_t*>(f.tower.data()), vt::DType::kF32,
                                               q.device, {E * N, K});
        wt.dtype = c.dtype;  // block dtype: elementwise strides are inert
        vt::MatmulBTQuantGrouped(q, ot, at, wt, eid);
      }

      // (B) P per-expert kMatmulBTQuant calls on the same slices/rows.
      std::vector<float> op(static_cast<size_t>(P * N), 0.0F);
      for (int64_t p = 0; p < P; ++p) {
        void* arow = adt == vt::DType::kF32
                         ? static_cast<void*>(a32.data() + p * K)
                         : static_cast<void*>(a16.data() + p * K);
        vt::Tensor at = vt::Tensor::Contiguous(arow, adt, q.device, {1, K});
        vt::Tensor ot = vt::Tensor::Contiguous(op.data() + p * N,
                                               vt::DType::kF32, q.device, {1, N});
        vt::Tensor wt = ExpertSlice(f, c, f.eids[static_cast<size_t>(p)], N, K,
                                    q.device);
        vt::MatmulBTQuant(q, ot, at, wt);
      }

      REQUIRE(og.size() == op.size());
      CHECK(std::memcmp(og.data(), op.data(), og.size() * sizeof(float)) == 0);
    }
  }
}

TEST_CASE("grouped keep-quant GEMM == per-expert slice for a BF16 output") {
  // `IsOutFloat` accepts bf16, and StoreOutF32 writes it — but the grouped
  // kernel must also STRIDE the output rows by the output dtype.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const int64_t E = 4;
  const int64_t N = 6;
  const int64_t P = 5;

  for (const WeightCase& c : kWeightCases) {
    CAPTURE(std::string(c.name));
    const int64_t K = 2 * c.block_elems;
    const GroupedFixture f = MakeGrouped(c, E, N, K, P, 0x1234U);

    std::vector<uint16_t> og(static_cast<size_t>(P * N), 0);
    {
      std::vector<int32_t> ids = f.eids;
      vt::Tensor at = vt::Tensor::Contiguous(
          const_cast<float*>(f.a_f32.data()), vt::DType::kF32, q.device, {P, K});
      vt::Tensor ot =
          vt::Tensor::Contiguous(og.data(), vt::DType::kBF16, q.device, {P, N});
      vt::Tensor eid =
          vt::Tensor::Contiguous(ids.data(), vt::DType::kI32, q.device, {P});
      vt::Tensor wt = vt::Tensor::Contiguous(const_cast<uint8_t*>(f.tower.data()), vt::DType::kF32,
                                             q.device, {E * N, K});
      wt.dtype = c.dtype;
      vt::MatmulBTQuantGrouped(q, ot, at, wt, eid);
    }

    std::vector<uint16_t> op(static_cast<size_t>(P * N), 0);
    for (int64_t p = 0; p < P; ++p) {
      vt::Tensor at = vt::Tensor::Contiguous(
          const_cast<float*>(f.a_f32.data()) + p * K, vt::DType::kF32, q.device,
          {1, K});
      vt::Tensor ot = vt::Tensor::Contiguous(op.data() + p * N, vt::DType::kBF16,
                                             q.device, {1, N});
      vt::Tensor wt =
          ExpertSlice(f, c, f.eids[static_cast<size_t>(p)], N, K, q.device);
      vt::MatmulBTQuant(q, ot, at, wt);
    }

    REQUIRE(og.size() == op.size());
    CHECK(std::memcmp(og.data(), op.data(), og.size() * sizeof(uint16_t)) == 0);
  }
}
