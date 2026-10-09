// EXL3 (exllamav3 trellis) device kernels, CPU arm — MODEL-DSV4-EXL3 W2a/W2b.
//
// PORTED 1:1 FROM exllamav3 @ 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT):
//   exllamav3_ext/quant/hadamard.cu:88-173          had_r_128, the host launcher
//   exllamav3_ext/quant/hadamard_inner.cuh:17-44    shuffle_had_f4x32
//   exllamav3_ext/quant/hadamard_inner.cuh:93-279   the hf / ff / fh inners
//   exllamav3_ext/quant/exl3_gemm_kernel.cuh:8-80   the fused chain the GEMM runs
//   exllamav3_ext/quant/exl3_dq.cuh + codebook.cuh  reached through W1a's
//                                                   Exl3DecodeTile (cpu_exl3_dequant.cpp)
//   exllamav3_ext/quant/exl3_moe_kernel.cuh:17-283  the fused MoE MLP (W2d)
//   exllamav3_ext/quant/hadamard_inner.cuh:284-473  its guad / d epilogues (W2d)
//
// WHY THE OPERATION ORDER IS COPIED AND NOT SIMPLIFIED. The obvious CPU
// Hadamard is a two-line loop nest. This one instead reproduces upstream's warp
// decomposition exactly: levels 1-2 on the four values a "lane" holds, levels
// 4-64 as five xor-partner steps over 32 lanes, the sign flip performed by
// XORing the f32 SIGN BIT, every add in f32, one multiply by `r_scale` at the
// end and one round at the store. That is what makes the claim in
// `.agents/specs/model-dsv4-exl3.md` `## W2 design` §1 tier 2 a BYTE claim
// rather than a tolerance: the CUDA arm and this arm perform the same f32
// operations on the same values in the same order, so a device-vs-host gate can
// require equality and a defect cannot hide inside a bound. Written as a plain
// loop it would be close, and close is not checkable.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "cpu_threadpool.h"

#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/unaligned.h"

namespace vt::cpu {
namespace {

// 1/sqrt(128), spelled as upstream spells it (hadamard.cu:107). Kept as the
// literal rather than computed, because the literal is what the device rounds
// to and a recomputed 1/sqrt(128) can differ in the last f32 bit.
constexpr float kInvSqrt128 = 0.088388347648f;

inline float NegBySignBit(float v) {
  uint32_t u;
  std::memcpy(&u, &v, sizeof(u));
  u ^= 0x80000000u;
  float r;
  std::memcpy(&r, &u, sizeof(r));
  return r;
}

// fp16 multiply, the CPU spelling of __hmul2. The exact product of two fp16
// significands is 22 bits and fits f32 exactly, so rounding once from f32 is
// the same value an fp16 multiply produces.
inline uint16_t MulF16(uint16_t a, uint16_t b) {
  return F32ToF16(F16ToF32(a) * F16ToF32(b));
}

// `shuffle_had_f4x32` (hadamard_inner.cuh:17-44) over a whole 32-"lane" warp.
// `h[j][t]` is lane t's j-th value; the element it stands for is column 4*t + j.
// Lane bit `i` therefore carries Hadamard level 4*i, which is what makes levels
// 4..64 exactly the five steps below.
void ShuffleHadWarp(float h[4][32]) {
  float next[4][32];
  for (int i = 1; i < 32; i <<= 1) {
    for (int t = 0; t < 32; ++t) {
      const int p = t ^ i;
      const bool flip = (t & i) != 0;
      for (int j = 0; j < 4; ++j) {
        const float own = flip ? NegBySignBit(h[j][t]) : h[j][t];
        next[j][t] = own + h[j][p];
      }
    }
    std::memcpy(h, next, sizeof(next));
  }
}

// The 128-element transform of ONE block, in upstream's order. `in` holds the
// 128 f32 values ALREADY pre-scaled (or not); `out` receives them transformed
// and multiplied by `r_scale`, before any post-scale.
void HadBlock128(const float* in, float* out, float r_scale) {
  float h[4][32];
  for (int t = 0; t < 32; ++t) {
    // hadamard_inner.cuh:118-129, the level-1 and level-2 butterflies.
    const float v0 = in[4 * t + 0];
    const float v1 = in[4 * t + 1];
    const float v2 = in[4 * t + 2];
    const float v3 = in[4 * t + 3];
    const float s0 = v0 + v1;
    const float d0 = v0 - v1;
    const float s1 = v2 + v3;
    const float d1 = v2 - v3;
    h[0][t] = s0 + s1;
    h[1][t] = d0 + d1;
    h[2][t] = s0 - s1;
    h[3][t] = d0 - d1;
  }
  ShuffleHadWarp(h);
  for (int t = 0; t < 32; ++t)
    for (int j = 0; j < 4; ++j) out[4 * t + j] = h[j][t] * r_scale;
}

// `had_hf_r_128_inner` (fp16 in, fp16 out), `had_ff_r_128_inner` (f32 in, f32
// out) and `had_fh_r_128_inner` (f32 in, fp16 out) differ in exactly three
// places: how the value is loaded, how a pre-scale multiplies it (fp16 multiply
// for the half input, f32 multiply for the float input — hadamard_inner.cuh:109
// vs :167), and how the post-scale applies after the store rounding. Everything
// between is the same f32 arithmetic, so the three share this body.
// The four (input, output) width combinations upstream's inners come in.
// `kHalfFloat` is `had_hf_r_128_d_inner` (hadamard_inner.cuh:418-473), the MoE
// epilogue: it reads the down GEMM's fp16 store, transforms in f32, and applies
// the post-scale in f32 on the way into the f32 accumulator.
enum class HadIo { kHalfHalf, kFloatFloat, kFloatHalf, kHalfFloat };

constexpr bool HadHalfIn(HadIo io) {
  return io == HadIo::kHalfHalf || io == HadIo::kHalfFloat;
}
constexpr bool HadHalfOut(HadIo io) {
  return io == HadIo::kHalfHalf || io == HadIo::kFloatHalf;
}

// THE SCALES ARE WEIGHTS AND MAY SIT AT ANY BYTE. `pre` and `post` are `suh`
// and `svh`, which `BorrowStTensorBytes` hands over as the mapping's own bytes.
// A safetensors payload starts at `8 + <JSON header length>`
// (`safetensors_reader.cpp:78`) and a header length is arbitrary, so an F16
// scale vector begins on an odd byte in roughly half of all checkpoints. That is
// an ordinary file, not a corrupt one. They are therefore `const void*` here and
// are read one element at a time with `vt::LoadUnaligned`: forming a
// `const uint16_t*` over an odd address is undefined even where x86 executes
// the load, and `-fsanitize=alignment` aborted `test_qwen35_exl3` on exactly
// this operand (#2558). The load is free — at `-O2` it is the same `movzwl`
// the raw index compiled to.
//
// `in` and `out` KEEP their typed pointers, and the asymmetry is the point:
// they are the activation and the engine's own scratch, which
// `cpu_backend.cpp:20,42` allocates through `std::aligned_alloc(64, ...)`. This
// is the polarity `cpu_layernorm.cpp:44-46` states — an output is an
// engine-allocated buffer, never a borrowed file mapping.
inline uint16_t ScaleAt(const void* scales, int64_t i) {
  return LoadUnaligned<uint16_t>(static_cast<const unsigned char*>(scales) + i * 2);
}

void HadRowBlock(HadIo io, const void* in, void* out, const void* pre, const void* post,
                 float r_scale, int64_t block_base) {
  float buf[128];
  if (HadHalfIn(io)) {
    const uint16_t* p = static_cast<const uint16_t*>(in);
    for (int i = 0; i < 128; ++i) {
      // pre_scale rides an fp16 multiply BEFORE the widen (hadamard_inner.cuh:112-114).
      const uint16_t v = pre != nullptr ? MulF16(p[i], ScaleAt(pre, block_base + i)) : p[i];
      buf[i] = F16ToF32(v);
    }
  } else {
    const float* p = static_cast<const float*>(in);
    for (int i = 0; i < 128; ++i) {
      // the float inners widen the fp16 scale and multiply in f32
      // (hadamard_inner.cuh:171-174).
      buf[i] = pre != nullptr ? p[i] * F16ToF32(ScaleAt(pre, block_base + i)) : p[i];
    }
  }

  float res[128];
  HadBlock128(buf, res, r_scale);

  if (!HadHalfOut(io)) {
    float* o = static_cast<float*>(out);
    for (int i = 0; i < 128; ++i)
      o[i] = post != nullptr ? res[i] * F16ToF32(ScaleAt(post, block_base + i)) : res[i];
  } else {
    // Both half-output inners round FIRST and apply the post-scale as an fp16
    // multiply afterwards (hadamard_inner.cuh:137-146 and :264-278).
    uint16_t* o = static_cast<uint16_t*>(out);
    for (int i = 0; i < 128; ++i) {
      const uint16_t r = F32ToF16(res[i]);
      o[i] = post != nullptr ? MulF16(r, ScaleAt(post, block_base + i)) : r;
    }
  }
}

void HadRows(HadIo io, const void* in, void* out, const void* pre, const void* post,
             float r_scale, int64_t rows, int64_t cols) {
  const bool half_in = HadHalfIn(io);
  const bool half_out = HadHalfOut(io);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t b = 0; b < cols; b += 128) {
      const void* ip = half_in ? static_cast<const void*>(static_cast<const uint16_t*>(in) + r * cols + b)
                               : static_cast<const void*>(static_cast<const float*>(in) + r * cols + b);
      void* op = half_out ? static_cast<void*>(static_cast<uint16_t*>(out) + r * cols + b)
                          : static_cast<void*>(static_cast<float*>(out) + r * cols + b);
      HadRowBlock(io, ip, op, pre, post, r_scale, b);
    }
  }
}

void Exl3HadR128KernelCpu(Queue& q, Tensor& out, const Tensor& in, const Exl3HadArgs& args) {
  (void)q;
  const int64_t rows = in.shape[0];
  const int64_t cols = in.shape[1];
  if (rows == 0 || cols == 0) return;
  // `.data`, not `Ptr<uint16_t>()`: these scales may be borrowed mapping bytes
  // at an odd address, so no typed pointer is formed over them at all.
  const void* pre = args.pre_scale != nullptr ? args.pre_scale->data : nullptr;
  const void* post = args.post_scale != nullptr ? args.post_scale->data : nullptr;
  const float r_scale = args.scale * kInvSqrt128;  // hadamard.cu:107
  HadRows(in.dtype == DType::kF16 ? HadIo::kHalfHalf : HadIo::kFloatFloat, in.data, out.data, pre,
          post, r_scale, rows, cols);
}

// The fused chain `exl3_gemm_kernel` runs (exl3_gemm_kernel.cuh:14-50 plus the
// output transform at exl3_gemm_inner.cuh:456-480):
//   A_had = had_r_128(A, pre_scale = suh)
//   C_raw = A_had @ reconstruct(trellis)          [f32 accumulation]
//   C     = had_r_128(C_raw, post_scale = svh)
// The device does the first and third INSIDE the GEMM launch; here they are the
// same three steps, and the middle one accumulates in f32 exactly as the mma
// accumulators do. The trellis is decoded a 16x16 TILE AT A TIME rather than
// materialised as a [k, n] matrix, because a real expert is k = 4096, n = 2048
// and the materialised f32 weight would be 32 MiB per projection.
void Exl3GemmKernelCpu(Queue& q, Tensor& c, const Tensor& a, const Tensor& trellis,
                       const Tensor& suh, const Tensor& svh, Tensor& a_had,
                       const Exl3GemmArgs& args) {
  (void)q;
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];
  const int64_t n = c.shape[1];
  if (m == 0 || k == 0 || n == 0) return;

  // 1. the input transform, into the caller's scratch (which may alias A).
  HadRows(HadIo::kHalfHalf, a.data, a_had.data, suh.data, nullptr, kInvSqrt128, m, k);

  // 2. the matmul against the decoded trellis, f32 accumulators.
  const uint16_t* ah = a_had.Ptr<uint16_t>();
  // The trellis is kI8 — opaque BYTES whose alignment requirement is 1 — and it
  // is borrowed straight out of the mapping, so the tile cursor advances in
  // bytes. `tile_bytes` is `tile_words * sizeof(uint16_t)`, and that factor of
  // two is the one thing this cursor must not lose (#2558).
  const auto* tw = static_cast<const unsigned char*>(trellis.data);
  const int64_t tiles_n = n / 16;
  const int64_t tile_words =
      16 * static_cast<int64_t>(args.bits) + (args.half ? 8 : 0);
  const int64_t tile_bytes = tile_words * static_cast<int64_t>(sizeof(uint16_t));
  std::vector<float> raw(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);

  // PARALLEL OVER OUTPUT TILES, and the loop order is inverted for it: `tj`
  // outermost so each worker owns a disjoint 16-column stripe of `raw`, with
  // the `ti` accumulation kept inside one worker. The original order had `ti`
  // outermost, which accumulates ACROSS workers into the same columns and
  // cannot be split without a reduction.
  //
  // This was the only CPU kernel in the tree running single-threaded, and it is
  // the one that decodes: the trellis is decoded inside the GEMM, so a forward
  // pass re-decodes every weight of the model. On the 1B stock EXL3 checkpoint
  // that is 1,235,746,816 weights per token, which at ~50M weights/s on one
  // core is the whole of the 0.040 tok/s W1b measured. The device arm is the
  // real answer (QUANT-EXL3 W3); this is what the FALLBACK costs when the
  // device declines, and every non-CUDA backend falls back here.
  cpu::ParallelForRows(cpu::CurrentThreadpool(), tiles_n, [&](int64_t j0, int64_t j1) {
    float tile[256];
    for (int64_t tj = j0; tj < j1; ++tj) {
      for (int64_t ti = 0; ti < k / 16; ++ti) {
        Exl3DecodeTile(tw + (ti * tiles_n + tj) * tile_bytes, args.bits, args.codebook,
                       tile, args.half);
        for (int64_t r = 0; r < m; ++r) {
          float* orow = &raw[static_cast<size_t>(r * n + tj * 16)];
          for (int rr = 0; rr < 16; ++rr) {
            const float xv = F16ToF32(ah[r * k + ti * 16 + rr]);
            if (xv == 0.0f) continue;
            const float* wrow = tile + rr * 16;
            for (int cc = 0; cc < 16; ++cc) orow[cc] += xv * wrow[cc];
          }
        }
      }
    }
  });

  // 3. the output transform. The device holds this tile in f32 shared memory and
  // finishes with had_ff (f32 C) or had_fh (fp16 C) — the same two arms here.
  HadRows(c.dtype == DType::kF32 ? HadIo::kFloatFloat : HadIo::kFloatHalf, raw.data(), c.data,
          nullptr, svh.data, kInvSqrt128, m, n);
}

// ── the fused MoE MLP, CPU arm (exl3_moe_kernel.cuh:17-283) ──────────────────
//
// The device kernel is a persistent cooperative launch: `concurrency` groups of
// blocks draw expert tickets from a self-resetting scheduler and meet at group
// barriers between the five stages. NONE of that is reproduced here, and nothing
// is lost by dropping it: a group is a DEVICE SCHEDULING UNIT, the experts are
// mutually independent, and the CPU arm walks them in order. What IS reproduced,
// stage for stage, is the DATA FLOW — because that is what the tier-4 gate
// compares and what a wrong transcription would move.
//
//   1. gather + input Hadamard for g and u  (:85-114)
//   2. the gate and up GEMMs                (:117-161)
//   3. output Hadamard for g and u, the activation and gate, then the input
//      Hadamard for d, all in ONE fp16 pass (:165-188, hadamard_inner.cuh:284-413)
//   4. the down GEMM                        (:191-233)
//   5. output Hadamard for d, times the routing weight, scatter-ADDED into the
//      f32 accumulator                      (:237-259, hadamard_inner.cuh:418-473)
//
// STAGE 3 IS WHY THE FUSED ARM AND THE LOOP ARM DIFFER AT ALL. Upstream holds
// the intermediate in fp16 from the gate GEMM's store through the activation to
// the down GEMM's load; the loop arm widens to f32, activates in f32 and rounds
// back. Same algebra, different rounding, and the spec's tier 4 is the bound on
// the difference.
void MoeGemm(const uint16_t* a_had, const void* trellis, float* raw, int64_t m, int64_t k,
             int64_t n, int bits, int codebook) {
  // The same tile walk `Exl3GemmKernelCpu` step 2 performs, over an m-row batch,
  // and the same byte cursor for the same reason: an expert's trellis is kI8
  // borrowed straight out of the mapping (#2558).
  const int64_t tiles_n = n / 16;
  const int64_t tile_bytes = 16 * static_cast<int64_t>(bits) * static_cast<int64_t>(sizeof(uint16_t));
  const auto* tw = static_cast<const unsigned char*>(trellis);
  float tile[256];
  for (int64_t i = 0; i < m * n; ++i) raw[i] = 0.0f;
  for (int64_t ti = 0; ti < k / 16; ++ti) {
    for (int64_t tj = 0; tj < tiles_n; ++tj) {
      Exl3DecodeTile(tw + (ti * tiles_n + tj) * tile_bytes, bits, codebook, tile);
      for (int64_t r = 0; r < m; ++r) {
        float* orow = &raw[r * n + tj * 16];
        for (int rr = 0; rr < 16; ++rr) {
          const float xv = F16ToF32(a_had[r * k + ti * 16 + rr]);
          if (xv == 0.0f) continue;
          const float* wrow = tile + rr * 16;
          for (int cc = 0; cc < 16; ++cc) orow[cc] += xv * wrow[cc];
        }
      }
    }
  }
}

// Stage 3's activation, in fp16, on ALREADY post-scaled g and u values. The four
// arms are `act_function` (`exl3_moe_common.cuh:6-8` plus ours); see
// `include/vt/ops.h` and the spec's `## W2cd design` W2d-2 for why the fourth
// exists and how it differs from `kSilu`.
//
// THE ONE PLACE WHERE NO BYTE CLAIM IS AVAILABLE. The device computes this in
// fp16 through `h2exp` and `h2rcp` (hadamard_inner.cuh:323-332), which are
// hardware approximations with no host equivalent; here the same fp16 inputs
// widen to f32, `x * sigmoid(x)` is evaluated there, and the result rounds once.
// Unlike the Hadamard — where the two arms run the same f32 operations in the
// same order and the gate is a byte gate — these two cannot agree bit for bit at
// any effort, so the device-vs-host gate for this op is the spec's tier-4 bound.
uint16_t MoeActivate(uint16_t vg, uint16_t vu, Exl3MoeAct act, float act_limit) {
  const float g = F16ToF32(vg);
  const float u = F16ToF32(vu);
  switch (act) {
    case Exl3MoeAct::kSiluAndMulClamp: {
      // vLLM's SiluAndMulWithClamp (activation.py:197-201) with alpha = 1,
      // beta = 0 (nvidia/model.py:131). The clamp is UNCONDITIONAL — upstream's
      // `act_limit != 0` guard belongs to the three arms below, and a zero limit
      // here means clamp to zero, which is vLLM's own degenerate case.
      const float gate = g < act_limit ? g : act_limit;              // MAX only
      float up = u < -act_limit ? -act_limit : u;                    // BOTH sides
      if (up > act_limit) up = act_limit;
      const float out = gate / (1.0f + std::exp(-gate)) * up;
      return F32ToF16(out);
    }
    case Exl3MoeAct::kRelu2NoGate: {
      // :100-103: the gate lane is synthesized from u by a max with zero, and
      // the gate multiply below then squares it.
      float vgn = u > 0.0f ? u : 0.0f;
      float vun = u;
      if (act_limit != 0.0f) {
        vun = vun < -act_limit ? -act_limit : (vun > act_limit ? act_limit : vun);
        vgn = vgn > act_limit ? act_limit : vgn;
      }
      return F32ToF16(vgn * vun);
    }
    case Exl3MoeAct::kGelu:
    case Exl3MoeAct::kSilu:
    default: {
      float vg_a = act == Exl3MoeAct::kGelu
                       ? 0.5f * g *
                             (1.0f + std::tanh(0.797884560803f *
                                               (g + 0.044715f * g * g * g)))
                       : g / (1.0f + std::exp(-g));
      float vu_a = u;
      // :110-118. Upstream applies the limit only when it is non-zero, and it
      // applies it AFTER the activation.
      if (act_limit != 0.0f) {
        vu_a = vu_a < -act_limit ? -act_limit : (vu_a > act_limit ? act_limit : vu_a);
        vg_a = vg_a > act_limit ? act_limit : vg_a;
      }
      return F32ToF16(vg_a * vu_a);
    }
  }
}

void Exl3MoeMlpKernelCpu(Queue& q, Tensor& output_state, const Tensor& hidden_state,
                         const Exl3MoeExpertTables& tables, const Exl3MoeRouting& routing,
                         const Exl3MoeTemps& temps, const Exl3MoeArgs& args) {
  (void)q;
  const int64_t bsz = hidden_state.shape[0];
  const int64_t hidden = hidden_state.shape[1];
  const int64_t interm = temps.intermediate_g->shape[2];
  const int64_t num_experts = routing.expert_count->Numel() - 1;
  const int64_t max_rows = temps.state_g->shape[1];
  if (bsz == 0 || hidden == 0 || interm == 0) return;

  const int64_t* count = routing.expert_count->Ptr<int64_t>();
  const int64_t* token_sorted = routing.token_sorted->Ptr<int64_t>();
  const uint16_t* weight_sorted = routing.weight_sorted->Ptr<uint16_t>();
  const uint16_t* hid = hidden_state.Ptr<uint16_t>();
  float* out = output_state.Ptr<float>();

  // `const void*`, not `const uint16_t*`: every one of these is a borrowed
  // safetensors payload and may begin at an odd byte, so no typed pointer is
  // formed over one (#2558). The callees below take byte cursors.
  auto table = [](const Tensor* tt, int64_t e) -> const void* {
    return reinterpret_cast<const void*>(
        static_cast<uintptr_t>(tt->Ptr<int64_t>()[e]));
  };

  // The GROUP 0 slice of the temp buffers. A CPU run has one group; the other
  // `concurrency - 1` slices exist because the caller sizes the buffers for a
  // device and the same buffers are handed to both arms.
  uint16_t* st_g = temps.state_g->Ptr<uint16_t>();
  uint16_t* st_u = temps.state_u->Ptr<uint16_t>();
  uint16_t* in_g = temps.intermediate_g->Ptr<uint16_t>();
  uint16_t* in_u = temps.intermediate_u->Ptr<uint16_t>();

  std::vector<float> raw_g(static_cast<size_t>(max_rows) * static_cast<size_t>(interm));
  std::vector<float> raw_u(raw_g.size());
  std::vector<float> raw_d(static_cast<size_t>(max_rows) * static_cast<size_t>(hidden));

  const bool gated = args.act != Exl3MoeAct::kRelu2NoGate;
  int64_t start = 0;
  for (int64_t e = 0; e < num_experts; ++e) {
    const int64_t tokens = count[e];
    const int64_t seg = start;
    start += tokens;
    // :65-66. No tokens, or more than the temp buffers hold — the caller's
    // per-expert path covers the second case, exactly as upstream's does.
    if (tokens == 0 || tokens > max_rows) continue;

    const void* g_tr = table(tables.gate_trellis, e);
    const void* g_su = table(tables.gate_suh, e);
    const void* g_sv = table(tables.gate_svh, e);
    const void* u_tr = table(tables.up_trellis, e);
    const void* u_su = table(tables.up_suh, e);
    const void* u_sv = table(tables.up_svh, e);
    const void* d_tr = table(tables.down_trellis, e);
    const void* d_su = table(tables.down_suh, e);
    const void* d_sv = table(tables.down_svh, e);

    // stage 1: gather + input Hadamard, one 128-block per warp upstream.
    for (int64_t r = 0; r < tokens; ++r) {
      const int64_t tok = token_sorted[seg + r];
      const uint16_t* src = hid + tok * hidden;
      if (gated)
        HadRows(HadIo::kHalfHalf, src, st_g + r * hidden, g_su, nullptr, kInvSqrt128, 1, hidden);
      HadRows(HadIo::kHalfHalf, src, st_u + r * hidden, u_su, nullptr, kInvSqrt128, 1, hidden);
    }

    // stage 2: the gate and up GEMMs. Upstream calls the inner tile loop with a
    // NULL svh (`exl3_moe_kernel.cuh:130`), so the output Hadamard is skipped
    // and the f32 accumulator is ROUNDED TO FP16 at the store. Reproducing that
    // rounding is not optional: skipping it would make this arm strictly more
    // accurate than the kernel it is the reference for, and a device-vs-host
    // gate would then be measuring the difference between two intentions.
    if (gated) MoeGemm(st_g, g_tr, raw_g.data(), tokens, hidden, interm, args.bits_gate, args.codebook);
    MoeGemm(st_u, u_tr, raw_u.data(), tokens, hidden, interm, args.bits_up, args.codebook);
    for (int64_t i = 0; i < tokens * interm; ++i) {
      if (gated) in_g[i] = F32ToF16(raw_g[static_cast<size_t>(i)]);
      in_u[i] = F32ToF16(raw_u[static_cast<size_t>(i)]);
    }

    // stage 3, all of `had_hf_r_128_guad_inner` (hadamard_inner.cuh:284-413) in
    // upstream's own order: the output Hadamard for u and g with their post
    // scales, the activation on g, the gate multiply, the down PRE scale, and
    // the down input Hadamard — every step in fp16 between f32 butterflies,
    // which is the whole reason tier 4 is a bound and not a byte claim.
    for (int64_t r = 0; r < tokens; ++r) {
      uint16_t* gr = in_g + r * interm;
      uint16_t* ur = in_u + r * interm;
      HadRows(HadIo::kHalfHalf, ur, ur, nullptr, u_sv, kInvSqrt128, 1, interm);
      if (gated) HadRows(HadIo::kHalfHalf, gr, gr, nullptr, g_sv, kInvSqrt128, 1, interm);
      for (int64_t i = 0; i < interm; ++i)
        gr[i] = MoeActivate(gated ? gr[i] : ur[i], ur[i], args.act, args.act_limit);
      // the down pre-scale rides the SAME fp16 multiply the input Hadamard's
      // pre_scale arm performs (hadamard_inner.cuh:112-114 vs :404-406).
      HadRows(HadIo::kHalfHalf, gr, gr, d_su, nullptr, kInvSqrt128, 1, interm);
    }

    // stage 4: the down GEMM, again with no output Hadamard and again rounding
    // its f32 accumulator to fp16 at the store — into `state_g`, which is the
    // buffer upstream reuses for it (`exl3_moe_kernel.cuh:233`).
    MoeGemm(in_g, d_tr, raw_d.data(), tokens, interm, hidden, args.bits_down, args.codebook);
    for (int64_t i = 0; i < tokens * hidden; ++i) st_g[i] = F32ToF16(raw_d[static_cast<size_t>(i)]);

    // stage 5: `had_hf_r_128_d_inner`. The routing weight is folded into
    // `r_scale` (`exl3_moe_kernel.cuh:254`) rather than multiplied afterwards,
    // so it lands in f32 BEFORE the post scale; the result is ADDED into the
    // f32 accumulator, which is what makes `output_state` f32.
    std::vector<float> row(static_cast<size_t>(hidden));
    for (int64_t r = 0; r < tokens; ++r) {
      const float w = F16ToF32(weight_sorted[seg + r]);
      HadRows(HadIo::kHalfFloat, st_g + r * hidden, row.data(), nullptr, d_sv, kInvSqrt128 * w, 1,
              hidden);
      float* dst = out + token_sorted[seg + r] * hidden;
      for (int64_t h = 0; h < hidden; ++h) dst[h] += row[static_cast<size_t>(h)];
    }
  }
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kExl3HadR128, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<Exl3HadR128Fn>(&Exl3HadR128KernelCpu)));
    RegisterOp(OpId::kExl3Gemm, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<Exl3GemmFn>(&Exl3GemmKernelCpu)));
    RegisterOp(OpId::kExl3MoeMlp, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<Exl3MoeMlpFn>(&Exl3MoeMlpKernelCpu)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
