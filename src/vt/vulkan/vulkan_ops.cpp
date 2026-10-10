// Vulkan backend — op kernels: descriptor binding + dispatch of the committed
// SPIR-V in vulkan_spirv.h, plus the `RegisterOp` table entries. BACKEND-VULKAN,
// W0 skeleton. Self-registering TU, copying the `src/vt/cpu/cpu_ops.cpp`
// Registrar idiom exactly, so adding this backend edited NO existing kernel file.
//
// WHAT THIS TU COVERS (deliberately a SEAM PROOF, not a model):
//   kAdd, kRelu, kSiluAndMul, kCastBf16, kCastF32, kLayerNorm, kRmsNorm and the
//   single kFusedChain registration that inherits the portable fusion catalog.
// That set spans every structural class the seam has to get right: flat
// elementwise, a rank-1 broadcast, a dtype-converting copy, TWO different row
// reductions, an optional in-place residual stream, and the recipe interpreter.
// It matches the Metal skeleton's set exactly, so the two backends are directly
// comparable through tests/vt/test_backend_cross_device.cpp.
//
// SINCE THEN this TU has grown the dense path (both GEMM orientations plus the
// decode GEMV and coopmat tactics), the attention block (paged attention, the KV
// write, the QKV split, the rotary apply), the two ends of the model (embedding
// and greedy argmax), and — this row, BACKEND-VULKAN-GDN — six of the GDN /
// conv1d glue ops that Qwen3.6-27B needs.
//
// WHAT IS STILL NOT REGISTERED: the quant tier, MoE, the sampler beyond greedy
// argmax, the GDN recurrences themselves, and the ops listed in the
// BACKEND-VULKAN-GDN block comment further down. None of them THROW any more:
// since the portable reference tier landed, a missed GetOp on this
// unified-memory device installs the CPU kernel as a priority -1000 provider, so
// an unregistered op is CORRECT AND SLOW rather than fatal.
//
// BINDING MODEL: every tensor operand occupies TWO consecutive descriptor
// bindings onto the SAME VkBuffer — a uint32_t view and a uint16_t view — and
// its BYTE OFFSET travels in the push constants. See
// src/vt/vulkan/shaders/vt_common.glsl § STORAGE MODEL for why.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "vulkan_buffers.h"
#include "vulkan_context.h"
#include "vt/ops.h"
#include "vt/quant.h"

namespace vt::vulkan {
namespace {

// Gated on the same flag as the dispatch profile; costs nothing when unset.
const bool kCoopMatWhy = [] {
  const char* v = std::getenv("VT_VULKAN_DISPATCH_STATS");
  return v != nullptr && std::strcmp(v, "0") != 0;
}();

// Storage dtype -> the shader-side code (vt_common.glsl VT_DT_*).
uint32_t DtypeCode(DType d) {
  switch (d) {
    case DType::kF32: return 0;
    case DType::kF16: return 1;
    case DType::kBF16: return 2;
    default: break;
  }
  VT_CHECK(false, "vulkan: unsupported storage dtype (f32/f16/bf16 only in the W0 skeleton)");
  return 0;
}

// Collects the (buffer, byte-offset) pairs for a dispatch. Each Add() appends
// the SAME buffer twice — bindings 2k and 2k+1, the u32 and u16 views — and
// returns the byte offset for the push-constant block.
class Binder {
 public:
  uint32_t Add(const Tensor& t, const char* what) {
    Resolved r = Resolve(t.data, what);
    buffers_.push_back(r.buffer);
    buffers_.push_back(r.buffer);
    // f32 access indexes a uint32_t[] view, so a f32 operand's byte offset must
    // be 4-byte aligned; 16-bit access only needs 2. Tensor storage always
    // satisfies this (allocations are 64-byte aligned and views advance by whole
    // elements), but a violation would silently read shifted data.
    VT_CHECK(t.dtype != DType::kF32 || r.offset % 4 == 0,
             std::string("vulkan: ") + what + " has a byte offset that is not 4-byte aligned");
    VT_CHECK(r.offset % 2 == 0,
             std::string("vulkan: ") + what + " has an odd byte offset");
    return r.offset;
  }
  // A raw buffer bound ONCE (no 16-bit view): the fused-chain step list.
  void AddRaw(void* buffer) { buffers_.push_back(buffer); }

  // ONE MORE VIEW of an operand already added, for a shader that reads the same
  // memory at a THIRD width (vt_matmul_vec's 64-bit packed loads). Deliberately
  // separate from Add: this appends a SINGLE binding, and it must land after
  // every operand's u32/u16 pair, so the caller is choosing the descriptor index
  // rather than getting one implicitly.
  uint32_t AddAlias(const Tensor& t, const char* what) {
    Resolved r = Resolve(t.data, what);
    buffers_.push_back(r.buffer);
    return r.offset;
  }

  // A tensor bound through the uint32_t view ONLY, for shaders that declare a
  // single binding per operand because the operand is integer (embedding ids,
  // sampler token ids) or f32-by-contract (logits). Binding the unused 16-bit
  // view would need the shader to declare it too, and a descriptor a shader does
  // not declare must not be written.
  uint32_t AddU32Only(const Tensor& t, const char* what) {
    Resolved r = Resolve(t.data, what);
    buffers_.push_back(r.buffer);
    VT_CHECK(r.offset % 4 == 0,
             std::string("vulkan: ") + what + " has a byte offset that is not 4-byte aligned");
    return r.offset;
  }

  // A tensor bound through the uint32_t view ONLY, for BYTE-granular reads: an
  // i8 operand (GDN's has_initial_state) may legitimately start at any byte, so
  // AddU32Only's 4-byte assertion would reject a valid tensor. The shader
  // recovers the byte with a shift, which is exact because every buffer is bound
  // WHOLE at offset 0 and the returned offset is therefore a plain byte address
  // into it (vt_common.glsl § STORAGE MODEL). Deliberately NOT VK_KHR_8bit_storage:
  // this backend does not probe for it, and requiring it would narrow the set of
  // devices the backend registers on for the sake of one boolean array.
  uint32_t AddByteView(const Tensor& t, const char* what) {
    Resolved r = Resolve(t.data, what);
    buffers_.push_back(r.buffer);
    return r.offset;
  }

  // A tensor bound through the uint16_t view ONLY, the mirror of AddU32Only for
  // an operand every consumer reads at 16 bits: an fp16 scale vector, or the
  // EXL3 trellis, whose i8 BYTES are read as the uint32 pairs exl3_dq.cuh reads
  // and never as f32. Binding the unused 32-bit view would need the shader to
  // declare it too, and glslang strips a block nothing references — which the
  // generator then reports as a binding HOLE rather than passing silently.
  // The 2-byte assertion is Add()'s, and it is the real constraint here.
  uint32_t AddU16Only(const Tensor& t, const char* what) {
    Resolved r = Resolve(t.data, what);
    buffers_.push_back(r.buffer);
    VT_CHECK(r.offset % 2 == 0,
             std::string("vulkan: ") + what + " has an odd byte offset");
    return r.offset;
  }

  const void* const* data() const { return buffers_.data(); }
  uint32_t count() const { return static_cast<uint32_t>(buffers_.size()); }

 private:
  std::vector<const void*> buffers_;
};

// ---- Host mirrors of the shaders' push-constant blocks. Field order and types
// must match the GLSL declarations EXACTLY. GLSL `uint`/`float` are 4-byte with
// 4-byte alignment and every block below is a run of 4-byte scalars, so the std430
// push-constant layout coincides with the C++ layout with no padding surprises.
struct AddParams {
  uint32_t n, d, a_dt, b_dt, out_dt, bcast, a_off, b_off, out_off;
};
struct UnaryParams {
  uint32_t n, a_dt, out_dt, a_off, out_off;
};
// vt_cast carries its dtype pair in specialization constants instead, so its
// push block is only the shape and the two offsets.
struct CastParams {
  uint32_t n, a_off, out_off;
};
struct MatmulParams {
  uint32_t m, n, k, a_off, b_off, out_off;
};
// VK4 keep-quant: native TQ2_0 keep-quant GEMV/gemm (vt_matmul_bt_tq2.comp).
struct MatmulBTQuantTQ2Params {
  uint32_t n;       // output columns (weight rows)
  uint32_t m;       // output rows (tokens)
  uint32_t nb;      // k / 256
  uint32_t a_off, w_off, out_off;
};
struct MatmulBTQuantTQ2GroupedParams {
  uint32_t m;       // output rows (tokens)
  uint32_t n;       // output columns (weight rows)
  uint32_t nb;      // k / 256
  uint32_t e;       // num experts
  uint32_t bcast;   // 1 = all rows share activation row 0
  uint32_t a_off, w_off, eid_off, out_off;
};
// VK4 keep-quant Phase 2: fused gate+up+SwiGLU grouped MoE (TQ2_0 weights, bf16
// activation, on-device Q8_K quantize, f32 output). Push-constant block for
// vt_moe_gate_up_swiglu_grouped_tq2.comp.
struct MoeGateUpSwiGLUGroupedTQ2Params {
  uint32_t m;       // P (output rows = T * top_k)
  uint32_t n;       // I (output cols = moe_intermediate_size)
  uint32_t nb;      // K / 256
  uint32_t e;       // num experts
  uint32_t bcast;   // 1 = all rows share activation row 0
  uint32_t gather_k; // >0 = activation is [T, H], row p reads row p/gather_k
  uint32_t a_off;   // activation (f32/bf16) byte offset
  uint32_t gw_off;  // gate weight (TQ2_0) byte offset
  uint32_t uw_off;  // up weight (TQ2_0) byte offset
  uint32_t eid_off; // expert_ids (i32) byte offset
  uint32_t out_off; // output (f32) byte offset
  float limit;      // SwiGLU clamp limit
};
struct EmbeddingParams {
  uint32_t t, h, table_off, ids_off, out_off;
};
struct ArgmaxParams {
  uint32_t n, v, logits_off, out_off;
};
struct QkvSplitParams {
  uint32_t tokens, q_dim, k_dim, v_dim, src_off, q_off, k_off, v_off;
};
struct RopeFromCacheParams {
  uint32_t tokens, half_dim, rotary_dim, hq, hk;
  uint32_t q_s0, q_s1, k_s0, k_s1;
  uint32_t q_off, k_off, c_off, p_off;
};
// VK4: the rotary TABLE build (vt_rope_cos_sin_cache.comp). The angle
// construction travels as f64 scalars — GLSL `double` push members are 8-byte
// aligned, so the five uints above land on a 40-byte boundary and the doubles
// pack from offset 40 with no internal padding (std430 scalar alignment).
struct RopeCosSinCacheParams {
  uint32_t tokens, half_dim, rotary_dim;
  uint32_t pos_off, out_off;
  double base;
  double l3_sf, l3_lo, l3_hi, l3_omax;
};
// VK4: vt_rope_neox.comp — the DIRECT-ANGLE apply (no cos|sin table). Same f64
// tail as the cache builder: three uints then five uints land the doubles on
// their 8-byte boundary with no internal padding.
// VK4: vt_moe_combine.comp — the routed-expert weighted sum. All-f32 accumulate,
// one invocation per output element, shared term via spec-const gate.
struct MoeCombineParams {
  uint32_t tokens, h, k;
  float routed_scale;
  uint32_t out_off, e_off, w_off, s_off;
};

// VK4: vt_moe_router_topk.comp — softmax+topk+renorm, one workgroup per row.
struct MoeRouterTopKParams {
  uint32_t tokens, e, k;
  uint32_t logits_off, w_off, i_off;
};


struct RopeNeoxParams {
  uint32_t tokens, half_dim, rotary_dim, hq, hk;
  uint32_t q_off, k_off, p_off;
  double base;
  double l3_sf, l3_lo, l3_hi, l3_omax;
};
struct ReshapeAndCacheParams {
  uint32_t num_slots, n_elems, block_size;
  uint32_t k_blk, k_pg, v_blk, v_pg;
  uint32_t k_tok, v_tok;
  uint32_t k_off, v_off, kc_off, vc_off, sm_off;
};
struct PagedAttnParams {
  uint32_t total_q, hq, d, block_size, qpk, num_reqs;
  uint32_t causal;
  int32_t window_left, window_right;
  uint32_t kc_blk, kc_pg, kc_hd;
  uint32_t vc_blk, vc_pg, vc_hd;
  uint32_t bt_row, bt_col;
  uint32_t q_off, k_off, v_off, out_off;
  uint32_t bt_off, sl_off, qsl_off;
  float scale;
  float softcap;
};
struct SiluMulParams {
  uint32_t t, d, x_dt, out_dt, x_off, out_off;
};
struct RmsParams {
  uint32_t t, h, x_dt, w_dt, out_dt, res_dt, has_res, gemma, x_off, w_off, out_off, res_off;
  float eps;
};
struct LayerNormParams {
  uint32_t rows, d, x_dt, w_dt, b_dt, out_dt, has_w, has_b, x_off, w_off, b_off, out_off;
  float eps;
};
struct FcParams {
  uint32_t t, h, nsteps, x_dt, w_dt, res_dt, out_dt, x_off, w_off, res_off, out_off;
  float eps;
};
// --- The GDN / conv1d family (BACKEND-VULKAN-GDN). Same rule as above: field
// order and types must match the GLSL push-constant blocks EXACTLY.
struct SigmoidGateParams {
  uint32_t n, a_off, g_off, o_off;
};
struct RmsNormGatedParams {
  uint32_t rows, d, x_dt, z_dt, w_dt, out_dt, sigmoid_gate, gate_group, gate_outer;
  uint32_t x_off, z_off, w_off, out_off;
  float eps;
};
struct GdnStateGatherParams {
  uint32_t rows, work_row, work_inner, cache_inner, cache_row, n_cache_rows;
  uint32_t work_dt, cache_dt, his_mode;
  uint32_t work_off, cache_off, idx_off, his_off;
};
struct GdnStateScatterParams {
  uint32_t rows, work_row, work_inner, cache_inner, cache_row, n_cache_rows;
  uint32_t work_dt, cache_dt;
  uint32_t cache_off, work_off, idx_off;
};
struct ConvUpdateParams {
  uint32_t batch, c_dim, k, width, state_len, x_rs, n_state_rows;
  uint32_t has_bias, has_idx, silu;
  uint32_t out_dt, x_dt, w_dt, bias_dt, st_dt;
  uint32_t out_off, x_off, w_off, bias_off, st_off, idx_off;
};
struct GdnPostConvParams {
  uint32_t t, hk, dk, hv, dv, key_dim, value_dim, conv_dim, a_rs, b_rs;
  uint32_t conv_off, q_off, k_off, v_off, a_off, b_off;
  uint32_t g_off, beta_off, alog_off, dtb_off;
  float eps;
};
// The fused full-attention preamble (BACKEND-VULKAN-QKNORM).
struct AttnQkNormRopeGateParams {
  uint32_t hq, hkv, dh, rot, half_dim, qgate_rs, kf_rs, gemma;
  uint32_t qg_off, kf_off, qo_off, ko_off, go_off, qn_off, kn_off, cs_off;
  float eps;
};
// ONE block for BOTH recurrences: vt_gdn_prefill and vt_gdn_decode share their
// step body through an include, so they must also share their push layout. The
// prefill shader ignores has_idx / n_state_rows (they are the decode state-row
// indirection) rather than each op carrying a block that drifts from the other.
struct GdnRecurrenceParams {
  uint32_t hk, dk, hv, dv, nv, ratio, has_idx, n_state_rows;
  uint32_t q_off, k_off, v_off, out_off, g_off, beta_off, state_off, meta_off;
  float scale;
};

// BACKEND-VULKAN-EXL3 (#2530). vt_exl3_had.comp and vt_exl3_gemm.comp.
struct Exl3HadParams {
  uint32_t nblocks, blocks_per_row, cols, total_groups, has_pre, has_post;
  uint32_t in_off, out_off, pre_off, post_off;
  float r_scale;
};
struct Exl3GemmParams {
  uint32_t m, k, n, bits, codebook, tiles_n, raw_off, ah_off, tr_off;
};

// Vulkan only GUARANTEES 128 bytes of push-constant space (maxPushConstantsSize);
// staying inside it is what keeps this backend portable without a probe.
static_assert(sizeof(RmsParams) <= 128, "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(LayerNormParams) <= 128, "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(FcParams) <= 128, "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(PagedAttnParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(ConvUpdateParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
// The widest block in the backend at 84 bytes (tied with ConvFwdParams): the
// fused post-conv carries ten
// operand offsets. If it ever needs an eleventh, the step list has to move to the
// scratch buffer the way vt_fused_chain's does.
static_assert(sizeof(GdnPostConvParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(RmsNormGatedParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(GdnStateGatherParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(GdnRecurrenceParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(AttnQkNormRopeGateParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(Exl3HadParams) <= 128, "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(Exl3GemmParams) <= 128, "push constants must fit the guaranteed 128 bytes");

// Maximum M per dispatch chunk for dense GEMM/GEMV. The Intel Arc hangcheck is
// ~2s. Measured at 2368 tokens: one scalar GEMM takes ~1.1s. So at 4096 rows
// a single chunk takes ~1.9s — too close to the limit. 2048 rows gives ~0.95s
// per chunk, leaving comfortable margin under the 2s hangcheck.
constexpr uint32_t kMaxMPerChunkDense = 2048;

// Maximum P per dispatch chunk for MoE kernels. Each workgroup processes ALL P
// rows, so per-workgroup time scales with P. Measured: at P=9472 (1184 tokens),
// the MoE takes ~1.5s per layer. So at P=2048, each chunk takes ~0.32s per
// layer — safe under the 2s hangcheck with margin for L2 cache effects.
constexpr uint32_t kMaxPPerChunkMoe = 2048;

// Chunk a matmul dispatch over M to prevent any single command buffer from
// exceeding the GPU hangcheck timeout. Each chunk adjusts a_off and out_off
// to point at the correct rows. The shader is unaware of chunking.
// Flushes between chunks only when the accumulated GPU time would exceed ~1.5s.
// Also flushes at the start when the total estimated time exceeds 1.5s, to
// isolate the GEMM from preceding dispatches in the same command buffer.
template <typename DispatchFn>
void ChunkedMatmulDispatch(int64_t m, int64_t n, int64_t k, DType a_dt,
                           DType out_dt, uint32_t a_off, uint32_t out_off,
                           DispatchFn dispatch) {
  const size_t a_elem = vt::SizeOf(a_dt);
  const size_t o_elem = vt::SizeOf(out_dt);
  const uint32_t m_total = static_cast<uint32_t>(m);
  // Estimate GPU time per row: 2*N*K flops at ~200 GFLOP/s
  const double flops_per_row = 2.0 * static_cast<double>(n) * static_cast<double>(k);
  const double gpu_time_per_row_sec = flops_per_row / 200e9;
  const double total_est_sec = m_total * gpu_time_per_row_sec;
  // Flush when accumulated time exceeds 1.0s (conservative margin under 2s
  // hangcheck, accounting for other ops that may share the command buffer).
  constexpr double kFlushAtAccumSec = 1.0;
  // Flush at the start if this GEMM alone would exceed the threshold, to
  // isolate it from preceding dispatches (e.g. attention or previous GEMM).
  if (total_est_sec >= kFlushAtAccumSec) {
    VulkanContext::Get().FlushBatch("chunked-matmul-start");
  }
  double accum_sec = 0.0;
  for (uint32_t i_start = 0; i_start < m_total; i_start += kMaxMPerChunkDense) {
    const uint32_t m_chunk = std::min(kMaxMPerChunkDense, m_total - i_start);
    const uint32_t a_chunk = a_off + i_start * static_cast<uint32_t>(k) *
                                      static_cast<uint32_t>(a_elem);
    const uint32_t out_chunk = out_off + i_start * static_cast<uint32_t>(n) *
                                          static_cast<uint32_t>(o_elem);
    if (accum_sec >= kFlushAtAccumSec) {
      VulkanContext::Get().FlushBatch("chunk-accum");
      accum_sec = 0.0;
    }
    dispatch(m_chunk, a_chunk, out_chunk);
    accum_sec += m_chunk * gpu_time_per_row_sec;
  }
}
template <typename P>
void Go(const char* name, const Binder& b, const P& p, uint32_t groups,
        const uint32_t* spec = nullptr, uint32_t spec_count = 0) {
  VulkanContext::Get().Dispatch(name, b.data(), b.count(), &p, sizeof(P), groups, spec,
                                spec_count);
}

// ---------------------------------------------------------------------------
// Kernels. Every argument was already validated by the vt:: wrapper in
// src/vt/ops.cpp before GetOp dispatched here, so these only translate.
// ---------------------------------------------------------------------------

// cpu_layernorm.cpp:87-99 AddKernel.
void AddKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t n = a.Numel();
  const int64_t d = a.rank == 0 ? 1 : a.shape[a.rank - 1];
  const bool bcast = b.rank == 1 && a.rank != 1;
  Binder bind;
  const uint32_t a_off = bind.Add(a, "add: a");
  const uint32_t b_off = bind.Add(b, "add: b");
  const uint32_t out_off = bind.Add(out, "add: out");
  AddParams p{static_cast<uint32_t>(n), static_cast<uint32_t>(d),
              DtypeCode(a.dtype),      DtypeCode(b.dtype),
              DtypeCode(out.dtype),    bcast ? 1u : 0u,
              a_off,                   b_off,
              out_off};
  Go("vt_add", bind, p, FlatGroupCount(n));
}

// cpu_layernorm.cpp:75-85 ReluKernel.
void ReluKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t n = x.Numel();
  Binder bind;
  const uint32_t x_off = bind.Add(x, "relu: x");
  const uint32_t out_off = bind.Add(out, "relu: out");
  UnaryParams p{static_cast<uint32_t>(n), DtypeCode(x.dtype), DtypeCode(out.dtype), x_off,
                out_off};
  Go("vt_relu", bind, p, FlatGroupCount(n));
}

// cpu_ops.cpp:1436-1451 CastBf16Kernel / CastF32Kernel — one shader serves both
// (the CPU pair is likewise the same LoadF32/StoreF32 body twice).
void CastKernel(Queue&, Tensor& out, const Tensor& in) {
  const int64_t n = out.Numel();
  Binder bind;
  const uint32_t in_off = bind.Add(in, "cast: in");
  const uint32_t out_off = bind.Add(out, "cast: out");
  // The dtype pair rides SPECIALIZATION CONSTANTS rather than push constants, so
  // the per-element dtype branch is folded away at pipeline creation and each
  // (src, dst) pair is its own cached pipeline. Ascending constantID order, which
  // is what GetPipeline binds against the module's declared SpecIds.
  const uint32_t spec[2] = {DtypeCode(in.dtype), DtypeCode(out.dtype)};
  CastParams p{static_cast<uint32_t>(n), in_off, out_off};
  Go("vt_cast", bind, p, FlatGroupCount(n), spec, 2);
}

// ─── EXL3, BACKEND-VULKAN-EXL3 (#2530) ───────────────────────────────────────
//
// The fused chain, step for step as `Exl3GemmKernelCpu` runs it:
//   1. A_had = had_r_128(A, pre_scale = suh)
//   2. C_raw = A_had @ reconstruct(trellis)        [f32 accumulation]
//   3. C     = had_r_128(C_raw, post_scale = svh)
// Argument validation (shapes, dtypes, contiguity, the 16/128 multiples) is done
// once in `vt::Exl3Gemm` for every backend and is not repeated here. What IS
// checked is what selects device code: `bits` and `codebook` reach the shader as
// values and an out-of-range one would decode SILENTLY, because a shader cannot
// throw.
//
// See src/vt/vulkan/shaders/vt_exl3_gemm.comp for why the donor is the portable
// CPU reference and not cuda_exl3.cu, and .agents/specs/backend-vulkan-exl3.md
// for the byte-equality contract this arm carries instead of the CUDA arm's
// 1.0e-3 RMS bound.

// 1/sqrt(128) spelled as upstream spells it (hadamard.cu:107) and as
// cpu_exl3_kernels.cpp spells it. Kept as the literal and NEVER recomputed: a
// recomputed 1/sqrt(128) can differ in the last f32 bit, and that bit is the
// difference between this arm's byte gate passing and failing.
constexpr float kExl3InvSqrt128 = 0.088388347648f;

// The f32 [m, n] staging buffer between steps 2 and 3. The CPU arm holds it in a
// std::vector; a device needs the same bytes somewhere. GROW-ONLY and
// process-wide, which is the discipline rocm_exl3.hip's EnsureRawScratch
// documents.
//
// NAMED AS A COST rather than hidden. A fused kernel would not need it. A fused
// kernel is the later speed row; this one buys the end of the reference tier.
//
// THE FLUSH BEFORE A GROW IS LOAD-BEARING. Dispatches are BATCHED into an open
// command buffer, so the previous buffer may still be bound by work that has not
// executed. Freeing it under an open batch is a use-after-free with no error and
// no crash — the same hazard Backend::Copy's FlushIfBatchTouches exists for.
float* EnsureExl3RawScratch(size_t need) {
  static std::mutex mu;
  static void* base = nullptr;
  static size_t bytes = 0;
  std::lock_guard<std::mutex> lock(mu);
  if (need <= bytes) return static_cast<float*>(base);
  void* buffer = nullptr;
  void* memory = nullptr;
  void* fresh = VulkanContext::Get().AllocBuffer(need, &buffer, &memory);
  RegisterAllocation(fresh, need, buffer, memory);
  if (base != nullptr) {
    VulkanContext::Get().FlushBatch("exl3 raw scratch grow");
    void* old_buffer = nullptr;
    void* old_memory = nullptr;
    if (UnregisterAllocation(base, &old_buffer, &old_memory))
      VulkanContext::Get().FreeBuffer(old_buffer, old_memory);
  }
  base = fresh;
  bytes = need;
  return static_cast<float*>(base);
}

// One `had_r_128` dispatch. `pre` and `post` are optional and at most one is set,
// which is upstream's own instantiation (hadamard.cu:112-172) and what
// `vt::Exl3HadR128` checks; an absent one is bound aliasing `in`, because a
// descriptor the shader statically uses must be valid even on the path that
// never reads it.
void Exl3HadDispatch(void* out, DType out_dtype, const void* in, DType in_dtype,
                     const Tensor* pre, const Tensor* post, const Device& device, float r_scale,
                     int64_t rows, int64_t cols) {
  const int64_t blocks_per_row = cols / 128;
  const int64_t nblocks = rows * blocks_per_row;
  if (nblocks <= 0) return;
  const int64_t total_groups = (nblocks + 3) / 4;

  Tensor tin = Tensor::Contiguous(const_cast<void*>(in), in_dtype, device, {rows, cols});
  Tensor tout = Tensor::Contiguous(out, out_dtype, device, {rows, cols});
  Binder bind;
  const uint32_t in_off = bind.Add(tin, "exl3_had: in");
  const uint32_t out_off = bind.Add(tout, "exl3_had: out");
  const uint32_t pre_off =
      pre != nullptr ? bind.AddU16Only(*pre, "exl3_had: pre_scale") : bind.AddU16Only(tin, "exl3_had: in");
  const uint32_t post_off = post != nullptr ? bind.AddU16Only(*post, "exl3_had: post_scale")
                                            : bind.AddU16Only(tin, "exl3_had: in");
  const uint32_t spec[2] = {in_dtype == DType::kF16 ? 1u : 0u,
                            out_dtype == DType::kF16 ? 1u : 0u};
  Exl3HadParams p{static_cast<uint32_t>(nblocks),
                  static_cast<uint32_t>(blocks_per_row),
                  static_cast<uint32_t>(cols),
                  static_cast<uint32_t>(total_groups),
                  pre != nullptr ? 1u : 0u,
                  post != nullptr ? 1u : 0u,
                  in_off,
                  out_off,
                  pre_off,
                  post_off,
                  r_scale};
  // The shader carries a grid-stride loop, so the workgroup count is a
  // PERFORMANCE choice and never a correctness one. 65535 is the Vulkan-
  // guaranteed maxComputeWorkGroupCount[0], so this launch is valid on any
  // conformant device no matter how large the tensor is.
  const uint32_t groups = static_cast<uint32_t>(total_groups > 65535 ? 65535 : total_groups);
  Go("vt_exl3_had", bind, p, groups, spec, 2);
}

void Exl3GemmKernelVulkan(Queue& q, Tensor& c, const Tensor& a, const Tensor& trellis,
                          const Tensor& suh, const Tensor& svh, Tensor& a_had,
                          const Exl3GemmArgs& args) {
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];
  const int64_t n = c.shape[1];
  if (m == 0 || k == 0 || n == 0) return;
  VT_CHECK(args.bits >= 1 && args.bits <= 8,
           "vt vulkan exl3: bits must be in [1, 8]; got " + std::to_string(args.bits));
  VT_CHECK(args.codebook >= 0 && args.codebook <= 2,
           "vt vulkan exl3: codebook " + std::to_string(args.codebook) +
               " is not implemented (0 == 3INST, 1 == MCG, 2 == mul1). Upstream defines "
               "no other value: `decode_3inst<cb>` (codebook.cuh:56-90) has arms for 0, "
               "1 and 2 and falls off the end for anything else.");

  // 1. the input transform, into the caller's scratch (which may alias A).
  Exl3HadDispatch(a_had.data, a_had.dtype, a.data, a.dtype, &suh, nullptr, q.device,
                  kExl3InvSqrt128, m, k);

  // 2. the matmul against the decoded trellis, f32 accumulators.
  float* raw = EnsureExl3RawScratch(static_cast<size_t>(m) * static_cast<size_t>(n) *
                                    sizeof(float));
  Tensor traw = Tensor::Contiguous(raw, DType::kF32, q.device, {m, n});
  Binder bind;
  const uint32_t raw_off = bind.AddU32Only(traw, "exl3_gemm: raw");
  const uint32_t ah_off = bind.AddU16Only(a_had, "exl3_gemm: a_had");
  const uint32_t tr_off = bind.AddU16Only(trellis, "exl3_gemm: trellis");
  Exl3GemmParams p{static_cast<uint32_t>(m),
                   static_cast<uint32_t>(k),
                   static_cast<uint32_t>(n),
                   static_cast<uint32_t>(args.bits),
                   static_cast<uint32_t>(args.codebook),
                   static_cast<uint32_t>(n / 16),
                   raw_off,
                   ah_off,
                   tr_off};
  // The grid is FLATTENED because Dispatch takes group_count_x only: one
  // workgroup per (16-column output tile, 8-row block), and the shader
  // decomposes the id with the same `tiles_n` it is handed here.
  const int64_t groups = (n / 16) * ((m + 7) / 8);
  Go("vt_exl3_gemm", bind, p, static_cast<uint32_t>(groups));

  // 3. the output transform — had_ff for an f32 C, had_fh for an fp16 one, the
  // same two arms the CPU reference picks between off `c.dtype`.
  Exl3HadDispatch(c.data, c.dtype, raw, DType::kF32, nullptr, &svh, q.device, kExl3InvSqrt128, m,
                  n);
}

// cpu_ops.cpp:252-264 SiluAndMulKernel.
void SiluAndMulKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t t = x.shape[0], d = x.shape[1] / 2;
  Binder bind;
  const uint32_t x_off = bind.Add(x, "silu_and_mul: x");
  const uint32_t out_off = bind.Add(out, "silu_and_mul: out");
  SiluMulParams p{static_cast<uint32_t>(t), static_cast<uint32_t>(d), DtypeCode(x.dtype),
                  DtypeCode(out.dtype), x_off, out_off};
  Go("vt_silu_and_mul", bind, p, FlatGroupCount(t * d));
}

// WHICH RmsNorm MODULE THIS DISPATCH USES.
//
// `vt_rms_norm` is the portable 128-invocation module; `vt_rms_norm_wide` is the
// same body at 1024 invocations with a subgroup reduction. The choice is a
// DEVICE CAPABILITY question (VulkanContext::wide_reduce), not a shape question:
// the wide module is never wrong, only unavailable.
//
// VulkanContext::rms_norm_override() is the A/B lever over that decision
// (VT_VULKAN_RMSNORM=base|wide, or the setter the unit gate uses); it exists so
// the two arms are switchable inside ONE binary, because a cross-BUILD comparison
// is the shape that produced a false 1.2x reading for the subgroup tactic earlier
// in this campaign (see vulkan_context.cpp § kRingDepth).
const char* RmsNormShader() {
  const VulkanContext& ctx = VulkanContext::Get();
  const int forced = ctx.rms_norm_override();
  if (forced < 0) return "vt_rms_norm";
  if (forced > 0 || ctx.wide_reduce()) {
    VT_CHECK(ctx.wide_reduce(),
             "vulkan: the wide RmsNorm module was forced but this device does not "
             "support 1024-invocation workgroups with compute subgroup arithmetic");
    return "vt_rms_norm_wide";
  }
  return "vt_rms_norm";
}

// cpu_ops.cpp:225-250 RmsNormKernel. One workgroup per token row.
void RmsNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                   const RmsNormArgs& args, Tensor* residual) {
  const int64_t t = x.shape[0], h = x.shape[1];
  Binder bind;
  const uint32_t x_off = bind.Add(x, "rmsnorm: x");
  const uint32_t w_off = bind.Add(w, "rmsnorm: weight");
  const uint32_t out_off = bind.Add(out, "rmsnorm: out");
  // Bindings 6/7 are always written: a descriptor a shader statically uses must
  // be valid even on the code path that never reads it. With has_res == 0 they
  // alias `out` and are dead.
  const uint32_t res_off =
      residual != nullptr ? bind.Add(*residual, "rmsnorm: residual") : bind.Add(out, "rmsnorm: out");
  RmsParams p{static_cast<uint32_t>(t),
              static_cast<uint32_t>(h),
              DtypeCode(x.dtype),
              DtypeCode(w.dtype),
              DtypeCode(out.dtype),
              residual != nullptr ? DtypeCode(residual->dtype) : 0u,
              residual != nullptr ? 1u : 0u,
              args.gemma ? 1u : 0u,
              x_off,
              w_off,
              out_off,
              res_off,
              args.eps};
  Go(RmsNormShader(), bind, p, static_cast<uint32_t>(t));
}

// cpu_layernorm.cpp:49-73 LayerNormKernel.
void LayerNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor* weight,
                     const Tensor* bias, const LayerNormArgs& args) {
  const int64_t d = x.shape[x.rank - 1];
  const int64_t rows = d == 0 ? 0 : x.Numel() / d;
  Binder bind;
  const uint32_t x_off = bind.Add(x, "layer_norm: x");
  const uint32_t w_off =
      weight != nullptr ? bind.Add(*weight, "layer_norm: weight") : bind.Add(x, "layer_norm: x");
  const uint32_t b_off =
      bias != nullptr ? bind.Add(*bias, "layer_norm: bias") : bind.Add(x, "layer_norm: x");
  const uint32_t out_off = bind.Add(out, "layer_norm: out");
  LayerNormParams p{static_cast<uint32_t>(rows),
                    static_cast<uint32_t>(d),
                    DtypeCode(x.dtype),
                    weight != nullptr ? DtypeCode(weight->dtype) : 0u,
                    bias != nullptr ? DtypeCode(bias->dtype) : 0u,
                    DtypeCode(out.dtype),
                    weight != nullptr ? 1u : 0u,
                    bias != nullptr ? 1u : 0u,
                    x_off,
                    w_off,
                    b_off,
                    out_off,
                    args.eps};
  Go("vt_layer_norm", bind, p, static_cast<uint32_t>(rows));
}

// cpu_ops.cpp:1649-1702 FusedChainInterpKernel — the Tier-1 interpreter. ONE
// registration; every Tier-1-able recipe in include/vt/recipes.h realizes
// through it, and every non-Tier-1 recipe realizes through the device-agnostic
// Tier-0 composite in src/vt/ops.cpp, which re-enters this backend's standalone
// ops. That is the whole "2 lines -> all 10 recipes" property the spike claims.
void FusedChainKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& weight,
                      Tensor* residual, const FusedRecipe& r, float eps) {
  const int64_t t = x.shape[0], h = x.shape[1];
  VT_CHECK(r.n >= 1 && r.n <= kMaxFusedSteps, "vulkan fused_chain: bad step count");

  // Words per step, matching VT_STEP_WORDS in vt_fused_chain.comp.
  constexpr uint32_t kStepWords = 5;
  std::vector<uint32_t> steps(static_cast<size_t>(r.n) * kStepWords, 0u);
  for (int s = 0; s < r.n; ++s) {
    const FStep& st = r.steps[s];
    uint32_t op = 0;
    switch (st.op) {
      case FOp::kAdd: op = 0; break;
      case FOp::kMul: op = 1; break;
      case FOp::kSilu: op = 2; break;
      case FOp::kSigmoid: op = 3; break;
      case FOp::kRmsNorm:
        // Mirrors the CPU interpreter's assertion (cpu_ops.cpp:1674): the shader
        // hard-codes the mean-square reduction, so any other kind must not reach it.
        VT_CHECK(st.reduce == FReduce::kMeanSquare,
                 "vulkan fused_chain: rmsnorm needs kMeanSquare");
        op = 4;
        break;
      default:
        VT_CHECK(false, "vulkan fused_chain: non-Tier-1 opcode reached the interpreter");
    }
    // Canonical operand indices (cpu_ops.cpp:1621-1643): 0=x 1=weight 2=residual
    // 3=out, with 2 and 3 the only writable slots.
    VT_CHECK(st.out == 2 || st.out == 3, "vulkan fused_chain: step writes a read-only operand");
    VT_CHECK(st.in[0] <= 3 && st.in[1] <= 3, "vulkan fused_chain: operand index out of range");
    VT_CHECK(residual != nullptr || (st.out != 2 && st.in[0] != 2 && st.in[1] != 2),
             "vulkan fused_chain: recipe touches the residual slot but none was bound");
    const size_t base = static_cast<size_t>(s) * kStepWords;
    steps[base + 0] = op;
    steps[base + 1] = st.out;
    steps[base + 2] = st.in[0];
    steps[base + 3] = st.in[1];
    steps[base + 4] = st.gemma ? 1u : 0u;
  }

  VulkanContext& ctx = VulkanContext::Get();
  const size_t step_bytes = steps.size() * sizeof(uint32_t);
  VT_CHECK(step_bytes <= VulkanContext::kScratchBytes,
           "vulkan fused_chain: step list exceeds the scratch buffer");
  std::memcpy(ctx.ScratchData(), steps.data(), step_bytes);

  Binder bind;
  const uint32_t x_off = bind.Add(x, "fused_chain: x");
  const uint32_t w_off = bind.Add(weight, "fused_chain: weight");
  const uint32_t res_off = residual != nullptr ? bind.Add(*residual, "fused_chain: residual")
                                               : bind.Add(out, "fused_chain: out");
  const uint32_t out_off = bind.Add(out, "fused_chain: out");
  bind.AddRaw(ctx.ScratchBuffer());
  FcParams p{static_cast<uint32_t>(t),
             static_cast<uint32_t>(h),
             static_cast<uint32_t>(r.n),
             DtypeCode(x.dtype),
             DtypeCode(weight.dtype),
             residual != nullptr ? DtypeCode(residual->dtype) : 0u,
             DtypeCode(out.dtype),
             x_off,
             w_off,
             res_off,
             out_off,
             eps};
  Go("vt_fused_chain", bind, p, static_cast<uint32_t>(t));
}

// cpu_ops.cpp:187-260 MatmulChunked / MatmulKernel / MatmulBTKernel. One
// invocation per OUTPUT ELEMENT with the whole K reduction on it, which is what
// the CPU kernel does too (it deliberately never splits a K reduction across
// threads), so the accumulation ORDER matches rather than merely the tolerance.
//
// The naive body is the portable correctness tier on purpose; the tiled and
// cooperative-matrix ports (llama.cpp mul_mm.comp / mul_mm_cm2.comp) are VK-C,
// which needs exactly this as its same-device A/B reference.
// TACTIC SELECTION (VK-C). Every condition below is a HARD requirement of the
// cooperative-matrix path, not a heuristic, and failing any one of them means the
// scalar kernel -- which is always correct -- runs instead:
//
//   * the device reports the EXACT configuration the committed coopmat SPIR-V is
//     written to (16x16x16, bf16/bf16/f32/f32, SUBGROUP). Vulkan matches
//     configurations exactly, so "close enough" does not exist;
//   * subgroup size is 32, because the shader's workgroup is a literal 32 (see
//     the shader for why the size cannot travel as a specialization constant at
//     this target);
//   * BOTH operands are bf16. Every configuration GB10 reports takes
//     bf16/f16/int8 inputs, so f32 operands can never use this path -- a hardware
//     constraint, not a policy;
//   * K is a multiple of 16. A ragged K tail cannot be masked inside a
//     cooperative-matrix load, and silently truncating it would drop terms from
//     the dot product. Ragged M and N are fine: the shader bounds-checks its
//     store.
//
// MEASURED: GB10 satisfies all four; llvmpipe -- the only Vulkan device CI can
// reach -- fails the first, so CI exercises the scalar tactic and this selection
// returning false is the property CI can actually gate.
bool CoopMatMatmulUsable(const Tensor& a, const Tensor& b, int64_t k, int64_t m, int64_t n) {
  // VT_VULKAN_COOPMAT=0 forces the scalar tactic. This exists for ONE reason: a
  // same-binary A/B. Comparing the two tactics across two builds would confound
  // the kernel with everything else that differs between them, and the project's
  // benchmark protocol wants the arms to differ in exactly one thing. Default is
  // ON -- absent or any value other than "0" leaves selection to the capability
  // probe, so production behaviour is unchanged by the lever's existence.
  static const bool kDisabled = [] {
    const char* v = std::getenv("VT_VULKAN_COOPMAT");
    return v != nullptr && std::strcmp(v, "0") == 0;
  }();
  if (kDisabled) return false;

  const VulkanContext& ctx = VulkanContext::Get();

  // WHY IT DECLINED, reported once per distinct reason under
  // VT_VULKAN_DISPATCH_STATS. A 27B prefill measured 99.9% of GPU time in the
  // UNTILED SCALAR kernel at ~96 GFLOP/s -- roughly 1% of what this device can do
  // -- because this predicate was returning false for every GEMM, and reading the
  // source did not reveal which clause. Shapes were whole tiles and activations
  // were bf16, so the obvious two candidates were both excluded by inspection and
  // the answer still had to be measured. A selection predicate that can silently
  // route an entire model onto the correctness tier should be able to say so.
  if (kCoopMatWhy) {
    const char* why = nullptr;
    if (!ctx.coopmat_bf16_f32()) why = "device reports no bf16->f32 16x16x16 SUBGROUP config";
    else if (ctx.subgroup_size() != 32) why = "subgroup size is not 32";
    else if (a.dtype != DType::kBF16) why = "operand a is not bf16";
    else if (b.dtype != DType::kBF16) why = "operand b is not bf16";
    else if (k % 16 != 0) why = "K is not a multiple of 16";
    else if (m < 16) why = "M is below one 16-row tile";
    else if (n % 16 != 0) why = "N is not a multiple of 16";
    if (why != nullptr) {
      static std::mutex seen_mu;
      static std::set<std::string> seen;
      std::string key = std::string(why) + "|" + std::to_string(static_cast<int>(a.dtype)) +
                        "," + std::to_string(static_cast<int>(b.dtype));
      std::lock_guard<std::mutex> g(seen_mu);
      if (seen.insert(key).second) {
        std::fprintf(stderr,
                     "[vt vulkan] coopmat DECLINED: %s  (a.dtype=%d b.dtype=%d "
                     "m=%lld k=%lld n=%lld)\n",
                     why, static_cast<int>(a.dtype), static_cast<int>(b.dtype),
                     (long long)m, (long long)k, (long long)n);
        std::fflush(stderr);
      }
    }
  }

  return ctx.coopmat_bf16_f32() && ctx.subgroup_size() == 32 &&
         a.dtype == DType::kBF16 && b.dtype == DType::kBF16 && k % 16 == 0 &&
         // M AND N MUST ALSO BE WHOLE TILES. `coopMatLoad` reads a FULL 16x16
         // tile with no masking, so a partial tile reads past the end of the
         // operand -- and the store being bounds-checked does not save it,
         // because the fault happens on the LOAD. MEASURED: lm_head at M=1
         // (single decode token) read 15 rows (~30 KB) past a small activation
         // buffer, faulted the GPU, and the fence NEVER SIGNALLED -- an infinite
         // vkWaitForFences, which presents as a hang, not as an error.
         //
         // The original correctness gate used M=20, N=12 precisely to exercise
         // ragged shapes and PASSED, because there the out-of-bounds read stayed
         // inside the allocation and its garbage rows were discarded by the
         // bounds-checked store. Raggedness alone was not enough; the read has to
         // leave the allocation to fault.
         //
         // M NEED ONLY BE AT LEAST ONE WHOLE TILE, not a multiple of one. The
         // shader slides a trailing tile back to start at M-16, so every read
         // stays in bounds and the shared rows recompute to identical values.
         //
         // Requiring m % 16 == 0 here is what fixed the original hang, and it
         // MEASURED as the entire prefill bottleneck: prompt length gives
         // m = tokens + 1, so 513 % 16 == 1 sent every 27B prefill GEMM to the
         // untiled scalar kernel -- 99.9% of GPU time at ~96 GFLOP/s, about 1% of
         // this device. N stays whole because a ragged N would need the same
         // treatment on the B operand and no shape in play needs it.
         m >= 16 && n % 16 == 0;
}

// GEMV TACTIC SELECTION (VK-F). Same shape of contract as the coopmat predicate
// above -- every requirement is a hard one, and failing any of them runs the
// always-correct scalar kernel instead.
//
// The problem this solves is COALESCING, measured: vt_matmul was ~55% of all GPU
// time in an e2e decode run. It puts one invocation on each output element and
// loops K there, so for MatmulBT lane j reads b[j*k + q] and adjacent lanes land
// k*2 bytes apart -- each pulling its own cache line to use 2 bytes of it. The
// GEMV shader instead gives each output element a workgroup whose lanes stride K,
// so adjacent lanes read adjacent addresses.
//
//   * MatmulBT ONLY. In the other orientation vt_matmul reads b[q*n + j], which
//     is ALREADY coalesced across lanes; the GEMV shape would make that strided
//     and strictly worse. This is not a universally better kernel and the
//     predicate does not pretend otherwise.
//   * m == 1, the decode shape. One workgroup per output element is the right
//     trade only when there are few of them: at prefill m*n workgroups would each
//     do k/128 multiplies, and prefill is the coopmat tactic's job anyway.
//   * k >= the workgroup width, so the strided loop actually has work for every
//     lane. Below that most lanes contribute a zero partial and the reduction
//     costs more than the loop saves.
//
// ACCUMULATION ORDER: the K reduction becomes a tree, so this tactic does NOT
// share the CPU's accumulation order -- it sits in the NMSE tier alongside
// coopmat. That is why it is gated on a token-exactness run and not on an NMSE
// bound alone.
bool GemvMatmulUsable(bool bt, int64_t k, int64_t m) {
  // VT_VULKAN_GEMV=0 forces the scalar tactic, for the same single reason the
  // coopmat lever exists: a same-binary A/B, so the arms differ in exactly one
  // thing. Default ON.
  static const bool kDisabled = [] {
    const char* v = std::getenv("VT_VULKAN_GEMV");
    return v != nullptr && std::strcmp(v, "0") == 0;
  }();
  if (kDisabled) return false;

  // WHY IT DECLINED, once per distinct reason, under VT_VULKAN_DISPATCH_STATS.
  // Same reasoning as the coopmat predicate above: a 27B decode profile showed the
  // UNTILED SCALAR kernel still taking 256 calls at 12.53 ms -- one per output
  // token, and the largest single per-call cost in decode -- and no amount of
  // reading the source says WHICH clause sent it there.
  if (kCoopMatWhy) {
    const char* why = nullptr;
    if (!bt) why = "not MatmulBT (b is [K,N]; that layout is already coalesced)";
    // M>1 now allowed for prefill — GEMV coalesces B reads
    else if (k < static_cast<int64_t>(kWorkgroupSize)) why = "K is below one workgroup width";
    if (why != nullptr) {
      static std::mutex gseen_mu;
      static std::set<std::string> gseen;
      std::lock_guard<std::mutex> g(gseen_mu);
      if (gseen.insert(std::string(why)).second) {
        std::fprintf(stderr, "[vt vulkan] gemv DECLINED: %s  (bt=%d m=%lld k=%lld)\n",
                     why, bt ? 1 : 0, (long long)m, (long long)k);
        std::fflush(stderr);
      }
    }
  }

  if (!bt) return false;  // MatmulBT only — M>1 now allowed for prefill
  return k >= static_cast<int64_t>(kWorkgroupSize);
}

// GEMV VARIANT AXES (row BACKEND-VULKAN-GEMVROWS). Both are specialization
// constants on the SAME committed module, so every arm of an A/B lives in one
// binary; see shaders/vt_matmul_vec.comp for what each one can and cannot buy.
//
// Defaults are named here rather than spelled inline so a measured flip is one
// edit and the env lever keeps meaning "override the measured default".
//
// MEASURED, GB10, benchmarks/vulkan_gemv_ab.cpp over the 27B's own decode shapes,
// 9 arms x 4 rotated passes, each arm paired against the rows=1/pack=0 baseline
// measured IN THE SAME PASS (the box drifted 15.5% peak-to-peak between passes,
// so an unpaired ranking would have been noise):
//
//   rows=1 pack=0  1.000x     rows=2 pack=0  0.966x     rows=4 pack=0  0.968x
//   rows=1 pack=1  1.025x     rows=2 pack=1  1.017x     rows=4 pack=1  1.014x
//   rows=1 pack=2  1.086x     rows=2 pack=2  1.047x     rows=4 pack=2  1.039x
//
// ROWS > 1 IS A MEASURED LOSS ON THIS DEVICE, in all four passes and at every
// pack width, so it ships OFF. It is kept as an axis rather than deleted because
// llama.cpp makes exactly this knob device-dependent -- ggml-vulkan.cpp:4705-4719
// sets `rm_stdq` to 2 on AMD GCN and on Intel and leaves it 1 elsewhere -- and the
// board this campaign is waiting on (VK-I) is one of the two it raises it for.
// Deleting the axis would mean rediscovering it there.
// The likely reason it loses here: halving the workgroup count halves the number
// of independent sequential read streams the memory controller sees, and each
// surviving workgroup interleaves reads from rows k*2 bytes apart. The activation
// re-reads it saves were L2 hits, which were never the constraint.

// BISECT HOOK (temporary): VT_VK_DISABLE=kMoeRouterTopK,kMoeCombine,... skips
// registering the named ops so they fall back to the reference tier.
bool VkOpDisabled(const char* op_name) {
  const char* v = std::getenv("VT_VK_DISABLE");
  if (v == nullptr || *v == '\0') return false;
  const std::string s(v);
  size_t pos = 0;
  while (pos <= s.size()) {
    size_t comma = s.find(',', pos);
    std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    if (tok == op_name) return true;
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return false;
}

constexpr uint32_t kGemvRowsDefault = 1;
constexpr uint32_t kGemvPackDefault = 2;

uint32_t EnvVariant(const char* name, uint32_t fallback, uint32_t lo, uint32_t hi) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(v, &end, 10);
  if (end == v || *end != '\0') return fallback;
  const uint32_t val = static_cast<uint32_t>(parsed);
  return (val >= lo && val <= hi) ? val : fallback;
}

// Once per distinct reason, exactly like the two DECLINED reporters above: a
// variant that silently stops applying is the failure mode this whole row has to
// be able to rule out, and reading the source cannot tell you which clause fired.
void VariantWhy(const char* kind, const char* why) {
  if (!kCoopMatWhy) return;
  static std::mutex mu;
  static std::set<std::string> seen;
  std::lock_guard<std::mutex> g(mu);
  if (seen.insert(std::string(kind) + "|" + why).second) {
    std::fprintf(stderr, "[vt vulkan] gemv %s DECLINED: %s\n", kind, why);
    std::fflush(stderr);
  }
}

// OUTPUT ELEMENTS PER WORKGROUP. Requires m == 1 (so the block cannot straddle
// two output rows) and n % rows == 0 (so the shader has no ragged tail to branch
// on in its inner loop). Both hold for every decode projection in the models this
// backend runs; when they do not, the request degrades to the next lower power of
// two rather than failing, and says so under VT_VULKAN_DISPATCH_STATS.
uint32_t GemvRows(int64_t m, int64_t n) {
  // ONLY 1, 2 AND 4 EXIST, and that is a correctness constraint rather than a
  // taste one: the shader materialises exactly four accumulator sets and guards
  // them with `VT_MM_ROWS >= 2` / `>= 4`, so a value of 3 would compute two
  // output elements while the host dispatched ceil(n/3) workgroups and silently
  // drop a third of the row. Anything else falls back to 1.
  static const uint32_t kWant = [] {
    const uint32_t v = EnvVariant("VT_VULKAN_GEMV_ROWS", kGemvRowsDefault, 1, 4);
    return (v == 1u || v == 2u || v == 4u) ? v : 1u;
  }();
  if (m != 1) return 1u;  // M>1 (prefill): rows=1 only — rows>1 was measured as a loss
  uint32_t rows = kWant;
  while (rows > 1u && (n % static_cast<int64_t>(rows)) != 0) rows >>= 1;
  if (rows < kWant) {
    VariantWhy("rows", "N is not a multiple of the requested row count");
  }
  return rows;
}

// LOAD WIDTH for 16-bit operands: 0 = one element per load, 1 = two through the
// 32-bit view, 2 = four through the 64-bit view. Every requirement is a hard one
// and a failure DEGRADES to the next narrower width rather than declining, so a
// shape that cannot take the widest load still takes the one it can.
//   * both operands 16-bit -- an f32 operand is already one element per 32-bit
//     word, so there is nothing to pack.
//   * byte offsets aligned to the load width -- the wide views index at
//     (off >> 2) and (off >> 3), so a misaligned offset would read a word
//     straddling two elements.
//   * k divisible by the element count -- the shader's pair/quad count is exact
//     with no ragged element left over, and a row start j*k must itself be
//     aligned for every j, which k carries.
uint32_t GemvPack(const Tensor& a, const Tensor& b, int64_t k, uint32_t a_off,
                  uint32_t b_off) {
  static const uint32_t kWant = EnvVariant("VT_VULKAN_GEMV_PACK", kGemvPackDefault, 0, 2);
  if (kWant == 0) return 0;
  const bool a16 = a.dtype == DType::kF16 || a.dtype == DType::kBF16;
  const bool b16 = b.dtype == DType::kF16 || b.dtype == DType::kBF16;
  if (!a16 || !b16) {
    VariantWhy("pack", "an operand is not 16-bit (nothing to pack)");
    return 0;
  }
  uint32_t level = kWant;
  if (level >= 2 &&
      ((a_off % 8u) != 0u || (b_off % 8u) != 0u || (k % 4) != 0)) {
    VariantWhy("pack", "K or an operand byte offset is not 4-element aligned (8 B)");
    level = 1;
  }
  if (level >= 1 &&
      ((a_off % 4u) != 0u || (b_off % 4u) != 0u || (k % 2) != 0)) {
    VariantWhy("pack", "K or an operand byte offset is not 2-element aligned (4 B)");
    level = 0;
  }
  return level;
}

template <bool kBT>
void MatmulGeneric(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t m = a.shape[0], k = a.shape[1];
  const int64_t n = kBT ? b.shape[0] : b.shape[1];
  if (m == 0 || n == 0) return;
  Binder bind;
  const uint32_t a_off = bind.Add(a, "matmul: a");
  const uint32_t b_off = bind.Add(b, "matmul: b");
  const uint32_t out_off = bind.Add(out, "matmul: out");

  if (CoopMatMatmulUsable(a, b, k, m, n)) {
    // One workgroup (= one subgroup) per 16x16 OUTPUT TILE. Deliberately not
    // FlatGroupCount, which divides an element count by the workgroup size: here
    // the whole subgroup cooperates on one tile.
    const uint32_t tiles =
        static_cast<uint32_t>(((m + 15) / 16) * ((n + 15) / 16));
    const uint32_t spec[2] = {kBT ? 1u : 0u, DtypeCode(out.dtype)};
    MatmulParams p{static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                   static_cast<uint32_t>(k), a_off, b_off, out_off};
    Go("vt_matmul_coopmat", bind, p, tiles, spec, 2);
    return;
  }

  // Tiled GEMM for prefill (M >= 16): shared memory blocking reuses B tiles
  // across 16 output rows, reducing B reads by 16x vs the scalar kernel.
  // Selected for both orientations when A and B are bf16 or f32 (not quantized).
  // Not bit-exact with the scalar kernel (different accumulation order), but
  // within f32 round-off tolerance.
  if (m >= 16 &&
      (a.dtype == DType::kBF16 || a.dtype == DType::kF32 || a.dtype == DType::kF16) &&
      (b.dtype == DType::kBF16 || b.dtype == DType::kF32 || b.dtype == DType::kF16) &&
      (out.dtype == DType::kBF16 || out.dtype == DType::kF32 || out.dtype == DType::kF16)) {
    const uint32_t spec[4] = {DtypeCode(a.dtype), DtypeCode(b.dtype),
                              DtypeCode(out.dtype), kBT ? 1u : 0u};
    // BM=16, BN=64: one workgroup per [16, 64] output tile.
    const uint32_t BM = 16, BN = 64;
    // Chunk over M to prevent hangcheck timeouts at large prefill sizes.
    ChunkedMatmulDispatch(
        m, n, k, a.dtype, out.dtype, a_off, out_off,
        [&](uint32_t m_chunk, uint32_t a_chunk, uint32_t out_chunk) {
          MatmulParams p{m_chunk, static_cast<uint32_t>(n),
                         static_cast<uint32_t>(k), a_chunk, b_off, out_chunk};
          const uint32_t chunk_tiles = static_cast<uint32_t>(
              ((m_chunk + BM - 1) / BM) * ((n + BN - 1) / BN));
          Go("vt_matmul_tiled", bind, p, chunk_tiles, spec, 4);
        });
    return;
  }

  if (GemvMatmulUsable(kBT, k, m)) {
    // ONE WORKGROUP PER `rows` OUTPUT ELEMENTS -- not FlatGroupCount, which would
    // divide the element count by the workgroup size and put the whole K
    // reduction back on a single lane. The workgroup cooperates on its elements.
    const uint32_t rows = GemvRows(m, n);
    // VT_VULKAN_GEMV_UNROLL=1 forces the un-unrolled body, for the same-binary A/B.
    static const uint32_t kUnroll = [] {
      const char* v = std::getenv("VT_VULKAN_GEMV_UNROLL");
      return (v != nullptr && std::strcmp(v, "1") == 0) ? 1u : 4u;
    }();
    const uint32_t pack = GemvPack(a, b, k, a_off, b_off);
    const uint32_t spec[6] = {DtypeCode(a.dtype), DtypeCode(b.dtype),
                              DtypeCode(out.dtype), kUnroll, rows, pack};
    // TWO EXTRA BINDINGS, and they are the same two buffers again. This shader
    // declares a 64-bit view of each input operand for the widest packed load;
    // a shader must declare every descriptor the host writes, so both views are
    // bound for every variant and the narrower ones simply never read them. The
    // Binder pushed a and b at bindings 0-3 already, so the aliases have to be
    // appended AFTER the output pair to land on 6 and 7.
    Binder wide = bind;
    wide.AddAlias(a, "matmul: a (64-bit view)");
    wide.AddAlias(b, "matmul: b (64-bit view)");
    // Chunk over M to prevent hangcheck timeouts at large prefill sizes.
    ChunkedMatmulDispatch(
        m, n, k, a.dtype, out.dtype, a_off, out_off,
        [&](uint32_t m_chunk, uint32_t a_chunk, uint32_t out_chunk) {
          MatmulParams p{m_chunk, static_cast<uint32_t>(n),
                         static_cast<uint32_t>(k), a_chunk, b_off, out_chunk};
          const uint64_t groups = (static_cast<uint64_t>(m_chunk) * n + rows - 1) / rows;
          // MSVC C4244: the ceil-div is 64-bit on purpose; Go takes uint32_t.
          Go("vt_matmul_vec", wide, p, static_cast<uint32_t>(groups), spec, 6);
        });
    return;
  }

  // Scalar tactic: the portable reference, and the only one whose accumulation
  // ORDER matches the CPU kernel's.
  //
  // COLUMN BLOCKING IS GATED TO bt == 0, and that gate is structural, not a
  // heuristic. It exists to give a workgroup a CONTIGUOUS run of b, and b is only
  // contiguous along the output columns in the [K,N] orientation. In [N,K]
  // contiguity runs along K instead -- which is exactly why MatmulBT at M=1 has
  // its own kernel -- so blocking columns there would stride every load and make
  // this strictly worse.
  //
  // It is NOT gated on m. Each of a lane's accumulators owns one output element
  // and runs the whole K reduction sequentially, so the result is bit-identical to
  // the flat body for every shape; there is no numeric tier to protect by
  // restricting it to the decode shape. Correctness of that claim is gated by a
  // bitwise memcmp of the two arms, not by a tolerance.
  const uint32_t ncols = kBT ? 1u : MatmulColumnsPerLane();

  // WHICH ARM RAN, once per distinct orientation, under VT_VULKAN_DISPATCH_STATS.
  // Column blocking is a PERFORMANCE axis: every arm is bit-identical, so a run's
  // OUTPUT can never say which one served it, and the per-shader histogram only
  // reports the module name -- every arm is `vt_matmul`. A measurement block on
  // this box produced one 215 ms/call leg against three at 12.4 across two runs of
  // each arm; with no marker in the log there was no way to distinguish an arm
  // that had not taken the flag from a run that hit the GB10 residency lottery
  // (replication showed it was the lottery). An A/B whose arms are not
  // self-identifying is not an A/B.
  if (kCoopMatWhy) {
    static std::mutex sseen_mu;
    static std::set<std::string> sseen;
    const std::string key = std::to_string(kBT ? 1 : 0) + "|" + std::to_string(ncols);
    std::lock_guard<std::mutex> g(sseen_mu);
    if (sseen.insert(key).second) {
      std::fprintf(stderr,
                   "[vt vulkan] scalar matmul ARM: bt=%d ncols=%u (first shape m=%lld k=%lld "
                   "n=%lld)\n",
                   kBT ? 1 : 0, ncols, (long long)m, (long long)k, (long long)n);
      std::fflush(stderr);
    }
  }

  // Ascending constantID order: a dtype, b dtype, out dtype, orientation, ncols.
  const uint32_t spec[5] = {DtypeCode(a.dtype), DtypeCode(b.dtype), DtypeCode(out.dtype),
                            kBT ? 1u : 0u, ncols};
  // Chunk over M to prevent hangcheck timeouts at large prefill sizes.
  const int64_t span = static_cast<int64_t>(kWorkgroupSize) * ncols;
  ChunkedMatmulDispatch(
      m, n, k, a.dtype, out.dtype, a_off, out_off,
      [&](uint32_t m_chunk, uint32_t a_chunk, uint32_t out_chunk) {
        MatmulParams p{m_chunk, static_cast<uint32_t>(n),
                       static_cast<uint32_t>(k), a_chunk, b_off, out_chunk};
        const uint32_t groups =
            ncols == 1u ? FlatGroupCount(static_cast<int64_t>(m_chunk) * n)
                        : static_cast<uint32_t>(
                              static_cast<int64_t>(m_chunk) *
                              ((n + span - 1) / span));
        Go("vt_matmul", bind, p, groups, spec, 5);
      });
}

// cpu_ops.cpp:661-672 EmbeddingKernel. One output ELEMENT per invocation.
// The id dtype (i32 vs i64) is a specialization constant rather than a
// per-element branch; see the shader for why only the low 32 bits are read.
void EmbeddingKernel(Queue&, Tensor& out, const Tensor& table, const Tensor& ids) {
  const int64_t t = ids.shape[0], h = table.shape[1];
  if (t == 0 || h == 0) return;
  VT_CHECK(ids.dtype == DType::kI32 || ids.dtype == DType::kI64,
           "vulkan embedding: ids must be i32 or i64");
  Binder bind;
  const uint32_t table_off = bind.Add(table, "embedding: table");
  const uint32_t ids_off = bind.Add(ids, "embedding: ids");
  const uint32_t out_off = bind.Add(out, "embedding: out");
  const uint32_t spec[3] = {DtypeCode(table.dtype), DtypeCode(out.dtype),
                            ids.dtype == DType::kI64 ? 1u : 0u};
  EmbeddingParams p{static_cast<uint32_t>(t), static_cast<uint32_t>(h), table_off, ids_off,
                    out_off};
  Go("vt_embedding", bind, p, FlatGroupCount(t * h), spec, 3);
}

// cpu_sample.cpp:40-56 GreedyArgmaxKernel. ONE INVOCATION PER ROW, because the
// tie-break (strict `>`, so the first maximum wins) is part of the token-exact
// contract and a tree reduction would have to carry the index and break ties
// toward the lower one at every merge. Rows are few at decode; the vocabulary
// scan is the slow axis and is deliberately left for a later change.
void GreedyArgmaxKernel(Queue&, Tensor& token_ids, const Tensor& logits) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  VT_CHECK(logits.dtype == DType::kF32, "vulkan greedy argmax: logits must be f32");
  VT_CHECK(token_ids.dtype == DType::kI64, "vulkan greedy argmax: token_ids must be i64");
  Binder bind;
  const uint32_t logits_off = bind.AddU32Only(logits, "argmax: logits");
  const uint32_t out_off = bind.AddU32Only(token_ids, "argmax: token_ids");
  ArgmaxParams p{static_cast<uint32_t>(n), static_cast<uint32_t>(v), logits_off, out_off};
  // ONE WORKGROUP PER ROW, matching vt_rms_norm's convention -- the shader
  // tree-reduces the vocabulary across the workgroup's lanes. NOT
  // FlatGroupCount(n), which would allot one INVOCATION per row and leave the
  // vocabulary scan serial; at decode n is 1, so that dispatched a single lane
  // and measured 10.03 ms per call.
  Go("vt_greedy_argmax", bind, p, static_cast<uint32_t>(n));
}

// cpu_paged_attn.cpp:52-171 PagedAttentionKernel. ONE WORKGROUP per (query
// token, query head), lanes splitting the head dimension; see the shader for why
// the CPU's three passes become one online-softmax recurrence (its `probs` array
// is one float per key in the window, which a shader cannot allocate).
//
// This is the only kernel in the backend with NO llama.cpp counterpart to port
// from: its Vulkan backend has no paged KV anywhere. The block-table indirection
// and windowing come from the CPU kernel above, the online-softmax skeleton from
// flash_attn.comp's shape.
void PagedAttentionKernel(Queue& q, Tensor& out, const Tensor& query, const Tensor& k_cache,
                          const Tensor& v_cache, const Tensor& block_table,
                          const Tensor& seq_lens, const Tensor& query_start_loc,
                          const PagedAttentionArgs& args) {
  const int64_t num_reqs = seq_lens.shape[0];
  const int64_t total_q = query.shape[0];
  const int64_t hq = query.shape[1], d = query.shape[2];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];

  // PER-CALL REFUSAL, not a silent regression. An fp8 KV cache stores 1-byte
  // pages that must be dequantised as Dequant(fp8) * k_scale|v_scale before the
  // f32 softmax (cpu_paged_attn.cpp:79-93). This shader reads f32/f16/bf16 only,
  // so rather than throw -- which would REMOVE a capability the portable
  // reference tier already provides -- it declines through the provider seam and
  // forwards to the next provider down, which is exactly what GetOpFallback is
  // for (op_provider.h:94-100: per-call refusal belongs in the kernel, because
  // GetOp has no shape or dtype to inspect).
  // BISECT: VT_VK_DISABLE_PAGED_ATTN=1 forces every call down to the portable tier.
  static const bool kForcePaFallback = [] {
    const char* e = std::getenv("VT_VK_DISABLE_PAGED_ATTN");
    return e != nullptr && e[0] != 0 && std::strcmp(e, "0") != 0;
  }();
  if (kForcePaFallback || args.kv_cache_dtype != vt::Fp8KVCacheDataType::kAuto) {
    auto next = reinterpret_cast<PagedAttentionFn>(
        GetOpFallback(OpId::kPagedAttention, DeviceType::kVULKAN, kNativeProviderName));
    next(q, out, query, k_cache, v_cache, block_table, seq_lens, query_start_loc, args);
    return;
  }

  if (total_q == 0 || hq == 0 || d == 0) return;
  // The shader keeps its accumulator in VT_PA_ACC_MAX slots per lane, one per
  // head-dim element the lane owns. Asserted rather than trusted: overflowing it
  // would write past a local array.
  VT_CHECK(d <= 8 * static_cast<int64_t>(kWorkgroupSize),
           "vulkan paged attention: head dim " + std::to_string(d) +
               " exceeds the per-lane accumulator (8 * workgroup)");
  VT_CHECK(num_kv_heads > 0 && hq % num_kv_heads == 0,
           "vulkan paged attention: query heads must be a multiple of kv heads");

  Binder bind;
  const uint32_t q_off = bind.Add(query, "paged_attn: query");
  const uint32_t k_off = bind.Add(k_cache, "paged_attn: k_cache");
  const uint32_t v_off = bind.Add(v_cache, "paged_attn: v_cache");
  const uint32_t out_off = bind.Add(out, "paged_attn: out");
  const uint32_t bt_off = bind.AddU32Only(block_table, "paged_attn: block_table");
  const uint32_t sl_off = bind.AddU32Only(seq_lens, "paged_attn: seq_lens");
  const uint32_t qsl_off = bind.AddU32Only(query_start_loc, "paged_attn: query_start_loc");

  const int64_t wl = args.window_size.has_value() ? args.window_size->left : -1;
  const int64_t wr = args.window_size.has_value() ? args.window_size->right : -1;

  const uint32_t spec[4] = {DtypeCode(query.dtype), DtypeCode(k_cache.dtype),
                            DtypeCode(v_cache.dtype), DtypeCode(out.dtype)};
  PagedAttnParams p{static_cast<uint32_t>(total_q),
                    static_cast<uint32_t>(hq),
                    static_cast<uint32_t>(d),
                    static_cast<uint32_t>(block_size),
                    static_cast<uint32_t>(hq / num_kv_heads),
                    static_cast<uint32_t>(num_reqs),
                    args.causal ? 1u : 0u,
                    static_cast<int32_t>(wl),
                    static_cast<int32_t>(wr),
                    static_cast<uint32_t>(k_cache.stride[0]),
                    static_cast<uint32_t>(k_cache.stride[1]),
                    static_cast<uint32_t>(k_cache.stride[2]),
                    static_cast<uint32_t>(v_cache.stride[0]),
                    static_cast<uint32_t>(v_cache.stride[1]),
                    static_cast<uint32_t>(v_cache.stride[2]),
                    static_cast<uint32_t>(block_table.stride[0]),
                    static_cast<uint32_t>(block_table.stride[1]),
                    q_off,
                    k_off,
                    v_off,
                    out_off,
                    bt_off,
                    sl_off,
                    qsl_off,
                    args.scale,
                    args.logits_soft_cap};
  // One workgroup per (token, head) -- NOT FlatGroupCount, which divides by the
  // workgroup size; here the whole workgroup cooperates on one output row.
  // Flush before attention at large contexts: the preceding dense GEMM chunks
  // (QKV projection) may have accumulated enough GPU time in the current command
  // buffer that adding the attention dispatch would exceed the hangcheck.
  if (total_q > 256) VulkanContext::Get().FlushBatch("paged-attn-pre");
  Go("vt_paged_attn", bind, p, static_cast<uint32_t>(total_q * hq), spec, 4);
}

// cpu_cache.cpp:33-72 ReshapeAndCacheKernel. Pure BYTE MOVEMENT -- the CPU
// kernel is two memcpys per token and converts nothing -- so the dtype selects
// only the storage WIDTH to copy at, and the gate for it is bit-exactness.
void ReshapeAndCacheKernel(Queue&, const Tensor& k, const Tensor& v, Tensor& k_cache,
                           Tensor& v_cache, const Tensor& slot_mapping) {
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = k_cache.shape[1];
  const int64_t n_elems = k_cache.shape[2] * k_cache.shape[3];  // one token's page
  if (num_slots == 0 || n_elems == 0) return;
  VT_CHECK(slot_mapping.dtype == DType::kI64,
           "vulkan reshape_and_cache: slot_mapping must be i64");
  VT_CHECK(k.dtype == k_cache.dtype && v.dtype == v_cache.dtype,
           "vulkan reshape_and_cache: source and cache dtypes must match (this op "
           "moves bytes and converts nothing)");

  Binder bind;
  const uint32_t k_off = bind.Add(k, "reshape_and_cache: k");
  const uint32_t v_off = bind.Add(v, "reshape_and_cache: v");
  const uint32_t kc_off = bind.Add(k_cache, "reshape_and_cache: k_cache");
  const uint32_t vc_off = bind.Add(v_cache, "reshape_and_cache: v_cache");
  const uint32_t sm_off = bind.AddU32Only(slot_mapping, "reshape_and_cache: slot_mapping");

  const uint32_t spec[1] = {k.dtype == DType::kF32 ? 0u : 1u};
  ReshapeAndCacheParams p{static_cast<uint32_t>(num_slots),
                          static_cast<uint32_t>(n_elems),
                          static_cast<uint32_t>(block_size),
                          static_cast<uint32_t>(k_cache.stride[0]),
                          static_cast<uint32_t>(k_cache.stride[1]),
                          static_cast<uint32_t>(v_cache.stride[0]),
                          static_cast<uint32_t>(v_cache.stride[1]),
                          static_cast<uint32_t>(k.stride[0]),
                          static_cast<uint32_t>(v.stride[0]),
                          k_off,
                          v_off,
                          kc_off,
                          vc_off,
                          sm_off};
  Go("vt_reshape_and_cache", bind, p, FlatGroupCount(num_slots * n_elems), spec, 1);
}

// vt::RopeFromCache — the APPLY half of vLLM's rotary split.
// Upstream: rotary_embedding/base.py:160-252, common.py:145-185 @ e24d1b24fe96;
// our reference is cpu_ops.cpp RopeFromCacheKernel (:751-802).
//
// vLLM's RotaryEmbedding builds cos_sin_cache once in __init__ and the forward
// only applies it, so kRopeCosSinCache (the table, built in double) stays on the
// portable tier and this native kernel is the per-token apply. See the shader for
// why that boundary is also the right one numerically.
void RopeFromCacheKernel(Queue& queue, Tensor& qs, Tensor* ks, const Tensor& positions,
                         const Tensor& cache, const RopeArgs& args) {
  // MROPE DECLINES rather than throws. Multimodal RoPE selects a different
  // position AXIS per pair (cpu_ops.cpp:769-771 via MropeAxisForPair, mirroring
  // vLLM mrope.py), which this shader does not implement -- and throwing would
  // REMOVE a capability the portable reference tier already provides. Forwarded
  // through the provider seam, the same per-call refusal fp8 KV uses.
  if (positions.rank == 2) {
    auto next = reinterpret_cast<RopeFromCacheFn>(
        GetOpFallback(OpId::kRopeFromCache, DeviceType::kVULKAN, kNativeProviderName));
    next(queue, qs, ks, positions, cache, args);
    return;
  }

  const int64_t tokens = qs.shape[0];
  const int64_t hq = qs.shape[1];
  const int64_t hk = ks == nullptr ? 0 : ks->shape[1];
  const int64_t half = args.rotary_dim / 2;
  if (tokens == 0 || half == 0 || (hq + hk) == 0) return;
  VT_CHECK(positions.dtype == DType::kI32 || positions.dtype == DType::kI64,
           "vulkan rope_from_cache: positions must be i32 or i64");

  Binder bind;
  const uint32_t q_off = bind.Add(qs, "rope_from_cache: q");
  // Bindings 2/3 are declared by the shader whether or not k exists, and a
  // descriptor a shader statically uses must be valid even on the path that never
  // reads it -- so with hk == 0 they alias q and are dead. Same arrangement the
  // rmsnorm kernel already uses for its optional residual.
  const uint32_t k_off = ks != nullptr ? bind.Add(*ks, "rope_from_cache: k")
                                       : bind.Add(qs, "rope_from_cache: q");
  const uint32_t c_off = bind.Add(cache, "rope_from_cache: cos_sin_cache");
  const uint32_t p_off = bind.AddU32Only(positions, "rope_from_cache: positions");

  const uint32_t spec[5] = {DtypeCode(qs.dtype),
                            ks != nullptr ? DtypeCode(ks->dtype) : DtypeCode(qs.dtype),
                            DtypeCode(cache.dtype),
                            args.is_neox_style ? 1u : 0u,
                            positions.dtype == DType::kI64 ? 1u : 0u};
  RopeFromCacheParams p{static_cast<uint32_t>(tokens),
                        static_cast<uint32_t>(half),
                        static_cast<uint32_t>(args.rotary_dim),
                        static_cast<uint32_t>(hq),
                        static_cast<uint32_t>(hk),
                        static_cast<uint32_t>(qs.stride[0]),
                        static_cast<uint32_t>(qs.stride[1]),
                        static_cast<uint32_t>(ks != nullptr ? ks->stride[0] : 0),
                        static_cast<uint32_t>(ks != nullptr ? ks->stride[1] : 0),
                        q_off,
                        k_off,
                        c_off,
                        p_off};
  Go("vt_rope_from_cache", bind, p, FlatGroupCount(tokens * half * (hq + hk)), spec, 5);
}

// ---------------------------------------------------------------------------
// VK4 (B60 maple row): the rotary TABLE BUILD goes native. The old note —
// "kRopeCosSinCache is deliberately host-side, GLSL has no f64" — was WRONG on
// both halves: GLSL has `double` with core Float64 support in SPIR-V 1.3+, and
// our own CUDA kernel already does the f64 angle construction ON DEVICE
// (cuda_ops.cu RopeCosSinCacheKernel), so device-side double is the established
// house pattern for this op. vt_rope_cos_sin_cache.comp transcribes the CPU
// oracle's f64 pow/cos/sin element-for-element; the only f32 rounding left is
// the one cast at the store, which is exactly what the CPU does.
//
// cpu_ops.cpp:1182-1200 RopeCosSinCacheKernel.
void RopeCosSinCacheKernel(Queue&, Tensor& cos_sin, const Tensor& positions,
                           const RopeArgs& args) {
  const int64_t t = cos_sin.shape[0];
  const int64_t half = args.rotary_dim / 2;
  if (t == 0 || half == 0) return;
  VT_CHECK(cos_sin.dtype == DType::kF32,
           "vulkan rope_cos_sin_cache: cos_sin must be f32");
  VT_CHECK(positions.dtype == DType::kI32 || positions.dtype == DType::kI64,
           "vulkan rope_cos_sin_cache: positions must be i32 or i64");
  Binder bind;
  const uint32_t out_off = bind.Add(cos_sin, "rope_cos_sin_cache: cos_sin");
  const uint32_t pos_off =
      bind.AddU32Only(positions, "rope_cos_sin_cache: positions");
  const bool l3 = args.llama3_scaling_factor > 0.0f;
  const uint32_t spec[2] = {positions.dtype == DType::kI64 ? 1u : 0u, l3 ? 1u : 0u};
  RopeCosSinCacheParams p{static_cast<uint32_t>(t),
                          static_cast<uint32_t>(half),
                          static_cast<uint32_t>(args.rotary_dim),
                          pos_off,
                          out_off,
                          static_cast<double>(args.base),
                          static_cast<double>(args.llama3_scaling_factor),
                          static_cast<double>(args.llama3_low_freq_factor),
                          static_cast<double>(args.llama3_high_freq_factor),
                          static_cast<double>(args.llama3_orig_max_position)};
  Go("vt_rope_cos_sin_cache", bind, p, FlatGroupCount(t * half), spec, 2);
}

// VK4 (B60 maple row): the DIRECT-ANGLE rotary apply goes native. The dense
// AttnBlock path calls vt::RopeNeox every layer for the SWA layers; leaving it
// on the reference tier made each decode step pay a host round-trip. Same
// numerics story as RopeCosSinCacheKernel above — f64 angle accumulation,
// tracked f32 pow/trig exception (see the shader header).
// cpu_ops.cpp:1025-1038 RopeNeoxKernel.
void RopeNeoxKernel(Queue& /*queue*/, Tensor& qs, Tensor& ks, const Tensor& positions,
                    const RopeArgs& args) {
  const int64_t tokens = qs.shape[0];
  const int64_t hq = qs.shape[1];
  const int64_t hk = ks.shape[1];
  const int64_t half = args.rotary_dim / 2;
  if (tokens == 0 || half == 0 || (hq + hk) == 0) return;
  VT_CHECK(qs.dtype == ks.dtype,
           "vulkan rope_neox: q/k must share a dtype");
  VT_CHECK(positions.dtype == DType::kI32 || positions.dtype == DType::kI64,
           "vulkan rope_neox: positions must be i32 or i64");

  Binder bind;
  const uint32_t q_off = bind.Add(qs, "rope_neox: q");
  const uint32_t k_off = bind.Add(ks, "rope_neox: k");
  const uint32_t p_off = bind.AddU32Only(positions, "rope_neox: positions");
  const bool l3 = args.llama3_scaling_factor > 0.0f;
  const uint32_t qhd = static_cast<uint32_t>(qs.shape[2]);
  const uint32_t khd = static_cast<uint32_t>(ks.shape[2]);
  const uint32_t spec[6] = {DtypeCode(qs.dtype),
                            DtypeCode(ks.dtype),
                            positions.dtype == DType::kI64 ? 1u : 0u,
                            l3 ? 1u : 0u,
                            qhd,
                            khd};
  RopeNeoxParams p{static_cast<uint32_t>(tokens),
                   static_cast<uint32_t>(half),
                   static_cast<uint32_t>(args.rotary_dim),
                   static_cast<uint32_t>(hq),
                   static_cast<uint32_t>(hk),
                   q_off,
                   k_off,
                   p_off,
                   static_cast<double>(args.base),
                   static_cast<double>(args.llama3_scaling_factor),
                   static_cast<double>(args.llama3_low_freq_factor),
                   static_cast<double>(args.llama3_high_freq_factor),
                   static_cast<double>(args.llama3_orig_max_position)};
  Go("vt_rope_neox", bind, p,
     FlatGroupCount(tokens * half * (hq + hk)), spec, 6);
}

// VK4: MoeCombine native. cpu_ops.cpp:2749-2765 MoeCombineKernel.
// The optional shared expert is bound unconditionally (aliased to expert_out
// when absent — dead reads on that path, the rope kernel's arrangement) and
// gated by a spec const, because a shader's declared descriptors must be valid
// whether or not this dispatch reads them.
void MoeCombineKernelVulkan(Queue&, Tensor& out, const Tensor& expert_out,
                            const Tensor& weights, const Tensor* shared,
                            float routed_scale) {
  const int64_t t = out.shape[0];
  const int64_t h = out.shape[1];
  const int64_t k = weights.shape[1];
  if (t == 0 || h == 0) return;
  Binder bind;
  const uint32_t out_off = bind.Add(out, "moe_combine: out");
  const uint32_t e_off = bind.Add(expert_out, "moe_combine: expert_out");
  // weights is f32-by-contract ([T,K] f32), so it takes the full u32+u16 pair
  // like every other operand — the shader declares 8 bindings total.
  const uint32_t w_off = bind.Add(weights, "moe_combine: weights");
  const uint32_t s_off = shared != nullptr
                             ? bind.Add(*shared, "moe_combine: shared")
                             : bind.Add(expert_out, "moe_combine: shared(dead)");
  const uint32_t spec[3] = {DtypeCode(out.dtype), DtypeCode(expert_out.dtype),
                            shared != nullptr ? 1u : 0u};
  MoeCombineParams p{static_cast<uint32_t>(t),
                     static_cast<uint32_t>(h),
                     static_cast<uint32_t>(k),
                     routed_scale,
                     out_off,
                     e_off,
                     w_off,
                     s_off};
  Go("vt_moe_combine", bind, p, FlatGroupCount(t * h), spec, 3);
}

// VK4: MoeRouterTopK native — the UNGROUPED softmax path only. Grouped
// top-k and a non-null bias DECLINE to the reference tier (same per-call
// refusal shape as fp8 KV / mrope): throwing would remove capability the
// portable tier provides. cpu_ops.cpp:2680-2721.
void MoeRouterTopKKernelVulkan(Queue& queue, Tensor& weights, Tensor& indices,
                               const Tensor& logits, const MoeRouterTopKArgs& args,
                               const Tensor* bias) {
  if (args.num_expert_group > 0 || bias != nullptr) {
    auto next = reinterpret_cast<MoeRouterTopKFn>(
        GetOpFallback(OpId::kMoeRouterTopK, DeviceType::kVULKAN, kNativeProviderName));
    next(queue, weights, indices, logits, args, bias);
    return;
  }
  const int64_t t = logits.shape[0];
  const int64_t e = logits.shape[1];
  VT_CHECK(e <= 1024, "vulkan moe_router_topk: E exceeds VT_MR_MAX_E (1024)");
  if (t == 0) return;
  Binder bind;
  // BINDING ORDER = the shader's declared indices: 0=W32(weights), 1=I32(indices),
  // 2/3 = the logits u32/u16 pair. The original order put the logits PAIR first,
  // so the shader's W32/I32 writes landed in the LOGITS buffer and its loads read
  // weights-as-logits — wrong numbers AND cross-buffer corruption.
  const uint32_t w_off = bind.AddU32Only(weights, "moe_router_topk: weights");
  const uint32_t i_off = bind.AddU32Only(indices, "moe_router_topk: indices");
  const uint32_t l_off = bind.Add(logits, "moe_router_topk: logits");
  // spec constants are per-pipeline; E/K vary only across models, so the
  // pipeline cache keys on them correctly. renorm rides args.renormalize.
  const uint32_t spec[4] = {static_cast<uint32_t>(e),
                            static_cast<uint32_t>(args.top_k),
                            args.renormalize ? 1u : 0u,
                            DtypeCode(logits.dtype)};
  MoeRouterTopKParams p{static_cast<uint32_t>(t),
                        static_cast<uint32_t>(e),
                        static_cast<uint32_t>(args.top_k),
                        l_off,
                        w_off,
                        i_off};
  Go("vt_moe_router_topk", bind, p, static_cast<uint32_t>(t), spec, 4);
}

// cpu_ops.cpp:2162-2176 QkvSplitKernel. Mirrors vLLM's QKVParallelLinear output
// split (qkv.split([q_size, kv_size, kv_size], dim=-1)); the three widths are
// independent because under GQA k and v are narrower than q. One invocation per
// OUTPUT element across all three destinations, so this is one dispatch.
void QkvSplitKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& qkv) {
  const int64_t t = qkv.shape[0];
  if (t == 0) return;
  const int64_t q_dim = q_out.Numel() / t;
  const int64_t k_dim = k_out.Numel() / t;
  const int64_t v_dim = v_out.Numel() / t;
  VT_CHECK(q_out.dtype == k_out.dtype && k_out.dtype == v_out.dtype,
           "vulkan qkv_split: the three destinations must share a dtype");
  Binder bind;
  const uint32_t src_off = bind.Add(qkv, "qkv_split: qkv");
  const uint32_t q_off = bind.Add(q_out, "qkv_split: q");
  const uint32_t k_off = bind.Add(k_out, "qkv_split: k");
  const uint32_t v_off = bind.Add(v_out, "qkv_split: v");
  const uint32_t spec[2] = {DtypeCode(qkv.dtype), DtypeCode(q_out.dtype)};
  QkvSplitParams p{static_cast<uint32_t>(t),     static_cast<uint32_t>(q_dim),
                   static_cast<uint32_t>(k_dim), static_cast<uint32_t>(v_dim),
                   src_off,                      q_off,
                   k_off,                        v_off};
  Go("vt_qkv_split", bind, p, FlatGroupCount(t * (q_dim + k_dim + v_dim)), spec, 2);
}

// ===========================================================================
// The GDN / conv1d family (BACKEND-VULKAN-GDN). Qwen3.6-27B is a GDN hybrid, so
// before this row every one of these ops fell to the PORTABLE CPU REFERENCE TIER
// on Vulkan — correct, and running on the host against shared memory.
//
// The two RECURRENCES themselves (kGdnPrefill / kGdnDecode) landed in the
// follow-up row BACKEND-VULKAN-GDN-CORE and are at the bottom of this section.
//
// WHAT IS DELIBERATELY NOT HERE, so a later row does not have to re-derive it:
//   * kRopeCosSinCache — the rotary TABLE BUILD, which constructs its angles in
//     `double` (cpu_ops.cpp RopeCosSinCacheKernel). GLSL has no f64 here and
//     emulating it would be a numerics divergence in the one place vLLM itself
//     keeps the work off the device (its RotaryEmbedding builds the cache once in
//     __init__). Leaving it on the host MIRRORS upstream; "implementing" it would
//     be a regression, and the assertion in tests/vt/test_vulkan_backend.cpp says
//     so out loud.
//   * kCausalConv1dFwd — the PREFILL conv — is NATIVE now, in the first of the
//     two shapes this comment used to name: one invocation per (sequence,
//     channel), serial over the whole token range, with the old state row copied
//     into a private array before the write-back. See CausalConv1dFwdKernel; an
//     opt-in token-block split exists and its own hazard is documented there.
// ===========================================================================

// cpu_ops.cpp:2272-2279 SigmoidGateBf16Kernel. Flat, one invocation per element.
void SigmoidGateBf16Kernel(Queue&, Tensor& out, const Tensor& attn, const Tensor& gate) {
  const int64_t n = out.Numel();
  if (n == 0) return;
  Binder bind;
  const uint32_t a_off = bind.Add(attn, "sigmoid_gate_bf16: attn");
  const uint32_t g_off = bind.Add(gate, "sigmoid_gate_bf16: gate");
  const uint32_t o_off = bind.Add(out, "sigmoid_gate_bf16: out");
  // Only the attention operand varies; `gate` is f32 and `out` is bf16 by the op
  // contract (src/vt/ops.cpp:3327-3334) and are compile-time constants in the
  // shader, so a violation fails the host VT_CHECK rather than silently working.
  const uint32_t spec[1] = {DtypeCode(attn.dtype)};
  SigmoidGateParams p{static_cast<uint32_t>(n), a_off, g_off, o_off};
  Go("vt_sigmoid_gate_bf16", bind, p, FlatGroupCount(n), spec, 1);
}

// cpu_ops.cpp:1210-1235 RmsNormGatedKernel. ONE WORKGROUP PER ROW (the workgroup
// tree-reduces the mean square), not FlatGroupCount — same convention as
// vt_rms_norm.
void RmsNormGatedKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& gate,
                        const Tensor& w, const RmsNormGatedArgs& args) {
  const int64_t d = x.shape[x.rank - 1];
  const int64_t rows = d == 0 ? 0 : x.Numel() / d;
  if (rows == 0 || d == 0) return;
  // The rank-3 gate is a padded-row [T,Hv,D] view of the merged qkvz z slice;
  // rank-2 degenerates to group 1 with outer stride d (cpu_ops.cpp:1218-1219).
  const int64_t gate_group = gate.rank == 3 ? gate.shape[1] : 1;
  const int64_t gate_outer = gate.stride[0];
  Binder bind;
  const uint32_t x_off = bind.Add(x, "rmsnorm_gated: x");
  const uint32_t z_off = bind.Add(gate, "rmsnorm_gated: gate");
  const uint32_t w_off = bind.Add(w, "rmsnorm_gated: weight");
  const uint32_t out_off = bind.Add(out, "rmsnorm_gated: out");
  RmsNormGatedParams p{static_cast<uint32_t>(rows),
                       static_cast<uint32_t>(d),
                       DtypeCode(x.dtype),
                       DtypeCode(gate.dtype),
                       DtypeCode(w.dtype),
                       DtypeCode(out.dtype),
                       args.sigmoid_gate ? 1u : 0u,
                       static_cast<uint32_t>(gate_group),
                       static_cast<uint32_t>(gate_outer),
                       x_off,
                       z_off,
                       w_off,
                       out_off,
                       args.eps};
  Go("vt_rms_norm_gated", bind, p, static_cast<uint32_t>(rows));
}

// Shared geometry of the two state-cache ops (cpu_ops.cpp:1676-1682 and
// :1723-1727 compute it identically). `mid` is the channels/heads per row and
// `cache_row` the row's PHYSICAL width, which exceeds work_row when the conv
// state has been widened for spec-decode rollback.
struct GdnStateGeom {
  int64_t rows, work_row, work_inner, cache_inner, cache_row;
};
GdnStateGeom GdnStateGeometry(const Tensor& working, const Tensor& cache,
                              const Tensor& state_idx) {
  GdnStateGeom g{};
  g.rows = state_idx.shape[0];
  if (g.rows == 0) return g;
  g.work_inner = working.shape[working.rank - 1];
  g.cache_inner = cache.shape[cache.rank - 1];
  g.work_row = working.Numel() / g.rows;
  const int64_t mid = g.work_inner == 0 ? 0 : g.work_row / g.work_inner;
  g.cache_row = mid * g.cache_inner;
  return g;
}

// cpu_ops.cpp:1666-1707 GdnStateGatherKernel.
void GdnStateGatherKernel(Queue&, Tensor& working, const Tensor& cache,
                          const Tensor& state_idx, const Tensor* has_initial_state) {
  const GdnStateGeom g = GdnStateGeometry(working, cache, state_idx);
  if (g.rows == 0 || g.work_row == 0) return;
  Binder bind;
  const uint32_t work_off = bind.Add(working, "gdn_state_gather: working");
  const uint32_t cache_off = bind.Add(cache, "gdn_state_gather: cache");
  const uint32_t idx_off = bind.AddU32Only(state_idx, "gdn_state_gather: state_idx");
  // Binding 5 is always written: a descriptor a shader statically uses must be
  // valid even on the path that never reads it. With his_mode == 0 it aliases
  // state_idx and is dead — the same arrangement vt_rms_norm uses for its
  // optional residual.
  const uint32_t his_off =
      has_initial_state != nullptr
          ? bind.AddByteView(*has_initial_state, "gdn_state_gather: has_initial_state")
          : bind.AddByteView(state_idx, "gdn_state_gather: state_idx");
  uint32_t his_mode = 0;
  if (has_initial_state != nullptr) {
    his_mode = has_initial_state->dtype == DType::kI8 ? 1u : 2u;
  }
  GdnStateGatherParams p{static_cast<uint32_t>(g.rows),
                         static_cast<uint32_t>(g.work_row),
                         static_cast<uint32_t>(g.work_inner),
                         static_cast<uint32_t>(g.cache_inner),
                         static_cast<uint32_t>(g.cache_row),
                         static_cast<uint32_t>(cache.shape[0]),
                         DtypeCode(working.dtype),
                         DtypeCode(cache.dtype),
                         his_mode,
                         work_off,
                         cache_off,
                         idx_off,
                         his_off};
  Go("vt_gdn_state_gather", bind, p, FlatGroupCount(g.rows * g.work_row));
}

// cpu_ops.cpp:1709-1745 GdnStateScatterKernel.
void GdnStateScatterKernel(Queue&, Tensor& cache, const Tensor& working,
                           const Tensor& state_idx) {
  const GdnStateGeom g = GdnStateGeometry(working, cache, state_idx);
  if (g.rows == 0 || g.work_row == 0) return;
  Binder bind;
  const uint32_t cache_off = bind.Add(cache, "gdn_state_scatter: cache");
  const uint32_t work_off = bind.Add(working, "gdn_state_scatter: working");
  const uint32_t idx_off = bind.AddU32Only(state_idx, "gdn_state_scatter: state_idx");
  GdnStateScatterParams p{static_cast<uint32_t>(g.rows),
                          static_cast<uint32_t>(g.work_row),
                          static_cast<uint32_t>(g.work_inner),
                          static_cast<uint32_t>(g.cache_inner),
                          static_cast<uint32_t>(g.cache_row),
                          static_cast<uint32_t>(cache.shape[0]),
                          DtypeCode(working.dtype),
                          DtypeCode(cache.dtype),
                          cache_off,
                          work_off,
                          idx_off};
  Go("vt_gdn_state_scatter", bind, p, FlatGroupCount(g.rows * g.work_row));
}

struct ConvFwdParams {
  uint32_t n, c_dim, k, width, x_rs;
  uint32_t max_t_len;
  uint32_t has_bias, his_is_i8, silu;
  uint32_t out_dt, x_dt, w_dt, bias_dt, st_dt;
  uint32_t out_off, x_off, w_off, bias_off, st_off, qsl_off, his_off;
};
static_assert(sizeof(ConvFwdParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");

// cpu_ops.cpp CausalConv1dFwdKernel, the GDN PREFILL conv.
//
// DEFAULT MAPPING: one invocation per (sequence, channel), serial over that
// sequence's whole token range -- the CPU kernel's own ForRows unit, and the
// first of the two shapes the block comment above this family names for this
// op. The op computes every output from the OLD state window and then
// overwrites that window; one invocation owns both the reads and the
// write-back for its (sequence, channel), and it copies the old window into a
// private array first (the CPU kernel's `old_row`), so no other invocation can
// observe a half-rolled state.
//
// Deliberately NOT llama.cpp's ssm_conv.comp shape by default, which also
// parallelises over tokens: that works there because ggml advances the conv
// state as a separate node, and here the same dispatch reads and advances it.
//
// OPTIONAL TOKEN BLOCKS, VT_VULKAN_CONV_TARGET_GROUPS=<g>: split each sequence
// into VT_CONV_BLOCKS token blocks in the same dispatch. Only tokens t < width
// read the carried state, and the shader gives every one of them to block 0 --
// the block that also writes the state back -- so block 0 is the only one that
// touches conv_state at all. An earlier split that did not do this raced: see
// the BLOCK 0 comment in vt_causal_conv1d_fwd.comp. The split changes the grid
// shape only, never the output, so it is off unless asked for.
//
// STATE ROWS: the CPU reference addresses conv_state with stride `width` and
// the shader mirrors it. The op layer also admits rows WIDER than K-1
// (speculative-decode taps, ops.cpp CheckConvCommon), where that stride and
// the row's physical stride disagree. This kernel declines those shapes
// rather than choose between them, so they keep the reference tier's answer.
void CausalConv1dFwdKernel(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                           const Tensor* bias, Tensor& conv_state, const Tensor& qsl,
                           const Tensor& his, const CausalConv1dArgs& args) {
  const int64_t c_dim = x.shape[1], k = w.shape[1], total = x.shape[0];
  const int64_t n = conv_state.shape[0];
  if (n == 0 || c_dim == 0) return;
  auto decline = [&] {
    auto next = reinterpret_cast<CausalConv1dFwdFn>(
        GetOpFallback(OpId::kCausalConv1dFwd, DeviceType::kVULKAN, kNativeProviderName));
    next(q, out, x, w, bias, conv_state, qsl, his, args);
  };
  // PER-CALL REFUSAL rather than a throw, the seam vt_paged_attn and the GDN
  // recurrences use: a shape this shader cannot serve forwards to the portable
  // reference tier, which is correct for every shape. Reasons to decline:
  //   * K = 1 (no carried state, so a zero-length state row to bind) or a
  //     kernel width past the shader's private window (VT_CONV_MAX_WIDTH).
  //   * conv_state rows wider than K-1 (see STATE ROWS above).
  //   * A storage dtype outside f32/f16/bf16 (conv_state: f32/bf16).
  //   * has_initial_state that is neither i8 nor i32.
  //   * More (sequence, channel) pairs than one dispatch can cover: the grid
  //     is 1-D and Dispatch refuses a count above the device's
  //     maxComputeWorkGroupCount[0].
  //   * Any index the shader forms that would not fit its uint32 arithmetic
  //     (checked after binding, below, because it includes the offsets).
  const auto is_float = [](DType d) {
    return d == DType::kF32 || d == DType::kF16 || d == DType::kBF16;
  };
  const int64_t max_invocations =
      static_cast<int64_t>(VulkanContext::Get().max_workgroup_count_x()) * kWorkgroupSize;
  const bool serve = n * c_dim <= max_invocations && k >= 2 && k - 1 <= 8 &&
                     conv_state.shape[2] == k - 1 && is_float(out.dtype) &&
                     is_float(x.dtype) && is_float(w.dtype) &&
                     (bias == nullptr || is_float(bias->dtype)) &&
                     (conv_state.dtype == DType::kF32 || conv_state.dtype == DType::kBF16) &&
                     (his.dtype == DType::kI8 || his.dtype == DType::kI32);
  if (!serve) {
    decline();
    return;
  }
  // query_start_loc is NOT read on the host: it is device memory, and reading
  // it here would need the pending batch drained first. The shader guards
  // memory, not semantics -- see the guard at the top of its main(). (A
  // DECLINED call does reach host code: GetOpFallback drains the batch and the
  // reference tier runs over host-visible device storage.)

  Binder bind;
  const uint32_t out_off = bind.Add(out, "causal_conv1d_fwd: out");
  const uint32_t x_off = bind.Add(x, "causal_conv1d_fwd: x");
  const uint32_t w_off = bind.Add(w, "causal_conv1d_fwd: weight");
  const uint32_t bias_off = bias != nullptr ? bind.Add(*bias, "causal_conv1d_fwd: bias")
                                            : bind.Add(w, "causal_conv1d_fwd: weight");
  const uint32_t st_off = bind.Add(conv_state, "causal_conv1d_fwd: conv_state");
  const uint32_t qsl_off = bind.AddU32Only(qsl, "causal_conv1d_fwd: query_start_loc");
  // has_initial_state may be i8, which is not 4-byte aligned, so it goes through
  // the byte view and the shader unpacks it -- the same reasoning the update
  // kernel gives for aliasing a bf16 state through AddByteView.
  const uint32_t his_off = bind.AddByteView(his, "causal_conv1d_fwd: has_initial_state");

  // F5 of the review: the shader forms every index in uint32. Decline when the
  // largest one -- the binding's base element plus the farthest element the
  // shader can address in it -- would not fit.
  {
    constexpr int64_t kU32 = 0xFFFFFFFFll;
    const auto elem = [](DType d) { return d == DType::kF32 ? 4 : 2; };
    const auto fits = [&](uint32_t off_bytes, int64_t esize, int64_t extent) {
      return static_cast<int64_t>(off_bytes) / esize + extent <= kU32;
    };
    const int64_t x_extent = (total > 0 ? (total - 1) * x.stride[0] : 0) + c_dim;
    const bool ok =
        // query_start_loc is i32 and its last entry is `total`, so every
        // sequence length the shader sees is below 2^31; asserting it here is
        // what makes the shader's block arithmetic provably wrap-free.
        total <= 0x7FFFFFFFll && x.stride[0] >= 0 && x.stride[0] <= kU32 &&
        n * c_dim * k <= kU32 &&
        fits(out_off, elem(out.dtype), total * c_dim) &&
        fits(x_off, elem(x.dtype), x_extent) &&
        fits(w_off, elem(w.dtype), c_dim * k) &&
        (bias == nullptr || fits(bias_off, elem(bias->dtype), c_dim)) &&
        fits(st_off, elem(conv_state.dtype), n * c_dim * (k - 1)) &&
        fits(qsl_off, 4, n + 1) && fits(his_off, 1, n + 4);
    if (!ok) {
      decline();
      return;
    }
  }

  ConvFwdParams p{static_cast<uint32_t>(n),
                  static_cast<uint32_t>(c_dim),
                  static_cast<uint32_t>(k),
                  static_cast<uint32_t>(k - 1),
                  static_cast<uint32_t>(x.stride[0]),
                  static_cast<uint32_t>(total),
                  bias != nullptr ? 1u : 0u,
                  his.dtype == DType::kI8 ? 1u : 0u,
                  args.silu_activation ? 1u : 0u,
                  DtypeCode(out.dtype),
                  DtypeCode(x.dtype),
                  DtypeCode(w.dtype),
                  bias != nullptr ? DtypeCode(bias->dtype) : DtypeCode(w.dtype),
                  DtypeCode(conv_state.dtype),
                  out_off,
                  x_off,
                  w_off,
                  bias_off,
                  st_off,
                  qsl_off,
                  his_off};
  // TOKEN BLOCKS: 1 unless VT_VULKAN_CONV_TARGET_GROUPS asks for a split. Read
  // on every call rather than cached, so one test process can run both
  // mappings -- the split path carries the block-0 rule and has to stay under
  // test even though it is not the default.
  int64_t blocks = 1;
  if (const char* v = std::getenv("VT_VULKAN_CONV_TARGET_GROUPS")) {
    const long g = std::strtol(v, nullptr, 10);
    if (g >= 1 && g <= 65536) {
      const int64_t base_groups = (n * c_dim + 127) / 128;
      // Never more blocks than there are tokens to give them: past that the
      // extra invocations are empty and only cost launch.
      const int64_t avg_len = std::max<int64_t>(1, total / n);
      blocks = (static_cast<int64_t>(g) + base_groups - 1) / base_groups;
      blocks = std::max<int64_t>(1, std::min<int64_t>(blocks, avg_len));
      // ...and never more than one dispatch can launch.
      blocks = std::min<int64_t>(blocks, max_invocations / (n * c_dim));
      // VT_CONV_BLOCKS is a specialization constant, so every distinct value is
      // a separately compiled pipeline; the cap bounds what this knob can
      // create to 64. A cap, NOT power-of-two rounding: with rounding, the
      // pre-fix race stopped reproducing on llvmpipe (4 blocks of one
      // (sequence, channel) sat inside one 8-lane subgroup in the test's
      // layout), so it would have hidden the hazard from the test that guards it.
      blocks = std::min<int64_t>(blocks, 64);
    }
  }
  const uint32_t conv_spec[1] = {static_cast<uint32_t>(blocks)};
  Go("vt_causal_conv1d_fwd", bind, p,
     FlatGroupCount(n * c_dim * blocks), conv_spec, 1);
}

// cpu_ops.cpp:1081-1127 CausalConv1dUpdateKernel. One invocation per
// (token, channel) — the CPU kernel's own row-chunking unit, and what makes the
// read-old-then-roll safe with no barrier.
void CausalConv1dUpdateKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                              const Tensor* bias, Tensor& conv_state,
                              const Tensor* conv_state_indices,
                              const CausalConv1dArgs& args) {
  const int64_t batch = x.shape[0], c_dim = x.shape[1], k = w.shape[1];
  if (batch == 0 || c_dim == 0) return;
  // The state is read AND written in place through the dtype-erased pair of
  // views, so a COMPRESSED (bf16) cache needs no caller-side gather/upcast. That
  // is what SupportsCompressedConvState() advertises for this backend
  // (src/vt/ops.cpp CheckConvCommon), and what lets the model's indexed
  // state-I/O path — which hands the kernel the cache itself plus the slot
  // indices — replace two host memcpys per GDN layer per token.
  VT_CHECK(conv_state.dtype == DType::kF32 || conv_state.dtype == DType::kBF16,
           "vulkan causal_conv1d_update: conv_state must be f32 or bf16");
  Binder bind;
  const uint32_t out_off = bind.Add(out, "causal_conv1d_update: out");
  const uint32_t x_off = bind.Add(x, "causal_conv1d_update: x");
  const uint32_t w_off = bind.Add(w, "causal_conv1d_update: weight");
  // Bindings 6/7 alias the weight when there is no bias; see the note above.
  const uint32_t bias_off = bias != nullptr ? bind.Add(*bias, "causal_conv1d_update: bias")
                                            : bind.Add(w, "causal_conv1d_update: weight");
  const uint32_t st_off = bind.Add(conv_state, "causal_conv1d_update: conv_state");
  const uint32_t idx_off =
      conv_state_indices != nullptr
          ? bind.AddU32Only(*conv_state_indices, "causal_conv1d_update: conv_state_indices")
          // Alias the state when there are no indices. AddByteView, not
          // AddU32Only: a bf16 state is only 2-byte aligned, and AddU32Only's
          // 4-byte assertion would reject a perfectly valid tensor on the path
          // where the shader never reads this binding at all (p.has_idx == 0).
          : bind.AddByteView(conv_state, "causal_conv1d_update: conv_state");
  ConvUpdateParams p{static_cast<uint32_t>(batch),
                     static_cast<uint32_t>(c_dim),
                     static_cast<uint32_t>(k),
                     static_cast<uint32_t>(k - 1),
                     static_cast<uint32_t>(conv_state.shape[2]),
                     static_cast<uint32_t>(x.stride[0]),
                     static_cast<uint32_t>(conv_state.shape[0]),
                     bias != nullptr ? 1u : 0u,
                     conv_state_indices != nullptr ? 1u : 0u,
                     args.silu_activation ? 1u : 0u,
                     DtypeCode(out.dtype),
                     DtypeCode(x.dtype),
                     DtypeCode(w.dtype),
                     bias != nullptr ? DtypeCode(bias->dtype) : DtypeCode(w.dtype),
                     DtypeCode(conv_state.dtype),
                     out_off,
                     x_off,
                     w_off,
                     bias_off,
                     st_off,
                     idx_off};
  Go("vt_causal_conv1d_update", bind, p, FlatGroupCount(batch * c_dim));
}

// cpu_ops.cpp:2337-2417 GdnPostConvKernel. ONE WORKGROUP PER (token, head slot)
// over Hk + Hv slots — upstream's own (L, H+HV) grid — not FlatGroupCount: the
// q/k slots tree-reduce an L2 norm across the workgroup's lanes.
void GdnPostConvKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& v_out, Tensor& g_out,
                       Tensor& beta_out, const Tensor& conv, const Tensor& araw,
                       const Tensor& braw, const Tensor& a_log, const Tensor& dt_bias,
                       const L2NormArgs& args) {
  const int64_t t = conv.shape[0];
  const int64_t hk = q_out.shape[1], dk = q_out.shape[2];
  const int64_t hv = v_out.shape[1], dv = v_out.shape[2];
  if (t == 0 || hk + hv == 0) return;
  const int64_t key_dim = hk * dk, value_dim = hv * dv;
  Binder bind;
  const uint32_t conv_off = bind.Add(conv, "gdn_post_conv: conv");
  const uint32_t q_off = bind.Add(q_out, "gdn_post_conv: q_out");
  const uint32_t k_off = bind.Add(k_out, "gdn_post_conv: k_out");
  const uint32_t v_off = bind.Add(v_out, "gdn_post_conv: v_out");
  const uint32_t a_off = bind.Add(araw, "gdn_post_conv: araw");
  const uint32_t b_off = bind.Add(braw, "gdn_post_conv: braw");
  // f32 BY CONTRACT (src/vt/ops.cpp:3459-3463), so one binding each rather than a
  // dtype-erased pair whose 16-bit half could never be taken.
  const uint32_t g_off = bind.AddU32Only(g_out, "gdn_post_conv: g_out");
  const uint32_t beta_off = bind.AddU32Only(beta_out, "gdn_post_conv: beta_out");
  const uint32_t alog_off = bind.AddU32Only(a_log, "gdn_post_conv: a_log");
  const uint32_t dtb_off = bind.AddU32Only(dt_bias, "gdn_post_conv: dt_bias");
  // Ascending constantID order: conv dtype, the shared q/k/v dtype, the shared
  // araw/braw dtype.
  const uint32_t spec[3] = {DtypeCode(conv.dtype), DtypeCode(q_out.dtype),
                            DtypeCode(araw.dtype)};
  GdnPostConvParams p{static_cast<uint32_t>(t),
                      static_cast<uint32_t>(hk),
                      static_cast<uint32_t>(dk),
                      static_cast<uint32_t>(hv),
                      static_cast<uint32_t>(dv),
                      static_cast<uint32_t>(key_dim),
                      static_cast<uint32_t>(value_dim),
                      static_cast<uint32_t>(2 * key_dim + value_dim),
                      static_cast<uint32_t>(araw.stride[0]),
                      static_cast<uint32_t>(braw.stride[0]),
                      conv_off,
                      q_off,
                      k_off,
                      v_off,
                      a_off,
                      b_off,
                      g_off,
                      beta_off,
                      alog_off,
                      dtb_off,
                      args.eps};
  Go("vt_gdn_post_conv", bind, p, static_cast<uint32_t>(t * (hk + hv)), spec, 3);
}

// ---------------------------------------------------------------------------
// The two GATED-DELTA RECURRENCES (BACKEND-VULKAN-GDN-CORE). These are not glue:
// they ARE Qwen3.6's linear-attention core, and with the glue above already
// native they were the whole of what the model still ran on the host — a 512
// token prompt spent ~280 s in kGdnPrefill on the reference tier.
//
// The shaders (src/vt/vulkan/shaders/vt_gdn_prefill.comp, vt_gdn_decode.comp and
// the shared vt_gdn_recurrence.glsl) carry the port provenance and the tile
// geometry. What the HOST has to get right is only the grid and the decline.
// ---------------------------------------------------------------------------

// Must equal VT_GDN_BV / VT_GDN_MAX_DK in vt_gdn_recurrence.glsl. Duplicated
// rather than shared because the shader constants are in GLSL and the SPIR-V is
// committed, so nothing can compute one from the other — the gate in
// tests/vt/test_vulkan_backend.cpp exercises a Dv that is NOT a multiple of the
// tile so a drift shows up as wrong numbers there rather than in a model run.
constexpr int64_t kGdnTileRows = 16;
constexpr int64_t kGdnMaxDk = 128;

// Shared grid + binding setup for the two recurrences. Returns false when this
// backend cannot serve the shape and the caller must decline to the next
// provider.
bool GdnRecurrenceCommon(const Tensor& out, const Tensor& q_in, const Tensor& k,
                         const Tensor& v, const Tensor& g, const Tensor& beta,
                         const Tensor& state, Binder& bind, GdnRecurrenceParams& p,
                         uint32_t spec[2], float scale) {
  const int64_t hv = state.shape[1], dv = state.shape[2], dk = state.shape[3];
  const int64_t hk = q_in.shape[1];
  // PER-CALL REFUSAL rather than a throw, the same seam vt_paged_attn uses for an
  // fp8 KV cache: declining forwards to the portable reference tier, which is
  // correct for every shape, instead of REMOVING a capability the backend already
  // had. Two reasons to decline.
  //   * Dk beyond the shared tile's compile-time extent. The tile has to be sized
  //     at compile time against Vulkan's GUARANTEED 16 KB of shared memory, and
  //     the real gate dim is 128.
  //   * q/k/v disagreeing on dtype. One specialization constant covers the three
  //     (they come out of one GdnPostConv dispatch and always agree), and CUDA
  //     asserts the same thing (cuda_gdn.cu:2577).
  if (dk > kGdnMaxDk || dk <= 0) return false;
  if (k.dtype != q_in.dtype || v.dtype != q_in.dtype) return false;
  const uint32_t q_off = bind.Add(q_in, "gdn recurrence: q");
  const uint32_t k_off = bind.Add(k, "gdn recurrence: k");
  const uint32_t v_off = bind.Add(v, "gdn recurrence: v");
  const uint32_t out_off = bind.Add(out, "gdn recurrence: out");
  // g, beta and the state are f32 by the op contract on this device
  // (src/vt/ops.cpp:1629-1643 — a compressed state is CUDA-only), so one binding
  // each rather than a dtype-erased pair whose 16-bit half could never be taken.
  const uint32_t g_off = bind.AddU32Only(g, "gdn recurrence: g");
  const uint32_t beta_off = bind.AddU32Only(beta, "gdn recurrence: beta");
  const uint32_t state_off = bind.AddU32Only(state, "gdn recurrence: state");
  spec[0] = DtypeCode(q_in.dtype);
  spec[1] = DtypeCode(out.dtype);
  p.hk = static_cast<uint32_t>(hk);
  p.dk = static_cast<uint32_t>(dk);
  p.hv = static_cast<uint32_t>(hv);
  p.dv = static_cast<uint32_t>(dv);
  p.nv = static_cast<uint32_t>((dv + kGdnTileRows - 1) / kGdnTileRows);
  p.ratio = static_cast<uint32_t>(hv / hk);
  p.has_idx = 0;
  p.n_state_rows = static_cast<uint32_t>(state.shape[0]);
  p.q_off = q_off;
  p.k_off = k_off;
  p.v_off = v_off;
  p.out_off = out_off;
  p.g_off = g_off;
  p.beta_off = beta_off;
  p.state_off = state_off;
  p.meta_off = 0;
  p.scale = scale;
  return true;
}

// cpu_ops.cpp:1331-1366 GdnPrefillKernel. ONE WORKGROUP PER
// (sequence, value-head, value-tile) — the CPU kernel's own (SEQUENCE,
// VALUE-HEAD) chunking plus the value-row tile our CUDA kernel already uses as
// its grid.x (cuda_gdn.cu:2421). NOT FlatGroupCount: the whole workgroup
// cooperates on one tile, and the sequence stays sequential inside it.
void GdnPrefillKernel(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                      const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                      const Tensor& query_start_loc, const GdnArgs& args) {
  const int64_t n = state.shape[0], hv = state.shape[1], dv = state.shape[2];
  if (n == 0 || hv == 0 || dv == 0) return;
  Binder bind;
  GdnRecurrenceParams p{};
  uint32_t spec[2] = {0, 0};
  if (!GdnRecurrenceCommon(out, q_in, k, v, g, beta, state, bind, p, spec, args.scale)) {
    auto next = reinterpret_cast<GdnPrefillFn>(
        GetOpFallback(OpId::kGdnPrefill, DeviceType::kVULKAN, kNativeProviderName));
    next(q, out, q_in, k, v, g, beta, state, query_start_loc, args);
    return;
  }
  p.meta_off = bind.AddU32Only(query_start_loc, "gdn_prefill: query_start_loc");
  Go("vt_gdn_prefill", bind, p, static_cast<uint32_t>(n * hv * p.nv), spec, 2);
}

// cpu_ops.cpp:1368-1396 GdnDecodeKernel, one step per batch token. Same grid with
// the sequence axis replaced by the batch — cuda_gdn.cu:2513's
// (NV, n*Hv) flattened.
void GdnDecodeKernel(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                     const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                     const Tensor* state_idx, const GdnArgs& args) {
  const int64_t batch = q_in.shape[0], hv = state.shape[1], dv = state.shape[2];
  if (batch == 0 || hv == 0 || dv == 0) return;
  Binder bind;
  GdnRecurrenceParams p{};
  uint32_t spec[2] = {0, 0};
  if (!GdnRecurrenceCommon(out, q_in, k, v, g, beta, state, bind, p, spec, args.scale)) {
    auto next = reinterpret_cast<GdnDecodeFn>(
        GetOpFallback(OpId::kGdnDecode, DeviceType::kVULKAN, kNativeProviderName));
    next(q, out, q_in, k, v, g, beta, state, state_idx, args);
    return;
  }
  // Binding 11 is always written: a descriptor a shader statically uses must be
  // valid even on the path that never reads it. With has_idx == 0 it aliases the
  // state buffer and is dead.
  if (state_idx != nullptr) {
    p.has_idx = 1;
    p.meta_off = bind.AddU32Only(*state_idx, "gdn_decode: state_idx");
  } else {
    p.meta_off = bind.AddU32Only(state, "gdn_decode: state");
  }
  Go("vt_gdn_decode", bind, p, static_cast<uint32_t>(batch * hv * p.nv), spec, 2);
}

// ---------------------------------------------------------------------------
// The FUSED FULL-ATTENTION PREAMBLE (BACKEND-VULKAN-QKNORM).
//
// WHY THIS OP AND NOT ANOTHER. With the GDN family native, a 27B decode step was
// MEASURED at ~30 command-buffer flushes per token of which ~28 were
// reference-tier: op_provider.cpp drains the whole recorded batch (submit +
// blocking fence) before it can hand a host kernel device memory, so every
// reference-tier op costs a full round trip regardless of how little arithmetic
// it does. The declines named exactly three ops, and only this one runs in
// DECODE on every step: kCausalConv1dFwd is prefill-only and kRopeCosSinCache is
// deliberately host-side (the double-precision table; see vt_rope_from_cache.comp).
// Qwen3.6-27B has 64 layers of which 48 are linear-attention, so this fires 16
// times per token.
//
// NO DECLINE PATH. Every dtype the op contract admits (ops.cpp:1524-1541) is a
// specialization axis here, the packed row strides are read from stride[0], and
// rotary_dim < Dh is the ordinary case rather than a special one — so there is no
// shape this kernel has to hand back. cpu_ops.cpp:956-1010 is the oracle.
void AttnQkNormRopeGateKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                              const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                              const Tensor& k_norm, const Tensor& cos_sin,
                              const RmsNormArgs& na, const RopeArgs& ra) {
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  const int64_t hkv = k_out.shape[1];
  if (t == 0 || hq + hkv == 0 || dh == 0) return;
  Binder bind;
  // Binding order IS the descriptor order: the shader declares qgate, kf, then
  // the three outputs as dtype-erased PAIRS, then the three f32-by-contract
  // operands as single bindings.
  const uint32_t qg_off = bind.Add(qgate, "attn_qk_norm_rope_gate: qgate");
  const uint32_t kf_off = bind.Add(kf, "attn_qk_norm_rope_gate: kf");
  const uint32_t qo_off = bind.Add(q_out, "attn_qk_norm_rope_gate: q_out");
  const uint32_t ko_off = bind.Add(k_out, "attn_qk_norm_rope_gate: k_out");
  const uint32_t go_off = bind.Add(gate_out, "attn_qk_norm_rope_gate: gate_out");
  const uint32_t qn_off = bind.AddU32Only(q_norm, "attn_qk_norm_rope_gate: q_norm");
  const uint32_t kn_off = bind.AddU32Only(k_norm, "attn_qk_norm_rope_gate: k_norm");
  const uint32_t cs_off = bind.AddU32Only(cos_sin, "attn_qk_norm_rope_gate: cos_sin");
  // Ascending constantID order: the shared qgate/kf dtype, the shared q/k out
  // dtype, the gate out dtype (its own axis — the FA-2 prefill combo keeps the
  // gate f32 while q/k are bf16).
  const uint32_t spec[3] = {DtypeCode(qgate.dtype), DtypeCode(q_out.dtype),
                            DtypeCode(gate_out.dtype)};
  AttnQkNormRopeGateParams p{static_cast<uint32_t>(hq),
                             static_cast<uint32_t>(hkv),
                             static_cast<uint32_t>(dh),
                             static_cast<uint32_t>(ra.rotary_dim),
                             static_cast<uint32_t>(ra.rotary_dim / 2),
                             static_cast<uint32_t>(qgate.stride[0]),
                             static_cast<uint32_t>(kf.stride[0]),
                             na.gemma ? 1u : 0u,
                             qg_off,
                             kf_off,
                             qo_off,
                             ko_off,
                             go_off,
                             qn_off,
                             kn_off,
                             cs_off,
                             na.eps};
  // ONE WORKGROUP PER (token, head) over Hq + Hkv slots — CUDA's dim3(t, hq+hkv)
  // grid (cuda_ops.cu:1394) flattened. NOT FlatGroupCount: the workgroup
  // tree-reduces one head row's mean square.
  Go("vt_attn_qk_norm_rope_gate", bind, p, static_cast<uint32_t>(t * (hq + hkv)), spec, 3);
}

// ---------------------------------------------------------------------------
// BACKEND-VULKAN-KEEPQUANT — the GGUF keep-quant tier, by CPU FALL-THROUGH.
//
// WHY A HOST KERNEL IS THE CORRECT VULKAN REGISTRATION (for now). The B60 is
// integrated: VulkanContext allocates every tensor from HOST_VISIBLE |
// HOST_COHERENT memory and VulkanBackend::DeviceMemoryIsHostAddressable()
// answers true unconditionally, so the host vec_dot kernels dereference the
// SAME bytes the rest of the graph dispatches shaders over. That is precisely
// the property the portable reference tier already relies on to run the CPU
// kMatmulBTQuant lazily at GetOp time (op_provider.cpp
// MaybeInstallReferenceTier) — this registration makes that arrangement
// EAGER, so it also flips vt::OpRegistered(kMatmulBTQuant, kVULKAN), which
// EXCLUDES the reference tier by design (op_provider.cpp OpRegistered:
// "fallback, not native"). GgufQuantComputeAvailable()
// (gguf_keep_quant.cpp:75) reads exactly that predicate to decide the GGUF
// loader's keep-quant DEFAULT; without a native entry the loader expands
// every TQ1_0/TQ2_0/K-quant block to bf16 at load and maple's ~4 GB of
// ternary weights become ~32 GB of anonymous bf16 — OOM on the 31 GB box
// before the first token. With it, blocks stay resident and the win is
// MEMORY, not speed: the compute itself is the plain CPU kernel, SLOW, which
// is why the body drains any pending device batch FIRST — the same contract
// ResolveFallback discharges for reference-tier ops (a deferred-submission
// backend must not hand a host kernel activations the GPU has not written),
// discharged here on the NATIVE path too because the native kernel here IS a
// host kernel.
//
// Mirrors cuda_quant_dot.cu:1835 (CUDA falls through to GetOp(...,kCPU) for
// dtypes its GPU kernel lacks; on unified memory that fall-through is free).
// Local activations-reader (mirrors cpu_quant_gemm.cpp LoadActF32): the keep-quant
// native path quantizes the raw activation to Q8_K before dispatch.
static float LoadActF32Local(const Tensor& t, int64_t i) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[i];
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[i]);
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[i]);
    default: VT_CHECK(false, "vulkan tq2: unsupported activation dtype");
  }
  return 0.0f;
}

// BACKEND-VULKAN-KEEPQUANT: NATIVE TQ2_0 DECODE GEMV. For the TQ2_0 weight
// dtype at decode (M==1) we dispatch vt_matmul_bt_tq2 instead of the scalar
// CPU fall-through, which was pathologically slow (minutes per prompt, no fence to
// time out — the "hang" behind the earlier box outages). The activation is
// quantized to Q8_K (TQ2_0's vec_dot_type), then a workgroup-per-output
// element kernel reads BlockTQ2_0 weights against it. Returns true if handled.

// Persistent host->GPU scratch for TQ2_0 keep-quant activation rows. Growing a
// buffer from scratch every dispatch (vkCreateBuffer + vkAllocateMemory +
// vkDestroyBuffer) is the dominant host cost at 600 dispatches/token. Keep one
// allocation that only (re)allocates when a request outgrows it. Single-threaded
// engine use; mutex-guarded for safety.
static void* Tq2Scratch(size_t need, void** out_buffer) {
  static std::mutex mu;
  static void* buf = nullptr, *mem = nullptr;
  static void* map = nullptr;
  static size_t cap = 0;
  std::lock_guard<std::mutex> g(mu);
  auto& ctx = VulkanContext::Get();
  if (need > cap) {
    if (buf != nullptr) ctx.FreeBuffer(buf, mem);
    size_t nc = need; nc += nc / 2;
    if (nc < 4096) nc = 4096;
    map = ctx.AllocBuffer(nc, &buf, &mem);
    cap = nc;
  }
  *out_buffer = buf;
  return map;
}

static bool TryNativeTQDecode(Queue& q, Tensor& out, const Tensor& a,
                              const Tensor& b) {
  (void)q;
  const int64_t m = a.shape[0], k = a.shape[1], n = b.shape[0];
  const bool is_tq2 = (b.dtype == DType::kTQ2_0);
  const bool is_tq1 = (b.dtype == DType::kTQ1_0);
  if ((!is_tq2 && !is_tq1) || m <= 0 || k <= 0 || k % 256 != 0 ||
      b.repacked) {
    return false;
  }
  const int64_t nb = k / 256;
  const char* dev_shader = is_tq2 ? "vt_matmul_bt_tq2_dev"
                                  : "vt_matmul_bt_tq1_0_dev";
  const char* host_shader = is_tq2 ? "vt_matmul_bt_tq2" : "vt_matmul_bt_tq1_0";
  const DType wdt = b.dtype;

  // GPU-side quantize path (Phase 2): when the activation is f32 or bf16 on the
  // device, dispatch the _dev shader which quantizes Q8_K INSIDE the shader.
  // No host read, no FlushBatch — eliminates the per-dispatch drain.
  if (a.dtype == DType::kF32 || a.dtype == DType::kBF16) {
    auto& ctx = VulkanContext::Get();
    std::vector<void*> buffers;
    uint32_t a_off = 0, w_off = 0, out_off = 0;
    {
      Resolved r = Resolve(a.data, "matmul-bt-quant-dev: act");
      buffers.push_back(r.buffer); buffers.push_back(r.buffer);
      a_off = r.offset;
    }
    {
      Resolved r = Resolve(b.data, "matmul-bt-quant-dev: weight");
      buffers.push_back(r.buffer); buffers.push_back(r.buffer);
      w_off = r.offset;
    }
    {
      Resolved r = Resolve(out.data, "matmul-bt-quant-dev: out");
      buffers.push_back(r.buffer); buffers.push_back(r.buffer);
      out_off = r.offset;
    }
    MatmulBTQuantTQ2Params cp{static_cast<uint32_t>(n), static_cast<uint32_t>(m),
                              static_cast<uint32_t>(nb), a_off, w_off, out_off};
    const uint32_t spec[2] = {DtypeCode(out.dtype), DtypeCode(a.dtype)};
    ctx.Dispatch(dev_shader, buffers.data(),
                 static_cast<uint32_t>(buffers.size()), &cp, sizeof(cp),
                 static_cast<uint32_t>((n + 3) / 4), spec, 2);
    return true;
  }

  // Host-quantize fallback (f16 activation or when the device path is disabled):
  // reads the activation from mapped memory, quantizes to Q8_K on the host, and
  // dispatches the original shader. Requires FlushBatch for coherence.
  auto& ctx = VulkanContext::Get();
  ctx.FlushBatch("matmul-bt-quant native");

  vt::cpu::FromFloatFn from_float = vt::cpu::BlockFromFloat(DType::kQ8_K);
  if (from_float == nullptr) return false;
  // Quantize ALL m activation rows to Q8_K (the [m, k] prompt/token matrix).
  std::vector<uint8_t> act(
      vt::cpu::QuantActScratchBytes(wdt, m, k));
  std::vector<float> row(static_cast<size_t>(k));
  const size_t qrow =
      static_cast<size_t>(vt::RowSizeBytes(DType::kQ8_K, k));
  for (int64_t r = 0; r < m; ++r) {
    for (int64_t p = 0; p < k; ++p)
      row[static_cast<size_t>(p)] = LoadActF32Local(a, r * k + p);
    from_float(row.data(), act.data() + qrow * static_cast<size_t>(r), k);
  }

  void* vbuf = nullptr;
  void* mapped = Tq2Scratch(std::max(act.size(), (size_t)4), &vbuf);
  std::memcpy(mapped, act.data(), act.size());

  std::vector<void*> buffers;
  uint32_t a_off = 0, w_off = 0, out_off = 0;
  buffers.push_back(vbuf); buffers.push_back(vbuf);
  {
    Resolved r = Resolve(b.data, "matmul-bt-quant: weight");
    buffers.push_back(r.buffer); buffers.push_back(r.buffer);
    w_off = r.offset;
  }
  {
    Resolved r = Resolve(out.data, "matmul-bt-quant: out");
    buffers.push_back(r.buffer); buffers.push_back(r.buffer);
    out_off = r.offset;
  }

  MatmulBTQuantTQ2Params cp{static_cast<uint32_t>(n), static_cast<uint32_t>(m),
                            static_cast<uint32_t>(nb), a_off, w_off, out_off};
  const uint32_t spec[1] = {DtypeCode(out.dtype)};
  ctx.Dispatch(host_shader, buffers.data(),
               static_cast<uint32_t>(buffers.size()), &cp, sizeof(cp),
               static_cast<uint32_t>(n), spec, 1);
  return true;
}

void MatmulBTQuantKernelVulkan(Queue& q, Tensor& out, const Tensor& a,
                               const Tensor& b) {
  if (TryNativeTQDecode(q, out, a, b)) return;
  // Drain before the host reads: same rationale as vulkan_backend.cpp
  // FlushPending ("reference-tier") — see the block comment above.
  VulkanContext::Get().FlushBatch("matmul-bt-quant cpu-fallthrough");
  reinterpret_cast<MatmulFn>(GetOp(OpId::kMatmulBTQuant, DeviceType::kCPU))(
      q, out, a, b);
}


// VK4 keep-quant: native grouped (per-token expert) TQ2_0 matmul. Reuses the
// M-loop arithmetic but each output row <row> selects expert eids[row] whose weight
// slice is weight[e][N][K] (TQ2_0), and its activation row (row, or 0 when
// broadcasting a single shared hidden across experts) quantized to Q8_K on the host.
static bool TryNativeTQGrouped(Queue& q, Tensor& out, const Tensor& act,
                               const Tensor& weight, const Tensor& expert_ids) {
  (void)q;
  const int64_t P = out.shape[0], N = out.shape[1], K = act.shape[1];
  const bool bcast = (act.shape[0] == 1 && P > 1);
  const bool is_tq2 = (weight.dtype == DType::kTQ2_0);
  const bool is_tq1 = (weight.dtype == DType::kTQ1_0);
  if ((!is_tq2 && !is_tq1) || P <= 0 || K <= 0 || K % 256 != 0 ||
      weight.repacked) {
    return false;
  }
  const int64_t nb = K / 256;
  const int64_t E = weight.shape[0] / N;  // rank-2 weight is [E*N, K]
  const char* dev_shader = is_tq2 ? "vt_matmul_bt_tq2_grouped_dev"
                                  : "vt_matmul_bt_tq1_0_grouped_dev";
  const char* host_shader = is_tq2 ? "vt_matmul_bt_tq2_grouped"
                                   : "vt_matmul_bt_tq1_0_grouped";
  const DType wdt = weight.dtype;

  // GPU-side quantize path (Phase 2): when the activation is f32 or bf16 on the
  // device, dispatch the _dev shader which quantizes Q8_K INSIDE the shader.
  // No host read, no FlushBatch, no scratch buffer — the dispatch batches
  // naturally with surrounding ops, eliminating the per-dispatch drain that was
  // the maple MoE bottleneck. Bit-exact vs the host-quantize path because the
  // shader's Q8_K quantize matches cpu_quant_act.cpp QuantizeRowQ8_K exactly
  // (same signed-extremum max, same iscale = -127/max, same roundEven).
  if (act.dtype == DType::kF32 || act.dtype == DType::kBF16) {
    auto& ctx = VulkanContext::Get();
    std::vector<void*> buffers;
    uint32_t a_off = 0, w_off = 0, eid_off = 0, out_off = 0;
    {
      Resolved r = Resolve(act.data, "matmul-bt-quant-grouped-dev: act");
      buffers.push_back(r.buffer); buffers.push_back(r.buffer);
      a_off = r.offset;
    }
    {
      Resolved r = Resolve(weight.data, "matmul-bt-quant-grouped-dev: weight");
      buffers.push_back(r.buffer); buffers.push_back(r.buffer);
      w_off = r.offset;
    }
    {
      Resolved r = Resolve(expert_ids.data, "matmul-bt-quant-grouped-dev: eids");
      buffers.push_back(r.buffer);
      eid_off = r.offset;
    }
    {
      Resolved r = Resolve(out.data, "matmul-bt-quant-grouped-dev: out");
      buffers.push_back(r.buffer); buffers.push_back(r.buffer);
      out_off = r.offset;
    }
    const uint32_t spec[2] = {DtypeCode(out.dtype), DtypeCode(act.dtype)};
    // Chunk over P to prevent hangcheck timeouts at large prefill sizes.
    // Each MoE workgroup processes ALL P rows, so per-WG time scales with P.
    // Host-side chunking: adjust a_off, eid_off, out_off per chunk so the
    // shader (unchanged) processes only [p_start, p_start+p_count) rows.
    // For gather_k>0, p_start must be aligned to gather_k so that the
    // shader's row/gather_k maps to the correct activation row. kMaxPPerChunkMoe
    // is a multiple of any reasonable top_k, so alignment is automatic.
    const uint32_t total_P = static_cast<uint32_t>(P);
    const uint32_t wg = static_cast<uint32_t>((N + 3) / 4);
    const size_t act_elem = vt::SizeOf(act.dtype);
    const size_t out_elem = vt::SizeOf(out.dtype);
    for (uint32_t p_start = 0; p_start < total_P; p_start += kMaxPPerChunkMoe) {
      const uint32_t p_count = std::min(kMaxPPerChunkMoe, total_P - p_start);
      // Flush before each MoE chunk: each workgroup processes ALL p_count rows,
      // and the kernel also does on-device quantization + SwiGLU, so per-chunk
      // GPU time is higher than a pure GEMM estimate. Flushing isolates each
      // chunk in its own command buffer, preventing hangcheck timeouts.
      if (p_start > 0) ctx.FlushBatch("moe-chunk");
      // Adjust offsets: shader loops row 0..p_count-1, so shift offsets to
      // point at the correct slice of activation, expert_ids, and output.
      // In bcast mode (act.shape[0]==1), all rows share activation row 0, so
      // a_off must NOT be shifted — the activation tensor is a single row.
      const uint32_t a_chunk = bcast
          ? a_off
          : a_off + p_start * static_cast<uint32_t>(K) *
                        static_cast<uint32_t>(act_elem);
      const uint32_t eid_chunk = eid_off + p_start * sizeof(uint32_t);
      const uint32_t out_chunk = out_off + p_start * static_cast<uint32_t>(N) *
                                             static_cast<uint32_t>(out_elem);
      MatmulBTQuantTQ2GroupedParams cp{p_count,
                                static_cast<uint32_t>(N),
                                static_cast<uint32_t>(nb),
                                static_cast<uint32_t>(E),
                                static_cast<uint32_t>(bcast ? 1 : 0),
                                a_chunk, w_off, eid_chunk, out_chunk};
      ctx.Dispatch(dev_shader, buffers.data(),
                   static_cast<uint32_t>(buffers.size()), &cp, sizeof(cp),
                   wg, spec, 2);
    }
    return true;
  }

  // Host-quantize fallback (f16 activation or when the device path is disabled):
  // reads the activation from mapped memory, quantizes to Q8_K on the host, and
  // dispatches the original grouped shader. Requires FlushBatch for coherence.
  auto& ctx = VulkanContext::Get();
  ctx.FlushBatch("matmul-bt-quant-grouped native");

  vt::cpu::FromFloatFn from_float = vt::cpu::BlockFromFloat(DType::kQ8_K);
  if (from_float == nullptr) return false;
  const int64_t nrows = bcast ? 1 : P;
  std::vector<uint8_t> act_q(
      vt::cpu::QuantActScratchBytes(wdt, nrows, K));
  std::vector<float> row(static_cast<size_t>(K));
  const size_t qrow = static_cast<size_t>(vt::RowSizeBytes(DType::kQ8_K, K));
  for (int64_t r = 0; r < nrows; ++r) {
    for (int64_t p = 0; p < K; ++p)
      row[static_cast<size_t>(p)] = LoadActF32Local(act, r * K + p);
    from_float(row.data(), act_q.data() + qrow * static_cast<size_t>(r), K);
  }

  void* vbuf = nullptr;
  void* mapped = Tq2Scratch(std::max(act_q.size(), (size_t)4), &vbuf);
  std::memcpy(mapped, act_q.data(), act_q.size());

  std::vector<void*> buffers;
  uint32_t a_off = 0, w_off = 0, eid_off = 0, out_off = 0;
  buffers.push_back(vbuf); buffers.push_back(vbuf);
  {
    Resolved r = Resolve(weight.data, "matmul-bt-quant-grouped: weight");
    buffers.push_back(r.buffer); buffers.push_back(r.buffer);
    w_off = r.offset;
  }
  {
    Resolved r = Resolve(expert_ids.data, "matmul-bt-quant-grouped: eids");
    buffers.push_back(r.buffer);
    eid_off = r.offset;
  }
  {
    Resolved r = Resolve(out.data, "matmul-bt-quant-grouped: out");
    buffers.push_back(r.buffer); buffers.push_back(r.buffer);
    out_off = r.offset;
  }

  MatmulBTQuantTQ2GroupedParams cp{static_cast<uint32_t>(P), static_cast<uint32_t>(N),
                            static_cast<uint32_t>(nb), static_cast<uint32_t>(E),
                            static_cast<uint32_t>(bcast ? 1 : 0),
                            a_off, w_off, eid_off, out_off};
  const uint32_t spec[1] = {DtypeCode(out.dtype)};
  ctx.Dispatch(host_shader, buffers.data(),
               static_cast<uint32_t>(buffers.size()), &cp, sizeof(cp),
               static_cast<uint32_t>(N), spec, 1);
  return true;
}

void MatmulBTQuantGroupedKernelVulkan(Queue& q, Tensor& out,
                                      const Tensor& act, const Tensor& weight,
                                      const Tensor& expert_ids) {
  if (TryNativeTQGrouped(q, out, act, weight, expert_ids)) return;
  VulkanContext::Get().FlushBatch("matmul-bt-quant-grouped cpu-fallthrough");
  reinterpret_cast<MatmulBTQuantGroupedFn>(
      GetOp(OpId::kMatmulBTQuantGrouped, DeviceType::kCPU))(q, out, act, weight,
                                                            expert_ids);
}

// VK4 keep-quant Phase 2: Vulkan provider for kMoeGateUpSwiGLUGrouped. Fuses
// gate+up grouped GEMMs + clamped SwiGLU into ONE dispatch with on-device Q8_K
// quantize (vt_moe_gate_up_swiglu_grouped_tq2.comp). No FlushBatch — the shader
// reads the bf16 activation straight from the device buffer, so the dispatch
// batches with surrounding ops and the whole MoE expert path runs drain-free.
// Bit-exact vs the CPU golden (cpu_quant_gemm.cpp MoeGateUpSwiGLUGroupedKernel)
// because the shader's Q8_K quantize matches QuantizeRowQ8_K exactly and the
// vec-dot / SwiGLU math is the same integer/f32 sequence.
static bool TryNativeMoeGateUpSwiGLUGroupedTQ(
    Queue& q, Tensor& out, const Tensor& act, const Tensor& gate_w,
    const Tensor& up_w, const Tensor& expert_ids, float limit) {
  (void)q;
  const int64_t P = out.shape[0], N = out.shape[1], K = act.shape[1];
  const bool bcast = (act.shape[0] == 1 && P > 1);
  // Prefill gather: activation is [T, H], output is [P, H] where P = T * top_k.
  // Each output row p reads activation row p / top_k. Infer top_k from shapes.
  const uint32_t gather_k =
      (!bcast && act.shape[0] > 1 && act.shape[0] < P && P % act.shape[0] == 0)
          ? static_cast<uint32_t>(P / act.shape[0])
          : 0u;
  const bool is_tq2 = (gate_w.dtype == DType::kTQ2_0 && up_w.dtype == DType::kTQ2_0);
  const bool is_tq1 = (gate_w.dtype == DType::kTQ1_0 && up_w.dtype == DType::kTQ1_0);
  if ((!is_tq2 && !is_tq1) ||
      P <= 0 || K <= 0 || K % 256 != 0 ||
      (act.dtype != DType::kBF16 && act.dtype != DType::kF32) ||
      gate_w.repacked || up_w.repacked) {
    return false;
  }
  const char* shader = is_tq2 ? "vt_moe_gate_up_swiglu_grouped_tq2"
                              : "vt_moe_gate_up_swiglu_grouped_tq1_0";
  const int64_t nb = K / 256;
  const int64_t E = gate_w.shape[0] / N;

  auto& ctx = VulkanContext::Get();
  std::vector<void*> buffers;
  uint32_t a_off = 0, gw_off = 0, uw_off = 0, eid_off = 0, out_off = 0;
  {
    Resolved r = Resolve(act.data, "moe-gate-up-swiglu-grouped: act");
    buffers.push_back(r.buffer); buffers.push_back(r.buffer);
    a_off = r.offset;
  }
  {
    Resolved r = Resolve(gate_w.data, "moe-gate-up-swiglu-grouped: gate_w");
    buffers.push_back(r.buffer); buffers.push_back(r.buffer);
    gw_off = r.offset;
  }
  {
    Resolved r = Resolve(up_w.data, "moe-gate-up-swiglu-grouped: up_w");
    buffers.push_back(r.buffer); buffers.push_back(r.buffer);
    uw_off = r.offset;
  }
  {
    Resolved r = Resolve(expert_ids.data, "moe-gate-up-swiglu-grouped: eids");
    buffers.push_back(r.buffer);
    eid_off = r.offset;
  }
  {
    Resolved r = Resolve(out.data, "moe-gate-up-swiglu-grouped: out");
    buffers.push_back(r.buffer); buffers.push_back(r.buffer);
    out_off = r.offset;
  }
  const uint32_t spec[2] = {DtypeCode(act.dtype), DtypeCode(out.dtype)};
  const uint32_t wg = static_cast<uint32_t>((N + 3) / 4);
  // Chunk over P to prevent hangcheck timeouts at large prefill sizes.
  // Host-side chunking: adjust a_off, eid_off, out_off per chunk so the
  // shader (unchanged) processes only [p_start, p_start+p_count) rows.
  // For gather_k>0, p_start must be aligned to gather_k so that the shader's
  // row/gather_k maps to the correct activation row. kMaxPPerChunkMoe is a
  // multiple of any reasonable top_k, so alignment is automatic.
  const uint32_t total_P = static_cast<uint32_t>(P);
  const size_t act_elem = vt::SizeOf(act.dtype);
  const size_t out_elem = vt::SizeOf(out.dtype);
  // Align chunk to gather_k so row/gather_k maps correctly with a_off adjustment.
  const uint32_t chunk = (gather_k > 0u)
      ? std::max(gather_k, kMaxPPerChunkMoe / gather_k * gather_k)
      : kMaxPPerChunkMoe;
  for (uint32_t p_start = 0; p_start < total_P; p_start += chunk) {
    const uint32_t p_count = std::min(chunk, total_P - p_start);
    // Flush before each MoE chunk: gate+up does 2x the work (2 GEMMs + SwiGLU),
    // so per-chunk GPU time is higher. Flushing isolates each chunk.
    if (p_start > 0) ctx.FlushBatch("moe-gate-up-chunk");
    // Adjust offsets for this chunk. For gather_k>0, the activation row is
    // row/gather_k, so shift a_off by (p_start/gather_k) rows. In bcast mode
    // (act.shape[0]==1), all rows share activation row 0, so a_off must NOT
    // be shifted.
    const uint32_t act_row_off = bcast ? 0u
        : (gather_k > 0u) ? (p_start / gather_k) : p_start;
    const uint32_t a_chunk = a_off + act_row_off * static_cast<uint32_t>(K) *
                                      static_cast<uint32_t>(act_elem);
    const uint32_t eid_chunk = eid_off + p_start * sizeof(uint32_t);
    const uint32_t out_chunk = out_off + p_start * static_cast<uint32_t>(N) *
                                           static_cast<uint32_t>(out_elem);
    MoeGateUpSwiGLUGroupedTQ2Params cp{
        p_count, static_cast<uint32_t>(N),
        static_cast<uint32_t>(nb), static_cast<uint32_t>(E),
        static_cast<uint32_t>(bcast ? 1 : 0), gather_k,
        a_chunk, gw_off, uw_off, eid_chunk, out_chunk, limit};
    ctx.Dispatch(shader, buffers.data(),
                 static_cast<uint32_t>(buffers.size()), &cp, sizeof(cp),
                 wg, spec, 2);
  }
  return true;
}

void MoeGateUpSwiGLUGroupedKernelVulkan(
    Queue& q, Tensor& out, const Tensor& act, const Tensor& gate_w,
    const Tensor& up_w, const Tensor& expert_ids, float limit) {
  if (TryNativeMoeGateUpSwiGLUGroupedTQ(q, out, act, gate_w, up_w, expert_ids,
                                         limit)) return;
  VulkanContext::Get().FlushBatch("moe-gate-up-swiglu-grouped cpu-fallthrough");
  reinterpret_cast<MoeGateUpSwiGLUGroupedFn>(
      GetOp(OpId::kMoeGateUpSwiGLUGrouped, DeviceType::kCPU))(
          q, out, act, gate_w, up_w, expert_ids, limit);
}

struct Registrar {
  Registrar() {
    // Same guard as the backend registrar: a Vulkan-enabled build on a host with
    // no loader or no conformant device registers nothing, so GetOp throws its
    // normal not-registered error.
    if (!VulkanContext::Available()) return;
    // static_cast against the ops.h aliases ties every kernel signature to the
    // registration contract at COMPILE time (the cpu_ops.cpp idiom).
    RegisterOp(OpId::kQkvSplit, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<QkvSplitFn>(&QkvSplitKernel)));
    RegisterOp(OpId::kRopeFromCache, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<RopeFromCacheFn>(&RopeFromCacheKernel)));
    if (!VkOpDisabled("kReshapeAndCache"))     RegisterOp(OpId::kReshapeAndCache, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<ReshapeAndCacheFn>(&ReshapeAndCacheKernel)));
    if (!VkOpDisabled("kPagedAttention"))     RegisterOp(OpId::kPagedAttention, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<PagedAttentionFn>(&PagedAttentionKernel)));
    RegisterOp(OpId::kEmbedding, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingKernel)));
    RegisterOp(OpId::kGreedyArgmax, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<GreedyArgmaxFn>(&GreedyArgmaxKernel)));
    RegisterOp(OpId::kMatmul, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulGeneric<false>)));
    RegisterOp(OpId::kMatmulBT, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulGeneric<true>)));
    RegisterOp(OpId::kAdd, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<AddFn>(&AddKernel)));
    RegisterOp(OpId::kRelu, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<ReluFn>(&ReluKernel)));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernel)));
    RegisterOp(OpId::kCastBf16, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<CastBf16Fn>(&CastKernel)));
    RegisterOp(OpId::kCastF32, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<CastF32Fn>(&CastKernel)));
    // BACKEND-VULKAN-EXL3 V1 (#2530). The THIRD sibling, and the same kernel: the
    // dtype pair is a specialization constant, `DtypeCode` already maps
    // DType::kF16 to VT_DT_F16, and vt_common.glsl's f16 codec is an integer
    // transcription of src/vt/dtype.cpp -- so this registration adds a shader
    // variant, not a shader. It was missing while its two siblings were present,
    // which is why an EXL3 checkpoint's f16 activation cast was one of the two
    // ops S1 measured falling to the portable CPU tier on a Vulkan queue.
    //
    // THE SHARED LIMITATION IS NAMED RATHER THAN INHERITED IN SILENCE. `vt::CastF16`
    // and `vt::CastBf16` both TOLERATE a packed strided input (the merged-QKV view)
    // whose rows are dense while the row stride spans a parent tensor, and
    // `CastKernel` above indexes FLAT from one byte offset, so it would read such an
    // input as contiguous. That is pre-existing for the two siblings and is not
    // widened here; src/vt/ops.cpp says the merge is CUDA-only, and
    // .agents/specs/backend-vulkan-exl3.md `## Owed` carries it.
    RegisterOp(OpId::kCastF16, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<CastF16Fn>(&CastKernel)));
    // BACKEND-VULKAN-EXL3 V2 (#2530). The other op S1 measured on the reference
    // tier, and the whole EXL3 forward. `kExl3HadR128` is deliberately NOT
    // registered: the shader that performs it ships here as steps 1 and 3 of this
    // GEMM, but no dense forward path calls that op, and a registration nothing
    // reaches is what `.agents/reachability.md` exists to prevent. The spec's
    // `## Owed` carries it.
    RegisterOp(OpId::kExl3Gemm, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<Exl3GemmFn>(&Exl3GemmKernelVulkan)));
    RegisterOp(OpId::kLayerNorm, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<LayerNormFn>(&LayerNormKernel)));
    RegisterOp(OpId::kRmsNorm, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernel)));
    RegisterOp(OpId::kFusedChain, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<FusedChainFn>(&FusedChainKernel)));
    // BACKEND-VULKAN-GDN: the GDN glue family, now including kCausalConv1dFwd
    // (the prefill conv), registered after the update op below.
    RegisterOp(OpId::kSigmoidGateBf16, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<SigmoidGateBf16Fn>(&SigmoidGateBf16Kernel)));
    RegisterOp(OpId::kRmsNormGated, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<RmsNormGatedFn>(&RmsNormGatedKernel)));
    RegisterOp(OpId::kGdnStateGather, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<GdnStateGatherFn>(&GdnStateGatherKernel)));
    RegisterOp(OpId::kGdnStateScatter, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<GdnStateScatterFn>(&GdnStateScatterKernel)));
    RegisterOp(
        OpId::kCausalConv1dUpdate, DeviceType::kVULKAN,
        reinterpret_cast<void*>(static_cast<CausalConv1dUpdateFn>(&CausalConv1dUpdateKernel)));
    // The PREFILL conv. VT_VULKAN_CONV_FWD=0 leaves it on the portable CPU
    // reference tier, where it lived before this kernel existed, so its effect
    // can be attributed with a same-binary A/B rather than across two builds --
    // the same shape as VT_VULKAN_COOPMAT.
    {
      const char* conv_env = std::getenv("VT_VULKAN_CONV_FWD");
      if (conv_env == nullptr || conv_env[0] != '0') {
        RegisterOp(OpId::kCausalConv1dFwd, DeviceType::kVULKAN,
                   reinterpret_cast<void*>(static_cast<CausalConv1dFwdFn>(&CausalConv1dFwdKernel)));
      }
    }
    RegisterOp(OpId::kGdnPostConv, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<GdnPostConvFn>(&GdnPostConvKernel)));
    // BACKEND-VULKAN-GDN-CORE: the two recurrences.
    RegisterOp(OpId::kGdnPrefill, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<GdnPrefillFn>(&GdnPrefillKernel)));
    RegisterOp(OpId::kGdnDecode, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<GdnDecodeFn>(&GdnDecodeKernel)));
    // BACKEND-VULKAN-QKNORM: the fused full-attention preamble, the last
    // per-decode-step reference-tier op on the 27B.
    RegisterOp(
        OpId::kAttnQkNormRopeGate, DeviceType::kVULKAN,
        reinterpret_cast<void*>(static_cast<AttnQkNormRopeGateFn>(&AttnQkNormRopeGateKernel)));
    // VK4 (B60 maple row): the four ops the maple graph was still draining to
    // the host for. The rotary TABLE BUILD joins its apply half; see
    // RopeCosSinCacheKernel above for why the old "no f64 in GLSL" note retired.
    if (!VkOpDisabled("kRopeCosSinCache"))     RegisterOp(
        OpId::kRopeCosSinCache, DeviceType::kVULKAN,
        reinterpret_cast<void*>(static_cast<RopeCosSinCacheFn>(&RopeCosSinCacheKernel)));
    // BACKEND-VULKAN-KEEPQUANT: the GGUF compute-in-quant GEMMs, by CPU
    // fall-through (see MatmulBTQuantKernelVulkan above). This is what makes
    // GgufQuantComputeAvailable() true on Vulkan, so a TQ1_0/TQ2_0/K-quant GGUF
    // keeps its blocks resident instead of expanding to bf16 at load (the maple
    // 32 GB OOM). The grouped twin rides along for MoE models.
    RegisterOp(OpId::kMatmulBTQuant, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTQuantKernelVulkan)));
    if (!VkOpDisabled("kRopeNeox"))     RegisterOp(OpId::kRopeNeox, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<RopeFn>(&RopeNeoxKernel)));
    if (!VkOpDisabled("kMoeCombine"))     RegisterOp(
        OpId::kMoeCombine, DeviceType::kVULKAN,
        reinterpret_cast<void*>(static_cast<MoeCombineFn>(&MoeCombineKernelVulkan)));
    if (!VkOpDisabled("kMoeRouterTopK"))     RegisterOp(OpId::kMoeRouterTopK, DeviceType::kVULKAN,
               reinterpret_cast<void*>(
                   static_cast<MoeRouterTopKFn>(&MoeRouterTopKKernelVulkan)));
    RegisterOp(OpId::kMatmulBTQuantGrouped, DeviceType::kVULKAN,
               reinterpret_cast<void*>(
                   static_cast<MatmulBTQuantGroupedFn>(&MatmulBTQuantGroupedKernelVulkan)));
    if (!VkOpDisabled("kMoeGateUpSwiGLUGrouped"))
      RegisterOp(OpId::kMoeGateUpSwiGLUGrouped, DeviceType::kVULKAN,
                 reinterpret_cast<void*>(static_cast<MoeGateUpSwiGLUGroupedFn>(
                     &MoeGateUpSwiGLUGroupedKernelVulkan)));
  }
} registrar;

}  // namespace

// OUTPUT COLUMNS PER LANE for the scalar matmul tactic (VK-G). 4 by default;
// VT_VULKAN_MATMUL_NCOLS=1|2|4|8 selects an arm.
//
// WHY 4 AND NOT 8, MEASURED on GB10 over three interleaved triples at 27B decode
// (ms/call for the lm_head, k=5120 n=248320): ncols 1 = 12.55, ncols 4 = 11.63,
// ncols 8 = 12.81. Blocking is a TRADE, not a monotone win -- it buys a longer
// contiguous run per workgroup and pays for it in workgroups. At 8 the dispatch
// is only ceil(248320/1024) = 243 workgroups of 128 lanes, about 31k threads,
// and the device runs out of work to hide memory latency with faster than the
// longer run buys back.
//
// The lever exists for ONE reason, the same one the coopmat and GEMV levers cite:
// a SAME-BINARY A/B. The factor is a specialization constant, so every arm is the
// same committed SPIR-V and two runs differ in exactly one thing; comparing two
// BUILDS is what produced a false 1.2x reading for the subgroup tactic earlier in
// this campaign. It is NOT a correctness switch -- every arm is bit-identical,
// which tests/vt/test_vulkan_backend.cpp asserts with memcmp.
//
// Atomic because ops dispatch from whichever thread the engine runs on; relaxed
// because nothing else is ordered against it -- a stale read would pick the other
// arm of a performance A/B, never a wrong result.
std::atomic<uint32_t>& MatmulColumnsSlot() {
  static std::atomic<uint32_t> slot{[] {
    const char* v = std::getenv("VT_VULKAN_MATMUL_NCOLS");
    if (v == nullptr) return 4u;
    if (std::strcmp(v, "1") == 0) return 1u;
    if (std::strcmp(v, "2") == 0) return 2u;
    if (std::strcmp(v, "8") == 0) return 8u;
    return 4u;
  }()};
  return slot;
}

uint32_t MatmulColumnsPerLane() { return MatmulColumnsSlot().load(std::memory_order_relaxed); }

void SetMatmulColumnsPerLane(uint32_t ncols) {
  // 1, 2, 4 or 8 only. The shader sizes its accumulator array with a COMPILE-TIME
  // bound of 8 -- a specialization constant cannot size it without the array
  // spilling to scratch memory, which would defeat the point -- so a larger value
  // would read past the array rather than fail.
  VT_CHECK(ncols == 1u || ncols == 2u || ncols == 4u || ncols == 8u,
           "vulkan: matmul columns-per-lane must be 1, 2, 4 or 8 (the shader's "
           "accumulator array is bounded at 8)");
  MatmulColumnsSlot().store(ncols, std::memory_order_relaxed);
}

}  // namespace vt::vulkan
