// Laguna-S-2.1 forward — W3 REAL host-reference composition. Replaces the W1/W2
// `VT_CHECK(false, ...)` stub with a runnable per-layer forward that COMPOSES the
// ~85-90% reuse + the three genuinely-NEW small host ops (laguna_ops.cpp):
//
//   x_res = x; h = RMSNorm(x, input_norm)                    [shared RMSNorm]
//   q/k/v = {q,k,v}_proj(h)   (q width is PER-LAYER: 48 global / 72 sliding)
//   q,k  = dual per-layer RoPE (global -> YaRN partial-64 [olmo3 inv_freq];
//          sliding -> plain-128), applied NeoX-style on the first rotary_dim dims
//   attn = GQA(q,k,v, mask = global ? full-causal : sliding-window(512))
//                                                            [gemma2/3 is_sliding]
//   attn = per-head SOFTPLUS out-gate(attn, g_proj(h))       [NEW op (a)]
//   x   += o_proj(attn)
//   h2   = RMSNorm(x, post_attn_norm)
//   f    = dense SwiGLU MLP (layer 0)  |  ungrouped sigmoid-noaux MoE (1..47):
//            sel = UngroupedRouterTopK(router(h2), e_score_bias, top_k, renorm,
//                                      routed_scaling)        [NEW op (b), ds2-minus-group]
//            f   = sum_i sel.w_i * expert_{sel.id_i}(h2) + shared_expert(h2)
//   x   += f
//   final: RMSNorm(x, norm) -> lm_head (untied) -> logits
//
// SCOPE HONESTY: this is the whole-sequence (prefill) REFERENCE forward in f32 —
// runnable + unit-gated on synthetic weights (test_laguna_scaffold laguna-fwd
// case), the vehicle that verifies the composition + the new ops fire and the
// per-layer variable-Q-head shapes flow. It does NOT consume the paged KV cache
// (`attn_kv`/`attn_meta` are accepted but the reference recomputes attention over
// the token window); the device/paged production path (bf16 vt:: ops off the ds4
// keep-quant towers, olmo2.cpp ForwardBody pattern) + the tower materialization +
// the strict dual-oracle greedy gate (llama.cpp-Q4_K token-exact + vLLM-NVFP4
// near-tie) are W4. See .agents/specs/laguna-s21-w3-2026-07-31.md.
#include "vllm/model_executor/models/laguna.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <mutex>
#include <unordered_map>

#include "vllm/model_executor/models/dense_nvfp4_gemm.h"  // Dev/DBuf/ResidentNvfp4 + Marlin (B2)
#include "vllm/model_executor/models/laguna_device.h"
#include "vllm/model_executor/models/laguna_ops.h"
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vt/backend.h"  // vt::GetBackend (device drain for the keep-quant GEMMs)
#include "vt/breakable_graph.h"  // ENG-CUDAGRAPH-BREAK W5: the shared capture seam
#include "vt/dtype.h"    // vt::IsBlockQuant / RowSizeBytes
#include "vt/ops.h"      // vt::MatmulBT (dispatches kMatmulBTQuant on block weights)
#include "vt/recipes.h"  // vt::kFusedAddRmsNormStd (L4 residual-add + RMSNorm fusion)
#include "vt/tensor.h"
#include "vt/unaligned.h"

namespace vllm {

void StageLagunaGraphEmbedding(const OwnedTensor& embed, int32_t token,
                               int64_t hidden_size, int64_t vocab_size,
                               float* destination) {
  VT_CHECK(token >= 0 && token < vocab_size,
           "laguna: token id out of range");
  VT_CHECK(embed.dtype == vt::DType::kF32 ||
               embed.dtype == vt::DType::kBF16,
           "laguna embed: table dtype must be f32/bf16");
  const size_t row_offset =
      static_cast<size_t>(token) * static_cast<size_t>(hidden_size);
  const uint8_t* row = embed.bytes.data() +
                       row_offset * vt::SizeOf(embed.dtype);
  if (embed.dtype == vt::DType::kBF16) {
    for (int64_t i = 0; i < hidden_size; ++i) {
      const uint32_t bits =
          static_cast<uint32_t>(vt::LoadUnaligned<uint16_t>(row + i * 2))
          << 16;
      std::memcpy(&destination[i], &bits, sizeof(float));
    }
  } else {
    std::memcpy(destination, row,
                static_cast<size_t>(hidden_size) * sizeof(float));
  }
}

namespace {

// LEVER A: host f32→bf16 round-to-nearest-even (matches CUDA __float2bfloat16 used by the
// on-device KV append, so the one-time prefill-KV migration rounds identically to the decode
// appends). NaN is preserved as a quiet bf16 NaN; KV values never hit it in practice.
inline uint16_t LagunaF32ToBf16Rne(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  if ((x & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((x >> 16) | 0x0040u);
  const uint32_t bias = ((x >> 16) & 1u) + 0x7fffu;  // round-to-nearest-even
  return static_cast<uint16_t>((x + bias) >> 16);
}

// LEVER A (VT_LAGUNA_KV_BF16, default OFF — opt-in): store the resident/graph decode KV at
// bf16 (see laguna.h k_dev16). Reads the SAME env var as the .cu-side LagunaKvBf16On() (a
// process-global toggle; both read it independently, mirroring the other Laguna*Enabled gates).
// Default OFF because a prior short-context attempt was a WASH + near-tie break; the win is
// context-linear (see the .cu comment). MUST agree with the .cu reader — same var, same logic.
inline bool LagunaKvBf16Enabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LAGUNA_KV_BF16");
    return (e != nullptr && e[0] == '1');  // default OFF; =1 opts in
  }();
  return on;
}

// Per-layer decode KV headroom (rows reserved above the prompt for generated tokens). Default
// 1024 (unchanged behavior). VT_LAGUNA_KV_HEADROOM raises it so the two-length long-context
// slope measurement (~2048 generated tokens) fits the fixed-capacity resident/graph KV without
// tripping the capacity VT_CHECK — measurement infrastructure, not a lever.
inline int64_t LagunaKvHeadroom() {
  static const int64_t rows = [] {
    const char* e = std::getenv("VT_LAGUNA_KV_HEADROOM");
    if (e == nullptr) return static_cast<int64_t>(1024);
    const long v = std::strtol(e, nullptr, 10);
    return (v > 0) ? static_cast<int64_t>(v) : static_cast<int64_t>(1024);
  }();
  return rows;
}

// ── VT_LAGUNA_RESIDENT_BF16W (default ON; =0 opts back to the retag A/B arm):
// stage the bf16/f32 decode projection
// weights TRUE device-resident (a cudaMalloc'd copy, uploaded ONCE and reused every
// step) instead of the unified-memory RETAG of the host bytes (w.View() +
// .device=dev). On GB10 the GPU reads system-allocated (ATS/unified) memory slower
// than cudaMalloc'd device memory — worst on the long-K low-parallelism o_proj GEMV
// — so the retagged host weight caps per-call GEMV bandwidth below vLLM's (whose
// weights are true device allocations). SAME bytes, SAME MatmulBT kernel, SAME
// invocation ⇒ byte-exact by construction. Mirrors the canonical
// dense_attn_block.h ResidentWeight d_dev seam and reuses OwnedTensor's existing
// mutable d_dev cache. CAPTURE-SAFE: the first touch of each weight happens in the
// eager cold warm-run (LagunaGraph gstate 0, before any BeginCapture — the same
// place the Marlin residents build lazily), so the captured replay reads a stable
// device pointer and does ZERO fresh cudaMalloc.
inline bool LagunaResidentBf16WEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LAGUNA_RESIDENT_BF16W");
    return (e == nullptr || e[0] != '0');  // default ON (parity enabler); =0 opts out
  }();
  return on;
}

// Device operand for a bf16/f32 projection weight consumed by a decode GEMV.
// Lever ON: lazily upload w.bytes → w.d_dev (cudaMalloc + one H2D copy) and return
// a view over the DEVICE copy. Lever OFF: the legacy unified-memory retag (host
// bytes, device-tagged). The operand is byte-identical either way — only the
// backing allocation (true-device vs system) differs. Starts from w.View() so the
// shape/stride/nk/repack metadata is exactly the retag path's; only .data changes.
inline vt::Tensor LagunaResidentBf16W(vt::Queue& q, const OwnedTensor& w, vt::Device dev) {
  vt::Tensor t = w.View();
  t.device = dev;
  if (!LagunaResidentBf16WEnabled()) return t;  // legacy unified-memory retag (byte-identical)
  if (!w.d_dev) {
    vt::Backend& b = vt::GetBackend(dev);
    const size_t nb = w.bytes.size();
    void* p = b.Alloc(nb);
    b.Copy(q, p, w.bytes.data(), nb);
    vt::Backend* bk = &b;
    w.d_dev = std::shared_ptr<void>(p, [bk](void* pp) { bk->Free(pp); });
  }
  t.data = w.d_dev.get();
  return t;
}

// Decode an OwnedTensor (host bytes) to a flat f32 vector. Reference path handles
// the two host dtypes a synthetic/dequantized tower carries: f32 and bf16.
std::vector<float> ReadF32(const OwnedTensor& t) {
  const int64_t n = t.Numel();
  std::vector<float> out(static_cast<size_t>(n));
  const uint8_t* raw = t.bytes.data();
  if (t.dtype == vt::DType::kF32) {
    std::memcpy(out.data(), raw, static_cast<size_t>(n) * sizeof(float));
  } else if (t.dtype == vt::DType::kBF16) {
    for (int64_t i = 0; i < n; ++i) {
      const uint32_t bits =
          static_cast<uint32_t>(vt::LoadUnaligned<uint16_t>(raw + i * 2))
          << 16;
      std::memcpy(&out[static_cast<size_t>(i)], &bits, sizeof(float));
    }
  } else {
    VT_CHECK(false, "laguna reference forward: weight dtype must be f32/bf16 "
                    "(quant keep-quant decode is the W4 device path)");
  }
  return out;
}

// out[T,N] = x[T,K] @ W_nk[N,K]^T  (raw-NK torch Linear weight, no bias).
std::vector<float> MatmulNK(const std::vector<float>& x, const std::vector<float>& w,
                            int64_t T, int64_t N, int64_t K) {
  VT_CHECK(static_cast<int64_t>(x.size()) == T * K, "laguna matmul: x shape");
  VT_CHECK(static_cast<int64_t>(w.size()) == N * K, "laguna matmul: w shape");
  std::vector<float> out(static_cast<size_t>(T * N), 0.0F);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t nn = 0; nn < N; ++nn) {
      float acc = 0.0F;
      const float* xr = x.data() + static_cast<size_t>(t * K);
      const float* wr = w.data() + static_cast<size_t>(nn * K);
      for (int64_t kk = 0; kk < K; ++kk) acc += xr[kk] * wr[kk];
      out[static_cast<size_t>(t * N + nn)] = acc;
    }
  return out;
}

// RMSNorm(x, weight, eps): variance in f32.
std::vector<float> RmsNorm(const std::vector<float>& x, const std::vector<float>& w,
                           int64_t T, int64_t H, float eps) {
  std::vector<float> out(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t) {
    const float* xr = x.data() + static_cast<size_t>(t * H);
    float ss = 0.0F;
    for (int64_t i = 0; i < H; ++i) ss += xr[i] * xr[i];
    const float inv = 1.0F / std::sqrt(ss / static_cast<float>(H) + eps);
    float* orow = out.data() + static_cast<size_t>(t * H);
    for (int64_t i = 0; i < H; ++i)
      orow[i] = xr[i] * inv * w[static_cast<size_t>(i)];
  }
  return out;
}

// Per-head RMSNorm over the head_dim axis (Laguna QK-norm, VERIFIED W4): x is
// [T, heads, Dh] flattened; each length-Dh head vector is RMS-normed with the
// shared weight w[Dh]. Variance in f32 (Qwen3/OLMo-2 qk_layernorm semantics).
void RmsNormHeads(std::vector<float>& x, const std::vector<float>& w, int64_t T,
                  int64_t heads, int64_t Dh, float eps) {
  for (int64_t r = 0; r < T * heads; ++r) {
    float* v = x.data() + static_cast<size_t>(r * Dh);
    float ss = 0.0F;
    for (int64_t d = 0; d < Dh; ++d) ss += v[d] * v[d];
    const float inv = 1.0F / std::sqrt(ss / static_cast<float>(Dh) + eps);
    for (int64_t d = 0; d < Dh; ++d) v[d] = v[d] * inv * w[static_cast<size_t>(d)];
  }
}

inline float Silu(float x) { return x / (1.0F + std::exp(-x)); }

// Separate gate/up SwiGLU: silu(gate)*up per element -> [T,I].
std::vector<float> GateUpSilu(const std::vector<float>& gate,
                              const std::vector<float>& up, int64_t T, int64_t I) {
  std::vector<float> act(static_cast<size_t>(T * I));
  for (int64_t i = 0; i < T * I; ++i)
    act[static_cast<size_t>(i)] = Silu(gate[static_cast<size_t>(i)]) * up[static_cast<size_t>(i)];
  return act;
}

// In-place NeoX partial RoPE over the first `rd` dims of each head. `cache` is
// [rows, rd] with the cos|sin half split (BuildLaguna*CosSin layout).
void ApplyRope(std::vector<float>& x, int64_t T, int64_t heads, int64_t Dh,
               int64_t rd, const std::vector<float>& cache,
               const std::vector<int32_t>& positions) {
  const int64_t half = rd / 2;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t pos = positions[static_cast<size_t>(t)];
    const float* crow = cache.data() + static_cast<size_t>(pos * rd);
    for (int64_t h = 0; h < heads; ++h) {
      float* xv = x.data() + static_cast<size_t>((t * heads + h) * Dh);
      for (int64_t i = 0; i < half; ++i) {
        const float c = crow[i];
        const float s = crow[half + i];
        const float x0 = xv[i];
        const float x1 = xv[half + i];
        xv[i] = x0 * c - x1 * s;
        xv[half + i] = x1 * c + x0 * s;
      }
    }
  }
}

// One expert / shared-MLP forward from SEPARATE gate/up/down weights (the real
// GGUF name-map): silu(gate@h)*(up@h) then down. [H] out.
std::vector<float> ExpertMlp(const std::vector<float>& h_row,
                             const std::vector<float>& gate_w,
                             const std::vector<float>& up_w,
                             const std::vector<float>& down_w, int64_t H,
                             int64_t I) {
  const std::vector<float> g = MatmulNK(h_row, gate_w, 1, I, H);
  const std::vector<float> u = MatmulNK(h_row, up_w, 1, I, H);
  const std::vector<float> act = GateUpSilu(g, u, 1, I);
  return MatmulNK(act, down_w, 1, H, I);  // [H]
}

// ── W5 keep-quant GEMM (mirror deepseek_v4.cpp Gemm/GemmRowSlice) ─────────────
// Drain a device queue's stream before the host reads the output (no-op on CPU).
inline void DrainQueue(vt::Queue& q) {
  if (q.device.type != vt::DeviceType::kCPU)
    vt::GetBackend(q.device).Synchronize(q);
}

// Y[T,N] = X[T,K] @ W[N,K]^T. W is a keep-quant (block) OR f32/bf16 OwnedTensor in
// the file's own [N,K] order (nk=true, as on GGUF disk). A block-quant weight
// routes to vt::MatmulBT's kMatmulBTQuant provider (CPU always; CUDA when the
// queue is a device queue — reads the unified-memory blocks in place). An f32/bf16
// weight (the router, or a bf16 oracle expansion) falls to the host MatmulNK
// reference — BIT-IDENTICAL to the pre-W5 path. Mirrors deepseek_v4.cpp:410.
std::vector<float> LqGemm(vt::Queue& q, const OwnedTensor& w,
                          const std::vector<float>& x, int64_t T, int64_t N,
                          int64_t K) {
  VT_CHECK(w.rank == 2 && w.shape[0] == N && w.shape[1] == K,
           "laguna keep-quant GEMM: weight shape mismatch");
  if (!vt::IsBlockQuant(w.dtype)) {
    // N5 lever #2: the bf16/f32 tower (attention/dense/router/shared/lm_head on the
    // NVFP4 arm) — the nsys trace found it runs the HOST MatmulNK reference even on
    // the CUDA queue, CPU-bound at ~4.8 s/tok (the lm_head [Vsz,H] MatmulNK + its
    // per-token ReadF32 dominate). On the DEVICE, keep the weight bf16 (no ReadF32),
    // cast the SMALL [T,K] f32 activation to bf16 on-GPU, and run the bf16×bf16->f32
    // MatmulBT (cuBLASLt nvjet on GB10; MatmulBT needs matched dtypes). f32 accum;
    // near-tie vs MatmulNK (different algo + bf16 activation rounding). The CPU path
    // keeps the exact MatmulNK reference (the run-gate stays byte-identical); the GGUF
    // keep-quant tower is block-quant here so this branch is nvfp4-arm-only.
    const bool bf16_dev = q.device.type != vt::DeviceType::kCPU && w.dtype == vt::DType::kBF16;
    if (!bf16_dev) return MatmulNK(x, ReadF32(w), T, N, K);
    std::vector<float> out(static_cast<size_t>(T) * N);
    std::vector<uint16_t> a_bf16(static_cast<size_t>(T) * K);  // bf16 activation (unified mem)
    vt::Tensor xf = vt::Tensor::Contiguous(const_cast<float*>(x.data()), vt::DType::kF32,
                                           q.device, {T, K});
    vt::Tensor ab = vt::Tensor::Contiguous(a_bf16.data(), vt::DType::kBF16, q.device, {T, K});
    vt::CastBf16(q, ab, xf);
    vt::Tensor o = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, q.device, {T, N});
    vt::Tensor wt = w.View();
    wt.device = q.device;  // unified-memory bf16 weight retag (like the block-quant path)
    vt::MatmulBT(q, o, ab, wt);
    DrainQueue(q);
    return out;
  }
  std::vector<float> out(static_cast<size_t>(T) * N);
  vt::Queue cpuq{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const bool on_dev = q.device.type != vt::DeviceType::kCPU;
  vt::Queue& gq = on_dev ? q : cpuq;
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(x.data()),
                                        vt::DType::kF32, gq.device, {T, K});
  vt::Tensor o = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, gq.device, {T, N});
  vt::Tensor wt = w.View();
  wt.device = gq.device;  // unified-memory block view retag (ds4 ops.cpp:198 check)
  vt::MatmulBT(gq, o, a, wt);
  if (on_dev) DrainQueue(gq);
  return out;
}

// N2 (task #230): TRUE-W4A4 NVFP4 GEMM for the safetensors arm's routed experts.
// Y[M,N] f32 = fp4(X[M,K] / input_divisor) @ dequant(w).T, accumulator * alpha.
// The activation is quantized to fp4 with w.input_global_scale_inv (NOT scale2),
// then vt::MatmulNvfp4Fp4 scales the fp4xfp4 accumulator by w.alpha (which folds
// BOTH the activation and weight reciprocated global scales). Mirrors LqGemm's
// unified-memory pattern (host ptrs as device tensors on GB10; raw weight view
// retagged). CPU-runnable (ScaledFp4QuantKernel + MatmulNvfp4Fp4Kernel exist), so
// the fixture gate exercises the exact numerics. If the DGX gate shows all-zeros,
// the whole-weight raw view needs ResidentWeight staging (keepquant-device-slice
// note) — a one-line switch to explicit residency.
std::vector<float> LqGemmNvfp4Fp4(vt::Queue& q, const Nvfp4Weight& w,
                                  const std::vector<float>& x, int64_t M, int64_t N,
                                  int64_t K) {
  VT_CHECK(w.n == N && w.k == K && K % 16 == 0,
           "laguna nvfp4 GEMM: weight shape / K%16 mismatch");
  VT_CHECK(w.IsTrueW4A4(), "laguna nvfp4 GEMM: expected true-W4A4 (alpha>0) weight");
  std::vector<float> out(static_cast<size_t>(M) * N);
  vt::Queue cpuq{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const bool on_dev = q.device.type != vt::DeviceType::kCPU;
  vt::Queue& gq = on_dev ? q : cpuq;
  // fp4 activation intermediates (unified-memory host buffers: i8 packed [M,K/2] +
  // fp8-e4m3 block scale [M,K/16]).
  std::vector<int8_t> ap_buf(static_cast<size_t>(M) * (K / 2));
  std::vector<int8_t> as_buf(static_cast<size_t>(M) * (K / 16));
  vt::Tensor xt = vt::Tensor::Contiguous(const_cast<float*>(x.data()),
                                         vt::DType::kF32, gq.device, {M, K});
  vt::Tensor ap = vt::Tensor::Contiguous(ap_buf.data(), vt::DType::kI8, gq.device, {M, K / 2});
  vt::Tensor as = vt::Tensor::Contiguous(as_buf.data(), vt::DType::kI8, gq.device, {M, K / 16});
  vt::ScaledFp4Quant(gq, ap, as, xt, w.input_global_scale_inv);  // kLinear layout
  vt::Tensor ot = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, gq.device, {M, N});
  vt::Tensor bp = w.packed.View();
  bp.device = gq.device;  // unified-memory retag (mirror LqGemm)
  vt::Tensor bs = w.scale.View();
  bs.device = gq.device;
  vt::MatmulNvfp4Fp4(gq, ot, ap, as, bp, bs, w.alpha);
  if (on_dev) DrainQueue(gq);
  return out;
}

// N5 lever (task #234): DEVICE-RESIDENT W4A4 MoE block for ONE decode token. The
// per-expert `LqGemmNvfp4Fp4` loop drains the queue after EVERY GEMM (it returns a
// host vector) → ~top_k×3 `cudaStreamSynchronize`/layer, the 78.6%-of-API-time wall
// the nsys found. Here the WHOLE token's routed-expert MoE runs as one async device
// chain — per expert: fp4-quant(hrow, w.igs) → MatmulNvfp4Fp4 for gate/up, MoeSiluMul,
// fp4-quant(act, down.igs) → MatmulNvfp4Fp4 for down into a stacked [top_k,H] device
// buffer — then ONE MoeCombine (weighted sum) and ONE DrainQueue. Unified-memory
// buffers (host ptrs retagged device); every op stays on `q` so CUDA orders them (the
// gate GEMM reads `a_p` before the up-quant overwrites it — same-stream serial). CUDA
// only (M=1 decode); the CPU/per-expert path stays the byte-exact reference.
std::vector<float> LagunaMoeResidentFp4(vt::Queue& q, const LagunaMoeWeights& moe,
                                        const std::vector<float>& hrow, int64_t moe_I,
                                        int64_t H, const LagunaRouterSelection& sel) {
  const int64_t Pk = static_cast<int64_t>(sel.ids.size());
  std::vector<float> acc(static_cast<size_t>(H), 0.0F);
  if (Pk == 0) return acc;
  const vt::Device dev = q.device;
  // fp4 activation intermediates (K=H for gate/up, K=moe_I for down) + per-expert
  // gate/up/silu buffers + the stacked expert-output [Pk,H] the combine reduces.
  std::vector<int8_t> ap_h((H / 2)), as_h((H / 16));        // gate/up activation fp4
  std::vector<int8_t> ap_d((moe_I / 2)), as_d((moe_I / 16));  // down activation fp4
  std::vector<float> dg(moe_I), du(moe_I), dact(moe_I);
  std::vector<float> deo(static_cast<size_t>(Pk) * H);       // [Pk,H] stacked outputs
  std::vector<float> wgt(static_cast<size_t>(Pk));
  for (int64_t s = 0; s < Pk; ++s) wgt[static_cast<size_t>(s)] = sel.weights[static_cast<size_t>(s)];
  auto wview = [&](const OwnedTensor& w) { vt::Tensor t = w.View(); t.device = dev; return t; };
  vt::Tensor xt = vt::Tensor::Contiguous(const_cast<float*>(hrow.data()), vt::DType::kF32, dev, {1, H});
  vt::Tensor ap = vt::Tensor::Contiguous(ap_h.data(), vt::DType::kI8, dev, {1, H / 2});
  vt::Tensor as = vt::Tensor::Contiguous(as_h.data(), vt::DType::kI8, dev, {1, H / 16});
  vt::Tensor apd = vt::Tensor::Contiguous(ap_d.data(), vt::DType::kI8, dev, {1, moe_I / 2});
  vt::Tensor asd = vt::Tensor::Contiguous(as_d.data(), vt::DType::kI8, dev, {1, moe_I / 16});
  vt::Tensor dgt = vt::Tensor::Contiguous(dg.data(), vt::DType::kF32, dev, {1, moe_I});
  vt::Tensor dut = vt::Tensor::Contiguous(du.data(), vt::DType::kF32, dev, {1, moe_I});
  vt::Tensor dat = vt::Tensor::Contiguous(dact.data(), vt::DType::kF32, dev, {1, moe_I});
  for (int64_t s = 0; s < Pk; ++s) {
    const int64_t id = sel.ids[static_cast<size_t>(s)];
    const Nvfp4Weight& gw = moe.experts_gate_fp4[static_cast<size_t>(id)];
    const Nvfp4Weight& uw = moe.experts_up_fp4[static_cast<size_t>(id)];
    const Nvfp4Weight& dw = moe.experts_down_fp4[static_cast<size_t>(id)];
    vt::ScaledFp4Quant(q, ap, as, xt, gw.input_global_scale_inv);
    vt::MatmulNvfp4Fp4(q, dgt, ap, as, wview(gw.packed), wview(gw.scale), gw.alpha);
    vt::ScaledFp4Quant(q, ap, as, xt, uw.input_global_scale_inv);
    vt::MatmulNvfp4Fp4(q, dut, ap, as, wview(uw.packed), wview(uw.scale), uw.alpha);
    vt::MoeSiluMul(q, dat, dgt, dut);
    vt::ScaledFp4Quant(q, apd, asd, dat, dw.input_global_scale_inv);
    vt::Tensor eo = vt::Tensor::Contiguous(deo.data() + static_cast<size_t>(s) * H,
                                           vt::DType::kF32, dev, {1, H});
    vt::MatmulNvfp4Fp4(q, eo, apd, asd, wview(dw.packed), wview(dw.scale), dw.alpha);
  }
  vt::Tensor eostack = vt::Tensor::Contiguous(deo.data(), vt::DType::kF32, dev, {1, Pk, H});
  vt::Tensor wt = vt::Tensor::Contiguous(wgt.data(), vt::DType::kF32, dev, {1, Pk});
  vt::Tensor at = vt::Tensor::Contiguous(acc.data(), vt::DType::kF32, dev, {1, H});
  vt::MoeCombine(q, at, eostack, wt);
  DrainQueue(q);  // the ONLY sync for the whole token's routed-expert MoE
  return acc;
}

// ── N5 campaign-B (task #234): route the routed experts to the MARLIN W4A16
//    grouped MoE GEMM — vLLM's ACTUAL 18.8-tok/s kernel (VLLM_TEST_FORCE_FP8_MARLIN=1),
//    LOW-M-optimized for decode. Mirrors qwen3_5.cpp BuildMoeMarlinResident +
//    MoeBlockFusedMarlinCuda (the validated 27B/35B path: default-ON there,
//    16/16 token-for-token vs oracle, +22% gate / +80% decode), reusing the SHARED
//    dense_nvfp4::Dev/DBuf/ResidentNvfp4 + the shared vt::cuda Marlin repack/align
//    ops + vt::MoeGroupedGemmNvfp4Marlin. The SACRED qwen3_5 path is BYTE-UNTOUCHED
//    (this is a Laguna-local reconstruction over the shared primitives, split-w13
//    layout for a simple first landing). Gated OFF by default (VT_LAGUNA_MARLIN_MOE=1
//    opt-in) until the DGX near-tie + ncu gate lands — so the current default GEMV
//    path is unchanged. CUDA-only (VT_MARLIN_NVFP4). ──────────────────────────────
// The Marlin-repacked resident MoE state + its repack/build helpers are GENUINELY
// device-coupled: the types and repack entry points they name exist only in the CUDA leg
// (src/vt/cuda), so a CPU build cannot compile the block at all. REPAIR OWED: hoist this
// behind the OpProvider seam (a `kLagunaMoeMarlin` op resolved at runtime) the way
// laguna_device.cpp already resolves the kLaguna glue table, so the shared model TU stops
// carrying a build-time device branch. Deferred: it restructures the resident MoE path and
// needs the GB10 Laguna re-gate. Same class as the 26 pre-existing VT_MARLIN_NVFP4 sites
// in qwen3_5.cpp that make up the DSR baseline.
// DSR-ALLOW(S1): Marlin resident MoE state/helpers, CUDA-leg types; repair owed above.
#ifdef VT_MARLIN_NVFP4
namespace {
// Resident Marlin-repacked routed experts. Mirror MoeMarlinResident (qwen3_5.cpp), incl.
// the LEVER B fused-w13 layout: gate+up concatenated along N into ONE size_n=2*moe_I
// operand (w_gu/s_gu/g_gu), populated INSTEAD of w_gate/w_up when fused_w13.
struct LagunaMoeMarlinResident {
  void* w_gate = nullptr;  // i32 [E, H/16, moe_I*2]  (split layout)
  void* w_up = nullptr;    // i32 [E, H/16, moe_I*2]
  void* w_down = nullptr;  // i32 [E, moe_I/16, H*2]
  void* s_gate = nullptr;  // u8  [E, H/16, moe_I]
  void* s_up = nullptr;
  void* s_down = nullptr;  // u8  [E, moe_I/16, H]
  void* g_gate = nullptr;  // f32 [E]
  void* g_up = nullptr;
  void* g_down = nullptr;
  // LEVER B fused-w13 (VT_MOE_FUSED_W13): gate+up CONCATENATED along N per expert →
  // ONE grouped Marlin GEMM of size_n=2*moe_I + one SiluAndMul (3 grouped GEMMs → 2).
  void* w_gu = nullptr;    // i32 [E, H/16, (2*moe_I)*2]
  void* s_gu = nullptr;    // u8  [E, H/16, 2*moe_I]
  void* g_gu = nullptr;    // f32 [E]  (gate scale2; == up scale2, checked at build)
  bool fused_w13 = false;
  void* workspace = nullptr;  // i32 [sms*4]
  int sms = 0;
  bool ready = false;
};

// LEVER B gate: fuse the routed-expert gate+up into ONE size_n=2N grouped GEMM. Mirrors
// qwen3_5.cpp MoeFusedW13Enabled (SAME env var VT_MOE_FUSED_W13, DEFAULT ON) — the
// frameworkization move onto the shared fused-w13 layout. Any gate/up scale2 mismatch
// falls back to split at build (token-exact; no degraded fusion).
inline bool LagunaMoeFusedW13Enabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_MOE_FUSED_W13");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

// Same lifetime fix as the Qwen3.5 resident slots (issue #237); ResidentSlot and
// the reasoning are in qwen3_5_weights.h, which this model's weights header
// already includes.
LagunaMoeMarlinResident& LagunaMoeMarlinResidentFor(const LagunaMoeWeights* w) {
  static std::mutex mu;
  std::lock_guard<std::mutex> lk(mu);
  if (!w->resident_marlin.state) {
    w->resident_marlin.state = std::make_shared<LagunaMoeMarlinResident>();
  }
  return *static_cast<LagunaMoeMarlinResident*>(w->resident_marlin.state.get());
}

// Repack every routed expert's fp4 gate/up/down into the resident Marlin layout —
// the per-expert body of qwen3_5.cpp BuildMoeMarlinResident, reusing the identical
// shared vt::cuda primitives. K=H (gate/up in), N=moe_I (gate/up out); down is K=moe_I,
// N=H. LEVER B: when fused_w13, gate+up are repacked as ONE size_n=2*moe_I operand.
void BuildLagunaMoeMarlinResident(vllm::dense_nvfp4::Dev d, const LagunaMoeWeights& moe, int E,
                                  int H, int I, LagunaMoeMarlinResident& mr) {
  void* stream = d.q.handle;
  mr.sms = vt::cuda::MarlinDeviceSms(d.q.device.index);
  const size_t wg_i32 = static_cast<size_t>(H / 16) * (I * 2);  // gate/up weight elems
  const size_t wd_i32 = static_cast<size_t>(I / 16) * (H * 2);  // down weight elems
  const size_t sg_b = static_cast<size_t>(H / 16) * I;          // gate/up scale bytes
  const size_t sd_b = static_cast<size_t>(I / 16) * H;          // down scale bytes

  // LEVER B fused-w13 (mirror qwen3_5.cpp): ONE Marlin B operand per expert with gate+up
  // concatenated along N needs ONE per-expert global scale for both halves. vLLM checks
  // allclose(w13_weight_scale_2[:,0], [:,1]) then uses [:,0] (modelopt "single gscale for
  // w13"); our token-exact gate forbids degraded fusion, so ANY gate/up scale2 mismatch
  // falls back to the split two-GEMM layout instead.
  bool fuse = LagunaMoeFusedW13Enabled();
  for (int e = 0; fuse && e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    if (moe.experts_gate_fp4[se].scale2 != moe.experts_up_fp4[se].scale2) {
      std::fprintf(stderr,
                   "vllm.cpp laguna: VT_MOE_FUSED_W13: expert %d gate/up scale2 differ "
                   "(%g vs %g) — falling back to the split w13 layout\n",
                   e, static_cast<double>(moe.experts_gate_fp4[se].scale2),
                   static_cast<double>(moe.experts_up_fp4[se].scale2));
      fuse = false;
    }
  }
  mr.fused_w13 = fuse;

  if (fuse) {  // same total bytes as the split w_gate+w_up / s_gate+s_up pair
    mr.w_gu = d.b.Alloc(static_cast<size_t>(E) * 2 * wg_i32 * 4);
    mr.s_gu = d.b.Alloc(static_cast<size_t>(E) * 2 * sg_b);
    mr.g_gu = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  } else {
    mr.w_gate = d.b.Alloc(static_cast<size_t>(E) * wg_i32 * 4);
    mr.w_up = d.b.Alloc(static_cast<size_t>(E) * wg_i32 * 4);
    mr.s_gate = d.b.Alloc(static_cast<size_t>(E) * sg_b);
    mr.s_up = d.b.Alloc(static_cast<size_t>(E) * sg_b);
    mr.g_gate = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
    mr.g_up = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  }
  mr.w_down = d.b.Alloc(static_cast<size_t>(E) * wd_i32 * 4);
  mr.s_down = d.b.Alloc(static_cast<size_t>(E) * sd_b);
  mr.g_down = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  mr.workspace = d.b.Alloc(static_cast<size_t>(mr.sms) * 4 * sizeof(int32_t));

  // combined_scale_factor: gate+up jointly (w13), down alone (w2).
  std::vector<const uint8_t*> gu_bufs, dn_bufs;
  std::vector<size_t> gu_lens, dn_lens;
  for (int e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    gu_bufs.push_back(reinterpret_cast<const uint8_t*>(moe.experts_gate_fp4[se].scale.bytes.data()));
    gu_lens.push_back(moe.experts_gate_fp4[se].scale.bytes.size());
    gu_bufs.push_back(reinterpret_cast<const uint8_t*>(moe.experts_up_fp4[se].scale.bytes.data()));
    gu_lens.push_back(moe.experts_up_fp4[se].scale.bytes.size());
    dn_bufs.push_back(reinterpret_cast<const uint8_t*>(moe.experts_down_fp4[se].scale.bytes.data()));
    dn_lens.push_back(moe.experts_down_fp4[se].scale.bytes.size());
  }
  const float sf_gu = vt::cuda::MarlinNvfp4CombinedScaleFactor(gu_bufs, gu_lens);
  const float sf_dn = vt::cuda::MarlinNvfp4CombinedScaleFactor(dn_bufs, dn_lens);

  // Fused-w13 concat staging (device, reused across experts — issued on the SAME stream,
  // so each expert's repack reads its staging bytes before the next expert overwrites).
  // vLLM w13 stack = w1 (gate) rows first, then w3 (up); silu reads [:N] as gate.
  const size_t pk_b = static_cast<size_t>(I) * (H / 2);  // one shard's packed fp4 bytes
  uint8_t* tmp_w = nullptr;
  uint8_t* tmp_s = nullptr;
  if (fuse) {
    tmp_w = static_cast<uint8_t*>(d.b.Alloc(2 * pk_b));
    tmp_s = static_cast<uint8_t*>(d.b.Alloc(2 * sg_b));
  }

  std::vector<float> gg(E), gu(E), gd(E);
  for (int e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    vllm::dense_nvfp4::Nvfp4Dev g = vllm::dense_nvfp4::ResidentNvfp4(d, moe.experts_gate_fp4[se]);
    vllm::dense_nvfp4::Nvfp4Dev u = vllm::dense_nvfp4::ResidentNvfp4(d, moe.experts_up_fp4[se]);
    vllm::dense_nvfp4::Nvfp4Dev dn = vllm::dense_nvfp4::ResidentNvfp4(d, moe.experts_down_fp4[se]);
    auto* wd = static_cast<uint32_t*>(mr.w_down) + se * wd_i32;
    auto* sdp = static_cast<uint8_t*>(mr.s_down) + se * sd_b;
    const auto* pg = static_cast<const uint8_t*>(g.packed.data);
    const auto* pu = static_cast<const uint8_t*>(u.packed.data);
    const auto* pd = static_cast<const uint8_t*>(dn.packed.data);
    if (fuse) {
      // ONE repack + ONE scale-process over the N-concatenated gate|up (size_n = 2*I) —
      // the per-expert body of vLLM's repack over the stacked w13 (size_n = num_shards*N).
      auto* wgu = static_cast<uint32_t*>(mr.w_gu) + se * 2 * wg_i32;
      auto* sgup = static_cast<uint8_t*>(mr.s_gu) + se * 2 * sg_b;
      d.b.Copy(d.q, tmp_w, pg, pk_b);
      d.b.Copy(d.q, tmp_w + pk_b, pu, pk_b);
      vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, wgu, tmp_w, H, 2 * I);
      d.b.Copy(d.q, tmp_s, g.scale.data, sg_b);
      d.b.Copy(d.q, tmp_s + sg_b, u.scale.data, sg_b);
      vt::cuda::MarlinProcessExpertScales(stream, tmp_s, sgup, H, 2 * I, sf_gu);
      gg[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(moe.experts_gate_fp4[se].scale2, sf_gu);
    } else {
      auto* wg = static_cast<uint32_t*>(mr.w_gate) + se * wg_i32;
      auto* wu = static_cast<uint32_t*>(mr.w_up) + se * wg_i32;
      auto* sgp = static_cast<uint8_t*>(mr.s_gate) + se * sg_b;
      auto* sup = static_cast<uint8_t*>(mr.s_up) + se * sg_b;
      vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, wg, pg, H, I);
      vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, wu, pu, H, I);
      vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(g.scale.data), sgp, H,
                                          I, sf_gu);
      vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(u.scale.data), sup, H,
                                          I, sf_gu);
      gg[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(moe.experts_gate_fp4[se].scale2, sf_gu);
      gu[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(moe.experts_up_fp4[se].scale2, sf_gu);
    }
    vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, wd, pd, I, H);
    vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(dn.scale.data), sdp, I, H,
                                        sf_dn);
    gd[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(moe.experts_down_fp4[se].scale2, sf_dn);
  }
  if (fuse) {
    d.b.Copy(d.q, mr.g_gu, gg.data(), gg.size() * sizeof(float));
  } else {
    d.b.Copy(d.q, mr.g_gate, gg.data(), gg.size() * sizeof(float));
    d.b.Copy(d.q, mr.g_up, gu.data(), gu.size() * sizeof(float));
  }
  d.b.Copy(d.q, mr.g_down, gd.data(), gd.size() * sizeof(float));
  d.b.Memset(d.q, mr.workspace, 0, static_cast<size_t>(mr.sms) * 4 * sizeof(int32_t));
  d.b.Synchronize(d.q);  // repack done → safe to free the fp4 originals
  if (fuse) {
    d.b.Free(tmp_w);
    d.b.Free(tmp_s);
  }

  // CRITICAL (matches qwen3_5.cpp BuildMoeMarlinResident tail): the Marlin resident
  // is now the committed compute path, so FREE the per-expert fp4 originals — both
  // the ResidentNvfp4 device transients (d_packed/d_scale, ~expert-tower-sized) and
  // the HOST mirror (packed/scale .bytes). Without this, peak = host copies + device
  // transients + Marlin resident ≈ 3× the expert tower, blowing past the 119 GiB
  // unified pool (a failed Alloc mid-build → null → silent device fault). Safe here:
  // BuildLagunaMoeMarlinResident runs ONLY under LagunaMarlinMoeEnabled(), so the
  // GEMV/CPU paths that read these bytes can never run in this process.
  for (int e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    moe.experts_gate_fp4[se].ReleaseResident();
    moe.experts_up_fp4[se].ReleaseResident();
    moe.experts_down_fp4[se].ReleaseResident();
    moe.experts_gate_fp4[se].packed.ReleaseHost();
    moe.experts_gate_fp4[se].scale.ReleaseHost();
    moe.experts_up_fp4[se].packed.ReleaseHost();
    moe.experts_up_fp4[se].scale.ReleaseHost();
    moe.experts_down_fp4[se].packed.ReleaseHost();
    moe.experts_down_fp4[se].scale.ReleaseHost();
  }
  mr.ready = true;
}
}  // namespace

// Single-token (T=1) routed-expert MoE via Marlin W4A16. Returns the combined
// routed-expert contribution [H] (shared expert handled by the caller, as for
// LagunaMoeResidentFp4). Mirrors MoeBlockFusedMarlinCuda's split-w13 body with
// T=1, top_k=Pk. bf16 activation (W4A16), so it IGNORES the W4A4 activation-quant
// fields and uses scale2 — exactly vLLM's Marlin config.
std::vector<float> LagunaMoeResidentMarlin(vt::Queue& q, const LagunaMoeWeights& moe,
                                           const std::vector<float>& hrow, int64_t moe_I, int64_t H,
                                           int64_t E, const LagunaRouterSelection& sel) {
  using vt::DType;
  const int64_t Pk = static_cast<int64_t>(sel.ids.size());
  std::vector<float> acc(static_cast<size_t>(H), 0.0F);
  if (Pk == 0) return acc;
  vt::Backend& bk = vt::GetBackend(q.device.type);
  vllm::dense_nvfp4::Dev d{bk, q};
  LagunaMoeMarlinResident& mr = LagunaMoeMarlinResidentFor(&moe);
  if (!mr.ready)
    BuildLagunaMoeMarlinResident(d, moe, static_cast<int>(E), static_cast<int>(H),
                                 static_cast<int>(moe_I), mr);
  void* stream = q.handle;
  const vt::Device dev = q.device;

  const int block = vt::cuda::MarlinMoeAlignBlockSizeSelect(1, static_cast<int>(Pk),
                                                            static_cast<int>(E));
  int max_tok = 0, max_blk = 0;
  vt::cuda::MarlinMoeAlignSizes(1, static_cast<int>(Pk), static_cast<int>(E), block, &max_tok,
                                &max_blk);

  // bf16 activation [1,H] (upload f32 hrow, cast on device).
  vllm::dense_nvfp4::DBuf dh(d, DType::kBF16, {1, H});
  {
    vllm::dense_nvfp4::DBuf xf(d, DType::kF32, {1, H}, hrow.data());
    vt::CastBf16(q, dh.t(), xf.t());
  }
  // router top-k ids/weights for this one token = the Pk selected experts.
  std::vector<int32_t> ids32(static_cast<size_t>(Pk));
  std::vector<float> w1(static_cast<size_t>(Pk));
  for (int64_t s = 0; s < Pk; ++s) {
    ids32[static_cast<size_t>(s)] = static_cast<int32_t>(sel.ids[static_cast<size_t>(s)]);
    w1[static_cast<size_t>(s)] = sel.weights[static_cast<size_t>(s)];
  }
  vllm::dense_nvfp4::DBuf dtid(d, DType::kI32, {1, Pk}, ids32.data());
  vllm::dense_nvfp4::DBuf dtw(d, DType::kF32, {1, Pk}, w1.data());
  vllm::dense_nvfp4::DBuf sorted(d, DType::kI32, {max_tok});
  vllm::dense_nvfp4::DBuf eids(d, DType::kI32, {max_blk});
  vllm::dense_nvfp4::DBuf npad(d, DType::kI32, {1});
  vt::cuda::MarlinMoeAlignBlockSize(stream, static_cast<const int32_t*>(dtid.ptr()), 1,
                                    static_cast<int>(Pk), static_cast<int>(E), block,
                                    static_cast<int32_t*>(sorted.ptr()),
                                    static_cast<int32_t*>(eids.ptr()),
                                    static_cast<int32_t*>(npad.ptr()));

  vt::Tensor wd = vllm::dense_nvfp4::MakeTensor(mr.w_down, DType::kI32, dev, {E, moe_I / 16, H * 2});
  vt::Tensor sd = vllm::dense_nvfp4::MakeTensor(mr.s_down, DType::kI8, dev, {E, moe_I / 16, H});
  vt::Tensor gd = vllm::dense_nvfp4::MakeTensor(mr.g_down, DType::kF32, dev, {E});
  vt::Tensor ws = vllm::dense_nvfp4::MakeTensor(mr.workspace, DType::kI32, dev, {mr.sms * 4});

  const int bi = block, Pki = static_cast<int>(Pk), Hi = static_cast<int>(H),
            Ii = static_cast<int>(moe_I);
  vllm::dense_nvfp4::DBuf dact(d, DType::kBF16, {Pk, moe_I});
  if (mr.fused_w13) {  // LEVER B: ONE size_n=2I grouped GEMM + SiluAndMul (see Into variant).
    vt::Tensor wgu =
        vllm::dense_nvfp4::MakeTensor(mr.w_gu, DType::kI32, dev, {E, H / 16, 2 * moe_I * 2});
    vt::Tensor sgu = vllm::dense_nvfp4::MakeTensor(mr.s_gu, DType::kI8, dev, {E, H / 16, 2 * moe_I});
    vt::Tensor ggu = vllm::dense_nvfp4::MakeTensor(mr.g_gu, DType::kF32, dev, {E});
    vllm::dense_nvfp4::DBuf dgu(d, DType::kBF16, {Pk, 2 * moe_I});
    vt::MoeGroupedGemmNvfp4Marlin(q, dgu.t(), dh.t(), wgu, sgu, ggu, ws, sorted.t(), eids.t(),
                                  npad.t(), dtw.t(),
                                  vt::MoeMarlinArgs{bi, Pki, 1, 2 * Ii, Hi, false});
    vt::SiluAndMul(q, dact.t(), dgu.t());
  } else {
    vt::Tensor wg =
        vllm::dense_nvfp4::MakeTensor(mr.w_gate, DType::kI32, dev, {E, H / 16, moe_I * 2});
    vt::Tensor wu = vllm::dense_nvfp4::MakeTensor(mr.w_up, DType::kI32, dev, {E, H / 16, moe_I * 2});
    vt::Tensor sg = vllm::dense_nvfp4::MakeTensor(mr.s_gate, DType::kI8, dev, {E, H / 16, moe_I});
    vt::Tensor su = vllm::dense_nvfp4::MakeTensor(mr.s_up, DType::kI8, dev, {E, H / 16, moe_I});
    vt::Tensor gg = vllm::dense_nvfp4::MakeTensor(mr.g_gate, DType::kF32, dev, {E});
    vt::Tensor gu = vllm::dense_nvfp4::MakeTensor(mr.g_up, DType::kF32, dev, {E});
    vllm::dense_nvfp4::DBuf dgate(d, DType::kBF16, {Pk, moe_I});
    vllm::dense_nvfp4::DBuf dup(d, DType::kBF16, {Pk, moe_I});
    vt::MoeGroupedGemmNvfp4Marlin(q, dgate.t(), dh.t(), wg, sg, gg, ws, sorted.t(), eids.t(),
                                  npad.t(), dtw.t(), vt::MoeMarlinArgs{bi, Pki, 1, Ii, Hi, false});
    vt::MoeGroupedGemmNvfp4Marlin(q, dup.t(), dh.t(), wu, su, gu, ws, sorted.t(), eids.t(),
                                  npad.t(), dtw.t(), vt::MoeMarlinArgs{bi, Pki, 1, Ii, Hi, false});
    vt::MoeSiluMul(q, dact.t(), dgate.t(), dup.t());
  }

  vllm::dense_nvfp4::DBuf ddown(d, DType::kBF16, {Pk, H});
  vt::MoeGroupedGemmNvfp4Marlin(q, ddown.t(), dact.t(), wd, sd, gd, ws, sorted.t(),
                                eids.t(), npad.t(), dtw.t(),
                                vt::MoeMarlinArgs{bi, 1, Pki, Hi, Ii, false});
  vt::Tensor expert_out = vllm::dense_nvfp4::MakeTensor(ddown.ptr(), DType::kBF16, dev, {1, Pk, H});
  vllm::dense_nvfp4::DBuf dout(d, DType::kBF16, {1, H});
  vt::MoeCombine(q, dout.t(), expert_out, dtw.t());

  // bf16 [H] -> f32 acc (bf16 is the top 16 bits of f32; exact).
  std::vector<uint16_t> hb(static_cast<size_t>(H));
  dout.Download(d, hb.data());
  for (int64_t i = 0; i < H; ++i) {
    const uint32_t u = static_cast<uint32_t>(hb[static_cast<size_t>(i)]) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    acc[static_cast<size_t>(i)] = f;
  }
  return acc;
}

// Device-in/device-out variant for the resident decode: takes a DEVICE f32
// activation [H] + DEVICE i32 ids[Pk] + DEVICE f32 weights[Pk] (from the on-device
// router), runs the SAME Marlin W4A16 grouped chain, and writes the f32 [H] combine
// into out_dev (no host download → no per-MoE-layer drain). Same kernels ⇒ same
// device-regime near-tie as LagunaMoeResidentMarlin.
void LagunaMoeResidentMarlinInto(vt::Queue& q, const LagunaMoeWeights& moe, const float* hn_dev,
                                 int64_t moe_I, int64_t H, int64_t E, const int32_t* ids_dev,
                                 const float* w_dev, int64_t Pk, float* out_dev,
                                 const void* hn_bf16 = nullptr, void* out_bf16 = nullptr) {
  using vt::DType;
  if (Pk == 0) return;
  vt::Backend& bk = vt::GetBackend(q.device.type);
  vllm::dense_nvfp4::Dev d{bk, q};
  LagunaMoeMarlinResident& mr = LagunaMoeMarlinResidentFor(&moe);
  if (!mr.ready)
    BuildLagunaMoeMarlinResident(d, moe, static_cast<int>(E), static_cast<int>(H),
                                 static_cast<int>(moe_I), mr);
  void* stream = q.handle;
  const vt::Device dev = q.device;
  const int block = vt::cuda::MarlinMoeAlignBlockSizeSelect(1, static_cast<int>(Pk),
                                                            static_cast<int>(E));
  int max_tok = 0, max_blk = 0;
  vt::cuda::MarlinMoeAlignSizes(1, static_cast<int>(Pk), static_cast<int>(E), block, &max_tok,
                                &max_blk);
  // Wrap the async-produced device inputs (hn_dev/ids_dev/w_dev, written on-stream by
  // the router GEMM + sigmoid_topk with NO drain before) with MakeTensor — read them
  // ON-STREAM (ordered), NOT via a DBuf host-Copy which would snapshot stale data.
  // VT_LAGUNA_MOE_ONECAST: when the caller passes a persistent bf16 copy of hn (cast once
  // per MoE layer), read it on-stream and SKIP the redundant internal CastBf16 — byte-
  // identical (vt::CastBf16 is a deterministic truncation). Else cast hn→bf16 here as before.
  vllm::dense_nvfp4::DBuf dh(d, DType::kBF16, {1, H});
  vt::Tensor act_bf16 = dh.t();
  if (hn_bf16 != nullptr) {
    act_bf16 = vllm::dense_nvfp4::MakeTensor(const_cast<void*>(hn_bf16), DType::kBF16, dev, {1, H});
  } else {
    vt::Tensor xf = vllm::dense_nvfp4::MakeTensor(const_cast<float*>(hn_dev), DType::kF32, dev,
                                                 {1, H});
    vt::CastBf16(q, dh.t(), xf);
  }
  vt::Tensor dtw = vllm::dense_nvfp4::MakeTensor(const_cast<float*>(w_dev), DType::kF32, dev,
                                               {1, Pk});
  vllm::dense_nvfp4::DBuf sorted(d, DType::kI32, {max_tok});
  vllm::dense_nvfp4::DBuf eids(d, DType::kI32, {max_blk});
  vllm::dense_nvfp4::DBuf npad(d, DType::kI32, {1});
  vt::cuda::MarlinMoeAlignBlockSize(stream, ids_dev, 1, static_cast<int>(Pk), static_cast<int>(E),
                                    block, static_cast<int32_t*>(sorted.ptr()),
                                    static_cast<int32_t*>(eids.ptr()),
                                    static_cast<int32_t*>(npad.ptr()));
  vt::Tensor wd = vllm::dense_nvfp4::MakeTensor(mr.w_down, DType::kI32, dev, {E, moe_I / 16, H * 2});
  vt::Tensor sd = vllm::dense_nvfp4::MakeTensor(mr.s_down, DType::kI8, dev, {E, moe_I / 16, H});
  vt::Tensor gd = vllm::dense_nvfp4::MakeTensor(mr.g_down, DType::kF32, dev, {E});
  vt::Tensor ws = vllm::dense_nvfp4::MakeTensor(mr.workspace, DType::kI32, dev, {mr.sms * 4});
  const int bi = block, Pki = static_cast<int>(Pk), Hi = static_cast<int>(H),
            Ii = static_cast<int>(moe_I);
  vllm::dense_nvfp4::DBuf dact(d, DType::kBF16, {Pk, moe_I});
  if (mr.fused_w13) {
    // LEVER B: ONE grouped GEMM over the N-concatenated w13 (size_n=2*moe_I → [Pk,2I]) +
    // one SiluAndMul over the halves — vLLM's marlin_moe shape (3 grouped GEMMs → 2).
    // SiluAndMul(gate=dgu[:,:I], up=dgu[:,I:]) is the SAME f32 silu + bf16 store as
    // MoeSiluMul, so per-element byte-identical to the split path given equal GEMM output.
    vt::Tensor wgu =
        vllm::dense_nvfp4::MakeTensor(mr.w_gu, DType::kI32, dev, {E, H / 16, 2 * moe_I * 2});
    vt::Tensor sgu = vllm::dense_nvfp4::MakeTensor(mr.s_gu, DType::kI8, dev, {E, H / 16, 2 * moe_I});
    vt::Tensor ggu = vllm::dense_nvfp4::MakeTensor(mr.g_gu, DType::kF32, dev, {E});
    vllm::dense_nvfp4::DBuf dgu(d, DType::kBF16, {Pk, 2 * moe_I});
    vt::MoeGroupedGemmNvfp4Marlin(q, dgu.t(), act_bf16, wgu, sgu, ggu, ws, sorted.t(), eids.t(),
                                  npad.t(), dtw, vt::MoeMarlinArgs{bi, Pki, 1, 2 * Ii, Hi, false});
    vt::SiluAndMul(q, dact.t(), dgu.t());
  } else {
    vt::Tensor wg =
        vllm::dense_nvfp4::MakeTensor(mr.w_gate, DType::kI32, dev, {E, H / 16, moe_I * 2});
    vt::Tensor wu = vllm::dense_nvfp4::MakeTensor(mr.w_up, DType::kI32, dev, {E, H / 16, moe_I * 2});
    vt::Tensor sg = vllm::dense_nvfp4::MakeTensor(mr.s_gate, DType::kI8, dev, {E, H / 16, moe_I});
    vt::Tensor su = vllm::dense_nvfp4::MakeTensor(mr.s_up, DType::kI8, dev, {E, H / 16, moe_I});
    vt::Tensor gg = vllm::dense_nvfp4::MakeTensor(mr.g_gate, DType::kF32, dev, {E});
    vt::Tensor gu = vllm::dense_nvfp4::MakeTensor(mr.g_up, DType::kF32, dev, {E});
    vllm::dense_nvfp4::DBuf dgate(d, DType::kBF16, {Pk, moe_I});
    vllm::dense_nvfp4::DBuf dup(d, DType::kBF16, {Pk, moe_I});
    vt::MoeGroupedGemmNvfp4Marlin(q, dgate.t(), act_bf16, wg, sg, gg, ws, sorted.t(), eids.t(),
                                  npad.t(), dtw, vt::MoeMarlinArgs{bi, Pki, 1, Ii, Hi, false});
    vt::MoeGroupedGemmNvfp4Marlin(q, dup.t(), act_bf16, wu, su, gu, ws, sorted.t(), eids.t(), npad.t(),
                                  dtw, vt::MoeMarlinArgs{bi, Pki, 1, Ii, Hi, false});
    vt::MoeSiluMul(q, dact.t(), dgate.t(), dup.t());
  }
  vllm::dense_nvfp4::DBuf ddown(d, DType::kBF16, {Pk, H});
  vt::MoeGroupedGemmNvfp4Marlin(q, ddown.t(), dact.t(), wd, sd, gd, ws, sorted.t(), eids.t(),
                                npad.t(), dtw, vt::MoeMarlinArgs{bi, 1, Pki, Hi, Ii, false});
  vt::Tensor expert_out = vllm::dense_nvfp4::MakeTensor(ddown.ptr(), DType::kBF16, dev, {1, Pk, H});
  // VT_LAGUNA_TAIL_FUSED: when the caller passes a persistent bf16 out buffer, MoeCombine
  // writes its bf16 result straight there and the caller's bf16-x1 fused_add2_rmsnorm
  // widens it in-kernel — SKIPS the standalone CastF32 node (byte-exact: same bf16 bytes,
  // same widen). Else the default bf16-combine + CastF32-to-f32 path (unchanged).
  if (out_bf16 != nullptr) {
    vt::Tensor ob = vllm::dense_nvfp4::MakeTensor(out_bf16, DType::kBF16, dev, {1, H});
    vt::MoeCombine(q, ob, expert_out, dtw);
  } else {
    vllm::dense_nvfp4::DBuf dout(d, DType::kBF16, {1, H});
    vt::MoeCombine(q, dout.t(), expert_out, dtw);
    vt::Tensor ot = vllm::dense_nvfp4::MakeTensor(out_dev, DType::kF32, dev, {1, H});
    vt::CastF32(q, ot, dout.t());  // bf16 -> f32, device (no download)
  }
}

// ── VT_LAGUNA_SHARED_FP4: the per-layer SHARED expert via the SAME Marlin W4A16
//    single-expert (num_experts=1) grouped GEMM the routed experts win on, instead
//    of the bf16 GEMV over the dequantized shared weights. Device-in (hn_dev [1,H]
//    f32, the post-attn norm) -> device-out (so_dev [1,H] f32, added to the residual
//    by the caller). W4A16 regime: bf16 activation, fp4 weight, scale2 — the fp4
//    weight is 4x fewer DRAM bytes than the bf16 copy on this DRAM-bound M=1 GEMV.
//    Reuses the dense_nvfp4 single-expert primitives (GateUpFusedMarlinD +
//    MatmulNvfp4MarlinD, the validated 35B W4A16 path). CAPTURE-SAFE: the per-weight
//    Marlin residents build lazily on the gstate-0 eager warm-run (their Synchronize
//    runs there, never inside capture), and the pooled DBuf transients are warmed the
//    same way LagunaMoeResidentMarlinInto's are. ──────────────────────────────────
void LagunaSharedExpertMarlinInto(vt::Queue& q, const LagunaMoeWeights& moe, const float* hn_dev,
                                  int64_t H, float* so_dev, const void* hn_bf16 = nullptr) {
  namespace dn = vllm::dense_nvfp4;
  using vt::DType;
  vt::Backend& bk = vt::GetBackend(q.device.type);
  dn::Dev d{bk, q};
  const vt::Device dev = q.device;
  const int64_t K = moe.shared_gate_fp4.k;  // == H (shared expert in-features)
  const int64_t N = moe.shared_gate_fp4.n;  // shared_expert_intermediate_size
  // cast the post-attn activation to bf16 [1,K] (the W4A16 GEMM's activation dtype).
  // VT_LAGUNA_MOE_ONECAST: reuse the caller's single bf16 hn (K==H) and skip this cast —
  // byte-identical to casting here (deterministic truncation of the same hn).
  dn::DBuf xb(d, DType::kBF16, {1, K});
  vt::Tensor act_bf16 = xb.t();
  if (hn_bf16 != nullptr) {
    act_bf16 = dn::MakeTensor(const_cast<void*>(hn_bf16), DType::kBF16, dev, {1, K});
  } else {
    vt::Tensor xf = dn::MakeTensor(const_cast<float*>(hn_dev), DType::kF32, dev, {1, K});
    vt::CastBf16(q, xb.t(), xf);
  }
  // gate_up: silu(x@gate.T) * (x@up.T) -> bf16 [1,N]. ONE fused Marlin GEMM when the
  // gate/up global scales match (vLLM's merged gate_up layout); else two GEMMs + a
  // bf16 MoeSiluMul (byte-equal given equal GEMM output — the routed-expert fallback).
  dn::DBuf act = [&]() -> dn::DBuf {
    if (dn::GateUpFusedEligible(moe.shared_gate_fp4, moe.shared_up_fp4))
      return dn::GateUpFusedMarlinD(d, act_bf16, moe.shared_gate_fp4, moe.shared_up_fp4);
    dn::DBuf g = dn::MatmulNvfp4MarlinD(d, act_bf16, moe.shared_gate_fp4, DType::kBF16);
    dn::DBuf u = dn::MatmulNvfp4MarlinD(d, act_bf16, moe.shared_up_fp4, DType::kBF16);
    dn::DBuf a(d, DType::kBF16, {1, N});
    vt::MoeSiluMul(q, a.t(), g.t(), u.t());
    return a;
  }();
  // down: act @ down.T -> bf16 [1,H]; upcast into the caller's persistent f32 so_dev.
  dn::DBuf sb = dn::MatmulNvfp4MarlinD(d, act.t(), moe.shared_down_fp4, DType::kBF16);
  vt::Tensor ot = dn::MakeTensor(so_dev, DType::kF32, dev, {1, H});
  vt::CastF32(q, ot, sb.t());
}
#endif  // VT_MARLIN_NVFP4

// N5 campaign-B gate: route the routed experts through the Marlin W4A16 grouped
// MoE GEMM (vLLM's 18.8 kernel). Default OFF (opt-in) until the DGX near-tie + ncu
// gate lands; then flip to default-ON per the parity-enablers-ship-as-defaults
// policy. VT_LAGUNA_MARLIN_MOE=1 enables. Only meaningful when VT_MARLIN_NVFP4
// compiled the path in.
// DSR-ALLOW(S1): Marlin-only environment gate; all consumers share this build guard.
#ifdef VT_MARLIN_NVFP4
inline bool LagunaMarlinMoeEnabled() {
  // DEFAULT ON: the Marlin W4A16 grouped MoE is vLLM's own 18.8-tok/s kernel and the
  // validated fast Laguna-NVFP4 decode path (reproduced 3× on GB10, golden-matching,
  // ~10 tok/s = ~1.5× the fp4-GEMV fallback). It just works with no env; only an
  // explicit VT_LAGUNA_MARLIN_MOE=0 opts back out to the fp4 GEMV/resident path (the
  // same-binary A/B escape hatch). CUDA + VT_MARLIN_NVFP4 only; a CPU/non-marlin build
  // never compiles the branch, so it falls to the fp4 path automatically.
  static const bool on = [] {
    const char* e = std::getenv("VT_LAGUNA_MARLIN_MOE");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}
#endif  // VT_MARLIN_NVFP4

// Keep-quant GEMM against a ROW-SLICE [row_off, row_off+N) of a stacked block
// weight `w` [E*out, K] — the per-expert (moe_*_exps) slice. Rows are whole
// blocks (RowSizeBytes), so the offset is a byte offset and no block is cut.
// Mirrors deepseek_v4.cpp GemmRowSlice.
std::vector<float> LqGemmRowSlice(vt::Queue& q, const OwnedTensor& w,
                                  const std::vector<float>& x, int64_t T, int64_t N,
                                  int64_t K, int64_t row_off) {
  VT_CHECK(vt::IsBlockQuant(w.dtype),
           "laguna keep-quant expert GEMM requires a block-quant stacked weight");
  VT_CHECK(!w.repacked,
           "laguna keep-quant expert slice requires non-repacked blocks");
  VT_CHECK(w.rank == 2 && row_off >= 0 && row_off + N <= w.shape[0] && w.shape[1] == K,
           "laguna keep-quant expert GEMM: slice out of range");
  const size_t row_bytes = vt::RowSizeBytes(w.dtype, K);
  std::vector<float> out(static_cast<size_t>(T) * N);
  vt::Queue cpuq{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const bool on_dev = q.device.type != vt::DeviceType::kCPU;
  vt::Queue& gq = on_dev ? q : cpuq;
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(x.data()),
                                        vt::DType::kF32, gq.device, {T, K});
  vt::Tensor o = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, gq.device, {T, N});
  vt::Tensor wt;
  wt.data = const_cast<uint8_t*>(w.bytes.data()) +
            static_cast<size_t>(row_off) * row_bytes;
  wt.dtype = w.dtype;
  wt.device = gq.device;
  wt.rank = 2;
  wt.shape[0] = N;
  wt.shape[1] = K;
  wt.stride[0] = K;
  wt.stride[1] = 1;
  vt::MatmulBT(gq, o, a, wt);
  if (on_dev) DrainQueue(gq);
  return out;
}

// W8 lever #2 (grouped-expert GEMM): default-ON, `VT_LAGUNA_GROUPED_MOE=0` restores
// the byte-exact per-expert loop in the SAME binary. Mirrors ds4 GroupedMoeEnabled.
inline bool LagunaGroupedMoeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LAGUNA_GROUPED_MOE");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

// The fused gate+up+SwiGLU grouped call: ONE launch, ONE activation quantize,
// producing the same `eact` the two-call arm builds with `GateUpSilu`.
//
// `limit = +inf` is the whole reason this substitution is legal at all. The shared
// op applies a CLAMPED SwiGLU, and its own contract records that an infinite limit
// reduces it to plain `silu(gate) * up` — which is exactly what Laguna's
// `GateUpSilu` computes. A finite limit here would silently clamp activations
// Laguna never clamped.
std::vector<float> LqGemmGroupedFusedGateUp(vt::Queue& q, const OwnedTensor& gate_w,
                                            const OwnedTensor& up_w,
                                            const std::vector<float>& act,
                                            const std::vector<int32_t>& expert_ids,
                                            int64_t P, int64_t N, int64_t K) {
  VT_CHECK(gate_w.dtype == up_w.dtype,
           "laguna fused gate/up: both expert towers must share a block-quant dtype");
  VT_CHECK(vt::IsBlockQuant(gate_w.dtype) && !gate_w.repacked && !up_w.repacked,
           "laguna fused gate/up requires non-repacked block-quant stacked weights");
  VT_CHECK(gate_w.rank == 2 && gate_w.shape[1] == K && up_w.rank == 2 && up_w.shape[1] == K,
           "laguna fused gate/up: weight K mismatch");
  VT_CHECK(static_cast<int64_t>(act.size()) == P * K,
           "laguna fused gate/up: act size mismatch");
  VT_CHECK(static_cast<int64_t>(expert_ids.size()) == P,
           "laguna fused gate/up: expert_ids size mismatch");

  std::vector<float> out(static_cast<size_t>(P) * N);
  vt::Queue cpuq{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const bool on_dev = q.device.type != vt::DeviceType::kCPU;
  vt::Queue& gq = on_dev ? q : cpuq;
  std::vector<int32_t> eids = expert_ids;  // stable buffer for the (unified) tensor
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(act.data()),
                                        vt::DType::kF32, gq.device, {P, K});
  vt::Tensor o = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, gq.device, {P, N});
  vt::Tensor eid =
      vt::Tensor::Contiguous(eids.data(), vt::DType::kI32, gq.device, {P});
  vt::Tensor gw = gate_w.View();
  vt::Tensor uw = up_w.View();
  gw.device = gq.device;  // unified-memory block view retag, as LqGemmGrouped does
  uw.device = gq.device;
  vt::MoeGateUpSwiGLUGrouped(gq, o, a, gw, uw, eid,
                             std::numeric_limits<float>::infinity());
  if (on_dev) DrainQueue(gq);
  return out;
}

// PERF-LAGUNA-FUSED-GATEUP W2 (#2061): route the grouped gate+up pair through the
// FUSED `vt::MoeGateUpSwiGLUGrouped`, which quantizes the activation ONCE where the
// two-call arm quantizes it twice. W11 measured `QuantizeQ8KKernel` at 12.4% of
// decode GPU time and half of that is the duplicate.
//
// DEFAULT-OFF, and the reason is measured rather than cautious. The fused epilogue
// is NOT bit-identical to `GateUpSilu`: the shared op computes
// `g * (1.0f/d) * u` while Laguna computes `(g/d) * u`, a reciprocal-then-multiply
// against a divide, which is two roundings against one. Sampled over 2e6 plausible
// pre-activations: 20.3% of values differ, max 2 ULP, max relative 2.4e-7. Making
// them byte-identical would mean changing the SHARED op's epilogue, which
// DeepSeek-V4 is also gated against, so the divergence is accepted and bounded
// here instead — and the arm stays off until a TOKEN gate says the difference does
// not move an output token.
inline bool LagunaFusedGateUpEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LAGUNA_FUSED_GATEUP");
    return e != nullptr && std::string(e) == "1";
  }();
  return on;
}

// The fused arm's PRECONDITION, checked per call because it is a property of the
// CHECKPOINT rather than of the build. `vt::MoeGateUpSwiGLUGrouped` requires both
// expert towers in the SAME block-quant dtype, and a UD quant varies type per
// tensor by design. W1 measured `Laguna-S-2.1 UD-Q4_K_XL` and found gate and up
// paired on all 47 expert layers, including the one layer that deviates to Q5_K on
// both — but that is one checkpoint, not a guarantee, so a mismatch falls back to
// the two-call arm rather than refusing the load.
inline bool LagunaFusedGateUpUsable(const OwnedTensor& gate_w, const OwnedTensor& up_w) {
  return gate_w.dtype == up_w.dtype && vt::IsBlockQuant(gate_w.dtype) &&
         !gate_w.repacked && !up_w.repacked;
}

// Grouped keep-quant expert GEMM over the stacked [E*N,K] tower: out[P,N] where
// out[p,:] = act[p,:] · weight[expert_ids[p]*N .. +N] (the block row-slice for that
// expert). ONE vt::MatmulBTQuantGrouped launch replaces P per-expert LqGemmRowSlice
// matvecs. BYTE-IDENTICAL to the per-expert path — the grouped op's CPU provider loops
// the EXACT kMatmulBTQuant per group; the CUDA provider is the same integer-dot core
// (ds4-gated byte-exact). Routes through the SHARED vt keep-quant grouped op (fold
// policy: no hand-rolled per-model kernel). Mirror of deepseek_v4.cpp
// GemmGroupedExpertsKq (drains before the local eids/out buffers leave scope).
std::vector<float> LqGemmGrouped(vt::Queue& q, const OwnedTensor& w,
                                 const std::vector<float>& act,
                                 const std::vector<int32_t>& expert_ids, int64_t P,
                                 int64_t N, int64_t K) {
  VT_CHECK(vt::IsBlockQuant(w.dtype) && !w.repacked,
           "laguna grouped expert GEMM requires a non-repacked block-quant stacked weight");
  VT_CHECK(w.rank == 2 && w.shape[1] == K,
           "laguna grouped expert GEMM: weight K mismatch");
  VT_CHECK(static_cast<int64_t>(act.size()) == P * K,
           "laguna grouped expert GEMM: act size mismatch");
  VT_CHECK(static_cast<int64_t>(expert_ids.size()) == P,
           "laguna grouped expert GEMM: expert_ids size mismatch");
  std::vector<float> out(static_cast<size_t>(P) * N);
  vt::Queue cpuq{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const bool on_dev = q.device.type != vt::DeviceType::kCPU;
  vt::Queue& gq = on_dev ? q : cpuq;
  std::vector<int32_t> eids = expert_ids;  // stable buffer for the (unified) tensor
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(act.data()),
                                        vt::DType::kF32, gq.device, {P, K});
  vt::Tensor o = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, gq.device, {P, N});
  vt::Tensor eid =
      vt::Tensor::Contiguous(eids.data(), vt::DType::kI32, gq.device, {P});
  vt::Tensor wt = w.View();
  wt.device = gq.device;  // unified-memory block view retag
  vt::MatmulBTQuantGrouped(gq, o, a, wt, eid);
  if (on_dev) DrainQueue(gq);
  return out;
}

// ── W6 shared building blocks (used by BOTH the stateless full-recompute and the
//    KV-cached incremental forward, so the two paths are bit-identical BY SHARING
//    the exact same float ops — the moved code is verbatim from the W5 forward). ──

// Embed gather: hidden[T,H] = embed_table[token_ids]. Gathers ONLY the T needed
// rows directly from the (f32/bf16) table bytes — BIT-IDENTICAL to the prior
// ReadF32(whole-table)-then-gather (same per-element f32/bf16→f32 conversion,
// same rows), but avoids materializing the full [Vsz,H] table (~1.23 GB, ~311M
// element-converts) on EVERY decode token — the dominant host-orchestration
// waste measured in the W7 speed profile (laguna-s21-w7-speed-2026-07-31.md #5).
std::vector<float> LagunaEmbed(const OwnedTensor& embed_t,
                               const std::vector<int32_t>& token_ids, int64_t H,
                               int64_t Vsz) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  std::vector<float> hidden(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t) {
    const int64_t tok = token_ids[static_cast<size_t>(t)];
    float* dst = hidden.data() + static_cast<size_t>(t * H);
    StageLagunaGraphEmbedding(embed_t, static_cast<int32_t>(tok), H, Vsz, dst);
  }
  return hidden;
}

// GQA attention of `Tq` queries (rows of `q`, global positions `q_pos`) against
// `kv_rows` cached K/V (rows of `k`/`v`, global positions `kv_pos`). window==0 =>
// full causal; window>0 => sliding (score only kv with 0 <= q_pos - kv_pos <
// window). Returns attn[Tq, Hq*Dh]. This is the VERBATIM W5 inner loop, generalized
// to distinct query/kv row sets so the KV-cached decode (Tq=1, kv=history) reuses
// the identical float ops as the full-recompute (Tq==kv_rows, q_pos==kv_pos).
std::vector<float> LagunaAttention(const std::vector<float>& q,
                                   const std::vector<float>& k,
                                   const std::vector<float>& v, int64_t Tq,
                                   int64_t kv_rows, int64_t Hq, int64_t Hkv,
                                   int64_t Dh, int64_t group,
                                   const std::vector<int64_t>& q_pos,
                                   const std::vector<int64_t>& kv_pos,
                                   int64_t window) {
  const int64_t qdim = Hq * Dh;
  std::vector<float> attn(static_cast<size_t>(Tq * qdim), 0.0F);
  const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
  for (int64_t h = 0; h < Hq; ++h) {
    const int64_t kvh = h / group;
    for (int64_t i = 0; i < Tq; ++i) {
      const int64_t pi = q_pos[static_cast<size_t>(i)];
      float maxs = -std::numeric_limits<float>::infinity();
      std::vector<float> logit(static_cast<size_t>(kv_rows),
                               -std::numeric_limits<float>::infinity());
      for (int64_t j = 0; j < kv_rows; ++j) {
        const int64_t pj = kv_pos[static_cast<size_t>(j)];
        if (pj > pi) continue;
        if (window > 0 && pi - pj >= window) continue;
        const float* qrow = q.data() + static_cast<size_t>((i * Hq + h) * Dh);
        const float* krow = k.data() + static_cast<size_t>((j * Hkv + kvh) * Dh);
        float dot = 0.0F;
        for (int64_t d = 0; d < Dh; ++d) dot += qrow[d] * krow[d];
        dot *= scale;
        logit[static_cast<size_t>(j)] = dot;
        maxs = std::max(maxs, dot);
      }
      float denom = 0.0F;
      for (int64_t j = 0; j < kv_rows; ++j) {
        if (logit[static_cast<size_t>(j)] == -std::numeric_limits<float>::infinity())
          continue;
        const float e = std::exp(logit[static_cast<size_t>(j)] - maxs);
        logit[static_cast<size_t>(j)] = e;
        denom += e;
      }
      float* ao = attn.data() + static_cast<size_t>((i * Hq + h) * Dh);
      for (int64_t j = 0; j < kv_rows; ++j) {
        const float ww = logit[static_cast<size_t>(j)];
        if (ww == -std::numeric_limits<float>::infinity() || ww == 0.0F) continue;
        const float pw = ww / denom;
        const float* vrow = v.data() + static_cast<size_t>((j * Hkv + kvh) * Dh);
        for (int64_t d = 0; d < Dh; ++d) ao[d] += pw * vrow[d];
      }
    }
  }
  return attn;
}

// FFN block: dense SwiGLU (layer 0) or ungrouped sigmoid-noaux MoE (layers 1..47).
// Consumes hn2[T,H] (post-attn RMSNorm) and returns f[T,H]. VERBATIM keep-quant W5
// FFN, shared by both forwards.
std::vector<float> LagunaFfnBlock(vt::Queue& q, const LagunaLayerWeights& lw,
                                  const LagunaParams& p,
                                  const std::vector<float>& hn2, int64_t T) {
  const int64_t H = p.hidden_size;
  std::vector<float> f(static_cast<size_t>(T * H), 0.0F);
  if (lw.is_dense) {
    const int64_t I = p.intermediate_size;
    const std::vector<float> g = LqGemm(q, lw.mlp.gate_proj, hn2, T, I, H);
    const std::vector<float> u = LqGemm(q, lw.mlp.up_proj, hn2, T, I, H);
    const std::vector<float> act = GateUpSilu(g, u, T, I);
    f = LqGemm(q, lw.mlp.down_proj, act, T, H, I);
    return f;
  }
  const int64_t moe_I = p.moe_intermediate_size;
  const std::vector<float> router_w = ReadF32(lw.moe.router);
  std::vector<float> bias;
  if (!lw.moe.e_score_correction_bias.Empty())
    bias = ReadF32(lw.moe.e_score_correction_bias);
  const bool has_shared = !lw.moe.shared_gate.Empty();
  // N2 (task #230): safetensors NVFP4 arm — routed experts are per-expert TRUE-W4A4
  // Nvfp4Weight (experts_*_fp4), not the stacked keep-quant OwnedTensor. Everything
  // else on this path (router BF16, shared expert BF16, attention BF16) flows through
  // ReadF32/LqGemm unchanged. The grouped fast-path is keep-quant-only (its op is
  // W4A16 grouped, wrong numerics for W4A4), so it is gated off when fp4.
  const bool fp4 = !lw.moe.experts_gate_fp4.empty();
  if (fp4) {
    VT_CHECK(lw.moe.experts_gate_fp4[0].IsTrueW4A4(),
             "laguna nvfp4 MoE: routed experts must be true-W4A4 (alpha>0)");
    VT_CHECK(lw.moe.experts_up_fp4.size() == lw.moe.experts_gate_fp4.size() &&
                 lw.moe.experts_down_fp4.size() == lw.moe.experts_gate_fp4.size(),
             "laguna nvfp4 MoE: gate/up/down expert counts must match");
  }
  for (int64_t i = 0; i < T; ++i) {
    std::vector<float> hrow(hn2.begin() + static_cast<int64_t>(i * H),
                            hn2.begin() + static_cast<int64_t>((i + 1) * H));
    const std::vector<float> rlog = MatmulNK(hrow, router_w, 1, p.num_experts, H);  // [E]
    const LagunaRouterSelection sel = LagunaUngroupedRouterTopK(
        rlog, bias, p.num_experts_per_tok, p.norm_topk_prob,
        p.moe_routed_scaling_factor);
    std::vector<float> acc(static_cast<size_t>(H), 0.0F);
    const int64_t Pk = static_cast<int64_t>(sel.ids.size());
    if (!fp4 && LagunaGroupedMoeEnabled() && Pk > 0) {
      // W8 lever #2: collapse this token's Pk per-expert gate/up/down matvecs into 3
      // grouped launches. Row s := (this token's hrow, expert sel.ids[s]) IN SLOT
      // ORDER, so eo[s] == the per-expert down output for expert sel.ids[s] and the
      // acc combine below runs in the exact same order as the per-expert fallback →
      // byte-identical. gate/up/down each = ONE vt::MatmulBTQuantGrouped.
      std::vector<int32_t> eids(sel.ids.begin(), sel.ids.end());
      std::vector<float> arep(static_cast<size_t>(Pk) * H);
      for (int64_t s = 0; s < Pk; ++s)
        std::memcpy(arep.data() + static_cast<size_t>(s) * H, hrow.data(),
                    static_cast<size_t>(H) * sizeof(float));
      // W2: one fused launch when the arm is on AND the towers pair; otherwise the
      // two-call arm, unchanged and still the reference.
      std::vector<float> eact;
      if (LagunaFusedGateUpEnabled() &&
          LagunaFusedGateUpUsable(lw.moe.experts_gate, lw.moe.experts_up)) {
        eact = LqGemmGroupedFusedGateUp(q, lw.moe.experts_gate, lw.moe.experts_up,
                                        arep, eids, Pk, moe_I, H);
      } else {
        const std::vector<float> eg =
            LqGemmGrouped(q, lw.moe.experts_gate, arep, eids, Pk, moe_I, H);
        const std::vector<float> eu =
            LqGemmGrouped(q, lw.moe.experts_up, arep, eids, Pk, moe_I, H);
        eact = GateUpSilu(eg, eu, Pk, moe_I);
      }
      const std::vector<float> eo =
          LqGemmGrouped(q, lw.moe.experts_down, eact, eids, Pk, H, moe_I);
      for (int64_t s = 0; s < Pk; ++s) {
        const float wgt = sel.weights[static_cast<size_t>(s)];
        const float* eor = eo.data() + static_cast<size_t>(s) * H;
        for (int64_t d = 0; d < H; ++d)
          acc[static_cast<size_t>(d)] += wgt * eor[d];
      }
    } else if (fp4 && q.device.type != vt::DeviceType::kCPU && Pk > 0) {
      // N5 device-resident MoE (task #234): the whole token's routed experts as ONE
      // async device chain, draining ONCE (vs ~Pk×3 per-GEMM syncs). Byte-neutral to
      // the per-expert path up to the fp4-op near-tie; the CPU path below stays the
      // reference so the run-gate is unchanged. Measures whether killing the syncs
      // wins EAGER on Laguna's fast-kernel profile (ds4's slow-glue precedent may not
      // transfer). VT_LAGUNA_RESIDENT_MOE=0 forces the per-expert path for an A/B.
      const char* dis = std::getenv("VT_LAGUNA_RESIDENT_MOE");
      // Dispatch branch onto the Marlin resident MoE above; carries an `else` fallback to
      // the per-expert path, so only the BRANCH is build-gated. Same repair owed.
      // DSR-ALLOW(S1): Marlin resident MoE dispatch branch (has an else fallback).
#ifdef VT_MARLIN_NVFP4
      if (LagunaMarlinMoeEnabled()) {
        // N5 campaign-B: route the routed experts through vLLM's 18.8 Marlin W4A16
        // grouped GEMM (opt-in VT_LAGUNA_MARLIN_MOE=1 until the DGX gate lands).
        const int64_t E = static_cast<int64_t>(lw.moe.experts_gate_fp4.size());
        const std::vector<float> racc =
            LagunaMoeResidentMarlin(q, lw.moe, hrow, moe_I, H, E, sel);
        for (int64_t d = 0; d < H; ++d) acc[static_cast<size_t>(d)] += racc[static_cast<size_t>(d)];
      } else
#endif
      if (dis == nullptr || dis[0] != '0') {
        const std::vector<float> racc = LagunaMoeResidentFp4(q, lw.moe, hrow, moe_I, H, sel);
        for (int64_t d = 0; d < H; ++d) acc[static_cast<size_t>(d)] += racc[static_cast<size_t>(d)];
      } else {
        for (size_t s = 0; s < sel.ids.size(); ++s) {
          const int64_t id = sel.ids[s];
          const std::vector<float> eg = LqGemmNvfp4Fp4(
              q, lw.moe.experts_gate_fp4[static_cast<size_t>(id)], hrow, 1, moe_I, H);
          const std::vector<float> eu = LqGemmNvfp4Fp4(
              q, lw.moe.experts_up_fp4[static_cast<size_t>(id)], hrow, 1, moe_I, H);
          const std::vector<float> eact = GateUpSilu(eg, eu, 1, moe_I);
          const std::vector<float> eo = LqGemmNvfp4Fp4(
              q, lw.moe.experts_down_fp4[static_cast<size_t>(id)], eact, 1, H, moe_I);
          const float wgt = sel.weights[s];
          for (int64_t d = 0; d < H; ++d) acc[static_cast<size_t>(d)] += wgt * eo[static_cast<size_t>(d)];
        }
      }
    } else {
      for (size_t s = 0; s < sel.ids.size(); ++s) {
        const int64_t id = sel.ids[s];
        // NVFP4 arm: per-expert TRUE-W4A4 GEMMs (fp4 activation + alpha-scaled
        // accumulate); keep-quant arm: stacked-block row-slice matvecs. Both
        // produce [1,moe_I] gate/up then [1,H] down, so the SwiGLU + combine below
        // are shared verbatim.
        const std::vector<float> eg =
            fp4 ? LqGemmNvfp4Fp4(q, lw.moe.experts_gate_fp4[static_cast<size_t>(id)],
                                 hrow, 1, moe_I, H)
                : LqGemmRowSlice(q, lw.moe.experts_gate, hrow, 1, moe_I, H, id * moe_I);
        const std::vector<float> eu =
            fp4 ? LqGemmNvfp4Fp4(q, lw.moe.experts_up_fp4[static_cast<size_t>(id)],
                                 hrow, 1, moe_I, H)
                : LqGemmRowSlice(q, lw.moe.experts_up, hrow, 1, moe_I, H, id * moe_I);
        const std::vector<float> eact = GateUpSilu(eg, eu, 1, moe_I);
        const std::vector<float> eo =
            fp4 ? LqGemmNvfp4Fp4(q, lw.moe.experts_down_fp4[static_cast<size_t>(id)],
                                 eact, 1, H, moe_I)
                : LqGemmRowSlice(q, lw.moe.experts_down, eact, 1, H, moe_I, id * H);
        const float wgt = sel.weights[s];
        for (int64_t d = 0; d < H; ++d)
          acc[static_cast<size_t>(d)] += wgt * eo[static_cast<size_t>(d)];
      }
    }
    if (has_shared) {
      const std::vector<float> sg = LqGemm(q, lw.moe.shared_gate, hrow, 1, moe_I, H);
      const std::vector<float> su = LqGemm(q, lw.moe.shared_up, hrow, 1, moe_I, H);
      const std::vector<float> sact = GateUpSilu(sg, su, 1, moe_I);
      const std::vector<float> so = LqGemm(q, lw.moe.shared_down, sact, 1, H, moe_I);
      for (int64_t d = 0; d < H; ++d)
        acc[static_cast<size_t>(d)] += so[static_cast<size_t>(d)];
    }
    std::copy(acc.begin(), acc.end(), f.begin() + static_cast<int64_t>(i * H));
  }
  return f;
}

// Final RMSNorm -> lm_head (untied keep-quant / tied f32) -> logits, with an
// optional gather over LOCAL row indices. VERBATIM W5 tail, shared by both forwards.
std::vector<float> LagunaFinalLogits(vt::Queue& q, const LagunaWeights& weights,
                                     const std::vector<float>& hidden, int64_t T,
                                     const std::vector<int32_t>& logits_indices) {
  const LagunaParams& p = weights.params;
  const int64_t H = p.hidden_size;
  const int64_t Vsz = p.vocab_size;
  const float eps = p.rms_norm_eps;
  const std::vector<float> hn = RmsNorm(hidden, ReadF32(weights.norm), T, H, eps);
  const bool gather =
      !logits_indices.empty() && static_cast<int64_t>(logits_indices.size()) < T;
  std::vector<float> src;
  int64_t n_out;
  if (gather) {
    n_out = static_cast<int64_t>(logits_indices.size());
    src.resize(static_cast<size_t>(n_out * H));
    for (int64_t r = 0; r < n_out; ++r)
      std::memcpy(src.data() + static_cast<size_t>(r * H),
                  hn.data() + static_cast<size_t>(logits_indices[static_cast<size_t>(r)] * H),
                  static_cast<size_t>(H) * sizeof(float));
  } else {
    n_out = T;
    src = hn;
  }
  const bool tied = p.tie_word_embeddings || weights.lm_head.Empty();
  if (tied) return MatmulNK(src, ReadF32(weights.embed), n_out, Vsz, H);
  // M=1 decode fast-path: a dedicated coalesced bf16 GEMV streams the ~616 MB lm_head
  // ONCE at ~roofline — cuBLASLt mis-routes LqGemm's M=1×N=vocab MatmulBT to a batched
  // wmma tile algo (~20% of roofline, the measured #1 Laguna decode GPU cost). Only the
  // T=1 decode (n_out==1) + bf16 tower lm_head (nvfp4 arm) on a CUDA queue with the
  // kLaguna table registered; the CPU / GGUF keep-quant / prefill (n_out>1) paths keep
  // the exact LqGemm reference. NEAR-TIE vs LqGemm (block-reduced dot reorders float
  // adds; accepted device regime, gated vs vLLM).
  if (n_out == 1 && weights.lm_head.dtype == vt::DType::kBF16 &&
      q.device.type != vt::DeviceType::kCPU && laguna::LagunaDeviceKernelsAvailable()) {
    std::vector<float> logits(static_cast<size_t>(Vsz));
    laguna::LagunaDevice()->lm_head_gemv(
        q, logits.data(), reinterpret_cast<const void*>(weights.lm_head.bytes.data()), src.data(),
        Vsz, H);
    DrainQueue(q);  // logits are consumed on the host by the caller
    return logits;
  }
  return LqGemm(q, weights.lm_head, src, n_out, Vsz, H);
}

}  // namespace

// N5 campaign-B: build ALL routed-expert Marlin residents at model-LOAD time
// (mirrors vLLM's process_weights_after_loading, marlin_utils_fp4.py), instead of
// lazily on the first forward — so the 48L×256E repack cost is paid once at load,
// not as a first-token TTFT spike. No-op unless the Marlin path is enabled + on GPU
// (+ built with VT_MARLIN_NVFP4). The forward's lazy `if (!mr.ready) Build…` then
// finds every resident ready. Safe: only runs under LagunaMarlinMoeEnabled(), so the
// GEMV/CPU paths (which read the fp4 originals it frees) can never run this process.
// External linkage (declared in laguna.h); the anon-namespace helpers it calls stay
// visible here via the anonymous namespace's implicit using-directive.
void LagunaBuildMarlinResidents(vt::Queue& q, const LagunaWeights& w) {
  // Body of the Marlin resident BUILD entry point — device-coupled for the same reason as
  // the struct above (CUDA-leg repack types). Note it ALREADY asks the runtime device
  // question on the next line (`q.device.type == kCPU`); the build-time gate exists only
  // because the enclosed types will not compile without CUDA. Same repair owed.
  // DSR-ALLOW(S1): Marlin resident build entry point, CUDA-leg repack types.
#ifdef VT_MARLIN_NVFP4
  if (q.device.type == vt::DeviceType::kCPU || !LagunaMarlinMoeEnabled()) return;
  vt::Backend& bk = vt::GetBackend(q.device.type);
  vllm::dense_nvfp4::Dev d{bk, q};
  for (const LagunaLayerWeights& lw : w.layers) {
    if (lw.is_dense || lw.moe.experts_gate_fp4.empty()) continue;
    const int E = static_cast<int>(lw.moe.experts_gate_fp4.size());
    const int N = static_cast<int>(lw.moe.experts_gate_fp4[0].n);  // moe_intermediate
    const int K = static_cast<int>(lw.moe.experts_gate_fp4[0].k);  // hidden_size
    LagunaMoeMarlinResident& mr = LagunaMoeMarlinResidentFor(&lw.moe);
    if (!mr.ready) BuildLagunaMoeMarlinResident(d, lw.moe, E, K, N, mr);
  }
#else
  (void)q;
  (void)w;
#endif
}

std::vector<float> LagunaModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const LagunaWeights& weights,
    const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  (void)attn_meta;
  (void)attn_kv;   // reference forward recomputes attention (paged path = W4)
  (void)config;
  (void)queue;
  const LagunaParams& p = weights.params;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = p.hidden_size;
  const int64_t Vsz = p.vocab_size;
  const int64_t Dh = p.head_dim;
  const int64_t Hkv = p.num_key_value_heads;
  const int64_t kvdim = Hkv * Dh;
  const float eps = p.rms_norm_eps;
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "laguna: positions length must match token_ids");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == p.num_hidden_layers,
           "laguna: one LagunaLayerWeights per layer required");

  // Size the RoPE caches to the max position (+1). Built once per regime.
  int64_t max_pos = 0;
  for (int32_t ps : positions) max_pos = std::max<int64_t>(max_pos, ps);
  const int64_t rope_rows = max_pos + 1;
  const std::vector<float> yarn_cache = BuildLagunaFullYarnCosSin(p, rope_rows);
  const std::vector<float> slide_cache = BuildLagunaSlidingCosSin(p, rope_rows);

  // Embed: hidden[T,H] = embed[token_ids].
  const std::vector<float> embed = ReadF32(weights.embed);
  std::vector<float> hidden(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t) {
    const int64_t tok = token_ids[static_cast<size_t>(t)];
    VT_CHECK(tok >= 0 && tok < Vsz, "laguna: token id out of range");
    std::memcpy(hidden.data() + static_cast<size_t>(t * H),
                embed.data() + static_cast<size_t>(tok * H),
                static_cast<size_t>(H) * sizeof(float));
  }

  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const LagunaLayerWeights& lw = weights.layers[static_cast<size_t>(l)];
    const int64_t Hq = p.QHeadsForLayer(l);
    const int64_t qdim = Hq * Dh;
    const int64_t group = p.GqaGroupForLayer(l);
    VT_CHECK(group > 0 && Hq == group * Hkv,
             "laguna: per-layer Q-head count must be a multiple of KV heads");
    const bool global = p.IsGlobalLayer(l);
    const int64_t rd = p.RotaryDimForLayer(l);
    const int64_t window = p.WindowForLayer(l);

    // --- attention ---
    const std::vector<float> hn =
        RmsNorm(hidden, ReadF32(lw.input_norm), T, H, eps);
    std::vector<float> q = MatmulNK(hn, ReadF32(lw.attn.q_proj), T, qdim, H);
    std::vector<float> k = MatmulNK(hn, ReadF32(lw.attn.k_proj), T, kvdim, H);
    std::vector<float> v = MatmulNK(hn, ReadF32(lw.attn.v_proj), T, kvdim, H);
    // Per-head QK-RMSNorm BEFORE RoPE (VERIFIED W4 from the GGUF attn_q/k_norm).
    if (p.has_qk_norm && !lw.attn.q_norm.Empty()) {
      RmsNormHeads(q, ReadF32(lw.attn.q_norm), T, Hq, Dh, eps);
      RmsNormHeads(k, ReadF32(lw.attn.k_norm), T, Hkv, Dh, eps);
    }
    const std::vector<float>& cache = global ? yarn_cache : slide_cache;
    ApplyRope(q, T, Hq, Dh, rd, cache, positions);
    ApplyRope(k, T, Hkv, Dh, rd, cache, positions);

    // GQA attention with the per-layer mask (global full-causal / sliding-window).
    std::vector<float> attn(static_cast<size_t>(T * qdim), 0.0F);
    const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
    for (int64_t h = 0; h < Hq; ++h) {
      const int64_t kvh = h / group;
      for (int64_t i = 0; i < T; ++i) {
        const int64_t pi = positions[static_cast<size_t>(i)];
        // score over all j with pos_j <= pos_i (causal) and, for sliding layers,
        // pos_i - pos_j < window (FA window convention).
        float maxs = -std::numeric_limits<float>::infinity();
        std::vector<float> logit(static_cast<size_t>(T),
                                 -std::numeric_limits<float>::infinity());
        for (int64_t j = 0; j < T; ++j) {
          const int64_t pj = positions[static_cast<size_t>(j)];
          if (pj > pi) continue;
          if (window > 0 && pi - pj >= window) continue;
          const float* qv = q.data() + static_cast<size_t>((i * Hq + h) * Dh);
          const float* kvp = k.data() + static_cast<size_t>((j * Hkv + kvh) * Dh);
          float dot = 0.0F;
          for (int64_t d = 0; d < Dh; ++d) dot += qv[d] * kvp[d];
          dot *= scale;
          logit[static_cast<size_t>(j)] = dot;
          maxs = std::max(maxs, dot);
        }
        float denom = 0.0F;
        for (int64_t j = 0; j < T; ++j) {
          if (logit[static_cast<size_t>(j)] ==
              -std::numeric_limits<float>::infinity())
            continue;
          const float e = std::exp(logit[static_cast<size_t>(j)] - maxs);
          logit[static_cast<size_t>(j)] = e;
          denom += e;
        }
        float* ao = attn.data() + static_cast<size_t>((i * Hq + h) * Dh);
        for (int64_t j = 0; j < T; ++j) {
          const float w = logit[static_cast<size_t>(j)];
          if (w == -std::numeric_limits<float>::infinity() || w == 0.0F) continue;
          const float pw = w / denom;
          const float* vv = v.data() + static_cast<size_t>((j * Hkv + kvh) * Dh);
          for (int64_t d = 0; d < Dh; ++d) ao[d] += pw * vv[d];
        }
      }
    }

    // NEW op (a): per-head softplus attention OUTPUT gate.
    const std::vector<float> glogits =
        MatmulNK(hn, ReadF32(lw.attn.g_proj), T, Hq, H);  // [T,Hq]
    for (int64_t i = 0; i < T; ++i) {
      std::vector<float> row(attn.begin() + static_cast<int64_t>(i * qdim),
                             attn.begin() + static_cast<int64_t>((i + 1) * qdim));
      std::vector<float> gl(glogits.begin() + static_cast<int64_t>(i * Hq),
                            glogits.begin() + static_cast<int64_t>((i + 1) * Hq));
      LagunaSoftplusHeadGate(row, gl, Hq, Dh);
      std::copy(row.begin(), row.end(),
                attn.begin() + static_cast<int64_t>(i * qdim));
    }

    // o_proj + residual.
    const std::vector<float> o = MatmulNK(attn, ReadF32(lw.attn.o_proj), T, H, qdim);
    for (int64_t i = 0; i < T * H; ++i) hidden[static_cast<size_t>(i)] += o[static_cast<size_t>(i)];

    // --- FFN: dense SwiGLU (layer 0) or ungrouped sigmoid-noaux MoE ---
    const std::vector<float> hn2 =
        RmsNorm(hidden, ReadF32(lw.post_attn_norm), T, H, eps);
    std::vector<float> f(static_cast<size_t>(T * H), 0.0F);
    if (lw.is_dense) {
      const int64_t I = p.intermediate_size;
      const std::vector<float> g = MatmulNK(hn2, ReadF32(lw.mlp.gate_proj), T, I, H);
      const std::vector<float> u = MatmulNK(hn2, ReadF32(lw.mlp.up_proj), T, I, H);
      const std::vector<float> act = GateUpSilu(g, u, T, I);
      f = MatmulNK(act, ReadF32(lw.mlp.down_proj), T, H, I);
    } else {
      const int64_t E = p.num_experts;
      const int64_t moe_I = p.moe_intermediate_size;
      const std::vector<float> router_w = ReadF32(lw.moe.router);
      std::vector<float> bias;
      if (!lw.moe.e_score_correction_bias.Empty())
        bias = ReadF32(lw.moe.e_score_correction_bias);
      const std::vector<float> exp_g = ReadF32(lw.moe.experts_gate);  // [E,moeI,H]
      const std::vector<float> exp_u = ReadF32(lw.moe.experts_up);    // [E,moeI,H]
      const std::vector<float> exp_dn = ReadF32(lw.moe.experts_down); // [E,H,moeI]
      const int64_t gu_stride = moe_I * H;
      const int64_t dn_stride = H * moe_I;
      const bool has_shared = !lw.moe.shared_gate.Empty();
      std::vector<float> shared_g, shared_u, shared_dn;
      if (has_shared) {
        shared_g = ReadF32(lw.moe.shared_gate);
        shared_u = ReadF32(lw.moe.shared_up);
        shared_dn = ReadF32(lw.moe.shared_down);
      }
      for (int64_t i = 0; i < T; ++i) {
        std::vector<float> hrow(hn2.begin() + static_cast<int64_t>(i * H),
                                hn2.begin() + static_cast<int64_t>((i + 1) * H));
        const std::vector<float> rlog = MatmulNK(hrow, router_w, 1, E, H);  // [E]
        const LagunaRouterSelection sel = LagunaUngroupedRouterTopK(
            rlog, bias, p.num_experts_per_tok, p.norm_topk_prob,
            p.moe_routed_scaling_factor);
        std::vector<float> acc(static_cast<size_t>(H), 0.0F);
        for (size_t s = 0; s < sel.ids.size(); ++s) {
          const int64_t id = sel.ids[s];
          std::vector<float> eg(exp_g.begin() + static_cast<int64_t>(id * gu_stride),
                                exp_g.begin() + static_cast<int64_t>((id + 1) * gu_stride));
          std::vector<float> eu(exp_u.begin() + static_cast<int64_t>(id * gu_stride),
                                exp_u.begin() + static_cast<int64_t>((id + 1) * gu_stride));
          std::vector<float> edn(exp_dn.begin() + static_cast<int64_t>(id * dn_stride),
                                 exp_dn.begin() + static_cast<int64_t>((id + 1) * dn_stride));
          const std::vector<float> eo = ExpertMlp(hrow, eg, eu, edn, H, moe_I);
          const float w = sel.weights[s];
          for (int64_t d = 0; d < H; ++d) acc[static_cast<size_t>(d)] += w * eo[static_cast<size_t>(d)];
        }
        if (has_shared) {
          const std::vector<float> so = ExpertMlp(hrow, shared_g, shared_u, shared_dn, H, moe_I);
          for (int64_t d = 0; d < H; ++d) acc[static_cast<size_t>(d)] += so[static_cast<size_t>(d)];
        }
        std::copy(acc.begin(), acc.end(),
                  f.begin() + static_cast<int64_t>(i * H));
      }
    }
    for (int64_t i = 0; i < T * H; ++i) hidden[static_cast<size_t>(i)] += f[static_cast<size_t>(i)];
  }

  // Final RMSNorm -> lm_head (untied) -> logits.
  const std::vector<float> hn = RmsNorm(hidden, ReadF32(weights.norm), T, H, eps);
  const bool gather =
      !logits_indices.empty() && static_cast<int64_t>(logits_indices.size()) < T;
  std::vector<float> src;
  int64_t n_out;
  if (gather) {
    n_out = static_cast<int64_t>(logits_indices.size());
    src.resize(static_cast<size_t>(n_out * H));
    for (int64_t r = 0; r < n_out; ++r)
      std::memcpy(src.data() + static_cast<size_t>(r * H),
                  hn.data() + static_cast<size_t>(logits_indices[static_cast<size_t>(r)] * H),
                  static_cast<size_t>(H) * sizeof(float));
  } else {
    n_out = T;
    src = hn;
  }
  const bool tied = p.tie_word_embeddings || weights.lm_head.Empty();
  return MatmulNK(src, ReadF32(tied ? weights.embed : weights.lm_head), n_out, Vsz, H);
}

ForwardLogits LagunaModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const LagunaWeights& weights,
    const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  // The device path reuses the host reference until the W4 device assembly lands
  // (mirrors ds4's ForwardDevice-after-Forward staging).
  return HostLogits(
      LagunaModel::Forward(token_ids, positions, attn_meta, attn_kv, weights,
                           config, queue, logits_indices),
      weights.params.vocab_size);
}

// ════════════════════════════════════════════════════════════════════════════
// W5 — the REAL keep-quant GGUF forward. The `LagunaModel::Forward` composition
// with the ~9 GEMM sites routed through the keep-quant LqGemm/LqGemmRowSlice
// (vt::MatmulBT on the block-typed weights) instead of MatmulNK(ReadF32). All the
// glue (dual-RoPE, per-head QK-RMSNorm, per-head softplus out-gate, ungrouped
// sigmoid-noaux router + routed_scaling, shared expert) is IDENTICAL to the
// unit-gated f32 reference — only the matmul operands change from f32 to
// keep-quant blocks. Stateless whole-sequence recompute (mirrors
// DeepseekV4ForwardGguf); the greedy driver loops it.
std::vector<float> LagunaForwardGguf(const LagunaWeights& weights, vt::Queue& q,
                                     const std::vector<int32_t>& token_ids,
                                     const std::vector<int32_t>& positions,
                                     const std::vector<int32_t>& logits_indices) {
  const LagunaParams& p = weights.params;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = p.hidden_size;
  const int64_t Vsz = p.vocab_size;
  const int64_t Dh = p.head_dim;
  const int64_t Hkv = p.num_key_value_heads;
  const int64_t kvdim = Hkv * Dh;
  const float eps = p.rms_norm_eps;
  VT_CHECK(weights.has_gguf_weights || weights.has_nvfp4_weights,
           "laguna forward: no keep-quant or nvfp4 tower");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "laguna gguf: positions length must match token_ids");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == p.num_hidden_layers,
           "laguna gguf: one LagunaLayerWeights per layer required");

  int64_t max_pos = 0;
  for (int32_t ps : positions) max_pos = std::max<int64_t>(max_pos, ps);
  const int64_t rope_rows = max_pos + 1;
  const std::vector<float> yarn_cache = BuildLagunaFullYarnCosSin(p, rope_rows);
  const std::vector<float> slide_cache = BuildLagunaSlidingCosSin(p, rope_rows);

  // Embed: hidden[T,H] = embed[token_ids] (f32 gather table).
  std::vector<float> hidden = LagunaEmbed(weights.embed, token_ids, H, Vsz);

  std::vector<int64_t> pos64(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) pos64[static_cast<size_t>(t)] = positions[static_cast<size_t>(t)];

  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const LagunaLayerWeights& lw = weights.layers[static_cast<size_t>(l)];
    const int64_t Hq = p.QHeadsForLayer(l);
    const int64_t qdim = Hq * Dh;
    const int64_t group = p.GqaGroupForLayer(l);
    VT_CHECK(group > 0 && Hq == group * Hkv,
             "laguna gguf: per-layer Q-head count must be a multiple of KV heads");
    const bool global = p.IsGlobalLayer(l);
    const int64_t rd = p.RotaryDimForLayer(l);
    const int64_t window = p.WindowForLayer(l);

    // --- attention ---
    const std::vector<float> hn = RmsNorm(hidden, ReadF32(lw.input_norm), T, H, eps);
    std::vector<float> qv = LqGemm(q, lw.attn.q_proj, hn, T, qdim, H);
    std::vector<float> kv = LqGemm(q, lw.attn.k_proj, hn, T, kvdim, H);
    std::vector<float> vv = LqGemm(q, lw.attn.v_proj, hn, T, kvdim, H);
    if (p.has_qk_norm && !lw.attn.q_norm.Empty()) {
      RmsNormHeads(qv, ReadF32(lw.attn.q_norm), T, Hq, Dh, eps);
      RmsNormHeads(kv, ReadF32(lw.attn.k_norm), T, Hkv, Dh, eps);
    }
    const std::vector<float>& cache = global ? yarn_cache : slide_cache;
    ApplyRope(qv, T, Hq, Dh, rd, cache, positions);
    ApplyRope(kv, T, Hkv, Dh, rd, cache, positions);

    std::vector<float> attn = LagunaAttention(qv, kv, vv, T, T, Hq, Hkv, Dh, group,
                                              pos64, pos64, window);

    // per-head softplus attention OUTPUT gate.
    const std::vector<float> glogits = LqGemm(q, lw.attn.g_proj, hn, T, Hq, H);
    for (int64_t i = 0; i < T; ++i) {
      std::vector<float> row(attn.begin() + static_cast<int64_t>(i * qdim),
                             attn.begin() + static_cast<int64_t>((i + 1) * qdim));
      std::vector<float> gl(glogits.begin() + static_cast<int64_t>(i * Hq),
                            glogits.begin() + static_cast<int64_t>((i + 1) * Hq));
      LagunaSoftplusHeadGate(row, gl, Hq, Dh);
      std::copy(row.begin(), row.end(), attn.begin() + static_cast<int64_t>(i * qdim));
    }

    const std::vector<float> o = LqGemm(q, lw.attn.o_proj, attn, T, H, qdim);
    for (int64_t i = 0; i < T * H; ++i) hidden[static_cast<size_t>(i)] += o[static_cast<size_t>(i)];

    // --- FFN: dense SwiGLU (layer 0) or ungrouped sigmoid-noaux MoE ---
    const std::vector<float> hn2 = RmsNorm(hidden, ReadF32(lw.post_attn_norm), T, H, eps);
    const std::vector<float> f = LagunaFfnBlock(q, lw, p, hn2, T);
    for (int64_t i = 0; i < T * H; ++i) hidden[static_cast<size_t>(i)] += f[static_cast<size_t>(i)];
  }

  // Final RMSNorm -> lm_head (untied keep-quant) -> logits.
  return LagunaFinalLogits(q, weights, hidden, T, logits_indices);
}

// ── W6: the KV-CACHED incremental forward. Same keep-quant composition as
//    LagunaForwardGguf, but binds a LagunaKvCache: this call's T tokens append their
//    per-layer POST-RoPE K + RAW V to the cache and the queries attend over the FULL
//    cached history (global layers) / the last-512 window (sliding layers, evicted
//    beyond the window). Prefill = first call (cache.len==0, all prompt tokens);
//    decode = later calls (ONE new token, positions={cache.len}). Token-IDENTICAL to
//    LagunaForwardGguf over the growing context — the KV-cache identity (a token's
//    cached K/V equal what full-recompute would recompute, since RoPE/QK-norm depend
//    only on the token's own position and attention is causal). Mirror of
//    DeepseekV4ForwardGgufCached, extended MLA-latent -> GQA multi-head K/V, plus the
//    NEW sliding-window eviction (grounded in gemma2/3 is_sliding).
// ── N5 device-resident T=1 decode (VT_LAGUNA_RESIDENT_DECODE, default OFF) ─────
// The NVFP4/Marlin arm's per-GEMM host drains (~432 cudaStreamSynchronize/token,
// MEASURED) are the 1.9x gap to vLLM 18.8. This v1 runs the ATTENTION block fully
// device-resident (the 5 kLaguna glue kernels + no-sync bf16 GEMMs), collapsing the
// ~5 per-layer attention-GEMM drains into 2; the FFN reuses the existing host path
// (its own minimal drains) for now (v2 = device-resident FFN). BYTE-EXACT: the
// kLaguna kernels use sequential reductions matching the host float order, and the
// GEMMs are the same bf16 MatmulBT as the current golden. Gated on the golden ids
// staying identical (VT_LAGUNA_RESIDENT_DECODE=0/1 A/B).
inline bool LagunaResidentDecodeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LAGUNA_RESIDENT_DECODE");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// ── Brick A2: capture the resident T=1 decode step into ONE CUDA graph and replay
// it — collapsing the ~700 host kernel launches/token into a single cudaGraphLaunch
// (the path to vLLM-NVFP4's 18.8 tok/s; the A1 eager resident measured ~14). Default
// OFF (VT_LAGUNA_DECODE_GRAPH=1 to enable); the A1 eager resident path stays the
// fallback. Mirror of deepseek_v4.cpp DecodeGraphEnabled()/VT_V4_DECODE_GRAPH. inline
// so an unused CPU build (VT_MARLIN_NVFP4 off, graph excluded) does not -Wunused.
inline bool LagunaDecodeGraphEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LAGUNA_DECODE_GRAPH");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// ── Attention-glue fusions on the resident decode (VT_LAGUNA_GLUE_FUSED, default ON).
// Recovers the device-time the CUDA graph does NOT hide (ramp/drain of ~450 tiny
// under-occupied sequential kernel-nodes/token) by BYTE-EXACT folds that shrink the
// captured node count: L1 folds the per-head softplus out-gate into the attention
// combine store (one fewer kernel/layer); L4 folds each residual-Add + RMSNorm pair
// into ONE vt::FusedChain(kFusedAddRmsNormStd) call (two fewer kernels/layer with the
// Tier-1 interpreter, VT_FUSED_TIER=1). Both are bit-for-bit identical to the toggle-
// off path (same f32 arithmetic, same 256-thread RMSNorm reduction order) — a same-
// binary A/B: VT_LAGUNA_GLUE_FUSED=0 restores the separate kernels. inline so a CPU
// build (VT_MARLIN_NVFP4 off) does not -Wunused.
inline bool LagunaGlueFusedEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LAGUNA_GLUE_FUSED");
    return !(e != nullptr && e[0] == '0');  // default ON; =0 opts out
  }();
  return on;
}

// ── LEVER A: fold the MoE routed-add into the trailing add_rms_norm (VT_LAGUNA_MOE_ADDNORM_FUSED,
// default ON). Under the glue-fused decode a MoE layer runs its residual update + next-layer
// input norm as TWO graph nodes — vt::Add(hidden,doutb) [routed AddKernel] then
// FusedChain(kFusedAddRmsNormStd)(hn,so,next_norm,hidden) [shared add + RMSNorm]. This lever
// collapses them into ONE fused_add2_rmsnorm node/MoE-layer (hidden=(hidden+doutb)+so;
// hn=rms_norm(hidden)*next_norm). BYTE-EXACT (IEEE add commutes + identical 256-thread norm
// reduction) — a same-binary A/B: =0 restores the split Add+FusedChain. inline so a CPU build
// does not -Wunused.
inline bool LagunaMoeAddNormFusedEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LAGUNA_MOE_ADDNORM_FUSED");
    return !(e != nullptr && e[0] == '0');  // default ON; =0 opts out
  }();
  return on;
}

// ── VT_LAGUNA_PREAMBLE_FUSED: fold the GRAPH attention preamble into ONE launch (default
// ON). The decode graph runs the per-layer preamble as FOUR under-occupied M=1 kernel
// nodes — rms_norm_seq(q) + rms_norm_seq(k) + rope_from_cache_g(q) + rope_from_cache_g(k)
// (2×nlayers RMSNorm + 2×nlayers RoPE launches/token). This lever collapses them into ONE
// fused_qk_norm_rope_g node/layer (3 fewer graph nodes/layer), recovering the ramp/drain
// device time the CUDA graph does NOT hide on tiny M=1 kernels — the same residual
// mechanism the glue fusion (L1/L4) already exploited. BYTE-EXACT: the fused kernel uses
// the IDENTICAL 256-thread shared-tree Σx² + 1/sqrtf reduction as rms_norm_seq and the
// IDENTICAL half-split RoPE as rope_from_cache_g (the normed pair stays in a register, an
// exact no-op vs the f32 memory round-trip) — a same-binary A/B: =0 restores the four
// separate kernels. Only taken when the layer has q/k norm (else the unfused 2-RoPE path
// stays). inline so a CPU build (VT_MARLIN_NVFP4 off, graph excluded) does not -Wunused.
inline bool LagunaPreambleFusedEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LAGUNA_PREAMBLE_FUSED");
    return !(e != nullptr && e[0] == '0');  // default ON; =0 opts out
  }();
  return on;
}

// ── On-device greedy sample + embed on the resident decode graph (VT_LAGUNA_ONDEV_SAMPLE).
// Removes the per-step host round-trip: the driver used to Synchronize the graph, download
// the whole [vocab] logits, argmax them on the HOST, then gather the next token's embedding
// on the HOST before the next replay — a measured ~527 us GPU-IDLE gap PER STEP (host
// argmax over the 100352-vocab + embed + the full drain, between two graph replays). With
// this on, the graph argmaxes its logits ON-DEVICE (vt::GreedyArgmax, lowest-index tie =
// the host ArgmaxLastRow winner) and gathers the next step's embedding ON-DEVICE
// (embed_gather), so replay N+1 launches without host work on step N's logits. WHERE argmax
// runs changes, NOT the math ⇒ BYTE-IDENTICAL token stream (=0 vs =1). DEFAULT ON
// (parity-enablers-ship-as-defaults): the DGX GB10 gate proved it byte-exact (160-id stream
// identical, ~/laguna-xs-nvfp4) AND faster — paired drop_caches decode wall +0.28% median
// (removes the ~150 us/step host argmax between graph replays) at GPU-busy parity (2-length
// nsys 27.44→27.42 ms/step); it also aligns Laguna's decode with vLLM's on-device sampling
// (the born-on-runner pattern the decode-framework-routing audit flagged). `=0` opts back
// out (same-binary A/B). inline so a CPU build (VT_MARLIN_NVFP4 off) does not -Wunused.
inline bool LagunaOndevSampleEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LAGUNA_ONDEV_SAMPLE");
    return !(e != nullptr && e[0] == '0');  // default ON; =0 opts out
  }();
  return on;
}

// ── VT_LAGUNA_MOE_ONECAST: cast the MoE input activation `hn` [1,H] to bf16 ONCE per MoE
// layer and reuse it, instead of re-casting the SAME f32 `hn` inside every consumer
// (default ON). Under the shared-fp4 decode a MoE layer casts hn→bf16 THREE times — the
// router GEMV (GemmBf16), LagunaMoeResidentMarlinInto, and LagunaSharedExpertMarlinInto
// each run their own vt::CastBf16 node over the identical hn. This lever casts hn into a
// persistent bf16 buffer once and passes that pointer to all three (2 fewer CastBf16 graph
// nodes/MoE-layer = ~78 fewer nodes/step at 39 MoE layers), recovering the ramp/drain
// device time the CUDA graph does NOT hide on the tiny M=1 cast kernels — same mechanism
// as the glue/preamble/addnorm folds. BYTE-EXACT: vt::CastBf16 is a deterministic
// truncation, so the single cast's bf16 bytes are bit-identical to each consumer's own
// cast of the same hn — a same-binary A/B: =0 restores the per-consumer casts. inline so a
// CPU build (VT_MARLIN_NVFP4 off, graph excluded) does not -Wunused.
inline bool LagunaMoeOnecastEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LAGUNA_MOE_ONECAST");
    return !(e != nullptr && e[0] == '0');  // default ON; =0 opts out
  }();
  return on;
}

// ── VT_LAGUNA_TAIL_FUSED: fold the routed-MoE output CastF32 into the trailing
// fused_add2_rmsnorm (default ON). Under the glue+addnorm-fused decode the routed expert
// path ends with vt::MoeCombine writing a bf16 [1,H] result, THEN a standalone vt::CastF32
// widening it to the f32 `doutb` the fused_add2_rmsnorm reads as x1 — one M=1 CastF32 graph
// node PER MoE layer (~39/step). This lever has MoeCombine write its bf16 result straight
// into a persistent bf16 buffer and feeds that to a bf16-x1 fused_add2_rmsnorm sibling that
// widens in-kernel, deleting the CastF32 node (recovers the ramp/drain the CUDA graph does
// NOT hide on the tiny cast — same mechanism as the onecast/preamble/addnorm folds).
// BYTE-EXACT: MoeCombine writes the IDENTICAL bf16 bytes either way, and the in-kernel
// __bfloat162float reproduces exactly what vt::CastF32 wrote (bf16 bits << 16) — a same-
// binary A/B: =0 restores MoeCombine(bf16)+CastF32+f32 fused_add2. Only taken in the
// glue+addnorm-fused regime (the f32 split path is unchanged otherwise). inline so a CPU
// build (VT_MARLIN_NVFP4 off, graph excluded) does not -Wunused.
inline bool LagunaTailFusedEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LAGUNA_TAIL_FUSED");
    return !(e != nullptr && e[0] == '0');  // default ON; =0 opts out
  }();
  return on;
}

// ── VT_LAGUNA_SHARED_AUX: run the per-MoE-layer SHARED expert on a SECOND CUDA
// stream, forked from the post-attn-norm hidden `hn` BEFORE the router GEMV, so
// the (fp4-Marlin) shared expert overlaps the router GEMV + sigmoid_topk + routed
// grouped-GEMM issued on the main stream. This mirrors vLLM's
// MULTI_STREAM_OVERLAPPED (fused_moe/runner/shared_experts.py:125-129 fork from
// the post-attn hidden, moe_runner.py:560-596,809 join before MoeCombine) — the
// SAME machinery the 35B ships ON by default (ENG-MOE-SHARED-AUX,
// qwen3_5.cpp:4408-4467). The routed grouped GEMM is low-occupancy on the GB10
// (ncu: ~14% mem-SoL, 23% occ, 1 wave), so its spare SMs run the shared MLP
// concurrently. DEFAULT ON: the in-situ GB10 A/B won it (byte-exact + real 2.34 ms/step
// concurrency, net GPU-active +2.9%, 38.15->39.27 tok/s, ~92% of vLLM — parity-enablers-
// ship-as-defaults). BYTE-EXACT: the two streams read the same `hn` and write
// disjoint buffers (routed->doutb on main, shared->so on aux) joined before the
// combine, so the result is bit-identical to the serial order; `=0` is the
// same-binary rollback. Only meaningful in the fp4-shared arm (VT_LAGUNA_SHARED_FP4,
// default ON), where the router GEMV is already SPLIT from the shared expert — the
// EARLY fork the prior fused-GEMV attempt could not achieve. inline so a CPU build
// (VT_MARLIN_NVFP4 off, graph excluded) does not -Wunused.
inline bool LagunaSharedAuxEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LAGUNA_SHARED_AUX");
    return !(e != nullptr && e[0] == '0');  // default ON; =0 opts out
  }();
  return on;
}

bool LagunaCanRunResidentDecode(const LagunaParams& p, vt::Queue& q, const LagunaWeights& w,
                                int64_t T) {
#ifndef VT_MARLIN_NVFP4
  (void)p; (void)q; (void)w; (void)T;
  return false;  // the resident FFN routes the MoE through the Marlin W4A16 path
#else
  if (!LagunaResidentDecodeEnabled() || T != 1) return false;
  if (q.device.type == vt::DeviceType::kCPU) return false;
  if (!w.has_nvfp4_weights) return false;
  if (p.tie_word_embeddings || w.lm_head.Empty()) return false;
  if (!laguna::LagunaDeviceKernelsAvailable()) return false;
  return true;
#endif
}

std::vector<float> LagunaForwardResidentDecode(const LagunaWeights& weights, vt::Queue& q,
                                               LagunaKvCache& cache,
                                               const std::vector<int32_t>& token_ids,
                                               const std::vector<int32_t>& positions,
                                               const std::vector<int32_t>& logits_indices) {
  using vt::DType;
  const LagunaParams& p = weights.params;
  const int64_t H = p.hidden_size, Vsz = p.vocab_size, Dh = p.head_dim;
  const int64_t Hkv = p.num_key_value_heads, kvdim = Hkv * Dh;
  const int64_t nlayers = p.num_hidden_layers;
  const float eps = p.rms_norm_eps;
  const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
  const vt::Device dev = q.device;
  const int64_t pos = positions[0];
  if (cache.k.empty()) cache.Reset(nlayers, Dh, Hkv);

  // Brick A1: migrate the (host-built) prefill KV into fixed-capacity device buffers
  // ONCE. Thereafter each token's K/V is appended ON-STREAM at dev_rows[l] (no host
  // std::vector insert, so no per-layer DrainQueue). Full-deck (no eviction) — the
  // decode_attn window mask handles sliding layers, and dev_first_pos stays frozen.
  // LEVER A: allocate the KV at bf16 (half the DRAM) or f32 (byte-exact fallback), never both.
  const bool kv_bf16 = LagunaKvBf16Enabled();
  if (!cache.resident_ready) {
    cache.max_cap = pos + LagunaKvHeadroom();  // decode headroom (VT_CHECK guards; one-time zero-init)
    cache.dev_first_pos.assign(static_cast<size_t>(nlayers), 0);
    cache.dev_rows.assign(static_cast<size_t>(nlayers), 0);
    if (kv_bf16) {
      cache.k_dev16.assign(static_cast<size_t>(nlayers), {});
      cache.v_dev16.assign(static_cast<size_t>(nlayers), {});
    } else {
      cache.k_dev.assign(static_cast<size_t>(nlayers), {});
      cache.v_dev.assign(static_cast<size_t>(nlayers), {});
    }
    for (int64_t l = 0; l < nlayers; ++l) {
      const size_t cap = static_cast<size_t>(cache.max_cap * kvdim);
      const std::vector<float>& kc = cache.k[static_cast<size_t>(l)];
      const std::vector<float>& vc = cache.v[static_cast<size_t>(l)];
      if (kv_bf16) {  // one-time prefill migration with RNE cast (matches the on-device append)
        cache.k_dev16[static_cast<size_t>(l)].assign(cap, 0);
        cache.v_dev16[static_cast<size_t>(l)].assign(cap, 0);
        std::vector<uint16_t>& kd = cache.k_dev16[static_cast<size_t>(l)];
        std::vector<uint16_t>& vd = cache.v_dev16[static_cast<size_t>(l)];
        for (size_t i = 0; i < kc.size(); ++i) kd[i] = LagunaF32ToBf16Rne(kc[i]);
        for (size_t i = 0; i < vc.size(); ++i) vd[i] = LagunaF32ToBf16Rne(vc[i]);
      } else {
        cache.k_dev[static_cast<size_t>(l)].assign(cap, 0.0F);
        cache.v_dev[static_cast<size_t>(l)].assign(cap, 0.0F);
        std::copy(kc.begin(), kc.end(), cache.k_dev[static_cast<size_t>(l)].begin());
        std::copy(vc.begin(), vc.end(), cache.v_dev[static_cast<size_t>(l)].begin());
      }
      cache.dev_first_pos[static_cast<size_t>(l)] = cache.first_pos[static_cast<size_t>(l)];
      cache.dev_rows[static_cast<size_t>(l)] = static_cast<int64_t>(kc.size()) / kvdim;
    }
    cache.resident_ready = true;
  }

  const laguna::LagunaDeviceKernels* LAG = laguna::LagunaDevice();

  int64_t Hq_max = p.num_attention_heads;
  for (int64_t l = 0; l < nlayers; ++l) Hq_max = std::max(Hq_max, p.QHeadsForLayer(l));
  const int64_t qdim_max = Hq_max * Dh;
  const int64_t E = p.num_experts, topk = p.num_experts_per_tok;
  const int64_t moe_I = p.moe_intermediate_size, dense_I = p.intermediate_size;
  const int64_t maxI = std::max(moe_I, dense_I);
  const int64_t maxK = std::max({H, qdim_max, moe_I, dense_I});

  // rope_from_cache reads only row `pos`; build that single row (pos0=pos, rows=1)
  // instead of the full [pos+1, rd] table — the full rebuild was O(pos)/token = O(n^2)
  // cumulative. Byte-identical to row `pos` of the full table; read below at index 0.
  const std::vector<float> yarn_cache = BuildLagunaFullYarnCosSin(p, /*rows=*/1, /*pos0=*/pos);
  const std::vector<float> slide_cache = BuildLagunaSlidingCosSin(p, /*rows=*/1, /*pos0=*/pos);

  std::vector<float> hidden = LagunaEmbed(weights.embed, token_ids, H, Vsz);
  std::vector<float> hn(static_cast<size_t>(H)), qv(static_cast<size_t>(qdim_max)),
      gl(static_cast<size_t>(Hq_max)), attn(static_cast<size_t>(qdim_max)),
      o(static_cast<size_t>(H));
  std::vector<uint16_t> abf(static_cast<size_t>(maxK));
  std::vector<float> knew(static_cast<size_t>(kvdim)), vnew(static_cast<size_t>(kvdim));
  // Fused-projection decode scratch (steady-decode lever): ONE wider GEMV writes
  // the stacked qkvg / router+shared outputs; downstream ops read pointer-offset
  // slices into these (no copy). qkvg = [q | k | v | g], rsg = [router | sg | su].
  std::vector<float> qkvg_buf(static_cast<size_t>(qdim_max + 2 * kvdim + Hq_max));
  std::vector<float> rsg_buf(static_cast<size_t>(E + 2 * moe_I));
  // device-FFN scratch (unified) + persistent f32 norm/bias keep-alive (no per-FFN drain)
  std::vector<float> gating(static_cast<size_t>(E)), dg(static_cast<size_t>(maxI)),
      du(static_cast<size_t>(maxI)), dact(static_cast<size_t>(maxI)), fdn(static_cast<size_t>(H)),
      so(static_cast<size_t>(H)), doutb(static_cast<size_t>(H));
  std::vector<int32_t> eids32(static_cast<size_t>(topk));
  std::vector<float> topw(static_cast<size_t>(topk));
  std::vector<std::vector<float>> keep;
  auto Keep = [&](std::vector<float> v) -> const float* {
    keep.push_back(std::move(v));
    return keep.back().data();
  };
  auto DevT = [&](float* p2, int64_t I) {
    return vt::Tensor::Contiguous(p2, DType::kF32, dev, {1, I});
  };
  auto SiluMul = [&](float* out, float* g, float* u, int64_t I) {
    vt::Tensor ot = DevT(out, I);
    vt::MoeSiluMul(q, ot, DevT(g, I), DevT(u, I));
  };
  auto AddInto = [&](float* a, float* b) {  // a += b  (both [1,H])
    vt::Tensor at = DevT(a, H);
    vt::Add(q, at, at, DevT(b, H));
  };

  // no-sync bf16 GEMM: out[1,N] f32 = cast(x[1,K])·w[N,K]^T (mirror LqGemm device path)
  auto GemmBf16Into = [&](float* out, const OwnedTensor& w, const float* x, int64_t N, int64_t K) {
    vt::Tensor xf = vt::Tensor::Contiguous(const_cast<float*>(x), DType::kF32, dev, {1, K});
    vt::Tensor ab = vt::Tensor::Contiguous(abf.data(), DType::kBF16, dev, {1, K});
    vt::CastBf16(q, ab, xf);
    vt::Tensor ot = vt::Tensor::Contiguous(out, DType::kF32, dev, {1, N});
    vt::Tensor wt = LagunaResidentBf16W(q, w, dev);  // VT_LAGUNA_RESIDENT_BF16W: device copy vs retag
    vt::MatmulBT(q, ot, ab, wt);
  };

  for (int64_t l = 0; l < nlayers; ++l) {
    const LagunaLayerWeights& lw = weights.layers[static_cast<size_t>(l)];
    const int64_t Hq = p.QHeadsForLayer(l), qdim = Hq * Dh;
    const int64_t group = p.GqaGroupForLayer(l);
    const bool global = p.IsGlobalLayer(l);
    const int64_t rd = p.RotaryDimForLayer(l);
    const int64_t window = p.WindowForLayer(l);
    const float* rcache = (global ? yarn_cache : slide_cache).data();

    // Norm weights via ReadF32 (bf16->f32 exact). Keep()'d so they outlive the now
    // fully-async layer loop (no per-layer drain) — freed at the step-boundary drain.
    const float* w_in = Keep(ReadF32(lw.input_norm));
    LAG->rms_norm_seq(q, hn.data(), hidden.data(), w_in, 1, H, eps, true);
    // LEVER 1 — fused q|k|v|g projection (ONE wider GEMV instead of 3+1 narrow
    // M=1 GEMVs). Slice the stacked output into qv/knew/vnew/gl pointer views (no
    // copy); the offsets MUST match the load-time stack order (q,k,v,g). Fallback
    // to the split GEMVs on the GGUF path (qkvg_proj Empty()).
    float* qvp;
    float* knp;
    float* vnp;
    float* glp;
    const bool fused_qkvg = !lw.attn.qkvg_proj.Empty();
    if (fused_qkvg) {
      const int64_t n_qkvg = qdim + 2 * kvdim + Hq;
      GemmBf16Into(qkvg_buf.data(), lw.attn.qkvg_proj, hn.data(), n_qkvg, H);
      qvp = qkvg_buf.data();
      knp = qkvg_buf.data() + qdim;
      vnp = qkvg_buf.data() + qdim + kvdim;
      glp = qkvg_buf.data() + qdim + 2 * kvdim;
    } else {
      GemmBf16Into(qv.data(), lw.attn.q_proj, hn.data(), qdim, H);
      GemmBf16Into(knew.data(), lw.attn.k_proj, hn.data(), kvdim, H);
      GemmBf16Into(vnew.data(), lw.attn.v_proj, hn.data(), kvdim, H);
      qvp = qv.data();
      knp = knew.data();
      vnp = vnew.data();
      glp = gl.data();
    }
    if (p.has_qk_norm && !lw.attn.q_norm.Empty()) {
      const float* w_qn = Keep(ReadF32(lw.attn.q_norm));
      const float* w_kn = Keep(ReadF32(lw.attn.k_norm));
      LAG->rms_norm_seq(q, qvp, qvp, w_qn, Hq, Dh, eps, true);
      LAG->rms_norm_seq(q, knp, knp, w_kn, Hkv, Dh, eps, true);
    }
    LAG->rope_from_cache(q, qvp, rcache, Hq, Dh, rd, /*pos=*/0);    // single-row cache
    LAG->rope_from_cache(q, knp, rcache, Hkv, Dh, rd, /*pos=*/0);  // (built for `pos`)
    // Brick A1: append K/V to the device cache ON-STREAM (ordered after the RoPE that
    // wrote knew/vnew, before decode_attn reads — same stream), NO DrainQueue, NO host
    // insert/eviction. Full-deck: dev_first_pos frozen, decode_attn's window mask evicts.
    const int64_t dr = cache.dev_rows[static_cast<size_t>(l)];
    VT_CHECK(dr < cache.max_cap, "laguna resident decode: KV capacity exceeded (raise max_cap)");
    // LEVER A: the cache is bf16 (k_dev16) or f32 (k_dev); pass its base to the kernels via
    // float* (address pun for bf16 — the launchers read the env and the kernels reinterpret).
    float* kdev = kv_bf16 ? reinterpret_cast<float*>(cache.k_dev16[static_cast<size_t>(l)].data())
                          : cache.k_dev[static_cast<size_t>(l)].data();
    float* vdev = kv_bf16 ? reinterpret_cast<float*>(cache.v_dev16[static_cast<size_t>(l)].data())
                          : cache.v_dev[static_cast<size_t>(l)].data();
    if (kv_bf16) {  // append CASTS f32→bf16 on device at host row offset dr (no shared device int)
      LAG->append_kv_row_cast(q, kdev, vdev, knp, vnp, kvdim, dr);
    } else {  // f32: raw f32→f32 Copy at the byte offset (byte-exact, unchanged path)
      const size_t rowbytes = static_cast<size_t>(kvdim) * sizeof(float);
      vt::GetBackend(dev).Copy(q, kdev + dr * kvdim, knp, rowbytes);
      vt::GetBackend(dev).Copy(q, vdev + dr * kvdim, vnp, rowbytes);
    }
    const int64_t rows = dr + 1;
    const int64_t fp = cache.dev_first_pos[static_cast<size_t>(l)];
    // LEVER 1: the fused GEMV already produced g (glp -> qkvg_buf); only the split
    // fallback still needs a dedicated g_proj GEMV. L1 folds the softplus out-gate into
    // decode_attn_gqa's store, which reads g — so produce it BEFORE attention (reorder of
    // two independent same-stream ops => byte-exact).
    if (!fused_qkvg) GemmBf16Into(gl.data(), lw.attn.g_proj, hn.data(), Hq, H);
    const bool glue_fused = LagunaGlueFusedEnabled();
    LAG->decode_attn_gqa(q, attn.data(), qvp, kdev, vdev, Hq, Hkv, Dh, group, rows, pos, fp,
                         window, scale, glue_fused ? glp : nullptr);
    cache.dev_rows[static_cast<size_t>(l)] = rows;  // advance this layer's cached-row count
    if (!glue_fused) LAG->softplus_head_gate(q, attn.data(), glp, Hq, Dh);  // L1 off => separate pass
    GemmBf16Into(o.data(), lw.attn.o_proj, attn.data(), H, qdim);
    {
      vt::Tensor ht = vt::Tensor::Contiguous(hidden.data(), DType::kF32, dev, {1, H});
      vt::Tensor ot = vt::Tensor::Contiguous(o.data(), DType::kF32, dev, {1, H});
      vt::Add(q, ht, ht, ot);
    }
    // post-attn RMSNorm (device); FFN fully device-resident — NO per-layer FFN drain.
    LAG->rms_norm_seq(q, hn.data(), hidden.data(), Keep(ReadF32(lw.post_attn_norm)), 1, H, eps,
                      true);
    if (lw.is_dense) {
      GemmBf16Into(dg.data(), lw.mlp.gate_proj, hn.data(), dense_I, H);
      GemmBf16Into(du.data(), lw.mlp.up_proj, hn.data(), dense_I, H);
      SiluMul(dact.data(), dg.data(), du.data(), dense_I);
      GemmBf16Into(fdn.data(), lw.mlp.down_proj, dact.data(), H, dense_I);
      AddInto(hidden.data(), fdn.data());
    } else {
      // Reads the fused router|shared_gate|shared_up weight that laguna_weights.cpp only
      // STACKS under the same build gate, so this branch must match it. Retires together
      // with that one when the concat moves to a runtime device question.
      // DSR-ALLOW(S1): pairs with the laguna_weights.cpp fused-projection gate.
#ifdef VT_MARLIN_NVFP4
      // LEVER 2 — fused router|shared_gate|shared_up projection (ONE GEMV): all
      // three read the SAME post-attn `hn`. Slice: [0,E)=router logits ->
      // sigmoid_topk; [E,E+moe_I)=shared_gate; [E+moe_I,E+2*moe_I)=shared_up
      // (pre-SiLU). Fallback to split GEMVs on the GGUF path (router_shared_gu Empty()).
      // VT_LAGUNA_SHARED_FP4: the shared expert stays fp4-resident (Marlin W4A16),
      // so the wide fused GEMV drops to a router-ONLY GEMV (the split `moe.router`)
      // and the shared gate/up/down run through the fp4 Marlin path — no bf16 shared
      // weights are read (the 4x-DRAM win). `shared_gate_fp4` non-empty => the fp4
      // tower was loaded (LagunaLoadSharedExpertFp4) for this MoE layer.
      const bool shared_fp4 = LagunaSharedFp4Enabled() && !lw.moe.shared_gate_fp4.Empty();
      float* gatingp;
      float* sgp;
      float* sup;
      const bool fused_rsg = !shared_fp4 && !lw.moe.router_shared_gu.Empty();
      if (fused_rsg) {
        GemmBf16Into(rsg_buf.data(), lw.moe.router_shared_gu, hn.data(), E + 2 * moe_I, H);
        gatingp = rsg_buf.data();
        sgp = rsg_buf.data() + E;
        sup = rsg_buf.data() + E + moe_I;
      } else {
        GemmBf16Into(gating.data(), lw.moe.router, hn.data(), E, H);  // router (bf16 weight)
        gatingp = gating.data();
        sgp = dg.data();
        sup = du.data();
      }
      const float* bias = lw.moe.e_score_correction_bias.Empty()
                              ? nullptr
                              : Keep(ReadF32(lw.moe.e_score_correction_bias));
      LAG->sigmoid_topk(q, eids32.data(), topw.data(), gatingp, bias, bias != nullptr, E,
                        topk, p.norm_topk_prob, p.moe_routed_scaling_factor);
      LagunaMoeResidentMarlinInto(q, lw.moe, hn.data(), moe_I, H, E, eids32.data(), topw.data(),
                                  topk, doutb.data());
      if (shared_fp4) {
        LagunaSharedExpertMarlinInto(q, lw.moe, hn.data(), H, so.data());  // fp4 shared expert
      } else {
        if (!fused_rsg) {  // split fallback still needs dedicated shared gate/up GEMVs
          GemmBf16Into(dg.data(), lw.moe.shared_gate, hn.data(), moe_I, H);  // shared expert
          GemmBf16Into(du.data(), lw.moe.shared_up, hn.data(), moe_I, H);
        }
        SiluMul(dact.data(), sgp, sup, moe_I);
        GemmBf16Into(so.data(), lw.moe.shared_down, dact.data(), H, moe_I);
      }
      AddInto(hidden.data(), doutb.data());  // hidden += routed
      AddInto(hidden.data(), so.data());     // hidden += shared
#else
      VT_CHECK(false, "laguna resident decode: MoE requires the VT_MARLIN_NVFP4 build");
#endif
    }
  }
  DrainQueue(q);  // the ONE step-boundary drain: hidden coherent for the final logits
  return LagunaFinalLogits(q, weights, hidden, 1, logits_indices);
}

// The decode CUDA-GRAPH capture class is irreducibly device-coupled at build time —
// the CHAIN it captures is `VT_MARLIN_NVFP4` CUDA code, so the class compiles only in
// that build. The REPAIR this note owed has LANDED: the portable `vt` capture/replay
// seam is `vt::GraphCaptureScope` + `vt::BreakableGraph`
// (`include/vt/breakable_graph.h`, row ENG-CUDAGRAPH-BREAK, #1163), and this class and
// `deepseek_v4.cpp`'s `V4Graph` — the sibling this note named — moved onto it together
// in W5 (#1335). What is left device-coupled is the captured CHAIN, not the capture.
// DSR-ALLOW(S1): decode CUDA-graph capture class; the CHAIN is CUDA/Marlin, not the seam.
#ifdef VT_MARLIN_NVFP4
namespace {
// ─── Brick A2: the DECODE CUDA GRAPH (mirror of deepseek_v4.cpp V4Graph) ──────────
// Capture the now-100%-device resident T=1 step (the A1 chain) into ONE graph and
// replay it — one cudaGraphLaunch/step instead of ~700 host launches (the A1 eager
// resident measured ~45% GPU-idle from host-launch gaps). The whole per-layer chain
// runs over PERSISTENT member buffers (never resized → stable addresses the captured
// graph bakes); the ONLY per-step-varying inputs (embed→hidden, position, KV length,
// the single-row RoPE cos/sin) live in persistent buffers whose CONTENTS the driver
// refreshes OUTSIDE capture before each replay. The growing KV is handled cudagraph-
// safely: fixed-capacity per-layer cache_k/cache_v seeded from the prefill KV; the
// kv-norm+RoPE writes the new token's K/V into the k/v sub-ranges of the FIXED
// per-layer qkvg[l] fused scratch (NOT the varying cache slot); decode_attn_gqa_g
// attends cache[0..len) + those k/v ranges with len/pos read from DEVICE buffers;
// and BETWEEN replays the driver async-copies the k/v ranges→cache_k[l]/cache_v[l]
// at the host-known slot (NOT inside capture) +
// advances the row count. cold→warm→captured: the cold step runs eager (warms the
// DevicePool + the per-stream GEMM scratch, grow-only → capture does zero fresh
// cudaMalloc, which stream capture forbids). [[cudagraph-capture-bakes-stack-
// addresses]]: EVERY captured input is a member buffer, never a stack temporary or a
// per-token ReadF32 (norms are pre-converted ONCE in the ctor). Gate BY TOKENS.
//
// Single-len_buf scope: correct while every layer keeps the SAME cached-row count and
// first_pos==0 — holds for a prompt P < the 512 sliding window (the P=6 benchmark).
// The ctor VT_CHECKs it; the per-layer ring-buffer graph (diverging sliding evictions)
// is a named follow-up (Brick A2b), NOT blocking here.
//
// NOVEL capture risk vs V4Graph (flagged for the GPU token-gate): V4Graph uses ZERO
// pooled scratch in-graph; Laguna's MoE (LagunaMoeResidentMarlinInto) allocates ~10
// DevicePool DBufs per call. Capture is safe iff (a) the cold warm-run already touched
// every DBuf size-class (so capture pops warm blocks, no cudaMalloc), and (b) nothing
// else Gets those size-classes between/after replays (single in-flight sequence). The
// pool is deterministic LIFO, so the baked pooled pointers stay valid across replays.
struct LagunaGraph {
  const LagunaWeights* w;
  const LagunaParams* p;
  vt::Queue* qp;
  vt::Device dev;
  int64_t H, Vsz, Dh, Hkv, kvdim, nlayers, E, topk, moe_I, dense_I;
  float eps, scale;
  int64_t max_cap = 0;
  int64_t kv_rows = 0;  // uniform cached-row count for ALL layers (grows 1/token)
  struct LayerC {
    int64_t Hq, qdim, group, rd, window;
    bool global, is_dense;
  };
  std::vector<LayerC> lc;
  // persistent per-layer device KV (seeded from prefill) + the fused new-row
  // scratch. LEVER 1: qkvg[l] is the per-layer [qdim_l + 2*kvdim + Hq_l] output of
  // the fused q|k|v|g GEMV; its k/v sub-ranges REPLACE the old separate knew[l]/
  // vnew[l] (Step() copies those slices into the growing cache between replays), so
  // each layer needs its OWN persistent buffer (a shared one would be overwritten
  // by the next layer before Step() drains it).
  std::vector<std::vector<float>> cache_k, cache_v, qkvg;
  // LEVER A (VT_LAGUNA_KV_BF16): bf16 alternative to cache_k/cache_v (half the KV DRAM,
  // matching vLLM). Only one of the pair is allocated per run (env-gated); the f32 path is
  // byte-identical when off. bf16 bits (uint16_t) passed to the kernels via the float* param.
  std::vector<std::vector<uint16_t>> cache_k16, cache_v16;
  bool kv_bf16 = false;  // resolved in the ctor from LagunaKvBf16Enabled()
  // persistent per-layer f32 norms/bias — pre-converted ONCE (kills per-token ReadF32
  // + the stack-baked-pointer hazard).
  std::vector<std::vector<float>> input_norm_f, q_norm_f, k_norm_f, post_norm_f, moe_bias_f;
  std::vector<float> final_norm_f;
  // LEVER A: device-resident position-indexed RoPE cos/sin tables — built ONCE (rows
  // [0,max_cap)), indexed by *pos_buf on-device via rope_from_cache_g. Replaces the old
  // single-row buffers that Step host-rebuilt every token (kills the per-step host cos/sin
  // recompute + copy). Row `pos` is byte-identical to the old single-row build for `pos`.
  std::vector<float> yarn_full, slide_full;
  // per-step device-read scalars (FIXED pointers, contents refreshed each token).
  std::vector<int> pos_buf, len_buf;
  // persistent working scratch (all members → stable captured addresses). LEVER 2:
  // rsg is the shared [E + 2*moe_I] fused router|shared_gate|shared_up output (a
  // single reused buffer suffices — unlike qkvg's K/V, none of its slices persist
  // across layers/replays). The old separate qv/gl/gating are now qkvg/rsg slices.
  std::vector<float> hidden, hn, attn, o, dg, du, dact, fdn, so, doutb, topw, logits, rsg;
  std::vector<uint16_t> abf;
  // VT_LAGUNA_MOE_ONECAST: persistent bf16 copy of the MoE input `hn` [1,H], cast ONCE
  // per MoE layer and reused by the router GEMM + routed-expert Marlin + shared-expert
  // Marlin (fixed address ⇒ capture-safe; unlike `abf` which each GemmBf16 overwrites).
  std::vector<uint16_t> hn_bf16;
  std::vector<uint16_t> doutb_bf16;  // VT_LAGUNA_TAIL_FUSED: bf16 routed-MoE out (no CastF32)
  std::vector<int32_t> eids32;
  // VT_LAGUNA_ONDEV_SAMPLE: single-element device buffer holding the greedy token id.
  // Step seeds it (host) with THIS step's input token; RunChain's embed_gather reads it
  // at replay start and its trailing GreedyArgmax overwrites it with the next token — read
  // back after the drain. i64 to match vt::GreedyArgmax's output dtype. `ondev` caches the
  // env flag; `last_sampled` is the token the caller consumes (-1 when off).
  const bool ondev = LagunaOndevSampleEnabled();
  // VT_LAGUNA_SHARED_FP4: run the shared expert through the fp4 Marlin W4A16 path
  // (set only when the fp4 shared tower was loaded — LagunaLoadSharedExpertFp4). When
  // true the per-MoE-layer GEMV is router-ONLY (`moe.router`); the shared gate/up/down
  // ride LagunaSharedExpertMarlinInto (no bf16 shared weights read). Set in the ctor
  // init-list (depends on the ctor's weights arg).
  const bool shared_fp4;
  // VT_LAGUNA_SHARED_AUX: the aux stream + fork/join events for the shared-expert
  // overlap (ENG-MOE-SHARED-AUX mirror). Created in the ctor — BEFORE any capture —
  // so the stream/events exist when the gstate-1 region is captured; the gstate-0
  // eager warm-run issues the aux shared expert once (building its fp4 Marlin
  // residents + warming AuxPool) so the captured replay does zero cudaMalloc. Only
  // armed when VT_LAGUNA_SHARED_AUX=1 AND the fp4-shared arm is active AND the
  // backend has a second stream (SupportsAuxStream). Serial (no fork) otherwise.
  bool shared_aux = false;
  vt::Queue aux_q{};      // the shared-expert stream (plain default-priority, like vLLM)
  vt::Event aux_fork{};   // recorded on main; aux waits it (hn ready — the fork point)
  vt::Event aux_done{};   // recorded on aux after the shared MLP; main waits it (join)
  std::vector<int64_t> argmax_id;
  int32_t last_sampled = -1;
  // ENG-CUDAGRAPH-BREAK W5 (#1335): the instantiated graph, its handle ownership
  // and its release live in the SHARED SEAM instead of in a raw `void*` this
  // class destroyed by hand. Every segment is released through
  // `Backend::DestroyGraph`, which is the routing that lets ENG-CUDAGRAPH-DEDUP
  // (#1162) interpose at the backend later without editing this file. `gstate`
  // STAYS: it is this driver's cold/warm/captured ladder, not a duplicate of
  // `captured()` — the seam has no notion of the eager warm-run that grows the
  // pool so a capture may allocate nothing.
  vt::BreakableGraph graph;
  int gstate = 0;  // 0 cold (eager warm-run), 1 warm (capture+replay), 2 captured (replay)

  LagunaGraph(const LagunaWeights& w_, vt::Queue& q, LagunaKvCache& cache)
      : w(&w_), p(&w_.params), qp(&q), dev(q.device),
        shared_fp4(LagunaSharedFp4Enabled() && LagunaHasFp4SharedExpert(w_)) {
    H = p->hidden_size; Vsz = p->vocab_size; Dh = p->head_dim;
    Hkv = p->num_key_value_heads; kvdim = Hkv * Dh;
    nlayers = p->num_hidden_layers;
    E = p->num_experts; topk = p->num_experts_per_tok;
    moe_I = p->moe_intermediate_size; dense_I = p->intermediate_size;
    eps = p->rms_norm_eps;
    scale = 1.0F / std::sqrt(static_cast<float>(Dh));
    int64_t Hq_max = p->num_attention_heads;
    for (int64_t l = 0; l < nlayers; ++l) Hq_max = std::max(Hq_max, p->QHeadsForLayer(l));
    const int64_t qdim_max = Hq_max * Dh;
    const int64_t maxI = std::max(moe_I, dense_I);
    const int64_t maxK = std::max({H, qdim_max, moe_I, dense_I});
    lc.resize(static_cast<size_t>(nlayers));
    for (int64_t l = 0; l < nlayers; ++l) {
      lc[static_cast<size_t>(l)] = {p->QHeadsForLayer(l), p->QHeadsForLayer(l) * Dh,
                                    p->GqaGroupForLayer(l), p->RotaryDimForLayer(l),
                                    p->WindowForLayer(l), p->IsGlobalLayer(l),
                                    w->layers[static_cast<size_t>(l)].is_dense};
    }
    // single-len_buf validity + the seed row count (uniform across layers for P<512).
    VT_CHECK(!cache.k.empty() && static_cast<int64_t>(cache.k.size()) == nlayers,
             "laguna decode graph: prefill KV missing (build after prefill only)");
    const int64_t rows0 = static_cast<int64_t>(cache.k[0].size()) / kvdim;
    for (int64_t l = 0; l < nlayers; ++l) {
      VT_CHECK(cache.first_pos[static_cast<size_t>(l)] == 0 &&
                   static_cast<int64_t>(cache.k[static_cast<size_t>(l)].size()) / kvdim == rows0,
               "laguna decode graph: single-len_buf requires uniform per-layer KV (prompt "
               "< sliding window 512); the per-layer ring-buffer graph is Brick A2b");
    }
    kv_rows = rows0;
    max_cap = rows0 + LagunaKvHeadroom();  // decode headroom (VT_LAGUNA_KV_HEADROOM for ~2k slope)
    // persistent per-layer KV seeded from the host prefill KV; per-layer fused
    // q|k|v|g new-row scratch (LEVER 1). Each qkvg[l] is sized to the layer's own
    // N_total = qdim_l + 2*kvdim + Hq_l; its k/v sub-ranges are the new-row scratch
    // Step() drains between replays.
    // LEVER A: bf16 or f32 KV, never both (env-gated; f32 byte-identical when off).
    kv_bf16 = LagunaKvBf16Enabled();
    qkvg.assign(static_cast<size_t>(nlayers), {});
    if (kv_bf16) {
      cache_k16.assign(static_cast<size_t>(nlayers), {});
      cache_v16.assign(static_cast<size_t>(nlayers), {});
    } else {
      cache_k.assign(static_cast<size_t>(nlayers), {});
      cache_v.assign(static_cast<size_t>(nlayers), {});
    }
    for (int64_t l = 0; l < nlayers; ++l) {
      const size_t cap = static_cast<size_t>(max_cap * kvdim);
      const std::vector<float>& kc = cache.k[static_cast<size_t>(l)];
      const std::vector<float>& vc = cache.v[static_cast<size_t>(l)];
      if (kv_bf16) {  // one-time prefill migration with RNE cast (matches append_kv_row)
        cache_k16[static_cast<size_t>(l)].assign(cap, 0);
        cache_v16[static_cast<size_t>(l)].assign(cap, 0);
        std::vector<uint16_t>& kd = cache_k16[static_cast<size_t>(l)];
        std::vector<uint16_t>& vd = cache_v16[static_cast<size_t>(l)];
        for (size_t i = 0; i < kc.size(); ++i) kd[i] = LagunaF32ToBf16Rne(kc[i]);
        for (size_t i = 0; i < vc.size(); ++i) vd[i] = LagunaF32ToBf16Rne(vc[i]);
      } else {
        cache_k[static_cast<size_t>(l)].assign(cap, 0.0F);
        cache_v[static_cast<size_t>(l)].assign(cap, 0.0F);
        std::copy(kc.begin(), kc.end(), cache_k[static_cast<size_t>(l)].begin());
        std::copy(vc.begin(), vc.end(), cache_v[static_cast<size_t>(l)].begin());
      }
      const LayerC& c = lc[static_cast<size_t>(l)];
      // The graph always fuses (NVFP4-arm-only path); the loader must have built
      // the stacked projections. Fail LOUDLY here rather than mis-slice at replay.
      const LagunaLayerWeights& L = w->layers[static_cast<size_t>(l)];
      VT_CHECK(!L.attn.qkvg_proj.Empty(),
               "laguna decode graph: fused qkvg_proj missing (NVFP4 loader must build it)");
      // VT_LAGUNA_SHARED_FP4 uses the split router GEMV + fp4 shared expert instead of
      // the fused router|shared_gate|shared_up projection (which is freed at load).
      if (shared_fp4)
        VT_CHECK(c.is_dense || (!L.moe.router.Empty() && !L.moe.shared_gate_fp4.Empty()),
                 "laguna decode graph: VT_LAGUNA_SHARED_FP4 needs split router + fp4 shared expert");
      else
        VT_CHECK(c.is_dense || !L.moe.router_shared_gu.Empty(),
                 "laguna decode graph: fused router_shared_gu missing (NVFP4 loader must build it)");
      qkvg[static_cast<size_t>(l)].assign(static_cast<size_t>(c.qdim + 2 * kvdim + c.Hq), 0.0F);
    }
    // persistent f32 norms/bias (pre-converted ONCE — no per-token ReadF32 in-graph).
    input_norm_f.resize(static_cast<size_t>(nlayers));
    q_norm_f.resize(static_cast<size_t>(nlayers));
    k_norm_f.resize(static_cast<size_t>(nlayers));
    post_norm_f.resize(static_cast<size_t>(nlayers));
    moe_bias_f.resize(static_cast<size_t>(nlayers));
    for (int64_t l = 0; l < nlayers; ++l) {
      const LagunaLayerWeights& L = w->layers[static_cast<size_t>(l)];
      input_norm_f[static_cast<size_t>(l)] = ReadF32(L.input_norm);
      post_norm_f[static_cast<size_t>(l)] = ReadF32(L.post_attn_norm);
      if (p->has_qk_norm && !L.attn.q_norm.Empty()) {
        q_norm_f[static_cast<size_t>(l)] = ReadF32(L.attn.q_norm);
        k_norm_f[static_cast<size_t>(l)] = ReadF32(L.attn.k_norm);
      }
      if (!L.is_dense && !L.moe.e_score_correction_bias.Empty())
        moe_bias_f[static_cast<size_t>(l)] = ReadF32(L.moe.e_score_correction_bias);
    }
    final_norm_f = ReadF32(w->norm);
    // LEVER A: build the FULL position-indexed RoPE tables ONCE (rows [0,max_cap)). pos
    // increments in lockstep with kv_rows (both == cache.len at decode), and Step VT_CHECKs
    // kv_rows+1 <= max_cap, so every decoded pos < max_cap ⇒ the tables cover every index.
    yarn_full = BuildLagunaFullYarnCosSin(*p, /*rows=*/max_cap, /*pos0=*/0);
    slide_full = BuildLagunaSlidingCosSin(*p, /*rows=*/max_cap, /*pos0=*/0);
    pos_buf.assign(1, 0);
    len_buf.assign(1, 0);
    // persistent working scratch. (qv/gl/gating are now qkvg/rsg slices.)
    hidden.assign(static_cast<size_t>(H), 0.0F);
    hn.assign(static_cast<size_t>(H), 0.0F);
    attn.assign(static_cast<size_t>(qdim_max), 0.0F);
    o.assign(static_cast<size_t>(H), 0.0F);
    rsg.assign(static_cast<size_t>(E + 2 * moe_I), 0.0F);
    dg.assign(static_cast<size_t>(maxI), 0.0F);
    du.assign(static_cast<size_t>(maxI), 0.0F);
    dact.assign(static_cast<size_t>(maxI), 0.0F);
    fdn.assign(static_cast<size_t>(H), 0.0F);
    so.assign(static_cast<size_t>(H), 0.0F);
    doutb.assign(static_cast<size_t>(H), 0.0F);
    topw.assign(static_cast<size_t>(topk), 0.0F);
    logits.assign(static_cast<size_t>(Vsz), 0.0F);
    abf.assign(static_cast<size_t>(maxK), 0);
    hn_bf16.assign(static_cast<size_t>(H), 0);  // VT_LAGUNA_MOE_ONECAST reuse buffer
    doutb_bf16.assign(static_cast<size_t>(H), 0);  // VT_LAGUNA_TAIL_FUSED reuse buffer
    eids32.assign(static_cast<size_t>(topk), 0);
    argmax_id.assign(1, 0);  // VT_LAGUNA_ONDEV_SAMPLE token buffer (device-accessible)
    // VT_LAGUNA_SHARED_AUX: create the aux stream + fork/join events NOW (ctor runs
    // before any BeginCapture). Only in the fp4-shared arm — the bf16 arm fuses the
    // shared gate/up INTO the router GEMV (router_shared_gu), so there is no split
    // shared expert to hand to a second stream. SupportsAuxStream is false on non-CUDA
    // backends (the graph itself is CUDA-only, but keep the guard honest).
    if (LagunaSharedAuxEnabled() && shared_fp4) {
      vt::Backend& b = vt::GetBackend(dev);
      if (b.SupportsAuxStream()) {
        aux_q = b.CreateQueue();
        aux_fork = b.CreateEvent();
        aux_done = b.CreateEvent();
        shared_aux = true;
      }
    }
  }
  ~LagunaGraph() {
    if (qp == nullptr) return;
    vt::Backend& b = vt::GetBackend(qp->device);
    // No `DestroyGraph` here: `graph` releases its own segments. The aux queue
    // and its two events are still this class's, because the SEAM tracks a fork
    // and never owns one — see `vt::GraphNoteFork`.
    if (shared_aux) {
      b.DestroyEvent(aux_fork);
      b.DestroyEvent(aux_done);
      b.DestroyQueue(aux_q);
    }
  }
  LagunaGraph(const LagunaGraph&) = delete;
  LagunaGraph& operator=(const LagunaGraph&) = delete;

  // no-sync bf16 GEMM over the persistent abf scratch (mirror of the eager
  // GemmBf16Into lambda): out[1,N] f32 = cast(x[1,K])·w[N,K]^T. Same-stream ⇒ reusing
  // one abf buffer across GEMMs is byte-identical (each cast→matmul completes before
  // the next cast overwrites), and the FIXED abf pointer is capture-safe.
  void GemmBf16(float* out, const OwnedTensor& wt, const float* x, int64_t N, int64_t K) {
    vt::Queue& q = *qp;
    vt::Tensor xf = vt::Tensor::Contiguous(const_cast<float*>(x), vt::DType::kF32, dev, {1, K});
    vt::Tensor ab = vt::Tensor::Contiguous(abf.data(), vt::DType::kBF16, dev, {1, K});
    vt::CastBf16(q, ab, xf);
    vt::Tensor ot = vt::Tensor::Contiguous(out, vt::DType::kF32, dev, {1, N});
    vt::Tensor wv = LagunaResidentBf16W(q, wt, dev);  // VT_LAGUNA_RESIDENT_BF16W
    vt::MatmulBT(q, ot, ab, wv);
  }

  // VT_LAGUNA_MOE_ONECAST: cast hn[1,H] f32 → the persistent hn_bf16 buffer ONCE (the
  // single node that replaces the per-consumer casts). Same CastBf16 truncation ⇒ the
  // bytes are identical to each consumer's own cast of the same hn.
  void CastHnBf16(const float* x) {
    vt::Queue& q = *qp;
    vt::Tensor xf = vt::Tensor::Contiguous(const_cast<float*>(x), vt::DType::kF32, dev, {1, H});
    vt::Tensor hb = vt::Tensor::Contiguous(hn_bf16.data(), vt::DType::kBF16, dev, {1, H});
    vt::CastBf16(q, hb, xf);
  }
  // GEMM whose activation is the ALREADY-cast persistent hn_bf16 (no per-call CastBf16):
  // out[1,N] f32 = hn_bf16[1,K]·w[N,K]^T. Byte-identical to GemmBf16(out,wt,hn,N,K) when
  // hn_bf16 == cast(hn). K must be H (hn_bf16 holds the [1,H] MoE input).
  void GemmBf16Pre(float* out, const OwnedTensor& wt, int64_t N, int64_t K) {
    vt::Queue& q = *qp;
    vt::Tensor ab = vt::Tensor::Contiguous(hn_bf16.data(), vt::DType::kBF16, dev, {1, K});
    vt::Tensor ot = vt::Tensor::Contiguous(out, vt::DType::kF32, dev, {1, N});
    vt::Tensor wv = LagunaResidentBf16W(q, wt, dev);  // VT_LAGUNA_RESIDENT_BF16W
    vt::MatmulBT(q, ot, ab, wv);
  }

  // The per-layer resident chain over the PERSISTENT buffers (the CAPTURE region) —
  // same math as LagunaForwardResidentDecode's loop body, but: norms read the pre-
  // converted f32 members; RoPE reads the persistent single-row cos/sin (index 0);
  // decode_attn uses decode_attn_gqa_g with len_buf/pos_buf (DEVICE); the new K/V
  // are the k/v sub-ranges of the FIXED per-layer qkvg[l] fused buffer (NOT appended
  // in-graph); final norm + lm_head are INSIDE the region, writing persistent logits.
  void RunChain() {
    vt::Queue& q = *qp;
    const laguna::LagunaDeviceKernels* LAG = laguna::LagunaDevice();
    auto DevT = [&](float* pp, int64_t I) {
      return vt::Tensor::Contiguous(pp, vt::DType::kF32, dev, {1, I});
    };
    auto SiluMul = [&](float* out, float* g, float* u, int64_t I) {
      vt::Tensor ot = DevT(out, I);
      vt::MoeSiluMul(q, ot, DevT(g, I), DevT(u, I));
    };
    auto AddInto = [&](float* a, float* b) {
      vt::Tensor at = DevT(a, H);
      vt::Add(q, at, at, DevT(b, H));
    };
    const bool glue_fused = LagunaGlueFusedEnabled();
    const bool moe_addnorm_fused = LagunaMoeAddNormFusedEnabled();  // LEVER A
    const bool preamble_fused = LagunaPreambleFusedEnabled();       // fused q/k norm+rope
    const bool moe_onecast = LagunaMoeOnecastEnabled();             // one hn→bf16 cast/MoE-layer
    // VT_LAGUNA_TAIL_FUSED: bf16 routed-MoE out widened in fused_add2 (drops routed CastF32).
    // Only meaningful in the glue+addnorm-fused regime (its fused_add2 tail is what absorbs
    // the widen); otherwise the f32 doutb + CastF32 path stays.
    const bool tail_fused =
        LagunaTailFusedEnabled() && glue_fused && moe_addnorm_fused;
    // VT_LAGUNA_ONDEV_SAMPLE: gather THIS step's input embedding ON-DEVICE from argmax_id
    // (seeded by Step, or the previous replay's argmax) into the persistent hidden buffer —
    // the in-graph replacement for the host embed loop in Step. Byte-identical widen (the
    // kernel does bits<<16 for bf16, plain copy for f32). hidden is fully overwritten (all H
    // elements), matching the host gather that reset it each step.
    if (ondev)
      LAG->embed_gather(q, hidden.data(), reinterpret_cast<const void*>(w->embed.bytes.data()),
                        w->embed.dtype == vt::DType::kBF16, argmax_id.data(), H);
    // L4: fold a residual-Add + STANDARD RMSNorm pair into ONE vt::FusedChain
    // (kFusedAddRmsNormStd): res += x; out = rms_norm(res)*w. BYTE-EXACT to
    // AddInto(res,x) + rms_norm_seq(out,res,w): the f32 residual add is the same
    // (commutative + ResRound<f32> is identity), and the RMSNorm uses the SAME 256-thread
    // strided sum + shared-tree reduction + 1/sqrtf as RmsNormSeqKernel. NODE-COUNT WIN in
    // BOTH tiers: the default Tier-0 composite collapses the chain to ONE
    // vt::RmsNorm(residual) launch (RmsNormRowKernel does the add inline), and Tier-1
    // (VT_FUSED_TIER=1) to one interpreter kernel — either way 2 kernels -> 1. All
    // operands are persistent member buffers => capture-safe (same temporary-Tensor
    // pattern as the AddInto lambda already captured in this graph).
    auto FusedAddNorm = [&](float* out, float* x, const float* wgt, float* res) {
      vt::Tensor o2 = DevT(out, H), x2 = DevT(x, H), r2 = DevT(res, H);
      vt::Tensor w1 = vt::Tensor::Contiguous(const_cast<float*>(wgt), vt::DType::kF32, dev, {H});
      vt::FusedChain(q, o2, x2, w1, &r2, vt::kFusedAddRmsNormStd, eps);
    };
    // L4: when fused, `hn` always holds rms_norm(hidden, input_norm[l]) on entry to layer
    // l — layer 0 is seeded here (no preceding residual add to fold), and every later
    // layer inherits it from the previous layer's fused Pair-2 tail (last MLP add folded
    // with the next input norm). When NOT fused, the input norm runs at each loop top.
    if (glue_fused)
      LAG->rms_norm_seq(q, hn.data(), hidden.data(), input_norm_f[0].data(), 1, H, eps, true);
    for (int64_t l = 0; l < nlayers; ++l) {
      const LagunaLayerWeights& lw = w->layers[static_cast<size_t>(l)];
      const LayerC& c = lc[static_cast<size_t>(l)];
      // LEVER 1: the fused q|k|v|g output lives in the per-layer qkvg[l] buffer;
      // slice it (offsets MUST match the load-time stack order q,k,v,g). kn/vn are
      // the k/v sub-ranges Step() drains into the growing cache between replays.
      float* base = qkvg[static_cast<size_t>(l)].data();
      float* qvp = base;
      float* kn = base + c.qdim;
      float* vn = base + c.qdim + kvdim;
      float* glp = base + c.qdim + 2 * kvdim;
      // LEVER A: the position-indexed full RoPE table for this layer's regime; the graph
      // RoPE indexes row *pos_buf on-device (no per-step host cos/sin rebuild).
      const float* rcache = (c.global ? yarn_full : slide_full).data();
      if (!glue_fused)  // L4: fused path carries hn in from the previous layer's Pair-2 tail
        LAG->rms_norm_seq(q, hn.data(), hidden.data(),
                          input_norm_f[static_cast<size_t>(l)].data(), 1, H, eps, true);
      GemmBf16(base, lw.attn.qkvg_proj, hn.data(), c.qdim + 2 * kvdim + c.Hq, H);
      const bool do_qk_norm = p->has_qk_norm && !lw.attn.q_norm.Empty();
      if (preamble_fused && do_qk_norm) {
        // ONE node: fused q/k RMSNorm + dual-head RoPE over the qkvg q/k slices, byte-exact
        // to the four kernels below (same 256-thread reduction, same half-split rope). Both
        // regimes handled by c.rd (global 64 / sliding 128) reading the layer's rcache.
        LAG->fused_qk_norm_rope_g(q, qvp, kn, q_norm_f[static_cast<size_t>(l)].data(),
                                  k_norm_f[static_cast<size_t>(l)].data(), rcache, c.Hq, Hkv, Dh,
                                  c.rd, eps, pos_buf.data());
      } else {
        if (do_qk_norm) {
          LAG->rms_norm_seq(q, qvp, qvp, q_norm_f[static_cast<size_t>(l)].data(), c.Hq,
                            Dh, eps, true);
          LAG->rms_norm_seq(q, kn, kn, k_norm_f[static_cast<size_t>(l)].data(), Hkv, Dh, eps, true);
        }
        LAG->rope_from_cache_g(q, qvp, rcache, c.Hq, Dh, c.rd, pos_buf.data());
        LAG->rope_from_cache_g(q, kn, rcache, Hkv, Dh, c.rd, pos_buf.data());
      }
      // GRAPH decode attention: cache_k[l][0..len) + kn/vn as the new row, with
      // len/pos read from the DEVICE buffers. first_pos=0 (uniform, P<512).
      // LEVER A: the cache base is bf16 (cache_k16) or f32 (cache_k), passed via float* (bf16
      // is an address pun — the launcher reads the env and the kernels reinterpret to bf16).
      float* ckl = kv_bf16 ? reinterpret_cast<float*>(cache_k16[static_cast<size_t>(l)].data())
                           : cache_k[static_cast<size_t>(l)].data();
      float* cvl = kv_bf16 ? reinterpret_cast<float*>(cache_v16[static_cast<size_t>(l)].data())
                           : cache_v[static_cast<size_t>(l)].data();
      LAG->decode_attn_gqa_g(q, attn.data(), qvp, ckl, cvl, kn, vn, c.Hq, Hkv, Dh, c.group,
                             /*first_pos=*/0, c.window, scale, len_buf.data(), pos_buf.data(),
                             glue_fused ? glp : nullptr);  // L1: fold softplus out-gate
      // LEVER A: append this token's post-RoPE K (kn) and raw V (vn) into the growing
      // cache at the DEVICE-read slot *len_buf, IN-GRAPH — folds the 2×nlayers between-
      // replay host Copy launches into the captured graph. Runs after decode_attn_gqa_g
      // consumed kn/vn (slot len is outside its [0,len) cache read ⇒ no intra-replay RAW;
      // the write is seen by the NEXT replay's attention via same-stream ordering).
      LAG->append_kv_row(q, ckl, cvl, kn, vn, kvdim, len_buf.data());
      // g was produced by the fused GEMV (glp -> qkvg[l]); untouched until here.
      if (!glue_fused)  // L1 off => separate softplus out-gate pass
        LAG->softplus_head_gate(q, attn.data(), glp, c.Hq, Dh);
      GemmBf16(o.data(), lw.attn.o_proj, attn.data(), H, c.qdim);
      // L4 Pair 1 (post-attn): hidden += o; hn = rms_norm(hidden, post_norm[l]).
      if (glue_fused) {
        FusedAddNorm(hn.data(), o.data(), post_norm_f[static_cast<size_t>(l)].data(), hidden.data());
      } else {
        AddInto(hidden.data(), o.data());
        LAG->rms_norm_seq(q, hn.data(), hidden.data(), post_norm_f[static_cast<size_t>(l)].data(), 1,
                          H, eps, true);
      }
      // L4 Pair 2: the LAST residual add of this layer folds with the NEXT layer's input
      // RMSNorm (or the post-loop final RMSNorm on the last layer) — computed into hn.
      const float* next_norm = (l + 1 < nlayers)
                                   ? input_norm_f[static_cast<size_t>(l + 1)].data()
                                   : final_norm_f.data();
      if (c.is_dense) {
        GemmBf16(dg.data(), lw.mlp.gate_proj, hn.data(), dense_I, H);
        GemmBf16(du.data(), lw.mlp.up_proj, hn.data(), dense_I, H);
        SiluMul(dact.data(), dg.data(), du.data(), dense_I);
        GemmBf16(fdn.data(), lw.mlp.down_proj, dact.data(), H, dense_I);
        if (glue_fused) FusedAddNorm(hn.data(), fdn.data(), next_norm, hidden.data());
        else AddInto(hidden.data(), fdn.data());
      } else {
        // LEVER 2: fused router|shared_gate|shared_up GEMV -> rsg slices. VT_LAGUNA_SHARED_FP4:
        // the shared expert is fp4-resident, so drop to a router-ONLY GEMV (`moe.router`) and
        // run gate/up/down through the fp4 Marlin path (no bf16 shared weights read).
        // VT_LAGUNA_MOE_ONECAST: cast hn→bf16 ONCE here and feed the persistent buffer to the
        // router GEMM + routed Marlin + shared Marlin (each else-cast the SAME hn otherwise).
        const void* hnb = moe_onecast ? static_cast<const void*>(hn_bf16.data()) : nullptr;
        // ── VT_LAGUNA_SHARED_AUX FORK ─────────────────────────────────────────────
        // Issue the FULL fp4 shared expert on the aux stream NOW — forked from `hn`
        // (the post-attn-norm hidden) BEFORE the main-stream router GEMV — so it runs
        // concurrently with router+sigmoid_topk+routed grouped GEMM. This is the EARLY
        // fork vLLM does (shared_experts.py:125-129: record on main, aux waits, aux runs
        // the shared MLP). The aux path reads `hn` (f32) and does its OWN hn->bf16 cast
        // (hnb=nullptr) rather than the main-stream hn_bf16 — deterministic truncation of
        // the same hn ⇒ byte-identical `so`, and it removes the only main->aux data
        // dependency, so the aux path waits ONLY the fork point (hn ready). Its scratch is
        // drawn from AuxPool (disjoint from the main routed Pool blocks) so the two
        // concurrent streams never alias a live block.
        const bool do_aux = shared_aux;  // shared_aux already implies shared_fp4
        if (do_aux) {
          vt::Backend& b = vt::GetBackend(dev);
          b.RecordEvent(aux_fork, q);          // event0.record() on the main stream (hn ready)
          b.QueueWaitEvent(aux_q, aux_fork);   // aux waits event0 before reading hn
          // D10 (ENG-CUDAGRAPH-BREAK W5, #1335): tell the capture scope that a
          // side queue is now participating in this capture. Closing a segment
          // while a fork is outstanding FAILS at `cudaStreamEndCapture`, and the
          // scope is the only thing that can see a segment about to close.
          // Upstream keeps its whole `torch.cuda.Stream.wait_stream` monkey-patch
          // (`breakable_cuda_graph.py:101-153`) for no other purpose than to know
          // this; ours is one call, because our fork is explicit. OUTSIDE a
          // capture it is a no-op making zero backend calls, so the eager arm is
          // byte-identical. `aux_done` is the join event, and it outlives every
          // scope because this class owns it.
          vt::GraphNoteFork(aux_q, aux_done);
          ActivePoolScope guard(&AuxPool(b));  // shared scratch from AuxPool (see device_pool.h)
          LagunaSharedExpertMarlinInto(aux_q, lw.moe, hn.data(), H, so.data());  // fp4 shared on aux
          b.RecordEvent(aux_done, aux_q);      // event1.record() on the aux stream (join target)
        }
        if (moe_onecast) CastHnBf16(hn.data());
        if (shared_fp4) {
          if (moe_onecast) GemmBf16Pre(rsg.data(), lw.moe.router, E, H);  // router only
          else GemmBf16(rsg.data(), lw.moe.router, hn.data(), E, H);
        } else {
          if (moe_onecast) GemmBf16Pre(rsg.data(), lw.moe.router_shared_gu, E + 2 * moe_I, H);
          else GemmBf16(rsg.data(), lw.moe.router_shared_gu, hn.data(), E + 2 * moe_I, H);
        }
        float* gatingp = rsg.data();
        float* sgp = rsg.data() + E;
        float* sup = rsg.data() + E + moe_I;
        const float* bias = moe_bias_f[static_cast<size_t>(l)].empty()
                                ? nullptr
                                : moe_bias_f[static_cast<size_t>(l)].data();
        LAG->sigmoid_topk(q, eids32.data(), topw.data(), gatingp, bias, bias != nullptr, E,
                          topk, p->norm_topk_prob, p->moe_routed_scaling_factor);
        // VT_LAGUNA_TAIL_FUSED: when on, MoeCombine writes the routed output as bf16 into
        // doutb_bf16 (no CastF32) and the bf16-x1 fused_add2 below widens it in-kernel.
        void* routed_bf16 = tail_fused ? static_cast<void*>(doutb_bf16.data()) : nullptr;
        LagunaMoeResidentMarlinInto(q, lw.moe, hn.data(), moe_I, H, E, eids32.data(), topw.data(),
                                    topk, doutb.data(), hnb, routed_bf16);
        if (shared_fp4) {
          // When do_aux, the shared expert was already issued on the aux stream above;
          // otherwise run it serially on the main stream here (fp4 shared).
          if (!do_aux)
            LagunaSharedExpertMarlinInto(q, lw.moe, hn.data(), H, so.data(), hnb);
        } else {
          SiluMul(dact.data(), sgp, sup, moe_I);
          GemmBf16(so.data(), lw.moe.shared_down, dact.data(), H, moe_I);
        }
        // ── VT_LAGUNA_SHARED_AUX JOIN ─────────────────────────────────────────────
        // Make the main stream wait for the aux shared MLP (event1.wait) so the combine
        // below reads a fully-computed `so`. Both the routed path (main) and the shared
        // path (aux) are now complete → the combine result is byte-identical to serial.
        if (do_aux) {
          vt::GetBackend(dev).QueueWaitEvent(q, aux_done);
          // D10: the model joined its own fork, so retire the registration. This
          // is the common arm and the one that must cost nothing — without it the
          // scope would issue a REDUNDANT second join before every segment close.
          vt::GraphNoteJoin(aux_q);
        }
        // LEVER A: hidden += routed + shared; hn = rms_norm(hidden)*next_norm in ONE node
        // (byte-exact vs AddInto(doutb) + FusedAddNorm(so)). Only in the glue-fused regime
        // (the split path is what it replaces); =0 restores the two-node split.
        if (tail_fused) {
          // routed contribution (x1) is bf16 in doutb_bf16, widened in-kernel; shared (x2) f32.
          LAG->fused_add2_rmsnorm_bf16x1(q, hn.data(), hidden.data(),
                                         static_cast<const void*>(doutb_bf16.data()), so.data(),
                                         next_norm, H, eps);
        } else if (glue_fused && moe_addnorm_fused) {
          LAG->fused_add2_rmsnorm(q, hn.data(), hidden.data(), doutb.data(), so.data(), next_norm,
                                  H, eps);
        } else {
          AddInto(hidden.data(), doutb.data());  // hidden += routed (always a plain add)
          // hidden += shared, folded with the next layer's input (or final) RMSNorm.
          if (glue_fused) FusedAddNorm(hn.data(), so.data(), next_norm, hidden.data());
          else AddInto(hidden.data(), so.data());
        }
      }
    }
    // final RMSNorm + lm_head INSIDE the captured region (device), into persistent
    // logits — the resident/non-graph path does this host-side (LagunaFinalLogits);
    // moving it on-device adds one block-reduced-norm near-tie point (accepted device
    // regime, gated vs vLLM). lm_head is the bf16 tower weight (nvfp4 arm). The
    // dedicated M=1 GEMV streams the ~616 MB weight ONCE at ~roofline — replacing the
    // GemmBf16 MatmulBT that cuBLASLt mis-routed to a batched wmma tile algo (~20% of
    // roofline, the measured #1 decode GPU cost). Fixed grid=Vsz + fixed pointers ⇒
    // capture-safe. (f32/quant lm_head would fall back to GemmBf16, but the nvfp4 arm
    // is always bf16 here.)
    if (!glue_fused)  // L4: fused path already produced hn=norm(hidden,final_norm) in the
                      // last layer's Pair-2 tail (next_norm==final_norm_f when l+1==nlayers)
      LAG->rms_norm_seq(q, hn.data(), hidden.data(), final_norm_f.data(), 1, H, eps, true);
    if (w->lm_head.dtype == vt::DType::kBF16) {
      // VT_LAGUNA_RESIDENT_BF16W: stream the ~616 MB lm_head from a true device copy
      // (uploaded once in the gstate-0 warm-run) instead of the unified host bytes.
      vt::Tensor lmw = LagunaResidentBf16W(q, w->lm_head, dev);
      LAG->lm_head_gemv(q, logits.data(), lmw.data, hn.data(), Vsz, H);
    } else
      GemmBf16(logits.data(), w->lm_head, hn.data(), Vsz, H);
    // VT_LAGUNA_ONDEV_SAMPLE: greedy argmax the logits ON-DEVICE into argmax_id (the next
    // step's input token). vt::GreedyArgmax uses a two-pass reduction with a LOWEST-index
    // tie-break — the exact winner ArgmaxLastRow(host) produces — so the token stream is
    // unchanged. Its partials scratch is grown ONCE by the gstate-0 eager warm-run (same
    // launcher sequence), so no cudaMalloc runs inside the gstate-1 capture. Fixed pointers
    // ⇒ capture-safe; runs AFTER the lm_head write (same-stream RAW on logits).
    if (ondev) {
      vt::Tensor lt = vt::Tensor::Contiguous(logits.data(), vt::DType::kF32, dev, {1, Vsz});
      vt::Tensor it = vt::Tensor::Contiguous(argmax_id.data(), vt::DType::kI64, dev, {1});
      vt::GreedyArgmax(q, it, lt);
    }
  }

  // One decode token: refresh the persistent inputs OUTSIDE capture (embed + the two
  // device-read scalars pos/len — the RoPE table and KV append are now BOTH in-graph),
  // run/replay the step, read logits. Mirror of V4Graph::Step.
  std::vector<float> Step(int32_t token, int32_t pos) {
    vt::Queue& q = *qp;
    VT_CHECK(kv_rows + 1 <= max_cap, "laguna decode graph: KV capacity exceeded");
    // (1) embed → persistent hidden. VT_LAGUNA_ONDEV_SAMPLE: the gather is ON-DEVICE inside
    // RunChain (embed_gather reads argmax_id); here we only SEED argmax_id with this step's
    // input token (an 8-byte host write, replacing the H-element host embed loop below).
    // token == the previous step's on-device argmax result, so the seed is value-identical
    // to letting the graph feed itself — but keeps the driver in control of the token.
    if (ondev) {
      argmax_id[0] = static_cast<int64_t>(token);
    } else {
      StageLagunaGraphEmbedding(w->embed, token, H, Vsz, hidden.data());
    }
    // (2) device-read scalars (the ONLY per-step host refresh now: pos indexes the
    // in-graph RoPE table on-device, len drives both the in-graph attention and KV append).
    pos_buf[0] = pos;
    len_buf[0] = static_cast<int>(kv_rows);
    vt::Backend& b = vt::GetBackend(dev);
    if (gstate == 0) {         // cold: eager warm-run (warms the pool + GEMM scratch)
      RunChain();
      gstate = 1;
    } else if (gstate == 1) {  // warm: capture the region once, then replay it
      // ENG-CUDAGRAPH-BREAK W5 (#1335): the capture is the SHARED SEAM's, not
      // this class's hand-rolled `BeginCapture`/`EndCaptureGraph` pair. The scope
      // owns the segment, the handle, its release, the drain a mid-capture throw
      // needs, the D10 fork set, and the G3 counters.
      //
      // kFULL, INHERITED FROM W2 AND NOT RE-ARGUED. vLLM's v1 default
      // `FULL_AND_PIECEWISE` (`vllm/config/compilation.py:63` @ pin
      // `5559679229`) is documented at `:630-632` as a FULL graph for DECODE
      // batches and a piecewise one for prefill and mixed batches, and
      // `decode_mode()` (`:65-66`) returns the full half. This is the T=1
      // resident decode step, so its capture is ONE segment with the attention
      // calls INSIDE it — byte-identical in shape to the region this replaces.
      {
        vt::GraphCaptureScope scope(b, q, graph, vt::GraphCaptureMode::kFull);
        RunChain();
      }  // ~GraphCaptureScope joins any outstanding fork, closes the segment,
         // and files it on `graph`
      // NOT CAPTURED covers TWO states and they mean opposite things.
      //
      //   * FAILED (`capture_failed() == true`): `Backend::EndCaptureGraph`
      //     threw. Under stream capture NOTHING between `BeginCapture` and the
      //     throw executed — every kernel was RECORDED — so `logits` (and
      //     `argmax_id` on the on-device-sample arm) hold whatever the pool last
      //     left there, and returning them would hand this step uncomputed
      //     device memory as its logits: no fault, and a token gate cannot see
      //     it. It PROPAGATES, carrying the runtime's own exception, which is
      //     what the pre-W5 unguarded `EndCaptureGraph` did.
      //   * INERT (`capture_failed() == false`): capture is unsupported here, or
      //     `VLLM_CPP_CUDAGRAPH=0`. The scope made no backend call, `RunChain`
      //     ran EAGERLY, and the buffers hold a real result — so this step
      //     returns normally and the driver stays in `gstate == 1`, running
      //     eager every step rather than pretending to hold a graph.
      if (!graph.captured()) {
        if (graph.capture_failed()) {
          const std::exception_ptr err = graph.capture_error();
          graph.Reset();  // clear the failure with the graph it described
          if (err) std::rethrow_exception(err);
          VT_CHECK(false,
                   "laguna decode graph: the capture was ABANDONED and its logits were "
                   "never computed; refusing to return uncaptured device memory");
        }
      } else {
        graph.Replay(q);
        gstate = 2;
      }
    } else {                   // captured: one cudaGraphLaunch
      // Through the seam's container, never `Backend::ReplayGraph` directly: it
      // replays its segments in order (one, here, because this capture is kFull)
      // and owns the G3 replay counter.
      graph.Replay(q);
    }
    // LEVER A: the per-layer K/V append is now IN-GRAPH (append_kv_row writes the new row
    // at the device-read slot *len_buf inside RunChain), so the old between-replay host
    // Copy loop (2×nlayers launches/step) is gone — only the row count advances here.
    kv_rows++;
    b.Synchronize(q);  // the ONE step-boundary drain: logits ready, appends complete
    // VT_LAGUNA_ONDEV_SAMPLE: the graph already argmaxed logits into argmax_id — read the
    // winning token (8 bytes) instead of the caller downloading the full [vocab] logits for
    // a host argmax. Return an empty vector (the driver consumes last_sampled). Off: the
    // full logits go back for the host argmax (unchanged path).
    if (ondev) {
      last_sampled = static_cast<int32_t>(argmax_id[0]);
      return {};
    }
    return logits;
  }
};
}  // namespace
#endif  // VT_MARLIN_NVFP4

std::vector<float> LagunaForwardGgufCached(const LagunaWeights& weights, vt::Queue& q,
                                           LagunaKvCache& cache,
                                           const std::vector<int32_t>& token_ids,
                                           const std::vector<int32_t>& positions,
                                           const std::vector<int32_t>& logits_indices) {
  const LagunaParams& p = weights.params;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = p.hidden_size;
  const int64_t Vsz = p.vocab_size;
  const int64_t Dh = p.head_dim;
  const int64_t Hkv = p.num_key_value_heads;
  const int64_t kvdim = Hkv * Dh;
  const float eps = p.rms_norm_eps;
  const int64_t nlayers = p.num_hidden_layers;
  VT_CHECK(weights.has_gguf_weights || weights.has_nvfp4_weights,
           "laguna cached forward: no keep-quant or nvfp4 tower");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "laguna gguf cached: positions length must match token_ids");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == nlayers,
           "laguna gguf cached: one LagunaLayerWeights per layer required");
  if (cache.k.empty()) cache.Reset(nlayers, Dh, Hkv);  // lazy init on first call
  VT_CHECK(static_cast<int64_t>(cache.k.size()) == nlayers && cache.head_dim == Dh &&
               cache.kv_heads == Hkv,
           "laguna gguf cached: cache not sized for this model");
  // The base global position of this call's first token = cache.len. Positions must
  // be contiguous from there (prefill: 0..T-1 with len==0; decode: {len}).
  const int64_t base = cache.len;
  VT_CHECK(positions[0] == static_cast<int32_t>(base),
           "laguna gguf cached: positions[0] must equal cache.len (contiguous decode)");

  // N5 device-resident T=1 decode fast path (VT_LAGUNA_RESIDENT_DECODE, default OFF).
  if (LagunaCanRunResidentDecode(p, q, weights, T) && logits_indices.size() == 1 &&
      logits_indices[0] == 0) {
    // DSR-ALLOW(S1): decode CUDA-graph dispatch; retires with the capture class above.
#ifdef VT_MARLIN_NVFP4
    // Brick A2 (VT_LAGUNA_DECODE_GRAPH=1): after prefill (cache.len>0), drive the
    // resident step through a captured CUDA graph — one cudaGraphLaunch/step. The
    // prefill step (cache.len==0, T>1) never reaches here (T!=1 fails the guard) and
    // fills cache.k/cache.v the graph seeds from. Default OFF; the A1 eager resident
    // path below is the fallback. Mirror of DeepseekV4ForwardGgufCached (deepseek_v4.
    // cpp:2173-2183).
    if (LagunaDecodeGraphEnabled() && cache.len > 0) {
      if (!cache.decode_graph) {
        cache.decode_graph = std::shared_ptr<void>(
            new LagunaGraph(weights, q, cache),
            [](void* g) { delete static_cast<LagunaGraph*>(g); });
      }
      auto* g = static_cast<LagunaGraph*>(cache.decode_graph.get());
      std::vector<float> lg = g->Step(token_ids[0], positions[0]);
      // VT_LAGUNA_ONDEV_SAMPLE: surface the graph's on-device argmax id to the driver so it
      // skips the host argmax over the empty logits (-1 when off ⇒ caller host-argmaxes lg).
      cache.last_sampled = g->last_sampled;
      cache.len += T;
      return lg;
    }
#endif  // VT_MARLIN_NVFP4
    std::vector<float> lg =
        LagunaForwardResidentDecode(weights, q, cache, token_ids, positions, logits_indices);
    cache.len += T;
    return lg;
  }

  int64_t max_pos = 0;
  for (int32_t ps : positions) max_pos = std::max<int64_t>(max_pos, ps);
  const int64_t rope_rows = max_pos + 1;
  const std::vector<float> yarn_cache = BuildLagunaFullYarnCosSin(p, rope_rows);
  const std::vector<float> slide_cache = BuildLagunaSlidingCosSin(p, rope_rows);

  std::vector<float> hidden = LagunaEmbed(weights.embed, token_ids, H, Vsz);

  std::vector<int64_t> q_pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) q_pos[static_cast<size_t>(t)] = positions[static_cast<size_t>(t)];

  for (int64_t l = 0; l < nlayers; ++l) {
    const LagunaLayerWeights& lw = weights.layers[static_cast<size_t>(l)];
    const int64_t Hq = p.QHeadsForLayer(l);
    const int64_t qdim = Hq * Dh;
    const int64_t group = p.GqaGroupForLayer(l);
    VT_CHECK(group > 0 && Hq == group * Hkv,
             "laguna gguf cached: per-layer Q-head count must be a multiple of KV heads");
    const bool global = p.IsGlobalLayer(l);
    const int64_t rd = p.RotaryDimForLayer(l);
    const int64_t window = p.WindowForLayer(l);

    // --- attention: project THIS call's T tokens' q/k/v, qk-norm + RoPE ---
    const std::vector<float> hn = RmsNorm(hidden, ReadF32(lw.input_norm), T, H, eps);
    std::vector<float> qv = LqGemm(q, lw.attn.q_proj, hn, T, qdim, H);
    std::vector<float> knew = LqGemm(q, lw.attn.k_proj, hn, T, kvdim, H);
    std::vector<float> vnew = LqGemm(q, lw.attn.v_proj, hn, T, kvdim, H);
    if (p.has_qk_norm && !lw.attn.q_norm.Empty()) {
      RmsNormHeads(qv, ReadF32(lw.attn.q_norm), T, Hq, Dh, eps);
      RmsNormHeads(knew, ReadF32(lw.attn.k_norm), T, Hkv, Dh, eps);
    }
    const std::vector<float>& rope = global ? yarn_cache : slide_cache;
    ApplyRope(qv, T, Hq, Dh, rd, rope, positions);
    ApplyRope(knew, T, Hkv, Dh, rd, rope, positions);

    // Append the T new K/V rows to this layer's cache. The cached K is POST-RoPE /
    // POST-QK-norm and V is raw — both position-only functions, so bit-exact.
    std::vector<float>& kc = cache.k[static_cast<size_t>(l)];
    std::vector<float>& vc = cache.v[static_cast<size_t>(l)];
    kc.insert(kc.end(), knew.begin(), knew.end());
    vc.insert(vc.end(), vnew.begin(), vnew.end());
    // Sliding-window eviction (gemma2/3 is_sliding): keep only the last `window`
    // rows — a query at global position pi attends kv with pi - pj < window, so once
    // more than `window` rows are cached the oldest can never be scored again. Global
    // layers (window==0) keep the whole history. first_pos tracks the evicted base.
    int64_t rows = static_cast<int64_t>(kc.size()) / kvdim;
    if (window > 0 && rows > window) {
      const int64_t drop = rows - window;
      const size_t off = static_cast<size_t>(drop * kvdim);
      kc.erase(kc.begin(), kc.begin() + static_cast<int64_t>(off));
      vc.erase(vc.begin(), vc.begin() + static_cast<int64_t>(off));
      cache.first_pos[static_cast<size_t>(l)] += drop;
      rows = window;
    }
    // Global positions of the cached rows: first_pos .. first_pos+rows-1.
    const int64_t fp = cache.first_pos[static_cast<size_t>(l)];
    std::vector<int64_t> kv_pos(static_cast<size_t>(rows));
    for (int64_t r = 0; r < rows; ++r) kv_pos[static_cast<size_t>(r)] = fp + r;

    std::vector<float> attn = LagunaAttention(qv, kc, vc, T, rows, Hq, Hkv, Dh, group,
                                              q_pos, kv_pos, window);

    // per-head softplus attention OUTPUT gate.
    const std::vector<float> glogits = LqGemm(q, lw.attn.g_proj, hn, T, Hq, H);
    for (int64_t i = 0; i < T; ++i) {
      std::vector<float> row(attn.begin() + static_cast<int64_t>(i * qdim),
                             attn.begin() + static_cast<int64_t>((i + 1) * qdim));
      std::vector<float> gl(glogits.begin() + static_cast<int64_t>(i * Hq),
                            glogits.begin() + static_cast<int64_t>((i + 1) * Hq));
      LagunaSoftplusHeadGate(row, gl, Hq, Dh);
      std::copy(row.begin(), row.end(), attn.begin() + static_cast<int64_t>(i * qdim));
    }

    const std::vector<float> o = LqGemm(q, lw.attn.o_proj, attn, T, H, qdim);
    for (int64_t i = 0; i < T * H; ++i) hidden[static_cast<size_t>(i)] += o[static_cast<size_t>(i)];

    // --- FFN: dense SwiGLU (layer 0) or ungrouped sigmoid-noaux MoE ---
    const std::vector<float> hn2 = RmsNorm(hidden, ReadF32(lw.post_attn_norm), T, H, eps);
    const std::vector<float> f = LagunaFfnBlock(q, lw, p, hn2, T);
    for (int64_t i = 0; i < T * H; ++i) hidden[static_cast<size_t>(i)] += f[static_cast<size_t>(i)];
  }

  cache.len += T;  // every layer appended its T rows
  return LagunaFinalLogits(q, weights, hidden, T, logits_indices);
}

}  // namespace vllm
