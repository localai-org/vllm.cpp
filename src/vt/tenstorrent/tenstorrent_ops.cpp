// Tenstorrent op providers — the ttnn adapter layer (BACKEND-TENSTORRENT,
// .agents/specs/tenstorrent-backend.md). vllm.cpp original; no upstream
// mirror (vLLM has no Tenstorrent platform). Op table: OPT-125m's 9 ops plus
// the Qwen3-dense deltas (kRmsNorm first; kSiluAndMul / RoPE / Cast next),
// matching Metal's OPT→Qwen3 sequencing. ttnn for compute where available;
// host-staged pure data-movement / attention for the remainder (see
// HOST-STAGED OPS note below).
//
// SCOPE: F32 for the W0 path unless noted. kAdd allows rank-1 bias
// broadcast; kLayerNorm optional rank-1 weight/bias; kRmsNorm weight +
// optional residual stream; kEmbedding i32/i64 ids. Every other shape/dtype
// is a VT_CHECK failure — no CPU reference tier (UnifiedMemory()==false).
//
// HOST-STAGED OPS (kReshapeAndCache, kPagedAttention): this backend's Alloc is
// host memory (tenstorrent_backend.cpp). ReshapeAndCache is a pure contiguous /
// stride-aware page write. PagedAttention uses the CPU-oracle f32 softmax over
// the host-resident paged cache; mapping vLLM's block-table contract onto
// ttnn::sdpa_decode is deferred. kQkvSplit is hybrid: device-slice when the
#include "vt/tenstorrent/tenstorrent_internal.h"

namespace vt::tenstorrent {
namespace {
// Publish a device result as the current value of `out` WITHOUT downloading
void CommitDevice2D(Tensor& out, ttnn::Tensor dev) {
  VT_CHECK(out.rank == 2 && out.IsContiguous(),
           "tenstorrent: CommitDevice2D expects contiguous rank-2 out");
  CommitDeviceLogical2D(out, std::move(dev), static_cast<uint32_t>(out.shape[0]),
                        static_cast<uint32_t>(out.shape[1]));
}

}  // namespace

// Host wrote `out` in place — drop any device shadow.
void CommitHost(Tensor& out) {
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(out.data);
  if (s == nullptr) return;
  s->host_current = true;
  s->device_current = false;
  s->device = std::nullopt;
  s->conv_transposed = false;
  s->device_reserved = false;  // real bytes now — the upload contract applies
}

namespace {

// Device compute: keep result on device (CommitDevice2D). Host round-trip only
// when the consumer is a host-staged op (EnsureHost) or an untracked buffer.
void MatmulKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  TT_OP_TRACE("Matmul");
  VT_CHECK(a.rank == 2 && b.rank == 2 && out.rank == 2,
           "tenstorrent kMatmul: only rank-2 tensors are supported in W0");
  VT_CHECK(IsFloatDType(a.dtype) && IsFloatDType(b.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kMatmul: float in, f32/bf16 out");
  const uint32_t M = static_cast<uint32_t>(a.shape[0]);
  const uint32_t K = static_cast<uint32_t>(a.shape[1]);
  const uint32_t N = static_cast<uint32_t>(b.shape[1]);
  VT_CHECK(b.shape[0] == K, "tenstorrent kMatmul: a/b inner dimension mismatch");
  VT_CHECK(out.shape[0] == M && out.shape[1] == N, "tenstorrent kMatmul: out shape mismatch");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "tenstorrent kMatmul: strided (non-contiguous) tensors are not supported in W0");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_a = EnsureDevice2D(a, device);
  ttnn::Tensor dev_b = EnsureMatmulWeightDevice(b, device);
  if (dev_b.dtype() == ttnn::DataType::BFLOAT8_B) Bfp8MatmulUse();
  ttnn::Tensor dev_c = ttnn::operations::matmul::matmul(dev_a, dev_b);
  CommitDevice2D(out, std::move(dev_c));
}

// kMatmulBT: `b` is a [N,K] row-major torch nn.Linear weight; computes
// `a @ b^T` (cpu_ops.cpp's MatmulBTKernel contract). ttnn's matmul() already
// exposes a transpose_b flag, so this is the same sequence as kMatmul with
// that flag flipped — no separate upload shape needed since `b` is uploaded
// in its native [N,K] layout and ttnn transposes on device.
void MatmulBTKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  TT_OP_TRACE("MatmulBT");
  VT_CHECK(a.rank == 2 && b.rank == 2 && out.rank == 2,
           "tenstorrent kMatmulBT: only rank-2 tensors are supported in W0");
  VT_CHECK(IsFloatDType(a.dtype) && IsFloatDType(b.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kMatmulBT: float in, f32/bf16 out");
  const uint32_t M = static_cast<uint32_t>(a.shape[0]);
  const uint32_t K = static_cast<uint32_t>(a.shape[1]);
  const uint32_t N = static_cast<uint32_t>(b.shape[0]);
  VT_CHECK(b.shape[1] == K, "tenstorrent kMatmulBT: a/b inner dimension mismatch");
  VT_CHECK(out.shape[0] == M && out.shape[1] == N, "tenstorrent kMatmulBT: out shape mismatch");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "tenstorrent kMatmulBT: strided (non-contiguous) tensors are not supported in W0");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_a = EnsureDevice2D(a, device);
  ttnn::Tensor dev_b = EnsureMatmulWeightDevice(b, device);
  if (dev_b.dtype() == ttnn::DataType::BFLOAT8_B) Bfp8MatmulUse();
  ttnn::Tensor dev_c =
      ttnn::operations::matmul::matmul(dev_a, dev_b, /*transpose_a=*/false, /*transpose_b=*/true);
  CommitDevice2D(out, std::move(dev_c));
}

// kAdd: elementwise add, plus the rank-1 `b` row-broadcast form used for
// nn.Linear bias (cpu_layernorm.cpp's AddKernel contract). ttnn::add needs
// same-rank operands, so the broadcast case uploads `b` replicated into a
// [rows, d] tile rather than relying on ttnn's own broadcast rules — keeps
// this kernel's behavior pinned to the CPU reference rather than to
// whatever ttnn::add happens to support today.
void AddKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  TT_OP_TRACE("Add");
  VT_CHECK(a.rank == 2 && out.rank == 2, "tenstorrent kAdd: `a`/`out` must be rank-2 in W0");
  VT_CHECK(b.rank == 2 || b.rank == 1, "tenstorrent kAdd: `b` must be rank-1 or rank-2 in W0");
  VT_CHECK(IsFloatDType(a.dtype) && IsFloatDType(b.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kAdd: float in, f32/bf16 out");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "tenstorrent kAdd: strided (non-contiguous) tensors are not supported in W0");
  const uint32_t rows = static_cast<uint32_t>(a.shape[0]);
  const uint32_t d = static_cast<uint32_t>(a.shape[1]);
  VT_CHECK(out.shape[0] == rows && out.shape[1] == d, "tenstorrent kAdd: out shape mismatch");
  const bool bcast = b.rank == 1;
  VT_CHECK(bcast ? b.shape[0] == d : (b.shape[0] == rows && b.shape[1] == d),
           "tenstorrent kAdd: `b` shape mismatch");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_a = EnsureDevice2D(a, device);
  ttnn::Tensor dev_b;
  if (bcast) {
    EnsureHost(b);
    std::vector<float> replicated(static_cast<size_t>(rows) * d);
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t c = 0; c < d; ++c)
        replicated[static_cast<size_t>(r) * d + c] = LoadElemF32(b, c);
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
      std::fprintf(stderr, "[TT-UP] AddKernel from_vector WRITE during capture\n");
    dev_b = ttnn::Tensor::from_vector<float>(replicated, TileSpecOf(rows, d), &device);
  } else {
    dev_b = EnsureDevice2D(b, device);
  }
  ttnn::Tensor dev_c = ttnn::add(dev_a, dev_b);
  CommitDevice2D(out, std::move(dev_c));
}

// kRelu: elementwise max(0, x) (cpu_layernorm.cpp's ReluKernel contract).
void ReluKernel(Queue&, Tensor& out, const Tensor& x) {
  VT_CHECK(x.rank == 2 && out.rank == 2, "tenstorrent kRelu: only rank-2 tensors are supported in W0");
  VT_CHECK(IsFloatDType(x.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kRelu: float in, f32/bf16 out");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1],
           "tenstorrent kRelu: out shape mismatch");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(),
           "tenstorrent kRelu: strided (non-contiguous) tensors are not supported in W0");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = EnsureDevice2D(x, device);
  ttnn::Tensor dev_y = ttnn::relu(dev_x);
  CommitDevice2D(out, std::move(dev_y));
}

}  // namespace

// ---- Persistent embedding-table shadows (ROW_MAJOR BF16 on device) --------
// The vocab table is multi-hundred MB for Qwen3; re-uploading every forward
// was a pure tax. Keyed by host table base; invalidated by MarkHostWritten /
// UnregisterHostBuffer. The EmbedTableShadow struct lives in
// tenstorrent_internal.h (split stage 5: the capture TU reads the map too).
std::mutex& EmbedTableMutex() {
  static std::mutex m;
  return m;
}
std::map<uintptr_t, EmbedTableShadow>& EmbedTableShadows() {
  static std::map<uintptr_t, EmbedTableShadow>* m = new std::map<uintptr_t, EmbedTableShadow>(); // never destroyed (#1486)
  return *m;
}
namespace {

// kEmbedding: row gather `out[i,:] = table[ids[i],:]` (cpu_ops.cpp
// EmbeddingKernel contract). Two layout departures from the TILE/BFLOAT16
// linear ops, forced by ttnn::embedding's validate path:
//   1. ids upload as ROW_MAJOR UINT32 (ttnn rejects i32/i64; vt still accepts
//      kI32/kI64 at the seam and converts host-side, matching Metal/Vulkan).
//   2. table is ROW_MAJOR BFLOAT16 (cached on device after first use).
// Parameter order at the ttnn call is (ids, table) — reversed from
// vt::EmbeddingFn's (table, ids). Output is TILE so the next matmul can keep
// the activation device-resident without a host round-trip.
void EmbeddingKernel(Queue&, Tensor& out, const Tensor& table, const Tensor& ids) {
  TT_OP_TRACE("Embedding");
  VT_CHECK(table.rank == 2 && ids.rank == 1 && out.rank == 2,
           "tenstorrent kEmbedding: table rank-2, ids rank-1, out rank-2");
  VT_CHECK(IsFloatDType(table.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kEmbedding: float table, f32/bf16 out");
  VT_CHECK(ids.dtype == DType::kI32 || ids.dtype == DType::kI64,
           "tenstorrent kEmbedding: ids must be i32 or i64");
  VT_CHECK(table.IsContiguous() && ids.IsContiguous() && out.IsContiguous(),
           "tenstorrent kEmbedding: strided (non-contiguous) tensors are not supported");
  const uint32_t vocab = static_cast<uint32_t>(table.shape[0]);
  const uint32_t h = static_cast<uint32_t>(table.shape[1]);
  const uint32_t t = static_cast<uint32_t>(ids.shape[0]);
  VT_CHECK(out.shape[0] == t && out.shape[1] == h, "tenstorrent kEmbedding: out shape mismatch");

  EnsureHost(ids);
  std::vector<uint32_t> host_ids(t);
  if (ids.dtype == DType::kI32) {
    const int32_t* p = ids.Ptr<int32_t>();
    for (uint32_t i = 0; i < t; ++i) {
      VT_CHECK(p[i] >= 0 && static_cast<uint32_t>(p[i]) < vocab,
               "tenstorrent kEmbedding: id out of range");
      host_ids[i] = static_cast<uint32_t>(p[i]);
    }
  } else {
    const int64_t* p = ids.Ptr<int64_t>();
    for (uint32_t i = 0; i < t; ++i) {
      VT_CHECK(p[i] >= 0 && static_cast<uint64_t>(p[i]) < vocab,
               "tenstorrent kEmbedding: id out of range");
      host_ids[i] = static_cast<uint32_t>(p[i]);
    }
  }
  MeshDevice& device = SharedMeshDevice();
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] EmbeddingKernel from_vector WRITE during capture\n");
  ttnn::Tensor dev_ids = ttnn::Tensor::from_vector<uint32_t>(
      host_ids, SpecOf(tt::tt_metal::Shape({t}), ttnn::DataType::UINT32, ttnn::Layout::ROW_MAJOR),
      &device);
  ttnn::Tensor dev_table = EnsureEmbedTableDevice(table, device);
  // TILE output → CommitDevice2D so the first residual/RMS/matmul reuses it.
  ttnn::Tensor dev_out = ttnn::embedding(dev_ids, dev_table, /*pad_token=*/std::nullopt,
                                         /*layout=*/ttnn::Layout::TILE);
  // embedding may return [t, h] or a higher-rank view; normalize to [t, h].
  if (dev_out.logical_shape().rank() != 2 ||
      dev_out.logical_shape()[0] != t || dev_out.logical_shape()[1] != h) {
    dev_out = ttnn::reshape(dev_out, ttnn::Shape({t, h}));
  }
  // HOST-FREE-DECODE: when the caller's buffer already carries a CURRENT
  // device shadow of the same shape (the decode-graph driver's PERSISTENT
  // hidden buffer), refresh that shadow IN PLACE (device->device copy) so
  // its device address never moves. A captured region reads the address
  // recorded at capture time; replacing the shadow here would leave every
  // replay reading the capture-step embedding. First call (no shadow yet)
  // commits normally.
  bool in_place = false;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(out.data);
    in_place = s != nullptr && s->device_current && s->device.has_value() &&
               s->dev_rows == t && s->dev_cols == h;
    if (in_place) {
      ttnn::copy(dev_out, *s->device);
      s->host_current = false;
    }
  }
  if (!in_place) CommitDevice2D(out, std::move(dev_out));
}

// kKeepQuantDecode: packed-block decode as a device compute chain — the packed
// stream is decoded on the device, not staged through a host decode. Contracts
// are DequantQ4_K/DequantQ5_K/DequantQ6_K/DequantQ8_0 (cpu_quant_dequant.cpp;
// llama.cpp dequantize_row_q4_K:1471, q5_K:1673, q6_K:1881, q8_0:495). Q4_K:
// per block_q4_K {f16 d; f16 dmin; u8 scales[12]; u8 qs[128]}, y[g*32+l] =
// d1*(nib) - m1 with d1 = d*sc, m1 = dmin*mm, sc/mm from GetScaleMinK4(is,
// scales), groups 0..7 over the 8 32-nibble lanes (low nibbles first, then
// high nibbles, of the same 32 bytes). Every unpack step is device work:
// bitwise masks/shifts for the nibbles, scale bytes, and f16 halves;
// reshape/permute only reorder lanes; d/dmin widen f16->f32 exactly via the
// integer bit-construction chain of vt::F16ToF32 (loader
// minimax_h3_vae_loader.cpp:47): normal/inf bits are sign | (exp+112)<<23 |
// mant<<13, subnormals ride the exact integer->f32 typecast times 2^-24,
// zero keeps its sign bit, and one i32->f32 bitcast materializes the f32.
// The one divergence from F16ToF32: a NaN scale's payload canonicalizes to
// inf across the SFPU bitcast (hardware pin) — corrupt-GGUF territory, and
// every finite pattern is bit-exact. nibbles and scale factors typecast to
// f32 exactly (values <= 255); d1/m1 and the final y = d1*nib - m1 are
// separate f32 ttnn ops (no FMA), the same IEEE order as the host under
// -ffp-contract=off.
//
// THE SHARED DECODE (KEEPQUANT W2, generalized W3): the packed-block -> f32
// chains, each returning the repaired f32 {rows, nb*elems} in ROW_MAJOR.
// KeepQuantDecode commits it to the host-visible output; the quant matmul
// (MatmulBTQuantKernel below) consumes it as the weight operand. One decode
// per encoding, two consumers, one numerics authority:
//   Q4_K 144B/256 and Q5_K 176B/256: {f16 d; f16 dmin; u8 scales[12]; ...}
//   — Q5_K inserts qh[32] between scales and qs (the 5th bit, +16) and its
//   tail is Q4_K's verbatim; Q6_K 210B/256: {u8 ql[128]; u8 qh[64]; i8
//   scales[16]; f16 d} — SIGNED 8-bit scales, 16 sub-blocks of 16, no min
//   term, q = nib6 - 32; Q8_0 34B/32: {f16 d; i8 qs[32]} — plain int8 times
//   the f16 scale. 210 and 34 are not multiples of four, so those blocks pad
//   to 53 / 9 words (2 pad bytes at the stream tail, zero-filled).
//
// STAGING (W3): the packed stream is staged as a resident i32 word shadow ONCE
// per weight (EnsureKeepQuantWords below, the EnsureMatmulWeightDevice
// persistent-shadow pattern), so the per-call decode runs entirely on-core
// from the resident words — no host repack, no from_vector upload. The -0.0f
// signed-zero repair constant is built by BIT ops from the zero-cache's +0
// (multiply canonicalizes -0 to +0 on the SFPU, so the sign bit is set below
// the float domain), because a per-call from_vector of the constant would be
// exactly the captured-graph write this wave removes.




// kLayerNorm: per-row mean/var over the last dim (cpu_layernorm.cpp
// LayerNormKernel / ATen native_layer_norm). Biased (1/N) variance; optional
// rank-1 weight/bias (elementwise_affine). Uses ttnn::layer_norm with the
// same TILE/BFLOAT16 upload path as the linear ops; eps comes from
// LayerNormArgs (OPT default 1e-5, not ttnn's 1e-12 default).
void LayerNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor* weight,
                     const Tensor* bias, const LayerNormArgs& args) {
  TT_OP_TRACE("LayerNorm");
  VT_CHECK(x.rank == 2 && out.rank == 2,
           "tenstorrent kLayerNorm: only rank-2 tensors are supported in this step");
  VT_CHECK(IsFloatDType(x.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kLayerNorm: float in, f32/bf16 out");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1],
           "tenstorrent kLayerNorm: out shape mismatch");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(),
           "tenstorrent kLayerNorm: strided (non-contiguous) tensors are not supported");
  VT_CHECK(args.eps >= 0.0f, "tenstorrent kLayerNorm: eps must be non-negative");
  const uint32_t d = static_cast<uint32_t>(x.shape[1]);
  for (const Tensor* p : {weight, bias}) {
    if (p == nullptr) continue;
    VT_CHECK(p->rank == 1 && p->shape[0] == d,
             "tenstorrent kLayerNorm: weight/bias must be rank-1 [D]");
    VT_CHECK(IsFloatDType(p->dtype), "tenstorrent kLayerNorm: float weight/bias");
    VT_CHECK(p->IsContiguous(), "tenstorrent kLayerNorm: weight/bias must be contiguous");
  }

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = EnsureDevice2D(x, device);
  std::optional<ttnn::Tensor> dev_w;
  std::optional<ttnn::Tensor> dev_b;
  if (weight != nullptr) dev_w = EnsureAffine1D(*weight, d, device);
  if (bias != nullptr) dev_b = EnsureAffine1D(*bias, d, device);
  ttnn::Tensor dev_y = ttnn::layer_norm(dev_x, args.eps, dev_w, dev_b);
  CommitDevice2D(out, std::move(dev_y));
}

// kRmsNorm: per-row RMS over the last dim (cpu_ops.cpp RmsNormKernel). First
// Qwen3-dense (`Qwen3ForCausalLM`) op beyond OPT's LayerNorm set — Qwen3 uses
// RMSNorm for input/post-attn/final norms and per-head q/k norms. Weight is
// always present at the seam; optional residual is the residual stream
// (pre-norm sum written back, then normed), matching CPU residual round-trip
// for bf16 faithfulness. Gemma style (w+1) is host-only for now — Qwen3 does
// not set gemma=true.
//
// Device path: ttnn::rms_norm after residual merge (when any) and weight
// upload via the same TILE [1,D] affine helper as kLayerNorm.
void RmsNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& weight,
                   const RmsNormArgs& args, Tensor* residual) {
  TT_OP_TRACE("RmsNorm");
  VT_CHECK(x.rank == 2 && out.rank == 2,
           "tenstorrent kRmsNorm: only rank-2 tensors are supported in this step");
  VT_CHECK(IsFloatDType(x.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kRmsNorm: float in, f32/bf16 out");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1],
           "tenstorrent kRmsNorm: out shape mismatch");
  VT_CHECK(x.IsContiguous() && out.IsContiguous() && weight.IsContiguous(),
           "tenstorrent kRmsNorm: strided (non-contiguous) tensors are not supported");
  VT_CHECK(args.eps >= 0.0f, "tenstorrent kRmsNorm: eps must be non-negative");
  const uint32_t rows = static_cast<uint32_t>(x.shape[0]);
  const uint32_t d = static_cast<uint32_t>(x.shape[1]);
  VT_CHECK(weight.rank == 1 && weight.shape[0] == d && IsFloatDType(weight.dtype),
           "tenstorrent kRmsNorm: weight must be rank-1 float [D]");
  if (residual != nullptr) {
    VT_CHECK(residual->rank == 2 && residual->shape[0] == rows && residual->shape[1] == d,
             "tenstorrent kRmsNorm: residual shape must match x");
    VT_CHECK(IsFloatDType(residual->dtype) && residual->IsContiguous(),
             "tenstorrent kRmsNorm: residual must be contiguous float");
  }

  // Host path for gemma (w+1) and for tiny residual merges: short decode
  // (rows=1) pays more for device add+rms launches than a host loop, and was
  // a measurable e2e regression vs host residual.
  constexpr uint32_t kDeviceResidualMinRows = 32;
  // HOST-FREE-FORWARD R1: force the residual merge + RMS device path at T=1 when
  // capture is desired (ttnn trace prohibits host ops in the captured region).
  // Default ON since the R5 flip; VT_TT_HOST_FREE_DECODE=0 opts out (the
  // pre-flip default). Numerics proven by BACKEND-TENSTORRENT-RESIDUAL-GOLDEN.
  const bool host_free_decode = HostFreeDecodeEnabled();
  const bool host_residual = !host_free_decode &&
      (args.gemma || (residual != nullptr && rows < kDeviceResidualMinRows));
  if (host_residual) {
    EnsureHost(x);
    EnsureHost(weight);
    if (residual != nullptr) EnsureHost(*residual);
    for (int64_t r = 0; r < static_cast<int64_t>(rows); ++r) {
      float sumsq = 0.0f;
      for (int64_t j = 0; j < static_cast<int64_t>(d); ++j) {
        const int64_t idx = r * static_cast<int64_t>(d) + j;
        float v = LoadElemF32(x, idx);
        if (residual != nullptr) {
          v += LoadElemF32(*residual, idx);
          StoreElemF32(*residual, idx, v);
          v = LoadElemF32(*residual, idx);
        }
        sumsq += v * v;
      }
      const float inv =
          1.0f / std::sqrt(sumsq / static_cast<float>(d) + args.eps);
      for (int64_t j = 0; j < static_cast<int64_t>(d); ++j) {
        const int64_t idx = r * static_cast<int64_t>(d) + j;
        float v =
            residual != nullptr ? LoadElemF32(*residual, idx) : LoadElemF32(x, idx);
        float wj = LoadElemF32(weight, j);
        if (args.gemma) wj += 1.0f;
        StoreElemF32(out, idx, v * inv * wj);
      }
    }
    CommitHost(out);
    if (residual != nullptr) CommitHost(*residual);
    return;
  }

  // Device path: plain rms, or residual stream (x+residual → residual, then
  // rms) when rows are large enough that launches amortize.
  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = EnsureDevice2D(x, device);
  // Gemma (w+1) is baked host-side in the oracle's f32 order — the same
  // treatment as the fused preamble's weff — because ttnn::rms_norm applies
  // gamma raw. The baked upload is TRANSIENT and deliberately bypasses the
  // EnsureAffine1D slot cache: the cached form is the RAW weight, and a
  // non-gemma consumer of the same buffer must never read the +1 version.
  // Qwen3 never sets gemma; Qwen3.5 sets it on every norm (BACKEND-TENSTORRENT-
  // QWEN35 W2b: dropping the +1 here collapsed ambient prefill to `,`).
  ttnn::Tensor dev_w;
  if (args.gemma) {
    // Gemma (w+1) is baked host-side in the oracle's f32 order — the same
    // treatment as the fused preamble's weff — because ttnn::rms_norm applies
    // gamma raw. The baked form is cached in its own slot field, distinct
    // from EnsureAffine1D's raw BF16 form: a non-gemma consumer of the same
    // buffer must never read the +1 version. Qwen3 never sets gemma; Qwen3.5
    // sets it on every norm (BACKEND-TENSTORRENT-QWEN35 W2b: dropping the +1
    // here collapsed ambient prefill to `,`). The cache is what makes the
    // captured arm legal: the first (warmup) call stages the baked tensor,
    // and every later call — including every captured step — reuses it
    // instead of issuing the per-call host write that trace capture
    // forbids (#2812: the Qwen3.5-0.8B captured battery fatalled here,
    // fd_mesh_command_queue.cpp:760, 2/2 deterministic).
    bool have_w = false;
    {
      std::lock_guard<std::mutex> g(SlotMutex());
      BufferSlot* s = FindSlot(weight.data);
      if (s != nullptr && weight.data == s->host &&
          s->gemma_device.has_value()) {
        dev_w = *s->gemma_device;
        have_w = true;
      }
    }
    if (!have_w) {
      EnsureHost(weight);
      std::vector<float> gw(static_cast<size_t>(d));
      for (uint32_t i = 0; i < d; ++i)
        gw[static_cast<size_t>(i)] =
            LoadElemF32(weight, static_cast<int64_t>(i)) + 1.0f;
      if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
        std::fprintf(stderr,
                     "[TT-UP] RmsNorm gemma from_vector WRITE during capture\n");
      dev_w = ttnn::Tensor::from_vector<float>(
          std::move(gw),
          SpecOf(tt::tt_metal::Shape({1, d}), ttnn::DataType::FLOAT32,
                 ttnn::Layout::TILE),
          &device);
      std::lock_guard<std::mutex> g(SlotMutex());
      BufferSlot* s = FindSlot(weight.data);
      if (s != nullptr && weight.data == s->host)
        s->gemma_device = dev_w;
    }
  } else {
    dev_w = EnsureAffine1D(weight, d, device);
  }
  ttnn::Tensor to_norm = dev_x;
  if (residual != nullptr) {
    ttnn::Tensor dev_r = EnsureDevice2D(*residual, device);
    to_norm = ttnn::add(dev_x, dev_r);
    CommitDevice2D(*residual, to_norm);
  }
  ttnn::Tensor dev_y = ttnn::rms_norm(to_norm, args.eps, dev_w);
  CommitDevice2D(out, std::move(dev_y));
}

// kFusedChain: dispatch kFusedAddRmsNormStd to the same device RmsNorm path
// (residual += x; out = rms_norm(residual, weight)). Other recipes fall
// through to the CPU interpreter (host round-trip). Without this registration,
// FusedChain falls back to the CPU kernel which reads HOST memory — fatal
// when the PA output is device-resident (VT_TT_HOST_FREE_DECODE).
void FusedChainKernel(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                      Tensor* residual, const FusedRecipe& r, float eps) {
  TT_OP_TRACE("FusedChain");
  // kFusedAddRmsNormStd: step0 kAdd(residual = x + residual),
  // step1 kRmsNorm(out = rms_norm(residual, weight)).
  // This is exactly RmsNormKernel with the residual parameter.
  if (r.n == 2 &&
      r.steps[0].op == FOp::kAdd && r.steps[0].out == 2 &&
      r.steps[1].op == FOp::kRmsNorm && r.steps[1].out == 3 &&
      r.steps[1].gemma == false) {
    RmsNormKernel(q, out, x, weight, RmsNormArgs{eps, false}, residual);
    return;
  }
  // kFusedAddRmsNorm (gemma variant): same but gemma=true.
  if (r.n == 2 &&
      r.steps[0].op == FOp::kAdd && r.steps[0].out == 2 &&
      r.steps[1].op == FOp::kRmsNorm && r.steps[1].out == 3 &&
      r.steps[1].gemma == true) {
    RmsNormKernel(q, out, x, weight, RmsNormArgs{eps, true}, residual);
    return;
  }
  // Unknown recipe: fall back to host (safe outside capture).
  VT_CHECK(!tt_capture_active(),
           "tenstorrent: unknown FusedChain recipe during capture");
  EnsureHost(out);
  EnsureHost(x);
  EnsureHost(weight);
  if (residual != nullptr) EnsureHost(*residual);
  // Delegate to the CPU interpreter by calling the registered CPU op.
  auto cpu_fn = reinterpret_cast<FusedChainFn>(
      GetOpFallback(OpId::kFusedChain, DeviceType::kTENSTORRENT, "vt-tenstorrent"));
  if (cpu_fn) {
    cpu_fn(q, out, x, weight, residual, r, eps);
  } else {
    VT_CHECK(false, "tenstorrent: no FusedChain fallback available");
  }
}

// kSiluAndMul: SwiGLU gate half — out[i,j] = silu(x[i,j]) * x[i,j+d]
// with d = x.shape[1]/2 (cpu_ops.cpp SiluAndMulKernel). Second Qwen3-dense
// op beyond OPT (MLP: gate_up GEMM -> SiluAndMul -> down GEMM). Device path
// keeps the gate_up → SiluAndMul → down GEMM chain on-device: slice the
// last-dim halves, ttnn::silu(gate), ttnn::multiply by up. BF16 tile path
// (same envelope as matmul/norm); not bit-exact vs host f32.
void SiluAndMulKernel(Queue&, Tensor& out, const Tensor& x) {
  TT_OP_TRACE("SiluAndMul");
  VT_CHECK(x.rank == 2 && out.rank == 2,
           "tenstorrent kSiluAndMul: only rank-2 tensors are supported");
  VT_CHECK(IsFloatDType(x.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kSiluAndMul: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(),
           "tenstorrent kSiluAndMul: strided (non-contiguous) tensors are not supported");
  VT_CHECK(x.shape[1] % 2 == 0, "tenstorrent kSiluAndMul: last dim must be even");
  const int64_t t = x.shape[0];
  const int64_t d = x.shape[1] / 2;
  VT_CHECK(out.shape[0] == t && out.shape[1] == d,
           "tenstorrent kSiluAndMul: out shape must be [T, D] with D = x.shape[1]/2");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = EnsureDevice2D(x, device);
  const uint32_t tu = static_cast<uint32_t>(t);
  const uint32_t du = static_cast<uint32_t>(d);
  // x = [gate | up] along last dim.
  ttnn::Tensor gate = ttnn::slice(dev_x, ttsl::SmallVector<uint32_t>{0, 0},
                                  ttsl::SmallVector<uint32_t>{tu, du},
                                  ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor up = ttnn::slice(dev_x, ttsl::SmallVector<uint32_t>{0, du},
                                ttsl::SmallVector<uint32_t>{tu, 2 * du},
                                ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor silu_gate = ttnn::silu(gate);
  ttnn::Tensor dev_y = ttnn::multiply(silu_gate, up);
  CommitDevice2D(out, std::move(dev_y));
}

// kMoeSiluMul: silu(gate) * up with SPLIT operands (ops.cpp MoeSiluMul -> id 63;
// cpu_ops.cpp MoeSiluMulKernel, cuda_moe.cu MoeSiluMulKernel). The split sibling
// of kSiluAndMul above: the GGUF dense MLP arm reaches the DenseMlpBlock tail
// with gate/up as TWO separate [T,I] GEMM outputs (GGUF stores ff_gate/ff_up
// unmerged), so there is no merged [T,2I] operand to slice. Same ttnn math
// (ttnn::silu + ttnn::multiply) and the same capture-safe EnsureDevice2D /
// CommitDevice2D staging as kSiluAndMul, so the captured e2e treats it
// identically. The vehicle path feeds f32 gate/up, where the CPU kernel's
// RoundThrough(gate.dtype) narrowing is the identity.
void MoeSiluMulKernel(Queue&, Tensor& out, const Tensor& gate, const Tensor& up) {
  TT_OP_TRACE("MoeSiluMul");
  VT_CHECK(gate.rank == 2 && up.rank == 2 && out.rank == 2,
           "tenstorrent kMoeSiluMul: only rank-2 tensors are supported");
  VT_CHECK(IsFloatDType(gate.dtype) && IsFloatDType(up.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kMoeSiluMul: float in, f32/bf16 out");
  VT_CHECK(gate.IsContiguous() && up.IsContiguous() && out.IsContiguous(),
           "tenstorrent kMoeSiluMul: strided (non-contiguous) tensors are not supported");
  VT_CHECK(gate.shape[0] == out.shape[0] && gate.shape[1] == out.shape[1] &&
               up.shape[0] == gate.shape[0] && up.shape[1] == gate.shape[1],
           "tenstorrent kMoeSiluMul: gate, up and out must share one [T, I] shape");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_g = EnsureDevice2D(gate, device);
  ttnn::Tensor dev_u = EnsureDevice2D(up, device);
  ttnn::Tensor dev_y = ttnn::multiply(ttnn::silu(dev_g), dev_u);
  CommitDevice2D(out, std::move(dev_y));
}

// kCastBf16 / kCastF32: elementwise dtype convert via Load/Store (cpu_ops
// CastBf16Kernel / CastF32Kernel). Qwen3 uses these for K/V cache dtype and
// the logits / rope-cache paths. Host-staged; bit-exact for supported pairs.
// The device-pure staging below rides helpers defined later in this TU.

void CastBf16Kernel(Queue&, Tensor& out, const Tensor& in) {
  TT_OP_TRACE("CastBf16");
  VT_CHECK(out.dtype == DType::kBF16, "tenstorrent kCastBf16: out must be bf16");
  VT_CHECK(IsFloatDType(in.dtype), "tenstorrent kCastBf16: in must be float");
  VT_CHECK(out.Numel() == in.Numel(), "tenstorrent kCastBf16: numel mismatch");
  VT_CHECK(out.IsContiguous() && in.IsContiguous(),
           "tenstorrent kCastBf16: contiguous required");
  // Captured arm: the producer (fused preamble / GEMM) left `in`
  // device-authoritative, so the cast must run on the device shadow — a host
  // EnsureHost here would enqueue exactly the readback the trace forbids.
  // The device typecast is the same single RNE round the host StoreElemF32
  // makes (the SigmoidGateBf16 doctrine), so both arms stay byte-identical.
  const uint32_t n = static_cast<uint32_t>(in.Numel());
  ttnn::Tensor dev_in;
  if (ServeDeviceShadowRaw(in, 1, n, dev_in)) {
    ttnn::Tensor dev = NormalizeDevF32Tile(std::move(dev_in), 1, n);
    // Commit the SERVED geometry (#2282): a rank-2 shadow keeps its native
    // logical shape, and the record must name the geometry actually stored.
    const auto dls = dev.logical_shape();
    CommitDeviceLogical2D(out, ttnn::typecast(dev, ttnn::DataType::BFLOAT16),
                          dls[dls.rank() - 2], dls[dls.rank() - 1]);
    return;
  }
  {
    // The host fallback below is legal mid-capture ONLY while the bytes are
    // already host-current (EnsureHost would no-op). An untracked tensor is
    // host-only by construction. Anything else would need exactly the
    // readback a trace capture forbids, so refuse it loudly instead.
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* cs = FindSlot(in.data);
    VT_CHECK(!tt_capture_active() || cs == nullptr || cs->host_current,
             "tenstorrent kCastBf16: input arrived without a servable device "
             "shadow during trace capture and is not host-current — the "
             "readback the host path would need is what capture forbids");
  }
  EnsureHost(in);
  const int64_t ne = out.Numel();
  for (int64_t i = 0; i < ne; ++i) StoreElemF32(out, i, LoadElemF32(in, i));
  CommitHost(out);
}

void CastF32Kernel(Queue&, Tensor& out, const Tensor& in) {
  TT_OP_TRACE("CastF32");
  VT_CHECK(out.dtype == DType::kF32, "tenstorrent kCastF32: out must be f32");
  VT_CHECK(IsFloatDType(in.dtype), "tenstorrent kCastF32: in must be float");
  VT_CHECK(out.Numel() == in.Numel(), "tenstorrent kCastF32: numel mismatch");
  VT_CHECK(out.IsContiguous() && in.IsContiguous(),
           "tenstorrent kCastF32: contiguous required");
  // Captured arm, identical to kCastBf16 above: the producer (the lm_head
  // GEMM) left `in` device-authoritative, so the cast must run on the device
  // shadow — a host EnsureHost here would enqueue exactly the readback the
  // trace forbids. NormalizeDevF32Tile yields f32 TILE, which is the target
  // dtype, so no typecast remains; the ROW_MAJOR round-trip is a pure copy.
  const uint32_t nf = static_cast<uint32_t>(in.Numel());
  ttnn::Tensor dev_in;
  if (ServeDeviceShadowRaw(in, 1, nf, dev_in)) {
    ttnn::Tensor dev = NormalizeDevF32Tile(std::move(dev_in), 1, nf);
    // SERVED geometry, as in kCastBf16 above.
    const auto dls = dev.logical_shape();
    CommitDeviceLogical2D(out, std::move(dev), dls[dls.rank() - 2],
                          dls[dls.rank() - 1]);
    return;
  }
  {
    // Same host-current discipline as kCastBf16 above.
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* cs = FindSlot(in.data);
    VT_CHECK(!tt_capture_active() || cs == nullptr || cs->host_current,
             "tenstorrent kCastF32: input arrived without a servable device "
             "shadow during trace capture and is not host-current — the "
             "readback the host path would need is what capture forbids");
  }
  EnsureHost(in);
  const int64_t n = out.Numel();
  for (int64_t i = 0; i < n; ++i) StoreElemF32(out, i, LoadElemF32(in, i));
  CommitHost(out);
}

// kSigmoidGateBf16: out[i] = F32ToBF16(attn[i] * sigmoid(gate[i])) — the
// full-attention o_proj gate (cpu_ops.cpp SigmoidGateBf16Kernel; wrapper
// ops.cpp:4136-4153). The gate must NOT be rounded before the sigmoid
// (ops.cpp:4140: "sigmoid input must not be rounded"), so both operands ride
// FLOAT32 tile shadows and the product runs in f32; the single RNE round to
// bf16 happens in ttnn::typecast (the device fp32->fp16b cast: +0x7FFF+lsb
// then mask, ckernel_sfpu_typecast.h:246-262 — bit-identical to F32ToBF16;
// ttnn::to_dtype is HOST-ONLY at this pin, tensor_ops.cpp:533). The
// only TT-vs-CPU delta is the SFPU f32 sigmoid (accurate exp +
// reciprocal_iter<2> vs std::exp) — a few f32 ULP that can flip at most one
// bf16 rounding; the doctest envelope is one bf16 ULP.
// (UploadTensor / DeviceRows / ServePostConvAB / CachedTile / ServeActF32
// forward declarations removed by the stage-3 split: their definitions
// gained external linkage and tenstorrent_internal.h declarations, and a
// declaration left inside this anonymous namespace would name a distinct,
// never-defined internal entity that shadows them.)
void SigmoidGateBf16Kernel(Queue&, Tensor& out, const Tensor& attn,
                           const Tensor& gate) {
  TT_OP_TRACE("SigmoidGateBf16");
  VT_CHECK(out.dtype == DType::kBF16,
           "tenstorrent kSigmoidGateBf16: out must be bf16");
  VT_CHECK((attn.dtype == DType::kF32 || attn.dtype == DType::kBF16) &&
               gate.dtype == DType::kF32,
           "tenstorrent kSigmoidGateBf16: attn must be f32/bf16, gate f32");
  VT_CHECK(out.Numel() == attn.Numel() && out.Numel() == gate.Numel(),
           "tenstorrent kSigmoidGateBf16: out/attn/gate same element count");
  VT_CHECK(out.IsContiguous() && attn.IsContiguous() && gate.IsContiguous(),
           "tenstorrent kSigmoidGateBf16: contiguous required");
  const uint32_t n = static_cast<uint32_t>(out.Numel());
  MeshDevice& device = SharedMeshDevice();
  // UploadTensor FLOAT32 (not the bf16 EnsureDevice2D shadow): bf16 attn
  // upcasts exactly, and f32 attn keeps its full mantissa — a bf16 shadow
  // would pre-round the attn operand and widen the envelope (the f32 arm of
  // the doctest exists to catch exactly that).
  // Capture staging: in the captured arm attn arrives device-resident (the
  // PA sdpa commit) and the gate rides the same device-authoritative commit
  // path, so ServeActF32 serves those f32 TILE shadows and refuses under
  // capture when neither a shadow nor host bytes are readable — a ToHostF32
  // here would enqueue exactly the readback the trace forbids. Eager keeps
  // the host staging (ServeActF32's fallback is byte-identical to it).
  ttnn::Tensor dev_attn = ServeActF32(attn, 1, n, "kSigmoidGateBf16 attn", device);
  ttnn::Tensor dev_gate = ServeActF32(gate, 1, n, "kSigmoidGateBf16 gate", device);
  ttnn::Tensor sig = ttnn::sigmoid(dev_gate);
  ttnn::Tensor prod = ttnn::multiply(dev_attn, sig);
  ttnn::Tensor dev_y = ttnn::typecast(prod, ttnn::DataType::BFLOAT16);
  CommitDeviceLogical2D(out, std::move(dev_y), 1, n);
}

// Content-checked serve of the fused preamble's cos|sin table. The host bytes
// of `cos_sin` are fresh (the in-region RopeCosSinCacheKernel refill ran
// before this op), so equality against the warmed table proves the persistent
// device tensor holds THIS step's content. Returns false on miss.
static bool LookupAttnCSTable(const Tensor& cos_sin, uint32_t t, uint32_t rot,
                              ttnn::Tensor& out) {
  EnsureHost(cos_sin);
  const int64_t n = static_cast<int64_t>(t) * rot;
  std::vector<float> host(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    host[static_cast<size_t>(i)] = LoadElemF32(cos_sin, i);
  std::lock_guard<std::mutex> g(AttnCSMutex());
  auto it = AttnCSCache().find(AttnCSKey(t, rot));
  if (tt_capture_active() && std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr,
                 "[TT-TRACE] attn-cs lookup key=%ux%u found=%d eq=%d "
                 "want_first=%f have_first=%f\n",
                 t, rot, it != AttnCSCache().end() ? 1 : 0,
                 it != AttnCSCache().end() && it->second.cs_host == host ? 1 : 0,
                 host.empty() ? -1.0f : host.front(),
                 (it == AttnCSCache().end() || it->second.cs_host.empty())
                     ? -1.0f
                     : it->second.cs_host.front());
  if (it == AttnCSCache().end() || it->second.cs_host != host) return false;
  out = it->second.cs;
  return true;
}

// kAttnQkNormRopeGate: fused full-attention preamble = split q|gate +
// per-head gemma qk-RMSNorm + partial NeoX RoPE-from-cos_sin + exact gate
// passthrough, in ONE launch (cpu_ops.cpp AttnQkNormRopeGateKernel:1216-1270;
// wrapper ops.cpp:1638-1689; production call qwen3_5.cpp:5206-5224, gemma=true,
// rot < Dh). qgate/kf arrive as merged-QKV strided views (rows
// inner-contiguous); the split into per-head [T*H, Dh] rows happens in the
// HOST gather, and each leg is uploaded already in its final logical shape
// (no device slice/reshape: measured on the P150, first green attempt, a
// column slice + ttnn::reshape chain returned wrong data for the qgate legs
// while the full-width kf leg stayed correct — only a fresh contiguous
// upload is a safe reshape input at this pin). Norm+rope ride FLOAT32 tiles
// end to end (the SigmoidGateBf16 doctrine): rsqrt, the weight mix and the
// cos/sin rotation never round before the single output typecast, so an f32
// out sits within reduction-order ULPs of the scalar-f32 oracle and a bf16
// out is its exact RNE round. The gate leg is a plain device copy — no
// arithmetic touches it, so the f32 passthrough is bit-exact (the sigmoid
// input must not be rounded, ops.cpp:1660-1662/4140). The gemma weight w+1
// is computed host-side in f32, the same add GemmaNormElem does
// (cpu_ops.cpp:1204-1208).
void AttnQkNormRopeGateKernel(Queue&, Tensor& q_out, Tensor& k_out,
                              Tensor& gate_out, const Tensor& qgate,
                              const Tensor& kf, const Tensor& q_norm,
                              const Tensor& k_norm, const Tensor& cos_sin,
                              const RmsNormArgs& na, const RopeArgs& ra) {
  TT_OP_TRACE("AttnQkNormRopeGate");
  VT_CHECK((q_out.dtype == DType::kF32 || q_out.dtype == DType::kBF16) &&
               k_out.dtype == q_out.dtype &&
               (gate_out.dtype == q_out.dtype ||
                (q_out.dtype == DType::kBF16 && gate_out.dtype == DType::kF32)),
           "tenstorrent kAttnQkNormRopeGate: q/k/gate out f32 or bf16 "
           "(gate f32 allowed with bf16 q/k)");
  VT_CHECK(IsFloatDType(qgate.dtype) && kf.dtype == qgate.dtype,
           "tenstorrent kAttnQkNormRopeGate: qgate/kf float, same dtype");
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() &&
               gate_out.IsContiguous() && qgate.stride[1] == 1 &&
               qgate.stride[0] >= qgate.shape[1] && kf.stride[1] == 1 &&
               kf.stride[0] >= kf.shape[1] && q_norm.IsContiguous() &&
               k_norm.IsContiguous() && cos_sin.IsContiguous(),
           "tenstorrent kAttnQkNormRopeGate: contiguous required "
           "(qgate/kf row views excepted)");
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  const int64_t hkv = k_out.shape[1];
  const int64_t rot = ra.rotary_dim, half = rot / 2;
  const int64_t qrow = qgate.shape[1], krow = kf.shape[1];
  VT_CHECK(qrow == hq * 2 * dh && krow == hkv * dh,
           "tenstorrent kAttnQkNormRopeGate: qgate [T, Hq*2*Dh], kf [T, Hkv*Dh]");

  MeshDevice& device = SharedMeshDevice();
  // Host-gather the strided merged-QKV rows straight into per-head [T*H, Dh]
  // legs (the q|gate split is the host half of this fused op). The FLOAT32
  // upload also upcasts a bf16 input exactly (LoadElemF32), so every later
  // leg is dtype-uniform.
  // ---- device-pure staging ------------------------------------------------
  // qgate is the merged q|gate view: one [q(dh)|gate(dh)] block per head.
  // Serving the legs is pure data movement on the committed shadow — a
  // ROW_MAJOR materialize (the safe reshape input at this pin), the
  // [t*hq, 2*dh] head-row metadata view, then the two TILE column slices
  // (the dev_conv slice precedent). kf normalizes straight to its rows.
  // Every fallback below stays in the eager arm: an EnsureHost on a
  // committed activation is exactly the readback the trace refuses, and
  // the gathered re-uploads are capture-time writes.
  ttnn::Tensor dev_q, dev_gate, dev_k;
  {
    const uint32_t tw = static_cast<uint32_t>(t);
    const uint32_t qw = static_cast<uint32_t>(qgate.shape[1]);
    const uint32_t kw = static_cast<uint32_t>(kf.shape[1]);
    ttnn::Tensor qg_shadow, kf_shadow;
    const bool qg_ok = ServeDeviceShadowRaw(qgate, tw, qw, qg_shadow) ||
                       ServeDeviceWindow(qgate, tw, qw, qg_shadow);
    const bool kf_ok = ServeDeviceShadowRaw(kf, tw, kw, kf_shadow) ||
                       ServeDeviceWindow(kf, tw, kw, kf_shadow);
    if (qg_ok && kf_ok) {
      auto rm_f32 = [](ttnn::Tensor x, uint32_t r, uint32_t c) {
        x = ttnn::to_layout(x, ttnn::Layout::ROW_MAJOR);
        const auto ls = x.logical_shape();
        if (ls.rank() != 2 || ls[0] != r || ls[1] != c)
          x = ttnn::reshape(x, ttnn::Shape({r, c}));
        if (x.dtype() != ttnn::DataType::FLOAT32)
          x = ttnn::typecast(x, ttnn::DataType::FLOAT32);
        return x;
      };
      const uint32_t rows_q = static_cast<uint32_t>(t * hq);
      ttnn::Tensor qg = rm_f32(std::move(qg_shadow), tw, qw);
      qg = ttnn::reshape(qg, ttnn::Shape({rows_q, static_cast<uint32_t>(2 * dh)}));
      qg = ttnn::to_layout(qg, ttnn::Layout::TILE);
      dev_q = ttnn::slice(qg, ttsl::SmallVector<uint32_t>{0, 0},
                          ttsl::SmallVector<uint32_t>{rows_q,
                                                      static_cast<uint32_t>(dh)},
                          ttsl::SmallVector<uint32_t>{1, 1});
      dev_gate = ttnn::slice(qg, ttsl::SmallVector<uint32_t>{0,
                                                             static_cast<uint32_t>(dh)},
                             ttsl::SmallVector<uint32_t>{rows_q,
                                                         static_cast<uint32_t>(2 * dh)},
                             ttsl::SmallVector<uint32_t>{1, 1});
      dev_k = NormalizeDevF32Tile(std::move(kf_shadow),
                                  static_cast<uint32_t>(t * hkv),
                                  static_cast<uint32_t>(dh));
      // NormalizeDevF32Tile preserves the shadow's native [t, hkv*dh] geometry
      // (PR #3206 L1 fix). Reshape to the intended [t*hkv, dh] — safe on TILE
      // (view_device guard at tensor_ops.cpp:398: layout != ROW_MAJOR
      // short-circuits before the page_size rebuild that corrupts ROW_MAJOR).
      {
        const auto k_target = ttnn::Shape({static_cast<uint32_t>(t * hkv),
                                            static_cast<uint32_t>(dh)});
        if (dev_k.logical_shape() != k_target)
          dev_k = CaptureSafeReshape(dev_k, k_target);
      }
    } else {
      VT_CHECK(!tt_capture_active(),
               "tenstorrent kAttnQkNormRopeGate: qgate/kf arrived without a "
               "servable device shadow during trace capture — the producer "
               "must commit device-side before the captured region");
      EnsureHost(qgate);
      EnsureHost(kf);
      std::vector<float> qh(static_cast<size_t>(t * hq * dh)),
          gh(qh.size()), kh(static_cast<size_t>(t * hkv * dh));
      for (int64_t i = 0; i < t; ++i) {
        for (int64_t h = 0; h < hq; ++h) {
          const int64_t base = i * qgate.stride[0] + h * 2 * dh;
          const size_t dst = static_cast<size_t>((i * hq + h) * dh);
          for (int64_t j = 0; j < dh; ++j) {
            qh[dst + j] = LoadElemF32(qgate, base + j);
            gh[dst + j] = LoadElemF32(qgate, base + dh + j);
          }
        }
        for (int64_t h = 0; h < hkv; ++h) {
          const int64_t base = i * kf.stride[0] + h * dh;
          const size_t dst = static_cast<size_t>((i * hkv + h) * dh);
          for (int64_t j = 0; j < dh; ++j)
            kh[dst + j] = LoadElemF32(kf, base + j);
        }
      }
      dev_q = UploadTensor(std::move(qh),
                           ttnn::Shape({static_cast<uint32_t>(t * hq),
                                        static_cast<uint32_t>(dh)}),
                           ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
      dev_gate = UploadTensor(std::move(gh),
                              ttnn::Shape({static_cast<uint32_t>(t * hq),
                                           static_cast<uint32_t>(dh)}),
                              ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
      dev_k = UploadTensor(std::move(kh),
                           ttnn::Shape({static_cast<uint32_t>(t * hkv),
                                        static_cast<uint32_t>(dh)}),
                           ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
    }
  }
  // Gemma effective weight (host f32 add, oracle-identical): immutable
  // model constants, so the tiles ride the content-checked cache (the warm
  // step builds them; an in-region rebuild is refused by name).
  EnsureHost(q_norm);
  EnsureHost(k_norm);
  auto weff = [&](const Tensor& w) {
    std::vector<float> v(static_cast<size_t>(dh));
    for (int64_t j = 0; j < dh; ++j) {
      const float wj = LoadElemF32(w, j);
      v[static_cast<size_t>(j)] = na.gemma ? wj + 1.0f : wj;
    }
    return v;
  };
  const uint32_t du = static_cast<uint32_t>(dh);
  ttnn::Tensor dev_wq = CachedTile(q_norm.data, du, na.gemma ? 1 : 0, 0,
                                   [&] { return weff(q_norm); },
                                   ttnn::Shape({1, du}), device);
  ttnn::Tensor dev_wk = CachedTile(k_norm.data, du, na.gemma ? 1 : 0, 0,
                                   [&] { return weff(k_norm); },
                                   ttnn::Shape({1, du}), device);
  // cos_sin: the driver stages the step's table device-side in the captured
  // arm; the eager arm falls back to the host bytes. Both arms then share
  // the same device expansion: the [cos|sin] halves are TILE column slices,
  // and the per-(token,head) repeat is a dim-0 gather under the cached
  // full-width u32 index (the GdnDecode head-map precedent) — pure data
  // movement, bit-exact against the host expansion it replaces.
  ttnn::Tensor dev_cs;
  {
    ttnn::Tensor cs_shadow;
    if (ServeDeviceShadowRaw(cos_sin, static_cast<uint32_t>(t),
                             static_cast<uint32_t>(rot), cs_shadow) ||
        ServeDeviceWindow(cos_sin, static_cast<uint32_t>(t),
                          static_cast<uint32_t>(rot), cs_shadow)) {
      dev_cs = NormalizeDevF32Tile(std::move(cs_shadow),
                                   static_cast<uint32_t>(t),
                                   static_cast<uint32_t>(rot));
    } else {
      // Capture: serve the persistent per-step table the driver's
      // WarmAttnCosSin refreshes OUTSIDE capture (WarmRopeCosSin pattern).
      ttnn::Tensor cs_cached;
      if (LookupAttnCSTable(cos_sin, static_cast<uint32_t>(t),
                            static_cast<uint32_t>(rot), cs_cached)) {
        dev_cs = std::move(cs_cached);  // already f32 TILE [t, rot]
      } else {
        VT_CHECK(!tt_capture_active(),
                 "tenstorrent kAttnQkNormRopeGate: cos_sin arrived without a "
                 "servable device shadow during trace capture — the "
                 "decode-graph driver must call WarmAttnCosSin for the step's "
                 "positions BEFORE BeginCapture (the WarmRopeCosSin pattern)");
        EnsureHost(cos_sin);
        std::vector<float> cs(static_cast<size_t>(t) * rot);
        for (int64_t i = 0; i < t; ++i)
          for (int64_t j = 0; j < rot; ++j)
            cs[static_cast<size_t>(i) * rot + j] = LoadElemF32(cos_sin, i * rot + j);
        dev_cs = UploadTensor(std::move(cs),
                              ttnn::Shape({static_cast<uint32_t>(t),
                                           static_cast<uint32_t>(rot)}),
                              ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
      }
    }
  }
  const uint32_t halfu = static_cast<uint32_t>(half);
  const uint32_t rotu = static_cast<uint32_t>(rot);
  const uint32_t tu = static_cast<uint32_t>(t);
  ttnn::Tensor dev_cosh = ttnn::slice(
      dev_cs, ttsl::SmallVector<uint32_t>{0, 0},
      ttsl::SmallVector<uint32_t>{tu, halfu}, ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor dev_sinh = ttnn::slice(
      dev_cs, ttsl::SmallVector<uint32_t>{0, halfu},
      ttsl::SmallVector<uint32_t>{tu, rotu}, ttsl::SmallVector<uint32_t>{1, 1});
  // normed = x * rsqrt(mean(x^2)+eps) * (w+gemma); NeoX half-split rotation of
  // the leading rot cols; tail [rot, Dh) passes through NORMED (not rotated).
  auto norm_rope = [&](ttnn::Tensor x, const ttnn::Tensor& dev_w,
                       const ttnn::Tensor& dev_cos,
                       const ttnn::Tensor& dev_sin, int64_t nrows) {
    const uint32_t nr = static_cast<uint32_t>(nrows);
    ttnn::Tensor sq = ttnn::multiply(x, x);
    ttnn::Tensor s = ttnn::sum(sq, ttsl::SmallVector<int>{1}, true);
    ttnn::Tensor denom =
        ttnn::add(ttnn::multiply(s, 1.0f / static_cast<float>(dh)), na.eps);
    ttnn::Tensor inv = ttnn::rsqrt(denom);
    ttnn::Tensor normed = ttnn::multiply(ttnn::multiply(x, inv), dev_w);
    ttnn::Tensor x1 = ttnn::slice(normed, ttsl::SmallVector<uint32_t>{0, 0},
                                  ttsl::SmallVector<uint32_t>{nr, halfu},
                                  ttsl::SmallVector<uint32_t>{1, 1});
    ttnn::Tensor x2 = ttnn::slice(normed,
                                   ttsl::SmallVector<uint32_t>{0, halfu},
                                   ttsl::SmallVector<uint32_t>{nr, rotu},
                                   ttsl::SmallVector<uint32_t>{1, 1});
    ttnn::Tensor o1 = ttnn::subtract(ttnn::multiply(x1, dev_cos),
                                     ttnn::multiply(x2, dev_sin));
    ttnn::Tensor o2 = ttnn::add(ttnn::multiply(x1, dev_sin),
                                ttnn::multiply(x2, dev_cos));
    ttnn::Tensor out =
        ttnn::concat(std::vector<ttnn::Tensor>{o1, o2}, /*dim=*/1);
    if (rot < dh) {
      ttnn::Tensor tail =
          ttnn::slice(normed, ttsl::SmallVector<uint32_t>{0, rotu},
                      ttsl::SmallVector<uint32_t>{nr, du},
                      ttsl::SmallVector<uint32_t>{1, 1});
      out = ttnn::concat(std::vector<ttnn::Tensor>{out, tail}, /*dim=*/1);
    }
    return out;
  };
  // Gate leg FIRST (plain device copy of its own upload — no math, so the
  // passthrough stays bit-exact), then the q/k norm+rope legs.
  if (gate_out.dtype == DType::kBF16)
    dev_gate = ttnn::typecast(dev_gate, ttnn::DataType::BFLOAT16);
  CommitDeviceLogical2D(gate_out, std::move(dev_gate),
                        static_cast<uint32_t>(t * hq),
                        static_cast<uint32_t>(dh));
  ttnn::Tensor cos_q = ttnn::gather(dev_cosh, /*dim=*/0,
                                    CachedRepeatIdx(t, hq, half, device),
                                    /*sparse_grad=*/false, std::nullopt);
  ttnn::Tensor sin_q = ttnn::gather(dev_sinh, /*dim=*/0,
                                    CachedRepeatIdx(t, hq, half, device),
                                    /*sparse_grad=*/false, std::nullopt);
  ttnn::Tensor q_dev = norm_rope(std::move(dev_q), dev_wq, cos_q, sin_q,
                                 t * hq);
  if (q_out.dtype == DType::kBF16)
    q_dev = ttnn::typecast(q_dev, ttnn::DataType::BFLOAT16);
  CommitDeviceLogical2D(q_out, std::move(q_dev), static_cast<uint32_t>(t * hq),
                        static_cast<uint32_t>(dh));
  ttnn::Tensor cos_k = ttnn::gather(dev_cosh, /*dim=*/0,
                                    CachedRepeatIdx(t, hkv, half, device),
                                    /*sparse_grad=*/false, std::nullopt);
  ttnn::Tensor sin_k = ttnn::gather(dev_sinh, /*dim=*/0,
                                    CachedRepeatIdx(t, hkv, half, device),
                                    /*sparse_grad=*/false, std::nullopt);
  ttnn::Tensor k_dev =
      norm_rope(std::move(dev_k), dev_wk, cos_k, sin_k, t * hkv);
  if (k_out.dtype == DType::kBF16)
    k_dev = ttnn::typecast(k_dev, ttnn::DataType::BFLOAT16);
  CommitDeviceLogical2D(k_out, std::move(k_dev),
                        static_cast<uint32_t>(t * hkv),
                        static_cast<uint32_t>(dh));
}



// Device NeoX apply: view [T,H,D] as [T*H,D], rotate leading `rot` cols via
// slice + mul/sub/add + concat. Reuses EnsureDevice2D so a prior RmsNorm on the
// [T*H,D] view leaves the shadow resident (no re-upload). BF16 tile path.
void RopeApplyDeviceNeox(Tensor& x3, const float* cos_t, const float* sin_t,
                         int64_t tokens, int64_t heads, int64_t d, int64_t rot,
                         MeshDevice& device) {
  VT_CHECK(x3.rank == 3 && x3.IsContiguous() && x3.shape[0] == tokens &&
               x3.shape[1] == heads && x3.shape[2] == d,
           "tenstorrent device rope: rank-3 contiguous [T,H,D]");
  VT_CHECK(rot > 0 && (rot % 2) == 0 && rot <= d, "tenstorrent device rope: rotary_dim");
  const int64_t half = rot / 2;
  const int64_t th = tokens * heads;
  Tensor x_mat = x3.View({th, d});
  ttnn::Tensor dev_x = EnsureDevice2D(x_mat, device);

  std::vector<float> cos_exp, sin_exp;
  ExpandCosSinPerHead(cos_t, sin_t, tokens, heads, half, cos_exp, sin_exp);
  const uint32_t thu = static_cast<uint32_t>(th);
  const uint32_t halfu = static_cast<uint32_t>(half);
  const uint32_t rotu = static_cast<uint32_t>(rot);
  const uint32_t du = static_cast<uint32_t>(d);
  // ITEM 5: persistent cos/sin — build outside capture, copy in-region.
  const std::string rk = RopeCSKey(thu, halfu);
  ttnn::Tensor dev_cos, dev_sin;
  bool cache_hit = false;
  {
    std::lock_guard<std::mutex> g(RopeCSMutex());
    auto it = RopeCSCache().find(rk);
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
      std::fprintf(stderr,
                   "[TT-TRACE] rope lookup key=%s found=%d content_eq=%d "
                   "(want first=%f n=%zu)\n",
                   rk.c_str(), it != RopeCSCache().end(),
                   it != RopeCSCache().end() && it->second.cos_host == cos_exp,
                   cos_exp.empty() ? -1.0f : cos_exp.front(), cos_exp.size());
    if (it != RopeCSCache().end() && it->second.cos_host == cos_exp) {
      dev_cos = it->second.cos;
      dev_sin = it->second.sin;
      cache_hit = true;
    }
  }
  if (!cache_hit) {
    VT_CHECK(!tt_capture_active(),
             "tenstorrent: rope cos/sin cache miss during capture — the "
             "table changed (positions moved); the decode-graph driver must "
             "call WarmRopeCosSin for the step's positions BEFORE BeginCapture "
             "(the SizeSlot::Refresh pattern)");
    dev_cos = UploadRows(cos_exp.data(), thu, halfu, device);
    dev_sin = UploadRows(sin_exp.data(), thu, halfu, device);
    std::lock_guard<std::mutex> g(RopeCSMutex());
    RopeCSEntry e;
    e.cos = dev_cos;
    e.sin = dev_sin;
    e.cos_host = cos_exp;
    RopeCSCache()[rk] = std::move(e);
  }
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-TRACE] rope cos/sin cache %s during capture "
                 "(key th=%u half=%u first=%f)\n",
                 cache_hit ? "HIT" : "MISS", thu, halfu,
                 cos_exp.empty() ? -1.0f : cos_exp.front());

  // x1 = x[..., :half], x2h = x[..., half:rot]  (NeoX half-split)
  ttnn::Tensor x1 = ttnn::slice(dev_x, ttsl::SmallVector<uint32_t>{0, 0},
                                ttsl::SmallVector<uint32_t>{thu, halfu},
                                ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor x2h = ttnn::slice(dev_x, ttsl::SmallVector<uint32_t>{0, halfu},
                                 ttsl::SmallVector<uint32_t>{thu, rotu},
                                 ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor o1 = ttnn::subtract(ttnn::multiply(x1, dev_cos), ttnn::multiply(x2h, dev_sin));
  ttnn::Tensor o2 = ttnn::add(ttnn::multiply(x1, dev_sin), ttnn::multiply(x2h, dev_cos));
  ttnn::Tensor rotated = ttnn::concat(std::vector<ttnn::Tensor>{o1, o2}, /*dim=*/1);
  ttnn::Tensor out_dev;
  if (rot == d) {
    out_dev = std::move(rotated);
  } else {
    ttnn::Tensor tail = ttnn::slice(dev_x, ttsl::SmallVector<uint32_t>{0, rotu},
                                    ttsl::SmallVector<uint32_t>{thu, du},
                                    ttsl::SmallVector<uint32_t>{1, 1});
    out_dev = ttnn::concat(std::vector<ttnn::Tensor>{rotated, tail}, /*dim=*/1);
  }
  CommitDevice2D(x_mat, std::move(out_dev));
}

// Gather per-token cos|sin from a [P, rot] cache via rank-1 positions.
void GatherCosSinRows(const Tensor& cache, const Tensor& positions, int64_t tokens,
                      int rot, std::vector<float>& cos_t, std::vector<float>& sin_t) {
  EnsureHost(cache);
  EnsureHost(positions);
  const int64_t half = rot / 2;
  cos_t.resize(static_cast<size_t>(tokens * half));
  sin_t.resize(static_cast<size_t>(tokens * half));
  for (int64_t t = 0; t < tokens; ++t) {
    const int64_t position = positions.dtype == DType::kI32
                                 ? static_cast<int64_t>(positions.Ptr<int32_t>()[t])
                                 : positions.Ptr<int64_t>()[t];
    VT_CHECK(position >= 0 && position < cache.shape[0],
             "tenstorrent rope: position outside cache");
    const int64_t cache_off = position * rot;
    for (int64_t i = 0; i < half; ++i) {
      cos_t[static_cast<size_t>(t * half + i)] = LoadElemF32(cache, cache_off + i);
      sin_t[static_cast<size_t>(t * half + i)] =
          LoadElemF32(cache, cache_off + half + i);
    }
  }
}


// Host NeoX/GPT-J apply from a precomputed cos|sin table. Fast path for short
// decode: many tiny device launches (slice/mul/concat × q/k) lose to this.
void RopeApplyHost(Tensor& qs, Tensor* ks, const float* cos_t, const float* sin_t,
                   int64_t tokens, int64_t hq, int64_t hk, int64_t d, int rot,
                   bool is_neox) {
  EnsureHost(qs);
  if (ks != nullptr) EnsureHost(*ks);
  const int64_t half = rot / 2;
  auto apply_one = [&](Tensor& x, int64_t heads) {
    for (int64_t token = 0; token < tokens; ++token) {
      for (int64_t pair = 0; pair < half; ++pair) {
        const float c = cos_t[static_cast<size_t>(token * half + pair)];
        const float s = sin_t[static_cast<size_t>(token * half + pair)];
        const int64_t first = is_neox ? pair : pair * 2;
        const int64_t second = is_neox ? pair + half : pair * 2 + 1;
        for (int64_t head = 0; head < heads; ++head) {
          const int64_t off = (token * heads + head) * d;
          const float xv = LoadElemF32(x, off + first);
          const float yv = LoadElemF32(x, off + second);
          StoreElemF32(x, off + first, xv * c - yv * s);
          StoreElemF32(x, off + second, xv * s + yv * c);
        }
      }
    }
  };
  apply_one(qs, hq);
  if (ks != nullptr) apply_one(*ks, hk);
  CommitHost(qs);
  if (ks != nullptr) CommitHost(*ks);
}

// Prefer device apply only when T*H amortizes the slice/mul/concat launches.
// Short Qwen3 decode (T=1,H=16) is host-faster even when Q is already on device
// (measured regression when always-device-for-resident was forced).

inline bool PreferDeviceRope(int64_t tokens, int64_t heads) {
  // HOST-FREE-FORWARD R1: force device RoPE at T=1 for capture (see RmsNorm note).
  if (HostFreeDecodeEnabled()) return true;
  return tokens * heads >= 64;
}

// kRopeNeox: Qwen3-dense RoPE. Device NeoX for large [T*H]; host for short decode.
void RopeNeoxKernel(Queue&, Tensor& qs, Tensor& ks, const Tensor& pos, const RopeArgs& args) {
  TT_OP_TRACE("RopeNeox");
  VT_CHECK(qs.rank == 3 && ks.rank == 3, "tenstorrent kRopeNeox: qs/ks rank-3");
  VT_CHECK(IsFloatDType(qs.dtype) && qs.dtype == ks.dtype,
           "tenstorrent kRopeNeox: qs/ks float same dtype");
  VT_CHECK(pos.rank == 1 && (pos.dtype == DType::kI32 || pos.dtype == DType::kI64),
           "tenstorrent kRopeNeox: positions rank-1 i32/i64");
  VT_CHECK(qs.IsContiguous() && ks.IsContiguous() && pos.IsContiguous(),
           "tenstorrent kRopeNeox: contiguous required");
  VT_CHECK(args.rotary_dim > 0 && (args.rotary_dim % 2) == 0 &&
               args.rotary_dim <= qs.shape[2],
           "tenstorrent kRopeNeox: rotary_dim must be even and <= head_dim");
  const int64_t t = qs.shape[0], hq = qs.shape[1], hk = ks.shape[1], d = qs.shape[2];
  VT_CHECK(ks.shape[0] == t && ks.shape[2] == d, "tenstorrent kRopeNeox: ks shape");
  VT_CHECK(pos.shape[0] == t, "tenstorrent kRopeNeox: positions length");

  std::vector<float> cos_t, sin_t;
  BuildCosSinFromPositions(pos, t, args.rotary_dim, static_cast<double>(args.base), args, cos_t,
                           sin_t);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-TRACE] rope kernel pos0=%d t=%lld hq=%lld cos_first=%f\n",
                 (int)(pos.dtype == DType::kI32 ? pos.Ptr<int32_t>()[0]
                                                : static_cast<int32_t>(pos.Ptr<int64_t>()[0])),
                 (long long)t, (long long)hq,
                 cos_t.empty() ? -1.0f : cos_t.front());
  if (PreferDeviceRope(t, hq)) {
    MeshDevice& device = SharedMeshDevice();
    RopeApplyDeviceNeox(qs, cos_t.data(), sin_t.data(), t, hq, d, args.rotary_dim, device);
    RopeApplyDeviceNeox(ks, cos_t.data(), sin_t.data(), t, hk, d, args.rotary_dim, device);
  } else {
    RopeApplyHost(qs, &ks, cos_t.data(), sin_t.data(), t, hq, hk, d, args.rotary_dim,
                  /*is_neox=*/true);
  }
}

// kRopeCosSinCache: per-step cos|sin table [T, rot] (cpu_ops RopeCosSinCacheKernel).
// Stays host — table is small and built once per step; apply is device.
void RopeCosSinCacheKernel(Queue&, Tensor& cos_sin, const Tensor& positions,
                           const RopeArgs& args) {
  VT_CHECK(cos_sin.rank == 2 && cos_sin.dtype == DType::kF32 && cos_sin.IsContiguous(),
           "tenstorrent kRopeCosSinCache: cos_sin contiguous f32 [T,rot]");
  VT_CHECK(positions.rank == 1 &&
               (positions.dtype == DType::kI32 || positions.dtype == DType::kI64) &&
               positions.IsContiguous(),
           "tenstorrent kRopeCosSinCache: positions rank-1 i32/i64");
  VT_CHECK(args.rotary_dim > 0 && (args.rotary_dim % 2) == 0,
           "tenstorrent kRopeCosSinCache: rotary_dim even > 0");
  const int64_t t = cos_sin.shape[0];
  const int rot = args.rotary_dim;
  VT_CHECK(cos_sin.shape[1] == rot && positions.shape[0] == t,
           "tenstorrent kRopeCosSinCache: shape mismatch");
  EnsureHost(positions);
  const int64_t half = rot / 2;
  const double base = static_cast<double>(args.base);
  for (int64_t i = 0; i < t; ++i) {
    const int64_t p = positions.dtype == DType::kI32 ? positions.Ptr<int32_t>()[i]
                                                     : positions.Ptr<int64_t>()[i];
    for (int64_t pair = 0; pair < half; ++pair) {
      double freq =
          std::pow(base, -2.0 * static_cast<double>(pair) / static_cast<double>(rot));
      freq = Llama3ScaleFreq(freq, args);
      const double angle = static_cast<double>(p) * freq;
      StoreElemF32(cos_sin, i * rot + pair, static_cast<float>(std::cos(angle)));
      StoreElemF32(cos_sin, i * rot + half + pair, static_cast<float>(std::sin(angle)));
    }
  }
  CommitHost(cos_sin);
}

// kRopeFromCache: apply precomputed cos|sin (cpu_ops RopeFromCacheKernel).
// Rank-1 positions only (Qwen3-dense); mrope deferred. DEFAULT Qwen3 path
// (VT_QWEN3_ROPE_CACHE). Device NeoX when T*H is large; host for short decode
// and GPT-J interleave.
void RopeFromCacheKernel(Queue&, Tensor& qs, Tensor* ks, const Tensor& positions,
                         const Tensor& cache, const RopeArgs& args) {
  VT_CHECK(qs.rank == 3 && IsFloatDType(qs.dtype) && qs.IsContiguous(),
           "tenstorrent kRopeFromCache: qs rank-3 contiguous float");
  VT_CHECK(positions.rank == 1 &&
               (positions.dtype == DType::kI32 || positions.dtype == DType::kI64) &&
               positions.IsContiguous(),
           "tenstorrent kRopeFromCache: rank-1 positions only (no mrope yet)");
  VT_CHECK(cache.rank == 2 && IsFloatDType(cache.dtype) && cache.IsContiguous(),
           "tenstorrent kRopeFromCache: cache rank-2 contiguous float");
  VT_CHECK(args.rotary_dim > 0 && (args.rotary_dim % 2) == 0 &&
               args.rotary_dim <= qs.shape[2],
           "tenstorrent kRopeFromCache: rotary_dim");
  const int64_t tokens = qs.shape[0];
  const int64_t hq = qs.shape[1];
  const int64_t d = qs.shape[2];
  const int64_t hk = ks == nullptr ? 0 : ks->shape[1];
  if (ks != nullptr) {
    VT_CHECK(ks->rank == 3 && ks->dtype == qs.dtype && ks->IsContiguous(),
             "tenstorrent kRopeFromCache: ks must match qs");
    VT_CHECK(ks->shape[0] == tokens && ks->shape[2] == d,
             "tenstorrent kRopeFromCache: ks shape");
  }
  VT_CHECK(positions.shape[0] == tokens, "tenstorrent kRopeFromCache: positions length");

  std::vector<float> cos_t, sin_t;
  GatherCosSinRows(cache, positions, tokens, args.rotary_dim, cos_t, sin_t);

  if (args.is_neox_style && PreferDeviceRope(tokens, hq)) {
    MeshDevice& device = SharedMeshDevice();
    RopeApplyDeviceNeox(qs, cos_t.data(), sin_t.data(), tokens, hq, d, args.rotary_dim, device);
    if (ks != nullptr) {
      RopeApplyDeviceNeox(*ks, cos_t.data(), sin_t.data(), tokens, hk, d, args.rotary_dim,
                          device);
    }
    return;
  }
  RopeApplyHost(qs, ks, cos_t.data(), sin_t.data(), tokens, hq, hk, d, args.rotary_dim,
                args.is_neox_style);
}

// kQkvSplit: column split of merged [T, q+k+v] into q/k/v (cpu_ops QkvSplitKernel).
//
// Device path when qkv already has a resident shadow (post MatmulBT): slice the
// last dim on-device and CommitDevice2D each shard so qk-RmsNorm can reshape-
// reuse without download+reupload. Host path (bit-exact memcpy) when qkv is
// host-only — unit tests and weight-load style callers.
void QkvSplitKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& qkv) {
  TT_OP_TRACE("QkvSplit");
  VT_CHECK(qkv.rank == 2 && IsFloatDType(qkv.dtype),
           "tenstorrent kQkvSplit: rank-2 float qkv required");
  VT_CHECK(q_out.dtype == qkv.dtype && k_out.dtype == qkv.dtype && v_out.dtype == qkv.dtype,
           "tenstorrent kQkvSplit: q/k/v out must match qkv dtype");
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() && v_out.IsContiguous() &&
               qkv.IsContiguous(),
           "tenstorrent kQkvSplit: contiguous required");
  const int64_t t = qkv.shape[0];
  const int64_t q_dim = q_out.Numel() / t;
  const int64_t k_dim = k_out.Numel() / t;
  const int64_t v_dim = v_out.Numel() / t;
  const int64_t total = q_dim + k_dim + v_dim;
  VT_CHECK(qkv.shape[1] == total, "tenstorrent kQkvSplit: inner dim mismatch");
  VT_CHECK(q_out.rank == 2 && k_out.rank == 2 && v_out.rank == 2 &&
               q_out.shape[0] == t && k_out.shape[0] == t && v_out.shape[0] == t &&
               q_out.shape[1] == q_dim && k_out.shape[1] == k_dim && v_out.shape[1] == v_dim,
           "tenstorrent kQkvSplit: out shapes must be [T, *]");

  const uint32_t tu = static_cast<uint32_t>(t);
  const uint32_t total_u = static_cast<uint32_t>(total);
  const uint32_t qd = static_cast<uint32_t>(q_dim);
  const uint32_t kd = static_cast<uint32_t>(k_dim);
  const uint32_t vd = static_cast<uint32_t>(v_dim);

  if (DeviceShadowExact(qkv, tu, total_u)) {
    MeshDevice& device = SharedMeshDevice();
    ttnn::Tensor dev = EnsureDevice2D(qkv, device);
    ttnn::Tensor dq = ttnn::slice(dev, ttsl::SmallVector<uint32_t>{0, 0},
                                  ttsl::SmallVector<uint32_t>{tu, qd},
                                  ttsl::SmallVector<uint32_t>{1, 1});
    ttnn::Tensor dk = ttnn::slice(dev, ttsl::SmallVector<uint32_t>{0, qd},
                                  ttsl::SmallVector<uint32_t>{tu, qd + kd},
                                  ttsl::SmallVector<uint32_t>{1, 1});
    ttnn::Tensor dv = ttnn::slice(dev, ttsl::SmallVector<uint32_t>{0, qd + kd},
                                  ttsl::SmallVector<uint32_t>{tu, qd + kd + vd},
                                  ttsl::SmallVector<uint32_t>{1, 1});
    CommitDevice2D(q_out, std::move(dq));
    CommitDevice2D(k_out, std::move(dk));
    CommitDevice2D(v_out, std::move(dv));
    return;
  }

  EnsureHost(qkv);
  const size_t esz = SizeOf(qkv.dtype);
  const auto* src = static_cast<const uint8_t*>(qkv.data);
  auto* qdst = static_cast<uint8_t*>(q_out.data);
  auto* kdst = static_cast<uint8_t*>(k_out.data);
  auto* vdst = static_cast<uint8_t*>(v_out.data);
  for (int64_t i = 0; i < t; ++i) {
    const uint8_t* row = src + static_cast<size_t>(i * total) * esz;
    std::memcpy(qdst + static_cast<size_t>(i * q_dim) * esz, row,
                static_cast<size_t>(q_dim) * esz);
    std::memcpy(kdst + static_cast<size_t>(i * k_dim) * esz, row + static_cast<size_t>(q_dim) * esz,
                static_cast<size_t>(k_dim) * esz);
    std::memcpy(vdst + static_cast<size_t>(i * v_dim) * esz,
                row + static_cast<size_t>(q_dim + k_dim) * esz, static_cast<size_t>(v_dim) * esz);
  }
  CommitHost(q_out);
  CommitHost(k_out);
  CommitHost(v_out);
}

// kGreedyArgmax: per-row lowest-index max of f32 logits (cpu_sample.cpp).
// OPT's lm_head produces F32 logits; host-staged, bit-exact with CPU.
void GreedyArgmaxKernel(Queue&, Tensor& token_ids, const Tensor& logits) {
  VT_CHECK(logits.rank == 2 && logits.dtype == DType::kF32 && logits.IsContiguous(),
           "tenstorrent kGreedyArgmax: logits must be contiguous f32 [N,V]");
  VT_CHECK(token_ids.rank == 1 && token_ids.dtype == DType::kI64 && token_ids.IsContiguous() &&
               token_ids.shape[0] == logits.shape[0],
           "tenstorrent kGreedyArgmax: token_ids must be i64 [N]");
  EnsureHost(logits);
  const int64_t n = logits.shape[0], v = logits.shape[1];
  const float* lp = logits.Ptr<float>();
  int64_t* out = token_ids.Ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) {
    const float* row = lp + i * v;
    int64_t best = 0;
    float best_v = row[0];
    for (int64_t j = 1; j < v; ++j) {
      if (row[j] > best_v) {
        best_v = row[j];
        best = j;
      }
    }
    out[i] = best;
  }
  CommitHost(token_ids);
}

// ---- GDN prefill op set (BACKEND-TENSTORRENT-GDN W1) -------------------------
// kL2Norm / kRmsNormGated / kCausalConv1dFwd / kGdnPrefill: the op chain the
// Qwen3.5-family GDN layer issues in prefill. The CPU f32 arm (cpu_ops.cpp
// GdnPrefillKernel / CausalConv1dFwdKernel / L2NormKernel / RmsNormGatedKernel)
// is the correctness oracle; the tt-metal substrate is the implementation.
// Device-composed: kL2Norm (square → row-sum → rsqrt → scale, bf16 tiles),
// kRmsNormGated (ttnn::rms_norm + silu/sigmoid gate eltwise), kGdnPrefill
// (ttnn::transformer::chunk_gated_delta_rule behind a varlen→dense adapter).
// Host-staged in W1: kCausalConv1dFwd (the varlen window build + rolling
// conv_state writeback is pure data movement at these shapes; the conv-state
// device shadow that would make a composed path win is W2's decode work).
}  // namespace

namespace {
struct Registrar {
  Registrar() {
    if (!DeviceAvailable()) return;
    RegisterOp(OpId::kMatmul, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulKernel)));
    RegisterOp(OpId::kMatmulBT, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTKernel)));
    RegisterOp(OpId::kAdd, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<AddFn>(&AddKernel)));
    RegisterOp(OpId::kRelu, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<ReluFn>(&ReluKernel)));
    RegisterOp(OpId::kEmbedding, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingKernel)));
    RegisterOp(OpId::kKeepQuantDecode, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<KeepQuantDecodeFn>(&KeepQuantDecodeKernel)));
    RegisterOp(OpId::kMatmulBTQuant, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTQuantKernel)));
    RegisterOp(OpId::kMatmulBTQuantGrouped, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<MatmulBTQuantGroupedFn>(
                   &MatmulBTQuantGroupedKernel)));
    RegisterOp(OpId::kLayerNorm, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<LayerNormFn>(&LayerNormKernel)));
    RegisterOp(OpId::kRmsNorm, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernel)));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernel)));
    RegisterOp(OpId::kMoeSiluMul, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<MoeSiluMulFn>(&MoeSiluMulKernel)));
    RegisterOp(OpId::kCastBf16, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<CastBf16Fn>(&CastBf16Kernel)));
    RegisterOp(OpId::kCastF32, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<CastF32Fn>(&CastF32Kernel)));
    RegisterOp(OpId::kRopeNeox, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RopeFn>(&RopeNeoxKernel)));
    RegisterOp(OpId::kRopeCosSinCache, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RopeCosSinCacheFn>(&RopeCosSinCacheKernel)));
    RegisterOp(OpId::kRopeFromCache, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RopeFromCacheFn>(&RopeFromCacheKernel)));
    RegisterOp(OpId::kQkvSplit, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<QkvSplitFn>(&QkvSplitKernel)));
    RegisterOp(OpId::kReshapeAndCache, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<ReshapeAndCacheFn>(&ReshapeAndCacheKernel)));
    RegisterOp(OpId::kPagedAttention, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<PagedAttentionFn>(&PagedAttentionKernel)));
    RegisterOp(OpId::kGreedyArgmax, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<GreedyArgmaxFn>(&GreedyArgmaxKernel)));
    RegisterOp(OpId::kFusedChain, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<FusedChainFn>(&FusedChainKernel)));
    RegisterOp(OpId::kL2Norm, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<L2NormFn>(&L2NormKernel)));
    RegisterOp(OpId::kRmsNormGated, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RmsNormGatedFn>(&RmsNormGatedKernel)));
    RegisterOp(OpId::kCausalConv1dFwd, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<CausalConv1dFwdFn>(&CausalConv1dFwdKernel)));
  RegisterOp(OpId::kGdnPrefill, DeviceType::kTENSTORRENT,
             reinterpret_cast<void*>(static_cast<GdnPrefillFn>(&GdnPrefillKernel)));
    RegisterOp(OpId::kCausalConv1dUpdate, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<CausalConv1dUpdateFn>(&CausalConv1dUpdateKernel)));
    RegisterOp(OpId::kGdnDecode, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<GdnDecodeFn>(&GdnDecodeKernel)));
    RegisterOp(OpId::kGdnStateGather, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<GdnStateGatherFn>(&GdnStateGatherKernel)));
    RegisterOp(OpId::kGdnStateScatter, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(
                   static_cast<GdnStateScatterFn>(&GdnStateScatterKernel)));
    RegisterOp(OpId::kGdnPostConv, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<GdnPostConvFn>(&GdnPostConvKernel)));
    RegisterOp(OpId::kSigmoidGateBf16, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<SigmoidGateBf16Fn>(&SigmoidGateBf16Kernel)));
    RegisterOp(OpId::kAttnQkNormRopeGate, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(
                   static_cast<AttnQkNormRopeGateFn>(&AttnQkNormRopeGateKernel)));
  }
} registrar;

}  // namespace


}  // namespace vt::tenstorrent
