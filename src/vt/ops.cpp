// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include "vt/ops.h"
#include "vt/recipes.h"
#include "vt/paged_attn_route.h"  // W10 repair (#1865): the uniform-spec shape guard

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// CheckConvCommon asks the BACKEND whether it can address a compressed
// conv_state in place, rather than naming a device type.
#include "vt/backend.h"

namespace vt {

namespace {
// The op table itself moved to src/vt/op_provider.cpp — it is now the
// acceleration-provider registry (include/vt/op_provider.h,
// .agents/specs/metal-mlx-reuse-study.md §6), not a flat one-slot-per-device
// array with silent last-writer-wins. `RegisterOp`, `GetOp` and `OpRegistered`
// keep their exact signatures and semantics, so every one of the ~70 op wrappers
// below is UNCHANGED: the seam is inserted at `GetOp` and picked up by all of
// them at once with zero call-site edits.
bool IsFloat(DType d) { return d == DType::kF32 || d == DType::kF16 || d == DType::kBF16; }
bool IsOutFloat(DType d) { return d == DType::kF32 || d == DType::kBF16; }
// Retained model values are scoped to ordinary weights, never activations.
void ValidateWeightValues(const Tensor& weight, const Tensor& other,
                          const Tensor& out, const Queue& q) {
  VT_CHECK(!other.weight_value_dtype && !out.weight_value_dtype,
           "weight_value_dtype is invalid on activation, ID, or output operands");
  if (!weight.weight_value_dtype) return;
  VT_CHECK(weight.dtype == DType::kF16 &&
               (*weight.weight_value_dtype == DType::kBF16 ||
                *weight.weight_value_dtype == DType::kF32),
           "weight_value_dtype requires F16 storage with BF16 or F32 values");
  VT_CHECK(q.device.type == DeviceType::kROCM,
           "weight_value_dtype requires the native ROCm provider");
}

void* ResolveWeightOp(OpId op, const Queue& q, const Tensor& weight) {
  void* fn = GetOp(op, q.device.type);
  if (weight.weight_value_dtype) {
    const char* selected = GetOpProviderStats(op, q.device.type).last_selected;
    VT_CHECK(selected && std::strcmp(selected, kNativeProviderName) == 0,
             "weight_value_dtype requires the native ROCm provider");
  }
  return fn;
}
}  // namespace

ScalarTypeId ToScalarType(DType dtype) {
  switch (dtype) {
    case DType::kF32: return scalar_type::kF32;
    case DType::kF16: return scalar_type::kF16;
    case DType::kBF16: return scalar_type::kBF16;
    case DType::kI8: return scalar_type::kI8;
    case DType::kI32: return scalar_type::kI32;
    case DType::kI64: return scalar_type::kI64;
    // Block-quantized encodings have no single scalar type: a block mixes
    // scales and packed codes. Kernels consume them through the quant traits
    // table, never through a KernelTensorDesc scalar type.
    case DType::kQ4_0:
    case DType::kQ5_0:
    case DType::kQ8_0:
    case DType::kQ2_K:
    case DType::kQ3_K:
    case DType::kQ4_K:
    case DType::kQ5_K:
    case DType::kQ6_K:
    case DType::kQ8_K:
    case DType::kIQ2_XXS:
    case DType::kIQ3_XXS:
    case DType::kIQ2_S:
    case DType::kIQ1_S:
    case DType::kIQ1_XXXS:
    case DType::kIQ1_M:
    case DType::kIQ4_NL:
    case DType::kMXFP4:
    case DType::kIQ2_XS:
    case DType::kIQ4_XS:
    case DType::kIQ3_S:
    case DType::kTQ2_0:
    case DType::kTQ1_0:
      break;
  }
  VT_CHECK(false, "unsupported storage dtype for scalar-type conversion");
  return scalar_type::kF32;
}

KernelTensorDesc Describe(const Tensor& tensor, ScalarTypeId semantic_type,
                          KernelLayout layout) {
  VT_CHECK(tensor.rank >= 1 && tensor.rank <= kMaxRank,
           "kernel tensor descriptor rank out of range");
  VT_CHECK(tensor.data != nullptr, "kernel tensor descriptor requires non-null data");
  for (int d = 0; d < tensor.rank; ++d) {
    VT_CHECK(tensor.shape[d] > 0, "kernel tensor descriptor requires positive dimensions");
    VT_CHECK(tensor.stride[d] >= 0, "kernel tensor descriptor rejects negative strides");
  }

  switch (layout) {
    case KernelLayout::kStrided:
      VT_CHECK(semantic_type == ToScalarType(tensor.dtype),
               "strided layout semantic type must match its storage dtype");
      break;
    case KernelLayout::kPackedTwoFp4PerByte:
      VT_CHECK(tensor.dtype == DType::kI8 && semantic_type == scalar_type::kFE2M1f,
               "packed-two-fp4 layout requires i8 storage with explicit FE2M1 semantics");
      break;
    case KernelLayout::kBlockScaleLinear:
    case KernelLayout::kBlockScaleSwizzled:
      VT_CHECK(tensor.dtype == DType::kI8 &&
                   (semantic_type == scalar_type::kFE4M3fn ||
                    semantic_type == scalar_type::kFE8M0fnu),
               "block-scale layout requires i8 storage with explicit FP8 scale semantics");
      break;
    case KernelLayout::kMarlinInterleaved:
      VT_CHECK(tensor.dtype == DType::kI8 &&
                   (semantic_type == scalar_type::kFE2M1f ||
                    semantic_type == scalar_type::kI4 || semantic_type == scalar_type::kU4),
               "Marlin layout requires i8 storage with an explicit 4-bit semantic type");
      break;
  }

  KernelTensorDesc desc;
  desc.data = tensor.data;
  desc.storage_dtype = tensor.dtype;
  desc.scalar_type = semantic_type;
  desc.device = tensor.device;
  desc.rank = tensor.rank;
  desc.layout = layout;
  for (int d = 0; d < kMaxRank; ++d) {
    desc.shape[d] = tensor.shape[d];
    desc.stride[d] = tensor.stride[d];
  }
  return desc;
}

WorkspaceKey MakeWorkspaceKey(const Queue& q, OpId op, WorkspaceSlot slot) {
  VT_CHECK(q.id != 0, "workspace key requires a live queue identity");
  return WorkspaceKey{q.device, q.id, reinterpret_cast<uintptr_t>(q.handle), op, slot};
}

void Matmul(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  ValidateWeightValues(b, a, out, q);
  VT_CHECK(a.rank == 2 && b.rank == 2 && out.rank == 2, "matmul: rank-2 tensors required");
  VT_CHECK(a.shape[1] == b.shape[0], "matmul: inner dims mismatch");
  VT_CHECK(out.shape[0] == a.shape[0] && out.shape[1] == b.shape[1],
           "matmul: output shape mismatch");
  VT_CHECK(IsFloat(a.dtype) && IsFloat(b.dtype) && IsOutFloat(out.dtype),
           "matmul: float inputs and f32/bf16 output required");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "matmul: contiguous tensors required");
  VT_CHECK(a.device == b.device && a.device == out.device && a.device == q.device,
           "matmul: device mismatch");
  reinterpret_cast<MatmulFn>(ResolveWeightOp(OpId::kMatmul, q, b))(q, out, a, b);
}

void DropinProbe(Queue& q, Tensor& out, const Tensor& in,
                 const DropinProbeArgs& args) {
  VT_CHECK(q.id != 0, "dropin_probe: live queue required");
  VT_CHECK(in.rank == 2 && out.rank == 2, "dropin_probe: rank-2 tensors required");
  VT_CHECK(in.shape[0] == out.shape[0] && in.shape[1] == out.shape[1],
           "dropin_probe: input/output shape mismatch");
  VT_CHECK(in.device == q.device && out.device == q.device,
           "dropin_probe: input/output/queue device mismatch");
  VT_CHECK(args.workspace_slot != args.scalar_slot,
           "dropin_probe: workspace and scalar slots must not alias");
  VT_CHECK(args.workspace_bytes >= sizeof(uint32_t),
           "dropin_probe: workspace must hold the raw-launch marker");
  (void)Describe(in, args.scalar_type, args.layout);
  (void)Describe(out, args.scalar_type, args.layout);
  VT_CHECK(args.scalar_type == scalar_type::kF32 && args.layout == KernelLayout::kStrided,
           "dropin_probe: W0 raw kernel supports f32 strided tensors only");
  GetTypedOp<DropinProbeFn>(OpId::kDropinProbe, q.device.type)(q, out, in, args);
}

void MatmulBT(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  ValidateWeightValues(b, a, out, q);
  // GGUF compute-in-quant (QUANT-GGUF-CIQ-GEMM work row G4). A block-quantized
  // weight is NOT an elementwise tensor — it has no per-element stride and
  // cannot be read by kMatmulBT — but it IS in exactly the [N, K] orientation
  // this entry point defines, which is ggml's src0 layout and GGUF's disk
  // order (ggml-cpu.c:1245-1443). Dispatching it to kMatmulBTQuant here is
  // what routes the MODEL: every matmul helper already sends an `nk == true`
  // weight to MatmulBT (qwen3_5.cpp:1067,1078 device-in/out and :743,760
  // host), so the keep-quant loader's block-typed OwnedTensor reaches the
  // quantized GEMM with no signature, call-site or forward change. Elementwise
  // weights fall through to the unchanged validation + kMatmulBT below, so
  // every safetensors path is bit-identical by construction.
  if (IsBlockQuant(b.dtype)) {
    MatmulBTQuant(q, out, a, b);
    return;
  }
  VT_CHECK(a.rank == 2 && b.rank == 2 && out.rank == 2, "matmul_bt: rank-2 tensors required");
  VT_CHECK(a.shape[1] == b.shape[1], "matmul_bt: inner dims mismatch (b is [N,K])");
  VT_CHECK(out.shape[0] == a.shape[0] && out.shape[1] == b.shape[0],
           "matmul_bt: output shape mismatch");
  VT_CHECK(IsFloat(a.dtype) && IsFloat(b.dtype) && IsOutFloat(out.dtype),
           "matmul_bt: float inputs and f32/bf16 output required");
  // The ACTIVATION may be ROW-STRIDED (relaxed at MLA campaign W6): upstream's
  // `kv_b_proj(kv_c)` inside `_compute_prefill_context`
  // (mla_attention.py:2141-2160) is applied to a COLUMN SLICE of the 576-wide
  // chunked-prefill workspace, i.e. a torch view whose row stride is 576 while
  // K is 512 — and `F.linear` accepts exactly that. Only the innermost dim must
  // be packed; for a CONTIGUOUS activation the row stride IS K, so every
  // existing caller passes byte-identical arguments (same cuBLASLt ld, so the
  // same algo, so bit-identical results).
  VT_CHECK(a.stride[1] == 1 && a.stride[0] >= a.shape[1],
           "matmul_bt: activation rows must be packed (innermost stride 1) and "
           "non-overlapping");
  VT_CHECK(b.IsContiguous() && out.IsContiguous(),
           "matmul_bt: contiguous weight and output required");
  VT_CHECK(a.device == b.device && a.device == out.device && a.device == q.device,
           "matmul_bt: device mismatch");
  reinterpret_cast<MatmulFn>(ResolveWeightOp(OpId::kMatmulBT, q, b))(q, out, a, b);
}

// vt::MatmulBTQuant — see ops.h. Validation mirrors MatmulBT except for the
// weight, whose block layout replaces the elementwise stride contract.
void MatmulBTQuant(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  VT_CHECK(a.rank == 2 && b.rank == 2 && out.rank == 2,
           "matmul_bt_quant: rank-2 tensors required");
  VT_CHECK(a.shape[1] == b.shape[1],
           "matmul_bt_quant: inner dims mismatch (b is [N,K])");
  VT_CHECK(out.shape[0] == a.shape[0] && out.shape[1] == b.shape[0],
           "matmul_bt_quant: output shape mismatch");
  VT_CHECK(IsBlockQuant(b.dtype),
           "matmul_bt_quant: weight must be a block-quantized dtype (use "
           "MatmulBT for elementwise weights)");
  VT_CHECK(IsFloat(a.dtype) && IsOutFloat(out.dtype),
           "matmul_bt_quant: float activation and f32/bf16 output required");
  // ggml_row_size asserts the row is whole blocks; a GEMM weight whose K is
  // not block-aligned is not keep-quant eligible in the first place.
  VT_CHECK(b.shape[1] % BlockElems(b.dtype) == 0,
           "matmul_bt_quant: K must be a whole number of weight blocks");
  // Same relaxed activation contract as MatmulBT: only the innermost dim must
  // be packed, so a column slice of a wider workspace is consumed as-is.
  VT_CHECK(a.stride[1] == 1 && a.stride[0] >= a.shape[1],
           "matmul_bt_quant: activation rows must be packed (innermost stride "
           "1) and non-overlapping");
  VT_CHECK(out.IsContiguous(), "matmul_bt_quant: contiguous output required");
  VT_CHECK(a.device == b.device && a.device == out.device && a.device == q.device,
           "matmul_bt_quant: device mismatch");
  reinterpret_cast<MatmulFn>(GetOp(OpId::kMatmulBTQuant, q.device.type))(q, out, a, b);
}

// vt::MatmulBTQuantGrouped — see ops.h. out[P,N], act[P,K], weight[E*N,K]
// block-quant, expert_ids[P] i32. The per-group weight row-block is selected by
// expert_ids[p]; validation mirrors MatmulBTQuant plus the expert-index contract.
void MatmulBTQuantGrouped(Queue& q, Tensor& out, const Tensor& act,
                          const Tensor& weight, const Tensor& expert_ids) {
  VT_CHECK(out.rank == 2 && act.rank == 2 && weight.rank == 2,
           "matmul_bt_quant_grouped: rank-2 out/act/weight required");
  const int64_t P = out.shape[0], N = out.shape[1], K = act.shape[1];
  VT_CHECK(act.shape[0] == P || act.shape[0] == 1,
           "matmul_bt_quant_grouped: act rows must be P (per-expert) or 1 (broadcast)");
  VT_CHECK(weight.shape[1] == K, "matmul_bt_quant_grouped: weight K mismatch (b is [E*N,K])");
  VT_CHECK(weight.shape[0] % N == 0,
           "matmul_bt_quant_grouped: weight rows must be a whole multiple of N");
  VT_CHECK(IsBlockQuant(weight.dtype),
           "matmul_bt_quant_grouped: weight must be a block-quantized dtype");
  VT_CHECK(IsFloat(act.dtype) && IsOutFloat(out.dtype),
           "matmul_bt_quant_grouped: float activation and f32/bf16 output required");
  VT_CHECK(weight.shape[1] % BlockElems(weight.dtype) == 0,
           "matmul_bt_quant_grouped: K must be a whole number of weight blocks");
  VT_CHECK(expert_ids.Numel() == P && expert_ids.dtype == DType::kI32,
           "matmul_bt_quant_grouped: expert_ids must be i32 [P]");
  VT_CHECK(act.stride[1] == 1 && act.stride[0] >= K,
           "matmul_bt_quant_grouped: activation rows must be packed (innermost stride 1)");
  VT_CHECK(out.IsContiguous() && expert_ids.IsContiguous(),
           "matmul_bt_quant_grouped: contiguous out + expert_ids required");
  VT_CHECK(act.device == q.device && weight.device == q.device &&
               out.device == q.device && expert_ids.device == q.device,
           "matmul_bt_quant_grouped: device mismatch");
  reinterpret_cast<MatmulBTQuantGroupedFn>(GetOp(OpId::kMatmulBTQuantGrouped, q.device.type))(
      q, out, act, weight, expert_ids);
}

// vt::MoeGateUpSwiGLUGrouped — see ops.h. out[P,N] f32, act[Pa,K] (Pa==1 broadcast),
// gate_w/up_w[E*N,K] SAME block-quant dtype, expert_ids[P] i32, float limit. Validation
// mirrors MatmulBTQuantGrouped for BOTH weight towers plus the same-dtype/f32-out
// contract the fused epilogue requires.
void MoeGateUpSwiGLUGrouped(Queue& q, Tensor& out, const Tensor& act, const Tensor& gate_w,
                            const Tensor& up_w, const Tensor& expert_ids, float limit) {
  VT_CHECK(out.rank == 2 && act.rank == 2 && gate_w.rank == 2 && up_w.rank == 2,
           "moe_gate_up_swiglu: rank-2 out/act/gate_w/up_w required");
  const int64_t P = out.shape[0], N = out.shape[1], K = act.shape[1];
  VT_CHECK(act.shape[0] == P || act.shape[0] == 1,
           "moe_gate_up_swiglu: act rows must be P (per-expert) or 1 (broadcast)");
  VT_CHECK(gate_w.shape[1] == K && up_w.shape[1] == K,
           "moe_gate_up_swiglu: gate_w/up_w K mismatch (both are [E*N,K])");
  VT_CHECK(gate_w.shape[0] % N == 0 && up_w.shape[0] == gate_w.shape[0],
           "moe_gate_up_swiglu: gate_w/up_w rows must be a whole multiple of N and equal");
  VT_CHECK(IsBlockQuant(gate_w.dtype) && gate_w.dtype == up_w.dtype,
           "moe_gate_up_swiglu: gate_w/up_w must be the SAME block-quantized dtype");
  VT_CHECK(IsFloat(act.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "moe_gate_up_swiglu: float activation and f32 output required");
  VT_CHECK(gate_w.shape[1] % BlockElems(gate_w.dtype) == 0,
           "moe_gate_up_swiglu: K must be a whole number of weight blocks");
  VT_CHECK(expert_ids.Numel() == P && expert_ids.dtype == DType::kI32,
           "moe_gate_up_swiglu: expert_ids must be i32 [P]");
  VT_CHECK(act.stride[1] == 1 && act.stride[0] >= K,
           "moe_gate_up_swiglu: activation rows must be packed (innermost stride 1)");
  VT_CHECK(out.IsContiguous() && expert_ids.IsContiguous(),
           "moe_gate_up_swiglu: contiguous out + expert_ids required");
  VT_CHECK(act.device == q.device && gate_w.device == q.device && up_w.device == q.device &&
               out.device == q.device && expert_ids.device == q.device,
           "moe_gate_up_swiglu: device mismatch");
  reinterpret_cast<MoeGateUpSwiGLUGroupedFn>(
      GetOp(OpId::kMoeGateUpSwiGLUGrouped, q.device.type))(q, out, act, gate_w, up_w, expert_ids,
                                                           limit);
}

// vt::BatchedMatmul — `torch.bmm` (mla_attention.py:789 q-side W_UK absorption,
// :1034 W_UV v-up-projection). Stride-driven on every operand because BOTH
// upstream call sites pass transposed views; only the innermost dim must be
// unit-stride.
void BatchedMatmul(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  VT_CHECK(a.rank == 3 && b.rank == 3 && out.rank == 3,
           "batched_matmul: rank-3 tensors required (out[G,M,N] = a[G,M,K] @ b[G,K,N])");
  VT_CHECK(a.shape[0] == b.shape[0] && a.shape[0] == out.shape[0],
           "batched_matmul: batch dim mismatch");
  VT_CHECK(a.shape[2] == b.shape[1], "batched_matmul: inner dims mismatch");
  VT_CHECK(out.shape[1] == a.shape[1] && out.shape[2] == b.shape[2],
           "batched_matmul: output shape mismatch");
  VT_CHECK(IsFloat(a.dtype) && IsFloat(b.dtype) && IsOutFloat(out.dtype),
           "batched_matmul: float inputs and f32/bf16 output required");
  VT_CHECK(a.dtype == b.dtype, "batched_matmul: a/b dtype must match");
  // Only the innermost stride is constrained; the batch/row strides are free so
  // a transposed view (upstream's `q_nope.transpose(0,1)` / `out.transpose(0,1)`)
  // is consumed without a copy.
  VT_CHECK(a.stride[2] == 1 && b.stride[2] == 1 && out.stride[2] == 1,
           "batched_matmul: innermost dimension must be unit-stride");
  VT_CHECK(a.stride[1] >= a.shape[2] && b.stride[1] >= b.shape[2] &&
               out.stride[1] >= out.shape[2],
           "batched_matmul: row stride must not overlap the next row");
  VT_CHECK(a.device == b.device && a.device == out.device && a.device == q.device,
           "batched_matmul: device mismatch");
  GetTypedOp<BatchedMatmulFn>(OpId::kBatchedMatmul, q.device.type)(q, out, a, b);
}

// vt::ConcatMlaNopeRope — the generalization of upstream's two MLA head-concat
// sites: `concat_mla_q` (cache_kernels.cu:1555-1600, the decode 512+64 query)
// and `_concat_k_nope_k_pe` (mla_attention.py:2063-2092, the prefill 128+64 key
// with a head-BROADCAST rope part). Stride checks mirror
// cache_kernels.cu:1572-1577.
void ConcatMlaNopeRope(Queue& q, Tensor& out, const Tensor& nope, const Tensor& rope) {
  VT_CHECK(out.rank == 3 && nope.rank == 3 && rope.rank == 3,
           "concat_mla_nope_rope: rank-3 [tokens, heads, dim] tensors required");
  const int64_t tokens = out.shape[0], heads = out.shape[1];
  const int64_t dn = nope.shape[2], dr = rope.shape[2];
  VT_CHECK(nope.shape[0] == tokens && rope.shape[0] == tokens,
           "concat_mla_nope_rope: token count mismatch");
  VT_CHECK(nope.shape[1] == heads, "concat_mla_nope_rope: nope head count mismatch");
  VT_CHECK(rope.shape[1] == heads || rope.shape[1] == 1,
           "concat_mla_nope_rope: rope must carry `heads` heads or exactly 1 "
           "(the single shared k_pe head, broadcast — mla_attention.py:2063-2092)");
  VT_CHECK(out.shape[2] == dn + dr,
           "concat_mla_nope_rope: out last dim must be nope_dim + rope_dim");
  // The NoPE case (GLM-5.3-Flash, W3, #2213): `qk_rope_head_dim == 0` means the
  // decoupled-rope slice does not exist, so the "concat" is the nope part alone.
  // Both kernels already do exactly that — their rope loop runs zero times — so
  // this is the wrapper admitting a shape the implementations always handled.
  // `dn == 0` stays refused: a concat with no nope part has no upstream form.
  VT_CHECK(dn > 0, "concat_mla_nope_rope: the nope part must be non-empty");
  VT_CHECK(dr >= 0, "concat_mla_nope_rope: the rope width must be >= 0");
  VT_CHECK(out.dtype == nope.dtype && out.dtype == rope.dtype,
           "concat_mla_nope_rope: all tensors must share one dtype");
  VT_CHECK(IsOutFloat(out.dtype) || out.dtype == DType::kF16,
           "concat_mla_nope_rope: f32/bf16/f16 only");
  VT_CHECK(out.stride[2] == 1 && nope.stride[2] == 1 && rope.stride[2] == 1,
           "concat_mla_nope_rope: innermost dimension must be unit-stride "
           "(upstream cache_kernels.cu:1572-1577)");
  VT_CHECK(out.device == q.device && nope.device == q.device && rope.device == q.device,
           "concat_mla_nope_rope: device mismatch");
  if (tokens == 0 || heads == 0) return;  // `if (num_tokens == 0) return;` (:1584)
  GetTypedOp<ConcatMlaNopeRopeFn>(OpId::kConcatMlaNopeRope, q.device.type)(q, out, nope, rope);
}

void MatmulNvfp4(Queue& q, Tensor& out, const Tensor& act, const Tensor& weight_packed,
                 const Tensor& weight_scale, float weight_scale_2) {
  VT_CHECK(act.rank == 2 && weight_packed.rank == 2 && weight_scale.rank == 2 && out.rank == 2,
           "matmul_nvfp4: act/weight_packed/weight_scale/out must be rank-2");
  const int64_t m = act.shape[0], k = act.shape[1], n = weight_packed.shape[0];
  VT_CHECK(k % 16 == 0, "matmul_nvfp4: K (act inner dim) must be a multiple of 16");
  VT_CHECK(weight_packed.shape[1] == k / 2,
           "matmul_nvfp4: weight_packed must be [N, K/2] (two fp4 codes per byte)");
  VT_CHECK(weight_scale.shape[0] == n && weight_scale.shape[1] == k / 16,
           "matmul_nvfp4: weight_scale must be [N, K/16] (one fp8 scale per 16-elem group)");
  VT_CHECK(out.shape[0] == m && out.shape[1] == n, "matmul_nvfp4: out must be [M, N]");
  VT_CHECK(IsFloat(act.dtype) && IsOutFloat(out.dtype),
           "matmul_nvfp4: float act, f32/bf16 out");
  VT_CHECK(weight_packed.dtype == DType::kI8 && weight_scale.dtype == DType::kI8,
           "matmul_nvfp4: weight_packed/weight_scale must be i8 (raw fp4/fp8 bytes)");
  VT_CHECK(act.IsContiguous() && weight_packed.IsContiguous() && weight_scale.IsContiguous() &&
               out.IsContiguous(),
           "matmul_nvfp4: contiguous tensors required");
  VT_CHECK(act.device == q.device && weight_packed.device == q.device &&
               weight_scale.device == q.device && out.device == q.device,
           "matmul_nvfp4: device mismatch (act/weight_packed/weight_scale/out/queue)");
  reinterpret_cast<MatmulNvfp4Fn>(GetOp(OpId::kMatmulNvfp4, q.device.type))(
      q, out, act, weight_packed, weight_scale, weight_scale_2);
}

void ScaledFp4Quant(Queue& q, Tensor& out_packed, Tensor& out_scale, const Tensor& x,
                    float input_global_scale_inv, Fp4ScaleLayout scale_layout) {
  VT_CHECK(x.rank == 2 && out_packed.rank == 2 && out_scale.rank == 2,
           "scaled_fp4_quant: x/out_packed/out_scale must be rank-2");
  const int64_t m = x.shape[0], k = x.shape[1];
  VT_CHECK(k % 16 == 0, "scaled_fp4_quant: K (inner dim) must be a multiple of 16");
  VT_CHECK(out_packed.shape[0] == m && out_packed.shape[1] == k / 2,
           "scaled_fp4_quant: out_packed must be [M, K/2]");
  const auto round_up = [](int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
  };
  if (scale_layout == Fp4ScaleLayout::kLinear) {
    VT_CHECK(out_scale.shape[0] == m && out_scale.shape[1] == k / 16,
             "scaled_fp4_quant: linear out_scale must be [M, K/16]");
  } else {
    VT_CHECK(scale_layout == Fp4ScaleLayout::kCutlassSwizzled,
             "scaled_fp4_quant: invalid scale layout");
    VT_CHECK(out_scale.shape[0] == round_up(m, 128) &&
                 out_scale.shape[1] == round_up(k / 16, 4),
             "scaled_fp4_quant: swizzled out_scale must be "
             "[round_up(M,128), round_up(K/16,4)]");
  }
  VT_CHECK(IsFloat(x.dtype), "scaled_fp4_quant: float x required");
  VT_CHECK(out_packed.dtype == DType::kI8 && out_scale.dtype == DType::kI8,
           "scaled_fp4_quant: out_packed/out_scale must be i8 (raw fp4/fp8 bytes)");
  VT_CHECK(x.IsContiguous() && out_packed.IsContiguous() && out_scale.IsContiguous(),
           "scaled_fp4_quant: contiguous tensors required");
  VT_CHECK(x.device == q.device && out_packed.device == q.device && out_scale.device == q.device,
           "scaled_fp4_quant: device mismatch (x/out_packed/out_scale/queue)");
  reinterpret_cast<ScaledFp4QuantFn>(GetOp(OpId::kScaledFp4Quant, q.device.type))(
      q, out_packed, out_scale, x, input_global_scale_inv, scale_layout);
}

void SiluMulFp4Quant(Queue& q, Tensor& out_packed, Tensor& out_scale, const Tensor& gate,
                     const Tensor& up, float input_global_scale_inv,
                     Fp4ScaleLayout scale_layout) {
  VT_CHECK(gate.rank == 2 && up.rank == 2 && out_packed.rank == 2 && out_scale.rank == 2,
           "silu_mul_fp4_quant: gate/up/out_packed/out_scale must be rank-2");
  const int64_t m = gate.shape[0], i = gate.shape[1];
  VT_CHECK(up.shape[0] == m && up.shape[1] == i, "silu_mul_fp4_quant: gate/up shape mismatch");
  VT_CHECK(i % 16 == 0, "silu_mul_fp4_quant: I (inner dim) must be a multiple of 16");
  VT_CHECK(out_packed.shape[0] == m && out_packed.shape[1] == i / 2,
           "silu_mul_fp4_quant: out_packed must be [M, I/2]");
  const auto round_up = [](int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
  };
  if (scale_layout == Fp4ScaleLayout::kLinear) {
    VT_CHECK(out_scale.shape[0] == m && out_scale.shape[1] == i / 16,
             "silu_mul_fp4_quant: linear out_scale must be [M, I/16]");
  } else {
    VT_CHECK(scale_layout == Fp4ScaleLayout::kCutlassSwizzled,
             "silu_mul_fp4_quant: invalid scale layout");
    VT_CHECK(out_scale.shape[0] == round_up(m, 128) &&
                 out_scale.shape[1] == round_up(i / 16, 4),
             "silu_mul_fp4_quant: swizzled out_scale must be "
             "[round_up(M,128), round_up(I/16,4)]");
  }
  VT_CHECK(IsFloat(gate.dtype) && gate.dtype == up.dtype,
           "silu_mul_fp4_quant: gate/up must be the same float dtype");
  VT_CHECK(out_packed.dtype == DType::kI8 && out_scale.dtype == DType::kI8,
           "silu_mul_fp4_quant: out_packed/out_scale must be i8 (raw fp4/fp8 bytes)");
  VT_CHECK(gate.IsContiguous() && up.IsContiguous() && out_packed.IsContiguous() &&
               out_scale.IsContiguous(),
           "silu_mul_fp4_quant: contiguous tensors required");
  VT_CHECK(gate.device == q.device && up.device == q.device && out_packed.device == q.device &&
               out_scale.device == q.device,
           "silu_mul_fp4_quant: device mismatch");
  reinterpret_cast<SiluMulFp4QuantFn>(GetOp(OpId::kSiluMulFp4Quant, q.device.type))(
      q, out_packed, out_scale, gate, up, input_global_scale_inv, scale_layout);
}

void SiluAndMulFp4Quant(Queue& q, Tensor& out_packed, Tensor& out_scale,
                        const Tensor& gate_up, float input_global_scale_inv,
                        Fp4ScaleLayout scale_layout) {
  VT_CHECK(gate_up.rank == 2 && out_packed.rank == 2 && out_scale.rank == 2,
           "silu_and_mul_fp4_quant: gate_up/out_packed/out_scale must be rank-2");
  const int64_t m = gate_up.shape[0];
  VT_CHECK(gate_up.shape[1] % 2 == 0,
           "silu_and_mul_fp4_quant: gate_up inner dim must be even");
  const int64_t i = gate_up.shape[1] / 2;
  VT_CHECK(i % 16 == 0,
           "silu_and_mul_fp4_quant: I (half inner dim) must be a multiple of 16");
  VT_CHECK(out_packed.shape[0] == m && out_packed.shape[1] == i / 2,
           "silu_and_mul_fp4_quant: out_packed must be [M, I/2]");
  const auto round_up = [](int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
  };
  if (scale_layout == Fp4ScaleLayout::kLinear) {
    VT_CHECK(out_scale.shape[0] == m && out_scale.shape[1] == i / 16,
             "silu_and_mul_fp4_quant: linear out_scale must be [M, I/16]");
  } else {
    VT_CHECK(scale_layout == Fp4ScaleLayout::kCutlassSwizzled,
             "silu_and_mul_fp4_quant: invalid scale layout");
    VT_CHECK(out_scale.shape[0] == round_up(m, 128) &&
                 out_scale.shape[1] == round_up(i / 16, 4),
             "silu_and_mul_fp4_quant: swizzled out_scale must be "
             "[round_up(M,128), round_up(I/16,4)]");
  }
  VT_CHECK(gate_up.dtype == DType::kF32 || gate_up.dtype == DType::kBF16,
           "silu_and_mul_fp4_quant: gate_up must be f32 or bf16");
  VT_CHECK(out_packed.dtype == DType::kI8 && out_scale.dtype == DType::kI8,
           "silu_and_mul_fp4_quant: outputs must be i8 (raw fp4/fp8 bytes)");
  VT_CHECK(gate_up.IsContiguous() && out_packed.IsContiguous() &&
               out_scale.IsContiguous(),
           "silu_and_mul_fp4_quant: contiguous tensors required");
  VT_CHECK(gate_up.device == q.device && out_packed.device == q.device &&
               out_scale.device == q.device,
           "silu_and_mul_fp4_quant: device mismatch");
  reinterpret_cast<SiluAndMulFp4QuantFn>(
      GetOp(OpId::kSiluAndMulFp4Quant, q.device.type))(
      q, out_packed, out_scale, gate_up, input_global_scale_inv, scale_layout);
}

void SigmoidGateFp4Quant(Queue& q, Tensor& out_packed, Tensor& out_scale,
                         const Tensor& attn, const Tensor& gate,
                         float input_global_scale_inv, Fp4ScaleLayout scale_layout) {
  VT_CHECK(attn.rank == 2 && gate.rank == 2 && out_packed.rank == 2 && out_scale.rank == 2,
           "sigmoid_gate_fp4_quant: attn/gate/out_packed/out_scale must be rank-2");
  const int64_t m = attn.shape[0], i = attn.shape[1];
  VT_CHECK(gate.shape[0] == m && gate.shape[1] == i,
           "sigmoid_gate_fp4_quant: attn/gate shape mismatch");
  VT_CHECK(i % 16 == 0, "sigmoid_gate_fp4_quant: K (inner dim) must be a multiple of 16");
  VT_CHECK(out_packed.shape[0] == m && out_packed.shape[1] == i / 2,
           "sigmoid_gate_fp4_quant: out_packed must be [M, K/2]");
  const auto round_up = [](int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
  };
  if (scale_layout == Fp4ScaleLayout::kLinear) {
    VT_CHECK(out_scale.shape[0] == m && out_scale.shape[1] == i / 16,
             "sigmoid_gate_fp4_quant: linear out_scale must be [M, K/16]");
  } else {
    VT_CHECK(scale_layout == Fp4ScaleLayout::kCutlassSwizzled,
             "sigmoid_gate_fp4_quant: invalid scale layout");
    VT_CHECK(out_scale.shape[0] == round_up(m, 128) &&
                 out_scale.shape[1] == round_up(i / 16, 4),
             "sigmoid_gate_fp4_quant: swizzled out_scale must be "
             "[round_up(M,128), round_up(K/16,4)]");
  }
  VT_CHECK(attn.dtype == DType::kF32 || attn.dtype == DType::kBF16,
           "sigmoid_gate_fp4_quant: attn must be f32 or bf16");
  VT_CHECK(gate.dtype == DType::kF32,
           "sigmoid_gate_fp4_quant: gate must be f32 (sigmoid input unrounded)");
  VT_CHECK(out_packed.dtype == DType::kI8 && out_scale.dtype == DType::kI8,
           "sigmoid_gate_fp4_quant: out_packed/out_scale must be i8 (raw fp4/fp8 bytes)");
  VT_CHECK(attn.IsContiguous() && gate.IsContiguous() && out_packed.IsContiguous() &&
               out_scale.IsContiguous(),
           "sigmoid_gate_fp4_quant: contiguous tensors required");
  VT_CHECK(attn.device == q.device && gate.device == q.device &&
               out_packed.device == q.device && out_scale.device == q.device,
           "sigmoid_gate_fp4_quant: device mismatch");
  reinterpret_cast<SigmoidGateFp4QuantFn>(GetOp(OpId::kSigmoidGateFp4Quant, q.device.type))(
      q, out_packed, out_scale, attn, gate, input_global_scale_inv, scale_layout);
}

void MatmulNvfp4Fp4(Queue& q, Tensor& out, const Tensor& a_packed, const Tensor& a_scale,
                    const Tensor& b_packed, const Tensor& b_scale, float alpha) {
  VT_CHECK(out.rank == 2 && a_packed.rank == 2 && a_scale.rank == 2 && b_packed.rank == 2 &&
               b_scale.rank == 2,
           "matmul_nvfp4_fp4: all tensors must be rank-2");
  const int64_t m = a_packed.shape[0], k = a_packed.shape[1] * 2, n = b_packed.shape[0];
  VT_CHECK(k % 16 == 0, "matmul_nvfp4_fp4: K (inner dim) must be a multiple of 16");
  VT_CHECK(a_scale.shape[0] == m && a_scale.shape[1] == k / 16,
           "matmul_nvfp4_fp4: a_scale must be [M, K/16]");
  VT_CHECK(b_packed.shape[1] == k / 2,
           "matmul_nvfp4_fp4: b_packed must be [N, K/2] (K matches a_packed)");
  VT_CHECK(b_scale.shape[0] == n && b_scale.shape[1] == k / 16,
           "matmul_nvfp4_fp4: b_scale must be [N, K/16]");
  VT_CHECK(out.shape[0] == m && out.shape[1] == n, "matmul_nvfp4_fp4: out must be [M, N]");
  VT_CHECK(IsOutFloat(out.dtype), "matmul_nvfp4_fp4: f32/bf16 out");
  VT_CHECK(a_packed.dtype == DType::kI8 && a_scale.dtype == DType::kI8 &&
               b_packed.dtype == DType::kI8 && b_scale.dtype == DType::kI8,
           "matmul_nvfp4_fp4: packed/scale operands must be i8 (raw fp4/fp8 bytes)");
  VT_CHECK(out.IsContiguous() && a_packed.IsContiguous() && a_scale.IsContiguous() &&
               b_packed.IsContiguous() && b_scale.IsContiguous(),
           "matmul_nvfp4_fp4: contiguous tensors required");
  VT_CHECK(out.device == q.device && a_packed.device == q.device && a_scale.device == q.device &&
               b_packed.device == q.device && b_scale.device == q.device,
           "matmul_nvfp4_fp4: device mismatch");
  reinterpret_cast<MatmulNvfp4Fp4Fn>(GetOp(OpId::kMatmulNvfp4Fp4, q.device.type))(
      q, out, a_packed, a_scale, b_packed, b_scale, alpha);
}

void SwizzleBlockscale(Queue& q, Tensor& out_swizzled, const Tensor& in_linear) {
  VT_CHECK(in_linear.rank == 2 && out_swizzled.rank == 2,
           "swizzle_blockscale: rank-2 tensors required");
  const int64_t rows = in_linear.shape[0], cols = in_linear.shape[1];
  auto round_up = [](int64_t x, int64_t y) { return (x + y - 1) / y * y; };
  VT_CHECK(out_swizzled.shape[0] == round_up(rows, 128) &&
               out_swizzled.shape[1] == round_up(cols, 4),
           "swizzle_blockscale: out must be [round_up(rows,128), round_up(cols,4)]");
  VT_CHECK(in_linear.dtype == DType::kI8 && out_swizzled.dtype == DType::kI8,
           "swizzle_blockscale: i8 (raw fp8) operands required");
  VT_CHECK(in_linear.IsContiguous() && out_swizzled.IsContiguous(),
           "swizzle_blockscale: contiguous tensors required");
  VT_CHECK(in_linear.device == q.device && out_swizzled.device == q.device,
           "swizzle_blockscale: device mismatch");
  reinterpret_cast<SwizzleBlockscaleFn>(GetOp(OpId::kSwizzleBlockscale, q.device.type))(
      q, out_swizzled, in_linear);
}

namespace {
void ValidateMatmulNvfp4Cutlass(Queue& q, Tensor& out,
                                const Tensor& a_packed,
                                const Tensor& a_sf_sw,
                                const Tensor& b_packed,
                                const Tensor& b_sf_sw) {
  VT_CHECK(out.rank == 2 && a_packed.rank == 2 && a_sf_sw.rank == 2 && b_packed.rank == 2 &&
               b_sf_sw.rank == 2,
           "matmul_nvfp4_cutlass: all tensors must be rank-2");
  const int64_t m = a_packed.shape[0], k = a_packed.shape[1] * 2, n = b_packed.shape[0];
  VT_CHECK(k % 32 == 0 && n % 32 == 0, "matmul_nvfp4_cutlass: K and N must be multiples of 32");
  VT_CHECK(b_packed.shape[1] == k / 2,
           "matmul_nvfp4_cutlass: b_packed must be [N, K/2] (K matches a_packed)");
  VT_CHECK(out.shape[0] == m && out.shape[1] == n, "matmul_nvfp4_cutlass: out must be [M, N]");
  VT_CHECK(out.dtype == DType::kBF16 || out.dtype == DType::kF32,
           "matmul_nvfp4_cutlass: out must be bf16 or f32 (bf16 epilogue, f32 via cast)");
  VT_CHECK(a_packed.dtype == DType::kI8 && a_sf_sw.dtype == DType::kI8 &&
               b_packed.dtype == DType::kI8 && b_sf_sw.dtype == DType::kI8,
           "matmul_nvfp4_cutlass: packed/scale operands must be i8 (raw fp4/fp8 bytes)");
  auto round_up = [](int64_t x, int64_t y) { return (x + y - 1) / y * y; };
  VT_CHECK(a_sf_sw.shape[0] == round_up(m, 128) && a_sf_sw.shape[1] == round_up(k / 16, 4),
           "matmul_nvfp4_cutlass: a_sf must be swizzled [round_up(M,128), round_up(K/16,4)]");
  VT_CHECK(b_sf_sw.shape[0] == round_up(n, 128) && b_sf_sw.shape[1] == round_up(k / 16, 4),
           "matmul_nvfp4_cutlass: b_sf must be swizzled [round_up(N,128), round_up(K/16,4)]");
  VT_CHECK(out.device == q.device && a_packed.device == q.device && a_sf_sw.device == q.device &&
               b_packed.device == q.device && b_sf_sw.device == q.device,
           "matmul_nvfp4_cutlass: device mismatch");
}

void DispatchMatmulNvfp4Cutlass(Queue& q, Tensor& out,
                                const Tensor& a_packed,
                                const Tensor& a_sf_sw,
                                const Tensor& b_packed,
                                const Tensor& b_sf_sw,
                                const Tensor* alpha_device,
                                float alpha_host) {
  reinterpret_cast<MatmulNvfp4CutlassFn>(GetOp(OpId::kMatmulNvfp4Cutlass, q.device.type))(
      q, out, a_packed, a_sf_sw, b_packed, b_sf_sw, alpha_device,
      alpha_host);
}
}  // namespace

void MatmulNvfp4Cutlass(Queue& q, Tensor& out, const Tensor& a_packed,
                        const Tensor& a_sf_sw, const Tensor& b_packed,
                        const Tensor& b_sf_sw, const Tensor& alpha) {
  ValidateMatmulNvfp4Cutlass(q, out, a_packed, a_sf_sw, b_packed, b_sf_sw);
  VT_CHECK(alpha.rank == 0 || alpha.rank == 1,
           "matmul_nvfp4_cutlass: alpha must be a rank-0 or rank-1 scalar tensor");
  VT_CHECK(alpha.Numel() == 1,
           "matmul_nvfp4_cutlass: alpha must contain exactly one element");
  VT_CHECK(alpha.dtype == DType::kF32,
           "matmul_nvfp4_cutlass: alpha must be f32");
  VT_CHECK(alpha.data != nullptr,
           "matmul_nvfp4_cutlass: alpha must have non-null storage");
  VT_CHECK(alpha.IsContiguous(),
           "matmul_nvfp4_cutlass: alpha must be contiguous");
  VT_CHECK(alpha.device == q.device,
           "matmul_nvfp4_cutlass: alpha device mismatch");
  DispatchMatmulNvfp4Cutlass(q, out, a_packed, a_sf_sw, b_packed,
                             b_sf_sw, &alpha, 0.0F);
}

void MatmulNvfp4Cutlass(Queue& q, Tensor& out, const Tensor& a_packed,
                        const Tensor& a_sf_sw, const Tensor& b_packed,
                        const Tensor& b_sf_sw, float alpha) {
  ValidateMatmulNvfp4Cutlass(q, out, a_packed, a_sf_sw, b_packed, b_sf_sw);
  DispatchMatmulNvfp4Cutlass(q, out, a_packed, a_sf_sw, b_packed,
                             b_sf_sw, nullptr, alpha);
}
void QuantFp8Static(Queue& q, Tensor& out_fp8, const Tensor& x, float input_scale) {
  VT_CHECK(x.rank == 2 && out_fp8.rank == 2, "quant_fp8_static: x/out must be rank-2");
  VT_CHECK(out_fp8.shape[0] == x.shape[0] && out_fp8.shape[1] == x.shape[1],
           "quant_fp8_static: out must match x shape [M,K]");
  VT_CHECK(IsFloat(x.dtype), "quant_fp8_static: float x (f32/bf16) required");
  VT_CHECK(out_fp8.dtype == DType::kI8, "quant_fp8_static: out must be i8 (raw fp8-e4m3fn bytes)");
  VT_CHECK(x.IsContiguous() && out_fp8.IsContiguous(),
           "quant_fp8_static: contiguous tensors required");
  VT_CHECK(x.device == q.device && out_fp8.device == q.device,
           "quant_fp8_static: device mismatch (x/out/queue)");
  reinterpret_cast<QuantFp8StaticFn>(GetOp(OpId::kQuantFp8Static, q.device.type))(q, out_fp8, x,
                                                                                  input_scale);
}
void QuantFp8Group(Queue& q, Tensor& out_fp8, Tensor& out_scale, const Tensor& x,
                   int group_size) {
  VT_CHECK(x.rank == 2 && out_fp8.rank == 2 && out_scale.rank == 2,
           "quant_fp8_group: x/out_fp8/out_scale must be rank-2");
  // group_size is validated BEFORE it divides anything: `K % 0` is undefined
  // behaviour, so a zero here must refuse rather than trap.
  VT_CHECK(group_size > 0, "quant_fp8_group: group_size must be positive");
  const int64_t m = x.shape[0], k = x.shape[1];
  // Mirrors upstream's assert text at
  // vllm/model_executor/layers/quantization/utils/fp8_utils.py:596-599.
  VT_CHECK(k % group_size == 0,
           "quant_fp8_group: the last dimension of x must be divisible by group_size");
  VT_CHECK(out_fp8.shape[0] == m && out_fp8.shape[1] == k,
           "quant_fp8_group: out_fp8 must match x shape [M,K]");
  VT_CHECK(out_scale.shape[0] == m && out_scale.shape[1] == k / group_size,
           "quant_fp8_group: out_scale must be [M, K/group_size]");
  VT_CHECK(IsFloat(x.dtype), "quant_fp8_group: float x (f32/bf16) required");
  VT_CHECK(out_fp8.dtype == DType::kI8,
           "quant_fp8_group: out_fp8 must be i8 (raw fp8-e4m3fn bytes)");
  // f32, not the model dtype: upstream allocates the scale f32 (fp8_utils.py:631)
  // and the block-scaled GEMM multiplies it into an f32 accumulator.
  VT_CHECK(out_scale.dtype == DType::kF32, "quant_fp8_group: out_scale must be f32");
  // Upstream asserts `x.stride(-1) == 1` (fp8_utils.py:600); a group that is not
  // contiguous would read across rows.
  VT_CHECK(x.IsContiguous() && out_fp8.IsContiguous() && out_scale.IsContiguous(),
           "quant_fp8_group: contiguous tensors required");
  VT_CHECK(x.device == q.device && out_fp8.device == q.device && out_scale.device == q.device,
           "quant_fp8_group: device mismatch (x/out_fp8/out_scale/queue)");
  reinterpret_cast<QuantFp8GroupFn>(GetOp(OpId::kQuantFp8Group, q.device.type))(
      q, out_fp8, out_scale, x, group_size);
}
void MatmulFp8BlockScaled(Queue& q, Tensor& out, const Tensor& a_fp8, const Tensor& a_scale,
                          const Tensor& b_fp8, const Tensor& b_scale, int block_n,
                          int block_k) {
  VT_CHECK(out.rank == 2 && a_fp8.rank == 2 && a_scale.rank == 2 && b_fp8.rank == 2 &&
               b_scale.rank == 2,
           "matmul_fp8_block_scaled: out/a_fp8/a_scale/b_fp8/b_scale must be rank-2");
  // Validated BEFORE either one divides anything: `x / 0` and `x % 0` are
  // undefined behaviour, so a zero must refuse rather than trap.
  VT_CHECK(block_n > 0 && block_k > 0,
           "matmul_fp8_block_scaled: block_n and block_k must be positive");
  const int64_t m = a_fp8.shape[0], k = a_fp8.shape[1];
  const int64_t n = b_fp8.shape[0];
  // Upstream: `assert A.shape[-1] == B.shape[-1]` (quant_utils.py:111).
  VT_CHECK(b_fp8.shape[1] == k,
           "matmul_fp8_block_scaled: a_fp8 [M,K] and b_fp8 [N,K] must share K");
  VT_CHECK(out.shape[0] == m && out.shape[1] == n, "matmul_fp8_block_scaled: out must be [M,N]");
  VT_CHECK(a_fp8.dtype == DType::kI8 && b_fp8.dtype == DType::kI8,
           "matmul_fp8_block_scaled: a_fp8/b_fp8 must be i8 (raw fp8-e4m3fn bytes)");
  // f32, not the model dtype: upstream refuses any other scale dtype on this
  // path (csrc/.../w8a8/cutlass/c3x/scaled_mm_helper.hpp:15-18) and the
  // accumulator these multiply into is f32.
  VT_CHECK(a_scale.dtype == DType::kF32 && b_scale.dtype == DType::kF32,
           "matmul_fp8_block_scaled: a_scale/b_scale must be f32");
  VT_CHECK(out.dtype == DType::kF32 || out.dtype == DType::kBF16,
           "matmul_fp8_block_scaled: out must be f32 or bf16");
  // CEIL on every tiling, so a ragged final block is legal: upstream asserts
  // `triton.cdiv(N, block_n) == Bs.shape[0]` and
  // `triton.cdiv(K, block_k) == Bs.shape[1]` (fp8_utils.py:935-936), and
  // `triton.cdiv(A.shape[-1], block_k) == As.shape[-1]` (fp8_utils.py:930).
  const int64_t k_tiles = (k + block_k - 1) / block_k;
  const int64_t n_tiles = (n + block_n - 1) / block_n;
  VT_CHECK(a_scale.shape[0] == m && a_scale.shape[1] == k_tiles,
           "matmul_fp8_block_scaled: a_scale must be [M, cdiv(K, block_k)]");
  VT_CHECK(b_scale.shape[0] == n_tiles && b_scale.shape[1] == k_tiles,
           "matmul_fp8_block_scaled: b_scale must be [cdiv(N, block_n), cdiv(K, block_k)]");
  VT_CHECK(out.IsContiguous() && a_fp8.IsContiguous() && a_scale.IsContiguous() &&
               b_fp8.IsContiguous() && b_scale.IsContiguous(),
           "matmul_fp8_block_scaled: contiguous tensors required");
  VT_CHECK(out.device == q.device && a_fp8.device == q.device && a_scale.device == q.device &&
               b_fp8.device == q.device && b_scale.device == q.device,
           "matmul_fp8_block_scaled: device mismatch (out/a_fp8/a_scale/b_fp8/b_scale/queue)");
  reinterpret_cast<MatmulFp8BlockScaledFn>(GetOp(OpId::kMatmulFp8BlockScaled, q.device.type))(
      q, out, a_fp8, a_scale, b_fp8, b_scale, block_n, block_k);
}
void RmsNormQuantFp8(Queue& q, Tensor& out_fp8, Tensor* out_bf16, const Tensor& x,
                     const Tensor& weight, const RmsNormArgs& args, Tensor* residual,
                     float input_scale) {
  VT_CHECK(x.rank == 2 && out_fp8.rank == 2 && weight.rank == 1,
           "rmsnorm_quant_fp8: x/out_fp8 rank-2, weight rank-1");
  VT_CHECK(x.shape[0] == out_fp8.shape[0] && x.shape[1] == out_fp8.shape[1],
           "rmsnorm_quant_fp8: out_fp8 must match x shape [T,H]");
  VT_CHECK(weight.shape[0] == x.shape[1], "rmsnorm_quant_fp8: weight size mismatch");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(weight.dtype), "rmsnorm_quant_fp8: float x/weight required");
  VT_CHECK(out_fp8.dtype == DType::kI8,
           "rmsnorm_quant_fp8: out_fp8 must be i8 (raw fp8-e4m3fn bytes)");
  VT_CHECK(x.IsContiguous() && out_fp8.IsContiguous() && weight.IsContiguous(),
           "rmsnorm_quant_fp8: contiguous tensors required");
  if (out_bf16 != nullptr) {
    VT_CHECK(out_bf16->dtype == DType::kBF16 && out_bf16->rank == 2 &&
                 out_bf16->shape[0] == x.shape[0] && out_bf16->shape[1] == x.shape[1] &&
                 out_bf16->IsContiguous() && out_bf16->device == x.device,
             "rmsnorm_quant_fp8: out_bf16 must be bf16 [T,H] contiguous on x's device");
  }
  if (residual != nullptr) {
    VT_CHECK((residual->dtype == DType::kF32 || residual->dtype == DType::kBF16) &&
                 residual->rank == 2 && residual->shape[0] == x.shape[0] &&
                 residual->shape[1] == x.shape[1] && residual->IsContiguous() &&
                 residual->device == x.device,
             "rmsnorm_quant_fp8: residual must be f32/bf16 [T,H] contiguous on x's device");
  }
  VT_CHECK(x.device == out_fp8.device && weight.device == x.device && x.device == q.device,
           "rmsnorm_quant_fp8: device mismatch (x/out_fp8/weight/queue)");
  reinterpret_cast<RmsNormQuantFp8Fn>(GetOp(OpId::kRmsNormQuantFp8, q.device.type))(
      q, out_fp8, out_bf16, x, weight, args, residual, input_scale);
}
void RmsNormGatedQuantFp8(Queue& q, Tensor& out_fp8, const Tensor& x, const Tensor& gate,
                          const Tensor& weight, const RmsNormGatedArgs& args, float input_scale) {
  const int64_t d = x.rank == 0 ? 0 : x.shape[x.rank - 1];
  VT_CHECK(weight.rank == 1 && weight.shape[0] == d,
           "rmsnorm_gated_quant_fp8: weight must be rank-1 [D] matching x's last dim");
  VT_CHECK(out_fp8.rank == x.rank, "rmsnorm_gated_quant_fp8: out_fp8 rank must match x");
  for (int i = 0; i < x.rank; ++i)
    VT_CHECK(out_fp8.shape[i] == x.shape[i],
             "rmsnorm_gated_quant_fp8: out_fp8 shape must match x");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(weight.dtype) && IsFloat(gate.dtype),
           "rmsnorm_gated_quant_fp8: float x/gate/weight required");
  VT_CHECK(gate.dtype == x.dtype && weight.dtype == x.dtype,
           "rmsnorm_gated_quant_fp8: gate/weight dtype must match x");
  VT_CHECK(out_fp8.dtype == DType::kI8,
           "rmsnorm_gated_quant_fp8: out_fp8 must be i8 (raw fp8-e4m3fn bytes)");
  VT_CHECK(x.IsContiguous() && out_fp8.IsContiguous() && weight.IsContiguous(),
           "rmsnorm_gated_quant_fp8: contiguous x/out_fp8/weight required");
  VT_CHECK(x.device == out_fp8.device && weight.device == x.device && gate.device == x.device &&
               x.device == q.device,
           "rmsnorm_gated_quant_fp8: device mismatch (x/out_fp8/gate/weight/queue)");
  reinterpret_cast<RmsNormGatedQuantFp8Fn>(GetOp(OpId::kRmsNormGatedQuantFp8, q.device.type))(
      q, out_fp8, x, gate, weight, args, input_scale);
}
void MatmulFp8Cutlass(Queue& q, Tensor& out, const Tensor& a_fp8, const Tensor& b_fp8,
                      float alpha) {
  VT_CHECK(out.rank == 2 && a_fp8.rank == 2 && b_fp8.rank == 2,
           "matmul_fp8_cutlass: all tensors must be rank-2");
  const int64_t m = a_fp8.shape[0], k = a_fp8.shape[1], n = b_fp8.shape[0];
  VT_CHECK(k % 16 == 0 && n % 16 == 0, "matmul_fp8_cutlass: K and N must be multiples of 16");
  VT_CHECK(b_fp8.shape[1] == k, "matmul_fp8_cutlass: b_fp8 must be [N, K] (K matches a_fp8)");
  VT_CHECK(out.shape[0] == m && out.shape[1] == n, "matmul_fp8_cutlass: out must be [M, N]");
  VT_CHECK(out.dtype == DType::kBF16 || out.dtype == DType::kF32,
           "matmul_fp8_cutlass: out must be bf16 or f32 (bf16 epilogue, f32 via cast)");
  VT_CHECK(a_fp8.dtype == DType::kI8 && b_fp8.dtype == DType::kI8,
           "matmul_fp8_cutlass: a_fp8/b_fp8 must be i8 (raw fp8-e4m3fn bytes)");
  VT_CHECK(out.IsContiguous() && a_fp8.IsContiguous() && b_fp8.IsContiguous(),
           "matmul_fp8_cutlass: contiguous tensors required");
  VT_CHECK(out.device == q.device && a_fp8.device == q.device && b_fp8.device == q.device,
           "matmul_fp8_cutlass: device mismatch");
  reinterpret_cast<MatmulFp8CutlassFn>(GetOp(OpId::kMatmulFp8Cutlass, q.device.type))(
      q, out, a_fp8, b_fp8, alpha);
}
void MatmulFp8CublasLt(Queue& q, Tensor& out, const Tensor& a_fp8, const Tensor& b_fp8,
                       float alpha, bool claims_splitk1_premise) {
  // Same argument contract as MatmulFp8Cutlass (drop-in fp8 dense GEMM).
  VT_CHECK(out.rank == 2 && a_fp8.rank == 2 && b_fp8.rank == 2,
           "matmul_fp8_cublaslt: all tensors must be rank-2");
  const int64_t m = a_fp8.shape[0], k = a_fp8.shape[1], n = b_fp8.shape[0];
  VT_CHECK(k % 16 == 0 && n % 16 == 0, "matmul_fp8_cublaslt: K and N must be multiples of 16");
  VT_CHECK(b_fp8.shape[1] == k, "matmul_fp8_cublaslt: b_fp8 must be [N, K] (K matches a_fp8)");
  VT_CHECK(out.shape[0] == m && out.shape[1] == n, "matmul_fp8_cublaslt: out must be [M, N]");
  VT_CHECK(out.dtype == DType::kBF16 || out.dtype == DType::kF32,
           "matmul_fp8_cublaslt: out must be bf16 or f32");
  VT_CHECK(a_fp8.dtype == DType::kI8 && b_fp8.dtype == DType::kI8,
           "matmul_fp8_cublaslt: a_fp8/b_fp8 must be i8 (raw fp8-e4m3fn bytes)");
  VT_CHECK(out.IsContiguous() && a_fp8.IsContiguous() && b_fp8.IsContiguous(),
           "matmul_fp8_cublaslt: contiguous tensors required");
  VT_CHECK(out.device == q.device && a_fp8.device == q.device && b_fp8.device == q.device,
           "matmul_fp8_cublaslt: device mismatch");
  reinterpret_cast<MatmulFp8CublasLtFn>(GetOp(OpId::kMatmulFp8CublasLt, q.device.type))(
      q, out, a_fp8, b_fp8, alpha, claims_splitk1_premise);
}
void MatmulFp8CublasLtAlphaVec(Queue& q, Tensor& out, const Tensor& a_fp8, const Tensor& b_fp8,
                               const Tensor& alpha_vec, bool claims_splitk1_premise) {
  // Same operand contract as MatmulFp8CublasLt, plus the per-column alpha.
  //
  // The output was f32 ONLY, because the fallback arm applies the vector with
  // vt::MulColVecF32 and that op was f32-typed -- accepting bf16 would have been
  // a capability the fallback could not honor. PERF-FP8-ALPHA-FOLD / #417 removed
  // that blocker (MulColVecF32 now carries a bf16 store arm), so bf16 is
  // admitted: the GEMM emits a bf16 D and the column pass runs at bf16, halving
  // the bytes it moves. bf16 is also what vLLM's ModelOptFp8LinearMethod emits
  // here (out_dtype = the model dtype, modelopt.py:458).
  //
  // A bf16 `out` always takes the TWO-LAUNCH arm, whatever
  // VT_FP8_ALPHA_VEC_EPILOGUE says -- see the CUDA implementation for why: at
  // bf16 the epilogue would round ONCE and the fallback rounds TWICE, so letting
  // the toggle choose between them would turn a performance switch into a
  // numerics switch. The toggle stays a pure performance A/B at every dtype.
  VT_CHECK(out.rank == 2 && a_fp8.rank == 2 && b_fp8.rank == 2,
           "matmul_fp8_cublaslt_alpha_vec: all tensors must be rank-2");
  const int64_t m = a_fp8.shape[0], k = a_fp8.shape[1], n = b_fp8.shape[0];
  VT_CHECK(k % 16 == 0 && n % 16 == 0,
           "matmul_fp8_cublaslt_alpha_vec: K and N must be multiples of 16");
  VT_CHECK(b_fp8.shape[1] == k,
           "matmul_fp8_cublaslt_alpha_vec: b_fp8 must be [N, K] (K matches a_fp8)");
  VT_CHECK(out.shape[0] == m && out.shape[1] == n,
           "matmul_fp8_cublaslt_alpha_vec: out must be [M, N]");
  VT_CHECK(out.dtype == DType::kF32 || out.dtype == DType::kBF16,
           "matmul_fp8_cublaslt_alpha_vec: out must be f32 or bf16");
  VT_CHECK(a_fp8.dtype == DType::kI8 && b_fp8.dtype == DType::kI8,
           "matmul_fp8_cublaslt_alpha_vec: a_fp8/b_fp8 must be i8 (raw fp8-e4m3fn bytes)");
  VT_CHECK(out.IsContiguous() && a_fp8.IsContiguous() && b_fp8.IsContiguous(),
           "matmul_fp8_cublaslt_alpha_vec: contiguous tensors required");
  // cuBLASLt reads the alpha vector as one entry per OUTPUT ROW of its
  // column-major D, which is our row-major out's COLUMN count, N.
  VT_CHECK(alpha_vec.rank == 1 && alpha_vec.shape[0] == n,
           "matmul_fp8_cublaslt_alpha_vec: alpha_vec must be [N] (one alpha per output column)");
  VT_CHECK(alpha_vec.dtype == DType::kF32 && alpha_vec.IsContiguous(),
           "matmul_fp8_cublaslt_alpha_vec: alpha_vec must be contiguous f32");
  VT_CHECK(out.device == q.device && a_fp8.device == q.device && b_fp8.device == q.device &&
               alpha_vec.device == q.device,
           "matmul_fp8_cublaslt_alpha_vec: device mismatch");
  reinterpret_cast<MatmulFp8CublasLtAlphaVecFn>(
      GetOp(OpId::kMatmulFp8CublasLtAlphaVec, q.device.type))(q, out, a_fp8, b_fp8, alpha_vec,
                                                             claims_splitk1_premise);
}

void MoeGroupedGemmNvfp4(Queue& q, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                         const Tensor* row_map, const Tensor& packed_ptrs,
                         const Tensor& scale_ptrs, const Tensor& scale2s) {
  VT_CHECK(out.rank == 2 && act.rank == 2, "moe_grouped_gemm_nvfp4: out/act must be rank-2");
  const int64_t p = out.shape[0], k = act.shape[1], e = scale2s.shape[0];
  VT_CHECK(k % 16 == 0, "moe_grouped_gemm_nvfp4: K (act inner dim) must be a multiple of 16");
  VT_CHECK(expert_ids.Numel() == p,
           "moe_grouped_gemm_nvfp4: expert_ids must have P entries (one per out row)");
  VT_CHECK(expert_ids.dtype == DType::kI32, "moe_grouped_gemm_nvfp4: expert_ids must be i32");
  VT_CHECK(packed_ptrs.Numel() == e && scale_ptrs.Numel() == e,
           "moe_grouped_gemm_nvfp4: packed_ptrs/scale_ptrs must have E entries");
  VT_CHECK(packed_ptrs.dtype == DType::kI64 && scale_ptrs.dtype == DType::kI64,
           "moe_grouped_gemm_nvfp4: packed_ptrs/scale_ptrs must be i64 (device pointers)");
  VT_CHECK(scale2s.dtype == DType::kF32, "moe_grouped_gemm_nvfp4: scale2s must be f32");
  VT_CHECK(IsFloat(act.dtype) && IsOutFloat(out.dtype),
           "moe_grouped_gemm_nvfp4: float act, f32/bf16 out");
  VT_CHECK(act.IsContiguous() && out.IsContiguous() && expert_ids.IsContiguous() &&
               packed_ptrs.IsContiguous() && scale_ptrs.IsContiguous() && scale2s.IsContiguous(),
           "moe_grouped_gemm_nvfp4: contiguous tensors required");
  VT_CHECK(act.device == q.device && out.device == q.device && expert_ids.device == q.device &&
               packed_ptrs.device == q.device && scale_ptrs.device == q.device &&
               scale2s.device == q.device,
           "moe_grouped_gemm_nvfp4: device mismatch");
  if (row_map != nullptr) {
    VT_CHECK(row_map->Numel() == p && row_map->dtype == DType::kI32 && row_map->IsContiguous() &&
                 row_map->device == q.device,
             "moe_grouped_gemm_nvfp4: row_map must be contiguous i32 [P] on the queue device");
  }
  reinterpret_cast<MoeGroupedGemmNvfp4Fn>(GetOp(OpId::kMoeGroupedGemmNvfp4, q.device.type))(
      q, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs, scale2s);
}

namespace {
void ValidateMoeGroupedGemmBf16(Queue& q, const Tensor& out, const Tensor& act, const Tensor& expert_ids,
                        const Tensor* row_map, const Tensor& weight_ptrs) {
  VT_CHECK(out.rank == 2 && act.rank == 2, "moe_grouped_gemm_bf16: out/act must be rank-2");
  const int64_t p = out.shape[0], e = weight_ptrs.shape[0];
  VT_CHECK(act.dtype == DType::kBF16, "moe_grouped_gemm_bf16: act must be bf16");
  VT_CHECK(IsOutFloat(out.dtype), "moe_grouped_gemm_bf16: out must be f32/bf16");
  VT_CHECK(expert_ids.Numel() == p,
           "moe_grouped_gemm_bf16: expert_ids must have P entries (one per out row)");
  VT_CHECK(expert_ids.dtype == DType::kI32, "moe_grouped_gemm_bf16: expert_ids must be i32");
  VT_CHECK(weight_ptrs.Numel() == e && weight_ptrs.dtype == DType::kI64,
           "moe_grouped_gemm_bf16: weight_ptrs must be i64 [E] (device pointers)");
  VT_CHECK(act.IsContiguous() && out.IsContiguous() && expert_ids.IsContiguous() &&
               weight_ptrs.IsContiguous(),
           "moe_grouped_gemm_bf16: contiguous tensors required");
  VT_CHECK(act.device == q.device && out.device == q.device && expert_ids.device == q.device &&
               weight_ptrs.device == q.device,
           "moe_grouped_gemm_bf16: device mismatch");
  if (row_map != nullptr) {
    VT_CHECK(row_map->Numel() == p && row_map->dtype == DType::kI32 && row_map->IsContiguous() &&
                 row_map->device == q.device,
             "moe_grouped_gemm_bf16: row_map must be contiguous i32 [P] on the queue device");
  }
}
}  // namespace

void MoeGroupedGemmBf16(Queue& q, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                        const Tensor* row_map, const Tensor& weight_ptrs) {
  ValidateMoeGroupedGemmBf16(q, out, act, expert_ids, row_map, weight_ptrs);
  reinterpret_cast<MoeGroupedGemmBf16Fn>(GetOp(OpId::kMoeGroupedGemmBf16, q.device.type))(
      q, out, act, expert_ids, row_map, weight_ptrs);
}

namespace {
void ValidateMoeGroupedGemmBf16GateUpSilu(Queue& q, const Tensor& out, const Tensor& act,
                                  const Tensor& expert_ids, const Tensor* row_map,
                                  const Tensor& gate_ptrs, const Tensor& up_ptrs) {
  VT_CHECK(out.rank == 2 && act.rank == 2,
           "moe_grouped_gemm_bf16_gate_up_silu: out/act must be rank-2");
  const int64_t p = out.shape[0], e = gate_ptrs.shape[0];
  VT_CHECK(act.dtype == DType::kBF16, "moe_grouped_gemm_bf16_gate_up_silu: act must be bf16");
  VT_CHECK(out.dtype == DType::kBF16,
           "moe_grouped_gemm_bf16_gate_up_silu: out must be bf16 (fused silu store)");
  VT_CHECK(expert_ids.Numel() == p,
           "moe_grouped_gemm_bf16_gate_up_silu: expert_ids must have P entries (one per out row)");
  VT_CHECK(expert_ids.dtype == DType::kI32,
           "moe_grouped_gemm_bf16_gate_up_silu: expert_ids must be i32");
  VT_CHECK(gate_ptrs.Numel() == e && gate_ptrs.dtype == DType::kI64 && up_ptrs.Numel() == e &&
               up_ptrs.dtype == DType::kI64,
           "moe_grouped_gemm_bf16_gate_up_silu: gate_ptrs/up_ptrs must be i64 [E] device pointers");
  VT_CHECK(act.IsContiguous() && out.IsContiguous() && expert_ids.IsContiguous() &&
               gate_ptrs.IsContiguous() && up_ptrs.IsContiguous(),
           "moe_grouped_gemm_bf16_gate_up_silu: contiguous tensors required");
  VT_CHECK(act.device == q.device && out.device == q.device && expert_ids.device == q.device &&
               gate_ptrs.device == q.device && up_ptrs.device == q.device,
           "moe_grouped_gemm_bf16_gate_up_silu: device mismatch");
  if (row_map != nullptr) {
    VT_CHECK(row_map->Numel() == p && row_map->dtype == DType::kI32 && row_map->IsContiguous() &&
                 row_map->device == q.device,
             "moe_grouped_gemm_bf16_gate_up_silu: row_map must be contiguous i32 [P] on the device");
  }
}
}  // namespace

void MoeGroupedGemmBf16GateUpSilu(Queue& q, Tensor& out, const Tensor& act,
                                  const Tensor& expert_ids, const Tensor* row_map,
                                  const Tensor& gate_ptrs, const Tensor& up_ptrs) {
  ValidateMoeGroupedGemmBf16GateUpSilu(q, out, act, expert_ids, row_map, gate_ptrs, up_ptrs);
  reinterpret_cast<MoeGroupedGemmBf16GateUpSiluFn>(
      GetOp(OpId::kMoeGroupedGemmBf16GateUpSilu, q.device.type))(q, out, act, expert_ids, row_map,
                                                                gate_ptrs, up_ptrs);
}

void MoeGroupedGemmBf16GateUpSiluNative(Queue& q, Tensor& out, const Tensor& act,
                                      const Tensor& expert_ids, const Tensor* row_map,
                                      const Tensor& gate_ptrs, const Tensor& up_ptrs) {
  ValidateMoeGroupedGemmBf16GateUpSilu(q, out, act, expert_ids, row_map, gate_ptrs, up_ptrs);
  reinterpret_cast<MoeGroupedGemmBf16GateUpSiluNativeFn>(
      GetOp(OpId::kMoeGroupedGemmBf16GateUpSiluNative, q.device.type))(
      q, out, act, expert_ids, row_map, gate_ptrs, up_ptrs);
}

void MoeGroupedGemmBf16Weighted(Queue& q, Tensor& out, const Tensor& act,
                               const Tensor& expert_ids, const Tensor* row_map,
                               const Tensor& weight_ptrs, const Tensor& route_weights) {
  ValidateMoeGroupedGemmBf16(q, out, act, expert_ids, row_map, weight_ptrs);
  VT_CHECK(route_weights.rank == 1 && route_weights.Numel() == out.shape[0] &&
               route_weights.dtype == DType::kF32 && route_weights.IsContiguous() &&
               route_weights.device == q.device,
           "moe_grouped_gemm_bf16_weighted: route_weights must be contiguous f32 [P] on the device");
  reinterpret_cast<MoeGroupedGemmBf16WeightedFn>(
      GetOp(OpId::kMoeGroupedGemmBf16Weighted, q.device.type))(
      q, out, act, expert_ids, row_map, weight_ptrs, route_weights);
}

void MoeCombinePreweighted(Queue& q, Tensor& out, const Tensor& expert_out,
                           const Tensor* shared, float routed_scale) {
  VT_CHECK(out.rank == 2 && expert_out.rank == 3 &&
               expert_out.shape[0] == out.shape[0] && expert_out.shape[2] == out.shape[1],
           "moe_combine_preweighted: expert_out [T,K,H] must match out [T,H]");
  VT_CHECK(expert_out.dtype == DType::kBF16 && IsOutFloat(out.dtype),
           "moe_combine_preweighted: expert_out must be bf16, out must be f32/bf16");
  VT_CHECK(expert_out.IsContiguous() && out.IsContiguous() &&
               expert_out.device == q.device && out.device == q.device,
           "moe_combine_preweighted: contiguous tensors on the queue device required");
  if (shared != nullptr) {
    VT_CHECK(shared->rank == 2 && shared->shape[0] == out.shape[0] &&
                 shared->shape[1] == out.shape[1] && IsOutFloat(shared->dtype) &&
                 shared->IsContiguous() && shared->device == q.device,
             "moe_combine_preweighted: shared must be f32/bf16 [T,H] on the queue device");
  }
  reinterpret_cast<MoeCombinePreweightedFn>(
      GetOp(OpId::kMoeCombinePreweighted, q.device.type))(q, out, expert_out, shared, routed_scale);
}

bool MoeGroupedBf16NativeAvailable(DeviceType device) {
  return OpRegistered(OpId::kMoeGroupedGemmBf16, device) &&
         OpRegistered(OpId::kMoeGroupedGemmBf16GateUpSilu, device) &&
         OpRegistered(OpId::kMoeGroupedGemmBf16GateUpSiluNative, device) &&
         OpRegistered(OpId::kMoeGroupedGemmBf16Weighted, device) &&
         OpRegistered(OpId::kMoeCombinePreweighted, device);
}

void MoeGroupedGemmNvfp4Marlin(Queue& q, Tensor& c, const Tensor& a, const Tensor& b_q_weight,
                               const Tensor& b_scales, const Tensor& global_scale,
                               Tensor& workspace, const Tensor& sorted_token_ids,
                               const Tensor& expert_ids, const Tensor& num_tokens_past_padded,
                               const Tensor& topk_weights, const MoeMarlinArgs& args) {
  VT_CHECK(a.rank == 2 && c.rank == 2, "moe_marlin: a/c must be rank-2");
  VT_CHECK(a.dtype == DType::kBF16 && c.dtype == DType::kBF16, "moe_marlin: a/c must be bf16");
  VT_CHECK(args.size_k % 16 == 0, "moe_marlin: size_k must be a multiple of 16 (group size)");
  VT_CHECK(a.shape[0] == args.size_m && a.shape[1] == args.size_k,
           "moe_marlin: a shape must be [size_m, size_k]");
  VT_CHECK(b_q_weight.rank == 3, "moe_marlin: b_q_weight must be rank-3 [E, K/16, N*8/pack]");
  VT_CHECK(expert_ids.dtype == DType::kI32 && sorted_token_ids.dtype == DType::kI32 &&
               num_tokens_past_padded.dtype == DType::kI32,
           "moe_marlin: align tensors must be i32");
  VT_CHECK(global_scale.dtype == DType::kF32 && topk_weights.dtype == DType::kF32,
           "moe_marlin: global_scale/topk_weights must be f32");
  VT_CHECK(workspace.dtype == DType::kI32, "moe_marlin: workspace must be i32 (reduction locks)");
  reinterpret_cast<MoeGroupedGemmNvfp4MarlinFn>(
      GetOp(OpId::kMoeGroupedGemmNvfp4Marlin, q.device.type))(
      q, c, a, b_q_weight, b_scales, global_scale, workspace, sorted_token_ids, expert_ids,
      num_tokens_past_padded, topk_weights, args);
}

void MarlinDenseGemm(Queue& q, Tensor& c, const Tensor& a, const Tensor& b_q_weight,
                     const Tensor& b_scales, const Tensor& global_scale, Tensor& workspace,
                     const MarlinDenseArgs& args) {
  VT_CHECK(a.rank == 2 && c.rank == 2, "marlin_dense: a/c must be rank-2");
  VT_CHECK(a.dtype == DType::kBF16 && c.dtype == DType::kBF16, "marlin_dense: a/c must be bf16");
  VT_CHECK(args.size_k % 16 == 0, "marlin_dense: size_k must be a multiple of 16 (group size)");
  VT_CHECK(args.group_size == 16 || args.group_size == 32,
           "marlin_dense: group_size must be 16 (nvfp4) or 32 (mxfp4)");
  VT_CHECK(a.shape[0] == args.size_m && a.shape[1] == args.size_k,
           "marlin_dense: a shape must be [size_m, size_k]");
  VT_CHECK(c.shape[0] == args.size_m && c.shape[1] == args.size_n,
           "marlin_dense: c shape must be [size_m, size_n]");
  VT_CHECK(b_q_weight.rank == 2, "marlin_dense: b_q_weight must be rank-2 [K/16, N*8/pack]");
  VT_CHECK(global_scale.dtype == DType::kF32, "marlin_dense: global_scale must be f32");
  VT_CHECK(workspace.dtype == DType::kI32, "marlin_dense: workspace must be i32 (reduction locks)");
  reinterpret_cast<MarlinDenseGemmFn>(GetOp(OpId::kMarlinDenseGemm, q.device.type))(
      q, c, a, b_q_weight, b_scales, global_scale, workspace, args);
}

void MoeSiluMul(Queue& q, Tensor& out, const Tensor& gate, const Tensor& up) {
  VT_CHECK(gate.Numel() == out.Numel() && up.Numel() == out.Numel(),
           "moe_silu_mul: out/gate/up must have the same element count");
  VT_CHECK(IsFloat(gate.dtype) && IsFloat(up.dtype) && IsOutFloat(out.dtype),
           "moe_silu_mul: float gate/up, f32/bf16 out");
  VT_CHECK(out.IsContiguous() && gate.IsContiguous() && up.IsContiguous(),
           "moe_silu_mul: contiguous tensors required");
  VT_CHECK(out.device == q.device && gate.device == q.device && up.device == q.device,
           "moe_silu_mul: device mismatch (out/gate/up/queue)");
  reinterpret_cast<MoeSiluMulFn>(GetOp(OpId::kMoeSiluMul, q.device.type))(q, out, gate, up);
}

void MoeRelu2(Queue& q, Tensor& out, const Tensor& x) {
  VT_CHECK(x.Numel() == out.Numel(), "moe_relu2: out/x must have the same element count");
  VT_CHECK(IsFloat(x.dtype) && IsOutFloat(out.dtype), "moe_relu2: float x, f32/bf16 out");
  VT_CHECK(out.IsContiguous() && x.IsContiguous(), "moe_relu2: contiguous tensors required");
  VT_CHECK(out.device == q.device && x.device == q.device,
           "moe_relu2: device mismatch (out/x/queue)");
  reinterpret_cast<MoeRelu2Fn>(GetOp(OpId::kMoeRelu2, q.device.type))(q, out, x);
}

void RmsNorm(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
             const RmsNormArgs& args, Tensor* residual) {
  VT_CHECK(x.rank == 2 && out.rank == 2 && weight.rank == 1, "rmsnorm: x/out rank-2, w rank-1");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1], "rmsnorm: shape mismatch");
  VT_CHECK(weight.shape[0] == x.shape[1], "rmsnorm: weight size mismatch");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(weight.dtype) && IsOutFloat(out.dtype),
           "rmsnorm: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous() && weight.IsContiguous(),
           "rmsnorm: contiguous required");
  if (residual != nullptr) {
    VT_CHECK((residual->dtype == DType::kF32 || residual->dtype == DType::kBF16) &&
                 residual->rank == 2 &&
                 residual->shape[0] == x.shape[0] && residual->shape[1] == x.shape[1] &&
                 residual->IsContiguous() && residual->device == x.device,
             "rmsnorm: residual must be f32/bf16 [T,H] contiguous on x's device");
  }
  VT_CHECK(x.device == out.device && weight.device == x.device && x.device == q.device,
           "rmsnorm: device mismatch (x/out/weight/queue)");
  reinterpret_cast<RmsNormFn>(GetOp(OpId::kRmsNorm, q.device.type))(q, out, x, weight, args,
                                                                    residual);
}

void ResidualRmsNorm(Queue& q, Tensor& out, const Tensor& a, const Tensor& base,
                     const Tensor* delta, const Tensor& weight,
                     const ResidualRmsNormArgs& args, Tensor* residual_out) {
  const auto& desc = args.descriptor;
  VT_CHECK(desc.expression == ResidualNormExpr::kAdd ||
               desc.expression == ResidualNormExpr::kDeltaPlusAdd,
           "residual_rmsnorm: invalid expression order");
  const bool triple = desc.expression == ResidualNormExpr::kDeltaPlusAdd;
  VT_CHECK(triple == (delta != nullptr), "residual_rmsnorm: expression operand count mismatch");
  VT_CHECK(desc.materialize_residual == (residual_out != nullptr),
           "residual_rmsnorm: materialization descriptor/output mismatch");
  VT_CHECK(!desc.output_alias_delta || triple,
           "residual_rmsnorm: output alias requires a delta operand");
  VT_CHECK(!desc.residual_alias_base || desc.materialize_residual,
           "residual_rmsnorm: base alias requires residual materialization");
  VT_CHECK(std::isfinite(args.eps) && args.eps >= 0.0f,
           "residual_rmsnorm: epsilon must be finite and nonnegative");
  VT_CHECK(a.rank == 2 && a.shape[0] >= 0 && a.shape[1] > 0,
           "residual_rmsnorm: expected nonnegative rows and positive hidden width");
  VT_CHECK(q.device.index >= 0, "residual_rmsnorm: queue device index must be nonnegative");
  const auto row_bytes = [&](const Tensor& tensor) -> size_t {
    VT_CHECK(tensor.rank == 2 && tensor.shape[0] == a.shape[0] &&
                 tensor.shape[1] == a.shape[1], "residual_rmsnorm: row shape mismatch");
    VT_CHECK(tensor.dtype == DType::kBF16, "residual_rmsnorm: BF16 activation/output required");
    VT_CHECK(tensor.device == q.device, "residual_rmsnorm: queue/tensor device mismatch");
    VT_CHECK(tensor.stride[1] == 1 && tensor.stride[0] >= tensor.shape[1],
             "residual_rmsnorm: unit inner stride and nonoverlapping rows required");
    if (tensor.shape[0] == 0) return 0;
    const auto rows = static_cast<uint64_t>(tensor.shape[0]);
    const auto width = static_cast<uint64_t>(tensor.shape[1]);
    const auto stride = static_cast<uint64_t>(tensor.stride[0]);
    const uint64_t max_elements = std::numeric_limits<size_t>::max() / sizeof(uint16_t);
    VT_CHECK(width <= max_elements && (rows - 1) <= (max_elements - width) / stride,
             "residual_rmsnorm: row storage span overflow");
    const size_t bytes = static_cast<size_t>((rows - 1) * stride + width) * sizeof(uint16_t);
    const uintptr_t start = reinterpret_cast<uintptr_t>(tensor.data);
    VT_CHECK(start != 0 && start % alignof(uint16_t) == 0 &&
                 start <= std::numeric_limits<uintptr_t>::max() - bytes,
             "residual_rmsnorm: invalid or unaligned row data");
    return bytes;
  };
  const size_t a_bytes = row_bytes(a), base_bytes = row_bytes(base), out_bytes = row_bytes(out);
  const size_t delta_bytes = delta == nullptr ? 0 : row_bytes(*delta);
  const size_t res_bytes = residual_out == nullptr ? 0 : row_bytes(*residual_out);
  VT_CHECK(weight.rank == 1 && weight.shape[0] == a.shape[1] && weight.stride[0] == 1 &&
               weight.dtype == DType::kBF16 && weight.device == q.device,
           "residual_rmsnorm: gamma must be contiguous BF16 [H] on the queue device");
  const uintptr_t w_start = reinterpret_cast<uintptr_t>(weight.data);
  const auto w_width = static_cast<uint64_t>(weight.shape[0]);
  VT_CHECK(w_width <= std::numeric_limits<size_t>::max() / sizeof(uint16_t),
           "residual_rmsnorm: gamma storage span overflow");
  const size_t w_bytes = static_cast<size_t>(w_width) * sizeof(uint16_t);
  VT_CHECK(w_start != 0 && w_start % alignof(uint16_t) == 0 &&
               w_start <= std::numeric_limits<uintptr_t>::max() - w_bytes,
           "residual_rmsnorm: invalid or unaligned gamma data");
  const auto overlaps = [](const Tensor& lhs, size_t lhs_bytes,
                            const Tensor& rhs, size_t rhs_bytes) {
    const uintptr_t left = reinterpret_cast<uintptr_t>(lhs.data);
    const uintptr_t right = reinterpret_cast<uintptr_t>(rhs.data);
    return lhs_bytes != 0 && rhs_bytes != 0 && left < right + rhs_bytes && right < left + lhs_bytes;
  };
  const auto same_view = [](const Tensor& lhs, const Tensor& rhs) {
    return lhs.data == rhs.data && lhs.stride[0] == rhs.stride[0];
  };
  for (const auto& input : {std::pair<const Tensor*, size_t>{&a, a_bytes},
                            {&base, base_bytes}, {delta, delta_bytes}, {&weight, w_bytes}}) {
    if (input.first == nullptr) continue;
    VT_CHECK(!overlaps(out, out_bytes, *input.first, input.second) ||
                 (input.first == delta && desc.output_alias_delta && same_view(out, *delta)),
             "residual_rmsnorm: normalized output overlaps a read-only input");
    if (residual_out != nullptr)
      VT_CHECK(!overlaps(*residual_out, res_bytes, *input.first, input.second) ||
                   (input.first == &base && desc.residual_alias_base && same_view(*residual_out, base)),
               "residual_rmsnorm: residual output overlaps a read-only input");
  }
  VT_CHECK(residual_out == nullptr || !overlaps(out, out_bytes, *residual_out, res_bytes),
           "residual_rmsnorm: outputs must not overlap");
  if (a.shape[0] == 0) return;
  reinterpret_cast<ResidualRmsNormFn>(GetOp(OpId::kResidualRmsNorm, q.device.type))(
      q, out, a, base, delta, weight, args, residual_out);
}

void RmsNormGroup(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                  const RmsNormGroupArgs& args) {
  VT_CHECK(x.rank == 2 && out.rank == 2 && weight.rank == 1,
           "rmsnorm_group: x/out rank-2, w rank-1");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1],
           "rmsnorm_group: shape mismatch");
  VT_CHECK(weight.shape[0] == x.shape[1], "rmsnorm_group: weight size mismatch");
  // group_size == 0 lands here rather than degenerating to a whole-row norm.
  // Upstream refuses the divisibility case by name (modeling_qwen4_exp.py:164-165
  // "hidden_size (...) must be divisible by group_size (...)"); the zero case is
  // ours, because upstream's `None` means "no grouping" and this op's default
  // must not silently mean that. See RmsNormGroupArgs::group_size.
  VT_CHECK(args.group_size >= 1,
           "rmsnorm_group: group_size must be >= 1; 0 is NOT 'the whole row' "
           "(that is vt::RmsNorm). Defaulting it to the whole row would make the "
           "most likely caller mistake indistinguishable from success, so the "
           "unset value is refused rather than interpreted");
  VT_CHECK(x.shape[1] % args.group_size == 0,
           "rmsnorm_group: group_size must divide the last dim "
           "(modeling_qwen4_exp.py:164-165)");
  VT_CHECK(args.eps > 0.0f, "rmsnorm_group: eps must be > 0");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(weight.dtype) && IsOutFloat(out.dtype),
           "rmsnorm_group: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous() && weight.IsContiguous(),
           "rmsnorm_group: contiguous required");
  VT_CHECK(x.device == out.device && weight.device == x.device && x.device == q.device,
           "rmsnorm_group: device mismatch (x/out/weight/queue)");
  reinterpret_cast<RmsNormGroupFn>(GetOp(OpId::kRmsNormGroup, q.device.type))(q, out, x, weight,
                                                                             args);
}

namespace {

// Fetch the tensor bound to operand slot `idx`, checked non-null.
Tensor* FusedOp(const FusedBinding& b, uint8_t idx, const char* what) {
  VT_CHECK(idx < b.n, "fused_chain: operand index out of range");
  VT_CHECK(b.op[idx] != nullptr, what);
  return b.op[idx];
}

// Tier-0 composite: walk the recipe DISPATCHING each opcode to the already-
// registered standalone vt:: op. Device-agnostic — every op self-dispatches on
// q.device, so the same walker realizes CPU and CUDA. Byte-exact by construction
// to the unfused standalone-op sequence the model hand-calls (that IS the golden).
// The residual-add idiom (kAdd writing the residual) folds into the following
// norm's RmsNorm(residual) call — the only form whose f32 add-then-normalize the
// standalone op reproduces bit-for-bit.
void FusedChainCompositeImpl(Queue& q, const FusedRecipe& r, const FusedBinding& b,
                             const FusedParams& p) {
  Tensor* add_x = nullptr;    // pending residual-add: x operand
  Tensor* add_res = nullptr;  // pending residual-add: residual operand (also the out)
  bool add_pending = false;
  for (int s = 0; s < r.n; ++s) {
    const FStep& st = r.steps[s];
    switch (st.op) {
      case FOp::kResidualRmsNorm: {
        VT_CHECK(!add_pending, "fused_chain: residual expression cannot consume a pending add");
        const bool triple = st.residual_norm.expression == ResidualNormExpr::kDeltaPlusAdd;
        VT_CHECK(st.nin == (triple ? 4 : 3), "fused_chain: residual expression input count");
        VT_CHECK((st.out2 != kNoOperand) == st.residual_norm.materialize_residual,
                 "fused_chain: residual expression materialization mismatch");
        VT_CHECK(st.reduce == FReduce::kMeanSquare && !st.gemma && !st.sigmoid_gate &&
                     !st.norm_full_width, "fused_chain: invalid residual expression modifiers");
        Tensor* out = FusedOp(b, st.out, "fused_chain: null residual norm output");
        Tensor* a = FusedOp(b, st.in[0], "fused_chain: null residual expression a");
        Tensor* base = FusedOp(b, st.in[1], "fused_chain: null residual expression base");
        Tensor* delta = triple ? FusedOp(b, st.in[2], "fused_chain: null delta") : nullptr;
        Tensor* weight = FusedOp(b, st.in[triple ? 3 : 2], "fused_chain: null residual gamma");
        Tensor* residual = st.out2 == kNoOperand ? nullptr : FusedOp(b, st.out2, "fused_chain: null materialized residual");
        ResidualRmsNorm(q, *out, *a, *base, delta, *weight,
                        ResidualRmsNormArgs{p.eps, st.residual_norm}, residual);
        break;
      }
      case FOp::kAdd:
        // Residual-add producing the residual stream: fold into the next kRmsNorm.
        VT_CHECK(st.nin == 2 && st.out == st.in[1],
                 "fused_chain composite: kAdd must be residual-add (out==in[1])");
        add_x = FusedOp(b, st.in[0], "fused_chain: null add input");
        add_res = FusedOp(b, st.out, "fused_chain: null residual");
        add_pending = true;
        break;
      case FOp::kRmsNorm: {
        Tensor* out = FusedOp(b, st.out, "fused_chain: null rmsnorm out");
        Tensor* w = FusedOp(b, st.in[1], "fused_chain: null rmsnorm weight");
        if (add_pending) {
          RmsNorm(q, *out, *add_x, *w, RmsNormArgs{p.eps, st.gemma}, add_res);
          add_pending = false;
        } else {
          Tensor* a = FusedOp(b, st.in[0], "fused_chain: null rmsnorm input");
          RmsNorm(q, *out, *a, *w, RmsNormArgs{p.eps, st.gemma}, nullptr);
        }
        break;
      }
      case FOp::kRmsNormGated: {
        Tensor* out = FusedOp(b, st.out, "fused_chain: null gated-norm out");
        Tensor* x = FusedOp(b, st.in[0], "fused_chain: null gated-norm x");
        Tensor* gate = FusedOp(b, st.in[1], "fused_chain: null gated-norm gate");
        Tensor* w = FusedOp(b, st.in[2], "fused_chain: null gated-norm weight");
        RmsNormGated(q, *out, *x, *gate, *w, RmsNormGatedArgs{p.eps, st.sigmoid_gate});
        break;
      }
      case FOp::kSiluMul: {
        Tensor* out = FusedOp(b, st.out, "fused_chain: null silu_mul out");
        Tensor* gate = FusedOp(b, st.in[0], "fused_chain: null silu_mul gate");
        Tensor* up = FusedOp(b, st.in[1], "fused_chain: null silu_mul up");
        MoeSiluMul(q, *out, *gate, *up);
        break;
      }
      case FOp::kSigmoidGate: {
        Tensor* out = FusedOp(b, st.out, "fused_chain: null sigmoid_gate out");
        Tensor* attn = FusedOp(b, st.in[0], "fused_chain: null sigmoid_gate attn");
        Tensor* gate = FusedOp(b, st.in[1], "fused_chain: null sigmoid_gate gate");
        SigmoidGateBf16(q, *out, *attn, *gate);
        break;
      }
      case FOp::kRope: {
        Tensor* qs = FusedOp(b, st.out, "fused_chain: null rope q");
        Tensor* cos_sin = FusedOp(b, st.in[1], "fused_chain: null rope cos_sin cache");
        Tensor* pos = FusedOp(b, st.in[2], "fused_chain: null rope positions");
        Tensor* ks = (st.out2 == kNoOperand) ? nullptr : b.op[st.out2];
        RopeFromCache(q, *qs, ks, *pos, *cos_sin, p.rope);
        break;
      }
      case FOp::kQuantFp8: {
        Tensor* out = FusedOp(b, st.out, "fused_chain: null fp8 out");
        Tensor* a = FusedOp(b, st.in[0], "fused_chain: null fp8 input");
        QuantFp8Static(q, *out, *a, p.quant_scale);
        break;
      }
      case FOp::kQuantFp4: {
        Tensor* packed = FusedOp(b, st.out, "fused_chain: null fp4 packed out");
        VT_CHECK(st.out2 != kNoOperand, "fused_chain: kQuantFp4 needs an out_scale (out2)");
        Tensor* scale = FusedOp(b, st.out2, "fused_chain: null fp4 scale out");
        Tensor* a = FusedOp(b, st.in[0], "fused_chain: null fp4 input");
        ScaledFp4Quant(q, *packed, *scale, *a, p.quant_scale, p.fp4_layout);
        break;
      }
      case FOp::kAttnQkNormRopeGate: {
        // Composite MACRO: dispatch the whole fused preamble to the one standalone
        // op. Operand order is fixed (recipes.h): [qgate, kf, q_norm, k_norm,
        // cos_sin, q_out, k_out, gate_out].
        VT_CHECK(r.n_operands == 8 && b.n == 8, "fused_chain: attn preamble needs 8 operands");
        AttnQkNormRopeGate(q, *FusedOp(b, 5, "q_out"), *FusedOp(b, 6, "k_out"),
                           *FusedOp(b, 7, "gate_out"), *FusedOp(b, 0, "qgate"),
                           *FusedOp(b, 1, "kf"), *FusedOp(b, 2, "q_norm"),
                           *FusedOp(b, 3, "k_norm"), *FusedOp(b, 4, "cos_sin"),
                           RmsNormArgs{p.eps, st.gemma}, p.rope);
        break;
      }
      case FOp::kMul:
      case FOp::kSilu:
      case FOp::kSigmoid:
        VT_CHECK(false,
                 "fused_chain composite: granular kMul/kSilu/kSigmoid have no standalone op "
                 "(Tier-1 vocabulary only)");
        break;
    }
  }
  VT_CHECK(!add_pending, "fused_chain composite: residual-add without a following rmsnorm");
}

// FAST realization (W2): dispatch a recipe bound to a bespoke single-launch fused
// kernel (recipe.fast_op) to that kernel via its existing vt:: wrapper. One switch
// case per fast realization — the additive surface (O(1), mirrors the composite's
// per-opcode switch). Each case unpacks the recipe's indexed operand table into the
// wrapper's arguments; the wrapper self-dispatches on q.device. Byte-exact to the
// composite by construction (it IS the kernel the composite is validated against).
void DispatchFusedFast(Queue& q, const FusedRecipe& r, const FusedBinding& b,
                       const FusedParams& p) {
  switch (static_cast<OpId>(r.fast_op)) {
    case OpId::kRmsNormQuantFp8: {
      // operands: 0=x, 1=weight, 2=residual?, 3=tmp_bf16 (optional out_bf16), 4=out_fp8
      Tensor* out_fp8 = FusedOp(b, 4, "fused_chain fast: null fp8 out");
      Tensor* x = FusedOp(b, 0, "fused_chain fast: null x");
      Tensor* w = FusedOp(b, 1, "fused_chain fast: null weight");
      Tensor* residual = b.op[2];  // optional
      Tensor* out_bf16 = b.op[3];  // optional (bf16 normed activation consumer)
      RmsNormQuantFp8(q, *out_fp8, out_bf16, *x, *w, RmsNormArgs{p.eps, r.steps[1].gemma},
                      residual, p.quant_scale);
      break;
    }
    case OpId::kAttnQkNormRope: {
      // operands: 0=q2[T*Hq,Dh] 1=q_norm 2=k2[T*Hkv,Dh] 3=k_norm
      //           4=q3[T,Hq,Dh] 5=k3[T,Hkv,Dh] 6=cos_sin 7=positions
      // The fused kernel norms and rotates q3/k3 IN PLACE; slots 0 and 2 are the
      // 2-D aliases of the same buffers and are not passed on.
      Tensor* q3 = FusedOp(b, 4, "fused_chain fast: null qk-norm-rope q3");
      Tensor* k3 = FusedOp(b, 5, "fused_chain fast: null qk-norm-rope k3");
      Tensor* qw = FusedOp(b, 1, "fused_chain fast: null qk-norm-rope q_norm");
      Tensor* kw = FusedOp(b, 3, "fused_chain fast: null qk-norm-rope k_norm");
      Tensor* cs = FusedOp(b, 6, "fused_chain fast: null qk-norm-rope cos_sin");
      Tensor* po = FusedOp(b, 7, "fused_chain fast: null qk-norm-rope positions");
      reinterpret_cast<AttnQkNormRopeFn>(GetOp(OpId::kAttnQkNormRope, q.device.type))(
          q, *q3, *k3, *qw, *kw, *cs, *po, RmsNormArgs{p.eps, r.steps[0].gemma}, p.rope);
      break;
    }
    case OpId::kRmsNormGatedQuantFp8: {
      // operands: 0=x, 1=gate, 2=weight, 3=tmp_bf16 (unused), 4=out_fp8
      Tensor* out_fp8 = FusedOp(b, 4, "fused_chain fast: null gated fp8 out");
      Tensor* x = FusedOp(b, 0, "fused_chain fast: null gated x");
      Tensor* gate = FusedOp(b, 1, "fused_chain fast: null gated gate");
      Tensor* w = FusedOp(b, 2, "fused_chain fast: null gated weight");
      RmsNormGatedQuantFp8(q, *out_fp8, *x, *gate, *w,
                           RmsNormGatedArgs{p.eps, r.steps[0].sigmoid_gate}, p.quant_scale);
      break;
    }
    case OpId::kSiluMulFp4Quant: {
      // operands: 0=gate, 1=up, 2=tmp_bf16 (unused), 3=out_packed, 4=out_scale
      Tensor* packed = FusedOp(b, 3, "fused_chain fast: null silu_mul packed");
      Tensor* scale = FusedOp(b, 4, "fused_chain fast: null silu_mul scale");
      Tensor* gate = FusedOp(b, 0, "fused_chain fast: null silu_mul gate");
      Tensor* up = FusedOp(b, 1, "fused_chain fast: null silu_mul up");
      SiluMulFp4Quant(q, *packed, *scale, *gate, *up, p.quant_scale, p.fp4_layout);
      break;
    }
    case OpId::kSigmoidGateFp4Quant: {
      // operands: 0=attn, 1=gate, 2=tmp_bf16 (unused), 3=out_packed, 4=out_scale
      Tensor* packed = FusedOp(b, 3, "fused_chain fast: null sigmoid_gate packed");
      Tensor* scale = FusedOp(b, 4, "fused_chain fast: null sigmoid_gate scale");
      Tensor* attn = FusedOp(b, 0, "fused_chain fast: null sigmoid_gate attn");
      Tensor* gate = FusedOp(b, 1, "fused_chain fast: null sigmoid_gate gate");
      SigmoidGateFp4Quant(q, *packed, *scale, *attn, *gate, p.quant_scale, p.fp4_layout);
      break;
    }
    default:
      VT_CHECK(false, "fused_chain: recipe.fast_op has no fast-realization case");
  }
}

}  // namespace

void FusedChainComposite(Queue& q, const FusedRecipe& recipe, const FusedBinding& binding,
                         const FusedParams& params) {
  VT_CHECK(recipe.n >= 1 && recipe.n <= kMaxFusedSteps, "fused_chain: empty/oversized recipe");
  VT_CHECK(recipe.n_operands == binding.n && binding.n <= kMaxFusedOperands,
           "fused_chain: binding size must match recipe operand count");
  FusedChainCompositeImpl(q, recipe, binding, params);
}

void FusedChain(Queue& q, const FusedRecipe& recipe, const FusedBinding& binding,
                const FusedParams& params) {
  VT_CHECK(recipe.n >= 1 && recipe.n <= kMaxFusedSteps, "fused_chain: empty/oversized recipe");
  VT_CHECK(recipe.n_operands == binding.n && binding.n <= kMaxFusedOperands,
           "fused_chain: binding size must match recipe operand count");

  // FAST realization: a recipe bound to an existing bespoke fused kernel dispatches
  // to it whenever the backend provides that OpId — the SAME single-launch kernel the
  // model called directly before W2 migration, so this is perf-neutral by
  // construction (no extra kernel, no getenv, no allocation on the path). A backend
  // that lacks the fast kernel falls through to the byte-exact composite below.
  if (recipe.fast_op != kNoFastOp &&
      OpRegistered(static_cast<OpId>(recipe.fast_op), q.device.type)) {
    DispatchFusedFast(q, recipe, binding, params);
    return;
  }

  // Tier-1 interpreter path: only for Tier-1-able recipes (all elementwise/rms)
  // over the canonical operand order [x, weight, residual?, out]. Everything else
  // (quant/rope/gated/attn) runs through the byte-exact composite.
  if (RecipeIsTier1Able(recipe) && FusedTier() == 1) {
    Tensor* x = FusedOp(binding, 0, "fused_chain: null x");
    Tensor* weight = FusedOp(binding, 1, "fused_chain: null weight");
    Tensor* residual =
        (binding.n >= 3 && recipe.operands[2].kind == FKind::kResidual) ? binding.op[2] : nullptr;
    Tensor* out = FusedOp(binding, static_cast<uint8_t>(recipe.steps[recipe.n - 1].out),
                          "fused_chain: null out");
    reinterpret_cast<FusedChainFn>(GetOp(OpId::kFusedChain, q.device.type))(
        q, *out, *x, *weight, residual, recipe, params.eps);
    return;
  }
  FusedChainComposite(q, recipe, binding, params);
}

void FusedChain(Queue& q, Tensor& out, const Tensor& a, const Tensor& base,
                const Tensor* delta, const Tensor& weight,
                const ResidualRmsNormArgs& args, Tensor* residual_out) {
  FusedBinding binding{};
  binding.n = 6;
  binding.op[0] = const_cast<Tensor*>(&a);
  binding.op[1] = const_cast<Tensor*>(&base);
  binding.op[2] = const_cast<Tensor*>(delta);
  binding.op[3] = const_cast<Tensor*>(&weight);
  binding.op[4] = &out;
  binding.op[5] = residual_out;
  // Preserve mismatched optional arguments until the typed validation can reject
  // them. A recipe cannot silently hide an unexpected delta or residual output.
  VT_CHECK((delta != nullptr) == (args.descriptor.expression == ResidualNormExpr::kDeltaPlusAdd),
           "fused_chain: residual expression operand count mismatch");
  VT_CHECK((residual_out != nullptr) == args.descriptor.materialize_residual,
           "fused_chain: residual expression materialization mismatch");
  FusedParams params{};
  params.eps = args.eps;
  FusedChain(q, ResidualRmsNormRecipe(args.descriptor), binding, params);
}

void FusedChain(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight, Tensor* residual,
                const FusedRecipe& recipe, float eps) {
  // Canonical 4-operand shape (the W0-adopted kFusedAddRmsNorm site): x/out [T,H],
  // weight [H], optional residual [T,H] f32/bf16. Validate at the chokepoint so a
  // bad call fails here, not inside a kernel, then bind [x, weight, residual, out]
  // and forward to the general entry.
  VT_CHECK(x.rank == 2 && out.rank == 2 && weight.rank == 1,
           "fused_chain: x/out rank-2, weight rank-1");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1],
           "fused_chain: out shape must match x");
  VT_CHECK(weight.shape[0] == x.shape[1], "fused_chain: weight size mismatch");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(weight.dtype) && IsOutFloat(out.dtype),
           "fused_chain: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous() && weight.IsContiguous(),
           "fused_chain: contiguous required");
  if (residual != nullptr) {
    VT_CHECK((residual->dtype == DType::kF32 || residual->dtype == DType::kBF16) &&
                 residual->rank == 2 && residual->shape[0] == x.shape[0] &&
                 residual->shape[1] == x.shape[1] && residual->IsContiguous() &&
                 residual->device == x.device,
             "fused_chain: residual must be f32/bf16 [T,H] contiguous on x's device");
  }
  VT_CHECK(x.device == out.device && weight.device == x.device && x.device == q.device,
           "fused_chain: device mismatch (x/out/weight/queue)");
  FusedBinding binding;
  binding.op[0] = const_cast<Tensor*>(&x);
  binding.op[1] = const_cast<Tensor*>(&weight);
  binding.op[2] = residual;
  binding.op[3] = &out;
  binding.n = 4;
  FusedChain(q, recipe, binding, FusedParams{eps, 1.0f, Fp4ScaleLayout::kLinear, RopeArgs{}});
}

void FusedChain(Queue& q, const FusedRecipe& recipe, Tensor& out_packed, Tensor& out_scale,
                const Tensor& a, const Tensor& b, float quant_scale,
                Fp4ScaleLayout scale_layout) {
  // Fp4-activation-quant shape (kSiluMulFp4Quant, kSigmoidGateFp4Quant): bind the
  // recipe's [a, b, tmp_bf16, out_packed, out_scale] table (tmp_bf16 = nullptr; the
  // fast realization never materializes it) and dispatch to the recipe's fast_op.
  FusedBinding binding;
  binding.op[0] = const_cast<Tensor*>(&a);
  binding.op[1] = const_cast<Tensor*>(&b);
  binding.op[2] = nullptr;  // tmp_bf16 (composite-only scratch)
  binding.op[3] = &out_packed;
  binding.op[4] = &out_scale;
  binding.n = 5;
  FusedChain(q, recipe, binding, FusedParams{1e-6f, quant_scale, scale_layout, RopeArgs{}});
}

void FusedChain(Queue& q, const FusedRecipe& recipe, Tensor& out_fp8, Tensor* out_bf16,
                const Tensor& x, const Tensor& weight, Tensor* residual, float eps,
                float input_scale) {
  // RmsNorm->fp8 shape (kRmsNormQuantFp8): bind [x, weight, residual, out_bf16,
  // out_fp8]. residual / out_bf16 pass through as-is (may be nullptr), matching the
  // bespoke RmsNormQuantFp8 contract; tmp_bf16 slot IS the optional out_bf16 here.
  FusedBinding binding;
  binding.op[0] = const_cast<Tensor*>(&x);
  binding.op[1] = const_cast<Tensor*>(&weight);
  binding.op[2] = residual;
  binding.op[3] = out_bf16;
  binding.op[4] = &out_fp8;
  binding.n = 5;
  FusedChain(q, recipe, binding, FusedParams{eps, input_scale, Fp4ScaleLayout::kLinear, RopeArgs{}});
}

void FusedChain(Queue& q, const FusedRecipe& recipe, Tensor& out_fp8, const Tensor& x,
                const Tensor& gate, const Tensor& weight, float eps, float input_scale) {
  // Gated-RmsNorm->fp8 shape (kRmsNormGatedQuantFp8): bind [x, gate, weight,
  // tmp_bf16, out_fp8] (tmp_bf16 = nullptr; fast realization skips it).
  FusedBinding binding;
  binding.op[0] = const_cast<Tensor*>(&x);
  binding.op[1] = const_cast<Tensor*>(&gate);
  binding.op[2] = const_cast<Tensor*>(&weight);
  binding.op[3] = nullptr;  // tmp_bf16 (composite-only scratch)
  binding.op[4] = &out_fp8;
  binding.n = 5;
  FusedChain(q, recipe, binding, FusedParams{eps, input_scale, Fp4ScaleLayout::kLinear, RopeArgs{}});
}

void FusedChain(Queue& q, const FusedRecipe& recipe, Tensor& q_out, Tensor& k_out,
                Tensor& gate_out, const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                const Tensor& k_norm, const Tensor& cos_sin, float eps, const RopeArgs& rope) {
  // Attn-preamble MACRO shape (kAttnQkNormRopeGate): bind the fixed operand table
  // [qgate, kf, q_norm, k_norm, cos_sin, q_out, k_out, gate_out]. No fast_op — the
  // composite macro dispatches to the single vt::AttnQkNormRopeGate kernel.
  FusedBinding binding;
  binding.op[0] = const_cast<Tensor*>(&qgate);
  binding.op[1] = const_cast<Tensor*>(&kf);
  binding.op[2] = const_cast<Tensor*>(&q_norm);
  binding.op[3] = const_cast<Tensor*>(&k_norm);
  binding.op[4] = const_cast<Tensor*>(&cos_sin);
  binding.op[5] = &q_out;
  binding.op[6] = &k_out;
  binding.op[7] = &gate_out;
  binding.n = 8;
  FusedChain(q, recipe, binding, FusedParams{eps, 1.0f, Fp4ScaleLayout::kLinear, rope});
}

void SiluAndMul(Queue& q, Tensor& out, const Tensor& x) {
  VT_CHECK(x.rank == 2 && out.rank == 2, "silu_and_mul: rank-2 required");
  VT_CHECK(x.shape[1] % 2 == 0, "silu_and_mul: inner dim must be even");
  VT_CHECK(out.shape[0] == x.shape[0] && out.shape[1] == x.shape[1] / 2,
           "silu_and_mul: output shape mismatch");
  VT_CHECK(IsFloat(x.dtype) && IsOutFloat(out.dtype), "silu_and_mul: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(), "silu_and_mul: contiguous required");
  VT_CHECK(x.device == out.device && x.device == q.device, "silu_and_mul: device mismatch");
  reinterpret_cast<SiluAndMulFn>(GetOp(OpId::kSiluAndMul, q.device.type))(q, out, x);
}

void ClampedSwiGLU(Queue& q, Tensor& out, const Tensor& gate_up, float limit) {
  VT_CHECK(gate_up.rank == 2 && out.rank == 2, "clamped_swiglu: rank-2 required");
  VT_CHECK(gate_up.shape[1] % 2 == 0, "clamped_swiglu: inner dim must be even");
  VT_CHECK(out.shape[0] == gate_up.shape[0] && out.shape[1] == gate_up.shape[1] / 2,
           "clamped_swiglu: output shape mismatch");
  VT_CHECK(IsFloat(gate_up.dtype) && IsOutFloat(out.dtype),
           "clamped_swiglu: float in, f32/bf16 out");
  VT_CHECK(gate_up.IsContiguous() && out.IsContiguous(), "clamped_swiglu: contiguous required");
  VT_CHECK(gate_up.device == out.device && gate_up.device == q.device,
           "clamped_swiglu: device mismatch");
  reinterpret_cast<ClampedSwiGLUFn>(GetOp(OpId::kClampedSwiGLU, q.device.type))(
      q, out, gate_up, limit);
}

void GeluAndMul(Queue& q, Tensor& out, const Tensor& x) {
  VT_CHECK(x.rank == 2 && out.rank == 2, "gelu_and_mul: rank-2 required");
  VT_CHECK(x.shape[1] % 2 == 0, "gelu_and_mul: inner dim must be even");
  VT_CHECK(out.shape[0] == x.shape[0] && out.shape[1] == x.shape[1] / 2,
           "gelu_and_mul: output shape mismatch");
  VT_CHECK(IsFloat(x.dtype) && IsOutFloat(out.dtype), "gelu_and_mul: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(), "gelu_and_mul: contiguous required");
  VT_CHECK(x.device == out.device && x.device == q.device, "gelu_and_mul: device mismatch");
  reinterpret_cast<GeluAndMulFn>(GetOp(OpId::kGeluAndMul, q.device.type))(q, out, x);
}

void MulScalar(Queue& q, Tensor& out, const Tensor& x, double scalar) {
  VT_CHECK(out.rank == x.rank, "mul_scalar: out rank must match x");
  for (int i = 0; i < x.rank; ++i)
    VT_CHECK(out.shape[i] == x.shape[i], "mul_scalar: out shape must match x");
  VT_CHECK(IsFloat(x.dtype) && IsOutFloat(out.dtype), "mul_scalar: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(), "mul_scalar: contiguous required");
  VT_CHECK(x.device == out.device && x.device == q.device, "mul_scalar: device mismatch");
  reinterpret_cast<MulScalarFn>(GetOp(OpId::kMulScalar, q.device.type))(q, out, x, scalar);
}

void SoftCap(Queue& q, Tensor& out, const Tensor& x, double cap) {
  VT_CHECK(out.rank == x.rank, "soft_cap: out rank must match x");
  for (int i = 0; i < x.rank; ++i)
    VT_CHECK(out.shape[i] == x.shape[i], "soft_cap: out shape must match x");
  VT_CHECK(cap > 0.0, "soft_cap: cap must be > 0");
  VT_CHECK(IsFloat(x.dtype) && IsOutFloat(out.dtype), "soft_cap: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(), "soft_cap: contiguous required");
  VT_CHECK(x.device == out.device && x.device == q.device, "soft_cap: device mismatch");
  reinterpret_cast<SoftCapFn>(GetOp(OpId::kSoftCap, q.device.type))(q, out, x, cap);
}

void LayerNorm(Queue& q, Tensor& out, const Tensor& x, const Tensor* weight,
               const Tensor* bias, const LayerNormArgs& args) {
  VT_CHECK(x.rank >= 1 && out.rank == x.rank, "layer_norm: out rank must match x");
  for (int i = 0; i < x.rank; ++i)
    VT_CHECK(out.shape[i] == x.shape[i], "layer_norm: out shape must match x");
  const int64_t d = x.shape[x.rank - 1];
  VT_CHECK(d > 0, "layer_norm: last dim must be non-empty");
  VT_CHECK(IsFloat(x.dtype) && IsOutFloat(out.dtype), "layer_norm: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(), "layer_norm: contiguous required");
  VT_CHECK(x.device == out.device && x.device == q.device, "layer_norm: device mismatch");
  for (const Tensor* p : {weight, bias}) {
    if (p == nullptr) continue;
    VT_CHECK(p->rank == 1 && p->shape[0] == d,
             "layer_norm: weight/bias must be rank-1 [D] matching x's last dim");
    VT_CHECK(IsFloat(p->dtype), "layer_norm: float weight/bias required");
    VT_CHECK(p->IsContiguous() && p->device == x.device,
             "layer_norm: weight/bias must be contiguous on x's device");
  }
  VT_CHECK(args.eps >= 0.0f, "layer_norm: eps must be non-negative");
  reinterpret_cast<LayerNormFn>(GetOp(OpId::kLayerNorm, q.device.type))(q, out, x, weight, bias,
                                                                       args);
}

void Relu(Queue& q, Tensor& out, const Tensor& x) {
  VT_CHECK(out.rank == x.rank, "relu: out rank must match x");
  for (int i = 0; i < x.rank; ++i)
    VT_CHECK(out.shape[i] == x.shape[i], "relu: out shape must match x");
  VT_CHECK(IsFloat(x.dtype) && IsOutFloat(out.dtype), "relu: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(), "relu: contiguous required");
  VT_CHECK(x.device == out.device && x.device == q.device, "relu: device mismatch");
  reinterpret_cast<ReluFn>(GetOp(OpId::kRelu, q.device.type))(q, out, x);
}

void GeluTanh(Queue& q, Tensor& out, const Tensor& x) {
  VT_CHECK(out.rank == x.rank, "gelu_tanh: out rank must match x");
  for (int i = 0; i < x.rank; ++i)
    VT_CHECK(out.shape[i] == x.shape[i], "gelu_tanh: out shape must match x");
  VT_CHECK(IsFloat(x.dtype) && IsOutFloat(out.dtype), "gelu_tanh: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(), "gelu_tanh: contiguous required");
  VT_CHECK(x.device == out.device && x.device == q.device, "gelu_tanh: device mismatch");
  reinterpret_cast<ReluFn>(GetOp(OpId::kGeluTanh, q.device.type))(q, out, x);
}

void GeluErf(Queue& q, Tensor& out, const Tensor& x) {
  VT_CHECK(out.rank == x.rank, "gelu_erf: out rank must match x");
  for (int i = 0; i < x.rank; ++i)
    VT_CHECK(out.shape[i] == x.shape[i], "gelu_erf: out shape must match x");
  VT_CHECK(IsFloat(x.dtype) && IsOutFloat(out.dtype), "gelu_erf: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(), "gelu_erf: contiguous required");
  VT_CHECK(x.device == out.device && x.device == q.device, "gelu_erf: device mismatch");
  reinterpret_cast<ReluFn>(GetOp(OpId::kGeluErf, q.device.type))(q, out, x);
}

void Add(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  VT_CHECK(a.rank >= 1 && out.rank == a.rank, "add: out rank must match a");
  for (int i = 0; i < a.rank; ++i)
    VT_CHECK(out.shape[i] == a.shape[i], "add: out shape must match a");
  const int64_t d = a.shape[a.rank - 1];
  // Two accepted b shapes: a's exact shape (elementwise) or rank-1 [D] over the
  // last dim (a nn.Linear bias row-broadcast).
  const bool broadcast = b.rank == 1 && a.rank != 1;
  if (broadcast) {
    VT_CHECK(b.shape[0] == d, "add: rank-1 b must match a's last dim (bias broadcast)");
  } else {
    VT_CHECK(b.rank == a.rank, "add: b must match a's rank or be rank-1 [D]");
    for (int i = 0; i < a.rank; ++i)
      VT_CHECK(b.shape[i] == a.shape[i], "add: b shape must match a");
  }
  VT_CHECK(IsFloat(a.dtype) && IsFloat(b.dtype) && IsOutFloat(out.dtype),
           "add: float in, f32/bf16 out");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "add: contiguous required");
  VT_CHECK(a.device == out.device && b.device == a.device && a.device == q.device,
           "add: device mismatch");
  reinterpret_cast<AddFn>(GetOp(OpId::kAdd, q.device.type))(q, out, a, b);
}

void Embedding(Queue& q, Tensor& out, const Tensor& table, const Tensor& ids) {
  ValidateWeightValues(table, ids, out, q);
  VT_CHECK(table.rank == 2 && ids.rank == 1 && out.rank == 2, "embedding: bad ranks");
  VT_CHECK(out.shape[0] == ids.shape[0] && out.shape[1] == table.shape[1],
           "embedding: output shape mismatch");
  VT_CHECK(ids.dtype == DType::kI32 || ids.dtype == DType::kI64, "embedding: ids i32/i64");
  // A BLOCK-QUANTIZED table is admitted alongside the elementwise ones: the
  // kernel then dequantizes ONE ROW per gathered id instead of loading it,
  // mirroring `ggml_compute_forward_get_rows_q` (llama.cpp @ b10451
  // ggml/src/ggml-cpu/ops.cpp:4850), which dispatches every quantized get_rows
  // through the type's `to_float`. This is what lets a gather table stay
  // COMPRESSED in memory; a 20 M-entry n-gram table has no other affordable
  // residency (Qwen3.8-Flash-Next: 28.8 GB of IQ4_NL against 102.4 GB of bf16).
  VT_CHECK(IsFloat(table.dtype) || IsBlockQuant(table.dtype),
           "embedding: table must be float or block-quantized");
  VT_CHECK(IsOutFloat(out.dtype), "embedding: f32/bf16 out");
  // `ggml_row_size` asserts a row is a whole number of blocks; a ragged K has no
  // row stride at all, so it refuses here rather than mis-striding every row
  // after the first. (This is also the reason the shipped table is IQ4_NL:
  // its row is 160, and no 256-element K-quant can encode it.)
  if (IsBlockQuant(table.dtype)) {
    VT_CHECK(table.shape[1] % BlockElems(table.dtype) == 0,
             "embedding: block table K must be a whole number of blocks");
  }
  // A block table's strides are logical (elements), exactly as for a
  // `kMatmulBTQuant` weight: they describe [V, K] row-major, and the kernel
  // converts to bytes through RowSizeBytes.
  VT_CHECK(table.IsContiguous() && ids.IsContiguous() && out.IsContiguous(),
           "embedding: contiguous required");
  VT_CHECK(table.device == out.device && ids.device == table.device && table.device == q.device,
           "embedding: device mismatch (table/out/ids/queue)");
  // A block table is a DIFFERENT capability, so it is a different op id. The
  // split is what lets the GGUF residency policy ask "can this device decode a
  // block row" through the provider table instead of naming devices in the
  // loader, and it is what makes a device that lacks the arm refuse BY NAME
  // here -- GetOp throws naming the op and the device -- rather than dispatch
  // into a kernel that would assert on the dtype one frame later.
  const OpId op = IsBlockQuant(table.dtype) ? OpId::kEmbeddingQuant : OpId::kEmbedding;
  reinterpret_cast<EmbeddingFn>(ResolveWeightOp(op, q, table))(q, out, table, ids);
}

void RopeNeox(Queue& q, Tensor& q_states, Tensor& k_states, const Tensor& positions,
              const RopeArgs& args) {
  VT_CHECK(q_states.rank == 3 && k_states.rank == 3, "rope: q/k rank-3 [T,H,D]");
  VT_CHECK(q_states.shape[0] == k_states.shape[0] && q_states.shape[2] == k_states.shape[2],
           "rope: q/k token count and head_dim must match");
  VT_CHECK(positions.rank == 1 && positions.shape[0] == q_states.shape[0],
           "rope: positions[T] mismatch");
  VT_CHECK(positions.dtype == DType::kI32 || positions.dtype == DType::kI64,
           "rope: positions i32/i64");
  VT_CHECK(IsOutFloat(q_states.dtype) && k_states.dtype == q_states.dtype,
           "rope: q/k must be f32 or bf16, same dtype");
  VT_CHECK(args.rotary_dim > 0 && args.rotary_dim % 2 == 0 &&
               args.rotary_dim <= q_states.shape[2],
           "rope: rotary_dim must be even and <= head_dim");
  VT_CHECK(q_states.IsContiguous() && k_states.IsContiguous() && positions.IsContiguous(),
           "rope: contiguous required");
  VT_CHECK(q_states.device == q.device && k_states.device == q.device &&
               positions.device == q.device,
           "rope: device mismatch (q/k/positions/queue)");
  reinterpret_cast<RopeFn>(GetOp(OpId::kRopeNeox, q.device.type))(q, q_states, k_states,
                                                                  positions, args);
}

void RopeFromCache(Queue& q, Tensor& q_states, Tensor* k_states,
                   const Tensor& positions, const Tensor& cos_sin_cache,
                   const RopeArgs& args) {
  VT_CHECK(q_states.rank == 3, "rope_from_cache: q must be rank-3 [T,H,D]");
  const int64_t tokens = q_states.shape[0];
  const int64_t head_dim = q_states.shape[2];
  if (k_states != nullptr) {
    VT_CHECK(k_states->rank == 3 && k_states->shape[0] == tokens &&
                 k_states->shape[2] == head_dim,
             "rope_from_cache: k must be [T,Hk,D] matching q");
    VT_CHECK(k_states->dtype == q_states.dtype,
             "rope_from_cache: q/k dtype mismatch");
    // MLA campaign W6: only the innermost dim must be unit-stride. DeepSeek's
    // k_pe is the trailing column block of the single fused
    // `kv_a_proj_with_mqa` output (deepseek_v2.py:511, mla.py:155-162), i.e. a
    // STRIDED view; requiring contiguity would force a copy per layer per step.
    VT_CHECK(k_states->stride[2] == 1,
             "rope_from_cache: k innermost dimension must be unit-stride");
    VT_CHECK(k_states->device == q.device,
             "rope_from_cache: k/queue device mismatch");
  }
  VT_CHECK(positions.rank == 1 || positions.rank == 2,
           "rope_from_cache: positions must be [T] or [3,T]");
  if (positions.rank == 1) {
    VT_CHECK(positions.shape[0] == tokens,
             "rope_from_cache: positions[T] mismatch");
  } else {
    VT_CHECK(positions.shape[0] == 3 && positions.shape[1] == tokens,
             "rope_from_cache: MRoPE positions must be [3,T]");
    int64_t section_sum = 0;
    for (int32_t section : args.mrope_section) {
      VT_CHECK(section >= 0,
               "rope_from_cache: negative mrope_section entry");
      section_sum += section;
    }
    VT_CHECK(section_sum == args.rotary_dim / 2,
             "rope_from_cache: mrope_section must sum to rotary_dim/2");
  }
  VT_CHECK(positions.dtype == DType::kI32 || positions.dtype == DType::kI64,
           "rope_from_cache: positions must be i32/i64");
  VT_CHECK(cos_sin_cache.rank == 2 && cos_sin_cache.shape[0] > 0 &&
               cos_sin_cache.shape[1] == args.rotary_dim,
           "rope_from_cache: cache must be [P,rotary_dim]");
  VT_CHECK(IsOutFloat(q_states.dtype) &&
               cos_sin_cache.dtype == q_states.dtype,
           "rope_from_cache: q/k/cache must share f32 or bf16 dtype");
  VT_CHECK(args.rotary_dim > 0 && args.rotary_dim % 2 == 0 &&
               args.rotary_dim <= head_dim,
           "rope_from_cache: rotary_dim must be even and <= head_dim");
  // q likewise only needs a unit-stride innermost dim: DeepSeek rotates the
  // TRAILING qk_rope_head_dim slice of each query head
  // (mla.py:160-167 passes `q[..., qk_nope_head_dim:]`), a strided view of the
  // [T, num_heads, qk_head_dim] query. For contiguous tensors the kernels'
  // strided offsets are integer-identical to the pre-W6 formula.
  VT_CHECK(q_states.stride[2] == 1,
           "rope_from_cache: q innermost dimension must be unit-stride");
  VT_CHECK(positions.IsContiguous() && cos_sin_cache.IsContiguous(),
           "rope_from_cache: contiguous positions/cache required");
  VT_CHECK(q_states.device == q.device && positions.device == q.device &&
               cos_sin_cache.device == q.device,
           "rope_from_cache: q/positions/cache/queue device mismatch");
  reinterpret_cast<RopeFromCacheFn>(
      GetOp(OpId::kRopeFromCache, q.device.type))(
      q, q_states, k_states, positions, cos_sin_cache, args);
}

void FusedNormRope(Queue& q, Tensor& latent_out, Tensor& pe_out, const Tensor& x,
                   const Tensor& norm_weight, const Tensor& positions,
                   const Tensor& cos_sin_cache, const RmsNormArgs& norm_args,
                   const RopeArgs& rope_args) {
  const int64_t off = norm_weight.shape[0];   // latent width (kv_lora_rank)
  const int64_t rot = rope_args.rotary_dim;   // decoupled-rope width
  VT_CHECK(x.rank == 2 && latent_out.rank == 2 && pe_out.rank == 2,
           "fused_norm_rope: x/latent_out/pe_out must be rank-2 [T, ...]");
  const int64_t t = x.shape[0];
  VT_CHECK(norm_weight.rank == 1 && off > 0, "fused_norm_rope: norm_weight must be [off]");
  VT_CHECK(rot > 0 && rot % 2 == 0, "fused_norm_rope: rotary_dim must be positive even");
  VT_CHECK(x.shape[1] == off + rot,
           "fused_norm_rope: x must be [T, off+rot] (merged latent|pe)");
  VT_CHECK(latent_out.shape[0] == t && latent_out.shape[1] == off,
           "fused_norm_rope: latent_out must be [T, off]");
  VT_CHECK(pe_out.shape[0] == t && pe_out.shape[1] == rot,
           "fused_norm_rope: pe_out must be [T, rot]");
  VT_CHECK(x.stride[1] == 1 && latent_out.stride[1] == 1 && pe_out.stride[1] == 1,
           "fused_norm_rope: x/latent_out/pe_out innermost dim must be unit-stride");
  VT_CHECK(IsOutFloat(x.dtype) && norm_weight.dtype == x.dtype &&
               latent_out.dtype == x.dtype && pe_out.dtype == x.dtype,
           "fused_norm_rope: x/norm_weight/latent_out/pe_out must share f32 or bf16 dtype");
  VT_CHECK(positions.rank == 1 && positions.shape[0] == t,
           "fused_norm_rope: positions must be [T]");
  VT_CHECK(positions.dtype == DType::kI32 || positions.dtype == DType::kI64,
           "fused_norm_rope: positions must be i32/i64");
  VT_CHECK(cos_sin_cache.rank == 2 && cos_sin_cache.shape[0] > 0 &&
               cos_sin_cache.shape[1] == rot && cos_sin_cache.dtype == x.dtype,
           "fused_norm_rope: cache must be [P,rot] with x's dtype");
  VT_CHECK(positions.IsContiguous() && cos_sin_cache.IsContiguous(),
           "fused_norm_rope: contiguous positions/cache required");
  VT_CHECK(x.device == q.device && norm_weight.device == q.device &&
               latent_out.device == q.device && pe_out.device == q.device &&
               positions.device == q.device && cos_sin_cache.device == q.device,
           "fused_norm_rope: all tensors must share the queue device");
  if (t == 0) return;
  reinterpret_cast<FusedNormRopeFn>(GetOp(OpId::kFusedNormRope, q.device.type))(
      q, latent_out, pe_out, x, norm_weight, positions, cos_sin_cache, norm_args, rope_args);
}

void RopeCosSinCache(Queue& q, Tensor& cos_sin, const Tensor& positions, const RopeArgs& args) {
  VT_CHECK(cos_sin.rank == 2, "rope_cos_sin_cache: cos_sin rank-2 [T, rotary_dim]");
  VT_CHECK(cos_sin.dtype == DType::kF32, "rope_cos_sin_cache: cos_sin must be f32");
  VT_CHECK(positions.rank == 1 && positions.shape[0] == cos_sin.shape[0],
           "rope_cos_sin_cache: positions[T] must match cos_sin leading dim");
  VT_CHECK(positions.dtype == DType::kI32 || positions.dtype == DType::kI64,
           "rope_cos_sin_cache: positions i32/i64");
  VT_CHECK(args.rotary_dim > 0 && args.rotary_dim % 2 == 0 && cos_sin.shape[1] == args.rotary_dim,
           "rope_cos_sin_cache: cos_sin second dim must equal an even rotary_dim");
  VT_CHECK(cos_sin.IsContiguous() && positions.IsContiguous(),
           "rope_cos_sin_cache: contiguous required");
  VT_CHECK(cos_sin.device == q.device && positions.device == q.device,
           "rope_cos_sin_cache: device mismatch (cos_sin/positions/queue)");
  VT_CHECK(std::isfinite(args.base) && args.base > 0.f,
           "rope_cos_sin_cache: base must be finite and positive");
  VT_CHECK(std::isfinite(args.linear_scaling_factor) && args.linear_scaling_factor >= 0.f,
           "rope_cos_sin_cache: linear factor must be finite and nonnegative");
  VT_CHECK(args.linear_scaling_factor == 0.f || args.llama3_scaling_factor <= 0.f,
           "rope_cos_sin_cache: linear and Llama-3 scaling are mutually exclusive");
  reinterpret_cast<RopeCosSinCacheFn>(GetOp(OpId::kRopeCosSinCache, q.device.type))(q, cos_sin,
                                                                                    positions, args);
}

void AttnQkNormRopeGate(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                        const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                        const Tensor& k_norm, const Tensor& cos_sin,
                        const RmsNormArgs& norm_args, const RopeArgs& rope_args) {
  VT_CHECK(q_out.rank == 3 && k_out.rank == 3 && gate_out.rank == 3,
           "attn_qk_norm_rope_gate: q_out/k_out/gate_out rank-3 [T,H,Dh]");
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  const int64_t hkv = k_out.shape[1];
  VT_CHECK(k_out.shape[0] == t && k_out.shape[2] == dh, "attn_qk_norm_rope_gate: k_out [T,Hkv,Dh]");
  VT_CHECK(gate_out.shape[0] == t && gate_out.shape[1] == hq && gate_out.shape[2] == dh,
           "attn_qk_norm_rope_gate: gate_out must match q_out [T,Hq,Dh]");
  VT_CHECK(qgate.rank == 2 && qgate.shape[0] == t && qgate.shape[1] == hq * 2 * dh,
           "attn_qk_norm_rope_gate: qgate must be [T, Hq*2*Dh]");
  VT_CHECK(kf.rank == 2 && kf.shape[0] == t && kf.shape[1] == hkv * dh,
           "attn_qk_norm_rope_gate: kf must be [T, Hkv*Dh]");
  VT_CHECK(q_norm.rank == 1 && q_norm.shape[0] == dh && k_norm.rank == 1 && k_norm.shape[0] == dh,
           "attn_qk_norm_rope_gate: q_norm/k_norm must be [Dh]");
  VT_CHECK(cos_sin.rank == 2 && cos_sin.shape[0] == t && cos_sin.shape[1] == rope_args.rotary_dim,
           "attn_qk_norm_rope_gate: cos_sin must be [T, rotary_dim]");
  VT_CHECK(rope_args.rotary_dim > 0 && rope_args.rotary_dim % 2 == 0 && rope_args.rotary_dim <= dh,
           "attn_qk_norm_rope_gate: rotary_dim must be even and <= Dh");
  // q/k share one out dtype; the gate may additionally stay f32 while q/k are
  // bf16 (the FA-2 prefill combo: bf16 q feeds FA-2 and bf16 k feeds the bf16
  // KV-cache write, but sigmoid(gate) must see the un-rounded f32 gate). All
  // kernel math is f32 either way; a bf16 store is the RN round of the same
  // value, so mixed out is bit-identical to f32-out + CastBf16 on q/k.
  VT_CHECK(IsOutFloat(q_out.dtype) && k_out.dtype == q_out.dtype &&
               (gate_out.dtype == q_out.dtype ||
                (q_out.dtype == DType::kBF16 && gate_out.dtype == DType::kF32)),
           "attn_qk_norm_rope_gate: q/k/gate out f32 or bf16 (gate f32 allowed with bf16 q/k)");
  VT_CHECK(IsFloat(qgate.dtype) && kf.dtype == qgate.dtype,
           "attn_qk_norm_rope_gate: qgate/kf float, same dtype");
  VT_CHECK(q_norm.dtype == DType::kF32 && k_norm.dtype == DType::kF32 &&
               cos_sin.dtype == DType::kF32,
           "attn_qk_norm_rope_gate: q_norm/k_norm/cos_sin must be f32");
  // QKVParallelLinear returns torch.split-style logical views: every Q/K row
  // is inner-contiguous, but stride(0) remains Q+K+V. Consume that layout
  // directly instead of adding split-copy kernels.
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() &&
               gate_out.IsContiguous() && qgate.stride[1] == 1 &&
               qgate.stride[0] >= qgate.shape[1] && kf.stride[1] == 1 &&
               kf.stride[0] >= kf.shape[1] && q_norm.IsContiguous() &&
               k_norm.IsContiguous() && cos_sin.IsContiguous(),
           "attn_qk_norm_rope_gate: outputs/weights/cache must be contiguous; "
           "qgate/kf rows must be inner-contiguous");
  VT_CHECK(q_out.device == q.device && k_out.device == q.device && gate_out.device == q.device &&
               qgate.device == q.device && kf.device == q.device && q_norm.device == q.device &&
               k_norm.device == q.device && cos_sin.device == q.device,
           "attn_qk_norm_rope_gate: device mismatch");
  reinterpret_cast<AttnQkNormRopeGateFn>(GetOp(OpId::kAttnQkNormRopeGate, q.device.type))(
      q, q_out, k_out, gate_out, qgate, kf, q_norm, k_norm, cos_sin, norm_args, rope_args);
}

void AttnQkNormRope(Queue& q, Tensor& q3, Tensor& k3, const Tensor& q_norm,
                    const Tensor& k_norm, const Tensor& cos_sin, const Tensor& positions,
                    const RmsNormArgs& norm_args, const RopeArgs& rope_args) {
  VT_CHECK(q3.rank == 3 && k3.rank == 3, "attn_qk_norm_rope: q3/k3 rank-3 [T,H,Dh]");
  const int64_t t = q3.shape[0], dh = q3.shape[2];
  VT_CHECK(k3.shape[0] == t && k3.shape[2] == dh, "attn_qk_norm_rope: k3 must be [T,Hkv,Dh]");
  VT_CHECK(q3.dtype == k3.dtype, "attn_qk_norm_rope: q3/k3 dtype");
  VT_CHECK(IsFloat(q3.dtype), "attn_qk_norm_rope: q3/k3 must be f32 or bf16");
  VT_CHECK(q_norm.rank == 1 && q_norm.shape[0] == dh && k_norm.rank == 1 &&
               k_norm.shape[0] == dh,
           "attn_qk_norm_rope: q_norm/k_norm must be [Dh]");
  VT_CHECK(positions.rank == 1 && positions.shape[0] == t,
           "attn_qk_norm_rope: positions must be [T]");
  VT_CHECK(rope_args.rotary_dim > 0 && rope_args.rotary_dim % 2 == 0 &&
               rope_args.rotary_dim <= dh,
           "attn_qk_norm_rope: rotary_dim must be even and <= Dh");
  VT_CHECK(cos_sin.rank == 2 && cos_sin.shape[0] > 0 &&
               cos_sin.shape[1] == rope_args.rotary_dim,
           "attn_qk_norm_rope: cos_sin must be [rows, rotary_dim]");
  VT_CHECK(q3.IsContiguous() && k3.IsContiguous() && q_norm.IsContiguous() &&
               k_norm.IsContiguous() && cos_sin.IsContiguous() && positions.IsContiguous(),
           "attn_qk_norm_rope: states/weights/cache/index must be contiguous");
  // The fused op rotates the in-place operands, so the 2-D alias the composite's
  // RmsNorm step would have normed is the same memory: [T*H,Dh] with stride Dh.
  // That alias is a RESHAPE of the rank-3 view, so on a row-major [T,H,Dh] tensor
  // the row stride is stride[1] == Dh and the INNER dimension is stride[2]. The
  // check named stride[1] and so refused every Dh > 1 operand — every real call,
  // the hand-call realization at dense_attn_block.h:648 among them.
  VT_CHECK(q3.stride[2] == 1 && k3.stride[2] == 1,
           "attn_qk_norm_rope: the head dimension must be the inner dimension");
  reinterpret_cast<AttnQkNormRopeFn>(GetOp(OpId::kAttnQkNormRope, q.device.type))(
      q, q3, k3, q_norm, k_norm, cos_sin, positions, norm_args, rope_args);
}

namespace {
// Shared shape/dtype/device validation for the two conv ops. x/out [T,C],
// weight [C,K], optional bias [C], conv_state [N,C,K-1] f32.
void CheckConvCommon(const Queue& q, const Tensor& out, const Tensor& x, const Tensor& weight,
                     const Tensor* bias, const Tensor& conv_state, const char* name) {
  VT_CHECK(x.rank == 2 && out.rank == 2 && weight.rank == 2 && conv_state.rank == 3,
           std::string(name) + ": x/out [T,C], weight [C,K], conv_state [N,C,K-1]");
  const int64_t c = x.shape[1], k = weight.shape[1];
  VT_CHECK(out.shape[0] == x.shape[0] && out.shape[1] == c,
           std::string(name) + ": out shape must match x");
  VT_CHECK(weight.shape[0] == c, std::string(name) + ": weight channel dim mismatch");
  VT_CHECK(k >= 1, std::string(name) + ": kernel width must be >= 1");
  // The conv row physical width is (K-1) for the ordinary (num_spec==0) cache,
  // but WIDENED to (K-1)+num_spec when speculative decode allocates the extra
  // rollback taps (mamba_utils.py:226). Non-spec ops operate on the LEADING
  // (K-1) sub-window with the row's PHYSICAL stride, mirroring vLLM's
  // `state_len = KERNEL_WIDTH - 1` + physical `stride_conv_state_tok`
  // (causal_conv1d.py:67-69,156, prefill writes leading [0,K-1)). At num_spec==0
  // the row is exactly (K-1), so this admits the identical shapes as before.
  VT_CHECK(conv_state.shape[1] == c && conv_state.shape[2] >= k - 1,
           std::string(name) + ": conv_state must be [N,C,(K-1)+num_spec>=K-1]");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(weight.dtype) && IsOutFloat(out.dtype),
           std::string(name) + ": float x/weight, f32/bf16 out");
  // bf16 conv_state is admitted wherever the BACKEND says its conv kernels can
  // address a compressed row in place (Backend::SupportsCompressedConvState) —
  // CUDA and Vulkan today. Asking the backend rather than naming a device is
  // what keeps this shared op layer device-agnostic; the alternative for a
  // backend that answers false is the caller's f32 gather/compute/scatter, which
  // is still exactly what CPU does.
  const Backend* conv_backend = TryGetBackend(q.device.type);
  VT_CHECK(conv_state.dtype == DType::kF32 ||
               (conv_state.dtype == DType::kBF16 && conv_backend != nullptr &&
                conv_backend->SupportsCompressedConvState()),
           std::string(name) +
               ": conv_state must be f32, or bf16 on a backend whose conv kernels "
               "support a compressed state in place (in/out, in place; bf16 = "
               "vLLM default mamba_cache_dtype, read/written in f32 registers)");
  // out/weight/conv_state stay fully contiguous. x may be a padded-row
  // (inner-contiguous, outer stride >= C) view: the merged qkvz projection feeds
  // the conv its mixed_qkv = mixed_qkvz[:, :conv_dim] slice without any copy
  // (qwen_gdn_linear_attn.py:929-936). The kernels honor x.stride[0] directly.
  VT_CHECK(x.stride[1] == 1 && x.stride[0] >= c && out.IsContiguous() &&
               weight.IsContiguous() && conv_state.IsContiguous(),
           std::string(name) +
               ": out/weight/conv_state contiguous; x rows inner-contiguous "
               "(padded outer stride allowed)");
  VT_CHECK(x.device == q.device && out.device == q.device && weight.device == q.device &&
               conv_state.device == q.device,
           std::string(name) + ": device mismatch (x/out/weight/conv_state/queue)");
  if (bias != nullptr) {
    VT_CHECK(bias->rank == 1 && bias->shape[0] == c && IsFloat(bias->dtype) &&
                 bias->IsContiguous() && bias->device == q.device,
             std::string(name) + ": bias must be float [C] contiguous on the queue device");
  }
}

// Shared validation for the two decomposed delta-rule ops. q_in/k
// [T,Hk,Dk], v [T,Hv,Dv], g/beta [T,Hv] f32, state [N,Hv,Dv,Dk]
// fp16/bf16/fp32 on CUDA (independent Mamba temporal-cache dtype), out
// [T,Hv,Dv]. CPU keeps the f32 recurrence reference.
void CheckGdnCommon(const Queue& q, const Tensor& out, const Tensor& q_in, const Tensor& k,
                    const Tensor& v, const Tensor& g, const Tensor& beta, const Tensor& state,
                    const GdnArgs& args, const char* name,
                    bool allow_compressed_state) {
  VT_CHECK(q_in.rank == 3 && k.rank == 3 && v.rank == 3 && out.rank == 3 && g.rank == 2 &&
               beta.rank == 2 && state.rank == 4,
           std::string(name) +
               ": q/k [T,Hk,Dk], v/out [T,Hv,Dv], g/beta [T,Hv], state [N,Hv,Dv,Dk]");
  const int64_t t = q_in.shape[0], hk = q_in.shape[1], dk = q_in.shape[2];
  const int64_t hv = v.shape[1], dv = v.shape[2];
  VT_CHECK(k.shape[0] == t && k.shape[1] == hk && k.shape[2] == dk,
           std::string(name) + ": k shape must match q");
  VT_CHECK(v.shape[0] == t, std::string(name) + ": v token count must match q");
  VT_CHECK(out.shape[0] == t && out.shape[1] == hv && out.shape[2] == dv,
           std::string(name) + ": out must be [T,Hv,Dv]");
  VT_CHECK(g.shape[0] == t && g.shape[1] == hv && beta.shape[0] == t && beta.shape[1] == hv,
           std::string(name) + ": g/beta must be [T,Hv]");
  VT_CHECK(hk >= 1 && hv % hk == 0,
           std::string(name) + ": Hv must be a multiple of Hk (GQA broadcast)");
  VT_CHECK(state.shape[1] == hv && state.shape[2] == dv && state.shape[3] == dk,
           std::string(name) + ": state must be [N,Hv,Dv,Dk]");
  VT_CHECK(IsFloat(q_in.dtype) && IsFloat(k.dtype) && IsFloat(v.dtype) &&
               IsOutFloat(out.dtype),
           std::string(name) + ": float q/k/v, f32/bf16 out");
  VT_CHECK(g.dtype == DType::kF32 && beta.dtype == DType::kF32,
           std::string(name) + ": g/beta must be f32 (upstream keeps them f32)");
  if (allow_compressed_state) {
    // Asking the backend (Backend::SupportsCompressedGdnState) rather than
    // naming a device — the same device-agnostic pattern CheckConvCommon
    // already uses for the conv state. CUDA answers for its existing kernels;
    // ROCm answers for the portable scan's f16/bf16 state arms.
    const Backend* gdn_backend = TryGetBackend(q.device.type);
    VT_CHECK(state.dtype == DType::kF32 ||
                 ((state.dtype == DType::kF16 || state.dtype == DType::kBF16) &&
                  gdn_backend != nullptr && gdn_backend->SupportsCompressedGdnState()),
             std::string(name) +
                 ": state must be f32, or fp16/bf16 on a backend whose GDN kernels "
                 "support a compressed state in place (in/out, in place; "
                 "read/written in f32 registers)");
  } else {
    VT_CHECK(state.dtype == DType::kF32,
             std::string(name) +
                 ": state must be f32; gather compressed cache rows first");
  }
  VT_CHECK(q_in.IsContiguous() && k.IsContiguous() && v.IsContiguous() && out.IsContiguous() &&
               g.IsContiguous() && beta.IsContiguous() && state.IsContiguous(),
           std::string(name) + ": contiguous required");
  VT_CHECK(q_in.device == q.device && k.device == q.device && v.device == q.device &&
               out.device == q.device && g.device == q.device && beta.device == q.device &&
               state.device == q.device,
           std::string(name) + ": device mismatch (q/k/v/out/g/beta/state/queue)");
  VT_CHECK(args.scale > 0.0f, std::string(name) + ": args.scale must be set (> 0)");
}

void CheckI32Meta(const Queue& q, const Tensor& t, int64_t expect_len, const char* name,
                  const char* what) {
  VT_CHECK(t.rank == 1 && t.shape[0] == expect_len && t.dtype == DType::kI32 &&
               t.IsContiguous() && t.device == q.device,
           std::string(name) + ": " + what + " must be i32 [" + std::to_string(expect_len) +
               "] contiguous on the queue device");
}

void CheckBoolMeta(const Queue& q, const Tensor& t, int64_t expect_len,
                   const char* name, const char* what) {
  VT_CHECK(t.rank == 1 && t.shape[0] == expect_len &&
               (t.dtype == DType::kI8 || t.dtype == DType::kI32) &&
               t.IsContiguous() && t.device == q.device,
           std::string(name) + ": " + what + " must be i8/i32 [" +
               std::to_string(expect_len) + "] contiguous on the queue device");
}
}  // namespace

void CausalConv1dFwd(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                     const Tensor* bias, Tensor& conv_state, const Tensor& query_start_loc,
                     const Tensor& has_initial_state, const CausalConv1dArgs& args) {
  CheckConvCommon(q, out, x, weight, bias, conv_state, "causal_conv1d_fwd");
  const int64_t n = conv_state.shape[0];
  CheckI32Meta(q, query_start_loc, n + 1, "causal_conv1d_fwd", "query_start_loc");
  CheckBoolMeta(q, has_initial_state, n, "causal_conv1d_fwd", "has_initial_state");
  VT_CHECK((args.batch_ptr == nullptr) == (args.token_chunk_offset_ptr == nullptr),
           "causal_conv1d_fwd: batch_ptr and token_chunk_offset_ptr must be supplied together");
  if (args.batch_ptr != nullptr) {
    const int64_t programs = args.batch_ptr->shape[0];
    CheckI32Meta(q, *args.batch_ptr, programs, "causal_conv1d_fwd", "batch_ptr");
    CheckI32Meta(q, *args.token_chunk_offset_ptr, programs, "causal_conv1d_fwd",
                 "token_chunk_offset_ptr");
    VT_CHECK(programs > 0,
             "causal_conv1d_fwd: exact chunk descriptor must not be empty");
  }
  reinterpret_cast<CausalConv1dFwdFn>(GetOp(OpId::kCausalConv1dFwd, q.device.type))(
      q, out, x, weight, bias, conv_state, query_start_loc, has_initial_state, args);
}

void CausalConv1dUpdate(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                        const Tensor* bias, Tensor& conv_state, const CausalConv1dArgs& args,
                        const Tensor* conv_state_indices) {
  CheckConvCommon(q, out, x, weight, bias, conv_state, "causal_conv1d_update");
  if (conv_state_indices == nullptr) {
    VT_CHECK(conv_state.shape[0] == x.shape[0],
             "causal_conv1d_update: one conv_state row per token required");
  } else {
    // In-place indexed path: conv_state is the FULL cache; one index per token.
    CheckI32Meta(q, *conv_state_indices, x.shape[0], "causal_conv1d_update",
                 "conv_state_indices");
    VT_CHECK(conv_state.shape[0] >= x.shape[0],
             "causal_conv1d_update: indexed cache must have >= batch rows");
  }
  reinterpret_cast<CausalConv1dUpdateFn>(GetOp(OpId::kCausalConv1dUpdate, q.device.type))(
      q, out, x, weight, bias, conv_state, conv_state_indices, args);
}

void Qwen4ExpPleConv(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                     Tensor& conv_state, const Tensor& query_start_loc,
                     const Tensor* conv_state_indices, const Qwen4ExpPleConvArgs& args) {
  constexpr const char* name = "qwen4_exp_ple_conv";
  VT_CHECK(x.rank == 2 && out.rank == 2 && weight.rank == 2 && conv_state.rank == 3,
           std::string(name) +
               ": x/out [T,C], weight [C,K], conv_state [N,C,(K-1)*dilation]");
  const int64_t T = x.shape[0], c = x.shape[1], k = weight.shape[1];
  VT_CHECK(out.shape[0] == T && out.shape[1] == c,
           std::string(name) + ": out shape must match x");
  VT_CHECK(weight.shape[0] == c, std::string(name) + ": weight channel dim mismatch");
  VT_CHECK(k >= 2, std::string(name) + ": kernel width must be >= 2");
  VT_CHECK(args.dilation >= 1,
           std::string(name) + ": dilation must be >= 1, got " +
               std::to_string(args.dilation));
  // THE ONE CHECK THIS OP EXISTS FOR. `CausalConv1dFwd` welds the state width to
  // `K - 1`; here it is `(K - 1) * dilation`, and the two agree only at
  // dilation 1. A caller that sized its cache with the Mamba formula and then
  // asked for dilation 3 gets a message naming both numbers, rather than a
  // plausible answer computed off nine columns of which six are somebody else's.
  const int64_t want_state = (k - 1) * args.dilation;
  VT_CHECK(conv_state.shape[1] == c && conv_state.shape[2] == want_state,
           std::string(name) + ": conv_state must be [N,C,(K-1)*dilation] = [N," +
               std::to_string(c) + "," + std::to_string(want_state) + "], got [N," +
               std::to_string(conv_state.shape[1]) + "," +
               std::to_string(conv_state.shape[2]) + "]");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(weight.dtype) && IsOutFloat(out.dtype),
           std::string(name) + ": float x/weight, f32/bf16 out");
  // THE STATE CARRIES THE MODEL DTYPE, AND THE ORACLE SETTLES IT (W5k, #2031).
  // This check read `conv_state.dtype == kF32` and argued that "no CUDA arm of
  // this op exists, so admitting a dtype nothing can produce would be a promise
  // with no kernel behind it". The premise was about which KERNELS exist; the
  // question is what UPSTREAM STORES, and those are different questions. The
  // second one is now answered from the running oracle rather than from the
  // shape of this tree.
  //
  // transformers 5.16.0 (the `qwen4_exp` lane pin, `.agents/oracles/transformers.md`)
  // types each cache slot from the tensor that FIRST reaches it, per slot and not
  // per layer: `cache_utils.py:1019-1023` allocates
  // `torch.zeros(..., dtype=conv_states.dtype, device=conv_states.device)`. The
  // tensor reaching the PLE conv slot is `hidden_states`
  // (`modeling_qwen4_exp.py:1157-1159`, `update_conv_state(..., state_idx=1)`), so
  // the ring carries the MODEL dtype. Observed, not inferred: the same fixture run
  // at `dtype=torch.bfloat16` reports `conv_states[1] dtype=torch.bfloat16`, and at
  // `float32` reports `float32`. It NEVER widens to f32.
  //
  // Admitting bf16 is therefore mirroring upstream, and refusing it was the
  // "dtype that is too wide" AGENTS.md names — the defect class a token gate
  // cannot see, because the tokens match while the path moves twice the bytes.
  // The CPU kernel reads and writes the ring through the same `LoadF32At` /
  // `StoreF32At` accessors it already used for `x` and `out`, so this admits no
  // dtype that has no kernel behind it. f32 stays accepted and every existing
  // f32 caller is byte-unchanged.
  VT_CHECK(conv_state.dtype == DType::kF32 || conv_state.dtype == DType::kBF16,
           std::string(name) +
               ": conv_state must be f32 or bf16 (upstream types the slot from "
               "the model dtype, cache_utils.py:1019-1023)");
  VT_CHECK(x.IsContiguous() && out.IsContiguous() && weight.IsContiguous() &&
               conv_state.IsContiguous(),
           std::string(name) + ": x/out/weight/conv_state must be contiguous");
  VT_CHECK(x.device == q.device && out.device == q.device && weight.device == q.device &&
               conv_state.device == q.device,
           std::string(name) + ": device mismatch (x/out/weight/conv_state/queue)");
  VT_CHECK(query_start_loc.rank == 1 && query_start_loc.shape[0] >= 2,
           std::string(name) + ": query_start_loc must be i32 [num_seqs + 1]");
  const int64_t n_seqs = query_start_loc.shape[0] - 1;
  CheckI32Meta(q, query_start_loc, n_seqs + 1, name, "query_start_loc");
  if (conv_state_indices != nullptr) {
    CheckI32Meta(q, *conv_state_indices, n_seqs, name, "conv_state_indices");
  } else {
    VT_CHECK(conv_state.shape[0] >= n_seqs,
             std::string(name) +
                 ": without conv_state_indices the cache needs one row per sequence");
  }
  if (q.device.type == DeviceType::kCPU) {
    const int32_t* qsl = query_start_loc.Ptr<int32_t>();
    VT_CHECK(qsl[0] == 0 && qsl[n_seqs] == T,
             std::string(name) + ": query_start_loc must run from 0 to T");
    for (int64_t i = 0; i < n_seqs; ++i) {
      VT_CHECK(qsl[i + 1] >= qsl[i],
               std::string(name) + ": query_start_loc must be non-decreasing");
    }
    if (conv_state_indices != nullptr) {
      const int32_t* rows = conv_state_indices->Ptr<int32_t>();
      for (int64_t i = 0; i < n_seqs; ++i) {
        VT_CHECK(rows[i] >= 0 && rows[i] < conv_state.shape[0],
                 std::string(name) + ": conv_state_indices out of range");
      }
    }
  }
  reinterpret_cast<Qwen4ExpPleConvFn>(GetOp(OpId::kQwen4ExpPleConv, q.device.type))(
      q, out, x, weight, conv_state, query_start_loc, conv_state_indices, args);
}

// vt::Qwen4ExpPleGate — `Qwen4ExpTextPLELayer.forward` :1181-1182 (+ :1184),
// transformers v5.16.0. The DOT that feeds it is vt::BatchedMatmul and is
// deliberately not here; see the kQwen4ExpPleGate comment in include/vt/ops.h.
void Qwen4ExpPleGate(Queue& q, Tensor& out, const Tensor& score, const Tensor& value,
                     const Qwen4ExpPleGateArgs& args) {
  constexpr const char* name = "qwen4_exp_ple_gate";
  VT_CHECK(out.rank == 2 && score.rank == 2 && value.rank == 2,
           std::string(name) + ": out [T,hc*H], score [T,hc], value [T,H]");
  const int64_t T = score.shape[0], hc = score.shape[1], h = value.shape[1];
  VT_CHECK(out.shape[0] == T && value.shape[0] == T,
           std::string(name) + ": out/score/value must agree on T");
  VT_CHECK(hc >= 1 && h >= 1, std::string(name) + ": hc and H must be >= 1");
  // THE ONE CHECK THIS OP EXISTS FOR, after the arithmetic itself. `out` is the
  // FLATTENED [T, hc*H] the conv and the norm downstream want, and a caller that
  // flattened (H, hc) instead of (hc, H) produces a buffer of exactly the right
  // size holding a transposed answer. The product is therefore named against
  // both factors, so the message says which two numbers were multiplied.
  VT_CHECK(out.shape[1] == hc * h,
           std::string(name) + ": out must be [T, hc*H] = [T," + std::to_string(hc) + "*" +
               std::to_string(h) + "] = [T," + std::to_string(hc * h) + "], got [T," +
               std::to_string(out.shape[1]) + "]");
  // f32 score only, the reason SigmoidGateBf16 gives for its own gate operand:
  // this value is the argument of a sigmoid AND of a square root, and rounding a
  // transcendental's input is a value change no downstream tolerance owns.
  VT_CHECK(score.dtype == DType::kF32,
           std::string(name) + ": score must be f32 (it is the sigmoid/sqrt argument)");
  VT_CHECK(IsFloat(value.dtype) && IsOutFloat(out.dtype),
           std::string(name) + ": float value, f32/bf16 out");
  VT_CHECK(args.gate_divisor > 0.0f,
           std::string(name) + ": gate_divisor must be > 0 (it is math.sqrt(hidden_size)), got " +
               std::to_string(args.gate_divisor));
  // 0 is NOT "no floor". Upstream's literal is 1e-6 and its whole effect is the
  // 1e-3 floor it puts on |output|; a zero here would silently mean "port the
  // line without the clamp", which is the defect the op is gated against.
  VT_CHECK(args.clamp_min > 0.0f,
           std::string(name) +
               ": clamp_min must be > 0; 0 is NOT 'no floor'. Upstream's literal is 1e-6 "
               "(modeling_qwen4_exp.py:1181) and it is applied BEFORE the sqrt, so the "
               "floor on |out| is its square root");
  VT_CHECK(out.IsContiguous() && score.IsContiguous() && value.IsContiguous(),
           std::string(name) + ": out/score/value must be contiguous");
  VT_CHECK(out.device == q.device && score.device == q.device && value.device == q.device,
           std::string(name) + ": device mismatch (out/score/value/queue)");
  reinterpret_cast<Qwen4ExpPleGateFn>(GetOp(OpId::kQwen4ExpPleGate, q.device.type))(
      q, out, score, value, args);
}

void CausalConv1dSpecUpdate(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                            const Tensor* bias, Tensor& conv_state,
                            const Tensor& conv_state_indices,
                            const Tensor& num_accepted_tokens, const Tensor& cu_seqlens,
                            const CausalConv1dArgs& args) {
  constexpr const char* name = "causal_conv1d_spec_update";
  // Same contract as CheckConvCommon EXCEPT the state width: a speculative row
  // carries the sliding window (K-1) + (max_query_len - 1) taps
  // (causal_conv1d.py:1181-1184), not K-1.
  VT_CHECK(x.rank == 2 && out.rank == 2 && weight.rank == 2 && conv_state.rank == 3,
           std::string(name) + ": x/out [T,C], weight [C,K], conv_state [N,C,state_len]");
  const int64_t c = x.shape[1], k = weight.shape[1];
  const int64_t state_len = conv_state.shape[2];
  VT_CHECK(out.shape[0] == x.shape[0] && out.shape[1] == c,
           std::string(name) + ": out shape must match x");
  VT_CHECK(weight.shape[0] == c, std::string(name) + ": weight channel dim mismatch");
  VT_CHECK(k >= 2, std::string(name) + ": kernel width must be >= 2");
  VT_CHECK(conv_state.shape[1] == c && state_len >= k - 1,
           std::string(name) + ": conv_state must be [N,C,state_len>=K-1]");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(weight.dtype) && IsOutFloat(out.dtype),
           std::string(name) + ": float x/weight, f32/bf16 out");
  VT_CHECK(conv_state.dtype == DType::kF32 ||
               (conv_state.dtype == DType::kBF16 && q.device.type == DeviceType::kCUDA),
           std::string(name) + ": conv_state must be f32, or bf16 on CUDA");
  VT_CHECK(x.stride[1] == 1 && x.stride[0] >= c && out.IsContiguous() &&
               weight.IsContiguous() && conv_state.IsContiguous(),
           std::string(name) +
               ": out/weight/conv_state contiguous; x rows inner-contiguous");
  VT_CHECK(x.device == q.device && out.device == q.device && weight.device == q.device &&
               conv_state.device == q.device,
           std::string(name) + ": device mismatch (x/out/weight/conv_state/queue)");
  if (bias != nullptr) {
    VT_CHECK(bias->rank == 1 && bias->shape[0] == c && IsFloat(bias->dtype) &&
                 bias->IsContiguous() && bias->device == q.device,
             std::string(name) + ": bias must be float [C] contiguous on the queue device");
  }
  VT_CHECK(cu_seqlens.rank == 1 && cu_seqlens.shape[0] >= 2, std::string(name) +
               ": cu_seqlens must be i32 [num_reqs + 1]");
  const int64_t num_reqs = cu_seqlens.shape[0] - 1;
  CheckI32Meta(q, cu_seqlens, num_reqs + 1, name, "cu_seqlens");
  CheckI32Meta(q, conv_state_indices, num_reqs, name, "conv_state_indices");
  CheckI32Meta(q, num_accepted_tokens, num_reqs, name, "num_accepted_tokens");
  // max_query_len is implied by the state width: state_len = (K-1) + (mql - 1).
  const int64_t max_query_len = state_len - (k - 1) + 1;
  VT_CHECK(max_query_len >= 1,
           std::string(name) + ": conv_state width too small for a spec step");
  if (q.device.type == DeviceType::kCPU) {
    const int32_t* cs = cu_seqlens.Ptr<int32_t>();
    const int32_t* nat = num_accepted_tokens.Ptr<int32_t>();
    const int32_t* idx = conv_state_indices.Ptr<int32_t>();
    VT_CHECK(cs[0] == 0 && cs[num_reqs] == x.shape[0],
             std::string(name) + ": bad cu_seqlens bounds");
    for (int64_t i = 0; i < num_reqs; ++i) {
      VT_CHECK(cs[i + 1] >= cs[i] && cs[i + 1] - cs[i] <= max_query_len,
               std::string(name) + ": query length exceeds the conv window");
      VT_CHECK(nat[i] >= 1 && nat[i] <= max_query_len,
               std::string(name) + ": num_accepted_tokens out of range");
      VT_CHECK(idx[i] < conv_state.shape[0],
               std::string(name) + ": conv_state_indices out of range");
    }
  }
  reinterpret_cast<CausalConv1dSpecUpdateFn>(
      GetOp(OpId::kCausalConv1dSpecUpdate, q.device.type))(
      q, out, x, weight, bias, conv_state, conv_state_indices, num_accepted_tokens,
      cu_seqlens, args);
}

void L2Norm(Queue& q, Tensor& out, const Tensor& x, const L2NormArgs& args) {
  VT_CHECK(x.rank == 2 || x.rank == 3, "l2norm: rank 2 or 3 required");
  VT_CHECK(out.rank == x.rank, "l2norm: out rank must match x");
  for (int d = 0; d < x.rank; ++d)
    VT_CHECK(out.shape[d] == x.shape[d], "l2norm: out shape must match x");
  VT_CHECK(IsFloat(x.dtype) && IsOutFloat(out.dtype), "l2norm: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(), "l2norm: contiguous required");
  VT_CHECK(x.device == q.device && out.device == q.device,
           "l2norm: device mismatch (x/out/queue)");
  reinterpret_cast<L2NormFn>(GetOp(OpId::kL2Norm, q.device.type))(q, out, x, args);
}

void RmsNormGated(Queue& q, Tensor& out, const Tensor& x, const Tensor& gate,
                  const Tensor& weight, const RmsNormGatedArgs& args) {
  // Rank-2 [rows, D] (contiguous split path) OR rank-3 [T, Hv, D] (merged qkvz
  // path): the gate is the padded-row z = mixed_qkvz[:, conv_dim:] slice viewed
  // as [T, Hv, Dv], so its inner [Hv,Dv] block stays contiguous while the token
  // stride may exceed Hv*Dv (qwen_gdn_linear_attn.py:929-936). Normalization is
  // always over the LAST dim; rows = numel/D either way.
  VT_CHECK((x.rank == 2 || x.rank == 3) && gate.rank == x.rank &&
               out.rank == x.rank && weight.rank == 1,
           "rmsnorm_gated: x/gate/out rank-2 or rank-3, weight rank-1");
  for (int dd = 0; dd < x.rank; ++dd)
    VT_CHECK(gate.shape[dd] == x.shape[dd] && out.shape[dd] == x.shape[dd],
             "rmsnorm_gated: x/gate/out shapes must match");
  const int64_t d = x.shape[x.rank - 1];
  VT_CHECK(weight.shape[0] == d, "rmsnorm_gated: weight size mismatch");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(gate.dtype) && IsFloat(weight.dtype) &&
               IsOutFloat(out.dtype),
           "rmsnorm_gated: float in, f32/bf16 out");
  // x/out/weight stay contiguous; the gate may carry a padded outer (token)
  // stride with all inner dims contiguous.
  VT_CHECK(x.IsContiguous() && weight.IsContiguous() && out.IsContiguous(),
           "rmsnorm_gated: x/out/weight contiguous required");
  bool gate_inner = true;
  int64_t gate_span = 1;
  for (int dd = gate.rank - 1; dd >= 1; --dd) {
    gate_inner = gate_inner && gate.stride[dd] == gate_span;
    gate_span *= gate.shape[dd];
  }
  gate_inner = gate_inner && gate.stride[0] >= gate_span;
  VT_CHECK(gate_inner,
           "rmsnorm_gated: gate rows inner-contiguous (padded outer stride "
           "allowed)");
  VT_CHECK(x.device == q.device && gate.device == q.device && weight.device == q.device &&
               out.device == q.device,
           "rmsnorm_gated: device mismatch (x/gate/weight/out/queue)");
  reinterpret_cast<RmsNormGatedFn>(GetOp(OpId::kRmsNormGated, q.device.type))(q, out, x, gate,
                                                                              weight, args);
}

// KERNEL-GDN-CHUNKED-MIRROR D3: the one flag, read in the shared op layer so
// every backend's GdnPrefill answers the same question. Lifted verbatim from
// cuda_gdn.cu's bespoke ChunkedPrefillEnabled(), which now delegates here.
// Read on each call: prefill is coarse-grained, so the getenv cost is
// negligible, and it lets one process drive both arms (T4).
bool GdnChunkedPrefillEnabled() {
  const char* e = std::getenv("VT_GDN_CHUNKED");
  return e == nullptr || e[0] != '0';
}

// KERNEL-GDN-CHUNKED-MIRROR D0. `chunked = (dtype == bf16) && VT_GDN_CHUNKED != 0`.
// The dtype term is NOT a concession to this tree's three f32 goldens; it is the
// mirror. Upstream has no chunked path at any dtype but bf16 to mirror, on
// either of its two implementations of the kernel, so f32 AND f16 route to the
// sequential recurrence that upstream's own f32-capable kernel computes.
//
// The spec's D0 states this as `dtype != f32`, and implementing it that way was
// a defect: f16 then took the chunked arm and was silently bf16-rounded at all
// nine placement sites. `== bf16` is the rule; see the reason on the body below.
bool GdnUseChunkedPrefill(DType q_dtype) {
  // BF16 ONLY, not merely "not f32". Upstream's chunked kernels accept exactly
  // one input dtype on BOTH implementations: the Triton wrapper asserts
  // `q.dtype != torch.float32` (chunk.py:213-215) and every FLA store is
  // `k.dtype`, while the CPU kernel type-checks `at::kBFloat16` outright
  // (csrc/cpu/sgl-kernels/fla.cpp:2205-2207). There is no f16 chunked gated
  // delta rule upstream to mirror, so f16 takes the sequential recurrence for
  // the same reason f32 does.
  return q_dtype == DType::kBF16 && GdnChunkedPrefillEnabled();
}

void GdnPrefill(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
                const Tensor& g, const Tensor& beta, Tensor& state,
                const Tensor& query_start_loc, const GdnArgs& args) {
  CheckGdnCommon(q, out, q_in, k, v, g, beta, state, args, "gdn_prefill",
                 /*allow_compressed_state=*/false);
  CheckI32Meta(q, query_start_loc, state.shape[0] + 1, "gdn_prefill", "query_start_loc");
  reinterpret_cast<GdnPrefillFn>(GetOp(OpId::kGdnPrefill, q.device.type))(
      q, out, q_in, k, v, g, beta, state, query_start_loc, args);
}

void KdaGatedDeltaRule(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                       const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                       const Tensor& query_start_loc, const GdnArgs& args) {
  constexpr const char* name = "kda_gated_delta_rule";
  // Same contracts as GdnPrefill EXCEPT g is per-K-channel [T,Hv,Dk].
  VT_CHECK(q_in.rank == 3 && k.rank == 3 && v.rank == 3 && out.rank == 3 && g.rank == 3 &&
               beta.rank == 2 && state.rank == 4,
           std::string(name) +
               ": q/k [T,Hk,Dk], v/out [T,Hv,Dv], g [T,Hv,Dk], beta [T,Hv], state [N,Hv,Dv,Dk]");
  const int64_t t = q_in.shape[0], hk = q_in.shape[1], dk = q_in.shape[2];
  const int64_t hv = v.shape[1], dv = v.shape[2];
  VT_CHECK(k.shape[0] == t && k.shape[1] == hk && k.shape[2] == dk,
           std::string(name) + ": k shape must match q");
  VT_CHECK(v.shape[0] == t, std::string(name) + ": v token count must match q");
  VT_CHECK(out.shape[0] == t && out.shape[1] == hv && out.shape[2] == dv,
           std::string(name) + ": out must be [T,Hv,Dv]");
  VT_CHECK(g.shape[0] == t && g.shape[1] == hv && g.shape[2] == dk,
           std::string(name) + ": g must be [T,Hv,Dk] (per-K-channel decay)");
  VT_CHECK(beta.shape[0] == t && beta.shape[1] == hv,
           std::string(name) + ": beta must be [T,Hv]");
  VT_CHECK(hk >= 1 && hv % hk == 0,
           std::string(name) + ": Hv must be a multiple of Hk (GQA broadcast)");
  VT_CHECK(state.shape[1] == hv && state.shape[2] == dv && state.shape[3] == dk,
           std::string(name) + ": state must be [N,Hv,Dv,Dk]");
  VT_CHECK(IsFloat(q_in.dtype) && IsFloat(k.dtype) && IsFloat(v.dtype) && IsOutFloat(out.dtype),
           std::string(name) + ": float q/k/v, f32/bf16 out");
  VT_CHECK(g.dtype == DType::kF32 && beta.dtype == DType::kF32,
           std::string(name) + ": g/beta must be f32 (upstream keeps them f32)");
  VT_CHECK(state.dtype == DType::kF32,
           std::string(name) + ": state must be f32 (fresh-zeros or persistent, read/written f32)");
  VT_CHECK(q_in.IsContiguous() && k.IsContiguous() && v.IsContiguous() && out.IsContiguous() &&
               g.IsContiguous() && beta.IsContiguous() && state.IsContiguous(),
           std::string(name) + ": contiguous required");
  VT_CHECK(q_in.device == q.device && k.device == q.device && v.device == q.device &&
               out.device == q.device && g.device == q.device && beta.device == q.device &&
               state.device == q.device,
           std::string(name) + ": device mismatch (q/k/v/out/g/beta/state/queue)");
  VT_CHECK(args.scale > 0.0f, std::string(name) + ": args.scale must be set (> 0)");
  CheckI32Meta(q, query_start_loc, state.shape[0] + 1, name, "query_start_loc");
  reinterpret_cast<KdaGatedDeltaRuleFn>(GetOp(OpId::kKdaGatedDeltaRule, q.device.type))(
      q, out, q_in, k, v, g, beta, state, query_start_loc, args);
}

void KdaChunkPrefill(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                     const Tensor& v, const Tensor& g_raw, const Tensor& beta,
                     const Tensor& a_log, const Tensor& dt_bias, Tensor& state,
                     const Tensor& query_start_loc, const GdnArgs& args) {
  constexpr const char* name = "kda_chunk_prefill";
  // Same tensor contracts as KdaGatedDeltaRule EXCEPT the gate is supplied RAW
  // (g_raw + a_log + dt_bias fused on-device by kda_gate_cumsum), not pre-gated.
  VT_CHECK(q_in.rank == 3 && k.rank == 3 && v.rank == 3 && out.rank == 3 && g_raw.rank == 3 &&
               beta.rank == 2 && state.rank == 4,
           std::string(name) +
               ": q/k [T,Hk,Dk], v/out [T,Hv,Dv], g_raw [T,Hv,Dk], beta [T,Hv], state [N,Hv,Dv,Dk]");
  const int64_t t = q_in.shape[0], hk = q_in.shape[1], dk = q_in.shape[2];
  const int64_t hv = v.shape[1], dv = v.shape[2];
  VT_CHECK(k.shape[0] == t && k.shape[1] == hk && k.shape[2] == dk,
           std::string(name) + ": k shape must match q");
  VT_CHECK(v.shape[0] == t, std::string(name) + ": v token count must match q");
  VT_CHECK(out.shape[0] == t && out.shape[1] == hv && out.shape[2] == dv,
           std::string(name) + ": out must be [T,Hv,Dv]");
  VT_CHECK(g_raw.shape[0] == t && g_raw.shape[1] == hv && g_raw.shape[2] == dk,
           std::string(name) + ": g_raw must be [T,Hv,Dk] (raw per-K-channel gate projection)");
  VT_CHECK(beta.shape[0] == t && beta.shape[1] == hv,
           std::string(name) + ": beta must be [T,Hv]");
  VT_CHECK(hk >= 1 && hv % hk == 0,
           std::string(name) + ": Hv must be a multiple of Hk (GQA broadcast)");
  VT_CHECK(state.shape[0] == 1 && state.shape[1] == hv && state.shape[2] == dv &&
               state.shape[3] == dk,
           std::string(name) + ": state must be [1,Hv,Dv,Dk] (single prefill sequence)");
  VT_CHECK(IsFloat(q_in.dtype) && IsFloat(k.dtype) && IsFloat(v.dtype) && IsOutFloat(out.dtype),
           std::string(name) + ": float q/k/v, f32/bf16 out");
  VT_CHECK(g_raw.dtype == DType::kF32 && beta.dtype == DType::kF32,
           std::string(name) + ": g_raw/beta must be f32");
  VT_CHECK(a_log.dtype == DType::kF32 && dt_bias.dtype == DType::kF32,
           std::string(name) + ": a_log/dt_bias must be f32");
  VT_CHECK(a_log.rank == 1 && a_log.shape[0] == hv,
           std::string(name) + ": a_log must be [Hv]");
  VT_CHECK(dt_bias.rank == 1 && (dt_bias.shape[0] == hv * dk || dt_bias.shape[0] == 0),
           std::string(name) + ": dt_bias must be [Hv*Dk] or empty");
  VT_CHECK(state.dtype == DType::kF32, std::string(name) + ": state must be f32");
  VT_CHECK(q_in.IsContiguous() && k.IsContiguous() && v.IsContiguous() && out.IsContiguous() &&
               g_raw.IsContiguous() && beta.IsContiguous() && state.IsContiguous(),
           std::string(name) + ": contiguous required");
  VT_CHECK(q_in.device == q.device && k.device == q.device && v.device == q.device &&
               out.device == q.device && g_raw.device == q.device && beta.device == q.device &&
               state.device == q.device,
           std::string(name) + ": device mismatch (q/k/v/out/g_raw/beta/state/queue)");
  VT_CHECK(args.scale > 0.0f, std::string(name) + ": args.scale must be set (> 0)");
  CheckI32Meta(q, query_start_loc, state.shape[0] + 1, name, "query_start_loc");
  reinterpret_cast<KdaChunkPrefillFn>(GetOp(OpId::kKdaChunkPrefill, q.device.type))(
      q, out, q_in, k, v, g_raw, beta, a_log, dt_bias, state, query_start_loc, args);
}

namespace {

// W1 lands `tp_world_size == 1` only. An unimplemented arm is REFUSED with the
// missing piece named, never silently mis-computed (mamba2-ssd.md §7): sharding
// `n_groups` across ranks needs `extra_groups_for_head_shards`
// (mamba_utils.py:187) plus `mamba_v2_sharded_weight_loader`
// (mamba_mixer2.py:174-236), neither of which exists here.
void CheckMamba2NoShard(int64_t tp_world_size, const char* name) {
  VT_CHECK(tp_world_size == 1,
           std::string(name) +
               ": tp_world_size > 1 is NOT implemented (mamba2-ssd.md W1 lands "
               "tp_world_size == 1); n_groups sharding needs "
               "extra_groups_for_head_shards (mamba_utils.py:187)");
}

// A Mamba2 activation/state operand: float in, f32/bf16 out, contiguous, on the
// queue's device. Every SSD tensor check funnels through here so a shape message
// can never disagree with a dtype message.
void CheckMamba2Operand(const Queue& q, const Tensor& t, const char* name, const char* what,
                        bool is_output) {
  VT_CHECK(is_output ? IsOutFloat(t.dtype) : IsFloat(t.dtype),
           std::string(name) + ": " + what + (is_output ? " must be f32/bf16" : " must be float"));
  VT_CHECK(t.IsContiguous(), std::string(name) + ": " + what + " must be contiguous");
  VT_CHECK(t.device == q.device, std::string(name) + ": " + what + " must be on the queue device");
}

}  // namespace

void Mamba2ChunkScan(Queue& q, Tensor& out, Tensor& final_states, const Tensor& x,
                     const Tensor& dt, const Tensor& A, const Tensor& B, const Tensor& C,
                     const Tensor* D, const Tensor* z, const Tensor* dt_bias,
                     const Tensor* initial_states, const Tensor& cu_seqlens,
                     const Tensor& cu_chunk_seqlens, const Tensor& last_chunk_indices,
                     const Tensor& seq_idx, const Mamba2Args& args) {
  constexpr const char* name = "mamba2_chunk_scan";
  CheckMamba2NoShard(args.tp_world_size, name);
  // chunk_size must be an integer power of two — `is_int_pow_2`, asserted at
  // ssd_combined.py:48 before any stage runs.
  VT_CHECK(args.chunk_size > 0 && (args.chunk_size & (args.chunk_size - 1)) == 0,
           std::string(name) + ": chunk_size must be an integer power of 2 "
                               "(ssd_combined.py:48)");
  // dt_limit is caller-supplied and upstream's default is `(0.0, inf)`
  // (ssd_combined.py:180). A negative `dt_min` would let `dt` go negative, which
  // makes `dA_cumsum` NON-monotonic within a chunk and turns upstream's
  // `min(., 0)` clamps (ssd_chunk_state.py:283-285, ssd_chunk_scan.py:339-341)
  // from algebraic no-ops into a silent truncation of the recurrence. The
  // companion precondition `A < 0` is checked against the tensor's contents in
  // the kernel, where the data lives.
  VT_CHECK(args.dt_min >= 0.0f && args.dt_max >= args.dt_min,
           std::string(name) +
               ": dt_limit must satisfy 0 <= dt_min <= dt_max (upstream default is "
               "(0.0, inf), ssd_combined.py:180); a negative dt_min admits a negative "
               "dt and breaks the dA_cumsum monotonicity the intra-chunk clamp rests on");
  VT_CHECK(x.rank == 3 && dt.rank == 2 && A.rank == 1 && B.rank == 3 && C.rank == 3 &&
               out.rank == 3 && final_states.rank == 4,
           std::string(name) +
               ": x/out [T,H,P], dt [T,H], A [H], B/C [T,G,N], final_states [S,H,P,N]");
  const int64_t t = x.shape[0], h = x.shape[1], p = x.shape[2];
  const int64_t g = B.shape[1], n = B.shape[2];
  VT_CHECK(dt.shape[0] == t && dt.shape[1] == h, std::string(name) + ": dt must be [T,H]");
  VT_CHECK(A.shape[0] == h, std::string(name) + ": A must be [H]");
  VT_CHECK(B.shape[0] == t && C.shape[0] == t && C.shape[1] == g && C.shape[2] == n,
           std::string(name) + ": B and C must both be [T,G,N] (ssd_combined.py:52-55)");
  VT_CHECK(g >= 1 && h % g == 0,
           std::string(name) + ": nheads must be divisible by ngroups (ssd_combined.py:51)");
  VT_CHECK(out.shape[0] == t && out.shape[1] == h && out.shape[2] == p,
           std::string(name) + ": out must be [T,H,P]");
  const int64_t s = final_states.shape[0];
  VT_CHECK(final_states.shape[1] == h && final_states.shape[2] == p &&
               final_states.shape[3] == n,
           std::string(name) + ": final_states must be [S,H,P,N]");
  CheckMamba2Operand(q, x, name, "x", false);
  CheckMamba2Operand(q, dt, name, "dt", false);
  CheckMamba2Operand(q, B, name, "B", false);
  CheckMamba2Operand(q, C, name, "C", false);
  CheckMamba2Operand(q, out, name, "out", true);
  // final_states carries `state_dtype` (ssd_combined.py:46,119,176), a knob that
  // is deliberately INDEPENDENT of the activation dtype — never derive one from
  // the other (mamba2-ssd.md §7).
  CheckMamba2Operand(q, final_states, name, "final_states", true);
  VT_CHECK(A.dtype == DType::kF32 && A.IsContiguous() && A.device == q.device,
           std::string(name) + ": A must be f32 contiguous on the queue device");
  if (D != nullptr) {
    VT_CHECK(D->dtype == DType::kF32 && D->IsContiguous() && D->device == q.device,
             std::string(name) + ": D must be f32 contiguous on the queue device");
    VT_CHECK((D->rank == 1 && D->shape[0] == h) ||
                 (D->rank == 2 && D->shape[0] == h && D->shape[1] == p),
             std::string(name) + ": D must be [H] or [H,P] (ssd_combined.py:56-57)");
  }
  if (z != nullptr) {
    VT_CHECK(z->rank == 3 && z->shape[0] == t && z->shape[1] == h && z->shape[2] == p,
             std::string(name) + ": z must be [T,H,P] (ssd_combined.py:54-55)");
    CheckMamba2Operand(q, *z, name, "z", false);
  }
  if (dt_bias != nullptr) {
    VT_CHECK(dt_bias->rank == 1 && dt_bias->shape[0] == h && dt_bias->dtype == DType::kF32 &&
                 dt_bias->IsContiguous() && dt_bias->device == q.device,
             std::string(name) + ": dt_bias must be f32 [H] contiguous on the queue device");
  }
  if (initial_states != nullptr) {
    VT_CHECK(initial_states->rank == 4 && initial_states->shape[0] == s &&
                 initial_states->shape[1] == h && initial_states->shape[2] == p &&
                 initial_states->shape[3] == n,
             std::string(name) +
                 ": initial_states must be [S,H,P,N] (ssd_combined.py:78-79, :194)");
    CheckMamba2Operand(q, *initial_states, name, "initial_states", true);
  }
  CheckI32Meta(q, cu_seqlens, s + 1, name, "cu_seqlens");
  CheckI32Meta(q, last_chunk_indices, s, name, "last_chunk_indices");
  const int64_t nchunks = cu_chunk_seqlens.rank == 1 ? cu_chunk_seqlens.shape[0] - 1 : -1;
  VT_CHECK(nchunks >= 0, std::string(name) + ": cu_chunk_seqlens must be i32 [nchunks+1]");
  CheckI32Meta(q, cu_chunk_seqlens, nchunks + 1, name, "cu_chunk_seqlens");
  // seq_idx is PER CHUNK, not per token — `seq_idx.shape == (nchunks,)`,
  // asserted at ssd_combined.py:60-61 and documented at :189.
  CheckI32Meta(q, seq_idx, nchunks, name, "seq_idx (PER CHUNK, ssd_combined.py:60-61)");
  reinterpret_cast<Mamba2ChunkScanFn>(GetOp(OpId::kMamba2ChunkScan, q.device.type))(
      q, out, final_states, x, dt, A, B, C, D, z, dt_bias, initial_states, cu_seqlens,
      cu_chunk_seqlens, last_chunk_indices, seq_idx, args);
}

void Mamba2StateUpdate(Queue& q, Tensor& out, Tensor& state, const Tensor& x,
                       const Tensor& dt, const Tensor& A, const Tensor& B, const Tensor& C,
                       const Tensor* D, const Tensor* z, const Tensor* dt_bias,
                       const Tensor* state_indices, const Mamba2Args& args) {
  constexpr const char* name = "mamba2_state_update";
  CheckMamba2NoShard(args.tp_world_size, name);
  VT_CHECK(x.rank == 3 && dt.rank == 2 && A.rank == 1 && B.rank == 3 && C.rank == 3 &&
               out.rank == 3 && state.rank == 4,
           std::string(name) +
               ": x/out [Nb,H,P], dt [Nb,H], A [H], B/C [Nb,G,N], state [S,H,P,N]");
  const int64_t nb = x.shape[0], h = x.shape[1], p = x.shape[2];
  const int64_t g = B.shape[1], n = B.shape[2];
  VT_CHECK(dt.shape[0] == nb && dt.shape[1] == h,
           std::string(name) + ": dt must be [Nb,H] (scalar per head; Mamba2 tie_hdim)");
  VT_CHECK(A.shape[0] == h, std::string(name) + ": A must be [H]");
  VT_CHECK(B.shape[0] == nb && C.shape[0] == nb && C.shape[1] == g && C.shape[2] == n,
           std::string(name) + ": B and C must both be [Nb,G,N]");
  VT_CHECK(g >= 1 && h % g == 0,
           std::string(name) + ": nheads must be divisible by ngroups (mamba_ssm.py:583)");
  VT_CHECK(out.shape[0] == nb && out.shape[1] == h && out.shape[2] == p,
           std::string(name) + ": out must be [Nb,H,P]");
  VT_CHECK(state.shape[1] == h && state.shape[2] == p && state.shape[3] == n,
           std::string(name) + ": state must be [S,H,P,N]");
  CheckMamba2Operand(q, x, name, "x", false);
  CheckMamba2Operand(q, dt, name, "dt", false);
  CheckMamba2Operand(q, B, name, "B", false);
  CheckMamba2Operand(q, C, name, "C", false);
  CheckMamba2Operand(q, out, name, "out", true);
  // The SSM cache dtype is its own knob (mamba2_state_dtype, mamba_utils.py:73-81).
  CheckMamba2Operand(q, state, name, "state", true);
  VT_CHECK(A.dtype == DType::kF32 && A.IsContiguous() && A.device == q.device,
           std::string(name) + ": A must be f32 contiguous on the queue device");
  if (D != nullptr) {
    VT_CHECK(D->rank == 1 && D->shape[0] == h && D->dtype == DType::kF32 &&
                 D->IsContiguous() && D->device == q.device,
             std::string(name) + ": D must be f32 [H] contiguous on the queue device");
  }
  if (z != nullptr) {
    VT_CHECK(z->rank == 3 && z->shape[0] == nb && z->shape[1] == h && z->shape[2] == p,
             std::string(name) + ": z must be [Nb,H,P]");
    CheckMamba2Operand(q, *z, name, "z", false);
  }
  if (dt_bias != nullptr) {
    VT_CHECK(dt_bias->rank == 1 && dt_bias->shape[0] == h && dt_bias->dtype == DType::kF32 &&
                 dt_bias->IsContiguous() && dt_bias->device == q.device,
             std::string(name) + ": dt_bias must be f32 [H] contiguous on the queue device");
  }
  if (state_indices == nullptr) {
    VT_CHECK(state.shape[0] == nb,
             std::string(name) +
                 ": without state_indices the state is compact, one row per token");
  } else {
    CheckI32Meta(q, *state_indices, nb, name, "state_indices");
  }
  reinterpret_cast<Mamba2StateUpdateFn>(GetOp(OpId::kMamba2StateUpdate, q.device.type))(
      q, out, state, x, dt, A, B, C, D, z, dt_bias, state_indices, args);
}

void RmsNormGatedGroup(Queue& q, Tensor& out, const Tensor& x, const Tensor& gate,
                       const Tensor* weight, const RmsNormGatedGroupArgs& args) {
  constexpr const char* name = "rms_norm_gated_group";
  CheckMamba2NoShard(args.tp_world_size, name);
  VT_CHECK(x.rank >= 2 && x.rank <= 3, std::string(name) + ": x must be rank 2 or 3");
  VT_CHECK(gate.rank == x.rank && out.rank == x.rank,
           std::string(name) + ": gate and out must have x's rank");
  for (int d = 0; d < x.rank; ++d) {
    VT_CHECK(gate.shape[d] == x.shape[d] && out.shape[d] == x.shape[d],
             std::string(name) + ": gate and out must have x's shape");
  }
  const int64_t hidden = x.shape[x.rank - 1];
  VT_CHECK(args.n_groups >= 1 && hidden % args.n_groups == 0,
           std::string(name) +
               ": n_groups must divide the last dim (group_size = hidden / n_groups, "
               "mamba_mixer2.py:80)");
  CheckMamba2Operand(q, x, name, "x", false);
  CheckMamba2Operand(q, gate, name, "gate", false);
  CheckMamba2Operand(q, out, name, "out", true);
  if (weight != nullptr) {
    VT_CHECK(weight->rank == 1 && weight->shape[0] == hidden,
             std::string(name) + ": weight must be [hidden]");
    CheckMamba2Operand(q, *weight, name, "weight", false);
  }
  VT_CHECK(args.eps > 0.0f, std::string(name) + ": eps must be > 0");
  reinterpret_cast<RmsNormGatedGroupFn>(GetOp(OpId::kRmsNormGatedGroup, q.device.type))(
      q, out, x, gate, weight, args);
}

namespace {

// The geometry checks both Qwen4-Exp gated-residual ops share. Split out so the
// two entry points cannot drift on what `hc_count`/`hidden_size` mean, which is
// exactly how a stream described one way by the read and another by the
// write-back would corrupt in silence.
void CheckQwen4ExpHc(const Qwen4ExpGatedResidualArgs& args, const char* name) {
  // Upstream `Qwen4ExpTextConfig.__post_init__` rejects hc_count <= 1
  // (configuration_qwen4_exp.py:196-197). Mirror it here rather than divide by a
  // number the model can never carry.
  VT_CHECK(args.hc_count > 1,
           std::string(name) + ": hc_count must be > 1 (upstream rejects <= 1), got " +
               std::to_string(args.hc_count));
  VT_CHECK(args.hidden_size > 0,
           std::string(name) + ": hidden_size must be positive, got " +
               std::to_string(args.hidden_size));
}

}  // namespace

void Qwen4ExpGatedResidual(Queue& q, Tensor& mixed, Tensor* injection, const Tensor& hyper,
                           const Tensor& hc_norm_w, const Tensor& mix_down,
                           const Tensor& mix_up, const Tensor* block_inject,
                           const Qwen4ExpGatedResidualArgs& args) {
  constexpr const char* name = "qwen4_exp_gated_residual";
  CheckQwen4ExpHc(args, name);
  VT_CHECK(args.lowrank > 0,
           std::string(name) + ": hc_lowrank must be positive, got " +
               std::to_string(args.lowrank));
  VT_CHECK(args.eps > 0.0f, std::string(name) + ": eps must be > 0");
  const int64_t flat = args.hc_count * args.hidden_size;
  VT_CHECK(hyper.rank == 2 && hyper.shape[1] == flat,
           std::string(name) + ": hyper must be [T, hc_count * hidden_size]");
  const int64_t T = hyper.shape[0];
  VT_CHECK(mixed.rank == 2 && mixed.shape[0] == T && mixed.shape[1] == args.hidden_size,
           std::string(name) + ": mixed must be [T, hidden_size]");
  VT_CHECK(hc_norm_w.rank == 1 && hc_norm_w.shape[0] == flat,
           std::string(name) + ": hc_norm weight must be [hc_count * hidden_size]");
  VT_CHECK(mix_down.rank == 2 && mix_down.shape[0] == args.lowrank &&
               mix_down.shape[1] == flat,
           std::string(name) + ": input_mix_weight_down must be [hc_lowrank, hc*H]");
  VT_CHECK(mix_up.rank == 2 && mix_up.shape[0] == flat && mix_up.shape[1] == args.lowrank,
           std::string(name) + ": input_mix_weight_up must be [hc*H, hc_lowrank]");
  // The `use_combine` switch is a PAIR. One of the two present alone is a caller
  // that has decided the arm one way and allocated for it the other way, which
  // would otherwise read as a silently missing injection.
  VT_CHECK((injection == nullptr) == (block_inject == nullptr),
           std::string(name) +
               ": `injection` and `block_inject` must be null TOGETHER (a null pair is "
               "upstream's use_combine=False mixer)");
  if (block_inject != nullptr) {
    VT_CHECK(block_inject->rank == 2 && block_inject->shape[0] == args.hc_count &&
                 block_inject->shape[1] == flat,
             std::string(name) + ": block_inject_weight must be [hc_count, hc*H]");
    VT_CHECK(injection->rank == 2 && injection->shape[0] == T &&
                 injection->shape[1] == args.hc_count,
             std::string(name) + ": injection must be [T, hc_count]");
  }
  const auto check_operand = [&](const Tensor& t, const char* what, bool is_out) {
    VT_CHECK(IsFloat(t.dtype) && (!is_out || IsOutFloat(t.dtype)),
             std::string(name) + ": " + what + " must be float (f32/bf16 for outputs)");
    VT_CHECK(t.IsContiguous(), std::string(name) + ": " + what + " must be contiguous");
    VT_CHECK(t.device == q.device, std::string(name) + ": " + what + " device mismatch");
  };
  // A PROJECTION OPERAND MAY KEEP THE FILE'S BLOCK ENCODING (W5p, #2031). The
  // released `unsloth/Qwen3.8-Flash-Next-GGUF` stores all 194 hyper-connection
  // mix weights as Q8_0 — `blk.N.hc_{attn,ffn}_{down,up}.weight` plus
  // `output_hc_{down,up}` — and the loader keeps their blocks, so demanding a
  // float here refused the released checkpoint at its first prefill.
  //
  // THE POLICY IS llama.cpp'S, AND IT SPLITS THIS OP'S OPERANDS IN TWO. vLLM
  // registered `qwen4_exp` on 2026-08-31 (#2489) but never loads a GGUF, so it
  // has no opinion on a block-typed operand; llama.cpp merged it on 2026-08-27
  // (`6c84c7d5d`, PR #27742) and runs this exact file. It declares each of the
  // SIX projections `GGML_OP_MUL_MAT` (`src/llama-arch.cpp:759,760,761,763,764,765`
  // — down, up AND inject, on both the attention and the feed-forward side) and
  // consumes them with a plain `build_lora_mm` on the file-typed tensor
  // (`src/models/qwen4exp.cpp:237-241`); it never dequantizes one. It declares
  // `hc_*_norm` `GGML_OP_MUL` (`:758`, `:762`), and where a weight of this
  // architecture meets an ELEMENTWISE multiply it casts to f32 first and says so
  // (`qwen4exp.cpp:1198-1202`, the PLE conv). Matmul operands keep the file's
  // type; elementwise operands get a cast. `hc_norm_w` therefore stays on
  // `check_operand` above, and refusing a block-typed gamma by name is a gated
  // behaviour, not an oversight.
  //
  // The BLOCK layout replaces the elementwise stride contract, exactly as
  // `MatmulBTQuant`'s own validation puts it: a block-typed tensor has no
  // per-element stride, so `IsContiguous()` is not the question — being a whole
  // number of blocks per row is. The shape checks above are unaffected, because
  // `Tensor.shape` is in ELEMENTS for a block dtype too.
  const auto check_projection = [&](const Tensor& t, const char* what) {
    if (!IsBlockQuant(t.dtype)) {
      check_operand(t, what, false);
      return;
    }
    VT_CHECK(t.shape[1] % BlockElems(t.dtype) == 0,
             std::string(name) + ": " + what +
                 " keeps its blocks, so K must be a whole number of them");
    VT_CHECK(t.device == q.device, std::string(name) + ": " + what + " device mismatch");
  };
  check_operand(hyper, "hyper", false);
  check_operand(hc_norm_w, "hc_norm weight", false);
  check_projection(mix_down, "input_mix_weight_down");
  check_projection(mix_up, "input_mix_weight_up");
  check_operand(mixed, "mixed", true);
  if (block_inject != nullptr) {
    check_projection(*block_inject, "block_inject_weight");
    check_operand(*injection, "injection", true);
  }
  reinterpret_cast<Qwen4ExpGatedResidualFn>(
      GetOp(OpId::kQwen4ExpGatedResidual, q.device.type))(
      q, mixed, injection, hyper, hc_norm_w, mix_down, mix_up, block_inject, args);
}

void Qwen4ExpGatedResidualWriteBack(Queue& q, Tensor& hyper, const Tensor& block_out,
                                    const Tensor& injection,
                                    const Qwen4ExpGatedResidualArgs& args) {
  constexpr const char* name = "qwen4_exp_gated_residual_write_back";
  CheckQwen4ExpHc(args, name);
  const int64_t flat = args.hc_count * args.hidden_size;
  VT_CHECK(hyper.rank == 2 && hyper.shape[1] == flat,
           std::string(name) + ": hyper must be [T, hc_count * hidden_size]");
  const int64_t T = hyper.shape[0];
  VT_CHECK(block_out.rank == 2 && block_out.shape[0] == T &&
               block_out.shape[1] == args.hidden_size,
           std::string(name) + ": block output must be [T, hidden_size]");
  VT_CHECK(injection.rank == 2 && injection.shape[0] == T &&
               injection.shape[1] == args.hc_count,
           std::string(name) + ": injection must be [T, hc_count]");
  const auto check_operand = [&](const Tensor& t, const char* what, bool is_out) {
    VT_CHECK(IsFloat(t.dtype) && (!is_out || IsOutFloat(t.dtype)),
             std::string(name) + ": " + what + " must be float (f32/bf16 for outputs)");
    VT_CHECK(t.IsContiguous(), std::string(name) + ": " + what + " must be contiguous");
    VT_CHECK(t.device == q.device, std::string(name) + ": " + what + " device mismatch");
  };
  check_operand(hyper, "hyper", true);
  check_operand(block_out, "block output", false);
  check_operand(injection, "injection", false);
  reinterpret_cast<Qwen4ExpGatedResidualWriteBackFn>(
      GetOp(OpId::kQwen4ExpGatedResidualWriteBack, q.device.type))(q, hyper, block_out,
                                                                  injection, args);
}

namespace {

// The operand checks both QSA ops share. Split out for the same reason the
// gated-residual pair shares one: two entry points that disagree about what
// "contiguous, float, on this queue" means is how a caller silently reads
// somebody else's device memory.
void CheckQsaOperand(const Queue& q, const Tensor& t, const char* name, const char* what,
                     bool is_out, bool require_contiguous = true) {
  VT_CHECK(IsFloat(t.dtype) && (!is_out || IsOutFloat(t.dtype)),
           std::string(name) + ": " + what + " must be float (f32/bf16 for outputs)");
  VT_CHECK(!require_contiguous || t.IsContiguous(),
           std::string(name) + ": " + what + " must be contiguous");
  VT_CHECK(t.device == q.device, std::string(name) + ": " + what + " device mismatch");
}

}  // namespace

void Qwen4ExpQsaCompress(Queue& q, Tensor& block_keys, const Tensor& raw_keys,
                         const Tensor& k_norm_weight, const Tensor& cos, const Tensor& sin,
                         const Qwen4ExpQsaCompressArgs& args) {
  constexpr const char* name = "qwen4_exp_qsa_compress";
  VT_CHECK(args.compress_ratio > 1,
           std::string(name) +
               ": compress_ratio must be > 1 (a ratio of 1 stores one state per token and "
               "is not a compressor), got " +
               std::to_string(args.compress_ratio));
  VT_CHECK(args.eps > 0.0f, std::string(name) + ": eps must be > 0");
  VT_CHECK(raw_keys.rank == 2 && k_norm_weight.rank == 1 && block_keys.rank == 2,
           std::string(name) + ": raw_keys/block_keys must be 2-D and k_norm_weight 1-D");
  const int64_t num_keys = raw_keys.shape[0];
  const int64_t D = raw_keys.shape[1];
  // configuration_qwen4_exp.py:225-231 — `rotary_dim = int(head_dim *
  // partial_rotary_factor)` must FIT the index head, and `rotate_half` needs an
  // even span. Both are refused here rather than at the read.
  VT_CHECK(args.rotary_dim >= 0 && args.rotary_dim <= D,
           std::string(name) + ": rotary_dim must fit the index head dim, got " +
               std::to_string(args.rotary_dim) + " for head dim " + std::to_string(D));
  VT_CHECK(args.rotary_dim % 2 == 0,
           std::string(name) + ": rotary_dim must be even (rotate_half), got " +
               std::to_string(args.rotary_dim));
  // The compressor early-exits unless `(position + 1) % compress_ratio == 0`
  // (compressor_utils.py:52), so a partial block writes NO state. A caller that
  // handed one in has confused the ragged tail — which is attended from the raw
  // KV — with a block, and would silently pool across the end of its own
  // sequence.
  VT_CHECK(num_keys % args.compress_ratio == 0,
           std::string(name) +
               ": raw_keys must be a whole number of COMPLETE blocks; the ragged tail is "
               "attended from the raw KV and writes no state. Got " +
               std::to_string(num_keys) + " keys at compress_ratio " +
               std::to_string(args.compress_ratio));
  const int64_t nb = num_keys / args.compress_ratio;
  VT_CHECK(block_keys.shape[0] == nb && block_keys.shape[1] == D,
           std::string(name) + ": block_keys must be [num_keys / compress_ratio, head_dim]");
  VT_CHECK(k_norm_weight.shape[0] == D,
           std::string(name) + ": k_layernorm weight must be [head_dim]");
  VT_CHECK(cos.rank == 2 && sin.rank == 2 && cos.shape[1] == args.rotary_dim &&
               sin.shape[1] == args.rotary_dim,
           std::string(name) + ": cos/sin must be [positions, rotary_dim]");
  // The tables are indexed at the BLOCK-START position `compress_ratio * b`, so
  // they have to cover every key position the caller handed in, not just nb rows.
  VT_CHECK(cos.shape[0] >= num_keys && sin.shape[0] >= num_keys,
           std::string(name) +
               ": cos/sin must cover every key position (the rope reads row "
               "compress_ratio * b)");
  CheckQsaOperand(q, raw_keys, name, "raw_keys", false);
  CheckQsaOperand(q, k_norm_weight, name, "k_layernorm weight", false);
  CheckQsaOperand(q, cos, name, "cos", false);
  CheckQsaOperand(q, sin, name, "sin", false);
  CheckQsaOperand(q, block_keys, name, "block_keys", true);
  VT_CHECK(cos.dtype == DType::kF32 && sin.dtype == DType::kF32,
           std::string(name) + ": cos/sin must be f32");
  reinterpret_cast<Qwen4ExpQsaCompressFn>(
      GetOp(OpId::kQwen4ExpQsaCompress, q.device.type))(q, block_keys, raw_keys,
                                                        k_norm_weight, cos, sin, args);
}

void Qwen4ExpQsaGatherAttention(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                                const Tensor& value, const Tensor& block_ids,
                                const Tensor& kv_lens, const Qwen4ExpQsaAttnArgs& args) {
  constexpr const char* name = "qwen4_exp_qsa_gather_attention";
  VT_CHECK(args.scale > 0.0f,
           std::string(name) + ": scale must be set explicitly (> 0), the head_dim^-0.5 of "
                               "the MODEL's attention head, not the indexer's");
  VT_CHECK(args.compress_ratio > 1,
           std::string(name) + ": compress_ratio must be > 1, got " +
               std::to_string(args.compress_ratio));
  // The PAGED address mode (W5d-3, #2249 item 2). It changes the RANK and the
  // CONTIGUITY of key/value and nothing else, so the checks below fork exactly
  // there; every other operand is validated once for both arms.
  const bool paged = args.kv_block_table != nullptr;
  VT_CHECK(paged == (args.kv_block_size > 0),
           std::string(name) +
               ": kv_block_table and a positive kv_block_size must be set TOGETHER — a "
               "page table with no page size cannot address a row, and a page size with "
               "no table is a paged read that silently falls back to a contiguous one");
  VT_CHECK(query.rank == 3 && out.rank == 3,
           std::string(name) + ": query/out must be [tokens, heads, head_dim]");
  VT_CHECK(key.rank == value.rank && key.rank == (paged ? 4 : 3),
           std::string(name) + ": key/value must be " +
               (paged ? "[num_pages, kv_block_size, num_kv_heads, head_dim] in the paged "
                        "address mode"
                      : "[max_kv, num_kv_heads, head_dim]"));
  const int64_t T = query.shape[0];
  const int64_t HQ = query.shape[1];
  const int64_t DH = query.shape[2];
  const int64_t HKV = key.shape[paged ? 2 : 1];
  VT_CHECK(HQ > 0 && HKV > 0 && DH > 0, std::string(name) + ": bad attention shape");
  VT_CHECK(HQ % HKV == 0,
           std::string(name) + ": GQA needs num_q_heads divisible by num_kv_heads, got " +
               std::to_string(HQ) + " over " + std::to_string(HKV));
  for (int i = 0; i < key.rank; ++i) {
    VT_CHECK(key.shape[i] == value.shape[i],
             std::string(name) + ": key and value must have the SAME shape");
  }
  VT_CHECK(key.shape[key.rank - 1] == DH,
           std::string(name) + ": the key/value head_dim must match the query's");
  if (paged) {
    const Tensor& bt = *args.kv_block_table;
    VT_CHECK(key.shape[1] == args.kv_block_size,
             std::string(name) + ": the cache view's page height " +
                 std::to_string(key.shape[1]) + " disagrees with kv_block_size " +
                 std::to_string(args.kv_block_size));
    VT_CHECK(bt.rank == 2 && bt.shape[0] == 1 && bt.shape[1] > 0,
             std::string(name) +
                 ": kv_block_table must be [1, max_pages] i32 — this op serves ONE "
                 "sequence per call, as the contiguous arm does");
    VT_CHECK(bt.dtype == DType::kI32, std::string(name) + ": kv_block_table must be i32");
    VT_CHECK(bt.IsContiguous(), std::string(name) + ": kv_block_table must be contiguous");
    VT_CHECK(bt.device == q.device, std::string(name) + ": kv_block_table device mismatch");
    // The row within a page is contiguous even though the PAGE stride is not
    // (K and V interleave at dim 1 of the flash cache), so the kernel resolves a
    // base offset from the strides and then reads `head_dim` elements running.
    VT_CHECK(key.stride[3] == 1 && value.stride[3] == 1,
             std::string(name) +
                 ": a paged key/value view must be contiguous WITHIN a head row");
  }
  VT_CHECK(out.shape[0] == T && out.shape[1] == HQ && out.shape[2] == DH,
           std::string(name) + ": out must match query's shape");
  VT_CHECK(block_ids.rank == 2 && block_ids.shape[0] == T,
           std::string(name) + ": block_ids must be [tokens, block_topk]");
  VT_CHECK(block_ids.dtype == DType::kI32,
           std::string(name) + ": block_ids must be i32 (vt::DsaTopkSelect's output)");
  VT_CHECK(kv_lens.rank == 1 && kv_lens.shape[0] == T,
           std::string(name) + ": kv_lens must be [tokens]");
  VT_CHECK(kv_lens.dtype == DType::kI32, std::string(name) + ": kv_lens must be i32");
  CheckQsaOperand(q, query, name, "query", false);
  // The paged views are STRIDED by construction — `dense_attn::KvSlice` gives
  // each of K and V a page stride of `2 * block_size * Hkv * Dh` — so the
  // contiguity half of the shared check cannot apply to them. Dtype and device
  // still must.
  CheckQsaOperand(q, key, name, "key", false, /*require_contiguous=*/!paged);
  CheckQsaOperand(q, value, name, "value", false, /*require_contiguous=*/!paged);
  CheckQsaOperand(q, out, name, "out", true);
  VT_CHECK(block_ids.IsContiguous() && kv_lens.IsContiguous(),
           std::string(name) + ": block_ids/kv_lens must be contiguous");
  VT_CHECK(block_ids.device == q.device && kv_lens.device == q.device,
           std::string(name) + ": block_ids/kv_lens device mismatch");
  reinterpret_cast<Qwen4ExpQsaGatherAttentionFn>(
      GetOp(OpId::kQwen4ExpQsaGatherAttention, q.device.type))(q, out, query, key, value,
                                                               block_ids, kv_lens, args);
}

void GdnDecode(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
               const Tensor& g, const Tensor& beta, Tensor& state, const GdnArgs& args,
               const Tensor* state_idx) {
  CheckGdnCommon(q, out, q_in, k, v, g, beta, state, args, "gdn_decode",
                 /*allow_compressed_state=*/true);
  if (state_idx == nullptr) {
    VT_CHECK(state.shape[0] == q_in.shape[0],
             "gdn_decode: one state row per token required (single-token sequences)");
  } else {
    // In-place indexed path: state is the FULL cache; one slot index per token.
    CheckI32Meta(q, *state_idx, q_in.shape[0], "gdn_decode", "state_idx");
    VT_CHECK(state.shape[0] >= q_in.shape[0],
             "gdn_decode: indexed cache must have >= batch rows");
  }
  reinterpret_cast<GdnDecodeFn>(GetOp(OpId::kGdnDecode, q.device.type))(
      q, out, q_in, k, v, g, beta, state, state_idx, args);
}

void GdnSpecDecode(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k, const Tensor& v,
                   const Tensor& g, const Tensor& beta, Tensor& state, const Tensor& cu_seqlens,
                   const Tensor& state_indices, const Tensor& num_accepted_tokens,
                   const GdnArgs& args) {
  constexpr const char* name = "gdn_spec_decode";
  CheckGdnCommon(q, out, q_in, k, v, g, beta, state, args, name,
                 /*allow_compressed_state=*/true);
  VT_CHECK(state_indices.rank == 2 && state_indices.dtype == DType::kI32 &&
               state_indices.IsContiguous() && state_indices.device == q.device,
           std::string(name) + ": state_indices must be i32 [num_reqs, num_cols] contiguous");
  const int64_t num_reqs = state_indices.shape[0];
  const int64_t num_cols = state_indices.shape[1];
  VT_CHECK(num_cols >= 1, std::string(name) + ": state_indices needs >= 1 column");
  CheckI32Meta(q, cu_seqlens, num_reqs + 1, name, "cu_seqlens");
  CheckI32Meta(q, num_accepted_tokens, num_reqs, name, "num_accepted_tokens");
  VT_CHECK(state.shape[0] >= 1, std::string(name) + ": state cache must be non-empty");
  if (q.device.type == DeviceType::kCPU) {
    // CPU can validate the engine-owned index metadata without a sync; the CUDA
    // kernel keeps its own per-slot bounds guard instead (no capture-breaking
    // D2H). Mirrors the GdnPackedDecode contract.
    const int32_t* cs = cu_seqlens.Ptr<int32_t>();
    const int32_t* nat = num_accepted_tokens.Ptr<int32_t>();
    const int32_t* idx = state_indices.Ptr<int32_t>();
    VT_CHECK(cs[0] == 0 && cs[num_reqs] == q_in.shape[0],
             std::string(name) + ": bad cu_seqlens bounds");
    for (int64_t i = 0; i < num_reqs; ++i) {
      VT_CHECK(cs[i + 1] >= cs[i] && cs[i + 1] - cs[i] <= num_cols,
               std::string(name) + ": query length exceeds the spec slot count");
      VT_CHECK(nat[i] >= 1 && nat[i] <= num_cols,
               std::string(name) + ": num_accepted_tokens out of range");
      for (int64_t c = 0; c < num_cols; ++c) {
        VT_CHECK(idx[i * num_cols + c] < state.shape[0],
                 std::string(name) + ": state_indices out of range");
      }
    }
  }
  reinterpret_cast<GdnSpecDecodeFn>(GetOp(OpId::kGdnSpecDecode, q.device.type))(
      q, out, q_in, k, v, g, beta, state, cu_seqlens, state_indices, num_accepted_tokens,
      args);
}

void GdnPackedDecode(Queue& q, Tensor& out, const Tensor& mixed_qkv,
                     const Tensor& a, const Tensor& b, const Tensor& a_log,
                     const Tensor& dt_bias, Tensor& state,
                     const Tensor& state_idx, const GdnArgs& args) {
  constexpr const char* name = "gdn_packed_decode";
  VT_CHECK(mixed_qkv.rank == 2 && a.rank == 2 && b.rank == 2 &&
               a_log.rank == 1 && dt_bias.rank == 1 && out.rank == 3 &&
               state.rank == 4,
           "gdn_packed_decode: mixed_qkv/a/b rank-2, A_log/dt_bias rank-1, "
           "out rank-3, state rank-4 required");
  const int64_t batch = mixed_qkv.shape[0];
  const int64_t hv = state.shape[1];
  const int64_t dv = state.shape[2];
  const int64_t dk = state.shape[3];
  VT_CHECK(a.shape[0] == batch && b.shape[0] == batch && a.shape[1] == hv &&
               b.shape[1] == hv,
           "gdn_packed_decode: a/b must be [B,Hv]");
  VT_CHECK(a_log.shape[0] == hv && dt_bias.shape[0] == hv,
           "gdn_packed_decode: A_log/dt_bias must be [Hv]");
  VT_CHECK(out.shape[0] == batch && out.shape[1] == hv && out.shape[2] == dv,
           "gdn_packed_decode: out must be [B,Hv,Dv]");
  CheckI32Meta(q, state_idx, batch, name, "state_idx");

  const int64_t qk_dim = mixed_qkv.shape[1] - hv * dv;
  VT_CHECK(qk_dim > 0 && qk_dim % 2 == 0,
           "gdn_packed_decode: invalid packed mixed_qkv width");
  const int64_t q_dim = qk_dim / 2;
  VT_CHECK(dk > 0 && q_dim % dk == 0,
           "gdn_packed_decode: packed Q width must be divisible by Dk");
  const int64_t hk = q_dim / dk;
  VT_CHECK(hk > 0 && hv % hk == 0,
           "gdn_packed_decode: Hv must be a multiple of inferred Hk");

  VT_CHECK(IsFloat(mixed_qkv.dtype) && a.dtype == mixed_qkv.dtype &&
               b.dtype == mixed_qkv.dtype && out.dtype == mixed_qkv.dtype,
           "gdn_packed_decode: mixed_qkv/a/b/out must share FP16/BF16/F32 dtype");
  VT_CHECK(IsFloat(state.dtype),
           "gdn_packed_decode: state must use an independent FP16/BF16/F32 dtype");
  VT_CHECK(IsFloat(a_log.dtype) && IsFloat(dt_bias.dtype),
           "gdn_packed_decode: A_log/dt_bias must each use a floating dtype");
  VT_CHECK(mixed_qkv.stride[1] == 1 &&
               mixed_qkv.stride[0] >= mixed_qkv.shape[1] && a.stride[1] == 1 &&
               a.stride[0] >= hv && b.stride[1] == 1 && b.stride[0] >= hv,
           "gdn_packed_decode: mixed_qkv/a/b require inner-contiguous non-overlapping rows");
  VT_CHECK(a_log.IsContiguous() && dt_bias.IsContiguous() && out.IsContiguous() &&
               state.IsContiguous(),
           "gdn_packed_decode: A_log/dt_bias/out/state must be contiguous");
  VT_CHECK(mixed_qkv.device == q.device && a.device == q.device &&
               b.device == q.device && a_log.device == q.device &&
               dt_bias.device == q.device && out.device == q.device &&
               state.device == q.device,
           "gdn_packed_decode: device mismatch");
  VT_CHECK(args.scale > 0.0f,
           "gdn_packed_decode: args.scale must be set (> 0)");

  // CPU can validate the index values without synchronization. CUDA metadata
  // is engine-owned and remains device-resident; its kernel independently
  // bounds-checks each slot before dereferencing it.
  if (q.device.type == DeviceType::kCPU) {
    std::vector<uint8_t> seen(static_cast<size_t>(state.shape[0]), 0);
    const int32_t* idx = state_idx.Ptr<int32_t>();
    for (int64_t i = 0; i < batch; ++i) {
      if (idx[i] < 0) continue;
      VT_CHECK(idx[i] < state.shape[0],
               "gdn_packed_decode: state_idx out of range");
      VT_CHECK(seen[static_cast<size_t>(idx[i])] == 0,
               "gdn_packed_decode: duplicate live state_idx");
      seen[static_cast<size_t>(idx[i])] = 1;
    }
  }

  reinterpret_cast<GdnPackedDecodeFn>(
      GetOp(OpId::kGdnPackedDecode, q.device.type))(
      q, out, mixed_qkv, a, b, a_log, dt_bias, state, state_idx, args);
}

namespace {
void CheckGdnStateIo(const Queue& q, const Tensor& working,
                     const Tensor& cache, const Tensor& state_idx,
                     const char* name) {
  VT_CHECK(cache.rank >= 2 && cache.rank <= kMaxRank,
           std::string(name) + ": cache rank must be in [2,4]");
  VT_CHECK(working.rank == cache.rank,
           std::string(name) + ": working/cache ranks must match");
  VT_CHECK(state_idx.rank == 1 && state_idx.dtype == DType::kI32 &&
               state_idx.IsContiguous(),
           std::string(name) + ": state_idx must be contiguous i32 [N]");
  VT_CHECK(working.shape[0] == state_idx.shape[0],
           std::string(name) + ": working rows must match state_idx");
  // The middle dims (channels/heads) must match exactly. The INNERMOST dim of
  // the cache may be WIDER than the working buffer: the GDN conv state row is
  // widened to (K-1)+num_spec taps for spec-decode rollback, while the non-spec
  // gather/scatter operates on the LEADING (K-1) sub-window per channel with the
  // cache row's physical stride (mirror vLLM `state_len=KERNEL_WIDTH-1`). At
  // num_spec==0 the inner dims are equal, so this is the previous exact-match
  // check byte-for-byte; the SSM state cache (rank 4) is never widened, so its
  // inner dim always matches and it always takes the contiguous fast path.
  for (int d = 1; d < cache.rank - 1; ++d) {
    VT_CHECK(working.shape[d] == cache.shape[d],
             std::string(name) + ": working/cache row shapes must match");
  }
  VT_CHECK(cache.shape[cache.rank - 1] >= working.shape[working.rank - 1],
           std::string(name) + ": cache inner dim must be >= working inner dim");
  VT_CHECK(working.dtype == DType::kF32,
           std::string(name) + ": working state must be f32");
  VT_CHECK(cache.dtype == DType::kF32 || cache.dtype == DType::kF16 ||
               cache.dtype == DType::kBF16,
           std::string(name) + ": cache must be fp16, bf16, or f32");
  VT_CHECK(working.IsContiguous() && cache.IsContiguous(),
           std::string(name) + ": working/cache must be contiguous");
  VT_CHECK(working.device == q.device && cache.device == q.device &&
               state_idx.device == q.device,
           std::string(name) + ": device mismatch");
}
}  // namespace

void GdnStateGather(Queue& q, Tensor& working, const Tensor& cache,
                    const Tensor& state_idx,
                    const Tensor* has_initial_state) {
  CheckGdnStateIo(q, working, cache, state_idx, "gdn_state_gather");
  if (has_initial_state != nullptr) {
    VT_CHECK(has_initial_state->rank == 1 &&
                 has_initial_state->shape[0] == state_idx.shape[0] &&
                 (has_initial_state->dtype == DType::kI8 ||
                  has_initial_state->dtype == DType::kI32) &&
                 has_initial_state->IsContiguous() &&
                 has_initial_state->device == q.device,
             "gdn_state_gather: has_initial_state must be contiguous i8/i32 [N] on device");
  }
  reinterpret_cast<GdnStateGatherFn>(
      GetOp(OpId::kGdnStateGather, q.device.type))(
      q, working, cache, state_idx, has_initial_state);
}

void GdnStateScatter(Queue& q, Tensor& cache, const Tensor& working,
                     const Tensor& state_idx) {
  CheckGdnStateIo(q, working, cache, state_idx, "gdn_state_scatter");
  reinterpret_cast<GdnStateScatterFn>(
      GetOp(OpId::kGdnStateScatter, q.device.type))(
      q, cache, working, state_idx);
}

namespace {
// Common shape/dtype contract for the row gather/scatter ops. `many` is the
// [M, D...] compact side (out for select, in for copy); `base` is the [N, D...]
// row-addressable side (in for select, out for copy). D... must match and the
// compact side must be contiguous; the base side may carry an outer row stride.
void CheckIndexRowOp(Queue& q, const Tensor& compact, const Tensor& base,
                     const Tensor& idx, const char* name) {
  VT_CHECK(compact.dtype == base.dtype,
           std::string(name) + ": dtype mismatch");
  VT_CHECK(compact.rank == base.rank && compact.rank >= 1,
           std::string(name) + ": rank mismatch / must be >= 1");
  VT_CHECK(idx.rank == 1 && idx.dtype == DType::kI32,
           std::string(name) + ": idx must be i32 [M]");
  VT_CHECK(idx.shape[0] == compact.shape[0],
           std::string(name) + ": idx length must equal the compact row count");
  for (int r = 1; r < compact.rank; ++r)
    VT_CHECK(compact.shape[r] == base.shape[r],
             std::string(name) + ": trailing dims must match");
  VT_CHECK(compact.IsContiguous(),
           std::string(name) + ": the compact [M,D] side must be contiguous");
  // The base side's inner dims (everything past dim 0) must be contiguous so a
  // row is one packed block; only the outer row stride may differ.
  int64_t inner = 1;
  for (int r = base.rank - 1; r >= 1; --r) {
    VT_CHECK(base.stride[r] == inner,
             std::string(name) + ": base inner dims must be contiguous");
    inner *= base.shape[r];
  }
  VT_CHECK(base.stride[0] >= inner,
           std::string(name) + ": base outer row stride too small");
  VT_CHECK(compact.device == q.device && base.device == q.device &&
               idx.device == q.device,
           std::string(name) + ": device mismatch");
}
}  // namespace

void IndexSelect(Queue& q, Tensor& out, const Tensor& in, const Tensor& idx) {
  CheckIndexRowOp(q, out, in, idx, "index_select");
  reinterpret_cast<IndexSelectFn>(GetOp(OpId::kIndexSelect, q.device.type))(
      q, out, in, idx);
}

void IndexCopy(Queue& q, Tensor& out, const Tensor& in, const Tensor& idx) {
  CheckIndexRowOp(q, in, out, idx, "index_copy");
  reinterpret_cast<IndexCopyFn>(GetOp(OpId::kIndexCopy, q.device.type))(
      q, out, in, idx);
}

void VHeadPermute(Queue& q, Tensor& out, const Tensor& in,
                  const VHeadPermuteArgs& args) {
  VT_CHECK(out.rank == 2 && in.rank == 2,
           "v_head_permute: out/in must be rank-2 [T, N]");
  VT_CHECK(out.dtype == in.dtype, "v_head_permute: dtype mismatch");
  VT_CHECK(out.device == q.device && in.device == q.device,
           "v_head_permute: device mismatch");
  VT_CHECK(out.shape[0] == in.shape[0] && out.shape[1] == in.shape[1],
           "v_head_permute: out/in shape mismatch");
  VT_CHECK(args.num_key_heads > 0 && args.heads_per_key > 0 &&
               args.head_width > 0 && args.prefix_elems >= 0,
           "v_head_permute: non-positive geometry");
  const int64_t body =
      args.num_key_heads * args.heads_per_key * args.head_width;
  VT_CHECK(args.prefix_elems + body == in.shape[1],
           "v_head_permute: prefix_elems + K*R*head_width != N (" +
               std::to_string(args.prefix_elems) + " + " +
               std::to_string(body) + " != " + std::to_string(in.shape[1]) +
               ")");
  // Inner-contiguous, outer row stride may be padded. ALIASING IS REFUSED
  // rather than tolerated: this is a gather, so an in-place call would read
  // elements it has already overwritten and the permutation is not an
  // involution at the released 16-vs-48 ratio (it IS one at K == R, which is
  // exactly the fixture shape that could not detect the difference).
  VT_CHECK(out.stride[1] == 1 && in.stride[1] == 1,
           "v_head_permute: inner dim must be contiguous");
  VT_CHECK(out.stride[0] >= out.shape[1] && in.stride[0] >= in.shape[1],
           "v_head_permute: outer row stride too small");
  VT_CHECK(out.data != in.data, "v_head_permute: out and in must not alias");
  if (in.shape[0] == 0) return;
  reinterpret_cast<VHeadPermuteFn>(GetOp(OpId::kVHeadPermute, q.device.type))(
      q, out, in, args);
}

void MoeRouterTopK(Queue& q, Tensor& weights, Tensor& indices, const Tensor& logits,
                   const MoeRouterTopKArgs& args,
                   const Tensor* e_score_correction_bias) {
  VT_CHECK(logits.rank == 2 && weights.rank == 2 && indices.rank == 2,
           "moe_router_topk: logits/weights/indices rank-2");
  const int64_t t = logits.shape[0], e = logits.shape[1];
  // --- grouped-topk (`noaux_tc`) argument contract (W3). num_expert_group == 0
  // is the pre-W3 ungrouped path and must reject every grouped-only knob, so an
  // args struct that was half-filled fails loudly instead of silently ignoring
  // the grouping the caller asked for.
  if (args.num_expert_group > 0) {
    VT_CHECK(e % args.num_expert_group == 0,
             "moe_router_topk: num_experts must be divisible by num_expert_group");
    VT_CHECK(args.topk_group >= 1 && args.topk_group <= args.num_expert_group,
             "moe_router_topk: topk_group must be in [1, num_expert_group]");
    // grouped_topk_router.py masks all but topk_group groups, so at most
    // topk_group * (E / n_group) experts remain selectable.
    VT_CHECK(static_cast<int64_t>(args.top_k) <=
                 static_cast<int64_t>(args.topk_group) * (e / args.num_expert_group),
             "moe_router_topk: top_k exceeds the experts surviving the group mask");
  } else {
    VT_CHECK(args.topk_group == 0,
             "moe_router_topk: topk_group requires num_expert_group > 0");
    VT_CHECK(args.scoring_func == MoeScoringFunc::kSoftmax,
             "moe_router_topk: sigmoid scoring is only defined on the grouped "
             "(noaux_tc) path");
    VT_CHECK(args.routed_scaling_factor == 1.0f,
             "moe_router_topk: routed_scaling_factor requires num_expert_group > 0");
    VT_CHECK(e_score_correction_bias == nullptr,
             "moe_router_topk: e_score_correction_bias requires num_expert_group > 0");
  }
  if (e_score_correction_bias != nullptr) {
    const Tensor& bias = *e_score_correction_bias;
    VT_CHECK(bias.rank == 1 && bias.shape[0] == e,
             "moe_router_topk: e_score_correction_bias must be [num_experts]");
    VT_CHECK(bias.dtype == DType::kF32,
             "moe_router_topk: e_score_correction_bias must be f32");
    VT_CHECK(bias.IsContiguous(),
             "moe_router_topk: e_score_correction_bias must be contiguous");
    VT_CHECK(bias.device == q.device,
             "moe_router_topk: device mismatch (e_score_correction_bias/queue)");
  }
  VT_CHECK(args.top_k >= 1 && args.top_k <= e,
           "moe_router_topk: top_k must be in [1, num_experts]");
  VT_CHECK(weights.shape[0] == t && weights.shape[1] == args.top_k,
           "moe_router_topk: weights must be [T, top_k]");
  VT_CHECK(indices.shape[0] == t && indices.shape[1] == args.top_k,
           "moe_router_topk: indices must be [T, top_k]");
  VT_CHECK(IsFloat(logits.dtype), "moe_router_topk: logits must be a float dtype");
  VT_CHECK(weights.dtype == DType::kF32, "moe_router_topk: weights must be f32");
  VT_CHECK(indices.dtype == DType::kI32, "moe_router_topk: indices must be i32");
  VT_CHECK(logits.IsContiguous() && weights.IsContiguous() && indices.IsContiguous(),
           "moe_router_topk: contiguous required");
  VT_CHECK(logits.device == q.device && weights.device == q.device &&
               indices.device == q.device,
           "moe_router_topk: device mismatch (logits/weights/indices/queue)");
  reinterpret_cast<MoeRouterTopKFn>(GetOp(OpId::kMoeRouterTopK, q.device.type))(
      q, weights, indices, logits, args, e_score_correction_bias);
}

void MoeCombine(Queue& q, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                const Tensor* shared, float routed_scale) {
  VT_CHECK(expert_out.rank == 3 && weights.rank == 2 && out.rank == 2,
           "moe_combine: expert_out [T,K,H], weights [T,K], out [T,H]");
  const int64_t t = out.shape[0], h = out.shape[1], k = weights.shape[1];
  VT_CHECK(expert_out.shape[0] == t && expert_out.shape[1] == k && expert_out.shape[2] == h,
           "moe_combine: expert_out must be [T,K,H] matching out/weights");
  VT_CHECK(weights.shape[0] == t, "moe_combine: weights token count must match out");
  VT_CHECK(IsFloat(expert_out.dtype) && IsOutFloat(out.dtype),
           "moe_combine: float expert_out, f32/bf16 out");
  VT_CHECK(weights.dtype == DType::kF32, "moe_combine: weights must be f32");
  VT_CHECK(expert_out.IsContiguous() && weights.IsContiguous() && out.IsContiguous(),
           "moe_combine: contiguous required");
  VT_CHECK(expert_out.device == q.device && weights.device == q.device &&
               out.device == q.device,
           "moe_combine: device mismatch (expert_out/weights/out/queue)");
  if (shared != nullptr) {
    VT_CHECK(shared->rank == 2 && shared->shape[0] == t && shared->shape[1] == h &&
                 IsFloat(shared->dtype) && shared->IsContiguous() &&
                 shared->device == q.device,
             "moe_combine: shared must be float [T,H] contiguous on the queue device");
  }
  reinterpret_cast<MoeCombineFn>(GetOp(OpId::kMoeCombine, q.device.type))(
      q, out, expert_out, weights, shared, routed_scale);
}

void MoeCombineGate(Queue& q, Tensor& out, const Tensor& expert_out, const Tensor& weights,
                    const Tensor& sd, const Tensor& gl) {
  VT_CHECK(expert_out.rank == 3 && weights.rank == 2 && out.rank == 2,
           "moe_combine_gate: expert_out [T,K,H], weights [T,K], out [T,H]");
  const int64_t t = out.shape[0], h = out.shape[1], k = weights.shape[1];
  VT_CHECK(expert_out.shape[0] == t && expert_out.shape[1] == k && expert_out.shape[2] == h,
           "moe_combine_gate: expert_out must be [T,K,H] matching out/weights");
  VT_CHECK(weights.shape[0] == t, "moe_combine_gate: weights token count must match out");
  VT_CHECK(IsFloat(expert_out.dtype) && IsOutFloat(out.dtype),
           "moe_combine_gate: float expert_out, f32/bf16 out");
  VT_CHECK(weights.dtype == DType::kF32, "moe_combine_gate: weights must be f32");
  // sd may be bf16: the shared down-proj GEMM produces bf16 and the gate
  // re-rounds through bf16 anyway, so reading it directly is bit-identical to
  // casting to f32 first (VT_SHARED_DOWN_BF16).
  VT_CHECK((sd.dtype == DType::kF32 || sd.dtype == DType::kBF16) && sd.rank == 2 &&
               sd.shape[0] == t && sd.shape[1] == h,
           "moe_combine_gate: sd must be f32/bf16 [T,H]");
  VT_CHECK(gl.dtype == DType::kF32 && gl.Numel() == t,
           "moe_combine_gate: gl must be f32 with T elements");
  VT_CHECK(expert_out.IsContiguous() && weights.IsContiguous() && out.IsContiguous() &&
               sd.IsContiguous() && gl.IsContiguous(),
           "moe_combine_gate: contiguous required");
  VT_CHECK(expert_out.device == q.device && weights.device == q.device &&
               out.device == q.device && sd.device == q.device && gl.device == q.device,
           "moe_combine_gate: device mismatch");
  reinterpret_cast<MoeCombineGateFn>(GetOp(OpId::kMoeCombineGate, q.device.type))(
      q, out, expert_out, weights, sd, gl);
}

void Attention(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
               const Tensor& value, const AttentionArgs& args) {
  VT_CHECK(query.rank == 3 && key.rank == 3 && value.rank == 3 && out.rank == 3,
           "attention: query/key/value/out rank-3 [T,H,D]");
  const int64_t t = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t hk = key.shape[1];
  VT_CHECK(key.shape[0] == t && value.shape[0] == t,
           "attention: query/key/value token count must match");
  VT_CHECK(key.shape[2] == d && value.shape[2] == d,
           "attention: key/value head_dim must match query");
  VT_CHECK(value.shape[1] == hk, "attention: key/value must share the kv-head count");
  VT_CHECK(out.shape[0] == t && out.shape[1] == hq && out.shape[2] == d,
           "attention: out must be [T,Hq,D] matching query");
  VT_CHECK(hk >= 1 && hq >= 1 && hq % hk == 0,
           "attention: Hq must be a positive multiple of Hk (GQA broadcast)");
  VT_CHECK(args.scale > 0.0f, "attention: scale must be set (> 0), e.g. head_dim^-0.5");
  VT_CHECK(IsFloat(query.dtype) && key.dtype == query.dtype && value.dtype == query.dtype,
           "attention: query/key/value must share one float dtype");
  VT_CHECK(IsOutFloat(out.dtype), "attention: out must be f32 or bf16");
  VT_CHECK(query.IsContiguous() && key.IsContiguous() && value.IsContiguous() &&
               out.IsContiguous(),
           "attention: contiguous tensors required");
  VT_CHECK(query.device == q.device && key.device == q.device && value.device == q.device &&
               out.device == q.device,
           "attention: device mismatch (query/key/value/out/queue)");
  reinterpret_cast<AttentionFn>(GetOp(OpId::kAttention, q.device.type))(q, out, query, key,
                                                                        value, args);
}

// Dense non-causal CROSS attention (LTX-2.5 L2). Same validation shape as
// vt::Attention, MINUS the token-count equality it enforces between query and
// key/value — that equality is precisely what a cross-attention cannot satisfy.
void AttentionCross(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                    const Tensor& value, const Tensor* bias, const AttentionCrossArgs& args) {
  VT_CHECK(query.rank == 3 && key.rank == 3 && value.rank == 3 && out.rank == 3,
           "attention_cross: query/key/value/out rank-3 [T,H,D]");
  const int64_t tq = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t s = key.shape[0], hk = key.shape[1];
  VT_CHECK(tq > 0 && s > 0, "attention_cross: query and key token counts must be positive");
  VT_CHECK(value.shape[0] == s, "attention_cross: key/value token counts must match");
  VT_CHECK(key.shape[2] == d && value.shape[2] == d,
           "attention_cross: key/value head_dim must match query");
  VT_CHECK(value.shape[1] == hk, "attention_cross: key/value must share the kv-head count");
  VT_CHECK(out.shape[0] == tq && out.shape[1] == hq && out.shape[2] == d,
           "attention_cross: out must be [Tq,Hq,D] matching query");
  VT_CHECK(hk >= 1 && hq >= 1 && hq % hk == 0,
           "attention_cross: Hq must be a positive multiple of Hkv (GQA broadcast)");
  VT_CHECK(args.scale > 0.0f, "attention_cross: scale must be set (> 0), e.g. head_dim^-0.5");
  VT_CHECK(IsFloat(query.dtype) && key.dtype == query.dtype && value.dtype == query.dtype,
           "attention_cross: query/key/value must share one float dtype");
  VT_CHECK(IsOutFloat(out.dtype), "attention_cross: out must be f32 or bf16");
  VT_CHECK(query.IsContiguous() && key.IsContiguous() && value.IsContiguous() &&
               out.IsContiguous(),
           "attention_cross: contiguous tensors required");
  VT_CHECK(query.device == q.device && key.device == q.device && value.device == q.device &&
               out.device == q.device,
           "attention_cross: device mismatch (query/key/value/out/queue)");
  if (bias != nullptr) {
    VT_CHECK(bias->rank == 2, "attention_cross: bias must be rank-2 [Tq or 1, S]");
    VT_CHECK(bias->shape[0] == tq || bias->shape[0] == 1,
             "attention_cross: bias rows must be Tq or 1 (key-only broadcast)");
    VT_CHECK(bias->shape[1] == s, "attention_cross: bias columns must equal the key count");
    VT_CHECK(bias->dtype == DType::kF32, "attention_cross: bias must be f32");
    VT_CHECK(bias->IsContiguous(), "attention_cross: bias must be contiguous");
    VT_CHECK(bias->device == q.device, "attention_cross: bias device mismatch");
  }
  reinterpret_cast<AttentionCrossFn>(GetOp(OpId::kAttentionCross, q.device.type))(
      q, out, query, key, value, bias, args);
}

// --- Conformer / FastConformer audio-encoder kernels (spike P1/P2/P3) --------
// Upstream mirror: transformers 5.3.0
// transformers/models/parakeet/modeling_parakeet.py (:357 subsampling Conv2d,
// :116 convolution module depthwise Conv1d, :259 relative-position attention),
// which is the module vLLM itself runs (parakeet.py:37,62). The validation here
// mirrors torch's own shape contracts for nn.Conv2d / nn.Conv1d(groups=C) so a
// caller that passes what the Python module passes is accepted verbatim.

void Conv2d(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight, const Tensor* bias,
            const Conv2dArgs& args) {
  VT_CHECK(x.rank == 4 && out.rank == 4, "conv2d: x/out must be rank-4 [N,C,H,W]");
  VT_CHECK(weight.rank == 4, "conv2d: weight must be rank-4 [Cout,Cin/groups,KH,KW]");
  const int64_t g = args.groups;
  VT_CHECK(g >= 1, "conv2d: groups must be >= 1");
  const int64_t n = x.shape[0], cin = x.shape[1], hin = x.shape[2], win = x.shape[3];
  const int64_t cout = weight.shape[0], cin_g = weight.shape[1];
  const int64_t kh = weight.shape[2], kw = weight.shape[3];
  VT_CHECK(n >= 0 && cin > 0 && hin > 0 && win > 0, "conv2d: x extents must be positive");
  VT_CHECK(cout > 0 && kh > 0 && kw > 0, "conv2d: weight extents must be positive");
  VT_CHECK(cin % g == 0 && cout % g == 0, "conv2d: groups must divide both Cin and Cout");
  VT_CHECK(cin_g == cin / g, "conv2d: weight dim 1 must be Cin/groups");
  VT_CHECK(args.stride_h >= 1 && args.stride_w >= 1, "conv2d: stride must be >= 1");
  VT_CHECK(args.dilation_h >= 1 && args.dilation_w >= 1, "conv2d: dilation must be >= 1");
  VT_CHECK(args.pad_h >= 0 && args.pad_w >= 0, "conv2d: padding must be >= 0");
  const int64_t hout = (hin + 2 * args.pad_h - args.dilation_h * (kh - 1) - 1) / args.stride_h + 1;
  const int64_t wout = (win + 2 * args.pad_w - args.dilation_w * (kw - 1) - 1) / args.stride_w + 1;
  VT_CHECK(hout > 0 && wout > 0, "conv2d: kernel/dilation larger than the padded input");
  VT_CHECK(out.shape[0] == n && out.shape[1] == cout && out.shape[2] == hout &&
               out.shape[3] == wout,
           "conv2d: out must be [N,Cout,Hout,Wout] for the given stride/padding/dilation");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(weight.dtype) && IsFloat(out.dtype),
           "conv2d: x/weight/out must be f32, f16 or bf16");
  VT_CHECK(x.IsContiguous() && weight.IsContiguous() && out.IsContiguous(),
           "conv2d: contiguous tensors required");
  VT_CHECK(x.device == q.device && weight.device == q.device && out.device == q.device,
           "conv2d: device mismatch (x/weight/out/queue)");
  if (bias != nullptr) {
    VT_CHECK(bias->rank == 1 && bias->shape[0] == cout, "conv2d: bias must be rank-1 [Cout]");
    VT_CHECK(IsFloat(bias->dtype) && bias->IsContiguous() && bias->device == q.device,
             "conv2d: bias must be a contiguous float tensor on the queue device");
  }
  reinterpret_cast<Conv2dFn>(GetOp(OpId::kConv2d, q.device.type))(q, out, x, weight, bias, args);
}

// --- General 3-D convolution (LTX25-DEVICE-RESIDENCY W5, #1007) --------------
// Mirrors torch `nn.Conv3d`'s own shape contract at batch 1, so a caller that
// passes what `CausalConv3d` passes (Lightricks/LTX-2 @ fd4ded7f2,
// video_vae/convolution.py:292-302) is accepted verbatim. The rank-5 shapes
// torch uses are expressed at rank 4 for the reason include/vt/ops.h states at
// vt::Conv3d; both foldings are CHECKED here rather than assumed.
void Conv3d(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight, const Tensor* bias,
            const Conv3dArgs& args) {
  VT_CHECK(x.rank == 4 && out.rank == 4,
           "conv3d: x/out must be rank-4 [C,T,H,W] — this op is BATCH 1, because vt::Tensor caps "
           "rank at 4; pass one clip per call rather than folding N into a channel axis");
  VT_CHECK(weight.rank == 4, "conv3d: weight must be rank-4 [Cout*Cin/groups,KT,KH,KW]");
  const int64_t g = args.groups;
  VT_CHECK(g >= 1, "conv3d: groups must be >= 1");
  const int64_t cin = x.shape[0], tin = x.shape[1], hin = x.shape[2], win = x.shape[3];
  const int64_t cout = out.shape[0];
  const int64_t kt = weight.shape[1], kh = weight.shape[2], kw = weight.shape[3];
  VT_CHECK(cin > 0 && tin > 0 && hin > 0 && win > 0, "conv3d: x extents must be positive");
  VT_CHECK(cout > 0 && kt > 0 && kh > 0 && kw > 0,
           "conv3d: out channels and kernel extents must be positive");
  VT_CHECK(cin % g == 0 && cout % g == 0, "conv3d: groups must divide both Cin and Cout");
  const int64_t cin_g = cin / g;
  VT_CHECK(weight.shape[0] == cout * cin_g,
           "conv3d: weight dim 0 must be Cout*(Cin/groups) — torch's [Cout,Cin/groups,KT,KH,KW] "
           "with its two leading axes merged");
  VT_CHECK(args.stride_t >= 1 && args.stride_h >= 1 && args.stride_w >= 1,
           "conv3d: stride must be >= 1");
  VT_CHECK(args.dilation_t >= 1 && args.dilation_h >= 1 && args.dilation_w >= 1,
           "conv3d: dilation must be >= 1");
  VT_CHECK(args.pad_t >= 0 && args.pad_h >= 0 && args.pad_w >= 0,
           "conv3d: padding must be >= 0");
  // The SPAN is separated from the division, and that is the whole of the
  // shape contract's agreement with torch (#1007 fresh review F7).
  //
  // torch FLOORS `(in + 2*pad - dilation*(k-1) - 1) / stride`; C++ integer
  // division TRUNCATES TOWARD ZERO. The two agree for a non-negative numerator
  // and disagree for a negative one whenever stride > 1: at
  // `tin = 2, k = 3, stride = 2, pad = 0` the numerator is -1, so torch gets
  // `floor(-1/2) + 1 = 0` and raises "Output size is too small" while truncation
  // gets `-1/2 + 1 = 1` and ACCEPTS an extent of 1, convolving over taps the
  // stride skipped. Refusing on a negative span makes the two identical without
  // a signed-division idiom, and it is the shape `Conv1dOutLength` below already
  // uses for the same reason.
  //
  // UNREACHABLE FROM LTX — `CausalConv3d` materialises a pad of at least the
  // kernel on every axis (video_vae/convolution.py:305-311), so the padded
  // extent never falls below the kernel. It is gated because this op is offered
  // as a SHARED SEAM and the contract at vt::Conv3d claims to mirror nn.Conv3d;
  // tests/vt/test_ops_conv3d.cpp holds it.
  const int64_t span_t = tin + 2 * args.pad_t - args.dilation_t * (kt - 1) - 1;
  const int64_t span_h = hin + 2 * args.pad_h - args.dilation_h * (kh - 1) - 1;
  const int64_t span_w = win + 2 * args.pad_w - args.dilation_w * (kw - 1) - 1;
  VT_CHECK(span_t >= 0 && span_h >= 0 && span_w >= 0,
           "conv3d: kernel/dilation larger than the padded input");
  const int64_t tout = span_t / args.stride_t + 1;
  const int64_t hout = span_h / args.stride_h + 1;
  const int64_t wout = span_w / args.stride_w + 1;
  VT_CHECK(tout > 0 && hout > 0 && wout > 0, "conv3d: kernel/dilation larger than the padded input");
  VT_CHECK(out.shape[1] == tout && out.shape[2] == hout && out.shape[3] == wout,
           "conv3d: out must be [Cout,Tout,Hout,Wout] for the given stride/padding/dilation");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(weight.dtype) && IsFloat(out.dtype),
           "conv3d: x/weight/out must be f32, f16 or bf16");
  VT_CHECK(x.IsContiguous() && weight.IsContiguous() && out.IsContiguous(),
           "conv3d: contiguous tensors required");
  VT_CHECK(x.device == q.device && weight.device == q.device && out.device == q.device,
           "conv3d: device mismatch (x/weight/out/queue)");
  if (bias != nullptr) {
    VT_CHECK(bias->rank == 1 && bias->shape[0] == cout, "conv3d: bias must be rank-1 [Cout]");
    VT_CHECK(IsFloat(bias->dtype) && bias->IsContiguous() && bias->device == q.device,
             "conv3d: bias must be a contiguous float tensor on the queue device");
  }
  // #1007 fresh review, non-blocking suggestion. The FIRST non-CPU kConv3d
  // dispatch in a process announces itself, once, on stderr.
  //
  // The CUDA arm of this op has never been compiled and has never been run —
  // there is no `nvcc` on the authoring host and no GPU runner in CI — and no
  // gate anywhere in this tree can catch a kernel that compiles and computes the
  // wrong pixels. This line does not remove that risk; it converts a SILENT
  // first execution of never-run code into an announced one, so whoever gets a
  // GPU first sees the moment it happened beside whatever the pixels look like.
  //
  // Deliberately on the DEVICE-TYPE rather than on CUDA: the same argument holds
  // for every accelerator arm this seam gains, and putting it here keeps it in
  // code that this box compiles and `test_diffusion_device_seam` executes,
  // rather than in a `.cu` file nothing here can build.
  if (q.device.type != DeviceType::kCPU) {
    static std::atomic<bool> announced{false};
    if (!announced.exchange(true)) {
      // The CUDA arm HAS now been run, so it must not claim otherwise. It was
      // compiled for sm_121a and executed on a GB10 under #1452, and
      // tests/vt/test_ops_conv3d.cpp gates it `memcmp`-identical to the CPU
      // provider over the whole shape table and under catastrophic
      // cancellation. Every OTHER accelerator type reaching this seam is still
      // unrun, and the announcement stays for them: that is why this is
      // narrowed rather than deleted.
      std::fprintf(stderr, "[vt] first non-CPU vt::Conv3d dispatch (device type %d). %s\n",
                   static_cast<int>(q.device.type),
                   q.device.type == DeviceType::kCUDA
                       ? "The CUDA arm is byte-gated against the CPU provider and was executed "
                         "on a GB10 (#1452). No SPEED claim attaches to it."
                       : "This arm has never been run on real hardware; see issue #1452.");
    }
  }
  reinterpret_cast<Conv3dFn>(GetOp(OpId::kConv3d, q.device.type))(q, out, x, weight, bias, args);
}

void DepthwiseConv1d(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                     const Tensor* bias, const DepthwiseConv1dArgs& args) {
  VT_CHECK(x.rank == 3 && out.rank == 3, "depthwise_conv1d: x/out must be rank-3 [N,C,L]");
  // torch's depthwise parameter is [C, 1, K]; the flat [C, K] form is accepted
  // because that is how vt::CausalConv1dFwd already carries a per-channel filter.
  VT_CHECK(weight.rank == 3 || weight.rank == 2,
           "depthwise_conv1d: weight must be [C,1,K] (torch) or [C,K]");
  const int64_t n = x.shape[0], c = x.shape[1], lin = x.shape[2];
  const int64_t k = weight.shape[weight.rank - 1];
  VT_CHECK(c > 0 && lin > 0 && k > 0, "depthwise_conv1d: C/L/K must be positive");
  VT_CHECK(weight.shape[0] == c, "depthwise_conv1d: weight dim 0 must be C (groups == C)");
  if (weight.rank == 3) {
    VT_CHECK(weight.shape[1] == 1, "depthwise_conv1d: weight dim 1 must be 1 (in_channels/groups)");
  }
  VT_CHECK(args.stride >= 1, "depthwise_conv1d: stride must be >= 1");
  VT_CHECK(args.dilation >= 1, "depthwise_conv1d: dilation must be >= 1");
  VT_CHECK(args.padding >= 0, "depthwise_conv1d: padding must be >= 0");
  const int64_t lout = (lin + 2 * args.padding - args.dilation * (k - 1) - 1) / args.stride + 1;
  VT_CHECK(lout > 0, "depthwise_conv1d: kernel/dilation larger than the padded input");
  VT_CHECK(out.shape[0] == n && out.shape[1] == c && out.shape[2] == lout,
           "depthwise_conv1d: out must be [N,C,Lout] for the given stride/padding/dilation");
  VT_CHECK(IsFloat(x.dtype) && IsFloat(weight.dtype) && IsFloat(out.dtype),
           "depthwise_conv1d: x/weight/out must be f32, f16 or bf16");
  VT_CHECK(x.IsContiguous() && weight.IsContiguous() && out.IsContiguous(),
           "depthwise_conv1d: contiguous tensors required");
  VT_CHECK(x.device == q.device && weight.device == q.device && out.device == q.device,
           "depthwise_conv1d: device mismatch (x/weight/out/queue)");
  if (bias != nullptr) {
    VT_CHECK(bias->rank == 1 && bias->shape[0] == c, "depthwise_conv1d: bias must be rank-1 [C]");
    VT_CHECK(IsFloat(bias->dtype) && bias->IsContiguous() && bias->device == q.device,
             "depthwise_conv1d: bias must be a contiguous float tensor on the queue device");
  }
  reinterpret_cast<DepthwiseConv1dFn>(GetOp(OpId::kDepthwiseConv1d, q.device.type))(
      q, out, x, weight, bias, args);
}

// --- BigVGAN / DAC vocoder 1-D convolutions (#672) --------------------------
// Upstream mirror: torch `nn.Conv1d` / `nn.ConvTranspose1d` as instantiated by
// minimax_music3_vocoder.py:42,44,55,89,98 and LTX-2.5 audio_vae/vocoder.py.
// The validation mirrors torch's own shape contracts, so a caller that passes
// what the Python module passes is accepted verbatim. See vt::Conv1d in
// include/vt/ops.h for the f64-accumulator + pinned-visit-order contract.

int64_t Conv1dOutLength(int64_t in_len, int64_t kernel, const Conv1dArgs& args) {
  if (args.stride < 1 || args.dilation < 1 || args.padding < 0 || kernel < 1) return 0;
  const int64_t effective = args.dilation * (kernel - 1) + 1;
  const int64_t span = in_len + 2 * args.padding - effective;
  if (span < 0) return 0;
  return span / args.stride + 1;
}

int64_t ConvTranspose1dOutLength(int64_t in_len, int64_t kernel, const ConvTranspose1dArgs& args) {
  if (args.stride < 1 || args.dilation < 1 || args.padding < 0 || args.output_padding < 0 ||
      kernel < 1 || in_len < 1) {
    return 0;
  }
  return (in_len - 1) * args.stride - 2 * args.padding + args.dilation * (kernel - 1) + 1 +
         args.output_padding;
}

void Conv1d(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight, const Tensor* bias,
            const Conv1dArgs& args) {
  VT_CHECK(x.rank == 3 && out.rank == 3, "conv1d: x/out must be rank-3 [N,C,L]");
  VT_CHECK(weight.rank == 3, "conv1d: weight must be rank-3 [Cout,Cin/groups,K]");
  const int64_t g = args.groups;
  VT_CHECK(g >= 1, "conv1d: groups must be >= 1");
  const int64_t n = x.shape[0], cin = x.shape[1], lin = x.shape[2];
  const int64_t cout = weight.shape[0], cin_g = weight.shape[1], k = weight.shape[2];
  VT_CHECK(cin > 0 && lin > 0, "conv1d: x extents must be positive");
  VT_CHECK(cout > 0 && k > 0, "conv1d: weight extents must be positive");
  VT_CHECK(cin % g == 0 && cout % g == 0, "conv1d: groups must divide both Cin and Cout");
  VT_CHECK(cin_g == cin / g, "conv1d: weight dim 1 must be Cin/groups");
  VT_CHECK(args.stride >= 1, "conv1d: stride must be >= 1");
  VT_CHECK(args.dilation >= 1, "conv1d: dilation must be >= 1");
  VT_CHECK(args.padding >= 0, "conv1d: padding must be >= 0");
  const int64_t lout = Conv1dOutLength(lin, k, args);
  VT_CHECK(lout > 0, "conv1d: kernel/dilation larger than the padded input");
  VT_CHECK(out.shape[0] == n && out.shape[1] == cout && out.shape[2] == lout,
           "conv1d: out must be [N,Cout,Lout] for the given stride/padding/dilation");
  // f32 ONLY, and refused by name rather than widened — see the header.
  VT_CHECK(x.dtype == DType::kF32 && weight.dtype == DType::kF32 && out.dtype == DType::kF32,
           "conv1d: x/weight/out must be f32 (f16/bf16 arms are not implemented; the four "
           "vocoder1d consumers are f32 host-reference paths and no golden covers a narrow one)");
  VT_CHECK(x.IsContiguous() && weight.IsContiguous() && out.IsContiguous(),
           "conv1d: contiguous tensors required");
  VT_CHECK(x.device == q.device && weight.device == q.device && out.device == q.device,
           "conv1d: device mismatch (x/weight/out/queue)");
  if (bias != nullptr) {
    VT_CHECK(bias->rank == 1 && bias->shape[0] == cout, "conv1d: bias must be rank-1 [Cout]");
    VT_CHECK(bias->dtype == DType::kF32 && bias->IsContiguous() && bias->device == q.device,
             "conv1d: bias must be a contiguous f32 tensor on the queue device");
  }
  reinterpret_cast<Conv1dFn>(GetOp(OpId::kConv1d, q.device.type))(q, out, x, weight, bias, args);
}

void ConvTranspose1d(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                     const Tensor* bias, const ConvTranspose1dArgs& args) {
  VT_CHECK(x.rank == 3 && out.rank == 3, "conv_transpose1d: x/out must be rank-3 [N,C,L]");
  // torch's ConvTranspose1d parameter is [Cin, Cout/groups, K] — dim 0 is the
  // INPUT channel, the opposite of nn.Conv1d. Getting this backwards still
  // produces finite, correctly shaped output, so it is checked here.
  VT_CHECK(weight.rank == 3, "conv_transpose1d: weight must be rank-3 [Cin,Cout/groups,K]");
  const int64_t g = args.groups;
  VT_CHECK(g >= 1, "conv_transpose1d: groups must be >= 1");
  const int64_t n = x.shape[0], cin = x.shape[1], lin = x.shape[2];
  const int64_t cout_g = weight.shape[1], k = weight.shape[2];
  VT_CHECK(cin > 0 && lin > 0, "conv_transpose1d: x extents must be positive");
  VT_CHECK(cout_g > 0 && k > 0, "conv_transpose1d: weight extents must be positive");
  VT_CHECK(weight.shape[0] == cin, "conv_transpose1d: weight dim 0 must be Cin");
  VT_CHECK(cin % g == 0, "conv_transpose1d: groups must divide Cin");
  const int64_t cout = cout_g * g;
  VT_CHECK(args.stride >= 1, "conv_transpose1d: stride must be >= 1");
  VT_CHECK(args.dilation >= 1, "conv_transpose1d: dilation must be >= 1");
  VT_CHECK(args.padding >= 0, "conv_transpose1d: padding must be >= 0");
  VT_CHECK(args.output_padding >= 0, "conv_transpose1d: output_padding must be >= 0");
  VT_CHECK(args.output_padding < args.stride || args.output_padding < args.dilation,
           "conv_transpose1d: output_padding must be smaller than stride or dilation (torch)");
  const int64_t lout = ConvTranspose1dOutLength(lin, k, args);
  VT_CHECK(lout > 0, "conv_transpose1d: padding crops the whole output away");
  VT_CHECK(out.shape[0] == n && out.shape[1] == cout && out.shape[2] == lout,
           "conv_transpose1d: out must be [N,Cout,Lout] for the given stride/padding/dilation");
  VT_CHECK(x.dtype == DType::kF32 && weight.dtype == DType::kF32 && out.dtype == DType::kF32,
           "conv_transpose1d: x/weight/out must be f32 (f16/bf16 arms are not implemented; the "
           "four vocoder1d consumers are f32 host-reference paths and no golden covers a narrow "
           "one)");
  VT_CHECK(x.IsContiguous() && weight.IsContiguous() && out.IsContiguous(),
           "conv_transpose1d: contiguous tensors required");
  VT_CHECK(x.device == q.device && weight.device == q.device && out.device == q.device,
           "conv_transpose1d: device mismatch (x/weight/out/queue)");
  if (bias != nullptr) {
    VT_CHECK(bias->rank == 1 && bias->shape[0] == cout,
             "conv_transpose1d: bias must be rank-1 [Cout]");
    VT_CHECK(bias->dtype == DType::kF32 && bias->IsContiguous() && bias->device == q.device,
             "conv_transpose1d: bias must be a contiguous f32 tensor on the queue device");
  }
  reinterpret_cast<ConvTranspose1dFn>(GetOp(OpId::kConvTranspose1d, q.device.type))(
      q, out, x, weight, bias, args);
}

void AttentionRelPos(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                     const Tensor& value, const Tensor& rel_key, const Tensor* bias_u,
                     const Tensor* bias_v, const Tensor* key_mask,
                     const AttentionRelPosArgs& args) {
  VT_CHECK(query.rank == 3 && key.rank == 3 && value.rank == 3 && out.rank == 3 &&
               rel_key.rank == 3,
           "attention_relpos: query/key/value/rel_key/out rank-3 [T,H,D]");
  const int64_t t = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t hk = key.shape[1];
  VT_CHECK(t > 0 && hq > 0 && d > 0, "attention_relpos: T/Hq/D must be positive");
  VT_CHECK(key.shape[0] == t && value.shape[0] == t,
           "attention_relpos: query/key/value token count must match");
  VT_CHECK(key.shape[2] == d && value.shape[2] == d,
           "attention_relpos: key/value head_dim must match query");
  VT_CHECK(value.shape[1] == hk, "attention_relpos: key/value must share the kv-head count");
  VT_CHECK(hk >= 1 && hq % hk == 0,
           "attention_relpos: Hq must be a positive multiple of Hk (GQA broadcast)");
  VT_CHECK(out.shape[0] == t && out.shape[1] == hq && out.shape[2] == d,
           "attention_relpos: out must be [T,Hq,D] matching query");
  // P == 2T-1: the relative-position table spans offsets T-1 .. -(T-1), exactly
  // what ParakeetEncoderRelPositionalEncoding emits (modeling_parakeet.py:78
  // `arange(seq_length-1, -seq_length, -1)`), and what _rel_shift assumes.
  VT_CHECK(rel_key.shape[0] == 2 * t - 1 && rel_key.shape[1] == hq && rel_key.shape[2] == d,
           "attention_relpos: rel_key must be [2*T-1, Hq, D]");
  VT_CHECK(args.scale > 0.0f, "attention_relpos: scale must be set (> 0), e.g. head_dim^-0.5");
  VT_CHECK(IsFloat(query.dtype) && IsFloat(key.dtype) && IsFloat(value.dtype) &&
               IsFloat(rel_key.dtype),
           "attention_relpos: query/key/value/rel_key must be f32, f16 or bf16");
  VT_CHECK(IsFloat(out.dtype), "attention_relpos: out must be f32, f16 or bf16");
  VT_CHECK(query.IsContiguous() && key.IsContiguous() && value.IsContiguous() &&
               rel_key.IsContiguous() && out.IsContiguous(),
           "attention_relpos: contiguous tensors required");
  VT_CHECK(query.device == q.device && key.device == q.device && value.device == q.device &&
               rel_key.device == q.device && out.device == q.device,
           "attention_relpos: device mismatch (query/key/value/rel_key/out/queue)");
  for (const Tensor* b : {bias_u, bias_v}) {
    if (b == nullptr) continue;
    VT_CHECK(b->rank == 2 && b->shape[0] == hq && b->shape[1] == d,
             "attention_relpos: bias_u/bias_v must be rank-2 [Hq,D]");
    VT_CHECK(IsFloat(b->dtype) && b->IsContiguous() && b->device == q.device,
             "attention_relpos: bias_u/bias_v must be contiguous float on the queue device");
  }
  if (key_mask != nullptr) {
    CheckBoolMeta(q, *key_mask, t, "attention_relpos", "key_mask");
  }
  reinterpret_cast<AttentionRelPosFn>(GetOp(OpId::kAttentionRelPos, q.device.type))(
      q, out, query, key, value, rel_key, bias_u, bias_v, key_mask, args);
}

void AttentionDenseFast(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                        const Tensor& value, const AttentionArgs& args) {
  VT_CHECK(query.rank == 3 && key.rank == 3 && value.rank == 3 && out.rank == 3,
           "attention-dense-fast: query/key/value/out rank-3 [T,H,D]");
  const int64_t t = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t hk = key.shape[1];
  VT_CHECK(key.shape[0] == t && value.shape[0] == t,
           "attention-dense-fast: query/key/value token count must match");
  VT_CHECK(key.shape[2] == d && value.shape[2] == d,
           "attention-dense-fast: key/value head_dim must match query");
  VT_CHECK(value.shape[1] == hk, "attention-dense-fast: key/value must share the kv-head count");
  VT_CHECK(out.shape[0] == t && out.shape[1] == hq && out.shape[2] == d,
           "attention-dense-fast: out must be [T,Hq,D] matching query");
  VT_CHECK(hk >= 1 && hq >= 1 && hq % hk == 0,
           "attention-dense-fast: Hq must be a positive multiple of Hk (GQA broadcast)");
  VT_CHECK(args.scale > 0.0f, "attention-dense-fast: scale must be set (> 0)");
  VT_CHECK(IsFloat(query.dtype) && key.dtype == query.dtype && value.dtype == query.dtype,
           "attention-dense-fast: query/key/value must share one float dtype");
  VT_CHECK(IsOutFloat(out.dtype), "attention-dense-fast: out must be f32 or bf16");
  VT_CHECK(query.IsContiguous() && key.IsContiguous() && value.IsContiguous() &&
               out.IsContiguous(),
           "attention-dense-fast: contiguous tensors required");
  VT_CHECK(query.device == q.device && key.device == q.device && value.device == q.device &&
               out.device == q.device,
           "attention-dense-fast: device mismatch (query/key/value/out/queue)");
  reinterpret_cast<AttentionFn>(GetOp(OpId::kAttentionDenseFast, q.device.type))(
      q, out, query, key, value, args);
}

void AttentionDenseFlash(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                         const Tensor& value, const AttentionArgs& args) {
  VT_CHECK(query.rank == 3 && key.rank == 3 && value.rank == 3 && out.rank == 3,
           "attention-dense-flash: query/key/value/out rank-3 [T,H,D]");
  const int64_t t = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t hk = key.shape[1];
  VT_CHECK(key.shape[0] == t && value.shape[0] == t,
           "attention-dense-flash: query/key/value token count must match");
  VT_CHECK(key.shape[2] == d && value.shape[2] == d,
           "attention-dense-flash: key/value head_dim must match query");
  VT_CHECK(value.shape[1] == hk, "attention-dense-flash: key/value must share the kv-head count");
  VT_CHECK(out.shape[0] == t && out.shape[1] == hq && out.shape[2] == d,
           "attention-dense-flash: out must be [T,Hq,D] matching query");
  VT_CHECK(hk >= 1 && hq >= 1 && hq % hk == 0,
           "attention-dense-flash: Hq must be a positive multiple of Hk (GQA broadcast)");
  VT_CHECK(args.scale > 0.0f, "attention-dense-flash: scale must be set (> 0)");
  VT_CHECK(IsFloat(query.dtype) && key.dtype == query.dtype && value.dtype == query.dtype,
           "attention-dense-flash: query/key/value must share one float dtype");
  VT_CHECK(IsOutFloat(out.dtype), "attention-dense-flash: out must be f32 or bf16");
  VT_CHECK(query.IsContiguous() && key.IsContiguous() && value.IsContiguous() &&
               out.IsContiguous(),
           "attention-dense-flash: contiguous tensors required");
  VT_CHECK(query.device == q.device && key.device == q.device && value.device == q.device &&
               out.device == q.device,
           "attention-dense-flash: device mismatch (query/key/value/out/queue)");
  reinterpret_cast<AttentionFn>(GetOp(OpId::kAttentionDenseFlash, q.device.type))(
      q, out, query, key, value, args);
}

void AttentionDenseFa2(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                       const Tensor& value, const AttentionArgs& args) {
  VT_CHECK(query.rank == 3 && key.rank == 3 && value.rank == 3 && out.rank == 3,
           "attention-dense-fa2: query/key/value/out rank-3 [T,H,D]");
  const int64_t t = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t hk = key.shape[1];
  VT_CHECK(key.shape[0] == t && value.shape[0] == t,
           "attention-dense-fa2: query/key/value token count must match");
  VT_CHECK(key.shape[2] == d && value.shape[2] == d,
           "attention-dense-fa2: key/value head_dim must match query");
  VT_CHECK(value.shape[1] == hk, "attention-dense-fa2: key/value must share the kv-head count");
  VT_CHECK(out.shape[0] == t && out.shape[1] == hq && out.shape[2] == d,
           "attention-dense-fa2: out must be [T,Hq,D] matching query");
  VT_CHECK(hk >= 1 && hq >= 1 && hq % hk == 0,
           "attention-dense-fa2: Hq must be a positive multiple of Hk (GQA broadcast)");
  VT_CHECK(args.scale > 0.0f, "attention-dense-fa2: scale must be set (> 0)");
  VT_CHECK(IsFloat(query.dtype) && key.dtype == query.dtype && value.dtype == query.dtype,
           "attention-dense-fa2: query/key/value must share one float dtype");
  VT_CHECK(IsOutFloat(out.dtype), "attention-dense-fa2: out must be f32 or bf16");
  VT_CHECK(query.IsContiguous() && key.IsContiguous() && value.IsContiguous() &&
               out.IsContiguous(),
           "attention-dense-fa2: contiguous tensors required");
  VT_CHECK(query.device == q.device && key.device == q.device && value.device == q.device &&
               out.device == q.device,
           "attention-dense-fa2: device mismatch (query/key/value/out/queue)");
  reinterpret_cast<AttentionFn>(GetOp(OpId::kAttentionDenseFa2, q.device.type))(
      q, out, query, key, value, args);
}

void DFlashBlockAttention(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                          const Tensor& value, const DFlashBlockAttentionArgs& args) {
  VT_CHECK(query.rank == 3 && key.rank == 3 && value.rank == 3 && out.rank == 3,
           "dflash-block-attn: query/key/value/out rank-3 [T,Hq/Hkv,D]");
  const int64_t tq = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t t = key.shape[0];
  const int64_t hk = key.shape[1];
  VT_CHECK(value.shape[0] == t, "dflash-block-attn: key/value token count must match");
  VT_CHECK(args.cu_seqlens_q != nullptr || tq == t,
           "dflash-block-attn: query/key token count must match unless cu_seqlens_q is set");
  VT_CHECK(key.shape[2] == d && value.shape[2] == d,
           "dflash-block-attn: key/value head_dim must match query");
  VT_CHECK(value.shape[1] == hk, "dflash-block-attn: key/value must share the kv-head count");
  VT_CHECK(out.shape[0] == tq && out.shape[1] == hq && out.shape[2] == d,
           "dflash-block-attn: out must be [Tq,Hq,D] matching query");
  VT_CHECK(hk >= 1 && hq >= 1 && hq % hk == 0,
           "dflash-block-attn: Hq must be a positive multiple of Hk (GQA broadcast)");
  VT_CHECK(args.scale > 0.0f, "dflash-block-attn: scale must be set (> 0), e.g. head_dim^-0.5");
  VT_CHECK(args.num_reqs >= 1 && args.cu_seqlens != nullptr,
           "dflash-block-attn: cu_seqlens (host, num_reqs+1) required");
  VT_CHECK(args.cu_seqlens[0] == 0 && args.cu_seqlens[args.num_reqs] == static_cast<int32_t>(t),
           "dflash-block-attn: cu_seqlens must span [0,T]");
  if (args.cu_seqlens_q != nullptr) {
    // D1 (#2087): the query block is the per-request SUFFIX of the key block, so
    // every request's query run must FIT its key run. A qlen > klen would make the
    // combined offset negative and read the previous request's keys.
    VT_CHECK(args.cu_seqlens_q[0] == 0 &&
                 args.cu_seqlens_q[args.num_reqs] == static_cast<int32_t>(tq),
             "dflash-block-attn: cu_seqlens_q must span [0,Tq]");
    for (int r = 0; r < args.num_reqs; ++r) {
      VT_CHECK(args.cu_seqlens_q[r + 1] >= args.cu_seqlens_q[r] &&
                   args.cu_seqlens[r + 1] >= args.cu_seqlens[r],
               "dflash-block-attn: cu_seqlens/cu_seqlens_q must be non-decreasing");
      VT_CHECK(args.cu_seqlens_q[r + 1] - args.cu_seqlens_q[r] <=
                   args.cu_seqlens[r + 1] - args.cu_seqlens[r],
               "dflash-block-attn: per-request query rows must not exceed key rows");
    }
  }
  VT_CHECK(IsFloat(query.dtype) && key.dtype == query.dtype && value.dtype == query.dtype,
           "dflash-block-attn: query/key/value must share one float dtype");
  VT_CHECK(IsOutFloat(out.dtype), "dflash-block-attn: out must be f32 or bf16");
  VT_CHECK(query.IsContiguous() && key.IsContiguous() && value.IsContiguous() &&
               out.IsContiguous(),
           "dflash-block-attn: contiguous tensors required");
  VT_CHECK(query.device == q.device && key.device == q.device && value.device == q.device &&
               out.device == q.device,
           "dflash-block-attn: device mismatch (query/key/value/out/queue)");
  reinterpret_cast<DFlashBlockAttentionFn>(GetOp(OpId::kDFlashBlockAttention, q.device.type))(
      q, out, query, key, value, args);
}

void DFlashPagedBlockAttention(Queue& q, Tensor& out, const Tensor& query,
                               const Tensor& block_key, const Tensor& block_value,
                               const Tensor& ctx_key, const Tensor& ctx_value,
                               const Tensor& cu_seqlens, const Tensor& seq_lens,
                               const Tensor& block_table,
                               const DFlashPagedBlockAttentionArgs& args) {
  VT_CHECK(query.rank == 3 && block_key.rank == 3 && block_value.rank == 3 && out.rank == 3,
           "dflash-paged-block-attn: query/block_key/block_value/out rank-3 [Nq,Hq/Hkv,D]");
  const int64_t nq = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t hk = block_key.shape[1];
  VT_CHECK(block_key.shape[0] == nq && block_value.shape[0] == nq,
           "dflash-paged-block-attn: block key/value row count must match query");
  VT_CHECK(block_key.shape[2] == d && block_value.shape[2] == d && block_value.shape[1] == hk,
           "dflash-paged-block-attn: block key/value head geometry mismatch");
  VT_CHECK(out.shape[0] == nq && out.shape[1] == hq && out.shape[2] == d,
           "dflash-paged-block-attn: out must be [Nq,Hq,D] matching query");
  VT_CHECK(hk >= 1 && hq >= 1 && hq % hk == 0,
           "dflash-paged-block-attn: Hq must be a positive multiple of Hk (GQA broadcast)");
  VT_CHECK(ctx_key.rank == 4 && ctx_value.rank == 4,
           "dflash-paged-block-attn: ctx key/value rank-4 [pages,block_size,Hkv,D]");
  VT_CHECK(ctx_key.shape[2] == hk && ctx_key.shape[3] == d && ctx_value.shape[2] == hk &&
               ctx_value.shape[3] == d,
           "dflash-paged-block-attn: ctx cache head geometry must match block");
  VT_CHECK(args.scale > 0.0f, "dflash-paged-block-attn: scale must be set (> 0)");
  VT_CHECK(args.num_reqs >= 1 && args.block_size > 0,
           "dflash-paged-block-attn: num_reqs>=1 and block_size>0 required");
  VT_CHECK(cu_seqlens.dtype == DType::kI32 && seq_lens.dtype == DType::kI32 &&
               block_table.dtype == DType::kI32,
           "dflash-paged-block-attn: cu_seqlens/seq_lens/block_table must be i32");
  VT_CHECK(cu_seqlens.shape[0] == args.num_reqs + 1 && seq_lens.shape[0] == args.num_reqs,
           "dflash-paged-block-attn: cu_seqlens[num_reqs+1] / seq_lens[num_reqs] required");
  VT_CHECK(block_table.rank == 2 && block_table.shape[0] == args.num_reqs,
           "dflash-paged-block-attn: block_table must be [num_reqs, max_pages]");
  VT_CHECK(IsFloat(query.dtype) && block_key.dtype == query.dtype &&
               block_value.dtype == query.dtype && ctx_key.dtype == query.dtype &&
               ctx_value.dtype == query.dtype,
           "dflash-paged-block-attn: query/block/ctx must share one float dtype");
  VT_CHECK(IsOutFloat(out.dtype), "dflash-paged-block-attn: out must be f32 or bf16");
  VT_CHECK(query.IsContiguous() && block_key.IsContiguous() && block_value.IsContiguous() &&
               ctx_key.IsContiguous() && ctx_value.IsContiguous() && out.IsContiguous(),
           "dflash-paged-block-attn: contiguous float tensors required");
  VT_CHECK(query.device == q.device && ctx_key.device == q.device &&
               cu_seqlens.device == q.device && seq_lens.device == q.device &&
               block_table.device == q.device,
           "dflash-paged-block-attn: device mismatch (tensors/queue)");
  reinterpret_cast<DFlashPagedBlockAttentionFn>(
      GetOp(OpId::kDFlashPagedBlockAttention, q.device.type))(
      q, out, query, block_key, block_value, ctx_key, ctx_value, cu_seqlens, seq_lens, block_table,
      args);
}

void DFlashGroupedConv(Queue& q, Tensor& out, const Tensor& x, const Tensor& coefficients,
                       const Tensor& base, const DFlashGroupedConvArgs& args) {
  VT_CHECK(args.taps >= 1 && args.num_groups >= 1 && args.group_size >= 1,
           "dflash2-grouped-conv: taps/num_groups/group_size must be >= 1");
  VT_CHECK(args.block_size >= 1, "dflash2-grouped-conv: block_size must be >= 1 (1 + k)");
  VT_CHECK(x.rank == 2 && out.rank == 2, "dflash2-grouped-conv: x/out must be rank-2 [T,H]");
  VT_CHECK(coefficients.rank == 4,
           "dflash2-grouped-conv: coefficients must be rank-4 [T,sides,taps,num_groups]");
  VT_CHECK(base.rank == 3, "dflash2-grouped-conv: base must be rank-3 [sides,taps,H]");
  const int64_t t = x.shape[0];
  const int64_t h = args.num_groups * args.group_size;
  const int64_t sides = coefficients.shape[1];
  VT_CHECK(x.shape[1] == h,
           "dflash2-grouped-conv: x hidden must be num_groups*group_size");
  VT_CHECK(out.shape[0] == t && out.shape[1] == h,
           "dflash2-grouped-conv: out must be [T,H] matching x");
  VT_CHECK(coefficients.shape[0] == t && coefficients.shape[2] == args.taps &&
               coefficients.shape[3] == args.num_groups,
           "dflash2-grouped-conv: coefficients must be [T,sides,taps,num_groups]");
  VT_CHECK(base.shape[0] == sides && base.shape[1] == args.taps && base.shape[2] == h,
           "dflash2-grouped-conv: base must be [sides,taps,H] with the coefficients' sides");
  VT_CHECK(args.side >= 0 && args.side < sides,
           "dflash2-grouped-conv: side must index the sides dimension "
           "(0 = prepare, 1 = finish)");
  // ONE dtype across all four. The rounding after each step is what makes the
  // CPU reference and the CUDA kernel bit-identical, and a mixed set would make
  // "the dtype" ambiguous rather than merely inconvenient.
  VT_CHECK(IsFloat(x.dtype) && coefficients.dtype == x.dtype && base.dtype == x.dtype &&
               out.dtype == x.dtype,
           "dflash2-grouped-conv: x/coefficients/base/out must share one float dtype");
  VT_CHECK(x.IsContiguous() && coefficients.IsContiguous() && base.IsContiguous() &&
               out.IsContiguous(),
           "dflash2-grouped-conv: contiguous tensors required");
  VT_CHECK(x.device == q.device && coefficients.device == q.device &&
               base.device == q.device && out.device == q.device,
           "dflash2-grouped-conv: device mismatch (x/coefficients/base/out/queue)");
  reinterpret_cast<DFlashGroupedConvFn>(GetOp(OpId::kDFlashGroupedConv, q.device.type))(
      q, out, x, coefficients, base, args);
}

void Dflash2SelectorEdges(Queue& q, Tensor& scores, const Tensor& pred_codebook,
                          const Tensor& succ_codebook, const Tensor& candidate_ids,
                          const Tensor& unary, const Tensor& hidden, const Tensor& anchors,
                          const Dflash2SelectorEdgesArgs& args) {
  VT_CHECK(args.top_k >= 1, "dflash2-selector-edges: top_k must be >= 1");
  VT_CHECK(pred_codebook.rank == 2 && succ_codebook.rank == 2,
           "dflash2-selector-edges: codebooks must be rank-2 [vocab, rank]");
  VT_CHECK(candidate_ids.rank == 3 && unary.rank == 3 && hidden.rank == 3,
           "dflash2-selector-edges: candidate_ids/unary must be [B,L,K] and hidden [B,L,R]");
  VT_CHECK(scores.rank == 4, "dflash2-selector-edges: scores must be rank-4 [B,L,K,K]");
  VT_CHECK(anchors.rank == 1, "dflash2-selector-edges: anchors must be rank-1 [B]");
  const int64_t b = candidate_ids.shape[0], l = candidate_ids.shape[1];
  const int64_t k = args.top_k, r = pred_codebook.shape[1];
  VT_CHECK(candidate_ids.shape[2] == k,
           "dflash2-selector-edges: candidate_ids last dim must be top_k");
  VT_CHECK(unary.shape[0] == b && unary.shape[1] == l && unary.shape[2] == k,
           "dflash2-selector-edges: unary must be [B,L,K] matching candidate_ids");
  VT_CHECK(hidden.shape[0] == b && hidden.shape[1] == l && hidden.shape[2] == r,
           "dflash2-selector-edges: hidden must be [B,L,rank]");
  VT_CHECK(scores.shape[0] == b && scores.shape[1] == l && scores.shape[2] == k &&
               scores.shape[3] == k,
           "dflash2-selector-edges: scores must be [B,L,K,K]");
  VT_CHECK(anchors.shape[0] == b, "dflash2-selector-edges: anchors must be [B]");
  VT_CHECK(succ_codebook.shape[0] == pred_codebook.shape[0] &&
               succ_codebook.shape[1] == r,
           "dflash2-selector-edges: the two codebooks must share [vocab, rank]");
  // ONE float dtype across the codebooks and the projected hidden — upstream's
  // params_dtype, the model dtype. `unary` and `scores` are f32 because upstream
  // makes the candidate values f32 in compute_candidates and torch then promotes
  // the bf16 einsum output to f32 on the add.
  VT_CHECK(IsFloat(pred_codebook.dtype) && succ_codebook.dtype == pred_codebook.dtype &&
               hidden.dtype == pred_codebook.dtype,
           "dflash2-selector-edges: codebooks and hidden must share one float dtype");
  VT_CHECK(unary.dtype == DType::kF32 && scores.dtype == DType::kF32,
           "dflash2-selector-edges: unary and scores must be f32");
  VT_CHECK(candidate_ids.dtype == DType::kI64 && anchors.dtype == DType::kI64,
           "dflash2-selector-edges: candidate_ids and anchors must be i64");
  VT_CHECK(pred_codebook.IsContiguous() && succ_codebook.IsContiguous() &&
               candidate_ids.IsContiguous() && unary.IsContiguous() &&
               hidden.IsContiguous() && anchors.IsContiguous() && scores.IsContiguous(),
           "dflash2-selector-edges: contiguous tensors required");
  VT_CHECK(pred_codebook.device == q.device && succ_codebook.device == q.device &&
               candidate_ids.device == q.device && unary.device == q.device &&
               hidden.device == q.device && anchors.device == q.device &&
               scores.device == q.device,
           "dflash2-selector-edges: device mismatch");
  reinterpret_cast<Dflash2SelectorEdgesFn>(
      GetOp(OpId::kDflash2SelectorEdges, q.device.type))(
      q, scores, pred_codebook, succ_codebook, candidate_ids, unary, hidden, anchors, args);
}

void Dflash2PathWalk(Queue& q, Tensor& tokens, const Tensor& scores,
                     const Tensor& candidate_ids, const Dflash2PathWalkArgs& args) {
  VT_CHECK(args.top_k >= 1, "dflash2-path-walk: top_k must be >= 1");
  VT_CHECK(scores.rank == 4, "dflash2-path-walk: scores must be rank-4 [B,L,K,K]");
  VT_CHECK(candidate_ids.rank == 3,
           "dflash2-path-walk: candidate_ids must be rank-3 [B,L,K]");
  VT_CHECK(tokens.rank == 2, "dflash2-path-walk: tokens must be rank-2 [B,L]");
  const int64_t b = candidate_ids.shape[0], l = candidate_ids.shape[1];
  const int64_t k = args.top_k;
  VT_CHECK(candidate_ids.shape[2] == k,
           "dflash2-path-walk: candidate_ids last dim must be top_k");
  // BOTH trailing axes are checked. They are the PREDECESSOR axis and the CHILD
  // axis and they have the same extent, so checking one and inferring the other
  // would admit a lattice indexed the wrong way round -- which reads plausible
  // scores from the wrong rows and moves acceptance without raising.
  VT_CHECK(scores.shape[0] == b && scores.shape[1] == l && scores.shape[2] == k &&
               scores.shape[3] == k,
           "dflash2-path-walk: scores must be [B,L,K,K] matching candidate_ids");
  VT_CHECK(tokens.shape[0] == b && tokens.shape[1] == l,
           "dflash2-path-walk: tokens must be [B,L] matching candidate_ids");
  VT_CHECK(scores.dtype == DType::kF32, "dflash2-path-walk: scores must be f32");
  VT_CHECK(candidate_ids.dtype == DType::kI64 && tokens.dtype == DType::kI64,
           "dflash2-path-walk: candidate_ids and tokens must be i64");
  VT_CHECK(scores.IsContiguous() && candidate_ids.IsContiguous() && tokens.IsContiguous(),
           "dflash2-path-walk: contiguous tensors required");
  VT_CHECK(scores.device == q.device && candidate_ids.device == q.device &&
               tokens.device == q.device,
           "dflash2-path-walk: device mismatch");
  reinterpret_cast<Dflash2PathWalkFn>(GetOp(OpId::kDflash2PathWalk, q.device.type))(
      q, tokens, scores, candidate_ids, args);
}

void TopKValuesIndices(Queue& q, Tensor& values, Tensor& indices, const Tensor& logits,
                       const TopKValuesIndicesArgs& args) {
  VT_CHECK(logits.rank == 2 && values.rank == 2 && indices.rank == 2,
           "topk-values-indices: logits/values/indices must be rank-2");
  const int64_t rows = logits.shape[0], v = logits.shape[1];
  const int64_t pad = args.num_org_vocab_padding;
  VT_CHECK(pad >= 0 && pad < v, "topk-values-indices: num_org_vocab_padding must be in [0, V)");
  VT_CHECK(args.k >= 1 && args.k <= v - pad,
           "topk-values-indices: k must be in [1, V - num_org_vocab_padding]");
  VT_CHECK(values.shape[0] == rows && indices.shape[0] == rows &&
               values.shape[1] == args.k && indices.shape[1] == args.k,
           "topk-values-indices: values/indices must be [rows, k]");
  VT_CHECK(logits.dtype == DType::kF32 && values.dtype == DType::kF32,
           "topk-values-indices: logits and values must be f32");
  VT_CHECK(indices.dtype == DType::kI64, "topk-values-indices: indices must be i64");
  VT_CHECK(logits.IsContiguous() && values.IsContiguous() && indices.IsContiguous(),
           "topk-values-indices: contiguous tensors required");
  VT_CHECK(logits.device == q.device && values.device == q.device &&
               indices.device == q.device,
           "topk-values-indices: device mismatch");
  reinterpret_cast<TopKValuesIndicesFn>(GetOp(OpId::kTopKValuesIndices, q.device.type))(
      q, values, indices, logits, args);
}

void ReshapeAndCache(Queue& q, const Tensor& k, const Tensor& v, Tensor& k_cache,
                     Tensor& v_cache, const Tensor& slot_mapping) {
  VT_CHECK(k.rank == 3 && v.rank == 3,
           "reshape_and_cache: k/v must be rank-3 [num_tokens,num_kv_heads,head_size]");
  VT_CHECK(k_cache.rank == 4 && v_cache.rank == 4,
           "reshape_and_cache: k_cache/v_cache must be rank-4 "
           "[num_blocks,block_size,num_kv_heads,head_size]");
  VT_CHECK(slot_mapping.rank == 1, "reshape_and_cache: slot_mapping must be rank-1 [num_slots]");
  const int64_t num_kv_heads = k.shape[1];
  const int64_t head_size_k = k.shape[2];
  const int64_t head_size_v = v.shape[2];
  VT_CHECK(v.shape[0] == k.shape[0] && v.shape[1] == num_kv_heads,
           "reshape_and_cache: k and v must share [num_tokens,num_kv_heads]; "
           "head_size may differ (MiMoV2 asymmetric K/V: Dh_k != Dh_v)");
  VT_CHECK(k_cache.shape[2] == num_kv_heads && k_cache.shape[3] == head_size_k,
           "reshape_and_cache: k_cache num_kv_heads/head_size must match k");
  VT_CHECK(v_cache.shape[0] == k_cache.shape[0] && v_cache.shape[1] == k_cache.shape[1] &&
               v_cache.shape[2] == num_kv_heads && v_cache.shape[3] == head_size_v,
           "reshape_and_cache: v_cache must match k_cache in num_blocks/block_size/"
           "num_kv_heads, and v_cache head_size must match v");
  // Upstream uses slot_mapping.size(0) as the token count: k/v may carry extra
  // trailing rows (CUDA-graph padding) that are ignored.
  VT_CHECK(k.shape[0] >= slot_mapping.shape[0],
           "reshape_and_cache: num_tokens (k.shape[0]) must be >= slot_mapping length");
  // KV-FP8 W3 — the loud end of the half-sized-block hazard. A `kI8` cache page
  // is one byte per element because the KV-cache spec was sized that way; a
  // float store into it would write two bytes per element at offsets computed
  // for one, which is wrong tokens rather than an out-of-bounds. Every
  // attention block that has been routed calls `vt::ReshapeAndCacheFp8` here
  // instead (`include/vllm/model_executor/models/kv_cache_route.h`), so reaching
  // this line means THIS architecture's attention block has not been routed —
  // and the message says so rather than reporting a dtype mismatch.
  VT_CHECK(k_cache.dtype != DType::kI8 && v_cache.dtype != DType::kI8,
           "reshape_and_cache: this cache is 1-byte fp8 storage (DType::kI8) but "
           "this is the float (auto) store; an fp8 KV cache must be written "
           "through vt::ReshapeAndCacheFp8. This model's attention block is not "
           "routed for fp8 KV (KV-FP8 W3) — run it on --kv-cache-dtype auto");
  VT_CHECK(IsFloat(k.dtype) && k.dtype == v.dtype && k_cache.dtype == k.dtype &&
               v_cache.dtype == k.dtype,
           "reshape_and_cache: k/v/k_cache/v_cache must share one float dtype (auto cache path)");
  VT_CHECK(slot_mapping.dtype == DType::kI64, "reshape_and_cache: slot_mapping must be i64");
  // The paged KV cache is ONE (num_blocks, 2, block_size, H, D) allocation;
  // k_cache/v_cache are the two dim-1 unbind slices, i.e. rank-4 STRIDED views
  // (block stride 2*bs*H*D, not bs*H*D). We therefore must NOT require the cache
  // to be contiguous — indexing is driven by k_cache/v_cache strides (mirroring
  // pinned cache_kernels.cu::reshape_and_cache_flash, which reads block/page/
  // head strides from key_cache.stride(0/1/2)). We only require what the copy
  // actually needs: the innermost element access is well-defined (elem stride 1)
  // and the per-token page is dense (head stride == head_size, i.e. dim-2/3
  // packed), which holds for the NHD unbind slice. Input K/V may likewise be
  // torch.split-style QKVParallelLinear views: each [H,D] token page is packed,
  // while stride(0) still spans Q+K+V. The kernels already consume the explicit
  // token strides, matching upstream reshape_and_cache_flash.
  VT_CHECK(k.stride[2] == 1 && v.stride[2] == 1 &&
               k.stride[1] == head_size_k && v.stride[1] == head_size_v &&
               k.stride[0] >= num_kv_heads * head_size_k &&
               v.stride[0] >= num_kv_heads * head_size_v &&
               slot_mapping.IsContiguous(),
           "reshape_and_cache: k/v token pages must be inner-contiguous and "
           "slot_mapping contiguous");
  VT_CHECK(k_cache.stride[3] == 1 && v_cache.stride[3] == 1,
           "reshape_and_cache: k_cache/v_cache innermost (head_size) stride must be 1");
  VT_CHECK(k_cache.stride[2] == head_size_k && v_cache.stride[2] == head_size_v,
           "reshape_and_cache: k_cache/v_cache page must be head-contiguous "
           "(stride[2] == head_size) — the NHD unbind-slice layout");
  VT_CHECK(k.device == q.device && v.device == q.device && k_cache.device == q.device &&
               v_cache.device == q.device && slot_mapping.device == q.device,
           "reshape_and_cache: device mismatch (k/v/k_cache/v_cache/slot_mapping/queue)");
  reinterpret_cast<ReshapeAndCacheFn>(GetOp(OpId::kReshapeAndCache, q.device.type))(
      q, k, v, k_cache, v_cache, slot_mapping);
}

void ReshapeAndCacheFp8(Queue& q, const Tensor& k, const Tensor& v, Tensor& k_cache,
                        Tensor& v_cache, const Tensor& slot_mapping, Fp8KVCacheDataType kind,
                        float k_scale, float v_scale) {
  // fp8 KV store. Same geometry contract as ReshapeAndCache; the K/V cache is
  // 1-byte fp8 (DType::kI8 — the dtype.h "the byte never guesses its semantic
  // type" rule) and the source K/V are the model float dtype.
  VT_CHECK(kind != Fp8KVCacheDataType::kAuto,
           "reshape_and_cache_fp8: kind must be an fp8 dtype (use ReshapeAndCache for auto)");
  VT_CHECK(kind == Fp8KVCacheDataType::kFp8E4M3,
           "reshape_and_cache_fp8: only fp8_e4m3 is implemented "
           "(fp8_e5m2 compute is a named later brick, spec W5)");
  VT_CHECK(k.rank == 3 && v.rank == 3,
           "reshape_and_cache_fp8: k/v must be rank-3 [num_tokens,num_kv_heads,head_size]");
  VT_CHECK(k_cache.rank == 4 && v_cache.rank == 4,
           "reshape_and_cache_fp8: k_cache/v_cache must be rank-4 "
           "[num_blocks,block_size,num_kv_heads,head_size]");
  VT_CHECK(slot_mapping.rank == 1,
           "reshape_and_cache_fp8: slot_mapping must be rank-1 [num_slots]");
  const int64_t num_kv_heads = k.shape[1];
  const int64_t head_size_k = k.shape[2];
  const int64_t head_size_v = v.shape[2];
  VT_CHECK(v.shape[0] == k.shape[0] && v.shape[1] == num_kv_heads,
           "reshape_and_cache_fp8: k and v must share [num_tokens,num_kv_heads]; "
           "head_size may differ (MiMoV2 asymmetric K/V: Dh_k != Dh_v)");
  VT_CHECK(k_cache.shape[2] == num_kv_heads && k_cache.shape[3] == head_size_k,
           "reshape_and_cache_fp8: k_cache num_kv_heads/head_size must match k");
  VT_CHECK(v_cache.shape[0] == k_cache.shape[0] && v_cache.shape[1] == k_cache.shape[1] &&
               v_cache.shape[2] == num_kv_heads && v_cache.shape[3] == head_size_v,
           "reshape_and_cache_fp8: v_cache must match k_cache in num_blocks/block_size/"
           "num_kv_heads, and v_cache head_size must match v");
  VT_CHECK(k.shape[0] >= slot_mapping.shape[0],
           "reshape_and_cache_fp8: num_tokens (k.shape[0]) must be >= slot_mapping length");
  // Source K/V are model floats; cache pages are fp8 bytes (kI8).
  VT_CHECK(IsFloat(k.dtype) && k.dtype == v.dtype,
           "reshape_and_cache_fp8: k/v must share one float dtype");
  VT_CHECK(k_cache.dtype == DType::kI8 && v_cache.dtype == DType::kI8,
           "reshape_and_cache_fp8: k_cache/v_cache must be 1-byte fp8 storage (DType::kI8)");
  VT_CHECK(slot_mapping.dtype == DType::kI64, "reshape_and_cache_fp8: slot_mapping must be i64");
  VT_CHECK(k_scale > 0.0f && v_scale > 0.0f,
           "reshape_and_cache_fp8: k_scale/v_scale must be > 0 (per-tensor fp8 KV scales)");
  // Same stride contract as ReshapeAndCache (inner-contiguous token pages, NHD
  // unbind-slice cache). Element size 1 for the cache; the source uses its float
  // element size.
  VT_CHECK(k.stride[2] == 1 && v.stride[2] == 1 &&
               k.stride[1] == head_size_k && v.stride[1] == head_size_v &&
               k.stride[0] >= num_kv_heads * head_size_k &&
               v.stride[0] >= num_kv_heads * head_size_v &&
               slot_mapping.IsContiguous(),
           "reshape_and_cache_fp8: k/v token pages must be inner-contiguous and "
           "slot_mapping contiguous");
  VT_CHECK(k_cache.stride[3] == 1 && v_cache.stride[3] == 1,
           "reshape_and_cache_fp8: k_cache/v_cache innermost (head_size) stride must be 1");
  VT_CHECK(k_cache.stride[2] == head_size_k && v_cache.stride[2] == head_size_v,
           "reshape_and_cache_fp8: k_cache/v_cache page must be head-contiguous "
           "(stride[2] == head_size) — the NHD unbind-slice layout");
  // NO device-class guard. W1 hard-refused every non-CPU queue here, which is
  // what kept the CUDA arm unreachable; W2 lands that arm (cuda_cache.cu), so
  // the op resolves through the provider table like every other op and a device
  // with no registered fp8-KV store refuses BY NAME in GetOp
  // (src/vt/op_provider.cpp:563-567) instead of by device class.
  VT_CHECK(k.device == q.device && v.device == q.device && k_cache.device == q.device &&
               v_cache.device == q.device && slot_mapping.device == q.device,
           "reshape_and_cache_fp8: device mismatch (k/v/k_cache/v_cache/slot_mapping/queue)");
  reinterpret_cast<ReshapeAndCacheFp8Fn>(GetOp(OpId::kReshapeAndCacheFp8, q.device.type))(
      q, k, v, k_cache, v_cache, slot_mapping, kind, k_scale, v_scale);
}

void ConcatAndCacheMla(Queue& q, const Tensor& kv_c, const Tensor& k_pe, Tensor& kv_cache,
                       const Tensor& slot_mapping) {
  VT_CHECK(kv_c.rank == 2 && k_pe.rank == 2,
           "concat_and_cache_mla: kv_c/k_pe must be rank-2 "
           "[num_tokens, kv_lora_rank] / [num_tokens, qk_rope_head_dim]");
  VT_CHECK(kv_cache.rank == 3,
           "concat_and_cache_mla: kv_cache must be rank-3 "
           "[num_blocks, block_size, kv_lora_rank + qk_rope_head_dim] — MLA has "
           "NO K/V axis (mla_attention.py:1216-1224)");
  VT_CHECK(slot_mapping.rank == 1,
           "concat_and_cache_mla: slot_mapping must be rank-1 [num_slots]");
  const int64_t kv_lora_rank = kv_c.shape[1];
  const int64_t pe_dim = k_pe.shape[1];
  VT_CHECK(kv_c.shape[0] == k_pe.shape[0],
           "concat_and_cache_mla: kv_c and k_pe must share num_tokens");
  // Upstream's host-side contract, cache_kernels.cu:876
  // (`STD_TORCH_CHECK(kv_cache.size(2) == kv_lora_rank + pe_dim)`).
  VT_CHECK(kv_cache.shape[2] == kv_lora_rank + pe_dim,
           "concat_and_cache_mla: kv_cache entry width must equal "
           "kv_lora_rank + qk_rope_head_dim");
  // `pe_dim == 0` is the NoPE cache row (GLM-5.3-Flash, W3, #2213): the entry IS
  // the latent, `kv_cache.shape[2] == kv_lora_rank`, and both kernels' second
  // copy loop runs zero times. `kv_lora_rank == 0` stays refused — an MLA cache
  // with no latent is not a geometry.
  VT_CHECK(kv_lora_rank > 0, "concat_and_cache_mla: kv_lora_rank must be > 0");
  VT_CHECK(pe_dim >= 0,
           "concat_and_cache_mla: qk_rope_head_dim must be >= 0 (0 is the NoPE "
           "cache row, whose entry width is kv_lora_rank exactly)");
  // Upstream uses slot_mapping.size(0) as the token count (`:855-863`): kv_c/k_pe
  // may carry extra trailing rows (CUDA-graph padding) that are ignored.
  VT_CHECK(kv_c.shape[0] >= slot_mapping.shape[0],
           "concat_and_cache_mla: num_tokens must be >= slot_mapping length");
  // The "auto" cache path ONLY. fp8_ds_mla (the 656-byte V3.2 layout,
  // cache_kernels.cu:866-875) and int4 are out of campaign scope, so a mismatched
  // cache dtype is REFUSED rather than silently mis-written.
  VT_CHECK(IsFloat(kv_c.dtype) && kv_c.dtype == k_pe.dtype && kv_cache.dtype == kv_c.dtype,
           "concat_and_cache_mla: kv_c/k_pe/kv_cache must share one float dtype "
           "(the auto cache path; fp8_ds_mla is out of scope)");
  VT_CHECK(slot_mapping.dtype == DType::kI64,
           "concat_and_cache_mla: slot_mapping must be i64");
  // Indexing is stride-driven (upstream reads kv_cache.stride(0)/stride(1) and
  // kv_c/k_pe.stride(0)), so the cache need not be contiguous — a strided view
  // and a split-projection source row are both fine. We require only what the
  // copy actually needs: unit innermost stride on every operand.
  VT_CHECK(kv_c.stride[1] == 1 && k_pe.stride[1] == 1 && kv_cache.stride[2] == 1 &&
               slot_mapping.IsContiguous(),
           "concat_and_cache_mla: kv_c/k_pe/kv_cache innermost stride must be 1 "
           "and slot_mapping contiguous");
  VT_CHECK(kv_c.stride[0] >= kv_lora_rank && k_pe.stride[0] >= pe_dim,
           "concat_and_cache_mla: kv_c/k_pe token rows must not overlap");
  VT_CHECK(kv_c.device == q.device && k_pe.device == q.device &&
               kv_cache.device == q.device && slot_mapping.device == q.device,
           "concat_and_cache_mla: device mismatch "
           "(kv_c/k_pe/kv_cache/slot_mapping/queue)");
  reinterpret_cast<ConcatAndCacheMlaFn>(GetOp(OpId::kConcatAndCacheMla, q.device.type))(
      q, kv_c, k_pe, kv_cache, slot_mapping);
}

void ConcatAndCacheDsMla(Queue& q, const Tensor& k, Tensor& kv_cache,
                         const Tensor& slot_mapping, int64_t block_size) {
  // Upstream: `assert k.dim() == 2 and k.shape[1] == 512` (cache_utils.py:167-169).
  VT_CHECK(k.rank == 2 && k.shape[1] == kFp8DsMlaInputDim,
           "concat_and_cache_ds_mla: k must be rank-2 [num_tokens, 512] — the "
           "compressed latent, NoPE in [0, 448) and the rotated RoPE in [448, 512)");
  // The page is RANK-2 BYTES. This is the difference from concat_and_cache_mla
  // and it is not a formality: a token's scales live in a different REGION of
  // the block from its data (cache_utils.py:59-66), so no (block, row, column)
  // indexing reaches both and a rank-3 page cannot describe this cache.
  VT_CHECK(kv_cache.rank == 2,
           "concat_and_cache_ds_mla: kv_cache must be rank-2 "
           "[num_blocks, block_bytes] — the fp8_ds_mla block is REGION-SPLIT "
           "(cache_utils.py:59-66), so it is not a rank-3 page");
  VT_CHECK(kv_cache.dtype == DType::kI8,
           "concat_and_cache_ds_mla: kv_cache must be DType::kI8 (a BYTE page). A "
           "float cache here would be the 3.5x overrun this op exists to stop: "
           "2048 f32 bytes per token against the 584 the spec declares");
  VT_CHECK(slot_mapping.rank == 1,
           "concat_and_cache_ds_mla: slot_mapping must be rank-1 [num_slots]");
  VT_CHECK(slot_mapping.dtype == DType::kI64,
           "concat_and_cache_ds_mla: slot_mapping must be i64");
  // Upstream asserts bf16 (`:170-171`). We accept any float dtype because the
  // encoder's first act is upstream's own fp32 -> bf16 round
  // (`:110-118`, "Load bf16 input"), so an f32 row and its bf16 rounding
  // produce identical bytes; a non-float source has no such reading.
  VT_CHECK(IsFloat(k.dtype),
           "concat_and_cache_ds_mla: k must be a float dtype (upstream asserts "
           "bf16; the encoder rounds to bf16 first, so f32/f16 agree bit for bit)");
  VT_CHECK(block_size > 0,
           "concat_and_cache_ds_mla: block_size must be > 0 (it is the STORAGE "
           "block size, block_size / compress_ratio)");
  // 584 bytes per token (kv_cache_interface.py:401-403). Everything past that is
  // the block's alignment padding (`:63`) and no store may reach it, so the page
  // must be at least the real size; a LARGER row is the padded page and is fine.
  const int64_t real_block_bytes = block_size * kFp8DsMlaTokenBytes;
  VT_CHECK(kv_cache.shape[1] >= real_block_bytes,
           "concat_and_cache_ds_mla: kv_cache row must hold block_size * 584 bytes");
  VT_CHECK(k.stride[1] == 1 && kv_cache.stride[1] == 1 && slot_mapping.IsContiguous(),
           "concat_and_cache_ds_mla: k/kv_cache innermost stride must be 1 and "
           "slot_mapping contiguous");
  VT_CHECK(k.stride[0] >= kFp8DsMlaInputDim,
           "concat_and_cache_ds_mla: k token rows must not overlap");
  VT_CHECK(kv_cache.stride[0] >= kv_cache.shape[1],
           "concat_and_cache_ds_mla: kv_cache blocks must not overlap");
  // Upstream takes the token count from slot_mapping (`:186-188`): the tail of
  // `k` is DP padding and is ignored.
  VT_CHECK(k.shape[0] >= slot_mapping.shape[0],
           "concat_and_cache_ds_mla: num_tokens must be >= slot_mapping length");
  VT_CHECK(k.device == q.device && kv_cache.device == q.device &&
               slot_mapping.device == q.device,
           "concat_and_cache_ds_mla: device mismatch (k/kv_cache/slot_mapping/queue)");
  reinterpret_cast<ConcatAndCacheDsMlaFn>(
      GetOp(OpId::kConcatAndCacheDsMla, q.device.type))(q, k, kv_cache, slot_mapping,
                                                        block_size);
}

void DequantAndGatherDsMla(Queue& q, Tensor& out, const Tensor& kv_cache,
                           const Tensor& seq_lens, const Tensor* gather_lens,
                           const Tensor& block_table,
                           const DequantAndGatherDsMlaArgs& args) {
  VT_CHECK(out.rank == 3 && out.shape[2] == kFp8DsMlaInputDim,
           "dequant_and_gather_ds_mla: out must be rank-3 "
           "[num_reqs, max_num_tokens, 512]");
  // Upstream stores bf16 (`cache_utils.py:328`, `:337`). f32 is the annotated
  // widening: the scratch dtype must equal the query dtype the following
  // vt::MlaDecodeAttention runs at, and our CPU MLA decode runs at f32.
  VT_CHECK(out.dtype == DType::kF32 || out.dtype == DType::kBF16,
           "dequant_and_gather_ds_mla: out must be f32 or bf16 (bf16 mirrors "
           "upstream exactly; f32 is the scratch dtype our CPU MLA decode consumes)");
  VT_CHECK(kv_cache.rank == 2 && kv_cache.dtype == DType::kI8,
           "dequant_and_gather_ds_mla: kv_cache must be rank-2 "
           "[num_blocks, block_bytes] DType::kI8 — the REGION-SPLIT byte page");
  VT_CHECK(seq_lens.rank == 1 && seq_lens.dtype == DType::kI32,
           "dequant_and_gather_ds_mla: seq_lens must be rank-1 [num_reqs] i32");
  VT_CHECK(block_table.rank == 2 && block_table.dtype == DType::kI32,
           "dequant_and_gather_ds_mla: block_table must be rank-2 "
           "[num_reqs, max_blocks_per_seq] i32");
  const int64_t num_reqs = out.shape[0];
  VT_CHECK(num_reqs > 0, "dequant_and_gather_ds_mla: num_reqs must be > 0");
  VT_CHECK(seq_lens.shape[0] == num_reqs && block_table.shape[0] == num_reqs,
           "dequant_and_gather_ds_mla: seq_lens/block_table must have num_reqs rows");
  // nullptr is upstream's `gather_lens_ptr is None` arm — gather the WHOLE
  // sequence (`:257-262`) — and NOT an error.
  if (gather_lens != nullptr) {
    VT_CHECK(gather_lens->rank == 1 && gather_lens->dtype == DType::kI32 &&
                 gather_lens->shape[0] == num_reqs,
             "dequant_and_gather_ds_mla: gather_lens must be rank-1 [num_reqs] i32");
    VT_CHECK(gather_lens->IsContiguous() && gather_lens->device == q.device,
             "dequant_and_gather_ds_mla: gather_lens must be contiguous and on the "
             "queue's device");
  }
  VT_CHECK(args.block_size > 0,
           "dequant_and_gather_ds_mla: args.block_size must be > 0 (the STORAGE "
           "block size, block_size / compress_ratio)");
  VT_CHECK(args.offset >= 0 && args.offset < out.shape[1],
           "dequant_and_gather_ds_mla: args.offset must index a column of out");
  VT_CHECK(kv_cache.shape[1] >= args.block_size * kFp8DsMlaTokenBytes,
           "dequant_and_gather_ds_mla: kv_cache row must hold block_size * 584 bytes");
  VT_CHECK(out.stride[2] == 1 && kv_cache.stride[1] == 1 && block_table.stride[1] == 1 &&
               seq_lens.IsContiguous(),
           "dequant_and_gather_ds_mla: out/kv_cache/block_table innermost stride must "
           "be 1 and seq_lens contiguous");
  VT_CHECK(out.device == q.device && kv_cache.device == q.device &&
               seq_lens.device == q.device && block_table.device == q.device,
           "dequant_and_gather_ds_mla: device mismatch "
           "(out/kv_cache/seq_lens/block_table/queue)");
  reinterpret_cast<DequantAndGatherDsMlaFn>(
      GetOp(OpId::kDequantAndGatherDsMla, q.device.type))(q, out, kv_cache, seq_lens,
                                                          gather_lens, block_table, args);
}

void MlaDecodeAttention(Queue& q, Tensor& out, Tensor* lse, const Tensor& query,
                        const Tensor& kv_cache, const Tensor& block_table,
                        const Tensor& seq_lens, const MlaDecodeAttentionArgs& args) {
  VT_CHECK(query.rank == 3 && out.rank == 3,
           "mla_decode_attention: query/out must be rank-3 "
           "[batch, num_q_heads, head_size] / [batch, num_q_heads, v_head_dim]");
  // The MLA cache has NO K/V axis and NO head axis (mla_attention.py:1216-1224).
  VT_CHECK(kv_cache.rank == 3,
           "mla_decode_attention: kv_cache must be rank-3 "
           "[num_blocks, block_size, head_size] — MLA has no K/V and no head axis");
  VT_CHECK(block_table.rank == 2, "mla_decode_attention: block_table rank-2 [batch, max_blocks]");
  VT_CHECK(seq_lens.rank == 1, "mla_decode_attention: seq_lens rank-1 [batch]");
  const int64_t batch = query.shape[0];
  const int64_t num_q_heads = query.shape[1];
  const int64_t head_size = query.shape[2];
  const int64_t v_head_dim = out.shape[2];
  VT_CHECK(batch > 0 && num_q_heads > 0 && head_size > 0 && v_head_dim > 0,
           "mla_decode_attention: batch/num_q_heads/head_size/v_head_dim must be > 0");
  VT_CHECK(out.shape[0] == batch && out.shape[1] == num_q_heads,
           "mla_decode_attention: out must be [batch, num_q_heads, v_head_dim]");
  VT_CHECK(kv_cache.shape[2] == head_size,
           "mla_decode_attention: kv_cache entry width must equal query head_size "
           "(kv_lora_rank + qk_rope_head_dim)");
  // V is the LEADING v_head_dim slice of the same latent row (triton_mla.py:236,
  // `kv_c_cache = kv_c_and_k_pe_cache[..., :self.kv_lora_rank]`).
  VT_CHECK(v_head_dim <= head_size,
           "mla_decode_attention: v_head_dim must be <= head_size (V is the leading "
           "kv_lora_rank slice of the SAME cache row)");
  VT_CHECK(kv_cache.shape[1] > 0, "mla_decode_attention: block_size must be > 0");
  VT_CHECK(block_table.shape[0] == batch && seq_lens.shape[0] == batch,
           "mla_decode_attention: block_table/seq_lens must have `batch` rows");
  VT_CHECK(block_table.dtype == DType::kI32 && seq_lens.dtype == DType::kI32,
           "mla_decode_attention: block_table/seq_lens must be i32");
  VT_CHECK(IsFloat(query.dtype) && kv_cache.dtype == query.dtype && out.dtype == query.dtype,
           "mla_decode_attention: query/kv_cache/out must share one float dtype "
           "(the auto cache path; the fp8 KV-cache branch is out of scope)");
  VT_CHECK(args.scale > 0.0f, "mla_decode_attention: args.scale must be > 0");
  VT_CHECK(args.num_kv_splits >= 0, "mla_decode_attention: args.num_kv_splits must be >= 0");
  // The sliding-window arm (dots3-note W4b-2, #699). `left` is the inclusive
  // distance behind the query, so the window WIDTH is `left + 1` and upstream's
  // `sliding_window_size` 513 arrives as `left == 512`. A zero-width window
  // would leave a decode row with no keys at all, which upstream cannot
  // produce (`WINDOW_SIZE` is a positive config field), so it is refused rather
  // than silently emitting zeros.
  if (args.window_size.has_value()) {
    VT_CHECK(args.window_size->left >= 0,
             "mla_decode_attention: window_size.left must be >= 0 (it is the INCLUSIVE "
             "distance behind the query; sliding_window 513 is left == 512)");
    VT_CHECK(args.window_size->right == 0,
             "mla_decode_attention: window_size.right must be 0 — an MLA decode query IS "
             "the last position of its own sequence, so a positive right bound could only "
             "admit keys that do not exist. Upstream's dots3-note window is "
             "(sliding_window - 1, 0) (attention.py:300 @ bc2d63e650).");
  }
  // The DSA SELECTED-SLOT arm (dots3-note W4b-3c, #699). Both null is the
  // ABSENT state and a NOT-TAKEN branch; exactly one present is a caller bug
  // that would otherwise silently serve dense attention on a sparse model, so
  // it is refused BY NAME rather than ignored.
  if (args.topk_indices != nullptr || args.valid_counts != nullptr) {
    VT_CHECK(args.topk_indices != nullptr && args.valid_counts != nullptr,
             "mla_decode_attention: topk_indices and valid_counts must be supplied "
             "TOGETHER — one without the other cannot describe a selection "
             "(upstream returns both from `triton_convert_req_index_to_global_index`, "
             "attention.py:760-767 @ bc2d63e650)");
    const Tensor& ti = *args.topk_indices;
    const Tensor& vc = *args.valid_counts;
    VT_CHECK(ti.rank == 2 && vc.rank == 1,
             "mla_decode_attention: topk_indices must be rank-2 [batch, topk] and "
             "valid_counts rank-1 [batch]");
    VT_CHECK(ti.shape[0] == batch && vc.shape[0] == batch,
             "mla_decode_attention: topk_indices/valid_counts must have `batch` rows");
    VT_CHECK(ti.shape[1] > 0,
             "mla_decode_attention: topk_indices must have at least one column — a "
             "topk of 0 selects nothing and upstream's `index_topk` is a positive "
             "config field");
    VT_CHECK(ti.dtype == DType::kI32 && vc.dtype == DType::kI32,
             "mla_decode_attention: topk_indices/valid_counts must be i32");
    VT_CHECK(ti.stride[1] == 1 && vc.IsContiguous(),
             "mla_decode_attention: topk_indices rows must be contiguous and "
             "valid_counts contiguous");
    VT_CHECK(ti.device == q.device && vc.device == q.device,
             "mla_decode_attention: topk_indices/valid_counts device mismatch");
    // Upstream cannot produce a windowed layer that also selects:
    // `Dots3NoteSlidingAttention` sets `self.indexer = None` / `is_sparse =
    // False` (model.py:432-434 @ bc2d63e650), so the sliding geometry carries no
    // indexer at all. Refusing the pair keeps the two arms' bounds from
    // silently composing into a key set upstream has no counterpart for.
    VT_CHECK(!args.window_size.has_value(),
             "mla_decode_attention: a sliding window and a DSA selection cannot be "
             "combined — upstream's sliding layers set `self.indexer = None` and "
             "`is_sparse = False` (model.py:432-434 @ bc2d63e650), so no layer "
             "carries both");
    // The COUNT bound. Reading `valid_counts` here would force a device
    // synchronization on every decode step of a sparse model, which is a
    // per-step cost on the model path, so the value check runs only where the
    // memory is host-readable. BOTH kernels additionally clamp `min(count,
    // topk)`, so an over-large count on a device tensor cannot read out of
    // bounds — it is refused where it can be seen and contained where it
    // cannot. Recorded as a deviation rather than left to be discovered.
    if (vc.device.type == DeviceType::kCPU) {
      const int32_t* counts = vc.Ptr<int32_t>();
      for (int64_t b = 0; b < batch; ++b) {
        VT_CHECK(counts[b] >= 0 && counts[b] <= ti.shape[1],
                 "mla_decode_attention: valid_counts[" + std::to_string(b) + "] is " +
                     std::to_string(counts[b]) + " but topk is " +
                     std::to_string(ti.shape[1]) +
                     " — a count past the row length names slots that do not exist");
      }
    }
  }
  // Indexing is stride-driven on the leading dims (a cross-layer cache view has
  // gaps — cf. upstream `_page_stride`, triton_decode_attention.py:59-65), so we
  // require only unit innermost strides.
  VT_CHECK(query.stride[2] == 1 && out.stride[2] == 1 && kv_cache.stride[2] == 1,
           "mla_decode_attention: query/out/kv_cache innermost stride must be 1");
  VT_CHECK(block_table.stride[1] == 1 && seq_lens.IsContiguous(),
           "mla_decode_attention: block_table rows must be contiguous and seq_lens contiguous");
  if (lse != nullptr) {
    VT_CHECK(lse->rank == 2 && lse->shape[0] == batch && lse->shape[1] == num_q_heads,
             "mla_decode_attention: lse must be rank-2 [batch, num_q_heads]");
    VT_CHECK(lse->dtype == DType::kF32,
             "mla_decode_attention: lse must be f32 (we keep the LSE in the "
             "accumulation dtype; upstream stores it in q.dtype)");
    VT_CHECK(lse->stride[1] == 1, "mla_decode_attention: lse innermost stride must be 1");
    VT_CHECK(lse->device == q.device, "mla_decode_attention: lse device mismatch");
  }
  VT_CHECK(out.device == q.device && query.device == q.device && kv_cache.device == q.device &&
               block_table.device == q.device && seq_lens.device == q.device,
           "mla_decode_attention: device mismatch "
           "(out/query/kv_cache/block_table/seq_lens/queue)");
  reinterpret_cast<MlaDecodeAttentionFn>(GetOp(OpId::kMlaDecodeAttention, q.device.type))(
      q, out, lse, query, kv_cache, block_table, seq_lens, args);
}

// The DSA "Lightning Indexer" selection pair (dots3-note W4b-3c, #699).
// Ported from vllm/v1/attention/ops/triton_fp8_mqa_logits.py:120-156 and
// vllm/model_executor/layers/sparse_attn_indexer.py:509 @ bc2d63e650. The
// host reference these mirror is vllm::deepseek_v4::DsaIndexerLogits /
// DsaTopkSelect, which remains the gate's oracle.
void DsaIndexerLogits(Queue& q, Tensor& logits, const Tensor& q_states, const Tensor& k,
                      const Tensor& weights, const Tensor& win_start,
                      const Tensor& win_end, const DsaIndexerLogitsArgs& args) {
  VT_CHECK(q_states.rank == 3,
           "dsa_indexer_logits: q must be rank-3 [num_tokens, index_n_heads, "
           "index_head_dim]");
  VT_CHECK(k.rank == 2,
           "dsa_indexer_logits: k must be rank-2 [num_keys, index_head_dim] — the "
           "indexer is MQA, so there is exactly ONE KV head (deepseek_v2.py:700-707 "
           "@ bc2d63e650)");
  VT_CHECK(logits.rank == 2, "dsa_indexer_logits: logits must be rank-2 [num_tokens, num_keys]");
  VT_CHECK(weights.rank == 2,
           "dsa_indexer_logits: weights must be rank-2 [num_tokens, index_n_heads]");
  VT_CHECK(win_start.rank == 1 && win_end.rank == 1,
           "dsa_indexer_logits: win_start/win_end must be rank-1 [num_tokens]");
  const int64_t T = q_states.shape[0];
  const int64_t H = q_states.shape[1];
  const int64_t D = q_states.shape[2];
  const int64_t S = k.shape[0];
  VT_CHECK(T > 0 && H > 0 && D > 0 && S > 0,
           "dsa_indexer_logits: num_tokens/index_n_heads/index_head_dim/num_keys "
           "must be > 0");
  VT_CHECK(k.shape[1] == D,
           "dsa_indexer_logits: k's width must equal q's index_head_dim (the dot "
           "spans the WHOLE 128-wide indexer head, triton_fp8_mqa_logits.py:125)");
  VT_CHECK(logits.shape[0] == T && logits.shape[1] == S,
           "dsa_indexer_logits: logits must be [num_tokens, num_keys]");
  VT_CHECK(weights.shape[0] == T && weights.shape[1] == H,
           "dsa_indexer_logits: weights must be [num_tokens, index_n_heads] (one gate "
           "per query token per indexer head)");
  VT_CHECK(win_start.shape[0] == T && win_end.shape[0] == T,
           "dsa_indexer_logits: win_start/win_end must have one entry per token");
  VT_CHECK(win_start.dtype == DType::kI32 && win_end.dtype == DType::kI32,
           "dsa_indexer_logits: win_start/win_end must be i32");
  VT_CHECK(logits.dtype == DType::kF32,
           "dsa_indexer_logits: logits must be f32 — upstream's MQA-logit kernel "
           "accumulates and stores f32 whatever the operand dtype "
           "(triton_fp8_mqa_logits.py:125 `input_precision=\"ieee\"`)");
  VT_CHECK(IsFloat(q_states.dtype) && k.dtype == q_states.dtype &&
               weights.dtype == q_states.dtype,
           "dsa_indexer_logits: q/k/weights must share one float dtype");
  VT_CHECK(args.softmax_scale > 0.0f && args.n_head_scale > 0.0f,
           "dsa_indexer_logits: softmax_scale and n_head_scale must both be > 0 "
           "(`head_dim**-0.5` deepseek_v2.py:709, `n_head**-0.5` :742)");
  VT_CHECK(q_states.stride[2] == 1 && k.stride[1] == 1 && weights.stride[1] == 1 &&
               logits.stride[1] == 1,
           "dsa_indexer_logits: q/k/weights/logits innermost stride must be 1");
  VT_CHECK(win_start.IsContiguous() && win_end.IsContiguous(),
           "dsa_indexer_logits: win_start/win_end must be contiguous");
  if (args.q_scale != nullptr) {
    VT_CHECK(args.q_scale->rank == 2 && args.q_scale->shape[0] == T &&
                 args.q_scale->shape[1] == H,
             "dsa_indexer_logits: q_scale must be [num_tokens, index_n_heads] — it is "
             "the per-token-per-head fp8 quantization scale folded into `weights` "
             "(deepseek_v2.py:838,:840 @ bc2d63e650)");
    VT_CHECK(args.q_scale->dtype == DType::kF32,
             "dsa_indexer_logits: q_scale must be f32 (per_token_group_quant_fp8 "
             "returns an fp32 scale, deepseek_v2.py:831-836)");
    VT_CHECK(args.q_scale->stride[1] == 1 && args.q_scale->device == q.device,
             "dsa_indexer_logits: q_scale innermost stride must be 1 and its device "
             "must match the queue");
  }
  VT_CHECK(logits.device == q.device && q_states.device == q.device &&
               k.device == q.device && weights.device == q.device &&
               win_start.device == q.device && win_end.device == q.device,
           "dsa_indexer_logits: device mismatch "
           "(logits/q/k/weights/win_start/win_end/queue)");
  reinterpret_cast<DsaIndexerLogitsFn>(GetOp(OpId::kDsaIndexerLogits, q.device.type))(
      q, logits, q_states, k, weights, win_start, win_end, args);
}

void DsaTopkSelect(Queue& q, Tensor& indices, Tensor& counts, const Tensor& logits,
                   const Tensor& win_start, const Tensor& win_end) {
  VT_CHECK(indices.rank == 2, "dsa_topk_select: indices must be rank-2 [num_tokens, topk]");
  VT_CHECK(counts.rank == 1, "dsa_topk_select: counts must be rank-1 [num_tokens]");
  VT_CHECK(logits.rank == 2, "dsa_topk_select: logits must be rank-2 [num_tokens, num_keys]");
  VT_CHECK(win_start.rank == 1 && win_end.rank == 1,
           "dsa_topk_select: win_start/win_end must be rank-1 [num_tokens]");
  const int64_t T = logits.shape[0];
  const int64_t S = logits.shape[1];
  const int64_t topk = indices.shape[1];
  VT_CHECK(T > 0 && S > 0, "dsa_topk_select: num_tokens/num_keys must be > 0");
  VT_CHECK(topk > 0,
           "dsa_topk_select: topk must be > 0 — upstream's `index_topk` is a positive "
           "config field (deepseek_v2.py:685)");
  VT_CHECK(indices.shape[0] == T && counts.shape[0] == T,
           "dsa_topk_select: indices/counts must have one row per query token");
  VT_CHECK(win_start.shape[0] == T && win_end.shape[0] == T,
           "dsa_topk_select: win_start/win_end must have one entry per token");
  VT_CHECK(indices.dtype == DType::kI32 && counts.dtype == DType::kI32 &&
               win_start.dtype == DType::kI32 && win_end.dtype == DType::kI32,
           "dsa_topk_select: indices/counts/win_start/win_end must be i32 — the pair "
           "feeds MlaDecodeAttentionArgs::topk_indices/valid_counts directly");
  VT_CHECK(logits.dtype == DType::kF32, "dsa_topk_select: logits must be f32");
  VT_CHECK(indices.stride[1] == 1 && logits.stride[1] == 1 && counts.IsContiguous() &&
               win_start.IsContiguous() && win_end.IsContiguous(),
           "dsa_topk_select: indices/logits rows must be contiguous and "
           "counts/win_start/win_end contiguous");
  VT_CHECK(indices.device == q.device && counts.device == q.device &&
               logits.device == q.device && win_start.device == q.device &&
               win_end.device == q.device,
           "dsa_topk_select: device mismatch (indices/counts/logits/win_start/win_end/queue)");
  reinterpret_cast<DsaTopkSelectFn>(GetOp(OpId::kDsaTopkSelect, q.device.type))(
      q, indices, counts, logits, win_start, win_end);
}

void Glm5NextKpoolCompress(Queue& q, Tensor& pool_keys, Tensor& pool_indices,
                           Tensor& pool_valid, Tensor& num_pools, const Tensor& packed,
                           const Tensor& ape) {
  VT_CHECK(packed.rank == 3,
           "glm5_next_kpool_compress: packed must be rank-3 [batch, kv_len, "
           "2 * index_head_dim + 1] — the indexer cache row is "
           "`concat[k, gate_scores, valid]` (modular_glm5_next.py:798-801 @ "
           "transformers v5.16.1), not the 128-wide key DeepSeek-V4 caches");
  VT_CHECK(ape.rank == 2,
           "glm5_next_kpool_compress: ape must be rank-2 [index_kpool, index_head_dim]");
  VT_CHECK(pool_keys.rank == 3 && pool_indices.rank == 3 && pool_valid.rank == 2 &&
               num_pools.rank == 1,
           "glm5_next_kpool_compress: pool_keys/pool_indices must be rank-3, pool_valid "
           "rank-2, num_pools rank-1");
  const int64_t batch = packed.shape[0];
  const int64_t kv_len = packed.shape[1];
  const int64_t kpool = ape.shape[0];
  const int64_t head_dim = ape.shape[1];
  VT_CHECK(batch > 0 && kv_len > 0 && head_dim > 0,
           "glm5_next_kpool_compress: batch/kv_len/index_head_dim must be > 0");
  VT_CHECK(kpool >= 1,
           "glm5_next_kpool_compress: index_kpool must be >= 1 "
           "(configuration_glm5_next.py:216-217); it is 4 on the published checkpoint "
           "and 16 in the config class, so a defaulted value is wrong by a factor of "
           "four");
  VT_CHECK(packed.shape[2] == 2 * head_dim + 1,
           "glm5_next_kpool_compress: packed's row must be 2 * index_head_dim + 1 wide");
  const int64_t np = (kv_len + kpool - 1) / kpool;
  VT_CHECK(pool_keys.shape[0] == batch && pool_keys.shape[1] == np &&
               pool_keys.shape[2] == head_dim,
           "glm5_next_kpool_compress: pool_keys must be [batch, ceil(kv_len / "
           "index_kpool), index_head_dim] — the STATIC upper bound, because the live "
           "width P is only known after the `keep` compaction (:968-970)");
  VT_CHECK(pool_indices.shape[0] == batch && pool_indices.shape[1] == np &&
               pool_indices.shape[2] == kpool,
           "glm5_next_kpool_compress: pool_indices must be [batch, np, index_kpool]");
  VT_CHECK(pool_valid.shape[0] == batch && pool_valid.shape[1] == np,
           "glm5_next_kpool_compress: pool_valid must be [batch, np]");
  VT_CHECK(num_pools.shape[0] == 1,
           "glm5_next_kpool_compress: num_pools must be a [1] DEVICE scalar — reading P "
           "back to the host is the round trip this op family exists to remove");
  VT_CHECK(packed.dtype == DType::kF32 && ape.dtype == DType::kF32 &&
               pool_keys.dtype == DType::kF32,
           "glm5_next_kpool_compress: packed/ape/pool_keys must be f32 — upstream scores "
           "and pools in fp32 (:823, :960-964)");
  VT_CHECK(pool_indices.dtype == DType::kI32 && pool_valid.dtype == DType::kI32 &&
               num_pools.dtype == DType::kI32,
           "glm5_next_kpool_compress: pool_indices/pool_valid/num_pools must be i32");
  VT_CHECK(packed.IsContiguous() && ape.IsContiguous() && pool_keys.IsContiguous() &&
               pool_indices.IsContiguous() && pool_valid.IsContiguous() &&
               num_pools.IsContiguous(),
           "glm5_next_kpool_compress: every operand must be contiguous");
  VT_CHECK(packed.device == q.device && ape.device == q.device &&
               pool_keys.device == q.device && pool_indices.device == q.device &&
               pool_valid.device == q.device && num_pools.device == q.device,
           "glm5_next_kpool_compress: device mismatch (operands/queue)");
  reinterpret_cast<Glm5NextKpoolCompressFn>(
      GetOp(OpId::kGlm5NextKpoolCompress, q.device.type))(q, pool_keys, pool_indices,
                                                          pool_valid, num_pools, packed,
                                                          ape);
}

void Glm5NextKpoolSelect(Queue& q, Tensor& topk_indices, Tensor& index_scores,
                         const Tensor& q_states, const Tensor& head_weights,
                         const Tensor& pool_keys, const Tensor& pool_indices,
                         const Tensor& pool_valid, const Tensor& num_pools,
                         const Tensor& valid_keys, const Tensor& q_mask,
                         const Glm5NextKpoolSelectArgs& args) {
  VT_CHECK(q_states.rank == 4,
           "glm5_next_kpool_select: q_states must be rank-4 [batch, seq_len, "
           "index_n_heads, index_head_dim]");
  VT_CHECK(head_weights.rank == 3,
           "glm5_next_kpool_select: head_weights must be rank-3 [batch, seq_len, "
           "index_n_heads] — the weights_proj output BEFORE the n_heads ** -0.5 scale, "
           "which this op applies (:827)");
  VT_CHECK(pool_keys.rank == 3 && pool_indices.rank == 3 && pool_valid.rank == 2,
           "glm5_next_kpool_select: the pooled candidate set must come from "
           "vt::Glm5NextKpoolCompress unchanged");
  VT_CHECK(num_pools.rank == 1 && num_pools.shape[0] == 1,
           "glm5_next_kpool_select: num_pools must be the [1] DEVICE scalar the compress "
           "op published");
  VT_CHECK(valid_keys.rank == 2 && q_mask.rank == 2,
           "glm5_next_kpool_select: valid_keys must be [batch, kv_len] and q_mask "
           "[batch, seq_len]");
  VT_CHECK(topk_indices.rank == 3 && index_scores.rank == 3,
           "glm5_next_kpool_select: topk_indices/index_scores must be rank-3");
  const int64_t batch = q_states.shape[0];
  const int64_t seq_len = q_states.shape[1];
  const int64_t n_heads = q_states.shape[2];
  const int64_t head_dim = q_states.shape[3];
  const int64_t np = pool_keys.shape[1];
  const int64_t kpool = pool_indices.shape[2];
  const int64_t kv_len = valid_keys.shape[1];
  VT_CHECK(batch > 0 && seq_len > 0 && n_heads > 0 && head_dim > 0 && kv_len > 0,
           "glm5_next_kpool_select: batch/seq_len/index_n_heads/index_head_dim/kv_len "
           "must be > 0");
  VT_CHECK(kpool >= 1, "glm5_next_kpool_select: index_kpool must be >= 1");
  VT_CHECK(args.index_topk > 0, "glm5_next_kpool_select: index_topk must be > 0");
  VT_CHECK(args.index_topk % kpool == 0,
           "glm5_next_kpool_select: index_topk must be divisible by index_kpool — the "
           "pool budget `index_topk // index_kpool` is exact upstream "
           "(configuration_glm5_next.py:219-220)");
  VT_CHECK(kv_len >= seq_len,
           "glm5_next_kpool_select: kv_len must be at least seq_len — the current window "
           "is always part of the key history it selects over");
  VT_CHECK(args.current_length >= seq_len,
           "glm5_next_kpool_select: current_length must be at least seq_len; the query at "
           "step s sits at current_length - seq_len + s (:892)");
  VT_CHECK(pool_keys.shape[0] == batch && pool_keys.shape[2] == head_dim,
           "glm5_next_kpool_select: pool_keys must be [batch, np, index_head_dim]");
  VT_CHECK(pool_indices.shape[0] == batch && pool_indices.shape[1] == np,
           "glm5_next_kpool_select: pool_indices must be [batch, np, index_kpool]");
  VT_CHECK(pool_valid.shape[0] == batch && pool_valid.shape[1] == np,
           "glm5_next_kpool_select: pool_valid must be [batch, np]");
  VT_CHECK(np == (kv_len + kpool - 1) / kpool,
           "glm5_next_kpool_select: np must be ceil(kv_len / index_kpool), the same "
           "static bound the compress op allocated against");
  VT_CHECK(head_weights.shape[0] == batch && head_weights.shape[1] == seq_len &&
               head_weights.shape[2] == n_heads,
           "glm5_next_kpool_select: head_weights must be [batch, seq_len, index_n_heads]");
  VT_CHECK(q_mask.shape[0] == batch && q_mask.shape[1] == seq_len,
           "glm5_next_kpool_select: q_mask must be [batch, seq_len]");
  VT_CHECK(valid_keys.shape[0] == batch,
           "glm5_next_kpool_select: valid_keys must be [batch, kv_len]");
  const int64_t out_w = args.index_topk + (args.always_select_tail ? kpool - 1 : 0);
  VT_CHECK(topk_indices.shape[0] == batch && topk_indices.shape[1] == seq_len &&
               topk_indices.shape[2] == out_w,
           "glm5_next_kpool_select: topk_indices must be [batch, seq_len, index_topk + "
           "index_kpool - 1] with the always-kept tail (:864-867) — 2051 and not 2048 on "
           "the published checkpoint; sizing it 2048 truncates the tail the model always "
           "keeps");
  VT_CHECK(index_scores.shape[0] == batch && index_scores.shape[1] == seq_len &&
               index_scores.shape[2] == np,
           "glm5_next_kpool_select: index_scores must be [batch, seq_len, np]");
  VT_CHECK(q_states.dtype == DType::kF32 && head_weights.dtype == DType::kF32 &&
               pool_keys.dtype == DType::kF32 && index_scores.dtype == DType::kF32,
           "glm5_next_kpool_select: q_states/head_weights/pool_keys/index_scores must be "
           "f32");
  VT_CHECK(pool_indices.dtype == DType::kI32 && pool_valid.dtype == DType::kI32 &&
               num_pools.dtype == DType::kI32 && valid_keys.dtype == DType::kI32 &&
               q_mask.dtype == DType::kI32 && topk_indices.dtype == DType::kI32,
           "glm5_next_kpool_select: every integer operand must be i32");
  VT_CHECK(q_states.IsContiguous() && head_weights.IsContiguous() &&
               pool_keys.IsContiguous() && pool_indices.IsContiguous() &&
               pool_valid.IsContiguous() && num_pools.IsContiguous() &&
               valid_keys.IsContiguous() && q_mask.IsContiguous() &&
               topk_indices.IsContiguous() && index_scores.IsContiguous(),
           "glm5_next_kpool_select: every operand must be contiguous");
  VT_CHECK(q_states.device == q.device && head_weights.device == q.device &&
               pool_keys.device == q.device && pool_indices.device == q.device &&
               pool_valid.device == q.device && num_pools.device == q.device &&
               valid_keys.device == q.device && q_mask.device == q.device &&
               topk_indices.device == q.device && index_scores.device == q.device,
           "glm5_next_kpool_select: device mismatch (operands/queue)");
  reinterpret_cast<Glm5NextKpoolSelectFn>(GetOp(OpId::kGlm5NextKpoolSelect,
                                                q.device.type))(
      q, topk_indices, index_scores, q_states, head_weights, pool_keys, pool_indices,
      pool_valid, num_pools, valid_keys, q_mask, args);
}

void MlaPrefillAttention(Queue& q, Tensor& out, Tensor* lse, const Tensor& query,
                         const Tensor& key, const Tensor& value, const Tensor& cu_seqlens_q,
                         const Tensor& cu_seqlens_k, const MlaPrefillAttentionArgs& args) {
  VT_CHECK(query.rank == 3 && key.rank == 3 && value.rank == 3 && out.rank == 3,
           "mla_prefill_attention: query/key/value/out must be rank-3 "
           "[total_q|total_k, num_heads, head_dim]");
  VT_CHECK(cu_seqlens_q.rank == 1 && cu_seqlens_k.rank == 1,
           "mla_prefill_attention: cu_seqlens_q/cu_seqlens_k must be rank-1 [num_reqs+1]");
  const int64_t total_q = query.shape[0];
  const int64_t total_k = key.shape[0];
  const int64_t num_heads = query.shape[1];
  const int64_t qk_head_dim = query.shape[2];
  const int64_t v_head_dim = value.shape[2];
  VT_CHECK(num_heads > 0 && qk_head_dim > 0 && v_head_dim > 0,
           "mla_prefill_attention: num_heads/qk_head_dim/v_head_dim must be > 0");
  VT_CHECK(key.shape[1] == num_heads && value.shape[1] == num_heads,
           "mla_prefill_attention: key/value must carry the SAME head count as query "
           "(MLA prefill is multi-head on both sides — the latent is up-projected; "
           "mla_attention.py:315-318)");
  VT_CHECK(key.shape[2] == qk_head_dim,
           "mla_prefill_attention: key head_dim must equal query head_dim "
           "(qk_nope_head_dim + qk_rope_head_dim)");
  VT_CHECK(value.shape[0] == total_k,
           "mla_prefill_attention: key and value must share total_k");
  VT_CHECK(out.shape[0] == total_q && out.shape[1] == num_heads && out.shape[2] == v_head_dim,
           "mla_prefill_attention: out must be [total_q, num_heads, v_head_dim]");
  // Upstream pads V UP to the QK width (flash_attn.py:164-168), so V may never
  // be WIDER than QK. For DeepSeek v_head_dim 128 <= qk_head_dim 192.
  VT_CHECK(v_head_dim <= qk_head_dim,
           "mla_prefill_attention: v_head_dim must be <= qk_head_dim (upstream ZERO-PADS "
           "V up to the QK width, flash_attn.py:164-168)");
  VT_CHECK(cu_seqlens_q.shape[0] >= 2 && cu_seqlens_q.shape[0] == cu_seqlens_k.shape[0],
           "mla_prefill_attention: cu_seqlens_q/cu_seqlens_k must both be [num_reqs+1]");
  VT_CHECK(cu_seqlens_q.dtype == DType::kI32 && cu_seqlens_k.dtype == DType::kI32,
           "mla_prefill_attention: cu_seqlens_q/cu_seqlens_k must be i32");
  VT_CHECK(cu_seqlens_q.IsContiguous() && cu_seqlens_k.IsContiguous(),
           "mla_prefill_attention: cu_seqlens_q/cu_seqlens_k must be contiguous");
  VT_CHECK(IsFloat(query.dtype) && key.dtype == query.dtype && value.dtype == query.dtype &&
               out.dtype == query.dtype,
           "mla_prefill_attention: query/key/value/out must share one float dtype "
           "(the fp8-prefill branch needs device-capability family 100 and is "
           "UNREACHABLE on sm_121 — mla_attention.py:1382-1385)");
  VT_CHECK(args.scale > 0.0f, "mla_prefill_attention: args.scale must be > 0");
  VT_CHECK(args.max_seqlen_q >= 0 && args.max_seqlen_k >= 0,
           "mla_prefill_attention: args.max_seqlen_q/max_seqlen_k must be >= 0");
  // The sliding-window arm (dots3-note W4b-2, #699). Upstream's only windowed
  // prefill call is `causal=True, window_size=(sliding_window - 1, 0)`
  // (attention.py:279-305 @ bc2d63e650). A NON-causal window is refused rather
  // than approximated: FlashAttention's local mask replaces the causal
  // specialization entirely, so "all keys forward, windowed backward" would need
  // an infinite right bound this struct cannot express.
  if (args.window_size.has_value()) {
    VT_CHECK(args.window_size->left >= 0,
             "mla_prefill_attention: window_size.left must be >= 0 (the INCLUSIVE distance "
             "behind the bottom-right aligned query position)");
    VT_CHECK(args.window_size->right == 0,
             "mla_prefill_attention: window_size.right must be 0 — upstream's windowed MLA "
             "prefill is the causal (sliding_window - 1, 0) pair (attention.py:300)");
    VT_CHECK(args.causal,
             "mla_prefill_attention: a window requires causal=true. FlashAttention's local "
             "mask REPLACES the causal specialization (is_causal = causal && !is_local), so "
             "a non-causal window cannot be spelled with a finite right bound. Upstream "
             "never asks for one (attention.py:299-301).");
  }
  // Stride-driven on the token/head axes (a workspace slice is a strided view);
  // the innermost head_dim must be packed.
  VT_CHECK(query.stride[2] == 1 && key.stride[2] == 1 && value.stride[2] == 1 &&
               out.stride[2] == 1,
           "mla_prefill_attention: query/key/value/out innermost stride must be 1");
  if (lse != nullptr) {
    // UNPADDED varlen LSE, upstream's [num_heads, total_q] layout.
    VT_CHECK(lse->rank == 2 && lse->shape[0] == num_heads && lse->shape[1] == total_q,
             "mla_prefill_attention: lse must be rank-2 [num_heads, total_q] (upstream's "
             "unpadded varlen LSE layout)");
    VT_CHECK(lse->dtype == DType::kF32, "mla_prefill_attention: lse must be f32");
    VT_CHECK(lse->stride[1] == 1, "mla_prefill_attention: lse innermost stride must be 1");
    VT_CHECK(lse->device == q.device, "mla_prefill_attention: lse device mismatch");
  }
  VT_CHECK(out.device == q.device && query.device == q.device && key.device == q.device &&
               value.device == q.device && cu_seqlens_q.device == q.device &&
               cu_seqlens_k.device == q.device,
           "mla_prefill_attention: device mismatch "
           "(out/query/key/value/cu_seqlens_q/cu_seqlens_k/queue)");
  reinterpret_cast<MlaPrefillAttentionFn>(GetOp(OpId::kMlaPrefillAttention, q.device.type))(
      q, out, lse, query, key, value, cu_seqlens_q, cu_seqlens_k, args);
}

void GatherMlaCache(Queue& q, Tensor& dst, const Tensor& src_cache, const Tensor& block_table,
                    const Tensor& cu_seq_lens, const Tensor& token_to_seq,
                    const Tensor* seq_starts, int64_t num_tokens) {
  VT_CHECK(dst.rank == 2, "gather_mla_cache: dst must be rank-2 [tot_tokens, head_dim]");
  VT_CHECK(src_cache.rank == 3,
           "gather_mla_cache: src_cache must be rank-3 [num_blocks, block_size, head_dim] "
           "— the 3-D MLA cache (mla_attention.py:1216-1224)");
  VT_CHECK(block_table.rank == 2, "gather_mla_cache: block_table rank-2 [batch, max_blocks]");
  VT_CHECK(cu_seq_lens.rank == 1, "gather_mla_cache: cu_seq_lens rank-1 [batch+1]");
  VT_CHECK(token_to_seq.rank == 1, "gather_mla_cache: token_to_seq rank-1 [max_tokens]");
  const int64_t batch = block_table.shape[0];
  const int64_t head_dim = src_cache.shape[2];
  VT_CHECK(batch > 0 && head_dim > 0, "gather_mla_cache: batch/head_dim must be > 0");
  VT_CHECK(dst.shape[1] == head_dim,
           "gather_mla_cache: dst entry width must equal the cache entry width");
  VT_CHECK(cu_seq_lens.shape[0] == batch + 1,
           "gather_mla_cache: cu_seq_lens must be [batch+1]");
  VT_CHECK(num_tokens >= 0, "gather_mla_cache: num_tokens must be >= 0");
  VT_CHECK(dst.shape[0] >= num_tokens, "gather_mla_cache: dst must hold num_tokens rows");
  VT_CHECK(token_to_seq.shape[0] >= num_tokens,
           "gather_mla_cache: token_to_seq must hold num_tokens entries");
  VT_CHECK(src_cache.shape[1] > 0, "gather_mla_cache: block_size must be > 0");
  VT_CHECK(block_table.dtype == DType::kI32 && cu_seq_lens.dtype == DType::kI32 &&
               token_to_seq.dtype == DType::kI32,
           "gather_mla_cache: block_table/cu_seq_lens/token_to_seq must be i32");
  // The auto path only — the fp8 dequant branch (cache_kernels.cu:1039-1047) is
  // out of campaign scope, exactly as the W3 cache write refuses fp8_ds_mla.
  VT_CHECK(IsFloat(dst.dtype) && src_cache.dtype == dst.dtype,
           "gather_mla_cache: dst/src_cache must share one float dtype (the auto path; "
           "the fp8 dequant branch is out of scope)");
  VT_CHECK(dst.stride[1] == 1 && src_cache.stride[2] == 1,
           "gather_mla_cache: dst/src_cache innermost stride must be 1");
  VT_CHECK(block_table.stride[1] == 1 && cu_seq_lens.IsContiguous() &&
               token_to_seq.IsContiguous(),
           "gather_mla_cache: block_table rows / cu_seq_lens / token_to_seq must be contiguous");
  VT_CHECK(dst.device == q.device && src_cache.device == q.device &&
               block_table.device == q.device && cu_seq_lens.device == q.device &&
               token_to_seq.device == q.device,
           "gather_mla_cache: device mismatch");
  if (seq_starts != nullptr) {
    VT_CHECK(seq_starts->rank == 1 && seq_starts->shape[0] == batch,
             "gather_mla_cache: seq_starts must be rank-1 [batch]");
    VT_CHECK(seq_starts->dtype == DType::kI32, "gather_mla_cache: seq_starts must be i32");
    VT_CHECK(seq_starts->IsContiguous(), "gather_mla_cache: seq_starts must be contiguous");
    VT_CHECK(seq_starts->device == q.device, "gather_mla_cache: seq_starts device mismatch");
  }
  reinterpret_cast<GatherMlaCacheFn>(GetOp(OpId::kGatherMlaCache, q.device.type))(
      q, dst, src_cache, block_table, cu_seq_lens, token_to_seq, seq_starts, num_tokens);
}

void MergeAttnStates(Queue& q, Tensor& output, Tensor* output_lse, const Tensor& prefix_output,
                     const Tensor& prefix_lse, const Tensor& suffix_output,
                     const Tensor& suffix_lse, int64_t prefill_tokens_with_context) {
  VT_CHECK(output.rank == 3 && prefix_output.rank == 3 && suffix_output.rank == 3,
           "merge_attn_states: output/prefix_output/suffix_output must be rank-3 "
           "[num_tokens, num_heads, head_size]");
  VT_CHECK(prefix_lse.rank == 2 && suffix_lse.rank == 2,
           "merge_attn_states: prefix_lse/suffix_lse must be rank-2 [num_heads, num_tokens]");
  const int64_t num_tokens = output.shape[0];
  const int64_t num_heads = output.shape[1];
  const int64_t head_size = output.shape[2];
  VT_CHECK(num_heads > 0 && head_size > 0,
           "merge_attn_states: num_heads/head_size must be > 0");
  VT_CHECK(prefix_output.shape[0] == num_tokens && prefix_output.shape[1] == num_heads &&
               prefix_output.shape[2] == head_size,
           "merge_attn_states: prefix_output shape must match output");
  VT_CHECK(suffix_output.shape[0] == num_tokens && suffix_output.shape[1] == num_heads &&
               suffix_output.shape[2] == head_size,
           "merge_attn_states: suffix_output shape must match output");
  VT_CHECK(prefix_lse.shape[0] == num_heads && prefix_lse.shape[1] == num_tokens &&
               suffix_lse.shape[0] == num_heads && suffix_lse.shape[1] == num_tokens,
           "merge_attn_states: prefix_lse/suffix_lse must be [num_heads, num_tokens]");
  VT_CHECK(prefix_lse.dtype == DType::kF32 && suffix_lse.dtype == DType::kF32,
           "merge_attn_states: prefix_lse/suffix_lse must be f32");
  VT_CHECK(IsFloat(output.dtype) && prefix_output.dtype == output.dtype &&
               suffix_output.dtype == output.dtype,
           "merge_attn_states: output/prefix_output/suffix_output must share one float dtype "
           "(the USE_FP8_OUTPUT branch is out of scope)");
  VT_CHECK(output.stride[2] == 1 && prefix_output.stride[2] == 1 &&
               suffix_output.stride[2] == 1 && prefix_lse.stride[1] == 1 &&
               suffix_lse.stride[1] == 1,
           "merge_attn_states: innermost strides must be 1");
  VT_CHECK(output.device == q.device && prefix_output.device == q.device &&
               prefix_lse.device == q.device && suffix_output.device == q.device &&
               suffix_lse.device == q.device,
           "merge_attn_states: device mismatch");
  if (output_lse != nullptr) {
    VT_CHECK(output_lse->rank == 2 && output_lse->shape[0] == num_heads &&
                 output_lse->shape[1] == num_tokens,
             "merge_attn_states: output_lse must be [num_heads, num_tokens]");
    VT_CHECK(output_lse->dtype == DType::kF32, "merge_attn_states: output_lse must be f32");
    VT_CHECK(output_lse->stride[1] == 1,
             "merge_attn_states: output_lse innermost stride must be 1");
    VT_CHECK(output_lse->device == q.device, "merge_attn_states: output_lse device mismatch");
  }
  reinterpret_cast<MergeAttnStatesFn>(GetOp(OpId::kMergeAttnStates, q.device.type))(
      q, output, output_lse, prefix_output, prefix_lse, suffix_output, suffix_lse,
      prefill_tokens_with_context);
}

namespace detail {
// W10 repair (#1865): the classified-arrival counter behind
// PagedAttnSpecClassifiedCount(). Function-local static so the count exists
// exactly once per process regardless of link order.
std::atomic<uint64_t>& PagedAttnSpecClassified() {
  static std::atomic<uint64_t> n{0};
  return n;
}
}  // namespace detail

void PagedAttention(Queue& q, Tensor& out, const Tensor& query, const Tensor& k_cache,
                    const Tensor& v_cache, const Tensor& block_table, const Tensor& seq_lens,
                    const Tensor& query_start_loc, const PagedAttentionArgs& args) {
  VT_CHECK(query.rank == 3 && out.rank == 3,
           "paged_attention: query/out rank-3 [num_actual_tokens,num_q_heads,head_size]");
  VT_CHECK(k_cache.rank == 4 && v_cache.rank == 4,
           "paged_attention: k_cache/v_cache rank-4 "
           "[num_blocks,block_size,num_kv_heads,head_size]");
  const int64_t num_tokens = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t num_kv_heads = k_cache.shape[2], head_size_k = k_cache.shape[3];
  // MiMoV2: V head dim can differ from K head dim (v_head_dim=128 vs
  // head_dim=192). The query's d == K head dim (QK dot-product width); V is
  // read from v_cache.shape[3].
  const int64_t head_size_v = v_cache.shape[3];
  const int64_t head_size = head_size_k;  // legacy alias used below
  VT_CHECK(out.shape[0] == num_tokens && out.shape[1] == hq && out.shape[2] == head_size_v,
           "paged_attention: out must match [num_tokens, num_q_heads, head_size_v]");
  VT_CHECK(d == head_size, "paged_attention: query head_size must match the K cache head_size");
  VT_CHECK(v_cache.shape[0] == k_cache.shape[0] && v_cache.shape[1] == k_cache.shape[1] &&
               v_cache.shape[2] == num_kv_heads,
           "paged_attention: k_cache and v_cache must share shape except head_size_v");
  VT_CHECK(hq >= 1 && num_kv_heads >= 1 && hq % num_kv_heads == 0,
           "paged_attention: num_q_heads must be a positive multiple of num_kv_heads (GQA)");
  VT_CHECK(args.scale > 0.0f, "paged_attention: scale must be set (> 0), e.g. head_size^-0.5");
  if (args.window_size.has_value()) {
    VT_CHECK(args.window_size->left >= 0 && args.window_size->right >= 0,
             "paged_attention: window_size left/right must both be >= 0");
  }
  VT_CHECK(IsFloat(query.dtype) && IsOutFloat(out.dtype),
           "paged_attention: float query, f32/bf16 out");
  // The KV cache may be a DIFFERENT float dtype than the query (Phase-1 bf16 KV
  // cache: f32 query · bf16 cache — the kernel converts bf16 cache reads to f32
  // and accumulates in f32). Require only that K and V share one float dtype.
  // fp8 KV (args.kv_cache_dtype != kAuto): the cache is 1-byte fp8 (kI8) and the
  // kernel dequantizes each read as Dequant(fp8) * k_scale|v_scale (KV-FP8 W1).
  if (args.kv_cache_dtype == Fp8KVCacheDataType::kAuto) {
    VT_CHECK(IsFloat(k_cache.dtype) && k_cache.dtype == v_cache.dtype,
             "paged_attention: k_cache/v_cache must share one float dtype");
  } else {
    VT_CHECK(args.kv_cache_dtype == Fp8KVCacheDataType::kFp8E4M3,
             "paged_attention: only the fp8_e4m3 KV read is implemented "
             "(the fp8_e5m2 read is a named later brick, spec W5)");
    VT_CHECK(k_cache.dtype == DType::kI8 && v_cache.dtype == DType::kI8,
             "paged_attention: fp8 KV read requires 1-byte fp8 cache (DType::kI8)");
    VT_CHECK(args.k_scale > 0.0f && args.v_scale > 0.0f,
             "paged_attention: fp8 KV read requires k_scale/v_scale > 0");
    // WHICH BACKENDS HAVE AN fp8 READ. Unlike the fp8 STORE — a separate OpId
    // that only the CPU, CUDA, and ROCm backends register, so an unimplemented
    // backend refuses by name inside GetOp — the fp8 read rides ADDITIVE fields
    // on PagedAttentionArgs of an op that kMETAL and kROCM already register for
    // the float path (metal_ops.mm, rocm_ops.hip). Nothing in the provider
    // table can tell those two apart, so without this list an fp8 cache would
    // reach a kernel that reads the same bytes as floats and returns silent
    // garbage. AGENTS.md: refuse an unimplemented arm with a message that names
    // the missing part. CPU landed in W1, CUDA in W2, ROCm in W6; Metal is owed.
    VT_CHECK(q.device.type == DeviceType::kCPU || q.device.type == DeviceType::kCUDA ||
                 q.device.type == DeviceType::kROCM,
             "paged_attention: the fp8 KV read is implemented on CPU (KV-FP8 W1), "
             "CUDA (KV-FP8 W2), and ROCm (KV-FP8 W6) only; this backend has no "
             "fp8 dequant on the cache read and would read the fp8 bytes as its "
             "float dtype");
  }
  // metadata: block_table [num_reqs, max_blocks] i32, seq_lens [num_reqs] i32,
  // query_start_loc [num_reqs+1] i32.
  VT_CHECK(seq_lens.rank == 1 && seq_lens.dtype == DType::kI32,
           "paged_attention: seq_lens must be i32 [num_reqs]");
  const int64_t num_reqs = seq_lens.shape[0];
  VT_CHECK(num_reqs >= 1, "paged_attention: num_reqs must be >= 1");
  VT_CHECK(block_table.rank == 2 && block_table.shape[0] == num_reqs &&
               block_table.dtype == DType::kI32,
           "paged_attention: block_table must be i32 [num_reqs, max_blocks]");
  VT_CHECK(query_start_loc.rank == 1 && query_start_loc.shape[0] == num_reqs + 1 &&
               query_start_loc.dtype == DType::kI32,
           "paged_attention: query_start_loc must be i32 [num_reqs+1]");
  // query/out contiguous; seq_lens/query_start_loc contiguous. The cache and
  // block_table are read via strides (the cache is the strided NHD unbind slice),
  // but the per-token page must be head-contiguous (elem stride 1, head stride
  // head_size) — same guarantee reshape_and_cache relies on.
  VT_CHECK(query.IsContiguous() && out.IsContiguous() && seq_lens.IsContiguous() &&
               query_start_loc.IsContiguous(),
           "paged_attention: query/out/seq_lens/query_start_loc must be contiguous");
  VT_CHECK(k_cache.stride[3] == 1 && v_cache.stride[3] == 1,
           "paged_attention: k_cache/v_cache innermost (head_size) stride must be 1");
  VT_CHECK(k_cache.stride[2] == head_size_k && v_cache.stride[2] == head_size_v,
           "paged_attention: k_cache/v_cache page must be head-contiguous "
           "(stride[2] == head_size) — the NHD unbind-slice layout");
  VT_CHECK(query.device == q.device && out.device == q.device && k_cache.device == q.device &&
               v_cache.device == q.device && block_table.device == q.device &&
               seq_lens.device == q.device && query_start_loc.device == q.device,
           "paged_attention: device mismatch (query/out/cache/block_table/seq_lens/"
           "query_start_loc/queue)");
  // W10 repair (#1865): count a spec-CLASSIFIED arrival (shape-consistent
  // classification present) at the ONE wrapper every backend's dispatch sits
  // behind, so the runner→model→args threading is observable on a CPU box.
  if (PagedAttnUniformSpecShape(num_tokens, num_reqs, args.uniform_spec_query_len))
    detail::PagedAttnSpecClassified().fetch_add(1, std::memory_order_relaxed);
  reinterpret_cast<PagedAttentionFn>(GetOp(OpId::kPagedAttention, q.device.type))(
      q, out, query, k_cache, v_cache, block_table, seq_lens, query_start_loc, args);
}

uint64_t PagedAttnSpecClassifiedCount() {
  return detail::PagedAttnSpecClassified().load(std::memory_order_relaxed);
}
void ResetPagedAttnSpecClassifiedCount() {
  detail::PagedAttnSpecClassified().store(0, std::memory_order_relaxed);
}

namespace {
// Shared checks for the sampling ops: logits [num_reqs, vocab] f32, contiguous,
// on the queue device. Returns num_reqs for downstream per-row-metadata checks.
int64_t CheckSamplingLogits(const Queue& q, const Tensor& logits, const char* name) {
  VT_CHECK(logits.rank == 2, std::string(name) + ": logits must be rank-2 [num_reqs, vocab]");
  VT_CHECK(logits.dtype == DType::kF32, std::string(name) + ": logits must be f32");
  VT_CHECK(logits.IsContiguous(), std::string(name) + ": logits must be contiguous");
  VT_CHECK(logits.device == q.device, std::string(name) + ": logits device must match queue");
  return logits.shape[0];
}
}  // namespace

void ApplyTemperature(Queue& q, Tensor& logits, const Tensor& temp, bool all_random) {
  const int64_t n = CheckSamplingLogits(q, logits, "apply_temperature");
  VT_CHECK(temp.rank == 1 && temp.shape[0] == n && temp.dtype == DType::kF32 &&
               temp.IsContiguous() && temp.device == q.device,
           "apply_temperature: temp must be f32 [num_reqs] contiguous on the queue device");
  reinterpret_cast<ApplyTemperatureFn>(GetOp(OpId::kApplyTemperature, q.device.type))(
      q, logits, temp, all_random);
}

void GreedyArgmax(Queue& q, Tensor& token_ids, const Tensor& logits) {
  const int64_t n = CheckSamplingLogits(q, logits, "greedy_argmax");
  VT_CHECK(token_ids.rank == 1 && token_ids.shape[0] == n && token_ids.dtype == DType::kI64 &&
               token_ids.IsContiguous() && token_ids.device == q.device,
           "greedy_argmax: token_ids must be i64 [num_reqs] contiguous on the queue device");
  reinterpret_cast<GreedyArgmaxFn>(GetOp(OpId::kGreedyArgmax, q.device.type))(q, token_ids,
                                                                              logits);
}

void ApplyTopKTopP(Queue& q, Tensor& logits, const Tensor* k, const Tensor* p) {
  const int64_t n = CheckSamplingLogits(q, logits, "apply_top_k_top_p");
  // Both None => no-op (upstream apply_top_k_top_p returns logits unchanged).
  if (k == nullptr && p == nullptr) return;
  if (k != nullptr) {
    VT_CHECK(k->rank == 1 && k->shape[0] == n && k->dtype == DType::kI32 && k->IsContiguous() &&
                 k->device == q.device,
             "apply_top_k_top_p: k must be i32 [num_reqs] contiguous on the queue device");
  }
  if (p != nullptr) {
    VT_CHECK(p->rank == 1 && p->shape[0] == n && p->dtype == DType::kF32 && p->IsContiguous() &&
                 p->device == q.device,
             "apply_top_k_top_p: p must be f32 [num_reqs] contiguous on the queue device");
  }
  reinterpret_cast<ApplyTopKTopPFn>(GetOp(OpId::kApplyTopKTopP, q.device.type))(q, logits, k, p);
}

void ComputeProbs(Queue& q, Tensor& probs, const Tensor& logits) {
  const int64_t n = CheckSamplingLogits(q, logits, "compute_probs");
  VT_CHECK(probs.rank == 2 && probs.shape[0] == n && probs.shape[1] == logits.shape[1] &&
               probs.dtype == DType::kF32 && probs.IsContiguous() && probs.device == q.device,
           "compute_probs: probs must be f32 [num_reqs, vocab] contiguous matching logits");
  reinterpret_cast<ComputeProbsFn>(GetOp(OpId::kComputeProbs, q.device.type))(q, probs, logits);
}

void ComputeLogprobs(Queue& q, Tensor& logprobs, const Tensor& logits) {
  const int64_t n = CheckSamplingLogits(q, logits, "compute_logprobs");
  VT_CHECK(logprobs.rank == 2 && logprobs.shape[0] == n && logprobs.shape[1] == logits.shape[1] &&
               logprobs.dtype == DType::kF32 && logprobs.IsContiguous() &&
               logprobs.device == q.device,
           "compute_logprobs: logprobs must be f32 [num_reqs, vocab] contiguous matching logits");
  reinterpret_cast<ComputeLogprobsFn>(GetOp(OpId::kComputeLogprobs, q.device.type))(q, logprobs,
                                                                                    logits);
}

void RandomSample(Queue& q, Tensor& token_ids, const Tensor& probs, const Tensor& seeds) {
  VT_CHECK(probs.rank == 2, "random_sample: probs must be rank-2 [num_reqs, vocab]");
  VT_CHECK(probs.dtype == DType::kF32, "random_sample: probs must be f32");
  VT_CHECK(probs.IsContiguous(), "random_sample: probs must be contiguous");
  VT_CHECK(probs.device == q.device, "random_sample: probs device must match queue");
  const int64_t n = probs.shape[0];
  VT_CHECK(token_ids.rank == 1 && token_ids.shape[0] == n && token_ids.dtype == DType::kI64 &&
               token_ids.IsContiguous() && token_ids.device == q.device,
           "random_sample: token_ids must be i64 [num_reqs] contiguous on the queue device");
  VT_CHECK(seeds.rank == 1 && seeds.shape[0] == n && seeds.dtype == DType::kI64 &&
               seeds.IsContiguous() && seeds.device == q.device,
           "random_sample: seeds must be i64 [num_reqs] contiguous on the queue device");
  reinterpret_cast<RandomSampleFn>(GetOp(OpId::kRandomSample, q.device.type))(q, token_ids, probs,
                                                                              seeds);
}

void GreedyRejectionSample(Queue& q, Tensor& sampled, Tensor& num_sampled, Tensor& target_argmax,
                           const Tensor& logits, const Tensor& draft_sampled,
                           const Tensor& cu_num_logits) {
  const int64_t num_logits = CheckSamplingLogits(q, logits, "greedy_rejection_sample");
  VT_CHECK(cu_num_logits.rank == 1 && cu_num_logits.shape[0] >= 1 &&
               cu_num_logits.dtype == DType::kI32 && cu_num_logits.IsContiguous() &&
               cu_num_logits.device == q.device,
           "greedy_rejection_sample: cu_num_logits must be i32 [num_reqs+1] contiguous on the "
           "queue device");
  const int64_t num_reqs = cu_num_logits.shape[0] - 1;
  VT_CHECK(draft_sampled.rank == 1 && draft_sampled.shape[0] == num_logits &&
               draft_sampled.dtype == DType::kI32 && draft_sampled.IsContiguous() &&
               draft_sampled.device == q.device,
           "greedy_rejection_sample: draft_sampled must be i32 [num_logits] contiguous on the "
           "queue device");
  VT_CHECK(sampled.rank == 2 && sampled.shape[0] == num_reqs && sampled.dtype == DType::kI32 &&
               sampled.IsContiguous() && sampled.device == q.device,
           "greedy_rejection_sample: sampled must be i32 [num_reqs, max_num_logits] contiguous on "
           "the queue device");
  VT_CHECK(num_sampled.rank == 1 && num_sampled.shape[0] == num_reqs &&
               num_sampled.dtype == DType::kI32 && num_sampled.IsContiguous() &&
               num_sampled.device == q.device,
           "greedy_rejection_sample: num_sampled must be i32 [num_reqs] contiguous on the queue "
           "device");
  // The per-row argmax scratch is the CALLER's buffer, for the lifetime reason
  // spelled out beside the declaration (SPEC-DFLASH2 A2-2, #2802). It is checked
  // exactly like the outputs, because it is one: the argmax kernel writes it and
  // the accept kernel reads it, both after this call returns on a discrete
  // backend.
  VT_CHECK(target_argmax.rank == 1 && target_argmax.shape[0] == num_logits &&
               target_argmax.dtype == DType::kI32 && target_argmax.IsContiguous() &&
               target_argmax.device == q.device,
           "greedy_rejection_sample: target_argmax must be i32 [num_logits] contiguous on the "
           "queue device");
  if (num_reqs == 0) return;
  reinterpret_cast<GreedyRejectionSampleFn>(GetOp(OpId::kGreedyRejectionSample, q.device.type))(
      q, sampled, num_sampled, target_argmax, logits, draft_sampled, cu_num_logits);
}

namespace {
// [num_reqs, vocab] tensor of a required dtype, contiguous, on the queue device.
void CheckSamplingMatrix(const Queue& q, const Tensor& t, int64_t n, int64_t v, DType dt,
                         const char* name, const char* what) {
  VT_CHECK(t.rank == 2 && t.shape[0] == n && t.shape[1] == v && t.dtype == dt &&
               t.IsContiguous() && t.device == q.device,
           std::string(name) + ": " + what + " must be [num_reqs, vocab] of the expected dtype, "
                                             "contiguous on the queue device");
}
// [num_reqs] f32 vector.
void CheckSamplingVec(const Queue& q, const Tensor& t, int64_t n, const char* name,
                      const char* what) {
  VT_CHECK(t.rank == 1 && t.shape[0] == n && t.dtype == DType::kF32 && t.IsContiguous() &&
               t.device == q.device,
           std::string(name) + ": " + what + " must be f32 [num_reqs] contiguous on the device");
}
}  // namespace

void ApplyPenalties(Queue& q, Tensor& logits, const Tensor& prompt_mask,
                    const Tensor& output_bin_counts, const Tensor& output_mask,
                    const Tensor& frequency_penalties, const Tensor& presence_penalties,
                    const Tensor& repetition_penalties) {
  const int64_t n = CheckSamplingLogits(q, logits, "apply_penalties");
  const int64_t v = logits.shape[1];
  CheckSamplingMatrix(q, prompt_mask, n, v, DType::kI8, "apply_penalties", "prompt_mask");
  CheckSamplingMatrix(q, output_mask, n, v, DType::kI8, "apply_penalties", "output_mask");
  CheckSamplingMatrix(q, output_bin_counts, n, v, DType::kI32, "apply_penalties",
                      "output_bin_counts");
  CheckSamplingVec(q, frequency_penalties, n, "apply_penalties", "frequency_penalties");
  CheckSamplingVec(q, presence_penalties, n, "apply_penalties", "presence_penalties");
  CheckSamplingVec(q, repetition_penalties, n, "apply_penalties", "repetition_penalties");
  reinterpret_cast<ApplyPenaltiesFn>(GetOp(OpId::kApplyPenalties, q.device.type))(
      q, logits, prompt_mask, output_bin_counts, output_mask, frequency_penalties,
      presence_penalties, repetition_penalties);
}

void ApplyMinP(Queue& q, Tensor& logits, const Tensor& min_p) {
  const int64_t n = CheckSamplingLogits(q, logits, "apply_min_p");
  CheckSamplingVec(q, min_p, n, "apply_min_p", "min_p");
  reinterpret_cast<ApplyMinPFn>(GetOp(OpId::kApplyMinP, q.device.type))(q, logits, min_p);
}

namespace {
// The (rows, cols) pair-list shape shared by ApplyLogitBias / ApplyTokenMask.
void CheckPairList(const Queue& q, const Tensor& rows, const Tensor& cols, const char* name) {
  VT_CHECK(rows.rank == 1 && cols.rank == 1 && rows.shape[0] == cols.shape[0],
           std::string(name) + ": rows and cols must be equal-length rank-1 [m]");
  VT_CHECK(rows.dtype == DType::kI32 && cols.dtype == DType::kI32,
           std::string(name) + ": rows/cols must be i32");
  VT_CHECK(rows.IsContiguous() && cols.IsContiguous() && rows.device == q.device &&
               cols.device == q.device,
           std::string(name) + ": rows/cols must be contiguous on the queue device");
}
}  // namespace

void ApplyLogitBias(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols,
                    const Tensor& biases) {
  CheckSamplingLogits(q, logits, "apply_logit_bias");
  CheckPairList(q, rows, cols, "apply_logit_bias");
  VT_CHECK(biases.rank == 1 && biases.shape[0] == rows.shape[0] && biases.dtype == DType::kF32 &&
               biases.IsContiguous() && biases.device == q.device,
           "apply_logit_bias: biases must be f32 [m] contiguous on the queue device");
  reinterpret_cast<ApplyLogitBiasFn>(GetOp(OpId::kApplyLogitBias, q.device.type))(q, logits, rows,
                                                                                  cols, biases);
}

void ApplyTokenMask(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols) {
  CheckSamplingLogits(q, logits, "apply_token_mask");
  CheckPairList(q, rows, cols, "apply_token_mask");
  reinterpret_cast<ApplyTokenMaskFn>(GetOp(OpId::kApplyTokenMask, q.device.type))(q, logits, rows,
                                                                                  cols);
}

void ApplyAllowedTokenIds(Queue& q, Tensor& logits, const Tensor& mask) {
  const int64_t n = CheckSamplingLogits(q, logits, "apply_allowed_token_ids");
  CheckSamplingMatrix(q, mask, n, logits.shape[1], DType::kI8, "apply_allowed_token_ids", "mask");
  reinterpret_cast<ApplyAllowedTokenIdsFn>(GetOp(OpId::kApplyAllowedTokenIds, q.device.type))(
      q, logits, mask);
}

// --- Qwen3.6 elementwise "glue" ops (M0.9 forward). ------------------------

void CastBf16(Queue& q, Tensor& out, const Tensor& in) {
  VT_CHECK(out.dtype == DType::kBF16, "cast_bf16: out must be bf16");
  VT_CHECK(in.dtype == DType::kF32, "cast_bf16: in must be f32");
  VT_CHECK(out.Numel() == in.Numel(), "cast_bf16: out/in must have the same element count");
  // Input may be a torch.split-style packed view (the merged QKV path): each
  // logical row is dense while the row stride spans the parent Q+K+V tensor.
  // Symmetric with CastF32; strided inputs only arise on CUDA (the merge is
  // CUDA-only), where the kernel honors the row stride.
  int64_t inner = 1;
  bool inner_contiguous = true;
  for (int dim = in.rank - 1; dim >= 1; --dim) {
    inner_contiguous = inner_contiguous && in.stride[dim] == inner;
    inner *= in.shape[dim];
  }
  inner_contiguous = inner_contiguous && in.rank >= 1 && in.stride[0] >= inner;
  VT_CHECK(out.IsContiguous() && inner_contiguous,
           "cast_bf16: out must be contiguous and input rows inner-contiguous");
  VT_CHECK(out.device == q.device && in.device == q.device,
           "cast_bf16: device mismatch (out/in/queue)");
  reinterpret_cast<CastBf16Fn>(GetOp(OpId::kCastBf16, q.device.type))(q, out, in);
}

void CastF16(Queue& q, Tensor& out, const Tensor& in) {
  VT_CHECK(out.dtype == DType::kF16, "cast_f16: out must be f16");
  VT_CHECK(in.dtype == DType::kF32 || in.dtype == DType::kBF16,
           "cast_f16: in must be f32 or bf16 (an f16 source is refused rather than copied)");
  VT_CHECK(out.Numel() == in.Numel(), "cast_f16: out/in must have the same element count");
  // Same packed-view tolerance as CastBf16: each logical row is dense while the
  // row stride may span a parent tensor (the merged-QKV shape).
  int64_t inner = 1;
  bool inner_contiguous = true;
  for (int dim = in.rank - 1; dim >= 1; --dim) {
    inner_contiguous = inner_contiguous && in.stride[dim] == inner;
    inner *= in.shape[dim];
  }
  inner_contiguous = inner_contiguous && in.rank >= 1 && in.stride[0] >= inner;
  VT_CHECK(out.IsContiguous() && inner_contiguous,
           "cast_f16: out must be contiguous and input rows inner-contiguous");
  VT_CHECK(out.device == q.device && in.device == q.device,
           "cast_f16: device mismatch (out/in/queue)");
  reinterpret_cast<CastF16Fn>(GetOp(OpId::kCastF16, q.device.type))(q, out, in);
}

void PermuteVHeads(Queue& q, Tensor& out, const Tensor& in,
                   int64_t T, int64_t num_k, int64_t rpk, int64_t dv) {
  VT_CHECK(out.dtype == DType::kBF16 && in.dtype == DType::kBF16,
           "permute_v_heads: both tensors must be bf16");
  VT_CHECK(out.Numel() == in.Numel(),
           "permute_v_heads: out/in must have the same element count");
  VT_CHECK(out.device == q.device && in.device == q.device,
           "permute_v_heads: device mismatch");
  reinterpret_cast<PermuteVHeadsFn>(GetOp(OpId::kPermuteVHeads, q.device.type))(
      q, out, in, T, num_k, rpk, dv);
}

void CastF32(Queue& q, Tensor& out, const Tensor& in) {
  VT_CHECK(out.dtype == DType::kF32, "cast_f32: out must be f32");
  VT_CHECK(in.dtype == DType::kBF16, "cast_f32: in must be bf16");
  VT_CHECK(out.Numel() == in.Numel(), "cast_f32: out/in must have the same element count");
  int64_t inner = 1;
  bool inner_contiguous = true;
  for (int dim = in.rank - 1; dim >= 1; --dim) {
    inner_contiguous = inner_contiguous && in.stride[dim] == inner;
    inner *= in.shape[dim];
  }
  inner_contiguous = inner_contiguous && in.rank >= 1 &&
                     in.stride[0] >= inner;
  VT_CHECK(out.IsContiguous() && inner_contiguous,
           "cast_f32: out must be contiguous and input rows inner-contiguous");
  VT_CHECK(out.device == q.device && in.device == q.device,
           "cast_f32: device mismatch (out/in/queue)");
  reinterpret_cast<CastF32Fn>(GetOp(OpId::kCastF32, q.device.type))(q, out, in);
}

void MulColVecF32(Queue& q, Tensor& x, const Tensor& col) {
  // PERF-FP8-ALPHA-FOLD / #417: x may be bf16 as well as f32. The op is named
  // for the ARITHMETIC (an f32 multiply by an f32 column vector), not for the
  // store width -- `col` stays f32 on both arms and the multiply is f32 on both
  // arms; only x's store rounds. Narrowing x halves the bytes this
  // read-modify-write moves, which is its entire measured cost (it runs at 77%
  // of the GB10's peak bandwidth on the merged FP8 GDN in_proj output). The
  // narrowing is NOT value-neutral, so it is opt-in at the model layer
  // (VT_GDN_FP8_IN_BF16, default OFF) and never chosen here.
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16,
           "mul_col_vec_f32: x must be f32 or bf16");
  VT_CHECK(col.dtype == DType::kF32, "mul_col_vec_f32: col must be f32");
  VT_CHECK(x.rank == 2, "mul_col_vec_f32: x must be rank-2 [M,N]");
  VT_CHECK(col.rank == 1, "mul_col_vec_f32: col must be rank-1 [N]");
  VT_CHECK(col.shape[0] == x.shape[1],
           "mul_col_vec_f32: col length must equal x columns");
  VT_CHECK(x.stride[1] == 1 && x.stride[0] >= x.shape[1],
           "mul_col_vec_f32: x rows must be inner-contiguous");
  VT_CHECK(col.IsContiguous(), "mul_col_vec_f32: col must be contiguous");
  VT_CHECK(x.device == q.device && col.device == q.device,
           "mul_col_vec_f32: device mismatch (x/col/queue)");
  reinterpret_cast<MulColVecF32Fn>(GetOp(OpId::kMulColVecF32, q.device.type))(q, x, col);
}

void AttnGateSplit(Queue& q, Tensor& q_out, Tensor& gate_out, const Tensor& qgate) {
  VT_CHECK(q_out.rank == 3 && gate_out.rank == 3, "attn_gate_split: q_out/gate_out rank-3 [T,Hq,Dh]");
  VT_CHECK(qgate.rank == 2, "attn_gate_split: qgate rank-2 [T, Hq*2*Dh]");
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  VT_CHECK(gate_out.shape[0] == t && gate_out.shape[1] == hq && gate_out.shape[2] == dh,
           "attn_gate_split: gate_out must match q_out [T,Hq,Dh]");
  VT_CHECK(qgate.shape[0] == t && qgate.shape[1] == hq * 2 * dh,
           "attn_gate_split: qgate must be [T, Hq*2*Dh]");
  // q_out is f32 OR bf16. bf16 is what vLLM's `torch.chunk` produces for a
  // checkpoint whose model dtype is bf16 (qwen4_exp: qwen3_next.py:430 then :437
  // hands `q` to q_norm unwidened), and an f32 destination there would move twice
  // the bytes to hold values a bf16 qgate already rounded. f32 is Qwen3.5's, whose
  // qk-norm/RoPE chain reads f32.
  //
  // gate_out is f32 and only f32: `vt::SigmoidGateBf16` — the only consumer of
  // this operand — states an f32 gate on each of its four backends. Narrowing it
  // is #2488's remaining half and is NOT admitted here, because a refusal that
  // widened ahead of its consumer would move the failure from this call to a
  // deeper one.
  VT_CHECK(q_out.dtype == DType::kF32 || q_out.dtype == DType::kBF16,
           "attn_gate_split: q_out must be f32 or bf16");
  VT_CHECK(gate_out.dtype == DType::kF32,
           "attn_gate_split: gate_out must be f32 (vt::SigmoidGateBf16 takes an f32 gate)");
  VT_CHECK(qgate.dtype == DType::kF32 || qgate.dtype == DType::kBF16,
           "attn_gate_split: qgate must be f32 or bf16 (bf16 = VT_BF16_GEMM_OUT q_proj)");
  VT_CHECK(q_out.IsContiguous() && gate_out.IsContiguous() && qgate.IsContiguous(),
           "attn_gate_split: contiguous required");
  VT_CHECK(q_out.device == q.device && gate_out.device == q.device && qgate.device == q.device,
           "attn_gate_split: device mismatch (q_out/gate_out/qgate/queue)");
  reinterpret_cast<AttnGateSplitFn>(GetOp(OpId::kAttnGateSplit, q.device.type))(q, q_out, gate_out,
                                                                                qgate);
}

void SigmoidGateBf16(Queue& q, Tensor& out, const Tensor& attn, const Tensor& gate) {
  VT_CHECK(out.dtype == DType::kBF16, "sigmoid_gate_bf16: out must be bf16");
  // attn may be bf16 (the FA-2 prefill path outputs bf16 attention); it is
  // upcast to f32 inside the kernel (exact), so bf16-attn is bit-identical to
  // f32-attn holding the same bf16-representable values. The gate stays f32
  // (sigmoid input must not be rounded).
  VT_CHECK((attn.dtype == DType::kF32 || attn.dtype == DType::kBF16) &&
               gate.dtype == DType::kF32,
           "sigmoid_gate_bf16: attn must be f32/bf16, gate f32");
  VT_CHECK(out.Numel() == attn.Numel() && out.Numel() == gate.Numel(),
           "sigmoid_gate_bf16: out/attn/gate must have the same element count");
  VT_CHECK(out.IsContiguous() && attn.IsContiguous() && gate.IsContiguous(),
           "sigmoid_gate_bf16: contiguous required");
  VT_CHECK(out.device == q.device && attn.device == q.device && gate.device == q.device,
           "sigmoid_gate_bf16: device mismatch (out/attn/gate/queue)");
  reinterpret_cast<SigmoidGateBf16Fn>(GetOp(OpId::kSigmoidGateBf16, q.device.type))(q, out, attn,
                                                                                    gate);
}

void GdnGBeta(Queue& q, Tensor& g_out, Tensor& beta_out, const Tensor& araw, const Tensor& braw,
              const Tensor& a_log, const Tensor& dt_bias) {
  VT_CHECK(g_out.rank == 2 && beta_out.rank == 2 && araw.rank == 2 && braw.rank == 2,
           "gdn_g_beta: g_out/beta_out/araw/braw rank-2 [T,Hv]");
  const int64_t t = g_out.shape[0], hv = g_out.shape[1];
  VT_CHECK(beta_out.shape[0] == t && beta_out.shape[1] == hv && araw.shape[0] == t &&
               araw.shape[1] == hv && braw.shape[0] == t && braw.shape[1] == hv,
           "gdn_g_beta: g_out/beta_out/araw/braw must all be [T,Hv]");
  VT_CHECK(a_log.rank == 1 && a_log.shape[0] == hv && dt_bias.rank == 1 && dt_bias.shape[0] == hv,
           "gdn_g_beta: a_log/dt_bias must be [Hv]");
  VT_CHECK(g_out.dtype == DType::kF32 && beta_out.dtype == DType::kF32 &&
               (araw.dtype == DType::kF32 || araw.dtype == DType::kBF16) &&
               braw.dtype == araw.dtype && a_log.dtype == DType::kF32 &&
               dt_bias.dtype == DType::kF32,
           "gdn_g_beta: g/beta/a_log/dt_bias f32; a/b must share f32 or bf16");
  VT_CHECK(g_out.IsContiguous() && beta_out.IsContiguous() &&
               araw.stride[1] == 1 && braw.stride[1] == 1 &&
               araw.stride[0] >= hv && braw.stride[0] >= hv &&
               a_log.IsContiguous() && dt_bias.IsContiguous(),
           "gdn_g_beta: outputs/vectors contiguous; a/b inner-contiguous row views required");
  VT_CHECK(g_out.device == q.device && beta_out.device == q.device && araw.device == q.device &&
               braw.device == q.device && a_log.device == q.device && dt_bias.device == q.device,
           "gdn_g_beta: device mismatch (g_out/beta_out/araw/braw/a_log/dt_bias/queue)");
  reinterpret_cast<GdnGBetaFn>(GetOp(OpId::kGdnGBeta, q.device.type))(q, g_out, beta_out, araw,
                                                                      braw, a_log, dt_bias);
}

void GdnConvSplit(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& conv) {
  VT_CHECK(conv.rank == 2, "gdn_conv_split: conv rank-2 [T, conv_dim]");
  const int64_t t = conv.shape[0];
  VT_CHECK(t > 0, "gdn_conv_split: T must be > 0");
  VT_CHECK(q_out.Numel() % t == 0 && k_out.Numel() % t == 0 && v_out.Numel() % t == 0,
           "gdn_conv_split: q_out/k_out/v_out element counts must be divisible by T");
  const int64_t key_dim = q_out.Numel() / t, value_dim = v_out.Numel() / t;
  VT_CHECK(k_out.Numel() / t == key_dim, "gdn_conv_split: q_out and k_out must share key_dim");
  VT_CHECK(conv.shape[1] == 2 * key_dim + value_dim,
           "gdn_conv_split: conv_dim must be 2*key_dim + value_dim");
  // q/k/v out may be f32 OR bf16 (coupled GDN bf16 path); conv may be f32 OR bf16
  // (input-side bf16 GDN path, VT_GDN_IN_BF16) — the kernel upcasts via Load().
  VT_CHECK((q_out.dtype == DType::kF32 || q_out.dtype == DType::kBF16) &&
               k_out.dtype == q_out.dtype && v_out.dtype == q_out.dtype &&
               (conv.dtype == DType::kF32 || conv.dtype == DType::kBF16),
           "gdn_conv_split: q/k/v out f32 or bf16 (same dtype), conv f32 or bf16");
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() && v_out.IsContiguous() &&
               conv.IsContiguous(),
           "gdn_conv_split: contiguous required");
  VT_CHECK(q_out.device == q.device && k_out.device == q.device && v_out.device == q.device &&
               conv.device == q.device,
           "gdn_conv_split: device mismatch (q_out/k_out/v_out/conv/queue)");
  reinterpret_cast<GdnConvSplitFn>(GetOp(OpId::kGdnConvSplit, q.device.type))(q, q_out, k_out,
                                                                              v_out, conv);
}

void QkvSplit(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& qkv) {
  VT_CHECK(qkv.rank == 2, "qkv_split: qkv rank-2 [T, q_dim+k_dim+v_dim]");
  const int64_t t = qkv.shape[0];
  VT_CHECK(t > 0, "qkv_split: T must be > 0");
  VT_CHECK(q_out.Numel() % t == 0 && k_out.Numel() % t == 0 && v_out.Numel() % t == 0,
           "qkv_split: q_out/k_out/v_out element counts must be divisible by T");
  const int64_t q_dim = q_out.Numel() / t, k_dim = k_out.Numel() / t, v_dim = v_out.Numel() / t;
  VT_CHECK(qkv.shape[1] == q_dim + k_dim + v_dim,
           "qkv_split: qkv inner dim must be q_dim + k_dim + v_dim");
  VT_CHECK((q_out.dtype == DType::kF32 || q_out.dtype == DType::kBF16) &&
               k_out.dtype == q_out.dtype && v_out.dtype == q_out.dtype &&
               qkv.dtype == q_out.dtype,
           "qkv_split: q/k/v out and qkv must share dtype (f32 or bf16)");
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() && v_out.IsContiguous() &&
               qkv.IsContiguous(),
           "qkv_split: contiguous required");
  VT_CHECK(q_out.device == q.device && k_out.device == q.device && v_out.device == q.device &&
               qkv.device == q.device,
           "qkv_split: device mismatch (q_out/k_out/v_out/qkv/queue)");
  reinterpret_cast<QkvSplitFn>(GetOp(OpId::kQkvSplit, q.device.type))(q, q_out, k_out, v_out, qkv);
}

void GdnPostConv(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out, Tensor& g_out,
                 Tensor& beta_out, const Tensor& conv, const Tensor& araw, const Tensor& braw,
                 const Tensor& a_log, const Tensor& dt_bias, const L2NormArgs& args) {
  // Fusion of GdnConvSplit + L2Norm(q) + L2Norm(k) + GdnGBeta; validation is the
  // union of those four ops (same shape/dtype/device/contiguity contracts).
  VT_CHECK(conv.rank == 2, "gdn_post_conv: conv rank-2 [T, conv_dim]");
  VT_CHECK(q_out.rank == 3 && k_out.rank == 3 && v_out.rank == 3,
           "gdn_post_conv: q_out/k_out/v_out rank-3 [T,H,D]");
  const int64_t t = conv.shape[0];
  VT_CHECK(t > 0, "gdn_post_conv: T must be > 0");
  VT_CHECK(q_out.shape[0] == t && k_out.shape[0] == t && v_out.shape[0] == t,
           "gdn_post_conv: q_out/k_out/v_out leading dim must be T");
  const int64_t hk = q_out.shape[1], dk = q_out.shape[2];
  const int64_t hv = v_out.shape[1], dv = v_out.shape[2];
  VT_CHECK(k_out.shape[1] == hk && k_out.shape[2] == dk,
           "gdn_post_conv: q_out and k_out must share [T,Hk,Dk]");
  const int64_t key_dim = hk * dk, value_dim = hv * dv;
  VT_CHECK(conv.shape[1] == 2 * key_dim + value_dim,
           "gdn_post_conv: conv_dim must be 2*key_dim + value_dim");
  VT_CHECK(g_out.rank == 2 && beta_out.rank == 2 && araw.rank == 2 && braw.rank == 2,
           "gdn_post_conv: g_out/beta_out/araw/braw rank-2 [T,Hv]");
  VT_CHECK(g_out.shape[0] == t && g_out.shape[1] == hv && beta_out.shape[0] == t &&
               beta_out.shape[1] == hv && araw.shape[0] == t && araw.shape[1] == hv &&
               braw.shape[0] == t && braw.shape[1] == hv,
           "gdn_post_conv: g_out/beta_out/araw/braw must all be [T,Hv]");
  VT_CHECK(a_log.rank == 1 && a_log.shape[0] == hv && dt_bias.rank == 1 && dt_bias.shape[0] == hv,
           "gdn_post_conv: a_log/dt_bias must be [Hv]");
  // q/k/v activations may be f32 OR bf16 (coupled GDN bf16 path, VT_GDN_BF16):
  // bf16 feeds the WMMA chunk-scan as native bf16 fragments. All three must share
  // one dtype (the scan requires q.dtype==k.dtype==v.dtype). g/beta and the
  // araw/braw may share f32 OR bf16 dtype. vLLM's packed BA projection emits
  // model-dtype bf16 and slicing it produces row-strided inner-contiguous
  // views; kernels upcast those loads while g/beta remain f32. conv may be f32
  // OR bf16 (input-side bf16 GDN path, VT_GDN_IN_BF16).
  VT_CHECK((q_out.dtype == DType::kF32 || q_out.dtype == DType::kBF16) &&
               k_out.dtype == q_out.dtype && v_out.dtype == q_out.dtype,
           "gdn_post_conv: q_out/k_out/v_out must be f32 or bf16, same dtype");
  VT_CHECK(conv.dtype == DType::kF32 || conv.dtype == DType::kBF16,
           "gdn_post_conv: conv must be f32 or bf16");
  VT_CHECK(g_out.dtype == DType::kF32 && beta_out.dtype == DType::kF32 &&
               (araw.dtype == DType::kF32 || araw.dtype == DType::kBF16) &&
               braw.dtype == araw.dtype && a_log.dtype == DType::kF32 &&
               dt_bias.dtype == DType::kF32,
           "gdn_post_conv: g/beta/a_log/dt_bias f32; a/b must share f32 or bf16");
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() && v_out.IsContiguous() &&
               g_out.IsContiguous() && beta_out.IsContiguous() && conv.IsContiguous() &&
               araw.stride[1] == 1 && braw.stride[1] == 1 &&
               araw.stride[0] >= hv && braw.stride[0] >= hv &&
               a_log.IsContiguous() && dt_bias.IsContiguous(),
           "gdn_post_conv: outputs/conv/vectors contiguous; a/b inner-contiguous row views required");
  VT_CHECK(q_out.device == q.device && k_out.device == q.device && v_out.device == q.device &&
               g_out.device == q.device && beta_out.device == q.device && conv.device == q.device &&
               araw.device == q.device && braw.device == q.device && a_log.device == q.device &&
               dt_bias.device == q.device,
           "gdn_post_conv: device mismatch");
  reinterpret_cast<GdnPostConvFn>(GetOp(OpId::kGdnPostConv, q.device.type))(
      q, q_out, k_out, v_out, g_out, beta_out, conv, araw, braw, a_log, dt_bias, args);
}

void SharedExpertGate(Queue& q, Tensor& out, const Tensor& sd, const Tensor& gl) {
  VT_CHECK(out.rank == 2, "shared_expert_gate: out rank-2 [T,H]");
  const int64_t t = out.shape[0], h = out.shape[1];
  VT_CHECK(out.dtype == DType::kBF16, "shared_expert_gate: out must be bf16");
  // sd may be bf16 (VT_SHARED_DOWN_BF16): widening in-kernel is exact and the
  // store is bf16 either way, so both forms are bit-identical.
  VT_CHECK((sd.dtype == DType::kF32 || sd.dtype == DType::kBF16) && gl.dtype == DType::kF32,
           "shared_expert_gate: sd must be f32/bf16 and gl f32");
  VT_CHECK(sd.Numel() == t * h, "shared_expert_gate: sd must have T*H elements matching out");
  VT_CHECK(gl.Numel() == t, "shared_expert_gate: gl must have T elements (one gate per token)");
  VT_CHECK(out.IsContiguous() && sd.IsContiguous() && gl.IsContiguous(),
           "shared_expert_gate: contiguous required");
  VT_CHECK(out.device == q.device && sd.device == q.device && gl.device == q.device,
           "shared_expert_gate: device mismatch (out/sd/gl/queue)");
  reinterpret_cast<SharedExpertGateFn>(GetOp(OpId::kSharedExpertGate, q.device.type))(q, out, sd,
                                                                                      gl);
}


// ─── EXL3 device kernels — MODEL-DSV4-EXL3 W2a / W2b ─────────────────────────
//
// Ported from exllamav3 @ 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT). The
// contracts, the parity tiers and the dtype decision are in include/vt/ops.h and
// in `.agents/specs/model-dsv4-exl3.md` `## W2 design`.
//
// Every refusal below NAMES the op and the thing it could not represent. That is
// the standing quant-arm rule: an arm we have not built refuses by name rather
// than being discovered later as a wrong number.

void Exl3HadR128(Queue& q, Tensor& out, const Tensor& in, const Exl3HadArgs& args) {
  VT_CHECK(in.rank == 2 && out.rank == 2, "exl3_had_r_128: in/out must be rank-2 [rows, cols]");
  VT_CHECK(in.shape[0] == out.shape[0] && in.shape[1] == out.shape[1],
           "exl3_had_r_128: in/out shapes must match");
  VT_CHECK(in.dtype == out.dtype, "exl3_had_r_128: in/out must share a dtype");
  VT_CHECK(in.dtype == DType::kF16 || in.dtype == DType::kF32,
           "exl3_had_r_128: dtype must be f16 or f32 (upstream hadamard.cu:172 "
           "refuses every other); got " + std::string(Name(in.dtype)));
  const int64_t cols = in.shape[1];
  // TORCH_CHECK_DIV(input, 1, 128) (hadamard.cu:102). The transform IS 128-wide;
  // there is no partial block to fall back to.
  VT_CHECK(cols % 128 == 0,
           "exl3_had_r_128: the row length must be a multiple of 128 (the transform "
           "is blockwise Hadamard-128); got " + std::to_string(cols));
  VT_CHECK(in.IsContiguous() && out.IsContiguous(), "exl3_had_r_128: contiguous required");
  VT_CHECK(in.device == q.device && out.device == q.device, "exl3_had_r_128: device mismatch");
  // hadamard.cu:112-172 instantiates <pre_scale, post_scale> and never with both
  // true, so "both" is not a mode this port can express.
  VT_CHECK(!(args.pre_scale != nullptr && args.post_scale != nullptr),
           "exl3_had_r_128: at most one of pre_scale/post_scale (upstream instantiates "
           "the kernel as one or the other, never both)");
  const Tensor* sc = args.pre_scale != nullptr ? args.pre_scale : args.post_scale;
  if (sc != nullptr) {
    VT_CHECK(sc->dtype == DType::kF16,
             "exl3_had_r_128: the scale vector is fp16 (upstream's `const half*` in both "
             "the half and the float kernels); got " + std::string(Name(sc->dtype)));
    VT_CHECK(sc->Numel() == cols,
             "exl3_had_r_128: the scale vector must have one entry per column; got " +
                 std::to_string(sc->Numel()) + " for " + std::to_string(cols));
    VT_CHECK(sc->device == q.device, "exl3_had_r_128: scale device mismatch");
  }
  reinterpret_cast<Exl3HadR128Fn>(GetOp(OpId::kExl3HadR128, q.device.type))(q, out, in, args);
}

void Exl3Gemm(Queue& q, Tensor& c, const Tensor& a, const Tensor& trellis, const Tensor& suh,
              const Tensor& svh, Tensor& a_had, const Exl3GemmArgs& args) {
  VT_CHECK(args.bits >= 1 && args.bits <= 8,
           "exl3_gemm: bits must be in [1, 8]; got " + std::to_string(args.bits));
  // ALL THREE CODEBOOKS UPSTREAM DEFINES: 0 (the original QTIP 3INST), 1 (MCG)
  // and 2 (`mul1`). `decode_3inst<cb>` (`codebook.cuh:56-90`) has arms for those
  // three and falls off the end for any other value, so a fourth is not an arm
  // this tree has yet to port -- it does not exist.
  //
  // The earlier narrowing to MCG here was WRONG rather than merely conservative,
  // and it was written when the only checkpoint in view was the SparkInfer
  // DeepSeek-V4 artifact, which ships an `mcg` marker. `LinearEXL3` derives the
  // codebook from tensor PRESENCE (`exl3.py:74-77`), so every stock
  // `turboderp/*-exl3` artifact -- shipping neither `mcg` nor `mul1` -- is cb 0,
  // and cb 0 is therefore the COMMON case rather than an exotic one.
  //
  // Cb 2 joined it for the same reason one step later:
  // `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` ships a `mul1` marker on every quantized
  // linear, so refusing cb 2 here refused that checkpoint whole
  // (QUANT-EXL3-MUL1, #2495).
  //
  // THIS IS THE SEAM CHECK, NOT THE ARM CHECK, and the difference matters. The
  // host arm decodes every (bits, codebook) pair; the DEVICE arm instantiates a
  // named few and refuses the rest at its own launcher (`cuda_exl3.cu`), which
  // is where an uninstantiated pair is caught. Widening this one does not widen
  // that one.
  VT_CHECK(args.codebook >= 0 && args.codebook <= 2,
           "exl3_gemm: codebook must be 0 (3INST), 1 (mcg) or 2 (mul1); codebook " +
               std::to_string(args.codebook) +
               " is not an arm upstream defines at all (QUANT-EXL3-MUL1, #2495)");
  // HALF-INTEGER rate (BACKEND-ROCM frac rates): K = KA + 0.5, `bits` = KA, and
  // the mul1 codebook is the only one the oracle defines the fractional rates
  // for (`frac.cu`). Anything else refuses here rather than decoding a frac
  // stream through an integer codebook.
  VT_CHECK(!args.half || (args.codebook == 2 && args.bits <= 7),
           "exl3_gemm: a half-integer rate (K+0.5) requires the mul1 codebook (2) and "
           "KA in [1, 7]; got codebook " + std::to_string(args.codebook) + " bits " +
               std::to_string(args.bits));
  VT_CHECK(a.rank == 2 && c.rank == 2, "exl3_gemm: A and C must be rank-2");
  // `ldmatrix.sync.aligned.m8n8.x4.shared.b16` + `mma...f16.f16` read fp16
  // fragments (ptx.cuh:52-74,203-212), so A has no dtype freedom at all.
  // ROCm accepts bf16 too: its input-Hadamard arm folds the caller's CastF16
  // into the load bit-identically (DF32ToF16(bf16->f32)), so the residual
  // stream feeds the transform directly. Every other backend needs the f16
  // staging the caller performs.
  const bool a_ok = a.dtype == DType::kF16 ||
                    (q.device.type == DeviceType::kROCM && a.dtype == DType::kBF16);
  VT_CHECK(a_ok,
           "exl3_gemm: A must be f16 (the tensor-core fragments are fp16); got " +
               std::string(Name(a.dtype)));
  VT_CHECK(a_had.dtype == DType::kF16,
           "exl3_gemm: A_had must be f16 (it holds the transformed A); got " +
               std::string(Name(a_had.dtype)));
  // bf16 C is ROCm-only for the same reason: its output transform folds the
  // trailing CastBf16 into the store (__float2bfloat16 of the f32 result).
  const bool c_ok = c.dtype == DType::kF16 || c.dtype == DType::kF32 ||
                    (q.device.type == DeviceType::kROCM && c.dtype == DType::kBF16);
  VT_CHECK(c_ok,
           "exl3_gemm: C must be f16 (the default, exl3.py:72) or f32 (upstream's "
           "c_fp32 arm, exl3_gemm.cu:134); got " + std::string(Name(c.dtype)));
  VT_CHECK(trellis.dtype == DType::kI8,
           "exl3_gemm: the trellis travels as opaque i8 BYTES; got " +
               std::string(Name(trellis.dtype)));
  VT_CHECK(trellis.rank == 3, "exl3_gemm: trellis must be rank-3 [k/16, n/16, 32*bits]");
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];
  const int64_t n = c.shape[1];
  VT_CHECK(c.shape[0] == m, "exl3_gemm: C rows must equal A rows");
  VT_CHECK(a_had.shape[0] == m && a_had.shape[1] == k, "exl3_gemm: A_had must be shaped like A");
  // Both sides were Hadamard-128 transformed at quantization time
  // (exl3_lib/quantize.py:15), so both must be multiples of 128 for the runtime
  // transform to be defined; upstream's own limits are k % 16 and n % 128
  // (exl3_gemm.cu:34-35) and the tighter k rule comes from the transform, not
  // the GEMM.
  VT_CHECK(k % 128 == 0 && n % 128 == 0,
           "exl3_gemm: k and n must be multiples of 128 (both sides carry a blockwise "
           "Hadamard-128); got k=" + std::to_string(k) + " n=" + std::to_string(n));
  VT_CHECK(trellis.shape[0] == k / 16 && trellis.shape[1] == n / 16 &&
               trellis.shape[2] ==
                   32 * static_cast<int64_t>(args.bits) + (args.half ? 16 : 0),
           "exl3_gemm: trellis shape must be [k/16, n/16, 32*bits] bytes (or 32*KA+16 for "
           "a half-integer rate) for k=" +
               std::to_string(k) + " n=" + std::to_string(n) +
               " bits=" + std::to_string(args.bits) +
               (args.half ? " half" : ""));
  VT_CHECK(suh.dtype == DType::kF16 && svh.dtype == DType::kF16,
           "exl3_gemm: suh/svh are fp16 sign+scale vectors (exl3.py:20-91)");
  VT_CHECK(suh.Numel() == k, "exl3_gemm: suh must have k entries");
  VT_CHECK(svh.Numel() == n, "exl3_gemm: svh must have n entries");
  VT_CHECK(a.IsContiguous() && c.IsContiguous() && a_had.IsContiguous() &&
               trellis.IsContiguous() && suh.IsContiguous() && svh.IsContiguous(),
           "exl3_gemm: contiguous required (the kernels read A as contiguous rows, "
           "exl3.py:129-131)");
  VT_CHECK(a.device == q.device && c.device == q.device && a_had.device == q.device &&
               trellis.device == q.device && suh.device == q.device && svh.device == q.device,
           "exl3_gemm: device mismatch");
  reinterpret_cast<Exl3GemmFn>(GetOp(OpId::kExl3Gemm, q.device.type))(q, c, a, trellis, suh, svh,
                                                                     a_had, args);
}

// QUANT-EXL3 W6. The reconstruct+cuBLAS dispatch, parallel to Exl3Gemm above.
// Same validation, plus w_scratch: fp16 [k, min(n, 32768)] holding the
// reconstructed weight (MAX_RECONSTRUCT_SLICE_N, exl3.py:11).
//
// QUANT-EXL3 W7: `w_scratch == nullptr` is the persistent-scratch spelling, whose
// scratch the backend owns, so there is nothing of the caller's to validate.
namespace {
void CheckExl3ReconstructGemm(const Queue& q, const Tensor& c, const Tensor& a,
                              const Tensor& trellis, const Tensor& suh, const Tensor& svh,
                              const Tensor& a_had, const Tensor* w_scratch,
                              const Exl3GemmArgs& args) {
  VT_CHECK(args.bits >= 1 && args.bits <= 8,
           "exl3_reconstruct_gemm: bits must be in [1, 8]; got " + std::to_string(args.bits));
  VT_CHECK(args.codebook >= 0 && args.codebook <= 2,
           "exl3_reconstruct_gemm: codebook must be 0, 1 or 2; got " +
               std::to_string(args.codebook));
  VT_CHECK(a.rank == 2 && c.rank == 2, "exl3_reconstruct_gemm: A and C must be rank-2");
  VT_CHECK(a.dtype == DType::kF16,
           "exl3_reconstruct_gemm: A must be f16; got " + std::string(Name(a.dtype)));
  VT_CHECK(a_had.dtype == DType::kF16,
           "exl3_reconstruct_gemm: A_had must be f16; got " + std::string(Name(a_had.dtype)));
  VT_CHECK(c.dtype == DType::kF16 || c.dtype == DType::kF32,
           "exl3_reconstruct_gemm: C must be f16 or f32; got " + std::string(Name(c.dtype)));
  VT_CHECK(trellis.dtype == DType::kI8,
           "exl3_reconstruct_gemm: trellis must be i8; got " + std::string(Name(trellis.dtype)));
  VT_CHECK(trellis.rank == 3, "exl3_reconstruct_gemm: trellis must be rank-3");
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];
  const int64_t n = c.shape[1];
  VT_CHECK(c.shape[0] == m, "exl3_reconstruct_gemm: C rows must equal A rows");
  VT_CHECK(a_had.shape[0] == m && a_had.shape[1] == k,
           "exl3_reconstruct_gemm: A_had must be shaped like A");
  if (w_scratch != nullptr) {
    const int64_t w_cols = n <= 32768 ? n : 32768;
    VT_CHECK(w_scratch->dtype == DType::kF16,
             "exl3_reconstruct_gemm: w_scratch must be f16; got " +
                 std::string(Name(w_scratch->dtype)));
    VT_CHECK(w_scratch->rank == 2 && w_scratch->shape[0] == k && w_scratch->shape[1] == w_cols,
             "exl3_reconstruct_gemm: w_scratch must be [k, min(n, 32768)]");
    VT_CHECK(w_scratch->IsContiguous() && w_scratch->device == q.device,
             "exl3_reconstruct_gemm: w_scratch must be contiguous and on the queue's device");
  }
  // Upstream requires this on EVERY reconstruct path, not only the fused one.
  // The unfused path calls `reconstruct` -> `reconstruct_slice`, which checks
  // N % 128 (exllamav3_ext/quant/reconstruct.cu:121), and `had_r_128` on the
  // input (K) and on the output (N), which checks cols % 128
  // (exllamav3_ext/quant/hadamard.cu:102). The fused `reconstruct_had_slice`
  // checks both (reconstruct.cu:348-349). exl3.py:170-174 gates only the fused
  // choice on it because EXL3 tensors always satisfy it. `Exl3Gemm` above
  // enforces the same rule, so no shape it serves is refused here.
  VT_CHECK(k % 128 == 0 && n % 128 == 0,
           "exl3_reconstruct_gemm: k and n must be multiples of 128");
  VT_CHECK(trellis.shape[0] == k / 16 && trellis.shape[1] == n / 16 &&
               trellis.shape[2] == 32 * static_cast<int64_t>(args.bits),
           "exl3_reconstruct_gemm: trellis shape must be [k/16, n/16, 32*bits]");
  // No frac reconstruct arm anywhere (BACKEND-ROCM, spec `## Owed`): a
  // half-integer rate (K+0.5, mul1) reconstructs through the INTEGER-KA decode
  // here and would be silently wrong, so it refuses by name. The fused dot arm
  // and the Exl3GemmK fallback serve these tensors instead.
  VT_CHECK(!args.half,
           "exl3_reconstruct_gemm: half-integer rates (K+0.5, mul1) have no reconstruct "
           "arm; the trellis last dim " + std::to_string(trellis.shape[2]) +
               " bytes is a frac tile (32*KA+16). This is an unimplemented arm, not an "
               "invalid tensor (BACKEND-ROCM frac rates, ## Owed).");
  VT_CHECK(suh.dtype == DType::kF16 && svh.dtype == DType::kF16,
           "exl3_reconstruct_gemm: suh/svh must be fp16");
  VT_CHECK(suh.Numel() == k, "exl3_reconstruct_gemm: suh must have k entries");
  VT_CHECK(svh.Numel() == n, "exl3_reconstruct_gemm: svh must have n entries");
  VT_CHECK(a.IsContiguous() && c.IsContiguous() && a_had.IsContiguous() &&
               trellis.IsContiguous() &&
               suh.IsContiguous() && svh.IsContiguous(),
           "exl3_reconstruct_gemm: contiguous required");
  VT_CHECK(a.device == q.device && c.device == q.device && a_had.device == q.device &&
               trellis.device == q.device &&
               suh.device == q.device && svh.device == q.device,
           "exl3_reconstruct_gemm: device mismatch");
}
}  // namespace

void Exl3ReconstructGemm(Queue& q, Tensor& c, const Tensor& a, const Tensor& trellis,
                          const Tensor& suh, const Tensor& svh, Tensor& a_had,
                          Tensor& w_scratch, const Exl3GemmArgs& args) {
  // A caller-owned scratch must be a real buffer: an empty one would silently
  // select the backend's persistent scratch through the spelling that says the
  // caller supplies it.
  VT_CHECK(w_scratch.data != nullptr,
           "exl3_reconstruct_gemm: w_scratch is empty; call the overload without w_scratch "
           "for the backend's persistent reconstruct scratch");
  CheckExl3ReconstructGemm(q, c, a, trellis, suh, svh, a_had, &w_scratch, args);
  reinterpret_cast<Exl3ReconstructGemmFn>(GetOp(OpId::kExl3ReconstructGemm, q.device.type))(
      q, c, a, trellis, suh, svh, a_had, w_scratch, args);
}

void Exl3ReconstructGemm(Queue& q, Tensor& c, const Tensor& a, const Tensor& trellis,
                          const Tensor& suh, const Tensor& svh, Tensor& a_had,
                          const Exl3GemmArgs& args) {
  CheckExl3ReconstructGemm(q, c, a, trellis, suh, svh, a_had, nullptr, args);
  Tensor persistent;  // data == nullptr: the backend's persistent scratch
  reinterpret_cast<Exl3ReconstructGemmFn>(GetOp(OpId::kExl3ReconstructGemm, q.device.type))(
      q, c, a, trellis, suh, svh, a_had, persistent, args);
}

// ─── The fused MoE MLP — MODEL-DSV4-EXL3 W2d ─────────────────────────────────
//
// The checks are `exl3_moe.cu:145-201`, in upstream's own order, with the torch
// spellings replaced by VT_CHECK so both arms share one set of refusals rather
// than duplicating them per backend. Every refusal names the op and what it
// could not represent.
void Exl3MoeMlp(Queue& q, Tensor& output_state, const Tensor& hidden_state,
                const Exl3MoeExpertTables& tables, const Exl3MoeRouting& routing,
                const Exl3MoeTemps& temps, const Exl3MoeArgs& args) {
  // exl3_moe.cu:142-143: "Nothing for the fused kernel to do". Returning before
  // the checks is upstream's own order and matters: a caller with no active
  // expert may legitimately pass buffers it never sized.
  if (args.num_active == 0) return;

  VT_CHECK(hidden_state.rank == 2, "exl3_moe: hidden_state must be rank-2 [bsz, hidden]");
  VT_CHECK(hidden_state.dtype == DType::kF16,
           "exl3_moe: hidden_state must be f16 (the tensor-core fragments are fp16); got " +
               std::string(Name(hidden_state.dtype)));
  const int64_t bsz = hidden_state.shape[0];
  const int64_t hidden_dim = hidden_state.shape[1];

  // exl3_moe.cu:151-152. f32 is UPSTREAM's width for the accumulator, not a
  // widening: the epilogue atomicAdds one contribution per (token, active
  // expert) into it.
  VT_CHECK(output_state.dtype == DType::kF32,
           "exl3_moe: output_state must be f32 (upstream's own width, exl3_moe.cu:151 — the "
           "scatter-add accumulates one contribution per active expert into it); got " +
               std::string(Name(output_state.dtype)));
  VT_CHECK(output_state.rank == 2 && output_state.shape[0] == bsz &&
               output_state.shape[1] == hidden_dim,
           "exl3_moe: output_state must be shaped like hidden_state");

  VT_CHECK(routing.expert_count != nullptr && routing.token_sorted != nullptr &&
               routing.weight_sorted != nullptr,
           "exl3_moe: expert_count, token_sorted and weight_sorted are all required");
  VT_CHECK(routing.expert_count->dtype == DType::kI64 &&
               routing.token_sorted->dtype == DType::kI64,
           "exl3_moe: expert_count and token_sorted are i64 (exl3_moe.cu:154,158)");
  VT_CHECK(routing.weight_sorted->dtype == DType::kF16,
           "exl3_moe: weight_sorted is f16 (exl3_moe.cu:56)");
  VT_CHECK(routing.expert_count->rank == 1 && routing.expert_count->Numel() >= 2,
           "exl3_moe: expert_count is [num_experts + 1]");
  const int64_t num_experts = routing.expert_count->Numel() - 1;
  VT_CHECK(routing.token_sorted->Numel() == routing.weight_sorted->Numel(),
           "exl3_moe: token_sorted and weight_sorted must have the same length");
  VT_CHECK(bsz > 0 && routing.token_sorted->Numel() % bsz == 0,
           "exl3_moe: token_sorted must hold a whole number of assignments per token");

  VT_CHECK(temps.state_g != nullptr && temps.state_u != nullptr &&
               temps.intermediate_g != nullptr && temps.intermediate_u != nullptr,
           "exl3_moe: all four temp buffers are required");
  const Tensor* four[4] = {temps.state_g, temps.state_u, temps.intermediate_g,
                           temps.intermediate_u};
  for (const Tensor* tt : four) {
    VT_CHECK(tt->dtype == DType::kF16, "exl3_moe: the temp buffers are f16 (exl3_moe.cu:163-174)");
    VT_CHECK(tt->rank == 3,
             "exl3_moe: the temp buffers are [concurrency, max_tokens_per_expert, dim]");
    VT_CHECK(tt->IsContiguous(), "exl3_moe: the temp buffers must be contiguous");
    VT_CHECK(tt->device == q.device, "exl3_moe: temp buffer device mismatch");
  }
  VT_CHECK(temps.state_g->shape[2] == hidden_dim && temps.state_u->shape[2] == hidden_dim,
           "exl3_moe: the state buffers' last dim must be hidden_dim (exl3_moe.cu:166)");
  const int64_t intermediate_dim = temps.intermediate_g->shape[2];
  VT_CHECK(temps.intermediate_u->shape[2] == intermediate_dim,
           "exl3_moe: the intermediate buffers must agree on their last dim");
  const int64_t max_tokens_per_expert = temps.state_g->shape[1];
  const int64_t concurrency = temps.state_g->shape[0];
  for (const Tensor* tt : four)
    VT_CHECK(tt->shape[0] == concurrency && tt->shape[1] == max_tokens_per_expert,
             "exl3_moe: the four temp buffers must agree on concurrency and "
             "max_tokens_per_expert");
  VT_CHECK(concurrency >= 1 && max_tokens_per_expert >= 1,
           "exl3_moe: concurrency and max_tokens_per_expert must both be at least 1");

  // Both sides carry a blockwise Hadamard-128, the same rule Exl3Gemm states,
  // and the gather/scatter epilogues are 128-wide per warp
  // (exl3_moe_kernel.cuh:87,167,239).
  VT_CHECK(hidden_dim % 128 == 0 && intermediate_dim % 128 == 0,
           "exl3_moe: hidden and intermediate must be multiples of 128 (both sides carry a "
           "blockwise Hadamard-128); got hidden=" + std::to_string(hidden_dim) +
               " intermediate=" + std::to_string(intermediate_dim));

  // exl3_moe.cu:184-185. cb 0 (3INST) and cb 2 (mul1) exist upstream and are
  // NOT ported: this checkpoint is mcg, and an unported arm refuses by name.
  VT_CHECK(args.codebook == 1,
           "exl3_moe: only codebook 1 (mcg) is implemented; codebook " +
               std::to_string(args.codebook) +
               " is an upstream arm this row has not ported (MODEL-DSV4-EXL3)");
  const int bits[3] = {args.bits_gate, args.bits_up, args.bits_down};
  for (int b : bits)
    VT_CHECK(b >= 1 && b <= 8,
             "exl3_moe: every bit width must be in [1, 8]; got " + std::to_string(b));

  const Tensor* nine[9] = {tables.gate_trellis, tables.gate_suh,   tables.gate_svh,
                           tables.up_trellis,   tables.up_suh,     tables.up_svh,
                           tables.down_trellis, tables.down_suh,   tables.down_svh};
  for (const Tensor* tt : nine) {
    VT_CHECK(tt != nullptr, "exl3_moe: all nine per-expert pointer tables are required");
    VT_CHECK(tt->dtype == DType::kI64,
             "exl3_moe: a pointer table is an i64 array of per-expert addresses "
             "(exl3_moe.cu:118-126 passes tensors of void*); got " +
                 std::string(Name(tt->dtype)));
    VT_CHECK(tt->Numel() == num_experts,
             "exl3_moe: every pointer table must have one entry per expert (" +
                 std::to_string(num_experts) + "); got " + std::to_string(tt->Numel()));
    VT_CHECK(tt->IsContiguous(), "exl3_moe: the pointer tables must be contiguous");
    VT_CHECK(tt->device == q.device, "exl3_moe: pointer table device mismatch");
  }

  VT_CHECK(hidden_state.IsContiguous() && output_state.IsContiguous() &&
               routing.expert_count->IsContiguous() && routing.token_sorted->IsContiguous() &&
               routing.weight_sorted->IsContiguous(),
           "exl3_moe: contiguous required");
  VT_CHECK(hidden_state.device == q.device && output_state.device == q.device &&
               routing.expert_count->device == q.device &&
               routing.token_sorted->device == q.device &&
               routing.weight_sorted->device == q.device,
           "exl3_moe: device mismatch");

  reinterpret_cast<Exl3MoeMlpFn>(GetOp(OpId::kExl3MoeMlp, q.device.type))(
      q, output_state, hidden_state, tables, routing, temps, args);
}

}  // namespace vt
