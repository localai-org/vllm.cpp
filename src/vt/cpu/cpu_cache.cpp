// Ported from: vllm/csrc/libtorch_stable/cache_kernels.cu @ e24d1b24
//   (reshape_and_cache_flash — the "auto"/contiguous-heads write semantics
//    only; the NHD cache layout is FlashAttentionBackend::get_kv_cache_shape's
//    (num_blocks, 2, block_size, num_kv_heads, head_size), NOT the HND cpu_attn
//    layout — see .agents/discipline.md and the M1.6 Task-2 layout trap note).
#include <cstdint>
#include <cstring>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4_compressor.h"  // the W8 slice-1 packer
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vt::cpu {
namespace {

// Write each new per-token K/V into the paged NHD cache at its slot id. The
// "auto" cache path is a raw element copy (cache dtype == k/v dtype); we copy
// bytes so f32/f16/bf16 are all bit-exact (upstream KV_T == CACHE_T in auto).
//
// CRITICAL (M1.6 Task-2): the destination is indexed from the TENSOR STRIDES,
// not from k_cache.shape. get_kv_cache_shape hands us ONE (num_blocks, 2,
// block_size, H, D) allocation and K/V are its two dim-1 unbind slices, so the
// block stride is 2*bs*H*D (NOT bs*H*D) and K/V are NON-contiguous rank-4 views.
// This mirrors pinned csrc/libtorch_stable/cache_kernels.cu @ e24d1b24:
//   host reshape_and_cache_flash (~L797-801): key_stride = key.stride(0);
//     block_stride = key_cache.stride(0); page_stride = key_cache.stride(1);
//     head_stride = key_cache.stride(2);
//   kernel reshape_and_cache_flash_kernel (~L337-347): key_src = key +
//     token_idx*key_stride; key_dst = key_cache + block_idx*block_stride +
//     block_offset*page_stride; is_contiguous_heads = (head_stride == head_size).
// The wrapper guarantees head_stride == head_size && elem stride == 1 (the NHD
// unbind slice), so the per-token page is one dense run of n_elems inside the
// block — the is_contiguous_heads fast path, i.e. a single memcpy per token.
void ReshapeAndCacheKernel(Queue&, const Tensor& k, const Tensor& v, Tensor& k_cache,
                           Tensor& v_cache, const Tensor& slot_mapping) {
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];
  const int64_t head_size_k = k_cache.shape[3];
  const int64_t head_size_v = v_cache.shape[3];
  const int64_t n_elems_k = num_kv_heads * head_size_k;  // one token's K page (NHD)
  const int64_t n_elems_v = num_kv_heads * head_size_v;  // one token's V page (NHD)
  // Destination strides come from the tensors (unbind-slice aware), each cache
  // with ITS OWN strides. Source token stride comes from k/v.stride(0); the
  // per-token [H, D] payload is packed (input k/v are contiguous rows).
  const int64_t k_block_stride = k_cache.stride[0];
  const int64_t k_page_stride = k_cache.stride[1];
  const int64_t v_block_stride = v_cache.stride[0];
  const int64_t v_page_stride = v_cache.stride[1];
  const int64_t k_tok_stride = k.stride[0];
  const int64_t v_tok_stride = v.stride[0];
  const size_t elem = SizeOf(k.dtype);

  const int64_t* slots = slot_mapping.Ptr<int64_t>();
  const auto* ksrc = static_cast<const uint8_t*>(k.data);
  const auto* vsrc = static_cast<const uint8_t*>(v.data);
  auto* kdst = static_cast<uint8_t*>(k_cache.data);
  auto* vdst = static_cast<uint8_t*>(v_cache.data);
  const size_t bytes_k = static_cast<size_t>(n_elems_k) * elem;
  const size_t bytes_v = static_cast<size_t>(n_elems_v) * elem;

  for (int64_t t = 0; t < num_slots; ++t) {
    const int64_t slot = slots[t];
    if (slot < 0) continue;  // padded token → skip (upstream NOTE: slot can be -1)
    const int64_t block = slot / block_size;
    const int64_t offset = slot % block_size;
    const int64_t kdst_off = block * k_block_stride + offset * k_page_stride;  // elements
    const int64_t vdst_off = block * v_block_stride + offset * v_page_stride;
    const int64_t ksrc_off = t * k_tok_stride;
    const int64_t vsrc_off = t * v_tok_stride;
    std::memcpy(kdst + static_cast<size_t>(kdst_off) * elem,
                ksrc + static_cast<size_t>(ksrc_off) * elem, bytes_k);
    std::memcpy(vdst + static_cast<size_t>(vdst_off) * elem,
                vsrc + static_cast<size_t>(vsrc_off) * elem, bytes_v);
  }
}

// MLA cache write — the CPU REFERENCE for vt::ConcatAndCacheMla, ported 1:1 from
// vllm/csrc/libtorch_stable/cache_kernels.cu:401-442
// (`concat_and_cache_mla_kernel`) @ e24d1b24. Upstream is literally two strided
// `copy(...)` lambda calls into one destination entry:
//   copy(kv_c, kv_cache, kv_c_stride, block_stride, kv_lora_rank, 0);       (:440)
//   copy(k_pe, kv_cache, k_pe_stride, block_stride, pe_dim, kv_lora_rank);  (:441)
// with dst_idx = block_idx*block_stride + block_offset*entry_stride + i + offset
// (`:431-433`) — note upstream passes `block_stride` as the SOURCE-side
// dst_stride argument and never uses it, the real destination stride pair is
// (block_stride, entry_stride). We reproduce the destination arithmetic exactly.
//
// The "auto" path (kv_dt == kAuto, `:435-436`) is a raw element copy, so we copy
// BYTES: f32/f16/bf16 are all bit-exact. The fp8 `scaled_convert` branch
// (`:437-440`) is out of scope and is refused by the op wrapper's dtype check.
// This is a CONCATENATION, not two writes to two tensors: the latent occupies
// columns [0, kv_lora_rank) of the single 576-wide entry and the decoupled rope
// part occupies [kv_lora_rank, kv_lora_rank + pe_dim).
void ConcatAndCacheMlaKernel(Queue&, const Tensor& kv_c, const Tensor& k_pe, Tensor& kv_cache,
                             const Tensor& slot_mapping) {
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = kv_cache.shape[1];
  const int64_t kv_lora_rank = kv_c.shape[1];
  const int64_t pe_dim = k_pe.shape[1];
  // Destination strides from the TENSOR (upstream `:882-884`), so a strided
  // cache view works without a copy.
  const int64_t block_stride = kv_cache.stride[0];
  const int64_t entry_stride = kv_cache.stride[1];
  const int64_t kv_c_stride = kv_c.stride[0];
  const int64_t k_pe_stride = k_pe.stride[0];
  const size_t elem = SizeOf(kv_c.dtype);

  const int64_t* slots = slot_mapping.Ptr<int64_t>();
  const auto* csrc = static_cast<const uint8_t*>(kv_c.data);
  const auto* psrc = static_cast<const uint8_t*>(k_pe.data);
  auto* dst = static_cast<uint8_t*>(kv_cache.data);

  for (int64_t t = 0; t < num_slots; ++t) {
    const int64_t slot = slots[t];
    if (slot < 0) continue;  // padded token → skip (upstream `:419-422`)
    const int64_t block = slot / block_size;
    const int64_t offset = slot % block_size;
    const int64_t entry = block * block_stride + offset * entry_stride;  // elements
    std::memcpy(dst + static_cast<size_t>(entry) * elem,
                csrc + static_cast<size_t>(t * kv_c_stride) * elem,
                static_cast<size_t>(kv_lora_rank) * elem);
    std::memcpy(dst + static_cast<size_t>(entry + kv_lora_rank) * elem,
                psrc + static_cast<size_t>(t * k_pe_stride) * elem,
                static_cast<size_t>(pe_dim) * elem);
  }
}

// Load one source K/V element as f32 (the model dtype: f32/f16/bf16).
float LoadSrcF32(const Tensor& t, int64_t elem_off) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[elem_off];
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[elem_off]);
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[elem_off]);
    default: VT_CHECK(false, "reshape_and_cache_fp8 LoadSrcF32: unsupported source dtype");
      return 0.0f;
  }
}

// fp8 KV STORE — the fp8 sibling of ReshapeAndCacheKernel. Ported from the fp8
// branch of vllm reshape_and_cache_flash_kernel (cache_kernels.cu:314-401) +
// CopyWithScaleOp (:241-252): each element is stored as Quantize(hp / scale)
// (the fp8::scaled_convert scale convention, quant_utils.cuh:296-308). The cache
// pages are 1-byte fp8 (kI8), so the destination element size is 1; the strides
// are the same NHD unbind-slice arithmetic as the auto path. Per-tensor k/v
// scales (BaseKVCacheMethod, kv_cache.py:108-191); per-head is a later brick.
void ReshapeAndCacheFp8Kernel(Queue&, const Tensor& k, const Tensor& v, Tensor& k_cache,
                              Tensor& v_cache, const Tensor& slot_mapping,
                              Fp8KVCacheDataType kind, float k_scale, float v_scale) {
  VT_CHECK(kind == Fp8KVCacheDataType::kFp8E4M3,
           "reshape_and_cache_fp8: CPU kernel implements fp8_e4m3 only");
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];
  const int64_t head_size_k = k_cache.shape[3];
  const int64_t head_size_v = v_cache.shape[3];
  const int64_t n_elems_k = num_kv_heads * head_size_k;  // one token's K page (NHD)
  const int64_t n_elems_v = num_kv_heads * head_size_v;  // one token's V page (NHD)
  const int64_t k_block_stride = k_cache.stride[0];
  const int64_t k_page_stride = k_cache.stride[1];
  const int64_t v_block_stride = v_cache.stride[0];
  const int64_t v_page_stride = v_cache.stride[1];
  const int64_t k_tok_stride = k.stride[0];
  const int64_t v_tok_stride = v.stride[0];

  const int64_t* slots = slot_mapping.Ptr<int64_t>();
  auto* kdst = k_cache.Ptr<uint8_t>();  // fp8 bytes (kI8)
  auto* vdst = v_cache.Ptr<uint8_t>();

  for (int64_t t = 0; t < num_slots; ++t) {
    const int64_t slot = slots[t];
    if (slot < 0) continue;  // padded token → skip (upstream: slot can be -1)
    const int64_t block = slot / block_size;
    const int64_t offset = slot % block_size;
    const int64_t kdst_base = block * k_block_stride + offset * k_page_stride;  // bytes/elems
    const int64_t vdst_base = block * v_block_stride + offset * v_page_stride;
    const int64_t ksrc_base = t * k_tok_stride;
    const int64_t vsrc_base = t * v_tok_stride;
    for (int64_t e = 0; e < n_elems_k; ++e) {
      kdst[kdst_base + e] = StoreKvFp8E4M3(LoadSrcF32(k, ksrc_base + e), k_scale);
    }
    for (int64_t e = 0; e < n_elems_v; ++e) {
      vdst[vdst_base + e] = StoreKvFp8E4M3(LoadSrcF32(v, vsrc_base + e), v_scale);
    }
  }
}

// ── The fp8_ds_mla paged K cache (KV-DSV4-MULTICACHE W8, #2455) ─────────────
// These two are the CPU arms of `vt::ConcatAndCacheDsMla` and
// `vt::DequantAndGatherDsMla`, and they are deliberately THIN: the byte layout
// lives once, in the W8 slice-1 host packer
// (`vllm/model_executor/models/deepseek_v4_compressor.{h,cpp}`), which the CUDA
// kernels of slice 5 are the other port of. Writing the region arithmetic a
// second time here is exactly the drift the shared packer exists to prevent.

// One source row of `k` as f32. The encoder's first act is upstream's own
// fp32 -> bf16 round (`cache_utils.py:110-118`), so widening f16/bf16 to f32
// here changes no stored byte.
void LoadDsMlaRow(const Tensor& k, int64_t token, std::vector<float>* head) {
  const int64_t width = k.shape[1];
  const int64_t off = token * k.stride[0];
  head->resize(static_cast<size_t>(width));
  switch (k.dtype) {
    case DType::kF32:
      std::memcpy(head->data(), k.Ptr<float>() + off,
                  static_cast<size_t>(width) * sizeof(float));
      break;
    case DType::kF16:
      for (int64_t d = 0; d < width; ++d)
        (*head)[static_cast<size_t>(d)] = F16ToF32(k.Ptr<uint16_t>()[off + d]);
      break;
    case DType::kBF16:
      for (int64_t d = 0; d < width; ++d)
        (*head)[static_cast<size_t>(d)] = BF16ToF32(k.Ptr<uint16_t>()[off + d]);
      break;
    default:
      VT_CHECK(false, "concat_and_cache_ds_mla: unsupported k dtype");
  }
}

const vllm::deepseek_v4::Fp8DsMlaLayout& DsMlaTokenLayout() {
  // 448 / 64 / 64 — upstream's TOKEN_FP8_DIM, TOKEN_BF16_DIM and
  // QUANT_BLOCK_SIZE (`cache_utils.py:180-183`), named once in vt/ops.h.
  static const vllm::deepseek_v4::Fp8DsMlaLayout layout =
      vllm::deepseek_v4::MakeFp8DsMlaLayout(kFp8DsMlaNopeDim, kFp8DsMlaRopeDim,
                                            kFp8DsMlaQuantBlock);
  return layout;
}

// `quantize_and_insert_k_kernel` (`cache_utils.py:36-159`), one program per
// token (`:71`), serialised.
void ConcatAndCacheDsMlaKernel(Queue&, const Tensor& k, Tensor& kv_cache,
                               const Tensor& slot_mapping, int64_t block_size) {
  const vllm::deepseek_v4::Fp8DsMlaLayout& L = DsMlaTokenLayout();
  const vllm::deepseek_v4::Fp8DsMlaPageLayout page =
      vllm::deepseek_v4::MakeFp8DsMlaPageLayout(L, block_size);
  // Upstream reads the block stride from the TENSOR (`:189`, `k_cache.stride(0)`),
  // so a padded page row (37440 for the 37376-byte C4A block) works unchanged.
  const int64_t block_stride = kv_cache.stride[0];
  // Upstream's token count is slot_mapping's, never k's (`:186-188`).
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t* slots = slot_mapping.Ptr<int64_t>();
  auto* base = kv_cache.Ptr<uint8_t>();

  std::vector<float> head;
  for (int64_t t = 0; t < num_slots; ++t) {
    const int64_t slot = slots[t];
    // `if slot_idx == -1: return` (`:77-78`). Load-bearing, and not merely
    // duplicated by `Fp8DsMlaStoreToken`'s own negative-position skip: C++
    // truncating division sends slot == -block_size to (block -1, pos 0), which
    // the packer would happily write — one whole block BELOW the page.
    if (slot < 0) continue;
    const int64_t block_idx = slot / block_size;   // `:80`
    const int64_t pos_in_block = slot % block_size;  // `:81`
    LoadDsMlaRow(k, t, &head);
    vllm::deepseek_v4::Fp8DsMlaStoreToken(base + block_idx * block_stride, page,
                                          pos_in_block,
                                          vllm::deepseek_v4::Fp8DsMlaEncodeToken(head, L));
  }
}

// `_dequantize_and_gather_k_kernel` (`cache_utils.py:228-341`), the (batch,
// worker) grid collapsed to one serial walk per request.
void DequantAndGatherDsMlaKernel(Queue&, Tensor& out, const Tensor& kv_cache,
                                 const Tensor& seq_lens, const Tensor* gather_lens,
                                 const Tensor& block_table,
                                 const DequantAndGatherDsMlaArgs& args) {
  const vllm::deepseek_v4::Fp8DsMlaLayout& L = DsMlaTokenLayout();
  const vllm::deepseek_v4::Fp8DsMlaPageLayout page =
      vllm::deepseek_v4::MakeFp8DsMlaPageLayout(L, args.block_size);
  const int64_t num_reqs = out.shape[0];
  const int64_t max_blocks_per_seq = block_table.shape[1];
  const int64_t block_stride = kv_cache.stride[0];  // `:388`
  const auto* base = kv_cache.Ptr<uint8_t>();
  const int32_t* lens = seq_lens.Ptr<int32_t>();
  const int32_t* glens = gather_lens != nullptr ? gather_lens->Ptr<int32_t>() : nullptr;
  const int32_t* table = block_table.Ptr<int32_t>();

  for (int64_t b = 0; b < num_reqs; ++b) {
    const int64_t seq_len = lens[b];
    // `gather_lens_ptr is None` gathers the whole sequence (`:257-262`).
    const int64_t gather_len = glens != nullptr ? glens[b] : seq_len;
    const int64_t start_pos = seq_len - gather_len;  // `:264`
    VT_CHECK(gather_len >= 0 && start_pos >= 0,
             "dequant_and_gather_ds_mla: gather_len must be in [0, seq_len]");
    VT_CHECK(args.offset + gather_len <= out.shape[1],
             "dequant_and_gather_ds_mla: offset + gather_len overruns out");
    for (int64_t i = 0; i < gather_len; ++i) {
      const int64_t pos = start_pos + i;                        // `:268`
      const int64_t block_in_seq = pos / args.block_size;       // `:271`
      const int64_t pos_in_block = pos % args.block_size;       // `:272`
      VT_CHECK(block_in_seq < max_blocks_per_seq,
               "dequant_and_gather_ds_mla: seq_len exceeds the block table");
      // `block_table_ptr + batch_idx * max_blocks_per_seq` (`:275-276`).
      const int64_t physical_block = table[b * block_table.stride[0] + block_in_seq];
      VT_CHECK(physical_block >= 0 && physical_block < kv_cache.shape[0],
               "dequant_and_gather_ds_mla: block table entry out of range");
      // The read consumes n_quant_blocks = 7 (`:385`), one FEWER than the store
      // writes; `Fp8DsMlaLoadToken` drops the pad byte exactly as the gather does.
      const std::vector<float> latent = vllm::deepseek_v4::Fp8DsMlaDecodeToken(
          vllm::deepseek_v4::Fp8DsMlaLoadToken(base + physical_block * block_stride,
                                               page, pos_in_block),
          L);
      // `out_ptr + batch_idx*out_stride0 + (offset + i)*out_stride1` (`:295-296`).
      const int64_t row = b * out.stride[0] + (args.offset + i) * out.stride[1];
      if (out.dtype == DType::kF32) {
        std::memcpy(out.Ptr<float>() + row, latent.data(),
                    latent.size() * sizeof(float));
      } else {
        uint16_t* dst = out.Ptr<uint16_t>() + row;
        for (size_t d = 0; d < latent.size(); ++d) dst[d] = F32ToBF16(latent[d]);
      }
    }
  }
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kReshapeAndCache, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ReshapeAndCacheFn>(&ReshapeAndCacheKernel)));
    RegisterOp(
        OpId::kReshapeAndCacheFp8, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<ReshapeAndCacheFp8Fn>(&ReshapeAndCacheFp8Kernel)));
    RegisterOp(
        OpId::kConcatAndCacheMla, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<ConcatAndCacheMlaFn>(&ConcatAndCacheMlaKernel)));
    RegisterOp(OpId::kConcatAndCacheDsMla, DeviceType::kCPU,
               reinterpret_cast<void*>(
                   static_cast<ConcatAndCacheDsMlaFn>(&ConcatAndCacheDsMlaKernel)));
    RegisterOp(OpId::kDequantAndGatherDsMla, DeviceType::kCPU,
               reinterpret_cast<void*>(
                   static_cast<DequantAndGatherDsMlaFn>(&DequantAndGatherDsMlaKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
