// KV-FP8 W3 — the ONE place a model's attention block decides whether it is
// writing and reading a float KV cache or an fp8 one.
//
// Ported from: vllm/model_executor/layers/attention/attention.py @ 555967922 —
// `Attention.forward` hands `layer._k_scale`/`layer._v_scale` and the layer's
// `kv_cache_dtype` to the backend impl, which routes to
// `reshape_and_cache_flash`'s fp8 branch (`csrc/libtorch_stable/
// cache_kernels.cu:241-252,314-401`) and to the scaled read
// (`csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh:419-429`). Upstream's
// routing lives inside one `Attention` module that every model instantiates;
// ours lives here because our attention preambles are per-architecture free
// functions, and a decision copied into each of them is a decision that will
// drift.
//
// WHY A HELPER AND NOT AN IF AT EACH SITE. The store and the read must agree
// with the BLOCK SIZING about how wide a KV element is. `ApplyCacheDType`
// (`kv_cache_interface.cpp`) sizes the page at `vt::SizeOf(spec->dtype)`; these
// two functions are what spend it. If one of them takes the float path against a
// half-sized page, the writes land at the wrong offsets and the model emits
// wrong tokens — no bounds check fires, because the buffer is exactly as large
// as the sizing said. So both decisions read the SAME `PagedKvCache::fp8_kind`
// field, and neither re-derives it from the storage dtype.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_KV_CACHE_ROUTE_H_
#define VLLM_MODEL_EXECUTOR_MODELS_KV_CACHE_ROUTE_H_

#include "vllm/model_executor/models/qwen3_5.h"  // PagedKvCache
#include "vt/dtype.h"
#include "vt/fp8_kv.h"
#include "vt/ops.h"

namespace vllm {
namespace dense_attn {

// True when this layer's paged cache holds fp8 bytes rather than floats. The
// two facts are asserted to agree here, once, rather than at every caller: a
// 1-byte storage dtype with no fp8 interpretation, or an fp8 interpretation over
// a float page, is a mis-sized cache and never a mode.
inline bool IsFp8KvCache(const PagedKvCache& kv) {
  const bool byte_storage = kv.dtype == vt::DType::kI8;
  const bool fp8_kind = kv.fp8_kind != vt::Fp8KVCacheDataType::kAuto;
  VT_CHECK(byte_storage == fp8_kind,
           "kv cache route: the paged cache's storage dtype and its fp8 "
           "interpretation disagree — a 1-byte page must carry an fp8 kind and "
           "a float page must not (KV-FP8 W3)");
  return fp8_kind;
}

// The dtype the ATTENTION presents for this store. An fp8 store is read through
// the bf16 dense dequant scratch (`Fp8PagedToDenseBf16Kernel`), so it presents
// bf16 exactly as a bf16 store does; a model that decides its query/out dtype
// from the store's dtype must ask THIS, or an fp8 store silently falls off the
// bf16 attention lanes (FA-2 included) onto the per-read CUDA-core path.
inline bool KvCachePresentsBf16(const PagedKvCache& kv) {
  return kv.dtype == vt::DType::kBF16 || IsFp8KvCache(kv);
}

// The KV STORE. `k`/`v` are the model-dtype [T, Hkv, Dh] tensors the attention
// preamble produced; `k_cache`/`v_cache` are this layer's `KvSlice` views.
//
// On the float path this is `vt::ReshapeAndCache` verbatim — the same op, the
// same arguments, in the same order — so every existing caller is byte-identical
// once routed through here. On the fp8 path it is `vt::ReshapeAndCacheFp8`,
// which takes the FLOAT k/v directly and does the `Quantize(hp / scale)`
// conversion itself (`quant_utils.cuh:296-300`); there is no cast to the cache
// dtype to do first, and a caller that tried would be casting to `kI8`.
inline void WriteKvCache(vt::Queue& q, const PagedKvCache& kv,
                         const vt::Tensor& k, const vt::Tensor& v,
                         vt::Tensor& k_cache, vt::Tensor& v_cache,
                         const vt::Tensor& slot_mapping) {
  if (IsFp8KvCache(kv)) {
    vt::ReshapeAndCacheFp8(q, k, v, k_cache, v_cache, slot_mapping, kv.fp8_kind,
                           kv.k_scale, kv.v_scale);
    return;
  }
  vt::ReshapeAndCache(q, k, v, k_cache, v_cache, slot_mapping);
}

// The KV READ. Sets the three additive `PagedAttentionArgs` fields W1 added, so
// the paged kernel dequantizes each cache read as `Dequant(fp8) * k_scale|
// v_scale`. Inert on a float cache: the fields keep their defaults and the op
// takes the same branch it always did (`src/vt/ops.cpp` — the read guard refuses
// an fp8 cache read with `kv_cache_dtype == kAuto`, which is what makes a
// forgotten call here a loud failure rather than silent garbage).
inline void ApplyKvCacheQuant(vt::PagedAttentionArgs& args,
                              const PagedKvCache& kv) {
  if (!IsFp8KvCache(kv)) return;
  args.kv_cache_dtype = kv.fp8_kind;
  args.k_scale = kv.k_scale;
  args.v_scale = kv.v_scale;
}

}  // namespace dense_attn
}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_KV_CACHE_ROUTE_H_
