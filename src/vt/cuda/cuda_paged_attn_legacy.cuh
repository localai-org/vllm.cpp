// Legacy CUDA decode widths used by TinyLlama and SmolLM2.
// Pinned llama.cpp 62318cd57 fattn-vec.cuh:86-118,274 uses vector loads and
// warp reductions without Tensor Cores. Included inside vt::cuda's namespace.

bool LegacyRegisterDecodeEnabled() {
  const auto& caps = GetDeviceCaps();
  return caps.valid && caps.sm_major == 6;
}

template <>
__device__ inline void LoadRowN<2, __nv_bfloat16>(const __nv_bfloat16* p,
                                                int64_t base, int lane, float* r) {
  const uint32_t word = reinterpret_cast<const uint32_t*>(p + base)[lane];
  const auto* values = reinterpret_cast<const __nv_bfloat16*>(&word);
  r[0] = __bfloat162float(values[0]);
  r[1] = __bfloat162float(values[1]);
}

template <>
__device__ inline void LoadRowN<2, float>(const float* p, int64_t base,
                                         int lane, float* r) {
  const float2 word = reinterpret_cast<const float2*>(p + base)[lane];
  r[0] = word.x;
  r[1] = word.y;
}

// Public paged caches may have padding between rows. Vector loads require
// alignment of every reachable row; retain the portable path otherwise.
template <typename TQ, typename TKV>
bool LegacyDecodeRowsAligned(const Tensor& query, const Tensor& key,
                             const Tensor& value, int elements) {
  const auto aligned = [elements](const Tensor& tensor, size_t element_bytes,
                                  bool cache) {
    const size_t bytes = static_cast<size_t>(elements) * element_bytes;
    if (reinterpret_cast<uintptr_t>(tensor.data) % bytes != 0) return false;
    if (!cache) return true;  // query is contiguous, and d is a vector multiple
    for (int axis = 0; axis < 3; ++axis)
      if (tensor.stride[axis] % elements != 0) return false;
    return true;
  };
  return aligned(query, sizeof(TQ), false) && aligned(key, sizeof(TKV), true) &&
         aligned(value, sizeof(TKV), true);
}
