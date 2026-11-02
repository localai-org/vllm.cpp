// Pascal Q4_K decode, adapted from llama-cpp 62318cd5 vecdotq.cuh:918-975
// (vec_dot_q4_K_q8_1) and ggml-quants.c:1530 (dequantize_row_q4_K layout).
// Each of the 32 lanes owns 8 of the 256 super-block values, so every lane
// issues two __dp4a instead of the generic kernel's 8-active-lane scalar walk.
// Prefill (m > 1) keeps its existing MMQ lane.
__device__ __forceinline__ void PascalQ4KScaleMin(const BlockQ4_K* w, int sb, int* sc, int* m) {
  if (sb < 4) {
    *sc = w->scales[sb] & 63;
    *m = w->scales[sb + 4] & 63;
  } else {
    *sc = (w->scales[sb + 4] & 0xF) | ((w->scales[sb - 4] >> 6) << 4);
    *m = (w->scales[sb + 4] >> 4) | ((w->scales[sb] >> 6) << 4);
  }
}

// Default on. The Q8_1 activation lane matches llama.cpp's decode arithmetic
// but reduces the super-block in a different order, so it can resolve a
// sub-milli-nat tie differently. VT_PASCAL_Q4K_DECODE=0 restores the generic
// Q8_K lane for a byte-identical rollback.
bool PascalQ4KDecodeEnabled() {
  static const bool enabled = [] {
    const char* e = std::getenv("VT_PASCAL_Q4K_DECODE");
    return !(e != nullptr && e[0] == '0');
  }();
  return enabled;
}

template <typename OutT>
__global__ void PascalQ4KDecode(OutT* out, const uint8_t* weight,
                                const PascalQ8_1* act, int64_t n, int64_t nsb) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (row >= n) return;
  const int lane = threadIdx.x;
  const int sb = lane / 4;       // 32-value sub-block inside the super-block
  const int q = lane % 4;        // which 8 of that sub-block's 32 values
  const int shift = (sb & 1) ? 4 : 0;
  const int aoff = q * 8;
  float partial = 0;
  for (int64_t b = 0; b < nsb; ++b) {
    const auto* w = reinterpret_cast<const BlockQ4_K*>(weight + (row * nsb + b) * 144);
    const auto& a = act[b * 8 + sb];
    const int* q4 = reinterpret_cast<const int*>(w->qs + (sb / 2) * 32 + aoff);
    const int* q8 = reinterpret_cast<const int*>(a.qs + aoff);
    int dot = 0;
    int suma = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
      const int v = (q4[l] >> shift) & 0x0F0F0F0F;
      dot = LegacyDp4a(v, q8[l], dot);
      suma = LegacyDp4a(0x01010101, q8[l], suma);
    }
    int sc = 0;
    int m = 0;
    PascalQ4KScaleMin(w, sb, &sc, &m);
    partial += DF16ToF32(a.d) *
        (DF16ToF32(w->d) * sc * dot - DF16ToF32(w->dmin) * m * suma);
  }
#pragma unroll
  for (int mask = 16; mask; mask >>= 1)
    partial += __shfl_down_sync(0xffffffffu, partial, mask);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[row] = partial;
    else out[row] = DF32ToBF16(partial);
  }
}

void LaunchPascalQ4KDecode(Tensor& out, const Tensor& a, const Tensor& weight,
                           int64_t n, int64_t nsb, cudaStream_t s) {
  auto* act = static_cast<PascalQ8_1*>(EnsureScratch(nsb * 8 * sizeof(PascalQ8_1), s));
  PascalQuantizeQ8_1<<<static_cast<unsigned>(nsb * 8), 32, 0, s>>>(
      act, a.data, ActDtOf(a.dtype), nsb * 8);
  const dim3 block(32, 4);
  const unsigned grid = static_cast<unsigned>((n + 3) / 4);
  if (out.dtype == DType::kF32)
    PascalQ4KDecode<<<grid, block, 0, s>>>(static_cast<float*>(out.data),
        static_cast<const uint8_t*>(weight.data), act, n, nsb);
  else
    PascalQ4KDecode<<<grid, block, 0, s>>>(static_cast<uint16_t*>(out.data),
        static_cast<const uint8_t*>(weight.data), act, n, nsb);
}
