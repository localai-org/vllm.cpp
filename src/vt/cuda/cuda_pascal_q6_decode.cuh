// llama-cpp 62318cd5 vecdotq.cuh:627-648,1019-1042.
// Each lane consumes four packed low bytes and their matching high bits.
// Q6_K blocks alternate two-byte alignment, so GetIntB2 is required.
template <typename OutT>
__global__ void PascalQ6Decode(OutT* out, const uint8_t* weight,
                               const PascalQ8_1* act, int64_t n, int64_t nsb) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (row >= n) return;
  const int lane = threadIdx.x;
  float partial = 0;
  for (int64_t b = 0; b < nsb; ++b) {
    const auto* w = reinterpret_cast<const BlockQ6_K*>(weight + (row * nsb + b) * 210);
    const int aoff = 4 * (lane / 16) + (lane % 16) / 8;
    const int soff = 8 * (lane / 16) + (lane % 16) / 4;
    const int hshift = 2 * ((lane % 16) / 8);
    const int vl = GetIntB2(reinterpret_cast<const int8_t*>(w->ql), lane);
    const int vh = GetIntB2(reinterpret_cast<const int8_t*>(w->qh),
                           8 * (lane / 16) + lane % 8) >> hshift;
    float sum = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int lo = (vl >> (4 * i)) & 0x0f0f0f0f;
      const int hi = ((vh >> (4 * i)) << 4) & 0x30303030;
      const int v = __vsub4(lo | hi, 0x20202020);
      const auto& a = act[b * 8 + aoff + 2 * i];
      const int u = GetIntB2(a.qs, lane % 8);
      sum += DF16ToF32(a.d) * (LegacyDp4a(v, u, 0) * w->scales[soff + 4 * i]);
    }
    partial += DF16ToF32(w->d) * sum;
  }
#pragma unroll
  for (int mask = 16; mask; mask >>= 1)
    partial += __shfl_down_sync(0xffffffffu, partial, mask);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[row] = partial;
    else out[row] = DF32ToBF16(partial);
  }
}

void LaunchPascalQ6Decode(Tensor& out, const Tensor& a, const Tensor& weight,
                           int64_t n, int64_t nsb, cudaStream_t s) {
  auto* act = static_cast<PascalQ8_1*>(EnsureScratch(nsb * 8 * sizeof(PascalQ8_1), s));
  PascalQuantizeQ8_1<<<static_cast<unsigned>(nsb * 8), 32, 0, s>>>(
      act, a.data, ActDtOf(a.dtype), nsb * 8);
  const dim3 block(32, 4);
  const unsigned grid = static_cast<unsigned>((n + 3) / 4);
  if (out.dtype == DType::kF32)
    PascalQ6Decode<<<grid, block, 0, s>>>(static_cast<float*>(out.data),
        static_cast<const uint8_t*>(weight.data), act, n, nsb);
  else
    PascalQ6Decode<<<grid, block, 0, s>>>(static_cast<uint16_t*>(out.data),
        static_cast<const uint8_t*>(weight.data), act, n, nsb);
}
