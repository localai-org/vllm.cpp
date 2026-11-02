// Pascal Q5_0 decode, adapted from llama-cpp 62318cd5:
// quantize.cu:64-101 and vecdotq.cuh:175-203. Prefill uses its existing MMQ lane.
struct PascalQ8_1 {
  uint16_t d;
  uint16_t sum;
  int8_t qs[32];
};

__global__ void PascalQuantizeQ8_1(PascalQ8_1* out, const void* a, ActDT adt,
                                   int64_t nb) {
  const int64_t b = blockIdx.x;
  const int lane = threadIdx.x;
  const float x = DLoadAct(a, adt, b * 32 + lane);
  float amax = fabsf(x);
  float sum = x;
#pragma unroll
  for (int mask = 16; mask; mask >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
    sum += __shfl_xor_sync(0xffffffffu, sum, mask);
  }
  const float d = amax / 127.0f;
  out[b].qs[lane] = amax == 0 ? 0 : static_cast<int8_t>(roundf(x / d));
  if (lane == 0) {
    out[b].d = DF32ToF16(d);
    out[b].sum = DF32ToF16(sum);
  }
}

template <typename OutT>
__global__ void PascalQ5Decode(OutT* out, const uint8_t* weight,
                               const PascalQ8_1* act, int64_t n, int64_t nb) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (row >= n) return;
  const int lane = threadIdx.x;
  float partial = 0;
  // Each lane computes the oracle's 16-value VDR=2 contribution.
  for (int64_t b = lane / 2; b < nb; b += 16) {
    const auto* w = reinterpret_cast<const BlockQ5_0*>(weight + (row * nb + b) * 22);
    const auto& a = act[b];
    const uint32_t qh = static_cast<uint32_t>(w->qh[0]) |
        (static_cast<uint32_t>(w->qh[1]) << 8) |
        (static_cast<uint32_t>(w->qh[2]) << 16) |
        (static_cast<uint32_t>(w->qh[3]) << 24);
    int sum = 0;
#pragma unroll
    for (int j = (lane % 2) * 8; j < (lane % 2 + 1) * 8; ++j) {
      const int lo = (w->qs[j] & 15) | (((qh >> j) & 1) << 4);
      const int hi = (w->qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4);
      sum += lo * a.qs[j] + hi * a.qs[j + 16];
    }
    partial += DF16ToF32(w->d) *
        (static_cast<float>(sum) * DF16ToF32(a.d) - 8.0f * DF16ToF32(a.sum));
  }
#pragma unroll
  for (int mask = 16; mask; mask >>= 1)
    partial += __shfl_down_sync(0xffffffffu, partial, mask);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[row] = partial;
    else out[row] = DF32ToBF16(partial);
  }
}

bool IsPascalQ5Decode() {
  int device = 0;
  CheckCuda(cudaGetDevice(&device), "Pascal Q5 decode device");
  thread_local int cached_device = -1;
  thread_local bool pascal = false;
  if (cached_device != device) {
    cudaDeviceProp prop{};
    CheckCuda(cudaGetDeviceProperties(&prop, device), "Pascal Q5 decode properties");
    pascal = prop.major == 6;
    cached_device = device;
  }
  return pascal;
}

void LaunchPascalQ5Decode(Tensor& out, const Tensor& a, const Tensor& weight,
                           int64_t n, int64_t nb, cudaStream_t s) {
  auto* act = static_cast<PascalQ8_1*>(EnsureScratch(nb * sizeof(PascalQ8_1), s));
  PascalQuantizeQ8_1<<<static_cast<unsigned>(nb), 32, 0, s>>>(act, a.data, ActDtOf(a.dtype), nb);
  const dim3 block(32, 4);
  const unsigned grid = static_cast<unsigned>((n + 3) / 4);
  if (out.dtype == DType::kF32)
    PascalQ5Decode<<<grid, block, 0, s>>>(static_cast<float*>(out.data),
        static_cast<const uint8_t*>(weight.data), act, n, nb);
  else
    PascalQ5Decode<<<grid, block, 0, s>>>(static_cast<uint16_t*>(out.data),
        static_cast<const uint8_t*>(weight.data), act, n, nb);
}
