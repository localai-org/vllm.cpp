// Pascal standard dense MMQ, llama.cpp 62318cd57:
// mmq.cuh:60-76,564-581; quantize.cu:457-550; mmq.cu:355-445.
// This is a numerical-contract port through the existing registered GEMM seam.
struct PascalMmqD4 {
  float d;
  int8_t qs[32];
};

__global__ void PascalQuantizeMmqD4(PascalMmqD4* out, const void* a, ActDT adt,
                                    int64_t nb, int64_t row_stride) {
  const int64_t b = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int lane = threadIdx.x;
  const float x = DLoadAct(a, adt, row * row_stride + b * 32 + lane);
  float amax = fabsf(x);
#pragma unroll
  for (int mask = 16; mask; mask >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
  // D4 stores F32 d, unlike MMVQ's half scale and original activation sum.
  const float inverse = amax == 0.0f ? 0.0f : 127.0f / amax;
  const float scale = inverse == 0.0f ? 0.0f : 1.0f / inverse;
  out[row * nb + b].qs[lane] = amax == 0.0f ? 0 : static_cast<int8_t>(roundf(x * inverse));
  if (lane == 0) out[row * nb + b].d = scale;
}

// The upstream dense MMQ selector requires a DP4A-compiled image and48KiB
// shared memory. Keep sm60, Volta, modern GPUs and repacked arms unchanged.
bool IsPascalMmqD4Eligible(const Tensor& out, const Tensor& a, const Tensor& b) {
  if (a.shape[0] <= 8 || a.shape[1] % 32 != 0 || !a.IsContiguous() ||
      !b.IsContiguous() || b.repacked || b.q8_0_aligned || b.elem_kn_repacked ||
      (a.dtype != DType::kF32 && a.dtype != DType::kBF16) ||
      (out.dtype != DType::kF32 && out.dtype != DType::kBF16)) return false;
  const auto& caps = GetDeviceCaps();
  if (!caps.valid || caps.sm_major != 6 || caps.sm_minor < 1 ||
      caps.max_shared_memory_per_block_optin < 48 * 1024) return false;
  thread_local int cached_device = -1;
  thread_local bool compiled_dp4a = false;
  if (cached_device != caps.device) {
    cudaFuncAttributes attributes{};
    CheckCuda(cudaFuncGetAttributes(&attributes, PascalQuantizeMmqD4), "Pascal MMQ compiled architecture");
    compiled_dp4a = attributes.binaryVersion >= 61;
    cached_device = caps.device;
  }
  return compiled_dp4a;
}

const PascalMmqD4* EnsurePascalMmqD4(const Tensor& a, int64_t m, int64_t nb,
                                    cudaStream_t stream) {
  auto* act = static_cast<PascalMmqD4*>(EnsureScratch(static_cast<size_t>(m * nb) * sizeof(PascalMmqD4), stream));
  const dim3 grid(static_cast<unsigned>(nb), static_cast<unsigned>(m));
  PascalQuantizeMmqD4<<<grid, 32, 0, stream>>>(act, a.data, ActDtOf(a.dtype), nb, a.stride[0]);
  return act;
}

template <bool Q5, typename OutT>
__global__ void PascalMmqD4Gemm(OutT* out, const uint8_t* weights,
                                const PascalMmqD4* act, int64_t m,
                                int64_t n, int64_t nb) {
  const int64_t output = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (output >= m * n) return;
  const int64_t row = output / n, col = output % n;
  const int lane = threadIdx.x;
  float partial = 0;
  for (int64_t b = lane; b < nb; b += 32) {
    const auto& ab = act[row * nb + b];
    int dot = 0;
    float wd;
    if constexpr (Q5) {
      const auto* w = reinterpret_cast<const BlockQ5_0*>(weights + (col * nb + b) * sizeof(BlockQ5_0));
      const uint32_t hi = uint32_t(w->qh[0]) | (uint32_t(w->qh[1]) << 8) |
          (uint32_t(w->qh[2]) << 16) | (uint32_t(w->qh[3]) << 24);
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int lo = (w->qs[j] & 15) + int((hi >> j) & 1) * 16 - 16;
        const int high = (w->qs[j] >> 4) + int((hi >> (j + 16)) & 1) * 16 - 16;
        dot += lo * ab.qs[j] + high * ab.qs[j + 16];
      }
      wd = DF16ToF32(w->d);
    } else {
      const auto* w = reinterpret_cast<const BlockQ8_0*>(weights + (col * nb + b) * sizeof(BlockQ8_0));
#pragma unroll
      for (int j = 0; j < 32; ++j) dot += int(w->qs[j]) * ab.qs[j];
      wd = DF16ToF32(w->d);
    }
    partial += (wd * ab.d) * static_cast<float>(dot);
  }
#pragma unroll
  for (int mask = 16; mask; mask >>= 1)
    partial += __shfl_down_sync(0xffffffffu, partial, mask);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[output] = partial;
    else out[output] = DF32ToBF16(partial);
  }
}

void LaunchPascalMmqD4(Tensor& out, const Tensor& a, const Tensor& b, cudaStream_t stream) {
  const int64_t m = a.shape[0], n = b.shape[0], nb = a.shape[1] / 32;
  const auto* act = EnsurePascalMmqD4(a, m, nb, stream);
  const dim3 block(32, 4);
  const unsigned grid = static_cast<unsigned>((m * n + 3) / 4);
  const auto* weight = static_cast<const uint8_t*>(b.data);
  if (b.dtype == DType::kQ5_0) {
    if (out.dtype == DType::kF32)
      PascalMmqD4Gemm<true><<<grid, block, 0, stream>>>(static_cast<float*>(out.data), weight, act, m, n, nb);
    else PascalMmqD4Gemm<true><<<grid, block, 0, stream>>>(static_cast<uint16_t*>(out.data), weight, act, m, n, nb);
  } else {
    if (out.dtype == DType::kF32)
      PascalMmqD4Gemm<false><<<grid, block, 0, stream>>>(static_cast<float*>(out.data), weight, act, m, n, nb);
    else PascalMmqD4Gemm<false><<<grid, block, 0, stream>>>(static_cast<uint16_t*>(out.data), weight, act, m, n, nb);
  }
}
