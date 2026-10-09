// Native EXL3: reference, packed and fused variants preserve the CPU reduction
// and Had128 rounding order. Only activations use global scratch.
#include "xpu_common.h"
#include "xpu_exl3.h"
#include "xpu_exl3_strategy.h"
#include "xpu_kernels.h"
#include "vt/xpu.h"
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string_view>
#include <nlohmann/json.hpp>

namespace vt::xpu {
namespace {
constexpr float kInvSqrt128 = 0.088388347648f;
float FlipSign(float x) { return sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(x) ^ 0x80000000u); }

// One accumulator per output preserves the reference's FP32 summation order.
// Specializing the real widths removes integer division by a runtime word
// count from every weight decode; aligned 32-bit loads replace byte assembly.
template<int Bits>
void PackedGemm(Queue& q, float* out, const sycl::half* in, const uint32_t* packed,
                int64_t m, int64_t k, int64_t n) {
  const auto event = NativeQueue(q).parallel_for(sycl::range<1>(m * n), [=](sycl::id<1> item) {
    const int64_t row = item[0] / n, col = item[0] % n;
    float acc = 0;
    for (int64_t tile_k = 0; tile_k < k / 16; ++tile_k) {
      const auto* tile = packed + (tile_k * (n / 16) + col / 16) * (8 * Bits);
      #pragma unroll
      for (int inner = 0; inner < 16; ++inner) {
        const float value = static_cast<float>(in[row * k + tile_k * 16 + inner]);
        if (value == 0.0f) continue;
        acc += value * exl3::Decode(exl3::PackedCodeword<Bits>(tile, inner, col % 16), 2);
      }
    }
    out[item[0]] = acc;
  });
  RecordProfileEvent(q, "exl3_packed_gemm", event);
}

// Each workgroup owns a complete 128-column Hadamard block. Register
// accumulators reuse each decoded weight across Rows tokens; 512*Rows bytes
// of SLM replace the global [M,N] temporary. Reductions retain the exact
// scalar order and the same four-values-per-lane output transform as Had().
template<int Bits, int Rows>
void FusedPackedGemm(Queue& q, const View out, const sycl::half* in, const uint32_t* packed,
                     const sycl::half* scales, int64_t m, int64_t k, int64_t n) {
  const int64_t tiles_n = n / 128, groups = ((m + Rows - 1) / Rows) * tiles_n;
  const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<float> shared(Rows * 128, h);
    h.parallel_for(sycl::nd_range<1>(groups * 128, 128), [=](sycl::nd_item<1> item) {
      const int lane = item.get_local_id(0);
      const int64_t row = (item.get_group(0) / tiles_n) * Rows;
      const int64_t col_base = (item.get_group(0) % tiles_n) * 128;
      const int64_t col = col_base + lane;
      float acc[Rows] = {};
      for (int64_t tile_k = 0; tile_k < k / 16; ++tile_k) {
        const auto* tile = packed + (tile_k * (n / 16) + col / 16) * (8 * Bits);
        #pragma unroll
        for (int inner = 0; inner < 16; ++inner) {
          const float weight = exl3::Decode(exl3::PackedCodeword<Bits>(tile, inner, col % 16), 2);
          #pragma unroll
          for (int r = 0; r < Rows; ++r) {
            if (row + r >= m) continue;
            const float value = static_cast<float>(in[(row + r) * k + tile_k * 16 + inner]);
            if (value != 0.0f) acc[r] += value * weight;
          }
        }
      }
      for (int r = 0; r < Rows; ++r) shared[r * 128 + lane] = acc[r];
      item.barrier(sycl::access::fence_space::local_space);
      for (int r = 0; r < Rows; ++r) {
        float v[4] = {};
        const int base = r * 128;
        if (lane < 32) {
          for (int j = 0; j < 4; ++j) v[j] = shared[base + lane * 4 + j];
          const float s0 = v[0] + v[1], d0 = v[0] - v[1];
          const float s1 = v[2] + v[3], d1 = v[2] - v[3];
          v[0] = s0 + s1; v[1] = d0 + d1; v[2] = s0 - s1; v[3] = d0 - d1;
          for (int j = 0; j < 4; ++j) shared[base + lane * 4 + j] = v[j];
        }
        item.barrier(sycl::access::fence_space::local_space);
        for (int step = 1; step < 32; step *= 2) {
          if (lane < 32) for (int j = 0; j < 4; ++j)
            v[j] = ((lane & step) ? FlipSign(v[j]) : v[j]) + shared[base + (lane ^ step) * 4 + j];
          item.barrier(sycl::access::fence_space::local_space);
          if (lane < 32) for (int j = 0; j < 4; ++j) shared[base + lane * 4 + j] = v[j];
          item.barrier(sycl::access::fence_space::local_space);
        }
        if (lane < 32 && row + r < m) for (int j = 0; j < 4; ++j) {
          const int64_t column = col_base + lane * 4 + j;
          const float value = Round(out.dtype == DType::kBF16 ? DType::kF32 : out.dtype,
                                    v[j] * kInvSqrt128) * static_cast<float>(scales[column]);
          Store(out, (row + r) * n + column, value);
        }
      }
    });
  });
  RecordProfileEvent(q, "exl3_fused_gemm_hadamard", event);
}

using exl3::Strategy;
Strategy AutomaticStrategy(Queue& q, int bits, int64_t k, int64_t n, int64_t m, DType dtype) {
  static std::mutex mutex;
  static std::map<int, exl3::StrategyDomain> domains;
  static exl3::StrategyCache cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto found = domains.find(q.device.index);
  if (found == domains.end()) {
    const auto info = nlohmann::json::parse(DeviceDescription(q.device.index));
    found = domains.emplace(q.device.index, exl3::StrategyDomain{
        info.value("device_id", -1), info.at("driver"), info.at("runtime"),
        info.at("compiler"), exl3::kKernelVersion}).first;
  }
  return cache.Get(found->second, {bits, k, n, m, dtype});
}

Strategy SelectedStrategy() {
  static const Strategy strategy = [] {
    const char* value = std::getenv("VT_XPU_EXL3_STRATEGY");
    const std::string_view name = value ? value : "auto";
    if (name == "reference") return Strategy::kReference;
    if (name == "packed") return Strategy::kPacked;
    if (name == "fused") return Strategy::kFused;
    if (name == "prefill") return Strategy::kPrefill;
    if (name == "panel") return Strategy::kPanel;
    if (name == "prefill_all_rows") return Strategy::kPrefillAllRows;
    VT_CHECK(name == "auto", "Invalid VT_XPU_EXL3_STRATEGY");
    return Strategy::kAuto;
  }();
  return strategy;
}

const char* StrategyName(Strategy strategy) {
  switch (strategy) {
    case Strategy::kAuto: return "auto";
    case Strategy::kReference: return "reference";
    case Strategy::kPacked: return "packed";
    case Strategy::kFused: return "fused";
    case Strategy::kPrefill: return "prefill";
    case Strategy::kPanel: return "panel";
    case Strategy::kPrefillAllRows: return "prefill_all_rows";
  }
  return "unknown";
}

void TraceExl3Dispatch(Queue& q, const Tensor& in, const Tensor& out, int bits, int codebook,
                       const char* name,
                       Strategy selected, const char* leaf, const char* reason) {
  static const bool enabled = [] {
    const char* value = std::getenv("VT_XPU_EXL3_TRACE");
    return value && std::string_view(value) == "1";
  }();
  if (!enabled) return;
  int layer = -1;
  if (name) {
    const std::string_view matrix(name);
    const size_t marker = matrix.find(".layers.");
    if (marker != std::string_view::npos) {
      const size_t first = marker + sizeof(".layers.") - 1;
      if (first < matrix.size() && matrix[first] >= '0' && matrix[first] <= '9')
        layer = std::atoi(name + first);
    }
  }
  const nlohmann::json event = {{"event", "exl3_dispatch"}, {"queue_id", q.id},
      {"matrix", name ? name : ""}, {"layer", layer},
      {"bits", bits}, {"codebook", codebook}, {"m", in.shape[0]}, {"k", in.shape[1]},
      {"n", out.shape[1]}, {"input_dtype", Name(in.dtype)}, {"output_dtype", Name(out.dtype)},
      {"requested", StrategyName(SelectedStrategy())}, {"selected", StrategyName(selected)},
      {"leaf", leaf}, {"rejection_reason", reason}};
  std::fprintf(stderr, "EXL3_DISPATCH %s\n", event.dump().c_str());
}

template<bool StridedOutput = false, bool CastInputToHalf = false>
void Had(Queue& q, Tensor& out, const Tensor& in, const Tensor* pre, const Tensor* post,
         float scale, const char* profile_stage = "exl3_hadamard") {
  if (in.Numel() == 0) return;
  const View src(in), dst(out);
  const auto cols = in.shape[1];
  const auto* pre_values = pre ? static_cast<const sycl::half*>(pre->data) : nullptr;
  const auto* post_values = post ? static_cast<const sycl::half*>(post->data) : nullptr;
  const auto blocks = static_cast<size_t>(in.Numel() / 128);
  const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<float> shared(128, h);
    h.parallel_for(sycl::nd_range<1>(blocks * 32, 32), [=](sycl::nd_item<1> item) {
      const int lane = item.get_local_id(0);
      const int64_t base = item.get_group(0) * 128;
      const int64_t col = base % cols;
      float v[4];
      for (int j = 0; j < 4; ++j) {
        v[j] = Load(src, base + lane * 4 + j);
        if constexpr (CastInputToHalf) v[j] = Round(DType::kF16, v[j]);
        if (pre_values) v[j] = Round(CastInputToHalf ? DType::kF16 : src.dtype,
                                     v[j] * static_cast<float>(pre_values[col + lane * 4 + j]));
      }
      const float s0 = v[0] + v[1], d0 = v[0] - v[1];
      const float s1 = v[2] + v[3], d1 = v[2] - v[3];
      v[0] = s0 + s1; v[1] = d0 + d1; v[2] = s0 - s1; v[3] = d0 - d1;
      for (int j = 0; j < 4; ++j) shared[lane * 4 + j] = v[j];
      item.barrier(sycl::access::fence_space::local_space);
      for (int step = 1; step < 32; step *= 2) {
        for (int j = 0; j < 4; ++j)
          v[j] = ((lane & step) ? FlipSign(v[j]) : v[j]) + shared[(lane ^ step) * 4 + j];
        item.barrier(sycl::access::fence_space::local_space);
        for (int j = 0; j < 4; ++j) shared[lane * 4 + j] = v[j];
        item.barrier(sycl::access::fence_space::local_space);
      }
      for (int j = 0; j < 4; ++j) {
        // Half output rounds BEFORE multiplying svh, then again at the store.
        // A BF16 target replaces the caller's F32 result + final cast. It must
        // not introduce an extra BF16 rounding BEFORE multiplying svh.
        float value = Round(dst.dtype == DType::kBF16 ? DType::kF32 : dst.dtype, v[j] * scale);
        if (post_values) value *= static_cast<float>(post_values[col + lane * 4 + j]);
        const auto index = base + lane * 4 + j;
        Store(dst, StridedOutput ? dst.offset(index) : index, value);
      }
    });
  });
  RecordProfileEvent(q, profile_stage, event);
}
}

void Exl3OutputHadPanel(Queue& q, Tensor& out, const Tensor& raw, const Tensor& svh, int64_t column) {
  auto panel = Tensor::Contiguous(static_cast<char*>(out.data) + column * SizeOf(out.dtype),
      out.dtype, out.device, {raw.shape[0], raw.shape[1]});
  panel.stride[0] = out.shape[1];
  auto scales = Tensor::Contiguous(static_cast<char*>(svh.data) + column * sizeof(sycl::half),
      svh.dtype, svh.device, {raw.shape[1]});
  Had<true>(q, panel, raw, nullptr, &scales, kInvSqrt128, "exl3_output_hadamard");
}

void Exl3HadR128Kernel(Queue& q, Tensor& out, const Tensor& in, const Exl3HadArgs& args) {
  TraceXpuOp(OpId::kExl3HadR128, q, {&out, &in, args.pre_scale, args.post_scale});
  const auto* sc = args.pre_scale ? args.pre_scale : args.post_scale;
  VT_CHECK(!sc || sc->IsContiguous(), "XPU EXL3 Had128 requires contiguous scale");
  // Exact in-place is safe: each workgroup reads its whole block before stores.
  if (out.data == in.data && (!sc || !Overlap(out, *sc))) {
    Had(q, out, in, args.pre_scale, args.post_scale, args.scale * kInvSqrt128,
        args.pre_scale ? "exl3_input_hadamard" : "exl3_output_hadamard");
  } else {
    WithOutput(q, out, {&in, sc}, [&](Tensor& target) {
      Had(q, target, in, args.pre_scale, args.post_scale, args.scale * kInvSqrt128,
          args.pre_scale ? "exl3_input_hadamard" : "exl3_output_hadamard");
    });
  }
}

static void Exl3GemmImpl(Queue& q, Tensor& out, const Tensor& in, const Tensor& trellis,
                         const Tensor& suh, const Tensor& svh, Tensor& in_had,
                         const Exl3GemmArgs& args, Tensor* replay_raw) {
  const ProfileMatrixScope profile_matrix(args.debug_name);
  TraceXpuOp(OpId::kExl3Gemm, q, {&out, &in, &trellis, &suh, &svh, &in_had});
  const int64_t m = in.shape[0], k = in.shape[1], n = out.shape[1];
  if (m == 0 || k == 0 || n == 0) return;
  VT_CHECK(!Overlap(in_had, trellis) && !Overlap(in_had, suh) && !Overlap(in_had, svh),
           "XPU EXL3 input scratch may not overwrite weights");
  VT_CHECK(!Overlap(out, trellis) && !Overlap(out, suh) && !Overlap(out, svh),
           "XPU EXL3 output may not overwrite weights");
  const auto* ah = static_cast<const sycl::half*>(in_had.data);
  const auto* packed = static_cast<const unsigned char*>(trellis.data);
  const int bits = args.bits, cb = args.codebook;
  auto strategy = SelectedStrategy();
  // The prefill override selects only large-M calls; the head gather and
  // subsequent decode retain the measured PR07 policy.
  if ((strategy == Strategy::kPrefill || strategy == Strategy::kPanel) && m < 128)
    strategy = Strategy::kAuto;
  if (strategy == Strategy::kPrefillAllRows && m < 512) strategy = Strategy::kAuto;
  if (strategy == Strategy::kAuto)
    strategy = AutomaticStrategy(q, bits, k, n, m, out.dtype == DType::kBF16 ? DType::kF32 : out.dtype);
  if (replay_raw) {
    VT_CHECK(strategy == Strategy::kPacked || strategy == Strategy::kReference,
             "EXL3 replay cannot capture fused/prefill intermediates; dispatch was not changed");
    VT_CHECK(replay_raw->device == q.device && replay_raw->dtype == DType::kF32 &&
                 replay_raw->rank == 2 && replay_raw->shape[0] == m && replay_raw->shape[1] == n &&
                 replay_raw->IsContiguous(), "EXL3 replay requires contiguous F32[M,N] on the owning device");
    for (const Tensor* tensor : std::initializer_list<const Tensor*>{&out, &in, &trellis, &suh, &svh, &in_had})
      VT_CHECK(!Overlap(*replay_raw, *tensor), "EXL3 replay raw storage may not overlap operands");
    RecordGraphWrite(q, replay_raw->data, Span(*replay_raw));
  }
  if (args.fuse_casts && in.dtype != DType::kF16) {
    WithOutput(q, in_had, {&in, &suh}, [&](Tensor& target) {
      Had<false, true>(q, target, in, &suh, nullptr, kInvSqrt128, "exl3_input_hadamard");
    });
  } else Exl3HadR128Kernel(q, in_had, in, Exl3HadArgs{&suh, nullptr, 1.0f});
  const bool specialized = strategy != Strategy::kReference && cb == 2 && bits >= 3 && bits <= 6 &&
      reinterpret_cast<uintptr_t>(packed) % alignof(uint32_t) == 0;
  if (specialized && (strategy == Strategy::kPrefill || strategy == Strategy::kPanel ||
      strategy == Strategy::kPrefillAllRows) && !Overlap(out, in_had) &&
      Exl3PrefillKernel(q, out, in_had, trellis, svh, bits,
                        strategy != Strategy::kPanel, strategy == Strategy::kPrefillAllRows)) {
    TraceExl3Dispatch(q, in, out, bits, cb, args.debug_name,
                      strategy, StrategyName(strategy), "none");
    return;
  }
  const bool fused = specialized && !Overlap(out, in_had) && strategy == Strategy::kFused;
  if (fused) {
    TraceExl3Dispatch(q, in, out, bits, cb, args.debug_name, strategy, "fused", "none");
    const auto* words = reinterpret_cast<const uint32_t*>(packed);
    const auto* scales = static_cast<const sycl::half*>(svh.data);
    auto launch = [&]<int Bits>() {
      if (m == 1) FusedPackedGemm<Bits, 1>(q, View(out), ah, words, scales, m, k, n);
      else FusedPackedGemm<Bits, 4>(q, View(out), ah, words, scales, m, k, n);
    };
    switch (bits) {
      case 3: launch.template operator()<3>(); break;
      case 4: launch.template operator()<4>(); break;
      case 5: launch.template operator()<5>(); break;
      case 6: launch.template operator()<6>(); break;
    }
    return;
  }
  const char* reason = "none";
  if (strategy == Strategy::kReference) reason = "reference_selected";
  else if (cb != 2) reason = "unsupported_codebook";
  else if (bits < 3 || bits > 6) reason = "unsupported_bits";
  else if (reinterpret_cast<uintptr_t>(packed) % alignof(uint32_t) != 0) reason = "unaligned_trellis";
  else if ((strategy == Strategy::kFused || strategy == Strategy::kPrefill ||
            strategy == Strategy::kPanel || strategy == Strategy::kPrefillAllRows) &&
           Overlap(out, in_had)) reason = "output_overlaps_input_scratch";
  else if (strategy == Strategy::kPrefill || strategy == Strategy::kPanel ||
           strategy == Strategy::kPrefillAllRows) reason = "prefill_kernel_rejected";
  TraceExl3Dispatch(q, in, out, bits, cb, args.debug_name,
                    strategy, specialized ? "packed" : "reference", reason);
  const size_t raw_bytes = m * n * sizeof(float);
  auto launch = [&](void* storage) {
    auto raw = Tensor::Contiguous(storage, DType::kF32, q.device, {m, n});
    auto* result = static_cast<float*>(raw.data);
    if (specialized) {
      const auto* words = reinterpret_cast<const uint32_t*>(packed);
      switch (bits) {
        case 3: PackedGemm<3>(q, result, ah, words, m, k, n); break;
        case 4: PackedGemm<4>(q, result, ah, words, m, k, n); break;
        case 5: PackedGemm<5>(q, result, ah, words, m, k, n); break;
        case 6: PackedGemm<6>(q, result, ah, words, m, k, n); break;
      }
    } else {
      const auto event = NativeQueue(q).parallel_for(sycl::range<1>(m * n), [=](sycl::id<1> item) {
      const int64_t row = item[0] / n, col = item[0] % n;
      float acc = 0;
      for (int64_t inner = 0; inner < k; ++inner) {
        const float value = static_cast<float>(ah[row * k + inner]);
        if (value == 0.0f) continue;  // same signed-zero behavior as CPU
        const auto* tile = packed + ((inner / 16) * (n / 16) + col / 16) * (32 * bits);
        const int t = exl3::Fragment(inner % 16, col % 16);
        acc += value * exl3::Decode(exl3::Codeword(tile, bits, t), cb);
      }
      result[item[0]] = acc;
      });
      RecordProfileEvent(q, "exl3_reference_gemm", event);
    }
    Had(q, out, raw, nullptr, &svh, kInvSqrt128, "exl3_output_hadamard");
  };
  if (replay_raw) {
    launch(replay_raw->data);
    return;
  }
  // Reuse the same bounded buffer as prefill. Decode graphs retain its address
  // and the backend orders all eager/graph users across queues. Larger eager
  // shapes or tight budgets keep the original temporary-buffer fallback.
  constexpr size_t workspace_bytes = 32 * 1024 * 1024;
  if (raw_bytes <= workspace_bytes && WithExl3Workspace(q, workspace_bytes, launch)) return;
  Scratch storage(q.device, raw_bytes);
  launch(storage.data);
  // Free is conservative and waits all owned queues; surface errors explicitly.
  GetBackend(q.device).Synchronize(q);
}

void Exl3GemmKernel(Queue& q, Tensor& out, const Tensor& in, const Tensor& trellis,
                    const Tensor& suh, const Tensor& svh, Tensor& in_had, const Exl3GemmArgs& args) {
  Exl3GemmImpl(q, out, in, trellis, suh, svh, in_had, args, nullptr);
}

void Exl3GemmReplayKernel(Queue& q, Tensor& out, const Tensor& in, const Tensor& trellis,
                          const Tensor& suh, const Tensor& svh, Tensor& in_had,
                          const Exl3GemmArgs& args, Tensor& raw) {
  Exl3GemmImpl(q, out, in, trellis, suh, svh, in_had, args, &raw);
}
}  // namespace vt::xpu
