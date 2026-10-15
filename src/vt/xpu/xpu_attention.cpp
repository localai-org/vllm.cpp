#include "xpu_common.h"
#include "xpu_kernels.h"
#include "xpu_fp8.h"
#include "xpu_qk_norm.h"
#include <sycl/ext/intel/math.hpp>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <limits>
#include <cstdlib>
#include <memory>
#include <string_view>
#include <vector>

namespace vt::xpu {
namespace {
int64_t Position(View pos, int64_t token) {
  return pos.dtype == DType::kI32 ? static_cast<const int32_t*>(pos.data)[token]
                                : static_cast<const int64_t*>(pos.data)[token];
}
double Frequency(int64_t pair, const RopeArgs& args) {
  double freq = sycl::pow(double(args.base), -2.0 * pair / args.rotary_dim);
  if (!(args.llama3_scaling_factor > 0)) return freq;
  constexpr double two_pi = 6.283185307179586476925286766559;
  const double low = args.llama3_low_freq_factor, high = args.llama3_high_freq_factor;
  const double original = args.llama3_orig_max_position, factor = args.llama3_scaling_factor;
  const double wavelength = two_pi / freq;
  if (wavelength < original / high) return freq;
  if (wavelength > original / low) return freq / factor;
  const double smooth = low == high ? 0.0 : (original / wavelength - low) / (high - low);
  return (1.0 - smooth) * freq / factor + smooth * freq;
}
void Rotate(View dst, int64_t token, int64_t head, int64_t first, int64_t second, float c, float s,
            bool fp16 = false) {
  const auto base = token * dst.stride[0] + head * dst.stride[1];
  const float x = Load(dst, base + first), y = Load(dst, base + second);
  if (fp16) {
    c = Round(DType::kF16, c); s = Round(DType::kF16, s);
    Store(dst, base + first, Round(DType::kF16, x * c) - Round(DType::kF16, y * s));
    Store(dst, base + second, Round(DType::kF16, x * s) + Round(DType::kF16, y * c));
  } else {
    Store(dst, base + first, x * c - y * s);
    Store(dst, base + second, x * s + y * c);
  }
}
void CopyElement(View dst, int64_t to, View src, int64_t from) {
  if (src.dtype == DType::kF32)
    static_cast<uint32_t*>(dst.data)[to] = static_cast<const uint32_t*>(src.data)[from];
  else
    static_cast<uint16_t*>(dst.data)[to] = static_cast<const uint16_t*>(src.data)[from];
}


}
void AttnGateSplitKernel(Queue& q, Tensor& queries, Tensor& gates, const Tensor& packed) {
  TraceXpuOp(OpId::kAttnGateSplit, q, {&queries, &gates, &packed});
  VT_CHECK(!Overlap(queries, packed) && !Overlap(gates, packed) && !Overlap(queries, gates),
           "XPU Q/gate split requires separate output storage");
  const View src(packed), dst(queries), gate(gates);
  const auto width = queries.shape[2];
  const auto event = NativeQueue(q).parallel_for(sycl::range<1>(queries.Numel()), [=](sycl::id<1> item) {
    const int64_t head = item[0] / width, col = item[0] % width;
    Store(dst, item[0], Load(src, head * 2 * width + col));
    Store(gate, item[0], Load(src, head * 2 * width + width + col));
  });
  RecordProfileEvent(q, "attn_gate_split", event);
}
void AttnQkNormRopeGateKernel(Queue& q, Tensor& q_out, Tensor& k_out,
                             Tensor& gate_out, const Tensor& qgate,
                             const Tensor& kf, const Tensor& q_norm,
                             const Tensor& k_norm, const Tensor& cos_sin,
                             const RmsNormArgs& norm_args, const RopeArgs& rope_args) {
  TraceXpuOp(OpId::kAttnQkNormRopeGate, q,
             {&q_out, &k_out, &gate_out, &qgate, &kf, &q_norm, &k_norm, &cos_sin});
  for (const Tensor* dst : {&q_out, &k_out, &gate_out})
    for (const Tensor* src : {&qgate, &kf, &q_norm, &k_norm, &cos_sin})
      VT_CHECK(!Overlap(*dst, *src), "XPU attention preamble requires separate outputs");
  VT_CHECK(!Overlap(q_out, k_out) && !Overlap(q_out, gate_out) &&
               !Overlap(k_out, gate_out),
           "XPU attention preamble outputs must not overlap");
  const View qo(q_out), ko(k_out), go(gate_out), qs(qgate), ks(kf),
             qw(q_norm), kw(k_norm), cs(cos_sin);
  const int64_t tokens = q_out.shape[0], hq = q_out.shape[1],
                hk = k_out.shape[1], dim = q_out.shape[2];
  const int64_t half = rope_args.rotary_dim / 2, rot = rope_args.rotary_dim;
  const float eps = norm_args.eps;
  const bool gemma = norm_args.gemma;
  const bool fp16 = rope_args.fp16_intermediates;
  // Optional focused observer of the actual kernel's mean/inverse boundary.
  // No allocation or synchronization when unset; never overwrite a receipt.
  const char* probe_prefix = std::getenv("VT_XPU_ATTN_NORM_PROBE");
  auto free_probe = [&q](float* p) { if (p) sycl::free(p, NativeQueue(q)); };
  std::unique_ptr<float, decltype(free_probe)> probe_owner(nullptr, free_probe);
  if (probe_prefix) probe_owner.reset(sycl::malloc_shared<float>(3 * tokens * (hq + hk), NativeQueue(q)));
  float* probe = probe_owner.get();
  VT_CHECK(!probe_prefix || probe != nullptr, "XPU attention norm probe allocation failed");
  const auto finish_probe = [&](sycl::event event) {
    if (!probe_prefix) return;
    event.wait_and_throw();
    static std::atomic<uint64_t> sequence{0};
    const std::string path = std::string(probe_prefix) + "." +
        std::to_string(sequence.fetch_add(1)) + ".t" + std::to_string(tokens) + ".f32";
    // Ordinary host storage is required for file I/O on Level Zero USM.
    const std::vector<float> host(probe, probe + 3 * tokens * (hq + hk));
    std::FILE* file = std::fopen(path.c_str(), "wbx");
    VT_CHECK(file != nullptr, "XPU attention norm probe requires a new writable path");
    const size_t written = std::fwrite(host.data(), sizeof(float), host.size(), file);
    const int closed = std::fclose(file);
    VT_CHECK(written == host.size() && closed == 0, "XPU attention norm probe write failed");
  };
  enum class PreambleMode { kAuto, kReference, kSubgroup };
  static const PreambleMode mode = [] {
    const char* value = std::getenv("VT_XPU_ATTN_PREAMBLE");
    const std::string_view name = value ? value : "auto";
    VT_CHECK(name == "auto" || name == "reference" || name == "subgroup",
             "Invalid VT_XPU_ATTN_PREAMBLE");
    return name == "reference" ? PreambleMode::kReference :
           name == "subgroup" ? PreambleMode::kSubgroup : PreambleMode::kAuto;
  }();
  const bool shape_ok = dim == 256 && rot > 0 && rot <= dim && half % 16 == 0 &&
      qgate.dtype == DType::kF16 && kf.dtype == DType::kF16 &&
      q_out.dtype == DType::kF16 && k_out.dtype == DType::kF16;
  bool subgroup = false;
  if (shape_ok && mode != PreambleMode::kReference) {
    const auto sizes = NativeQueue(q).get_device().get_info<sycl::info::device::sub_group_sizes>();
    subgroup = std::find(sizes.begin(), sizes.end(), 16) != sizes.end();
  }
  VT_CHECK(mode != PreambleMode::kSubgroup || subgroup,
           "VT_XPU_ATTN_PREAMBLE=subgroup requires F16 D256, RoPE half divisible by 16 and SG16");
  if (subgroup) {
    constexpr int lanes = 16, tile = 16, workgroup = 128;
    const auto heads = tokens * (hq + hk);
    const auto global = ((heads * lanes + workgroup - 1) / workgroup) * workgroup;
    // Separate kernel instantiations preserve the producer root boundary:
    // runtime selection of rsqrt versus legacy 1/sqrt changed observed F32
    // inverses on the pinned B70 toolchain, despite identical variances.
    const auto launch = [&]<bool Producer>() {
      return NativeQueue(q).parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(workgroup)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
      constexpr bool fp16 = Producer;
      const int64_t index = item.get_global_linear_id() / lanes;
      if (index >= heads) return;
      const int lane = item.get_local_linear_id() % lanes;
      const int64_t token = index / (hq + hk), head = index % (hq + hk);
      const bool query = head < hq;
      const int64_t local_head = query ? head : head - hq;
      const View src = query ? qs : ks, weight = query ? qw : kw, dst = query ? qo : ko;
      const int64_t src_base = token * src.stride[0] + local_head * (query ? 2 * dim : dim);
      const int64_t out_base = (token * (query ? hq : hk) + local_head) * dim;
      float values[tile], norm[tile], sum = 0.0f;
      for (int j = 0; j < tile; ++j) {
        values[j] = Load(src, src_base + lane + lanes * j);
        sum += values[j] * values[j];
      }
      sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<float>());
      const float mean = fp16 ? ProducerQkMean256(src, src_base, item.get_sub_group(),
                                                 lane, tokens * (query ? hq : hk))
                              : sum / static_cast<float>(dim);
      const float variance = mean + eps;
      // Keep the producer rsqrt operation distinct from the legacy 1/sqrt.
      const float inv = fp16 ? sycl::rsqrt(variance) : 1.0f / sycl::sqrt(variance);
      if (probe && lane == 0) {
        probe[3 * index] = mean; probe[3 * index + 1] = variance; probe[3 * index + 2] = inv;
      }
      for (int j = 0; j < tile; ++j) {
        const int col = lane + lanes * j;
        const float w = Load(weight, col);
        norm[j] = fp16 ? ProducerQkNormValue(values[j], inv, w, gemma)
                       : values[j] * inv * (gemma ? 1.0f + w : w);
        if (fp16) norm[j] = Round(DType::kF16, norm[j]);
      }
      for (int j = 0; j < tile; ++j) {
        const int col = lane + lanes * j;
        float value = norm[j];
        if (col < rot) {
          const int pair = col < half ? col : col - half;
          const int first = pair / lanes, second = (pair + half) / lanes;
          const float c = Load(cs, token * rot + pair);
          const float s = Load(cs, token * rot + half + pair);
          if (fp16) {
            const float hc = Round(DType::kF16, c), hs = Round(DType::kF16, s);
            value = col < half ? Round(DType::kF16, norm[first] * hc) - Round(DType::kF16, norm[second] * hs)
                               : Round(DType::kF16, norm[first] * hs) + Round(DType::kF16, norm[second] * hc);
          } else {
            value = col < half ? norm[first] * c - norm[second] * s
                               : norm[first] * s + norm[second] * c;
          }
        }
        Store(dst, out_base + col, value);
        if (query) Store(go, out_base + col, Load(qs, src_base + dim + col));
      }
      });
    };
    const auto event = fp16 ? launch.operator()<true>() : launch.operator()<false>();
    RecordProfileEvent(q, "attn_qk_norm_rope_gate_subgroup", event);
    finish_probe(event);
    return;
  }
  const auto launch = [&]<bool Producer>() {
    return NativeQueue(q).parallel_for(
      sycl::range<1>(tokens * (hq + hk)), [=](sycl::id<1> item) {
    constexpr bool fp16 = Producer;
    const int64_t token = item[0] / (hq + hk), head = item[0] % (hq + hk);
    const bool query = head < hq;
    const int64_t local_head = query ? head : head - hq;
    const View src = query ? qs : ks, weight = query ? qw : kw,
               dst = query ? qo : ko;
    const int64_t src_base = token * src.stride[0] +
        local_head * (query ? 2 * dim : dim);
    const int64_t out_base = (token * (query ? hq : hk) + local_head) * dim;
    float sum = 0;
    for (int64_t i = 0; i < dim; ++i) {
      const float value = Load(src, src_base + i);
      sum += value * value;
    }
    const float mean = fp16 && dim == 256
        ? ProducerQkMean256Scalar(src, src_base, tokens * (query ? hq : hk))
        : sum / static_cast<float>(dim);
    const float variance = mean + eps;
    const float inv = fp16 ? sycl::rsqrt(variance) : 1.0f / sycl::sqrt(variance);
    if (probe) {
      probe[3 * item[0]] = mean; probe[3 * item[0] + 1] = variance; probe[3 * item[0] + 2] = inv;
    }
    for (int64_t i = 0; i < dim; ++i) {
      const auto norm = [&](int64_t col) {
        const float w = Load(weight, col);
        const float value = fp16 ? ProducerQkNormValue(Load(src, src_base + col), inv, w, gemma)
                                 : Load(src, src_base + col) * inv * (gemma ? 1.0f + w : w);
        return fp16 ? Round(DType::kF16, value) : value;
      };
      float value = norm(i);
      if (i < rot) {
        const int64_t pair = i < half ? i : i - half;
        const float first = norm(pair), second = norm(pair + half);
        const float c = Load(cs, token * rot + pair);
        const float s = Load(cs, token * rot + half + pair);
        if (fp16) {
          const float hc = Round(DType::kF16, c), hs = Round(DType::kF16, s);
          // Torch's half pointwise multiply uses F32 opmath before the F16
          // store. Without an explicit F32 operation this scalar kernel
          // lowered these products to half FMul and lost the observed boundary.
          const auto product = [](float a, float b) {
            return Round(DType::kF16, ProducerFloatProduct(a, b));
          };
          value = i < half ? product(first, hc) - product(second, hs)
                           : product(first, hs) + product(second, hc);
        } else value = i < half ? first * c - second * s : first * s + second * c;
      }
      Store(dst, out_base + i, value);
      if (query) Store(go, out_base + i, Load(qs, src_base + dim + i));
    }
    });
  };
  const auto event = fp16 ? launch.operator()<true>() : launch.operator()<false>();
  RecordProfileEvent(q, "attn_qk_norm_rope_gate", event);
  finish_probe(event);
}
void RopeNeoxKernel(Queue& q, Tensor& queries, Tensor& keys, const Tensor& positions, const RopeArgs& args) {
  TraceXpuOp(OpId::kRopeNeox, q, {&queries, &keys, &positions});
  VT_CHECK(NativeQueue(q).get_device().has(sycl::aspect::fp64), "XPU legacy RoPE requires FP64 frequency math");
  if (!args.rotary_dim) return;
  const View qs(queries), ks(keys), pos(positions);
  const int64_t hq = queries.shape[1], hk = keys.shape[1], half = args.rotary_dim / 2;
  const auto event = NativeQueue(q).parallel_for(sycl::range<1>(queries.shape[0] * (hq + hk) * half), [=](sycl::id<1> item) {
    const int64_t pair = item[0] % half, head = (item[0] / half) % (hq + hk), token = item[0] / (half * (hq + hk));
    const double angle = double(Position(pos, token)) * Frequency(pair, args);
    const float c = float(sycl::cos(angle)), s = float(sycl::sin(angle));
    Rotate(head < hq ? qs : ks, token, head < hq ? head : head - hq, pair, pair + half, c, s);
  });
  RecordProfileEvent(q, "rope_neox", event);
}
void RopeCosSinCacheKernel(Queue& q, Tensor& cache, const Tensor& positions, const RopeArgs& args) {
  TraceXpuOp(OpId::kRopeCosSinCache, q, {&cache, &positions});
  VT_CHECK(NativeQueue(q).get_device().has(sycl::aspect::fp64), "XPU legacy RoPE requires FP64 frequency math");
  if (!args.rotary_dim) return;
  const View dst(cache), pos(positions);
  const auto rot = args.rotary_dim, half = rot / 2;
  const bool multi_axis = positions.rank == 2;
  const int64_t tokens = cache.shape[0];
  const auto event = NativeQueue(q).parallel_for(sycl::range<1>(cache.shape[0] * half), [=](sycl::id<1> item) {
    const int64_t row = item[0] / half, pair = item[0] % half;
    int axis = 0;
    if (multi_axis) {
      if (args.mrope_interleaved) {
        // The executing Qwen3.5 M-RoPE selects H/W every third pair;
        // pairs beyond each section's extent retain the temporal axis.
        if (pair % 3 == 1 && pair < int64_t(args.mrope_section[1]) * 3) axis = 1;
        else if (pair % 3 == 2 && pair < int64_t(args.mrope_section[2]) * 3) axis = 2;
      } else if (pair >= args.mrope_section[0]) {
        axis = pair < int64_t(args.mrope_section[0]) + args.mrope_section[1] ? 1 : 2;
      }
    }
    const int64_t p = Position(pos, int64_t(axis) * tokens + row);
    float c, s;
    if (args.linear_scaling_factor > 0) {
      // Round both pow and its reciprocal to F32 before the F32 angle.
      // B70 FP64 avoids the larger device float-pow approximation.
      const float exponent = float(2 * pair) / float(rot);
      // The pinned Torch XPU regeneration uses F32 power and reciprocal.
      // Its half-coefficient boundary distinguishes this from the legacy
      // high-precision power. Cache initialization device is not witnessed.
      const float power = args.fp16_intermediates
          ? sycl::pow(args.base, exponent)
          : float(sycl::pow(double(args.base), double(exponent)));
      // Request round-to-nearest explicitly at the producer F32 reciprocal
      // boundary. Ordinary division (even widened to F64) did not preserve
      // that boundary in the fused cache kernel on the pinned B70 toolchain.
      const float inv = args.fp16_intermediates
          ? sycl::ext::intel::math::fdiv_rn(1.0f, power)
          : float(1.0 / double(power));
      const float angle = (float(p) / args.linear_scaling_factor) * inv;
      c = sycl::cos(angle); s = sycl::sin(angle);
    } else {
      const double angle = double(p) * Frequency(pair, args);
      c = float(sycl::cos(angle)); s = float(sycl::sin(angle));
    }
    Store(dst, row * rot + pair, c); Store(dst, row * rot + half + pair, s);
  });
  RecordProfileEvent(q, "rope_cache_produce", event);
}
void RopeFromCacheKernel(Queue& q, Tensor& queries, Tensor* keys, const Tensor& positions,
                          const Tensor& cache, const RopeArgs& args) {
  TraceXpuOp(OpId::kRopeFromCache, q, {&queries, keys, &positions, &cache});
  VT_CHECK(positions.rank == 1, "XPU text RoPE does not yet implement multi-axis vision positions");
  if (!args.rotary_dim) return;
  const View qs(queries), ks(keys ? *keys : queries), pos(positions), cs(cache);
  const auto tokens = queries.shape[0], hq = queries.shape[1], hk = keys ? keys->shape[1] : 0;
  const auto half = args.rotary_dim / 2, rot = args.rotary_dim;
  const auto count = cache.shape[0];
  CheckDeviceMetadata(q, [=] {
    for (int64_t i = 0; i < tokens; ++i) if (Position(pos, i) < 0 || Position(pos, i) >= count) return false;
    return true;
  }, "XPU RoPE position outside cache", {&positions});
  const bool neox = args.is_neox_style;
  const bool fp16 = args.fp16_intermediates;
  VT_CHECK(!fp16 || (queries.dtype == DType::kF16 && (!keys || keys->dtype == DType::kF16)),
           "XPU FP16 RoPE intermediates require FP16 Q/K");
  const auto event = NativeQueue(q).parallel_for(sycl::range<1>(tokens * (hq + hk) * half), [=](sycl::id<1> item) {
    const int64_t pair = item[0] % half, head = (item[0] / half) % (hq + hk), token = item[0] / (half * (hq + hk));
    const auto base = Position(pos, token) * rot;
    Rotate(head < hq ? qs : ks, token, head < hq ? head : head - hq,
           neox ? pair : 2 * pair, neox ? pair + half : 2 * pair + 1,
           Load(cs, base + pair), Load(cs, base + half + pair), fp16);
  });
  RecordProfileEvent(q, "rope_cache_consume", event);
}
void VisionRopeApplyKernel(Queue& q, Tensor& queries, Tensor& keys, const Tensor& cache) {
  TraceXpuOp(OpId::kVisionRopeApply, q, {&queries, &keys, &cache});
  const View qs(queries), ks(keys), cs(cache);
  const int64_t heads = queries.shape[1], dim = queries.shape[2];
  const int64_t half = dim / 2;
  const auto event = NativeQueue(q).parallel_for(sycl::range<1>(queries.Numel()), [=](sycl::id<1> item) {
    const int64_t pair = item[0] % half;
    const int64_t head = (item[0] / half) % (2 * heads);
    const int64_t token = item[0] / (half * (2 * heads));
    Rotate(head < heads ? qs : ks, token, head < heads ? head : head - heads,
           pair, pair + half, Load(cs, token * dim + pair),
           Load(cs, token * dim + half + pair), true);
  });
  RecordProfileEvent(q, "vision_rope_apply", event);
}

namespace {
template<bool Fp8>
void CacheWrite(Queue& q, const Tensor& keys, const Tensor& values, Tensor& key_cache,
                Tensor& value_cache, const Tensor& slots, float k_scale, float v_scale) {
  const int64_t count = slots.Numel(), page = key_cache.shape[1], blocks = key_cache.shape[0];
  const auto elements = keys.shape[1] * keys.shape[2];
  const auto head_dim = keys.shape[2];
  const auto* ids = static_cast<const int64_t*>(slots.data);
  CheckDeviceMetadata(q, [=] {
    for (int64_t t = 0; t < count; ++t) if (ids[t] >= blocks * page) return false;
    return true;
  }, "XPU KV slot outside cache", {&slots});
  if (!count || !elements) return;
  const View ks(keys), vs(values), kc(key_cache), vc(value_cache);
  const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<int> keep(1, h);
    h.parallel_for(sycl::nd_range<1>(count * 128, 128), [=](sycl::nd_item<1> item) {
      const int64_t token = item.get_group(0), slot = ids[token];
      if (item.get_local_id(0) == 0) {
        // Last writer wins, checked once per token rather than per K/V value.
        int active = slot >= 0;
        for (int64_t later = token + 1; active && later < count; ++later)
          if (ids[later] == slot) active = 0;
        keep[0] = active;
      }
      item.barrier(sycl::access::fence_space::local_space);
      if (!keep[0]) return;
      const auto block = slot / page, offset = slot % page;
      for (int64_t col = item.get_local_id(0); col < elements; col += 128) {
        const auto head = col / head_dim, channel = col % head_dim;
        const auto kd = block * kc.stride[0] + offset * kc.stride[1] + head * kc.stride[2] + channel;
        const auto vd = block * vc.stride[0] + offset * vc.stride[1] + head * vc.stride[2] + channel;
        if constexpr (Fp8) {
          static_cast<uint8_t*>(kc.data)[kd] = EncodeE4M3(Load(ks, token * ks.stride[0] + col) / k_scale);
          static_cast<uint8_t*>(vc.data)[vd] = EncodeE4M3(Load(vs, token * vs.stride[0] + col) / v_scale);
        } else {
          CopyElement(kc, kd, ks, token * ks.stride[0] + col);
          CopyElement(vc, vd, vs, token * vs.stride[0] + col);
        }
      }
    });
  });
  RecordProfileEvent(q, Fp8 ? "kv_write_fp8" : "kv_write", event);
}
}
void ReshapeAndCacheKernel(Queue& q, const Tensor& keys, const Tensor& values, Tensor& key_cache,
                            Tensor& value_cache, const Tensor& slots) {
  TraceXpuOp(OpId::kReshapeAndCache, q, {&keys, &values, &key_cache, &value_cache, &slots});
  CacheWrite<false>(q, keys, values, key_cache, value_cache, slots, 1, 1);
}
void ReshapeAndCacheFp8Kernel(Queue& q, const Tensor& keys, const Tensor& values, Tensor& key_cache,
                               Tensor& value_cache, const Tensor& slots, Fp8KVCacheDataType kind,
                               float k_scale, float v_scale) {
  TraceXpuOp(OpId::kReshapeAndCacheFp8, q, {&keys, &values, &key_cache, &value_cache, &slots});
  VT_CHECK(kind == Fp8KVCacheDataType::kFp8E4M3 && std::isfinite(k_scale) && std::isfinite(v_scale),
           "XPU FP8 KV requires E4M3 and finite positive scales");
  CacheWrite<true>(q, keys, values, key_cache, value_cache, slots, k_scale, v_scale);
}

void AttentionKernel(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                     const Tensor& value, const AttentionArgs& args) {
  TraceXpuOp(OpId::kAttention, q, {&out, &query, &key, &value});
  const int64_t tokens = query.shape[0];
  if (!tokens) return;
  VT_CHECK(tokens <= std::numeric_limits<int32_t>::max(),
           "XPU attention token count exceeds I32 metadata");
  // A contiguous sequence is one page. Borrow Q/K/V directly and reuse the
  // native paged implementation, including its output-alias handling. No KV
  // copy or quadratic score allocation is needed for this unpaged entry.
  auto keys = Tensor::Contiguous(key.data, key.dtype, key.device,
                                {1, tokens, key.shape[1], key.shape[2]});
  auto values = Tensor::Contiguous(value.data, value.dtype, value.device,
                                  {1, tokens, value.shape[1], value.shape[2]});
  Scratch metadata(q.device, 4 * sizeof(int32_t));
  auto* data = static_cast<int32_t*>(metadata.data);
  const int32_t length = static_cast<int32_t>(tokens);
  NativeQueue(q).single_task([=] {
    data[0] = 0; data[1] = length; data[2] = 0; data[3] = length;
  });
  auto table = Tensor::Contiguous(data, DType::kI32, q.device, {1, 1});
  auto lengths = Tensor::Contiguous(data + 1, DType::kI32, q.device, {1});
  auto offsets = Tensor::Contiguous(data + 2, DType::kI32, q.device, {2});
  PagedAttentionArgs paged;
  paged.scale = args.scale;
  paged.causal = args.causal;
  vt::PagedAttention(q, out, query, keys, values, table, lengths, offsets, paged);
  // Scratch release drains its last use, including any selected fast kernel.
}

void PagedAttentionKernel(Queue& q, Tensor& out, const Tensor& query, const Tensor& key_cache,
                           const Tensor& value_cache, const Tensor& block_table,
                           const Tensor& seq_lens, const Tensor& query_start_loc,
                           const PagedAttentionArgs& args) {
  TraceXpuOp(OpId::kPagedAttention, q, {&out, &query, &key_cache, &value_cache,
                                          &block_table, &seq_lens, &query_start_loc});
  VT_CHECK(args.kv_cache_dtype == Fp8KVCacheDataType::kAuto ||
               (args.kv_cache_dtype == Fp8KVCacheDataType::kFp8E4M3 &&
                std::isfinite(args.k_scale) && std::isfinite(args.v_scale)),
           "XPU paged attention requires float KV or E4M3 with finite positive scales");
  const auto tokens = query.shape[0], heads = query.shape[1], dim = query.shape[2];
  const auto page = key_cache.shape[1], blocks = key_cache.shape[0], ratio = heads / key_cache.shape[2];
  const auto requests = seq_lens.Numel(), columns = block_table.shape[1];
  const auto bt_row = block_table.stride[0], bt_col = block_table.stride[1];
  VT_CHECK(page > 0 && dim > 0, "XPU paged attention requires positive page/head size");
  const auto* lengths = static_cast<const int32_t*>(seq_lens.data);
  const auto* offsets = static_cast<const int32_t*>(query_start_loc.data);
  const auto* table = static_cast<const int32_t*>(block_table.data);
  const char* setting = std::getenv("VT_XPU_ATTENTION");
  const std::string_view mode = setting ? setting : "auto";
  const int64_t max_length = args.max_seq_len;
  const auto check_metadata = [&](int64_t packed_verify_rows) {
    CheckDeviceMetadata(q, [=] {
      if (offsets[0] != 0 || offsets[requests] != tokens) return false;
      for (int64_t r = 0; r < requests; ++r) {
        const int64_t first = offsets[r], end = offsets[r + 1], length = lengths[r];
        if (first < 0 || end < first || end > tokens) return false;
        // An admitted packed donor uses a uniform host hint. Prove it against
        // fresh device offsets inside the existing eager/graph metadata check;
        // never trust total-token division or add a per-layer host readback.
        if (packed_verify_rows && (first != r * packed_verify_rows ||
            end != (r + 1) * packed_verify_rows ||
            (max_length > 0 && length > max_length))) return false;
        if (end == first) continue;  // Padded/inactive rows need no valid cache entries.
        if (length < end - first) return false;
        const auto needed = (length + page - 1) / page;
        if (needed > columns) return false;
        for (int64_t b = 0; b < needed; ++b) {
          const auto id = table[r * bt_row + b * bt_col];
          if (id < 0 || id >= blocks) return false;
        }
      }
      return true;
    }, "XPU paged attention invalid sequence offsets, lengths or block table", {&seq_lens, &query_start_loc, &block_table});
  };
  if (!tokens) { check_metadata(0); return; }
  size_t lanes = 1;
  while (lanes < static_cast<uint64_t>(dim)) lanes *= 2;
  VT_CHECK(lanes <= NativeQueue(q).get_device().get_info<sycl::info::device::max_work_group_size>(),
           "XPU paged attention head exceeds workgroup limit");
  const float scale = args.scale, cap = args.logits_soft_cap;
  const float k_scale = args.k_scale, v_scale = args.v_scale;
  const bool causal = args.causal;
  const int64_t left = args.window_size ? args.window_size->left : -1;
  const int64_t right = args.window_size ? args.window_size->right : -1;
  WithOutput(q, out, {&query, &key_cache, &value_cache, &block_table, &seq_lens, &query_start_loc}, [&](Tensor& target) {
    VT_CHECK(mode == "auto" || mode == "reference" || mode == "split" ||
                 mode == "prefill" || mode == "verify" || mode == "exl3_onednn", "Invalid VT_XPU_ATTENTION");
    const auto device = NativeQueue(q).get_device();
    // Qualified against the pinned checkpoint's answer and probability corpus.
    // E4M3 cache storage is chosen by the caller; auto may use the fast
    // attention kernels without changing that choice or its scales.
    const bool automatic = mode == "auto" && query.shape[1] == 24 && dim == 256 &&
        key_cache.shape[2] == 4 &&
        (key_cache.dtype == DType::kBF16 || key_cache.dtype == DType::kF16 ||
         (key_cache.dtype == DType::kI8 &&
          args.kv_cache_dtype == Fp8KVCacheDataType::kFp8E4M3)) &&
        device.has(sycl::aspect::ext_intel_device_id) &&
        device.get_info<sycl::ext::intel::info::device::device_id>() == 57891 &&
        std::string_view(__VERSION__) == "Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724)" &&
        device.get_info<sycl::info::device::driver_version>() == "1.17.39758+10" &&
        device.get_platform().get_info<sycl::info::platform::version>() == "1.17";
    int64_t packed_verify_rows = 0;
#ifdef VLLM_CPP_XPU_XE2_VERIFY
    const char* verify_setting = std::getenv("VT_XPU_XE2_VERIFY");
    const bool packed_requested = mode == "verify" ||
        (automatic && verify_setting && std::string_view(verify_setting) == "1");
    if (requests > 1 && packed_requested &&
        CanUsePagedAttentionXe2Verify(q, target, query, key_cache, value_cache,
                                     block_table, seq_lens, query_start_loc, args))
      packed_verify_rows = PagedAttentionXe2VerifyQueryLength(tokens, requests, args.query_start_loc_host);
#endif
    // Generic fallback reads fresh lengths and has no packed maximum-length
    // specialization. Do not bake a capture-time bound into that route just
    // because a caller requested a fast kernel that cannot admit its tensors.
    check_metadata(packed_verify_rows);
    bool onednn = false;
#ifdef VLLM_CPP_XPU_ONEDNN
    if (mode == "exl3_onednn" || (automatic && tokens > 128))
      onednn = PagedAttentionExl3OneDnnKernel(q, target, query, key_cache, value_cache,
          block_table, seq_lens, query_start_loc, args);
#endif
    bool verify = false;
#ifdef VLLM_CPP_XPU_XE2_VERIFY
    // Original short C1 decode uses FP16 probabilities and XMX P*V, with
    // one split below16 KV tiles. The same native donor is already used for
    // optional packed verification; its own admission guards both routes.
    if ((mode == "verify" || automatic) && tokens == 1)
      verify = PagedAttentionXe2DecodeKernel(
          q, target, query, key_cache, value_cache,
          block_table, seq_lens, query_start_loc, args);
    if (!verify && (mode == "verify" || (automatic && verify_setting &&
                              std::string_view(verify_setting) == "1")))
      verify = PagedAttentionXe2VerifyKernel(
          q, target, query, key_cache, value_cache,
          block_table, seq_lens, query_start_loc, args);
#endif
    const bool try_split = !onednn && !verify &&
        (automatic || mode == "split" || mode == "prefill" || mode == "verify" || mode == "exl3_onednn");
    const bool split = try_split && PagedAttentionSplitKernel(
        q, target, query, key_cache, value_cache,
        block_table, seq_lens, query_start_loc, args);
#ifdef VLLM_CPP_XPU_XE2_PREFILL
    const bool xe2 = !onednn && !verify && !split && (automatic || mode == "prefill") &&
        PagedAttentionXe2PrefillKernel(q, target, query, key_cache, value_cache,
                                      block_table, seq_lens, query_start_loc, args);
#else
    const bool xe2 = false;
#endif
    const bool try_prefill = !onednn && !verify && !split && !xe2 &&
        (automatic || mode == "prefill");
    const bool prefill = try_prefill && PagedAttentionPrefillKernel(
        q, target, query, key_cache, value_cache,
        block_table, seq_lens, query_start_loc, args);
    if (const char* trace = std::getenv("VT_XPU_TRACE_FAST_PATH");
        trace != nullptr && trace[0] == '1' && trace[1] == '\0') {
      const char* reason = onednn || verify || split || xe2 || prefill ? "eligible" :
          mode == "reference" ? "mode_reference" :
          mode == "auto" && !automatic ? "stack_gate_or_shape" :
          "kernel_declined";
      std::fprintf(stderr,
                   "{\"event\":\"xpu_fast_path\",\"operator\":\"paged_attention\","
                   "\"selected\":\"%s\",\"reason\":\"%s\","
                   "\"mode\":\"%.*s\",\"auto_eligible\":%s,"
                   "\"tokens\":%lld}\n",
                   onednn ? "exl3_onednn" : verify ? "xe2_verify" : split ? "split" : xe2 ? "xe2_prefill" :
                   prefill ? "prefill" : "reference", reason,
                   static_cast<int>(mode.size()), mode.data(),
                   automatic ? "true" : "false", static_cast<long long>(tokens));
    }
    if (onednn || verify || split || xe2 || prefill) return;
    const View qs(query), kc(key_cache), vc(value_cache), dst(target);
    const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
      sycl::local_accessor<float, 1> partial(sycl::range<1>(lanes), h);
      // One group per query/head; each lane owns one value component. Scores
      // reduce in F32, and online softmax never materializes a scores matrix.
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(tokens * heads * lanes), sycl::range<1>(lanes)),
                     [=](sycl::nd_item<1> item) {
        const auto lane = item.get_local_id(0);
        const bool active = lane < static_cast<size_t>(dim);
        const int64_t token = item.get_group(0) / heads, head = item.get_group(0) % heads;
        int64_t request = 0;
        while (token >= offsets[request + 1]) ++request;
        const int64_t position = lengths[request] - (offsets[request + 1] - offsets[request])
                                   + token - offsets[request];
        const int64_t first = left < 0 ? 0 : sycl::max(int64_t{0}, position - left);
        int64_t last = causal ? position : int64_t(lengths[request]) - 1;
        if (right >= 0) last = sycl::min(last, position + right);
        const auto qbase = (token * heads + head) * dim;
        const float qvalue = active ? Load(qs, qbase + lane) : 0;
        float maximum = -std::numeric_limits<float>::infinity(), denominator = 0, accumulator = 0;
        for (int64_t key = first; key <= last; ++key) {
          const int64_t block = table[request * bt_row + (key / page) * bt_col];
          const auto kbase = block * kc.stride[0] + (key % page) * kc.stride[1] + (head / ratio) * kc.stride[2];
          const auto vbase = block * vc.stride[0] + (key % page) * vc.stride[1] + (head / ratio) * vc.stride[2];
          partial[lane] = active ? qvalue * LoadKV(kc, kbase + lane, k_scale) : 0;
          item.barrier(sycl::access::fence_space::local_space);
          for (size_t step = lanes / 2; step > 0; step /= 2) {
            if (lane < step) partial[lane] += partial[lane + step];
            item.barrier(sycl::access::fence_space::local_space);
          }
          float score = partial[0] * scale;
          if (cap > 0) score = cap * sycl::tanh(score / cap);
          const float next = sycl::max(maximum, score);
          const float old_scale = sycl::exp(maximum - next), probability = sycl::exp(score - next);
          denominator = denominator * old_scale + probability;
          if (active) accumulator = accumulator * old_scale + probability * LoadKV(vc, vbase + lane, v_scale);
          maximum = next;
          item.barrier(sycl::access::fence_space::local_space);
        }
        if (active) Store(dst, qbase + lane, accumulator / denominator);
      });
    });
    RecordProfileEvent(q, "attention_reference", event);
  });
}
}  // namespace vt::xpu
