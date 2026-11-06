// CPU group32 compressed-tensors repack. Q4_0/Q8_0 scales stay bit-exact.
#pragma once

#include <cstring>
#include <limits>

#include "vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_config.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"

namespace vllm::gemma4_ct {

class Reader {
 public:
  Reader(const TensorResolver& get, const nlohmann::json& raw)
      : get_(get),
        config_(layers::compressed_tensors::Config::FromHfConfigRaw(raw)),
        quant_(layers::compressed_tensors::QuantizationConfigOf(raw)) {
    if (quant_ != nullptr) {
      const auto kv = quant_->find("kv_cache_scheme");
      VT_CHECK(kv == quant_->end() || kv->is_null(),
               "gemma4 CT: kv_cache_scheme is not implemented for the CPU arm");
    }
  }

  OwnedTensor Plain(const std::string& name, const std::vector<int64_t>& shape,
                    bool preserve_f16 = false) const {
    const StTensor& t = get_(name);
    VT_CHECK(t.shape == shape && (t.dtype == "F16" || t.dtype == "BF16"),
             "gemma4 CT: expected F16/BF16 parameter shape for " + name);
    const size_t count = Count(shape, name);
    VT_CHECK(t.data && t.nbytes == count * 2,
             "gemma4 CT: parameter byte-size mismatch for " + name);
    OwnedTensor o = dense_loaders::MakeOwned(
        preserve_f16 && t.dtype == "F16" ? vt::DType::kF16 : vt::DType::kBF16,
        shape);
    if (t.dtype == "BF16" || preserve_f16) {
      std::memcpy(o.bytes.data(), t.data, t.nbytes);
    } else {
      for (size_t i = 0; i < count; ++i) {
        const uint16_t value = vt::F32ToBF16(vt::F16ToF32(
            vt::LoadUnaligned<uint16_t>(t.data + i * 2)));
        std::memcpy(o.bytes.data() + i * 2, &value, 2);
      }
    }
    return o;
  }

  OwnedTensor Linear(const std::string& module, int64_t n, int64_t k) const {
    const auto scheme = config_.Resolve(module);
    if (!scheme.matched || scheme.ignored) {
      OwnedTensor o = Plain(module + ".weight", {n, k});
      o.nk = true;
      return o;
    }
    const auto& w = scheme.weights;
    const auto& group = quant_->at("config_groups").at(scheme.group);
    const auto output = group.find("output_activations");
    VT_CHECK(output == group.end() || output->is_null(),
             "gemma4 CT: quantized output activations are not implemented for " +
                 module);
    VT_CHECK(scheme.format == "pack-quantized" && w.present && w.type == "int" &&
                 (w.num_bits == 4 || w.num_bits == 8) && w.strategy == "group" &&
                 w.group_size == 32 && w.symmetric && !w.dynamic &&
                 w.dynamic_str.empty() &&
                 (w.actorder.empty() || w.actorder == "none") &&
                 !scheme.input_activations.present,
             "gemma4 CT: unsupported quantization scheme for " + module);
    VT_CHECK(n > 0 && k > 0 && n <= std::numeric_limits<int64_t>::max() / k &&
                 k % 32 == 0,
             "gemma4 CT: group32 dimensions required for " + module);
    (void)Count({n, k}, module);
    const StTensor& p = get_(module + ".weight_packed");
    const StTensor& s = get_(module + ".weight_scale");
    const StTensor& sh = get_(module + ".weight_shape");
    const int bits = w.num_bits;
    const int factor = 32 / bits;
    VT_CHECK(p.dtype == "I32" &&
                 p.shape == std::vector<int64_t>({n, k / factor}) && p.data &&
                 p.nbytes == static_cast<size_t>(n * k / factor) * 4,
             "gemma4 CT: packed shape/byte-size mismatch for " + module);
    VT_CHECK(s.dtype == "F16" &&
                 s.shape == std::vector<int64_t>({n, k / 32}) && s.data &&
                 s.nbytes == static_cast<size_t>(n * k / 32) * 2,
             "gemma4 CT: F16 group scale shape/byte-size mismatch for " + module);
    VT_CHECK(sh.dtype == "I32" && sh.shape == std::vector<int64_t>({2}) && sh.data &&
                 sh.nbytes == 8 && vt::LoadUnaligned<int32_t>(sh.data) == n &&
                 vt::LoadUnaligned<int32_t>(sh.data + 4) == k,
             "gemma4 CT: weight_shape mismatch for " + module);
    OwnedTensor o;
    o.dtype = bits == 4 ? vt::DType::kQ4_0 : vt::DType::kQ8_0;
    o.rank = 2;
    o.shape[0] = n;
    o.shape[1] = k;
    o.nk = true;
    const size_t blocks = static_cast<size_t>(n * k / 32);
    const size_t block_bytes = bits == 4 ? 18 : 34;
    VT_CHECK(blocks <= std::numeric_limits<size_t>::max() / block_bytes,
             "gemma4 CT: quant byte-size overflow for " + module);
    o.bytes.resize(blocks * block_bytes);
    // Converter: convert_gguf_q4_0_to_ct.py::_convert_q4_0_blocks_to_ct
    // and _convert_q8_0_blocks_to_ct. Reverse packing without requantization.
    for (size_t b = 0; b < blocks; ++b) {
      uint8_t* dst = o.bytes.data() + b * block_bytes;
      std::memcpy(dst, s.data + b * 2, 2);
      const uint8_t* src = p.data + b * static_cast<size_t>(bits) * 4;
      const auto code = [&](int i) {
        const uint32_t word = vt::LoadUnaligned<uint32_t>(src + (i / factor) * 4);
        return static_cast<uint8_t>((word >> ((i % factor) * bits)) &
                                    ((1u << bits) - 1));
      };
      if (bits == 4) {
        for (int i = 0; i < 16; ++i)
          dst[2 + i] = static_cast<uint8_t>(code(i) | (code(i + 16) << 4));
      } else {
        for (int i = 0; i < 32; ++i) dst[2 + i] = code(i) ^ 0x80u;
      }
    }
    return o;
  }

 private:
  static size_t Count(const std::vector<int64_t>& shape, const std::string& name) {
    size_t n = 1;
    for (int64_t d : shape) {
      VT_CHECK(d > 0 && static_cast<uint64_t>(d) <=
                            std::numeric_limits<size_t>::max() / n / 2,
               "gemma4 CT: invalid/overflowing shape for " + name);
      n *= static_cast<size_t>(d);
    }
    return n;
  }

  const TensorResolver& get_;
  layers::compressed_tensors::Config config_;
  const nlohmann::json* quant_;
};

}  // namespace vllm::gemma4_ct
