#include "vllm/model_executor/model_loader/gptq4_weight.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <string>

#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vt/backend.h"
#include "vt/unaligned.h"

namespace vllm {
namespace {

constexpr int64_t kGroupSize = 128;

void CheckTensor(const StTensor& tensor, const std::string& name,
                 const char* dtype, const std::vector<int64_t>& shape,
                 size_t bytes) {
  VT_CHECK(tensor.dtype == dtype, "gptq4: wrong dtype for " + name);
  VT_CHECK(tensor.shape == shape, "gptq4: wrong shape for " + name);
  VT_CHECK(tensor.nbytes == bytes && tensor.data != nullptr,
           "gptq4: wrong byte span for " + name);
}

size_t CheckedBytes(int64_t rows, int64_t cols, size_t element_bytes) {
  VT_CHECK(rows > 0 && cols > 0 &&
               static_cast<uint64_t>(rows) <=
                   std::numeric_limits<size_t>::max() / element_bytes /
                       static_cast<uint64_t>(cols),
           "gptq4: invalid or overflowing geometry");
  return static_cast<size_t>(rows) * static_cast<size_t>(cols) * element_bytes;
}

void CopyProjection(const TensorResolver& get, const std::string& projection,
                    int64_t k, int64_t n, int64_t n_total, int64_t n_offset,
                    Gptq4Weight& result) {
  const std::string qname = projection + ".qweight";
  const std::string sname = projection + ".scales";
  const std::string zname = projection + ".qzeros";
  const std::string gname = projection + ".g_idx";
  const StTensor& q = get(qname);
  const StTensor& s = get(sname);
  const StTensor& z = get(zname);
  const StTensor& g = get(gname);
  CheckTensor(q, qname, "I32", {k / 8, n}, CheckedBytes(k / 8, n, 4));
  CheckTensor(s, sname, "F16", {k / kGroupSize, n},
              CheckedBytes(k / kGroupSize, n, 2));
  CheckTensor(z, zname, "I32", {k / kGroupSize, n / 8},
              CheckedBytes(k / kGroupSize, n / 8, 4));
  CheckTensor(g, gname, "I32", {k}, CheckedBytes(k, 1, 4));

  for (int64_t i = 0; i < k; ++i) {
    const int32_t group = vt::LoadUnaligned<int32_t>(g.data + i * 4);
    VT_CHECK(group == i / kGroupSize,
             "gptq4: non-identity g_idx for " + projection);
  }
  // GPTQ v1 stores effective zero minus one. A symmetric effective zero of 8
  // therefore has the raw nibble 7 at every output and group.
  for (size_t i = 0; i < z.nbytes; i += 4) {
    VT_CHECK(vt::LoadUnaligned<uint32_t>(z.data + i) == 0x77777777u,
             "gptq4: qzeros do not encode effective zero 8 for " + projection);
  }
  for (size_t i = 0; i < s.nbytes; i += 2) {
    const uint16_t bits = vt::LoadUnaligned<uint16_t>(s.data + i);
    const float scale = vt::F16ToF32(bits);
    VT_CHECK(scale > 0.0f && scale < std::numeric_limits<float>::infinity(),
             "gptq4: non-positive or non-finite scale for " + projection);
  }

  const size_t words_per_output = static_cast<size_t>(k / 8);
  for (int64_t output = 0; output < n; ++output) {
    auto* dst = result.qweight.bytes.data() +
                (static_cast<size_t>(n_offset + output) * words_per_output) * 4;
    for (size_t word = 0; word < words_per_output; ++word) {
      std::memcpy(dst + word * 4,
                  q.data + (word * static_cast<size_t>(n) + output) * 4, 4);
    }
  }
  for (int64_t group = 0; group < k / kGroupSize; ++group) {
    std::memcpy(result.scales.bytes.data() +
                    (static_cast<size_t>(group * n_total + n_offset)) * 2,
                s.data + (static_cast<size_t>(group * n)) * 2,
                static_cast<size_t>(n) * 2);
  }
  MaybeReleaseSourcePages(q.data, q.nbytes);
  MaybeReleaseSourcePages(s.data, s.nbytes);
  MaybeReleaseSourcePages(z.data, z.nbytes);
  MaybeReleaseSourcePages(g.data, g.nbytes);
}

}  // namespace

Gptq4Weight LoadMergedGptq4Weight(
    const TensorResolver& get, const std::vector<std::string>& projections,
    int64_t k, const std::vector<int64_t>& output_widths) {
  VT_CHECK(!projections.empty() && projections.size() == output_widths.size(),
           "gptq4: projection names and widths must be nonempty and aligned");
  VT_CHECK(k > 0 && k % kGroupSize == 0,
           "gptq4: K must be positive and divisible by 128");
  int64_t n_total = 0;
  for (int64_t n : output_widths) {
    VT_CHECK(n > 0 && n % 8 == 0 &&
                 n_total <= std::numeric_limits<int64_t>::max() - n,
             "gptq4: output widths must be positive, divisible by 8, and fit");
    n_total += n;
  }
  CheckedBytes(n_total, k / 8, 4);
  CheckedBytes(k / kGroupSize, n_total, 2);
  Gptq4Weight result;
  result.k = k;
  result.n = n_total;
  result.qweight = dense_loaders::MakeOwned(vt::DType::kI32, {n_total, k / 8});
  result.scales = dense_loaders::MakeOwned(vt::DType::kF16,
                                          {k / kGroupSize, n_total});
  result.zero_point = dense_loaders::MakeOwned(vt::DType::kI8, {1});
  result.zero_point.bytes.data()[0] = 8;
  int64_t offset = 0;
  for (size_t i = 0; i < projections.size(); ++i) {
    CopyProjection(get, projections[i], k, output_widths[i], n_total, offset,
                   result);
    offset += output_widths[i];
  }
  return result;
}

Gptq4Weight LoadGptq4Weight(const TensorResolver& get,
                            const std::string& projection, int64_t k,
                            int64_t n) {
  return LoadMergedGptq4Weight(get, {projection}, k, {n});
}

Gptq4Weight QuantizeMergedGptq4Weight(
    const std::vector<StTensor>& sources, int64_t k) {
  VT_CHECK(!sources.empty() && k > 0 && k % kGroupSize == 0,
           "gptq4: dense source K must be positive and divisible by 128");
  int64_t n_total = 0;
  for (const StTensor& source : sources) {
    VT_CHECK((source.dtype == "BF16" || source.dtype == "F16") &&
                 source.shape.size() == 2 && source.shape[0] > 0 &&
                 source.shape[1] == k && source.data != nullptr &&
                 source.nbytes == CheckedBytes(source.shape[0], k, 2) &&
                 n_total <= std::numeric_limits<int64_t>::max() - source.shape[0],
             "gptq4: dense source must be finite BF16/F16 Linear [N,K]");
    n_total += source.shape[0];
  }
  VT_CHECK(n_total % 8 == 0,
           "gptq4: merged dense output width must be divisible by 8");
  CheckedBytes(n_total, k / 8, 4);
  CheckedBytes(k / kGroupSize, n_total, 2);
  Gptq4Weight result;
  result.k = k;
  result.n = n_total;
  result.qweight = dense_loaders::MakeOwned(vt::DType::kI32,
                                            {n_total, k / 8});
  result.scales = dense_loaders::MakeOwned(vt::DType::kF16,
                                           {k / kGroupSize, n_total});
  result.zero_point = dense_loaders::MakeOwned(vt::DType::kI8, {1});
  result.zero_point.bytes.data()[0] = 8;
  int64_t output_offset = 0;
  for (const StTensor& source : sources) {
    for (int64_t row = 0; row < source.shape[0]; ++row) {
      const int64_t out_row = output_offset + row;
      for (int64_t group = 0; group < k / kGroupSize; ++group) {
        float values[kGroupSize];
        float maxabs = 0.0F;
        for (int64_t i = 0; i < kGroupSize; ++i) {
          const size_t index = static_cast<size_t>(row * k +
                                                   group * kGroupSize + i);
          const uint16_t bits = vt::LoadUnaligned<uint16_t>(source.data + 2 * index);
          const float value = source.dtype == "F16" ? vt::F16ToF32(bits)
                                                       : vt::BF16ToF32(bits);
          VT_CHECK(std::isfinite(value), "gptq4: non-finite dense source weight");
          values[i] = value;
          maxabs = std::max(maxabs, std::abs(value));
        }
        // The production recipe uses maxabs/7 in FP32 and stores FP16 scales.
        // An all-zero group has no defined division; encode exact zero with a
        // positive FP16 scale so oneDNN still accepts the packed owner.
        const float scale = maxabs == 0.0F ? 0x1p-24F : maxabs / 7.0F;
        const uint16_t scale_bits = vt::F32ToF16(scale);
        VT_CHECK(vt::F16ToF32(scale_bits) > 0.0F &&
                     std::isfinite(vt::F16ToF32(scale_bits)),
                 "gptq4: dense source scale is outside FP16 range");
        std::memcpy(result.scales.bytes.data() +
                        2 * static_cast<size_t>(group * n_total + out_row),
                    &scale_bits, sizeof(scale_bits));
        for (int64_t word = 0; word < kGroupSize / 8; ++word) {
          uint32_t packed = 0;
          for (int nibble = 0; nibble < 8; ++nibble) {
            const float value = values[word * 8 + nibble];
            const int quantized = maxabs == 0.0F
                                      ? 0
                                      : std::clamp(static_cast<int>(std::nearbyint(value / scale)),
                                                   -8, 7);
            packed |= static_cast<uint32_t>(quantized + 8) << (4 * nibble);
          }
          const size_t word_offset = static_cast<size_t>(out_row * (k / 8) +
                                                          group * (kGroupSize / 8) + word);
          std::memcpy(result.qweight.bytes.data() + 4 * word_offset,
                      &packed, sizeof(packed));
        }
      }
    }
    output_offset += source.shape[0];
  }
  return result;
}

Gptq4Weight QuantizeGptq4Weight(const StTensor& source, int64_t k) {
  return QuantizeMergedGptq4Weight({source}, k);
}

size_t Gptq4Weight::ResidentBytes() const {
  if (!resident) return 0;
  return CheckedBytes(n, k / 8, 4) +
         CheckedBytes(k / group_size, n, 2) + 1;
}

Gptq4ResidentViews PrepareGptq4Resident(const Gptq4Weight& weight,
                                        vt::Queue& queue,
                                        bool release_host) {
  VT_CHECK(weight.k > 0 && weight.n > 0 && weight.group_size == kGroupSize &&
               weight.packing_version == Gptq4Weight::kPackingVersion &&
               weight.disk_zero_offset == 1,
           "gptq4: unsupported or empty packed owner");
  VT_CHECK(!weight.resident || weight.resident_device == queue.device,
           "gptq4: packed owner is already resident on a different device");
  const auto views = [&weight, &queue] {
    return Gptq4ResidentViews{
        weight.qweight.ViewOn(weight.qweight.d_dev.get(), queue.device),
        weight.scales.ViewOn(weight.scales.d_dev.get(), queue.device),
        weight.zero_point.ViewOn(weight.zero_point.d_dev.get(), queue.device)};
  };
  if (weight.resident) {
    VT_CHECK(weight.qweight.d_dev && weight.scales.d_dev &&
                 weight.zero_point.d_dev,
             "gptq4: incomplete packed resident");
    if (release_host) {
      weight.qweight.ReleaseHost();
      weight.scales.ReleaseHost();
      weight.zero_point.ReleaseHost();
    }
    return views();
  }
  auto& backend = vt::GetBackend(queue.device.type);
  const auto upload = [&backend, &queue](const OwnedTensor& tensor) {
    if (tensor.d_dev) return;
    VT_CHECK(tensor.HasHostBytes(),
             "gptq4: host bytes were released before device upload");
    const size_t bytes = tensor.bytes.size();
    void* raw = backend.Alloc(bytes);
    auto* backend_ptr = &backend;
    std::shared_ptr<void> device(raw, [backend_ptr](void* p) {
      backend_ptr->Free(p);
    });
    backend.Copy(queue, raw, tensor.bytes.data(), bytes);
    load_stats::AddDeviceUpload(bytes);
    tensor.d_dev = std::move(device);
  };
  upload(weight.qweight);
  upload(weight.scales);
  upload(weight.zero_point);
  // Once all three copies are queued, the host buffers must stay live until
  // completion. This wait is load-time only; inference reuses the resident.
  backend.Synchronize(queue);
  weight.resident_device = queue.device;
  weight.resident = true;
  if (release_host) {
    weight.qweight.ReleaseHost();
    weight.scales.ReleaseHost();
    weight.zero_point.ReleaseHost();
  }
  return views();
}

}  // namespace vllm
