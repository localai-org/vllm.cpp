#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_weights.h"

namespace vllm {

// The reference oneDNN U4 layout: packed int32 words [N,K/8], with the eight
// low-to-high nibbles of each word covering consecutive K values. Scales stay
// [K/128,N]. Disk qzeros and g_idx are checked before being discarded.
struct Gptq4Weight {
  static constexpr int kPackingVersion = 1;
  int64_t k = 0;
  int64_t n = 0;
  int group_size = 128;
  int packing_version = kPackingVersion;
  int disk_zero_offset = 1;
  mutable bool resident = false;
  mutable vt::Device resident_device;
  OwnedTensor qweight;
  OwnedTensor scales;
  OwnedTensor zero_point;  // scalar effective I8 value 8

  size_t ResidentBytes() const;
};

struct Gptq4ResidentViews {
  vt::Tensor qweight;
  vt::Tensor scales;
  vt::Tensor zero_point;
};

// Upload once on the supplied VT queue, then optionally discard host buffers
// after the three asynchronous copies have completed. Repeated calls reuse
// the same owner; moving it to a second device is rejected.
Gptq4ResidentViews PrepareGptq4Resident(const Gptq4Weight& weight,
                                        vt::Queue& queue,
                                        bool release_host = true);

// Load one projection, or concatenate projections in output order directly
// into a single owner. The caller supplies the expected dimensions from the
// model configuration; unexpected checkpoint shapes fail before upload.
Gptq4Weight LoadGptq4Weight(const TensorResolver& get,
                            const std::string& projection, int64_t k,
                            int64_t n);
Gptq4Weight LoadMergedGptq4Weight(
    const TensorResolver& get, const std::vector<std::string>& projections,
    int64_t k, const std::vector<int64_t>& output_widths);

// One-time symmetric G128 packing of dense checkpoint Linear weights [N,K].
// Matches the production draft's q=round(weight/(maxabs/7)), clamp[-8,7],
// q+8 nibble layout. Sources are concatenated in output-row order.
Gptq4Weight QuantizeMergedGptq4Weight(
    const std::vector<StTensor>& sources, int64_t k);
Gptq4Weight QuantizeGptq4Weight(const StTensor& source, int64_t k);

}  // namespace vllm
