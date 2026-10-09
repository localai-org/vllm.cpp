#pragma once

#include <cstddef>
#include <cstdint>

#include "vt/device.h"
#include "vt/tensor.h"

namespace dnnl { struct engine; struct stream; }

namespace vt::xpu {

struct Gptq4RuntimeStats {
  size_t engine_count = 0;
  size_t primitive_count = 0;
  size_t scratchpad_allocation_count = 0;
  size_t scratchpad_capacity_bytes = 0;
};

Gptq4RuntimeStats GetGptq4RuntimeStats(int device_index);
void ReleaseGptq4Queue(const Queue& queue);

// Signed row-major INT8 operands; F16 destination may have a larger row
// stride. Apply the static weight scale and binary per-row activation scale
// before the F16 destination rounding, matching the pinned EXL3 DNNL donor.
void Exl3W8A8Matmul(Queue&, Tensor& out, const Tensor& activation,
                   const Tensor& weights, const Tensor& row_scales,
                   const Tensor& weight_scale);

// The EXL3 SDPA graph uses this same pinned device engine, queue stream and
// accounted scratchpad. Fixed contiguous graph layouts are independent of the
// physical paged-cache strides used by the caller's gather.
dnnl::engine& Exl3OneDnnEngine(Queue&);
dnnl::stream& Exl3OneDnnStream(Queue&);
void* Exl3OneDnnScratchpad(Queue&, size_t bytes);

}  // namespace vt::xpu
