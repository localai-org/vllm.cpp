#pragma once
#include "vt/ops.h"

namespace vt::xpu {
void VisionAttentionDenseFlashKernel(Queue&, Tensor&, const Tensor&, const Tensor&,
                                     const Tensor&, const AttentionArgs&);
bool VisionAttentionXe2Kernel(Queue&, Tensor&, const Tensor&, const Tensor&,
                              const Tensor&, const AttentionArgs&);
}
