// Shared exact F16 materialization; caller retains the three tensor owners.
#pragma once
namespace vt {
struct Queue;
struct Tensor;
namespace xpu {
// Contiguous [T,H] inputs -> contiguous [T,2H] output. Two-byte alignment,
// matching XPU devices, checked sizes and no output/input alias required.
// Raw storage copies preserve signed zeros and all NaN payloads. No allocation
// or host wait; ordinary queue dependencies also apply during graph capture.
void ConcatRowsF16(Queue& queue, Tensor& out, const Tensor& left,
                   const Tensor& right);
}  // namespace xpu
}  // namespace vt
