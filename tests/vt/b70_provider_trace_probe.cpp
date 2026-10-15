// Separate process: trace opt-in is sampled once, and fake host-addressable XPU
// registration must never coexist with a real XPU backend in this probe.
#include "vt/backend.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include <stdexcept>

namespace {
int executions = 0;
void Kernel() { ++executions; }
class HostBackend final : public vt::Backend {
 public:
  void* Alloc(size_t n) override { return vt::GetBackend(vt::DeviceType::kCPU).Alloc(n); }
  void Free(void* p) override { vt::GetBackend(vt::DeviceType::kCPU).Free(p); }
  void Memset(vt::Queue& q, void* p, int v, size_t n) override {
    vt::GetBackend(vt::DeviceType::kCPU).Memset(q, p, v, n);
  }
  void Copy(vt::Queue& q, void* dst, const void* src, size_t n) override {
    vt::GetBackend(vt::DeviceType::kCPU).Copy(q, dst, src, n);
  }
  vt::Queue CreateQueue() override { return {}; }
  bool UnifiedMemory() const override { return true; }
  bool DeviceMemoryIsHostAddressable() const override { return true; }
};
}

int main() {
  using vt::DeviceType;
  using vt::OpId;
  HostBackend host;
  vt::RegisterBackend(DeviceType::kXPU, &host);
  vt::RegisterOp(OpId::kSiluAndMul, DeviceType::kCPU, reinterpret_cast<void*>(&Kernel));
  vt::RegisterOp(OpId::kAdd, DeviceType::kXPU, reinterpret_cast<void*>(&Kernel));
  for (int i = 0; i < 3; ++i) {
    reinterpret_cast<void (*)()>(vt::GetOp(OpId::kAdd, DeviceType::kXPU))();
    reinterpret_cast<void (*)()>(vt::GetOp(OpId::kSiluAndMul, DeviceType::kXPU))();
  }
  vt::RegisterOp(OpId::kSiluAndMul, DeviceType::kXPU, reinterpret_cast<void*>(&Kernel));
  reinterpret_cast<void (*)()>(vt::GetOpFallback(OpId::kSiluAndMul, DeviceType::kXPU, vt::kNativeProviderName))();
  if (executions != 7 || vt::GetReferenceTierHits() != 4)
    throw std::runtime_error("provider dispatch/reference selection counts differ");
}
