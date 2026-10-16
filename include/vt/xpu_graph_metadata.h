#pragma once
#include <cstddef>
#include <memory>

namespace vt { struct Queue; }
namespace vt::xpu {
// Host-only queries/registration; no synchronization or vendor types.
bool IsGraphCapturing(Queue&);
// Already validated immutable model metadata needs no replay check kernel.
// Preserve graph write-overlap protection and pin its allocation until retire.
void RecordGraphImmutableRead(Queue&, const void* data, size_t bytes,
                              const std::shared_ptr<void>& owner, const char* message);
}
