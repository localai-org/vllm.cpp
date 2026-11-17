#pragma once

namespace vt { class Backend; }
namespace vllm {
class DevicePool;

// Internal lifecycle diagnostic access to the existing decode-input pool.
// Observe live owners normally; Drain only after every engine/graph/consumer
// using this backend has been destroyed. Not a serving-path release API.
DevicePool& Qwen3_5PersistentDecodePoolForDiagnostics(vt::Backend& backend);
} // namespace vllm
