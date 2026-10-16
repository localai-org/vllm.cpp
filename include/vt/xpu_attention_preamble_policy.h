#pragma once

namespace vt {
// Same FP32 arithmetic nodes as the RN-helper control, without device-library
// calls. Share this default with graph retirement so dispatch cannot drift.
inline constexpr const char* kXpuAttentionPreambleNativeFp32Default = "1";
}  // namespace vt
