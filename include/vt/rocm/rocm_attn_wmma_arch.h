#pragma once

#include <string_view>

#include "vt/rocm/rocm_arch.h"

namespace vt::rocm {

// The gfx1100 decode prerequisite is independent of the prefill control.
constexpr bool GcnArchNameHasGemmaDecodeWmma(std::string_view arch) {
  return arch == "gfx1100" || arch.starts_with("gfx1100:");
}

// Attention admission is separate from quantized WMMA admission. Every admitted
// target must compile the attention body as well as pass this host predicate.
// gfx1101 (Navi32, RX 7700/7800 XT) is the same gfx11 WMMA family as gfx1100 —
// admitted for prefill only; decode admission stays on
// GcnArchNameHasGemmaDecodeWmma and is unchanged.
constexpr bool GcnArchNameHasSharedKAttentionWmma(std::string_view arch) {
  auto prefix_ok = [](std::string_view s, std::string_view stem) {
    if (s.size() < stem.size()) return false;
    if (s.substr(0, stem.size()) != stem) return false;
    if (s.size() == stem.size()) return true;
    const char c = s[stem.size()];
    return c < '0' || c > '9';
  };
  return GcnArchNameHasGemmaDecodeWmma(arch) || prefix_ok(arch, "gfx1101") ||
         GcnArchNameIsGfx12PrefillWmma(arch);
}

// Default on for validated gfx1100 and the existing gfx12 targets. Preserve
// the environment control's first-character rule.
constexpr bool SharedKAttentionWmmaEnabled(std::string_view arch, const char* override_value) {
  return GcnArchNameHasSharedKAttentionWmma(arch) && (!override_value || override_value[0] != '0');
}

}  // namespace vt::rocm
