// Strict environment-variable parsing for ROCm-invented knobs.
//
// Knobs that mirror an upstream lever keep upstream's `atoi` semantics (see
// Exl3GemvParseMode's comment in src/vt/exl3_policy.cpp:181: a knob that
// behaves differently from the oracle's cannot reproduce its run). Knobs this
// backend invents have no oracle parity to keep, and `atoi` makes every
// typo'd value silently meaningful — 'x' parses to 0, which on a several-state
// knob flips a kernel arm with no diagnostic. The helpers below parse the
// whole token and fall back to the declared default with a stderr warning on
// any junk, so a misspelling is visible in the log instead of invisible in
// the result.
#pragma once

#include <cstdlib>
#include <cstdio>

namespace vt::rocm {

// Integer knob. `def` is the value when unset, empty, or unparseable.
inline int EnvIntStrict(const char* name, int def) {
  const char* e = std::getenv(name);
  if (e == nullptr || e[0] == '\0') return def;
  char* end = nullptr;
  const long v = std::strtol(e, &end, 10);
  if (end == e || *end != '\0') {
    std::fprintf(stderr,
                 "vt rocm: %s='%s' is not an integer; using default %d\n", name,
                 e, def);
    return def;
  }
  if (v > 2147483647L) return 2147483647;
  if (v < -2147483648L) return -2147483648;
  return static_cast<int>(v);
}

// Boolean knob. `def` is the value when unset or empty. An unparseable value
// warns and falls back to `def`; '0' is the only spelled-off value (matching
// every sibling arm's contract), '1' and every other integer spell on/off by
// their truthiness like the incumbent parsers do.
inline bool EnvBoolStrict(const char* name, bool def) {
  const char* e = std::getenv(name);
  if (e == nullptr || e[0] == '\0') return def;
  if (e[0] == '0' && e[1] == '\0') return false;
  if (e[0] == '1' && e[1] == '\0') return true;
  std::fprintf(stderr,
               "vt rocm: %s='%s' is not 0 or 1; using default %d\n", name, e,
               static_cast<int>(def));
  return def;
}

}  // namespace vt::rocm
