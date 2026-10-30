#include "vt/cpu/cpu_isa_arm.h"

#include <sstream>

#if defined(__aarch64__)
#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#endif

#if defined(__linux__) && !defined(HWCAP_ASIMD)
#define HWCAP_ASIMD (1UL << 1)
#endif
#if defined(__linux__) && !defined(HWCAP_ASIMDDP)
#define HWCAP_ASIMDDP (1UL << 20)
#endif
#if defined(__linux__) && !defined(HWCAP2_I8MM)
#define HWCAP2_I8MM (1UL << 13)
#endif
#if defined(__linux__) && !defined(HWCAP2_BF16)
#define HWCAP2_BF16 (1UL << 14)
#endif

namespace vt::cpu {
namespace {

constexpr std::array<ArmIsaTierRequirement, 4> kInventory{{
    {ArmIsaTier::kPortable, "portable", "scalar", "none", "none", "none"},
    {ArmIsaTier::kNeon, "neon", "elementwise-gemm", "neon",
     "getauxval:AT_HWCAP:HWCAP_ASIMD", "aarch64-baseline:neon"},
    {ArmIsaTier::kDotProd, "dotprod", "q8-dot", "neon,dotprod",
     "getauxval:AT_HWCAP:HWCAP_ASIMDDP",
     "sysctl:hw.optional.arm.FEAT_DotProd"},
    {ArmIsaTier::kI8mm, "i8mm", "quant-dot,quant-repack",
     "neon,dotprod,i8mm", "getauxval:AT_HWCAP2:HWCAP2_I8MM",
     "sysctl:hw.optional.arm.FEAT_I8MM"},
}};

#if defined(__aarch64__) && defined(__APPLE__)
bool DarwinFeature(const char* name) {
  int value = 0;
  size_t size = sizeof(value);
  return sysctlbyname(name, &value, &size, nullptr, 0) == 0 && value != 0;
}
#endif

}  // namespace

const std::array<ArmIsaTierRequirement, 4>& ArmIsaTierInventory() {
  return kInventory;
}

const char* ArmIsaTierName(ArmIsaTier tier) {
  for (const auto& item : kInventory) {
    if (item.tier == tier) return item.name.data();
  }
  return "unknown";
}

// BFDOT (ARMv8.6 FEAT_BF16) probe, same shape as the i8mm one: AT_HWCAP2 on
// Linux, the sysctl feature question on Darwin. Used by the paged-attention
// BFDOT arm (cpu_paged_attn_bfdot_arm.cpp), which is handed out at runtime
// only after this detector proves the hardware.
bool DetectArmBf16() {
#if defined(__aarch64__) && defined(__linux__)
  return (getauxval(AT_HWCAP2) & HWCAP2_BF16) != 0;
#elif defined(__aarch64__) && defined(__APPLE__)
  return DarwinFeature("hw.optional.arm.FEAT_BF16");
#else
  return false;
#endif
}

ArmIsaCaps DetectArmIsaCaps() {
  ArmIsaCaps caps{};
#if defined(__aarch64__) && defined(__linux__)
  const unsigned long hwcap = getauxval(AT_HWCAP);
  const unsigned long hwcap2 = getauxval(AT_HWCAP2);
  caps.neon = (hwcap & HWCAP_ASIMD) != 0;
  caps.dotprod = (hwcap & HWCAP_ASIMDDP) != 0;
  caps.i8mm = (hwcap2 & HWCAP2_I8MM) != 0;
#elif defined(__aarch64__) && defined(__APPLE__)
  caps.neon = true;
  caps.dotprod = DarwinFeature("hw.optional.arm.FEAT_DotProd");
  caps.i8mm = DarwinFeature("hw.optional.arm.FEAT_I8MM");
#endif
  return caps;
}

bool ArmIsaTierSupported(const ArmIsaCaps& caps, ArmIsaTier tier) {
  switch (tier) {
    case ArmIsaTier::kPortable: return true;
    case ArmIsaTier::kNeon: return caps.neon;
    case ArmIsaTier::kDotProd: return caps.neon && caps.dotprod;
    case ArmIsaTier::kI8mm:
      return caps.neon && caps.dotprod && caps.i8mm;
  }
  return false;
}

bool SelectArmIsaTier(const ArmIsaCaps& caps, std::string_view forced,
                      ArmIsaTier* selected, std::string* error) {
  ArmIsaTier requested = ArmIsaTier::kPortable;
  if (forced.empty()) {
    for (auto it = kInventory.rbegin(); it != kInventory.rend(); ++it) {
      if (ArmIsaTierSupported(caps, it->tier)) {
        *selected = it->tier;
        return true;
      }
    }
  } else {
    bool found = false;
    for (const auto& item : kInventory) {
      if (item.name == forced) {
        requested = item.tier;
        found = true;
        break;
      }
    }
    if (!found) {
      if (error != nullptr) *error = "unknown Arm ISA tier '" + std::string(forced) + "'";
      return false;
    }
    if (ArmIsaTierSupported(caps, requested)) {
      *selected = requested;
      return true;
    }
  }

  if (error != nullptr) {
    const auto& requirement = kInventory[static_cast<std::size_t>(requested)];
    std::ostringstream message;
    message << "unsupported Arm ISA tier '" << requirement.name
            << "': requires CPU features " << requirement.cpu_features;
    *error = message.str();
  }
  return false;
}

bool ResolveArmIsaToggle(const ArmIsaCaps& caps, ArmIsaTier tier,
                         std::string_view value, bool* enabled,
                         std::string* error) {
  if (value.empty() || value == "auto") {
    *enabled = ArmIsaTierSupported(caps, tier);
    return true;
  }
  if (value == "portable" || value == "0" || value == "off" ||
      value == "false") {
    *enabled = false;
    return true;
  }
  if (value != ArmIsaTierName(tier) && value != "1" && value != "on" &&
      value != "true") {
    if (error != nullptr) {
      *error = "unknown force value '" + std::string(value) +
               "' for Arm ISA tier '" + ArmIsaTierName(tier) + "'";
    }
    return false;
  }
  if (!ArmIsaTierSupported(caps, tier)) {
    if (error != nullptr) {
      *error = "unsupported forced Arm ISA tier '" +
               std::string(ArmIsaTierName(tier)) + "'";
    }
    return false;
  }
  *enabled = true;
  return true;
}

}  // namespace vt::cpu
