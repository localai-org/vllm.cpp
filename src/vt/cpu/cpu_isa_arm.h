#pragma once

#include <array>
#include <string>
#include <string_view>

namespace vt::cpu {

enum class ArmIsaTier { kPortable, kNeon, kDotProd, kI8mm };

struct ArmIsaCaps {
  bool neon = false;
  bool dotprod = false;
  bool i8mm = false;
};

struct ArmIsaTierRequirement {
  ArmIsaTier tier;
  std::string_view name;
  std::string_view kernel_families;
  std::string_view cpu_features;
  std::string_view linux_probe;
  std::string_view darwin_probe;
};

bool DetectArmBf16();
const std::array<ArmIsaTierRequirement, 4>& ArmIsaTierInventory();
const char* ArmIsaTierName(ArmIsaTier tier);
ArmIsaCaps DetectArmIsaCaps();
bool ArmIsaTierSupported(const ArmIsaCaps& caps, ArmIsaTier tier);
bool SelectArmIsaTier(const ArmIsaCaps& caps, std::string_view forced,
                      ArmIsaTier* selected, std::string* error);
bool ResolveArmIsaToggle(const ArmIsaCaps& caps, ArmIsaTier tier,
                         std::string_view value, bool* enabled,
                         std::string* error);

}  // namespace vt::cpu
