#pragma once
#include <string>

#define VLLM_CPP_VERSION_MAJOR 0
#define VLLM_CPP_VERSION_MINOR 0
#define VLLM_CPP_VERSION_PATCH 3
#define VLLM_CPP_BUILD_VERSION "0.0.3"

namespace vllm {
// Returns the exact build identity, plus "+cuda" when built with that backend.
std::string Version();
}  // namespace vllm
