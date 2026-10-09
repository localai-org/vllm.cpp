#pragma once

namespace gdn {

static constexpr float l2norm_eps = 0.000001f;

#ifdef VLLM_XPU_ENABLE_XE2
static constexpr int chunk_size_xe2 = 64;
#endif

enum class ActMode {
  silu = 0,
  swish = 1,
};

}  // namespace gdn