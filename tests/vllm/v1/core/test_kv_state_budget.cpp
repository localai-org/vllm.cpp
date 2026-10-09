// vllm.cpp original. The recurrent-state budget check (issue #371).
//
// Ported semantics: vllm/v1/core/kv_cache_utils.py:751-787
// (`_check_enough_kv_cache_memory` refuses with needed-vs-available and the
// knobs; it never auto-clamps) plus
// vllm/v1/kv_cache_interface.py:713-718, where `MambaSpec.max_memory_usage_bytes`
// counts `num_speculative_blocks` — the term our estimator omitted.
//
// WHY THIS EXISTS. `ResolveMaxModelLen` guarded only the PAGED ATTENTION pool
// and skipped entirely when `bytes_per_block == 0`, reasoning that a Mamba/GDN
// model's state "is sized per sequence slot rather than per block, so ... has
// nothing to run out of". It does run out. Speculation widens that state to
// k+1 snapshot slots per sequence (`gdn_state_slots_ = max_num_reqs *
// (num_spec+1)`, runner.cpp:449-451), so a draft whose checkpoint block_size is
// 15 demands SIXTEEN times the GDN state of spec-off. On 2026-08-11 that took
// down two machines four times (two Jetson Thor reboots, one dgx reboot) rather
// than refusing: we computed the demand and allocated it.
//
// The numbers below are the REAL Qwen3.6-27B ones (48 linear-attention layers,
// linear_num_value_heads=48, key/value_head_dim=128, mamba_ssm_dtype=float32,
// conv_kernel=4), so the arithmetic in the failure report is what the test pins.
#include <doctest/doctest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/recurrent_prefix_snapshot.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

using namespace vllm::v1;

namespace {

// One Qwen3.6-27B linear-attention layer's state, exactly as
// MakeQwen3_5KVCacheSpec builds it: an SSM page [Hv, Dv, Dk] plus a conv page
// [conv_dim, conv_kernel-1+num_spec].
constexpr int kLinearLayers = 48;
constexpr int64_t kHv = 48, kDk = 128, kDv = 128;
constexpr int64_t kKeyDim = 16 * 128, kValueDim = 48 * 128;  // 16 k-heads, 48 v-heads
constexpr int64_t kConvDim = 2 * kKeyDim + kValueDim;
constexpr int kConvKernel = 4;

KVCacheConfig Qwen27bConfig(int num_spec) {
  KVCacheConfig kv;
  kv.num_blocks = 256;
  std::vector<std::string> layers;
  layers.reserve(kLinearLayers);
  for (int i = 0; i < kLinearLayers; ++i) {
    layers.push_back("model.layers." + std::to_string(i) + ".linear_attn");
  }
  kv.kv_cache_groups.emplace_back(
      layers,
      std::make_shared<MambaSpec>(
          /*block_size=*/1,
          std::vector<std::vector<int64_t>>{{kConvDim, kConvKernel - 1 + num_spec},
                                            {kHv, kDv, kDk}},
          std::vector<vt::DType>{vt::DType::kBF16, vt::DType::kF32},
          /*page_size_padded=*/std::nullopt,
          /*mamba_cache_mode=*/"none",
          /*num_speculative_blocks=*/num_spec));
  return kv;
}

}  // namespace

TEST_CASE("recurrent state is counted per sequence slot, widened by k+1") {
  // The formula the runner actually allocates from (runner.cpp:449-451,670-679):
  // max_num_seqs * (num_spec + 1) * per-layer state, summed over the group.
  const int64_t off = recurrent_state_bytes(Qwen27bConfig(0), /*max_num_seqs=*/32);
  const int64_t k8 = recurrent_state_bytes(Qwen27bConfig(8), /*max_num_seqs=*/32);
  const int64_t k15 = recurrent_state_bytes(Qwen27bConfig(15), /*max_num_seqs=*/32);

  CHECK(off > 0);
  // k+1 slots per sequence: 9x and 16x the spec-off state, plus the conv row
  // widening (conv_kernel-1+num_spec), so strictly MORE than the slot ratio.
  CHECK(k8 > off * 9);
  CHECK(k15 > off * 16);
  // The headline from issue #371: the 27B draft's block_size=15 demands tens of
  // GiB of state on its own.
  CHECK(k15 > 70LL * 1024 * 1024 * 1024);
  CHECK(off < 6LL * 1024 * 1024 * 1024);
}

TEST_CASE("max_num_seqs scales the state linearly") {
  const int64_t one = recurrent_state_bytes(Qwen27bConfig(15), 1);
  const int64_t many = recurrent_state_bytes(Qwen27bConfig(15), 32);
  CHECK(many == one * 32);
}

TEST_CASE("PR11 recurrent prefix snapshots reserve additional state memory") {
  auto config = Qwen27bConfig(0);
  const auto working = recurrent_state_bytes(config, 4);
  config.recurrent_prefix_snapshots = std::make_shared<RecurrentPrefixSnapshotIndex>(4, 16);
  CHECK(recurrent_state_bytes(config, 4) == 2 * working);
  CHECK_THROWS_AS(check_enough_state_memory(working, recurrent_state_bytes(config, 4), 4, 0), std::invalid_argument);
}

TEST_CASE("a state budget that does not fit is REFUSED, not clamped") {
  // Upstream raises ValueError rather than reducing concurrency silently
  // (kv_cache_utils.py:778-787). Ours throws std::invalid_argument.
  const int64_t needed = recurrent_state_bytes(Qwen27bConfig(15), 32);
  const int64_t available = 40LL * 1024 * 1024 * 1024;  // a 119 GiB box, target resident
  CHECK_THROWS_AS(
      check_enough_state_memory(available, needed, /*max_num_seqs=*/32,
                                /*num_spec=*/15),
      std::invalid_argument);

  // Fits => silent, exactly as upstream returns without touching the config.
  CHECK_NOTHROW(check_enough_state_memory(needed + 1, needed, 32, 15));
}

TEST_CASE("the refusal names the arithmetic and the knobs a user can turn") {
  // A message that says only "out of memory" sends someone hunting the wrong
  // thing; this crash looked like a machine fault for a whole night.
  const int64_t needed = recurrent_state_bytes(Qwen27bConfig(15), 32);
  try {
    check_enough_state_memory(1LL << 30, needed, /*max_num_seqs=*/32, /*num_spec=*/15);
    FAIL("expected a refusal");
  } catch (const std::invalid_argument& e) {
    const std::string msg = e.what();
    CHECK(msg.find("recurrent") != std::string::npos);
    CHECK(msg.find("speculative") != std::string::npos);
    CHECK(msg.find("max-num-seqs") != std::string::npos);  // the actionable knob
    CHECK(msg.find("16") != std::string::npos);            // k+1 slots per sequence
  }
}

TEST_CASE("a model with no recurrent state is unaffected") {
  // The guard must stay inert for a pure attention model: zero state, no
  // refusal, whatever the speculative width.
  KVCacheConfig attention_only;
  attention_only.num_blocks = 256;
  CHECK(recurrent_state_bytes(attention_only, 32) == 0);
  CHECK_NOTHROW(check_enough_state_memory(0, 0, 32, 15));
}
