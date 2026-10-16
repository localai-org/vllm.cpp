// The recurrent-state budget (issue #1983).
//
// EVERY CASE IS DRIVEN FROM `MakeQwen3_5KVCacheSpec`, the REAL registry builder
// the engine calls, and never from a hand-assembled `KVCacheConfig`. That is
// deliberate and it is the lesson of #1963: `tests/vllm/v1/test_kv_cache_interface.cpp`
// builds its groups by hand with the layer-name lists the registry does NOT
// emit, so a divisor that under-counts by the layer count passed every case in
// that file for months. A fixture that hands the code a shape no registry emits
// gates the fixture.
//
// The geometry below is the REAL Qwen3.8-27B one, read from
// `.agents/specs/qwen36-forward-notes.md` §1 (which records it from the
// checkpoint's own config.json): 64 layers as `[LA,LA,LA,FA] x 16`,
// `num_key_value_heads = 4`, `head_dim = 256`, GDN `Hk/Hv/Dk/Dv/conv =
// 16/48/128/128/4`, `mamba_ssm_dtype = float32`.
#include "vllm/v1/core/hybrid_kv_budget.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_common.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_interface.h"

using vllm::HfConfig;
using vllm::MakeQwen3_5KVCacheSpec;
using vllm::v1::ComputeHybridKvBudget;
using vllm::v1::ClampMaxNumSeqsToStateBudget;
using vllm::v1::HybridKvBudget;
using vllm::v1::KVCacheConfig;
using vllm::v1::kStateSeqsUnbounded;

namespace {

constexpr int kBlockSize = 32;

HfConfig Qwen27bConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForConditionalGeneration"};
  c.num_hidden_layers = 64;
  c.num_attention_heads = 24;
  c.num_key_value_heads = 4;
  c.head_dim = 256;
  c.linear_num_key_heads = 16;
  c.linear_num_value_heads = 48;
  c.linear_key_head_dim = 128;
  c.linear_value_head_dim = 128;
  c.linear_conv_kernel_dim = 4;
  c.mamba_ssm_dtype = "float32";
  return c;
}

// One GDN layer's state, one slot, computed independently of the production
// code: SSM `[Hv, Dv, Dk]` in f32 plus conv `[2*Kdim + Vdim, K-1+k]` in bf16.
int64_t MambaPageBytes(int num_spec) {
  const int64_t ssm = 48LL * 128 * 128 * 4;
  const int64_t conv = (2LL * 16 * 128 + 48LL * 128) * (4 - 1 + num_spec) * 2;
  return ssm + conv;
}

// One attention layer's whole paged pool, computed independently: K and V at
// `num_kv_heads * head_dim` each, bf16, over every block.
int64_t PagedBytesPerAttentionLayer(int num_blocks) {
  return static_cast<int64_t>(num_blocks) * kBlockSize * 4 * (256 + 256) * 2;
}

}  // namespace

TEST_CASE("hybrid budget: the 27B state pool never exceeds one attention layer's pool") {
  // THE INVARIANT, and it is upstream's own: a hybrid model's tensors are
  // `page_size * num_blocks` and each is `shared_by` ONE layer from EACH group
  // (`kv_cache_utils.py::_get_kv_cache_config_uniform_page_size` :1399-1416), so
  // every layer -- attention or mamba -- gets exactly `num_blocks` pages of the
  // same size. A GDN layer's state pool can therefore never cost more than an
  // attention layer's paged pool.
  //
  // Both sides here are computed from the geometry, not from the code under
  // test: the left from `MambaPageBytes` above, the right from
  // `PagedBytesPerAttentionLayer`.
  const HfConfig c = Qwen27bConfig();
  for (const int num_spec : {0, 4, 8}) {
    CAPTURE(num_spec);
    for (const int num_blocks : {512, 3072, 8192}) {
      CAPTURE(num_blocks);
      const KVCacheConfig kv =
          MakeQwen3_5KVCacheSpec(c, kBlockSize, num_blocks, num_spec);
      const HybridKvBudget b = ComputeHybridKvBudget(kv);
      REQUIRE(b.max_state_seqs != kStateSeqsUnbounded);
      REQUIRE(b.slots_per_seq == num_spec + 1);

      const int64_t state_pool_per_layer = static_cast<int64_t>(b.max_state_seqs) *
                                           b.slots_per_seq * MambaPageBytes(num_spec);
      CHECK(state_pool_per_layer <= PagedBytesPerAttentionLayer(num_blocks));
    }
  }
}

TEST_CASE("hybrid budget: the 27B arithmetic, term by term") {
  // The whole derivation pinned on the operator's own configuration
  // (`--num-blocks 3072`, block_size 32, `num_speculative_tokens 8`), so a
  // change to any intermediate is visible rather than absorbed by the quotient.
  const KVCacheConfig kv =
      MakeQwen3_5KVCacheSpec(Qwen27bConfig(), kBlockSize, 3072, /*num_spec=*/8);
  const HybridKvBudget b = ComputeHybridKvBudget(kv);

  CHECK(b.mamba_page_bytes == 3371008);        // 3,145,728 SSM + 225,280 conv
  CHECK(b.attn_bytes_per_token == 4096);       // 2 * 4 heads * 256 dim * 2 B
  CHECK(b.slots_per_seq == 9);                 // k + 1
  // interface.py:896-901 -- align 32, ceil(3371008 / (32*4096)) == 26.
  CHECK(b.unified_block_tokens == 32 * 26);    // 832
  CHECK(b.unified_num_blocks == (3072 * 32) / 832);  // 118
  CHECK(b.max_state_seqs == 118 / 9);          // 13

  // 43.40 GiB was the measured allocation at --max-num-seqs 32. The clamp turns
  // that into 13 seats.
  CHECK(ClampMaxNumSeqsToStateBudget(32, b) == 13);
  CHECK(ClampMaxNumSeqsToStateBudget(16, b) == 13);
  // Below the ceiling, the configured value passes through untouched.
  CHECK(ClampMaxNumSeqsToStateBudget(4, b) == 4);
}

TEST_CASE("hybrid budget: XPU FP8 Qwen page follows Python MTP depth") {
  HfConfig c = Qwen27bConfig();
  c.torch_dtype = "float16";
  c.raw = {{"quantization_config", {{"quant_method", "gptq"}}}};
  for (int depth : {0, 4}) {
    auto kv = MakeQwen3_5KVCacheSpec(c, /*block_size=*/64,
                                    /*num_blocks=*/1, depth);
    const auto dtype = vllm::v1::ParseCacheDType("fp8", vt::DType::kF16);
    vllm::v1::ApplyCacheDType(kv, dtype, 1.0F, 1.0F);
    const HybridKvBudget b = ComputeHybridKvBudget(kv, /*alignment=*/64);
    CAPTURE(depth);
    CHECK(b.attn_bytes_per_token == 2048);
    CHECK(b.mamba_page_bytes == MambaPageBytes(depth));
    CHECK(b.unified_block_tokens == (depth == 4 ? 1664 : 1600));
  }
}

TEST_CASE("hybrid budget: the SPEC-OFF 27B engine is not clamped at all") {
  // The defect is a speculation-x-concurrency product. Without speculation each
  // sequence owns ONE state slot and the same pool holds far more of them than
  // any configured concurrency, so this row must be inert on the production
  // default. If this case ever starts clamping, the bound has drifted tighter
  // than upstream and that is a regression, not a tightening.
  const KVCacheConfig kv =
      MakeQwen3_5KVCacheSpec(Qwen27bConfig(), kBlockSize, 3072, /*num_spec=*/0);
  const HybridKvBudget b = ComputeHybridKvBudget(kv);
  CHECK(b.slots_per_seq == 1);
  CHECK(b.max_state_seqs >= 32);
  CHECK(ClampMaxNumSeqsToStateBudget(32, b) == 32);
}

TEST_CASE("hybrid budget: the seat count does NOT move with the concurrency cap") {
  // The whole point. `max_num_seqs` sizes no allocation anywhere in vLLM; the
  // budget is a function of the KV pool and the spec geometry alone, and this
  // case is what the reachability mutation kills.
  const KVCacheConfig kv =
      MakeQwen3_5KVCacheSpec(Qwen27bConfig(), kBlockSize, 3072, /*num_spec=*/8);
  const HybridKvBudget b = ComputeHybridKvBudget(kv);
  for (const int configured : {13, 16, 24, 28, 32, 64, 1024}) {
    CAPTURE(configured);
    CHECK(ClampMaxNumSeqsToStateBudget(configured, b) == 13);
  }
}

TEST_CASE("hybrid budget: a bigger KV pool buys more seats, proportionally") {
  // The bound has to be a function OF the budget, not a constant that happens to
  // sit below it. Doubling the pool doubles the pages and so the seats.
  const HfConfig c = Qwen27bConfig();
  const HybridKvBudget small =
      ComputeHybridKvBudget(MakeQwen3_5KVCacheSpec(c, kBlockSize, 3072, 8));
  const HybridKvBudget big =
      ComputeHybridKvBudget(MakeQwen3_5KVCacheSpec(c, kBlockSize, 6144, 8));
  CHECK(big.unified_block_tokens == small.unified_block_tokens);
  CHECK(big.unified_num_blocks == 2 * small.unified_num_blocks);
  CHECK(big.max_state_seqs > small.max_state_seqs);
  CHECK(big.max_state_seqs >= 2 * small.max_state_seqs - 1);
}

TEST_CASE("hybrid budget: an attention-only config carries no bound") {
  // Every non-hybrid engine must pass its configured concurrency through
  // untouched, or this row has changed models it has no business touching.
  KVCacheConfig kv;
  kv.num_blocks = 3072;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<vllm::v1::FullAttentionSpec>(kBlockSize, 4, 256,
                                                    vt::DType::kBF16));
  const HybridKvBudget b = ComputeHybridKvBudget(kv);
  CHECK(b.max_state_seqs == kStateSeqsUnbounded);
  CHECK(ClampMaxNumSeqsToStateBudget(1024, b) == 1024);
}

TEST_CASE("hybrid budget: a pure-recurrent config carries no bound either") {
  // No attention page to unify against; upstream's unification has nothing to do
  // there either. Reporting a bound here would mean inventing a page. Owed in
  // the row's spec, and pinned so the disposition is deliberate rather than a
  // gap someone reads as a bug.
  KVCacheConfig kv;
  kv.num_blocks = 3072;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"gdn"},
      std::make_shared<vllm::v1::MambaSpec>(
          kBlockSize, std::vector<std::vector<int64_t>>{{128, 4}, {8, 16, 16}},
          std::vector<vt::DType>{vt::DType::kBF16, vt::DType::kF32}));
  const HybridKvBudget b = ComputeHybridKvBudget(kv);
  CHECK(b.mamba_page_bytes > 0);
  CHECK(b.max_state_seqs == kStateSeqsUnbounded);
  CHECK(ClampMaxNumSeqsToStateBudget(32, b) == 32);
}

TEST_CASE("hybrid budget: the seat count never falls below one") {
  // A pool too small for one sequence's state is a REFUSAL the #371 guard owns
  // (`check_enough_state_memory`). Serving zero sequences is not a disposition
  // this engine has, so the clamp floors at one and lets that guard speak.
  const KVCacheConfig kv =
      MakeQwen3_5KVCacheSpec(Qwen27bConfig(), kBlockSize, 8, /*num_spec=*/8);
  const HybridKvBudget b = ComputeHybridKvBudget(kv);
  CHECK(b.max_state_seqs == 0);
  CHECK(ClampMaxNumSeqsToStateBudget(32, b) == 1);
}
