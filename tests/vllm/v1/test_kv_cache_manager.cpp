// Tests for the TOP-LEVEL KVCacheManager (M1.3 Task 4) — get_computed_blocks
// (prefix reuse), allocate_slots (the central allocator + None-on-OOM), free
// (the LRU eviction order), reset_prefix_cache, and the hybrid full-attn+mamba
// allocation.
//
// Ported from vllm/tests/v1/core/test_prefix_caching.py @ e24d1b24:
//   - test_prefill: get_computed_blocks returns the cached prefix across two
//     requests (req1 hits req0's 3-block prefix) + allocate_slots success
//     returning ([1, 2, 3, 4],) and reducing the free-block count.
//   - test_evict: the LITERAL manager-level free order (M1.2 LRU carry-forward)
//     `[6, 10, 5, 4, 3, 2, 1]` then `[6, 10, 5, 4, 3, 2, 1, 9, 8, 7]`.
//   - test_prefill_hybrid_model: the 2-group (full-attn + GDN mamba) allocate.
// The allocate_slots OOM -> std::nullopt case (the signal the Scheduler uses to
// preempt) has no single dedicated upstream test; it is the return contract
// asserted directly here.
//
// Only sha256_cbor is ported (M1.2), so these use it; the block ids / free order
// are hash-function-independent.
#include <doctest/doctest.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_manager.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/recurrent_prefix_snapshot.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vt/dtype.h"

using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheBlocks;
using vllm::v1::KVCacheConfig;
using vllm::v1::KVCacheManager;
using vllm::v1::MambaSpec;
using vllm::v1::Request;
using vllm::v1::sha256_cbor;
using vt::DType;

namespace {

KVCacheConfig MakeFullConfig(int block_size, int num_blocks) {
  KVCacheConfig cfg;
  cfg.num_blocks = num_blocks;
  cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(block_size, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));
  return cfg;
}

KVCacheConfig MakeHybridConfig(int block_size, int num_blocks) {
  KVCacheConfig cfg;
  cfg.num_blocks = num_blocks;
  cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"full"},
      std::make_shared<FullAttentionSpec>(block_size, 1, 1, DType::kF32));
  cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"mamba"},
      std::make_shared<MambaSpec>(block_size,
                                  std::vector<std::vector<int64_t>>{{1, 1}},
                                  std::vector<DType>{DType::kF32}));
  return cfg;
}

std::unique_ptr<KVCacheManager> MakeManager(KVCacheConfig cfg, int block_size) {
  return std::make_unique<KVCacheManager>(
      std::move(cfg), /*max_model_len=*/8192,
      /*scheduler_block_size=*/block_size, /*hash_block_size=*/block_size);
}

// Single full-attention group manager with an explicit watermark and (optional)
// max_model_len — the watermark/reserved/lookahead/admission-cap machinery the
// existing tests never exercise (they all run with watermark == 0).
std::unique_ptr<KVCacheManager> MakeFullManager(int block_size, int num_blocks,
                                                double watermark = 0.0,
                                                int max_model_len = 8192) {
  return std::make_unique<KVCacheManager>(
      MakeFullConfig(block_size, num_blocks), max_model_len,
      /*scheduler_block_size=*/block_size, /*hash_block_size=*/block_size,
      /*max_num_batched_tokens=*/std::nullopt, /*enable_caching=*/true,
      /*use_eagle=*/false, /*log_stats=*/false,
      /*enable_kv_cache_events=*/false, /*dcp_world_size=*/1,
      /*pcp_world_size=*/1, watermark);
}

// Build a request whose token ids are `tokens`, wired with the cbor block hasher
// so its block_hashes populate at block_size granularity.
Request MakeRequest(const std::string& id, const std::vector<int32_t>& tokens,
                    int block_size) {
  return Request(id, tokens, vllm::SamplingParams{}, /*arrival_time=*/0.0,
                 get_request_block_hasher(block_size, sha256_cbor));
}

std::vector<int32_t> Range(int lo, int hi) {
  std::vector<int32_t> v;
  for (int i = lo; i < hi; ++i) v.push_back(i);
  return v;
}

std::vector<int> FreeBlockIds(KVCacheManager& mgr) {
  std::vector<int> ids;
  for (const auto* blk : mgr.block_pool.free_block_queue.get_all_free_blocks()) {
    ids.push_back(blk->block_id);
  }
  return ids;
}

}  // namespace

TEST_CASE("PR11 B70 admission bounds four maximum contexts and preserves reserved pages") {
  constexpr int Block = 16, Context = 32768, BlocksPerContext = Context / Block;
  KVCacheConfig cfg;
  cfg.num_blocks = 2 * BlocksPerContext + 1; // two contexts, plus the null sentinel
  cfg.kv_cache_groups.emplace_back(std::vector<std::string>{"attention"},
      std::make_shared<FullAttentionSpec>(Block, 4, 256, DType::kI8));
  KVCacheManager manager(cfg, Context, Block, Block, /*max_num_batched_tokens=*/4096,
                         /*enable_caching=*/false);
  std::vector<std::unique_ptr<Request>> requests;
  for (int i = 0; i < 4; ++i) {
    requests.push_back(std::make_unique<Request>(std::to_string(i), std::vector<int32_t>(Context, i + 1),
                                               vllm::SamplingParams{}, 0.0));
    auto allocated = manager.allocate_slots(*requests.back(), Context);
    CHECK(allocated.has_value() == (i < 2));
    CHECK(manager.block_pool.get_num_free_blocks() >= 0);
  }
  CHECK(manager.block_pool.get_num_free_blocks() == 0);
  manager.free(*requests[0]);
  // Retain one whole context for the other in-flight request: admission fails
  // without touching the pool, then succeeds after the reservation is released.
  auto reserved = manager.allocate_slots(*requests[2], Context, 0, std::nullopt, 0, 0, false, 0, false, BlocksPerContext);
  CHECK_FALSE(reserved.has_value());
  CHECK(manager.block_pool.get_num_free_blocks() == BlocksPerContext);
  REQUIRE(manager.allocate_slots(*requests[2], Context).has_value());
  CHECK(manager.block_pool.get_num_free_blocks() == 0);
  manager.free(*requests[1]); manager.free(*requests[2]);
  CHECK(manager.block_pool.get_num_free_blocks() == cfg.num_blocks - 1);
}

TEST_CASE("PR11 hybrid prefix hits require a published snapshot and retain it until restore") {
  init_none_hash(sha256_cbor);
  auto cfg = MakeHybridConfig(16, 40);
  cfg.recurrent_prefix_snapshots = std::make_shared<vllm::v1::RecurrentPrefixSnapshotIndex>(2, 16);
  auto index = cfg.recurrent_prefix_snapshots;
  auto manager = MakeManager(cfg, 16);
  auto first = MakeRequest("a", Range(0, 65), 16);
  REQUIRE(manager->allocate_slots(first, 65).has_value());
  manager->free(first); manager->new_step_starts();
  auto second = MakeRequest("b", Range(0, 66), 16);
  CHECK(manager->get_computed_blocks(second).second == 0); // KV alone is insufficient
  auto snapshot = index->Reserve(first.block_hashes[1], 32); REQUIRE(snapshot.has_value());
  CHECK(manager->get_computed_blocks(second).second == 0); // GPU copy not published yet
  index->Publish(*snapshot);
  auto [blocks, tokens] = manager->get_computed_blocks(second);
  CHECK(tokens == 32);
  REQUIRE(index->Pinned("b").has_value());
  CHECK(index->Pinned("b")->hash == first.block_hashes[1]);
  CHECK_FALSE(manager->reset_prefix_cache());
  CHECK(manager->get_computed_blocks(second).second == tokens);
  // A waiting request can lose its KV hit before admission. Its previous
  // recurrent pin must then be released rather than exhausting the bounded pool.
  auto miss = MakeRequest("b", Range(100, 166), 16);
  CHECK(manager->get_computed_blocks(miss).second == 0);
  CHECK_FALSE(index->Pinned("b").has_value());
  CHECK(manager->get_computed_blocks(second).second == tokens);
  REQUIRE(manager->allocate_slots(second, 34, tokens, blocks).has_value());
  manager->free(second); CHECK_FALSE(index->Pinned("b").has_value());
  REQUIRE(manager->reset_prefix_cache()); CHECK_FALSE(index->Contains(first.block_hashes[1]));
}

// ---------------------------------------------------------------------------
// get_computed_blocks: prefix reuse across two requests (test_prefill)
// ---------------------------------------------------------------------------

TEST_CASE(
    "KVCacheManager: get_computed_blocks returns the cached prefix across two "
    "requests (ported test_prefill)") {
  init_none_hash(sha256_cbor);
  const int block_size = 16;
  auto mgr = MakeManager(MakeFullConfig(block_size, 11), block_size);

  // 3 full blocks (48 common tokens) + 7 unique = 55 tokens.
  std::vector<int32_t> common;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < block_size; ++j) common.push_back(i);
  std::vector<int32_t> tokens0 = common;
  for (int i = 0; i < 7; ++i) tokens0.push_back(3);

  Request req0 = MakeRequest("0", tokens0, block_size);
  REQUIRE(req0.block_hashes.size() == 3);
  auto [computed0, n0] = mgr->get_computed_blocks(req0);
  CHECK(computed0.blocks[0].empty());  // full cache miss
  CHECK(n0 == 0);

  int free_before = mgr->block_pool.get_num_free_blocks();
  auto blocks0 = mgr->allocate_slots(req0, /*num_new_tokens=*/55,
                                     /*num_new_computed_tokens=*/n0,
                                     std::optional<KVCacheBlocks>(computed0));
  REQUIRE(blocks0.has_value());
  // cdiv(55, 16) = 4 blocks: [1, 2, 3, 4].
  CHECK(blocks0->get_block_ids() == std::vector<std::vector<int>>{{1, 2, 3, 4}});
  CHECK(mgr->block_pool.get_num_free_blocks() == free_before - 4);

  // req1 shares the 3-block (48-token) prefix.
  std::vector<int32_t> tokens1 = common;
  for (int i = 0; i < 5; ++i) tokens1.push_back(3);
  Request req1 = MakeRequest("1", tokens1, block_size);
  auto [computed1, n1] = mgr->get_computed_blocks(req1);
  CHECK(computed1.get_block_ids() == std::vector<std::vector<int>>{{1, 2, 3}});
  CHECK(n1 == 3 * block_size);

  auto blocks1 = mgr->allocate_slots(req1, /*num_new_tokens=*/53 - 3 * 16, n1,
                                     std::optional<KVCacheBlocks>(computed1));
  REQUIRE(blocks1.has_value());
  CHECK(blocks1->get_block_ids() == std::vector<std::vector<int>>{{5}});
  // The shared prefix blocks are now referenced by both requests.
  for (auto* blk : computed1.blocks[0]) CHECK(blk->ref_cnt == 2);
}

// ---------------------------------------------------------------------------
// KV-PREFIX-CACHE W2: extra_keys (cache_salt) partition the cache — RED-first
// no-false-share at the MANAGER level. Two requests with IDENTICAL tokens but
// different cache_salt must NOT share a cached block: the second is a full miss.
// A third request with the SAME salt as the first DOES hit — proving the salt is
// the only differentiator and the cache is otherwise live. Were extra_keys
// dropped (the pre-W2 stub), the differently-salted request would false-hit the
// first request's blocks and be served the wrong tenant's KV.
// Ported behaviour from vllm/tests/v1/core/test_prefix_caching.py
// test_cache_key_salting @ 555967922.
// ---------------------------------------------------------------------------

TEST_CASE(
    "KVCacheManager: cache_salt partitions the prefix cache (no false-share)") {
  init_none_hash(sha256_cbor);
  const int block_size = 16;
  auto mgr = MakeManager(MakeFullConfig(block_size, 11), block_size);

  // 3 full blocks (48 tokens) of a shared prompt + a unique tail.
  std::vector<int32_t> common;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < block_size; ++j) common.push_back(i);
  std::vector<int32_t> tokens = common;
  for (int i = 0; i < 7; ++i) tokens.push_back(3);

  auto make_salted = [&](const std::string& id, const std::string& salt) {
    // Set cache_salt BEFORE installing the hasher (fields precede the first
    // hash, as FromEngineCoreRequest does).
    Request req(id, tokens, vllm::SamplingParams{}, /*arrival_time=*/0.0);
    req.cache_salt = salt;
    req.block_hasher_ = get_request_block_hasher(block_size, sha256_cbor);
    req.update_block_hashes();
    return req;
  };

  // req0 (tenant A) populates the cache.
  Request req0 = make_salted("0", "tenantA");
  REQUIRE(req0.block_hashes.size() == 3);
  auto [c0, n0] = mgr->get_computed_blocks(req0);
  CHECK(c0.blocks[0].empty());
  CHECK(n0 == 0);
  auto b0 = mgr->allocate_slots(req0, /*num_new_tokens=*/55, n0,
                                std::optional<KVCacheBlocks>(c0));
  REQUIRE(b0.has_value());

  // req1 (tenant B) has the SAME tokens but a different salt -> FULL MISS.
  Request req1 = make_salted("1", "tenantB");
  // Different salt -> different block hashes than tenant A.
  CHECK(req1.block_hashes != req0.block_hashes);
  auto [c1, n1] = mgr->get_computed_blocks(req1);
  CHECK(c1.blocks[0].empty());
  CHECK(n1 == 0);  // no false-share across the salt boundary

  // req2 (tenant A again) HITS req0's cached 3-block prefix.
  Request req2 = make_salted("2", "tenantA");
  CHECK(req2.block_hashes == req0.block_hashes);
  auto [c2, n2] = mgr->get_computed_blocks(req2);
  CHECK(c2.get_block_ids() == std::vector<std::vector<int>>{{1, 2, 3}});
  CHECK(n2 == 3 * block_size);  // the cache is live for the matching salt
}

// ---------------------------------------------------------------------------
// allocate_slots: OOM -> std::nullopt (the Scheduler preempts)
// ---------------------------------------------------------------------------

TEST_CASE("KVCacheManager: allocate_slots returns std::nullopt on OOM") {
  init_none_hash(sha256_cbor);
  const int block_size = 16;
  // 4 usable blocks (num_blocks 5, minus the null block).
  auto mgr = MakeManager(MakeFullConfig(block_size, 5), block_size);

  // Ask for 10 blocks worth of tokens -> cannot fit -> nullopt.
  Request req = MakeRequest("0", Range(0, 10 * block_size), block_size);
  auto [computed, n] = mgr->get_computed_blocks(req);
  auto blocks = mgr->allocate_slots(req, /*num_new_tokens=*/10 * block_size, n,
                                    std::optional<KVCacheBlocks>(computed));
  CHECK(!blocks.has_value());
  // No blocks were consumed by the failed allocation.
  CHECK(mgr->block_pool.get_num_free_blocks() == 4);
}

// ---------------------------------------------------------------------------
// free: the LITERAL LRU eviction order (ported test_evict)
// ---------------------------------------------------------------------------

TEST_CASE(
    "KVCacheManager: free returns blocks in the pinned LRU order "
    "[6, 10, 5, 4, 3, 2, 1] (ported test_evict)") {
  init_none_hash(sha256_cbor);
  const int block_size = 16;
  auto mgr = MakeManager(MakeFullConfig(block_size, 11), block_size);

  int last = 5 * 16 + 7;  // 5 full + 1 partial block
  Request req0 = MakeRequest("0", Range(0, last), block_size);
  auto [c0, n0] = mgr->get_computed_blocks(req0);
  CHECK(c0.blocks[0].empty());
  CHECK(n0 == 0);
  auto b0 = mgr->allocate_slots(req0, last, n0, std::optional<KVCacheBlocks>(c0));
  REQUIRE(b0.has_value());
  CHECK(b0->blocks[0].size() == 6);  // 5 full + 1 partial

  Request req1 = MakeRequest("1", Range(last, last + 3 * 16), block_size);
  auto [c1, n1] = mgr->get_computed_blocks(req1);
  CHECK(c1.blocks[0].empty());
  CHECK(n1 == 0);
  auto b1 = mgr->allocate_slots(req1, 3 * 16, n1, std::optional<KVCacheBlocks>(c1));
  REQUIRE(b1.has_value());
  CHECK(b1->blocks[0].size() == 3);  // 3 full blocks

  // 10 usable - (6 + 3) == 1
  CHECK(mgr->block_pool.free_block_queue.num_free_blocks == 1);

  mgr->free(req0);
  // Partial block (unhashed, id 6) at head for immediate reuse; the pre-existing
  // free block (10) next; then the hashed blocks (5, 4, 3, 2, 1) at the tail.
  CHECK(FreeBlockIds(*mgr) == std::vector<int>{6, 10, 5, 4, 3, 2, 1});

  mgr->free(req1);
  CHECK(mgr->block_pool.free_block_queue.num_free_blocks == 10);
  CHECK(FreeBlockIds(*mgr) == std::vector<int>{6, 10, 5, 4, 3, 2, 1, 9, 8, 7});
}

// ---------------------------------------------------------------------------
// reset_prefix_cache
// ---------------------------------------------------------------------------

TEST_CASE("KVCacheManager: reset_prefix_cache clears cached prefixes") {
  init_none_hash(sha256_cbor);
  const int block_size = 16;
  auto mgr = MakeManager(MakeFullConfig(block_size, 11), block_size);

  std::vector<int32_t> tokens = Range(0, 3 * block_size);
  Request req0 = MakeRequest("0", tokens, block_size);
  auto [c0, n0] = mgr->get_computed_blocks(req0);
  auto b0 = mgr->allocate_slots(req0, 3 * block_size, n0,
                                std::optional<KVCacheBlocks>(c0));
  REQUIRE(b0.has_value());
  mgr->free(req0);

  // A second request now hits the cached 3-block prefix.
  Request req1 = MakeRequest("1", tokens, block_size);
  auto [c1, n1] = mgr->get_computed_blocks(req1);
  CHECK(n1 == 2 * block_size);  // last token withheld -> 2 full blocks hit

  // After a reset the prefix cache is empty (no in-use blocks -> succeeds).
  CHECK(mgr->reset_prefix_cache());
  Request req2 = MakeRequest("2", tokens, block_size);
  auto [c2, n2] = mgr->get_computed_blocks(req2);
  CHECK(n2 == 0);
  CHECK(c2.blocks[0].empty());
}

// ---------------------------------------------------------------------------
// Hybrid full-attn + mamba allocation (ported test_prefill_hybrid_model)
// ---------------------------------------------------------------------------

TEST_CASE("KVCacheManager: allocate_slots over a hybrid full-attn + mamba model") {
  init_none_hash(sha256_cbor);
  const int block_size = 16;
  auto mgr = MakeManager(MakeHybridConfig(block_size, 21), block_size);
  REQUIRE(mgr->num_kv_cache_groups == 2);

  // 3 full blocks (48 tokens) + 7 unique = 55 tokens.
  Request req0 = MakeRequest("0", Range(0, 55), block_size);
  auto [c0, n0] = mgr->get_computed_blocks(req0);
  CHECK(n0 == 0);

  int free_before = mgr->block_pool.get_num_free_blocks();
  auto b0 = mgr->allocate_slots(req0, 55, n0, std::optional<KVCacheBlocks>(c0));
  REQUIRE(b0.has_value());
  REQUIRE(b0->blocks.size() == 2);
  // Both groups allocate cdiv(55, 16) = 4 blocks.
  CHECK(b0->blocks[0].size() == 4);  // full attention
  CHECK(b0->blocks[1].size() == 4);  // mamba
  CHECK(mgr->block_pool.get_num_free_blocks() == free_before - 8);

  mgr->free(req0);
  CHECK(mgr->block_pool.get_num_free_blocks() == free_before);
}

// ---------------------------------------------------------------------------
// allocate_slots: the watermark / reserved / lookahead / admission-cap paths.
//
// These exercise the upstream allocate_slots branches (kv_cache_manager.py @
// e24d1b24) that M1.4's preemption drives but which no existing test touches
// (all above run with watermark == 0). Block math is derived from the pinned
// source; the pool reserves the null block, so num_blocks=N => N-1 usable free.
// ---------------------------------------------------------------------------

// (1) NON-ZERO WATERMARK. Upstream: watermark_blocks = int(watermark *
// num_blocks); it is added to required_blocks ONLY for WAITING/PREEMPTED
// requests AND only when has_scheduled_reqs is true. num_blocks=11 => 10 usable
// free; watermark 0.5 => watermark_blocks = int(0.5*11) = 5. A fresh WAITING
// request needing cdiv(96,16)=6 blocks: required = 6 + 5 = 11 > 10 free -> None,
// even though the 6 blocks would fit without the watermark.
TEST_CASE(
    "KVCacheManager: allocate_slots applies the watermark only to "
    "WAITING/PREEMPTED with scheduled reqs") {
  init_none_hash(sha256_cbor);
  const int block_size = 16;
  const int num_blocks = 11;    // 10 usable free
  const double watermark = 0.5;  // watermark_blocks = int(0.5*11) = 5

  // 6 blocks worth of tokens; 6 <= 10 free, but 6 + 5 (watermark) = 11 > 10.
  const std::vector<int32_t> tokens = Range(0, 6 * block_size);

  {
    auto mgr = MakeFullManager(block_size, num_blocks, watermark);
    CHECK(mgr->watermark_blocks == 5);
    Request req = MakeRequest("w", tokens, block_size);  // status defaults WAITING
    // has_scheduled_reqs=true (default) + WAITING -> watermark applied -> None.
    auto blocks = mgr->allocate_slots(req, /*num_new_tokens=*/6 * block_size);
    CHECK(!blocks.has_value());
    CHECK(mgr->block_pool.get_num_free_blocks() == 10);  // nothing consumed
  }

  {
    // Same request, but RUNNING -> watermark NOT applied -> the 6 blocks fit.
    auto mgr = MakeFullManager(block_size, num_blocks, watermark);
    Request req = MakeRequest("r", tokens, block_size);
    req.status = vllm::v1::RequestStatus::kRunning;
    auto blocks = mgr->allocate_slots(req, /*num_new_tokens=*/6 * block_size);
    REQUIRE(blocks.has_value());
    CHECK(blocks->blocks[0].size() == 6);
    CHECK(mgr->block_pool.get_num_free_blocks() == 10 - 6);
  }

  {
    // WAITING but has_scheduled_reqs=false -> watermark NOT applied -> fits.
    auto mgr = MakeFullManager(block_size, num_blocks, watermark);
    Request req = MakeRequest("ns", tokens, block_size);
    auto blocks = mgr->allocate_slots(
        req, /*num_new_tokens=*/6 * block_size, /*num_new_computed_tokens=*/0,
        /*new_computed_blocks=*/std::nullopt, /*num_lookahead_tokens=*/0,
        /*num_external_computed_tokens=*/0, /*delay_cache_blocks=*/false,
        /*num_encoder_tokens=*/0, /*full_sequence_must_fit=*/false,
        /*reserved_blocks=*/0, /*has_scheduled_reqs=*/false);
    REQUIRE(blocks.has_value());
    CHECK(blocks->blocks[0].size() == 6);
  }
}

// (2) reserved_blocks. Upstream: available_blocks = free - reserved_blocks;
// required_blocks (= num_blocks_to_allocate, watermark 0 here) must be <=
// available. A 6-block request fits against 10 free but not against
// (10 - reserved=5) = 5.
TEST_CASE("KVCacheManager: allocate_slots honors reserved_blocks") {
  init_none_hash(sha256_cbor);
  const int block_size = 16;
  const std::vector<int32_t> tokens = Range(0, 6 * block_size);  // 6 blocks

  {  // reserved_blocks=0 -> available 10 -> 6 fits.
    auto mgr = MakeFullManager(block_size, /*num_blocks=*/11);
    Request req = MakeRequest("res0", tokens, block_size);
    auto blocks = mgr->allocate_slots(
        req, /*num_new_tokens=*/6 * block_size, 0, std::nullopt, 0, 0, false, 0,
        false, /*reserved_blocks=*/0);
    REQUIRE(blocks.has_value());
    CHECK(blocks->blocks[0].size() == 6);
  }

  {  // reserved_blocks=5 -> available 10-5=5 -> 6 > 5 -> None.
    auto mgr = MakeFullManager(block_size, /*num_blocks=*/11);
    Request req = MakeRequest("res5", tokens, block_size);
    auto blocks = mgr->allocate_slots(
        req, /*num_new_tokens=*/6 * block_size, 0, std::nullopt, 0, 0, false, 0,
        false, /*reserved_blocks=*/5);
    CHECK(!blocks.has_value());
    CHECK(mgr->block_pool.get_num_free_blocks() == 10);  // nothing consumed
  }
}

// (3) num_lookahead_tokens. Upstream: num_tokens_need_slot =
// min(num_tokens_main_model + num_lookahead_tokens, max_model_len). A 1-block
// (16-token) request with a 16-token lookahead needs slots for 32 tokens => 2
// blocks; without lookahead it is 1 block. A huge lookahead is clamped to
// max_model_len.
TEST_CASE("KVCacheManager: allocate_slots allocates for num_lookahead_tokens") {
  init_none_hash(sha256_cbor);
  const int block_size = 16;
  const std::vector<int32_t> tokens = Range(0, block_size);  // 1 block

  {  // lookahead=0 -> 1 block.
    auto mgr = MakeFullManager(block_size, /*num_blocks=*/11);
    Request req = MakeRequest("la0", tokens, block_size);
    auto blocks = mgr->allocate_slots(req, /*num_new_tokens=*/block_size);
    REQUIRE(blocks.has_value());
    CHECK(blocks->blocks[0].size() == 1);
    CHECK(mgr->block_pool.get_num_free_blocks() == 10 - 1);
  }

  {  // lookahead=16 -> need_slot=32 -> 2 blocks.
    auto mgr = MakeFullManager(block_size, /*num_blocks=*/11);
    Request req = MakeRequest("la16", tokens, block_size);
    auto blocks = mgr->allocate_slots(req, /*num_new_tokens=*/block_size,
                                      /*num_new_computed_tokens=*/0,
                                      /*new_computed_blocks=*/std::nullopt,
                                      /*num_lookahead_tokens=*/block_size);
    REQUIRE(blocks.has_value());
    CHECK(blocks->blocks[0].size() == 2);
    CHECK(mgr->block_pool.get_num_free_blocks() == 10 - 2);
  }

  {  // Huge lookahead clamps to max_model_len=32 -> need_slot=32 -> 2 blocks.
    auto mgr = MakeFullManager(block_size, /*num_blocks=*/11, /*watermark=*/0.0,
                               /*max_model_len=*/32);
    Request req = MakeRequest("laclamp", tokens, block_size);
    auto blocks = mgr->allocate_slots(req, /*num_new_tokens=*/block_size,
                                      /*num_new_computed_tokens=*/0,
                                      /*new_computed_blocks=*/std::nullopt,
                                      /*num_lookahead_tokens=*/1000);
    REQUIRE(blocks.has_value());
    CHECK(blocks->blocks[0].size() == 2);  // clamped, not 1000/16+1 blocks
  }
}

// (4) full_sequence_must_fit admission cap. Upstream: when true, allocate_slots
// first checks the FULL sequence (min(request.num_tokens, max_model_len)) and
// returns None up front if it cannot fit, regardless of the incremental
// num_new_tokens. A 10-block request into a 4-usable-block pool is rejected even
// when only the first 1-block chunk is being scheduled; without the gate that
// first chunk would be admitted.
TEST_CASE(
    "KVCacheManager: allocate_slots full_sequence_must_fit rejects an "
    "over-large sequence up front") {
  init_none_hash(sha256_cbor);
  const int block_size = 16;
  const std::vector<int32_t> full = Range(0, 10 * block_size);  // 10 blocks

  {  // 4 usable free; full sequence needs 10 -> rejected before allocating.
    auto mgr = MakeFullManager(block_size, /*num_blocks=*/5);  // 4 free
    Request req = MakeRequest("cap", full, block_size);
    auto blocks = mgr->allocate_slots(
        req, /*num_new_tokens=*/block_size,  // only the first 1-block chunk
        0, std::nullopt, 0, 0, false, 0,
        /*full_sequence_must_fit=*/true);
    CHECK(!blocks.has_value());
    CHECK(mgr->block_pool.get_num_free_blocks() == 4);  // nothing consumed
  }

  {  // Without the admission cap, the first 1-block chunk is admitted.
    auto mgr = MakeFullManager(block_size, /*num_blocks=*/5);
    Request req = MakeRequest("nocap", full, block_size);
    auto blocks = mgr->allocate_slots(req, /*num_new_tokens=*/block_size);
    REQUIRE(blocks.has_value());
    CHECK(blocks->blocks[0].size() == 1);
  }
}
