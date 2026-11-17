// Tests for the per-group KV-cache managers — FullAttentionManager (standard
// paged K+V), SlidingWindowManager / ChunkedLocalAttentionManager
// (recycling-aware local K+V), and MambaManager (the GDN single-state manager).
//
// Ported from vllm/tests/v1/core/test_single_type_kv_cache_manager.py @ e24d1b24:
//   - test_sliding_window_possible_cached_prefix
//   - test_sliding_window_remove_skipped_blocks
//   - test_chunked_local_attention_possible_cached_prefix
//   - test_chunked_local_attention_remove_skipped_blocks
//   - test_chunked_local_attention_get_num_blocks_to_allocate
//   - test_get_num_blocks_to_allocate
//   - test_evictable_cached_blocks_not_double_allocated -> "evictable cached
//       blocks are not double allocated"
//   - test_predictor_matches_allocator_blocks_calculation_with_admission_cap
// The FullAttention and Mamba find_longest_cache_hit / lifecycle behaviors
// below follow the upstream
// mock-the-block-pool pattern (block_pool.cached_block_hash_to_block insertion)
// applied to the T0 managers, asserting the upstream method semantics.
//
// Object-identity comparisons (`block is ...`) are ported as pointer equality.
#include <doctest/doctest.h>

#include <deque>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/core/block_pool.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/single_type_kv_cache_manager.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vt/dtype.h"

using vllm::v1::BlockHash;
using vllm::v1::BlockPool;
using vllm::v1::ChunkedLocalAttentionManager;
using vllm::v1::ChunkedLocalAttentionSpec;
using vllm::v1::FullAttentionManager;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheBlock;
using vllm::v1::KVCacheSpec;
using vllm::v1::make_block_hash_with_group_id;
using vllm::v1::MambaManager;
using vllm::v1::MambaSpec;
using vllm::v1::MLAAttentionSpec;
using vllm::v1::Request;
using vllm::v1::sha256_cbor;
using vllm::v1::SlidingWindowManager;
using vllm::v1::SlidingWindowSpec;
using vt::DType;

namespace {

// block_size=2, num_kv_heads=1, head_size=1, float32 (upstream test dims).
std::shared_ptr<FullAttentionSpec> MakeFullSpec(int block_size = 2) {
  return std::make_shared<FullAttentionSpec>(block_size, /*num_kv_heads=*/1,
                                             /*head_size=*/1, DType::kF32);
}

std::shared_ptr<MambaSpec> MakeMambaSpec(int block_size = 2,
                                         const std::string& mode = "none",
                                         int num_speculative_blocks = 0) {
  return std::make_shared<MambaSpec>(
      block_size, std::vector<std::vector<int64_t>>{{2, 4}},
      std::vector<DType>{DType::kF32}, /*page_size_padded=*/std::nullopt, mode,
      num_speculative_blocks);
}

// MLA latent cache: block_size=2, ONE kv head, head_size=576
// (kv_lora_rank 512 + qk_rope_head_dim 64). kind() == kMlaAttention, yet the
// registry routes it to FullAttentionManager (kv_cache_spec_registry.cpp:75),
// exactly as upstream registers MLAAttentionSpec -> FullAttentionManager
// (single_type_kv_cache_manager.py:1538-1540).
std::shared_ptr<MLAAttentionSpec> MakeMlaSpec(int block_size = 2,
                                              int head_size = 576) {
  return std::make_shared<MLAAttentionSpec>(block_size, head_size, DType::kF32);
}

std::shared_ptr<SlidingWindowSpec> MakeSlidingSpec(
    int block_size = 2, int sliding_window = 4) {
  return std::make_shared<SlidingWindowSpec>(
      block_size, /*num_kv_heads=*/1, /*head_size=*/1, DType::kF32,
      sliding_window);
}

std::shared_ptr<ChunkedLocalAttentionSpec> MakeChunkedLocalSpec(
    int block_size = 2, int attention_chunk_size = 4) {
  return std::make_shared<ChunkedLocalAttentionSpec>(
      block_size, /*num_kv_heads=*/1, /*head_size=*/1, DType::kF32,
      attention_chunk_size);
}

// Insert a cached block for `bh` in group 0 (mirrors upstream's direct
// cached_block_hash_to_block.insert used by the manager tests).
void MockCache(BlockPool& pool, const BlockHash& bh, KVCacheBlock* blk) {
  pool.cached_block_hash_to_block[make_block_hash_with_group_id(bh, 0)]
                                 [blk->block_id] = blk;
}

Request MakeRequest(const std::string& id, const std::vector<int32_t>& tokens,
                    int block_size) {
  return Request(id, tokens, vllm::SamplingParams{}, /*arrival_time=*/0.0,
                 get_request_block_hasher(block_size, sha256_cbor));
}

std::vector<int32_t> Iota(int n) {
  std::vector<int32_t> v;
  for (int i = 0; i < n; ++i) v.push_back(i);
  return v;
}

// Pointer-stable store for standalone KVCacheBlock objects (upstream's
// KVCacheBlock(id) test fixtures).
struct BlockStore {
  std::deque<KVCacheBlock> storage;
  KVCacheBlock* make(int id) {
    storage.emplace_back(id);
    return &storage.back();
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// FullAttentionManager
// ---------------------------------------------------------------------------

TEST_CASE("R11 MTP shifted boundary: cache waits for the next committed token") {
  init_none_hash(sha256_cbor);
  for (bool recurrent : {false, true}) {
    CAPTURE(recurrent);
    BlockPool pool(100, true, 4);
    std::unique_ptr<vllm::v1::SingleTypeKVCacheManager> manager;
    if (recurrent)
      manager = std::make_unique<MambaManager>(MakeMambaSpec(4), pool, true, 0, 4);
    else
      manager = std::make_unique<FullAttentionManager>(MakeFullSpec(4), pool, true, 0, 4);
    Request req("shifted", {1, 2, 3, 4, 5, 6, 7, 8}, {}, 0,
                get_request_block_hasher(4, sha256_cbor, true));
    REQUIRE(req.block_hashes.size() == 1);
    const auto blocks = manager->allocate_new_blocks(req.request_id, 8, 8);
    REQUIRE(blocks.size() == 2);
    manager->cache_blocks(req, 8);
    CHECK(manager->num_cached_block.at(req.request_id) == 1);
    CHECK(blocks[0]->block_hash().has_value());
    CHECK_FALSE(blocks[1]->block_hash().has_value());
    // Retrying before the lookahead becomes committed must not publish/reset it.
    manager->cache_blocks(req, 8);
    CHECK(manager->num_cached_block.at(req.request_id) == 1);
    req.AppendOutputToken(9);
    REQUIRE(req.block_hashes.size() == 2);
    manager->cache_blocks(req, 8);
    CHECK(manager->num_cached_block.at(req.request_id) == 2);
    REQUIRE(blocks[1]->block_hash().has_value());
    CHECK(*blocks[1]->block_hash() == make_block_hash_with_group_id(req.block_hashes[1], 0));
    manager->cache_blocks(req, 8);
    CHECK(manager->num_cached_block.at(req.request_id) == 2);
    manager->free(req.request_id);
  }
  // Ordinary hashes still publish the exact full boundary immediately.
  BlockPool pool(100, true, 4);
  FullAttentionManager ordinary(MakeFullSpec(4), pool, true, 0, 4);
  Request req = MakeRequest("ordinary", Iota(8), 4);
  ordinary.allocate_new_blocks(req.request_id, 8, 8);
  ordinary.cache_blocks(req, 8);
  CHECK(ordinary.num_cached_block.at(req.request_id) == 2);
}

TEST_CASE("R11 MTP shifted boundary: invalid publication fails before cache mutation") {
  init_none_hash(sha256_cbor);
  BlockPool pool(100, true, 4);
  Request req("missing", {1, 2, 3, 4}, {}, 0,
              get_request_block_hasher(4, sha256_cbor, true));
  const auto blocks = pool.get_new_blocks(1);
  CHECK_THROWS_AS(pool.cache_full_blocks(req, blocks, 0, 1, 4, 0), std::runtime_error);
  CHECK_FALSE(blocks[0]->block_hash().has_value());
  CHECK(pool.cached_block_hash_to_block.empty());
  req.AppendOutputToken(5);
  REQUIRE(req.block_hashes.size() == 1);
  CHECK_THROWS_AS(pool.cache_full_blocks(req, {}, 0, 1, 4, 0), std::runtime_error);
  CHECK(pool.cached_block_hash_to_block.empty());
  pool.free_blocks(blocks);
}

TEST_CASE(
    "FullAttentionManager: allocate_new_blocks reduces pool free count and "
    "records req_to_blocks") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeFullSpec();
  FullAttentionManager mgr(spec, pool, /*enable_caching=*/true,
                           /*kv_cache_group_id=*/0, /*scheduler_block_size=*/2);

  CHECK(pool.get_num_free_blocks() == 99);
  auto blocks = mgr.allocate_new_blocks("r1", /*num_tokens=*/6,
                                        /*num_tokens_main_model=*/6);
  CHECK(blocks.size() == 3);  // cdiv(6, 2)
  CHECK(mgr.req_to_blocks["r1"].size() == 3);
  CHECK(pool.get_num_free_blocks() == 96);

  // Full attention records new block ids for the worker's block-table update.
  auto ids = mgr.take_new_block_ids();
  CHECK(ids.size() == 3);
  CHECK(mgr.take_new_block_ids().empty());
}

TEST_CASE("FullAttentionManager: get_num_blocks_to_allocate accounting") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeFullSpec();
  FullAttentionManager mgr(spec, pool, true, 0, 2);

  BlockStore store;
  std::vector<KVCacheBlock*> cached_blocks_1;
  for (int i = 0; i < 10; ++i) cached_blocks_1.push_back(store.make(i + 1));
  std::vector<KVCacheBlock*> cached_blocks_2;
  for (int i = 0; i < 5; ++i) cached_blocks_2.push_back(pool.null_block);
  for (int i = 0; i < 5; ++i) cached_blocks_2.push_back(store.make(i + 1));

  // 10 required-suffix blocks + 10 evictable computed blocks.
  CHECK(mgr.get_num_blocks_to_allocate("1", /*num_tokens=*/40, cached_blocks_1,
                                       /*total_computed_tokens=*/0,
                                       /*num_tokens_main_model=*/40) == 20);
  // 5 of the computed blocks are null (not evictable) -> 10 + 5.
  CHECK(mgr.get_num_blocks_to_allocate("2", 40, cached_blocks_2, 0, 40) == 15);
}

TEST_CASE("FullAttentionManager: evictable cached blocks are not double allocated") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeFullSpec();
  FullAttentionManager mgr(spec, pool, /*enable_caching=*/true, 0, 2);

  KVCacheBlock* evictable_block = &pool.blocks[1];  // ref_cnt 0, eviction cand.

  int n = mgr.get_num_blocks_to_allocate(
      "req", /*num_tokens=*/4, {evictable_block},
      /*total_computed_tokens=*/2, /*num_tokens_main_model=*/4);
  // Free-capacity check counts the evictable block, but only 1 truly new block.
  CHECK(n == 2);

  mgr.add_local_computed_blocks("req", {evictable_block},
                                /*num_local_computed_tokens=*/2,
                                /*num_external_computed_tokens=*/0);
  auto new_blocks = mgr.allocate_new_blocks("req", 4, 4);
  CHECK(new_blocks.size() == 1);
  CHECK(mgr.req_to_blocks["req"].size() == 2);
}

TEST_CASE("FullAttentionManager: find_longest_cache_hit walks the multi-block prefix") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeFullSpec();
  FullAttentionManager mgr(spec, pool, true, 0, 2);

  std::vector<BlockHash> bh = {"h0", "h1", "h2"};

  SUBCASE("prefix hit stops at the first miss") {
    MockCache(pool, "h0", &pool.blocks[10]);
    MockCache(pool, "h1", &pool.blocks[11]);
    // h2 not cached.
    auto computed = mgr.find_longest_cache_hit(
        bh, /*max_length=*/6, {0}, pool, *spec, /*drop_eagle_block=*/false,
        /*alignment_tokens=*/2)[0];
    CHECK(computed.size() == 2);
    CHECK(computed[0] == &pool.blocks[10]);
    CHECK(computed[1] == &pool.blocks[11]);
  }

  SUBCASE("full hit returns every block") {
    MockCache(pool, "h0", &pool.blocks[10]);
    MockCache(pool, "h1", &pool.blocks[11]);
    MockCache(pool, "h2", &pool.blocks[12]);
    auto computed =
        mgr.find_longest_cache_hit(bh, 6, {0}, pool, *spec, false, 2)[0];
    CHECK(computed.size() == 3);
    CHECK(computed[2] == &pool.blocks[12]);
  }

  SUBCASE("no hit returns empty") {
    auto computed =
        mgr.find_longest_cache_hit(bh, 6, {0}, pool, *spec, false, 2)[0];
    CHECK(computed.empty());
  }
}

TEST_CASE(
    "FullAttentionManager: cache_blocks then find_longest_cache_hit returns the "
    "cached prefix (end-to-end)") {
  init_none_hash(sha256_cbor);
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeFullSpec();
  FullAttentionManager mgr(spec, pool, true, 0, 2);

  Request req = MakeRequest("r1", Iota(6), /*block_size=*/2);
  REQUIRE(req.block_hashes.size() == 3);

  auto allocated = mgr.allocate_new_blocks("r1", 6, 6);
  REQUIRE(allocated.size() == 3);
  mgr.cache_blocks(req, /*num_tokens=*/6);

  auto computed = mgr.find_longest_cache_hit(req.block_hashes, 6, {0}, pool,
                                             *spec, false, 2)[0];
  CHECK(computed.size() == 3);
  for (int i = 0; i < 3; ++i) {
    CHECK(computed[static_cast<size_t>(i)]->block_id == allocated[static_cast<size_t>(i)]->block_id);
  }
}

// ---------------------------------------------------------------------------
// MLAAttentionSpec via FullAttentionManager — the MLA prefix-cache-hit path.
//
// Regression for the DeepSeek-V2 SACRED-gate abort: FullAttentionManager is the
// registered manager for MLAAttentionSpec (kind() == kMlaAttention), exactly as
// upstream registers MLAAttentionSpec -> FullAttentionManager
// (single_type_kv_cache_manager.py:1538-1540) and its find_longest_cache_hit
// admits `FullAttentionSpec | ChunkedLocalAttentionSpec` (:578-582) — MLA IS-A
// FullAttentionSpec subclass, so the isinstance check passes for it upstream.
// Our C++ kind()-equality precondition previously accepted ONLY kFullAttention,
// so with automatic prefix caching ON (DeepSeek's default) the very first
// cache-hit lookup aborted the assert. These cases prove (a) the lookup does not
// abort for an MLA group and (b) the restored prefix is byte-identical to the
// cold allocation — the same physical KV pages, i.e. a pure reuse of already
// computed latent rows, which is what makes a cached-prefix decode reproduce the
// uncached tokens exactly.
// ---------------------------------------------------------------------------
TEST_CASE(
    "FullAttentionManager (MLA spec): find_longest_cache_hit walks the prefix "
    "without aborting the kind assertion") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeMlaSpec();
  REQUIRE(spec->kind() == vllm::v1::KVCacheSpecKind::kMlaAttention);
  FullAttentionManager mgr(spec, pool, /*enable_caching=*/true,
                           /*kv_cache_group_id=*/0, /*scheduler_block_size=*/2);

  std::vector<BlockHash> bh = {"h0", "h1", "h2"};
  MockCache(pool, "h0", &pool.blocks[10]);
  MockCache(pool, "h1", &pool.blocks[11]);
  // h2 not cached: the walk must stop after the two-block prefix, not abort.
  auto computed = mgr.find_longest_cache_hit(
      bh, /*max_length=*/6, {0}, pool, *spec, /*drop_eagle_block=*/false,
      /*alignment_tokens=*/2)[0];
  REQUIRE(computed.size() == 2);
  CHECK(computed[0] == &pool.blocks[10]);
  CHECK(computed[1] == &pool.blocks[11]);
}

TEST_CASE(
    "FullAttentionManager (MLA spec): cache_blocks then find_longest_cache_hit "
    "restores the byte-identical prefix (end-to-end MLA prefix reuse)") {
  init_none_hash(sha256_cbor);
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeMlaSpec();
  FullAttentionManager mgr(spec, pool, true, 0, 2);

  // Request A fills three whole latent blocks and caches them.
  Request reqA = MakeRequest("A", Iota(6), /*block_size=*/2);
  REQUIRE(reqA.block_hashes.size() == 3);
  auto allocated = mgr.allocate_new_blocks("A", 6, 6);
  REQUIRE(allocated.size() == 3);
  mgr.cache_blocks(reqA, /*num_tokens=*/6);

  // Request B shares A's full 6-token prefix: the deterministic block hashes hit
  // A's cached latent pages. The restored blocks must be the SAME physical pages
  // A wrote — a cached-prefix decode then reads byte-identical latent rows and
  // reproduces the cold-run tokens exactly.
  Request reqB = MakeRequest("B", Iota(6), /*block_size=*/2);
  REQUIRE(reqB.block_hashes == reqA.block_hashes);
  auto computed = mgr.find_longest_cache_hit(reqB.block_hashes, 6, {0}, pool,
                                             *spec, false, 2)[0];
  REQUIRE(computed.size() == 3);
  for (int i = 0; i < 3; ++i) {
    CHECK(computed[static_cast<size_t>(i)]->block_id ==
          allocated[static_cast<size_t>(i)]->block_id);
  }
}

TEST_CASE("FullAttentionManager: free returns blocks to the pool") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeFullSpec();
  FullAttentionManager mgr(spec, pool, true, 0, 2);

  mgr.allocate_new_blocks("r1", 6, 6);
  CHECK(pool.get_num_free_blocks() == 96);
  mgr.free("r1");
  CHECK(pool.get_num_free_blocks() == 99);
  CHECK(mgr.req_to_blocks.count("r1") == 0);
  CHECK(mgr.num_cached_block.count("r1") == 0);
}

TEST_CASE("FullAttentionManager: get_num_common_prefix_blocks") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeFullSpec();
  FullAttentionManager mgr(spec, pool, true, 0, 2);

  mgr.allocate_new_blocks("r1", 6, 6);  // 3 blocks, ref_cnt 1 each.
  // Single request: all its blocks are common (ref_cnt == #requests == 1).
  CHECK(mgr.get_num_common_prefix_blocks("r1") == 3);

  // A second, unshared request drops the common count: r1's blocks have
  // ref_cnt 1 != 2.
  mgr.allocate_new_blocks("r2", 4, 4);
  CHECK(mgr.get_num_common_prefix_blocks("r1") == 0);
}

// ---------------------------------------------------------------------------
// SlidingWindowManager
// ---------------------------------------------------------------------------

TEST_CASE("SlidingWindowManager: possible cached prefix searches right-to-left") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeSlidingSpec(/*block_size=*/2, /*sliding_window=*/4);
  SlidingWindowManager mgr(spec, pool, true, 0, 2,
                           /*max_admission_blocks_per_request=*/1000000000);

  auto run_case = [&](const std::vector<bool>& cached, int expected_length) {
    pool.cached_block_hash_to_block.clear();
    std::vector<BlockHash> hashes;
    for (size_t i = 0; i < cached.size(); ++i) {
      hashes.push_back("h" + std::to_string(i));
      if (cached[i]) {
        MockCache(pool, hashes.back(), &pool.blocks[i + 10]);
      }
    }

    auto computed = mgr.find_longest_cache_hit(
        hashes, static_cast<int>(hashes.size()) * 2, {0}, pool, *spec,
        /*drop_eagle_block=*/false, /*alignment_tokens=*/2)[0];
    REQUIRE(computed.size() == static_cast<size_t>(expected_length));
    for (int i = 0; i < expected_length - 2; ++i) {
      CHECK(computed[static_cast<size_t>(i)] == pool.null_block);
    }
    for (int offset = 0; offset < 2 && offset < expected_length; ++offset) {
      const int index = expected_length - offset - 1;
      CHECK(computed[static_cast<size_t>(index)]->block_id == index + 10);
    }
  };

  run_case(std::vector<bool>(10, false), 0);
  run_case({true}, 1);
  run_case({true, false}, 1);
  run_case({true, true}, 2);
  run_case({true, true, false}, 2);
  run_case({true, true, true}, 3);
  run_case({true, true, true, false}, 3);
  run_case({true, true, false, true, false, false, true, true, false, true,
            true, true},
           12);
  run_case({true, true, false, true, false, false, true, true, false, false,
            false},
           8);
  run_case({true, true, false, true, false, false, true, true, false, false,
            false, true},
           8);
}

TEST_CASE("SlidingWindowManager: alignment and EAGLE require the lookahead block") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/8);
  auto spec = MakeSlidingSpec(/*block_size=*/8, /*sliding_window=*/8);
  SlidingWindowManager mgr(spec, pool, true, 0,
                           /*scheduler_block_size=*/32,
                           /*max_admission_blocks_per_request=*/100);
  std::vector<BlockHash> hashes = {"h0", "h1", "h2", "h3", "h4"};

  SUBCASE("non-EAGLE hit lands on the aligned boundary") {
    MockCache(pool, "h3", &pool.blocks[13]);
    auto hit = mgr.find_longest_cache_hit(hashes, 40, {0}, pool, *spec,
                                          /*drop_eagle_block=*/false,
                                          /*alignment_tokens=*/32)[0];
    REQUIRE(hit.size() == 4);
    CHECK(hit[3] == &pool.blocks[13]);
  }

  SUBCASE("EAGLE caches one block past the boundary and drops it") {
    MockCache(pool, "h3", &pool.blocks[13]);
    MockCache(pool, "h4", &pool.blocks[14]);
    auto hit = mgr.find_longest_cache_hit(hashes, 40, {0}, pool, *spec,
                                          /*drop_eagle_block=*/true,
                                          /*alignment_tokens=*/32)[0];
    REQUIRE(hit.size() == 4);
    CHECK(hit[3] == &pool.blocks[13]);
  }

  SUBCASE("EAGLE rejects a tail without its post-boundary lookahead") {
    MockCache(pool, "h3", &pool.blocks[13]);
    auto hit = mgr.find_longest_cache_hit(hashes, 40, {0}, pool, *spec,
                                          /*drop_eagle_block=*/true,
                                          /*alignment_tokens=*/32)[0];
    CHECK(hit.empty());
  }
}

TEST_CASE("SlidingWindowManager: reachable mask mirrors dense and sparse tails") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/16);
  auto spec = MakeSlidingSpec(/*block_size=*/16, /*sliding_window=*/16);
  SlidingWindowManager mgr(spec, pool, true, 0, 16, 100);

  // Dense default: every 16-token boundary is reachable.
  CHECK_FALSE(mgr.reachable_block_mask(0, 16, 16).has_value());

  // retention=64: segment tails 3/7/11/15 plus the exact-prompt replay tail 14.
  auto sparse = mgr.reachable_block_mask(
      0, 16, 16, /*retention_interval=*/64,
      /*num_prompt_tokens=*/256);
  REQUIRE(sparse.has_value());
  std::vector<int> set_indices;
  for (int i = 0; i < 16; ++i) {
    if ((*sparse)[static_cast<size_t>(i)]) set_indices.push_back(i);
  }
  CHECK(set_indices == std::vector<int>{3, 7, 11, 14, 15});

  // retention=0 keeps only the latest replay tail.
  auto latest_only = mgr.reachable_block_mask(0, 16, 16, 0, 256);
  REQUIRE(latest_only.has_value());
  set_indices.clear();
  for (int i = 0; i < 16; ++i) {
    if ((*latest_only)[static_cast<size_t>(i)]) set_indices.push_back(i);
  }
  CHECK(set_indices == std::vector<int>{14});

  // EAGLE shifts each tail across the boundary and requires two contiguous
  // blocks for a one-block window (tail + lookahead).
  auto eagle_spec = MakeSlidingSpec(/*block_size=*/8, /*sliding_window=*/8);
  SlidingWindowManager eagle(eagle_spec, pool, true, 0, 32, 100);
  eagle.use_eagle = true;
  auto eagle_mask = eagle.reachable_block_mask(0, 9, 32);
  REQUIRE(eagle_mask.has_value());
  set_indices.clear();
  for (int i = 0; i < 9; ++i) {
    if ((*eagle_mask)[static_cast<size_t>(i)]) set_indices.push_back(i);
  }
  CHECK(set_indices == std::vector<int>{3, 4, 7, 8});
}

TEST_CASE("SlidingWindowManager: skipped blocks recycle whole pages only") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeSlidingSpec(/*block_size=*/2, /*sliding_window=*/4);
  SlidingWindowManager mgr(spec, pool, true, 0, 2, 100);
  auto allocated = mgr.allocate_new_blocks("test", /*num_tokens=*/22,
                                            /*num_tokens_main_model=*/22);
  REQUIRE(allocated.size() == 11);
  std::vector<int> original_ids;
  for (KVCacheBlock* block : allocated) original_ids.push_back(block->block_id);

  auto check_prefix = [&](int null_count) {
    const auto& blocks = mgr.req_to_blocks["test"];
    for (int i = 0; i < 11; ++i) {
      if (i < null_count) {
        CHECK(blocks[static_cast<size_t>(i)] == pool.null_block);
      } else {
        CHECK(blocks[static_cast<size_t>(i)]->block_id ==
              original_ids[static_cast<size_t>(i)]);
      }
    }
  };

  mgr.remove_skipped_blocks("test", 0);
  check_prefix(0);
  mgr.remove_skipped_blocks("test", 4);
  check_prefix(0);  // one skipped token is not a whole page.
  mgr.remove_skipped_blocks("test", 5);
  check_prefix(1);
  mgr.remove_skipped_blocks("test", 6);
  check_prefix(1);
  mgr.remove_skipped_blocks("test", 7);
  check_prefix(2);
  mgr.remove_skipped_blocks("test", 11);
  check_prefix(4);

  CHECK(mgr.get_num_skipped_tokens(0) == 0);
  CHECK(mgr.get_num_skipped_tokens(4) == 1);
  CHECK(mgr.get_num_skipped_tokens(7) == 4);
  CHECK(mgr.get_num_common_prefix_blocks("test") == 0);
}

TEST_CASE("SlidingWindowManager: allocation and admission-cap accounting") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeSlidingSpec(/*block_size=*/2, /*sliding_window=*/4);
  SlidingWindowManager mgr(spec, pool, true, 0, 2,
                           /*max_admission_blocks_per_request=*/4);
  BlockStore store;
  std::vector<KVCacheBlock*> cached_1;
  for (int i = 0; i < 10; ++i) cached_1.push_back(store.make(i + 1));
  std::vector<KVCacheBlock*> cached_2(5, pool.null_block);
  for (int i = 0; i < 5; ++i) cached_2.push_back(store.make(i + 20));

  CHECK(mgr.get_num_blocks_to_allocate("1", 40, cached_1, 0, 40) == 20);
  CHECK(mgr.get_num_blocks_to_allocate("2", 40, cached_2, 0, 40) == 15);

  // The startup full-sequence admission check is capped, while the per-step
  // predictor remains uncapped to match allocate_new_blocks.
  CHECK(mgr.get_num_blocks_to_allocate("admit", 40, {}, 0, 40,
                                       /*apply_admission_cap=*/true) == 4);
  CHECK(mgr.get_num_blocks_to_allocate("step", 40, {}, 0, 40,
                                       /*apply_admission_cap=*/false) == 20);
}

TEST_CASE("SlidingWindowManager: randomized predictor and recycling properties") {
  std::mt19937 rng(0x5a17u);
  for (int trial = 0; trial < 40; ++trial) {
    const int block_size = trial % 2 == 0 ? 2 : 4;
    const int sliding_window = block_size * (2 + trial % 5);
    const int max_batched_tokens =
        block_size * (1 + trial % 4) - trial % block_size;
    const int max_model_len =
        block_size * (16 + trial % 17) + trial % block_size;
    auto spec = MakeSlidingSpec(block_size, sliding_window);
    const int cap = spec->max_admission_blocks_per_request(
        max_batched_tokens, max_model_len);
    BlockPool pool(/*num_gpu_blocks=*/256, /*enable_caching=*/false,
                   /*hash_block_size=*/block_size);
    SlidingWindowManager mgr(spec, pool, false, 0, block_size, cap);
    const std::string request_id = "r";
    std::unordered_map<int, std::vector<int>> block_contents;

    CHECK(mgr.get_num_blocks_to_allocate(
              request_id, max_model_len, {}, 0, max_model_len,
              /*apply_admission_cap=*/true) <= cap);

    int computed = 0;
    while (computed < max_model_len) {
      mgr.remove_skipped_blocks(request_id, computed);
      const int chunk =
          1 + static_cast<int>(rng() % max_batched_tokens);
      const int num_tokens = std::min(computed + chunk, max_model_len);
      const int predicted = mgr.get_num_blocks_to_allocate(
          request_id, num_tokens, {}, computed, num_tokens);
      auto allocated =
          mgr.allocate_new_blocks(request_id, num_tokens, num_tokens);
      CHECK(predicted == static_cast<int>(allocated.size()));
      computed = num_tokens;

      int held_blocks = 0;
      const auto& table = mgr.req_to_blocks[request_id];
      for (size_t logical_block = 0; logical_block < table.size();
           ++logical_block) {
        KVCacheBlock* block = table[logical_block];
        if (block == pool.null_block) continue;
        ++held_blocks;
        CHECK(block->ref_cnt > 0);
        CHECK(block->prev_free_block == nullptr);
        CHECK(block->next_free_block == nullptr);

        // Model the KV writes for this logical page. A recycled physical block
        // overwrites its old contents, just as the worker kernels do.
        auto& contents = block_contents[block->block_id];
        contents.assign(static_cast<size_t>(block_size), -1);
        for (int offset = 0; offset < block_size; ++offset) {
          const int token =
              static_cast<int>(logical_block) * block_size + offset;
          if (token < computed) {
            contents[static_cast<size_t>(offset)] = token;
          }
        }
      }
      CHECK(held_blocks <= cap);

      // Compare every token visible to the next attention step with the
      // full-allocation oracle. This checks that recycling changes storage,
      // never the logical token suffix presented to attention.
      const int visible_start =
          std::max(0, computed - sliding_window + 1);
      for (int token = visible_start; token < computed; ++token) {
        const size_t logical_block =
            static_cast<size_t>(token / block_size);
        REQUIRE(logical_block < table.size());
        KVCacheBlock* block = table[logical_block];
        REQUIRE(block != pool.null_block);
        REQUIRE(block_contents.count(block->block_id) == 1);
        CHECK(block_contents[block->block_id]
                            [static_cast<size_t>(token % block_size)] == token);
      }
    }

    // Free/preempt the request, then prove those physical pages can be
    // reallocated cleanly to a fresh request without a live page remaining in
    // the free queue.
    mgr.free(request_id);
    CHECK(mgr.req_to_blocks.count(request_id) == 0);
    CHECK(pool.get_num_free_blocks() == 255);
    auto fresh = mgr.allocate_new_blocks("fresh", max_batched_tokens,
                                         max_batched_tokens);
    CHECK_FALSE(fresh.empty());
    for (KVCacheBlock* block : fresh) {
      CHECK(block->ref_cnt > 0);
      CHECK(block->prev_free_block == nullptr);
      CHECK(block->next_free_block == nullptr);
    }
    mgr.free("fresh");
    CHECK(pool.get_num_free_blocks() == 255);
  }
}

// ---------------------------------------------------------------------------
// ChunkedLocalAttentionManager
// ---------------------------------------------------------------------------

TEST_CASE("ChunkedLocalAttentionManager: possible cached prefix stays in one chunk") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeChunkedLocalSpec(/*block_size=*/2,
                                   /*attention_chunk_size=*/4);
  ChunkedLocalAttentionManager mgr(
      spec, pool, true, 0, 2,
      /*max_admission_blocks_per_request=*/1000000000);

  auto run_case = [&](const std::vector<bool>& cached, int tail_token,
                      int expected_length) {
    pool.cached_block_hash_to_block.clear();
    std::vector<BlockHash> hashes;
    for (size_t i = 0; i < cached.size(); ++i) {
      hashes.push_back("h" + std::to_string(i));
      if (cached[i]) {
        MockCache(pool, hashes.back(), &pool.blocks[i + 10]);
      }
    }

    const int max_length = static_cast<int>(hashes.size()) * 2 + tail_token;
    auto computed = mgr.find_longest_cache_hit(
        hashes, max_length, {0}, pool, *spec,
        /*drop_eagle_block=*/false, /*alignment_tokens=*/2)[0];
    REQUIRE(computed.size() == static_cast<size_t>(expected_length));
    const int chunk_start_block = max_length / 4 * 4 / 2;
    for (int i = 0; i < std::min(chunk_start_block, expected_length); ++i) {
      CHECK(computed[static_cast<size_t>(i)] == pool.null_block);
    }
    for (int i = chunk_start_block; i < expected_length; ++i) {
      REQUIRE(cached[static_cast<size_t>(i)]);
      CHECK(computed[static_cast<size_t>(i)] == &pool.blocks[i + 10]);
    }
  };

  // Exact vectors from upstream
  // test_chunked_local_attention_possible_cached_prefix.
  run_case({true}, 0, 1);
  run_case({true}, 1, 1);
  run_case({true, false}, 0, 2);
  run_case({true, false}, 1, 2);
  run_case({true, true}, 0, 2);
  run_case({true, true}, 1, 2);
  run_case({true, true, false}, 0, 2);
  run_case({true, true, false}, 1, 2);
  run_case({true, true, true}, 0, 3);
  run_case({true, true, true}, 1, 3);
  run_case({true, true, true, false}, 0, 4);
  run_case({true, true, true, false}, 1, 4);
  run_case({false, true, false, true, false, true, false, true, true}, 1, 9);
  run_case({true, false, true, false, true, false, true, false, false}, 1, 8);
  run_case({false, true, false, true, false, true, false, true, true, true}, 1,
           10);
  run_case({true, false, true, false, true, false, true, false, true, false}, 0,
           10);
  run_case({true, false, true, false, true, false, true, false, true, false}, 1,
           10);
  run_case({false, true, false, true, false, true, false, true, false, true}, 0,
           10);
  run_case({false, true, false, true, false, true, false, true, false, true}, 1,
           10);
  run_case({true, false, true, false, true, false, true, false, false, false}, 0,
           10);
  run_case({true, false, true, false, true, false, true, false, false, false}, 1,
           10);
}

TEST_CASE("ChunkedLocalAttentionManager: skipped blocks recycle at chunk boundaries") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeChunkedLocalSpec(/*block_size=*/2,
                                   /*attention_chunk_size=*/4);
  ChunkedLocalAttentionManager mgr(spec, pool, true, 0, 2, 100);
  auto allocated = mgr.allocate_new_blocks("test", /*num_tokens=*/22,
                                            /*num_tokens_main_model=*/22);
  REQUIRE(allocated.size() == 11);
  std::vector<int> original_ids;
  for (KVCacheBlock* block : allocated) original_ids.push_back(block->block_id);

  auto check_prefix = [&](int null_count) {
    const auto& blocks = mgr.req_to_blocks["test"];
    for (int i = 0; i < 11; ++i) {
      if (i < null_count) {
        CHECK(blocks[static_cast<size_t>(i)] == pool.null_block);
      } else {
        CHECK(blocks[static_cast<size_t>(i)]->block_id ==
              original_ids[static_cast<size_t>(i)]);
      }
    }
  };

  mgr.remove_skipped_blocks("test", 0);
  check_prefix(0);
  mgr.remove_skipped_blocks("test", 4);
  check_prefix(2);
  mgr.remove_skipped_blocks("test", 6);
  check_prefix(2);
  mgr.remove_skipped_blocks("test", 12);
  check_prefix(6);

  CHECK(mgr.get_num_skipped_tokens(0) == 0);
  CHECK(mgr.get_num_skipped_tokens(3) == 0);
  CHECK(mgr.get_num_skipped_tokens(4) == 4);
  CHECK(mgr.get_num_skipped_tokens(13) == 12);
  CHECK(mgr.get_num_common_prefix_blocks("test") == 0);
}

TEST_CASE("ChunkedLocalAttentionManager: allocation and admission-cap accounting") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeChunkedLocalSpec(/*block_size=*/2,
                                   /*attention_chunk_size=*/4);
  ChunkedLocalAttentionManager mgr(
      spec, pool, true, 0, 2,
      /*max_admission_blocks_per_request=*/4);
  BlockStore store;
  std::vector<KVCacheBlock*> cached_1;
  for (int i = 0; i < 10; ++i) cached_1.push_back(store.make(i + 1));
  std::vector<KVCacheBlock*> cached_2(5, pool.null_block);
  for (int i = 0; i < 5; ++i) cached_2.push_back(store.make(i + 20));

  CHECK(mgr.get_num_blocks_to_allocate("1", 40, cached_1, 0, 40) == 20);
  CHECK(mgr.get_num_blocks_to_allocate("2", 40, cached_2, 0, 40) == 15);
  CHECK(mgr.get_num_blocks_to_allocate("admit", 40, {}, 0, 40,
                                       /*apply_admission_cap=*/true) == 4);
  CHECK(mgr.get_num_blocks_to_allocate("step", 40, {}, 0, 40,
                                       /*apply_admission_cap=*/false) == 20);
}

TEST_CASE("ChunkedLocalAttentionManager: rejects unsupported EAGLE and parallel layouts") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeChunkedLocalSpec();
  ChunkedLocalAttentionManager mgr(spec, pool, true, 0, 2, 100);
  std::vector<BlockHash> hashes = {"h0", "h1"};

  CHECK_THROWS_AS(mgr.find_longest_cache_hit(
                      hashes, 4, {0}, pool, *spec,
                      /*drop_eagle_block=*/true, 2),
                  std::invalid_argument);
  CHECK_THROWS_AS(mgr.find_longest_cache_hit(
                      hashes, 4, {0}, pool, *spec, false, 2,
                      /*dcp_world_size=*/2, /*pcp_world_size=*/1),
                  std::invalid_argument);
  CHECK_THROWS_AS(mgr.find_longest_cache_hit(
                      hashes, 4, {0}, pool, *spec, false, 2,
                      /*dcp_world_size=*/1, /*pcp_world_size=*/2),
                  std::invalid_argument);
  CHECK_THROWS_AS(mgr.find_longest_cache_hit(
                      hashes, 4, {0}, pool, *spec, false,
                      /*alignment_tokens=*/4),
                  std::invalid_argument);
}

TEST_CASE("ChunkedLocalAttentionManager: randomized predictor and recycling properties") {
  std::mt19937 rng(0xc4a11u);
  for (int trial = 0; trial < 40; ++trial) {
    const int block_size = trial % 2 == 0 ? 2 : 4;
    const int attention_chunk_size = block_size * (2 + trial % 5);
    const int max_batched_tokens =
        block_size * (1 + trial % 4) - trial % block_size;
    const int max_model_len =
        block_size * (16 + trial % 17) + trial % block_size;
    auto spec = MakeChunkedLocalSpec(block_size, attention_chunk_size);
    const int cap = spec->max_admission_blocks_per_request(
        max_batched_tokens, max_model_len);
    BlockPool pool(/*num_gpu_blocks=*/512, /*enable_caching=*/false,
                   /*hash_block_size=*/block_size);
    ChunkedLocalAttentionManager mgr(spec, pool, false, 0, block_size, cap);
    const std::string request_id = "r";
    std::unordered_map<int, std::vector<int>> block_contents;

    CHECK(mgr.get_num_blocks_to_allocate(
              request_id, max_model_len, {}, 0, max_model_len,
              /*apply_admission_cap=*/true) <= cap);

    int computed = 0;
    while (computed < max_model_len) {
      mgr.remove_skipped_blocks(request_id, computed);
      const int batch = 1 + static_cast<int>(rng() % max_batched_tokens);
      const int num_tokens = std::min(computed + batch, max_model_len);
      const int predicted = mgr.get_num_blocks_to_allocate(
          request_id, num_tokens, {}, computed, num_tokens);
      auto newly_allocated =
          mgr.allocate_new_blocks(request_id, num_tokens, num_tokens);
      CHECK(predicted == static_cast<int>(newly_allocated.size()));
      computed = num_tokens;

      int held_blocks = 0;
      const auto& table = mgr.req_to_blocks[request_id];
      for (size_t logical_block = 0; logical_block < table.size();
           ++logical_block) {
        KVCacheBlock* block = table[logical_block];
        if (block == pool.null_block) continue;
        ++held_blocks;
        CHECK(block->ref_cnt > 0);
        CHECK(block->prev_free_block == nullptr);
        CHECK(block->next_free_block == nullptr);

        auto& contents = block_contents[block->block_id];
        contents.assign(static_cast<size_t>(block_size), -1);
        for (int offset = 0; offset < block_size; ++offset) {
          const int token =
              static_cast<int>(logical_block) * block_size + offset;
          if (token < computed) {
            contents[static_cast<size_t>(offset)] = token;
          }
        }
      }
      CHECK(held_blocks <= cap);

      const int visible_start =
          computed / attention_chunk_size * attention_chunk_size;
      for (int token = visible_start; token < computed; ++token) {
        const size_t logical_block =
            static_cast<size_t>(token / block_size);
        REQUIRE(logical_block < table.size());
        KVCacheBlock* block = table[logical_block];
        REQUIRE(block != pool.null_block);
        REQUIRE(block_contents.count(block->block_id) == 1);
        CHECK(block_contents[block->block_id]
                            [static_cast<size_t>(token % block_size)] == token);
      }
    }

    mgr.free(request_id);
    CHECK(mgr.req_to_blocks.count(request_id) == 0);
    CHECK(pool.get_num_free_blocks() == 511);
  }
}

// ---------------------------------------------------------------------------
// MambaManager
// ---------------------------------------------------------------------------

TEST_CASE("MambaManager: find_longest_cache_hit keeps only the rightmost state") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeMambaSpec();
  MambaManager mgr(spec, pool, true, 0, 2);

  std::vector<BlockHash> bh = {"h0", "h1", "h2"};

  SUBCASE("all cached: [null, null, state] — single tail state") {
    MockCache(pool, "h0", &pool.blocks[10]);
    MockCache(pool, "h1", &pool.blocks[11]);
    MockCache(pool, "h2", &pool.blocks[12]);
    auto computed = mgr.find_longest_cache_hit(bh, /*max_length=*/6, {0}, pool,
                                               *spec, false, 2)[0];
    REQUIRE(computed.size() == 3);
    CHECK(computed[0] == pool.null_block);
    CHECK(computed[1] == pool.null_block);
    CHECK(computed[2] == &pool.blocks[12]);
  }

  SUBCASE("rightmost cached is h1 (h2 miss): [null, state]") {
    MockCache(pool, "h0", &pool.blocks[10]);
    MockCache(pool, "h1", &pool.blocks[11]);
    auto computed =
        mgr.find_longest_cache_hit(bh, 6, {0}, pool, *spec, false, 2)[0];
    REQUIRE(computed.size() == 2);
    CHECK(computed[0] == pool.null_block);
    CHECK(computed[1] == &pool.blocks[11]);
  }

  SUBCASE("no hit returns empty") {
    auto computed =
        mgr.find_longest_cache_hit(bh, 6, {0}, pool, *spec, false, 2)[0];
    CHECK(computed.empty());
  }
  SUBCASE("PR11 draft drop excludes the final recurrent snapshot") {
    MockCache(pool, "h0", &pool.blocks[10]);
    MockCache(pool, "h1", &pool.blocks[11]);
    MockCache(pool, "h2", &pool.blocks[12]);
    const auto computed = mgr.find_longest_cache_hit(bh, 6, {0}, pool, *spec, true, 2)[0];
    REQUIRE(computed.size() == 2);
    CHECK(computed[0] == pool.null_block);
    CHECK(computed[1] == &pool.blocks[11]);
    CHECK(mgr.find_longest_cache_hit(bh, 2, {0}, pool, *spec, true, 2)[0].empty());
    // Published pages are read-only throughout the search.
    CHECK(pool.get_cached_block("h2", {0}).value()[0] == &pool.blocks[12]);
  }
}

TEST_CASE("MambaManager: allocate_new_blocks (mode none) does not record block ids") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeMambaSpec();
  MambaManager mgr(spec, pool, true, 0, 2);

  auto blocks = mgr.allocate_new_blocks("r", 6, 6);
  CHECK(blocks.size() == 3);
  CHECK(mgr.req_to_blocks["r"].size() == 3);
  CHECK(pool.get_num_free_blocks() == 96);
  // Mamba is not a full-attention spec, so new_block_ids stays empty.
  CHECK(mgr.take_new_block_ids().empty());

  mgr.free("r");
  CHECK(pool.get_num_free_blocks() == 99);
}

TEST_CASE("MambaManager: get_num_common_prefix_blocks is always 0") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeMambaSpec();
  MambaManager mgr(spec, pool, true, 0, 2);
  mgr.allocate_new_blocks("r", 6, 6);
  CHECK(mgr.get_num_common_prefix_blocks("r") == 0);
}

TEST_CASE("MambaManager: remove_skipped_blocks frees all but the tail state") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeMambaSpec();
  MambaManager mgr(spec, pool, true, 0, 2);

  mgr.allocate_new_blocks("r", 6, 6);  // 3 blocks.
  int free_before = pool.get_num_free_blocks();

  // get_num_skipped_tokens(6) = 5 -> 5 // 2 = 2 leading blocks freed.
  mgr.remove_skipped_blocks("r", /*num_computed_tokens=*/6);
  auto& blocks = mgr.req_to_blocks["r"];
  CHECK(blocks[0] == pool.null_block);
  CHECK(blocks[1] == pool.null_block);
  CHECK(blocks[2] != pool.null_block);
  CHECK(pool.get_num_free_blocks() == free_before + 2);
}

// ---------------------------------------------------------------------------
// MambaManager mode "none" prefix caching (the GDN gate-model path). These
// exercise the manager end-to-end against a real (non-mocked) BlockPool cache
// populated by cache_blocks, mirroring upstream MambaManager semantics
// (single-recurrent-state reuse + the same-step reuse deferral).
// ---------------------------------------------------------------------------

TEST_CASE(
    "MambaManager (mode none): find_longest_cache_hit -> add_local_computed_blocks"
    " -> allocate_new_blocks end-to-end") {
  init_none_hash(sha256_cbor);
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeMambaSpec();  // mode "none"
  MambaManager mgr(spec, pool, /*enable_caching=*/true, 0, 2);

  // Request A: 6 prompt tokens -> 3 full blocks; allocate + cache its state.
  Request reqA = MakeRequest("A", Iota(6), /*block_size=*/2);
  REQUIRE(reqA.block_hashes.size() == 3);
  auto allocatedA = mgr.allocate_new_blocks("A", 6, 6);
  REQUIRE(allocatedA.size() == 3);
  mgr.cache_blocks(reqA, /*num_tokens=*/6);
  // Advance to the next scheduler step so A's just-cached state is reusable
  // (clears cached_blocks_this_step; the same-step guard is exercised below).
  mgr.new_step_starts();

  // Request B shares A's first 6 tokens and extends to 8 (4 blocks).
  Request reqB = MakeRequest("B", Iota(8), 2);
  REQUIRE(reqB.block_hashes.size() == 4);

  // find_longest_cache_hit keeps only the single rightmost state:
  // [null, null, state].
  auto computed = mgr.find_longest_cache_hit(reqB.block_hashes, /*max_length=*/8,
                                             {0}, pool, *spec,
                                             /*drop_eagle_block=*/false, 2)[0];
  REQUIRE(computed.size() == 3);
  CHECK(computed[0] == pool.null_block);
  CHECK(computed[1] == pool.null_block);
  CHECK(computed[2] == allocatedA[2]);  // A's tail state block.

  // add_local_computed_blocks null-front-pads: hit_length = 3 * 2 = 6 tokens,
  // get_num_skipped_tokens(6) = 5 -> 2 skipped blocks. It drops the 2 leading
  // nulls from the hit and re-adds 2 null pads, leaving [null, null, state].
  mgr.add_local_computed_blocks("B", computed,
                                /*num_local_computed_tokens=*/6,
                                /*num_external_computed_tokens=*/0);
  auto& bBlocks = mgr.req_to_blocks["B"];
  REQUIRE(bBlocks.size() == 3);
  CHECK(bBlocks[0] == pool.null_block);
  CHECK(bBlocks[1] == pool.null_block);
  CHECK(bBlocks[2] == allocatedA[2]);
  CHECK(mgr.num_cached_block["B"] == 3);
  // The reused state block was touched: ref_cnt = A(1) + B(1) = 2.
  CHECK(allocatedA[2]->ref_cnt == 2);

  // allocate_new_blocks completes B to 8 tokens (cdiv(8,2)=4 blocks): exactly
  // 1 genuinely new block on top of the 3-block hit.
  int free_before = pool.get_num_free_blocks();
  auto newB = mgr.allocate_new_blocks("B", 8, 8);
  REQUIRE(newB.size() == 1);
  CHECK(mgr.req_to_blocks["B"].size() == 4);
  CHECK(pool.get_num_free_blocks() == free_before - 1);
  // Mamba is not a full-attention spec, so no worker block-ids are recorded.
  CHECK(mgr.take_new_block_ids().empty());
}

TEST_CASE(
    "MambaManager (mode none): get_num_blocks_to_allocate defers a same-step "
    "state reuse, then clears after new_step_starts") {
  init_none_hash(sha256_cbor);
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeMambaSpec();
  MambaManager mgr(spec, pool, /*enable_caching=*/true, 0, 2);

  // Request A caches its state THIS step.
  Request reqA = MakeRequest("A", Iota(6), 2);
  auto allocatedA = mgr.allocate_new_blocks("A", 6, 6);
  REQUIRE(allocatedA.size() == 3);
  mgr.cache_blocks(reqA, 6);
  REQUIRE(!mgr.cached_blocks_this_step.empty());

  // Request B's hit lands on the very block A cached this step.
  Request reqB = MakeRequest("B", Iota(8), 2);
  auto computed =
      mgr.find_longest_cache_hit(reqB.block_hashes, 8, {0}, pool, *spec,
                                 false, 2)[0];
  REQUIRE(computed.size() == 3);
  REQUIRE(computed.back() == allocatedA[2]);

  // Same-step deferral: return num_gpu_blocks + 1 so the request is not
  // scheduled onto an in-flight state block another request cached this step.
  CHECK(mgr.get_num_blocks_to_allocate("B", /*num_tokens=*/8, computed,
                                       /*total_computed_tokens=*/6,
                                       /*num_tokens_main_model=*/8) ==
        pool.num_gpu_blocks + 1);

  // After the step boundary the deferral clears; normal accounting yields the
  // single new block needed to reach cdiv(8,2)=4 blocks from the 3-block hit.
  mgr.new_step_starts();
  CHECK(mgr.cached_blocks_this_step.empty());
  CHECK(mgr.get_num_blocks_to_allocate("B", 8, computed, 6, 8) == 1);
}

TEST_CASE(
    "MambaManager (mode none): get_num_skipped_tokens (n-1), "
    "remove_skipped_blocks frees all but the tail, common prefix is 0") {
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeMambaSpec();
  MambaManager mgr(spec, pool, /*enable_caching=*/true, 0, 2);

  // Mamba keeps only the last token's state: get_num_skipped_tokens(n) == n-1.
  CHECK(mgr.get_num_skipped_tokens(6) == 5);
  CHECK(mgr.get_num_skipped_tokens(1) == 0);

  mgr.allocate_new_blocks("r", 6, 6);  // 3 blocks.
  int free_before = pool.get_num_free_blocks();

  // num_skipped_tokens(6)=5, 5 // 2 = 2 leading blocks freed; tail survives.
  mgr.remove_skipped_blocks("r", /*num_computed_tokens=*/6);
  auto& blocks = mgr.req_to_blocks["r"];
  REQUIRE(blocks.size() == 3);
  CHECK(blocks[0] == pool.null_block);
  CHECK(blocks[1] == pool.null_block);
  CHECK(blocks[2] != pool.null_block);
  CHECK(pool.get_num_free_blocks() == free_before + 2);

  // Cascade attention is not supported by mamba.
  CHECK(mgr.get_num_common_prefix_blocks("r") == 0);
}

TEST_CASE(
    "MambaManager (mode none): cache_blocks populates cached_blocks_this_step, "
    "new_step_starts clears it") {
  init_none_hash(sha256_cbor);
  BlockPool pool(/*num_gpu_blocks=*/100, /*enable_caching=*/true,
                 /*hash_block_size=*/2);
  auto spec = MakeMambaSpec();
  MambaManager mgr(spec, pool, /*enable_caching=*/true, 0, 2);

  Request req = MakeRequest("r", Iota(6), 2);
  REQUIRE(req.block_hashes.size() == 3);
  mgr.allocate_new_blocks("r", 6, 6);
  CHECK(mgr.cached_blocks_this_step.empty());

  mgr.cache_blocks(req, /*num_tokens=*/6);
  // One entry per newly-cached (non-null, hashed) block.
  CHECK(mgr.cached_blocks_this_step.size() == 3);
  CHECK(mgr.cached_blocks_this_step.count(
            make_block_hash_with_group_id(req.block_hashes[0], 0)) == 1);
  CHECK(mgr.cached_blocks_this_step.count(
            make_block_hash_with_group_id(req.block_hashes[2], 0)) == 1);

  mgr.new_step_starts();
  CHECK(mgr.cached_blocks_this_step.empty());
}

TEST_CASE("SWA contiguous-KV packing cases are tracked by KV-PREFIX-CACHE" *
          doctest::skip(true) *
          doctest::description(
              "tests/v1/core/test_contiguous_kv_packing.py requires the "
              "KV-PREFIX-CACHE multi-block-size packing path")) {
  MESSAGE("SKIP KV-PREFIX-CACHE: contiguous KV packing is not ported");
}

TEST_CASE("SWA KV offload cases are tracked by KV-OFFLOAD" *
          doctest::skip(true) *
          doctest::description(
              "tests/v1/kv_offload SWA cases require the KV-OFFLOAD row")) {
  MESSAGE("SKIP KV-OFFLOAD: CPU-tiered SWA cache is not ported");
}

TEST_CASE("SWA connector cases are tracked by KV-CONNECTORS" *
          doctest::skip(true) *
          doctest::description(
              "tests/v1/kv_connector SWA cases require KV-CONNECTORS")) {
  MESSAGE("SKIP KV-CONNECTORS: external/disaggregated SWA cache is not ported");
}

TEST_CASE("chunked-local contiguous-KV packing is tracked by KV-PREFIX-CACHE" *
          doctest::skip(true) *
          doctest::description(
              "chunked-local multi-block-size packing requires KV-PREFIX-CACHE")) {
  MESSAGE("SKIP KV-PREFIX-CACHE: chunked-local contiguous packing is unported");
}

TEST_CASE("chunked-local KV offload is tracked by KV-OFFLOAD" *
          doctest::skip(true) *
          doctest::description(
              "chunked-local tests/v1/kv_offload cases require KV-OFFLOAD")) {
  MESSAGE("SKIP KV-OFFLOAD: chunked-local offload is not ported");
}

TEST_CASE("chunked-local connector cases are tracked by KV-CONNECTORS" *
          doctest::skip(true) *
          doctest::description(
              "chunked-local connector cases require KV-CONNECTORS")) {
  MESSAGE("SKIP KV-CONNECTORS: chunked-local external cache is not ported");
}
