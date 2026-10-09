// Tests for KVCacheBlock + FreeKVCacheBlockQueue (M1.2 Task 1) and the
// parent-chained, group-aware prefix-cache block hashing (M1.2 Task 2), ported
// from vllm/tests/v1/core/test_kv_cache_utils.py @ e24d1b24.
//
// Ported cases (the FreeKVCacheBlockQueue / KVCacheBlock behavioral oracle):
//   - test_kv_cache_block
//   - test_free_kv_cache_block_queue_initialization
//   - test_free_kv_cache_block_queue_operations
//   - test_free_kv_cache_block_queue_append_n
//   - test_free_kv_cache_block_queue_prepend_n
//   - test_free_kv_cache_block_queue_popleft_n
//   - test_free_kv_cache_block_queue_get_all_free_blocks
//
// Ported hashing cases (Task 2 behavioral oracle):
//   - test_none_hash (seeded reproducibility; different seeds differ)
//   - test_hash_block_tokens (byte-exact vs sha256_cbor)
//   - test_request_block_hasher (mm extra keys, chaining; byte-exact)
//   - test_hash_request_tokens_no_mm_inputs (chaining; byte-exact)
// Plus structural checks the upstream parametrization asserts across hashers:
// parent chaining changes the hash, group id differentiates, the partial last
// block is not hashed, and NONE_HASH is used as the first-block parent.
//
// C5 W1/W3 additionally port every local-attention arm of
// test_unify_hybrid_kv_cache_specs from
// vllm/tests/v1/core/test_kv_cache_utils.py:2420-2487 @ e24d1b24.
//
// HASH-FUNCTION FIDELITY: upstream parametrizes these over `sha256` (pickle) and
// `sha256_cbor` (cbor2 canonical). We port `sha256_cbor` BYTE-FOR-BYTE (see the
// header) and pin absolute goldens against it; `sha256` (pickle) is out of scope
// for byte-parity but its structural invariants are identical and are covered
// here. Goldens were generated with cbor2 6.1.2 + hashlib (canonical=True); the
// raw-CBOR checks below independently prove the serialization matches
// `cbor2.dumps(x, canonical=True)`.
//
// Object-identity comparisons in the upstream tests (`block is blocks[i]`,
// `get_all_free_blocks() == blocks`) are ported as pointer-equality checks,
// since the queue links the same KVCacheBlock objects the pool owns.
//
// NOTE: test_kv_cache_block upstream builds the hash via
// make_block_hash_with_group_id; Task 2 now provides it, and the case below uses
// it (Task 1 constructed the packed key directly).
#include <doctest/doctest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vt/dtype.h"

using vllm::v1::BlockHash;
using vllm::v1::BlockHashWithGroupId;
using vllm::v1::CborValue;
using vllm::v1::ChunkedLocalAttentionSpec;
using vllm::v1::ExtraKey;
using vllm::v1::ExtraKeys;
using vllm::v1::FreeKVCacheBlockQueue;
using vllm::v1::hash_block_tokens;
using vllm::v1::hash_request_tokens;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheBlock;
using vllm::v1::KVCacheSpec;
using vllm::v1::FullAttentionSpec;
using vllm::v1::make_block_hash_with_group_id;
using vllm::v1::MambaSpec;
using vllm::v1::sha256_cbor;
using vllm::v1::SlidingWindowSpec;
using vllm::v1::unify_hybrid_kv_cache_specs;

namespace {

// Lowercase hex of a byte string, for comparing digests to cbor2+hashlib
// goldens.
std::string ToHex(const std::string& bytes) {
  static const char* kDigits = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (unsigned char c : bytes) {
    out.push_back(kDigits[c >> 4]);
    out.push_back(kDigits[c & 0xf]);
  }
  return out;
}

// Build n KVCacheBlocks with block_id = 0..n-1. Reserve up front so no
// reallocation happens; moving the returned vector preserves element addresses
// (the buffer is stolen, elements are not relocated).
std::vector<KVCacheBlock> MakeBlocks(int n) {
  std::vector<KVCacheBlock> blocks;
  blocks.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    blocks.emplace_back(i);
  }
  return blocks;
}

std::vector<KVCacheBlock*> Ptrs(std::vector<KVCacheBlock>& blocks) {
  std::vector<KVCacheBlock*> ptrs;
  ptrs.reserve(blocks.size());
  for (auto& b : blocks) {
    ptrs.push_back(&b);
  }
  return ptrs;
}

}  // namespace

TEST_CASE("test_kv_cache_block") {
  // Test KVCacheBlock initialization.
  KVCacheBlock block(0);
  CHECK(block.block_id == 0);
  CHECK(block.ref_cnt == 0);
  CHECK_FALSE(block.block_hash().has_value());

  // Test reference count manipulation (direct, as upstream; plus the added
  // incr_ref/decr_ref helpers).
  block.ref_cnt += 1;
  CHECK(block.ref_cnt == 1);
  block.ref_cnt -= 1;
  CHECK(block.ref_cnt == 0);
  block.incr_ref();
  CHECK(block.ref_cnt == 1);
  block.decr_ref();
  CHECK(block.ref_cnt == 0);

  // Test block hash setting and resetting. Upstream:
  // make_block_hash_with_group_id(BlockHash(b"abc"), 0).
  BlockHashWithGroupId block_hash = make_block_hash_with_group_id("abc", 0);
  CHECK(block_hash == std::string("abc") + std::string("\x00\x00\x00\x00", 4));
  block.set_block_hash(block_hash);
  CHECK(block.block_hash().has_value());
  CHECK(block.block_hash().value() == block_hash);

  block.reset_hash();
  CHECK_FALSE(block.block_hash().has_value());
}

TEST_CASE("test_free_kv_cache_block_queue_initialization") {
  // Test with a single block.
  std::vector<KVCacheBlock> blocks = MakeBlocks(1);
  auto ptrs = Ptrs(blocks);
  FreeKVCacheBlockQueue queue(ptrs);
  CHECK(queue.num_free_blocks == 1);
  CHECK(queue.fake_free_list_head.next_free_block == ptrs[0]);
  CHECK(queue.fake_free_list_tail.prev_free_block == ptrs[0]);
}

TEST_CASE("test_free_kv_cache_block_queue_operations") {
  std::vector<KVCacheBlock> blocks = MakeBlocks(5);
  auto b = Ptrs(blocks);
  FreeKVCacheBlockQueue queue(b);

  // Check initial state.
  CHECK(queue.num_free_blocks == 5);
  CHECK(queue.fake_free_list_head.next_free_block == b[0]);
  CHECK(queue.fake_free_list_tail.prev_free_block == b[4]);

  // Pop the first block.
  KVCacheBlock* block1 = queue.popleft();
  CHECK(block1 == b[0]);
  CHECK(queue.num_free_blocks == 4);
  CHECK(queue.fake_free_list_head.next_free_block == b[1]);
  CHECK(queue.fake_free_list_tail.prev_free_block == b[4]);

  // Remove a block from the middle.
  KVCacheBlock* block_to_remove = b[2];
  queue.remove(block_to_remove);
  CHECK(queue.num_free_blocks == 3);
  CHECK(b[1]->next_free_block == b[3]);
  CHECK(b[3]->prev_free_block == b[1]);

  // Append a block back.
  queue.append(block_to_remove);
  CHECK(queue.num_free_blocks == 4);
  CHECK(queue.fake_free_list_tail.prev_free_block == block_to_remove);
  CHECK(block_to_remove->prev_free_block == b[4]);
  CHECK(block_to_remove->next_free_block == &queue.fake_free_list_tail);

  // Pop blocks until empty.
  for (int i = 0; i < 4; ++i) {
    queue.popleft();
  }
  CHECK(queue.num_free_blocks == 0);
  CHECK(queue.fake_free_list_head.next_free_block == &queue.fake_free_list_tail);
  CHECK(queue.fake_free_list_tail.prev_free_block == &queue.fake_free_list_head);

  // Attempt to pop from an empty queue.
  CHECK_THROWS_WITH_AS(queue.popleft(), "No free blocks available",
                       std::runtime_error);
}

TEST_CASE("test_free_kv_cache_block_queue_append_n") {
  // Create an empty FreeKVCacheBlockQueue.
  std::vector<KVCacheBlock*> empty;
  FreeKVCacheBlockQueue queue(empty);
  std::vector<KVCacheBlock> blocks = MakeBlocks(6);
  auto b = Ptrs(blocks);

  // Append 0 block: fake_head->fake_tail
  queue.append_n({});
  CHECK(queue.num_free_blocks == 0);
  CHECK(queue.fake_free_list_head.next_free_block == &queue.fake_free_list_tail);
  CHECK(queue.fake_free_list_tail.prev_free_block == &queue.fake_free_list_head);

  // Append 1 block: fake_head->b0->fake_tail
  queue.append_n({b[0]});
  CHECK(queue.num_free_blocks == 1);
  CHECK(queue.fake_free_list_head.next_free_block == b[0]);
  CHECK(b[0]->prev_free_block == &queue.fake_free_list_head);
  CHECK(b[0]->next_free_block == &queue.fake_free_list_tail);
  CHECK(queue.fake_free_list_tail.prev_free_block == b[0]);

  // Append 2 blocks: fake_head->b0->b4->b5->fake_tail
  queue.append_n({b[4], b[5]});
  CHECK(queue.num_free_blocks == 3);
  CHECK(queue.fake_free_list_head.next_free_block == b[0]);
  CHECK(b[0]->prev_free_block == &queue.fake_free_list_head);
  CHECK(b[0]->next_free_block == b[4]);
  CHECK(b[4]->prev_free_block == b[0]);
  CHECK(b[4]->next_free_block == b[5]);
  CHECK(b[5]->prev_free_block == b[4]);
  CHECK(b[5]->next_free_block == &queue.fake_free_list_tail);
  CHECK(queue.fake_free_list_tail.prev_free_block == b[5]);

  // Append 3 blocks: fake_head->b0->b4->b5->b1->b2->b3->fake_tail
  queue.append_n({b[1], b[2], b[3]});
  CHECK(queue.num_free_blocks == 6);
  CHECK(queue.fake_free_list_head.next_free_block == b[0]);
  CHECK(b[0]->next_free_block == b[4]);
  CHECK(b[4]->next_free_block == b[5]);
  CHECK(b[5]->next_free_block == b[1]);
  CHECK(b[1]->prev_free_block == b[5]);
  CHECK(b[1]->next_free_block == b[2]);
  CHECK(b[2]->prev_free_block == b[1]);
  CHECK(b[2]->next_free_block == b[3]);
  CHECK(b[3]->prev_free_block == b[2]);
  CHECK(b[3]->next_free_block == &queue.fake_free_list_tail);
  CHECK(queue.fake_free_list_tail.prev_free_block == b[3]);
}

TEST_CASE("test_free_kv_cache_block_queue_prepend_n") {
  // Seed the queue with one block so prepend has an existing head to splice in
  // front of (fake_head->b0->fake_tail).
  std::vector<KVCacheBlock> blocks = MakeBlocks(6);
  auto b = Ptrs(blocks);
  std::vector<KVCacheBlock*> seed{b[0]};
  FreeKVCacheBlockQueue queue(seed);

  // Prepend 0 blocks is a no-op.
  queue.prepend_n({});
  CHECK(queue.num_free_blocks == 1);
  CHECK(queue.fake_free_list_head.next_free_block == b[0]);

  // Prepend 2 blocks: fake_head->b4->b5->b0->fake_tail
  queue.prepend_n({b[4], b[5]});
  CHECK(queue.num_free_blocks == 3);
  CHECK(queue.fake_free_list_head.next_free_block == b[4]);
  CHECK(b[4]->prev_free_block == &queue.fake_free_list_head);
  CHECK(b[4]->next_free_block == b[5]);
  CHECK(b[5]->prev_free_block == b[4]);
  CHECK(b[5]->next_free_block == b[0]);
  CHECK(b[0]->prev_free_block == b[5]);
  CHECK(b[0]->next_free_block == &queue.fake_free_list_tail);
  CHECK(queue.fake_free_list_tail.prev_free_block == b[0]);

  // A second prepend goes ahead of everything previously prepended.
  // fake_head->b1->b2->b4->b5->b0->fake_tail
  queue.prepend_n({b[1], b[2]});
  CHECK(queue.num_free_blocks == 5);
  CHECK(queue.fake_free_list_head.next_free_block == b[1]);
  CHECK(b[1]->next_free_block == b[2]);
  CHECK(b[2]->next_free_block == b[4]);

  // The popleft order reflects the front-to-back queue order.
  std::vector<int> popped;
  for (int i = 0; i < 5; ++i) {
    popped.push_back(queue.popleft()->block_id);
  }
  CHECK(popped == std::vector<int>{1, 2, 4, 5, 0});
  CHECK(queue.num_free_blocks == 0);
}

TEST_CASE("test_free_kv_cache_block_queue_popleft_n") {
  std::vector<KVCacheBlock> blocks = MakeBlocks(6);
  auto b = Ptrs(blocks);
  std::vector<KVCacheBlock*> order{b[1], b[3], b[5], b[4], b[0], b[2]};
  FreeKVCacheBlockQueue queue(order);
  CHECK(queue.num_free_blocks == 6);
  CHECK(queue.fake_free_list_head.next_free_block == b[1]);
  CHECK(b[1]->prev_free_block == &queue.fake_free_list_head);
  CHECK(b[1]->next_free_block == b[3]);
  CHECK(b[3]->prev_free_block == b[1]);
  CHECK(b[3]->next_free_block == b[5]);
  CHECK(b[5]->prev_free_block == b[3]);
  CHECK(b[5]->next_free_block == b[4]);
  CHECK(b[4]->prev_free_block == b[5]);
  CHECK(b[4]->next_free_block == b[0]);
  CHECK(b[0]->prev_free_block == b[4]);
  CHECK(b[0]->next_free_block == b[2]);
  CHECK(b[2]->prev_free_block == b[0]);
  CHECK(b[2]->next_free_block == &queue.fake_free_list_tail);
  CHECK(queue.fake_free_list_tail.prev_free_block == b[2]);

  // Pop 0 block: fake_head->b1->b3->b5->b4->b0->b2->fake_tail
  CHECK(queue.popleft_n(0).empty());
  CHECK(queue.num_free_blocks == 6);

  // Pop 1 block: fake_head->b3->b5->b4->b0->b2->fake_tail
  auto r1 = queue.popleft_n(1);
  CHECK(queue.num_free_blocks == 5);
  CHECK(r1.size() == 1);
  CHECK(r1[0] == b[1]);
  for (auto* blk : r1) {
    CHECK(blk->prev_free_block == nullptr);
    CHECK(blk->next_free_block == nullptr);
  }

  // Pop 2 blocks: fake_head->b4->b0->b2->fake_tail
  auto r2 = queue.popleft_n(2);
  CHECK(r2.size() == 2);
  CHECK(queue.num_free_blocks == 3);
  CHECK(r2[0] == b[3]);
  CHECK(r2[1] == b[5]);
  for (auto* blk : r2) {
    CHECK(blk->prev_free_block == nullptr);
    CHECK(blk->next_free_block == nullptr);
  }

  // Pop 3 blocks: fake_head->fake_tail
  auto r3 = queue.popleft_n(3);
  CHECK(r3.size() == 3);
  CHECK(queue.num_free_blocks == 0);
  CHECK(r3[0] == b[4]);
  CHECK(r3[1] == b[0]);
  CHECK(r3[2] == b[2]);
  for (auto* blk : r3) {
    CHECK(blk->prev_free_block == nullptr);
    CHECK(blk->next_free_block == nullptr);
  }
}

TEST_CASE("test_free_kv_cache_block_queue_get_all_free_blocks") {
  std::vector<KVCacheBlock> blocks = MakeBlocks(5);
  auto b = Ptrs(blocks);
  FreeKVCacheBlockQueue queue(b);

  // Check all blocks are correctly retrieved.
  CHECK(queue.get_all_free_blocks() == b);

  // Pop a block and check again.
  queue.popleft();
  CHECK(queue.get_all_free_blocks() ==
        std::vector<KVCacheBlock*>{b[1], b[2], b[3], b[4]});

  // Remove a block and check again.
  KVCacheBlock* block_to_remove = b[2];
  queue.remove(block_to_remove);
  CHECK(queue.get_all_free_blocks() ==
        std::vector<KVCacheBlock*>{b[1], b[3], b[4]});

  // Append a block back and check again.
  queue.append(block_to_remove);
  CHECK(queue.get_all_free_blocks() ==
        std::vector<KVCacheBlock*>{b[1], b[3], b[4], b[2]});
}

// ---------------------------------------------------------------------------
// Task 2: block hashing.
// ---------------------------------------------------------------------------

// Independently prove CborValue::Encode() == cbor2.dumps(x, canonical=True).
TEST_CASE("cbor_value_canonical_encoding") {
  CHECK(ToHex(CborValue::Bytes("123").Encode()) == "43313233");
  CHECK(ToHex(CborValue::Array({CborValue::Int(1), CborValue::Int(2),
                                CborValue::Int(3)})
                  .Encode()) == "83010203");
  CHECK(ToHex(CborValue::Text("seed42").Encode()) == "66736565643432");
  CHECK(ToHex(CborValue::Null().Encode()) == "f6");
  // Negative ints (multi-modal offsets can be negative).
  CHECK(ToHex(CborValue::Int(-1).Encode()) == "20");
  CHECK(ToHex(CborValue::Int(-3).Encode()) == "22");
  CHECK(ToHex(CborValue::Int(-257).Encode()) == "390100");
  // Minimal-width integer arguments.
  CHECK(ToHex(CborValue::UInt(23).Encode()) == "17");
  CHECK(ToHex(CborValue::UInt(24).Encode()) == "1818");
  CHECK(ToHex(CborValue::UInt(255).Encode()) == "18ff");
  CHECK(ToHex(CborValue::UInt(256).Encode()) == "190100");
  CHECK(ToHex(CborValue::UInt(65536).Encode()) == "1a00010000");
  // The full (parent, tokens, extra_keys) tuple upstream feeds the hasher.
  CborValue tuple = CborValue::Array(
      {CborValue::Bytes("123"),
       CborValue::Array({CborValue::Int(1), CborValue::Int(2), CborValue::Int(3)}),
       CborValue::Array({CborValue::Text("key1"), CborValue::Text("key2")})});
  CHECK(ToHex(tuple.Encode()) ==
        "83433132338301020382646b657931646b657932");
}

// test_none_hash: seeded init is reproducible (== sha256_cbor(text(seed))),
// yields non-empty 32-byte hashes, and different seeds differ.
TEST_CASE("test_none_hash") {
  init_none_hash(sha256_cbor, "seed42");
  CHECK(vllm::v1::NONE_HASH.size() == 32);
  CHECK(ToHex(vllm::v1::NONE_HASH) ==
        "4dee92a73b42a5488db700713f330b42f7c8b90ae46143db43aef808eb2099d7");
  CHECK(vllm::v1::NONE_HASH == sha256_cbor(CborValue::Text("seed42")));

  const BlockHash seed42 = vllm::v1::NONE_HASH;
  init_none_hash(sha256_cbor, "different seed");
  CHECK(vllm::v1::NONE_HASH != seed42);

  // Unseeded: 32 random bytes, non-empty (upstream os.urandom(32)).
  init_none_hash(sha256_cbor);
  CHECK(vllm::v1::NONE_HASH.size() == 32);
}

// test_hash_block_tokens: block_hash == hash_fn((parent, tokens, extra_keys)),
// pinned byte-for-byte against sha256_cbor.
TEST_CASE("test_hash_block_tokens") {
  const BlockHash parent = "123";
  const std::vector<int32_t> tokens = {1, 2, 3};
  const ExtraKeys extra = std::vector<ExtraKey>{ExtraKey(std::string("key1")),
                                                ExtraKey(std::string("key2"))};

  const BlockHash block_hash = hash_block_tokens(sha256_cbor, parent, tokens, extra);
  const BlockHash expected = sha256_cbor(CborValue::Array(
      {CborValue::Bytes("123"),
       CborValue::Array({CborValue::Int(1), CborValue::Int(2), CborValue::Int(3)}),
       CborValue::Array({CborValue::Text("key1"), CborValue::Text("key2")})}));
  CHECK(block_hash == expected);
  CHECK(ToHex(block_hash) ==
        "75f96f75f082518549ca45da83610d619f8c63528c78f8aef470d43f1647317d");
}

// test_hash_request_tokens_no_mm_inputs: 6 tokens at block_size 3 -> 2 hashes,
// each chaining the previous (block 1 depends on block 0), pinned byte-for-byte.
TEST_CASE("test_hash_request_tokens_no_mm_inputs") {
  init_none_hash(sha256_cbor, "seed42");
  const std::vector<int32_t> tokens = {0, 1, 2, 3, 4, 5};
  const std::vector<BlockHash> hashes = hash_request_tokens(sha256_cbor, 3, tokens);

  REQUIRE(hashes.size() == 2);
  CHECK(hashes[0] == sha256_cbor(CborValue::Array(
                         {CborValue::Bytes(vllm::v1::NONE_HASH),
                          CborValue::Array({CborValue::Int(0), CborValue::Int(1),
                                            CborValue::Int(2)}),
                          CborValue::Null()})));
  CHECK(hashes[1] == sha256_cbor(CborValue::Array(
                         {CborValue::Bytes(hashes[0]),
                          CborValue::Array({CborValue::Int(3), CborValue::Int(4),
                                            CborValue::Int(5)}),
                          CborValue::Null()})));
  CHECK(ToHex(hashes[0]) ==
        "210559fa9941d0cd01ae4176a437cd8865c50b64e06af48138364280356c9aa9");
  CHECK(ToHex(hashes[1]) ==
        "855499304e237ca9bbbfff065f17f73d26744f5f30766249bb6e8f80f1bdf89a");
}

// test_request_block_hasher: per-block mm extra keys (("hashN", 0),), chained,
// pinned byte-for-byte.
TEST_CASE("test_request_block_hasher") {
  init_none_hash(sha256_cbor, "seed42");
  const std::vector<int32_t> tokens = {0, 1, 2, 3, 4, 5};
  const std::vector<ExtraKeys> per_block = {
      std::vector<ExtraKey>{ExtraKey(std::pair<std::string, int64_t>("hash1", 0))},
      std::vector<ExtraKey>{ExtraKey(std::pair<std::string, int64_t>("hash2", 0))},
  };
  const std::vector<BlockHash> hashes =
      hash_request_tokens(sha256_cbor, 3, tokens, per_block);

  REQUIRE(hashes.size() == 2);
  CHECK(ToHex(hashes[0]) ==
        "ae58025bbbd9b4375741f34b9ac177fe0ce7fa4bac8c57483ee86116bb92ee92");
  CHECK(ToHex(hashes[1]) ==
        "3b3218c497c0e23fa619550c8d57781f7f740d70214c4e1e6a54593a5da40911");
  // Different mm keys -> different hashes than the no-mm chain.
  const std::vector<BlockHash> no_mm = hash_request_tokens(sha256_cbor, 3, tokens);
  CHECK(hashes[0] != no_mm[0]);
  CHECK(hashes[1] != no_mm[1]);
}

// ---------------------------------------------------------------------------
// generate_block_hash_extra_keys (KV-PREFIX-CACHE W2), ported 1:1 from
// vllm/tests/v1/core/test_kv_cache_utils.py @ 555967922:
//   - test_generate_block_hash_extra_keys
//   - test_generate_block_hash_extra_keys_no_mm_inputs
//   - test_generate_block_hash_extra_keys_cache_salt
//   - test_generate_block_hash_extra_keys_lora
// (The prompt_embeds cases are NOT ported: this tree has no prompt-embeds path;
//  see the DEFERRED note in include/vllm/v1/request.h.)
// ---------------------------------------------------------------------------

namespace {

using vllm::v1::generate_block_hash_extra_keys;
using vllm::v1::Request;

// Build a text Request (no block hasher) whose mm_features are the given
// (offset, length) placeholders with the given mm hashes (identifiers).
Request MakeMmRequest(const std::string& id, int num_tokens,
                      const std::vector<std::pair<int, int>>& mm_positions,
                      const std::vector<std::string>& mm_hashes) {
  std::vector<int32_t> tokens;
  for (int i = 0; i < num_tokens; ++i) tokens.push_back(i);
  Request req(id, tokens, vllm::SamplingParams{}, /*arrival_time=*/0.0);
  for (size_t i = 0; i < mm_positions.size(); ++i) {
    vllm::multimodal::MultiModalFeatureSpec f;
    f.mm_hash = mm_hashes[i];
    f.offset = mm_positions[i].first;
    f.length = mm_positions[i].second;
    req.mm_features.push_back(f);
  }
  return req;
}

// A single mm extra key (identifier, offset).
ExtraKey MmKey(const std::string& id, int64_t offset) {
  return ExtraKey(std::pair<std::string, int64_t>(id, offset));
}

}  // namespace

TEST_CASE("test_generate_block_hash_extra_keys") {
  Request req = MakeMmRequest("0", /*num_tokens=*/20,
                              {{0, 5}, {10, 5}}, {"hash1", "hash2"});

  // No extra keys beyond the first mm input.
  {
    auto [keys, next] = generate_block_hash_extra_keys(req, 0, 5, 0);
    REQUIRE(keys.has_value());
    REQUIRE(keys->size() == 1);
    CHECK((*keys)[0] == MmKey("hash1", 0));
    CHECK(next == 1);
  }
  // Partial overlap -> negative relative offset.
  {
    auto [keys, next] = generate_block_hash_extra_keys(req, 3, 8, 0);
    REQUIRE(keys.has_value());
    REQUIRE(keys->size() == 1);
    CHECK((*keys)[0] == MmKey("hash1", -3));
    CHECK(next == 1);
  }
  // No overlap -> None, but the mm cursor still advanced past hash1.
  {
    auto [keys, next] = generate_block_hash_extra_keys(req, 6, 10, 0);
    CHECK(!keys.has_value());
    CHECK(next == 1);
  }
  // Multiple extra keys in one block.
  {
    auto [keys, next] = generate_block_hash_extra_keys(req, 0, 15, 0);
    REQUIRE(keys.has_value());
    REQUIRE(keys->size() == 2);
    CHECK((*keys)[0] == MmKey("hash1", 0));
    CHECK((*keys)[1] == MmKey("hash2", 10));
    CHECK(next == 2);
  }
}

TEST_CASE("test_generate_block_hash_extra_keys_no_mm_inputs") {
  Request req = MakeMmRequest("0", /*num_tokens=*/6, {}, {});
  auto [keys, next] = generate_block_hash_extra_keys(req, 0, 5, 0);
  CHECK(!keys.has_value());
  CHECK(next == 0);
}

TEST_CASE("test_generate_block_hash_extra_keys_cache_salt") {
  Request req = MakeMmRequest("0", /*num_tokens=*/6, {}, {});
  req.cache_salt = "salt";

  // Salt is added for the first block only.
  {
    auto [keys, next] = generate_block_hash_extra_keys(req, 0, 1, 0);
    REQUIRE(keys.has_value());
    REQUIRE(keys->size() == 1);
    CHECK((*keys)[0] == ExtraKey(std::string("salt")));
    (void)next;
  }
  {
    auto [keys, next] = generate_block_hash_extra_keys(req, 0, 10, 0);
    REQUIRE(keys.has_value());
    CHECK((*keys)[0] == ExtraKey(std::string("salt")));
    (void)next;
  }
  // No salt for non-first blocks.
  {
    auto [keys, next] = generate_block_hash_extra_keys(req, 1, 2, 0);
    CHECK(!keys.has_value());
    (void)next;
  }
  {
    auto [keys, next] = generate_block_hash_extra_keys(req, 6, 10, 0);
    CHECK(!keys.has_value());
    (void)next;
  }

  // Salt composes with mm keys, in order mm -> salt.
  Request req_mm = MakeMmRequest("0", /*num_tokens=*/20, {{0, 5}}, {"hash1"});
  req_mm.cache_salt = "salt";
  auto [keys, next] = generate_block_hash_extra_keys(req_mm, 0, 5, 0);
  REQUIRE(keys.has_value());
  REQUIRE(keys->size() == 2);
  CHECK((*keys)[0] == MmKey("hash1", 0));
  CHECK((*keys)[1] == ExtraKey(std::string("salt")));
  CHECK(next == 1);
}

TEST_CASE("test_generate_block_hash_extra_keys_lora") {
  Request req = MakeMmRequest("0", /*num_tokens=*/6, {}, {});
  req.lora_name = "test_lora_adapter";
  {
    auto [keys, next] = generate_block_hash_extra_keys(req, 0, 3, 0);
    REQUIRE(keys.has_value());
    REQUIRE(keys->size() == 1);
    CHECK((*keys)[0] == ExtraKey(std::string("test_lora_adapter")));
    (void)next;
  }
  // Cleared -> no extra keys.
  req.lora_name = std::nullopt;
  {
    auto [keys, next] = generate_block_hash_extra_keys(req, 0, 3, 0);
    CHECK(!keys.has_value());
    (void)next;
  }
}

// Order is lora -> mm -> cache_salt (upstream generate_block_hash_extra_keys
// concatenation; R8(b) in the prefix-caching-parity spike).
TEST_CASE("generate_block_hash_extra_keys ordering lora->mm->salt") {
  Request req = MakeMmRequest("0", /*num_tokens=*/20, {{0, 5}}, {"hashX"});
  req.lora_name = "adapterY";
  req.cache_salt = "saltZ";
  auto [keys, next] = generate_block_hash_extra_keys(req, 0, 5, 0);
  REQUIRE(keys.has_value());
  REQUIRE(keys->size() == 3);
  CHECK((*keys)[0] == ExtraKey(std::string("adapterY")));  // lora first
  CHECK((*keys)[1] == MmKey("hashX", 0));                   // mm second
  CHECK((*keys)[2] == ExtraKey(std::string("saltZ")));      // salt last
  CHECK(next == 1);
}

// RED-first no-false-share at the HASH level: two requests with IDENTICAL tokens
// but different extra keys (mm hash / cache salt / lora) must hash to DIFFERENT
// block hashes. The negative control (same tokens, no extra keys) hashes the
// SAME, proving the extra key is the load-bearing differentiator. If the W2 stub
// were still in place (extra_keys always None), every pair below would collide
// and the two requests would false-share a cache block.
TEST_CASE("extra_keys differentiate block hashes (no false-share)") {
  using vllm::v1::get_request_block_hasher;
  init_none_hash(sha256_cbor, "seed42");
  const int block_size = 4;
  std::vector<int32_t> tokens;
  for (int i = 0; i < 8; ++i) tokens.push_back(i);  // 2 full blocks

  auto hashes_of = [&](const std::optional<std::string>& salt,
                       const std::optional<std::string>& lora,
                       const std::vector<std::pair<int, int>>& mm_pos,
                       const std::vector<std::string>& mm_hashes) {
    // Set the extra-key inputs BEFORE installing the hasher (fields must be set
    // before the first hash, as the real FromEngineCoreRequest path does).
    Request req("r", tokens, vllm::SamplingParams{}, 0.0);
    req.cache_salt = salt;
    req.lora_name = lora;
    for (size_t i = 0; i < mm_pos.size(); ++i) {
      vllm::multimodal::MultiModalFeatureSpec f;
      f.mm_hash = mm_hashes[i];
      f.offset = mm_pos[i].first;
      f.length = mm_pos[i].second;
      req.mm_features.push_back(f);
    }
    req.block_hasher_ = get_request_block_hasher(block_size, sha256_cbor);
    req.update_block_hashes();
    return req.block_hashes;
  };

  const auto base = hashes_of(std::nullopt, std::nullopt, {}, {});
  REQUIRE(base.size() == 2);

  // cache_salt differentiates (and transitively chains into block 1).
  const auto salted = hashes_of("tenantA", std::nullopt, {}, {});
  CHECK(salted[0] != base[0]);
  CHECK(salted[1] != base[1]);
  // A different salt differs again.
  const auto salted_b = hashes_of("tenantB", std::nullopt, {}, {});
  CHECK(salted_b[0] != salted[0]);

  // lora name differentiates every block.
  const auto lora = hashes_of(std::nullopt, "adapter1", {}, {});
  CHECK(lora[0] != base[0]);
  CHECK(lora[1] != base[1]);
  CHECK(lora != hashes_of(std::nullopt, "adapter2", {}, {}));

  // mm hash differentiates the block it overlaps.
  const auto mm1 = hashes_of(std::nullopt, std::nullopt, {{0, 4}}, {"img1"});
  const auto mm2 = hashes_of(std::nullopt, std::nullopt, {{0, 4}}, {"img2"});
  CHECK(mm1[0] != base[0]);
  CHECK(mm1[0] != mm2[0]);
}

// Parent chaining: block N's hash depends on block N-1's (and on NONE_HASH for
// block 0).
TEST_CASE("hash_block_tokens_parent_chaining") {
  init_none_hash(sha256_cbor, "seed42");
  const std::vector<int32_t> block0 = {0, 1, 2};
  const std::vector<int32_t> block1 = {3, 4, 5};

  const BlockHash h0 = hash_block_tokens(sha256_cbor, std::nullopt, block0);
  // A falsy parent (nullopt or empty) uses NONE_HASH as the parent.
  CHECK(hash_block_tokens(sha256_cbor, BlockHash(), block0) == h0);
  CHECK(h0 == hash_block_tokens(sha256_cbor, vllm::v1::NONE_HASH, block0));

  const BlockHash h1_after_h0 = hash_block_tokens(sha256_cbor, h0, block1);
  const BlockHash h1_after_none = hash_block_tokens(sha256_cbor, std::nullopt, block1);
  // Same tokens, different parent -> different hash.
  CHECK(h1_after_h0 != h1_after_none);
}

// Group id differentiation via make_block_hash_with_group_id / get_block_hash /
// get_group_id.
TEST_CASE("block_hash_with_group_id") {
  const BlockHash bh = hash_block_tokens(sha256_cbor, std::nullopt, {1, 2, 3});
  const BlockHashWithGroupId g0 = make_block_hash_with_group_id(bh, 0);
  const BlockHashWithGroupId g1 = make_block_hash_with_group_id(bh, 1);

  // Same block hash, different group id -> different packed key.
  CHECK(g0 != g1);
  CHECK(g0.size() == bh.size() + 4);

  // Round-trip both components.
  CHECK(vllm::v1::get_block_hash(g0) == bh);
  CHECK(vllm::v1::get_block_hash(g1) == bh);
  CHECK(vllm::v1::get_group_id(g0) == 0u);
  CHECK(vllm::v1::get_group_id(g1) == 1u);

  // 4-byte big-endian group-id encoding.
  const BlockHashWithGroupId big = make_block_hash_with_group_id("abc", 0x01020304u);
  CHECK(big == std::string("abc") + std::string("\x01\x02\x03\x04", 4));
  CHECK(vllm::v1::get_group_id(big) == 0x01020304u);
}

// Partial last block is not hashed: only full blocks produce a hash.
TEST_CASE("hash_request_tokens_partial_block_not_hashed") {
  init_none_hash(sha256_cbor, "seed42");
  // 7 tokens at block_size 3 -> 2 full blocks, trailing partial ignored.
  const std::vector<BlockHash> h7 =
      hash_request_tokens(sha256_cbor, 3, {0, 1, 2, 3, 4, 5, 6});
  CHECK(h7.size() == 2);
  // Fewer tokens than a block -> no hashes.
  CHECK(hash_request_tokens(sha256_cbor, 3, {0, 1}).empty());
  // Exactly one block.
  CHECK(hash_request_tokens(sha256_cbor, 3, {0, 1, 2}).size() == 1);
  // The full-block prefix of a longer sequence matches the shorter one.
  const std::vector<BlockHash> h6 =
      hash_request_tokens(sha256_cbor, 3, {0, 1, 2, 3, 4, 5});
  CHECK(h7 == h6);
}

// get_request_block_hasher (the incremental Request._block_hasher): the hashes a
// Request accumulates must equal the batch hash_request_tokens oracle
// (byte-exact vs the pin), and appending tokens extends block_hashes only when a
// new full block completes.
TEST_CASE("get_request_block_hasher matches hash_request_tokens (byte-exact)") {
  using vllm::v1::get_request_block_hasher;
  using vllm::v1::Request;
  init_none_hash(sha256_cbor, "seed42");
  const int block_size = 4;

  // 14 tokens -> 3 full blocks; the ctor runs update_block_hashes once.
  std::vector<int32_t> tokens;
  for (int i = 0; i < 14; ++i) tokens.push_back(i);
  Request req("0", tokens, vllm::SamplingParams{}, /*arrival_time=*/0.0,
              get_request_block_hasher(block_size, sha256_cbor));

  const std::vector<BlockHash> oracle =
      hash_request_tokens(sha256_cbor, block_size, tokens);
  REQUIRE(oracle.size() == 3);
  CHECK(req.block_hashes == oracle);

  // Appending 2 tokens completes a 4th full block (16 tokens total).
  req.AppendOutputToken(std::vector<int32_t>{14, 15});
  std::vector<int32_t> tokens16 = tokens;
  tokens16.push_back(14);
  tokens16.push_back(15);
  const std::vector<BlockHash> oracle16 =
      hash_request_tokens(sha256_cbor, block_size, tokens16);
  REQUIRE(oracle16.size() == 4);
  CHECK(req.block_hashes == oracle16);
  // The first 3 hashes are unchanged (chaining is prefix-stable).
  CHECK(req.block_hashes[0] == oracle[0]);
  CHECK(req.block_hashes[1] == oracle[1]);
  CHECK(req.block_hashes[2] == oracle[2]);

  // Appending 1 token (17 total) does not complete a new full block.
  req.AppendOutputToken(16);
  CHECK(req.block_hashes.size() == 4);

  // A Request with no hasher (upstream block_hasher=None) never hashes.
  Request no_hash("1", tokens, vllm::SamplingParams{}, /*arrival_time=*/0.0);
  CHECK(no_hash.block_hashes.empty());
  no_hash.AppendOutputToken(std::vector<int32_t>{14, 15});
  CHECK(no_hash.block_hashes.empty());
}

TEST_CASE("R09 MTP prefix hash preserves shifted boundary identity") {
  using vllm::v1::get_request_block_hasher;
  using vllm::v1::Request;
  init_none_hash(sha256_cbor, "seed42");
  const auto ordinary = get_request_block_hasher(4, sha256_cbor);
  const auto shifted = get_request_block_hasher(4, sha256_cbor, true);
  Request base("base", {1, 2, 3, 4, 5, 6, 7, 8}, {}, 0, shifted);
  Request next_changed("next", {1, 2, 3, 4, 9, 6, 7, 8}, {}, 0, shifted);
  Request later_changed("later", {1, 2, 3, 4, 5, 9, 7, 8}, {}, 0, shifted);
  Request old("ordinary", {1, 2, 3, 4, 5, 6, 7, 8}, {}, 0, ordinary);
  REQUIRE(base.block_hashes.size() == 1);
  REQUIRE(next_changed.block_hashes.size() == 1);
  REQUIRE(later_changed.block_hashes.size() == 1);
  CHECK(base.block_hashes != next_changed.block_hashes);
  CHECK(base.block_hashes == later_changed.block_hashes);
  CHECK(old.block_hashes.size() == 2);
  CHECK(base.block_hashes[0] != old.block_hashes[0]);
  CHECK(old.block_hashes == hash_request_tokens(sha256_cbor, 4, {1, 2, 3, 4, 5, 6, 7, 8}));
  // Cold construction and incremental output append yield identical keys.
  Request growing("growing", {1, 2, 3, 4}, {}, 0, shifted);
  CHECK(growing.block_hashes.empty());
  growing.AppendOutputToken(5);
  CHECK(growing.block_hashes == base.block_hashes);
  growing.AppendOutputToken(std::vector<int32_t>{6, 7, 8});
  CHECK(growing.block_hashes == base.block_hashes);
  growing.AppendOutputToken(9);
  Request whole("whole", {1, 2, 3, 4, 5, 6, 7, 8, 9}, {}, 0, shifted);
  REQUIRE(growing.block_hashes.size() == 2);
  CHECK(growing.block_hashes == whole.block_hashes);
  CHECK(growing.block_hashes[0] == base.block_hashes[0]);
  growing.AppendOutputToken(10);
  CHECK(growing.block_hashes == whole.block_hashes);
  // Salt and LoRA still separate otherwise-compatible target/draft contexts.
  Request salted("salted", {1, 2, 3, 4, 5, 6, 7, 8}, {}, 0);
  salted.cache_salt = "other-checkpoint-context";
  salted.lora_name = "adapter";
  const auto salted_hashes = shifted(salted);
  CHECK(salted_hashes.size() == 1);
  CHECK(salted_hashes != base.block_hashes);
  // One-token pages exercise the same next-token dependency at every position.
  Request one("one", {1}, {}, 0, get_request_block_hasher(1, sha256_cbor, true));
  CHECK(one.block_hashes.empty());
  one.AppendOutputToken(2);
  CHECK(one.block_hashes.size() == 1);
  one.AppendOutputToken(3);
  CHECK(one.block_hashes.size() == 2);
}

TEST_CASE("unify_hybrid_kv_cache_specs converts sliding storage to full") {
  auto full = std::make_shared<FullAttentionSpec>(
      /*block_size=*/16, /*num_kv_heads=*/2, /*head_size=*/64,
      vt::DType::kF32);
  auto sliding = std::make_shared<SlidingWindowSpec>(
      /*block_size=*/16, /*num_kv_heads=*/2, /*head_size=*/64,
      vt::DType::kF32, /*sliding_window=*/1024,
      /*head_size_v=*/32, vllm::v1::KVQuantMode::kNone,
      /*page_size_padded=*/32768);
  std::unordered_map<std::string, std::shared_ptr<KVCacheSpec>> specs{
      {"layer_1", full}, {"layer_2", sliding}};

  unify_hybrid_kv_cache_specs(specs);
  CHECK(specs["layer_1"] == full);
  auto converted =
      std::dynamic_pointer_cast<FullAttentionSpec>(specs["layer_2"]);
  REQUIRE(converted != nullptr);
  CHECK(converted->block_size == 16);
  CHECK(converted->num_kv_heads == 2);
  CHECK(converted->head_size == 64);
  CHECK(converted->head_size_v == 32);
  CHECK(converted->page_size_padded == std::optional<int64_t>{32768});
  CHECK(converted->sliding_window == std::optional<int>{1024});
}

TEST_CASE("unify_hybrid_kv_cache_specs leaves uniform sliding specs unchanged") {
  auto sliding_a = std::make_shared<SlidingWindowSpec>(
      16, 2, 64, vt::DType::kF32, /*sliding_window=*/1024);
  auto sliding_b = std::make_shared<SlidingWindowSpec>(
      16, 4, 32, vt::DType::kF32, /*sliding_window=*/1024);
  std::unordered_map<std::string, std::shared_ptr<KVCacheSpec>> specs{
      {"layer_1", sliding_a}, {"layer_2", sliding_b}};
  unify_hybrid_kv_cache_specs(specs);
  CHECK(specs["layer_1"] == sliding_a);
  CHECK(specs["layer_2"] == sliding_b);
}

TEST_CASE("unify_hybrid_kv_cache_specs converts chunked-local storage to full") {
  auto full = std::make_shared<FullAttentionSpec>(
      /*block_size=*/16, /*num_kv_heads=*/2, /*head_size=*/64,
      vt::DType::kF32);
  auto chunked = std::make_shared<ChunkedLocalAttentionSpec>(
      /*block_size=*/16, /*num_kv_heads=*/2, /*head_size=*/64,
      vt::DType::kF32, /*attention_chunk_size=*/512,
      vllm::v1::KVQuantMode::kNone, /*page_size_padded=*/32768);
  std::unordered_map<std::string, std::shared_ptr<KVCacheSpec>> specs{
      {"layer_1", full}, {"layer_2", chunked}};

  unify_hybrid_kv_cache_specs(specs);
  CHECK(specs["layer_1"] == full);
  auto converted =
      std::dynamic_pointer_cast<FullAttentionSpec>(specs["layer_2"]);
  REQUIRE(converted != nullptr);
  CHECK(converted->block_size == 16);
  CHECK(converted->num_kv_heads == 2);
  CHECK(converted->head_size == 64);
  CHECK(converted->head_size_v == 64);
  CHECK(converted->page_size_padded == std::optional<int64_t>{32768});
  CHECK(converted->sliding_window == std::nullopt);
  CHECK(converted->attention_chunk_size == std::optional<int>{512});
}

TEST_CASE("unify_hybrid_kv_cache_specs converts both local policies with full") {
  auto full = std::make_shared<FullAttentionSpec>(16, 2, 64, vt::DType::kF32);
  auto sliding = std::make_shared<SlidingWindowSpec>(
      16, 2, 64, vt::DType::kF32, /*sliding_window=*/1024);
  auto chunked = std::make_shared<ChunkedLocalAttentionSpec>(
      16, 2, 64, vt::DType::kF32, /*attention_chunk_size=*/512);
  std::unordered_map<std::string, std::shared_ptr<KVCacheSpec>> specs{
      {"full", full}, {"swa", sliding}, {"chunked", chunked}};

  unify_hybrid_kv_cache_specs(specs);
  auto converted_sliding =
      std::dynamic_pointer_cast<FullAttentionSpec>(specs["swa"]);
  auto converted_chunked =
      std::dynamic_pointer_cast<FullAttentionSpec>(specs["chunked"]);
  REQUIRE(converted_sliding != nullptr);
  REQUIRE(converted_chunked != nullptr);
  CHECK(converted_sliding->sliding_window == std::optional<int>{1024});
  CHECK(converted_chunked->attention_chunk_size == std::optional<int>{512});
}

TEST_CASE("unify_hybrid_kv_cache_specs leaves uniform chunked-local specs unchanged") {
  auto chunked_a = std::make_shared<ChunkedLocalAttentionSpec>(
      16, 2, 64, vt::DType::kF32, /*attention_chunk_size=*/512);
  auto chunked_b = std::make_shared<ChunkedLocalAttentionSpec>(
      16, 4, 32, vt::DType::kF32, /*attention_chunk_size=*/512);
  std::unordered_map<std::string, std::shared_ptr<KVCacheSpec>> specs{
      {"layer_1", chunked_a}, {"layer_2", chunked_b}};
  unify_hybrid_kv_cache_specs(specs);
  CHECK(specs["layer_1"] == chunked_a);
  CHECK(specs["layer_2"] == chunked_b);
}

TEST_CASE("unify_hybrid_kv_cache_specs rejects local policies without full") {
  auto sliding = std::make_shared<SlidingWindowSpec>(
      16, 2, 64, vt::DType::kF32, /*sliding_window=*/1024);
  auto chunked = std::make_shared<ChunkedLocalAttentionSpec>(
      16, 2, 64, vt::DType::kF32, /*attention_chunk_size=*/512);
  std::unordered_map<std::string, std::shared_ptr<KVCacheSpec>> specs{
      {"swa", sliding}, {"chunked", chunked}};
  CHECK_THROWS_WITH_AS(
      unify_hybrid_kv_cache_specs(specs),
      "Hybrid KV cache manager is disabled but failed to convert the KV cache "
      "specs to one unified type.",
      std::invalid_argument);
}

TEST_CASE("unify_hybrid_kv_cache_specs rejects an unconvertible mixed policy") {
  auto sliding_a = std::make_shared<SlidingWindowSpec>(
      16, 2, 64, vt::DType::kF32, /*sliding_window=*/1024);
  auto sliding_b = std::make_shared<SlidingWindowSpec>(
      16, 2, 64, vt::DType::kF32, /*sliding_window=*/256);
  std::unordered_map<std::string, std::shared_ptr<KVCacheSpec>> specs{
      {"layer_1", sliding_a}, {"layer_2", sliding_b}};
  CHECK_THROWS_WITH_AS(
      unify_hybrid_kv_cache_specs(specs),
      "Hybrid KV cache manager is disabled but failed to convert the KV cache "
      "specs to one unified type.",
      std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Startup KV sizing: max_blocks_per_request / max_memory_usage_bytes_from_groups
// / estimate_max_model_len_from_groups / check_enough_kv_cache_memory /
// auto_fit_max_model_len.
// Ported from vllm/v1/core/kv_cache_utils.py @ 555967922 (:751-788) and, per
// GROUP, @ e126687a9a (:933-967, :2029-2157, :2304-2327). Issue #83 M4; the
// guard external PR #227 needed; FIX-KV-POOL-MIN-FIT
// (ISSUE-LOCAL-01M36QVG0KGKEP4MSMKT18MMZ4) for the per-group sum and the null
// block.
// ---------------------------------------------------------------------------
namespace {

// Upstream `new_kv_cache_spec()` (tests/v1/core/test_kv_cache_utils.py):
// block_size 16, 2 KV heads, head_size 64, float32 -> a 16384-byte page.
std::shared_ptr<FullAttentionSpec> NewKvCacheSpec(int block_size = 16) {
  return std::make_shared<FullAttentionSpec>(block_size, /*num_kv_heads=*/2,
                                             /*head_size=*/64, vt::DType::kF32);
}

// Upstream `new_mamba_spec()`: shapes ((2, 512), (3, 32, 32)), float32.
std::shared_ptr<MambaSpec> NewMambaSpec(int block_size, int num_speculative_blocks,
                                        const std::string& mode = "none") {
  return std::make_shared<MambaSpec>(
      block_size, std::vector<std::vector<int64_t>>{{2, 512}, {3, 32, 32}},
      std::vector<vt::DType>{vt::DType::kF32, vt::DType::kF32},
      /*page_size_padded=*/std::nullopt, mode, num_speculative_blocks);
}

vllm::v1::KVCacheConfig OneLayerPerGroup(
    std::vector<std::shared_ptr<KVCacheSpec>> specs, int num_blocks = 0) {
  vllm::v1::KVCacheConfig cfg{};
  cfg.num_blocks = num_blocks;
  int i = 0;
  for (auto& spec : specs) {
    cfg.kv_cache_groups.emplace_back(
        std::vector<std::string>{"model.layers." + std::to_string(i++) + ".attn"},
        std::move(spec));
  }
  return cfg;
}

constexpr int64_t kPage = 16 * 2 * 64 * 4 * 2;  // upstream mem_per_block_per_layer

}  // namespace

TEST_CASE("max_blocks_per_request mirrors each spec kind's per-request usage") {
  // FullAttentionSpec.max_memory_usage_bytes (kv_cache_interface.py:464-469):
  // whole blocks.
  const auto full = NewKvCacheSpec(32);
  CHECK(vllm::v1::max_blocks_per_request(*full, 32, 2048) == 1);
  CHECK(vllm::v1::max_blocks_per_request(*full, 33, 2048) == 2);
  CHECK(vllm::v1::max_blocks_per_request(*full, 64, 2048) == 2);
  CHECK(vllm::v1::max_blocks_per_request(*full, 1, 2048) == 1);
  CHECK(vllm::v1::max_blocks_per_request(*full, 0, 2048) == 0);

  // SlidingWindowSpec (:704-738): the admission-cap function, window-bounded.
  const SlidingWindowSpec swa(16, 2, 64, vt::DType::kF32, /*sliding_window=*/64);
  CHECK(vllm::v1::max_blocks_per_request(swa, 8192, 32) ==
        swa.max_admission_blocks_per_request(32, 8192));
  CHECK(vllm::v1::max_blocks_per_request(swa, 8192, 32) == (64 - 1 + 32 + 15) / 16 + 1);
  CHECK(vllm::v1::max_blocks_per_request(swa, 16, 32) == 2);  // min(.., len) + 1

  // MambaSpec (:883-894). `align`: 2 + k.
  CHECK(vllm::v1::max_blocks_per_request(*NewMambaSpec(16, 2, "align"), 8192, 2048) ==
        4);
  // `none` at upstream's resolved geometry (block_size = max_model_len,
  // config.py:657) is upstream's `1 + k`...
  CHECK(vllm::v1::max_blocks_per_request(*NewMambaSpec(8192, 2), 8192, 2048) == 3);
  // ...and at this tree's attention block size it is what MambaManager claims
  // for a fresh request, cdiv(len + bs*k, bs) = cdiv(len, bs) + k.
  CHECK(vllm::v1::max_blocks_per_request(*NewMambaSpec(32, 3), 128, 2048) == 4 + 3);
  CHECK(vllm::v1::max_blocks_per_request(*NewMambaSpec(32, 0), 128, 2048) == 4);
}

TEST_CASE("max_memory_usage_bytes_from_groups sums EVERY group's blocks") {
  // _max_memory_usage_bytes_from_groups (kv_cache_utils.py:2029-2058): each
  // group claims its own blocks from the one pool. A Qwen3.5 speculative shape
  // (fa, gdn, fa_draft) needs three tables, not one.
  const auto cfg = OneLayerPerGroup(
      {NewKvCacheSpec(32), NewMambaSpec(32, 3), NewKvCacheSpec(32)});
  const int64_t bpb = vllm::v1::KVBytesPerBlock(cfg);
  REQUIRE(bpb == 2 * kPage * 2);  // two attention layers, 32-token pages
  CHECK(vllm::v1::max_memory_usage_bytes_from_groups(cfg, 128, 2048) ==
        bpb * (4 + (4 + 3) + 4));
  // No paged KV: nothing to size.
  CHECK(vllm::v1::max_memory_usage_bytes_from_groups(
            OneLayerPerGroup({NewMambaSpec(32, 3)}), 128, 2048) == 0);
}

TEST_CASE("estimate_max_model_len_from_groups is the longest length that fits") {
  // _estimate_max_model_len_from_groups (:2061-2093): upstream's binary search.
  const auto cfg = OneLayerPerGroup({NewKvCacheSpec(16)});
  const int64_t bpb = vllm::v1::KVBytesPerBlock(cfg);
  CHECK(vllm::v1::estimate_max_model_len_from_groups(cfg, bpb * 10, 4096, 2048) ==
        160);
  // A partial block buys nothing.
  CHECK(vllm::v1::estimate_max_model_len_from_groups(cfg, bpb * 10 + bpb - 1, 4096,
                                                     2048) == 160);
  // Capped at the length being searched.
  CHECK(vllm::v1::estimate_max_model_len_from_groups(cfg, bpb * 10, 100, 2048) == 100);
  // Not even one block: 0, which suppresses the "estimated maximum" clause.
  CHECK(vllm::v1::estimate_max_model_len_from_groups(cfg, bpb - 1, 4096, 2048) == 0);
  CHECK(vllm::v1::estimate_max_model_len_from_groups(cfg, 0, 4096, 2048) == 0);

  // Agreement with the predicate it searches, across a grid, on the
  // three-group shape.
  const auto spec3 = OneLayerPerGroup(
      {NewKvCacheSpec(32), NewMambaSpec(32, 3), NewKvCacheSpec(32)});
  const int64_t bpb3 = vllm::v1::KVBytesPerBlock(spec3);
  for (int64_t blocks = 0; blocks <= 40; ++blocks) {
    const int64_t avail = blocks * bpb3;
    const int64_t est =
        vllm::v1::estimate_max_model_len_from_groups(spec3, avail, 4096, 2048);
    if (est > 0) {
      CHECK(vllm::v1::max_memory_usage_bytes_from_groups(spec3, est, 2048) <= avail);
      CHECK(vllm::v1::max_memory_usage_bytes_from_groups(spec3, est + 1, 2048) >
            avail);
    } else {
      CHECK(vllm::v1::max_memory_usage_bytes_from_groups(spec3, 1, 2048) > avail);
    }
  }
}

TEST_CASE("check_enough_kv_cache_memory admits a pool that holds the context") {
  CHECK_NOTHROW(vllm::v1::check_enough_kv_cache_memory(
      /*available_memory=*/1000, /*needed_memory=*/1000,
      /*max_model_len=*/320, /*estimated_max_model_len=*/320));
  CHECK_NOTHROW(vllm::v1::check_enough_kv_cache_memory(1000, 100, 32, 320));
  // An attention-free model needs no KV at all, so nothing is short.
  CHECK_NOTHROW(vllm::v1::check_enough_kv_cache_memory(1, 0, 32, 32));
}

TEST_CASE("check_enough_kv_cache_memory refuses a pool that cannot") {
  // :769-788. The message names both sizes, the estimate, and the remedies.
  CHECK_THROWS_AS(
      vllm::v1::check_enough_kv_cache_memory(/*available_memory=*/1000,
                                             /*needed_memory=*/8000,
                                             /*max_model_len=*/2560,
                                             /*estimated_max_model_len=*/320),
      std::invalid_argument);
  try {
    vllm::v1::check_enough_kv_cache_memory(1000, 8000, 2560, 320);
    FAIL("expected the KV sizing check to refuse this configuration");
  } catch (const std::invalid_argument& e) {
    const std::string msg = e.what();
    CHECK(msg.find("To serve at least one request with the model's max seq "
                   "len (2560)") != std::string::npos);
    CHECK(msg.find("larger than the available KV cache memory") !=
          std::string::npos);
    CHECK(msg.find("estimated maximum model length is 320") !=
          std::string::npos);
    CHECK(msg.find("decreasing `max_model_len`") != std::string::npos);
    CHECK(msg.find("--num-blocks") != std::string::npos);
  }

  // estimated <= 0 suppresses the estimate clause, exactly as upstream's
  // `if estimated_max_len > 0` does (:775-780).
  try {
    vllm::v1::check_enough_kv_cache_memory(50, 8000, 2560, 0);
    FAIL("expected a refusal");
  } catch (const std::invalid_argument& e) {
    CHECK(std::string(e.what()).find("Based on the available memory") ==
          std::string::npos);
  }
}

TEST_CASE("check_enough_kv_cache_memory refuses an empty pool by name") {
  // :757-765, the `available_memory <= 0` arm — a different message.
  CHECK_THROWS_WITH_AS(
      vllm::v1::check_enough_kv_cache_memory(0, 100, 32, 0),
      doctest::Contains("No available memory for the cache blocks"),
      std::invalid_argument);
}

TEST_CASE("auto_fit_max_model_len reduces the context to what the pool holds") {
  // _auto_fit_max_model_len (:2096-2157) after the null-block reservation
  // (:2304-2311). 11 blocks = 10 usable x 16 = 160 tokens.
  const auto cfg = OneLayerPerGroup({NewKvCacheSpec(16)});
  const int64_t bpb = vllm::v1::KVBytesPerBlock(cfg);
  CHECK(vllm::v1::auto_fit_max_model_len(cfg, bpb * 11, /*derived=*/4096, 2048) ==
        160);
  // The full context fits: keep it, never round it UP to the pool.
  CHECK(vllm::v1::auto_fit_max_model_len(cfg, bpb * 11, 128, 2048) == 128);
  CHECK(vllm::v1::auto_fit_max_model_len(cfg, bpb * 11, 160, 2048) == 160);
  // Attention-free (no paged KV at all): nothing to fit against (:2113-2120).
  CHECK(vllm::v1::auto_fit_max_model_len(vllm::v1::KVCacheConfig{}, 0, 4096, 2048) ==
        4096);
  // Only the null block: not even one token fits (:2133-2137).
  CHECK_THROWS_WITH_AS(vllm::v1::auto_fit_max_model_len(cfg, bpb, 4096, 2048),
                       doctest::Contains("Cannot auto-fit max_model_len"),
                       std::invalid_argument);
}

// Upstream test_auto_fit_max_model_len_with_hybrid
// (tests/v1/core/test_kv_cache_utils.py:2754-2776 @ e126687a9a). ADAPTATION,
// recorded in .agents/specs/kv-pool-min-fit.md: the Mamba spec is built at the
// block size upstream resolves for `none` mode, `max_model_len`
// (config.py:657), because this port counts the allocator's claim
// `cdiv(len, bs) + k`, which is upstream's `1 + k` only at that geometry.
TEST_CASE("auto_fit_max_model_len with a hybrid (Mamba + attention) config") {
  const int gamma = 2;
  const auto cfg =
      OneLayerPerGroup({NewMambaSpec(/*block_size=*/8192, gamma), NewKvCacheSpec()});
  REQUIRE(vllm::v1::KVBytesPerBlock(cfg) == kPage);
  // One extra block on top of what a 1024-token request needs: the pool
  // reserves one block as the null block.
  const int64_t available_memory = kPage * (1024 / 16 + 1 + gamma + 1);
  CHECK(vllm::v1::auto_fit_max_model_len(cfg, available_memory, 8192, 2048) == 1024);
}

// Upstream test_auto_fit_max_model_len_reserves_null_block (:3189-3207).
TEST_CASE("auto_fit_max_model_len reserves the null block") {
  const auto cfg = OneLayerPerGroup({NewKvCacheSpec(16)});
  // Exactly the 1024 / 16 = 64 blocks a full-length request would need.
  CHECK(vllm::v1::auto_fit_max_model_len(cfg, kPage * 64, 1024, 2048) == 63 * 16);
}

// Upstream test_check_enough_kv_cache_memory_reserves_null_block (:3210-3229)
// and test_kv_cache_reserves_null_block_for_max_model_len (:3164-3186).
TEST_CASE("check_enough_kv_cache_memory reserves the null block") {
  const auto cfg = OneLayerPerGroup({NewKvCacheSpec(16)});
  // 512 / 16 = 32 blocks needed. 32 blocks -> 31 usable: one short -> reject.
  CHECK_THROWS_WITH_AS(
      vllm::v1::check_enough_kv_cache_memory(cfg, kPage * 32, 512, 2048),
      doctest::Contains("max seq len"), std::invalid_argument);
  // 33 blocks -> 32 usable -> accept.
  CHECK_NOTHROW(vllm::v1::check_enough_kv_cache_memory(cfg, kPage * 33, 512, 2048));
}

TEST_CASE("check_enough_kv_cache_memory counts every group of a hybrid config") {
  // The FIX-KV-POOL-MIN-FIT defect at unit scale: fa + gdn(k=3) + fa_draft at
  // 128 tokens needs 4 + 7 + 4 = 15 blocks plus the null block.
  const auto cfg = OneLayerPerGroup(
      {NewKvCacheSpec(32), NewMambaSpec(32, 3), NewKvCacheSpec(32)});
  const int64_t bpb = vllm::v1::KVBytesPerBlock(cfg);
  CHECK_THROWS_WITH_AS(vllm::v1::check_enough_kv_cache_memory(cfg, bpb * 15, 128, 2048),
                       doctest::Contains("max seq len (128)"), std::invalid_argument);
  CHECK_NOTHROW(vllm::v1::check_enough_kv_cache_memory(cfg, bpb * 16, 128, 2048));
  // No paged KV: never refused (upstream's `if kv_cache_spec:`).
  CHECK_NOTHROW(vllm::v1::check_enough_kv_cache_memory(
      OneLayerPerGroup({NewMambaSpec(32, 3)}), 0, 128, 2048));
}
