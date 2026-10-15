// Ported from: vllm/v1/core/block_pool.py @ e24d1b24
// See include/vllm/v1/core/block_pool.h for scope, the pinned-API fidelity
// note, and recorded deferrals / deviations.
#include "vllm/v1/core/block_pool.h"

#include <cassert>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vllm/v1/request.h"

namespace vllm::v1 {
namespace {

// Build num_gpu_blocks KVCacheBlocks with block_id = 0..n-1. Reserve up front so
// the buffer never reallocates: FreeKVCacheBlockQueue and cached_block_hash_to_block
// hold raw pointers into these elements, so their addresses must stay stable.
std::vector<KVCacheBlock> MakeBlocks(int64_t num_gpu_blocks) {
  std::vector<KVCacheBlock> blocks;
  if (num_gpu_blocks > 0) {
    blocks.reserve(static_cast<size_t>(num_gpu_blocks));
    for (int64_t idx = 0; idx < num_gpu_blocks; ++idx) {
      blocks.emplace_back(static_cast<int>(idx));
    }
  }
  return blocks;
}

// A KVCacheBlock* view over `blocks`, used to seed FreeKVCacheBlockQueue.
std::vector<KVCacheBlock*> BlockPtrs(std::vector<KVCacheBlock>& blocks) {
  std::vector<KVCacheBlock*> ptrs;
  ptrs.reserve(blocks.size());
  for (auto& block : blocks) {
    ptrs.push_back(&block);
  }
  return ptrs;
}

}  // namespace

BlockPool::BlockPool(int64_t num_gpu_blocks, bool enable_caching,
                     int hash_block_size, bool enable_kv_cache_events)
    : num_gpu_blocks(num_gpu_blocks),
      enable_caching(enable_caching),
      hash_block_size(hash_block_size),
      blocks(MakeBlocks(num_gpu_blocks)),
      free_block_queue(BlockPtrs(blocks)),
      null_block(nullptr),
      enable_kv_cache_events(enable_kv_cache_events) {
  assert(num_gpu_blocks > 0);

  // To represent a placeholder block with block_id=0. The ref_cnt of null_block
  // is not maintained; it is popped out of the free queue here so it can never
  // be allocated. (Special care is needed elsewhere to avoid freeing it.)
  null_block = free_block_queue.popleft();
  null_block->is_null = true;
}

std::optional<std::vector<KVCacheBlock*>> BlockPool::get_cached_block(
    const BlockHash& block_hash, const std::vector<int>& kv_cache_group_ids) {
  std::vector<KVCacheBlock*> cached_blocks;
  for (int group_id : kv_cache_group_ids) {
    const BlockHashWithGroupId block_hash_with_group_id =
        make_block_hash_with_group_id(block_hash,
                                      static_cast<uint32_t>(group_id));
    auto it = cached_block_hash_to_block.find(block_hash_with_group_id);
    if (it == cached_block_hash_to_block.end() || it->second.empty()) {
      return std::nullopt;
    }
    // If there are duplicated blocks, return the first block in the cache.
    cached_blocks.push_back(it->second.begin()->second);
  }
  return cached_blocks;
}

void BlockPool::cache_full_blocks(
    const Request& request, const std::vector<KVCacheBlock*>& blocks_arg,
    int num_cached_blocks, int num_full_blocks, int block_size,
    int kv_cache_group_id, const std::optional<std::vector<bool>>& block_mask) {
  if (num_cached_blocks >= num_full_blocks) {
    return;
  }
  // new_full_blocks = blocks[num_cached_blocks:num_full_blocks] (kept as an
  // index range into blocks_arg to avoid a copy).
  const int num_new_full = num_full_blocks - num_cached_blocks;
  assert(!block_mask.has_value() ||
         static_cast<int>(block_mask->size()) == num_new_full);

  // Common case: block_size == hash_block_size, block_hashes = request's list.
  // The align path (block_size a multiple of hash_block_size, upstream's
  // BlockHashListWithBlockSize) is DEFERRED (see header).
  if (block_size != hash_block_size) {
    throw std::runtime_error(
        "BlockPool::cache_full_blocks: align mode "
        "(block_size != hash_block_size) not yet ported");
  }
  const std::vector<BlockHash>& block_hashes = request.block_hashes;
  if (static_cast<int>(block_hashes.size()) < num_full_blocks ||
      static_cast<int>(blocks_arg.size()) < num_full_blocks) {
    throw std::runtime_error(
        "BlockPool::cache_full_blocks: full pages require committed hashes "
        "and allocated blocks");
  }

  // new_block_hashes = block_hashes[num_cached_blocks:]; indexed as
  // block_hashes[num_cached_blocks + i] below.
  // new_hashes collects the external hashes for the BlockStored event, in the
  // same block order (only when events are enabled). Mirrors block_pool.py:268.
  std::vector<distributed::ExternalBlockHash> new_hashes;
  for (int i = 0; i < num_new_full; ++i) {
    KVCacheBlock* blk = blocks_arg[num_cached_blocks + i];
    // Some blocks may be null or masked out (sparse/sliding-window attention or
    // mamba align mode). Skip them.
    if (blk->is_null || (block_mask.has_value() && !(*block_mask)[i])) {
      continue;
    }
    const BlockHash& block_hash = block_hashes[num_cached_blocks + i];
    const int num_hash_tokens = (num_cached_blocks + i + 1) * block_size;

    const BlockHashWithGroupId block_hash_with_group_id =
        make_block_hash_with_group_id(block_hash,
                                      static_cast<uint32_t>(kv_cache_group_id));
    if (blk->block_hash().has_value()) {
      // The only valid case where a "new full block" already has a hash is a
      // partial->full promotion of the same cache block — DEFERRED (requires the
      // partial primitives). See header. (Upstream emits BlockRemoved for the
      // superseded hashes here; unreachable while promotion throws.)
      throw std::runtime_error(
          "BlockPool::cache_full_blocks: partial->full promotion not yet "
          "ported");
    }
    _insert_block_hash(block_hash_with_group_id, blk, num_hash_tokens);
    if (enable_kv_cache_events) {
      new_hashes.push_back(maybe_convert_block_hash(block_hash));
    }
  }

  // BlockStored emission (block_pool.py:301-342). Guarded: the default
  // (events-off) path never runs any of this, so it stays byte-identical.
  if (!enable_kv_cache_events) {
    return;
  }
  std::optional<distributed::ExternalBlockHash> parent_block_hash;
  if (num_cached_blocks != 0) {
    parent_block_hash =
        maybe_convert_block_hash(block_hashes[num_cached_blocks - 1]);
  }
  const int start_token_idx = num_cached_blocks * block_size;
  const int end_token_idx = num_full_blocks * block_size;

  // Per-block extra keys (one entry per non-null / non-masked block), matching
  // the length of new_hashes. block_pool.py:317-329.
  std::vector<distributed::EventExtraKey> extra_keys_list;
  int curr_mm_idx = 0;
  for (int i = num_cached_blocks; i < num_full_blocks; ++i) {
    if (blocks_arg[i]->is_null) {
      continue;
    }
    if (block_mask.has_value() && !(*block_mask)[i - num_cached_blocks]) {
      continue;
    }
    const int block_start = i * block_size;
    const int block_end = block_start + block_size;
    std::pair<ExtraKeys, int> extra = generate_block_hash_extra_keys(
        request, block_start, block_end, curr_mm_idx);
    curr_mm_idx = extra.second;
    extra_keys_list.push_back(std::move(extra.first));
  }

  kv_event_queue.emplace_back(_build_block_stored_event(
      request, std::move(new_hashes), parent_block_hash, start_token_idx,
      end_token_idx, block_size, kv_cache_group_id, std::move(extra_keys_list)));
}

distributed::BlockStored BlockPool::_build_block_stored_event(
    const Request& request,
    std::vector<distributed::ExternalBlockHash> block_hashes,
    std::optional<distributed::ExternalBlockHash> parent_block_hash,
    int start_token_idx, int end_token_idx, int block_size,
    int kv_cache_group_id,
    std::vector<distributed::EventExtraKey> extra_keys_list) const {
  // block_pool.py:344-371.
  const std::vector<int32_t> all_tokens = request.AllTokenIds();
  std::vector<int64_t> token_ids;
  const int lo = start_token_idx < 0 ? 0 : start_token_idx;
  const int hi = end_token_idx > static_cast<int>(all_tokens.size())
                     ? static_cast<int>(all_tokens.size())
                     : end_token_idx;
  token_ids.reserve(static_cast<size_t>(hi > lo ? hi - lo : 0));
  for (int i = lo; i < hi; ++i) {
    token_ids.push_back(static_cast<int64_t>(all_tokens[i]));
  }

  distributed::BlockStored event;
  event.block_hashes = std::move(block_hashes);
  event.parent_block_hash = std::move(parent_block_hash);
  event.token_ids = std::move(token_ids);
  event.block_size = block_size;
  // lora_id: the DEPRECATED int adapter id. The T0 Request carries only
  // lora_name (LORA-RUNTIME defers the adapter id), so this stays None — exactly
  // upstream's value on the base-model path (lora_request is None). On the
  // text/base path lora_name is also None, so the event is byte-identical.
  event.lora_id = std::nullopt;
  event.medium = std::string(distributed::MEDIUM_GPU);
  event.lora_name = request.lora_name;
  // extra_keys=extra_keys_list if extra_keys_list else None (block_pool.py:369).
  if (!extra_keys_list.empty()) {
    event.extra_keys = std::move(extra_keys_list);
  }
  event.group_idx = kv_cache_group_id;
  return event;
}

void BlockPool::emit_cached_block_events(const Request& request,
                                         int num_cached_blocks, int block_size,
                                         int kv_cache_group_id) {
  // block_pool.py:373-443. Reuse events for prefix-cache hits; state unchanged.
  if (!enable_kv_cache_events || num_cached_blocks == 0) {
    return;
  }
  // Common path: block_size == hash_block_size, so resolve_block_hashes is the
  // request's own block_hashes (the align path is DEFERRED, as in
  // cache_full_blocks).
  if (block_size != hash_block_size) {
    throw std::runtime_error(
        "BlockPool::emit_cached_block_events: align mode "
        "(block_size != hash_block_size) not yet ported");
  }
  const std::vector<BlockHash>& block_hashes = request.block_hashes;

  std::vector<distributed::ExternalBlockHash> cached_hashes;
  std::vector<distributed::EventExtraKey> extra_keys_list;
  int curr_mm_idx = 0;
  for (int i = 0; i < num_cached_blocks; ++i) {
    const int block_start = i * block_size;
    const int block_end = block_start + block_size;
    cached_hashes.push_back(maybe_convert_block_hash(block_hashes[i]));
    std::pair<ExtraKeys, int> extra = generate_block_hash_extra_keys(
        request, block_start, block_end, curr_mm_idx);
    curr_mm_idx = extra.second;
    extra_keys_list.push_back(std::move(extra.first));
  }
  if (cached_hashes.empty()) {
    return;
  }

  // Prefix-cache hits always form a contiguous prefix from block 0, so the
  // parent is None.
  const int start_token_idx = 0;
  const int end_token_idx = num_cached_blocks * block_size;
  kv_event_queue.emplace_back(_build_block_stored_event(
      request, std::move(cached_hashes), /*parent=*/std::nullopt,
      start_token_idx, end_token_idx, block_size, kv_cache_group_id,
      std::move(extra_keys_list)));
}

std::optional<BlockHashWithGroupId> BlockPool::cache_partial_block(
    const Request& /*request*/, KVCacheBlock* /*block*/, int /*num_tokens*/,
    int /*kv_cache_group_id*/, int /*block_size*/) {
  // DEFERRED 1:1 stub (partial prefix-cache primitives). See header.
  throw std::runtime_error(
      "BlockPool::cache_partial_block: partial primitives not yet ported");
}

void BlockPool::evict_blocks(const std::set<int>& block_ids) {
  // KV-OFFLOAD W4 (jointly with KV-BLOCK-POOL). 1:1 with block_pool.py:637-654:
  // evict the given blocks from the PREFIX CACHE hash table only. Blocks with
  // ref_cnt > 0 stay in the pool (not freed) — this drops only their cache
  // reachability, exactly as _maybe_evict_cached_block does. Used when a
  // connector reports blocks whose external copy has become authoritative.
  for (int block_id : block_ids) {
    if (block_id < 0 || block_id >= static_cast<int>(blocks.size())) {
      // A connector should only report scheduler-allocated ids; upstream asserts
      // (block_pool.py:648-652). We THROW rather than corrupt on an out-of-range
      // id, which is the safe side.
      throw std::out_of_range(
          "BlockPool::evict_blocks: block_id out of range (connector bug)");
    }
    _maybe_evict_cached_block(&blocks[block_id]);
  }
}

std::vector<BlockHashWithGroupId> BlockPool::_remove_cached_block_hashes(
    KVCacheBlock* block) {
  std::vector<BlockHashWithGroupId> block_hashes;
  if (block->block_hash().has_value()) {
    block_hashes.push_back(block->block_hash().value());
  }
  auto by_block = cached_block_hashes_by_block.find(block->block_id);
  if (by_block != cached_block_hashes_by_block.end()) {
    block_hashes.insert(block_hashes.end(), by_block->second.begin(),
                        by_block->second.end());
    cached_block_hashes_by_block.erase(by_block);
  }
  if (block_hashes.empty()) {
    return {};
  }

  std::vector<BlockHashWithGroupId> removed_hashes;
  for (const BlockHashWithGroupId& block_hash : block_hashes) {
    // cached_block_hash_to_block.pop(block_hash, block_id): remove block_id from
    // the key's inner map; drop the key when its inner map empties; the pop
    // "succeeds" only when block_id was present.
    auto it = cached_block_hash_to_block.find(block_hash);
    if (it == cached_block_hash_to_block.end()) {
      continue;
    }
    auto inner = it->second.find(block->block_id);
    if (inner == it->second.end()) {
      continue;
    }
    it->second.erase(inner);
    if (it->second.empty()) {
      cached_block_hash_to_block.erase(it);
    }
    removed_hashes.push_back(block_hash);
  }
  block->reset_hash();
  return removed_hashes;
}

void BlockPool::_insert_block_hash(
    const BlockHashWithGroupId& block_hash_with_group_id, KVCacheBlock* block,
    std::optional<int> num_tokens) {
  if (block->block_hash().has_value() &&
      block->block_hash().value() == block_hash_with_group_id) {
    return;
  }
  auto it = cached_block_hash_to_block.find(block_hash_with_group_id);
  if (it != cached_block_hash_to_block.end() &&
      it->second.count(block->block_id) != 0) {
    return;
  }

  if (!block->block_hash().has_value()) {
    block->set_block_hash(block_hash_with_group_id, num_tokens);
  } else {
    cached_block_hashes_by_block[block->block_id].insert(
        block_hash_with_group_id);
  }
  cached_block_hash_to_block[block_hash_with_group_id][block->block_id] = block;
}

std::vector<KVCacheBlock*> BlockPool::get_new_blocks(int num_blocks) {
  if (num_blocks > get_num_free_blocks()) {
    throw std::runtime_error("Cannot get free blocks from the pool");
  }

  std::vector<KVCacheBlock*> ret = free_block_queue.popleft_n(num_blocks);

  // In order to only iterate the list once, we duplicate a bit of code.
  if (enable_caching) {
    for (KVCacheBlock* block : ret) {
      _maybe_evict_cached_block(block);
      assert(block->ref_cnt == 0);
      block->incr_ref();
      // metrics_collector.on_block_allocated — DEFERRED (see header).
    }
  } else {
    for (KVCacheBlock* block : ret) {
      assert(block->ref_cnt == 0);
      block->incr_ref();
      // metrics_collector.on_block_allocated — DEFERRED (see header).
    }
  }
  return ret;
}

void BlockPool::_emit_block_removed_events(
    const std::vector<BlockHashWithGroupId>& block_hashes) {
  // block_pool.py:592-605.
  if (!enable_kv_cache_events) {
    return;
  }
  for (const BlockHashWithGroupId& bh : block_hashes) {
    distributed::BlockRemoved event;
    event.block_hashes.push_back(maybe_convert_block_hash(get_block_hash(bh)));
    event.medium = std::string(distributed::MEDIUM_GPU);
    event.group_idx = static_cast<int64_t>(get_group_id(bh));
    kv_event_queue.emplace_back(std::move(event));
  }
}

bool BlockPool::_maybe_evict_cached_block(KVCacheBlock* block) {
  // metrics_collector.on_block_evicted — DEFERRED (see header).
  std::vector<BlockHashWithGroupId> evicted_hashes =
      _remove_cached_block_hashes(block);
  if (evicted_hashes.empty()) {
    // The block doesn't have a hash, eviction is not needed.
    return false;
  }
  _emit_block_removed_events(evicted_hashes);
  return true;
}

void BlockPool::touch(const std::vector<KVCacheBlock*>& blocks_arg) {
  for (KVCacheBlock* block : blocks_arg) {
    // ref_cnt=0 means this block is in the free list (i.e. an eviction
    // candidate), so remove it.
    if (block->ref_cnt == 0 && !block->is_null) {
      free_block_queue.remove(block);
    }
    block->incr_ref();
  }
}

void BlockPool::free_blocks(const std::vector<KVCacheBlock*>& ordered_blocks) {
  // Identify blocks with a hash (LRU cache candidates) and without one (they
  // can never match in APC, so they should be evicted first). Mirrors upstream
  // block_pool.py free_blocks 1:1.
  std::vector<KVCacheBlock*> blocks_with_hash;
  std::vector<KVCacheBlock*> blocks_without_hash;
  for (KVCacheBlock* block : ordered_blocks) {
    block->decr_ref();
    if (block->ref_cnt == 0 && !block->is_null) {
      if (!block->block_hash().has_value()) {
        blocks_without_hash.push_back(block);
      } else {
        blocks_with_hash.push_back(block);
      }
    }
  }

  // Blocks without a hash always get evicted first: prepend them to the FRONT
  // (immediate reuse). Blocks with a hash go to the tail (LRU order).
  free_block_queue.prepend_n(blocks_without_hash);
  free_block_queue.append_n(blocks_with_hash);
}

bool BlockPool::reset_prefix_cache() {
  const int64_t num_used_blocks = num_gpu_blocks - get_num_free_blocks();
  if (num_used_blocks != 1) {  // The null block is always marked as used.
    // Some blocks are still in use; cannot reset (upstream logs a warning).
    return false;
  }

  // Remove all hashes so that no new blocks will hit.
  cached_block_hash_to_block.clear();
  cached_block_hashes_by_block.clear();

  // Remove all hashes from all blocks.
  for (KVCacheBlock& block : blocks) {
    block.reset_hash();
  }

  if (enable_kv_cache_events) {
    kv_event_queue.emplace_back(distributed::AllBlocksCleared{});
  }
  return true;
}

int BlockPool::get_num_free_blocks() const {
  return free_block_queue.num_free_blocks;
}

double BlockPool::get_usage() const {
  // Subtract 1 to account for the null block.
  const int64_t total_gpu_blocks = num_gpu_blocks - 1;
  if (total_gpu_blocks == 0) {
    return 0.0;
  }
  return 1.0 - (static_cast<double>(get_num_free_blocks()) /
                static_cast<double>(total_gpu_blocks));
}

std::vector<KVCacheEvent> BlockPool::take_events() {
  // block_pool.py:820-830: atomically drain kv_event_queue (KV-EVENTS).
  if (!enable_kv_cache_events) {
    return {};
  }
  std::vector<KVCacheEvent> events = std::move(kv_event_queue);
  kv_event_queue.clear();
  return events;
}

}  // namespace vllm::v1
