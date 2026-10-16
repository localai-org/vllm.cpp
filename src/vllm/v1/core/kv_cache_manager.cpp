#include "vllm/v1/core/recurrent_prefix_snapshot.h"
// Ported from: vllm/v1/core/kv_cache_manager.py @ e24d1b24
// See include/vllm/v1/core/kv_cache_manager.h for scope + deviations.

#include "vllm/v1/core/kv_cache_manager.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "vllm/v1/request.h"

namespace vllm::v1 {

// --- KVCacheBlocks ----------------------------------------------------------

KVCacheBlocks KVCacheBlocks::operator+(const KVCacheBlocks& other) const {
  KVCacheBlocksTuple out;
  out.reserve(blocks.size());
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    std::vector<KVCacheBlock*> merged = blocks[i];
    const auto& rhs = other.blocks[i];
    merged.insert(merged.end(), rhs.begin(), rhs.end());
    out.push_back(std::move(merged));
  }
  return KVCacheBlocks{std::move(out)};
}

std::vector<std::vector<int>> KVCacheBlocks::get_block_ids() const {
  std::vector<std::vector<int>> out;
  out.reserve(blocks.size());
  for (const auto& group : blocks) {
    std::vector<int> ids;
    ids.reserve(group.size());
    for (const KVCacheBlock* blk : group) ids.push_back(blk->block_id);
    out.push_back(std::move(ids));
  }
  return out;
}

std::optional<std::vector<std::vector<int>>> KVCacheBlocks::get_block_ids(
    bool allow_none) const {
  if (allow_none) {
    bool all_empty = true;
    for (const auto& group : blocks) {
      if (!group.empty()) {
        all_empty = false;
        break;
      }
    }
    if (all_empty) return std::nullopt;
  }
  return get_block_ids();
}

std::vector<int> KVCacheBlocks::get_unhashed_block_ids() const {
  if (blocks.size() != 1) {
    throw std::runtime_error("Only one group is supported");
  }
  std::vector<int> out;
  for (const KVCacheBlock* block : blocks[0]) {
    if (!block->block_hash().has_value()) out.push_back(block->block_id);
  }
  return out;
}

std::vector<std::vector<int>> KVCacheBlocks::get_unhashed_block_ids_all_groups()
    const {
  std::vector<std::vector<int>> out;
  out.reserve(blocks.size());
  for (const auto& group : blocks) {
    std::vector<int> ids;
    for (const KVCacheBlock* block : group) {
      if (!block->block_hash().has_value() && !block->is_null) {
        ids.push_back(block->block_id);
      }
    }
    out.push_back(std::move(ids));
  }
  return out;
}

KVCacheBlocks KVCacheBlocks::new_empty() const {
  return KVCacheBlocks{KVCacheBlocksTuple(blocks.size())};
}

// --- KVCacheManager ---------------------------------------------------------

KVCacheManager::KVCacheManager(KVCacheConfig kv_cache_config_in,
                               int max_model_len_in, int scheduler_block_size,
                               int hash_block_size,
                               std::optional<int> max_num_batched_tokens,
                               bool enable_caching_in, bool use_eagle_in,
                               bool log_stats_in, bool enable_kv_cache_events,
                               int dcp_world_size, int pcp_world_size,
                               double watermark)
    : max_model_len(max_model_len_in),
      enable_caching(enable_caching_in),
      use_eagle(use_eagle_in),
      log_stats(log_stats_in),
      enable_kv_cache_events(enable_kv_cache_events),
      num_kv_cache_groups(
          static_cast<int>(kv_cache_config_in.kv_cache_groups.size())),
      kv_cache_config(kv_cache_config_in),
      // Set below (needs num_blocks); placeholder to satisfy member order.
      watermark_blocks(0),
      // When unset, fall back to max_model_len so the recycling-aware cap
      // collapses to the prior (uncapped) admission behavior.
      coordinator(get_kv_cache_coordinator(
          std::move(kv_cache_config_in), max_model_len_in,
          max_num_batched_tokens.value_or(max_model_len_in), use_eagle_in,
          enable_caching_in, enable_kv_cache_events, dcp_world_size,
          pcp_world_size, scheduler_block_size, hash_block_size)),
      block_pool(coordinator->block_pool),
      empty_kv_cache_blocks(
          KVCacheBlocks{KVCacheBlocksTuple(kv_cache_config.kv_cache_groups.size())}) {
  if (watermark < 0.0) {
    throw std::runtime_error("watermark must be non-negative");
  }
  watermark_blocks =
      static_cast<int>(watermark * static_cast<double>(kv_cache_config.num_blocks));
}

double KVCacheManager::usage() const { return block_pool.get_usage(); }

std::pair<KVCacheBlocks, int> KVCacheManager::get_computed_blocks(
    const Request& request) {
  // Skip the prefix-cache lookup when caching is disabled. (Upstream also
  // skips on request.skip_reading_prefix_cache, DEFERRED — not on T0 Request.)
  if (!enable_caching) {
    return {empty_kv_cache_blocks, 0};
  }

  // NOTE: When all tokens hit the cache, we must recompute the last token to
  // obtain logits, so cap max_cache_hit_length at num_tokens - 1.
  int max_cache_hit_length = request.NumTokens() - 1;
  auto [computed_blocks, num_new_computed_tokens] =
      coordinator->find_longest_cache_hit(request.block_hashes,
                                          max_cache_hit_length);

  if (auto index = kv_cache_config.recurrent_prefix_snapshots) {
    const int boundary = num_new_computed_tokens / index->block_tokens();
    if (num_new_computed_tokens % index->block_tokens() != 0 || boundary <= 0 ||
        size_t(boundary) > request.block_hashes.size() ||
        !index->Pin(request.request_id, request.block_hashes[size_t(boundary - 1)])) {
      computed_blocks = empty_kv_cache_blocks.blocks;
      num_new_computed_tokens = 0;
      index->Release(request.request_id);
    }
  }

  // kv_cache_manager.py:262-280. Under the non-default report mode "full", the
  // blocks this request REUSED from the prefix cache are re-reported as
  // BlockStored, so an external prefix-cache-aware router learns that this
  // engine holds them (the default "incremental" mode reports only blocks that
  // were newly stored). Emits nothing on the default path: report mode is
  // "incremental" unless the request asked otherwise, and events are off unless
  // configured, so both guards are false in production.
  if (num_new_computed_tokens > 0 && enable_kv_cache_events &&
      request.kv_cache_report_mode == "full") {
    for (std::size_t group_idx = 0; group_idx < computed_blocks.size();
         ++group_idx) {
      const int num_blocks = static_cast<int>(computed_blocks[group_idx].size());
      if (num_blocks > 0) {
        const int group_block_size =
            kv_cache_config.kv_cache_groups[group_idx].kv_cache_spec->block_size;
        block_pool.emit_cached_block_events(request, num_blocks,
                                            group_block_size,
                                            static_cast<int>(group_idx));
      }
    }
  }

  // Upstream vllm/v1/core/kv_cache_manager.py:234-240. queries/hits count
  // TOKENS; a previously-preempted request is routed into the
  // mutually-exclusive preempted_* triple so its guaranteed second-pass hit
  // does not inflate the headline rate.
  if (log_stats) {
    prefix_cache_stats.record(request.NumTokens(), num_new_computed_tokens,
                              request.num_preemptions > 0);
  }

  return {create_kv_cache_blocks(computed_blocks), num_new_computed_tokens};
}

int KVCacheManager::num_matched_prefix_tokens(const Request& request) {
  // Pure read of the longest cached-prefix match — the same lookup
  // get_computed_blocks performs, MINUS the prefix_cache_stats.record side
  // effect (so LPM ordering never double-counts) and minus block allocation.
  // find_longest_cache_hit itself is a non-mutating walk of the hash chain (no
  // ref-count / LRU touch — those happen only in allocate_slots), so this is
  // safe to call for every waiting request each step.
  if (!enable_caching) {
    return 0;
  }
  const int max_cache_hit_length = request.NumTokens() - 1;
  if (max_cache_hit_length <= 0) {
    return 0;
  }
  auto [computed_blocks, num_new_computed_tokens] =
      coordinator->find_longest_cache_hit(request.block_hashes,
                                          max_cache_hit_length);
  (void)computed_blocks;
  return num_new_computed_tokens;
}

std::optional<KVCacheBlocks> KVCacheManager::allocate_slots(
    const Request& request, int num_new_tokens, int num_new_computed_tokens,
    std::optional<KVCacheBlocks> new_computed_blocks, int num_lookahead_tokens,
    int num_external_computed_tokens, bool delay_cache_blocks,
    int num_encoder_tokens, bool full_sequence_must_fit, int reserved_blocks,
    bool has_scheduled_reqs) {
  // When loading KV data asynchronously, we may have zero new tokens to compute
  // while still allocating slots for externally computed tokens.
  if (num_new_tokens == 0 && num_external_computed_tokens == 0) {
    throw std::runtime_error(
        "num_new_tokens must be greater than 0 when there are no external "
        "computed tokens");
  }

  const bool has_new_computed = new_computed_blocks.has_value();
  const KVCacheBlocksTuple& new_computed_block_list =
      has_new_computed ? new_computed_blocks->blocks
                       : empty_kv_cache_blocks.blocks;

  // The number of computed tokens is the number of computed tokens plus the new
  // prefix caching hits.
  const int num_local_computed_tokens =
      request.num_computed_tokens + num_new_computed_tokens;
  const int total_computed_tokens = std::min(
      num_local_computed_tokens + num_external_computed_tokens, max_model_len);

  int watermark_blocks_applied = 0;
  // The watermark is applied to waiting/preempted requests only, and only when
  // there's at least one request already scheduled.
  if (has_scheduled_reqs && (request.status == RequestStatus::kWaiting ||
                             request.status == RequestStatus::kPreempted)) {
    watermark_blocks_applied = watermark_blocks;
  }

  if (full_sequence_must_fit) {
    // First check and fail if the full request sequence won't fit.
    const int full_num_tokens = std::min(request.NumTokens(), max_model_len);
    const int num_blocks_full = coordinator->get_num_blocks_to_allocate(
        request.request_id, full_num_tokens, new_computed_block_list,
        num_encoder_tokens, total_computed_tokens, full_num_tokens,
        /*apply_admission_cap=*/true);
    const int required_full = num_blocks_full + watermark_blocks_applied;
    if (required_full > block_pool.get_num_free_blocks()) {
      return std::nullopt;
    }
  }

  const int num_tokens_main_model = total_computed_tokens + num_new_tokens;
  const int num_tokens_need_slot =
      std::min(num_tokens_main_model + num_lookahead_tokens, max_model_len);

  // Free the blocks that are skipped during the attention computation (e.g.
  // tokens outside the sliding window). Do this before allocating new blocks to
  // reduce the number of evicted blocks.
  coordinator->remove_skipped_blocks(request.request_id, total_computed_tokens,
                                     request.num_prompt_tokens);

  const int num_blocks_to_allocate = coordinator->get_num_blocks_to_allocate(
      request.request_id, num_tokens_need_slot, new_computed_block_list,
      num_encoder_tokens,
      num_local_computed_tokens + num_external_computed_tokens,
      num_tokens_main_model);

  // Keep reserved_blocks free for other in-flight sequences, and an additional
  // watermark of headroom for waiting/preempted admissions.
  const int available_blocks =
      block_pool.get_num_free_blocks() - reserved_blocks;
  const int required_blocks = num_blocks_to_allocate + watermark_blocks_applied;
  if (required_blocks > available_blocks) {
    // Cannot allocate new blocks (OOM -> the Scheduler preempts).
    return std::nullopt;
  }

  if (has_new_computed || num_external_computed_tokens > 0) {
    // Append the new computed blocks to the request blocks until now to avoid
    // the case where the new blocks cannot be allocated.
    coordinator->allocate_new_computed_blocks(
        request.request_id, new_computed_block_list, num_local_computed_tokens,
        num_external_computed_tokens);
  }

  KVCacheBlocksTuple new_blocks = coordinator->allocate_new_blocks(
      request.request_id, num_tokens_need_slot, num_tokens_main_model,
      num_encoder_tokens);

  // P/D: delay caching blocks if we have to recv from remote.
  if (!enable_caching || delay_cache_blocks) {
    return create_kv_cache_blocks(new_blocks);
  }

  // Commit (cache) up to total_computed_tokens + num_new_tokens, but cap at
  // request.num_tokens so only "finalized" tokens are cached (draft tokens that
  // could be rejected are excluded).
  const int num_tokens_to_cache =
      std::min(total_computed_tokens + num_new_tokens, request.NumTokens());
  coordinator->cache_blocks(request, num_tokens_to_cache);

  return create_kv_cache_blocks(new_blocks);
}

void KVCacheManager::free(const Request& request) {
  if (auto index = kv_cache_config.recurrent_prefix_snapshots) index->Release(request.request_id);
  coordinator->free(request.request_id);
}

void KVCacheManager::remove_skipped_blocks(
    const std::string& request_id, int total_computed_tokens,
    std::optional<int> num_prompt_tokens) {
  coordinator->remove_skipped_blocks(request_id, total_computed_tokens,
                                     num_prompt_tokens);
}

std::vector<KVCacheBlock*> KVCacheManager::pop_blocks_for_free(
    const Request& request) {
  if (auto index = kv_cache_config.recurrent_prefix_snapshots) index->Release(request.request_id);
  return coordinator->pop_blocks_for_free(request.request_id);
}

void KVCacheManager::evict_blocks(const std::set<int>& block_ids) {
  block_pool.evict_blocks(block_ids);
}

bool KVCacheManager::reset_prefix_cache() {
  const auto index = kv_cache_config.recurrent_prefix_snapshots;
  const auto reset_pages = [&] { return block_pool.reset_prefix_cache(); };
  if (!(index ? index->Reset(reset_pages) : reset_pages())) {
    return false;
  }
  // Upstream vllm/v1/core/kv_cache_manager.py:522-524: flag the RESET so the
  // consuming CachingMetrics window clears before folding in the next
  // observation, rather than blending pre- and post-reset hit rates.
  if (log_stats) {
    prefix_cache_stats.reset = true;
  }
  return true;
}

std::optional<PrefixCacheStats> KVCacheManager::make_prefix_cache_stats() {
  if (!log_stats) {
    return std::nullopt;
  }
  PrefixCacheStats stats = prefix_cache_stats;
  prefix_cache_stats = PrefixCacheStats();
  return stats;
}

std::vector<int> KVCacheManager::get_num_common_prefix_blocks(
    const std::string& running_request_id) {
  return coordinator->get_num_common_prefix_blocks(running_request_id);
}

std::vector<KVCacheEvent> KVCacheManager::take_events() {
  // DEFERRED: the kv_cache_event_metadata annotation is omitted (events are
  // deferred at the BlockPool); forward the pool's (empty) event list.
  return block_pool.take_events();
}

KVCacheBlocks KVCacheManager::get_blocks(const std::string& request_id) {
  return create_kv_cache_blocks(coordinator->get_blocks(request_id));
}

std::vector<std::vector<int>> KVCacheManager::get_block_ids(
    const std::string& request_id) {
  return get_blocks(request_id).get_block_ids();
}

void KVCacheManager::cache_blocks(const Request& request,
                                  int num_computed_tokens) {
  if (enable_caching) {
    coordinator->cache_blocks(request, num_computed_tokens);
  }
}

KVCacheBlocks KVCacheManager::create_kv_cache_blocks(
    const KVCacheBlocksTuple& blocks) const {
  // Only create a new KVCacheBlocks for non-empty blocks.
  for (const auto& group : blocks) {
    if (!group.empty()) return KVCacheBlocks{blocks};
  }
  return empty_kv_cache_blocks;
}

std::vector<int> KVCacheManager::take_new_block_ids() {
  std::vector<int> ids;
  for (const auto& mgr : coordinator->single_type_managers) {
    std::vector<int> group_ids = mgr->take_new_block_ids();
    ids.insert(ids.end(), group_ids.begin(), group_ids.end());
  }
  return ids;
}

void KVCacheManager::new_step_starts() { coordinator->new_step_starts(); }

}  // namespace vllm::v1
