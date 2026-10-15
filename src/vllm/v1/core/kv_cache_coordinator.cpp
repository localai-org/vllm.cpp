#include "vllm/v1/core/recurrent_prefix_snapshot.h"
// Ported from: vllm/v1/core/kv_cache_coordinator.py @ e24d1b24
// See include/vllm/v1/core/kv_cache_coordinator.h for the scope / deferred list
// and the CROSS-GROUP find_longest_cache_hit algorithm this task turns on.
#include "vllm/v1/core/kv_cache_coordinator.h"

#include <algorithm>
#include <cassert>

#include "vllm/v1/request.h"

namespace vllm::v1 {

namespace {

// Value equality of two specs (upstream frozen-dataclass __eq__), restricted to
// the ported spec kinds. Used to batch groups that share a spec.
bool spec_equal(const KVCacheSpec& a, const KVCacheSpec& b) {
  if (a.kind() != b.kind() || a.block_size != b.block_size) {
    return false;
  }
  switch (a.kind()) {
    case KVCacheSpecKind::kFullAttention: {
      const auto& fa = static_cast<const FullAttentionSpec&>(a);
      const auto& fb = static_cast<const FullAttentionSpec&>(b);
      return fa.num_kv_heads == fb.num_kv_heads &&
             fa.head_size == fb.head_size && fa.head_size_v == fb.head_size_v &&
             fa.dtype == fb.dtype && fa.kv_quant_mode == fb.kv_quant_mode &&
             fa.page_size_padded == fb.page_size_padded &&
             fa.indexes_kv_by_block_stride ==
                 fb.indexes_kv_by_block_stride &&
             fa.sliding_window == fb.sliding_window &&
             fa.attention_chunk_size == fb.attention_chunk_size &&
             fa.non_causal == fb.non_causal;
    }
    // #1974: `kMlaAttention` and `kSlidingWindowMla` used to fall into
    // `default:` and compare UNEQUAL to a structurally identical twin, so two
    // identical MLA groups never merged into one SpecGroup. Upstream cannot
    // answer that: every spec class is `@dataclass(frozen=True, kw_only=True)`
    // (`vllm/v1/kv_cache_interface.py:380-381` MLAAttentionSpec, `:610-611`
    // SlidingWindowMLASpec), so `__eq__` is generated over ALL fields — the
    // parent's plus the four DeepSeek-V4 fields — and two identical specs are
    // equal. Latent until KV-DSV4-MULTICACHE W2 (#1973) published three
    // `kMlaAttention` and four `kSlidingWindowMla` groups for DeepSeek-V4,
    // because every MLA model in the tree publishes exactly one MLA group.
    case KVCacheSpecKind::kMlaAttention: {
      const auto& ma = static_cast<const MLAAttentionSpec&>(a);
      const auto& mb = static_cast<const MLAAttentionSpec&>(b);
      return ma.num_kv_heads == mb.num_kv_heads &&
             ma.head_size == mb.head_size && ma.head_size_v == mb.head_size_v &&
             ma.dtype == mb.dtype && ma.kv_quant_mode == mb.kv_quant_mode &&
             ma.page_size_padded == mb.page_size_padded &&
             ma.indexes_kv_by_block_stride ==
                 mb.indexes_kv_by_block_stride &&
             ma.sliding_window == mb.sliding_window &&
             ma.attention_chunk_size == mb.attention_chunk_size &&
             ma.non_causal == mb.non_causal &&
             ma.cache_dtype_str == mb.cache_dtype_str &&
             ma.alignment == mb.alignment &&
             ma.compress_ratio == mb.compress_ratio &&
             ma.model_version == mb.model_version;
    }
    case KVCacheSpecKind::kSlidingWindowMla: {
      const auto& sa = static_cast<const SlidingWindowMLASpec&>(a);
      const auto& sb = static_cast<const SlidingWindowMLASpec&>(b);
      return sa.num_kv_heads == sb.num_kv_heads &&
             sa.head_size == sb.head_size && sa.head_size_v == sb.head_size_v &&
             sa.dtype == sb.dtype && sa.kv_quant_mode == sb.kv_quant_mode &&
             sa.page_size_padded == sb.page_size_padded &&
             sa.indexes_kv_by_block_stride ==
                 sb.indexes_kv_by_block_stride &&
             sa.sliding_window == sb.sliding_window &&
             sa.cache_dtype_str == sb.cache_dtype_str &&
             sa.alignment == sb.alignment &&
             sa.compress_ratio == sb.compress_ratio &&
             sa.model_version == sb.model_version;
    }
    case KVCacheSpecKind::kSlidingWindow: {
      const auto& sa = static_cast<const SlidingWindowSpec&>(a);
      const auto& sb = static_cast<const SlidingWindowSpec&>(b);
      return sa.num_kv_heads == sb.num_kv_heads &&
             sa.head_size == sb.head_size && sa.head_size_v == sb.head_size_v &&
             sa.dtype == sb.dtype && sa.kv_quant_mode == sb.kv_quant_mode &&
             sa.page_size_padded == sb.page_size_padded &&
             sa.indexes_kv_by_block_stride ==
                 sb.indexes_kv_by_block_stride &&
             sa.sliding_window == sb.sliding_window;
    }
    case KVCacheSpecKind::kChunkedLocalAttention: {
      const auto& ca = static_cast<const ChunkedLocalAttentionSpec&>(a);
      const auto& cb = static_cast<const ChunkedLocalAttentionSpec&>(b);
      return ca.num_kv_heads == cb.num_kv_heads &&
             ca.head_size == cb.head_size && ca.dtype == cb.dtype &&
             ca.kv_quant_mode == cb.kv_quant_mode &&
             ca.page_size_padded == cb.page_size_padded &&
             ca.indexes_kv_by_block_stride ==
                 cb.indexes_kv_by_block_stride &&
             ca.attention_chunk_size == cb.attention_chunk_size;
    }
    case KVCacheSpecKind::kMamba: {
      const auto& ma = static_cast<const MambaSpec&>(a);
      const auto& mb = static_cast<const MambaSpec&>(b);
      return ma.shapes == mb.shapes && ma.dtypes == mb.dtypes &&
             ma.page_size_padded == mb.page_size_padded &&
             ma.mamba_cache_mode == mb.mamba_cache_mode &&
             ma.num_speculative_blocks == mb.num_speculative_blocks;
    }
    default:
      return false;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// KVCacheCoordinator (base)
// ---------------------------------------------------------------------------

KVCacheCoordinator::KVCacheCoordinator(KVCacheConfig kv_cache_config,
                                       int max_model_len,
                                       int max_num_batched_tokens,
                                       bool use_eagle, bool enable_caching,
                                       bool enable_kv_cache_events,
                                       int dcp_world_size, int pcp_world_size,
                                       int scheduler_block_size,
                                       int hash_block_size)
    : kv_cache_config(std::move(kv_cache_config)),
      max_model_len(max_model_len),
      enable_caching(enable_caching),
      scheduler_block_size(scheduler_block_size),
      block_pool(this->kv_cache_config.num_blocks, enable_caching,
                 hash_block_size, enable_kv_cache_events),
      max_num_batched_tokens_(max_num_batched_tokens),
      dcp_world_size_(dcp_world_size),
      pcp_world_size_(pcp_world_size),
      hash_block_size_(hash_block_size) {
  // scheduler_block_size must be a multiple of hash_block_size and every group's
  // block size (the scheduling granularity, LCM of all group block sizes).
  assert(scheduler_block_size % hash_block_size == 0);
  for (const auto& g : this->kv_cache_config.kv_cache_groups) {
    assert(scheduler_block_size % g.kv_cache_spec->block_size == 0);
    (void)g;
  }

  for (std::size_t i = 0; i < this->kv_cache_config.kv_cache_groups.size();
       ++i) {
    if (this->kv_cache_config.kv_cache_groups[i].is_eagle_group) {
      eagle_group_ids.insert(static_cast<int>(i));
    }
  }
  // Shared EXL3 MTP KV uses the target group's physical page identities.
  // Draft prefill shifts input IDs by one, so the final draft KV of a hashed
  // page depends on the next token. Apply the existing last-hit-page drop
  // before choosing the joint attention/recurrent restore boundary.
  // Keep the target layer-name discriminator unchanged: is_eagle_group also
  // identifies draft-only storage during layer-name expansion.
  if (this->kv_cache_config.mtp_draft_shares_target_pages) {
    eagle_group_ids.insert(0);
  }
  // Conservatively fall back to flag all groups when no group is flagged.
  if (use_eagle && eagle_group_ids.empty()) {
    for (std::size_t i = 0; i < this->kv_cache_config.kv_cache_groups.size();
         ++i) {
      eagle_group_ids.insert(static_cast<int>(i));
    }
  }

  for (std::size_t i = 0; i < this->kv_cache_config.kv_cache_groups.size();
       ++i) {
    single_type_managers.push_back(get_manager_for_kv_cache_spec(
        this->kv_cache_config.kv_cache_groups[i].kv_cache_spec,
        max_num_batched_tokens_, max_model_len, block_pool, enable_caching,
        static_cast<int>(i), scheduler_block_size));
    if (auto* mamba = dynamic_cast<MambaManager*>(single_type_managers.back().get()))
      mamba->prefix_snapshots = this->kv_cache_config.recurrent_prefix_snapshots;
  }

  // retention_interval stays nullopt (dense caching); env read deferred.
}

int KVCacheCoordinator::get_num_blocks_to_allocate(
    const std::string& request_id, int num_tokens,
    const KVCacheBlocksTuple& new_computed_blocks, int /*num_encoder_tokens*/,
    int total_computed_tokens, int num_tokens_main_model,
    bool apply_admission_cap) {
  int num_blocks_to_allocate = 0;
  for (std::size_t i = 0; i < single_type_managers.size(); ++i) {
    // CrossAttentionManager encoder branch is DEFERRED (see header).
    num_blocks_to_allocate +=
        single_type_managers[i]->get_num_blocks_to_allocate(
            request_id, num_tokens, new_computed_blocks[i],
            total_computed_tokens, num_tokens_main_model, apply_admission_cap);
  }
  return num_blocks_to_allocate;
}

void KVCacheCoordinator::allocate_new_computed_blocks(
    const std::string& request_id,
    const KVCacheBlocksTuple& new_computed_blocks,
    int num_local_computed_tokens, int num_external_computed_tokens) {
  // A running request is already tracked in num_cached_block and won't have new
  // prefix-cache hits, so this is a no-op for it.
  bool any_running = false;
  for (const auto& manager : single_type_managers) {
    if (manager->num_cached_block.count(request_id) != 0) {
      any_running = true;
      break;
    }
  }
  if (any_running) {
    for (const auto& blocks : new_computed_blocks) {
      assert(blocks.empty());
      (void)blocks;
    }
    return;
  }

  // Two-phase allocation (issue #33775): touch every group's local hit blocks
  // first, then allocate external blocks for every group.
  for (std::size_t i = 0; i < single_type_managers.size(); ++i) {
    single_type_managers[i]->add_local_computed_blocks(
        request_id, new_computed_blocks[i], num_local_computed_tokens,
        num_external_computed_tokens);
  }
  if (num_external_computed_tokens > 0) {
    for (const auto& manager : single_type_managers) {
      manager->allocate_external_computed_blocks(
          request_id, num_local_computed_tokens, num_external_computed_tokens);
    }
  }
}

KVCacheBlocksTuple KVCacheCoordinator::allocate_new_blocks(
    const std::string& request_id, int num_tokens, int num_tokens_main_model,
    int /*num_encoder_tokens*/) {
  KVCacheBlocksTuple result;
  result.reserve(single_type_managers.size());
  for (const auto& manager : single_type_managers) {
    // CrossAttentionManager encoder branch is DEFERRED (see header).
    result.push_back(manager->allocate_new_blocks(request_id, num_tokens,
                                                  num_tokens_main_model));
  }
  return result;
}

void KVCacheCoordinator::cache_blocks(const Request& request,
                                      int num_computed_tokens) {
  for (const auto& manager : single_type_managers) {
    manager->cache_blocks(request, num_computed_tokens, retention_interval);
  }
}

void KVCacheCoordinator::free(const std::string& request_id) {
  for (const auto& manager : single_type_managers) {
    manager->free(request_id);
  }
}

std::vector<KVCacheBlock*> KVCacheCoordinator::pop_blocks_for_free(
    const std::string& request_id) {
  std::vector<KVCacheBlock*> blocks;
  for (const auto& manager : single_type_managers) {
    std::vector<KVCacheBlock*> group_blocks =
        manager->pop_blocks_for_free(request_id);
    blocks.insert(blocks.end(), group_blocks.begin(), group_blocks.end());
  }
  return blocks;
}

std::vector<int> KVCacheCoordinator::get_num_common_prefix_blocks(
    const std::string& running_request_id) {
  std::vector<int> result;
  result.reserve(single_type_managers.size());
  for (const auto& manager : single_type_managers) {
    result.push_back(manager->get_num_common_prefix_blocks(running_request_id));
  }
  return result;
}

void KVCacheCoordinator::remove_skipped_blocks(
    const std::string& request_id, int total_computed_tokens,
    std::optional<int> num_prompt_tokens) {
  for (const auto& manager : single_type_managers) {
    manager->remove_skipped_blocks(request_id, total_computed_tokens,
                                   num_prompt_tokens);
  }
}

KVCacheBlocksTuple KVCacheCoordinator::get_blocks(
    const std::string& request_id) {
  KVCacheBlocksTuple result;
  result.reserve(single_type_managers.size());
  for (const auto& manager : single_type_managers) {
    auto it = manager->req_to_blocks.find(request_id);
    if (it != manager->req_to_blocks.end()) {
      result.push_back(it->second);
    } else {
      result.emplace_back();
    }
  }
  return result;
}

void KVCacheCoordinator::new_step_starts() {
  for (const auto& manager : single_type_managers) {
    manager->new_step_starts();
  }
}

// ---------------------------------------------------------------------------
// KVCacheCoordinatorNoPrefixCache
// ---------------------------------------------------------------------------

KVCacheCoordinatorNoPrefixCache::KVCacheCoordinatorNoPrefixCache(
    KVCacheConfig kv_cache_config, int max_model_len,
    int max_num_batched_tokens, bool use_eagle,
    bool enable_kv_cache_events, int dcp_world_size, int pcp_world_size,
    int scheduler_block_size, int hash_block_size)
    : KVCacheCoordinator(std::move(kv_cache_config), max_model_len,
                         max_num_batched_tokens, use_eagle,
                         /*enable_caching=*/false, enable_kv_cache_events,
                         dcp_world_size, pcp_world_size, scheduler_block_size,
                         hash_block_size),
      num_single_type_managers_(single_type_managers.size()) {}

std::vector<int>
KVCacheCoordinatorNoPrefixCache::get_num_common_prefix_blocks(
    const std::string& /*running_request_id*/) {
  return std::vector<int>(num_single_type_managers_, 0);
}

std::pair<KVCacheBlocksTuple, int>
KVCacheCoordinatorNoPrefixCache::find_longest_cache_hit(
    const std::vector<BlockHash>& /*block_hashes*/,
    int /*max_cache_hit_length*/) {
  return {KVCacheBlocksTuple(num_single_type_managers_), 0};
}

// ---------------------------------------------------------------------------
// UnitaryKVCacheCoordinator
// ---------------------------------------------------------------------------

UnitaryKVCacheCoordinator::UnitaryKVCacheCoordinator(
    KVCacheConfig kv_cache_config, int max_model_len,
    int max_num_batched_tokens, bool use_eagle, bool enable_caching,
    bool enable_kv_cache_events, int dcp_world_size, int pcp_world_size,
    int scheduler_block_size, int hash_block_size)
    : KVCacheCoordinator(std::move(kv_cache_config), max_model_len,
                         max_num_batched_tokens, use_eagle, enable_caching,
                         enable_kv_cache_events, dcp_world_size, pcp_world_size,
                         scheduler_block_size, hash_block_size) {
  kv_cache_spec_ = this->kv_cache_config.kv_cache_groups[0].kv_cache_spec.get();
  block_size_ = kv_cache_spec_->block_size;
  if (dcp_world_size > 1) {
    block_size_ *= dcp_world_size;
  }
  if (pcp_world_size > 1) {
    block_size_ *= pcp_world_size;
  }
  // For mamba-only models with caching disabled, block_size == max_model_len and
  // hash validation is skipped; here caching is enabled (see factory).
  assert(!enable_caching || hash_block_size == block_size_);
  assert(this->kv_cache_config.kv_cache_groups.size() == 1);
  single_type_managers[0]->use_eagle = eagle_group_ids.count(0) != 0;
}

std::pair<KVCacheBlocksTuple, int>
UnitaryKVCacheCoordinator::find_longest_cache_hit(
    const std::vector<BlockHash>& block_hashes, int max_cache_hit_length) {
  KVCacheBlocksTuple hit_blocks =
      single_type_managers[0]->find_longest_cache_hit(
          block_hashes, max_cache_hit_length, /*kv_cache_group_ids=*/{0},
          block_pool, *kv_cache_spec_,
          /*drop_eagle_block=*/eagle_group_ids.count(0) != 0,
          /*alignment_tokens=*/block_size_, dcp_world_size_, pcp_world_size_);
  int hit_length = static_cast<int>(hit_blocks[0].size()) * block_size_;
  return {std::move(hit_blocks), hit_length};
}

// ---------------------------------------------------------------------------
// HybridKVCacheCoordinator
// ---------------------------------------------------------------------------

HybridKVCacheCoordinator::HybridKVCacheCoordinator(
    KVCacheConfig kv_cache_config, int max_model_len,
    int max_num_batched_tokens, bool use_eagle, bool enable_caching,
    bool enable_kv_cache_events, int dcp_world_size, int pcp_world_size,
    int scheduler_block_size, int hash_block_size)
    : KVCacheCoordinator(std::move(kv_cache_config), max_model_len,
                         max_num_batched_tokens, use_eagle, enable_caching,
                         enable_kv_cache_events, dcp_world_size, pcp_world_size,
                         scheduler_block_size, hash_block_size) {
  for (const auto& g : this->kv_cache_config.kv_cache_groups) {
    assert(g.kv_cache_spec->block_size % hash_block_size == 0 &&
           "block_size must be divisible by hash_block_size");
    // The BlockHashListWithBlockSize (differing block sizes) path is DEFERRED
    // (see header); T0 gate models use a uniform block size.
    assert(g.kv_cache_spec->block_size == hash_block_size &&
           "differing group/hash block sizes are DEFERRED (M1.3 Task 3)");
    (void)g;
  }
  assert(dcp_world_size == 1 && "DCP not support hybrid attn now.");
  assert(pcp_world_size == 1 && "PCP not support hybrid attn now.");
  verify_and_split_kv_cache_groups();
}

void HybridKVCacheCoordinator::verify_and_split_kv_cache_groups() {
  attention_groups.clear();
  for (std::size_t i = 0; i < kv_cache_config.kv_cache_groups.size(); ++i) {
    KVCacheSpec* spec = kv_cache_config.kv_cache_groups[i].kv_cache_spec.get();
    bool use_eagle = eagle_group_ids.count(static_cast<int>(i)) != 0;

    // Try to find an existing group with the same spec.
    bool merged = false;
    for (auto& group : attention_groups) {
      if (spec_equal(*group.spec, *spec)) {
        group.group_ids.push_back(static_cast<int>(i));
        if (use_eagle && !group.use_eagle) {
          group.use_eagle = true;
        }
        merged = true;
        break;
      }
    }
    if (!merged) {
      attention_groups.push_back(SpecGroup{
          spec, {static_cast<int>(i)}, single_type_managers[i].get(),
          use_eagle});
    }
  }

  assert(attention_groups.size() > 1 &&
         "HybridKVCacheCoordinator requires at least two attention groups.");

  // Put full attention first: its efficient left-to-right scan provides a
  // tighter initial bound, reducing work for subsequent groups. std::stable_sort
  // preserves relative order (matches Python's stable list.sort).
  std::stable_sort(attention_groups.begin(), attention_groups.end(),
                   [](const SpecGroup& a, const SpecGroup& b) {
                     bool a_full =
                         a.spec->kind() == KVCacheSpecKind::kFullAttention;
                     bool b_full =
                         b.spec->kind() == KVCacheSpecKind::kFullAttention;
                     // key = not isinstance(FullAttentionSpec): full attn => 0.
                     return static_cast<int>(!a_full) <
                            static_cast<int>(!b_full);
                   });

  // Propagate the eagle bit to each manager.
  for (const auto& group : attention_groups) {
    if (group.use_eagle) {
      for (int gid : group.group_ids) {
        single_type_managers[static_cast<std::size_t>(gid)]->use_eagle = true;
      }
    }
  }
}

void HybridKVCacheCoordinator::cache_blocks(const Request& request,
                                            int num_computed_tokens) {
  // Cache hits are always a multiple of scheduler_block_size tokens (see
  // find_longest_cache_hit); align down before caching per group.
  int aligned_num_computed_tokens =
      num_computed_tokens / scheduler_block_size * scheduler_block_size;
  for (const auto& manager : single_type_managers) {
    int num_tokens_to_cache = aligned_num_computed_tokens;
    // EAGLE groups match one block past each aligned boundary and drop it; make
    // that lookahead block eligible to be cached.
    if (manager->use_eagle && aligned_num_computed_tokens > 0) {
      num_tokens_to_cache = std::min(
          num_computed_tokens, aligned_num_computed_tokens + manager->block_size);
    }
    manager->cache_blocks(request, num_tokens_to_cache, retention_interval);
  }
}

std::pair<KVCacheBlocksTuple, int>
HybridKVCacheCoordinator::find_longest_cache_hit(
    const std::vector<BlockHash>& block_hashes, int max_cache_hit_length) {
  // _get_block_hashes: with a uniform block size (the differing-block-size /
  // BlockHashListWithBlockSize path is DEFERRED), the hashes pass through.

  int num_groups = static_cast<int>(kv_cache_config.kv_cache_groups.size());
  int hit_length = max_cache_hit_length;
  int longest_hit_length = 0;
  // Per-group hit blocks; std::nullopt distinguishes "not yet looked up" (the
  // upstream `None` sentinel that gates the full-attention fast path).
  std::vector<std::optional<std::vector<KVCacheBlock*>>> hit_blocks_by_group(
      static_cast<std::size_t>(num_groups), std::nullopt);

  // Simple hybrid (1 full attn + 1 other): one iteration suffices. Full attn is
  // always first if it exists.
  bool is_simple_hybrid =
      attention_groups.size() == 2 &&
      attention_groups[0].spec->kind() == KVCacheSpecKind::kFullAttention;

  // Attention-group indices whose EAGLE drop is verified at the current
  // curr_hit_length (each applies its drop at most once per candidate length).
  std::set<int> eagle_verified;

  while (true) {
    int curr_hit_length = hit_length;

    for (std::size_t idx = 0; idx < attention_groups.size(); ++idx) {
      const SpecGroup& group = attention_groups[idx];
      KVCacheSpec* spec = group.spec;
      const std::vector<int>& group_ids = group.group_ids;
      bool use_eagle = group.use_eagle;

      const std::optional<std::vector<KVCacheBlock*>>& cached_blocks =
          hit_blocks_by_group[static_cast<std::size_t>(group_ids[0])];
      if (spec->kind() == KVCacheSpecKind::kFullAttention &&
          cached_blocks.has_value()) {
        // Full attention is downward-closed: look up once, then just trim to the
        // (reduced) current hit length on later iterations.
        curr_hit_length =
            curr_hit_length / spec->block_size * spec->block_size;
        continue;
      }

      bool drop_eagle_block =
          use_eagle && eagle_verified.count(static_cast<int>(idx)) == 0;

      int max_length = curr_hit_length;
      if (drop_eagle_block) {
        // Eagle needs to match one more block and then pop the last.
        max_length =
            std::min(curr_hit_length + spec->block_size, max_cache_hit_length);
      }
      std::vector<std::vector<KVCacheBlock*>> hit_blocks =
          group.representative->find_longest_cache_hit(
              block_hashes, max_length, group_ids, block_pool, *spec,
              drop_eagle_block, /*alignment_tokens=*/scheduler_block_size);
      int new_hit_length =
          static_cast<int>(hit_blocks[0].size()) * spec->block_size;
      if (drop_eagle_block) {
        eagle_verified.insert(static_cast<int>(idx));
      } else if (new_hit_length < curr_hit_length) {
        // length shrunk; invalidate previous eagle verifications.
        eagle_verified.clear();
      }
      curr_hit_length = new_hit_length;
      for (std::size_t g = 0; g < group_ids.size(); ++g) {
        hit_blocks_by_group[static_cast<std::size_t>(group_ids[g])] =
            hit_blocks[g];
      }

      longest_hit_length = std::max(longest_hit_length, curr_hit_length);
    }

    if (curr_hit_length >= hit_length) {
      break;
    }
    hit_length = curr_hit_length;
    if (is_simple_hybrid) {
      break;
    }
  }

  // Truncate full attention blocks to the final hit_length (if present).
  const SpecGroup& first_group = attention_groups[0];
  if (first_group.spec->kind() == KVCacheSpecKind::kFullAttention) {
    int num_blocks = hit_length / first_group.spec->block_size;
    for (int group_id : first_group.group_ids) {
      auto& blks = hit_blocks_by_group[static_cast<std::size_t>(group_id)];
      if (blks.has_value() &&
          static_cast<int>(blks->size()) > num_blocks) {
        blks->resize(static_cast<std::size_t>(num_blocks));
      }
    }
  }

  // Uncached shared prefix detection: if any group cached a longer prefix than
  // the coordinated hit, it is an uncached common prefix across requests.
  num_uncached_common_prefix_tokens = longest_hit_length - hit_length;

  KVCacheBlocksTuple result;
  result.reserve(static_cast<std::size_t>(num_groups));
  for (auto& blocks : hit_blocks_by_group) {
    if (blocks.has_value()) {
      result.push_back(std::move(*blocks));
    } else {
      result.emplace_back();
    }
  }
  return {std::move(result), hit_length};
}

// ---------------------------------------------------------------------------
// get_kv_cache_coordinator (factory)
// ---------------------------------------------------------------------------

std::unique_ptr<KVCacheCoordinator> get_kv_cache_coordinator(
    KVCacheConfig kv_cache_config, int max_model_len,
    int max_num_batched_tokens, bool use_eagle, bool enable_caching,
    bool enable_kv_cache_events, int dcp_world_size, int pcp_world_size,
    int scheduler_block_size, int hash_block_size) {
  if (!enable_caching) {
    return std::make_unique<KVCacheCoordinatorNoPrefixCache>(
        std::move(kv_cache_config), max_model_len, max_num_batched_tokens,
        use_eagle, enable_kv_cache_events, dcp_world_size, pcp_world_size,
        scheduler_block_size, hash_block_size);
  }
  std::size_t num_groups = kv_cache_config.kv_cache_groups.size();
  if (num_groups == 1) {
    return std::make_unique<UnitaryKVCacheCoordinator>(
        std::move(kv_cache_config), max_model_len, max_num_batched_tokens,
        use_eagle, enable_caching, enable_kv_cache_events, dcp_world_size,
        pcp_world_size, scheduler_block_size, hash_block_size);
  }
  return std::make_unique<HybridKVCacheCoordinator>(
      std::move(kv_cache_config), max_model_len, max_num_batched_tokens,
      use_eagle, enable_caching, enable_kv_cache_events, dcp_world_size,
      pcp_world_size, scheduler_block_size, hash_block_size);
}

}  // namespace vllm::v1
