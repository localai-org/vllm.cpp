// Ported from: vllm/v1/core/single_type_kv_cache_manager.py @ e24d1b24
//
// Scope (M1.3 Task 2): the per-group KV-cache managers — the `SingleTypeKV
// CacheManager` abstract base plus `FullAttentionManager` (standard paged K+V
// blocks), `SlidingWindowManager` and `ChunkedLocalAttentionManager`
// (recycling-aware local K+V blocks), and `MambaManager` (the GDN/mamba
// single-recurrent-state manager). Each manager
// owns the per-request block table (`req_to_blocks`) for ONE kv cache group and
// calls the M1.2 BlockPool to allocate / cache / free physical blocks.
// Behavioral only: no CUDA, no model.
//
// THE FullAttention-vs-Mamba DIFFERENCE (the load-bearing bit of this task):
//   * FullAttentionManager.find_longest_cache_hit does a LEFT-TO-RIGHT
//     multi-block prefix walk: it appends every consecutively-cached block from
//     the front and stops at the first miss (a real reusable prefix of length N
//     -> N blocks). get_num_common_prefix_blocks counts leading blocks whose
//     ref_cnt == #requests (cascade attention).
//   * MambaManager keeps only ONE recurrent state. Its find_longest_cache_hit
//     searches RIGHT-TO-LEFT and early-stops on the FIRST (rightmost) cached
//     block, returning a list padded with `null_block` up to that index and the
//     single state block at the end (e.g. [NULL, NULL, state]). So mamba does
//     NOT do FullAttention's contiguous multi-block prefix reuse: it reuses at
//     most the single latest state block. get_num_skipped_tokens returns
//     num_computed - 1 (only the last token's state is needed), so
//     remove_skipped_blocks frees everything but the tail state; and
//     get_num_common_prefix_blocks is always 0 (no cascade for mamba).
//   MambaManager also carries the `mamba_cache_mode` handling: mode "none"
//   defers to the base block accounting (plus num_speculative_blocks padding);
//   mode "align" keeps a single running state at a fixed tail offset, tracks
//   `last_state_block_idx` / `_allocated_block_reqs`, and reuses/frees state
//   blocks across steps (the alignment-preserving allocation the SSM kernels
//   require). Both modes are ported exactly.
//
// DEVIATIONS, recorded:
//   - Upstream `find_longest_cache_hit` / `reachable_block_mask` are Python
//     @classmethods. C++ has no abstract/virtual static methods, so they are
//     ported as VIRTUAL INSTANCE methods (their bodies still only read the
//     passed-in args / this manager's own immutable fields, never per-request
//     state, so calling them on any manager instance of the right type is
//     equivalent to upstream's classmethod call). Task 3's coordinator invokes
//     them through the per-group manager instance it already holds.
//   - `BlockHashList` (upstream `list[BlockHash] | BlockHashListWithBlockSize`)
//     is taken as `const std::vector<BlockHash>&` — the common
//     block_size == hash_block_size case (the align/multi-block-size
//     BlockHashListWithBlockSize path is DEFERRED with BlockPool, M1.2).
//   - `kv_cache_spec` is held as `std::shared_ptr<KVCacheSpec>` (shared with the
//     owning KVCacheGroupSpec) and TREATED AS CONST — never mutated — so two
//     groups sharing a spec cannot leak state into each other (M1.3 Task 1
//     carry-forward).
//   - `block_pool` is held by reference (the pool is non-movable and outlives
//     the managers); managers are therefore non-copyable.
//   - `retention_interval` (int | None) is carried as std::optional<int>; the
//     dense default (nullopt) is the only path the gate models exercise.
//
// DEFERRED (marked; the gate models never exercise these — later tasks add them
// without reshaping the base):
//   - RSWAManager / CrossAttentionManager / SinkFullAttentionManager (T1/T2) —
//     omitted. Their
//     spec types are already deferred in kv_cache_interface.h.
//   - Out-of-tree platform registration callbacks are not wired to a platform
//     object yet. The core registry, built-in registration and inherited-spec
//     lookup are ported in kv_cache_spec_registry.*.
//   - dcp_world_size / pcp_world_size (decode/prefill context parallelism): the
//     find_longest_cache_hit params are carried for signature fidelity and
//     asserted == 1 (mamba) / honored in the block-size scaling (full attn), but
//     the > 1 paths are not otherwise exercised.
//   - add_local_computed_blocks' external-KV-connector siblings beyond the T0
//     needs are ported for fidelity but only the local path is tested here.
#ifndef VLLM_V1_CORE_SINGLE_TYPE_KV_CACHE_MANAGER_H_
#define VLLM_V1_CORE_SINGLE_TYPE_KV_CACHE_MANAGER_H_

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vllm/v1/core/block_pool.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/kv_cache_interface.h"

namespace vllm::v1 {

struct Request;

// Abstract base class for a manager that handles the kv cache management logic
// of one specific type of attention layer. (Upstream SingleTypeKVCacheManager.)
class SingleTypeKVCacheManager {
 public:
  SingleTypeKVCacheManager(std::shared_ptr<KVCacheSpec> kv_cache_spec,
                           BlockPool& block_pool, bool enable_caching,
                           int kv_cache_group_id, int scheduler_block_size,
                           std::optional<int> max_admission_blocks_per_request =
                               std::nullopt);
  virtual ~SingleTypeKVCacheManager() = default;

  // Non-copyable (holds a BlockPool reference).
  SingleTypeKVCacheManager(const SingleTypeKVCacheManager&) = delete;
  SingleTypeKVCacheManager& operator=(const SingleTypeKVCacheManager&) = delete;

  // sum(blk.ref_cnt == 0 && !blk.is_null for blk in blocks).
  static int _get_num_evictable_blocks(
      const std::vector<KVCacheBlock*>& blocks);

  // Number of blocks that must be allocated for the request. See upstream for
  // the num_skipped / num_evictable accounting.
  virtual int get_num_blocks_to_allocate(
      const std::string& request_id, int num_tokens,
      const std::vector<KVCacheBlock*>& new_computed_blocks,
      int total_computed_tokens, int num_tokens_main_model,
      bool apply_admission_cap = false);

  // Add the locally cached (prefix-hit) blocks to the request: touch them, pad
  // skipped blocks with nulls, then append the remaining computed blocks.
  virtual void add_local_computed_blocks(
      const std::string& request_id,
      const std::vector<KVCacheBlock*>& new_computed_blocks,
      int num_local_computed_tokens, int num_external_computed_tokens);

  // Allocate new blocks for external (KV-connector) computed tokens.
  virtual void allocate_external_computed_blocks(const std::string& request_id,
                                                 int num_local_computed_tokens,
                                                 int num_external_computed_tokens);

  // Allocate new blocks so the request has at least num_tokens slots.
  virtual std::vector<KVCacheBlock*> allocate_new_blocks(
      const std::string& request_id, int num_tokens, int num_tokens_main_model);

  // Drain and return block IDs allocated since the last call.
  std::vector<int> take_new_block_ids();

  // Cache the request's full blocks for prefix caching.
  virtual void cache_blocks(const Request& request, int num_tokens,
                            std::optional<int> retention_interval = std::nullopt);

  // Per-block mask for cache_full_blocks. nullopt => cache every (non-null)
  // block (the full-attention default). (Upstream classmethod; here a virtual
  // instance method reading this->block_size / this->scheduler_block_size.)
  virtual std::optional<std::vector<bool>> reachable_block_mask(
      int start_block, int end_block, std::optional<int> alignment_tokens,
      std::optional<int> retention_interval = std::nullopt,
      std::optional<int> num_prompt_tokens = std::nullopt);

  // Pop the request's bookkeeping and return its blocks (NOT yet returned to the
  // pool). Caller frees them in reverse order.
  virtual std::vector<KVCacheBlock*> pop_blocks_for_free(
      const std::string& request_id);

  // Free the request's blocks (reverse order so tail blocks are freed first).
  void free(const std::string& request_id);

  // Number of common prefix blocks for all requests with allocated KV cache.
  virtual int get_num_common_prefix_blocks(
      const std::string& running_request_id) = 0;

  // Longest cache-hit prefix (per kv cache group). Customized per attention
  // type (see the file header for the FullAttention vs Mamba difference).
  virtual std::vector<std::vector<KVCacheBlock*>> find_longest_cache_hit(
      const std::vector<BlockHash>& block_hashes, int max_length,
      const std::vector<int>& kv_cache_group_ids, BlockPool& block_pool,
      const KVCacheSpec& kv_cache_spec, bool drop_eagle_block,
      int alignment_tokens, int dcp_world_size = 1,
      int pcp_world_size = 1) = 0;

  // Free blocks in [first_block, last_block) and replace with null_block.
  void _remove_blocks_in_range(const std::string& request_id, int first_block,
                               int last_block);

  // Remove and free the blocks no longer needed for attention (replaced by
  // null_block). Depends on get_num_skipped_tokens.
  virtual void remove_skipped_blocks(
      const std::string& request_id, int total_computed_tokens,
      std::optional<int> num_prompt_tokens = std::nullopt);

  // Number of tokens skipped for attention (0 by default — full attention).
  virtual int get_num_skipped_tokens(int num_computed_tokens);

  // Per-step reset hook (no-op by default).
  virtual void new_step_starts();

  // --- Public state (mirrors upstream's accessible attributes). --------------

  int scheduler_block_size;
  // Block size for actual allocation (dcp/pcp scaling deferred).
  int block_size;
  // Shared with the owning KVCacheGroupSpec; treated as CONST (never mutated).
  std::shared_ptr<KVCacheSpec> kv_cache_spec;
  BlockPool& block_pool;
  bool enable_caching;
  // Recycling-aware reservation cap. nullopt for managers that retain their
  // full history (full attention and mamba).
  std::optional<int> _max_admission_blocks_per_request;
  std::vector<int> new_block_ids;

  // request_id -> the blocks allocated for that request (its block table).
  std::unordered_map<std::string, std::vector<KVCacheBlock*>> req_to_blocks;
  // request_id -> number of cached blocks (RUNNING requests only).
  std::unordered_map<std::string, int> num_cached_block;

  int kv_cache_group_id;
  KVCacheBlock* _null_block;
  bool use_eagle = false;
};

// The standard paged manager: full-attention K+V blocks. (Upstream
// FullAttentionManager.)
class FullAttentionManager : public SingleTypeKVCacheManager {
 public:
  using SingleTypeKVCacheManager::SingleTypeKVCacheManager;

  // Left-to-right multi-block prefix walk: append every consecutively-cached
  // block from the front, stop at the first miss.
  std::vector<std::vector<KVCacheBlock*>> find_longest_cache_hit(
      const std::vector<BlockHash>& block_hashes, int max_length,
      const std::vector<int>& kv_cache_group_ids, BlockPool& block_pool,
      const KVCacheSpec& kv_cache_spec, bool drop_eagle_block,
      int alignment_tokens, int dcp_world_size = 1,
      int pcp_world_size = 1) override;

  // Count leading blocks whose ref_cnt == #requests (cascade attention).
  int get_num_common_prefix_blocks(
      const std::string& running_request_id) override;
};

// Sliding-window manager. It retains only the live window's physical blocks,
// finds reusable cache tails right-to-left, and never participates in cascade
// attention. (Upstream SlidingWindowManager.)
class SlidingWindowManager : public SingleTypeKVCacheManager {
 public:
  SlidingWindowManager(
      std::shared_ptr<KVCacheSpec> kv_cache_spec, BlockPool& block_pool,
      bool enable_caching, int kv_cache_group_id, int scheduler_block_size,
      std::optional<int> max_admission_blocks_per_request = std::nullopt);

  static int _contiguous_blocks_for_hit(int window_size, int block_size,
                                        bool use_eagle);

  std::vector<std::vector<KVCacheBlock*>> find_longest_cache_hit(
      const std::vector<BlockHash>& block_hashes, int max_length,
      const std::vector<int>& kv_cache_group_ids, BlockPool& block_pool,
      const KVCacheSpec& kv_cache_spec, bool drop_eagle_block,
      int alignment_tokens, int dcp_world_size = 1,
      int pcp_world_size = 1) override;

  std::optional<std::vector<bool>> reachable_block_mask(
      int start_block, int end_block, std::optional<int> alignment_tokens,
      std::optional<int> retention_interval = std::nullopt,
      std::optional<int> num_prompt_tokens = std::nullopt) override;

  int get_num_skipped_tokens(int num_computed_tokens) override;

  // Always 0: cascade attention is not supported by sliding-window layers.
  int get_num_common_prefix_blocks(
      const std::string& running_request_id) override;

  int sliding_window;
};

// Fixed-chunk local-attention manager. Prefix blocks before the current chunk
// are represented by nulls without cache lookup; only the contiguous suffix in
// the current chunk must hit. Physical blocks from older chunks are recycled,
// and cascade attention is disabled. (Upstream ChunkedLocalAttentionManager.)
class ChunkedLocalAttentionManager : public SingleTypeKVCacheManager {
 public:
  ChunkedLocalAttentionManager(
      std::shared_ptr<KVCacheSpec> kv_cache_spec, BlockPool& block_pool,
      bool enable_caching, int kv_cache_group_id, int scheduler_block_size,
      std::optional<int> max_admission_blocks_per_request = std::nullopt);

  std::vector<std::vector<KVCacheBlock*>> find_longest_cache_hit(
      const std::vector<BlockHash>& block_hashes, int max_length,
      const std::vector<int>& kv_cache_group_ids, BlockPool& block_pool,
      const KVCacheSpec& kv_cache_spec, bool drop_eagle_block,
      int alignment_tokens, int dcp_world_size = 1,
      int pcp_world_size = 1) override;

  int get_num_skipped_tokens(int num_computed_tokens) override;

  // Always 0: cascade attention is not supported by chunked-local layers.
  int get_num_common_prefix_blocks(
      const std::string& running_request_id) override;

  int attention_chunk_size;
};

// The GDN/mamba single-recurrent-state manager. (Upstream MambaManager.)
class MambaManager : public SingleTypeKVCacheManager {
 public:
  MambaManager(std::shared_ptr<KVCacheSpec> kv_cache_spec, BlockPool& block_pool,
               bool enable_caching, int kv_cache_group_id,
               int scheduler_block_size);

  // Right-to-left search; keep ONLY the single rightmost cached state block,
  // padded with nulls up to its index. Early-stops on the first match.
  std::vector<std::vector<KVCacheBlock*>> find_longest_cache_hit(
      const std::vector<BlockHash>& block_hashes, int max_length,
      const std::vector<int>& kv_cache_group_ids, BlockPool& block_pool,
      const KVCacheSpec& kv_cache_spec, bool drop_eagle_block,
      int alignment_tokens, int dcp_world_size = 1,
      int pcp_world_size = 1) override;

  std::optional<std::vector<bool>> reachable_block_mask(
      int start_block, int end_block, std::optional<int> alignment_tokens,
      std::optional<int> retention_interval = std::nullopt,
      std::optional<int> num_prompt_tokens = std::nullopt) override;

  void remove_skipped_blocks(
      const std::string& request_id, int num_computed_tokens,
      std::optional<int> num_prompt_tokens = std::nullopt) override;

  // Always 0: cascade attention is not supported by mamba.
  int get_num_common_prefix_blocks(
      const std::string& running_request_id) override;

  int get_num_blocks_to_allocate(
      const std::string& request_id, int num_tokens,
      const std::vector<KVCacheBlock*>& new_computed_blocks,
      int total_computed_tokens, int num_tokens_main_model,
      bool apply_admission_cap = false) override;

  std::vector<KVCacheBlock*> allocate_new_blocks(
      const std::string& request_id, int num_tokens,
      int num_tokens_main_model) override;

  std::vector<KVCacheBlock*> pop_blocks_for_free(
      const std::string& request_id) override;

  // num_computed_tokens - 1 (only the last token's state is needed).
  int get_num_skipped_tokens(int num_computed_tokens) override;

  void cache_blocks(const Request& request, int num_tokens,
                    std::optional<int> retention_interval = std::nullopt) override;

  void new_step_starts() override;

  std::shared_ptr<RecurrentPrefixSnapshotIndex> prefix_snapshots;

  // --- Mamba-specific state (mirrors upstream). ------------------------------
  std::unordered_set<BlockHashWithGroupId> cached_blocks_this_step;
  std::string mamba_cache_mode;
  int num_speculative_blocks;
  // Only used in "align" mode:
  std::unordered_map<std::string, int> last_state_block_idx;
  std::unordered_set<std::string> _allocated_block_reqs;
};

// Registry-backed manager construction (upstream get_manager_for_kv_cache_spec).
std::unique_ptr<SingleTypeKVCacheManager> get_manager_for_kv_cache_spec(
    const std::shared_ptr<KVCacheSpec>& kv_cache_spec,
    int max_num_batched_tokens, int max_model_len, BlockPool& block_pool,
    bool enable_caching, int kv_cache_group_id, int scheduler_block_size);

}  // namespace vllm::v1

#endif  // VLLM_V1_CORE_SINGLE_TYPE_KV_CACHE_MANAGER_H_
