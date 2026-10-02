// Ported from: vllm/v1/worker/gpu/model_runner.py @ e24d1b24
// (initialize_kv_cache / execute_model / sample_tokens / sample /
// postprocess_sampled — the T0 slice) + the decode-first reorder from
// vllm/v1/attention/backends/utils.py::reorder_batch_to_split_decodes_and_prefills.
// See include/vllm/v1/worker/gpu/runner.h for scope, the V1-algorithm / MRV2-
// contract composition, the four-way ordering contract, and the deferred paths.
#include "vllm/v1/worker/gpu/runner.h"

#include "vllm/multimodal/utils.h"  // GetMmFeaturesInWindow (P2, #2379)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"  // SPEC-MTP I5d-pre: Qwen3_5MTPModel complete type for the owned draft member
#include "vllm/model_executor/models/qwen3_dflash_internal.h"  // W13 (#2112) the draft-lane route counters the readout reports
#include "vllm/platforms/interface.h"  // GetPlatform(device.type) per-tensor memory-model seam
#include "vllm/v1/attention/backend.h"  // AttentionBackend / get_kv_cache_shape (M3)
#include "vllm/v1/attention/registry.h"  // SelectAttentionBackendName / MakeAttentionBackend (M3)
#include "vllm/v1/kv_cache_dtype.h"  // ResolveKvCacheDType (VT_KV_CACHE_F32 A/B)
#include "vllm/v1/kv_offload/lmcache/lmcache_connector.h"  // KV-EXTERNAL-CACHE worker store/load
#include "vllm/v1/sample/ops/bad_words.h"  // apply_allowed_token_ids (-inf mask)
#include "vllm/v1/worker/gpu/async_runner_flag.h"  // VT_ASYNC_RUNNER predicate
#include "vllm/v1/worker/gpu/cudagraph_dispatch.h"  // W6 (#1374) the graph-eligibility predicate
#include "vllm/v1/spec_decode/rejection_sampler.h"  // SPEC-REJECTION I3 verify half
#include "vllm/v1/worker/gpu/spec_decode/mtp/speculator.h"  // SPEC-MTP I5d MtpProposePrefill
#include "vllm/v1/worker/gpu/spec_decode/dflash/speculator.h"  // SPEC-DFLASH D5 SampleDflashBlockDrafts
#include "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.h"  // SPEC-DFLASH2 W3/W4: Dflash2SelectCandidates, Dflash2WalkPath
#include "vllm/v1/worker/gpu/spec_decode/dspark/speculator.h"  // SPEC-DSPARK W5 SampleDsparkBlockDrafts
#include "vllm/v1/spec_decode/ngram_proposer.h"  // SPEC-NGRAM D3 NgramPropose
#include "vt/backend.h"  // vt::Backend / GetBackend (VT_GPU_SAMPLE=0 download)
#include "vt/dtype.h"  // VT_CHECK
#include "vt/tensor.h"
#ifdef VLLM_CPP_CUDA
#include "vt/cuda/combine_tokens.h"  // W3 device combine/scatter (removes the sync)
#endif

namespace vllm::v1 {

// Logits-gather A/B toggle (perf). Default ON: the forward gathers the
// per-request last-token hidden rows BEFORE lm_head (prefill/mixed), so lm_head
// runs on num_reqs rows and only [num_reqs,vocab] is Downloaded. VT_LOGITS_GATHER=0
// restores the old full [num_actual_tokens,vocab] path (lm_head over all tokens,
// full D2H, host re-gather in sample_tokens). Both are token-for-token identical.
static bool LogitsGatherEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LOGITS_GATHER");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

// GPU-sampling A/B toggle (perf). Default ON: the forward keeps the [num_reqs,
// vocab] logits ON DEVICE and the sampler (argmax / temperature / top-k/top-p —
// all device kernels, mirroring vllm/v1/sample/sampler.py which never copies the
// full logits to host) reads them directly; only the sampled token ids (~num_reqs
// * 4 bytes) cross to host. VT_GPU_SAMPLE=0 restores the OLD path: Download the
// full [num_reqs, vocab] logits to host, then sample from the host copy — for the
// A/B on the same binary. Token-for-token identical (same on-device sampler
// kernels either way; only the logits' residence differs). Requires the
// gather-before-lm_head path (default); with VT_LOGITS_GATHER=0 the forward
// already returns host logits, so this toggle is a no-op there.
static bool GpuSampleEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_GPU_SAMPLE");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

// Async-scheduling device-input default (ENG-ASYNC-SCHED W3 runner leaf).
// VT_ASYNC_RUNNER gates the combine_sampled_and_draft_tokens device-input +
// async sampler-output path at runner construction, which advertises
// runner_supports_async() so LoadedEngine resolves an AsyncScheduler + mcb=2.
// DEFAULT ON since the 2026-07-17 flip (mirror vLLM's async-scheduling default,
// vllm/config/vllm.py:992-1044; DGX-proven token-exact); VT_ASYNC_RUNNER=0 is the
// runner-level rollback to the synchronous host path (byte-identical pre-flip
// streams). Read at CONSTRUCTION (not per-step), honoring the env value live at
// each runner build so the DGX A/B — and the CPU construction-matrix test — can
// flip it per engine. The parse is factored into the pure, CPU-unit-tested
// AsyncRunnerFlagIsOn predicate (async_runner_flag.h). Tests may also toggle the
// flag directly via set_async_input_combine.
static bool AsyncRunnerEnvDefault() {
  return AsyncRunnerFlagIsOn(std::getenv("VT_ASYNC_RUNNER"));
}

// Async input-combine reads the sampled token id back on the host between
// steps. Whether that read is VALID is a backend CAPABILITY, not a device name:
// ask the backend (vt::Backend::SupportsAsyncSampledTokenReadback, backend.h),
// which answers true for CPU (host and device memory are one allocation) and
// CUDA (the sampled id is device-mirrored, async_device_mirror()), and false for
// a DISCRETE non-CUDA GPU (e.g. ROCm gfx1201) whose sample_tokens_async leg
// host-dereferences a device Alloc — the root cause of the "!" tokens on the lab
// R9700 (2026-08-07). An absent backend (device not built into this binary)
// yields nullptr and therefore false, which also subsumes the old
// #ifdef VLLM_CPP_CUDA guard. Keeping the question on the backend is what stops
// this device-agnostic shared layer from naming a device (check-device-leakage).
static bool QueueSupportsAsyncInputCombine(const vt::Queue& queue) {
  const vt::Backend* backend = vt::TryGetBackend(queue.device.type);
  return backend != nullptr && backend->SupportsAsyncSampledTokenReadback();
}

// GDN step-geometry diagnostic (default OFF). When VT_GDN_DIAG_STEP_LOG=1, each
// execute_model step prints the request count and the live/free recurrent-state
// slot geometry to std::cerr. Read ONCE (never per-step getenv); bounded to the
// packed-GDN c16 diagnostic checkpoint (see .agents/specs/gdn-packed-decode.md).
static bool GdnDiagStepLogEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_GDN_DIAG_STEP_LOG");
    return e != nullptr && e[0] == '1' && e[1] == '\0';
  }();
  return on;
}

namespace {

// SPEC-DFLASH2 W13, closing #2112. Emits both counter families on the readout's
// cadence, from the shared step path.
//
// BOTH FAMILIES, ALWAYS, when the dump is due -- including a `[dflash-route]`
// line of zeros on an engine that carries no DFlash draft. That is the #2089
// lesson applied to the readout rather than to the counter: "an instrument that
// only counts the fast path cannot report that a slow path exists", and a line
// that appeared only once a lane had run could not report that the lane is
// dead. A reader who asked for the dump is better served by a zero than by a
// missing line they have to interpret.
//
// The period is read fresh (`GraphStatsDumpPeriod`), so the readout cannot
// depend on which caller ran first in a process, and the due-check is one
// integer modulo over a total the counters already hold.
void MaybeDumpStepStats() {
  const int64_t period = GraphStatsDumpPeriod();
  if (period <= 0) return;
  const GraphDispatchStats stats = GetGraphDispatchStats();
  if (!GraphStatsDumpDue(GraphDispatchTotalSteps(stats), period)) return;
  std::fprintf(stderr, "%s\n", FormatGraphDispatchStats(stats).c_str());
  std::fprintf(stderr, "%s\n",
               vllm::detail::FormatDflashBlockRouteStats(
                   vllm::detail::GetDflashBlockRouteStats())
                   .c_str());
}

}  // namespace

// ─── Decode-first reorder (utils.py::reorder_batch_to_split_decodes_and_prefills)
bool reorder_batch_to_split_decodes_and_prefills(
    InputBatch& input_batch, const SchedulerOutput& scheduler_output,
    int decode_threshold) {
  const int num_reqs = input_batch.num_reqs();
  if (num_reqs <= 1) {
    return false;
  }

  // Per-request classification, in the current (dense) input_batch order.
  std::vector<int32_t> req_regions(static_cast<size_t>(num_reqs), 0);
  int num_decodes = 0, num_short = 0, num_long = 0, num_prefills = 0;
  for (int i = 0; i < num_reqs; ++i) {
    const std::string& req_id = *input_batch.req_ids[static_cast<size_t>(i)];
    const int num_scheduled = scheduler_output.num_scheduled_tokens.at(req_id);
    const int num_computed =
        input_batch.num_computed_tokens_cpu[static_cast<size_t>(i)];
    const int num_prompt =
        input_batch.num_prompt_tokens[static_cast<size_t>(i)];
    const bool has_context = num_computed > 0;
    const bool is_below = num_scheduled <= decode_threshold;
    const bool done_prefilling = num_computed >= num_prompt;
    // Mutually exclusive (exactly one True). Desired order:
    // decode(0) -> short_extend(1) -> long_extend(2) -> prefill(3).
    if (!has_context) {
      req_regions[static_cast<size_t>(i)] = 3;  // pure prefill (first chunk)
      ++num_prefills;
    } else if (!is_below) {
      req_regions[static_cast<size_t>(i)] = 2;  // long_extend
      ++num_long;
    } else if (!done_prefilling) {
      req_regions[static_cast<size_t>(i)] = 1;  // short_extend
      ++num_short;
    } else {
      req_regions[static_cast<size_t>(i)] = 0;  // decode
      ++num_decodes;
    }
  }

  // target_regions = repeat([0,1,2,3], [nd, ns, nl, np]).
  std::vector<int32_t> target_regions(static_cast<size_t>(num_reqs), 0);
  {
    int off = 0;
    const int counts[4] = {num_decodes, num_short, num_long, num_prefills};
    for (int region = 0; region < 4; ++region) {
      for (int k = 0; k < counts[region]; ++k) {
        target_regions[static_cast<size_t>(off++)] =
            static_cast<int32_t>(region);
      }
    }
  }

  // orig_indices = ascending indices whose region != target (need to move).
  std::vector<int> orig_indices;
  for (int i = 0; i < num_reqs; ++i) {
    if (req_regions[static_cast<size_t>(i)] !=
        target_regions[static_cast<size_t>(i)]) {
      orig_indices.push_back(i);
    }
  }
  if (orig_indices.empty()) {
    return false;
  }

  // src_indices = orig_indices sorted (stable) by their region — the source
  // order the swap chains consume. Stable keeps ascending index within a region.
  std::vector<int> src_indices = orig_indices;
  std::stable_sort(src_indices.begin(), src_indices.end(), [&](int a, int b) {
    return req_regions[static_cast<size_t>(a)] <
           req_regions[static_cast<size_t>(b)];
  });

  // src_dest_map = {src: dst} in src_indices insertion order (dst = orig_indices
  // by position). Iterate in that insertion order, following each swap chain.
  std::unordered_map<int, int> dest;
  for (size_t k = 0; k < src_indices.size(); ++k) {
    dest[src_indices[k]] = orig_indices[k];
  }
  for (int src : src_indices) {
    int dst = dest[src];
    while (src != dst) {
      input_batch.swap_states(src, dst);
      const auto it = dest.find(dst);
      const int next_dst = (it != dest.end()) ? it->second : dst;
      dest[dst] = dst;  // mark dst done
      dst = next_dst;
    }
  }
  return true;
}

// ─── apply_grammar_bitmask ──────────────────────────────────────────────────
// Ported from: vllm/v1/structured_output/utils.py::apply_grammar_bitmask @
// e24d1b24. See runner.h for the compacted-vs-dense contract + the bit sense.
void apply_grammar_bitmask(
    const GrammarOutput& grammar_output,
    const std::vector<std::string>& req_ids,
    const std::map<std::string, std::vector<int32_t>>&
        scheduled_spec_decode_tokens,
    vt::Queue& queue, vt::Tensor& logits) {
  const TokenBitmask& bitmask = grammar_output.grammar_bitmask;
  // No structured request scheduled this step => nothing to mask (no-op).
  if (grammar_output.structured_output_request_ids.empty() ||
      bitmask.num_seqs == 0) {
    return;
  }
  const int64_t num_logits = logits.shape[0];
  const int64_t vocab = logits.shape[1];

  // struct_out_req_batch_indices: for each structured req, the logit row = its
  // dense batch index + the cumulative spec-token offset ahead of it
  // (utils.py:112-120). At T0 scheduled_spec_decode_tokens is empty, so the
  // offset stays 0 and logit_index == batch_index.
  const std::set<std::string> struct_out_req_ids(
      grammar_output.structured_output_request_ids.begin(),
      grammar_output.structured_output_request_ids.end());
  std::unordered_map<std::string, int> struct_out_req_batch_indices;
  {
    int cumulative_offset = 0;
    for (int batch_index = 0;
         batch_index < static_cast<int>(req_ids.size()); ++batch_index) {
      const std::string& req_id = req_ids[static_cast<size_t>(batch_index)];
      const int logit_index = batch_index + cumulative_offset;
      const auto sit = scheduled_spec_decode_tokens.find(req_id);
      if (sit != scheduled_spec_decode_tokens.end()) {
        cumulative_offset += static_cast<int>(sit->second.size());
      }
      if (struct_out_req_ids.count(req_id) != 0) {
        struct_out_req_batch_indices[req_id] = logit_index;
      }
    }
  }

  // Reorder the compacted bitmask onto the dense logits rows and unpack it into a
  // per-row EXCLUDE mask (utils.py:124-140 reorder + the bit unpack). Rows for
  // non-structured requests stay all-false (all tokens allowed). The
  // apply_allowed_token_ids op reads TRUE == "exclude this token" (-> -inf), so a
  // token is excluded exactly when its grammar bit is CLEAR (forbidden).
  std::vector<std::vector<uint8_t>> exclude(
      static_cast<size_t>(num_logits),
      std::vector<uint8_t>(static_cast<size_t>(vocab), 0));
  int cumulative_index = 0;
  for (const std::string& req_id :
       grammar_output.structured_output_request_ids) {
    int num_spec_tokens = 0;
    const auto sit = scheduled_spec_decode_tokens.find(req_id);
    if (sit != scheduled_spec_decode_tokens.end()) {
      num_spec_tokens = static_cast<int>(sit->second.size());
    }
    const auto bit = struct_out_req_batch_indices.find(req_id);
    if (bit != struct_out_req_batch_indices.end()) {
      const int logit_idx = bit->second;
      for (int i = 0; i < 1 + num_spec_tokens; ++i) {
        const int bitmask_row = cumulative_index + i;
        const int logit_row = logit_idx + i;
        if (logit_row < 0 || logit_row >= static_cast<int>(num_logits) ||
            bitmask_row < 0 || bitmask_row >= bitmask.num_seqs) {
          continue;
        }
        const int32_t* words =
            bitmask.data.data() +
            static_cast<size_t>(bitmask_row) *
                static_cast<size_t>(bitmask.num_words);
        std::vector<uint8_t>& row = exclude[static_cast<size_t>(logit_row)];
        for (int64_t t = 0; t < vocab; ++t) {
          const int32_t word = words[static_cast<size_t>(t >> 5)];
          const bool allowed =
              ((word >> static_cast<int>(t & 31)) & 1) != 0;
          if (!allowed) row[static_cast<size_t>(t)] = 1;
        }
      }
    }
    cumulative_index += 1 + num_spec_tokens;
  }

  // Set every forbidden token's logit to -inf (reuse the M1.7 sampler op; CPU +
  // CUDA counterparts both exist — CUDA path is dgx-pending like the sampler).
  apply_allowed_token_ids(queue, logits, exclude);
}

// ─── GPUModelRunner ─────────────────────────────────────────────────────────

namespace {
// Per-group block sizes from the KVCacheConfig groups, in group order (the
// InputBatch / MultiGroupBlockTable expect one entry per KV cache group).
std::vector<int> group_block_sizes(const KVCacheConfig& cfg) {
  std::vector<int> sizes;
  sizes.reserve(cfg.kv_cache_groups.size());
  for (const auto& g : cfg.kv_cache_groups) {
    sizes.push_back(g.kv_cache_spec->block_size);
  }
  return sizes;
}

// ── #810: per-layer KV membership from the GROUP the model published ────────
//
// Upstream never parses a per-layer config string to decide what a layer's
// cache is. `get_kv_cache_spec()` walks the instantiated modules
// (`gpu_model_runner.py:7785-7787`), groups layers by their SPEC OBJECT
// (`kv_cache_utils.py:1209-1211` `same_type_layers[layer_spec].append(name)`),
// and then fans metadata out BY LAYER NAME (`gpu_model_runner.py:2548-2549`).
// `KVCacheGroupSpec.layer_names` (`kv_cache_interface.py:938-947`, mirrored at
// `include/vllm/v1/kv_cache_interface.h:343`) is the durable signal; a config
// spelling such as `layer_types` is not, and a hybrid whose config does not
// speak Qwen3.5's dialect (NemotronH ships `layers_block_type`, no
// `layer_types`) has an EMPTY one.
//
// Our runner still indexes buffers by layer POSITION, so a published name has
// to be resolved back to an index. `KVCacheLayerIndexOfName` does exactly that
// and nothing else: the integer of the `.layers.<N>.` segment of an
// upstream-style module path ("backbone.layers.5.mixer",
// "model.layers.12.self_attn").
//
// It deliberately returns nullopt for a PLACEHOLDER group name — "fa", "gdn",
// "mla", "kda", "fa_draft" — because such a name carries no layer identity at
// all. That is what keeps this additive: a group that does not publish
// per-layer names falls back to the historical `config_.layer_types` predicate,
// byte for byte.
//
// FIX-KV-GROUP-LAYER-COUNT (#1963, #1966) moved the body to
// `vllm::v1::KVCacheLayerIndexOfName` (kv_cache_interface.cpp), because the
// SIZING path now has to tell a placeholder name from a real one as well, and
// two copies of one rule are what disagree. It is also the reason the
// placeholder is no longer what this function usually sees: the loader resolves
// the placeholders into real names before the config reaches the runner, so a
// registry that publishes nothing still arrives here named.
using vllm::v1::KVCacheLayerIndexOfName;

// The enumerator's own name, for the refusal below. Nothing else in the tree
// spells a KVCacheSpecKind, and a refusal that prints an integer names nothing.
const char* KVCacheSpecKindName(KVCacheSpecKind kind) {
  switch (kind) {
    case KVCacheSpecKind::kFullAttention: return "kFullAttention";
    case KVCacheSpecKind::kMlaAttention: return "kMlaAttention";
    case KVCacheSpecKind::kSlidingWindow: return "kSlidingWindow";
    case KVCacheSpecKind::kSlidingWindowMla: return "kSlidingWindowMla";
    case KVCacheSpecKind::kMamba: return "kMamba";
    case KVCacheSpecKind::kChunkedLocalAttention:
      return "kChunkedLocalAttention";
    case KVCacheSpecKind::kSinkFullAttention: return "kSinkFullAttention";
    case KVCacheSpecKind::kEncoderOnlyAttention:
      return "kEncoderOnlyAttention";
    case KVCacheSpecKind::kCrossAttention: return "kCrossAttention";
    case KVCacheSpecKind::kUnknown: return "kUnknown";
  }
  return "kUnknown";
}

// The per-layer membership mask of one KV cache group, or nullopt when the
// group does not publish per-layer names.
//
// ALL-OR-NOTHING on purpose: a group is only read by name when EVERY one of its
// names resolves to a distinct in-range layer index. A partially-parseable
// group would silently drop the layers whose names did not parse, which is the
// silent-wrong-answer shape this whole row exists to remove.
std::optional<std::vector<bool>> GroupLayerMask(const KVCacheGroupSpec& group,
                                                int64_t num_layers) {
  if (group.layer_names.empty()) return std::nullopt;
  std::vector<bool> mask(static_cast<size_t>(num_layers), false);
  for (const std::string& name : group.layer_names) {
    const std::optional<int64_t> l = KVCacheLayerIndexOfName(name);
    if (!l.has_value() || *l < 0 || *l >= num_layers) return std::nullopt;
    if (mask[static_cast<size_t>(*l)]) return std::nullopt;  // duplicate index
    mask[static_cast<size_t>(*l)] = true;
  }
  return mask;
}
}  // namespace

GPUModelRunner::GPUModelRunner(
    const HfConfig& config, LoadedModel& model,
    const KVCacheConfig& kv_cache_config, vt::Queue queue, int max_num_reqs,
    int max_model_len, int max_num_batched_tokens,
    std::optional<vllm::SpeculativeConfig> spec_config,
    std::unique_ptr<vllm::Qwen3_5MTPModel> draft_model,
    std::vector<PagedKvCache> draft_kv)
    : config_(config),
      model_(&model),
      spec_config_(std::move(spec_config)),
      draft_model_(std::move(draft_model)),
      draft_attn_kv_(std::move(draft_kv)),
      queue_(queue),
      // SPEC-DFLASH2 A2-3 (#2911): size the per-req_state draft buffer's row by
      // this runner's resolved k, mirroring `RequestStates(..., num_speculative_
      // steps)` (states.py:34,71-77). `num_spec()` reads `spec_config_` only,
      // which is declared (and so initialized) above `input_batch_`. 0 without a
      // speculator, which leaves the buffer empty and every non-speculative
      // runner byte-identical.
      // FIX-BLOCK-TABLE-ROW-WIDTH: each group's row width comes from its spec
      // (gpu_model_runner.py:7340-7360 @ e126687a9a), so a speculative GDN
      // group gets its `+ k` blocks.
      input_batch_(max_num_reqs, max_model_len, max_num_batched_tokens,
                   static_cast<int>(config.vocab_size),
                   group_block_sizes(kv_cache_config),
                   group_block_sizes(kv_cache_config), num_spec(),
                   block_table_geometry(kv_cache_config, max_model_len).max_num_blocks,
                   block_table_geometry(kv_cache_config, max_model_len)
                       .slot_mapping_modes) {
  max_num_reqs_ = max_num_reqs;
  max_num_batched_tokens_ = max_num_batched_tokens;
  // SPEC-MTP I5e: the async input-combine splices the device-resident
  // last_sampled token over each decode row's input id. A speculator keeps the
  // sync HOST INPUT path instead (its drafts are spliced into token_ids_cpu by
  // update_req_spec_token_ids + prepare_inputs, and its sampler is the sync
  // one, so the host arrays stay fresh).
  // Since SPEC-DFLASH2 W7 (#1824) this veto is INPUT-side only: async
  // SCHEDULING stays on for the Eagle-type family via async_sched_supported_
  // below. Byte-identical for non-spec (spec_config_ is nullopt there, so
  // this is AsyncRunnerEnvDefault()).
  //
  // SPEC-DFLASH2 A2-1 (#2644) REMOVED the reason this comment used to lead
  // with. The combine was not spec-aware and overwrote a verify step's last
  // draft with the committed token; it now takes each row's num_logits from
  // cu_num_logits, so a verify step keeps that draft. Do not read the veto as
  // waiting on the combine's arithmetic — that half has landed.
  //
  // SPEC-DFLASH2 A2-2 (#2802) removed the OTHER half this comment used to lead
  // with. sample_tokens_async now HAS a verify arm, routing on the same
  // StepRoutesToVerify(step.num_draft_tokens) that sample_tokens routes on, so
  // a spec engine on the async front would no longer sample the EXPANDED
  // (1 + k) verify rows as if they were decode rows. Do not read the veto as
  // waiting on the async sampler either — that half has landed too, and its arm
  // is UNREACHED precisely because this line stands (spec `## Owed`).
  //
  // SPEC-DFLASH2 A2-3 (#2911) removed the reason this comment carried LAST. The
  // draft buffer is wired: `input_batch_.draft_tokens` is a per-req_state
  // [max_num_reqs, k] buffer that `set_draft_tokens` produces and that BOTH the
  // async placeholder fill and all three combine call sites read. The combine's
  // "not wired yet" refusal is gone, replaced by the sizing invariant that makes
  // the CUDA scatter's unbounded row read safe. Do not read the veto as waiting
  // on the draft buffer either. Its repair also gave the two CUDA arms the real
  // per-request `cu_num_logits` the host arm always had; they passed a null,
  // which those kernels read as an arange, and an arange is reason A on the
  // first verify step this line would admit.
  //
  // ONE reason keeps it standing, and it is NOT this wave's:
  //   `sample_tokens_async`'s DECODE arm never proposes. Its verify arm calls
  //   `propose_after_verify` (A2-2), but the arm that runs when the step drafts
  //   nothing — which is every spec engine's FIRST step, before any drafts exist
  //   — has no `spec_on()` branch at all, where the synchronous `sample_tokens`
  //   has one. So with this line deleted a spec engine on the `AsyncLLM` front
  //   proposes nothing on its first decode step and the next step's fill refuses.
  //   That is an unreached arm's gap, not a draft-buffer gap; the row's spec
  //   lists it under `## Owed` against #2911 and A2-5 cannot flip this line until
  //   it is closed.
  // Measured on the CPU tier at THIS head, deleting !spec_config_.has_value() at
  // both construction sites: test_mtp_depth goes to 9 of 10 cases passed, 1
  // failed, 118 assertions passed, exit 1 — against 5 of 10 failed on 29
  // assertions at the A2-2 head. "async input combine:" does not appear in the
  // output at all, so the combine now accepts the verify step and its scatter
  // runs. The single remaining failure is the W7 sync-vs-async identity case
  // throwing the FILL's refusal, "proposed 0 drafts but the scheduler placed 2
  // placeholders", which is the decode-arm gap above. The wave order and the
  // gates are in .agents/specs/dflash2-async-spec-sampler.md.
  // The A2-3 repair moved that message, not that count: with the fill's
  // freshness refusal restored the same case throws "async draft fill: no drafts
  // proposed for request 'req' (placeholders scheduled without a matching
  // propose)" from this file instead, still 9 of 10 on 118 assertions, and the
  // new wording names the decode-arm gap rather than its downstream symptom.
  async_input_combine_ = AsyncRunnerEnvDefault() && !spec_config_.has_value() &&
                         QueueSupportsAsyncInputCombine(queue_);
  // SPEC-DFLASH2 W7 (#1824): async SCHEDULING capability is the same
  // env/backend predicate WITHOUT the spec veto above — a spec engine keeps
  // the sync host input path (the combine is not draft-aware) while still
  // advertising the scheduler overlap, mirroring upstream keeping async
  // scheduling ON for the Eagle-type family (vllm/config/vllm.py:1064-1112).
  async_sched_supported_ =
      AsyncRunnerEnvDefault() && QueueSupportsAsyncInputCombine(queue_);
  if (std::getenv("VT_ASYNC_DEBUG"))
    fprintf(stderr, "[async-debug] env=%d queue_combine=%d -> sched_supported=%d\n",
            (int)AsyncRunnerEnvDefault(),
            (int)QueueSupportsAsyncInputCombine(queue_),
            (int)async_sched_supported_);
  // ARCH-ONE-SURFACE ROW 6 (mirror gpu/model_runner.py:368-369): a POOLING
  // model's runner pools instead of sampling — build the PoolingRunner over
  // the model-owned Pooler. Null for every text arch (byte-identical).
  if (model_->registration().info.is_pooling_model &&
      model_->pooler() != nullptr) {
    pooling_runner_ = std::make_unique<vllm::PoolingRunner>(*model_->pooler());
  }
  initialize_kv_cache(kv_cache_config);
  ModelRegistry::Prepare(*model_, config_, queue_);
}

GPUModelRunner::GPUModelRunner(
    const HfConfig& config, std::unique_ptr<LoadedModel> owned_model,
    const KVCacheConfig& kv_cache_config, vt::Queue queue, int max_num_reqs,
    int max_model_len, int max_num_batched_tokens,
    std::optional<vllm::SpeculativeConfig> spec_config,
    std::unique_ptr<vllm::Qwen3_5MTPModel> draft_model,
    std::vector<PagedKvCache> draft_kv)
    : config_(config),
      owned_model_(std::move(owned_model)),
      model_(owned_model_.get()),
      spec_config_(std::move(spec_config)),
      draft_model_(std::move(draft_model)),
      draft_attn_kv_(std::move(draft_kv)),
      queue_(queue),
      // SPEC-DFLASH2 A2-3 (#2911): size the per-req_state draft buffer's row by
      // this runner's resolved k, mirroring `RequestStates(..., num_speculative_
      // steps)` (states.py:34,71-77). `num_spec()` reads `spec_config_` only,
      // which is declared (and so initialized) above `input_batch_`. 0 without a
      // speculator, which leaves the buffer empty and every non-speculative
      // runner byte-identical.
      // FIX-BLOCK-TABLE-ROW-WIDTH: each group's row width comes from its spec
      // (gpu_model_runner.py:7340-7360 @ e126687a9a), so a speculative GDN
      // group gets its `+ k` blocks.
      input_batch_(max_num_reqs, max_model_len, max_num_batched_tokens,
                   static_cast<int>(config.vocab_size),
                   group_block_sizes(kv_cache_config),
                   group_block_sizes(kv_cache_config), num_spec(),
                   block_table_geometry(kv_cache_config, max_model_len).max_num_blocks,
                   block_table_geometry(kv_cache_config, max_model_len)
                       .slot_mapping_modes) {
  max_num_reqs_ = max_num_reqs;
  max_num_batched_tokens_ = max_num_batched_tokens;
  // SPEC-MTP I5e: the async input-combine splices the device-resident
  // last_sampled token over each decode row's input id. A speculator keeps the
  // sync HOST INPUT path instead (its drafts are spliced into token_ids_cpu by
  // update_req_spec_token_ids + prepare_inputs, and its sampler is the sync
  // one, so the host arrays stay fresh).
  // Since SPEC-DFLASH2 W7 (#1824) this veto is INPUT-side only: async
  // SCHEDULING stays on for the Eagle-type family via async_sched_supported_
  // below. Byte-identical for non-spec (spec_config_ is nullopt there, so
  // this is AsyncRunnerEnvDefault()).
  //
  // SPEC-DFLASH2 A2-1 (#2644) REMOVED the reason this comment used to lead
  // with. The combine was not spec-aware and overwrote a verify step's last
  // draft with the committed token; it now takes each row's num_logits from
  // cu_num_logits, so a verify step keeps that draft. Do not read the veto as
  // waiting on the combine's arithmetic — that half has landed.
  //
  // SPEC-DFLASH2 A2-2 (#2802) removed the OTHER half this comment used to lead
  // with. sample_tokens_async now HAS a verify arm, routing on the same
  // StepRoutesToVerify(step.num_draft_tokens) that sample_tokens routes on, so
  // a spec engine on the async front would no longer sample the EXPANDED
  // (1 + k) verify rows as if they were decode rows. Do not read the veto as
  // waiting on the async sampler either — that half has landed too, and its arm
  // is UNREACHED precisely because this line stands (spec `## Owed`).
  //
  // SPEC-DFLASH2 A2-3 (#2911) removed the reason this comment carried LAST. The
  // draft buffer is wired: `input_batch_.draft_tokens` is a per-req_state
  // [max_num_reqs, k] buffer that `set_draft_tokens` produces and that BOTH the
  // async placeholder fill and all three combine call sites read. The combine's
  // "not wired yet" refusal is gone, replaced by the sizing invariant that makes
  // the CUDA scatter's unbounded row read safe. Do not read the veto as waiting
  // on the draft buffer either. Its repair also gave the two CUDA arms the real
  // per-request `cu_num_logits` the host arm always had; they passed a null,
  // which those kernels read as an arange, and an arange is reason A on the
  // first verify step this line would admit.
  //
  // ONE reason keeps it standing, and it is NOT this wave's:
  //   `sample_tokens_async`'s DECODE arm never proposes. Its verify arm calls
  //   `propose_after_verify` (A2-2), but the arm that runs when the step drafts
  //   nothing — which is every spec engine's FIRST step, before any drafts exist
  //   — has no `spec_on()` branch at all, where the synchronous `sample_tokens`
  //   has one. So with this line deleted a spec engine on the `AsyncLLM` front
  //   proposes nothing on its first decode step and the next step's fill refuses.
  //   That is an unreached arm's gap, not a draft-buffer gap; the row's spec
  //   lists it under `## Owed` against #2911 and A2-5 cannot flip this line until
  //   it is closed.
  // Measured on the CPU tier at THIS head, deleting !spec_config_.has_value() at
  // both construction sites: test_mtp_depth goes to 9 of 10 cases passed, 1
  // failed, 118 assertions passed, exit 1 — against 5 of 10 failed on 29
  // assertions at the A2-2 head. "async input combine:" does not appear in the
  // output at all, so the combine now accepts the verify step and its scatter
  // runs. The single remaining failure is the W7 sync-vs-async identity case
  // throwing the FILL's refusal, "proposed 0 drafts but the scheduler placed 2
  // placeholders", which is the decode-arm gap above. The wave order and the
  // gates are in .agents/specs/dflash2-async-spec-sampler.md.
  // The A2-3 repair moved that message, not that count: with the fill's
  // freshness refusal restored the same case throws "async draft fill: no drafts
  // proposed for request 'req' (placeholders scheduled without a matching
  // propose)" from this file instead, still 9 of 10 on 118 assertions, and the
  // new wording names the decode-arm gap rather than its downstream symptom.
  async_input_combine_ = AsyncRunnerEnvDefault() && !spec_config_.has_value() &&
                         QueueSupportsAsyncInputCombine(queue_);
  // SPEC-DFLASH2 W7 (#1824): async SCHEDULING capability is the same
  // env/backend predicate WITHOUT the spec veto above — a spec engine keeps
  // the sync host input path (the combine is not draft-aware) while still
  // advertising the scheduler overlap, mirroring upstream keeping async
  // scheduling ON for the Eagle-type family (vllm/config/vllm.py:1064-1112).
  async_sched_supported_ =
      AsyncRunnerEnvDefault() && QueueSupportsAsyncInputCombine(queue_);
  if (std::getenv("VT_ASYNC_DEBUG"))
    fprintf(stderr, "[async-debug] env=%d queue_combine=%d -> sched_supported=%d\n",
            (int)AsyncRunnerEnvDefault(),
            (int)QueueSupportsAsyncInputCombine(queue_),
            (int)async_sched_supported_);
  // ARCH-ONE-SURFACE ROW 6 (mirror gpu/model_runner.py:368-369): a POOLING
  // model's runner pools instead of sampling — build the PoolingRunner over
  // the model-owned Pooler. Null for every text arch (byte-identical).
  if (model_->registration().info.is_pooling_model &&
      model_->pooler() != nullptr) {
    pooling_runner_ = std::make_unique<vllm::PoolingRunner>(*model_->pooler());
  }
  initialize_kv_cache(kv_cache_config);
  ModelRegistry::Prepare(*model_, config_, queue_);
}

GPUModelRunner::GPUModelRunner(const HfConfig& config,
                               const Qwen3_5MoeWeights& weights,
                               const KVCacheConfig& kv_cache_config,
                               vt::Queue queue, int max_num_reqs,
                               int max_model_len, int max_num_batched_tokens)
    : GPUModelRunner(config, BorrowQwen3_5MoeLoadedModel(weights),
                     kv_cache_config, queue, max_num_reqs, max_model_len,
                     max_num_batched_tokens) {}

GPUModelRunner::GPUModelRunner(const HfConfig& config,
                               const Qwen3_5DenseWeights& weights,
                               const KVCacheConfig& kv_cache_config,
                               vt::Queue queue, int max_num_reqs,
                               int max_model_len, int max_num_batched_tokens)
    : GPUModelRunner(config, BorrowQwen3_5DenseLoadedModel(weights),
                     kv_cache_config, queue, max_num_reqs, max_model_len,
                     max_num_batched_tokens) {}

GPUModelRunner::CacheBuffer::CacheBuffer(vt::Device device, vt::Queue& queue,
                                         size_t bytes,
                                         bool backend_resident)
    : device_(device), backend_resident_(backend_resident), bytes_(bytes) {
  if (!backend_resident_) {
    host_data_.assign(bytes, uint8_t{0});
    return;
  }

  backend_data_ = vt::Alloc(device_, std::max<size_t>(bytes, 1));
  try {
    if (bytes != 0) {
      vt::GetBackend(device_.type).Memset(queue, backend_data_, 0, bytes);
    }
  } catch (...) {
    vt::Free(device_, backend_data_);
    backend_data_ = nullptr;
    throw;
  }
}

GPUModelRunner::CacheBuffer::~CacheBuffer() {
  if (backend_data_ != nullptr) {
    vt::Free(device_, backend_data_);
  }
}

// FIX-KV-GROUP-LAYER-COUNT (#1963, #1966): the measurement half of the fix. See
// runner.h for what each of the two sums is meant to equal.
int64_t GPUModelRunner::kv_cache_allocated_paged_bytes() const {
  int64_t total = 0;
  for (const auto& b : full_attn_buf_) total += static_cast<int64_t>(b->bytes());
  for (const auto& b : draft_attn_buf_) total += static_cast<int64_t>(b->bytes());
  return total;
}

int64_t GPUModelRunner::kv_cache_allocated_bytes() const {
  int64_t total = kv_cache_allocated_paged_bytes();
  // ENG-RECURRENT-MULTISTATE (#2131): over EVERY state the group published, not
  // over a hardcoded conv + temporal pair. A state this loop did not visit is a
  // short KV report, which is the accounting half of #1963.
  for (const auto& layer : recurrent_state_buf_)
    for (const auto& b : layer) total += static_cast<int64_t>(b->bytes());
  return total;
}

void GPUModelRunner::initialize_kv_cache(const KVCacheConfig& kv_cache_config) {
  num_blocks_ = kv_cache_config.num_blocks;
  // ENGINE-LEVEL ATTENTION-BACKEND SELECTION (M3, issue #41) happens INSIDE the
  // full-attention region below, never here: a pure-GDN / pooling model that
  // caches no paged KV must not pay selection, and a platform whose priority
  // list yields no dense backend must fail loudly only for models that actually
  // need one (the empty-list loud-throw design, rocm.cpp W0). The resolution
  // block lives in the full-attn region; the per-group validation in the view
  // loop below.
  // Resolved LAZILY per group kind, on first use in the view loop below: a
  // pure-MLA model never resolves (or validates) a dense backend, and a dense
  // model never resolves MLA. `dense_backend` throws loudly if the platform has
  // no registered dense backend (the empty-list loud-throw design); `mla_backend`
  // stays empty on a device with no registered MLA backend (op-driven MLA).
  std::string dense_backend;
  std::string mla_backend;
  bool dense_backend_resolved = false;
  bool mla_backend_resolved = false;
  // GDN mamba-state slots = max concurrent sequences (one recurrent state per
  // sequence), decoupled from the attention num_blocks. Guard against a 0 (e.g.
  // a test path that skipped the ctor arg) by falling back to num_blocks.
  //
  // SPEC-MTP I5d: under speculation each sequence needs num_spec+1 CONSECUTIVE
  // GDN state slots (the k+1 draft-timestep snapshots the recurrent rollback
  // selects among, spec §3). Size the compact pool max_num_reqs*(num_spec+1) and
  // let remap_gdn_state_slots hand each sequence a base of num_spec+1 slots.
  const int spec_cols = spec_on() ? num_spec() + 1 : 1;
  const int64_t base_slots = max_num_reqs_ > 0 ? max_num_reqs_ : num_blocks_;
  gdn_state_slots_ = base_slots * spec_cols;
  gdn_slot_of_req_.clear();
  gdn_free_slots_.clear();
  gdn_free_slots_.reserve(static_cast<size_t>(base_slots));
  // The free list holds each sequence's BASE state slot. Under speculation a
  // base owns spec_cols (== num_spec+1) consecutive slots, so bases step by
  // spec_cols; without speculation spec_cols==1 and this is the pre-spec pool.
  for (int64_t b = base_slots - 1; b >= 0; --b)
    gdn_free_slots_.push_back(static_cast<int32_t>(b * spec_cols));

  // Resolve the full-attn + GDN(mamba) KV group ids (T0 gate models: exactly one
  // of each). The block-table group order == kv_cache_groups order.
  //
  // SPEC-MTP I5d-pre LATENT-BUG FIX: the previous loop assigned
  // `full_attn_group_id_` on EVERY full-attn/MLA group, so it kept the LAST one.
  // With num_spec>0 the MTP draft adds a THIRD group (`fa_draft`, appended at
  // index num_hidden_layers — qwen3_5_common.cpp:99-104), which the old code
  // would then wrongly select as the TARGET attention group. Select the TARGET
  // deterministically instead: the FIRST (index 0) non-eagle full-attn/MLA group,
  // so a later-appended draft group can never displace it. With num_spec==0
  // (every run today) there is exactly ONE full-attn group, so this is
  // byte-identical to the old behavior — the first and only group is chosen.
  //
  // KV-DSV4-MULTICACHE W3 (#2068) ALSO collects every group, in publication
  // order: `attn_group_ids_` for every non-eagle group whose spec is an
  // `AttentionSpec` and `recurrent_group_ids_` for every `kMamba` group.
  // `full_attn_group_id_` and `gdn_group_id_` are assigned by exactly the
  // conditions they were assigned by before, so their values are unchanged for
  // every topology; the vectors are additional, and only the multi-cache
  // allocation below reads them.
  attn_group_ids_.clear();
  recurrent_group_ids_.clear();
  for (int g = 0; g < static_cast<int>(kv_cache_config.kv_cache_groups.size());
       ++g) {
    const auto& group = kv_cache_config.kv_cache_groups[static_cast<size_t>(g)];
    const KVCacheSpecKind kind = group.kv_cache_spec->kind();
    // W3: `AttentionSpec` — not a kind whitelist — is the predicate, because
    // `page_size_bytes()` is the one allocation accessor EVERY `AttentionSpec`
    // inherits and NONE of them overrides: it is defined once, on `AttentionSpec`
    // itself (`vllm/v1/kv_cache_interface.py:196-201`), and all eleven subclasses
    // at the pin take it as inherited. (The file defines the name three more
    // times — `KVCacheSpec:109`, `MambaSpec:699`, `UniformTypeKVCacheSpecs:828` —
    // none of them an `AttentionSpec`, so the anchor is stated with its class
    // rather than by line alone.) The VALUE is kind-DEPENDENT and deliberately
    // so: five subclasses override `real_page_size_bytes` and MLA additionally
    // pads through `_apply_alignment_padding` (`:345-351`, typed
    // `MLAAttentionSpec | SlidingWindowMLASpec`). Uniform ACCESSOR over a
    // kind-dependent value is exactly what lets this cast stand in for a
    // whitelist, and it is what makes `kSlidingWindow` and
    // `kChunkedLocalAttention` allocatable with no further branch.
    const bool is_attention_spec =
        dynamic_cast<const AttentionSpec*>(group.kv_cache_spec.get()) != nullptr;
    if (is_attention_spec && !group.is_eagle_group) attn_group_ids_.push_back(g);
    // MLA campaign W7: an `MLAAttentionSpec` group IS the model's attention
    // group. Upstream registers MLA against the ordinary FullAttentionManager
    // (`vllm/v1/core/single_type_kv_cache_manager.py:1539`), so block table,
    // prefix caching and eviction are identical and the ONLY difference is the
    // page cost/shape — which the spec-driven allocation below already reads
    // off the spec. Additive condition: no existing arch changes group.
    if (kind == KVCacheSpecKind::kFullAttention ||
        kind == KVCacheSpecKind::kMlaAttention) {
      // Skip a draft (eagle) group and keep the FIRST match as the target.
      if (!group.is_eagle_group && full_attn_group_id_ < 0) {
        full_attn_group_id_ = g;
      }
    } else if (kind == KVCacheSpecKind::kMamba) {
      recurrent_group_ids_.push_back(g);
      gdn_group_id_ = g;
    }
  }

  // KV-DSV4-MULTICACHE W2 (#1973): REFUSE a published group this runner does
  // not allocate, instead of dropping it in silence.
  //
  // The loop above has exactly two arms and NO `else`. A `kSlidingWindowMla`
  // group matched nothing, and a SECOND `kMlaAttention` group is passed over by
  // the `full_attn_group_id_ < 0` guard — and neither produced a buffer or a
  // message. That is not "one group is missing": `membership_by_name` below is
  // reached only when the model has a recurrent group, so a model without one
  // (DeepSeek-V4 has none) falls into `is_full_attn = !is_gdn` and allocates
  // ONE buffer per HIDDEN LAYER, every one of them sized from the TARGET
  // group's page. Publishing DeepSeek-V4's seven groups without this check
  // allocates 43 buffers of one page for a model that needs 167 of seven, and
  // reports nothing. A silently short KV allocation is a wrong-tokens failure,
  // not a crash, which is why this refuses rather than warns
  // (AGENTS.md: "Refuse an unimplemented arm with a message that names the
  // missing part").
  //
  // A published group is exactly one of four things: the TARGET attention
  // group, the RECURRENT group, the single `fa_draft` draft-KV slot allocated
  // at `draft_attn_buf_` below, or a defect. The draft slot is recognised on
  // its KIND, mirroring that block's own predicate (`kind() != kFullAttention`
  // => continue), so a `fa_draft` group published with speculation OFF is
  // tolerated here and still unallocated — tightening that belongs with the
  // generalization of `full_attn_group_id_` (row KV-DSV4-MULTICACHE W3, which
  // is where a positive per-group accounting becomes possible).
  //
  // BYTE-NEUTRAL for every model shipping today: each publishes exactly the
  // groups this runner consumes, which `test_runner.cpp` asserts directly
  // rather than leaves to be inferred.
  //
  // KV-DSV4-MULTICACHE W3 (#2068) REPLACES the refusal for the topologies it can
  // now carry, and keeps it for the ones it cannot.
  //
  // THE ENTRY PREDICATE, and the reason byte-neutrality here is a property of
  // the control flow rather than a claim: the generalized path is entered ONLY
  // when a published group is left over after the target attention group, the
  // recurrent group and the `fa_draft` slot. That leftover set is exactly what
  // W2's loop below computes, and it is EMPTY for every model shipping today
  // (36 group-emplacement sites across 31 factories: 27 single-group, plus
  // `kimi_linear` MLA+Mamba, `nemotron_h` FA+Mamba, `qwen3_5_common`
  // FA+Mamba+`fa_draft`, and `deepseek_v4`). So every one of them takes the
  // legacy path below, unchanged, and none of the W3 code runs at all. This is
  // the `per_layer_attn_specs` empty-means-unchanged contract
  // (`include/vllm/v1/kv_cache_interface.h`) applied to the GROUP SET instead of
  // the layer list.
  //
  // WHAT STILL REACHES THE REFUSAL. Four shapes the generalized path cannot
  // represent, each of which is reachable from any future registry and each of
  // which would otherwise be a short KV allocation, i.e. wrong tokens rather
  // than a crash: a group whose published names do not ALL resolve to distinct
  // in-range layer indices, a SECOND `kMamba` group, a group whose spec is
  // neither an `AttentionSpec` nor a `MambaSpec`, and an EAGLE group.
  //
  // THE EAGLE SHAPE IS THE ONE THIS LOOP MISSED (#2084). `attn_group_ids_` above
  // excludes `is_eagle_group` by construction, because a draft KV is allocated
  // by the `draft_attn_buf_` block below rather than by the generalized loop —
  // and that block is itself OFF on a multi-cache topology (see its own guard).
  // So an eagle `AttentionSpec` group here satisfied every other arm, kept `why`
  // empty, and then received no buffer: NINE of ten published caches allocated,
  // in silence. That is the exact failure the message below names, arriving
  // through the one predicate the message did not test. Refusing is the right
  // direction rather than dropping the `!group.is_eagle_group` filter at the
  // collection above, because allocating a draft group as an ordinary named
  // cache is a decision about how speculation shares a multi-cache topology, and
  // that decision belongs to the wave that gates the speculative path (W4/W5),
  // not to one whose obligation is that nothing about it moves.
  const int64_t num_layers_for_groups = config_.num_hidden_layers;
  std::vector<std::vector<int32_t>> group_layer_index(
      kv_cache_config.kv_cache_groups.size());
  bool multi_cache_topology = false;
  {
    int leftover_count = 0;
    bool draft_slot_taken = false;
    for (int g = 0;
         g < static_cast<int>(kv_cache_config.kv_cache_groups.size()); ++g) {
      if (g == full_attn_group_id_ || g == gdn_group_id_) continue;
      const auto& group =
          kv_cache_config.kv_cache_groups[static_cast<size_t>(g)];
      const KVCacheSpecKind kind = group.kv_cache_spec->kind();
      if (kind == KVCacheSpecKind::kFullAttention && !draft_slot_taken) {
        draft_slot_taken = true;  // the `fa_draft` draft-KV slot
        continue;
      }
      ++leftover_count;
    }
    multi_cache_topology = leftover_count > 0;
  }
  if (multi_cache_topology) {
    std::string unallocated;
    int unallocated_count = 0;
    int recurrent_seen = 0;
    for (int g = 0;
         g < static_cast<int>(kv_cache_config.kv_cache_groups.size()); ++g) {
      const auto& group =
          kv_cache_config.kv_cache_groups[static_cast<size_t>(g)];
      const KVCacheSpecKind kind = group.kv_cache_spec->kind();
      const bool is_attn =
          dynamic_cast<const AttentionSpec*>(group.kv_cache_spec.get()) !=
          nullptr;
      const bool is_recurrent = kind == KVCacheSpecKind::kMamba;
      if (is_recurrent) ++recurrent_seen;
      std::string why;
      if (!is_attn && !is_recurrent) {
        why = "spec is neither an AttentionSpec nor a MambaSpec";
      } else if (is_recurrent && recurrent_seen > 1) {
        why = "a SECOND recurrent group";
      } else if (group.is_eagle_group) {
        why =
            "an EAGLE draft group, which the multi-cache path does not "
            "allocate";
      } else {
        const std::optional<std::vector<bool>> mask =
            GroupLayerMask(group, num_layers_for_groups);
        if (!mask.has_value()) {
          why =
              "its published layer names do not all resolve to distinct "
              "in-range layer indices";
        } else {
          // Resolve the per-name layer indices ONCE, in the group's own order,
          // so the allocation below never parses a name a second time.
          std::vector<int32_t>& idx = group_layer_index[static_cast<size_t>(g)];
          idx.reserve(group.layer_names.size());
          for (const std::string& name : group.layer_names) {
            idx.push_back(static_cast<int32_t>(*KVCacheLayerIndexOfName(name)));
          }
        }
      }
      if (why.empty()) continue;
      ++unallocated_count;
      unallocated += "\n  group ";
      unallocated += std::to_string(g);
      unallocated += " kind=";
      unallocated += KVCacheSpecKindName(kind);
      unallocated += " layers=";
      unallocated += std::to_string(group.layer_names.size());
      unallocated += " first='";
      unallocated += group.layer_names.empty() ? std::string("<unnamed>")
                                               : group.layer_names.front();
      unallocated += "' page_size_bytes=";
      unallocated +=
          std::to_string(group.kv_cache_spec->page_size_bytes());
      unallocated += " (";
      unallocated += why;
      unallocated += ")";
    }
    VT_CHECK(unallocated_count == 0,
             std::string("runner: ") + std::to_string(unallocated_count) +
                 " published KV cache group(s) get NO cache from this runner. "
                 "Refusing rather than allocating a SUBSET of the published "
                 "topology in silence. A multi-cache topology can carry any "
                 "number of non-eagle AttentionSpec groups and ONE MambaSpec "
                 "group, and every group has to publish per-layer names that "
                 "resolve. "
                 "Unallocated:" +
                 unallocated +
                 "\n(row KV-DSV4-MULTICACHE; #1973, #2068, #2084)");
  }

  // Allocate one PagedKvCache per full-attn layer and one GdnStateCache per GDN
  // layer, in LAYER ORDER (matches Qwen3_5Model::Forward's per-layer fa_idx /
  // gdn_idx indexing). As in upstream, MambaSpec is the source of truth for the
  // recurrent tensors' order, shapes, dtypes, and page bytes.
  //
  // #810: SPEC-DRIVEN, exactly like the attention half below. This block used
  // to rebuild the geometry from `config_.linear_num_key_heads` /
  // `linear_conv_kernel_dim` and then REFUSE when the model's own published
  // spec disagreed with that reconstruction — so the comment above stated a
  // polarity the code did not have, and every hybrid whose config does not
  // speak Qwen3.5's `linear_*` dialect (NemotronH ships `mamba_num_heads`,
  // `mamba_head_dim`, `ssm_state_size`, `conv_kernel`) was refused by Qwen3.5's
  // name for shapes it never claimed.
  //
  // Upstream cannot express that cross-check at all. Its runner allocates raw
  // BYTES (`gpu_model_runner.py:7311-7313` `torch.zeros(..., dtype=torch.int8)`)
  // and hands the Mamba layer an untyped page view (`:7429-7441`); the unpack
  // into typed conv/SSM views happens in the LAYER
  // (`mamba/abstract.py:38-43`: `nbytes = prod(shape) * get_dtype_size(dtype)`).
  // The runner never holds a conv shape, so it has nothing to compare. That is
  // why the check is DELETED rather than widened or put behind a per-arch
  // switch: it was a SECOND, independent derivation of a number the model had
  // already published, and a second derivation is the thing that can disagree.
  // It is also why `MambaSpec` (`kv_cache_interface.py:690-703`, mirrored at
  // `include/vllm/v1/kv_cache_interface.h:302-323`) deliberately carries no
  // num_heads/head_dim/conv_dim: the geometry IS `shapes`.
  //
  // SPEC-MTP I5d is now free: the conv row widened to (K-1)+num_spec under
  // speculation is simply what `MakeQwen3_5KVCacheSpec` publishes
  // (mamba_utils.py:226), so reading the spec picks it up with no spec_on()
  // branch here at all.
  //
  // ENG-RECURRENT-MULTISTATE (#2131): the state list is N long, not TWO.
  // This block used to read `shapes[0]` and `shapes[1]` behind a
  // `shapes.size() == 2` refusal, and a recurrent layer in this tree could
  // therefore not hold a third state at all. Upstream has no such assumption
  // and never had one: `MambaBase.kv_cache` is `tuple[torch.Tensor, ...]`
  // (`vllm/model_executor/layers/mamba/abstract.py:26`) and `bind_kv_cache`
  // (`:29-43`) unpacks ONE page into as many states as
  // `zip(get_state_shape(), get_state_dtype())` yields, each with its own shape
  // and its own dtype at a running byte offset. Three values of N ship at the
  // pin `5559679229`: 1 (`short_conv.py:87`), 2, and 5 (`mamba_mixer2.py:517-520`,
  // whose ring states are rank 3 / rank 2 / rank 3 with a `torch.float32`
  // between two activation dtypes, `mamba_utils.py:84-93` and `:202-221`).
  // The runner side is dtype- and count-blind by construction: it allocates
  // `num_blocks * page_size_bytes` raw int8 and hands the layer one untyped
  // page (`gpu_model_runner.py:7429-7440`).
  //
  // The loop below is that zip. `state_shapes` / `state_dtypes` /
  // `state_row_elems` are parallel and in SPEC ORDER, which is the order
  // `bind_kv_cache` slices in and the order every buffer and view below is
  // built in.
  std::vector<std::vector<int64_t>> state_shapes;  // per SLOT, spec order
  std::vector<vt::DType> state_dtypes;             // parallel to state_shapes
  std::vector<int64_t> state_row_elems;            // parallel to state_shapes

  const MambaSpec* mamba_spec = nullptr;
  if (gdn_group_id_ >= 0) {
    mamba_spec = dynamic_cast<const MambaSpec*>(
        kv_cache_config.kv_cache_groups[static_cast<size_t>(gdn_group_id_)]
            .kv_cache_spec.get());
    VT_CHECK(mamba_spec != nullptr,
             "runner: recurrent cache group must carry a MambaSpec");
    // `>= 2` rather than `== 2`, and the lengths must agree. The widening is
    // justified by the upstream anchor above and by expressibility, and by
    // nothing else: it closes no bug. A two-shape/one-dtype spec was already
    // refused before this row, by the CONJUNCTIVE `shapes.size() == 2 &&
    // dtypes.size() == 2` that stood here, and again by the
    // `shapes.size() != dtypes.size()` throw in `MambaSpec::page_size_bytes`
    // (`src/vllm/v1/kv_cache_interface.cpp:210-213`) ahead of its own zip.
    // The lower bound stays 2 because `GdnStateCache` still publishes
    // `conv_state` and `ssm_state` as named fields that every model consumer in
    // this tree reads; a one-state group (upstream `ShortConv`) is refused
    // rather than truncated, and is recorded under `## Owed` in
    // `.agents/specs/recurrent-multistate.md`.
    VT_CHECK(mamba_spec->shapes.size() >= 2 &&
                 mamba_spec->shapes.size() == mamba_spec->dtypes.size(),
             "runner: recurrent MambaSpec must carry at least a conv and a "
             "temporal state, with one dtype per shape");
    const size_t num_states = mamba_spec->shapes.size();
    state_shapes.reserve(num_states);
    state_dtypes.reserve(num_states);
    state_row_elems.reserve(num_states);
    int64_t page_bytes = 0;
    for (size_t i = 0; i < num_states; ++i) {
      const std::vector<int64_t>& shape = mamba_spec->shapes[i];
      const vt::DType dtype = mamba_spec->dtypes[i];
      // The GdnStateCache view prepends the SLOT dim, so a state shape may
      // carry at most kMaxRank-1 dims. Refuse rather than silently truncate.
      VT_CHECK(!shape.empty() &&
                   shape.size() < static_cast<size_t>(vt::kMaxRank),
               "runner: MambaSpec state shapes must be 1..kMaxRank-1 dims");
      int64_t row = 1;
      for (int64_t d : shape) row *= d;
      VT_CHECK(row > 0, "runner: MambaSpec state shapes must be positive");
      // Upstream imposes NO dtype constraint here: `bind_kv_cache` reinterprets
      // the page bytes with a bare `.view(dtype)`. The predicate this tree used
      // to carry — floating only — was justified by "all-zero bytes are +0.0f
      // for every supported floating storage type", which is equally true of an
      // integer zero, and it made an INTEGER state inexpressible (a token-id
      // history holds `input_ids.long()`, not activations). What genuinely
      // cannot work is a BLOCK-QUANT encoding, which has no per-element size at
      // all, so that is what the predicate names.
      VT_CHECK(!vt::IsBlockQuant(dtype),
               "runner: a recurrent MambaSpec state dtype has no per-element "
               "size (block-quantized encodings are storage-only)");
      page_bytes += row * static_cast<int64_t>(vt::SizeOf(dtype));
      state_shapes.push_back(shape);
      state_dtypes.push_back(dtype);
      state_row_elems.push_back(row);
    }
    gdn_conv_cache_dtype_ = state_dtypes[0];
    gdn_ssm_cache_dtype_ = state_dtypes[1];
    // The per-slot byte cost the allocator will use IS the spec's page size —
    // upstream's `MambaSpec.page_size_bytes` is the sum of `prod(shape) *
    // dtype_size` over the state tensors (`kv_cache_interface.py:698-707`).
    // Assert the identity rather than re-deriving it anywhere else. Summed over
    // ALL N states, so a third state that the allocator forgot would show up
    // here instead of as a short allocation nothing reports.
    VT_CHECK(page_bytes == mamba_spec->page_size_bytes(),
             "runner: MambaSpec page_size_bytes disagrees with its own "
             "shapes and dtypes");
  }

  // SPEC-DRIVEN attention-cache sizing and layout (MLA campaign W1).
  //
  // Upstream never reconstructs the KV shape from the HF config: every layer's
  // cache bytes come from its SPEC (`vllm/v1/kv_cache_interface.py:380-398`
  // `real_page_size_bytes` -> `page_size_bytes`), and the tensor shape comes
  // from the BACKEND (`get_kv_cache_shape`). That abstraction is exactly why
  // `vllm/v1/worker/gpu_model_runner.py` contains **no `use_mla` branch at all**
  // (only an import at :58, an isinstance at :977 and comments at :1085/:7133)
  // even though MLA's cache is 1-head, 576-wide and has NO separate V — MLA is
  // registered against the ordinary `FullAttentionManager`
  // (`vllm/v1/core/single_type_kv_cache_manager.py:1539`), so block table,
  // prefix caching and eviction are untouched and the whole cost lands in the
  // ALLOCATOR and the ops.
  //
  // We previously hardcoded `num_blocks * 2 * block * Hkv * Dh` with Hkv/Dh
  // read from `config_` — the factor 2 (K+V) and the HF-config reconstruction
  // are the two things MLA cannot express. Now: bytes come from
  // `spec->page_size_bytes()` and the cache VIEW comes from the spec's own
  // fields. Behaviour-preserving by construction for every existing model —
  // `FullAttentionSpec::real_page_size_bytes` is
  // `block * num_kv_heads * (head_size + head_size_v) * es`, which with
  // `head_size_v == head_size` is byte-for-byte the old expression.
  int64_t fa_block_size = 0;
  int64_t Hkv = 0;
  int64_t Dh = 0;
  int64_t fa_page_bytes = 0;
  vt::DType kv_dtype = ResolveKvCacheDType();
  // KV-FP8 W3: the group-level fp8 interpretation + scales, defaulting to the
  // inert float path. Read off the SAME AttentionSpec as `kv_dtype` below.
  vt::Fp8KVCacheDataType kv_fp8_kind = vt::Fp8KVCacheDataType::kAuto;
  float kv_k_scale = 1.0F;
  float kv_v_scale = 1.0F;
  if (full_attn_group_id_ >= 0) {
    const KVCacheSpec* fa_spec =
        kv_cache_config.kv_cache_groups[static_cast<size_t>(full_attn_group_id_)]
            .kv_cache_spec.get();
    const auto* attn_spec = dynamic_cast<const AttentionSpec*>(fa_spec);
    VT_CHECK(attn_spec != nullptr,
             "runner: full-attention cache group must carry an AttentionSpec");
    fa_block_size = attn_spec->block_size;
    Hkv = attn_spec->num_kv_heads;
    Dh = attn_spec->head_size;
    kv_dtype = attn_spec->dtype;
    kv_fp8_kind = attn_spec->fp8_kind;
    kv_k_scale = attn_spec->k_scale;
    kv_v_scale = attn_spec->v_scale;
    fa_page_bytes = attn_spec->page_size_bytes();
    // MiMoV2: asymmetric V head dim is now carried through PagedKvCache's
    // `head_size_v` field, so the guard that refused it is removed.
    [[maybe_unused]] int64_t Dh_v = attn_spec->head_size;  // default: symmetric
    if (const auto* full_spec = dynamic_cast<const FullAttentionSpec*>(fa_spec)) {
      Dh_v = full_spec->head_size_v;
    }
    VT_CHECK(fa_page_bytes > 0,
             "runner: full-attention spec reported a non-positive page size");

    // ENGINE-LEVEL ATTENTION-BACKEND SELECTION (M3, issue #41) is the first
    // runtime call of the selection seam, and it happens PER GROUP in the view
    // loop below — lazily per kind, inside the full-attn region, never for a
    // pure-GDN / pooling model (which has no full-attn groups and therefore no
    // paged KV to validate).
    //
    // Dense: LOUD. A model with dense full-attention groups needs a dense
    // backend; a platform whose priority list yields none (how Vulkan and ROCm
    // started) fails at init instead of silently running unlabelled. On ROCm
    // this resolves "ROCM_ATTN" (backend.cpp, M3); on CPU "CPU_ATTN"
    // (cpu_attn.cpp, issue #1371 — upstream's own CPU answer at cpu.py:75-87);
    // on CUDA/Metal/Vulkan "FLASH_ATTN". All four names report the same NHD KV
    // layout, which is the layout every one of those device kernels reads — the
    // name changed on CPU, the geometry validated below did not.
    // Mirrors upstream resolving get_attn_backend_cls per attention layer
    // (gpu_model_runner.py:6994-7099); we group by KV-cache kind because this
    // engine allocates exactly one layout per kind.
    //
    // MLA: TOLERANT. The engine executes MLA through TritonMLAImpl on a fused
    // 3-dim cache regardless of the registry (deepseek_v2.cpp:576-578), so on a
    // device with no registered MLA backend (CPU, ROCm today) the name stays
    // empty and the group keeps running op-driven — a loud throw would regress
    // working MLA paths. On CUDA this resolves "TRITON_MLA", whose
    // get_kv_cache_shape is exactly the fused view the engine allocates.

    // Positive signal that the SPEC (not the HF config) drove this allocation:
    // opt-in, one line, never on the hot path.
    if (const char* dbg = std::getenv("VT_KV_ALLOC_LOG");
        dbg != nullptr && dbg[0] == '1') {
      std::fprintf(stderr,
                   "[kv-alloc] source=spec kind=%d block_size=%lld "
                   "num_kv_heads=%lld head_size=%lld dtype=%d "
                   "page_size_bytes=%lld num_blocks=%lld\n",
                   static_cast<int>(fa_spec->kind()),
                   static_cast<long long>(fa_block_size),
                   static_cast<long long>(Hkv), static_cast<long long>(Dh),
                   static_cast<int>(kv_dtype),
                   static_cast<long long>(fa_page_bytes),
                   static_cast<long long>(num_blocks_));
    }
  }
  // Recorded for the gates: the exact per-block byte cost the allocator used,
  // sourced from the spec. `fa_page_size_bytes() > 0` is the runtime proof that
  // the spec-driven path RAN (a compiled-but-unexercised path leaves it 0).
  fa_page_size_bytes_ = fa_page_bytes;

  const vt::Device dev = queue_.device;
  const char* device_cache_env = std::getenv("VT_DEVICE_KV_CACHE");
  // W0b-1 / work row M3a: this read `is_cuda()`, which is the SAME defect the
  // `dense_attn_block.h` ResidentWeight fix corrected — "not NVIDIA" was being
  // used to mean "no device memory". On kMETAL (and kVULKAN, kXPU) the KV cache
  // fell into `host_data_`, and the first device kernel to touch it —
  // vt::ReshapeAndCache — was handed a HOST pointer. On Metal that surfaces as
  // "k_cache points outside every Metal allocation"; on a discrete-memory
  // backend it would be a silent wrong answer or a fault.
  //
  // Device residency is a property of HAVING a device, not of the vendor, so the
  // predicate is `!is_cpu()`. kCPU keeps the host vector (it has no device pool
  // and the host IS the device) and kCUDA behaviour is bit-identical — the dgx
  // regression set is the evidence, not this comment.
  kv_cache_backend_resident_ =
      !vllm::platforms::GetPlatform(dev.type).is_cpu() &&
      (device_cache_env == nullptr || device_cache_env[0] != '0');
  full_attn_buf_.clear();
  recurrent_state_buf_.clear();
  // A full-attention-only model (e.g. dense Qwen3ForCausalLM) has NO
  // linear-attention (GDN/Mamba) KV group and an EMPTY layer_types — indexing
  // layer_types[l] would be out of bounds. Drive "is this layer GDN?" off the
  // resolved KV-group structure (gdn_group_id_ >= 0 ⇔ the model has a mamba
  // group), then the per-layer layer_types tag. This is model-shape-agnostic:
  // the hybrid gate models keep a GDN group, so their path is byte-identical.
  const bool has_mamba_group = gdn_group_id_ >= 0;
  // #810: PER-LAYER MEMBERSHIP FROM THE PUBLISHED GROUP, when the model
  // publishes one. `KVCacheGroupSpec::layer_names` is what upstream keys
  // per-layer KV on end to end (see LayerIndexOfName above); the runner keyed
  // on `config_.layer_types[l] == "linear_attention"` instead, which is a
  // Qwen3.5 config spelling. NemotronH's `layer_types` is EMPTY, so all 52
  // layers classified as full attention: zero recurrent buffers and ~8.7x the
  // attention pages actually needed (52 against 6 real GQA layers).
  //
  // BYTE-NEUTRALITY CONTRACT, mirroring the one `per_layer_attn_specs` states
  // at `include/vllm/v1/kv_cache_interface.h`: the by-name path is entered ONLY
  // when the recurrent group — and the target attention group, if there is one —
  // publish per-layer names that all resolve to distinct in-range layer indices.
  // This is a capability probe on the record the model published, NOT a
  // per-architecture switch: a hybrid that publishes real names is routed
  // correctly with no new branch, which is the whole point
  // (`hf_config.cpp:484-528` synthesizing Qwen3.5's dialect for Kimi-Linear is
  // the anti-pattern this replaces).
  //
  // FIX-KV-GROUP-LAYER-COUNT (#1963, #1966): the registries still publish a
  // single PLACEHOLDER name per group ("fa"/"gdn", "mla"/"kda", "fa_draft"),
  // but `ResolveKVCacheGroupLayerNames` now rewrites those into real per-layer
  // names in the LOADER, before the config reaches this function, using exactly
  // the `layer_types` predicate the fallback below uses. So a hybrid arrives
  // here named and takes the by-name path, and the classification it gets is
  // the one the fallback would have produced — which is what makes that
  // rewrite byte-neutral for the allocation and, at the same time, what makes
  // `KVBytesPerBlock` and `recurrent_state_bytes` count the layers this loop
  // actually allocates for. Before that fix they counted ONE, and a 1 GiB
  // budget bought 8.5 GiB of buffers.
  const int64_t num_layers = config_.num_hidden_layers;
  std::optional<std::vector<bool>> gdn_layer_mask;
  std::optional<std::vector<bool>> attn_layer_mask;
  if (has_mamba_group) {
    gdn_layer_mask = GroupLayerMask(
        kv_cache_config.kv_cache_groups[static_cast<size_t>(gdn_group_id_)],
        num_layers);
    // KV-DSV4-MULTICACHE W3 (#2068): on the multi-cache path there is no single
    // "the attention group" to cross-check against — every attention group names
    // its own layers and the representability check above already proved all of
    // them resolve. The cross-check below is the LEGACY path's, unchanged.
    if (!multi_cache_topology && gdn_layer_mask.has_value() &&
        full_attn_group_id_ >= 0) {
      attn_layer_mask =
          GroupLayerMask(kv_cache_config
                             .kv_cache_groups[static_cast<size_t>(
                                 full_attn_group_id_)],
                         num_layers);
      // A recurrent group that names its layers next to an attention group that
      // does not leaves the non-recurrent layers unclassifiable. Fall back
      // wholesale rather than guess.
      if (!attn_layer_mask.has_value()) gdn_layer_mask.reset();
    }
  }
  const bool membership_by_name = gdn_layer_mask.has_value();
  if (membership_by_name && attn_layer_mask.has_value()) {
    for (int64_t l = 0; l < num_layers; ++l) {
      VT_CHECK(!((*gdn_layer_mask)[static_cast<size_t>(l)] &&
                 (*attn_layer_mask)[static_cast<size_t>(l)]),
               "runner: a layer is named by BOTH the attention and the "
               "recurrent KV cache group");
    }
  }
  // PER-LAYER KV head_dim (Gemma-4 G1b). When the model publishes a per-layer
  // attention spec (heterogeneous head_dim: sliding 256 / global 512), each
  // non-GDN layer allocates + views its OWN head_size/num_kv_heads/page_size.
  // EMPTY for every uniform-KV model => `has_per_layer` is false => the loop
  // uses the single group spec (Hkv/Dh/fa_page_bytes) for every layer, which is
  // byte-identical to before this field existed (proven by the CPU paged-engine
  // suite staying green). The block table / manager are head_dim-independent, so
  // a single group + per-layer allocation is correct with no per-group changes.
  const bool has_per_layer = !kv_cache_config.per_layer_attn_specs.empty();
  if (has_per_layer) {
    VT_CHECK(kv_cache_config.per_layer_attn_specs.size() ==
                 static_cast<size_t>(config_.num_hidden_layers),
             "runner: per_layer_attn_specs must have one entry per hidden layer");
  }
  // Per-full-attn-buffer view geometry, parallel to full_attn_buf_ (built here
  // so the views loop below need not re-derive per-layer dims). In the uniform
  // case every entry is {Hkv, Dh, kv_dtype} — identical to today.
  struct FaDims {
    int64_t num_kv_heads;
    int64_t head_size;
    // MiMoV2: V head dim can differ from K head dim. 0 means
    // `head_size` (the byte-identical legacy path every existing model
    // takes).
    int64_t head_size_v;
    vt::DType dtype;
    // KV-DSV4-MULTICACHE W3 (#2068): the entry's OWN page geometry. On the
    // legacy path every entry gets the single `fa_block_size` this loop already
    // used, so the view built from it is byte-identical; on the multi-cache path
    // DeepSeek-V4 needs 256, 64, 8 and 4 in one model.
    int64_t block_size;
    // KV-FP8 W3: the fp8 interpretation + per-tensor scales, carried from the
    // same spec that supplied `dtype` and `page_size_bytes()`. They travel with
    // the dtype because the page width and the byte's meaning are one decision.
    vt::Fp8KVCacheDataType fp8_kind;
    float k_scale;
    float v_scale;
    // KV-DSV4-MULTICACHE W8 slice 4 (#2455): the entry's OWN page in bytes, the
    // same `page_size_bytes()` this loop already spends on the allocation. It
    // travels beside the view geometry because the two disagree for any spec
    // whose page is not `block_size * head_size * sizeof(dtype)` — see the
    // field's comment on `PagedKvCache`.
    int64_t page_size_bytes;
  };
  std::vector<FaDims> fa_dims;
  // Parallel to fa_dims: 1 when the layer's spec kind is kMlaAttention (the
  // fused 3-dim cache view) vs 0 for a dense NHD layer.
  std::vector<char> mla_layer_mask;
  // KV-DSV4 PER-GROUP DISPATCH (ISSUE-LOCAL-01M2EMPC6T63TVDPQ90GVPRC5F):
  // parallel to `fa_dims` / `mla_layer_mask`, one entry per allocated cache —
  // the backend the GROUP names, i.e. upstream's `AttentionGroupKey.attn_backend`
  // taken from the layer (`gpu_model_runner.py:7150`, `:7170`). EMPTY for every
  // group that does not name one, which is every group of every other model, so
  // the selector path below is entered exactly as often as it was before.
  std::vector<std::string> named_backend;
  layer_kv_class_.assign(static_cast<size_t>(num_layers), LayerKvClass::kNone);
  layer_attn_kv_indices_.clear();
  attn_kv_layer_names_.clear();
  kv_index_layer_names_.clear();
  kv_index_group_ids_.clear();
  kv_index_layer_indices_.clear();
  kv_index_payload_kinds_.clear();
  kv_index_payload_slots_.clear();
  multi_cache_topology_ = multi_cache_topology;
  multi_kv_index_ = vllm::MultiKvCacheIndex{};
  group_block_tables_.clear();
  group_block_table_cols_.clear();

  // ── KV-DSV4-MULTICACHE W3 (#2068): ONE BUFFER PER PUBLISHED CACHE ──────────
  //
  // Upstream allocates per registered layer NAME, never per model layer: a cache
  // is an `AttentionLayerBase` registered under its own prefix and
  // `get_kv_cache_spec()` returns a `dict[str, KVCacheSpec]` keyed by that prefix
  // (`vllm/v1/worker/gpu_model_runner.py:7785-7801`). One DeepSeek-V4 C4A layer
  // therefore contributes FOUR entries — the compressed MLA latent
  // (`vllm/models/deepseek_v4/attention.py:626-645`), the sliding-window cache
  // (`:315-321`), the indexer key cache (`:761-767`) and the compressor state
  // (`vllm/models/deepseek_v4/compressor.py:290-295`) — and the loop below is
  // that shape.
  //
  // ENTERED ONLY on a multi-cache topology (see the entry predicate above), so
  // no model shipping today reaches it.
  //
  // ENG-MULTIKV-BYNAME extends it to the RECURRENT half. W3 built the by-name
  // index off `attn_group_ids_`, so a `MambaSpec` group's layers were allocated
  // (below) and then named nowhere: their states reached the forward through
  // `gdn_state` positionally and `Find()` answered -1 for every one of them.
  // Upstream has no such split — the Mamba page goes into the SAME
  // `kv_caches: dict[str, torch.Tensor]` as every attention cache
  // (`gpu_model_runner.py:7429-7441`, "Keeping one tensor per layer lets the KV
  // connector register it without special-casing Mamba"), and `:7318-7326`
  // asserts that dict's keys are every layer name of every published group.
  std::vector<int32_t> recurrent_slot_of_layer;
  if (multi_cache_topology) {
    layer_attn_kv_indices_.assign(static_cast<size_t>(num_layers),
                                  std::vector<int32_t>{});
    // The recurrent group first, in LAYER order, sized exactly as the legacy
    // path sizes it. Nothing in the tree publishes a recurrent group beside a
    // multi-cache attention set; it is carried rather than refused because
    // expressing it costs one loop and refusing it would be a hole.
    if (membership_by_name && has_mamba_group) {
      recurrent_slot_of_layer.assign(static_cast<size_t>(num_layers), -1);
      for (int64_t l = 0; l < num_layers; ++l) {
        if (!(*gdn_layer_mask)[static_cast<size_t>(l)]) continue;
        VT_CHECK(mamba_spec != nullptr,
                 "runner: linear-attention layer has no MambaSpec");
        layer_kv_class_[static_cast<size_t>(l)] = LayerKvClass::kRecurrent;
        // ENG-MULTIKV-BYNAME: the slot this layer's state lands in, recorded
        // BEFORE the push so it is the index of the entry about to be created.
        // `gdn_state_` is built one-for-one from `recurrent_state_buf_` at the
        // end of this function, so this IS the index the by-name channel hands
        // a forward. It is deliberately not `l`: the mask is over the model's
        // layers and the buffer is over the recurrent ones, and on every hybrid
        // in the tree the two differ.
        recurrent_slot_of_layer[static_cast<size_t>(l)] =
            static_cast<int32_t>(recurrent_state_buf_.size());
        alloc_recurrent_layer_states(dev, state_dtypes, state_row_elems);
      }
    }
    // Then one paged buffer per (group x published name), in PUBLICATION order,
    // each sized and viewed from its OWN group's spec. `page_size_bytes()` is
    // the single allocation accessor every `AttentionSpec` inherits and none
    // overrides (`vllm/v1/kv_cache_interface.py:196-201`, on `AttentionSpec`),
    // which is why nothing here switches on the kind except the fused-view flag
    // — the accessor is uniform even though the value it returns is not.
    for (int g : attn_group_ids_) {
      const auto& group = kv_cache_config.kv_cache_groups[static_cast<size_t>(g)];
      const auto* spec =
          dynamic_cast<const AttentionSpec*>(group.kv_cache_spec.get());
      VT_CHECK(spec != nullptr, "runner: attention group lost its AttentionSpec");
      const int64_t page = spec->page_size_bytes();
      VT_CHECK(page > 0,
               "runner: a published attention group reported a non-positive "
               "page size");
      // `SlidingWindowMLASpec` holds ONE vector rather than K + V
      // (`vllm/models/deepseek_v4/compressor.py:193`, and W1's
      // `real_page_size_bytes` multiplies `head_size` alone for it), so it takes
      // the FUSED view exactly as `MLAAttentionSpec` does. Unreachable on the
      // legacy path, where a `kSlidingWindowMla` group is a leftover.
      const bool fused = spec->kind() == KVCacheSpecKind::kMlaAttention ||
                         spec->kind() == KVCacheSpecKind::kSlidingWindowMla;
      const std::vector<int32_t>& idx = group_layer_index[static_cast<size_t>(g)];
      VT_CHECK(idx.size() == group.layer_names.size(),
               "runner: group layer indices out of sync with its names");
      for (size_t k = 0; k < group.layer_names.size(); ++k) {
        const int32_t l = idx[k];
        VT_CHECK(layer_kv_class_[static_cast<size_t>(l)] !=
                     LayerKvClass::kRecurrent,
                 "runner: a layer is named by BOTH an attention and the "
                 "recurrent KV cache group");
        layer_attn_kv_indices_[static_cast<size_t>(l)].push_back(
            static_cast<int32_t>(full_attn_buf_.size()));
        layer_kv_class_[static_cast<size_t>(l)] = LayerKvClass::kMultiCache;
        attn_kv_layer_names_.push_back(group.layer_names[k]);
        full_attn_buf_.push_back(std::make_unique<CacheBuffer>(
            dev, queue_,
            static_cast<size_t>(num_blocks_) * static_cast<size_t>(page),
            kv_cache_backend_resident_));
        fa_dims.push_back(FaDims{spec->num_kv_heads, spec->head_size,
                                 [&]() -> int64_t {
                                   if (const auto* fs = dynamic_cast<const FullAttentionSpec*>(spec))
                                     return static_cast<int64_t>(fs->head_size_v);
                                   if (const auto* sw = dynamic_cast<const SlidingWindowSpec*>(spec))
                                     return static_cast<int64_t>(sw->head_size_v);
                                   return static_cast<int64_t>(spec->head_size);
                                 }(),
                                 spec->dtype, spec->block_size, spec->fp8_kind,
                                 spec->k_scale, spec->v_scale, page});
        mla_layer_mask.push_back(static_cast<char>(fused));
        named_backend.push_back(group.attn_backend);
      }
    }

    // ── ENG-MULTIKV-BYNAME: THE BY-NAME INDEX OVER EVERY PUBLISHED CACHE ────
    //
    // One pass over the groups in PUBLICATION order, emitting each group's
    // names in the group's own order. That is upstream's insertion order for
    // its single `kv_caches` dict
    // (`vllm/v1/worker/gpu_model_runner.py:7365-7372`, `for group in ...: for
    // layer_name in group.layer_names`), and it is why a recurrent group
    // published between two attention groups lands BETWEEN them here rather
    // than being appended: an entry's index is its place among the published
    // caches, not its slot in a payload container.
    //
    // EVERY group is one of the two arms. The refusal above already rejected a
    // spec that is neither an `AttentionSpec` nor a `MambaSpec`, a second
    // recurrent group, an eagle group and a group whose names do not resolve,
    // so `group_layer_index[g]` is filled for every group that reaches here.
    // The `VT_CHECK`s restate that rather than assume it, because a future
    // shape that slips past the refusal must land as a refusal here and not as
    // a cache nobody can address.
    {
      int32_t paged_slot = 0;
      for (int g = 0;
           g < static_cast<int>(kv_cache_config.kv_cache_groups.size()); ++g) {
        const auto& group =
            kv_cache_config.kv_cache_groups[static_cast<size_t>(g)];
        const bool is_recurrent =
            group.kv_cache_spec->kind() == KVCacheSpecKind::kMamba;
        VT_CHECK(is_recurrent || dynamic_cast<const AttentionSpec*>(
                                     group.kv_cache_spec.get()) != nullptr,
                 "runner: a published group on the multi-cache path is neither "
                 "an AttentionSpec nor a MambaSpec group");
        const std::vector<int32_t>& idx =
            group_layer_index[static_cast<size_t>(g)];
        VT_CHECK(idx.size() == group.layer_names.size(),
                 "runner: group layer indices out of sync with its names");
        for (size_t k = 0; k < group.layer_names.size(); ++k) {
          kv_index_layer_names_.push_back(group.layer_names[k]);
          kv_index_group_ids_.push_back(static_cast<int32_t>(g));
          kv_index_layer_indices_.push_back(idx[k]);
          if (is_recurrent) {
            const int32_t slot =
                recurrent_slot_of_layer.empty()
                    ? -1
                    : recurrent_slot_of_layer[static_cast<size_t>(idx[k])];
            VT_CHECK(slot >= 0,
                     "runner: a published recurrent cache was given no state "
                     "slot, so no forward could address it by name");
            kv_index_payload_kinds_.push_back(
                static_cast<uint8_t>(vllm::KvCachePayload::kRecurrent));
            kv_index_payload_slots_.push_back(slot);
          } else {
            kv_index_payload_kinds_.push_back(
                static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
            kv_index_payload_slots_.push_back(paged_slot++);
          }
        }
      }
      // The paged half of the index, in order, IS `attn_kv_layer_names_` — two
      // derivations of one order, and a second derivation is the thing that can
      // disagree. The allocation loop above walks `attn_group_ids_`; this pass
      // walks every group and filters. They agree because the refusal makes the
      // two sets equal, and this states that agreement as a refusal rather than
      // leaving it as a property of two loops nobody compares.
      size_t j = 0;
      for (size_t i = 0; i < kv_index_layer_names_.size(); ++i) {
        if (static_cast<vllm::KvCachePayload>(kv_index_payload_kinds_[i]) !=
            vllm::KvCachePayload::kPaged)
          continue;
        VT_CHECK(j < attn_kv_layer_names_.size() &&
                     kv_index_layer_names_[i] == attn_kv_layer_names_[j] &&
                     kv_index_payload_slots_[i] == static_cast<int32_t>(j),
                 "runner: the by-name index's paged half disagrees with the "
                 "order attn_kv was allocated in");
        ++j;
      }
      VT_CHECK(j == attn_kv_layer_names_.size(),
               "runner: the by-name index does not cover every allocated paged "
               "cache");
    }
  } else {
    // THE LEGACY PATH, byte-for-byte. Every model shipping today lands here.
    for (int64_t l = 0; l < num_layers; ++l) {
      bool is_gdn = false;
      bool is_full_attn = false;
      if (membership_by_name) {
        is_gdn = (*gdn_layer_mask)[static_cast<size_t>(l)];
        is_full_attn = !is_gdn && attn_layer_mask.has_value() &&
                       (*attn_layer_mask)[static_cast<size_t>(l)];
      } else {
        // The historical predicate, unchanged. A full-attention-only model (e.g.
        // dense Qwen3ForCausalLM) has NO recurrent group and an EMPTY
        // layer_types — indexing layer_types[l] would be out of bounds.
        is_gdn = has_mamba_group && !config_.layer_types.empty() &&
                 config_.layer_types[static_cast<size_t>(l)] ==
                     "linear_attention";
        is_full_attn = !is_gdn;
      }
      layer_kv_class_[static_cast<size_t>(l)] =
          is_gdn ? LayerKvClass::kRecurrent
                 : (is_full_attn ? LayerKvClass::kFullAttention
                                 : LayerKvClass::kNone);
      if (is_gdn) {
        VT_CHECK(mamba_spec != nullptr,
                 "runner: linear-attention layer has no MambaSpec");
        // Raw buffers use their independent per-state cache dtypes. Zero bytes
        // are +0.0f in every floating storage type and integer 0 in every
        // integer one. Per-slot element counts come from the SPEC's shapes
        // (#810), not from HF-config arithmetic, and there are as many of them
        // as the spec published (#2131), not two.
        alloc_recurrent_layer_states(dev, state_dtypes, state_row_elems);
      } else if (is_full_attn) {
        // Bytes come from the SPEC, not from HF-config arithmetic: exactly
        // `num_blocks * spec->page_size_bytes()`, mirroring upstream's
        // `kv_cache_interface.py:380-398` sizing contract. For a symmetric
        // FullAttentionSpec this is byte-identical to the old
        // `num_blocks * 2 * block * Hkv * Dh * sizeof(kv_dtype)`; for a future
        // MLAAttentionSpec it drops the factor 2 with no allocator change.
        // 0 bytes == 0.0 in both bf16 and f32.
        //
        // PER-LAYER (Gemma-4 G1b): the layer's own spec supplies its page bytes +
        // view geometry when published; otherwise the single group spec (uniform
        // path). `l_page` collapses to `fa_page_bytes` and `{l_Hkv,l_Dh,l_dtype}`
        // to `{Hkv,Dh,kv_dtype}` when `has_per_layer` is false — byte-identical.
        int64_t l_Hkv = Hkv;
        int64_t l_Dh = Dh;
        int64_t l_Dh_v = Dh;  // MiMoV2: asymmetric V head dim (default = symmetric)
        int64_t l_page = fa_page_bytes;
        vt::DType l_dtype = kv_dtype;
        vt::Fp8KVCacheDataType l_fp8_kind = kv_fp8_kind;
        float l_k_scale = kv_k_scale;
        float l_v_scale = kv_v_scale;
        if (has_per_layer) {
          const std::shared_ptr<AttentionSpec>& sp =
              kv_cache_config.per_layer_attn_specs[static_cast<size_t>(l)];
          VT_CHECK(sp != nullptr,
                   "runner: non-GDN layer has no per-layer attention spec");
          l_Hkv = sp->num_kv_heads;
          l_Dh = sp->head_size;
          l_dtype = sp->dtype;
          l_fp8_kind = sp->fp8_kind;
          l_k_scale = sp->k_scale;
          l_v_scale = sp->v_scale;
          l_page = sp->page_size_bytes();
          // MiMoV2: asymmetric V head dim is now carried through
          // PagedKvCache's `head_size_v` field, so the guard that refused
          // it is removed.
          if (const auto* full_sp = dynamic_cast<const FullAttentionSpec*>(sp.get())) {
            l_Dh_v = full_sp->head_size_v;
          }
          VT_CHECK(l_page > 0,
                   "runner: per-layer attention spec reported a non-positive page");
        }
        full_attn_buf_.push_back(std::make_unique<CacheBuffer>(
            dev, queue_,
            static_cast<size_t>(num_blocks_) * static_cast<size_t>(l_page),
            kv_cache_backend_resident_));
        fa_dims.push_back(FaDims{l_Hkv, l_Dh, l_Dh_v, l_dtype, fa_block_size, l_fp8_kind,
                                 l_k_scale, l_v_scale, l_page});
        // Per-layer MLA flag, parallel to fa_dims: the view loop picks the right
        // backend name (TRITON_MLA for an MLA group) and the right expected KV
        // shape (fused 3-dim, not the NHD 5-dim) per group.
        const KVCacheSpecKind layer_kind = has_per_layer
            ? kv_cache_config
                  .per_layer_attn_specs[static_cast<size_t>(l)]->kind()
            : kv_cache_config
                  .kv_cache_groups[static_cast<size_t>(full_attn_group_id_)]
                  .kv_cache_spec->kind();
        mla_layer_mask.push_back(layer_kind == KVCacheSpecKind::kMlaAttention);
        // The legacy (single-group) path: the group's own name, which is empty
        // for every model that reaches this branch.
        named_backend.push_back(
            kv_cache_config
                .kv_cache_groups[static_cast<size_t>(full_attn_group_id_)]
                .attn_backend);
      }
      // else: this layer is named by NO KV cache group, so it caches nothing.
      // Reachable only on the by-name path, and it is the correct answer there:
      // NemotronH's 52 blocks are 6 attention + 23 Mamba2 + 23 MoE, and an MoE
      // block registers no attention module at all, so upstream's
      // `get_kv_cache_spec()` yields no entry for it
      // (`gpu_model_runner.py:7785-7801` walks `AttentionLayerBase` instances).
      // The `layer_types` fallback cannot express this third class — every
      // non-recurrent layer there is an attention layer — which is why it
      // allocated 52 attention pages for a model that needs 6.
    }
  }

  // Build the views over the (now stable) backing storage. `fa_dims` is parallel
  // to `full_attn_buf_`; in the uniform case every entry is {Hkv, Dh, kv_dtype}.
  VT_CHECK(fa_dims.size() == full_attn_buf_.size(),
           "runner: per-layer KV view geometry out of sync with buffers");
  VT_CHECK(named_backend.size() == fa_dims.size(),
           "runner: the per-group attention-backend names are out of sync with "
           "the KV view geometry");
  attn_kv_.clear();
  attn_backend_names_.clear();
  for (size_t i = 0; i < full_attn_buf_.size(); ++i) {
    PagedKvCache kv;
    kv.data = full_attn_buf_[i]->data();
    kv.dtype = fa_dims[i].dtype;
    kv.num_blocks = num_blocks_;
    // KV-DSV4-MULTICACHE W3 (#2068): the entry's OWN block size. Identical to
    // `fa_block_size` for every entry on the legacy path.
    kv.block_size = fa_dims[i].block_size;
    kv.num_kv_heads = fa_dims[i].num_kv_heads;
    kv.head_size = fa_dims[i].head_size;
    // MiMoV2: carry the V head dim when it differs from K. 0 means
    // `head_size` (the byte-identical legacy path every existing model takes).
    kv.head_size_v = fa_dims[i].head_size_v;
    // KV-FP8 W3: the fp8 interpretation + scales reach the model's attention
    // block ONLY through this view, which is what makes `--kv-cache-dtype fp8`
    // a served capability rather than a resized allocation.
    kv.fp8_kind = fa_dims[i].fp8_kind;
    kv.k_scale = fa_dims[i].k_scale;
    kv.v_scale = fa_dims[i].v_scale;
    // KV-DSV4-MULTICACHE W8 slice 4 (#2455): the allocated page, so a consumer
    // of a packed or compressed page can build a view over the bytes that were
    // actually reserved instead of the bytes the rank-3 geometry implies.
    kv.page_size_bytes = fa_dims[i].page_size_bytes;
    // M3: the backend selection resolved for THIS group must describe the view
    // geometry the engine allocates + KvSlice reads — the NHD 5-dim
    // (num_blocks, 2, block_size, num_kv_heads, head_size) for a dense group,
    // the fused MLA 3-dim (num_blocks, block_size, head_size) for an MLA group
    // (vllm::v1::CheckKvCacheShape). An empty name (MLA on a device with no
    // registered MLA backend) means op-driven execution — nothing to validate.
    // A future backend with a different layout fails LOUDLY here at init.
    const bool is_mla =
        mla_layer_mask[static_cast<size_t>(i)] != 0;
    // #1332 M1: the selector now applies the full validate_configuration
    // capability surface, so the request it is asked has to BE the request. The
    // three fields this site can answer come straight from the geometry it just
    // resolved. `dtype` (the model/query dtype) is NOT available here — the
    // runner resolves only ResolveKvCacheDType() — so it keeps its bf16 default;
    // that is owed to #1332 M4 and recorded under `## Owed` in
    // .agents/specs/attn-validate-configuration.md.
    //
    // AND READ THIS BEFORE READING A GREEN SELECTION AS A WORKING BACKEND: the
    // name resolved here still reaches only attn_backend_names_, the
    // VT_ATTN_SELECT_LOG print below and CheckKvCacheShape. dense_attn::AttnBlock
    // calls vt::PagedAttention unconditionally. Nothing DISPATCHES on this. #1332
    // M4 owns that, and until it lands a valid name is a claim, not a route.
    vllm::platforms::AttnSelectorConfig cfg;
    cfg.head_size = static_cast<int>(fa_dims[i].head_size);
    cfg.num_heads = static_cast<int>(fa_dims[i].num_kv_heads);
    cfg.block_size = static_cast<int>(fa_dims[i].block_size);
    // KV-FP8 W3: `KvCacheDTypeName` cannot answer this one, and deliberately so
    // — a bare `kI8` byte does not know its semantic type (`vt/dtype.h:20-32`),
    // exactly as upstream stores every fp8 flavour as `torch.uint8`
    // (`torch_utils.py:38-40`) and reads the flavour off the layer. The
    // interpretation is the thing that knows, so ask it.
    cfg.kv_cache_dtype =
        fa_dims[i].fp8_kind == vt::Fp8KVCacheDataType::kFp8E4M3
            ? "fp8_e4m3"
            : (fa_dims[i].fp8_kind == vt::Fp8KVCacheDataType::kFp8E5M2
                   ? "fp8_e5m2"
                   : vllm::v1::KvCacheDTypeName(fa_dims[i].dtype));
    cfg.quantized_kv_cache = vllm::v1::IsQuantizedKvCacheName(cfg.kv_cache_dtype);

    std::string name;
    // THE GROUP'S OWN BACKEND WINS, and it is taken DIRECTLY rather than
    // through the capability walk. That is upstream's shape: a layer that
    // names its backend (`compressor.py:189-190`, `sparse_swa.py:116-118`,
    // `indexer.py:183-196`) is never offered to `get_attn_backend_cls`, and the
    // named class is in no priority list. It is also the whole repair: the
    // resolution below is cached per CLASS, so DeepSeek-V4's 256-token latent
    // group resolved TRITON_MLA and the 4/4/8 compressor and indexer groups
    // then inherited a `% 16` refusal that was never about them, which is what
    // killed a real checkpoint at engine construction
    // (ISSUE-LOCAL-01M2EMPC6T63TVDPQ90GVPRC5F).
    //
    // A name no device registers throws HERE, at init, naming the device and
    // the backend — not later and not silently.
    if (!named_backend[i].empty()) {
      VT_CHECK(vllm::v1::HasAttentionBackend(queue_.device.type,
                                             named_backend[i]),
               std::string("runner: a KV cache group names attention backend '") +
                   named_backend[i] +
                   "', which is not registered for this device type");
      name = named_backend[i];
    } else if (is_mla) {
      if (!mla_backend_resolved) {
        mla_backend_resolved = true;
        vllm::platforms::AttnSelectorConfig mla_cfg = cfg;
        mla_cfg.use_mla = true;
        try {
          mla_backend = vllm::v1::SelectAttentionBackendName(
              vllm::platforms::GetPlatform(queue_.device.type), "", mla_cfg);
        } catch (const std::exception&) {
          // Op-driven MLA (no registered MLA backend for this device) —
          // recorded, not an error; see attn_backend_names_ in runner.h.
        }
      }
      name = mla_backend;
    } else {
      if (!dense_backend_resolved) {
        dense_backend_resolved = true;
        dense_backend = vllm::v1::SelectAttentionBackendName(
            vllm::platforms::GetPlatform(queue_.device.type), "", cfg);
      }
      name = dense_backend;
    }
    attn_backend_names_.push_back(name);
    if (const char* dbg = std::getenv("VT_ATTN_SELECT_LOG");
        dbg != nullptr && dbg[0] == '1') {
      if (is_mla) {
        std::fprintf(stderr,
                     "[attn-select] kind=mla backend=%s device=%d "
                     "shape=[%lld,%lld,%lld]\n",
                     name.empty() ? "(op-driven)" : name.c_str(),
                     static_cast<int>(queue_.device.type),
                     static_cast<long long>(num_blocks_),
                     static_cast<long long>(fa_dims[i].block_size),
                     static_cast<long long>(fa_dims[i].head_size));
      } else {
        std::fprintf(stderr,
                     "[attn-select] kind=dense backend=%s device=%d "
                     "shape=[%lld,2,%lld,%lld,%lld]\n",
                     name.c_str(), static_cast<int>(queue_.device.type),
                     static_cast<long long>(num_blocks_),
                     static_cast<long long>(fa_dims[i].block_size),
                     static_cast<long long>(fa_dims[i].num_kv_heads),
                     static_cast<long long>(fa_dims[i].head_size));
      }
    }
    if (!name.empty()) {
      vllm::v1::CheckKvCacheShape(queue_.device.type, name, num_blocks_,
                                  fa_dims[i].block_size,
                                  fa_dims[i].num_kv_heads,
                                  fa_dims[i].head_size, is_mla);
    }
    attn_kv_.push_back(kv);
  }

  // KV-DSV4-MULTICACHE W3 (#2068): PUBLISH the third channel, now that
  // `attn_kv_` is stable. The vectors are owned by this runner, so the pointers
  // stay valid for every forward it drives. Left at its default (all-null,
  // `size() == 0`) on every uniform topology, which is what keeps
  // `ModelForwardInput::multi_kv` null there.
  if (multi_cache_topology_) {
    VT_CHECK(attn_kv_layer_names_.size() == attn_kv_.size(),
             "runner: the paged name list is out of sync with attn_kv");
    // ENG-MULTIKV-BYNAME: the five vectors are parallel to EACH OTHER and cover
    // every published cache, so the size they are checked against is the index's
    // own length, NOT `attn_kv_.size()`. That distinction is the row: checking
    // against `attn_kv_` is exactly what made a recurrent entry inexpressible.
    const size_t n = kv_index_layer_names_.size();
    VT_CHECK(kv_index_group_ids_.size() == n &&
                 kv_index_layer_indices_.size() == n &&
                 kv_index_payload_kinds_.size() == n &&
                 kv_index_payload_slots_.size() == n,
             "runner: the by-name KV index vectors are not parallel");
    // EVERY allocated cache is addressable, stated as an equality over the two
    // payload containers. `gdn_state_` is built one-for-one from
    // `recurrent_state_buf_` further down this function, so the recurrent count
    // is read from the buffer list that already exists rather than from a
    // vector that is still empty at this point.
    VT_CHECK(n == attn_kv_.size() + recurrent_state_buf_.size(),
             "runner: the by-name KV index does not cover exactly the caches "
             "this runner allocated");
    multi_kv_index_.layer_names = &kv_index_layer_names_;
    multi_kv_index_.group_ids = &kv_index_group_ids_;
    multi_kv_index_.layer_indices = &kv_index_layer_indices_;
    multi_kv_index_.payload_kinds = &kv_index_payload_kinds_;
    multi_kv_index_.payload_slots = &kv_index_payload_slots_;
    // W5c-2 (#2249 item 3): the FOURTH vector — one block table per PUBLISHED
    // group. Sized here (by GROUP, not by cache) and filled per step by
    // `gather_group_block_tables`; the pointers are published once because the
    // vectors are members whose storage the refill reuses.
    group_block_tables_.assign(kv_cache_config.kv_cache_groups.size(), {});
    group_block_table_cols_.assign(kv_cache_config.kv_cache_groups.size(), 0);
    multi_kv_index_.group_block_tables = &group_block_tables_;
    multi_kv_index_.group_block_table_cols = &group_block_table_cols_;
  }

  // SPEC-MTP I5d: allocate the MTP draft's own paged KV layer (the `fa_draft`
  // group). It is sized exactly like a target full-attn layer and the propose
  // forward reuses the target's block table / slot mapping over it
  // (speculator.py:222-234). Allocated ONLY when speculation is on and the ctor
  // did not already supply a draft KV (tests may). num_spec==0 has no fa_draft
  // group, so this is never entered on the production default path.
  draft_attn_buf_.clear();
  // KV-DSV4-MULTICACHE W3 (#2068): NOT on the multi-cache path. This block picks
  // "the first kFullAttention group that is not the target" as the draft slot,
  // which on a multi-cache topology would be a real published group the loop
  // above already allocated, and would double-allocate it. Nothing publishes a
  // multi-cache topology WITH speculation today.
  //
  // SAY WHAT THIS GUARD DOES, because an earlier draft of this comment claimed
  // it "refuses" and it does not. With `multi_cache_topology && spec_on()` the
  // guard is false, `draft_attn_buf_` stays empty, and `propose_drafts_block`
  // returns early on `draft_attn_kv_.empty()` — so speculation turns ITSELF off
  // and says nothing. That costs throughput, never tokens: no drafts means the
  // engine decodes normally, which is the same output the speculative path is
  // required to produce. It is still a silent degradation, and it is listed
  // under `## Owed` against W4 in `.agents/specs/kv-dsv4-multicache.md` rather
  // than described here as a refusal it is not. Making it a real `VT_CHECK` is
  // the right end state and is deliberately NOT done here: no test surface can
  // reach `spec_on()` on this path today (the weights-based ctor takes no
  // `SpeculativeConfig`), and a refusal nothing can drive red is a claim rather
  // than a guarantee.
  if (!multi_cache_topology && spec_on() && draft_attn_kv_.empty() &&
      full_attn_group_id_ >= 0 && fa_page_bytes > 0) {
    for (int g = 0;
         g < static_cast<int>(kv_cache_config.kv_cache_groups.size()); ++g) {
      if (g == full_attn_group_id_) continue;
      const auto& group = kv_cache_config.kv_cache_groups[static_cast<size_t>(g)];
      if (group.kv_cache_spec->kind() != KVCacheSpecKind::kFullAttention) {
        continue;  // the GDN group and any non-attn group are not the draft.
      }
      draft_attn_buf_.push_back(std::make_unique<CacheBuffer>(
          dev, queue_,
          static_cast<size_t>(num_blocks_) * static_cast<size_t>(fa_page_bytes),
          kv_cache_backend_resident_));
      PagedKvCache dkv;
      dkv.data = draft_attn_buf_.back()->data();
      dkv.dtype = kv_dtype;
      dkv.num_blocks = num_blocks_;
      dkv.block_size = fa_block_size;
      dkv.num_kv_heads = Hkv;
      dkv.head_size = Dh;
      // The draft layer is sized from `fa_page_bytes` — the TARGET group's page
      // — so it must carry the target's storage dtype AND its fp8 interpretation
      // or the two disagree about element width over one shared block table.
      dkv.fp8_kind = kv_fp8_kind;
      dkv.k_scale = kv_k_scale;
      dkv.v_scale = kv_v_scale;
      // The draft buffer is allocated at `fa_page_bytes` three lines above, so
      // that is its page. Carrying the target's value here is the same choice
      // the dtype and fp8 fields already make, and for the same reason: both
      // sides index one shared block table.
      dkv.page_size_bytes = fa_page_bytes;
      draft_attn_kv_.push_back(dkv);
      break;  // exactly one fa_draft group at k=1.
    }
  }

  // #810: the recurrent VIEWS carry the spec's own per-slot shape with the slot
  // dim prepended — the mirror of `mamba/abstract.py:38-43`'s
  // `state.view(-1, *shape)`. Rank-general up to vt::kMaxRank (checked above),
  // so a 2-D conv state and a 3-D temporal state are both expressible without
  // the runner knowing what either MEANS.
  //
  // ENG-RECURRENT-MULTISTATE (#2131): and COUNT-general too. One view per
  // published state, in SPEC ORDER, which is upstream's
  // `self.kv_cache = tuple(states)` (`mamba/abstract.py:43`). `conv_state` and
  // `ssm_state` are `states[0]` and `states[1]` — assigned FROM the list rather
  // than beside it, so the two names cannot drift from the list every future
  // consumer reads.
  const auto slot_major_view = [&](void* data, vt::DType dtype,
                                   const std::vector<int64_t>& shape) {
    switch (shape.size()) {
      case 1:
        return vt::Tensor::Contiguous(data, dtype, dev,
                                      {gdn_state_slots_, shape[0]});
      case 2:
        return vt::Tensor::Contiguous(data, dtype, dev,
                                      {gdn_state_slots_, shape[0], shape[1]});
      default:
        return vt::Tensor::Contiguous(
            data, dtype, dev,
            {gdn_state_slots_, shape[0], shape[1], shape[2]});
    }
  };
  gdn_state_.clear();
  for (auto& layer : recurrent_state_buf_) {
    VT_CHECK(layer.size() == state_shapes.size(),
             "runner: a recurrent layer allocated a different number of states "
             "than its MambaSpec published");
    GdnStateCache gs;
    gs.states.reserve(layer.size());
    for (size_t i = 0; i < layer.size(); ++i)
      gs.states.push_back(
          slot_major_view(layer[i]->data(), state_dtypes[i], state_shapes[i]));
    gs.conv_state = gs.states[0];
    gs.ssm_state = gs.states[1];
    gdn_state_.push_back(std::move(gs));
  }
}

// ENG-RECURRENT-MULTISTATE (#2131): one buffer per published state, in SPEC
// ORDER. Upstream allocates one contiguous page per layer and slices it in that
// order (`mamba/abstract.py:29-43`); this tree holds one buffer per state
// instead, so the ORDER is what carries the correspondence.
void GPUModelRunner::alloc_recurrent_layer_states(
    vt::Device dev, const std::vector<vt::DType>& state_dtypes,
    const std::vector<int64_t>& state_row_elems) {
  std::vector<std::unique_ptr<CacheBuffer>> layer;
  layer.reserve(state_dtypes.size());
  for (size_t i = 0; i < state_dtypes.size(); ++i)
    layer.push_back(std::make_unique<CacheBuffer>(
        dev, queue_,
        static_cast<size_t>(gdn_state_slots_ * state_row_elems[i]) *
            vt::SizeOf(state_dtypes[i]),
        kv_cache_backend_resident_));
  recurrent_state_buf_.push_back(std::move(layer));
}

std::vector<int32_t> GPUModelRunner::gather_block_table(int group_id,
                                                        int num_reqs,
                                                        int* num_cols) const {
  // NO SUCH GROUP IS AN EMPTY TABLE, and it used to be an out-of-bounds read.
  //
  // `full_attn_group_id_` and `gdn_group_id_` are -1 SENTINELS meaning "this
  // model published no group of that kind". The GDN call site guards on its
  // sentinel; the full-attention one does not, and
  // `MultiGroupBlockTable::operator[]` casts the index to `size_t`, so
  // `block_tables[-1]` read a `BlockTable` object that does not exist. The
  // `max_num_blocks_per_req` it produced then decided the step: a garbage 0
  // gathered an empty table and the request went on to the model, while a
  // garbage negative made `num_reqs * cols` a ~1.8e19 `size_t` and the engine's
  // busy loop died with `std::length_error` before any forward ran. Which one
  // happened moved with the BINARY'S LAYOUT rather than with anything about the
  // request -- adding one earlier test case to the same suite flipped it -- and
  // that is issue #3027's `gather_block_table` signature.
  //
  // DeepSeek-V4 publishes no `kFullAttention` and no `kMlaAttention` group, so
  // `full_attn_group_id_` is -1 on EVERY served request for that architecture
  // and the read above happened on all of them. Whether that group should be
  // classified as the target attention group is a separate question, owed by
  // row KV-DSV4-MULTICACHE W3 (#2068); this only makes the sentinel mean what
  // it says.
  //
  // `MakeCommonAttentionMetadata` already tolerates the same sentinel one line
  // later -- its `group < slot_mapping.size()` is false for -1, so the group's
  // slot mapping is left empty -- so an empty table is what the rest of the
  // step is already written against. BYTE-NEUTRAL for every model that
  // publishes a full-attention group, which is every model shipping today.
  if (group_id < 0 || static_cast<size_t>(group_id) >=
                          input_batch_.block_table.block_tables.size()) {
    *num_cols = 0;
    return {};
  }
  const BlockTable& bt = input_batch_.block_table[group_id];
  const int cols = bt.max_num_blocks_per_req;
  *num_cols = cols;
  const std::vector<int32_t>& dev = bt.get_device_tensor();  // committed rows
  const size_t n = static_cast<size_t>(num_reqs) * static_cast<size_t>(cols);
  return std::vector<int32_t>(dev.begin(),
                              dev.begin() + static_cast<std::ptrdiff_t>(n));
}

// MODEL-MM-QWEN4-EXP W5c-2 (#2249 item 3) — the mirror of upstream's per-group
// metadata loop (`vllm/v1/worker/gpu_model_runner.py:2551-2567` @ pin
// 5559679229). Every published group gets its own committed table, including the
// TARGET attention group. Upstream gathers group 0 too, but ONCE and BEFORE the
// loop (`block_table_gid_0 = _get_block_table(0)` at `:2337`), carrying it into
// every iteration on `cm_base`; the loop body is guarded by `if kv_cache_gid >
// 0:` (`:2565`), so upstream does NOT re-gather group 0 inside the loop and
// this function does. That guard is an OPTIMISATION and not a second
// convention for "which groups are special" — the table it skips rebuilding is
// the same table. Re-gathering here therefore costs ONE EXTRA COPY of a table
// this step already built, and buys a vector with no index the reader has to
// know is special. The two named-id gathers in `execute_model`
// keep their own copies because the GDN one is REWRITTEN in place by
// `remap_gdn_state_slots`, and this vector must carry what the block table
// actually says, not a remapped state-slot view of it.
void GPUModelRunner::gather_group_block_tables(int num_reqs) {
  for (size_t g = 0; g < group_block_tables_.size(); ++g) {
    int cols = 0;
    group_block_tables_[g] =
        gather_block_table(static_cast<int>(g), num_reqs, &cols);
    group_block_table_cols_[g] = cols;
  }
}

void GPUModelRunner::remap_gdn_state_slots(
    std::vector<int32_t>& gdn_bt, int gdn_cols, int num_reqs,
    const std::vector<std::optional<std::string>>& req_ids) {
  if (gdn_cols <= 0 || num_reqs <= 0) return;
  // The persistent batch holds EVERY live sequence in its [0, num_reqs) prefix.
  // Key the compact state slot on the sequence IDENTITY (req_id), never the
  // block-table col-0 block-id: once a sequence exceeds one mamba block that
  // column collapses to the shared null block-id 0 (MambaManager skips all but
  // the last block), so block-id keying maps every long concurrent sequence to
  // ONE slot — the captured c16 "duplicate live GDN state index" fatal and,
  // pre-validator, silent cross-request recurrent-state corruption.
  // Reused member scratch (buckets persist across steps) — no per-step set
  // allocation. Cleared then refilled with this step's live request ids.
  std::unordered_set<std::string>& alive = gdn_alive_scratch_;
  alive.clear();
  alive.reserve(static_cast<size_t>(num_reqs));
  for (int r = 0; r < num_reqs; ++r) {
    // req_ids[r] is populated for every active [0, num_reqs) row after condense.
    VT_CHECK(req_ids[static_cast<size_t>(r)].has_value(),
             "GDN remap: active batch row is missing its request id");
    alive.insert(*req_ids[static_cast<size_t>(r)]);
  }
  // Reclaim slots of sequences no longer in the batch (finished / preempted):
  // a slot is released only after its owning request leaves.
  for (auto it = gdn_slot_of_req_.begin(); it != gdn_slot_of_req_.end();) {
    if (alive.find(it->first) == alive.end()) {
      gdn_free_slots_.push_back(it->second);
      it = gdn_slot_of_req_.erase(it);
    } else {
      ++it;
    }
  }
  // Assign/reuse a compact BASE slot per live sequence. Without speculation the
  // base is written into col 0 (the only column the GDN metadata builder reads).
  // SPEC-MTP I5d: under speculation each sequence owns num_spec+1 CONSECUTIVE
  // slots [base, base+num_spec]; write them into cols 0..num_spec so the spec
  // GDN builder reads the k+1 draft-timestep state slots (gdn_attn.py:266-269).
  const int spec_cols = spec_on() ? num_spec() + 1 : 1;
  for (int r = 0; r < num_reqs; ++r) {
    const std::string& rid = *req_ids[static_cast<size_t>(r)];
    const size_t off = static_cast<size_t>(r) * static_cast<size_t>(gdn_cols);
    auto it = gdn_slot_of_req_.find(rid);
    int32_t base;
    if (it != gdn_slot_of_req_.end()) {
      base = it->second;
    } else {
      VT_CHECK(!gdn_free_slots_.empty(),
               "GDN state slots exhausted: live sequences exceed max_num_reqs");
      base = gdn_free_slots_.back();
      gdn_free_slots_.pop_back();
      gdn_slot_of_req_.emplace(rid, base);
    }
    for (int c = 0; c < spec_cols && c < gdn_cols; ++c) {
      gdn_bt[off + static_cast<size_t>(c)] = base + c;
    }
  }
}

// ─── ENG-MM-INPUT-PIPELINE P2 (#2379): the runner's multimodal path ──────────
//
// Three hops separate `Request.mm_features` from `ModelForwardInput::mm`. The
// scheduler owns the first two (the payload and the encoder admission); these
// five functions are the third. Every one of them returns immediately for a
// request with no multimodal items, and none of them is CALLED at all unless the
// registration declares the encode/embed pair — see `supports_mm_inputs()`.

void GPUModelRunner::free_evicted_encoder_outputs(
    const SchedulerOutput& scheduler_output) {
  // The worker half of the scheduler's eviction. `free_encoder_mm_hashes` names
  // the entries the EncoderCacheManager actually evicted since the previous step
  // (encoder_cache_manager.py:255), and the two caches would otherwise disagree
  // about what is resident: the scheduler would stop reserving space for an
  // entry this map still holds, and the runner would hold vision embeddings for
  // a request that finished a thousand steps ago.
  for (const std::string& mm_hash : scheduler_output.free_encoder_mm_hashes) {
    encoder_cache_.erase(mm_hash);
  }
}

void GPUModelRunner::init_mrope_positions(
    const SchedulerOutput& scheduler_output) {
  // `_init_mrope_positions` (gpu_model_runner.py:1654). Computed ONCE, at
  // admission, because `get_mrope_input_positions` needs the WHOLE prompt: the
  // temporal/height/width rows of an image span are assigned relative to the
  // running maximum over everything before it, so a per-step recomputation over
  // a chunk would produce different numbers for the same token.
  for (const NewRequestData& new_req : scheduler_output.scheduled_new_reqs) {
    if (new_req.mm_features.empty()) continue;
    const auto it = req_states_.find(new_req.req_id);
    if (it == req_states_.end()) continue;
    CachedRequestState& state = it->second;
    MropePromptPositions positions = ModelRegistry::MropePromptPositionsFor(
        *model_, config_, state.prompt_token_ids, state.mm_features);
    state.mrope_positions = std::move(positions.positions);
    state.mrope_position_delta = positions.delta;
  }
}

void GPUModelRunner::execute_mm_encoder(
    const SchedulerOutput& scheduler_output) {
  // `_execute_mm_encoder` (gpu_model_runner.py:2998). THE TOWER CALL. The
  // scheduler already decided which items run this step and already reserved
  // cache space for each of them, so this loop neither budgets nor evicts: it
  // runs exactly what it was told to and stores the result under the item's
  // MM_HASH (gpu_model_runner.py:2995), never under (req_id, input_id). Two
  // requests carrying the same image therefore share one encoder run, which is
  // the whole reason the key is a content hash.
  for (const auto& [req_id, input_ids] :
       scheduler_output.scheduled_encoder_inputs) {
    const auto state_it = req_states_.find(req_id);
    if (state_it == req_states_.end()) continue;
    const CachedRequestState& state = state_it->second;
    for (const int input_id : input_ids) {
      VT_CHECK(input_id >= 0 &&
                   input_id < static_cast<int>(state.mm_features.size()),
               "runner mm encoder: the scheduler named multimodal item " +
                   std::to_string(input_id) + " for request '" + req_id +
                   "', which carries " +
                   std::to_string(state.mm_features.size()) +
                   " items. ENG-MM-INPUT-PIPELINE P2 (#2379).");
      const multimodal::MultiModalFeatureSpec& item =
          state.mm_features[static_cast<size_t>(input_id)];
      // Already encoded by an earlier step or by another request holding the
      // same hash. The scheduler's `check_and_update_cache` normally answers
      // this first and never names the item, so reaching it means two requests
      // were admitted in the SAME step for one hash.
      if (encoder_cache_.count(item.mm_hash) != 0) continue;
      encoder_cache_[item.mm_hash] =
          ModelRegistry::EncodeMm(*model_, config_, queue_, item);
    }
  }
}

bool GPUModelRunner::batch_carries_mm() const {
  // The second half of the multimodal predicate. It asks whether any request in
  // the batch HAS multimodal items, not whether this step's window covers one:
  // a decode step of an image request covers no placeholder row at all, and
  // taking the text path for it would hand a merged-embeds-only forward
  // (Qwen3-VL's) a step it refuses by name.
  const int num_reqs = input_batch_.num_reqs();
  for (int i = 0; i < num_reqs; ++i) {
    const std::optional<std::string>& req_id =
        input_batch_.req_ids[static_cast<size_t>(i)];
    if (!req_id.has_value()) continue;
    const auto it = req_states_.find(*req_id);
    if (it == req_states_.end()) continue;
    if (!it->second.mm_features.empty()) return true;
  }
  return false;
}

GPUModelRunner::MmGather GPUModelRunner::gather_mm_embeddings(
    const SchedulerOutput& scheduler_output, int total_num_scheduled_tokens) {
  // `_gather_mm_embeddings` (gpu_model_runner.py:3220). For every request in
  // batch order, take the multimodal items whose placeholder span overlaps the
  // token window this step covers, and slice the ROWS of their cached encoder
  // output that the window reaches.
  MmGather out;
  out.is_mm_embed.assign(static_cast<size_t>(total_num_scheduled_tokens), 0);

  // Batch order, bounded by num_reqs: the slots past it are stale nullopt
  // entries a removal left behind, and `req_start_idx` below only advances in
  // step with a SCHEDULED request.
  int req_start_idx = 0;
  const int num_reqs = input_batch_.num_reqs();
  for (int index = 0; index < num_reqs; ++index) {
    const std::string& req_id = *input_batch_.req_ids[static_cast<size_t>(index)];
    const int num_scheduled_tokens =
        scheduler_output.num_scheduled_tokens.at(req_id);
    const auto state_it = req_states_.find(req_id);
    VT_CHECK(state_it != req_states_.end(),
             "runner mm gather: no cached state for scheduled request '" +
                 req_id + "'. ENG-MM-INPUT-PIPELINE P2 (#2379).");
    const CachedRequestState& state = state_it->second;
    const int num_computed_tokens = state.num_computed_tokens;

    const std::pair<int, int> window = multimodal::GetMmFeaturesInWindow(
        state.mm_features, num_computed_tokens,
        num_computed_tokens + num_scheduled_tokens);
    for (int i = window.first; i < window.second; ++i) {
      const multimodal::MultiModalFeatureSpec& item =
          state.mm_features[static_cast<size_t>(i)];
      const int start_pos = item.offset;
      const int num_encoder_tokens = item.length;
      const int start_idx = std::max(num_computed_tokens - start_pos, 0);
      const int end_idx =
          std::min(num_computed_tokens - start_pos + num_scheduled_tokens,
                   num_encoder_tokens);
      if (end_idx - start_idx <= 0) continue;

      const auto cached = encoder_cache_.find(item.mm_hash);
      // THE MISS THE SCHEDULER MAKES UNREACHABLE. `_try_schedule_encoder_inputs`
      // truncates `num_new_tokens` to stop before an item whose encoder could
      // not run, so a decoder chunk can never cover placeholder rows with no
      // encoder output. Without that clamp this fires, and refusing here is the
      // only honest answer: splicing whatever the embedding table gave the
      // placeholder rows produces fluent, confidently WRONG tokens.
      VT_CHECK(cached != encoder_cache_.end(),
               "Encoder cache miss for " + item.mm_hash +
                   ". The scheduler admitted decoder tokens across a multimodal "
                   "span whose encoder was never run — see "
                   "Scheduler::try_schedule_encoder_inputs. "
                   "ENG-MM-INPUT-PIPELINE P2 (#2379).");

      // The row slice, as a BORROWED view into the cached device buffer. The
      // encoder output is [num_embeds, width] where `width` is whatever the
      // model's tower produced for one placeholder row — for Qwen3-VL that is
      // hidden * (1 + deepstack_levels), because DeepStack rides INSIDE the
      // encoder output and is unpacked model-side (`grep -c deepstack` over
      // upstream's runner is 0). The runner slices rows and never reads a
      // column, so it stays generic over that width.
      const vt::Tensor& full = cached->second.embeds;
      VT_CHECK(full.rank == 2 && full.data != nullptr,
               "runner mm gather: encoder output for " + item.mm_hash +
                   " is not a 2-D device tensor");
      VT_CHECK(end_idx <= static_cast<int>(full.shape[0]),
               "runner mm gather: encoder output for " + item.mm_hash +
                   " has " + std::to_string(full.shape[0]) + " rows, but the "
                   "placeholder span asks for row " + std::to_string(end_idx));
      const int64_t width = full.shape[1];
      vt::Tensor slice = full;
      slice.data = static_cast<char*>(full.data) +
                   static_cast<size_t>(start_idx) *
                       static_cast<size_t>(full.stride[0]) *
                       vt::SizeOf(full.dtype);
      slice.shape[0] = end_idx - start_idx;
      slice.shape[1] = width;
      out.mm_embeds.push_back(slice);

      const int req_start_pos = req_start_idx + start_pos - num_computed_tokens;
      for (int t = req_start_pos + start_idx; t < req_start_pos + end_idx; ++t) {
        VT_CHECK(t >= 0 && t < total_num_scheduled_tokens,
                 "runner mm gather: placeholder row " + std::to_string(t) +
                     " is outside this step's " +
                     std::to_string(total_num_scheduled_tokens) + " tokens");
        out.is_mm_embed[static_cast<size_t>(t)] = 1;
      }
    }
    req_start_idx += num_scheduled_tokens;
  }
  return out;
}

std::vector<int32_t> GPUModelRunner::calc_mrope_positions(
    const SchedulerOutput& scheduler_output, int total_num_scheduled_tokens) {
  // `_calc_mrope_positions` (gpu_model_runner.py:2748). [3, T] row-major.
  std::vector<int32_t> out(static_cast<size_t>(3) *
                           static_cast<size_t>(total_num_scheduled_tokens));
  int ptr = 0;
  const int num_reqs = input_batch_.num_reqs();
  for (int index = 0; index < num_reqs; ++index) {
    const std::string& req_id = *input_batch_.req_ids[static_cast<size_t>(index)];
    const auto state_it = req_states_.find(req_id);
    VT_CHECK(state_it != req_states_.end(),
             "runner M-RoPE: no cached state for scheduled request '" + req_id +
                 "'. ENG-MM-INPUT-PIPELINE P2 (#2379).");
    const CachedRequestState& state = state_it->second;
    const int num_computed_tokens =
        input_batch_.num_computed_tokens_cpu[static_cast<size_t>(index)];
    const int num_scheduled_tokens =
        scheduler_output.num_scheduled_tokens.at(req_id);
    const int num_prompt_tokens = state.num_prompt_tokens;

    int prompt_part_len = num_scheduled_tokens;
    int completion_part_len = 0;
    if (num_computed_tokens + num_scheduled_tokens > num_prompt_tokens) {
      prompt_part_len = std::max(0, num_prompt_tokens - num_computed_tokens);
      completion_part_len = std::max(0, num_scheduled_tokens - prompt_part_len);
    }

    if (prompt_part_len > 0) {
      // The prompt part is pre-computed; this is a SLICE, never a recomputation.
      VT_CHECK(static_cast<int>(state.mrope_positions.size()) >=
                   3 * (num_computed_tokens + prompt_part_len),
               "runner M-RoPE: request '" + req_id +
                   "' has no pre-computed prompt positions for this window. A "
                   "request that carries multimodal items on an M-RoPE model "
                   "must have gone through init_mrope_positions at admission.");
      for (int axis = 0; axis < 3; ++axis) {
        for (int j = 0; j < prompt_part_len; ++j) {
          out[static_cast<size_t>(axis) *
                  static_cast<size_t>(total_num_scheduled_tokens) +
              static_cast<size_t>(ptr + j)] =
              state.mrope_positions[static_cast<size_t>(axis) *
                                        static_cast<size_t>(num_prompt_tokens) +
                                    static_cast<size_t>(num_computed_tokens + j)];
        }
      }
      ptr += prompt_part_len;
    }
    if (completion_part_len > 0) {
      // `get_next_input_positions_tensor`: NOTHING stores a completion position.
      // All three axes take `context_len + i + delta`, so one int per request
      // carries M-RoPE across a thousand decode steps.
      const int64_t context_len = num_computed_tokens + prompt_part_len;
      for (int axis = 0; axis < 3; ++axis) {
        for (int j = 0; j < completion_part_len; ++j) {
          out[static_cast<size_t>(axis) *
                  static_cast<size_t>(total_num_scheduled_tokens) +
              static_cast<size_t>(ptr + j)] =
              static_cast<int32_t>(context_len + j + state.mrope_position_delta);
        }
      }
      ptr += completion_part_len;
    }
  }
  VT_CHECK(ptr == total_num_scheduled_tokens,
           "runner M-RoPE: filled " + std::to_string(ptr) + " of " +
               std::to_string(total_num_scheduled_tokens) + " positions");
  return out;
}

std::optional<ModelRunnerOutput> GPUModelRunner::execute_model(
    const SchedulerOutput& scheduler_output) {
  // ENG-ASYNC-SCHED depth-2 LIFETIME GUARD. Under async scheduling the previous
  // step went through sample_tokens_async, which DEFERS the main queue's D2H to
  // the consuming step's get_output() (one step_with_batch_queue call later). So
  // on entry here the previous step's forward / sample / scatter kernels may
  // still be IN FLIGHT, and they still reference the state we are about to
  // mutate: exec_state_ (the device logits + the StepInputs host arrays the
  // forward/sampler read) and input_batch_.last_sampled_tokens (the scatter
  // writes it; update_states' condense/swap read+move it). Freeing/mutating that
  // state now — while those kernels run on GB10's unified memory — is a
  // use-after-free that corrupts the host heap. It only manifests with REAL GPU
  // overlap: the CPU eager backend and compute-sanitizer both serialize the
  // queue and never see it, which is exactly why the CPU gates stayed green while
  // the served 35B aborted.
  //
  // Two dispositions of that drain, selected by async_device_mirror():
  //  - DEFAULT (mirror OFF, byte-identical production): last_sampled_tokens is a
  //    HOST array the scatter writes and update_states' condense reorders, so the
  //    drain MUST sit here, at the TOP, before update_states touches it.
  //  - MIRROR ON (VT_ASYNC_DEVICE_MIRROR, device-resident sampled tokens): the
  //    scatter writes and the combine reads the DEVICE mirror (dev->last_sampled),
  //    and condense only reorders a now-stale host array + records ops replayed
  //    on-queue (input_batch_.last_sampled_ops). The host condense<->scatter
  //    read-after-write is gone, so the drain no longer has to PRECEDE
  //    update_states. It MOVES to just before the forward (below): update_states +
  //    prepare_inputs + the attention/GDN metadata build (the bulk host prep, all
  //    of which read input_batch_ only, never exec_state_) then overlap the
  //    previous step's GPU tail, while exec_state_ and the persistent decode-graph
  //    buffers stay untouched — and therefore UAF-safe — until that later drain.
  //    This is a drain MOVE, not a drain REMOVAL: removal would additionally need
  //    double-buffered exec_state_ + block-table device buffers, which the moved
  //    drain makes unnecessary because nothing is freed/overwritten before it.
  //    Mirrors vLLM keeping input_batch.prev_sampled_token_ids device-resident
  //    (states.py:64) with the _prepare_input_ids GPU gather
  //    (gpu_model_runner.py:1786-1881) so _update_states never reads a
  //    device-written buffer and the sync only guards the input staging.
  const bool mirror = async_device_mirror();
  if (async_forward_in_flight_ && !mirror) {
    vt::GetBackend(queue_.device.type).Synchronize(queue_);
    async_forward_in_flight_ = false;
  }

  // update_states: admit new reqs (incl. prefill_token_ids) + apply cached diffs
  // + remove finished/unscheduled + condense (M1.5). Reads/mutates input_batch_
  // only (never exec_state_), so on the mirror path it is safe to run while the
  // previous step's kernels still read exec_state_.
  //
  // ENG-MM-INPUT-PIPELINE P2 (#2379): `req_states_` is upstream's
  // `self.requests`, and it is passed ONLY for a model that declares the
  // multimodal seam. A text engine passes null and update_states is
  // byte-identical.
  update_states(input_batch_, scheduler_output,
                supports_mm_inputs() ? &req_states_ : nullptr);

  // ENG-MM-INPUT-PIPELINE P2 (#2379): the encoder half of the step, in
  // upstream's order — drop what the scheduler evicted, compute the M-RoPE
  // prompt positions of every newly admitted multimodal request, then run the
  // TOWER over exactly the items the scheduler admitted. All three iterate
  // scheduler-output collections that are EMPTY on every text step, so a text
  // engine pays one predicate.
  if (supports_mm_inputs()) {
    free_evicted_encoder_outputs(scheduler_output);
    if (uses_mrope()) init_mrope_positions(scheduler_output);
    execute_mm_encoder(scheduler_output);
  }

  // Reset the stash. A 0-token step (e.g. an aborted-request flush) runs no
  // forward — mark num_reqs == 0 so sample_tokens returns an empty output
  // (mirrors upstream execute_model's total==0 early return). MIRROR ON DEFERS
  // this reset (and the drain) to just before the forward, because exec_state_
  // still holds the previous step's device logits the in-flight sampler reads;
  // freeing it here would be the very UAF the guard exists to prevent.
  if (!mirror) exec_state_ = ExecuteModelState{};
  if (scheduler_output.total_num_scheduled_tokens == 0 ||
      input_batch_.num_reqs() == 0) {
    // Mirror path deferred the drain + reset; do them now before returning. The
    // recorded structural ops (this step's condense) survive to the next step
    // that runs a combine — see the last_sampled_ops note below.
    if (async_forward_in_flight_) {
      vt::GetBackend(queue_.device.type).Synchronize(queue_);
      async_forward_in_flight_ = false;
    }
    if (mirror) exec_state_ = ExecuteModelState{};
    return std::nullopt;
  }

  // DECODE-FIRST REORDER (four-way ordering contract) — before any metadata.
  // #2129: the threshold is the mirrored one, not the declaration default.
  // Upstream passes its resolved `reorder_batch_threshold` here
  // (gpu_model_runner.py:1126-1130 @ pin 5559679229), and a spec-as-decode
  // backend raises it to `1 + (parallel_drafting ? 2 : 1) * k`. At k=8 with
  // parallel drafting that is 17: a 1+k == 9-token verify row is a DECODE, not
  // a long extend sorted among the chunked-prefill continuations. Exactly 1 —
  // the pre-#2129 value — on every non-speculative configuration.
  reorder_batch_to_split_decodes_and_prefills(input_batch_, scheduler_output,
                                              reorder_batch_threshold());

  // SPEC-MTP I5d: splice the scheduler's drafts for THIS verify step into
  // token_ids_cpu after each request's committed prefix (gpu_input_batch.py:
  // 484-509 update_req_spec_token_ids) so prepare_inputs reads the k draft tokens
  // at the verify positions. No-op on the default path (empty map / no speculator).
  //
  // SPEC-DFLASH2 W7 (#1824), the async draft-in-output WORKER half. Under
  // async scheduling two things differ from the sync flow, both handled here:
  //
  // (a) COMPUTED-TOKEN CORRECTION, OPTIMISTIC HALF (SPEC-DFLASH2 A2-4, #3004).
  //     The scheduler's num_computed_tokens for a request whose PREVIOUS step
  //     scheduled drafts may still include that step's rejected drafts — the
  //     rollback runs in update_from_output, which the depth-2 loop applies
  //     AFTER this schedule.
  //
  //     W7 answered that with a STRUCTURAL value, the newest committed token's
  //     position `num_tokens_no_spec - 1`, and it was right for one reason: our
  //     rejection walk ran on the HOST before this step was scheduled, so the
  //     exact value was already known here. A2-2 began removing that property
  //     and A2-5 finishes removing it, so the rule was correct and
  //     self-invalidating — which is what this wave is for.
  //
  //     What replaces it is upstream's, in upstream's two halves
  //     (gpu_model_runner.py:1356-1396 @ pin 5559679229). HERE: assume every one
  //     of last step's drafts was accepted and write that value, which needs no
  //     accept result at all — `prev_num_computed_tokens + 1 +
  //     prev_num_draft_len`, all of it this runner's own bookkeeping, and
  //     therefore the same number in BOTH engine orders rather than a rule that
  //     has to be indifferent to which one ran. THEN, in
  //     `apply_deferred_spec_decode_corrections`, subtract what was rejected.
  //     Upstream's optimism is the same one: `optimistic_num_accepted =
  //     req_state.prev_num_draft_len` (:1376).
  //
  // (b) PLACEHOLDER FILL. The scheduler ships -1 placeholders
  //     (async_scheduler.py:43-45); the real values are the drafts THIS
  //     runner proposed at the previous step's sampling. The fill patches a
  //     LOCAL copy for the splice: the engine-side SchedulerOutput keeps its
  //     placeholders, exactly as upstream's worker-side scatter leaves the
  //     scheduler's copy untouched (gpu_input_batch.py:520-523).
  //
  //     SPEC-DFLASH2 A2-3 (#2911) changed WHERE it reads them from. It used to
  //     read `pending_drafts_` — host, ragged, req_id-keyed, and the same object
  //     `take_draft_token_ids` moves out to the scheduler. It now reads the
  //     per-req_state `input_batch_.draft_tokens`, which is the buffer the
  //     combine's draft scatter reads and which survives the out-of-band pull.
  //     One residence, one producer (`set_draft_tokens`), two readers, which is
  //     upstream's shape (model_runner.py:1548,1553-1556).
  if (spec_on()) {
    if (use_async_scheduling_) {
      deferred_spec_corrections_.clear();
      const CachedRequestData& cached = scheduler_output.scheduled_cached_reqs;
      for (int ci = 0; ci < cached.num_reqs(); ++ci) {
        const std::string& req_id = cached.req_ids[static_cast<size_t>(ci)];
        const auto prev_it = prev_num_draft_len_.find(req_id);
        if (prev_it == prev_num_draft_len_.end() || prev_it->second <= 0) {
          continue;  // no drafts scheduled for it last step: value is exact.
        }
        const auto idx_it = input_batch_.req_id_to_index.find(req_id);
        if (idx_it == input_batch_.req_id_to_index.end()) {
          continue;  // not in the persistent batch (resumed-as-new path).
        }
        const auto pos_it = prev_num_computed_tokens_.find(req_id);
        if (pos_it == prev_num_computed_tokens_.end()) {
          continue;  // this runner recorded no position for it last step.
        }
        const int req_index = idx_it->second;
        // OPTIMISTICALLY ASSUME ALL ACCEPTED (gpu_model_runner.py:1373-1381).
        // The position this runner wrote last step, plus the token that step's
        // input carried, plus every draft it scheduled. No accept result reads
        // into this expression, which is the whole point of the wave.
        const int optimistic_num_accepted = prev_it->second;
        const int optimistic = pos_it->second + 1 + optimistic_num_accepted;
        const int sent =
            input_batch_.num_computed_tokens_cpu[static_cast<size_t>(req_index)];
        // The scheduler's value is the optimistic one when its rollback has not
        // yet run (the depth-2 batch-queue order) and the corrected one when it
        // has (the depth-1 LLMEngine::step order), so it brackets ours from
        // above and below. Anything outside that bracket is a bookkeeping
        // defect — refuse loudly, exactly as W7's range check did.
        VT_CHECK(optimistic - sent >= 0 &&
                     optimistic - sent <= optimistic_num_accepted,
                 "async spec computed-token correction out of range for '" +
                     req_id + "': scheduler sent " + std::to_string(sent) +
                     ", optimistic value " + std::to_string(optimistic) +
                     ", prev drafts " + std::to_string(optimistic_num_accepted));
        input_batch_.num_computed_tokens_cpu[static_cast<size_t>(req_index)] =
            optimistic;
        deferred_spec_corrections_.push_back(
            {req_id, req_index, optimistic_num_accepted});
      }
    }

    const std::map<std::string, std::vector<int32_t>>* sched_spec =
        &scheduler_output.scheduled_spec_decode_tokens;
    std::map<std::string, std::vector<int32_t>> filled_spec;
    if (use_async_scheduling_) {
      // SPEC-DFLASH2 A2-3 (#2911): THE FILL READS THE DRAFT BUFFER, not the
      // out-of-band stash. `pending_drafts_` is a delivery to the scheduler and
      // `take_draft_token_ids` MOVES it out, so it can never be the residence two
      // in-step readers share; `input_batch_.draft_tokens` survives the pull and
      // is the same buffer `combine_sampled_and_draft_tokens`' draft scatter
      // reads. That is upstream's arrangement (`req_states.draft_tokens` for the
      // workers, the `DraftTokensHandler` copy for the scheduler,
      // model_runner.py:1548,1553-1556) and it is what stops the fill and the
      // scatter from disagreeing about what this runner drafted — a disagreement
      // that costs acceptance and raises nothing, verify being lossless.
      //
      // THE WHOLE STEP IS ONE CALL, and this is the A2-3 repair-round-3 shape.
      // The per-request rule is `FillDraftsForRow` and the STEP rule —
      // freshness, the slot lookup, and the ledger's consume — is
      // `FillDraftsForStep` (both in prepare_inputs.h), so the loop and the
      // consume cannot be separated by an edit here. They were: the consume used
      // to sit inside `use_async_scheduling_ && !sched_spec->empty()`, so a step
      // with no placeholders consumed nothing and left the ledger answering for
      // a propose two steps old — the exact state this design exists to make
      // unrepresentable, and one no suite in this tree could see. The mechanism
      // it replaced never behaved that way: `take_draft_token_ids` moves
      // `pending_drafts_` out on EVERY deferred-batch step (core_proc.cpp:234).
      //
      // Called on every async-scheduling step, EMPTY MAP INCLUDED. That is not
      // an accident of structure: the empty step is precisely the one that must
      // still consume.
      filled_spec = FillDraftsForStep(
          proposed_drafts_, *sched_spec, input_batch_.draft_tokens,
          input_batch_.num_speculative_steps, input_batch_.req_id_to_index,
          input_batch_.num_valid_draft_tokens);
      sched_spec = &filled_spec;
    }

    const int nr = input_batch_.num_reqs();
    for (int i = 0; i < nr; ++i) {
      const std::string& req_id = *input_batch_.req_ids[static_cast<size_t>(i)];
      input_batch_.update_req_spec_token_ids(i, req_id, *sched_spec);
    }

    if (use_async_scheduling_) {
      // SPEC-DFLASH2 A2-4 (#3004). `prev_num_draft_len` is written where the
      // drafts are spliced, which is upstream's site verbatim
      // (`gpu_input_batch.py:517` @ pin 5559679229,
      // `request.prev_num_draft_len = num_spec_tokens`, inside
      // `update_req_spec_token_ids`).
      //
      // `prev_num_computed_tokens_` is recorded BESIDE it and from the same
      // step, so the next step's optimistic value is this runner's own number
      // rather than the scheduler's. It is read here AFTER the optimistic pass
      // above wrote it, which is deliberate: the value the next step must
      // extend is the one this step actually ran on. The deferred correction
      // below rewrites it once the accept result is in, and its own tail
      // re-records it, so the two writers cannot disagree.
      prev_num_draft_len_.clear();
      for (const auto& [rid, toks] : *sched_spec) {
        prev_num_draft_len_[rid] = static_cast<int>(toks.size());
      }
      prev_num_computed_tokens_.clear();
      for (int i = 0; i < nr; ++i) {
        prev_num_computed_tokens_[*input_batch_.req_ids[static_cast<size_t>(i)]] =
            input_batch_.num_computed_tokens_cpu[static_cast<size_t>(i)];
      }
    }
  }

  // SPEC-DFLASH2 A2-4 (#3004): the DEFERRED half of the computed-token
  // correction, which turns each optimistic value above into the exact one.
  // Upstream calls this after the model LAUNCH (gpu_model_runner.py:4524-4525)
  // and keeps the in-step positions right with a device kernel inside
  // `_prepare_inputs` (:2148-2166); our positions are computed on the host from
  // `num_computed_tokens_cpu`, so it must run before `prepare_inputs` reads
  // them, and it does — one line above the read it protects. Moving it past the
  // launch is A2-5's, with that kernel; see the declaration.
  apply_deferred_spec_decode_corrections();

  // Build the flattened dense-order step inputs (M1.5).
  StepInputs step = prepare_inputs(input_batch_, scheduler_output);
  const int num_reqs = input_batch_.num_reqs();

  // Async-scheduling device-input path (ENG-ASYNC-SCHED W3 runner leaf). When
  // engaged, overwrite each decode row's input token id with the GPU-resident-
  // analog last_sampled_tokens (combine_sampled_and_draft_tokens) instead of the
  // host token_ids_cpu value prepare_inputs read — so step N+1 need not wait on
  // step N's sampled token crossing to the host. idx_mapping is identity: our
  // persistent batch is already condensed dense (batch row == req_state slot).
  // Runs on the HOST side of input prep, BEFORE the forward and OUTSIDE any
  // CUDA-graph capture (input prep always precedes the decode graph replay), so
  // it is capture-safe. The mirror is DEFAULT ON (ROW-SERVE-ASYNC-LLM P0 fix);
  // VT_ASYNC_DEVICE_MIRROR=0 rolls back to the host-array combine, whose host
  // decode-graph read races this combine's device write (the bug the flip fixes).
  // Non-null on the W4 mirror path below (the default; integrated OR discrete):
  // the device input-id buffer the combine patched, handed to the forward so it
  // embeds the spliced ids instead of the (deliberately stale) host vector.
  const int32_t* device_input_ids = nullptr;
  if (async_input_combine_ && num_reqs > 0) {
    // The step's draft-token total is a sum of per-request counts, which are
    // sizes. A negative total is a corrupt step, not a decode step
    // (SPEC-DFLASH2 A2-2, #2802).
    VT_CHECK(step.num_draft_tokens >= 0,
             "async input combine: the step's draft-token total is negative, which "
             "is a corrupt step (it is a sum of per-request counts)");
    // ─── SPEC-DFLASH2 A2-3 (#2911): WHAT REPLACED THE "NOT WIRED YET" REFUSAL ──
    //
    // A2-1 made `combine_sampled_and_draft_tokens` draft-aware and A2-2 left the
    // refusal below reading `!StepRoutesToVerify(step.num_draft_tokens)` — a
    // BLANKET refusal of every verify step, because the buffer its scatter reads
    // did not exist and every call site passed an empty one. A2-3 supplies that
    // buffer, so the refusal's stated reason is now FALSE and it cannot stand as
    // written. It is not deleted, because the precondition it stood in for did
    // not go away; it CHANGED, from "a verify step is impossible here" to "a
    // verify step must arrive with a draft buffer that covers every req_state row
    // it will read", and that is what is asserted instead.
    //
    // THE INVARIANT IS THE ONE THAT MAKES THE CUDA ARM SAFE. That arm's kernel
    // reads `draft_tokens[req_state_idx * stride + b]` with NOTHING bounding it
    // (the row's spec records the gap and tells this wave to close it by sizing).
    // A buffer sized by this step's request count lets a high req_state slot
    // index past the allocation with every host check satisfied; the quiet
    // outcome is garbage drafts, which costs acceptance and nothing else. The
    // check below is that sizing, stated where the unbounded read is issued.
    //
    // The condition still reads the ROUTE predicate itself, negated, rather than
    // a re-derivation of it: `StepRoutesToVerify` is the same per-STEP function
    // `sample_tokens` routes on, so no input can route one way here and another
    // way there — the split shape #2710 shipped.
    VT_CHECK(
        !StepRoutesToVerify(step.num_draft_tokens) ||
            (input_batch_.num_speculative_steps > 0 &&
             input_batch_.draft_tokens.size() >=
                 static_cast<std::size_t>(input_batch_.max_num_reqs) *
                     static_cast<std::size_t>(
                         input_batch_.num_speculative_steps)),
        "async input combine: this step scheduled draft tokens, but the draft "
        "buffer the combine scatters from does not cover the req_state pool "
        "(SPEC-DFLASH2 A2-3, #2911) — the CUDA scatter's row read is unbounded, "
        "so a buffer sized by num_reqs is an out-of-bounds device read");
    // ─── SPEC-DFLASH2 A2-3 REPAIR (#2911): ONE cu_num_logits FOR EVERY ARM ───
    //
    // A2-3 wired the draft buffer into all three arms and left the two CUDA arms
    // passing `/*cu_num_logits=*/nullptr` while the host arm passed the real
    // `step.cu_num_logits`. A `## Owed` entry called that "the same call with a
    // different residence", and it was not: same buffer, DIFFERENT per-request
    // count, and the difference is silent.
    //
    // WHAT THE NULL DOES. `cuda_combine_tokens.cu` reads a null `cu_num_logits`
    // as arange(num_reqs + 1), so `num_logits` is 1 and `num_draft_tokens` is 0
    // for every row. On a decode step that is the right answer. On a VERIFY step
    // `logits_start` becomes `query_end - 1`, which IS the last draft slot: the
    // committed token lands on top of the last draft and NOTHING is scattered,
    // while the host arm writes it at `query_end - (1 + k)` and scatters the k
    // drafts. Reason A, measured on this branch as `draft=[ 6 18 5 ]` ->
    // `[ 6 18 22 ]` and `[ 5 ]` -> `[ 3 ]` — the previous step's `emit` — with
    // the `emit=` column unchanged OVER THOSE ROWS and the suite's identity
    // assertions passing 118 of 118 in both arms.
    //
    // "ONLY ACCEPTANCE FALLS" IS SCOPED TO THOSE ROWS AND IS NOT A GLOBAL CLAIM.
    // A round-3 review measured the whole run and it moves both ways: at `k=1`,
    // positions 9 and 11, the pre-repair shape reads `ns=2 acc=1
    // draft=[ 1 ] emit=[ 1 1 ]` where the repaired shape reads `ns=1 acc=0
    // emit=[ 1 ]`, because the corrupted slot holds the COMMITTED token, which
    // then matches and is ACCEPTED. Flattened, the pre-repair run emits 198
    // tokens against the repaired run's 197 — one inserted `1`, the rest
    // identical, verify being lossless — and `[SPECTRACE]` prints 185 lines
    // against 192. So acceptance and tokens-per-step at a fixed step budget do
    // differ, and a claim that nothing but per-row acceptance moves is wrong.
    //
    // That discriminator is NOT a gate this tree can ship, which is why the
    // guard below exists instead: reaching the pre-repair shape at all needs
    // mutation M — the veto deleted at BOTH `async_input_combine_` assignments —
    // so the difference is only observable under a mutation and never on a
    // committed configuration. What IS committed is the size guard below plus
    // `test_combine_tokens.cpp`'s G2, which drives the combine directly and
    // compares against the proposer's own ids. No END-TO-END token gate in this
    // tree sees the class (#1366).
    //
    // WHY IT IS FIXED AND NOT REFUSED. A refusal would be UNBUILT here (no nvcc
    // on this box), so its correctness would rest on reading, and it would leave
    // A2-5 having to make exactly this edit at exactly these two call sites with
    // no mutation to check it against. Passing the value removes the shape
    // instead of deferring it: `step.cu_num_logits` is the arange on the
    // non-speculative path (prepare_inputs.cpp:213), so this is INERT for every
    // engine running today and correct for the first verify step A2-5 admits.
    //
    // THE GUARD IS NOT THIS COMMENT. `cu_num_logits` is the ONLY source of a
    // row's `num_logits` in all three arms, and a short or absent one degenerates
    // every row to 1 — the same corruption by another route. That is asserted
    // here, on the host, where the CPU tier builds and gates it, ahead of the
    // arm selection so no arm can be reached without it.
    VT_CHECK(static_cast<int>(step.cu_num_logits.size()) == num_reqs + 1,
             "async input combine: cu_num_logits must hold num_reqs + 1 entries "
             "and holds " + std::to_string(step.cu_num_logits.size()) +
                 " for " + std::to_string(num_reqs) +
                 " requests (it is every arm's only source of a row's "
                 "num_logits; a short one degenerates every row to 1, which on a "
                 "verify step splices the committed token over the LAST DRAFT "
                 "SLOT and scatters no drafts — SPEC-DFLASH2 reason A, which "
                 "costs acceptance and raises nothing)");
#ifdef VLLM_CPP_CUDA
    // The host-side source both device arms start from. It is NOT true that they
    // therefore cannot be edited apart, and the round-3 review was right to say
    // so: only the UMA arm passes this pointer through. The mirror arm passes
    // `dev->cu_num_logits`, a DEVICE buffer that equals it only while the
    // `stage_upload(*dev, dev->cu_num_logits, cu_num_logits_host, num_reqs + 1)`
    // below is present and correctly sized. Delete or mis-size that upload and
    // the mirror arm alone reverts to reason A — unbuilt on this box and caught
    // by nothing. What the shared local does buy is that neither arm can pick up
    // a DIFFERENT host source than the host arm below, which takes the same
    // vector by reference. Inside the #ifdef because only the device arms
    // want a pointer; the assertion above it is outside, where the CPU tier
    // builds and gates it.
    const int32_t* const cu_num_logits_host = step.cu_num_logits.data();
    // W4 device-resident sampled tokens. Preferred whenever engaged
    // (async_device_mirror(): CUDA + VT_ASYNC_DEVICE_MIRROR, INTEGRATED OR
    // DISCRETE). `last_sampled` is already on the device (the previous step's
    // scatter wrote it there and nothing read it back), so the three host-known
    // inputs plus the freshly built input_ids are uploaded through the staging
    // copy, the recorded structural edits are replayed first so the mirror's row
    // order matches this step's batch, and the combine patches the DEVICE
    // input_ids the forward will embed. On the INTEGRATED path this is what keeps
    // the sampled tokens OFF the host array update_states' condense reorders,
    // which is what lets the depth-2 drain move past update_states (execute_model
    // top). Mirrors vLLM's prev_sampled_token_ids device tensor + _prepare_input_
    // ids GPU gather (gpu_model_runner.py:1786-1881).
    //
    // Ordering, all on the MAIN queue and therefore exact: replay -> uploads ->
    // combine -> forward. The forward is handed `device_input_ids` below, so the
    // host copy of step.input_token_ids is deliberately left stale for decode
    // rows; nothing on this path reads it (the rejection-sampler path that does
    // is spec-only, and a speculator keeps async_input_combine_ OFF — since W7
    // that is the runner-INPUT lever alone, async SCHEDULING staying on; the
    // spec host arrays stay fresh because the spec sampler is the sync one).
    if (AsyncDeviceInputs* dev = get_or_create_async_device_inputs();
        dev != nullptr) {
      replay_last_sampled_ops(*dev);
      const int64_t num_tokens =
          static_cast<int64_t>(step.input_token_ids.size());
      VT_CHECK(num_tokens <= dev->input_ids_capacity,
               "async device mirror: step tokens exceed max_num_batched_tokens");
      VT_CHECK(num_reqs <= dev->max_reqs,
               "async device mirror: step requests exceed max_num_reqs");
      stage_upload(*dev, dev->input_ids, step.input_token_ids.data(), num_tokens);
      stage_upload(*dev, dev->query_start_loc, step.query_start_loc.data(),
                   static_cast<int64_t>(num_reqs) + 1);
      stage_upload(*dev, dev->seq_lens, step.seq_lens.data(), num_reqs);
      stage_upload(*dev, dev->prefill_len, input_batch_.prefill_len.data(),
                   num_reqs);
      // SPEC-DFLASH2 A2-3 (#2911): the draft buffer is wired. It is uploaded the
      // way `prefill_len` is — the host array is authoritative because our
      // propose is host-synchronous — and the WHOLE req_state pool goes up, not
      // this step's `num_reqs` rows, because the kernel indexes it by req_state
      // slot and a slot above `num_reqs` is addressable after any abort or
      // finish reorder. Inert without a speculator: the stride is 0, the pointer
      // is null, and the kernel degenerates to the single splice exactly as
      // before.
      if (dev->draft_stride > 0 && dev->draft_tokens != nullptr) {
        stage_upload(*dev, dev->draft_tokens, input_batch_.draft_tokens.data(),
                     static_cast<int64_t>(dev->max_reqs) * dev->draft_stride);
      }
      // SPEC-DFLASH2 A2-3 REPAIR (#2911): cu_num_logits is UPLOADED and PASSED,
      // exactly like query_start_loc two calls up and for the same `num_reqs + 1`
      // extent. It used to be null here, which the kernel reads as arange and
      // which is reason A on any verify step — see the block above the #ifdef.
      stage_upload(*dev, dev->cu_num_logits, cu_num_logits_host,
                   static_cast<int64_t>(num_reqs) + 1);
      vt::cuda::LaunchCombineSampledAndDraftTokens(
          queue_, dev->input_ids, /*idx_mapping=*/nullptr, dev->last_sampled,
          dev->query_start_loc, dev->seq_lens, dev->prefill_len,
          dev->draft_tokens, dev->draft_stride,
          dev->cu_num_logits, num_reqs,
          /*num_new_sampled_tokens=*/1);
      device_input_ids = dev->input_ids;
    } else if (vllm::platforms::GetPlatform(queue_.device.type)
                   .is_integrated_gpu()) {
      // DEVICE combine, HOST-ARRAY residence (W3 DGX leaf; mirror OFF = the GB10
      // production default). Splice each decode row's input id from the device-
      // ADDRESSABLE host last_sampled_tokens (pageable on GB10's UMA) on the MAIN
      // queue, BEFORE the forward (which embeds input_token_ids on the same queue →
      // sees the patch) and OUTSIDE any decode-graph capture. idx_mapping identity
      // (condensed-dense batch), nullptr. num_new_sampled_tokens == 1 (T0
      // non-spec). All inputs live in `step` / persistent InputBatch members, so
      // they outlive the async launch. Its scatter counterpart writes the SAME
      // host array update_states' condense reorders, which is why the drain must
      // sit at execute_model's top on this path (the mirror path lifts that).
      // is_integrated_gpu() decouples a future discrete GPU (answers false → host
      // combine below, the right path there since its host arrays are not
      // device-addressable).
      // SPEC-DFLASH2 A2-3 (#2911): the draft buffer is wired here too, and on
      // this arm it is the HOST array read directly — device-addressable on a
      // UMA part, which is the same reason `last_sampled_tokens` and
      // `prefill_len` are passed as host pointers two lines up. It is an
      // `InputBatch` member, so it outlives the async launch exactly as they do.
      // Null and zero-strided without a speculator, which is every engine on
      // this path while the veto stands.
      //
      // SPEC-DFLASH2 A2-3 REPAIR (#2911): cu_num_logits is the SAME host vector
      // the host arm below reads, handed over as a pointer for the same reason
      // `last_sampled_tokens` and `prefill_len` are — device-addressable on a UMA
      // part, and living in `step`, which outlives the async launch. It used to
      // be null here, which the kernel reads as arange; this is GB10's
      // production default, so that null was reason A on the integrated
      // production path from the first verify step A2-5 would admit.
      vt::cuda::LaunchCombineSampledAndDraftTokens(
          queue_, step.input_token_ids.data(), /*idx_mapping=*/nullptr,
          input_batch_.last_sampled_tokens.data(), step.query_start_loc.data(),
          step.seq_lens.data(), input_batch_.prefill_len.data(),
          input_batch_.draft_tokens.empty()
              ? nullptr
              : input_batch_.draft_tokens.data(),
          input_batch_.num_speculative_steps,
          cu_num_logits_host, num_reqs,
          /*num_new_sampled_tokens=*/1);
    } else
#endif
    {
      std::vector<int32_t> idx_mapping(static_cast<size_t>(num_reqs));
      std::iota(idx_mapping.begin(), idx_mapping.end(), 0);
      // SPEC-DFLASH2 A2-3 (#2911): the host arm reads the same per-req_state
      // buffer the two device arms above read, and this is the arm the CPU tier
      // actually builds and gates. Since the A2-3 repair it also reads the same
      // `cu_num_logits`: the two device arms passed a null here, and this is the
      // arm whose mutation to that shape reproduces reason A.
      combine_sampled_and_draft_tokens(
          step.input_token_ids, idx_mapping, input_batch_.last_sampled_tokens,
          step.query_start_loc, step.seq_lens, input_batch_.prefill_len,
          input_batch_.draft_tokens, input_batch_.num_speculative_steps,
          step.cu_num_logits,
          /*num_new_sampled_tokens=*/1);
    }
  }

  // W4: the structural-op log has exactly one consumer, the device mirror's
  // replay above. On every other configuration (host-array integrated path, CPU
  // backend, the VT_ASYNC_DEVICE_MIRROR=0 rollback, async off) nothing drains it,
  // so drop it here rather than let it grow for the life of a serving process.
  // Deliberately NOT cleared when the mirror is on: a step with no requests
  // replays nothing, and its ops must survive to the next step that does.
  if (!mirror) input_batch_.last_sampled_ops.clear();

  // Full-attention KV group metadata (M1.6 MakeCommonAttentionMetadata).
  int fa_cols = 0;
  const std::vector<int32_t> fa_bt =
      gather_block_table(full_attn_group_id_, num_reqs, &fa_cols);
  CommonAttentionMetadata attn_meta = MakeCommonAttentionMetadata(
      step, fa_bt, fa_cols, /*causal=*/true, full_attn_group_id_);

  // W5c-2 (#2249 item 3): THE PRODUCTION CALL SITE for the per-group gather.
  // Beside the two named-id gathers rather than instead of them, and gated on
  // the multi-cache topology so every model shipping today enters no new code.
  // A published group whose table never leaves the runner has a cache that is
  // allocated and unaddressable, which is what left the `qwen4_exp` QSA indexer
  // side cache unread.
  if (multi_cache_topology_) gather_group_block_tables(num_reqs);

  // GDN KV group metadata: the same step over the GDN group's block table,
  // segmented decode-first by the GDN builder (M1.6 Task 4). GATED on the model
  // HAVING a linear-attention (GDN/Mamba) KV group: a full-attention-only model
  // (dense Qwen3ForCausalLM) has gdn_group_id_ < 0 and no GDN block-table group,
  // so gather_block_table(gdn_group_id_) / remap / the metadata build must be
  // skipped — gdn_meta stays default-empty (num_prefill_tokens == 0, all state
  // arrays nullopt) and no GDN state is wired. The hybrid gate models keep a GDN
  // group (gdn_group_id_ >= 0), so this block runs exactly as before —
  // behavior-preserving / byte-identical for the hybrid path.
  GDNAttentionMetadata gdn_meta;
  if (gdn_group_id_ >= 0) {
    int gdn_cols = 0;
    std::vector<int32_t> gdn_bt =
        gather_block_table(gdn_group_id_, num_reqs, &gdn_cols);
    // SPEC-MTP I5d: the spec GDN builder reads cols 0..num_spec of the block
    // table as the k+1 draft-timestep state slots. Our compact per-sequence pool
    // OWNS the slot ids (the physical mamba block-id is irrelevant, see the remap
    // rationale), so widen the block-table view to num_spec+1 columns when the
    // real GDN block table is narrower, and let remap fill all k+1 slots.
    if (spec_on() && gdn_cols < num_spec() + 1) {
      const int wide = num_spec() + 1;
      std::vector<int32_t> widened(
          static_cast<size_t>(num_reqs) * static_cast<size_t>(wide), 0);
      for (int r = 0; r < num_reqs; ++r) {
        widened[static_cast<size_t>(r) * static_cast<size_t>(wide)] =
            gdn_bt[static_cast<size_t>(r) * static_cast<size_t>(gdn_cols)];
      }
      gdn_bt = std::move(widened);
      gdn_cols = wide;
    }
    // Remap col 0 to a compact per-sequence state slot in [0, gdn_state_slots_),
    // keyed on the request identity so the GDN state cache is sized by
    // max_num_reqs (one recurrent state per sequence) rather than the attention
    // num_blocks, and no two live sequences ever collide on one slot. Only col 0
    // (state indices) is read downstream.
    remap_gdn_state_slots(gdn_bt, gdn_cols, num_reqs, input_batch_.req_ids);
    if (GdnDiagStepLogEnabled()) {
      std::cerr << "[VT_GDN_DIAG] step num_reqs=" << num_reqs
                << " gdn_free_slots=" << gdn_free_slots_.size()
                << " gdn_live_slots=" << gdn_slot_of_req_.size() << "\n";
    }
    const CommonAttentionMetadata gdn_cam = MakeCommonAttentionMetadata(
        step, gdn_bt, gdn_cols, /*causal=*/true, gdn_group_id_);
    if (spec_on()) {
      // SPEC-MTP I5d: feed the GDN builder's spec overload (gdn_attn.py:189-326).
      // num_decode_draft_tokens_cpu[i] = -1 for a non-spec row, else the number of
      // drafts this verify step scheduled for req i (mamba_hybrid.py:247-264);
      // num_accepted_tokens[i] is the PREVIOUS step's accepted count (seeded 1,
      // overwritten by the rejection sampler). The remapped gdn_bt cols 0..k hold
      // this request's k+1 GDN state slots (see remap_gdn_state_slots spec branch).
      std::vector<int32_t> num_decode_draft_tokens(static_cast<size_t>(num_reqs), -1);
      for (int i = 0; i < num_reqs; ++i) {
        const std::string& req_id = *input_batch_.req_ids[static_cast<size_t>(i)];
        const auto it = scheduler_output.scheduled_spec_decode_tokens.find(req_id);
        if (it != scheduler_output.scheduled_spec_decode_tokens.end() &&
            !it->second.empty()) {
          num_decode_draft_tokens[static_cast<size_t>(i)] =
              static_cast<int32_t>(it->second.size());
        }
      }
      std::vector<int32_t> num_accepted(
          input_batch_.num_accepted_tokens.begin(),
          input_batch_.num_accepted_tokens.begin() + num_reqs);
      GDNAttentionMetadataBuilder gdn_builder(num_spec());
      gdn_meta = gdn_builder.build(/*common_prefix_len=*/0, gdn_cam, &num_accepted,
                                   &num_decode_draft_tokens);
    } else {
      GDNAttentionMetadataBuilder gdn_builder;
      gdn_meta = gdn_builder.build(/*common_prefix_len=*/0, gdn_cam);
    }
  }

  // Flattened dense-order forward inputs (positions int64 -> int32 for RoPE).
  const std::vector<int32_t>& token_ids = step.input_token_ids;
  std::vector<int32_t> positions;
  positions.reserve(step.positions.size());
  for (const int64_t position : step.positions) {
    positions.push_back(static_cast<int32_t>(position));
  }

  // ENG-ASYNC-DEVICE-IDS-REFUSAL (#2710) / ENG-MM-EMBED-DEVICE-IDS (#2730):
  // WHETHER the host vector actually disagrees with the device buffer for this
  // step. The pointer alone says the device buffer is authoritative; it does not
  // say the two differ, and on an all-prefill step they do not.
  //
  // COMPUTED ONCE, HERE, and read by BOTH channels below -- `MmEmbedInputs` and
  // `ModelForwardInput`. It used to sit beside the `ModelForwardInput`
  // assignment, which is AFTER the multimodal embed, so the embed could not have
  // been told. Two calls would be two derivations that can disagree about one
  // step, and the embed's guard and the forward's guard must not.
  //
  // It shares `v1::CombineSplicesRow` with the combine that does the splicing,
  // so the guards' predicate is the route's predicate. `idx_mapping` is null for
  // the same reason the combine launch above passes null: the persistent batch is
  // condensed dense, so batch row == req_state slot. Only meaningful when the
  // pointer is live, so it is computed only there -- false on every non-mirror
  // path, where both guards are inert anyway.
  const bool host_token_ids_stale =
      device_input_ids != nullptr
          ? v1::AnyRowSplicedByCombine(step.seq_lens, input_batch_.prefill_len,
                                       /*idx_mapping=*/nullptr, num_reqs)
          : false;

  // ─── ENG-MM-INPUT-PIPELINE P2 (#2379): what FILLS ModelForwardInput::mm ────
  //
  // The gather takes the encoder-output rows this step's window reaches; the
  // model's `embed_mm` (upstream's `SupportsMultiModal.embed_input_ids`,
  // interfaces.py:383, whose `multimodal_embeddings` parameter `grep -c` == 1 in
  // that file) embeds the ids and splices those rows into the masked positions.
  // The merge is a MODEL method upstream and it is a model hook here, because
  // what a placeholder row means — and what else rides in the encoder output,
  // DeepStack included — is architecture knowledge the runner must not carry.
  //
  // `mm_buffers` OWNS the device staging the model produced and must outlive the
  // forward, so it is declared here rather than inside the branch.
  std::optional<MmForwardBuffers> mm_buffers;
  if (supports_mm_inputs() && batch_carries_mm()) {
    // The flattened token count, taken from the array the model is about to
    // embed rather than from the attention metadata, so the mask and the ids
    // cannot disagree about length.
    const int num_step_tokens = static_cast<int>(token_ids.size());
    const MmGather gathered =
        gather_mm_embeddings(scheduler_output, num_step_tokens);
    std::vector<int32_t> mrope_positions;
    if (uses_mrope()) {
      mrope_positions = calc_mrope_positions(scheduler_output, num_step_tokens);
    }
    MmEmbedInputs embed_inputs;
    embed_inputs.token_ids = &token_ids;
    embed_inputs.mm_embeds = &gathered.mm_embeds;
    embed_inputs.is_mm_embed = &gathered.is_mm_embed;
    // EMPTY, not null, when the model declares no M-RoPE hook — upstream's
    // `uses_mrope == False`, where the model reads the 1-D positions instead.
    embed_inputs.mrope_positions = &mrope_positions;
    // ENG-MM-EMBED-DEVICE-IDS (#2730): THE assignment this row exists for.
    // `token_ids` above is the host step vector, which this runner deliberately
    // leaves STALE for decode rows -- the combine spliced each decode row's
    // sampled token into `device_input_ids` on the main queue and never wrote it
    // back. `batch_carries_mm()` is true on those decode steps BY DESIGN (see its
    // own comment), so without these two lines `EmbedMm` merged `inputs_embeds`
    // out of token id 0 on every one of them. Null / false on every non-mirror
    // step, where the hooks are byte-identical to their pre-#2730 selves.
    embed_inputs.device_token_ids = device_input_ids;
    embed_inputs.host_token_ids_stale = host_token_ids_stale;
    mm_buffers = ModelRegistry::EmbedMm(*model_, config_, queue_, embed_inputs);
  }

  // THE FORWARD (Task 3, over the persistent KV caches). Returns f32 logits
  // (lm_head already applied): [num_reqs, vocab] when the gather-before-lm_head
  // path is on (prefill/mixed) or pure-decode, else the full
  // [num_actual_tokens, vocab] (VT_LOGITS_GATHER=0).
  //
  // DECODE CUDA-GRAPH path (M2.5): the registered model forward routes a
  // PURE-DECODE fp4/CUDA step through its model-specific graph, whose state now
  // lives behind LoadedModel. It captures once per padded batch size and
  // replays per step; real-row output remains bit-identical to eager Forward.
  //
  // BATCHED (num_reqs>1) — the gate-#1 lever: at conc-64 kernel-launch overhead
  // is ~24% of the decode wall (~1.4k cudaLaunchKernel/step -> 1 cudaGraphLaunch).
  // The batch is padded up to the nearest captured size {1,2,4,8,16,32,64}
  // (mirrors vLLM cudagraph_capture_sizes + pad-to-nearest,
  // compilation/cuda_graph.py); the padded rows are inert (BuildPaddedDecode).
  // The decode forward is row-independent, so padding cannot perturb the real
  // rows. Beyond 64 (kMaxDecodeGraphBatch) / prefill / mixed / bf16 / CPU stay
  // eager. The DENSE (27B) registration owns its sibling decode graph under the
  // same gate.
  const bool pure_decode = attn_meta.num_actual_tokens == num_reqs &&
                           gdn_meta.num_prefill_tokens == 0;
  // ENG-CUDAGRAPH-BREAK W6 (#1374): THE GRAPH-ELIGIBILITY PREDICATE, and this is
  // the line the row exists to move. Until here the runner shipped ONE boolean
  // that means "query length is 1", and a model wanting anything wider had to
  // re-derive the whole test for itself -- which two of them did, in twenty
  // duplicated lines each (`qwen3_5_moe.cpp`, `qwen3_5_dense.cpp` @ #442). The
  // runner now names the step's ACTUAL uniform query length once and every model
  // reads the answer.
  //
  // IT IS ALSO THE FIX FOR [#1020]. Those two copies compared the uniform length
  // against the CONFIGURED `num_spec()`, so a step the scheduler clamped to a
  // shorter -- but still perfectly uniform -- draft depth missed the predicate
  // and ran its verify eager, with no log and no counter.
  // `ActualUniformDecodeQueryLen` reads the length the step HAS, bounded above
  // by `1 + num_spec()` because nothing in this tree captures a longer one.
  //
  // The GDN prefill conjunct is `pure_decode`'s and stays: a step with GDN
  // prefill tokens carries recurrent-prefill segmentation no decode capture was
  // built for, and the two model copies each tested it separately.
  //
  // THE ARM ABOVE 1 IS A SPECULATIVE VERIFY AND NOTHING ELSE, and the shape
  // alone does not say so. A single request prefilling three tokens is uniform
  // at query length 3 by every arithmetic test upstream applies, and at k >= 2
  // it would pass a bare `q <= 1 + k` bound straight into a DECODE capture --
  // measured on this tree's own CPU spec fixture, where a 20-token run reported
  // 19 "uniform spec" steps before this conjunct existed. So the widened arm
  // additionally requires that EVERY request in the step is verifying at exactly
  // `q - 1` drafts, read off the scheduler's own per-request draft counts. That
  // is narrower than the shape test, never wider, and it is what makes
  // `uniform_query_len > 1` mean what its comment says it means.
  const std::optional<int64_t> uniform_qlen =
      gdn_meta.num_prefill_tokens == 0
          ? v1::GraphEligibleQueryLen(num_reqs, attn_meta.num_actual_tokens,
                                      attn_meta.max_query_len, num_spec(),
                                      step.num_draft_tokens_per_req)
          : std::nullopt;
  // #1020 is titled on the word SILENTLY. A step that finds no captured shape
  // now moves a counter, on the shared path every registered model reaches.
  //
  // SPEC-DFLASH2 W13 (#2117): and it now says WHY it found none. The shape is
  // computed only on the ragged arm, because on the uniform arm every request
  // is a verify row by the predicate's own per-request conjunct and the answer
  // is already known.
  const v1::RaggedStepShape ragged_shape =
      uniform_qlen.has_value()
          ? v1::RaggedStepShape{}
          : v1::ClassifyStepRows(attn_meta.query_start_loc, num_reqs,
                                 step.num_draft_tokens_per_req);
  v1::NoteGraphDispatch(uniform_qlen.value_or(0),
                        v1::UniformDecodeQueryLen(num_spec()), ragged_shape);
  // SPEC-DFLASH2 W10 (#1857): classify the VERIFIED uniform verify width onto
  // the DECODE attention class through the mirrored reorder threshold
  // (1 + (parallel_drafting ? 2 : 1) * k, backend.py:718-736 @ b389ac2946;
  // dflash/dspark are parallel drafting). The value rides
  // CommonAttentionMetadata into the model's PagedAttentionArgs, where the CUDA
  // dispatch keeps the classified batch on the split-KV DECODE lane instead of
  // the num_splits=1 prefill ladder (include/vt/paged_attn_route.h) — the +9
  // ms/step the #1574 K-ladder attributed. The input is GraphEligibleQueryLen's
  // answer, never a raw shape: uniform by arithmetic AND every request
  // verifying at exactly q-1 drafts. 0 (non-verify, ragged, prefill) leaves
  // every consumer byte-identical, and the counter makes the decision visible
  // to a CPU gate the way every dispatch decision here is.
  attn_meta.uniform_spec_query_len = static_cast<int>(v1::SpecAsDecodeQueryLen(
      uniform_qlen.value_or(0), num_spec(),
      spec_config_.has_value() && spec_config_->parallel_drafting));
  if (attn_meta.uniform_spec_query_len > 0) v1::NoteSpecAsDecode();
  // SPEC-DFLASH2 W13, closing [#2112](https://github.com/mudler/vllm.cpp/issues/2112):
  // THE PRODUCTION READOUT, and this call is the whole of what that issue owes.
  // Both counter families existed and neither had a reader outside `tests/`, so
  // E6 of `.agents/specs/dflash2-batch-propose.md` was not runnable and #2117's
  // ragged-step share could not be observed on a server at all. The dump is here
  // rather than beside the `[spec-phase]` prints because THIS is the shared step
  // path every registered model reaches, speculative or not, and a readout that
  // only fires on the propose would report nothing for the engines that never
  // draft. OFF by default; `VT_GRAPH_STATS=N` prints every Nth step.
  MaybeDumpStepStats();
  // Gather-before-lm_head indices (the SAME last-token rows sample_tokens uses).
  // Empty when the toggle is off → old full-logits path. The eager forwards skip
  // the gather when it is a no-op (pure decode: len == num_actual_tokens).
  // SAMPLE-PROMPT-LOGPROBS: a request that asked for prompt logprobs needs an
  // lm_head row at each of its prompt positions, not just at the one row the
  // sampler consumes. The gather seam cannot express that: every model's
  // gather-before-lm_head is guarded on `logits_indices.size() < T` — it is an
  // optimization for "far fewer rows than tokens", and scoring a whole prompt
  // is the opposite. So a step that owes prompt logits takes the full-logits
  // path instead, which is the same shape upstream computes anyway
  // (gpu_model_runner.py:5680-5682 runs compute_logits over the prompt slice).
  //
  // step.prompt_logprob_indices is EMPTY on every step where no request asked,
  // and then this is exactly the expression it has always been.
  const bool gather =
      LogitsGatherEnabled() && step.prompt_logprob_indices.empty();
  const std::vector<int32_t> kNoGather;
  const std::vector<int32_t>& gather_li = gather ? step.logits_indices : kNoGather;

  // THE FORWARD. DEFAULT (gather ON): the *Device entry points keep the
  // [num_reqs,vocab] logits ON DEVICE and hand them to the sampler with no
  // full-logits D2H. VT_LOGITS_GATHER=0: the host Forward returns full
  // [T,vocab] logits and sample_tokens re-gathers on host (unchanged path).
  ModelForwardInput forward_input{
      .token_ids = token_ids,
      .positions = positions,
      .attn_meta = attn_meta,
      .gdn_meta = gdn_meta,
      .attn_kv = attn_kv_,
      .gdn_state = gdn_state_,
      .config = config_,
      .queue = queue_,
      .logits_indices = gather_li,
      .num_reqs = num_reqs,
      .gdn_state_slots = gdn_state_slots_,
      .pure_decode = pure_decode,
      // SPEC-DSPARK W8 (#442): the decode-graph gate mirrors vLLM's UNIFORM
      // decode predicate, whose captured length is 1 + num_speculative_tokens
      // (cudagraph_dispatcher.py:37). 0 when speculation is off, which makes
      // the predicate reduce to today's pure-decode shape.
      .num_speculative_tokens = num_spec(),
      // W6 (#1374): the eligibility answer itself. 0 == no captured decode graph
      // in this tree serves this step.
      .uniform_query_len = uniform_qlen.value_or(0),
      .gather_logits = gather,
      // SPEC-MTP I5d: capture the target's post-final-norm [T,H] hidden for the
      // MTP drafter. Non-null only when a speculator is configured — the Qwen3.5
      // dense/MoE forward then routes to ForwardDeviceTap (byte-identical logits).
      // Null on every default step, so the forward path is byte-identical.
      // SPEC-DFLASH D5: the DFlash drafter conditions on the MULTI-tap instead,
      // so it uses aux_tap (below) and leaves hidden_tap null. SPEC-NGRAM: the
      // draft-free proposer needs no hidden state at all, so leave it null too.
      .hidden_tap = (spec_on() && !use_dflash() && !use_ngram())
                        ? &exec_state_.spec_hidden
                        : nullptr,
  };
  // SPEC-DFLASH D5: on the verify forward capture the D1 multi-tap (the residual
  // stream at the draft's target_layer_ids) as [T, H×taps]. Mutually exclusive
  // with hidden_tap; routes the dense/MoE forward to ForwardDeviceMultiTap
  // (byte-identical logits). Null on every non-dflash step.
  if (use_dflash()) {
    exec_state_.spec_aux.layer_ids = dflash_tap_layer_ids_;
    forward_input.aux_tap = &exec_state_.spec_aux;
  }
  // W4: non-null whenever the mirror is engaged (integrated OR discrete — see
  // `async_device_mirror()`; the comment here used to say "discrete-CUDA" and
  // that was wrong, #2710), where the combine above patched the DEVICE ids and
  // `token_ids` is deliberately stale for decode rows. Set after construction
  // because the field sits at the END of the struct, where it cannot shift the
  // positional aggregate initializers other callers use.
  forward_input.device_token_ids = device_input_ids;
  // ENG-ASYNC-DEVICE-IDS-REFUSAL (#2710): and WHETHER the host vector actually
  // disagrees with that buffer, which is what `ModelRegistry::Forward` refuses
  // on. The one derivation, taken above the multimodal embed so that
  // `MmEmbedInputs` could be told the same fact (#2730), and READ here rather
  // than recomputed.
  forward_input.host_token_ids_stale = host_token_ids_stale;
  // ENG-MM-INPUT-PIPELINE P2 (#2379): THE assignment this row exists for. Set
  // after aggregate construction for the same reason `device_token_ids` is —
  // the field sits past the positional initializers other callers use. nullopt
  // on every text step, and the forward is then byte-identical.
  if (mm_buffers.has_value()) forward_input.mm = mm_buffers->mm;
  // KV-DSV4-MULTICACHE W3 (#2068): the third cache channel. Null on every
  // uniform topology (`multi_kv_index_` is default-constructed there), so every
  // existing forward receives exactly what it received before.
  if (multi_cache_topology_) forward_input.multi_kv = &multi_kv_index_;

  // ENG-ASYNC-SCHED depth-2 LIFETIME GUARD — the MOVED drain (mirror path only).
  // The bulk host prep above (update_states + prepare_inputs + attention/GDN
  // metadata + forward_input assembly) has overlapped the previous step's GPU
  // tail. Drain it NOW, before we (1) free the previous step's exec_state_ — the
  // device logits the in-flight sampler still reads — and (2) let the forward
  // stage this step's inputs / replay the decode graph into the persistent buffers
  // the previous step's kernels read. `device_input_ids` (dev->input_ids) and the
  // combine/replay this step already enqueued live in the mirror's own persistent
  // buffers, not exec_state_, so the reset does not disturb them; the forward's
  // CopyInPlace of dev->input_ids into the graph inputs is main-queue-ordered after
  // this step's combine. On the DEFAULT (mirror OFF) path async_forward_in_flight_
  // was already cleared at the top and exec_state_ already reset, so this block is
  // skipped (mirror==false) and the path is byte-identical.
  if (mirror) {
    // VT_ASYNC_EXECUTOR (decode-graph slot double-buffer): if the previous step's
    // stashed logits are a NON-OWNING decode-graph slot view, the exec_state_ reset
    // frees nothing the in-flight async sampler reads (hazard-A — the eager owning
    // WrapDeviceLogits pool block — is absent for a view), and the model's 2-slot
    // parity ring + per-slot reuse event already guard the persistent decode-graph
    // host inputs against this step's replay (hazard-C). Both hazards handled, so
    // SKIP the main-queue Synchronize and let this step's host prep + replay overlap
    // the previous step's GPU tail — the c16/c32 unlock. An eager/mixed previous
    // step still owns its logits pool block, whose reset WOULD UAF the async
    // sampler, so it drains exactly as before. The reset runs on BOTH paths: on the
    // skip path it releases the no-op-deleter view and clears req_ids/discard for
    // this step's stash WITHOUT touching the still-live, still-referenced slot
    // buffer. Inert unless VT_ASYNC_EXECUTOR=1 (async_executor() default OFF), where
    // skip_drain is always false and the block is byte-identical to the drain.
    const bool skip_drain = async_executor() && exec_state_.logits.on_device() &&
                            exec_state_.logits.non_owning_view;
    if (skip_drain && std::getenv("VT_ASYNC_EXECUTOR_TRACE") != nullptr) {
      // Diagnostic (test-only): confirm the overlap path is actually engaged — a
      // skip count of 0 would mean async_executor()/non_owning_view never resolved
      // and the lever is inert. Single engine thread, so a plain static suffices.
      static long long kSkips = 0;
      ++kSkips;
      if (kSkips == 1 || kSkips % 200 == 0)
        std::fprintf(stderr, "[VT_ASYNC_EXECUTOR] drain skipped x%lld\n", kSkips);
    }
    if (!skip_drain && async_forward_in_flight_) {
      vt::GetBackend(queue_.device.type).Synchronize(queue_);
      async_forward_in_flight_ = false;
    }
    exec_state_ = ExecuteModelState{};
  }

  // KV-EXTERNAL-CACHE (LMCache): apply any external-prefix loads recorded by the
  // scheduler's connector for THIS step into the freshly-allocated KV blocks
  // BEFORE the forward reads them (load-before-compute, base.py:293). Inert when
  // no worker-capable connector is set — the KV cache is byte-identical to
  // production and no sync is issued.
  if (kv_connector_ != nullptr) {
    ConnectorLoadExternalKv();
  }

  ForwardLogits logits = ModelRegistry::Forward(*model_, forward_input);

  // KV-EXTERNAL-CACHE (LMCache): after the forward has written this step's KV,
  // STORE every newly-complete prompt block to the external cache (the worker
  // half of LMCacheConnectorV1's synchronous save). Inert (no sync, no I/O)
  // unless a worker-capable connector is set.
  if (kv_connector_ != nullptr) {
    ConnectorStorePromptKv(scheduler_output);
  }

  // Stash for sample_tokens (upstream ExecuteModelState).
  exec_state_.num_actual_tokens = scheduler_output.total_num_scheduled_tokens;
  exec_state_.num_reqs = num_reqs;
  exec_state_.logits = std::move(logits);
  exec_state_.step = std::move(step);
  exec_state_.attn_meta = std::move(attn_meta);
  exec_state_.gdn_meta = std::move(gdn_meta);
  exec_state_.req_ids.reserve(static_cast<size_t>(num_reqs));
  for (int i = 0; i < num_reqs; ++i) {
    exec_state_.req_ids.push_back(*input_batch_.req_ids[static_cast<size_t>(i)]);
  }

  // discard_request_mask (gpu_model_runner.py:2029-2051): a scheduled request
  // whose optimistic seq_len (num_computed + num_scheduled == step.seq_lens[i])
  // has not yet reached its total known token count (num_tokens_no_spec[i] ==
  // request.num_tokens) is still consuming prefill and must NOT sample this step.
  // NOTE: under async scheduling step.seq_lens[i] == num_tokens + placeholders
  // for a decode request (the reserved in-flight token), so `< num_tokens`
  // strictly selects prefill chunks only. sample_tokens / get_output() clear the
  // sampled token for these rows, honoring the scheduler contract that a
  // still-prefilling request returns EMPTY token ids (scheduler.py:1888-1890).
  exec_state_.discard.assign(static_cast<size_t>(num_reqs), 0);
  for (int i = 0; i < num_reqs; ++i) {
    const int32_t seq_len = exec_state_.step.seq_lens[static_cast<size_t>(i)];
    const int32_t num_tokens =
        input_batch_.num_tokens_no_spec[static_cast<size_t>(i)];
    exec_state_.discard[static_cast<size_t>(i)] =
        (seq_len < num_tokens) ? 1 : 0;
  }

  return std::nullopt;  // MRV2 split: forward done, sample separately.
}

// Assemble the [num_reqs, vocab] logits + apply the grammar bitmask. Extracted
// verbatim from sample_tokens (same ops, same order) so sample_tokens and
// sample_tokens_async share ONE assembly path (byte-identical sync behavior).
vt::Tensor GPUModelRunner::assemble_sample_logits(
    const std::optional<GrammarOutput>& grammar_output,
    std::vector<float>& sampled_logits) {
  // The number of LOGIT ROWS the sampler runs on. Non-speculative (the
  // production default) this is exactly num_reqs — prepare_inputs sets
  // cu_num_logits = arange(num_reqs+1), so cu_num_logits.back() == num_reqs and
  // every expression below is byte-identical to the pre-SPEC-REJECTION code.
  // Under spec decode it is Σ(1 + k_i) (spec §2.4 / StepInputs::cu_num_logits).
  const int num_logits = step_num_logits();
  const int64_t vocab = config_.vocab_size;

  // Assemble the [num_logits, vocab] logits the sampler runs on. Three cases:
  //
  //  (A) DEVICE path (default, gather ON): the forward already produced exactly
  //      num_logits rows in request order, ON DEVICE — hand the sampler the device
  //      tensor directly (its argmax / temperature / top-k/top-p kernels run
  //      on-device; only the sampled token ids come back). No full-logits D2H.
  //  (A') VT_GPU_SAMPLE=0 A/B: Download the device [num_logits,vocab] logits to a
  //      host buffer and sample from it (the OLD download-then-sample path).
  //  (B) HOST path (VT_LOGITS_GATHER=0): full [num_actual_tokens,vocab] host
  //      logits — re-gather the per-request rows on host via logits_indices,
  //      exactly as before.
  vt::Tensor logits;
  ForwardLogits& fl = exec_state_.logits;
  if (fl.on_device()) {
    VT_CHECK(fl.rows == num_logits,
             "sample_tokens: device logits rows must equal the expanded logits rows");
    if (GpuSampleEnabled()) {
      logits = fl.device_tensor;  // (A) sample straight off device
    } else {
      // (A') download then sample (A/B: reproduce the pre-change host path).
      sampled_logits.resize(static_cast<size_t>(num_logits) *
                            static_cast<size_t>(vocab));
      vt::Backend& b = vt::GetBackend(queue_.device.type);
      b.Copy(queue_, sampled_logits.data(), fl.device_tensor.data,
             sampled_logits.size() * sizeof(float));
      b.Synchronize(queue_);
      logits = sample_logits_staging_.Stage(
          queue_.device, queue_, sampled_logits.data(), vt::DType::kF32,
          {static_cast<int64_t>(num_logits), vocab});
    }
  } else if (fl.rows == num_logits) {
    logits = sample_logits_staging_.Stage(
        queue_.device, queue_, fl.host.data(), vt::DType::kF32,
        {static_cast<int64_t>(num_logits), vocab});
  } else {
    // (B) VT_LOGITS_GATHER=0: re-gather the logits rows from full [T,vocab] host.
    sampled_logits.resize(static_cast<size_t>(num_logits) *
                          static_cast<size_t>(vocab));
    for (int i = 0; i < num_logits; ++i) {
      const int row = exec_state_.step.logits_indices[static_cast<size_t>(i)];
      std::memcpy(
          sampled_logits.data() + static_cast<size_t>(i) * static_cast<size_t>(vocab),
          fl.host.data() +
              static_cast<size_t>(row) * static_cast<size_t>(vocab),
          static_cast<size_t>(vocab) * sizeof(float));
    }
    logits = sample_logits_staging_.Stage(
        queue_.device, queue_, sampled_logits.data(), vt::DType::kF32,
        {static_cast<int64_t>(num_logits), vocab});
  }

  // Apply the structured-output grammar bitmask (utils.py apply_grammar_bitmask)
  // to the gathered [num_logits, vocab] logits BEFORE sampling, when a structured
  // request is scheduled this step (gpu_model_runner.py:4462-4466). The grammar
  // bitmask over the EXPANDED spec rows (a bitmask row per draft position) is
  // DEFERRED with SPEC-MTP (spec §Protocol-compliance "Grammar bitmask under
  // spec decode: OUT of scope"), so the spec-token map stays empty (per-req
  // offset 0) — correct while num_draft_tokens == 0, which is the only state
  // the runner can reach today.
  if (grammar_output.has_value()) {
    apply_grammar_bitmask(*grammar_output, exec_state_.req_ids, {}, queue_,
                          logits);
  }
  return logits;
}

// step_num_logits: the number of expanded logit rows this step (spec §2.4).
// cu_num_logits is arange(num_reqs+1) on the non-speculative path, so this is
// exactly num_reqs there; the `.empty()` guard covers a StepInputs built by an
// older direct-runner test that never went through prepare_inputs.
int GPUModelRunner::step_num_logits() const {
  const std::vector<int32_t>& cu = exec_state_.step.cu_num_logits;
  return cu.empty() ? exec_state_.num_reqs : cu.back();
}

// collect_prompt_logprobs — 1:1 vllm/v1/worker/gpu_model_runner.py:5612-5719
// (`_get_prompt_logprobs_dict`), minus the row selection, which prepare_inputs
// did (see StepInputs::PromptLogprobRows). What is left is upstream's scoring
// (:5688-5697), its slice-by-slice accumulation (:5698-5706) and its
// emit-and-drop on the final chunk (:5665-5667, :5709-5712).
void GPUModelRunner::collect_prompt_logprobs(
    std::map<std::string, LogprobsTensors>& prompt_logprobs_dict) {
  const std::vector<StepInputs::PromptLogprobRows>& rows =
      exec_state_.step.prompt_logprob_rows;
  // An in-progress entry can outlive its request: an ABORT mid-prompt drops the
  // request from the input batch, and upstream frees the same state with the
  // request object at :1199. Swept on every call, including the early return
  // below, so a stale tensor cannot survive to the next request with that id.
  if (!in_progress_prompt_logprobs_.empty()) drop_stale_prompt_logprobs();
  if (rows.empty()) return;

  const int64_t vocab = config_.vocab_size;
  ForwardLogits& fl = exec_state_.logits;

  for (const StepInputs::PromptLogprobRows& r : rows) {
    // The accumulated tensor covers num_prompt_tokens - 1 positions: the first
    // prompt token has no logprob because nothing precedes it (:5646-5651, and
    // logprobs.py:162-167 which fills that leading None). Created on first
    // sight and filled slice by slice across chunks.
    auto [it, inserted] = in_progress_prompt_logprobs_.try_emplace(r.req_id);
    if (inserted) {
      // The height comes from the REQUEST, not from this chunk: prompt length
      // less one. dst_start + num_rows reaches exactly that on the final chunk,
      // and prepare_inputs tiles [0, num_prompt_tokens - 1) in order.
      it->second.num_positions = prompt_logprob_positions(r.req_id);
      it->second.num_tokens_per_position = r.num_prompt_logprobs + 1;
      it->second.logprob_token_ids.assign(
          static_cast<size_t>(it->second.num_positions) *
              static_cast<size_t>(it->second.num_tokens_per_position),
          0);
      it->second.logprobs.assign(it->second.logprob_token_ids.size(), 0.0f);
      it->second.selected_token_ranks.assign(
          static_cast<size_t>(it->second.num_positions), 0);
    }
    LogprobsTensors& acc = it->second;

    if (r.num_rows > 0) {
      // The step that OWES rows took the full-logits path in execute_model, so
      // there is one row per scheduled token and a prompt row is found by its
      // token-stream index. This belongs HERE, at the slice, not at the top of
      // the function: prepare_inputs deliberately keeps a final-chunk entry with
      // num_rows == 0 on the exact-prefill edge (:5668-5673), and that entry
      // contributes no gather index, so the step correctly kept the gathered
      // lm_head and has num_reqs rows, not num_actual_tokens. Asserting on
      // `rows` non-empty instead of on the slice threw whenever any OTHER
      // request in the same step contributed more than one token — and VT_CHECK
      // throws out of engine.step(), taking the whole batch down with it, not
      // just the asking request. Regression: test_llm_engine §9(h).
      VT_CHECK(fl.rows == exec_state_.num_actual_tokens,
               "collect_prompt_logprobs: a step that scores prompt rows must "
               "carry full logits");
      // A request's prompt rows are the CONTIGUOUS run starting at its first
      // scheduled token (prepare_inputs built them as query_start + [0,
      // num_rows)), so this is one slice, on device or on host.
      const int64_t first = exec_state_.step.prompt_logprob_indices
                                [static_cast<size_t>(r.src_start)];
      vt::Tensor view;
      if (fl.on_device()) {
        view = fl.device_tensor.Slice(0, first, first + r.num_rows);
      } else {
        view = prompt_logprobs_staging_.Stage(
            queue_.device, queue_,
            fl.host.data() +
                static_cast<size_t>(first) * static_cast<size_t>(vocab),
            vt::DType::kF32,
            {static_cast<int64_t>(r.num_rows), vocab});
      }

      const LogprobsTensors chunk = sampler_.compute_prompt_logprobs(
          queue_, view, r.num_prompt_logprobs, r.target_token_ids);

      // Copy into slice(start_idx, start_idx + num_logits) of the accumulated
      // tensor (:5698-5706).
      const size_t width =
          static_cast<size_t>(acc.num_tokens_per_position);
      VT_CHECK(chunk.num_tokens_per_position == acc.num_tokens_per_position,
               "collect_prompt_logprobs: chunk width must match the request's");
      VT_CHECK(r.dst_start + r.num_rows <= acc.num_positions,
               "collect_prompt_logprobs: chunk overruns the prompt tensor");
      const size_t dst = static_cast<size_t>(r.dst_start) * width;
      std::copy(chunk.logprob_token_ids.begin(), chunk.logprob_token_ids.end(),
                acc.logprob_token_ids.begin() +
                    static_cast<std::ptrdiff_t>(dst));
      std::copy(chunk.logprobs.begin(), chunk.logprobs.end(),
                acc.logprobs.begin() + static_cast<std::ptrdiff_t>(dst));
      std::copy(chunk.selected_token_ranks.begin(),
                chunk.selected_token_ranks.end(),
                acc.selected_token_ranks.begin() +
                    static_cast<std::ptrdiff_t>(r.dst_start));
    }

    if (r.final_chunk) {
      // The prompt is fully scored: hand the tensor to the output and forget
      // the request on BOTH sides (:5709-5712).
      prompt_logprobs_dict[r.req_id] = std::move(acc);
      in_progress_prompt_logprobs_.erase(it);
      input_batch_.num_prompt_logprobs.erase(r.req_id);
    }
  }
}

// The height of a request's prompt-logprob tensor: num_prompt_tokens - 1
// (gpu_model_runner.py:5648-5650). Read off the input batch, which is where the
// request's prompt length lives for us.
int GPUModelRunner::prompt_logprob_positions(const std::string& req_id) const {
  const auto it = input_batch_.req_id_to_index.find(req_id);
  VT_CHECK(it != input_batch_.req_id_to_index.end(),
           "prompt_logprob_positions: request is not in the batch");
  return input_batch_.num_prompt_tokens[static_cast<size_t>(it->second)] - 1;
}

// Drop in-progress prompt logprobs for requests the input batch no longer
// carries — an abort mid-prompt. Upstream frees the same state with the request
// object itself (gpu_model_runner.py:1199).
void GPUModelRunner::drop_stale_prompt_logprobs() {
  for (auto it = in_progress_prompt_logprobs_.begin();
       it != in_progress_prompt_logprobs_.end();) {
    if (input_batch_.num_prompt_logprobs.count(it->first) == 0) {
      it = in_progress_prompt_logprobs_.erase(it);
    } else {
      ++it;
    }
  }
}

// The SPEC-DECODE VERIFY half (SPEC-REJECTION I3). Ported from
// gpu/model_runner.py:1069-1077 (the rejection_sampler call) +
// rejection_sampler.py:111 (draft_sampled = input_ids[logits_indices]) +
// postprocess_sampled's num_accepted scatter (model_states/mamba_hybrid.py:
// 290-310, `max(num_sampled, 1)`).
//
// UNREACHABLE ON THE PRODUCTION DEFAULT: the only caller gates on
// exec_state_.step.num_draft_tokens > 0, which requires the scheduler to have
// populated scheduled_spec_decode_tokens, which requires a configured
// SpeculativeConfig. See the greedy accept rule in
// include/vllm/v1/spec_decode/rejection_sampler.h.
ModelRunnerOutput GPUModelRunner::sample_tokens_with_rejection(
    vt::Tensor& logits, VerifyDownload download) {
  const int num_reqs = exec_state_.num_reqs;
  const StepInputs& step = exec_state_.step;
  const int num_logits = step_num_logits();

  // draft_sampled = input_ids[logits_indices] (rejection_sampler.py:111): the
  // scheduled input token at each expanded row. Row cu[r]+i+1 is request r's
  // i-th draft token (the +1 the accept walk reads).
  std::vector<int32_t> draft_sampled(static_cast<size_t>(num_logits));
  for (int j = 0; j < num_logits; ++j) {
    draft_sampled[static_cast<size_t>(j)] =
        step.input_token_ids[static_cast<size_t>(step.logits_indices[static_cast<size_t>(j)])];
  }
  // A row still consuming its prefill chunk samples and rejects nothing
  // (_get_num_sampled_and_rejected_kernel; our discard mask is the same predicate).
  std::vector<char> chunked_prefilling(static_cast<size_t>(num_reqs), 0);
  for (int i = 0; i < num_reqs && i < static_cast<int>(exec_state_.discard.size()); ++i) {
    chunked_prefilling[static_cast<size_t>(i)] =
        exec_state_.discard[static_cast<size_t>(i)] != 0 ? 1 : 0;
  }

  int max_k = 0;
  for (const int32_t k : step.num_draft_tokens_per_req) max_k = k > max_k ? k : max_k;
  const RejectionSampler rejection_sampler(max_k);
  // SPEC-DFLASH2 A2-2: the accept walk is the same kernel on the same queue on
  // both routes; `download` decides only where the caller waits for its two
  // outputs. kMainQueueDrain goes through `forward`, which is byte-for-byte the
  // pre-split behaviour.
  RejectionSamplerOutput rs;
  if (download == VerifyDownload::kMainQueueDrain) {
    rs = rejection_sampler.forward(queue_, logits, draft_sampled, step.cu_num_logits,
                                   chunked_prefilling);
  } else {
    // The COPY-QUEUE route. `verify` issues the walk on the MAIN queue and
    // returns with its outputs still on the device and nothing waited on. We
    // then do exactly what AsyncOutput does for the non-spec sampled ids
    // (async_utils.py:29-44): record a fork event on the main queue, make the
    // copy queue wait it, issue the two D2H copies THERE into PAGE-LOCKED host
    // memory, record a completion event on the copy queue, and block the host on
    // that event alone. The main queue is never synchronized; the host still is,
    // right here, and the paragraph below says what that does and does not buy.
    //
    // The destination is `verify_download_staging_` and not a `std::vector` on
    // purpose: a device-to-host `cudaMemcpyAsync` into PAGEABLE memory is
    // host-synchronous, so a pageable destination would make the two events
    // above decorative and no later wave could obtain any overlap from them
    // (PinnedGrowStaging, runner.h). The host vectors below are filled from that
    // block AFTER the wait, which is a host-to-host copy of a few hundred ints.
    //
    // `dev_out` outlives the wait, so the buffers the kernel writes and the
    // staging it reads are alive for the whole in-flight window — the lifetime
    // contract RejectionSamplerDeviceOutput exists to make explicit.
    //
    // WHAT THIS DOES NOT YET DO, stated here so nobody reads more into it. The
    // host still waits IN STEP, because the write-back and `propose_drafts`
    // below both need these numbers on the host, and that is A2-3's and A2-4's
    // to remove. Waiting on `verify_ready_event_` is not free just because it is
    // an event: the copy waited the fork event, so by the time it completes the
    // main queue has drained anyway. What changed is the SHAPE — the copy is no
    // longer a main-stream operation and the wait is on a copy-queue event that
    // a later wave can move past the propose. The drain is REMOVABLE from here;
    // it is not removed here, and no overlap is claimed on this head.
    RejectionSamplerDeviceOutput dev_out =
        rejection_sampler.verify(queue_, logits, draft_sampled, step.cu_num_logits);
    const int64_t rows = dev_out.num_reqs();
    const int64_t width = dev_out.width();
    std::vector<int32_t> host_sampled(static_cast<size_t>(rows * width));
    std::vector<int32_t> host_num_sampled(static_cast<size_t>(rows));
    if (rows != 0) {
      vt::Backend& backend = vt::GetBackend(dev_out.device().type);
      vt::Queue& copy_q = get_or_create_async_copy_queue();
      ensure_verify_events();
      // One page-locked block for both copies: `sampled` [rows, width] first,
      // then `num_sampled` [rows].
      const size_t sampled_elems = static_cast<size_t>(rows * width);
      int32_t* pinned =
          verify_download_staging_.Get(backend, sampled_elems + static_cast<size_t>(rows));
      backend.RecordEvent(verify_fork_event_, queue_);
      backend.QueueWaitEvent(copy_q, verify_fork_event_);
      dev_out.CopyToHost(copy_q, pinned, pinned + sampled_elems);
      backend.RecordEvent(verify_ready_event_, copy_q);
      backend.SynchronizeEvent(verify_ready_event_);
      std::copy(pinned, pinned + sampled_elems, host_sampled.begin());
      std::copy(pinned + sampled_elems, pinned + sampled_elems + static_cast<size_t>(rows),
                host_num_sampled.begin());
    }
    rs = RejectionSampler::finalize(host_sampled, width, host_num_sampled,
                                    step.cu_num_logits, chunked_prefilling);
  }

  ModelRunnerOutput out;
  out.req_ids.reserve(static_cast<size_t>(num_reqs));
  out.sampled_token_ids.reserve(static_cast<size_t>(num_reqs));
  // SPEC-DFLASH2 A2-4 (#3004): rebuilt per verify step, exactly like
  // `prev_num_draft_len_`. Cleared HERE and not at the correction, because the
  // correction runs at the START of the next step and a map that outlived its
  // step would answer for a verify two steps old — the state `ProposedDraft
  // Ledger` exists to make unrepresentable one seam over (prepare_inputs.h).
  prev_valid_sampled_count_.clear();
  for (int i = 0; i < num_reqs; ++i) {
    const std::string& req_id = exec_state_.req_ids[static_cast<size_t>(i)];
    out.req_ids.push_back(req_id);
    out.req_id_to_index[req_id] = i;

    // num_accepted_tokens[i] = max(num_sampled, 1) — the GDN recurrent-state
    // slot select reads it next step (spec §3 step 3 / §2.4).
    const int32_t ns = rs.num_sampled[static_cast<size_t>(i)];
    input_batch_.num_accepted_tokens[static_cast<size_t>(i)] = ns > 1 ? ns : 1;
    // Acceptance telemetry: k drafts verified this row, (num_sampled-1) accepted
    // (the trailing token is the bonus/replacement, not a draft). Chunked-prefill
    // rows carry no drafts.
    const int32_t kr = step.num_draft_tokens_per_req.empty()
                           ? 0
                           : step.num_draft_tokens_per_req[static_cast<size_t>(i)];
    // The PER-ROW reading, and it is deliberately not the route predicate: this
    // decides only whether row i contributes to the acceptance telemetry.
    if (RowCarriesDraftTokens(kr) && !chunked_prefilling[static_cast<size_t>(i)]) {
      spec_drafts_proposed_ += kr;
      spec_drafts_accepted_ += (ns > 1 ? ns - 1 : 0);
      // SPEC-MTP-K-GT-1 (#81): the same accounting split by DEPTH. The rejection
      // sampler accepts a PREFIX of the draft, so with `ns - 1` drafts accepted
      // the draft at 0-based depth d was accepted exactly when `d < ns - 1`.
      // #81's M1 asks for per-depth acceptance, and it is the only signal an
      // acceptance-driven depth policy could read: the aggregate ratio cannot
      // tell "every request accepted its first draft" from "one request accepted
      // three". The vectors stay EMPTY when nothing speculates.
      if (static_cast<int32_t>(spec_drafts_proposed_by_depth_.size()) < kr) {
        spec_drafts_proposed_by_depth_.resize(static_cast<size_t>(kr), 0);
        spec_drafts_accepted_by_depth_.resize(static_cast<size_t>(kr), 0);
      }
      const int32_t accepted_drafts = ns > 1 ? ns - 1 : 0;
      for (int32_t d = 0; d < kr; ++d) {
        spec_drafts_proposed_by_depth_[static_cast<size_t>(d)] += 1;
        if (d < accepted_drafts) {
          spec_drafts_accepted_by_depth_[static_cast<size_t>(d)] += 1;
        }
      }
      // PER-BLOCK acceptance trace (VT_SPEC_TRACE=1), off by default. The
      // aggregate proposed/accepted totals cannot distinguish a diffuse
      // per-block difference from one displaced block, which is exactly the
      // question a cross-draft acceptance delta raises. This prints the whole
      // proposal and what the target did with it, so the answer is read off the
      // log rather than argued. `pos` is the request's token count BEFORE this
      // step's write-back, so blocks from two runs line up by position.
      static const bool spec_trace = [] {
        const char* v = std::getenv("VT_SPEC_TRACE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
      }();
      if (spec_trace) {
        const int32_t base = step.cu_num_logits[static_cast<size_t>(i)];
        std::string drafts;
        for (int32_t j = 0; j < kr; ++j) {
          drafts += std::to_string(
              draft_sampled[static_cast<size_t>(base) + static_cast<size_t>(j) + 1]);
          drafts += ' ';
        }
        std::string emitted;
        for (const int32_t t : rs.sampled_token_ids[static_cast<size_t>(i)]) {
          emitted += std::to_string(t);
          emitted += ' ';
        }
        std::fprintf(
            stderr,
            "[SPECTRACE] req=%s pos=%d k=%d ns=%d acc=%d draft=[ %s] emit=[ %s]\n",
            req_id.c_str(),
            static_cast<int>(input_batch_.num_tokens_no_spec[static_cast<size_t>(i)]),
            static_cast<int>(kr), static_cast<int>(ns),
            static_cast<int>(ns > 1 ? ns - 1 : 0), drafts.c_str(), emitted.c_str());
      }
    }

    if (chunked_prefilling[static_cast<size_t>(i)]) {
      out.sampled_token_ids.push_back({});
      continue;
    }
    const std::vector<int32_t>& toks = rs.sampled_token_ids[static_cast<size_t>(i)];
    out.sampled_token_ids.push_back(toks);
    // SPEC-DFLASH2 A2-4 (#3004): `valid_sampled_token_count` for this row —
    // how many tokens the accept walk emitted, which is upstream's
    // `_get_valid_sampled_token_count()` (gpu_model_runner.py:1530 @ pin
    // 5559679229). The NEXT step's deferred correction reads it, so it is keyed
    // by req_id and not by slot: `condense` and `swap_states` move slots between
    // the two steps. Written only for a row that ran the accept walk — a
    // chunked-prefilling row `continue`s above and records nothing, which is
    // what makes the correction's refusal on a missing entry meaningful.
    prev_valid_sampled_count_[req_id] = static_cast<int>(toks.size());

    // Write-back: append every emitted token (accepted drafts + the bonus /
    // replacement) to slot i's token row, exactly as the non-spec path does for
    // its single token.
    for (const int32_t tok : toks) {
      const int n = input_batch_.num_tokens_no_spec[static_cast<size_t>(i)];
      const size_t idx = static_cast<size_t>(i) *
                             static_cast<size_t>(input_batch_.max_model_len) +
                         static_cast<size_t>(n);
      if (idx < input_batch_.token_ids_cpu.size()) {
        input_batch_.token_ids_cpu[idx] = tok;
      }
      input_batch_.num_tokens_no_spec[static_cast<size_t>(i)] = n + 1;
      auto& out_ids = input_batch_.req_output_token_ids[static_cast<size_t>(i)];
      if (out_ids.has_value()) {
        out_ids->push_back(tok);
      }
    }
    if (!toks.empty()) {
      input_batch_.last_sampled_tokens[static_cast<size_t>(i)] = toks.back();
    }
  }
  return out;
}

// ─── SPEC-DFLASH2 A2-4 (#3004): apply_deferred_spec_decode_corrections ──────
// See the header for what this replaces, where upstream calls it, and the one
// host dependency it still carries.

void GPUModelRunner::apply_deferred_spec_decode_corrections() {
  if (deferred_spec_corrections_.empty()) return;
  for (const DeferredSpecDecodeCorrection& c : deferred_spec_corrections_) {
    const auto vit = prev_valid_sampled_count_.find(c.req_id);
    // A request whose previous step scheduled drafts ran the accept walk, so it
    // has a valid-sampled count. Refuse rather than skip: a skipped correction
    // leaves an OPTIMISTIC position in `num_computed_tokens_cpu`, which
    // re-attends the rejected drafts' positions and costs acceptance while
    // emitting identical text — the class of defect no token gate here can see
    // (#1366). A silent `continue` and a refusal are indistinguishable until
    // something mutates them.
    VT_CHECK(vit != prev_valid_sampled_count_.end() && vit->second >= 1,
             "async spec deferred correction has no accept result for '" +
                 c.req_id + "': its previous step scheduled " +
                 std::to_string(c.optimistic_num_accepted) +
                 " drafts and recorded no emitted-token count");
    // num_accepted = valid_sampled_token_count - 1 (the trailing token is the
    // bonus/replacement, not an accepted draft), and
    // correction = optimistic_num_accepted - num_accepted, verbatim
    // gpu_model_runner.py:1543-1546 @ pin 5559679229.
    const int num_accepted = vit->second - 1;
    const int correction = c.optimistic_num_accepted - num_accepted;
    // The slot is the one the optimistic pass read, carried rather than looked
    // up again. Upstream re-reads `req_id_to_index` here
    // (gpu_model_runner.py:1547) because its closure runs after the model
    // LAUNCH and after `condense()`; ours runs in the same step, between the
    // splice and `prepare_inputs`, with no `condense` or `swap_states` in
    // between, so a second lookup would return the same number and would add a
    // second reading of a fact this step already established. When the call
    // moves past the launch (A2-5), it moves back to the lookup.
    const size_t s = static_cast<size_t>(c.req_index);
    VT_CHECK(s < input_batch_.num_computed_tokens_cpu.size(),
             "async spec deferred correction indexes outside the batch for '" +
                 c.req_id + "'");
    input_batch_.num_computed_tokens_cpu[s] -= correction;
    // The next step's optimistic value extends the CORRECTED position, not the
    // optimistic one the splice recorded a moment ago.
    prev_num_computed_tokens_[c.req_id] = input_batch_.num_computed_tokens_cpu[s];

    // THE CROSS-CHECK, and it is the reason this wave can replace W7's rule
    // without moving a token. While the accept result is still host-visible in
    // step, W7's structural value — the newest committed token's position,
    // `num_tokens_no_spec - 1` — is computable right here, and the optimistic
    // value minus this correction MUST equal it. It is an equality between two
    // independent derivations of the same number, so it fires on an arithmetic
    // slip in either half, and it is what makes the replacement checkable rather
    // than argued. A2-5 removes the host-visible input this compares against; it
    // does not remove the correction.
    const int structural =
        input_batch_.num_tokens_no_spec[s] - 1;
    VT_CHECK(input_batch_.num_computed_tokens_cpu[s] == structural,
             "async spec deferred correction disagrees with the structural "
             "value for '" + c.req_id + "': corrected " +
                 std::to_string(input_batch_.num_computed_tokens_cpu[s]) +
                 ", structural " + std::to_string(structural) +
                 " (optimistic accepted " +
                 std::to_string(c.optimistic_num_accepted) + ", emitted " +
                 std::to_string(vit->second) + ")");
  }
  deferred_spec_corrections_.clear();
}

// SPEC-MTP I5d: propose the next verify step's drafts after committing this
// step's accepted tokens. The accept accounting lives in num_accepted_tokens
// (num_sampled = accepted, seeded/overwritten there); num_rejected is derived.
//
// SPEC-DFLASH2 A2-2 lifted this out of `sample_tokens` so `sample_tokens_async`
// calls the SAME derivation rather than writing a second copy of it. Upstream
// has exactly one, at model_runner.py:1533-1546, because upstream has exactly
// one sampling entry point.
void GPUModelRunner::propose_after_verify(int num_reqs) {
  if (!spec_on()) return;
  std::vector<int32_t> num_sampled(static_cast<size_t>(num_reqs), 1);
  std::vector<int32_t> num_rejected(static_cast<size_t>(num_reqs), 0);
  for (int i = 0; i < num_reqs; ++i) {
    const int32_t acc = input_batch_.num_accepted_tokens[static_cast<size_t>(i)];
    const int32_t k = exec_state_.step.num_draft_tokens_per_req.empty()
                          ? 0
                          : exec_state_.step.num_draft_tokens_per_req[static_cast<size_t>(i)];
    num_sampled[static_cast<size_t>(i)] = acc;              // == accepted count
    num_rejected[static_cast<size_t>(i)] = k - (acc - 1);   // (1+k) - num_sampled
  }
  propose_drafts(num_sampled, num_rejected);
}

// SPEC-MTP I5d: propose drafts after a plain (no-draft, e.g. FIRST) decode step
// so the next step verifies them. Each generating row sampled exactly one token
// (num_sampled=1, num_rejected=0); discarded prefill-chunk rows are skipped
// inside propose_drafts.
//
// SPEC-DFLASH2 A2-4 (#2920) lifted this out of `sample_tokens` so
// `sample_tokens_async`'s decode arm calls the SAME derivation rather than
// writing a second copy of it — the shape A2-2 already established for
// `propose_after_verify` above. Upstream needs neither, because its single
// `if self.speculator is not None:` tail (gpu/model_runner.py:1524-1547 @ pin
// 5559679229) runs after BOTH of `sample()`'s routes.
void GPUModelRunner::propose_after_decode(int num_reqs) {
  if (!spec_on()) return;
  std::vector<int32_t> num_sampled(static_cast<size_t>(num_reqs), 1);
  std::vector<int32_t> num_rejected(static_cast<size_t>(num_reqs), 0);
  propose_drafts(num_sampled, num_rejected);
}

// ARCH-ONE-SURFACE ROW 6: the pooling counterpart of sample_tokens. Mirror of
// gpu/model_runner.py:1586-1607 (pool instead of sample) over the landed
// PoolingRunner (pool/pooling_runner.py:29-42). The stashed forward result of
// the pooling arch is the [rows, hidden] POST-FINAL-NORM HIDDEN (the model has
// no lm_head — adapters.py:135-151), already gathered at logits_indices on the
// default path, which for LAST pooling IS upstream's
// `hidden_states[input_batch.logits_indices]` (pooling_runner.py:36).
ModelRunnerOutput GPUModelRunner::pool_tokens() {
  const int num_reqs = exec_state_.num_reqs;
  const int64_t hidden = exec_state_.logits.vocab;  // == hidden_size here
  ForwardLogits& fl = exec_state_.logits;
  VT_CHECK(!fl.on_device(),
           "pool_tokens: the pooling forward returns a HOST hidden carrier");

  // One hidden row per request. Default (gather ON): the forward already
  // gathered the per-request last-token rows. VT_LOGITS_GATHER=0: re-gather on
  // host from the full [num_actual_tokens, hidden] rows, exactly as the text
  // host path re-gathers logits.
  std::vector<float> gathered;
  const float* rows = nullptr;
  if (fl.rows == num_reqs) {
    rows = fl.host.data();
  } else {
    gathered.resize(static_cast<size_t>(num_reqs) * static_cast<size_t>(hidden));
    for (int i = 0; i < num_reqs; ++i) {
      const int row = exec_state_.step.logits_indices[static_cast<size_t>(i)];
      std::memcpy(gathered.data() + static_cast<size_t>(i) *
                                        static_cast<size_t>(hidden),
                  fl.host.data() + static_cast<size_t>(row) *
                                       static_cast<size_t>(hidden),
                  static_cast<size_t>(hidden) * sizeof(float));
    }
    rows = gathered.data();
  }
  vt::Tensor hidden_rows = vt::Tensor::Contiguous(
      const_cast<float*>(rows), vt::DType::kF32, vt::Device{vt::DeviceType::kCPU, 0},
      {static_cast<int64_t>(num_reqs), hidden});

  // PoolingMetadata over the GATHERED buffer: one row per sequence (first ==
  // last == i), task embed, activation ON — the unconditional L2 normalize of
  // pooling_runner.py:38 (a per-request use_activation knob is the matryoshka/
  // dimensions residual, named in the row spec).
  vllm::PoolingMetadata md;
  for (int i = 0; i < num_reqs; ++i) {
    md.pooling_cursor.first_token_indices.push_back(i);
    md.pooling_cursor.last_token_indices.push_back(i);
    md.pooling_cursor.prompt_lens.push_back(1);
    md.pooling_cursor.seq_lens.push_back(1);
    md.pooling_cursor.num_scheduled_tokens.push_back(1);
    vllm::PoolingParams pp;
    pp.task = vllm::PoolingTask::kEmbed;
    pp.use_activation = true;
    md.pooling_params.push_back(pp);
    md.tasks.push_back(vllm::PoolingTask::kEmbed);
  }
  vllm::PoolerOutput pooled = pooling_runner_->Pool(hidden_rows, md);
  VT_CHECK(static_cast<int>(pooled.size()) == num_reqs,
           "pool_tokens: pooler must return one vector per request");

  // Validity = the request's whole prompt has been seen (seq_lens == prompt_len,
  // pooling_runner.py:40-41). Our discard mask is the SAME predicate
  // (step.seq_lens[i] < num_tokens_no_spec[i] == still consuming prefill), so a
  // chunked-prefill row reports nullopt and the request keeps running.
  ModelRunnerOutput out;
  out.req_ids.reserve(static_cast<size_t>(num_reqs));
  out.sampled_token_ids.reserve(static_cast<size_t>(num_reqs));
  out.pooler_output.reserve(static_cast<size_t>(num_reqs));
  for (int i = 0; i < num_reqs; ++i) {
    const std::string& req_id = exec_state_.req_ids[static_cast<size_t>(i)];
    out.req_ids.push_back(req_id);
    out.req_id_to_index[req_id] = i;
    out.sampled_token_ids.push_back({});  // a pooling step samples NOTHING
    if (exec_state_.discard[static_cast<size_t>(i)] != 0) {
      out.pooler_output.push_back(std::nullopt);
    } else {
      out.pooler_output.push_back(std::move(pooled[static_cast<size_t>(i)]));
    }
  }
  return out;
}

// --- #2534 INSTRUMENT: the final-logit dump ---------------------------------
//
// ONE implementation, called from BOTH sampling paths. It has to be both,
// because `async_input_combine_` is assigned DIRECTLY at the two ctor sites
// below (`AsyncRunnerEnvDefault() && ...`) and `AsyncRunnerFlagIsOn` answers
// TRUE when VT_ASYNC_RUNNER is unset -- so the production default is the
// DEVICE-RESIDENT branch of `sample_tokens_async`, not `sample_tokens`.
//
// Instrumenting only `sample_tokens` cost a 40-minute lease on 2026-09-02: the
// dump wrote nothing, and so did `VT_DEBUG_SAMPLED`, which sits a few lines
// below it in the same function and predates this row. Two silent probes in a
// function the engine never calls is what "wrong function" looks like. Grepping
// the SETTER `set_async_input_combine` and finding no caller is what made it
// look like the sync path; the field is not written through the setter.
void GPUModelRunner::dump_step_logits(const vt::Tensor& logits) {
  //
  // `VT_DUMP_LOGITS=<dir>` appends this step's full logit row for every live
  // request to `<dir>/ours_<req_id>.f32`, and its argmax to
  // `<dir>/ours_<req_id>.ids.txt`. Off unless the variable is set, read ONCE at
  // static init like `VT_DEBUG_SAMPLED` below -- never a per-step getenv in the
  // sampling loop.
  //
  // WHY HERE, and not inside the sampler. The comment above says sampling
  // mutates `logits` IN PLACE, so a dump taken after it would not be the vector
  // argmax saw; and `Sampler` has no request identity at all (SamplingMetadata
  // is keyed by dense index). This point is the last one that has BOTH the
  // untouched vector and `exec_state_.req_ids`, and it is after
  // `assemble_sample_logits` has applied the grammar bitmask, so it is the
  // vector the greedy path selects from.
  //
  // WHY IT EXISTS: the arm's whole remaining question is how big OUR per-step
  // logit delta against llama.cpp b10451 is, against the oracle's own measured
  // 0.2020-to-1.3657 self-perturbation band. Four candidate causes have died on
  // this row because nothing here could produce a logit vector. See
  // .agents/specs/qwen38-27b-q4km-logit-dump.md.
  //
  // The file layout is deliberately the one `job/oracle_logits.cpp` already
  // writes -- one file per sequence, steps concatenated, `n_steps * n_vocab`
  // little-endian f32 -- so `cmp_logits.c` diffs the two sides unmodified.
  static const char* const kDumpLogitsDir = [] {
    const char* e = std::getenv("VT_DUMP_LOGITS");
    return (e != nullptr && e[0] != '\0') ? e : nullptr;
  }();
  if (kDumpLogitsDir != nullptr) {
    // NO SILENT SKIP. The first version of this guard folded `rank == 2 &&
    // shape[0] > 0` into the `if`, so an unexpected shape wrote nothing, raised
    // nothing, and left an empty dump directory that read exactly like "the
    // instrument is off". That cost a 40-minute lease on 2026-09-02: the run
    // completed, both engines exited 0, and the alignment control reported six
    // MISSING sidecars with no way to tell which condition had failed. An
    // instrument states why it did not fire.
    VT_CHECK(logits.rank == 2,
             "VT_DUMP_LOGITS: the sample logits are not rank 2");
    VT_CHECK(logits.dtype == vt::DType::kF32,
             "VT_DUMP_LOGITS: the sample logits are not f32");
    const int64_t rows = logits.shape[0];
    const int64_t vocab = logits.shape[1];
    // Printed ONCE, so the log says what the instrument saw rather than leaving
    // an empty directory to be interpreted.
    static bool announced = false;
    if (!announced) {
      announced = true;
      std::fprintf(stderr,
                   "vt-dump-logits: dir=%s rows=%lld vocab=%lld req_ids=%zu "
                   "discard=%zu\n",
                   kDumpLogitsDir, static_cast<long long>(rows),
                   static_cast<long long>(vocab), exec_state_.req_ids.size(),
                   exec_state_.discard.size());
      for (size_t k = 0; k < exec_state_.req_ids.size() && k < 8; ++k) {
        std::fprintf(stderr, "vt-dump-logits: req_ids[%zu]=\"%s\"\n", k,
                     exec_state_.req_ids[k].c_str());
      }
    }
    VT_CHECK(rows > 0, "VT_DUMP_LOGITS: no sample rows");
    VT_CHECK(!exec_state_.req_ids.empty(),
             "VT_DUMP_LOGITS: exec_state_.req_ids is EMPTY at the sample point, "
             "so no row can be keyed by request");

    // Same staging the reviewed logits-processor path uses
    // (sample/logits_processor/builtin.cpp): a backend that answers the NARROW
    // host-addressable predicate is wrapped in place, everything else -- CPU
    // included -- copies down. Read-only here, so nothing is copied back.
    vt::Backend& b = vt::GetBackend(queue_.device.type);
    const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(vocab);
    std::vector<float> staging;
    const float* host = nullptr;
    if (b.DeviceMemoryIsHostAddressable()) {
      host = static_cast<const float*>(logits.data);
    } else {
      staging.resize(total);
      if (total != 0) b.Copy(queue_, staging.data(), logits.data, total * sizeof(float));
      b.Synchronize(queue_);
      host = staging.data();
    }

    for (int64_t i = 0; i < rows && i < static_cast<int64_t>(exec_state_.req_ids.size()); ++i) {
      // A row still consuming prefill samples a token the scheduler throws
      // away, so dumping it would put a step in the file that never appears in
      // `--output-token-ids` and silently shift every later step.
      if (i < static_cast<int64_t>(exec_state_.discard.size()) &&
          exec_state_.discard[static_cast<size_t>(i)]) {
        continue;
      }
      const std::string& req_id = exec_state_.req_ids[static_cast<size_t>(i)];
      const float* row = host + static_cast<size_t>(i) * static_cast<size_t>(vocab);

      const std::string base = std::string(kDumpLogitsDir) + "/ours_" + req_id;
      // FAIL LOUDLY. A dump that silently writes nothing when the directory is
      // missing is an instrument whose failure looks like a result: the operator
      // gets no files, no error, and then diffs nothing against the oracle. The
      // directory is the caller's to create.
      std::FILE* f = std::fopen((base + ".f32").c_str(), "ab");
      VT_CHECK(f != nullptr,
               "VT_DUMP_LOGITS: cannot open the dump file -- does the directory "
               "exist? The instrument does not create it.");
      const size_t wrote = std::fwrite(row, sizeof(float),
                                       static_cast<size_t>(vocab), f);
      std::fclose(f);
      VT_CHECK(wrote == static_cast<size_t>(vocab),
               "VT_DUMP_LOGITS: short write; the dump would be misaligned");
      // The ALIGNMENT control the spec requires: this argmax must equal the id
      // `--output-token-ids` records for the same step, or the two sides are not
      // describing the same context and no delta computed from them means
      // anything.
      int64_t best = 0;
      for (int64_t v = 1; v < vocab; ++v) {
        if (row[v] > row[best]) best = v;
      }
      const int step = input_batch_.num_tokens_no_spec[static_cast<size_t>(i)] -
                       input_batch_.num_prompt_tokens[static_cast<size_t>(i)];
      std::FILE* g = std::fopen((base + ".ids.txt").c_str(), "a");
      VT_CHECK(g != nullptr, "VT_DUMP_LOGITS: cannot open the ids sidecar");
      std::fprintf(g, "%d %lld\n", step, static_cast<long long>(best));
      std::fclose(g);
    }
  }
}

ModelRunnerOutput GPUModelRunner::sample_tokens(
    const std::optional<GrammarOutput>& grammar_output) {
  ModelRunnerOutput out;
  const int num_reqs = exec_state_.num_reqs;
  if (num_reqs == 0) {
    return out;  // 0-token flush step (nothing sampled).
  }

  // POOLING ROUTING (ARCH-ONE-SURFACE ROW 6), mirroring the model-level task
  // split of gpu/model_runner.py:1586-1607: on a POOLING model the step's
  // output is the POOLED DATA, never a sampled token. pooling_runner_ is set
  // iff the registration declares is_pooling_model (ctor), so every text arch
  // takes the sampler path below byte-identically.
  if (pooling_runner_ != nullptr) {
    return pool_tokens();
  }

  std::vector<float> sampled_logits;  // host buffer; outlives the sampler when used
  vt::Tensor logits = assemble_sample_logits(grammar_output, sampled_logits);

  dump_step_logits(logits);

  // SAMPLE-PROMPT-LOGPROBS (gpu_model_runner.py:3841-3845): score the prompt
  // rows off the SAME forward result. Done before sampling because sampling
  // mutates `logits` in place; the prompt rows are outside that view, but
  // reading them first keeps the two independent of each other. Inert unless a
  // request asked for prompt logprobs.
  collect_prompt_logprobs(out.prompt_logprobs_dict);

  // SPEC-DECODE VERIFY ROUTING (SPEC-REJECTION I3), mirroring
  // gpu/model_runner.py:1065-1077 (`if input_batch.num_draft_tokens == 0 or
  // self.rejection_sampler is None: sampler(...) else rejection_sampler(...)`).
  // exec_state_.step.num_draft_tokens is > 0 ONLY when the scheduler populated
  // scheduled_spec_decode_tokens, which needs an actually-configured
  // SpeculativeConfig. On the production default it is 0 on every step, so this
  // branch is never taken and the sampler path below is byte-identical.
  if (StepRoutesToVerify(exec_state_.step.num_draft_tokens)) {
    ModelRunnerOutput out_rej = sample_tokens_with_rejection(logits);
    out_rej.prompt_logprobs_dict = std::move(out.prompt_logprobs_dict);
    propose_after_verify(num_reqs);
    return out_rej;
  }

  // SamplingMetadata in the SAME dense [0, num_reqs) order (M1.7; CLOSES the
  // make_sampling_metadata wiring dep). Then Sampler.forward.
  const SamplingMetadata sm = input_batch_.make_sampling_metadata();
  const SamplerOutput sampler_output = sampler_.forward(queue_, logits, sm);

  // ModelRunnerOutput.logprobs (gpu_model_runner.py:3842-3851 / outputs.py):
  // the sampler's batch-wide gather_logprobs result (one row per num_logits
  // position, req order). None unless a request asked for logprobs; the
  // scheduler slices it per request (slice_request). ROAD-V1-C7 SAMPLE-LOGPROBS.
  out.logprobs = sampler_output.logprobs_tensors;

  // Build the ModelRunnerOutput + WRITE the sampled tokens BACK into the
  // InputBatch (upstream postprocess_sampled / post_update) so the next step's
  // prepare_inputs reads them at the decode position.
  out.req_ids.reserve(static_cast<size_t>(num_reqs));
  out.sampled_token_ids.reserve(static_cast<size_t>(num_reqs));
  for (int i = 0; i < num_reqs; ++i) {
    const std::string& req_id = exec_state_.req_ids[static_cast<size_t>(i)];
    out.req_ids.push_back(req_id);
    out.req_id_to_index[req_id] = i;

    // discard_request_mask (vllm/v1/outputs.py:303 valid_sampled_token_ids[i]
    // .clear()): a request still consuming prefill tokens samples a garbage
    // token at its prefill position — emit EMPTY and skip the write-back, so the
    // scheduler appends no output token for it (scheduler.py:1888-1890).
    if (i < static_cast<int>(exec_state_.discard.size()) &&
        exec_state_.discard[static_cast<size_t>(i)]) {
      out.sampled_token_ids.push_back({});
      continue;
    }

    const std::vector<int32_t>& toks = sampler_output.sampled_token_ids[
        static_cast<size_t>(i)];
    out.sampled_token_ids.push_back(toks);

    // Lab debug: VT_DEBUG_SAMPLED=1 prints every greedy token id. Read ONCE at
    // static-init — never a per-token getenv in the sampling hot loop (matches
    // the "read once, never per-step getenv" discipline stated above).
    static const bool kDebugSampled = [] {
      const char* e = std::getenv("VT_DEBUG_SAMPLED");
      return e != nullptr && e[0] == '1';
    }();
    if (kDebugSampled && !toks.empty()) {
      std::fprintf(stderr, "vt-debug sampled req=%d tok=%d\n", i, toks.front());
    }

    // Write-back: append each sampled token to slot i's token row so it becomes
    // the input at its position next step. num_tokens_no_spec is the next free
    // column (== num_prompt + generated-so-far). T0 non-spec decode = 1 token.
    for (const int32_t tok : toks) {
      const int n = input_batch_.num_tokens_no_spec[static_cast<size_t>(i)];
      const size_t idx = static_cast<size_t>(i) *
                             static_cast<size_t>(input_batch_.max_model_len) +
                         static_cast<size_t>(n);
      if (idx < input_batch_.token_ids_cpu.size()) {
        input_batch_.token_ids_cpu[idx] = tok;
      }
      input_batch_.num_tokens_no_spec[static_cast<size_t>(i)] = n + 1;
      auto& out_ids = input_batch_.req_output_token_ids[static_cast<size_t>(i)];
      if (out_ids.has_value()) {
        out_ids->push_back(tok);
      }
    }
    // Async-scheduling: record this step's last sampled id per req_state so the
    // next step's combine_sampled_and_draft_tokens can build its decode input id
    // without a sampled-id host round-trip (post_update / states.py, T0 non-spec:
    // one token). Inert on the sync path; the write is cheap and keeps the async
    // buffer coherent for the DGX A/B. (On CUDA this becomes the on-GPU
    // last_sampled_tokens scatter — no D2H.)
    if (!toks.empty()) {
      input_batch_.last_sampled_tokens[static_cast<size_t>(i)] = toks.back();
    }
  }
  // ENG-ASYNC-SCHED W4: keep the DEVICE mirror coherent when the SYNCHRONOUS
  // sampler ran.
  //
  // The mirror exists so the async sampler can leave the ids on the device. But
  // whether the engine drives sample_tokens() or sample_tokens_async() is the
  // ENGINE's choice, not the runner's: LLMEngine::step() (which vllm-bench and
  // every synchronous embedding of the library use) calls this function, while
  // only AsyncLLM's depth-2 step_with_batch_queue calls the async one. Feeding
  // the mirror from just one of them leaves the device combine reading a buffer
  // nobody wrote — every decode row splices a zero, and the output stream is
  // garbage from the first generated token. So whichever sampler ran, the mirror
  // is fed here from the values this loop just wrote.
  //
  // This is an upload rather than a device scatter on purpose: the synchronous
  // path ALREADY has the ids on the host (it downloaded them to build `out`), so
  // there is no round-trip left to remove, and a few hundred bytes of H2D is
  // cheaper and simpler than a second kernel. It also makes the mirror exactly
  // the host array on this path, which is what makes the ON/OFF token-identity
  // gate meaningful.
  if (AsyncDeviceInputs* dinp = get_or_create_async_device_inputs();
      dinp != nullptr && num_reqs > 0) {
    stage_upload(*dinp, dinp->last_sampled,
                 input_batch_.last_sampled_tokens.data(), num_reqs);
  }
  // SPEC-MTP I5d: propose drafts after a plain (no-draft, e.g. first) decode
  // step so the next step verifies them. SPEC-DFLASH2 A2-4 (#2920) named the
  // derivation `propose_after_decode` so `sample_tokens_async`'s decode arm
  // applies it instead of writing it out a second time. No-op unless a
  // speculator is configured.
  propose_after_decode(num_reqs);
  return out;
}

// ─── SPEC-DFLASH2 A2-3 (#2911): the drafts' ONE producer seam ───────────────
//
// Mirrors `self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens`
// (gpu/model_runner.py:1548) immediately followed by
// `self.draft_tokens_handler.set_draft_tokens(input_batch, ...)` (:1553-1556) @
// pin 5559679229bc961848b121ccdeaa8fa5d79bec98. Upstream writes both in the same
// place for the same reason we do: the workers read the per-req_state buffer and
// the scheduler reads the handler's copy, and a propose arm that wrote one and
// forgot the other would leave the two disagreeing about what was drafted —
// which, verify being lossless, costs acceptance and raises nothing.
//
// The scatter is BY req_state SLOT, resolved through `req_id_to_index`. Our
// persistent batch is condensed-dense so the slot equals the batch row today,
// but the lookup is the `idx_mapping` indirection upstream applies and it is what
// stays correct after an abort or finish reorders req_states against the batch.
// A request that is not in the persistent batch is skipped rather than refused:
// the propose runs over `exec_state_.req_ids`, and a request can finish between
// the sampling and this call. Its drafts still ride out in `pending_drafts_`,
// where `update_draft_token_ids` skips a finished request by name
// (scheduler.cpp, mirroring scheduler.py:2082-2085).
//
// WHY IT DOES NOT ZERO THE SLOTS IT DID NOT WRITE, which is the question a
// reader arrives at: every propose arm builds `out` over the WHOLE step batch,
// `i` in [0, exec_state_.num_reqs), giving a discarded row an empty list rather
// than omitting it. So every slot the fill can reach is refreshed here each
// step, because the fill resolves a slot through `req_id_to_index` and that only
// yields slots below `num_reqs`. A slot above the batch has no live request; it
// is zeroed when `add_request` hands it out. Zeroing the rest unconditionally
// would cost per-step work for an unreachable state AND would mask a real defect
// if a propose arm ever stopped covering the batch, so the count is maintained
// per slot instead — seeded at admission, moved by `condense`, swapped by
// `swap_states`, which is upstream's arrangement.
void GPUModelRunner::set_draft_tokens(DraftTokenIds&& drafts) {
  const int stride = input_batch_.num_speculative_steps;
  if (stride > 0) {
    const std::size_t n =
        std::min(drafts.req_ids.size(), drafts.draft_token_ids.size());
    for (std::size_t i = 0; i < n; ++i) {
      const auto slot_it = input_batch_.req_id_to_index.find(drafts.req_ids[i]);
      if (slot_it == input_batch_.req_id_to_index.end()) continue;
      const int slot = slot_it->second;
      // THE ROW WRITE IS ONE NAMED RULE (A2-3 repair round 3, #2911). Payload
      // and count used to be two adjacent statements here, and deleting the
      // PAYLOAD one left every target in this row green: verify is lossless, so
      // a zeroed draft costs acceptance and raises nothing, while deleting the
      // COUNT reds loudly at the fill's refusal. `WriteDraftRow`
      // (prepare_inputs.h) makes them one call, which
      // `tests/vllm/v1/worker/test_draft_fill.cpp` gates as a round trip against
      // `FillDraftsForRow` — the producer's write read back by the consumer.
      WriteDraftRow(input_batch_.draft_tokens,
                    input_batch_.num_valid_draft_tokens, stride, slot,
                    drafts.draft_token_ids[i], drafts.req_ids[i]);
    }
  }
  // SPEC-DFLASH2 A2-3 repair (#2911): the FILL'S FRESHNESS SOURCE. Recorded
  // OUTSIDE the `stride > 0` block above, because it answers "did a propose run
  // for this request" and not "what did it write" — the two questions used to be
  // one only because both were read off `pending_drafts_`, and the fill needs
  // the first one to stay askable after `take_draft_token_ids` has moved that
  // object out. See the member's declaration for the freshness the pre-A2-3
  // refusal had and this restores.
  proposed_drafts_.Record(drafts.req_ids);
  pending_drafts_ = std::move(drafts);
}

void GPUModelRunner::clear_draft_tokens() {
  pending_drafts_.reset();
  // This propose produced nothing, so nothing is fresh: the fill must refuse
  // every placeholder by name rather than read a row an earlier step wrote.
  proposed_drafts_.Clear();
  // Zero the COUNTS and not the rows: a stale count is what would make the next
  // fill read a row this step did not write, and upstream governs the same way —
  // `req_states.draft_tokens` is left holding whatever it held and the valid
  // count decides how much of it is real.
  std::fill(input_batch_.num_valid_draft_tokens.begin(),
            input_batch_.num_valid_draft_tokens.end(), 0);
}

// SPEC-MTP I5d: the k=1 MTP propose, run after each step's sampling
// (gpu/model_runner.py:1455-1489 speculator.propose + set_draft_tokens). Uses the
// stashed target hidden tap + this step's verify attention metadata; the draft
// reuses the target's metadata over its OWN paged KV layer (speculator.py:222-234).
// Stashes the resulting per-request drafts in pending_drafts_ for the engine
// core's out-of-band pull. UNREACHABLE unless a speculator is configured.
void GPUModelRunner::propose_drafts(const std::vector<int32_t>& num_sampled_in,
                                    const std::vector<int32_t>& num_rejected_in) {
  const int num_reqs = exec_state_.num_reqs;
  // SPEC-NGRAM (ROAD-V1-D3): the draft-FREE proposer has no draft model, no hidden
  // tap and no draft KV — it matches the sequence's own suffix. Route first.
  if (use_ngram()) {
    propose_drafts_ngram(num_sampled_in, num_rejected_in);
    return;
  }
  // SPEC-DFLASH D5: the block-diffusion drafter has no MTP draft_model_/
  // draft_attn_kv_ (it recomputes context K/V inline); route to its own propose.
  // SPEC-DSPARK W5: a DSpark draft sets BOTH predicates (dflash_weights_ points
  // at its inherited backbone, so the shared machinery runs unchanged), so it is
  // checked first and only the propose tail differs.
  if (use_dspark()) {
    propose_drafts_dspark(num_sampled_in, num_rejected_in);
    return;
  }
  if (use_dflash()) {
    propose_drafts_dflash(num_sampled_in, num_rejected_in);
    return;
  }
  if (num_reqs == 0 || draft_model_ == nullptr || draft_attn_kv_.empty()) {
    clear_draft_tokens();
    return;
  }
  VT_CHECK(exec_state_.spec_hidden.tensor.data != nullptr,
           "propose_drafts: missing target hidden tap (hidden_tap not captured)");

  // idx_mapping is identity: our persistent batch is condensed-dense (batch row
  // == req_state slot), so last_sampled/next_prefill index directly by row.
  std::vector<int32_t> idx_mapping(static_cast<size_t>(num_reqs));
  std::iota(idx_mapping.begin(), idx_mapping.end(), 0);

  // num_sampled == 0 tells prepare_prefill_inputs to splice next_prefill_tokens
  // instead of last_sampled (a still-prefilling chunk). Force it for discarded
  // rows (which sampled no committed token this step). next_prefill_tokens[i] is
  // the token at the request's next prefill position; unused for generating rows.
  std::vector<int32_t> num_sampled = num_sampled_in;
  std::vector<int32_t> num_rejected = num_rejected_in;
  std::vector<int32_t> next_prefill(static_cast<size_t>(num_reqs), 0);
  for (int i = 0; i < num_reqs; ++i) {
    if (i < static_cast<int>(exec_state_.discard.size()) &&
        exec_state_.discard[static_cast<size_t>(i)]) {
      num_sampled[static_cast<size_t>(i)] = 0;
      num_rejected[static_cast<size_t>(i)] = 0;
    }
    const int32_t seq_len = exec_state_.step.seq_lens[static_cast<size_t>(i)];
    next_prefill[static_cast<size_t>(i)] =
        input_batch_.token_id(i, seq_len);
  }

  // SPEC-MTP-K-GT-1 (#81): the depth THIS step drafts at. Resolved here, once,
  // and passed down as a value rather than read from spec_config_ inside the
  // propose loop, so a scheduler-supplied depth (upstream decides it at
  // scheduler.py:1122-1126) has one place to come from. Today it is the
  // configured k, and no depth policy exists.
  const int k = num_spec();
  VT_CHECK(k >= 1, "propose_drafts: a configured speculator must draft >= 1");
  VT_CHECK(!draft_attn_kv_.empty() && draft_attn_kv_[0].block_size > 0,
           "propose_drafts: the draft KV group has no block geometry");

  const MtpDraftProposal proposal = MtpProposeDrafts(
      *draft_model_, exec_state_.attn_meta, draft_attn_kv_[0],
      exec_state_.spec_hidden.tensor, exec_state_.step.input_token_ids,
      exec_state_.step.positions, idx_mapping,
      input_batch_.last_sampled_tokens, next_prefill, num_sampled, num_rejected,
      /*max_num_reqs=*/num_reqs, /*num_speculative_tokens=*/k,
      /*max_model_len=*/input_batch_.max_model_len,
      /*block_size=*/static_cast<int>(draft_attn_kv_[0].block_size), queue_);
  const std::vector<int32_t>& drafts = proposal.draft_tokens;
  VT_CHECK(drafts.size() ==
               static_cast<size_t>(num_reqs) * static_cast<size_t>(k),
           "propose_drafts: the MTP propose must return k drafts per request");
  // SPEC-MTP-K-GT-1 (#81): the WORK witness, recorded here because this is the
  // only place that knows both the configured k and the forwards the propose
  // actually ran. The check above is a SHAPE check and cannot stand in for it:
  // a propose that ran one forward and padded all k columns passes it, emits the
  // same greedy tokens, and grows the per-depth counters to size k. The exact
  // relation these two counters carry is
  // `spec_mtp_draft_decode_forwards_ == spec_mtp_propose_calls_ * (k - 1)`,
  // which no draft-list shape can produce. Counted on every reaching call,
  // including the discarded-row case below, because the forwards ran either way.
  ++spec_mtp_propose_calls_;
  spec_mtp_draft_decode_forwards_ += proposal.num_draft_decode_forwards;

  // SPEC-MTP-K-GT-1 (#81): the RESULT witness, and the reason it is computed
  // HERE rather than inside the propose. The equality above counts the forwards
  // a propose RAN, and says nothing about whether their results reached this
  // array. A propose that runs all k-1 forwards, throws the sampled tokens away
  // and writes its step-0 draft into every column satisfies that equality
  // exactly, and was measured green on the whole depth suite. Read at the
  // CONSUMER, on the bytes the proposer handed over, such a row is a pure
  // function of its own first column and this counter stays 0 at every k. The
  // accessor in runner.h states what it does NOT prove: per-column provenance,
  // which the DGX gate owes. A non-zero acceptance count at depth does not show
  // it either, because a padded row is accepted at column 1 whenever the
  // target's greedy continuation repeats its own token. Only an acceptance-RATE
  // comparison against a padded control separates the two.
  for (int i = 0; i < num_reqs; ++i) {
    const size_t base = static_cast<size_t>(i) * static_cast<size_t>(k);
    bool varied = false;
    for (int j = 1; j < k && !varied; ++j) {
      varied = drafts[base + static_cast<size_t>(j)] != drafts[base];
    }
    if (varied) {
      ++spec_mtp_proposals_with_varied_drafts_;
      break;
    }
  }

  // Stash each request's k drafts, in draft order, for the out-of-band pull. The
  // DraftTokenIds seam already carries variable-length drafts (the n-gram
  // proposer returns 0..k), so nothing downstream changes shape with depth.
  // A discarded (still-prefilling) row gets no draft — an empty list clears its
  // spec tokens (scheduler.update_draft_token_ids skips prefill-chunk requests).
  DraftTokenIds out;
  out.req_ids.reserve(static_cast<size_t>(num_reqs));
  out.draft_token_ids.reserve(static_cast<size_t>(num_reqs));
  for (int i = 0; i < num_reqs; ++i) {
    out.req_ids.push_back(exec_state_.req_ids[static_cast<size_t>(i)]);
    if (i < static_cast<int>(exec_state_.discard.size()) &&
        exec_state_.discard[static_cast<size_t>(i)]) {
      out.draft_token_ids.push_back({});
    } else {
      const size_t base = static_cast<size_t>(i) * static_cast<size_t>(k);
      out.draft_token_ids.emplace_back(
          drafts.begin() + static_cast<std::ptrdiff_t>(base),
          drafts.begin() + static_cast<std::ptrdiff_t>(base) + k);
    }
  }
  set_draft_tokens(std::move(out));
}

// SPEC-DFLASH D5: the DFlash branch — the shared block body with the 1 + k
// fill-in layout and the parallel per-row argmax (dflash/speculator.py:242-273).
void GPUModelRunner::propose_drafts_dflash(
    const std::vector<int32_t>& num_sampled_in,
    const std::vector<int32_t>& num_rejected_in) {
  // num_sampled_in is derivable from (T_req - num_rejected) per request, so only
  // num_rejected_in drives the append.
  (void)num_sampled_in;
  if (dflash_weights_ == nullptr) {
    clear_draft_tokens();
    return;
  }
  const int k = dflash_k_;
  const int64_t draft_vocab = dflash_weights_->draft_vocab_size;
  propose_drafts_block(
      num_rejected_in, *dflash_weights_, *dflash_config_,
      /*num_query_per_req=*/1 + k,
      [k, draft_vocab, weights = dflash_weights_](
          const std::vector<float>& block_logits, int P,
          const std::vector<int32_t>& anchors) {
        (void)anchors;  // DFlash's anchor is a bonus token, never a prediction.
        // SPEC-DFLASH2 W4 (#1314): the guard lives HERE, inside the DFlash1
        // sampler's own closure, rather than beside the walk it protects. A
        // guard adjacent to the call site it defends is deleted in the same edit
        // that deletes that call site; one function away, losing the walk still
        // costs a named throw. `propose_drafts_block` enters this callback only
        // when the DFlash2 branch produced nothing, and a DFlash2 block that
        // reaches the per-slot argmax proposes worse tokens with NO visible
        // symptom -- the verify is lossless, so only acceptance falls.
        vllm::v1::RefuseDflash1ArgmaxOnDflash2Block(*weights);
        return SampleDflashBlockDrafts(block_logits, P, k, draft_vocab);
      });
}

// SPEC-DSPARK W5: the DSpark branch — the SAME shared block body (context
// accumulation, device KV store, block forward are inherited unchanged) with the
// anchor-aware query layout and the sequential Markov sampler
// (dspark/speculator.py:100-169).
void GPUModelRunner::propose_drafts_dspark(
    const std::vector<int32_t>& num_sampled_in,
    const std::vector<int32_t>& num_rejected_in) {
  (void)num_sampled_in;
  if (dspark_weights_ == nullptr) {
    clear_draft_tokens();
    return;
  }
  vllm::v1::DsparkBlockLayout layout;
  layout.num_speculative_steps = dflash_k_;
  layout.sample_from_anchor = dspark_sample_from_anchor_;
  const vllm::Qwen3DSparkWeights* weights = dspark_weights_;
  propose_drafts_block(
      num_rejected_in, weights->backbone, *dflash_config_,
      layout.num_query_per_req(),
      [this, layout, weights](const std::vector<float>& block_logits, int P,
                              const std::vector<int32_t>& anchors) {
        VT_CHECK(static_cast<int>(anchors.size()) == P,
                 "propose_drafts_dspark: one anchor token per proposing row");
        return vllm::v1::SampleDsparkBlockDrafts(block_logits, anchors, layout,
                                                 *weights, queue_);
      });
}

// SPEC-NGRAM (ROAD-V1-D3): the draft-FREE propose. Ported from
// vllm/v1/spec_decode/ngram_proposer.py NgramProposer.propose (via NgramPropose).
// Runs the host-side suffix-ngram matcher over each generating request's OWN
// committed context and stashes the variable-length (0..k) drafts for the engine
// core's out-of-band pull. No draft model / hidden tap / draft KV. The drafts are
// verified next step by the target model through the SAME rejection/verify loop as
// MTP/DFlash. UNREACHABLE unless method=="ngram".
void GPUModelRunner::propose_drafts_ngram(
    const std::vector<int32_t>& num_sampled_in,
    const std::vector<int32_t>& /*num_rejected_in*/) {
  const int num_reqs = exec_state_.num_reqs;
  if (num_reqs == 0) {
    clear_draft_tokens();
    return;
  }
  // valid_ngram_requests filter: a row proposes only if it committed >= 1 token
  // this step AND is not a discarded (still-prefilling) chunk (ngram_proposer.py:
  // 157-172; sampled_token_ids empty -> skip). num_tokens_no_spec / token_ids_cpu
  // are already updated with this step's committed tokens by the caller.
  vllm::v1::spec_decode::NgramConfig cfg;
  cfg.min_n = spec_config_->prompt_lookup_min.value_or(0);
  cfg.max_n = spec_config_->prompt_lookup_max.value_or(0);
  cfg.max_model_len = input_batch_.max_model_len;
  const int k = num_spec();

  std::vector<bool> has_sampled(static_cast<size_t>(num_reqs), false);
  std::vector<int32_t> num_tokens_no_spec(static_cast<size_t>(num_reqs), 0);
  std::vector<const int32_t*> token_rows(static_cast<size_t>(num_reqs), nullptr);
  for (int i = 0; i < num_reqs; ++i) {
    const bool discarded = i < static_cast<int>(exec_state_.discard.size()) &&
                           exec_state_.discard[static_cast<size_t>(i)];
    has_sampled[static_cast<size_t>(i)] =
        !discarded && num_sampled_in[static_cast<size_t>(i)] > 0;
    num_tokens_no_spec[static_cast<size_t>(i)] =
        input_batch_.num_tokens_no_spec[static_cast<size_t>(i)];
    token_rows[static_cast<size_t>(i)] =
        input_batch_.token_ids_cpu.data() +
        static_cast<size_t>(i) * static_cast<size_t>(input_batch_.max_model_len);
  }

  std::vector<std::vector<int32_t>> drafts = vllm::v1::spec_decode::NgramPropose(
      cfg, k, has_sampled, num_tokens_no_spec, token_rows);

  // Stash the per-request drafts (variable length 0..k) for the out-of-band pull.
  // An empty list clears a request's spec tokens (a plain decode next step).
  DraftTokenIds out;
  out.req_ids.reserve(static_cast<size_t>(num_reqs));
  out.draft_token_ids.reserve(static_cast<size_t>(num_reqs));
  for (int i = 0; i < num_reqs; ++i) {
    out.req_ids.push_back(exec_state_.req_ids[static_cast<size_t>(i)]);
    out.draft_token_ids.push_back(std::move(drafts[static_cast<size_t>(i)]));
  }
  set_draft_tokens(std::move(out));
}

// SPEC-DFLASH D5 (DF-ENGINE-INTEGRATION): wire the separately-loaded z-lab DFlash
// draft into the runner. Resolves the D1 multi-tap capture indices
// (target_layer_ids) from the draft config once and sizes the per-request
// combined-feature context accumulation.
void GPUModelRunner::set_dflash_draft(const vllm::Qwen3DFlashWeights* weights,
                                      const vllm::HfConfig* config, int k) {
  dflash_weights_ = weights;
  dflash_config_ = config;
  dflash_k_ = k;
  dflash_tap_layer_ids_.clear();
  if (config != nullptr && config->raw.is_object() &&
      config->raw.contains("dflash_config") &&
      config->raw.at("dflash_config").contains("target_layer_ids")) {
    for (const auto& id : config->raw.at("dflash_config").at("target_layer_ids")) {
      dflash_tap_layer_ids_.push_back(id.get<int32_t>());
    }
  }
  dflash_ctx_.clear();

  // #1919: resolve the draft context store's capacity from THIS engine's
  // advertised context, and say so. Before this, the capacity was a
  // compile-time 4096 unrelated to `--max-model-len`, nothing at startup
  // mentioned it, and the first thing that did was a throw out of the middle of
  // an EngineCore step on the first prompt above 4096 tokens.
  //
  // `k + 1` is the (1+k) query block DFlash always emits. DSpark's
  // anchor-as-first-prediction layout emits k rows, so `k + 1` is an upper
  // bound for both, which is what a capacity headroom wants.
  if (weights == nullptr || config == nullptr) {
    // Unwiring leaves no sizing behind: a stale one would outlive the draft it
    // was resolved for, and `MakeDeviceKVStore` refuses a zero capacity by name.
    dflash_ctx_sizing_ = Qwen3DFlashModel::DflashCtxStoreSizing{};
    return;
  }
  dflash_ctx_sizing_ = Qwen3DFlashModel::ResolveCtxStoreSizing(
      *config, input_batch_.max_model_len, static_cast<int64_t>(k) + 1,
      input_batch_.max_num_reqs);
  const auto& z = dflash_ctx_sizing_;
  const double mib = static_cast<double>(z.bytes_per_request) / (1024.0 * 1024.0);
  // #1919 review: state the AGGREGATE, not only the per-request cost. One store
  // is built per batch row, so the device holds `bytes_per_request *
  // max_num_reqs` and `gpu_memory_utilization` accounts none of it. A reader
  // given only the per-request number has to know that multiplication exists to
  // find the term, and a concurrency ladder that does not know it is there
  // measures it as noise.
  const double total_mib = static_cast<double>(z.bytes_total) / (1024.0 * 1024.0);
  std::cerr << "vllm.cpp: draft speculative context is limited to " << z.slots
            << " tokens (" << z.slots / z.page_size << " pages x " << z.page_size
            << ", " << mib << " MiB per concurrent request across "
            << config->num_hidden_layers << " draft layers, so " << total_mib
            << " MiB of device memory at max_num_seqs " << z.max_num_reqs
            << " — NOT counted by gpu_memory_utilization). max_model_len is "
            << input_batch_.max_model_len << " and the (1+k) draft block needs "
            << (k + 1) << " more";
  if (z.capped) {
    std::cerr << ", so this is CAPPED below the "
              << z.want_slots << " tokens that context asks for"
              << (z.overridden
                      ? " (VT_DFLASH_CTX_MAX_TOKENS)"
                      : " (the default aggregate budget of " +
                            std::to_string(z.budget_bytes / (1024 * 1024)) +
                            " MiB across max_num_seqs; raise or lower the per-request "
                            "share with VT_DFLASH_CTX_MAX_TOKENS)")
              << ". A request whose context passes " << z.slots
              << " tokens keeps running, WITHOUT speculation, on the target alone";
  } else {
    std::cerr << ", so the whole advertised context is speculated";
  }
  std::cerr << ".\n";
}

// SPEC-DSPARK W5: wire a DSpark draft. The inherited backbone goes through
// set_dflash_draft, so the aux multi-tap capture, the per-request device KV
// store and the context-aware block forward are the SAME code the landed DFlash
// lane runs; only the propose tail (use_dspark()) differs.
void GPUModelRunner::set_dspark_draft(const vllm::Qwen3DSparkWeights* weights,
                                      const vllm::HfConfig* config, int k,
                                      bool sample_from_anchor) {
  dspark_weights_ = weights;
  dspark_sample_from_anchor_ = sample_from_anchor;
  set_dflash_draft(weights == nullptr ? nullptr : &weights->backbone, config, k);
}

// SPEC-DFLASH D5: the DFlash branch of propose_drafts. Ported from
// dflash/speculator.py::propose (:300-413). Where vLLM writes each step's
// combined target features into the draft's incremental KV cache
// (precompute_and_store_context_kv) and excludes rejected positions via
// valid_ctx_end = ctx_end - num_rejected (:518), the inline D3/D4 path instead
// ACCUMULATES the per-request combined-feature context on the host and honors the
// rollback by appending only the num_sampled accepted-prefix features each step
// (the rejected drafts' features are simply never appended). Then it runs the
// non-autoregressive (1+k) block forward over that context (DflashProposeBlock)
// and stashes the k drafts/request. Reachable only when use_dflash().
void GPUModelRunner::propose_drafts_block(
    const std::vector<int32_t>& num_rejected_in,
    const vllm::Qwen3DFlashWeights& backbone, const vllm::HfConfig& config,
    int num_query_per_req,
    const std::function<std::vector<std::vector<int32_t>>(
        const std::vector<float>&, int, const std::vector<int32_t>&)>& sample) {
  const int num_reqs = exec_state_.num_reqs;
  if (num_reqs == 0) {
    clear_draft_tokens();
    return;
  }
  VT_CHECK(exec_state_.spec_aux.tensor.data != nullptr,
           "propose_drafts_block: missing target aux multi-tap (aux_tap not "
           "captured on the verify forward)");
  const int64_t H = config.hidden_size;
  const int taps = static_cast<int>(dflash_tap_layer_ids_.size());
  const int32_t mask_id = backbone.mask_token_id;
  // The number of NOISE/mask rows after the anchor row. DFlash always emits
  // 1 + k rows (anchor + k masks). DSpark's anchor-as-first-prediction layout
  // emits k rows total, so it carries k - 1 masks.
  const int num_mask_rows = num_query_per_req - 1;
  const StepInputs& step = exec_state_.step;

  // #2008 — THE DRAFT CONTEXT BELONGS TO THE REQUEST, NOT TO THE BATCH ROW.
  //
  // Release the entries of requests that have left the batch, then resolve one
  // pointer per row for the rest of this function. `InputBatch` is the authority
  // on residency, not `exec_state_.req_ids`, which lists only the rows SCHEDULED
  // this step; a resident request that happens not to be scheduled must keep its
  // context. The scan is bounded by `max_num_seqs` and runs once per step, off
  // the per-token path.
  //
  // The pruning is not housekeeping: each entry owns a device allocation sized
  // from `dflash_ctx_sizing_.slots`, so a leaked entry is leaked device memory.
  if (dflash_ctx_.size() > static_cast<size_t>(input_batch_.num_reqs())) {
    for (auto it = dflash_ctx_.begin(); it != dflash_ctx_.end();) {
      it = (input_batch_.req_id_to_index.count(it->first) == 0)
               ? dflash_ctx_.erase(it)
               : std::next(it);
    }
  }
  // Resolving here replaces the reused-slot test this loop used to open with.
  // "Has this row's occupant changed" was only ever a proxy for "is this state
  // this request's", and the proxy is what #2008 broke: after a condense move
  // the answer was yes for a request whose context was perfectly valid, and the
  // reset threw it away. Keyed by id the question cannot be asked wrongly — a
  // first sight of a request id constructs its entry, and THAT is the reset.
  std::vector<DflashReqCtx*> row_ctx(static_cast<size_t>(num_reqs), nullptr);
  for (int i = 0; i < num_reqs; ++i) {
    const std::string& req_id = exec_state_.req_ids[static_cast<size_t>(i)];
    DflashReqCtx& c = dflash_ctx_[req_id];
    // #2008's "first sight of a request id", named so #2042's classification
    // below can be gated on it explicitly and so that gate can be mutated on its
    // own. A request the batch merely MOVED is not first-seen.
    const bool first_sight = c.store == nullptr;
    if (first_sight) {
      c.store = Qwen3DFlashModel::MakeDeviceKVStore(config, queue_,
                                                    dflash_ctx_sizing_.slots);
    }
    if (first_sight) {
      // #2042 — PREFIX CACHING, CLASSIFIED AT THE ONE POINT THAT CAN CLASSIFY
      // IT. This branch is #2008's "first sight of a request id", and first
      // sight is exactly the question this needs: has this runner ever held
      // context for this request. Under the row-indexed form it could not be
      // asked — a request the batch had merely MOVED presented identically to a
      // request that had never been seen — so the answer below would have been
      // given for #2008's defect too, trading a loud refusal for a silent
      // acceptance loss. Keyed by id, the two states are distinct and this
      // branch sees only the second.
      //
      // A request admitted on a prefix-cache (or KV-connector) hit starts at the
      // CACHED length, not at 0: the scheduler sets `num_computed_tokens` to the
      // hit before the request runs a single token (`sched/scheduler.cpp`, the
      // waiting-admission `get_computed_blocks` arm) and the worker turns that
      // straight into absolute positions, `positions[t] =
      // num_computed_tokens_cpu[r] + query_pos[t]` (`prepare_inputs.cpp`). The
      // TARGET is served from cache, so it never runs those tokens, so there is
      // no aux hidden state for them, so there is no combined feature to project
      // and append. The store holds zero context rows while the target has
      // committed N — and no LATER step can supply the missing ones, because
      // those features were never computed. The invariant below is therefore a
      // DETECTOR here, refusing a state that really is inconsistent, and the
      // repair belongs in this accounting rather than in the check.
      //
      // UPSTREAM NEVER REACHES THIS STATE, and the reason is our divergence and
      // not the check: its DFlash draft keeps no private store at all. The
      // context K/V goes into the engine's own paged KV cache through
      // `attn.impl.do_kv_cache_update(...)`
      // (`vllm/model_executor/models/qwen3_dflash.py:601-619` at pin
      // `5559679229`) on a slot mapping built from the TARGET's block table
      // (`vllm/v1/spec_decode/dflash.py:145-153`,
      // `block_table_ptr=cad.block_table_tensor`), attending over the full
      // `cad.seq_lens`. A prefix hit therefore hands upstream's draft its context
      // for free: the block hash keys on token ids, and the draft layers' K/V for
      // that prefix is already resident. Moving our store into the paged
      // allocator is the real repair and is owed under #1919.
      //
      // So this is upstream's OTHER answer, the one for a proposer that cannot
      // serve a request: an EMPTY draft and the target running alone
      // (`vllm/v1/spec_decode/ngram_proposer.py:150-159`,
      // `vllm/v1/spec_decode/suffix_decoding.py:55-62` — both `continue` past the
      // request, neither raises). It is #1919's fallback, reached from a second
      // place, so the request is served and the engine lives.
      //
      // WHAT IT COSTS IS DECODE THROUGHPUT, AND THAT IS THE POINT OF THE NOTICE.
      // The request keeps prefix caching's TTFT win — the target still skips the
      // cached prefill — and gives up speculation for the rest of its life. On a
      // workload with a shared system prompt that is most requests, so this is
      // not a corner: it is prefix caching and DFlash2 being mutually exclusive
      // in effect until the store is paged. Said once per request, on stderr,
      // rather than left to be discovered in a throughput number.
      //
      // The position is read from `step.positions` rather than from
      // `num_computed_tokens_cpu` because the spec-decode arm of `execute_model`
      // CORRECTS that array in place before this runs; `step.positions` is what
      // the invariant below compares against and is therefore the same fact.
      const int seg0 = step.query_start_loc[static_cast<size_t>(i)];
      const int seg1 = step.query_start_loc[static_cast<size_t>(i + 1)];
      int64_t first_pos = 0;
      for (int t = seg0; t < seg1; ++t) {
        const int64_t pos = step.positions[static_cast<size_t>(t)];
        if (t == seg0 || pos < first_pos) first_pos = pos;
      }
      if (seg1 > seg0 && first_pos > 0) {
        c.disabled = true;
        std::cerr << "vllm.cpp: request " << req_id << " was admitted with "
                  << first_pos
                  << " tokens already computed (a prefix-cache or KV-connector "
                     "hit). The DFlash draft keeps its own context store and the "
                     "target does not recompute a cached prefix, so the draft can "
                     "never obtain those tokens' features. It keeps running "
                     "WITHOUT speculation, on the target alone — prefix caching "
                     "buys this request TTFT and costs it draft acceptance. "
                     "Disable prefix caching to speculate over the whole prompt "
                     "(#2042).\n";
      }
    }
    row_ctx[static_cast<size_t>(i)] = &c;
  }

  // SPEC-DFLASH2 W8 (#1838): the propose pre-phase timer. Before W8 everything
  // up to `t_fwd0` was untimed, which is how ~31 ms/step of aux round trips and
  // hard syncs stayed unattributed in `VT_SPEC_TRACE`.
  const auto t_pre0 = std::chrono::steady_clock::now();

  // 1+2. fc(cat(aux)) -> [T_total, H] combined features (combine_hidden_states),
  // straight off the DEVICE aux tap (SPEC-DFLASH2 W8, #1838). The pre-W8 chain —
  // bf16 D2H + full queue drain + host scalar cast + f32 H2D + cast + GEMM +
  // f32 D2H — was a sequence of exact bf16<->f32 round trips around this same
  // GEMM, so this is bit-identical and moves nothing across the host boundary.
  // Upstream: `combine_hidden_states(torch.cat(aux_hidden_states, dim=-1))`
  // consuming the target's device tensors (dflash/speculator.py::propose).
  const int64_t T_total = exec_state_.num_actual_tokens;
  const vt::Tensor& aux = exec_state_.spec_aux.tensor;
  VT_CHECK(aux.shape[0] == T_total && aux.shape[1] == H * taps,
           "propose_drafts_block: aux tap shape mismatch");
  const Qwen3DFlashModel::DflashCombinedDevice combined =
      Qwen3DFlashModel::CombineAuxFeaturesDevice(aux, backbone, config, queue_);

  // 3. Per request: reset a reused slot, PROJECT+APPEND only the newly-accepted
  //    rows to the persistent per-request KV store (D9 — no full recompute), and
  //    (for a generating row) build the next (1+k) mask block.
  std::vector<int32_t> blk_ids;                // [P*num_query_per_req]
  std::vector<int32_t> blk_pos;                // [P*num_query_per_req]
  std::vector<int32_t> blk_cu = {0};           // [P+1]
  std::vector<int> propose_rows;               // batch rows in the propose batch
  std::vector<int32_t> anchors;                // [P] each proposing row's anchor token
  // #1919: rows whose context has outgrown the draft store this step or an
  // earlier one, and which are DECODING (a still-prefilling row carries no
  // scheduled draft slots at all). Section 4 turns these into this batch's
  // fallback drafts.
  std::vector<bool> fell_back(static_cast<size_t>(num_reqs), false);

  for (int i = 0; i < num_reqs; ++i) {
    // #2008: no reused-slot reset here any more. This request's state was
    // resolved by id above, so it is this request's whether or not the batch
    // moved it, and a request seen for the first time started empty.
    DflashReqCtx& ctx = *row_ctx[static_cast<size_t>(i)];
    // #1919: a request whose context has outgrown the store stops speculating
    // for the rest of its life, and this test comes BEFORE the two invariants
    // below because a disabled row stops maintaining both: it neither appends
    // nor advances its `ctx_len`, so that counter and the target's committed
    // positions legitimately diverge from here on.
    //
    // What such a row PROPOSES is decided in section 4, which is also where the
    // two scheduling modes part company; `fell_back` is how this loop tells it
    // which rows are in that state and are decoding rather than still
    // prefilling.
    if (ctx.disabled) {
      fell_back[static_cast<size_t>(i)] =
          !(i < static_cast<int>(exec_state_.discard.size()) &&
            exec_state_.discard[static_cast<size_t>(i)]);
      continue;
    }
    const int seg0 = step.query_start_loc[static_cast<size_t>(i)];
    const int seg1 = step.query_start_loc[static_cast<size_t>(i + 1)];
    const int T_req = seg1 - seg0;
    if (T_req <= 0) continue;
    // Rows in ascending absolute position (== committed sequence order).
    std::vector<int> rows(static_cast<size_t>(T_req));
    std::iota(rows.begin(), rows.end(), seg0);
    std::sort(rows.begin(), rows.end(), [&](int a, int c) {
      return step.positions[static_cast<size_t>(a)] <
             step.positions[static_cast<size_t>(c)];
    });
    const int nrej = (i < static_cast<int>(num_rejected_in.size()))
                         ? num_rejected_in[static_cast<size_t>(i)]
                         : 0;
    int append = T_req - nrej;
    if (append < 0) append = 0;
    // Invariant: this step's first committed token sits at absolute position L
    // (== current context length). A violation means the accumulation lost sync
    // (the I5e async-input-combine bug class) — assert rather than corrupt.
    const int64_t L = ctx.ctx_len;
    VT_CHECK(step.positions[static_cast<size_t>(rows[0])] == L,
             "propose_drafts_block: context position discontinuity (accumulation "
             "out of sync with the target's committed positions)");
    // SPEC-DFLASH2 W8 (#1838): the runner's counter and the DEVICE store must
    // agree, or the append is dead. The W8 mutation run proved the check above
    // cannot see that state: with the append call deleted, `ctx_len`
    // kept advancing, the store stayed empty, every propose ran CONTEXT-FREE,
    // and every gate stayed green — well-formed drafts, lossless verify, only
    // ACCEPTANCE falls, the exact invisible-defect class this row exists to
    // remove. This host integer comparison is what makes that state loud.
    VT_CHECK(L == Qwen3DFlashModel::DeviceKVNumCtx(*ctx.store),
             "propose_drafts_block: the runner's context length and the device "
             "store's num_ctx disagree — the context-KV append is dead or "
             "double-run (SPEC-DFLASH2 W8, #1838)");
    // Gather this step's `append` accepted-prefix combined features (in ascending
    // position order) + their absolute positions [L, L+append), then project+append
    // to the persistent KV store. This projects ONLY the new rows (D9) — bit-identical
    // to the D5/D7 full recompute of the whole context by per-row projection independence.
    // #1919 — THE FALLBACK DECISION, and the only place production asks it.
    //
    // The store is a fixed pool; once this request's context plus its (1+k)
    // query block no longer fits, the speculator cannot serve it. That is not a
    // fatal condition and it is not grounds to refuse the request: the TARGET
    // can serve it, and a prompt inside `max_model_len` is a prompt vLLM and
    // SGLang both answer. Upstream's answer where a speculator cannot serve a
    // request is an EMPTY draft and the target running alone
    // (`vllm/v1/spec_decode/ngram_proposer.py:156-159`,
    // `vllm/v1/spec_decode/suffix_decoding.py:59-62`); it never raises. Before
    // this branch existed, `AppendContextKVDeviceRows` threw from inside the
    // EngineCore step instead, and #1919 measured what that costs: the engine
    // died and every later request on that server, of any size, came back
    // `request submitted to a stopped AsyncLLM` while the HTTP process stayed
    // alive and answering /health.
    //
    // The `+ num_query_per_req` is the block's own headroom, not slack. When the
    // sizing is UNCAPPED this reads exactly as upstream's condition, because the
    // capacity is then `max_model_len + num_query_per_req` and the test reduces
    // to `L + append > max_model_len`. When it IS capped it costs one block's
    // worth of context and keeps the W11 paged fast route, which
    // `ClassifyDflashBlockAttn` would otherwise drop below its capacity
    // conjunct.
    const int64_t capacity = Qwen3DFlashModel::DeviceKVCapacity(*ctx.store);
    if (L + append + num_query_per_req > capacity) {
      ctx.disabled = true;
      std::cerr << "vllm.cpp: request " << exec_state_.req_ids[static_cast<size_t>(i)]
                << " has outgrown the draft speculative context (" << (L + append)
                << " context tokens plus a " << num_query_per_req
                << "-row draft block, store capacity " << capacity
                << "). It keeps running WITHOUT speculation, on the target alone; "
                   "raise VT_DFLASH_CTX_MAX_TOKENS to speculate further.\n";
      fell_back[static_cast<size_t>(i)] =
          !(i < static_cast<int>(exec_state_.discard.size()) &&
            exec_state_.discard[static_cast<size_t>(i)]);
      continue;
    }
    // SPEC-DFLASH2 W8 (#1838): the gather is a device IndexSelect over the row
    // indices this host loop already determined — the accepted-prefix decision
    // stays host integer bookkeeping; only the floats stop commuting.
    std::vector<int32_t> new_rows(static_cast<size_t>(append));
    std::vector<int32_t> new_pos(static_cast<size_t>(append));
    for (int j = 0; j < append; ++j) {
      new_rows[static_cast<size_t>(j)] = static_cast<int32_t>(rows[static_cast<size_t>(j)]);
      new_pos[static_cast<size_t>(j)] = static_cast<int32_t>(L + j);
    }
    Qwen3DFlashModel::AppendContextKVDeviceRows(*ctx.store, combined.tensor, new_rows,
                                                new_pos, backbone, config, queue_);
    ctx.ctx_len = static_cast<int32_t>(L + append);

    // A discarded (still-prefilling chunk) row commits its chunk's features but
    // proposes no draft — it has no valid last_sampled anchor yet.
    const bool discarded = i < static_cast<int>(exec_state_.discard.size()) &&
                           exec_state_.discard[static_cast<size_t>(i)];
    if (discarded) continue;

    // Block: anchor = last_sampled (the bonus/last committed token, re-embedded),
    // then k mask tokens; positions L' .. L'+k (L' = the new context length).
    const int64_t Lp = ctx.ctx_len;
    const int32_t anchor = input_batch_.last_sampled_tokens[static_cast<size_t>(i)];
    blk_ids.push_back(anchor);
    blk_pos.push_back(static_cast<int32_t>(Lp));
    for (int j = 1; j <= num_mask_rows; ++j) {
      blk_ids.push_back(mask_id);
      blk_pos.push_back(static_cast<int32_t>(Lp + j));
    }
    blk_cu.push_back(static_cast<int32_t>(blk_ids.size()));
    propose_rows.push_back(i);
    anchors.push_back(anchor);
  }

  // 4. Non-autoregressive (1+k) block propose over the PERSISTENT context KV store.
  //    Concatenate the propose rows' per-layer stores (ctx_cu order) into one combined
  //    PrecomputedContextKV and run the context-aware block forward with NO re-projection.
  DraftTokenIds out;
  out.req_ids.reserve(static_cast<size_t>(num_reqs));
  out.draft_token_ids.assign(static_cast<size_t>(num_reqs), {});
  for (int i = 0; i < num_reqs; ++i)
    out.req_ids.push_back(exec_state_.req_ids[static_cast<size_t>(i)]);

  // #1919 — WHAT A FALLEN-BACK REQUEST PROPOSES, AND WHY THE TWO SCHEDULING
  // MODES DIFFER.
  //
  // Under SYNCHRONOUS scheduling the answer is nothing at all: the empty list
  // left by the assign above IS upstream's own skip
  // (`draft_token_ids.append([]); continue`,
  // `vllm/v1/spec_decode/suffix_decoding.py:59-62` at pin `5559679229`). The
  // scheduler installs it through `update_draft_token_ids`, schedules no spec
  // slots for that request next step, and the request decodes one token per
  // step on the target alone. That is the cheapest correct outcome and it is
  // exactly what the oracle does.
  //
  // Under ASYNC scheduling an empty list is not available, and the reason is
  // structural rather than a policy choice. The AsyncScheduler assigns
  // `num_spec_tokens_to_schedule` placeholder drafts to EVERY non-prefill-chunk
  // request at schedule time — one step BEFORE the propose that would fill them
  // (`async_scheduler.cpp`, mirroring `async_scheduler.py:_update_after_schedule`
  // :36-42) — and under async `update_draft_token_ids` is deliberately never
  // called (`core.cpp:120-123`), so the request state keeps those placeholders
  // and, IN THIS ENGINE AS IT STANDS, the count cannot shrink in response to a
  // propose. The scheduler has already budgeted `1 + k` verify positions for the
  // request; delivering fewer draft tokens than it budgeted is the mismatch
  // `execute_model`'s async draft fill refuses by name.
  //
  // That last clause is a property of this tree and not of the design. Upstream's
  // worker DOES shrink the count under async, by mutating the `SchedulerOutput`
  // directly rather than routing an empty draft back through the scheduler — see
  // the next paragraph for which mechanism, and #1943 for why it is not here.
  //
  // So the async arm keeps the draft's SHAPE and neutralises its CONTENT, with
  // the draft's own mask token: a real vocabulary entry the draft block already
  // feeds itself, and one this runner's fill can put in front of an embedding
  // gather, which upstream's `-1` placeholder is not.
  //
  // UPSTREAM HAS NO ANALOGUE ON THIS PATH, and the honest statement of what it
  // does instead is worth more than a near-match. Its DFlash draft keeps no
  // private context store — the context K/V goes into the engine's own paged KV
  // cache (`vllm/model_executor/models/qwen3_dflash.py:604-620`) — so its
  // proposer cannot fail for capacity and it carries no fallback state at all.
  // The nearest upstream mechanism for a proposer that delivered FEWER drafts
  // than the scheduler optimistically budgeted is
  // `update_scheduler_for_invalid_drafts`
  // (`vllm/v1/spec_decode/ngram_proposer_gpu.py:475-515` at pin 5559679229,
  // called from `vllm/v1/worker/gpu_model_runner.py:1333-1344` and gated on
  // `use_ngram_gpu()`), and it does NOT neutralise the tokens: it TRIMS the
  // schedule in the worker, decrementing `num_scheduled_tokens` and
  // `total_num_scheduled_tokens` and popping the request out of
  // `scheduled_spec_decode_tokens` at `valid_k == 0`, keeping
  // `original_num_spec_per_req` for the rejection correction.
  //
  // THIS CHANGE DELIBERATELY DOES NOT PORT THAT TRIM (#1943, owned by
  // SPEC-DFLASH2, listed under `## Owed` in the wave spec). Porting it moves the
  // scheduler/worker contract, and it needs a red-before gate on the SCHEDULE
  // rather than on the tokens. The cost of not porting it is stated rather than
  // hidden: a fallen-back request keeps being scheduled (1+k) verify positions
  // on every later step at ~zero acceptance, about 9x the target compute per
  // emitted token at k=8, for the rest of that request's life.
  //
  // NEITHER ARM CAN EMIT A WRONG TOKEN. The verify is lossless: a draft token
  // is accepted only where it equals what the target itself would have emitted,
  // so a junk draft costs acceptance and nothing else. That is also why the
  // waste above is invisible to a token gate.
  if (use_async_scheduling_) {
    const int32_t neutral = mask_id >= 0 ? mask_id : 0;
    const int k_drafts = num_spec();
    for (int i = 0; i < num_reqs; ++i) {
      if (!fell_back[static_cast<size_t>(i)]) continue;
      out.draft_token_ids[static_cast<size_t>(i)].assign(
          static_cast<size_t>(k_drafts), neutral);
    }
  }

  // VT_SPEC_TRACE=1: how many rows actually proposed this step, and what the
  // first row's drafts were. The aggregate acceptance trace on the VERIFY side
  // cannot distinguish "proposed nothing" from "proposed and everything was
  // rejected", which is exactly the question the first DSpark e2e raised.
  //
  // SPEC-DFLASH2 W9 (#1849): the value is a LEVEL now. Any nonzero value keeps
  // exactly what "1" always meant; ">= 2" additionally brackets the DFlash2
  // draft phase's four seams with queue synchronizes and prints the
  // per-segment `[spec-phase-dev]` split below — the level-1 `sample=` figure
  // aggregates pre-phase device work, graph replay, selector and walk into one
  // number, which is what left #1849's flat ~23 ms unattributable. The syncs
  // run ONLY at level >= 2, so every existing level-1 recipe keeps its
  // overlap-preserving shape; synchronization changes no value anywhere.
  static const int propose_trace_level = [] {
    const char* v = std::getenv("VT_SPEC_TRACE");
    if (v == nullptr || v[0] == '\0' || v[0] == '0') return 0;
    const int lvl = std::atoi(v);
    return lvl > 0 ? lvl : 1;  // a non-numeric nonzero value stays level 1
  }();
  const bool propose_trace = propose_trace_level >= 1;
  if (!propose_rows.empty()) {
    const int P = static_cast<int>(propose_rows.size());
    // D11 A-wire: run the block forward straight off the per-request DEVICE stores
    // (ForwardBlockLogitsWithDeviceKV) — no host concat, no D<->H round-trip. The
    // device-side IndexCopy concat reproduces the exact ascending-position layout
    // the D9 host path built, so tokens+acceptance are bit-identical. ctx_cu =
    // running sums of each store's on-device context length (DeviceKVNumCtx).
    std::vector<vllm::DflashDeviceKVStore*> stores;
    stores.reserve(static_cast<size_t>(P));
    std::vector<int32_t> ctx_cu = {0};
    int64_t total_ctx = 0;
    for (int r = 0; r < P; ++r) {
      vllm::DflashDeviceKVStore* st =
          row_ctx[static_cast<size_t>(propose_rows[static_cast<size_t>(r)])]->store.get();
      stores.push_back(st);
      total_ctx += Qwen3DFlashModel::DeviceKVNumCtx(*st);
      ctx_cu.push_back(static_cast<int32_t>(total_ctx));
    }
    // SPEC-DFLASH2 W3 (#1314), reshaped by W8 (#1837): a DFlash2 draft ALSO
    // captures the post-final-norm hidden off this forward -- the candidate
    // selector's `hidden_projection` input. Upstream's `_generate_draft` takes
    // both from one forward, and it must: the selector projects the SAME hidden
    // states these logits came from. Since W8 both come back as DEVICE handles
    // (`DflashBlockDeviceOut`), which is what re-arms the single-request PAGED
    // fast path and its CUDA-graph capture for a DFlash2 draft — the W4-era
    // `final_out` host contract disqualified that branch and cost every DFlash2
    // step the graph lane plus a full-logits download. A DFlash1 draft passes
    // nullptr and this call is byte-for-byte what it was.
    const bool dflash2 = backbone.IsDflash2();
    // SPEC-DFLASH2 W9 (#1849): the level-2 device-segment split. Each
    // Synchronize drains the queue at a phase seam so the wall clock between
    // seams IS that phase's device work; on the CPU backend the sync is a
    // no-op and the segments are the host time of each call, which is the
    // same number the level-1 line already reports in aggregate.
    const bool dev_trace = dflash2 && propose_trace_level >= 2;
    vt::Backend& trace_b = vt::GetBackend(queue_.device.type);
    if (dev_trace) trace_b.Synchronize(queue_);
    const auto t_fwd0 = std::chrono::steady_clock::now();
    Qwen3DFlashModel::DflashBlockDeviceOut dev_out;
    const std::vector<float> block_logits =
        Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(
            stores, ctx_cu, blk_ids, blk_pos, blk_cu, backbone, config, queue_,
            nullptr, nullptr, dflash2 ? &dev_out : nullptr);
    if (dev_trace) trace_b.Synchronize(queue_);
    const auto t_fwd1 = std::chrono::steady_clock::now();
    // SPEC-DFLASH2 W4 (#1314): the PRODUCTION draft of a DFlash2 block, end to
    // end. The block forward above ran the draft's grouped dynamic convolution
    // (W2); `Dflash2SelectCandidates` runs the target head's top-K, the codebook
    // lattice and the edge scores (W3); `Dflash2WalkPath` walks that lattice
    // from the verified anchor and IS what produces this draft's tokens (W4).
    // Both are the SAME functions `DflashProposeBlock` calls, so this path and
    // the one a gate can drive are one implementation rather than two.
    //
    // `sample` -- the DFlash1 per-slot argmax -- must NOT run for a DFlash2
    // block. It would succeed and propose worse tokens with no visible symptom,
    // because the verify is lossless and the emitted tokens stay the target's;
    // only acceptance falls. The fallback below is therefore entered on
    // EMPTINESS, and guarded, so that deleting this branch is loud.
    std::vector<std::vector<int32_t>> drafts;
    auto t_sel = t_fwd1;
    if (dflash2) {
      // SPEC-DFLASH2 W8 (#1837): the selector and the walk run DEVICE-TO-DEVICE
      // — sample-row gather, top-K, value scalars, projection, edge lattice,
      // walk — and download ONLY the [P, k] drafted token ids, which is all
      // upstream's `_generate_draft` brings back either. The host-vector
      // entries these replaced are marshaling shells over the same cores, so
      // the drafts are bit-identical to theirs.
      const vllm::v1::Dflash2ProposeStateDevice selected =
          vllm::v1::Dflash2SelectCandidatesDevice(dev_out.logits, dev_out.hidden, anchors,
                                                  P, num_query_per_req - 1, backbone,
                                                  config, queue_);
      if (dev_trace) trace_b.Synchronize(queue_);
      t_sel = std::chrono::steady_clock::now();
      // The walk's own [P, k] download synchronizes, so `t_smp1 - t_sel` below
      // is the walk segment with no extra seam needed.
      drafts = vllm::v1::Dflash2WalkPathDevice(selected, queue_).draft_token_ids;
    }
    if (drafts.empty()) drafts = sample(block_logits, P, anchors);
    const auto t_smp1 = std::chrono::steady_clock::now();
    if (dev_trace) {
      // SPEC-DFLASH2 W9 (#1849): the draft phase, split at its seams. `pre` is
      // the pre-phase device work (aux combine + accepted-prefix append) plus
      // the block assembly's host bookkeeping; `fwd` the block forward (embed
      // refresh + graph replay, or the eager paged body); `select` the device
      // selector chain; `walk` the path walk plus the one token download.
      std::fprintf(stderr,
                   "[spec-phase-dev] pre=%.2fms fwd=%.2fms select=%.2fms walk=%.2fms\n",
                   std::chrono::duration<double, std::milli>(t_fwd0 - t_pre0).count(),
                   std::chrono::duration<double, std::milli>(t_fwd1 - t_fwd0).count(),
                   std::chrono::duration<double, std::milli>(t_sel - t_fwd1).count(),
                   std::chrono::duration<double, std::milli>(t_smp1 - t_sel).count());
    }
    if (propose_trace) {
      // Splits the draft step into the pre-phase (aux combine + accepted-prefix
      // append; W8 made it attributable), the parallel backbone forward and the
      // sampler. For DSpark the sampler is a k-iteration host loop, each
      // iteration a Markov GEMV plus a device->host download plus a host argmax
      // over the draft vocab; upstream captures the WHOLE draft step in one CUDA
      // graph instead (dspark/speculator.py:22-24). `logits` is the block
      // forward's logits numel — device-resident for a DFlash2 draft, a host
      // vector for DFlash1/DSpark.
      const size_t logits_numel =
          dflash2 ? static_cast<size_t>(dev_out.logits.Numel()) : block_logits.size();
      std::fprintf(stderr,
                   "[spec-phase] pre=%.2fms backbone=%.2fms sample=%.2fms logits=%zu\n",
                   std::chrono::duration<double, std::milli>(t_fwd0 - t_pre0).count(),
                   std::chrono::duration<double, std::milli>(t_fwd1 - t_fwd0).count(),
                   std::chrono::duration<double, std::milli>(t_smp1 - t_fwd1).count(),
                   logits_numel);
    }
    for (int r = 0; r < P; ++r) {
      const int row = propose_rows[static_cast<size_t>(r)];
      out.draft_token_ids[static_cast<size_t>(row)] = drafts[static_cast<size_t>(r)];
    }
    if (propose_trace) {
      std::string first;
      for (int32_t id : drafts[0]) {
        first += std::to_string(id);
        first += ' ';
      }
      std::fprintf(stderr, "[spec-propose] rows=%d nqpr=%d drafts/row=%zu first=[%s]\n",
                   P, num_query_per_req, drafts[0].size(), first.c_str());
    }
  }
  if (propose_trace && propose_rows.empty()) {
    std::fprintf(stderr, "[spec-propose] NO proposing rows this step (num_reqs=%d)\n",
                 num_reqs);
  }
  set_draft_tokens(std::move(out));
}

GPUModelRunner::~GPUModelRunner() {
  // ─── SPEC-DFLASH2 A2-4 (#3004): THE DRAIN COMES FIRST ──────────────────────
  //
  // A2-3 put a drain in this destructor and scoped it to the mirror's buffers,
  // and its own comment recorded that the copy queue and the two events were
  // destroyed ABOVE it and therefore protected by nothing. That ordering smell
  // was booked to this wave, and this is it: the drain is hoisted to the top, so
  // everything the body releases below — the copy queue, the fork/ready pair,
  // the mirror's device buffers — is released after the main queue is idle. The
  // page-locked staging blocks are freed later still, by `~PinnedGrowStaging`
  // running after this body, which is the order their retain rule needs.
  //
  // WHY IT IS NOT CONDITIONAL ON THE MIRROR. The queue and the events are used
  // by the async COPY paths, which run whether or not the mirror is engaged
  // (`DownloadCommittedIds` and the verify D2H both record `verify_fork_event_`
  // on this queue), so gating the drain on `async_device_inputs_ != nullptr`
  // would leave exactly the two objects the smell named unprotected. It costs
  // one drain of an idle queue per runner destruction, at teardown, off every
  // serving path — the same trade `RejectionSamplerDeviceOutput::Release` took.
  //
  // IT IS EXECUTED ON NEITHER TIER in the sense that matters: on the CPU build
  // `vt::cpu::CpuBackend::Synchronize` is a no-op, so this line runs and proves
  // nothing about CUDA. Stated here rather than implied, because the version
  // this replaces read as coverage it did not have.
  vt::GetBackend(queue_.device.type).Synchronize(queue_);
  // Release the lazily-created async-output copy queue (gpu_model_runner.py has
  // no explicit teardown; our vt::Queue owns a CUDA stream that must be freed).
  if (async_copy_queue_.id != 0) {
    vt::DestroyQueue(async_copy_queue_);
  }
  // SPEC-DFLASH2 A2-2: the verify D2H's fork/ready events, created once and
  // re-recorded per step, released with the queue that carried them.
  if (verify_events_created_) {
    vt::Backend& b = vt::GetBackend(queue_.device.type);
    b.DestroyEvent(verify_fork_event_);
    b.DestroyEvent(verify_ready_event_);
    verify_events_created_ = false;
  }
  // W4 device mirror. Freed here rather than leaked like the scratch pool: these
  // are per-runner, and a serving process can construct more than one runner.
  if (async_device_inputs_ != nullptr) {
    vt::Backend& b = vt::GetBackend(queue_.device.type);
    // DRAIN BEFORE THE FREE (SPEC-DFLASH2 A2-3 repair, #2911). This is the ONLY
    // site that frees any of them, and until this repair it argued its own
    // safety from "the runner and therefore its queue are going away". That is
    // not an argument: the runner does not own `queue_` and does not destroy it,
    // only `async_copy_queue_` is destroyed here, and a combine or a forward
    // queued against these buffers is still queued when this destructor runs.
    // Freeing memory a queued kernel reads is exactly what `20d225e54` repaired
    // in `RejectionSamplerDeviceOutput::Release` one level down; `main` sets
    // that standard and this is the same tree's other half of it.
    //
    // WHAT IT COSTS: one full drain of the main queue per runner destruction,
    // once in a process's life for the ordinary single-runner server, and it is
    // paid at TEARDOWN with no request in flight. That is the same trade the
    // `Release` repair took — an observable stall instead of an unobservable
    // wrong answer — and here the stall is off every serving path entirely.
    //
    // IT IS EXECUTED ON NEITHER TIER, and the commit that landed it said "the
    // mirror's allocation and its entry in the destructor's free list are
    // outside the `#ifdef` and ARE compiled here", which is true and reads as
    // coverage it does not have. `async_device_mirror()` has its whole body
    // inside `#ifdef VLLM_CPP_CUDA`, so on the CPU build it always answers false,
    // `async_device_inputs_` is always null, and this entire block — the
    // `Synchronize` included — never runs. On the CUDA build it would run and is
    // not compiled here (no nvcc on this box). Deleting the drain leaves all
    // eleven of this row's targets green. It rests on READING: sole destruction
    // path, `queue_` is the queue `stage_upload` and both combine launches use,
    // correctly scoped, no double drain, no deadlock.
    //
    // THE ORDERING SMELL A2-3 RECORDED HERE IS CLOSED (SPEC-DFLASH2 A2-4,
    // #3004). `DestroyQueue(async_copy_queue_)` and the two `DestroyEvent` calls
    // used to run BEFORE this drain, so the drain protected the mirror's buffers
    // and neither of them. The drain now runs at the TOP of this destructor and
    // covers all three; this call is kept because it is free on an already-idle
    // queue and because deleting it would make the free below depend on a line
    // thirty lines away for its safety.
    b.Synchronize(queue_);
    // `draft_tokens` and `cu_num_logits` are in the list because they are in the
    // struct; see the struct's declaration for why A2-4 must not move a free
    // here by moving a wait.
    for (int32_t* p : {async_device_inputs_->last_sampled,
                       async_device_inputs_->prefill_len,
                       async_device_inputs_->query_start_loc,
                       async_device_inputs_->seq_lens,
                       async_device_inputs_->input_ids,
                       async_device_inputs_->ops,
                       async_device_inputs_->draft_tokens,
                       async_device_inputs_->cu_num_logits}) {
      if (p != nullptr) b.Free(p);
    }
  }
}

vt::Queue& GPUModelRunner::get_or_create_async_copy_queue() {
  // _get_or_create_async_output_copy_stream (gpu_model_runner.py:1137-1141): a
  // single dedicated copy stream on the runner's device, created on first use so
  // the synchronous production path never allocates it.
  if (async_copy_queue_.id == 0) {
    async_copy_queue_ = vt::CreateQueue(queue_.device);
  }
  return async_copy_queue_;
}

// ─── PinnedGrowStaging (SPEC-DFLASH2 A2-2, #2802) ───────────────────────────
// See the header for why the destination of an unwaited D2H must be page-locked
// and why a grow keeps the block it replaces.

PinnedGrowStaging::~PinnedGrowStaging() {
  if (backend_ == nullptr) return;
  for (int32_t* block : blocks_) {
    if (block != nullptr) backend_->FreePinned(block);
  }
  blocks_.clear();
}

int32_t* PinnedGrowStaging::Get(vt::Backend& backend, size_t elems) {
  if (elems == 0) elems = 1;
  if (!blocks_.empty() && elems <= elems_ && backend_ == &backend) {
    return blocks_.back();
  }
  // A backend swap can only happen if the runner's device changed, which it
  // cannot; assert it by construction rather than silently mixing allocators.
  VT_CHECK(backend_ == nullptr || backend_ == &backend,
           "verify download staging: the backend must not change under a live block");
  backend_ = &backend;
  blocks_.push_back(static_cast<int32_t*>(backend.AllocPinned(elems * sizeof(int32_t))));
  elems_ = elems;
  return blocks_.back();
}

// ─── DownloadCommittedIds (SPEC-DFLASH2 A2-4, #3004) ────────────────────────
// See the header for why it is unconditional, and for the ownership contract.

const int64_t* DownloadCommittedIds(vt::Backend& backend, vt::Queue& main_q,
                                    vt::Queue& copy_q, vt::Event& fork,
                                    vt::Event& ready, const void* dev_ids,
                                    int num_reqs, PinnedGrowStaging& staging) {
  if (num_reqs <= 0) return nullptr;
  VT_CHECK(dev_ids != nullptr,
           "DownloadCommittedIds: the step sampled " + std::to_string(num_reqs) +
               " ids into no buffer");
  // Two int32 elements per int64 id. `Get` returns a block at least this large
  // and retains any block it grows past, so a copy already in flight into an
  // older block is writing memory this object still owns.
  const size_t elems = static_cast<size_t>(num_reqs) * 2;
  int32_t* pinned = staging.Get(backend, elems);
  backend.RecordEvent(fork, main_q);
  backend.QueueWaitEvent(copy_q, fork);
  backend.Copy(copy_q, pinned, dev_ids,
               static_cast<size_t>(num_reqs) * sizeof(int64_t));
  backend.RecordEvent(ready, copy_q);
  backend.SynchronizeEvent(ready);
  return reinterpret_cast<const int64_t*>(pinned);
}

void GPUModelRunner::ensure_verify_events() {
  // SPEC-DFLASH2 A2-2. Two events, created on the first copy-queue verify and
  // re-recorded every step after that. `blocking=false` matches the async
  // sampled-id path: at depth-1 the host is going to wait on this copy
  // immediately, so a spin is the right wait.
  if (verify_events_created_) return;
  vt::Backend& b = vt::GetBackend(queue_.device.type);
  verify_fork_event_ = b.CreateEvent();
  verify_ready_event_ = b.CreateEvent();
  verify_events_created_ = true;
}

AsyncOutputPool& GPUModelRunner::get_or_create_async_output_pool() {
  // Persistent per-step overlap buffers, sized to the batch bound (max_num_reqs)
  // and pre-seeded with a few slots (depth-2 in flight + grammar-deferral
  // headroom), created on first async use. This removes ALL per-step
  // cudaMalloc/cudaHostAlloc/cudaEventCreate from sample_tokens_async — the raw
  // device-syncing allocator calls that serialized the depth-2 overlap.
  if (async_output_pool_ == nullptr) {
    const int cap = max_num_reqs_ > 0 ? max_num_reqs_ : 1;
    async_output_pool_ =
        std::make_unique<AsyncOutputPool>(queue_.device, cap, /*initial_slots=*/4);
  }
  return *async_output_pool_;
}

// ─── ENG-ASYNC-SCHED W4: CUDA device-resident async inputs ───────────────────

// W4, DEFAULT ON since the 2026-08-06 CORRECTNESS flip (ROW-SERVE-ASYNC-LLM P0).
// It landed default OFF (W3's A/B precedent: prove the speed win before flipping),
// and the c16 A/B came back NEUTRAL (0.999x — the drain MOVE overlaps only host
// prep). But the token-exactness probe that A/B ran uncovered a latent, SHIPPING
// correctness bug the OFF path CANNOT avoid: on the async serving loop
// (AsyncLLM -> step_with_batch_queue -> sample_tokens_async) the sampled token is
// not written to token_ids_cpu synchronously (see sample_tokens_async), so the
// next step's prepare_inputs reads a stale/zero decode-row placeholder and RELIES
// on the device combine. On the OFF integrated path the combine patches
// step.input_token_ids on the MAIN QUEUE while the Qwen3.5 decode graph reads that
// same host vector on the CPU (BuildPaddedDecode -> CopyInPlace -> EmbedInto's
// host->device upload) with NO intervening sync — an unsynchronized
// device-write/host-read race that nondeterministically embeds the zero
// placeholder and degenerates batch-1 greedy decode into repeated token-0 garbage.
// Making step.input_token_ids device-resident (this mirror) routes the combine's
// output into the embed via ApplyDeviceTokenIdsOverride, main-queue-ordered after
// the combine, so the embed never does the racing host read. The only OFF-path
// alternative is a per-step Synchronize after the combine — which reintroduces the
// host sync the async path exists to remove (a c16 REGRESSION) and is a GB10-only
// band-aid, not parity. This is exactly what upstream does on every platform
// (states.py:64 keeps prev_sampled_token_ids a GPU tensor; gpu_model_runner.py's
// _prepare_input_ids gathers on the GPU), so flipping the default ON is MIRROR-vLLM
// parity, speed-neutral, and gated: VT_ASYNC_DEVICE_MIRROR=0 is the rollback to the
// (racy) host-array path for an A/B. The async-serving token-exact gate
// (test_qwen36_async_serving.cpp) is RED on =0 and GREEN on the default.
//
// Distinct from VT_ASYNC_RUNNER, which would also turn off async scheduling
// itself; keeping them separate is what makes an honest A/B of W4 alone possible —
// same binary, same scheduler, one mechanism.
#ifdef VLLM_CPP_CUDA
// Guarded with its only use below: on a CPU build the mirror cannot exist, and
// an unused static function is a -Werror=unused-function break there. DEFAULT ON:
// on unless VT_ASYNC_DEVICE_MIRROR is explicitly "0" (the rollback), mirroring the
// AsyncRunnerFlagIsOn convention.
static bool AsyncDeviceMirrorEnvDefault() {
  const char* value = std::getenv("VT_ASYNC_DEVICE_MIRROR");
  return value == nullptr || value[0] != '0';
}
#endif

bool GPUModelRunner::async_device_mirror() const {
  if (async_device_mirror_cached_ >= 0) return async_device_mirror_cached_ != 0;
  bool on = false;
#ifdef VLLM_CPP_CUDA
  // Engage on any real CUDA GPU, integrated OR discrete — NOT the CPU backend.
  //  - DISCRETE (separate memory, !UnifiedMemory): the mirror is REQUIRED, because
  //    the host fallback would main-stream Synchronize to read the sampled ids.
  //  - INTEGRATED (GB10, UnifiedMemory AND is_integrated_gpu): the mirror is not
  //    required for addressability (host arrays ARE device-addressable), but
  //    making last_sampled_tokens device-resident removes the host condense<->
  //    scatter read-after-write that pins the async drain to the TOP of
  //    execute_model, letting the bulk host prep overlap the GPU tail (a drain
  //    MOVE) AND — the reason it is now DEFAULT ON — removes the unsynchronized
  //    combine-write/host-read race the OFF path has with the decode-graph embed
  //    (ROW-SERVE-ASYNC-LLM P0). VT_ASYNC_DEVICE_MIRROR=0 rolls back to it.
  //  - The CPU backend (UnifiedMemory AND not an integrated GPU) keeps the in-place
  //    host path — mirroring there would only add copies with no GPU tail to hide.
  // is_integrated_gpu() is already read at the combine/scatter sites in this file,
  // so it introduces no new device-type token to the leakage ratchet.
  on = async_input_combine_ && AsyncDeviceMirrorEnvDefault() &&
       (!vt::GetBackend(queue_.device.type).UnifiedMemory() ||
        vllm::platforms::GetPlatform(queue_.device.type).is_integrated_gpu());
#endif
  async_device_mirror_cached_ = on ? 1 : 0;
  return on;
}

// VT_ASYNC_EXECUTOR (decode-graph slot double-buffer, ENG-ASYNC-SCHED c16/c32
// overlap unlock). DEFAULT OFF — an opt-in speed lever. Engages only where the
// depth-2 moved drain and the decode graph both exist: the device mirror path on
// a real CUDA GPU. On the CPU backend / mirror-OFF path the drain does not move
// and there is no decode graph, so the lever is inert. Memoized. Reading a plain
// "not 0" env keeps a value of "1" the canonical enable; anything else (unset,
// "0") leaves the drain in place, which is byte-identical production.
bool GPUModelRunner::async_executor() const {
  if (async_executor_cached_ >= 0) return async_executor_cached_ != 0;
  bool on = false;
#ifdef VLLM_CPP_CUDA
  const char* value = std::getenv("VT_ASYNC_EXECUTOR");
  on = value != nullptr && value[0] == '1' && value[1] == '\0' &&
       async_device_mirror();
#endif
  async_executor_cached_ = on ? 1 : 0;
  return on;
}

GPUModelRunner::AsyncDeviceInputs*
GPUModelRunner::get_or_create_async_device_inputs() {
  if (!async_device_mirror()) return nullptr;
  if (async_device_inputs_ != nullptr) return async_device_inputs_.get();

  const int reqs = max_num_reqs_ > 0 ? max_num_reqs_ : 1;
  const int toks = max_num_batched_tokens_ > 0 ? max_num_batched_tokens_ : 1;
  vt::Backend& b = vt::GetBackend(queue_.device.type);
  auto dev = std::make_unique<AsyncDeviceInputs>();
  dev->max_reqs = reqs;
  dev->input_ids_capacity = toks;

  auto alloc_i32 = [&](int64_t count) {
    void* p = b.Alloc(static_cast<size_t>(count) * sizeof(int32_t));
    b.Memset(queue_, p, 0, static_cast<size_t>(count) * sizeof(int32_t));
    return static_cast<int32_t*>(p);
  };
  dev->last_sampled = alloc_i32(reqs);
  dev->prefill_len = alloc_i32(reqs);
  dev->query_start_loc = alloc_i32(static_cast<int64_t>(reqs) + 1);
  // SPEC-DFLASH2 A2-3 REPAIR (#2911): the same shape as query_start_loc, and
  // allocated unconditionally for the same reason — on a decode step it holds
  // the arange a null used to stand for, so the arm has one expression for
  // every step instead of one that is only correct while the veto stands.
  dev->cu_num_logits = alloc_i32(static_cast<int64_t>(reqs) + 1);
  dev->seq_lens = alloc_i32(reqs);
  dev->input_ids = alloc_i32(toks);
  dev->ops = alloc_i32(4LL * reqs);
  // SPEC-DFLASH2 A2-3 (#2911): the device mirror of the per-req_state draft
  // buffer. Sized by the req_state POOL for the reason the header states — the
  // kernel's row read is unbounded — and allocated once, freed only in the
  // destructor, so no later wave can free it by moving a wait. Nothing is
  // allocated without a speculator: `num_speculative_steps` is 0 and both the
  // pointer and the stride stay at their inert defaults.
  dev->draft_stride = input_batch_.num_speculative_steps;
  if (dev->draft_stride > 0) {
    dev->draft_tokens =
        alloc_i32(static_cast<int64_t>(reqs) * dev->draft_stride);
  }

  // The mirror starts from whatever the host array already holds. In production
  // that is all zeros (no request has been admitted yet), but seeding from the
  // host makes the mirror correct even if a runner is switched on mid-flight,
  // and it costs one copy for the process.
  b.Copy(queue_, dev->last_sampled, input_batch_.last_sampled_tokens.data(),
         static_cast<size_t>(reqs) * sizeof(int32_t));
  // The seeds recorded so far are already reflected by that copy; dropping them
  // here keeps the log from replaying them a second time.
  input_batch_.last_sampled_ops.clear();

  async_device_inputs_ = std::move(dev);
  return async_device_inputs_.get();
}

void GPUModelRunner::stage_upload(AsyncDeviceInputs& dev, int32_t* dst,
                                  const int32_t* src, int64_t count) {
  (void)dev;
  if (count <= 0) return;
  // Copied straight from the caller's PAGEABLE host buffer, deliberately: for a
  // pageable source the driver stages the bytes before cudaMemcpyAsync returns,
  // so the caller may reuse or destroy that buffer immediately, while the copy
  // itself stays ordered on the queue.
  //
  // A shared PINNED staging buffer would be the usual optimization and is WRONG
  // here: pinned copies are truly asynchronous, so the next upload's memcpy into
  // the shared buffer could overwrite bytes an in-flight DMA had not yet read,
  // and with a depth-2 scheduler that window spans steps. Making that safe needs
  // per-upload regions plus an event per step; these arrays are a few kilobytes
  // on the front of a step, so the staged pageable copy is the better trade.
  vt::GetBackend(queue_.device.type)
      .Copy(queue_, dst, src, static_cast<size_t>(count) * sizeof(int32_t));
}

void GPUModelRunner::replay_last_sampled_ops(AsyncDeviceInputs& dev) {
#ifdef VLLM_CPP_CUDA
  std::vector<InputBatch::LastSampledOp>& ops = input_batch_.last_sampled_ops;
  if (ops.empty()) return;
  // Flatten to (kind, a, b, value) quads. The log is bounded by the number of
  // admissions/removals in one step, so it fits the [4 * max_num_reqs] buffer;
  // if a step ever exceeded that, replaying a truncated prefix would silently
  // corrupt the mirror, so drain it in whole chunks instead.
  const int64_t cap_ops = 4LL * dev.max_reqs;
  size_t done = 0;
  std::vector<int32_t> flat;
  while (done < ops.size()) {
    const size_t chunk =
        std::min(ops.size() - done, static_cast<size_t>(dev.max_reqs));
    flat.clear();
    flat.reserve(chunk * 4);
    for (size_t i = 0; i < chunk; ++i) {
      const InputBatch::LastSampledOp& op = ops[done + i];
      flat.push_back(op.kind);
      flat.push_back(op.a);
      flat.push_back(op.b);
      flat.push_back(op.value);
    }
    VT_CHECK(static_cast<int64_t>(flat.size()) <= cap_ops,
             "async device mirror: structural-op chunk exceeds its buffer");
    stage_upload(dev, dev.ops, flat.data(), static_cast<int64_t>(flat.size()));
    vt::cuda::LaunchApplyLastSampledOps(queue_, dev.last_sampled, dev.ops,
                                        static_cast<int>(chunk));
    done += chunk;
  }
  ops.clear();
#else
  (void)dev;
#endif
}

std::unique_ptr<AsyncModelRunnerOutput> GPUModelRunner::sample_tokens_async(
    const std::optional<GrammarOutput>& grammar_output) {
  // ARCH-ONE-SURFACE ROW 6: pooling models resolve async scheduling OFF
  // (config/vllm.py:1068-1073 mirror in LoadedEngine::ResolveAsyncEnabled), so
  // the depth-2 async sampler must never see one — refuse loudly rather than
  // run the device sampler over hidden states.
  VT_CHECK(pooling_runner_ == nullptr,
           "sample_tokens_async: pooling models use the synchronous scheduler "
           "(async scheduling is disabled for pooling, config/vllm.py:1068)");
  // When async is NOT engaged (production default), degenerate to the byte-
  // identical synchronous path wrapped as a ready output — so a caller in the
  // depth-2 loop can always call sample_tokens_async without branching, yet the
  // sync behavior is unchanged. The device-resident overlap path below runs only
  // under VT_ASYNC_RUNNER / set_async_input_combine.
  if (!async_input_combine_) {
    return std::make_unique<ReadyModelRunnerOutput>(sample_tokens(grammar_output));
  }

  const vt::Device dev = queue_.device;
  const int num_reqs = exec_state_.num_reqs;

  ModelRunnerOutput skeleton;
  AsyncOutputPool& pool = get_or_create_async_output_pool();
  vt::Queue& copy_q = get_or_create_async_copy_queue();
  if (num_reqs == 0) {
    // 0-token flush step: nothing sampled. Return an already-complete async
    // output (zero rows) so the seam is uniform. A pool slot is borrowed and
    // released with no copy — no per-step allocation.
    AsyncOutputSlot* slot = pool.Acquire();
    return std::make_unique<AsyncGPUModelRunnerOutput>(
        std::move(skeleton), dev, pool, slot, /*num_reqs=*/0, queue_, copy_q);
  }

  std::vector<float> sampled_logits;
  vt::Tensor logits = assemble_sample_logits(grammar_output, sampled_logits);
  // #2534: the SAME dump, on the branch the production default actually takes.
  dump_step_logits(logits);
  // SAMPLE-PROMPT-LOGPROBS: same seam as the synchronous path. The prompt rows
  // are read off this step's forward result HERE, synchronously, because that
  // is where the logits still are — the async output only defers the sampled
  // IDS. Inert unless a request asked.
  collect_prompt_logprobs(skeleton.prompt_logprobs_dict);

  // ─── SPEC-DECODE VERIFY ROUTING (SPEC-DFLASH2 A2-2, #2802) ────────────────
  //
  // THE SAME PREDICATE `sample_tokens` ROUTES ON, read from the same field
  // through the same function (`StepRoutesToVerify`, prepare_inputs.h). Upstream
  // has one sampling entry point and asks this once
  // (model_runner.py:1129 @ pin 5559679229); we have two, so the rule is asked
  // twice and must be the one expression, not two agreeing readings.
  //
  // Without this arm the async sampler would take the expanded Σ(1 + k_i) verify
  // rows straight to `assemble_sample_logits` and sample them as if they were
  // decode rows: no draft would ever be verified, no next block proposed,
  // `pending_drafts_` would stay empty and the async scheduler's `-1`
  // placeholders would have nothing to fill. That is reason B of this row's
  // veto, and this arm is what closes it.
  //
  // The download route is the wave's payload. `kCopyQueueEvent` forks the copy
  // queue off the main queue with an event and waits only that event, so the
  // accept walk's copy is no longer a main-stream operation and its wait is a
  // copy-queue event — mirroring how upstream gets its spec-step ids across
  // (AsyncOutput, model_runner.py:1492-1499). The host STILL waits in step, and
  // the shape is what changed, not the stall: A2-4 and A2-5 are the waves that
  // move the wait. The TOKENS are unchanged: same kernel, same numbers.
  if (StepRoutesToVerify(exec_state_.step.num_draft_tokens)) {
    ModelRunnerOutput out_rej =
        sample_tokens_with_rejection(logits, VerifyDownload::kCopyQueueEvent);
    out_rej.prompt_logprobs_dict = std::move(skeleton.prompt_logprobs_dict);
    propose_after_verify(num_reqs);
    // A verify step emits a RAGGED token list per request (accepted drafts plus
    // the bonus/replacement), which the single-id AsyncGPUModelRunnerOutput
    // cannot carry: its pinned buffer is one int64 per request. The tokens are
    // already host-resident above, so the result is handed back ready. Deferring
    // the ragged materialization is A2-4's half of this wave (the optimistic
    // `prev_num_draft_len` plus the deferred correction), and it needs the
    // propose to stop consuming host num_sampled first, which is A2-3.
    return std::make_unique<ReadyModelRunnerOutput>(std::move(out_rej));
  }

  const SamplingMetadata sm = input_batch_.make_sampling_metadata();

  // Sample DEVICE-RESIDENT: the sampler writes the ids into the pool slot's
  // PERSISTENT device buffer (the pool guarantees the slot is not reused until
  // this step's async output releases it, so a depth-2 next step cannot tear this
  // snapshot). The sampler writes it device-resident (all-greedy: no host
  // download inside the sampler), and the async output issues the single
  // sampled-id copy to host on the copy queue. Mirrors sampler_output.
  // sampled_token_ids staying GPU-side (async_utils.py:31). No per-step
  // cudaMalloc here — the slot buffer is reused across steps.
  AsyncOutputSlot* slot = pool.Acquire();
  void* dev_ids = slot->device_sampled_ids;
  vt::Tensor dev_ids_t = vt::Tensor::Contiguous(
      dev_ids, vt::DType::kI64, dev, {static_cast<int64_t>(num_reqs)});
  SamplerOutput sampler_output = sampler_.forward(queue_, logits, sm, &dev_ids_t);
  // The sampler's logprobs ride on the async output, as upstream's
  // AsyncGPUModelRunnerOutput carries logprobs_tensors and get_output returns
  // them (gpu_model_runner.py:264-295, 320-325 @ 5559679229). A request that
  // asks for logprobs already took the sampler's host path, so they are host
  // rows here; nullopt when nobody asked. Discarding them emptied every
  // AsyncLLM logprobs request (ISSUE-LOCAL-01M3SE6RVKD6SCMA2YBS7F8X0R).
  skeleton.logprobs = std::move(sampler_output.logprobs_tensors);

  // post_update (input_batch.py:457-543 post_update / states.py): record this
  // step's last sampled id per req_state so the NEXT step's
  // combine_sampled_and_draft_tokens builds its decode input id, and advance the
  // write-back column counter. The token VALUE append to token_ids_cpu /
  // req_output_token_ids is DELETED on the async path (the runner does the
  // post_update only; host token bookkeeping for detok/penalties is fed by the
  // scheduler's update_from_output when get_output() materializes).
  //
  skeleton.req_ids.reserve(static_cast<size_t>(num_reqs));

  // SPEC-DFLASH2 A2-4 (#3004): WHERE THIS STEP'S COMMITTED IDS ENDED UP, AND
  // WHY THE PROPOSE TAIL NO LONGER ASKS.
  //
  // The three write-back branches below differ in one way the propose tail used
  // to care about: whether the ids this step committed are readable ON THE HOST
  // when the tail runs. The host branch Synchronizes and writes
  // `input_batch_.last_sampled_tokens` itself, so they are. Both CUDA branches
  // scatter them with a kernel queued on the main queue and nothing waits it —
  // the mirror branch deliberately leaves the host array's VALUES stale, and the
  // UMA branch writes the host array from a kernel that has not run yet.
  //
  // #2920 asked the question with a `committed_ids_on_host` flag the branch set
  // and the tail refused on. That refusal is what blocks A2-5: on GB10's
  // integrated default the UMA branch runs and hits it. A2-4 DELETES THE
  // QUESTION instead of answering it — the tail below downloads the ids itself,
  // unconditionally and outside every branch, so no branch decides anything and
  // a fourth one cannot forget to. The branches keep their write-backs (the
  // mirror and the device array are what the NEXT step's combine reads); what
  // they no longer own is the propose's view of them.
#ifdef VLLM_CPP_CUDA
  // W4 device-resident scatter. Preferred whenever the mirror is engaged
  // (async_device_mirror(): CUDA + VT_ASYNC_DEVICE_MIRROR, INTEGRATED OR DISCRETE):
  // write each row's sampled id into the DEVICE mirror (dinp->last_sampled) on the
  // MAIN queue, main-stream-ordered with the next step's combine, so the sampled
  // ids never round-trip the host. On the INTEGRATED path this is what keeps the
  // sampled tokens OFF the host array update_states' condense reorders (the drain-
  // move lever). The host `last_sampled_tokens` is deliberately NOT updated here —
  // reading the ids back is the cost being removed; its rows still move with
  // condense/swap and those moves are replayed onto the mirror, but its VALUES are
  // stale and nothing reads them (the sampled ids reach the engine through the
  // async output's own copy, as upstream does). Runs OUTSIDE any CUDA-graph capture.
  if (AsyncDeviceInputs* dinp = get_or_create_async_device_inputs();
      dinp != nullptr) {
    vt::cuda::LaunchScatterLastSampled(queue_, dinp->last_sampled,
                                       static_cast<const int64_t*>(dev_ids),
                                       /*idx_mapping=*/nullptr, num_reqs);
    for (int i = 0; i < num_reqs; ++i) {
      const std::string& req_id = exec_state_.req_ids[static_cast<size_t>(i)];
      skeleton.req_ids.push_back(req_id);
      skeleton.req_id_to_index[req_id] = i;
      if (i < static_cast<int>(exec_state_.discard.size()) &&
          exec_state_.discard[static_cast<size_t>(i)]) {
        continue;
      }
      input_batch_.num_tokens_no_spec[static_cast<size_t>(i)] += 1;
    }
  } else if (vllm::platforms::GetPlatform(dev.type).is_integrated_gpu()) {
    // DEVICE scatter, HOST-ARRAY residence (W3 DGX leaf; mirror OFF = GB10 default):
    // write each row's sampled id into the device-ADDRESSABLE host
    // last_sampled_tokens (pageable on GB10's UMA) on the MAIN queue, main-stream-
    // ordered with the next step's device combine, so the ids never round-trip the
    // host. DELETES the pre-scatter `Synchronize` the host-fallback needed (its ONLY
    // purpose was making the host read of dev_ids well-defined). idx_mapping
    // identity (nullptr); the host bookkeeping below needs no device read. Because
    // this is the array condense reorders, its scatter pins the drain to
    // execute_model's top (the mirror path lifts that). is_integrated_gpu()
    // decouples a future discrete GPU (false -> host bookkeeping below).
    vt::cuda::LaunchScatterLastSampled(
        queue_, input_batch_.last_sampled_tokens.data(),
        static_cast<const int64_t*>(dev_ids), /*idx_mapping=*/nullptr, num_reqs);
    for (int i = 0; i < num_reqs; ++i) {
      const std::string& req_id = exec_state_.req_ids[static_cast<size_t>(i)];
      skeleton.req_ids.push_back(req_id);
      skeleton.req_id_to_index[req_id] = i;
      // discard_request_mask: a still-prefilling request produced no output
      // token this step, so do NOT advance its write-back column (bumping it
      // would desync num_tokens from the scheduler). The stale last_sampled row
      // the scatter wrote is never read while the request is prefilling (combine
      // skips seq_len <= prefill_len) and is overwritten on its first decode.
      if (i < static_cast<int>(exec_state_.discard.size()) &&
          exec_state_.discard[static_cast<size_t>(i)]) {
        continue;
      }
      input_batch_.num_tokens_no_spec[static_cast<size_t>(i)] += 1;
    }
  } else
#endif
  {
    // HOST path (CPU backend): the sampler already ran synchronously, but on any
    // non-CUDA async-degeneration backend `Synchronize` keeps the host read of
    // `dev_ids` well-defined (no read-before-write race). This is the ONE ordering
    // cost the host-array combine pays; the CUDA device-kernel branch above removes
    // it. Runs OUTSIDE any CUDA-graph capture (post-sample host-side).
    vt::GetBackend(dev.type).Synchronize(queue_);
    const int64_t* ids = static_cast<const int64_t*>(dev_ids);
    for (int i = 0; i < num_reqs; ++i) {
      const std::string& req_id = exec_state_.req_ids[static_cast<size_t>(i)];
      skeleton.req_ids.push_back(req_id);
      skeleton.req_id_to_index[req_id] = i;
      // discard_request_mask: skip the write-back for a still-prefilling request
      // (no output token generated this step).
      if (i < static_cast<int>(exec_state_.discard.size()) &&
          exec_state_.discard[static_cast<size_t>(i)]) {
        continue;
      }
      input_batch_.last_sampled_tokens[static_cast<size_t>(i)] =
          static_cast<int32_t>(ids[i]);
      input_batch_.num_tokens_no_spec[static_cast<size_t>(i)] += 1;
    }
  }

  // ─── SPEC-DFLASH2 A2-4 (#2920): PROPOSE AFTER A PLAIN DECODE STEP ──────────
  //
  // A2-2 gave this function a verify arm and put `propose_after_verify` inside
  // it, so a step that DRAFTS proposed correctly and a step that did not
  // proposed nothing at all. A speculative engine's FIRST step carries no
  // drafts — there is no previous step to have proposed — so with the veto
  // lifted that engine would sample, propose nothing, and every step after it
  // would find nothing to verify: `pending_drafts_` stays empty, the async
  // scheduler's `-1` placeholders have nothing to fill, and `FillDraftsForStep`
  // refuses by name.
  //
  // Upstream cannot have this gap because it has ONE propose tail for both
  // routes: `sample()` picks the sampler on `input_batch.num_draft_tokens == 0`
  // (gpu/model_runner.py:1129 @ pin 5559679229) and the single
  // `if self.speculator is not None:` at :1524-1547 proposes from whichever
  // sampler ran. `propose_after_decode` is that tail's decode half, named once
  // and applied here and in `sample_tokens`, so the two entry points cannot
  // derive `num_sampled` / `num_rejected` differently. The ROUTE stays the one
  // expression too: this is simply the fall-through of the
  // `StepRoutesToVerify(...)` branch above, not a second reading of it.
  if (spec_on()) {
    // SPEC-DFLASH2 A2-4 (#3004): THE IDS, WHERE THEY LIVE. One download, on
    // every write-back branch, on the copy queue, waited on its own event.
    // #2920 asked which branch had left them host-readable and refused when the
    // answer was "neither"; asking is what blocked A2-5, because on GB10's
    // integrated default the answer is "neither" every step. There is no
    // question left to answer here and therefore no refusal to remove later.
    //
    // The fork event is recorded on the MAIN queue at this point, so the copy is
    // ordered after `sampler_.forward` AND after whichever write-back scatter
    // ran; `dev_ids` is the sampler's own output buffer, which is what both
    // scatters read. See `DownloadCommittedIds` (runner.h) for the ownership
    // contract on the returned pointer.
    ensure_verify_events();
    const int64_t* committed = DownloadCommittedIds(
        vt::GetBackend(dev.type), queue_, get_or_create_async_copy_queue(),
        verify_fork_event_, verify_ready_event_, dev_ids, num_reqs,
        committed_ids_staging_);
    // THE HOST STATE EVERY PROPOSE ARM READS, written from that one array: the
    // block drafters' `last_sampled_tokens` anchor and the token-row column
    // `propose_drafts_ngram` matches over. #2920 restored only the second, from
    // `last_sampled_tokens` — which is exactly the array the mirror branch
    // leaves STALE, so that restore was correct only on the branch that did not
    // need it. The rule is stated once in `prepare_inputs.h` over arrays, not
    // looped here, because a per-request rule feeding a per-step route is how
    // this row shipped a silent wrong answer (#2710).
    ApplyCommittedIdsForPropose(committed, num_reqs, exec_state_.discard,
                                input_batch_.num_tokens_no_spec,
                                input_batch_.max_model_len,
                                &input_batch_.last_sampled_tokens,
                                &input_batch_.token_ids_cpu);
    propose_after_decode(num_reqs);
  }

  // discard_request_mask (gpu_model_runner.py:3625-3628 -> outputs.py:303): the
  // rows get_output() must clear to empty. Prefill-chunk requests sampled a
  // garbage token at their prefill position; the scheduler must see EMPTY ids.
  std::vector<int32_t> invalid_req_indices;
  for (int i = 0; i < num_reqs; ++i) {
    if (i < static_cast<int>(exec_state_.discard.size()) &&
        exec_state_.discard[static_cast<size_t>(i)]) {
      invalid_req_indices.push_back(i);
    }
  }

  // ENG-ASYNC-SCHED depth-2 LIFETIME GUARD: this path issued the sampled-id D2H
  // on the COPY queue and left the forward / sample / scatter on the MAIN queue
  // UNSYNCED (that overlap is the whole point). Those kernels still reference
  // exec_state_ and write input_batch_.last_sampled_tokens, so the NEXT
  // execute_model() must drain the main queue before it resets exec_state_ or
  // mutates input_batch_. Flag it. (The host-fallback branch above already
  // Synchronized, so its flag is a no-op next step — harmless.)
  async_forward_in_flight_ = true;

  // Issue the non-blocking sampled-id D2H on the COPY queue + record the event.
  // The async output BORROWS the pool slot (device buffer already holds the
  // argmax ids) and releases it on consume — no per-step free.
  return std::make_unique<AsyncGPUModelRunnerOutput>(
      std::move(skeleton), dev, pool, slot, num_reqs, queue_, copy_q,
      std::move(invalid_req_indices));
}

// ─── KV-EXTERNAL-CACHE (LMCache) worker-side store/load ──────────────────────
//
// The full-attention page is [2, block_size, num_kv_heads, head_size] per block
// (backend.h:241), so within a block the K plane is the first half and the V
// plane the second — each exactly [block_size, hidden_dim] contiguous bytes,
// which is byte-for-byte the LMCache KV_2LTD per-layer plane the connector's
// PutKv2ltd/GetKv2ltd expect. We store/load each half at its own page offset, so
// an our-store -> our-load cycle restores the block bit-for-bit (the whole point
// of the output-invariance gate). Device<->host movement goes through the
// backend Copy (a UMA memcpy on GB10), synchronized before the bytes are read.

namespace {
// The full-attention page geometry a store/load needs. num_layers == the number
// of full-attention KV buffers, hidden_dim == num_kv_heads*head_size, plane_bytes
// == one K (or V) plane, page_bytes == the whole [K|V] block.
struct FaGeom {
  int num_layers = 0;
  int block_size = 0;
  int hidden_dim = 0;
  std::size_t elem_size = 0;
  std::size_t plane_bytes = 0;
  std::size_t page_bytes = 0;
};
}  // namespace

void GPUModelRunner::ConnectorLoadExternalKv() {
  using kv_offload::lmcache::LMCacheConnector;
  auto* lm = dynamic_cast<LMCacheConnector*>(kv_connector_);
  if (lm == nullptr) return;  // a base/non-worker connector: nothing to load
  std::vector<kv_offload::lmcache::LmcacheLoadJob> loads =
      lm->TakeConnectorLoads();
  if (loads.empty()) return;
  VT_CHECK(!attn_kv_.empty(),
           "LMCache load: model has no full-attention KV group");

  FaGeom g;
  g.num_layers = static_cast<int>(full_attn_buf_.size());
  g.block_size = static_cast<int>(attn_kv_[0].block_size);
  g.hidden_dim = static_cast<int>(attn_kv_[0].num_kv_heads * attn_kv_[0].head_size);
  g.elem_size = vt::SizeOf(attn_kv_[0].dtype);
  g.plane_bytes = static_cast<std::size_t>(g.block_size) *
                  static_cast<std::size_t>(g.hidden_dim) * g.elem_size;
  g.page_bytes = 2 * g.plane_bytes;
  VT_CHECK(g.page_bytes == static_cast<std::size_t>(fa_page_size_bytes_),
           "LMCache load: full-attention page is padded/asymmetric — refusing "
           "to move KV under an unexpected layout");
  const auto& c = lm->config();
  VT_CHECK(c.num_layers == g.num_layers && c.hidden_dim == g.hidden_dim &&
               c.chunk_tokens == g.block_size && c.elem_size == g.elem_size,
           "LMCache load: connector geometry disagrees with the runner's KV "
           "cache — refusing to load (identity safety)");

  vt::Backend& backend = vt::GetBackend(queue_.device.type);
  std::vector<std::string> kp, vp;
  for (const auto& job : loads) {
    VT_CHECK(job.keys.size() == job.gpu_block_ids.size(),
             "LMCache load: key/block-id count mismatch");
    for (std::size_t j = 0; j < job.keys.size(); ++j) {
      kp.clear();
      vp.clear();
      // A chunk the scheduler recorded as a HIT that is now absent is a real
      // bug (torn store, evicted mid-decode), never silently recomputed here.
      const bool got = lm->LoadChunk(job.keys[j], &kp, &vp);
      VT_CHECK(got,
               "LMCache load: a chunk recorded as an external-prefix hit is "
               "absent on the server");
      VT_CHECK(static_cast<int>(kp.size()) == g.num_layers &&
                   static_cast<int>(vp.size()) == g.num_layers,
               "LMCache load: plane count != num_layers");
      const int block_id = job.gpu_block_ids[j];
      for (int l = 0; l < g.num_layers; ++l) {
        VT_CHECK(kp[static_cast<std::size_t>(l)].size() == g.plane_bytes &&
                     vp[static_cast<std::size_t>(l)].size() == g.plane_bytes,
                 "LMCache load: plane byte count != expected");
        auto* base = static_cast<uint8_t*>(full_attn_buf_[static_cast<std::size_t>(l)]->data()) +
                     static_cast<std::size_t>(block_id) * g.page_bytes;
        backend.Copy(queue_, base, kp[static_cast<std::size_t>(l)].data(),
                     g.plane_bytes);
        backend.Copy(queue_, base + g.plane_bytes,
                     vp[static_cast<std::size_t>(l)].data(), g.plane_bytes);
      }
    }
  }
  backend.Synchronize(queue_);
}

void GPUModelRunner::ConnectorStorePromptKv(
    const SchedulerOutput& scheduler_output) {
  using kv_offload::lmcache::LMCacheConnector;
  auto* lm = dynamic_cast<LMCacheConnector*>(kv_connector_);
  if (lm == nullptr) return;
  if (full_attn_group_id_ < 0 || attn_kv_.empty()) return;
  const int num_reqs = input_batch_.num_reqs();
  if (num_reqs == 0) return;

  FaGeom g;
  g.num_layers = static_cast<int>(full_attn_buf_.size());
  g.block_size = static_cast<int>(attn_kv_[0].block_size);
  g.hidden_dim = static_cast<int>(attn_kv_[0].num_kv_heads * attn_kv_[0].head_size);
  g.elem_size = vt::SizeOf(attn_kv_[0].dtype);
  g.plane_bytes = static_cast<std::size_t>(g.block_size) *
                  static_cast<std::size_t>(g.hidden_dim) * g.elem_size;
  g.page_bytes = 2 * g.plane_bytes;
  VT_CHECK(g.page_bytes == static_cast<std::size_t>(fa_page_size_bytes_),
           "LMCache store: full-attention page is padded/asymmetric");
  const auto& cfg = lm->config();
  VT_CHECK(cfg.num_layers == g.num_layers && cfg.hidden_dim == g.hidden_dim &&
               cfg.chunk_tokens == g.block_size && cfg.elem_size == g.elem_size,
           "LMCache store: connector geometry disagrees with the runner's KV "
           "cache");

  int fa_cols = 0;
  const std::vector<int32_t> fa_bt =
      gather_block_table(full_attn_group_id_, num_reqs, &fa_cols);

  vt::Backend& backend = vt::GetBackend(queue_.device.type);
  // The forward's ReshapeAndCache writes may still be in flight on the queue;
  // the device reads below must see the committed KV.
  backend.Synchronize(queue_);

  std::vector<std::string> kp(static_cast<std::size_t>(g.num_layers));
  std::vector<std::string> vp(static_cast<std::size_t>(g.num_layers));
  for (int i = 0; i < num_reqs; ++i) {
    const std::string& req_id = *input_batch_.req_ids[static_cast<std::size_t>(i)];
    const int num_prompt =
        input_batch_.num_prompt_tokens[static_cast<std::size_t>(i)];
    const int computed_before =
        input_batch_.num_computed_tokens_cpu[static_cast<std::size_t>(i)];
    const int num_scheduled = scheduler_output.num_scheduled_tokens.at(req_id);
    // Post-step computed count, clamped to the prompt (offload_prompt_only).
    const int prompt_done = std::min(computed_before + num_scheduled, num_prompt);
    const int complete_blocks = prompt_done / g.block_size;
    int already = connector_stored_blocks_[req_id];  // 0 if unseen
    if (complete_blocks <= already) continue;

    const int need = complete_blocks * g.block_size;
    std::vector<int32_t> toks;
    toks.reserve(static_cast<std::size_t>(need));
    for (int col = 0; col < need; ++col) {
      toks.push_back(input_batch_.token_id(i, col));
    }
    const std::vector<uint64_t> folds = lm->ChunkFolds(toks);
    for (int b = already; b < complete_blocks &&
                          b < static_cast<int>(folds.size());
         ++b) {
      const int block_id = fa_bt[static_cast<std::size_t>(i) *
                                     static_cast<std::size_t>(fa_cols) +
                                 static_cast<std::size_t>(b)];
      for (int l = 0; l < g.num_layers; ++l) {
        kp[static_cast<std::size_t>(l)].resize(g.plane_bytes);
        vp[static_cast<std::size_t>(l)].resize(g.plane_bytes);
        const auto* base =
            static_cast<const uint8_t*>(full_attn_buf_[static_cast<std::size_t>(l)]->data()) +
            static_cast<std::size_t>(block_id) * g.page_bytes;
        backend.Copy(queue_, kp[static_cast<std::size_t>(l)].data(), base,
                     g.plane_bytes);
        backend.Copy(queue_, vp[static_cast<std::size_t>(l)].data(),
                     base + g.plane_bytes, g.plane_bytes);
      }
      backend.Synchronize(queue_);  // planes host-valid before the wire PUT
      lm->StoreChunk(lm->ChunkKey(folds[static_cast<std::size_t>(b)]), kp, vp);
    }
    connector_stored_blocks_[req_id] = complete_blocks;
  }
}

}  // namespace vllm::v1
