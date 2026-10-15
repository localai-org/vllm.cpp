// Ported from: vllm/v1/core/sched/scheduler.py @ e24d1b24
// See include/vllm/v1/core/sched/scheduler.h for the T0 scope + deferred list.
#include "vllm/v1/core/sched/scheduler.h"

#include "vllm/multimodal/utils.h"  // GetMmFeaturesInWindow (P2, #2379)

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "vllm/v1/core/sched/utils.h"           // check_stop
#include "vllm/v1/kv_offload/kv_connector.h"    // KVConnector abstract ABI (W5)
#include "vllm/v1/structured_output/manager.h"  // StructuredOutputManager

namespace vllm::v1 {

namespace {

// Prefill progress (chunked prefill). ON if VT_SERVER_PREFILL_PROGRESS=1 or
// VT_SERVER_VERBOSE=1. Rate-limited ~2.5 Hz per request.
//
// The progress line is emitted in two halves (external PR #227). `begin` is
// written at SCHEDULE time, because that is the only moment that proves the
// request left the waiting queue at all. `running` / `done` are written AFTER
// execute_model, because schedule() advances num_computed_tokens BEFORE the GPU
// has run: timing them at schedule time reported the token rate of an empty
// step and made a slow prefill look instantaneous.
bool PrefillProgressEnabled() {
  static const bool on = [] {
    const char* p = std::getenv("VT_SERVER_PREFILL_PROGRESS");
    if (p && p[0] == '0') return false;
    if (p && p[0] == '1') return true;
    const char* v = std::getenv("VT_SERVER_VERBOSE");
    return v && v[0] == '1';
  }();
  return on;
}

using PrefillClock = std::chrono::steady_clock;

struct PrefillState {
  PrefillClock::time_point first{};  // first scheduled chunk (t0 for the rate)
  PrefillClock::time_point last{};   // last emitted `running` line
  int first_computed = -1;
  int last_computed = -1;
  bool logged_begin = false;
  bool logged_done = false;
};

std::mutex& PrefillMu() {
  static std::mutex mu;
  return mu;
}

std::unordered_map<std::string, PrefillState>& PrefillStates() {
  static std::unordered_map<std::string, PrefillState> states;
  return states;
}

// Bound the per-request state map. It is keyed by request id, so without this it
// grows for the process lifetime — one entry per request ever served.
// Called with PrefillMu() held.
//
// Two eviction classes, because `logged_done` alone does not bound it: a request
// ABORTED or preempted mid-prefill never reaches done and would pin its entry
// forever. Anything no longer known to the scheduler is gone for good.
void PrefillEvictLocked(
    const std::map<std::string, std::unique_ptr<Request>>& live_requests) {
  if (PrefillStates().size() <= 64) return;
  auto& states = PrefillStates();
  for (auto it = states.begin(); it != states.end();) {
    if (it->second.logged_done || live_requests.find(it->first) ==
                                      live_requests.end()) {
      it = states.erase(it);
    } else {
      ++it;
    }
  }
}

// Emitted when tokens are SCHEDULED (before the GPU runs), so `first` anchors
// the elapsed-time base at the moment work actually started.
void PrefillMarkScheduled(const Request& request, int scheduled_this_step) {
  if (!PrefillProgressEnabled() || scheduled_this_step <= 0) return;
  const int prompt = request.num_prompt_tokens > 0 ? request.num_prompt_tokens
                                                   : request.NumTokens();
  if (prompt <= 0) return;
  const int computed = request.num_computed_tokens;
  if (computed >= prompt) return;  // already past prefill

  std::lock_guard<std::mutex> lock(PrefillMu());
  PrefillState& st = PrefillStates()[request.request_id];
  const auto now = PrefillClock::now();
  if (st.first.time_since_epoch().count() == 0) {
    st.first = now;
    st.first_computed = computed;
  }
  if (st.logged_begin) return;
  st.logged_begin = true;
  st.last = now;
  st.last_computed = computed;
  const int remaining = prompt - computed;
  std::cerr << "INFO prefill id=" << request.request_id << " status=begin"
            << " prompt_tokens=" << prompt << " already_computed=" << computed
            << " remaining=" << remaining
            << " scheduling=" << scheduled_this_step
            << (remaining > scheduled_this_step ? " chunked=1" : " chunked=0")
            << "\n";
  std::cerr.flush();
}

// Called AFTER execute_model, so elapsed_s covers real GPU prefill time.
void PrefillLogAfterExecute(
    const SchedulerOutput& out,
    const std::map<std::string, std::unique_ptr<Request>>& reqs) {
  if (!PrefillProgressEnabled()) return;
  if (out.num_scheduled_tokens.empty()) return;
  const auto now = PrefillClock::now();
  std::lock_guard<std::mutex> lock(PrefillMu());
  for (const auto& [req_id, scheduled] : out.num_scheduled_tokens) {
    if (scheduled <= 0) continue;
    const auto rit = reqs.find(req_id);
    if (rit == reqs.end() || !rit->second) continue;
    const Request& request = *rit->second;
    const int prompt = request.num_prompt_tokens > 0 ? request.num_prompt_tokens
                                                     : request.NumTokens();
    if (prompt <= 0) continue;
    const int computed = request.num_computed_tokens;
    // Decode phase: computed exceeds prompt once generation tokens append.
    if (computed > prompt && !request.is_prefill_chunk) continue;

    PrefillState& st = PrefillStates()[req_id];
    if (st.first.time_since_epoch().count() == 0) {
      st.first = now;
      st.first_computed = std::max(0, computed - scheduled);
    }
    const int shown = std::min(computed, prompt);
    const double elapsed_s = std::chrono::duration<double>(now - st.first).count();
    const int made = std::max(0, shown - std::max(0, st.first_computed));
    const double avg_tok_s =
        elapsed_s > 1e-3 ? static_cast<double>(std::max(made, 1)) / elapsed_s
                         : static_cast<double>(scheduled);

    const bool done = !request.is_prefill_chunk && computed >= prompt;
    if (done) {
      if (!st.logged_done) {
        const double el = std::max(elapsed_s, 1e-4);
        const double tok_s = static_cast<double>(std::max(made, scheduled)) / el;
        std::cerr << "INFO prefill id=" << req_id << " computed=" << shown << "/"
                  << prompt << " (100%) status=done elapsed_s=" << el
                  << " prefill_tok_s=" << tok_s << " avg_tok_s=" << avg_tok_s
                  << " scheduled_last=" << scheduled << "\n";
        std::cerr.flush();
        st.logged_done = true;
      }
      continue;
    }

    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - st.last)
            .count();
    if (st.last_computed >= 0 && ms < 400 && (computed - st.last_computed) < 256) {
      continue;
    }
    const double window_s = st.last.time_since_epoch().count() == 0
                                ? elapsed_s
                                : std::chrono::duration<double>(now - st.last).count();
    const int delta =
        st.last_computed < 0 ? scheduled : std::max(0, computed - st.last_computed);
    const double inst_tok_s = window_s > 1e-4
                                  ? static_cast<double>(delta) / window_s
                                  : static_cast<double>(scheduled);
    st.last = now;
    st.last_computed = computed;
    const int remain = prompt - shown;
    const double pct =
        100.0 * static_cast<double>(shown) / static_cast<double>(prompt);
    std::cerr << "INFO prefill id=" << req_id << " computed=" << shown << "/"
              << prompt << " (" << static_cast<int>(pct + 0.5)
              << "%) status=running remaining=" << remain
              << " elapsed_s=" << elapsed_s << " inst_tok_s=" << inst_tok_s
              << " avg_tok_s=" << avg_tok_s << " scheduled=" << scheduled;
    if (avg_tok_s > 1.0 && remain > 0) {
      std::cerr << " eta_s=" << (static_cast<double>(remain) / avg_tok_s);
    }
    std::cerr << "\n";
    std::cerr.flush();
  }
  PrefillEvictLocked(reqs);
}

// Map the config-level policy onto the request-queue policy.
//   kLPM (ENG-SGLANG-BEHAVIOR-FLAG) rides on the FCFS deque: the queue mechanics
//   are identical to fcfs (append / popleft / prepend), and the cache-aware
//   ordering is imposed by Scheduler::schedule() reordering the deque each step
//   (waiting->reorder), exactly as SGLang's SchedulePolicy.calc_priority sorts
//   the waiting list in place (schedule_policy.py:176,205).
SchedulingPolicy ToQueuePolicy(SchedulerPolicy policy) {
  switch (policy) {
    case SchedulerPolicy::kFCFS:
    case SchedulerPolicy::kLPM:
      return SchedulingPolicy::kFCFS;
    case SchedulerPolicy::kPriority:
      return SchedulingPolicy::kPriority;
  }
  return SchedulingPolicy::kFCFS;
}

}  // namespace

Scheduler::Scheduler(SchedulerConfig scheduler_config,
                     KVCacheConfig kv_cache_config, int block_size,
                     bool enable_caching,
                     StructuredOutputManager* structured_output_manager,
                     std::optional<SpeculativeConfig> speculative_config,
                     const distributed::KVEventsConfig* kv_events_config,
                     int data_parallel_rank, int hash_block_size)
    : max_num_running_reqs(scheduler_config.max_num_seqs),
      max_num_scheduled_tokens(
          scheduler_config.ResolvedMaxNumScheduledTokens()),
      max_model_len(scheduler_config.max_model_len),
      structured_output_manager_(structured_output_manager),
      scheduler_config_(scheduler_config),
      kv_cache_config_(kv_cache_config),
      long_prefill_token_threshold_(
          scheduler_config.long_prefill_token_threshold),
      enable_chunked_prefill_(scheduler_config.enable_chunked_prefill),
      scheduler_reserve_full_isl_(
          scheduler_config.scheduler_reserve_full_isl),
      block_size_(block_size) {
  // num_lookahead_tokens (scheduler.py:275-292): 0 when no speculator is
  // configured (the default — every spec-decode path stays inert), else derived
  // from the SpeculativeConfig (k for MTP). Threaded into allocate_slots below.
  if (speculative_config.has_value()) {
    num_lookahead_tokens_ = speculative_config->NumLookaheadTokens();
    // num_spec_tokens (scheduler.py:241): the per-step draft count, feeding
    // SchedulerOutput::num_spec_tokens_to_schedule (W7 #1824). 0 by default.
    num_spec_tokens_ = speculative_config->ResolvedNumSpeculativeTokens();
  }

  // Scheduling policy -> the waiting (FCFS) queue.
  waiting = create_request_queue(ToQueuePolicy(scheduler_config.policy));

  // KV-cache events (scheduler.py:116-119,155-158). A null config is upstream's
  // `None`: events off, so this resolves to the `false` that used to be
  // hard-coded below, and the factory hands back a no-op NullEventPublisher.
  // The config pointee is consumed here and NOT retained.
  enable_kv_cache_events_ = kv_events_config != nullptr &&
                            kv_events_config->enable_kv_cache_events;
  kv_event_publisher_ = distributed::EventPublisherFactory::create(
      kv_events_config, data_parallel_rank);

  // Build the KV cache manager (upstream scheduler.py ctor). use_eagle off and
  // dcp/pcp world sizes 1 at T0.
  //
  // `hash_block_size` is the CALLER'S when it gave one and `block_size`
  // otherwise, which is upstream's `if hash_block_size is None:
  // hash_block_size = block_size` (scheduler.py:268-270). This line used to
  // spell `block_size` unconditionally, and that is what aborted `vllm serve`
  // on DeepSeek-V4: the engine strides 256 while the architecture's SWA group
  // pages at 64, and `UnitaryKVCacheCoordinator` asserts the hash granularity
  // equals the group's block size (kv_cache_coordinator.py:516 @ e126687a9a).
  // A hard-coded equality cannot express a model whose groups page smaller.
  const int resolved_hash_block_size =
      hash_block_size > 0 ? hash_block_size : block_size;

  // log_stats is ON: upstream's `disable_log_stats` defaults False, and the
  // benchmark protocol VOIDS any caching arm that cannot report queries/hits.
  // Cost is three integer adds per admitted request.
  kv_cache_manager = std::make_unique<KVCacheManager>(
      kv_cache_config_, max_model_len, /*scheduler_block_size=*/block_size,
      resolved_hash_block_size,
      /*max_num_batched_tokens=*/scheduler_config.max_num_batched_tokens,
      enable_caching, /*use_eagle=*/false, /*log_stats=*/true,
      enable_kv_cache_events_, /*dcp_world_size=*/1,
      /*pcp_world_size=*/1, scheduler_config.watermark);

  // ENG-MM-INPUT-PIPELINE P2 (#2379): the encoder cache + its per-step compute
  // budget (scheduler.py:226-238). Upstream builds the manager unconditionally
  // and passes cache_size 0 when the model declares no multimodal budget; the
  // manager is then consulted only through `if request.has_encoder_inputs`, so
  // a text engine never reaches it either way. The EncoderDecoderCacheManager
  // arm (Whisper cross-attention) is not ported: is_encoder_decoder is not a
  // config this scheduler carries.
  max_num_encoder_input_tokens_ = scheduler_config.max_num_encoder_input_tokens;
  disable_chunked_mm_input_ = scheduler_config.disable_chunked_mm_input;
  encoder_cache_manager = std::make_unique<core::EncoderCacheManager>(
      scheduler_config.encoder_cache_size);
}

void Scheduler::shutdown() {
  // scheduler.py:2456-2459. The publisher is never null (the factory returns a
  // NullEventPublisher when events are off), so this needs no guard; upstream's
  // `if self.kv_event_publisher:` guards an attribute that can be unset in
  // partially-constructed teardown, which cannot happen here. The connector /
  // ec_connector shutdowns upstream also performs belong to their own rows.
  kv_event_publisher_->shutdown();
}

void Scheduler::add_request(std::unique_ptr<Request> request) {
  const std::string request_id = request->request_id;
  waiting->add_request(request.get());
  // QUEUED event (scheduler.py:2135): stamp the wait-queue entry time so the
  // frontend can measure the queued interval (scheduled_ts - queued_ts). Default
  // (nullopt) timestamp -> record_event stamps the current monotonic time.
  if (log_stats_) {
    request->record_event(EngineCoreEventType::kQueued);
  }
  requests[request_id] = std::move(request);
}

void Scheduler::finish_requests(const std::string& request_id,
                                RequestStatus finished_status) {
  assert(vllm::v1::IsFinished(finished_status) &&
         "finish_requests requires a finished status");
  auto it = requests.find(request_id);
  if (it == requests.end() || it->second->IsFinished()) {
    // Invalid / already-finished request id.
    return;
  }
  Request* request = it->second.get();

  // Remove from the queue it currently lives in.
  if (request->status == RequestStatus::kRunning) {
    running.erase(std::remove(running.begin(), running.end(), request),
                  running.end());
  } else {
    waiting->remove_request(request);
  }

  // _free_request (T0 subset): mark finished, record the id, free the KV blocks,
  // and erase the owning entry (which destroys the Request).
  request->status = finished_status;
  finished_req_ids.insert(request_id);
  kv_cache_manager->free(*request);
  free_request_encoder_inputs(*request);
  requests.erase(it);
}

void Scheduler::preempt_request(Request* request, double timestamp) {
  assert(request->status == RequestStatus::kRunning &&
         "Only running requests can be preempted");
  // _free_request_blocks (T0: defer_block_free off -> free immediately).
  kv_cache_manager->free(*request);
  // ENG-MM-INPUT-PIPELINE P2 (#2379): `self.encoder_cache_manager.free(request)`
  // (scheduler.py:1213), in the SAME place and for the same reason as the block
  // free on the line above. A preemption resets num_computed_tokens to 0, so the
  // resumed request re-schedules every encoder input from the start; the
  // references it held before the preemption would then never be dropped by
  // `_free_encoder_inputs` (whose test is `offset + length <=
  // num_computed_tokens`, and num_computed_tokens just went to zero). Each
  // preemption of an image request would leak one cache entry until the request
  // finished, and `CanAllocate` would start refusing admissions the cache had
  // room for. No-op for a request with no mm_features.
  free_request_encoder_inputs(*request);
  request->status = RequestStatus::kPreempted;
  request->num_computed_tokens = 0;
  // scheduler.py:1217-1218 (W7 #1824): drop un-verified drafts — real values
  // under the sync flow, -1 placeholders under async. A resumed request
  // re-enters through prefill; stale drafts scheduled beside it would verify
  // garbage (and, under async, would not pair with any worker-kept drafts).
  if (!request->spec_token_ids.empty()) {
    request->spec_token_ids.clear();
  }
  // Upstream Request.num_preemptions (read by PrefixCacheStats.record at
  // vllm/v1/core/kv_cache_manager.py:239).
  request->num_preemptions += 1;
  // PREEMPTED event (scheduler.py:1221): a preemption occurred; the frontend
  // bumps its num_preemptions counter and the interval derivations fold the
  // preempted span into prefill/inference. Uses schedule()'s shared timestamp.
  if (log_stats_) {
    request->record_event(EngineCoreEventType::kPreempted, timestamp);
  }
  // Put the request back to the FRONT of the waiting queue (FCFS retry).
  waiting->prepend_request(request);
  reset_preempted_req_ids.insert(request->request_id);
}

void Scheduler::maybe_reorder_waiting_for_lpm() {
  // ENG-SGLANG-BEHAVIOR-FLAG (SW1) — SGLang cache-aware LPM admission ordering,
  // ported from SchedulePolicy.calc_priority + _sort_by_longest_prefix
  // (python/sglang/srt/managers/schedule_policy.py:176,205 @ v0.5.15 f63458b),
  // mapped onto OUR waiting deque + OUR block-hash APC longest-match (reusing the
  // same index the admission loop matches against — NOT a second trie).
  if (scheduler_config_.policy != SchedulerPolicy::kLPM) {
    return;  // fcfs / priority: no cache-aware reorder (byte-identical default).
  }
  // `lpm` is meaningless with prefix caching off (no cache to match) -> fcfs.
  // (The server also warns at load; here we simply leave arrival order intact.)
  if (kv_cache_manager == nullptr || !kv_cache_manager->enable_caching) {
    return;
  }
  const std::size_t n = waiting->size();
  if (n <= 1) {
    return;  // nothing to reorder.
  }
  // Large-queue fallback (schedule_policy.py:229-233): skip the expensive
  // per-request match + sort when the waiting queue is large; stay fcfs.
  if (n > kLpmMaxWaitingQueue) {
    return;
  }

  // Snapshot the current (arrival / fcfs) order and compute each request's
  // longest cached-prefix match as a PURE read (no stats, no LRU touch).
  std::vector<Request*> order = waiting->ToList();
  std::unordered_map<const Request*, int> matched;
  matched.reserve(order.size());
  for (Request* r : order) {
    matched[r] = kv_cache_manager->num_matched_prefix_tokens(*r);
  }

  // SW2 — in-batch prefix-collision de-prioritization. Ported 1:1 from SGLang
  // SchedulePolicy._compute_prefix_matches (schedule_policy.py:253-301 @ f63458b).
  // We walk the waiting queue in ARRIVAL order (the pre-sort order — SGLang runs
  // this BEFORE _sort_by_longest_prefix) and build up an ephemeral set of block-
  // hash APC keys seen so far (SGLang's `waiting_queue_radix_tree`, a simulated
  // radix built only from the waiting queue — here reused via OUR block hashes,
  // NOT a second trie). A request whose REAL cached match is small
  // (<= kInBatchCheckThreshold; schedule_policy.py:277) and whose longest prefix
  // against the already-seen requests is large (>= kInBatchDeprioritizeThreshold;
  // schedule_policy.py:288-292) is temporarily de-prioritized; otherwise its
  // prefix is inserted into the seen set so a later collider matches it
  // (schedule_policy.py:293-300). Block hashes are chained over the prefix, so a
  // shared k-block prefix yields k identical leading hashes: the in-batch match
  // is the contiguous run of leading hashes already in the seen set.
  std::unordered_set<const Request*> temporary_deprioritized;
  {
    std::unordered_set<BlockHash> seen;  // block-hash keys of inserted prefixes.
    for (Request* r : order) {
      // schedule_policy.py:277 — only check requests with a small real match.
      if (matched[r] > kInBatchCheckThreshold) {
        continue;
      }
      // Longest in-batch prefix match: contiguous leading block hashes of r that
      // some earlier-seen request already contributed (chained hashes make the
      // run contiguous-from-front). Measured in tokens for the threshold.
      int in_batch_match_tokens = 0;
      for (const BlockHash& h : r->block_hashes) {
        if (seen.find(h) == seen.end()) {
          break;
        }
        in_batch_match_tokens += block_size_;
      }
      if (in_batch_match_tokens >= kInBatchDeprioritizeThreshold) {
        // schedule_policy.py:292 — collides with an in-flight uncached prefix.
        temporary_deprioritized.insert(r);
      } else {
        // schedule_policy.py:295-300 — first seer of this prefix; record it.
        for (const BlockHash& h : r->block_hashes) {
          seen.insert(h);
        }
      }
    }
  }

  // _sort_by_longest_prefix (schedule_policy.py:303-314): sort DESCENDING by
  // num_matched_prefix_tokens, but a de-prioritized request takes the sort key
  // `float("inf")` — i.e. it sorts BEHIND every non-de-prioritized request. A
  // STABLE sort keeps arrival (fcfs) order among equal-key requests, matching
  // Python's list.sort.
  std::stable_sort(
      order.begin(), order.end(),
      [&matched, &temporary_deprioritized](const Request* a, const Request* b) {
        const bool a_dep = temporary_deprioritized.count(a) != 0;
        const bool b_dep = temporary_deprioritized.count(b) != 0;
        if (a_dep != b_dep) {
          return !a_dep;  // non-de-prioritized sorts ahead of de-prioritized.
        }
        return matched[a] > matched[b];
      });
  waiting->reorder(order);
}


// _try_schedule_encoder_inputs (scheduler.py:1379). See scheduler.h for why the
// num_new_tokens clamp is a correctness guard rather than an optimization.
std::vector<int> Scheduler::try_schedule_encoder_inputs(
    const Request& request, int num_computed_tokens, int* num_new_tokens,
    int* encoder_compute_budget) {
  if (*num_new_tokens == 0 || request.mm_features.empty()) {
    return {};
  }
  std::vector<int> encoder_inputs_to_schedule;
  // Per-STEP trackers, because the scheduler decides at request granularity
  // while the budget is spent at item granularity (upstream's own NOTE at
  // :1414-1416).
  std::set<std::string> mm_hashes_to_schedule;
  int64_t num_embeds_to_schedule = 0;

  const std::pair<int, int> window = multimodal::GetMmFeaturesInWindow(
      request.mm_features, num_computed_tokens,
      num_computed_tokens + *num_new_tokens);

  for (int i = window.first; i < window.second; ++i) {
    const multimodal::MultiModalFeatureSpec& feature =
        request.mm_features[static_cast<size_t>(i)];
    const int start_pos = feature.offset;
    const int num_encoder_tokens = feature.length;
    // get_num_embeds() upstream. Our MultiModalFeatureSpec carries no `is_embed`
    // mask (that is the use_audio_in_video interleave, which nothing here
    // produces), so every placeholder token IS an embedding row and the two
    // counts coincide. Recorded rather than assumed, because if an interleaved
    // modality ever lands, this is the line that has to change with it.
    const int64_t num_encoder_embeds = num_encoder_tokens;

    if (mm_hashes_to_schedule.count(feature.mm_hash) != 0) {
      // The same encoder input is already scheduled this step (two placeholders
      // that hash equal share one encoder run).
      continue;
    }
    if (encoder_cache_manager->CheckAndUpdateCache(request.request_id, i,
                                                   feature.mm_hash)) {
      // Already computed and cached by an earlier step (or another request).
      continue;
    }

    if (disable_chunked_mm_input_ && num_computed_tokens < start_pos &&
        (num_computed_tokens + *num_new_tokens) <
            (start_pos + num_encoder_tokens)) {
      // Roll the whole step back to before this item rather than covering part
      // of its span.
      *num_new_tokens = std::max(0, start_pos - num_computed_tokens);
      break;
    }

    if (!encoder_cache_manager->CanAllocate(num_encoder_embeds,
                                            *encoder_compute_budget,
                                            num_embeds_to_schedule)) {
      // The encoder cache is full or the encoder budget is exhausted. NOTE
      // (woosuk): encoder input tokens are processed altogether because the
      // encoder is bidirectional, so there is no partial answer to fall back to.
      if (num_computed_tokens < start_pos) {
        // Schedule only the decoder tokens ahead of the encoder input.
        *num_new_tokens = start_pos - num_computed_tokens;
      } else {
        // Prefix caching pushed num_computed_tokens past start_pos while this
        // item's encoder output is not available, so nothing can be scheduled
        // for this request this step.
        *num_new_tokens = 0;
      }
      break;
    }

    // How much of this item's placeholder span the step actually covers. Zero
    // means the window overlap is empty for this item and the encoder run is not
    // needed yet (upstream's get_embeds_indices_in_range collapse).
    const int start_idx_rel = std::max(0, num_computed_tokens - start_pos);
    const int end_idx_rel = std::min(
        num_encoder_tokens, num_computed_tokens + *num_new_tokens - start_pos);
    if (end_idx_rel - start_idx_rel <= 0) {
      continue;
    }

    num_embeds_to_schedule += num_encoder_embeds;
    *encoder_compute_budget -= static_cast<int>(num_encoder_embeds);
    mm_hashes_to_schedule.insert(feature.mm_hash);
    encoder_inputs_to_schedule.push_back(i);
  }
  return encoder_inputs_to_schedule;
}

// _free_encoder_inputs (scheduler.py:2017).
void Scheduler::free_encoder_inputs(const Request& request) {
  if (request.mm_features.empty()) return;
  const std::unordered_set<int> cached =
      encoder_cache_manager->GetCachedInputIds(request.request_id);
  if (cached.empty()) return;
  // use_eagle is false in this tree, so the drafter's +1 look-ahead shift
  // upstream applies (:2026-2028) is 0. Named rather than dropped: a later
  // EAGLE row has to put it back or an entry frees one token too early.
  const int spec_lookahead = 0;
  // Copy: FreeEncoderInput mutates the set this iterates (upstream's own
  // `list(set)` for the same reason).
  const std::vector<int> input_ids(cached.begin(), cached.end());
  for (const int input_id : input_ids) {
    if (input_id < 0 ||
        input_id >= static_cast<int>(request.mm_features.size())) {
      continue;
    }
    const multimodal::MultiModalFeatureSpec& feature =
        request.mm_features[static_cast<size_t>(input_id)];
    if (feature.offset + feature.length + spec_lookahead <=
        request.num_computed_tokens - request.num_output_placeholders) {
      encoder_cache_manager->FreeEncoderInput(request.request_id, input_id,
                                              feature.mm_hash, feature.length);
    }
  }
}

// encoder_cache_manager.free(request) (encoder_cache_manager.py:243).
void Scheduler::free_request_encoder_inputs(const Request& request) {
  if (request.mm_features.empty()) return;
  const std::unordered_set<int> cached =
      encoder_cache_manager->GetCachedInputIds(request.request_id);
  const std::vector<int> input_ids(cached.begin(), cached.end());
  for (const int input_id : input_ids) {
    if (input_id < 0 ||
        input_id >= static_cast<int>(request.mm_features.size())) {
      continue;
    }
    const multimodal::MultiModalFeatureSpec& feature =
        request.mm_features[static_cast<size_t>(input_id)];
    encoder_cache_manager->FreeEncoderInput(request.request_id, input_id,
                                            feature.mm_hash, feature.length);
  }
}

SchedulerOutput Scheduler::schedule() {
  current_step_ += 1;
  // scheduled_timestamp (scheduler.py:461): one monotonic reading for the whole
  // step, shared by every SCHEDULED event and by any PREEMPTED event a
  // scheduling-time preemption records, so a request scheduled and a request
  // preempted in the same step carry a consistent instant.
  const double scheduled_timestamp = MonotonicSeconds();
  // NOTE(woosuk): there is no "prefill" nor "decode" phase — each request just
  // has num_computed_tokens and num_tokens, and each step assigns tokens so the
  // former catches up to the latter (covers chunked prefill uniformly).

  std::vector<Request*> scheduled_new_reqs;
  std::vector<Request*> scheduled_resumed_reqs;
  std::vector<Request*> scheduled_running_reqs;
  std::vector<Request*> preempted_reqs;

  std::map<std::string, KVCacheBlocks> req_to_new_blocks;
  std::map<std::string, int> num_scheduled_tokens;
  // Speculative decode: req_id -> the draft token ids scheduled for verification
  // this step (scheduler.py:593-609). Stays empty when no request carries drafts
  // (num_lookahead_tokens == 0 -> byte-identical to the pre-SPEC-MTP path).
  std::map<std::string, std::vector<int32_t>> scheduled_spec_decode_tokens;
  // ENG-MM-INPUT-PIPELINE P2 (#2379): req_id -> mm_features indices to encode
  // this step (scheduler.py:453-454), and the per-step encoder compute budget
  // both admission loops spend down. Both stay empty/untouched on a text step.
  std::map<std::string, std::vector<int>> scheduled_encoder_inputs;
  int encoder_compute_budget = max_num_encoder_input_tokens_;
  int token_budget = max_num_scheduled_tokens;

  kv_cache_manager->new_step_starts();

  // Whether the priority policy is in effect (else FCFS). Chosen once per step
  // (mirrors scheduler.py reading self.policy inside schedule()).
  const bool priority_policy =
      scheduler_config_.policy == SchedulerPolicy::kPriority;

  // `long_prefill_token_threshold` exists to stop a long prefill from
  // starving other requests of the token budget. When it is the only
  // request there is nobody to starve, so let it use the whole budget.
  // (scheduler.py:602-607, upstream commit c64b15cde5)
  const int long_prefill_threshold =
      (running.size() + waiting->size() > 1)
          ? long_prefill_token_threshold_
          : 0;

  // First, schedule the RUNNING requests. req_index is a signed int so the
  // priority-preemption `req_index -= 1` fix-up (upstream scheduler.py:570) is
  // well-defined.
  int req_index = 0;
  while (req_index < static_cast<int>(running.size()) && token_budget > 0) {
    Request* request = running[req_index];

    // Async scheduling (scheduler.py:445-459): avoid scheduling an extra step
    // once we are sure the previous step reached max_tokens. `+2` is
    // (num_computed + 1) - (num_output_placeholders - 1) — placeholders are in
    // the computed count, so we subtract (placeholders - 1) to drop draft
    // tokens. INERT while num_output_placeholders == 0 (synchronous Scheduler).
    if (request->num_output_placeholders > 0 &&
        request->num_computed_tokens + 2 - request->num_output_placeholders >=
            request->num_prompt_tokens + request->MaxTokens()) {
      req_index += 1;
      continue;
    }
    // V2+PP+async decode cadence (scheduler.py:461-465): enforce pp_size steps
    // between same-req decodes. INERT while next_decode_eligible_step == 0.
    if (current_step_ < request->next_decode_eligible_step) {
      req_index += 1;
      continue;
    }

    // num_tokens_with_spec adds the request's pending draft tokens so they are
    // scheduled for verification alongside the sampled token (scheduler.py:475).
    // Equals num_tokens when spec_token_ids is empty (no speculator / drafts
    // already consumed) -> byte-identical to the pre-SPEC-MTP path. Under async
    // scheduling num_output_placeholders reserves the in-flight sampled token(s)
    // (0 for the synchronous Scheduler, so this is unchanged there).
    int num_new_tokens = request->NumTokensWithSpec() +
                         request->num_output_placeholders -
                         request->num_computed_tokens;
    if (0 < long_prefill_threshold &&
        long_prefill_threshold < num_new_tokens) {
      num_new_tokens = long_prefill_threshold;
    }
    num_new_tokens = std::min(num_new_tokens, token_budget);
    // Do not let the input position exceed the max model len.
    num_new_tokens =
        std::min(num_new_tokens, max_model_len - request->num_computed_tokens -
                                     num_sampled_tokens_per_step_);

    // Schedule encoder inputs (scheduler.py:522-536). May CLAMP num_new_tokens
    // — including to 0, which the `continue` below then handles exactly as
    // upstream's reason 2/3 ("the encoder budget/cache is exhausted"). Empty and
    // budget-neutral for a request with no mm_features. `new_encoder_budget` is
    // a COPY that only becomes the running budget once the request is actually
    // scheduled, mirroring upstream's new_encoder_compute_budget.
    std::vector<int> encoder_inputs_to_schedule;
    int new_encoder_budget = encoder_compute_budget;
    if (!request->mm_features.empty()) {
      encoder_inputs_to_schedule = try_schedule_encoder_inputs(
          *request, request->num_computed_tokens, &num_new_tokens,
          &new_encoder_budget);
    }

    if (num_new_tokens == 0) {
      // Nothing to schedule for this request (e.g. it has reached its cap).
      // NOTE(woosuk): `continue` (not `break`) — do not strictly follow FCFS,
      // let lower-priority requests be scheduled.
      req_index += 1;
      continue;
    }

    // Schedule the KV blocks, preempting on OOM until the request fits.
    std::optional<KVCacheBlocks> new_blocks;
    while (true) {
      new_blocks = kv_cache_manager->allocate_slots(
          *request, num_new_tokens, /*num_new_computed_tokens=*/0,
          /*new_computed_blocks=*/std::nullopt,
          /*num_lookahead_tokens=*/num_lookahead_tokens_);
      if (new_blocks.has_value()) {
        break;  // The request can be scheduled.
      }

      // The request cannot be scheduled — preempt one running request.
      // (scheduler.py:546-572.)
      Request* preempted_req = nullptr;
      if (priority_policy) {
        // Preempt the lowest-priority running request: the max of running by
        // (priority, arrival_time). std::max_element returns the FIRST maximum,
        // matching Python max()'s first-encountered tie behavior.
        auto victim = std::max_element(
            running.begin(), running.end(), [](Request* a, Request* b) {
              return std::make_pair(a->priority, a->arrival_time) <
                     std::make_pair(b->priority, b->arrival_time);
            });
        preempted_req = *victim;
        running.erase(victim);
        // If the victim was already scheduled earlier this step, undo it:
        // restore its token budget, drop its block reservation, and step
        // req_index back one (running shrank in front of the cursor).
        auto sit = std::find(scheduled_running_reqs.begin(),
                             scheduled_running_reqs.end(), preempted_req);
        if (sit != scheduled_running_reqs.end()) {
          const std::string& preempted_id = preempted_req->request_id;
          scheduled_running_reqs.erase(sit);
          token_budget += num_scheduled_tokens[preempted_id];
          num_scheduled_tokens.erase(preempted_id);
          req_to_new_blocks.erase(preempted_id);
          // Restore the encoder compute budget the victim's already-scheduled
          // encoder inputs spent, and drop them from this step
          // (scheduler.py:590-600). (The spec-decode undo is deferred at T0.)
          const auto enc_it = scheduled_encoder_inputs.find(preempted_id);
          if (enc_it != scheduled_encoder_inputs.end()) {
            for (const int i : enc_it->second) {
              encoder_compute_budget += static_cast<int>(
                  preempted_req->mm_features[static_cast<size_t>(i)].length);
            }
            scheduled_encoder_inputs.erase(enc_it);
          }
          req_index -= 1;
        }
      } else {
        // FCFS: preempt the tail (lowest scheduling priority = last arrival).
        preempted_req = running.back();
        running.pop_back();
      }

      preempt_request(preempted_req, scheduled_timestamp);
      preempted_reqs.push_back(preempted_req);
      if (preempted_req == request) {
        // No more requests to preempt; cannot schedule this request.
        break;
      }
    }

    if (!new_blocks.has_value()) {
      // Cannot schedule this request.
      break;
    }

    // Schedule the request.
    scheduled_running_reqs.push_back(request);
    const std::string request_id = request->request_id;
    req_to_new_blocks[request_id] = *new_blocks;
    num_scheduled_tokens[request_id] = num_new_tokens;
    // Progress `begin`, while num_computed_tokens is still the PRE-step count
    // (update_after_schedule advances it below). Inert during decode.
    PrefillMarkScheduled(*request, num_new_tokens);
    token_budget -= num_new_tokens;
    req_index += 1;

    // Speculative decode related (scheduler.py:593-609). If the request carries
    // pending draft tokens, record how many of them were actually scheduled this
    // step (num_new_tokens beyond the request's own sampled-token target is the
    // draft-verification budget) and clear them off the request — the drafter
    // will re-propose in update_draft_token_ids before the next step. Inert when
    // spec_token_ids is empty (default path), so nothing is recorded and the map
    // stays empty.
    if (!request->spec_token_ids.empty()) {
      const int num_scheduled_spec_tokens =
          num_new_tokens + request->num_computed_tokens - request->NumTokens() -
          request->num_output_placeholders;
      static const bool spec_sched_trace = std::getenv("VT_SPEC_TRACE") != nullptr;
      if (spec_sched_trace) {
        std::fprintf(stderr,
                     "[spec-sched] req=%s have=%zu num_new=%d computed=%d "
                     "NumTokens=%d placeholders=%d -> sched=%d\n",
                     request_id.c_str(), request->spec_token_ids.size(),
                     num_new_tokens, request->num_computed_tokens,
                     request->NumTokens(), request->num_output_placeholders,
                     num_scheduled_spec_tokens);
      }
      if (num_scheduled_spec_tokens > 0) {
        std::vector<int32_t> spec_ids = request->spec_token_ids;
        // Chunked-prefill / budget clamping may fit only a prefix of the drafts.
        if (static_cast<int>(spec_ids.size()) > num_scheduled_spec_tokens) {
          spec_ids.resize(static_cast<std::size_t>(num_scheduled_spec_tokens));
        }
        scheduled_spec_decode_tokens[request_id] = std::move(spec_ids);
      }
      // New spec tokens will be set in update_draft_token_ids before the next
      // step when applicable.
      request->spec_token_ids.clear();
    }

    // Encoder-related (scheduler.py:642-650). Reserve cache space for every
    // encoder input this step will run, and only NOW commit the budget the
    // decision above computed.
    if (!encoder_inputs_to_schedule.empty()) {
      for (const int i : encoder_inputs_to_schedule) {
        const multimodal::MultiModalFeatureSpec& feature =
            request->mm_features[static_cast<size_t>(i)];
        encoder_cache_manager->Allocate(request_id, i, feature.mm_hash,
                                        feature.length);
      }
      scheduled_encoder_inputs[request_id] = std::move(encoder_inputs_to_schedule);
      encoder_compute_budget = new_encoder_budget;
    }
  }

  // Next, schedule the WAITING requests (skipped entirely if any preemption
  // happened this step, matching upstream).
  // KV-OFFLOAD W4: requests deferred by the connector's "not ready, re-ask"
  // (nullopt) third state. Popped off waiting this step, re-queued to the front
  // after the loop so they are re-asked next step (scheduler.py:744-750,1017).
  std::vector<Request*> connector_skipped_waiting;
  if (preempted_reqs.empty()) {
    // ENG-SGLANG-BEHAVIOR-FLAG (SW1): cache-aware `lpm` admission ordering. A
    // no-op under fcfs/priority or with caching off (the byte-identical default
    // path). Reorders the waiting deque by longest cached-prefix match so the
    // admission loop below (unchanged) admits the highest-hit requests first.
    maybe_reorder_waiting_for_lpm();
    while (!waiting->empty() && token_budget > 0) {
      if (static_cast<int>(running.size()) >= max_num_running_reqs) {
        break;
      }

      Request* request = waiting->peek_request();
      const std::string request_id = request->request_id;

      // Get already-cached (prefix) tokens.
      KVCacheBlocks new_computed_blocks = kv_cache_manager->empty_kv_cache_blocks;
      int num_new_local_computed_tokens = 0;
      int num_computed_tokens = 0;
      if (request->num_computed_tokens == 0) {
        auto computed = kv_cache_manager->get_computed_blocks(*request);
        new_computed_blocks = computed.first;
        num_new_local_computed_tokens = computed.second;
        num_computed_tokens = num_new_local_computed_tokens;
      } else {
        // Resumed reqs with num_computed_tokens > 0 come only from the deferred
        // KV-transfer path; at T0 preempted reqs are reset to 0, so this branch
        // is inert (kept for structural fidelity).
        num_computed_tokens = request->num_computed_tokens;
      }

      // KV-OFFLOAD W4: ask the connector how many EXTERNAL (tier-cached) tokens
      // load beyond the local prefix, so a cross-request / restarted-process
      // prefix HIT shortcuts prefill (scheduler.py:737-762). Null connector =
      // zero change. The nullopt THIRD state ("a disk->CPU promotion is still in
      // flight") pops the request and re-asks next step; treating it as 0 would
      // spin or serve a partial prefix (§Risks R5).
      int num_external_computed_tokens = 0;
      if (kv_connector_ != nullptr && request->num_computed_tokens == 0) {
        const auto match = kv_connector_->get_num_new_matched_tokens(
            *request, num_new_local_computed_tokens);
        if (!match.num_matched_tokens.has_value()) {
          waiting->pop_request();
          connector_skipped_waiting.push_back(request);
          continue;
        }
        num_external_computed_tokens = *match.num_matched_tokens;
        // The external tokens are treated as computed (their KV is loaded before
        // compute by the worker), so prefill shrinks by exactly this many.
        num_computed_tokens += num_external_computed_tokens;
      }

      // Number of tokens to schedule (num_tokens covers resumed reqs' output).
      int num_new_tokens = request->NumTokens() - num_computed_tokens;
      if (0 < long_prefill_threshold &&
          long_prefill_threshold < num_new_tokens) {
        num_new_tokens = long_prefill_threshold;
      }
      // With chunked prefill disabled, we cannot split -> stop scheduling here.
      if (!enable_chunked_prefill_ && num_new_tokens > token_budget) {
        break;
      }
      num_new_tokens = std::min(num_new_tokens, token_budget);
      assert(num_new_tokens > 0);

      // Schedule encoder inputs (scheduler.py:862-876). The CLAMP is what keeps
      // a chunked prefill from covering placeholder rows whose encoder could not
      // run; `num_new_tokens == 0` means this request cannot be scheduled at all
      // this step and upstream BREAKS out of the waiting loop (not `continue` —
      // the waiting queue is FCFS here).
      std::vector<int> encoder_inputs_to_schedule;
      int new_encoder_budget = encoder_compute_budget;
      if (!request->mm_features.empty()) {
        encoder_inputs_to_schedule = try_schedule_encoder_inputs(
            *request, num_computed_tokens, &num_new_tokens, &new_encoder_budget);
        if (num_new_tokens == 0) {
          break;
        }
      }

      std::optional<KVCacheBlocks> new_blocks = kv_cache_manager->allocate_slots(
          *request, num_new_tokens, num_new_local_computed_tokens,
          new_computed_blocks, /*num_lookahead_tokens=*/num_lookahead_tokens_,
          num_external_computed_tokens, /*delay_cache_blocks=*/false,
          /*num_encoder_tokens=*/0,
          /*full_sequence_must_fit=*/scheduler_reserve_full_isl_,
          /*reserved_blocks=*/0, /*has_scheduled_reqs=*/!running.empty());
      if (!new_blocks.has_value()) {
        // The request cannot be scheduled. Untouch it from the encoder cache
        // manager: CheckAndUpdateCache above may have added a reference for a
        // request that is not going to run (scheduler.py:938-939).
        free_request_encoder_inputs(*request);
        break;
      }

      request = waiting->pop_request();
      running.push_back(request);
      // SCHEDULED event (scheduler.py:1003): this request was admitted to the
      // running batch this step (from WAITING or resumed from PREEMPTED). The
      // frontend keeps the FIRST such timestamp as scheduled_ts (later resume
      // SCHEDULED events are ignored), the boundary between the queued and
      // prefill intervals. Uses the shared step timestamp.
      if (log_stats_) {
        request->record_event(EngineCoreEventType::kScheduled,
                              scheduled_timestamp);
      }
      if (request->status == RequestStatus::kWaiting) {
        scheduled_new_reqs.push_back(request);
      } else if (request->status == RequestStatus::kPreempted) {
        scheduled_resumed_reqs.push_back(request);
      } else {
        throw std::runtime_error("Invalid request status in waiting loop");
      }

      req_to_new_blocks[request_id] = kv_cache_manager->get_blocks(request_id);
      // KV-OFFLOAD W4: register the load of the external prefix into the blocks
      // just allocated (load-before-compute). ext==0 makes this a no-op
      // (scheduler.py:932-937). The worker executes the load before those tokens
      // are read as computed.
      if (kv_connector_ != nullptr && num_external_computed_tokens > 0) {
        kv_connector_->update_state_after_alloc(
            *request, req_to_new_blocks[request_id].get_block_ids(),
            num_external_computed_tokens);
      }
      num_scheduled_tokens[request_id] = num_new_tokens;
      token_budget -= num_new_tokens;
      request->status = RequestStatus::kRunning;
      request->num_computed_tokens = num_computed_tokens;
      // Progress `begin` for a freshly admitted request. AFTER the assignment
      // above so `already_computed` reports the prefix-cache hit, and still
      // before update_after_schedule adds this step's tokens.
      PrefillMarkScheduled(*request, num_new_tokens);

      // Encoder-related (scheduler.py:1030-1038).
      if (!encoder_inputs_to_schedule.empty()) {
        for (const int i : encoder_inputs_to_schedule) {
          const multimodal::MultiModalFeatureSpec& feature =
              request->mm_features[static_cast<size_t>(i)];
          encoder_cache_manager->Allocate(request_id, i, feature.mm_hash,
                                          feature.length);
        }
        scheduled_encoder_inputs[request_id] =
            std::move(encoder_inputs_to_schedule);
        encoder_compute_budget = new_encoder_budget;
      }
    }
    // KV-OFFLOAD W4: re-queue the connector-deferred requests to the FRONT of
    // waiting (reverse so FCFS order is preserved) to be re-asked next step.
    for (auto it = connector_skipped_waiting.rbegin();
         it != connector_skipped_waiting.rend(); ++it) {
      waiting->prepend_request(*it);
    }
  }

  // Check the scheduling constraints are satisfied.
  int total_num_scheduled_tokens = 0;
  for (const auto& [id, n] : num_scheduled_tokens) {
    total_num_scheduled_tokens += n;
  }
  assert(total_num_scheduled_tokens <= max_num_scheduled_tokens);
  assert(token_budget >= 0);
  assert(static_cast<int>(running.size()) <= max_num_running_reqs);

  // Longest common prefix among running requests (for cascade attention).
  std::vector<int> num_common_prefix_blocks(kv_cache_config_.kv_cache_groups.size(),
                                            0);
  if (!running.empty()) {
    num_common_prefix_blocks =
        kv_cache_manager->get_num_common_prefix_blocks(running[0]->request_id);
  }

  // Construct the scheduler output (MRV2 / V2 model-runner path — upstream
  // scheduler.py:1047-1080 `if self.use_v2_model_runner:` branch). Resumed
  // (PREEMPTED->RUNNING this step) requests fold into scheduled_new_reqs and
  // are re-sent as FULL new reqs carrying prefill_token_ids, NOT through the
  // cached diff: the V2 gpu runner (M1.5) asserts every new req has
  // prefill_token_ids and re-seeds its per-request all_token_ids from it. The
  // MRV1 path (resumed-as-cached + CachedRequestData.all_token_ids) is NOT
  // ported (the V2 runner never reads CachedRequestData.all_token_ids).
  scheduled_new_reqs.insert(scheduled_new_reqs.end(),
                            scheduled_resumed_reqs.begin(),
                            scheduled_resumed_reqs.end());
  scheduled_resumed_reqs.clear();

  std::vector<NewRequestData> new_reqs_data;
  new_reqs_data.reserve(scheduled_new_reqs.size());
  for (Request* req : scheduled_new_reqs) {
    // prefill_token_ids = the request's full token ids (prompt+output; at
    // prefill this is the prompt). Upstream passes req._all_token_ids.
    new_reqs_data.push_back(NewRequestData::from_request(
        *req, req_to_new_blocks.at(req->request_id).get_block_ids(),
        req->AllTokenIds()));
  }

  // The cached diff now covers only the already-running reqs; scheduled_resumed_
  // reqs was cleared above (kept in the signature for upstream fidelity — the V2
  // branch still passes it, now empty).
  CachedRequestData cached_reqs_data = make_cached_request_data(
      scheduled_running_reqs, scheduled_resumed_reqs, num_scheduled_tokens,
      req_to_new_blocks);

  SchedulerOutput scheduler_output;
  scheduler_output.scheduled_new_reqs = std::move(new_reqs_data);
  scheduler_output.scheduled_cached_reqs = std::move(cached_reqs_data);
  // MOVE the num_scheduled_tokens map (dead after this) and the finished_req_ids
  // set (member, re-assigned fresh in update_after_schedule) instead of COPYING
  // them — vLLM passes the dicts by reference (rescan §6 item e, container
  // plumbing only, zero policy change). The local map is not read after this;
  // update_after_schedule reads scheduler_output.num_scheduled_tokens (the moved
  // destination) and clears the finished_req_ids member (`= {}`) so its
  // moved-from state is unobservable.
  scheduler_output.num_scheduled_tokens = std::move(num_scheduled_tokens);
  scheduler_output.total_num_scheduled_tokens = total_num_scheduled_tokens;
  // scheduled_spec_decode_tokens (scheduler.py:593-609): populated above only for
  // running requests carrying drafts; empty on the default (no-speculator) path.
  scheduler_output.scheduled_encoder_inputs = std::move(scheduled_encoder_inputs);
  scheduler_output.scheduled_spec_decode_tokens =
      std::move(scheduled_spec_decode_tokens);
  scheduler_output.num_common_prefix_blocks = std::move(num_common_prefix_blocks);
  // finished_req_ids is EXISTING scheduler state (finished between the prior and
  // current step) — moved out here, then flushed in _update_after_schedule.
  scheduler_output.finished_req_ids = std::move(finished_req_ids);
  // free_encoder_mm_hashes (scheduler.py:1084 mirror): the mm_hash entries the
  // encoder cache actually EVICTED since the previous step. The worker drops
  // exactly these from its own mm_hash-keyed cache, so the two caches cannot
  // disagree about what is resident. Empty on every text step.
  scheduler_output.free_encoder_mm_hashes =
      encoder_cache_manager->GetFreedMmHashes();

  // num_spec_tokens_to_schedule (scheduler.py:1123-1156, W7 #1824): the count
  // the AsyncScheduler's update_after_schedule below turns into -1 placeholder
  // drafts for the NEXT step. Flat num_spec_tokens (dynamic-SD deferred); 0 on
  // the no-speculator default.
  scheduler_output.num_spec_tokens_to_schedule = num_spec_tokens_;

  update_after_schedule(scheduler_output);

  // Fold this step's prefix-cache lookups into the sliding-window hit-rate
  // aggregate. Upstream does this in the FRONTEND (SchedulerStats ->
  // LoggingStatLogger.observe, vllm/v1/metrics/loggers.py); we have no
  // SchedulerStats plumbing, so the aggregation lives on the Scheduler.
  // Behaviourally identical: make_prefix_cache_stats() is a take-and-swap of
  // the SAME per-step delta upstream ships to the logger, and CachingMetrics
  // ignores empty observations, so idle steps are free and cannot slide useful
  // history out of the window.
  if (auto stats = kv_cache_manager->make_prefix_cache_stats()) {
    prefix_cache_metrics_.observe(*stats);
    // Stash the SAME per-step delta so make_stats() can hand it to the
    // Prometheus logger without a second take-and-swap (which would return an
    // empty observation). scheduler.py builds SchedulerStats from this delta.
    last_prefix_cache_stats_ = *stats;
  } else {
    // No observation this step (prefix caching off): make_stats() must not
    // re-report the prior step's delta.
    last_prefix_cache_stats_ = PrefixCacheStats{};
  }

  // KV-OFFLOAD W4: build_connector_meta drains + RESETS the connector's per-step
  // batch state (base.py:516-517); on_schedule_end flushes deferred disk->CPU
  // promotions and polls transfers (tiering manager on_schedule_end). Null
  // connector = no-op. In a live engine the worker consumes the returned load
  // jobs; W4 wires the scheduler seam (default-off, provably inert), and the
  // load execution is exercised end-to-end in the connector correctness harness.
  if (kv_connector_ != nullptr) {
    kv_connector_->build_connector_meta();
    kv_connector_->on_schedule_end();
  }
  return scheduler_output;
}

std::optional<GrammarOutput> Scheduler::get_grammar_bitmask(
    const SchedulerOutput& scheduler_output) {
  // scheduler.py:1477-1499.
  // Collect the scheduled request ids that use structured output. The
  // corresponding rows of the bitmask will be in this order.
  if (!scheduler_output.has_structured_output_requests) {
    return std::nullopt;
  }
  // Without a manager wired, structured output is a no-op (backward-compat with
  // the M1.4/M1.8 tests). Upstream always has a manager.
  if (structured_output_manager_ == nullptr) {
    return std::nullopt;
  }

  std::vector<std::string> structured_output_request_ids;
  for (const auto& [req_id, num_tokens] : scheduler_output.num_scheduled_tokens) {
    (void)num_tokens;
    auto it = requests.find(req_id);
    if (it != requests.end() && it->second->use_structured_output() &&
        !it->second->is_prefill_chunk) {
      structured_output_request_ids.push_back(req_id);
    }
  }
  if (structured_output_request_ids.empty()) {
    return std::nullopt;
  }

  std::optional<TokenBitmask> bitmask = structured_output_manager_->grammar_bitmask(
      requests, structured_output_request_ids,
      scheduler_output.scheduled_spec_decode_tokens);
  if (!bitmask.has_value()) {
    return std::nullopt;
  }

  GrammarOutput out;
  out.structured_output_request_ids = std::move(structured_output_request_ids);
  out.grammar_bitmask = std::move(*bitmask);
  return out;
}

EngineCoreOutputs Scheduler::update_from_output(
    const SchedulerOutput& scheduler_output,
    const ModelRunnerOutput& model_runner_output) {
  const std::vector<std::vector<int32_t>>& sampled_token_ids =
      model_runner_output.sampled_token_ids;
  const std::map<std::string, int>& num_scheduled_tokens =
      scheduler_output.num_scheduled_tokens;

  std::vector<EngineCoreOutput> outputs;
  // Requests that stopped this step, split by the queue they must be removed
  // from (upstream stopped_running_reqs / stopped_preempted_reqs). The KV blocks
  // are freed and finished_req_ids updated inside the loop, but the owning
  // requests-map erase is deferred until after these pointers are used to filter
  // running/waiting (so the Request* stays valid — upstream relies on Python GC).
  std::set<Request*> stopped_running_reqs;
  std::set<Request*> stopped_preempted_reqs;
  std::vector<std::string> finished_ids_to_erase;

  // scheduler.py:1828 `spec_decoding_stats: SpecDecodingStats | None = None`.
  // Lazily constructed by the first request that verified a draft this step and
  // stashed for make_stats() at the end (#2770). Stays empty on every
  // non-speculative step, which is the polarity SchedulerStats carries.
  std::optional<spec_decode::SpecDecodingStats> spec_decoding_stats;

  // NOTE(woosuk): upstream iterates num_scheduled_tokens.items() (dict/schedule
  // order); std::map iterates in sorted key order. The set of outputs is the
  // same — only their order in the returned vector differs, which is benign
  // (each EngineCoreOutput is keyed by request_id).
  for (const auto& [req_id, num_tokens_scheduled] : num_scheduled_tokens) {
    assert(num_tokens_scheduled > 0);
    (void)num_tokens_scheduled;

    auto it = requests.find(req_id);
    if (it == requests.end() || it->second->IsFinished()) {
      // Already finished — e.g. aborted while the model was executing it.
      continue;
    }
    Request* request = it->second.get();

    const int req_index = model_runner_output.req_id_to_index.at(req_id);
    // sampled_token_ids[req_index] if sampled_token_ids else []. A request still
    // being prefilled gets an empty list from the runner.
    std::vector<int32_t> new_token_ids =
        sampled_token_ids.empty()
            ? std::vector<int32_t>{}
            : sampled_token_ids[static_cast<std::size_t>(req_index)];

    // Speculative-decode acceptance / num_computed rollback (scheduler.py:1580-
    // 1612). If this step verified draft tokens for the request, the runner
    // returned 1 + num_accepted tokens (the sampled/bonus token plus every
    // accepted draft); num_rejected of the scheduled drafts were computed but not
    // kept, so num_computed_tokens (and, under async scheduling, the matching
    // num_output_placeholders) is rewound by that many. Inert when the request
    // had no scheduled drafts (the map lookup misses -> whole block skipped), so
    // the default path is byte-identical.
    // W7 (#1824): skip the whole block while the request is draining stale
    // in-flight frames (scheduler.py:1670-1675 `async_tokens_to_discard == 0`):
    // a discarded frame's pre-reset rejection count would underflow both
    // counters. 0 on the synchronous path and whenever no force-preemption is
    // in flight, so the existing arms are unchanged.
    auto spec_it = scheduler_output.scheduled_spec_decode_tokens.find(req_id);
    if (spec_it != scheduler_output.scheduled_spec_decode_tokens.end() &&
        (!new_token_ids.empty() || num_sampled_tokens_per_step_ == 0) &&
        request->async_tokens_to_discard == 0) {
      const int num_draft_tokens = static_cast<int>(spec_it->second.size());
      // num_accepted = generated - num_sampled, floored at 0 so an empty
      // (aborted / error) output does not underflow (regression: upstream
      // test_spec_decoding_stats_empty_output).
      const int num_accepted = std::max(
          static_cast<int>(new_token_ids.size()) - num_sampled_tokens_per_step_,
          0);
      const int num_rejected = num_draft_tokens - num_accepted;
      // num_computed_tokens counts tokens processed this step including the
      // scheduled drafts; a rejection rewinds it so the next step recomputes from
      // the last accepted position (the paged KV slots are simply overwritten;
      // GDN linear-state rollback is the §4 kernel path, not here).
      if (request->num_computed_tokens > 0) {
        request->num_computed_tokens -= num_rejected;
      }
      // Under async scheduling num_output_placeholders also reserved the
      // scheduled draft count, so it is rewound identically (0 -> inert on the
      // synchronous path).
      if (request->num_output_placeholders > 0) {
        request->num_output_placeholders -= num_rejected;
      }
      // scheduler.py:1918-1924 make_spec_decoding_stats: the SAME num_draft_tokens
      // and num_accepted the rollback above just computed, folded into this
      // step's aggregate. This is the production call site the /metrics
      // spec_decode families are fed from (#2770); delete it and the exposition
      // never moves.
      spec_decoding_stats = make_spec_decoding_stats(
          std::move(spec_decoding_stats), num_draft_tokens, num_accepted,
          scheduler_output.num_invalid_spec_tokens, req_id);
    }
    // Free encoder inputs only AFTER the step has actually executed
    // (scheduler.py:1699-1701). Freeing at schedule time would drop an entry the
    // forward is still about to gather.
    free_encoder_inputs(*request);

    bool stopped = false;
    const RequestStatus status_before_stop = request->status;

    // _update_request_with_output (scheduler.py:1627-1630): append + check_stop +
    // trim. Called only when the runner produced tokens (upstream `if
    // new_token_ids:`). The virtual hook lets AsyncScheduler wrap it (stale-frame
    // discard + placeholder drain + block caching).
    if (!new_token_ids.empty()) {
      auto result =
          update_request_with_output(*request, std::move(new_token_ids));
      new_token_ids = std::move(result.first);
      stopped = result.second;
    }
    // Pooling stop (ARCH-ONE-SURFACE ROW 6; scheduler.py:1718-1721 `elif
    // request.pooling_params and pooler_output is not None`): a POOLING request
    // finishes as soon as the runner produced its pooled output. The runner
    // reports nullopt for a row still consuming prefill chunks (the
    // is_valid=false rows, pooling_runner.py:40-41), so such a request keeps
    // running. pooler_output is EMPTY on every generation step -> the text path
    // above is byte-identical.
    std::optional<std::vector<float>> pooler_output;
    if (!model_runner_output.pooler_output.empty() &&
        req_index < static_cast<int>(model_runner_output.pooler_output.size())) {
      pooler_output =
          model_runner_output.pooler_output[static_cast<std::size_t>(req_index)];
    }
    if (new_token_ids.empty() && request->pooling_params.has_value() &&
        pooler_output.has_value()) {
      request->status = RequestStatus::kFinishedStopped;
      stopped = true;
    }

    // scheduler.py:1636-1651: advance the structured-output FSM by the sampled
    // tokens. Only when the request produced tokens and the manager says the FSM
    // should advance (should_advance is true for any structured request at T0).
    // A grammar that rejects the sampled tokens is unexpected: terminate the
    // request with FINISHED_ERROR. No-op when no manager is wired.
    if (!new_token_ids.empty() && structured_output_manager_ != nullptr &&
        structured_output_manager_->should_advance(*request)) {
      auto& struct_output_request = request->structured_output_request;
      assert(struct_output_request.has_value());
      assert(struct_output_request->grammar != nullptr);
      if (!struct_output_request->grammar->accept_tokens(req_id, new_token_ids)) {
        request->status = RequestStatus::kFinishedError;
        // (upstream also sets request.resumable = false; resumable is deferred.)
        stopped = true;
      }
    }

    std::optional<FinishReason> finish_reason;
    if (stopped) {
      // Capture the finish reason before freeing (upstream captures it before
      // _handle_stopped_request, which may reset the status for resumable reqs —
      // resumable/streaming is deferred, so _handle_stopped_request is always
      // "finished" at T0).
      finish_reason = request->GetFinishedReason();
      // _free_request + _free_blocks (T0 subset): free the KV blocks and record
      // the finished id now; defer the requests-map erase (see above).
      kv_cache_manager->free(*request);
      free_request_encoder_inputs(*request);
      finished_req_ids.insert(request->request_id);
      finished_ids_to_erase.push_back(request->request_id);
      if (status_before_stop == RequestStatus::kRunning) {
        stopped_running_reqs.insert(request);
      } else {
        stopped_preempted_reqs.insert(request);
      }
    }

    // Extract sample logprobs if needed (scheduler.py:1815-1821). Only when the
    // request asked for logprobs and the runner produced them this step; slice
    // this request's rows out of the batch-wide LogprobsLists. The gate is
    // upstream's `num_logprobs` PROPERTY (:1818), not the raw `logprobs` field,
    // so a generative-scoring request — which sets logprob_token_ids and leaves
    // `logprobs` unset — is sliced too.
    std::optional<LogprobsTensors> new_logprobs;
    if (request->sampling_params.num_logprobs().has_value() &&
        model_runner_output.logprobs.has_value() &&
        model_runner_output.logprobs->num_positions > 0) {
      new_logprobs = model_runner_output.logprobs->slice_request(
          req_index, static_cast<int>(new_token_ids.size()));
    }
    // Get prompt logprobs for this request (scheduler.py:1826-1827). Populated
    // only during prefill for a prompt_logprobs request (runner source pending).
    std::optional<LogprobsTensors> new_prompt_logprobs_tensors;
    {
      auto plp_it = model_runner_output.prompt_logprobs_dict.find(req_id);
      if (plp_it != model_runner_output.prompt_logprobs_dict.end()) {
        new_prompt_logprobs_tensors = plp_it->second;
      }
    }
    // num_nans_in_logits: deferred.

    // Emit an EngineCoreOutput only when the request produced tokens or finished
    // (upstream's `if new_token_ids or ... or stopped`). A partial-prefill
    // request that produced neither is skipped: "EngineCore returns no partial
    // prefill outputs".
    if (!new_token_ids.empty() || pooler_output.has_value() || stopped) {
      EngineCoreOutput out;
      out.request_id = req_id;
      out.new_token_ids = new_token_ids;
      out.finish_reason = finish_reason;
      // Pooled data rides the output to the frontend (scheduler.py:1837
      // `pooling_output=pooler_output`); nullopt on every generation output.
      out.pooling_output = std::move(pooler_output);
      out.new_logprobs = std::move(new_logprobs);
      out.new_prompt_logprobs_tensors = std::move(new_prompt_logprobs_tensors);
      // stop_reason is int|str|None upstream; our EngineCoreOutput carries an
      // optional<string> (see engine/types.h). Only a stop_token_ids match sets
      // request.stop_reason at T0 — stringify that token id; otherwise nullopt.
      if (request->stop_reason.has_value()) {
        out.stop_reason = std::to_string(*request->stop_reason);
      }
      // events=request.take_events() (scheduler.py:1839): drain the request's
      // accumulated QUEUED/SCHEDULED/PREEMPTED events onto this output and clear
      // them. nullopt (omit-default) when log_stats is off or none fired.
      out.events = request->take_events();
      outputs.push_back(std::move(out));
    }
  }

  // Remove the stopped requests from the running list and the waiting queue.
  if (!stopped_running_reqs.empty()) {
    running.erase(
        std::remove_if(running.begin(), running.end(),
                       [&](Request* r) {
                         return stopped_running_reqs.count(r) > 0;
                       }),
        running.end());
  }
  if (!stopped_preempted_reqs.empty()) {
    // Rare (a stopped-while-preempted request); remove each from waiting.
    std::vector<Request*> to_remove(stopped_preempted_reqs.begin(),
                                    stopped_preempted_reqs.end());
    waiting->remove_requests(to_remove);
  }
  // Now that no queue references them, drop the owning entries (destroys the
  // finished Request objects — upstream _free_blocks' `del self.requests[...]`).
  for (const std::string& id : finished_ids_to_erase) {
    requests.erase(id);
  }

  // Collect and publish this step's KV-cache events (scheduler.py:1901-1915).
  // This is the LAST thing before the outputs are assembled, exactly as
  // upstream: eviction events raised by allocate_slots during schedule() and
  // store events raised by cache_blocks at the end of the step must land in the
  // SAME batch, so the drain belongs at the end of the step, not in schedule().
  //
  // With events disabled (the default) take_events() returns the pool's empty
  // queue and the `if` below never fires, so this is inert.
  //
  // DEFERRED: upstream also merges self.connector.take_events()
  // (scheduler.py:1903-1910) into the batch. Our kv_offload::KVConnector has no
  // KVCacheEvent-producing take_events -- the nearest thing,
  // OffloadingManager::take_events, yields OffloadingEvent, a different type
  // owned by KV-OFFLOAD. Bridging them is that row's work; coercing one into the
  // other here would hide a real gap behind a plausible-looking merge.
  std::vector<KVCacheEvent> kv_events = kv_cache_manager->take_events();
  if (!kv_events.empty()) {
    distributed::KVEventBatch batch;
    // time.time(): wall-clock seconds since the epoch (NOT a steady clock) --
    // a consumer correlates the batch against its own wall clock.
    batch.ts = std::chrono::duration<double>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
    batch.events = std::move(kv_events);
    kv_event_publisher_->publish(batch);
  }

  // scheduler.py:2211 hands spec_decoding_stats to make_stats in the same call.
  // EngineCore calls our make_stats() AFTER this function returns, so the value
  // is stashed instead — the identical arrangement schedule() already uses for
  // the prefix-cache delta. Unconditional, so a step that verified no draft
  // CLEARS the stash and cannot re-report the previous step's aggregate.
  last_spec_decoding_stats_ = std::move(spec_decoding_stats);

  EngineCoreOutputs engine_core_outputs;
  engine_core_outputs.outputs = std::move(outputs);
  return engine_core_outputs;
}

std::optional<spec_decode::SpecDecodingStats> Scheduler::make_spec_decoding_stats(
    std::optional<spec_decode::SpecDecodingStats> spec_decoding_stats,
    int num_draft_tokens, int num_accepted_tokens,
    const std::map<std::string, int>& num_invalid_spec_tokens,
    const std::string& request_id) const {
  // scheduler.py:2722-2723 `if not self.log_stats or not num_draft_tokens:
  // return None`. Returning EMPTY rather than the accumulated value is upstream's
  // own shape, and it is load-bearing: a step whose only drafted request was
  // dropped reports nothing at all instead of a zero-valued draft.
  if (!log_stats_ || num_draft_tokens <= 0) {
    return std::nullopt;
  }
  if (!spec_decoding_stats.has_value()) {
    // scheduler.py:2724-2725 SpecDecodingStats.new(self.num_spec_tokens).
    spec_decoding_stats = spec_decode::SpecDecodingStats::New(num_spec_tokens_);
  }
  // scheduler.py:2726-2727: a draft position the drafter could not fill was
  // scheduled as a -1 placeholder and must not count as drafted.
  const auto invalid = num_invalid_spec_tokens.find(request_id);
  if (invalid != num_invalid_spec_tokens.end()) {
    num_draft_tokens -= invalid->second;
  }
  // scheduler.py:2728-2730 observe_draft.
  spec_decoding_stats->ObserveDraft(num_draft_tokens, num_accepted_tokens);
  return spec_decoding_stats;
}

CachedRequestData Scheduler::make_cached_request_data(
    const std::vector<Request*>& running_reqs,
    const std::vector<Request*>& resumed_reqs,
    const std::map<std::string, int>& num_scheduled_tokens,
    const std::map<std::string, KVCacheBlocks>& req_to_new_blocks) {
  (void)num_scheduled_tokens;  // Used only by the deferred PP token_ids path.
  CachedRequestData data;
  const std::size_t num_running_reqs = running_reqs.size();

  std::size_t idx = 0;
  auto handle = [&](Request* req) {
    const std::string& req_id = req->request_id;
    data.req_ids.push_back(req_id);
    // NOTE: use_pp is false at T0, so the PP new_token_ids payload is omitted
    // (the model runner caches the sampled tokens itself).
    if (idx >= num_running_reqs) {
      data.resumed_req_ids.insert(req_id);
    }
    // MRV2 (V2 model runner): CachedRequestData.all_token_ids is never read by
    // the runner, so the MRV1 prev-step-gated all_token_ids propagation is
    // dropped (the field stays empty). Resumed reqs no longer flow here at all
    // (folded into scheduled_new_reqs with prefill_token_ids upstream).
    data.new_block_ids.push_back(
        req_to_new_blocks.at(req_id).get_block_ids(/*allow_none=*/true));
    data.num_computed_tokens.push_back(req->num_computed_tokens);
    // num_output_placeholders == 0 at T0.
    data.num_output_tokens.push_back(req->NumOutputTokens());
    idx += 1;
  };

  for (Request* req : running_reqs) {
    handle(req);
  }
  for (Request* req : resumed_reqs) {
    handle(req);
  }
  return data;
}

std::pair<std::vector<int32_t>, bool> Scheduler::update_request_with_output(
    Request& request, std::vector<int32_t> new_token_ids) {
  // scheduler.py:1886-1902: append each generated token, run check_stop after
  // each, and trim any tokens generated past the stop.
  bool stopped = false;
  for (std::size_t num_new = 1; num_new <= new_token_ids.size(); ++num_new) {
    request.AppendOutputToken(new_token_ids[num_new - 1]);
    stopped = check_stop(request, max_model_len);
    if (stopped) {
      new_token_ids.resize(num_new);  // del new_token_ids[num_new:]
      break;
    }
  }
  return {std::move(new_token_ids), stopped};
}

void Scheduler::update_after_schedule(SchedulerOutput& scheduler_output) {
  // Advance num_computed_tokens AFTER building the output so a chunked prefill
  // resumes on the next step; refresh is_prefill_chunk.
  for (const auto& [req_id, num_scheduled_token] :
       scheduler_output.num_scheduled_tokens) {
    Request* request = requests.at(req_id).get();
    request->num_computed_tokens += num_scheduled_token;
    // Under async scheduling num_output_placeholders extends the target token
    // count (the in-flight sampled token(s) not yet appended); 0 for the
    // synchronous Scheduler, so this is unchanged there.
    request->is_prefill_chunk =
        request->num_computed_tokens <
        request->NumTokens() + request->num_output_placeholders;
    // scheduler.py:1186: a structured request that is past its prefill chunk
    // needs a grammar bitmask this step.
    scheduler_output.has_structured_output_requests |=
        request->use_structured_output() && !request->is_prefill_chunk;
  }
  // Flush the finished / preempted id sets (assign fresh sets so the already
  // copied-out scheduler_output is unaffected).
  finished_req_ids = {};
  reset_preempted_req_ids = {};
}

void Scheduler::LogPrefillAfterExecute(const SchedulerOutput& scheduler_output) {
  PrefillLogAfterExecute(scheduler_output, requests);
}

void Scheduler::update_draft_token_ids(const DraftTokenIds& draft_token_ids) {
  // scheduler.py:1937-1957. Install the drafter's freshly-proposed spec tokens
  // onto their requests for the NEXT verify step.
  const std::size_t n =
      std::min(draft_token_ids.req_ids.size(),
               draft_token_ids.draft_token_ids.size());
  static const bool spec_entry_trace = std::getenv("VT_SPEC_TRACE") != nullptr;
  if (spec_entry_trace) {
    std::fprintf(stderr, "[spec-update] called n=%zu\n", n);
  }
  for (std::size_t i = 0; i < n; ++i) {
    const std::string& req_id = draft_token_ids.req_ids[i];
    auto it = requests.find(req_id);
    if (spec_entry_trace) {
      std::fprintf(stderr,
                   "[spec-update] req=%s found=%d finished=%d prefill_chunk=%d "
                   "drafts=%zu\n",
                   req_id.c_str(), it != requests.end() ? 1 : 0,
                   (it != requests.end() && it->second->IsFinished()) ? 1 : 0,
                   (it != requests.end() && it->second->is_prefill_chunk) ? 1 : 0,
                   draft_token_ids.draft_token_ids[i].size());
    }
    if (it == requests.end() || it->second->IsFinished()) {
      // The request may have been finished. Skip.
      continue;
    }
    Request* request = it->second.get();

    if (request->is_prefill_chunk) {
      // Ignore draft tokens for prefill chunks (a request mid-prefill must not
      // schedule drafts alongside its remaining prompt — scheduler.py:1948-1951).
      if (!request->spec_token_ids.empty()) {
        request->spec_token_ids.clear();
      }
      continue;
    }

    request->spec_token_ids = draft_token_ids.draft_token_ids[i];
    if (structured_output_manager_ != nullptr &&
        structured_output_manager_->should_advance(*request)) {
      request->spec_token_ids = request->structured_output_request->grammar
                                   ->validate_tokens(request->spec_token_ids);
    }
    static const bool spec_trace = std::getenv("VT_SPEC_TRACE") != nullptr;
    if (spec_trace) {
      std::fprintf(stderr, "[spec-install] req=%s installed=%zu lookahead=%d\n",
                   req_id.c_str(), request->spec_token_ids.size(),
                   num_lookahead_tokens_);
    }
  }
}

void Scheduler::update_draft_token_ids_in_output(
    const DraftTokenIds& draft_token_ids, SchedulerOutput& scheduler_output) {
  // scheduler.py:2072-2107 (SPEC-DFLASH2 W7, #1824). The async draft-in-output
  // variant: under async scheduling the request state carries only -1
  // placeholders, so the drafts are rewritten INTO the SchedulerOutput the
  // deferred (structured-output) sampling path is about to consume. The
  // Validate the draft prefix before padding back to the scheduled row count.
  std::map<std::string, int> num_invalid_spec_tokens;
  std::map<std::string, std::vector<int32_t>>& sched_spec_tokens =
      scheduler_output.scheduled_spec_decode_tokens;
  const std::size_t n = std::min(draft_token_ids.req_ids.size(),
                                 draft_token_ids.draft_token_ids.size());
  for (std::size_t i = 0; i < n; ++i) {
    const std::string& req_id = draft_token_ids.req_ids[i];
    const auto req_it = requests.find(req_id);
    if (req_it == requests.end() || req_it->second->IsFinished()) {
      continue;  // the request may have been finished; skip (:2082-2085).
    }
    const auto sched_it = sched_spec_tokens.find(req_id);
    if (sched_it == sched_spec_tokens.end() || sched_it->second.empty()) {
      continue;  // nothing scheduled for it this step (:2087-2089).
    }
    const std::size_t orig_num_spec_tokens = sched_it->second.size();
    std::vector<int32_t> spec_token_ids = draft_token_ids.draft_token_ids[i];
    if (spec_token_ids.size() > orig_num_spec_tokens) {
      // Trim to the scheduled count (the chunked-prefill case, :2091-2094).
      spec_token_ids.resize(orig_num_spec_tokens);
    }
    if (structured_output_manager_ != nullptr &&
        structured_output_manager_->should_advance(*req_it->second)) {
      spec_token_ids = req_it->second->structured_output_request->grammar
                           ->validate_tokens(spec_token_ids);
    }
    if (spec_token_ids.size() < orig_num_spec_tokens) {
      // Pad back to the scheduled count with -1 and record the invalid tail;
      // the grammar bitmask computation skips the -1 slots (:2099-2103).
      const int num_invalid =
          static_cast<int>(orig_num_spec_tokens - spec_token_ids.size());
      spec_token_ids.resize(orig_num_spec_tokens, -1);
      num_invalid_spec_tokens[req_id] = num_invalid;
    }
    sched_it->second = std::move(spec_token_ids);
  }
  // REPLACE the whole map each call (upstream builds a fresh dict, :2075,2107).
  scheduler_output.num_invalid_spec_tokens = std::move(num_invalid_spec_tokens);
}

int Scheduler::get_num_unfinished_requests() const {
  return static_cast<int>(waiting->size()) + static_cast<int>(running.size());
}

std::pair<int, int> Scheduler::get_request_counts() const {
  return {static_cast<int>(running.size()),
          static_cast<int>(waiting->size())};
}

bool Scheduler::has_finished_requests() const {
  return !finished_req_ids.empty();
}

SchedulerStats Scheduler::make_stats() const {
  // scheduler.py:2399-2436. The always-on core: running/waiting counts, KV-cache
  // usage, and this step's prefix-cache token delta (stashed by schedule()). The
  // advanced SchedulerStats fields upstream carries (num_skipped_waiting_reqs,
  // connector/spec/cudagraph/perf stats, kv_cache_eviction_events) are deferred
  // with their config-gated metric families.
  SchedulerStats stats;
  stats.num_running_reqs = static_cast<int64_t>(running.size());
  stats.num_waiting_reqs = static_cast<int64_t>(waiting->size());
  stats.kv_cache_usage = kv_cache_manager->usage();
  stats.prefix_cache_stats = last_prefix_cache_stats_;
  // scheduler.py:2696,2708 `spec_stats = spec_decoding_stats` ->
  // SchedulerStats(spec_decoding_stats=spec_stats). Empty unless
  // update_from_output verified a draft this step (#2770).
  stats.spec_decoding_stats = last_spec_decoding_stats_;
  return stats;
}

}  // namespace vllm::v1
