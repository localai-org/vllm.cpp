// Ported from: vllm/v1/sample/sampler.py @ e24d1b24. See sampler.h for the full
// 9-step order + the deferred stubs. This file assembles the ordered pipeline by
// composing the Task 2/3 vt ops over the [num_reqs, vocab] f32 logits.
#include "vllm/v1/sample/sampler.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include "vllm/v1/sample/device_scratch.h"
#include "vllm/v1/sample/logits_processor/builtin.h"
#include "vllm/v1/sample/ops/bad_words.h"
#include "vllm/v1/sample/ops/penalties.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/sample_common.h"

namespace vllm::v1 {

namespace {

// Owns a device-side output buffer of the given shape and downloads it to host.
// Mirrors the DeviceScratch materialization (unified vs discrete) but for OUTPUT
// tensors the vt ops write into: on unified backends Alloc is host-addressable
// and Copy is memcpy, so download() works uniformly on CPU and CUDA.
class DeviceBuffer {
 public:
  DeviceBuffer(vt::Device device, vt::Queue& q, vt::DType dtype,
               std::initializer_list<int64_t> shape)
      : backend_(&vt::GetBackend(device.type)), q_(q) {
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dtype);
    owned_ = backend_->Alloc(bytes_ == 0 ? 1 : bytes_);
    tensor_ = vt::Tensor::Contiguous(owned_, dtype, device, shape);
  }
  ~DeviceBuffer() { backend_->Free(owned_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  vt::Tensor& tensor() { return tensor_; }
  void download(void* dst) {
    if (bytes_ != 0) backend_->Copy(q_, dst, owned_, bytes_);
    backend_->Synchronize(q_);
  }

 private:
  vt::Backend* backend_ = nullptr;
  vt::Queue& q_;
  void* owned_ = nullptr;
  size_t bytes_ = 0;
  vt::Tensor tensor_;
};

// Materialize the per-req top_k / top_p optionals (SamplingMetadata) into the
// [num_reqs] tensors ApplyTopKTopP consumes; pass nullptr when a predicate says
// none. The DeviceScratch objects are locals whose lifetime spans the op call.
void ApplyTopKTopPFromMeta(vt::Queue& q, vt::Tensor& logits,
                           const std::optional<std::vector<int32_t>>& top_k,
                           const std::optional<std::vector<float>>& top_p) {
  const int64_t n = logits.shape[0];
  if (top_k.has_value())
    VT_CHECK(static_cast<int64_t>(top_k->size()) == n,
             "sampler: top_k must have num_reqs rows");
  if (top_p.has_value())
    VT_CHECK(static_cast<int64_t>(top_p->size()) == n,
             "sampler: top_p must have num_reqs rows");

  if (top_k.has_value() && top_p.has_value()) {
    DeviceScratch k(logits.device, q, top_k->data(), vt::DType::kI32, {n});
    DeviceScratch p(logits.device, q, top_p->data(), vt::DType::kF32, {n});
    vt::ApplyTopKTopP(q, logits, &k.tensor(), &p.tensor());
  } else if (top_k.has_value()) {
    DeviceScratch k(logits.device, q, top_k->data(), vt::DType::kI32, {n});
    vt::ApplyTopKTopP(q, logits, &k.tensor(), nullptr);
  } else if (top_p.has_value()) {
    DeviceScratch p(logits.device, q, top_p->data(), vt::DType::kF32, {n});
    vt::ApplyTopKTopP(q, logits, nullptr, &p.tensor());
  }
}

// gather_logprobs (sampler.py::Sampler.gather_logprobs). Over the raw (pre-
// mutation) logprobs snapshot: top-k values+indices, the sampled token's
// logprob, ranks via batched_count_greater_than, concat [sampled | topk].
// batched_count_greater_than (ops/logprobs.py) is `(x >= values).sum(-1)`, so
// the rank COUNTS the sampled token itself (>=) — it is 1-BASED and the
// max-logprob token has rank 1.
LogprobsTensors GatherLogprobs(const std::vector<float>& raw_logprobs, int64_t n,
                               int64_t vocab, int num_logprobs,
                               const std::vector<int64_t>& sampled) {
  // #249 defense-in-depth: validation (InputProcessor::ValidateParams) rejects
  // k > vocab before we get here, mirroring upstream's _validate_logprobs. Clamp
  // anyway — partial_sort(idx.begin(), idx.begin() + k, ...) below indexes a
  // vocab-sized array, so an unvalidated caller walking past the end is a crash,
  // and no internal path should be able to reach that from a request field.
  const int k = static_cast<int>(std::min<int64_t>(num_logprobs, vocab));
  const int width = k + 1;
  LogprobsTensors lt;
  lt.num_positions = static_cast<int>(n);
  lt.num_tokens_per_position = width;
  lt.logprob_token_ids.resize(static_cast<size_t>(n) * static_cast<size_t>(width));
  lt.logprobs.resize(static_cast<size_t>(n) * static_cast<size_t>(width));
  lt.selected_token_ranks.resize(static_cast<size_t>(n));

  std::vector<int32_t> idx(static_cast<size_t>(vocab));
  for (int64_t i = 0; i < n; ++i) {
    const float* row = &raw_logprobs[static_cast<size_t>(i * vocab)];
    const int32_t tok = static_cast<int32_t>(sampled[static_cast<size_t>(i)]);
    const float tok_lp = row[tok];

    // Rank: count of logprobs >= the sampled token's logprob (1-based).
    int32_t rank = 0;
    for (int64_t j = 0; j < vocab; ++j)
      if (row[j] >= tok_lp) ++rank;
    lt.selected_token_ranks[static_cast<size_t>(i)] = rank;

    // Top-k by value desc, lowest index on ties (torch.topk sorted order).
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [row](int32_t a, int32_t b) {
                        if (row[a] != row[b]) return row[a] > row[b];
                        return a < b;
                      });

    const size_t base = static_cast<size_t>(i) * static_cast<size_t>(width);
    lt.logprob_token_ids[base] = tok;  // column 0: the sampled token
    lt.logprobs[base] = tok_lp;
    for (int c = 0; c < k; ++c) {
      lt.logprob_token_ids[base + 1 + static_cast<size_t>(c)] = idx[static_cast<size_t>(c)];
      lt.logprobs[base + 1 + static_cast<size_t>(c)] = row[idx[static_cast<size_t>(c)]];
    }
  }
  return lt;
}

// gather_specific_token_logprobs (sampler.py:151-225). Generative scoring:
// return the logprobs of an EXPLICIT set of vocab ids per request instead of a
// top-k. Upstream builds a padded [batch, max_num_tokens + 1] int64 id matrix
// with the sampled token in column 0 (always valid), each request's ids at
// columns 1..n, `torch.gather`s the logprobs at those ids, masks the padded
// positions to -inf, and computes the sampled token's rank with
// batched_count_greater_than over the FULL vocab — not over the requested
// subset, so a rank stays comparable across requests (:207-215).
//
// Rows absent from `token_ids` still get column 0 (their sampled token) and
// -inf padding everywhere else, exactly as upstream's zero-initialized matrix +
// `valid_mask[:, 0] = True` does.
//
// BOUNDS: torch.gather RAISES on an out-of-range index; the equivalent here
// would read past the row, so every id is checked into [0, vocab) and every key
// into [0, n) before it indexes anything. (This is the defect class of issue
// #249, whose instance — GatherLogprobs' unbounded `k` — is a separate row and
// is deliberately not touched here.)
LogprobsTensors GatherSpecificTokenLogprobs(
    const std::vector<float>& raw_logprobs, int64_t n, int64_t vocab,
    const std::map<int, std::vector<int32_t>>& token_ids,
    const std::vector<int64_t>& sampled) {
  // max_num_tokens = max(len(tids) ...) (:180).
  size_t max_num_tokens = 0;
  for (const auto& [req_idx, ids] : token_ids) {
    VT_CHECK(0 <= req_idx && static_cast<int64_t>(req_idx) < n,
             "sampler: logprob_token_ids key is not a request index");
    for (int32_t tid : ids) {
      VT_CHECK(0 <= tid && static_cast<int64_t>(tid) < vocab,
               "sampler: logprob_token_ids contains an out-of-vocab token id");
    }
    max_num_tokens = std::max(max_num_tokens, ids.size());
  }
  const size_t width = max_num_tokens + 1;  // + the sampled token in column 0

  LogprobsTensors lt;
  lt.num_positions = static_cast<int>(n);
  lt.num_tokens_per_position = static_cast<int>(width);
  lt.logprob_token_ids.assign(static_cast<size_t>(n) * width, 0);
  lt.logprobs.assign(static_cast<size_t>(n) * width,
                     -std::numeric_limits<float>::infinity());
  lt.selected_token_ranks.resize(static_cast<size_t>(n));

  for (int64_t i = 0; i < n; ++i) {
    const float* row = &raw_logprobs[static_cast<size_t>(i * vocab)];
    const size_t base = static_cast<size_t>(i) * width;

    // Column 0: the sampled token, valid for EVERY row (:190).
    const int32_t tok = static_cast<int32_t>(sampled[static_cast<size_t>(i)]);
    VT_CHECK(0 <= tok && static_cast<int64_t>(tok) < vocab,
             "sampler: sampled token id is out of vocab range");
    const float tok_lp = row[tok];
    lt.logprob_token_ids[base] = tok;
    lt.logprobs[base] = tok_lp;

    // Columns 1..n: this request's ids, in REQUEST order. The rest stay at the
    // -inf / id 0 the assign() above wrote (upstream's masked padding).
    const auto it = token_ids.find(static_cast<int>(i));
    if (it != token_ids.end()) {
      for (size_t c = 0; c < it->second.size(); ++c) {
        const int32_t tid = it->second[c];
        lt.logprob_token_ids[base + 1 + c] = tid;
        lt.logprobs[base + 1 + c] = row[tid];
      }
    }

    // Rank over the FULL vocab, 1-based (:213-215), the same
    // batched_count_greater_than GatherLogprobs uses.
    int32_t rank = 0;
    for (int64_t j = 0; j < vocab; ++j)
      if (row[j] >= tok_lp) ++rank;
    lt.selected_token_ranks[static_cast<size_t>(i)] = rank;
  }
  return lt;
}

}  // namespace

// PERSISTENT greedy-argmax scratch: a device buffer (the argmax ids) + a pinned
// host buffer (the D2H destination), both grow-only and reused across steps, so
// the greedy decode hot path pays NO per-step cudaMalloc/cudaFree/cudaHostAlloc
// (each of which device-syncs). Frees ONCE in the dtor. Mirrors vLLM's
// persistent sampled-id + pinned buffers (gpu_model_runner.py:873-878).
struct Sampler::GreedyArgmaxScratch {
  vt::Backend* backend = nullptr;
  vt::Device device{};
  void* device_ids = nullptr;  // capacity int64 elements on `device`
  int64_t* pinned_host = nullptr;
  int64_t capacity = 0;

  ~GreedyArgmaxScratch() {
    if (pinned_host != nullptr) backend->FreePinned(pinned_host);
    if (device_ids != nullptr) vt::Free(device, device_ids);
  }
  // Ensure the buffers hold at least `n` int64 ids on `dev` (grow-only).
  void Ensure(vt::Device dev, int64_t n) {
    if (backend == nullptr) {
      backend = &vt::GetBackend(dev.type);
      device = dev;
    }
    if (n <= capacity) return;
    if (pinned_host != nullptr) backend->FreePinned(pinned_host);
    if (device_ids != nullptr) vt::Free(device, device_ids);
    const size_t bytes = static_cast<size_t>(n) * sizeof(int64_t);
    device_ids = vt::Alloc(device, bytes);
    pinned_host = static_cast<int64_t*>(backend->AllocPinned(bytes));
    capacity = n;
  }
};

Sampler::Sampler(LogprobsMode logprobs_mode) : logprobs_mode_(logprobs_mode) {
  std::random_device entropy;
  default_seed_ = (uint64_t(entropy()) << 32) ^ uint64_t(entropy());
}
Sampler::~Sampler() = default;

std::vector<int64_t> Sampler::greedy_argmax_host(vt::Queue& q,
                                                 const vt::Tensor& logits,
                                                 int64_t n) const {
  if (greedy_scratch_ == nullptr) {
    greedy_scratch_ = std::make_unique<GreedyArgmaxScratch>();
  }
  greedy_scratch_->Ensure(logits.device, n < 1 ? 1 : n);
  vt::Tensor ids = vt::Tensor::Contiguous(greedy_scratch_->device_ids,
                                          vt::DType::kI64, logits.device, {n});
  vt::GreedyArgmax(q, ids, logits);
  vt::Backend& b = vt::GetBackend(logits.device.type);
  if (n != 0) {
    b.Copy(q, greedy_scratch_->pinned_host, greedy_scratch_->device_ids,
           static_cast<size_t>(n) * sizeof(int64_t));
  }
  b.Synchronize(q);
  return std::vector<int64_t>(greedy_scratch_->pinned_host,
                              greedy_scratch_->pinned_host + n);
}

std::vector<int64_t> Sampler::sample(vt::Queue& q, vt::Tensor& logits,
                                     const SamplingMetadata& sm,
                                     std::vector<float>* processed_out) const {
  const int64_t n = logits.shape[0];
  const int64_t vocab = logits.shape[1];
  // processed_* snapshot (sampler.py:262-271,286-302). Non-null only when the
  // caller is in one of those two modes AND a request asked for logprobs.
  const bool want_processed = processed_out != nullptr;
  auto snapshot_processed = [&]() {
    processed_out->resize(static_cast<size_t>(n) * static_cast<size_t>(vocab));
    if (logprobs_mode_ == LogprobsMode::kProcessedLogits) {
      // processed_logits: the mutated logits tensor itself (:267-268).
      vt::Backend& b = vt::GetBackend(logits.device.type);
      b.Copy(q, processed_out->data(), logits.data,
             processed_out->size() * sizeof(float));
      b.Synchronize(q);
    } else {
      // processed_logprobs: compute_logprobs of it (:269-270).
      DeviceBuffer plp(logits.device, q, vt::DType::kF32, {n, vocab});
      vt::ComputeLogprobs(q, plp.tensor(), logits);
      plp.download(processed_out->data());
    }
  };

  VT_CHECK(!(sm.all_greedy && sm.all_random),
           "sampler: all_greedy and all_random are mutually exclusive");

  // 7a. Greedy snapshot (unless all_random). For an all-greedy batch, return it.
  std::vector<int64_t> greedy_sampled;
  const bool have_greedy = !sm.all_random;
  if (have_greedy) {
    greedy_sampled = greedy_argmax_host(q, logits, n);
    if (sm.all_greedy) {
      // Upstream returns HERE, before temperature (:262-271), so an all-greedy
      // batch's "processed" tensor is the post-logits-processor one.
      if (want_processed) snapshot_processed();
      return greedy_sampled;
    }
  }

  VT_CHECK(sm.temperature.has_value(),
           "sampler: temperature is required unless all_greedy");
  const std::vector<float>& temp = *sm.temperature;
  VT_CHECK(static_cast<int64_t>(temp.size()) == n,
           "sampler: temperature must have num_reqs rows");

  // 7b. Temperature (temp<eps->1.0 guard applied inside when !all_random).
  {
    DeviceScratch t(logits.device, q, temp.data(), vt::DType::kF32, {n});
    vt::ApplyTemperature(q, logits, t.tensor(), sm.all_random);
  }

  // 7c. Argmax-invariant procs: min_p (kMinPArgmaxInvariant == true).
  if (!sm.min_p.empty()) apply_min_p(q, logits, sm.min_p);

  // 7d. top_k and/or top_p (materialize the per-req optionals; nullptr => skip).
  ApplyTopKTopPFromMeta(q, logits, sm.top_k, sm.top_p);

  // Upstream takes the processed snapshot inside topk_topp_sampler, i.e. right
  // here: after temperature, min_p and top-k/top-p, before the sampling draw
  // (:286-290). A token masked away by top-k/top-p therefore reads -inf.
  if (want_processed) snapshot_processed();

  // 7e. probs = softmax(logits); random_sample (exponential-noise gumbel-max).
  DeviceBuffer probs(logits.device, q, vt::DType::kF32, {n, vocab});
  vt::ComputeProbs(q, probs.tensor(), logits);

  // Unseeded draws advance the sampler's stream. Explicit seeds instead use
  // accepted output positions, so replay/resume and batch permutations agree.
  // This remains VT's SplitMix64 stream, not torch Philox token parity.
  VT_CHECK(sm.output_token_positions.empty() || sm.output_token_positions.size() == size_t(n),
           "sampler: output token positions must have num_reqs rows");
  const auto batch_seed = vt::sample::SplitMix64(default_seed_ + random_step_++);
  std::vector<int64_t> seeds(static_cast<size_t>(n), static_cast<int64_t>(batch_seed));
  for (const auto& [i, seed] : sm.generators) {
    VT_CHECK(i >= 0 && static_cast<int64_t>(i) < n,
             "sampler: generator request index out of range");
    const auto position = sm.output_token_positions.empty() ? 0 : sm.output_token_positions[size_t(i)];
    // Cancel RandomSample's row salt only for explicit per-request generators.
    seeds[size_t(i)] = static_cast<int64_t>(vt::sample::SplitMix64(seed + position) -
                                          0x9E3779B97F4A7C15ULL * uint64_t(i));
  }
  std::vector<int64_t> random_sampled(static_cast<size_t>(n));
  {
    DeviceScratch s(logits.device, q, seeds.data(), vt::DType::kI64, {n});
    DeviceBuffer rs(logits.device, q, vt::DType::kI64, {n});
    vt::RandomSample(q, rs.tensor(), probs.tensor(), s.tensor());
    rs.download(random_sampled.data());
  }

  // 7f. Mixed batch: where(temp < eps, greedy, random) per row.
  if (!have_greedy) return random_sampled;
  std::vector<int64_t> sampled(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    sampled[static_cast<size_t>(i)] = (temp[static_cast<size_t>(i)] < vt::kSamplingEps)
                                          ? greedy_sampled[static_cast<size_t>(i)]
                                          : random_sampled[static_cast<size_t>(i)];
  }
  return sampled;
}

SamplerOutput Sampler::forward(vt::Queue& q, vt::Tensor& logits,
                               const SamplingMetadata& sm,
                               vt::Tensor* sampled_ids_out) const {
  VT_CHECK(logits.rank == 2, "sampler: logits must be [num_reqs, vocab]");
  VT_CHECK(logits.dtype == vt::DType::kF32, "sampler: logits must be f32");
  const int64_t n = logits.shape[0];
  const int64_t vocab = logits.shape[1];
  if (sampled_ids_out != nullptr) {
    VT_CHECK(sampled_ids_out->dtype == vt::DType::kI64,
             "sampler: sampled_ids_out must be int64 (GreedyArgmax id dtype)");
    VT_CHECK(sampled_ids_out->Numel() == n,
             "sampler: sampled_ids_out must have num_reqs elements");
  }

  // 1. RAW snapshot, BEFORE any mutation (sampler.py:85-93). Only the raw_*
  //    modes snapshot here; the processed_* pair is taken inside sample(),
  //    after the mutations they are named for, and overwrites this below
  //    (sampler.py:104-106).
  //    sampler.py:86: `if num_logprobs is not None or
  //    sampling_metadata.logprob_token_ids:` — generative scoring needs the
  //    same snapshot, and a request may set ONLY logprob_token_ids. Python
  //    truthiness makes `{}` falsy just like None, hence the !empty().
  //    logprobs_mode and logprob_token_ids are ORTHOGONAL and COMPOSE: the mode
  //    selects WHICH tensor the snapshot holds (raw vs processed), the ids
  //    select WHICH entries step 8 reads out of it.
  const std::optional<int> num_logprobs = sm.max_num_logprobs;
  const bool want_token_ids =
      sm.logprob_token_ids.has_value() && !sm.logprob_token_ids->empty();
  const bool want_logprobs = num_logprobs.has_value() || want_token_ids;
  const bool processed_mode = logprobs_mode_ == LogprobsMode::kProcessedLogprobs ||
                              logprobs_mode_ == LogprobsMode::kProcessedLogits;
  std::vector<float> raw_logprobs;  // host [n*vocab] when want_logprobs
  if (want_logprobs && !processed_mode) {
    raw_logprobs.resize(static_cast<size_t>(n) * static_cast<size_t>(vocab));
    if (logprobs_mode_ == LogprobsMode::kRawLogits) {
      // raw_logits (sampler.py:90-93): the logits THEMSELVES, no log_softmax.
      // They are already f32 (checked above), so upstream's `.clone()` /
      // `.to(torch.float32)` is exactly this copy out — and it must happen here,
      // before the processors below mutate the tensor in place.
      vt::Backend& b = vt::GetBackend(logits.device.type);
      b.Copy(q, raw_logprobs.data(), logits.data,
             raw_logprobs.size() * sizeof(float));
      b.Synchronize(q);
    } else {
      DeviceBuffer rlp(logits.device, q, vt::DType::kF32, {n, vocab});
      vt::ComputeLogprobs(q, rlp.tensor(), logits);
      rlp.download(raw_logprobs.data());
    }
  }

  // 2. float32 (already f32; checked above).
  // 3-6. Logits processors, in upstream order (apply_logits_processors):
  //   allowed_token_ids -> bad_words -> non-argmax-invariant (min_tokens,
  //   logit_bias) -> penalties.
  if (sm.allowed_token_ids_mask.has_value()) {
    apply_allowed_token_ids(q, logits, *sm.allowed_token_ids_mask);
  }
  if (!sm.bad_words_token_ids.empty()) {
    apply_bad_words(q, logits, sm.bad_words_token_ids, sm.output_token_ids);
  }
  // Non-argmax-invariant procs (kMinTokensArgmaxInvariant / kLogitBiasArgmax-
  // Invariant == false): min_tokens then logit_bias.
  apply_min_tokens(q, logits, sm.min_tokens, sm.output_token_ids);
  apply_logit_bias(q, logits, sm.logit_bias);
  // Custom host logits processors (ROAD-V1-C7 `custom_logit_processor`). Run at
  // the END of the non-argmax-invariant stage — after the builtin min_tokens /
  // logit_bias, before penalties — mirroring vLLM's ordering (builtins precede
  // custom procs in the non_argmax_invariant list; sampler.py:399). Absent =>
  // no-op (byte-identical default).
  apply_logits_processors(q, logits, sm.logits_processors, sm.output_token_ids);
  // Penalties (repetition, frequency, presence).
  if (!sm.no_penalties) {
    VT_CHECK(sm.prompt_token_ids.has_value(),
             "sampler: prompt_token_ids required when penalties are active");
    apply_all_penalties(q, logits, *sm.prompt_token_ids, sm.presence_penalties,
                        sm.frequency_penalties, sm.repetition_penalties,
                        sm.output_token_ids);
  }

  // 7. Sample.
  // ENG-ASYNC-SCHED W3 device-resident fast path: an all-greedy batch with no
  // logprobs writes the sampled ids straight into the caller's device tensor via
  // GreedyArgmax — NO host download, NO main-queue synchronize. The async output
  // owns the single D2H later (async_utils.py:31 keeps sampled ids GPU-side).
  if (sampled_ids_out != nullptr && sm.all_greedy && !want_logprobs) {
    vt::GreedyArgmax(q, *sampled_ids_out, logits);
    SamplerOutput out;
    out.sampled_on_device = true;  // host sampled_token_ids intentionally empty
    return out;
  }
  // processed_* modes take their snapshot INSIDE sample(), where the mutations
  // they are named for have happened; it then replaces the (empty) raw one, as
  // upstream's `if processed_logprobs is not None` does (:104-106).
  std::vector<float> processed;
  const std::vector<int64_t> sampled = sample(
      q, logits, sm, (want_logprobs && processed_mode) ? &processed : nullptr);
  if (!processed.empty()) raw_logprobs = std::move(processed);
  // Async path fallback (random rows or logprobs requested): the host `sample()`
  // above already materialized the ids, so mirror them into the device out-tensor
  // for a uniform async-output D2H. Correct, but no zero-copy win (documented; the
  // greedy gate is the overlap target).
  if (sampled_ids_out != nullptr) {
    vt::GetBackend(logits.device.type)
        .Copy(q, sampled_ids_out->data, sampled.data(),
              static_cast<size_t>(n) * sizeof(int64_t));
  }

  // 8. Gather logprobs.
  //
  // 8a. logprob_token_ids first (sampler.py:111-118): the explicit-id gather,
  //     which sampler.py:119-136 then lets WIN over any top-k gather.
  std::optional<LogprobsTensors> token_ids_tensors;
  if (want_token_ids) {
    token_ids_tensors = GatherSpecificTokenLogprobs(
        raw_logprobs, n, vocab, *sm.logprob_token_ids, sampled);
  }

  std::optional<LogprobsTensors> logprobs_tensors;
  if (!num_logprobs.has_value()) {
    // sampler.py:120-121: no count requested, so the explicit-id gather is the
    // whole answer (or nothing, when neither was requested).
    logprobs_tensors = std::move(token_ids_tensors);
  } else if (token_ids_tensors.has_value()) {
    // sampler.py:133-136: both were requested; the explicit ids are more
    // specific and win. Computing the top-k gather first and discarding it, as
    // upstream literally does, would be pure waste — the branch below is the
    // only consumer of it, so it is simply not entered.
    logprobs_tensors = std::move(token_ids_tensors);
  } else {
    if (*num_logprobs == -1) {
      // Full unsorted/unranked raw logprobs (upstream LogprobsTensors(empty,
      // raw_logprobs, empty), sampler.py:122-125). Minimal 1:1 form: the
      // [n, vocab] raw logprobs with empty ids/ranks.
      //
      // UNREACHABLE, exactly as upstream's is: max_num_logprobs comes from
      // InputBatch::num_logprobs, whose values add_request already widened from
      // `-1` to vocab_size (gpu_input_batch.py:434-440), so a live request for
      // "all logprobs" arrives here as a concrete count and takes the gather
      // below. Kept because upstream keeps it; a caller that hand-builds
      // SamplingMetadata can still reach it. Note the shape differs from the
      // gathered one — ids and ranks are empty — so any NEW consumer must
      // branch on it rather than index blindly (issue #231).
      LogprobsTensors lt;
      lt.num_positions = static_cast<int>(n);
      lt.num_tokens_per_position = static_cast<int>(vocab);
      lt.logprobs = raw_logprobs;
      logprobs_tensors = std::move(lt);
    } else {
      logprobs_tensors = GatherLogprobs(raw_logprobs, n, vocab, *num_logprobs, sampled);
    }
  }

  // 9. SamplerOutput: sampled_token_ids [num_reqs, 1] (list-of-lists form).
  SamplerOutput out;
  out.sampled_token_ids.resize(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    out.sampled_token_ids[static_cast<size_t>(i)] = {
        static_cast<int32_t>(sampled[static_cast<size_t>(i)])};
  }
  out.logprobs_tensors = std::move(logprobs_tensors);
  return out;
}

// compute_prompt_logprobs (gpu_model_runner.py:5688-5697). The prompt half of
// step 1 + step 8 above, with nothing in between: no logits processor, no
// temperature, no sampling. Upstream spells that out at :5691-5693 — "prompt
// tokens skip sampling processors, so processed_* and raw_* yield the same
// scores here" — which is why this reuses the exact ops rather than a
// prompt-specific variant.
LogprobsTensors Sampler::compute_prompt_logprobs(
    vt::Queue& q, const vt::Tensor& logits, int num_logprobs,
    const std::vector<int64_t>& target_token_ids) const {
  const int64_t n = logits.shape[0];
  const int64_t vocab = logits.shape[1];
  VT_CHECK(static_cast<int64_t>(target_token_ids.size()) == n,
           "compute_prompt_logprobs: one target token id per prompt row");
  VT_CHECK(num_logprobs >= 0 && static_cast<int64_t>(num_logprobs) <= vocab,
           "compute_prompt_logprobs: num_logprobs must be in [0, vocab]");
  VT_CHECK(logits.dtype == vt::DType::kF32 && logits.rank == 2,
           "compute_prompt_logprobs: logits must be f32 [n, vocab]");

  LogprobsTensors lt;
  if (n == 0) {
    lt.num_positions = 0;
    lt.num_tokens_per_position = num_logprobs + 1;
    return lt;
  }

  DeviceBuffer rlp(logits.device, q, vt::DType::kF32, {n, vocab});
  vt::ComputeLogprobs(q, rlp.tensor(), logits);
  std::vector<float> raw_logprobs(static_cast<size_t>(n) *
                                  static_cast<size_t>(vocab));
  rlp.download(raw_logprobs.data());
  return GatherLogprobs(raw_logprobs, n, vocab, num_logprobs, target_token_ids);
}

}  // namespace vllm::v1
