// Ported from: vllm/v1/sample/logits_processor/builtin.py @ e24d1b24.
#include "vllm/v1/sample/logits_processor/builtin.h"

#include <cstddef>
#include <vector>

#include "vllm/v1/sample/device_scratch.h"
#include "vllm/v1/sample/ops/bad_words.h"
#include "vllm/v1/sample/ops/penalties.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm::v1 {

void apply_min_tokens(vt::Queue& q, vt::Tensor& logits,
                      const std::map<int, MinTokensState>& min_tokens,
                      const std::vector<std::vector<int32_t>>& output_token_ids) {
  VT_CHECK(logits.rank == 2, "apply_min_tokens: logits must be [num_reqs, vocab]");
  if (min_tokens.empty()) return;

  // Build the (request, stop-token) -inf slice. Upstream pre-filters min_toks so
  // only under-floor requests remain; we re-check output_len < min_tokens so the
  // function is correct on its own (the same set upstream masks).
  std::vector<int32_t> rows;
  std::vector<int32_t> cols;
  for (const auto& [i, state] : min_tokens) {
    VT_CHECK(i >= 0 && static_cast<size_t>(i) < output_token_ids.size(),
             "apply_min_tokens: request index out of range");
    const int output_len = static_cast<int>(output_token_ids[static_cast<size_t>(i)].size());
    if (output_len >= state.min_tokens) continue;
    for (int32_t tok : state.stop_token_ids) {
      rows.push_back(i);
      cols.push_back(tok);
    }
  }

  if (rows.empty()) return;
  const int64_t m = static_cast<int64_t>(rows.size());
  DeviceScratch r(logits.device, q, rows.data(), vt::DType::kI32, {m});
  DeviceScratch c(logits.device, q, cols.data(), vt::DType::kI32, {m});
  vt::ApplyTokenMask(q, logits, r.tensor(), c.tensor());
}

void apply_logit_bias(vt::Queue& q, vt::Tensor& logits,
                      const std::map<int, std::map<int32_t, float>>& logit_bias) {
  VT_CHECK(logits.rank == 2, "apply_logit_bias: logits must be [num_reqs, vocab]");
  if (logit_bias.empty()) return;

  std::vector<int32_t> rows;
  std::vector<int32_t> cols;
  std::vector<float> biases;
  for (const auto& [i, tok_bias] : logit_bias) {
    for (const auto& [tok, bias] : tok_bias) {
      rows.push_back(i);
      cols.push_back(tok);
      biases.push_back(bias);
    }
  }

  if (rows.empty()) return;
  const int64_t m = static_cast<int64_t>(rows.size());
  DeviceScratch r(logits.device, q, rows.data(), vt::DType::kI32, {m});
  DeviceScratch c(logits.device, q, cols.data(), vt::DType::kI32, {m});
  DeviceScratch b(logits.device, q, biases.data(), vt::DType::kF32, {m});
  vt::ApplyLogitBias(q, logits, r.tensor(), c.tensor(), b.tensor());
}

void apply_speculative_logit_filters(
    vt::Queue& q, vt::Tensor& logits, const SamplingMetadata& metadata,
    const std::vector<int32_t>& cu_num_logits) {
  if (!metadata.allowed_token_ids_mask.has_value() &&
      metadata.logit_bias.empty() && metadata.min_tokens.empty()) return;
  VT_CHECK(logits.rank == 2 && logits.dtype == vt::DType::kF32,
           "speculative logit processors require [rows, vocab] F32 logits");
  VT_CHECK(cu_num_logits.size() >= 2 && cu_num_logits.front() == 0 &&
               cu_num_logits.back() == logits.shape[0],
           "speculative logit processors require valid row offsets");
  const size_t num_reqs = cu_num_logits.size() - 1;
  VT_CHECK(!metadata.allowed_token_ids_mask.has_value() ||
               metadata.allowed_token_ids_mask->size() == num_reqs,
           "speculative allowed-token mask requires one row per request");
  VT_CHECK(metadata.min_tokens.empty() ||
               metadata.output_token_positions.size() == num_reqs,
           "speculative min-tokens requires accepted output positions");
  std::vector<std::vector<uint8_t>> allowed_rows;
  if (metadata.allowed_token_ids_mask.has_value())
    allowed_rows.reserve(static_cast<size_t>(logits.shape[0]));
  std::vector<int32_t> bias_rows, bias_cols, mask_rows, mask_cols;
  std::vector<float> biases;
  for (size_t req = 0; req < num_reqs; ++req) {
    const int32_t begin = cu_num_logits[req], end = cu_num_logits[req + 1];
    VT_CHECK(begin >= 0 && end >= begin && end <= logits.shape[0],
             "speculative logit row offsets must be monotonic");
    const auto bias = metadata.logit_bias.find(static_cast<int>(req));
    const auto floor = metadata.min_tokens.find(static_cast<int>(req));
    for (int32_t row = begin; row < end; ++row) {
      if (metadata.allowed_token_ids_mask.has_value())
        allowed_rows.push_back((*metadata.allowed_token_ids_mask)[req]);
      if (bias != metadata.logit_bias.end()) {
        for (const auto& [token, value] : bias->second) {
          bias_rows.push_back(row);
          bias_cols.push_back(token);
          biases.push_back(value);
        }
      }
      if (floor != metadata.min_tokens.end() &&
          metadata.output_token_positions[req] + uint64_t(row - begin) <
              static_cast<uint64_t>(floor->second.min_tokens)) {
        for (int32_t token : floor->second.stop_token_ids) {
          mask_rows.push_back(row);
          mask_cols.push_back(token);
        }
      }
    }
  }
  if (!allowed_rows.empty())
    apply_allowed_token_ids(q, logits, allowed_rows);
  if (!bias_rows.empty()) {
    const int64_t count = static_cast<int64_t>(bias_rows.size());
    DeviceScratch r(logits.device, q, bias_rows.data(), vt::DType::kI32, {count});
    DeviceScratch c(logits.device, q, bias_cols.data(), vt::DType::kI32, {count});
    DeviceScratch b(logits.device, q, biases.data(), vt::DType::kF32, {count});
    vt::ApplyLogitBias(q, logits, r.tensor(), c.tensor(), b.tensor());
  }
  if (!mask_rows.empty()) {
    const int64_t count = static_cast<int64_t>(mask_rows.size());
    DeviceScratch r(logits.device, q, mask_rows.data(), vt::DType::kI32, {count});
    DeviceScratch c(logits.device, q, mask_cols.data(), vt::DType::kI32, {count});
    vt::ApplyTokenMask(q, logits, r.tensor(), c.tensor());
  }
}

void apply_speculative_logits_processors(
    vt::Queue& q, vt::Tensor& logits, const SamplingMetadata& metadata,
    const std::vector<int32_t>& cu_num_logits,
    const std::vector<int32_t>& draft_input_ids) {
  const bool history_needed = !metadata.no_penalties ||
      !metadata.bad_words_token_ids.empty() || !metadata.logits_processors.empty();
  if (!history_needed && !metadata.allowed_token_ids_mask.has_value() &&
      metadata.logit_bias.empty() && metadata.min_tokens.empty()) return;
  VT_CHECK(logits.rank == 2 && logits.dtype == vt::DType::kF32 &&
               logits.IsContiguous(),
           "speculative processors require contiguous [rows, vocab] F32 logits");
  VT_CHECK(cu_num_logits.size() >= 2 && cu_num_logits.front() == 0 &&
               cu_num_logits.back() == logits.shape[0],
           "speculative processors require valid row offsets");
  const size_t requests = cu_num_logits.size() - 1;
  for (size_t i = 0; i < requests; ++i)
    VT_CHECK(cu_num_logits[i] >= 0 && cu_num_logits[i + 1] >= cu_num_logits[i],
             "speculative processor row offsets must be monotonic");
  VT_CHECK(!history_needed ||
               (metadata.output_token_ids.size() == requests &&
                draft_input_ids.size() == static_cast<size_t>(logits.shape[0])),
           "speculative history processors require committed histories and draft inputs");
  VT_CHECK(!metadata.allowed_token_ids_mask.has_value() ||
               metadata.allowed_token_ids_mask->size() == requests,
           "speculative allowed-token mask requires one row per request");
  VT_CHECK(metadata.min_tokens.empty() ||
               metadata.output_token_positions.size() == requests,
           "speculative min-tokens requires accepted output positions");
  VT_CHECK(metadata.no_penalties ||
               (metadata.prompt_token_ids.has_value() &&
                metadata.prompt_token_ids->size() == requests &&
                metadata.presence_penalties.size() == requests &&
                metadata.frequency_penalties.size() == requests &&
                metadata.repetition_penalties.size() == requests),
           "speculative penalties require per-request prompts and penalty values");
  const auto check_keys = [&](const auto& map) {
    for (const auto& entry : map)
      VT_CHECK(entry.first >= 0 && static_cast<size_t>(entry.first) < requests,
               "speculative processor request index out of range");
  };
  check_keys(metadata.bad_words_token_ids); check_keys(metadata.logits_processors);
  check_keys(metadata.logit_bias); check_keys(metadata.min_tokens);

  SamplingMetadata expanded;
  if (metadata.allowed_token_ids_mask.has_value())
    expanded.allowed_token_ids_mask.emplace();
  if (!metadata.no_penalties) expanded.prompt_token_ids.emplace();
  std::vector<int32_t> mask_rows, mask_cols;
  for (size_t req = 0; req < requests; ++req) {
    const int32_t begin = cu_num_logits[req], end = cu_num_logits[req + 1];
    std::vector<int32_t> history;
    if (history_needed) history = metadata.output_token_ids[req];
    for (int32_t row = begin; row < end; ++row) {
      if (history_needed) {
        if (row > begin) history.push_back(draft_input_ids[static_cast<size_t>(row)]);
        expanded.output_token_ids.push_back(history);
      }
      if (expanded.allowed_token_ids_mask)
        expanded.allowed_token_ids_mask->push_back((*metadata.allowed_token_ids_mask)[req]);
      if (const auto it = metadata.bad_words_token_ids.find(static_cast<int>(req));
          it != metadata.bad_words_token_ids.end()) expanded.bad_words_token_ids[row] = it->second;
      if (const auto it = metadata.logits_processors.find(static_cast<int>(req));
          it != metadata.logits_processors.end()) expanded.logits_processors[row] = it->second;
      if (const auto it = metadata.logit_bias.find(static_cast<int>(req));
          it != metadata.logit_bias.end()) expanded.logit_bias[row] = it->second;
      if (const auto it = metadata.min_tokens.find(static_cast<int>(req));
          it != metadata.min_tokens.end()) {
        VT_CHECK(it->second.min_tokens >= 0, "speculative min_tokens must be nonnegative");
        const uint64_t floor = static_cast<uint64_t>(it->second.min_tokens);
        const uint64_t position = metadata.output_token_positions[req];
        // Subtraction after the range check avoids position+depth overflow.
        if (position < floor && static_cast<uint64_t>(row - begin) < floor - position)
          for (int32_t token : it->second.stop_token_ids) {
            mask_rows.push_back(row); mask_cols.push_back(token);
          }
      }
      if (!metadata.no_penalties) {
        expanded.prompt_token_ids->push_back((*metadata.prompt_token_ids)[req]);
        expanded.presence_penalties.push_back(metadata.presence_penalties[req]);
        expanded.frequency_penalties.push_back(metadata.frequency_penalties[req]);
        expanded.repetition_penalties.push_back(metadata.repetition_penalties[req]);
      }
    }
  }
  // Same order as Sampler::forward: allowed, bad words, min-tokens, bias,
  // custom processors, then penalties. Only registered host callbacks stage
  // logits to the host; builtins keep their existing native device operations.
  if (expanded.allowed_token_ids_mask && !expanded.allowed_token_ids_mask->empty())
    apply_allowed_token_ids(q, logits, *expanded.allowed_token_ids_mask);
  if (!expanded.bad_words_token_ids.empty())
    apply_bad_words(q, logits, expanded.bad_words_token_ids, expanded.output_token_ids);
  if (!mask_rows.empty()) {
    const int64_t count = static_cast<int64_t>(mask_rows.size());
    DeviceScratch r(logits.device, q, mask_rows.data(), vt::DType::kI32, {count});
    DeviceScratch c(logits.device, q, mask_cols.data(), vt::DType::kI32, {count});
    vt::ApplyTokenMask(q, logits, r.tensor(), c.tensor());
  }
  apply_logit_bias(q, logits, expanded.logit_bias);
  apply_logits_processors(q, logits, expanded.logits_processors, expanded.output_token_ids);
  if (!metadata.no_penalties)
    apply_all_penalties(q, logits, *expanded.prompt_token_ids,
                        expanded.presence_penalties, expanded.frequency_penalties,
                        expanded.repetition_penalties, expanded.output_token_ids);
}

void apply_min_p(vt::Queue& q, vt::Tensor& logits, const std::vector<float>& min_p) {
  const int64_t n = logits.shape[0];
  VT_CHECK(logits.rank == 2, "apply_min_p: logits must be [num_reqs, vocab]");
  VT_CHECK(static_cast<int64_t>(min_p.size()) == n, "apply_min_p: min_p must have num_reqs rows");
  DeviceScratch mp(logits.device, q, min_p.data(), vt::DType::kF32, {n});
  vt::ApplyMinP(q, logits, mp.tensor());
}

void apply_logits_processors(
    vt::Queue& q, vt::Tensor& logits,
    const std::map<int, LogitsProcessorCallback>& procs,
    const std::vector<std::vector<int32_t>>& output_token_ids) {
  if (procs.empty()) return;
  VT_CHECK(logits.rank == 2, "apply_logits_processors: logits must be [num_reqs, vocab]");
  const int64_t n = logits.shape[0];
  const int64_t vocab = logits.shape[1];

  vt::Backend& b = vt::GetBackend(logits.device.type);
  // The callbacks read+mutate the logits on the HOST, so any prior async op that
  // produced/mutated the logits must complete first (a no-op on the synchronous
  // CPU backend; a real sync on CUDA).
  b.Synchronize(q);

  // Obtain a host-addressable view of the [n, vocab] logits. The callbacks are
  // ABI code that LOADS AND STORES through whatever pointer we hand them, so the
  // question is whether the HOST MAY DEREFERENCE what `Backend::Alloc` returned
  // -- `DeviceMemoryIsHostAddressable()` -- and NOT whether host and device
  // happen to sit on the same physical RAM, which is the wider `UnifiedMemory()`.
  //
  // THIS USED TO ASK `UnifiedMemory()`, AND THAT IS #1746. It is the same
  // mistake `src/vt/op_provider.cpp` records beside `ReferenceTierEligible`,
  // where it cost two crashes (#844, #1435) and a third report (#960). CUDA
  // reports unified memory on GB10 because host and device address the same
  // physical RAM, yet `CudaBackend::Alloc` returns a plain `cudaMalloc` pointer
  // and CUDA never overrides the narrow predicate, so it keeps the `false`
  // default in include/vt/backend.h. Asking the wide question here therefore
  // handed the ABI callback a device pointer to store through, on the one box
  // the old comment named as safe.
  //
  // A backend that answers the narrow predicate `true` -- Vulkan, Metal
  // StorageModeShared, integrated ROCm -- keeps the zero-copy in-place wrap:
  // `logits.data` IS host memory there. Every other backend, CPU included,
  // stages down, runs the callbacks, and copies the edited logits back. CPU
  // pays that bounce because `CpuBackend` has never opted in to the narrow
  // predicate; that is a correct-but-conservative cost, and it is charged only
  // when a request actually registers a processor (the empty map returns above).
  const bool host_addressable = b.DeviceMemoryIsHostAddressable();
  const size_t total = static_cast<size_t>(n) * static_cast<size_t>(vocab);
  std::vector<float> staging;
  float* host = nullptr;
  if (host_addressable) {
    host = static_cast<float*>(logits.data);
  } else {
    staging.resize(total);
    if (total != 0) b.Copy(q, staging.data(), logits.data, total * sizeof(float));
    b.Synchronize(q);
    host = staging.data();
  }

  static const std::vector<int32_t> kNoTokens;
  for (const auto& [i, cb] : procs) {
    if (cb.fn == nullptr) continue;
    VT_CHECK(i >= 0 && static_cast<int64_t>(i) < n,
             "apply_logits_processors: request index out of range");
    const std::vector<int32_t>& toks =
        (static_cast<size_t>(i) < output_token_ids.size())
            ? output_token_ids[static_cast<size_t>(i)]
            : kNoTokens;
    float* row = host + static_cast<size_t>(i) * static_cast<size_t>(vocab);
    cb.fn(toks.data(), static_cast<int32_t>(toks.size()), row,
          static_cast<int32_t>(vocab), cb.user_data);
  }

  if (!host_addressable) {
    if (total != 0) b.Copy(q, logits.data, staging.data(), total * sizeof(float));
    b.Synchronize(q);
  }
}

}  // namespace vllm::v1
