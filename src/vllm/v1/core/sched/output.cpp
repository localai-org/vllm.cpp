// Ported from: vllm/v1/core/sched/output.py @ e24d1b24
// See include/vllm/v1/core/sched/output.h for the new-vs-cached diff protocol,
// the deferred-field list, and the recorded deviations.
#include "vllm/v1/core/sched/output.h"

#include <utility>

#include "vllm/v1/request.h"

namespace vllm::v1 {

NewRequestData NewRequestData::from_request(
    const Request& request, std::vector<std::vector<int>> block_ids,
    std::optional<std::vector<int32_t>> prefill_token_ids) {
  NewRequestData data;
  data.req_id = request.request_id;
  data.prompt_token_ids = request.prompt_token_ids;
  data.sampling_params = request.sampling_params;
  data.block_ids = std::move(block_ids);
  data.num_computed_tokens = request.num_computed_tokens;
  data.prefill_token_ids = std::move(prefill_token_ids);
  // ENG-MM-INPUT-PIPELINE P2 (#2379): the hop that used to drop the multimodal
  // payload. Upstream's dataclass carries `mm_features=request.mm_features`
  // (output.py NewRequestData.from_request); ours dropped it, so the worker
  // never learned a request had an image. Empty vector on every text request.
  data.mm_features = request.mm_features;
  data.prompt_block_hashes = request.block_hashes;
  return data;
}

bool CachedRequestData::is_context_phase(const std::string& req_id) const {
  for (std::size_t i = 0; i < req_ids.size(); ++i) {
    if (req_ids[i] == req_id) {
      return i < num_output_tokens.size() && num_output_tokens[i] == 0;
    }
  }
  return false;
}

CachedRequestData CachedRequestData::make_empty() {
  return CachedRequestData{};
}

SchedulerOutput SchedulerOutput::make_empty() {
  SchedulerOutput output;
  output.scheduled_cached_reqs = CachedRequestData::make_empty();
  return output;
}

}  // namespace vllm::v1
