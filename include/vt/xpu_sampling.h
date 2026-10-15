#pragma once

#include "vt/tensor.h"

namespace vt {
struct Queue;
}

namespace vt::xpu {

// Exact small-top-k candidate for the production k=20 path. Returns false
// without changing logits when ties/non-finite values or a top-p boundary
// require the general stable-sort route. `p` is one F32 value per row.
bool ApplyTopK20TopP(Queue& q, Tensor& logits, const Tensor& p);

// Greedy one-hot drafter, sampled target. All tensors live on the XPU.
// `probs` is the target distribution after temperature and sampling processors
// for every expanded verification row. `proposal[row]` is the draft id for that
// row, or -1 for a bonus row. A rejected proposal is replaced by a draw from
// the target distribution with that id excluded. `seeds` are request-local,
// position-specific independent keys, one per row.
// `greedy` is an i8 flag per expanded row. Greedy rows use the target argmax
// for proposal acceptance, correction and bonus; random rows use the target
// distribution. `choices` and `accepted` are caller-owned scratch, kept alive through the
// queued operation; `sampled` and `num_sampled` are final outputs.
void SampleOneHotRejection(Queue& q, Tensor& sampled, Tensor& num_sampled,
                           Tensor& choices, Tensor& accepted,
                           const Tensor& probs, const Tensor& proposal,
                           const Tensor& cu_num_logits, const Tensor& seeds,
                           const Tensor& greedy);

}  // namespace vt::xpu
