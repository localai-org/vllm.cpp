#pragma once

#include "vt/tensor.h"

namespace vt {
struct Queue;
}

namespace vt::xpu {

// Exact scheduling alternative for nonaliasing F32[1,248320] raw snapshot.
// Two F32 state values are caller-owned until all four kernels/consumers finish.
// No allocation, host wait or retirement here. The general public log-softmax,
// including aliases/other shapes, remains unchanged.
void SplitRawLogprobsM1(Queue&, Tensor& output, const Tensor& logits, Tensor& state);

struct alignas(8) SelectedLogprobRecord {
  int64_t token_id;
  float logprob;
  int32_t rank;
};
static_assert(sizeof(SelectedLogprobRecord) == 16);

// Gather from the existing rounded F32 log-softmax snapshot. Counts >= the
// selected stored value, without changing softmax arithmetic. IDs are I64[R],
// output is aligned I8[R*sizeof(SelectedLogprobRecord)], scratch is caller-owned
// I8 sized by XpuCompactLogprobsWorkspaceBytes. Retain all owners through both
// queued kernels/copies/graph replays. Invalid device IDs produce rank -1 and
// logprob0 without reading out of bounds; the host consumer must reject them.
// No allocation, host readback or wait occurs here.
void CompactRawLogprobs0(Queue& q, Tensor& output, const Tensor& raw_logprobs,
                         const Tensor& selected_ids, Tensor& scratch);

// Exact two-stage comparison reduction. Scratch is caller-owned contiguous I8
// storage, sized by XpuTiledArgmaxWorkspaceBytes and aligned to 16 bytes. Retain
// it through both queued kernels and retire captured graphs before freeing or
// resizing it. No allocation, queue wait or host readback occurs in this call.
// F32 logits [R,V]; I64 or I32 output [R], lowest-ID ties and CPU NaN semantics.
void TiledGreedyArgmax(Queue& q, Tensor& token_ids, const Tensor& logits,
                      Tensor& scratch, int tile);

// Same caller-owned workspace/lifetime as TiledGreedyArgmax. F32 [R,V]
// logits and I32 [V] map produce I32 [R] GLOBAL token IDs, smallest global ID
// on ties. Any nonfinite logit or invalid mapping anywhere in a row yields -1.
void TiledMappedGreedyArgmax(Queue& q, Tensor& token_ids, const Tensor& logits,
                            const Tensor& global_ids, int64_t target_vocab,
                            Tensor& scratch, int tile);

// Greedy verification with the same acceptance walk and live-offset checks as
// GreedyRejectionSample. The caller owns both target_argmax and tiled scratch
// through the final accept consumer and any copies, including graph replays.
void TiledGreedyRejectionSample(Queue& q, Tensor& sampled, Tensor& num_sampled,
                                Tensor& target_argmax, const Tensor& logits,
                                const Tensor& draft_sampled, const Tensor& cu_num_logits,
                                Tensor& scratch, int tile);

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
