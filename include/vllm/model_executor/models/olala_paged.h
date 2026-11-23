// vllm.cpp ORIGINAL — the `olala` PAGED engine forward (NEW-CANDIDATE-PORTS
// wiring unit, issue ISSUE-LOCAL-01M835QTXAG5P2SD26QZYT0Z5W). Reference:
// modeling_olala.py @ f781f124 (OVHaiLLM/Olala-v1.0-mini-7A1B); the numeric
// pipeline is the W2d/W2e host op set (olala_ops.h), composed per token against
// the runner-owned caches — the same fold NemotronH's A2-P paged forward took
// (nemotron_h_device.cpp NemotronHPagedForward) for a hybrid Mamba+attention
// model whose per-token step ops already exist.
#pragma once

#include "vllm/model_executor/models/olala.h"

namespace vllm {

struct ModelForwardInput;
struct ForwardLogits;

// The paged forward takes `ModelForwardInput` WHOLE, mirroring
// `NemotronHPagedForward` / `KimiLinearModel::ForwardPaged`. Host f32 compute
// end to end (the weights are OwnedTensor keep-quant/bf16 bytes); the runner's
// caches are read and written through their host-addressable views, so the
// CPU engine is the supported device — a CUDA queue refuses BY NAME (the
// device arm is owed).
ForwardLogits OlalaPagedForward(const OlalaWeights& w, const OlalaParams& p,
                                const ModelForwardInput& input);

}  // namespace vllm
