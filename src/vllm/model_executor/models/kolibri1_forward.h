// Kolibri-1 — CPU forward pass (MODEL-TEXT-kolibri-1 W2, private header).
//
// Declares the hybrid SWA/full-attention forward the registry's
// `ForwardKolibri1ForCausalLM` hook delegates to. CPU queue only; the GPU arm
// is a separate owed row. See .agents/specs/kolibri-1-cpu.md.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/kolibri1_weights.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // PagedKvCache, ForwardLogits

namespace vllm {

// Runs one forward step: embedding -> 50 (tiny: N) sandwich-norm layers
// (sliding: RoPE + 513 window; full: RNoPE, no positional encoding; GQA with
// per-head qk-norm) -> MoE on EVERY layer (sigmoid-logit-add router, ungated
// shared expert) -> final norm -> untied lm_head.
//
// `multi_kv` resolves each layer's PagedKvCache by layer name; null falls back
// to `attn_kv[layer]` (the single-group layout). Logits gather follows
// `logits_indices` when it is a strict subset of the step.
ForwardLogits ForwardKolibri1Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const Kolibri1Weights& weights,
    const MultiKvCacheIndex* multi_kv, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices);

}  // namespace vllm
