// Kolibri-1 — CPU forward pass (MODEL-TEXT-kolibri-1 W2, private header).
//
// Declares the hybrid SWA/full-attention forward the registry's
// `ForwardKolibri1ForCausalLM` hook delegates to. CPU queue only; the GPU arm
// is a separate owed row. See .agents/specs/kolibri-1-cpu.md.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"  // Dev, DBuf
#include "vllm/model_executor/models/kolibri1_dequant_cache.h"  // Cache
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
// The PRODUCTION UNCACHED fp8-block dequant: the exact bytes the CPU
// forward feeds to the GEMM when the dequant cache is disabled (the
// `DequantFp8Block` body in kolibri1_forward.cpp). Exposed for the
// byte-identity gate, which compares the CACHED bytes against the UNCACHED
// ones on the same weight: the cache is a caching change only, so the two
// must be byte-identical.
dense_attn::DBuf DequantFp8BlockUncached(dense_attn::Dev d,
                                         const Fp8BlockWeight& w);

// The PRODUCTION resolve of one fp8-block projection: the CACHED path (the
// dequant cache's lease) when the cache is enabled, the UNCACHED path
// (`DequantFp8Block`) otherwise and after a cache DEGRADATION (the cache
// could not hold its bytes). `cache` is a parameter so the gate can drive a
// degraded cache without touching the process-wide one; the production call
// passes `kolibri1_dequant_cache::ProcessCache()`.
struct ResolvedFp8Projection {
  dense_attn::DBuf buf;   // the uncached block (empty when the cache served)
  dense_attn::Tensor wt;  // the weight the GEMM reads
  bool cached = false;    // true when the bytes came from the cache
};
ResolvedFp8Projection ResolveFp8Projection(dense_attn::Dev d,
                                           const Fp8BlockWeight& w,
                                           kolibri1_dequant_cache::Cache& cache);

ForwardLogits ForwardKolibri1Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const Kolibri1Weights& weights,
    const MultiKvCacheIndex* multi_kv, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices);

}  // namespace vllm
