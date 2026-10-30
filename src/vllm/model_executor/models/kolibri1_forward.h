// Kolibri-1 — CPU forward pass (MODEL-TEXT-kolibri-1 W2, private header).
//
// Declares the hybrid SWA/full-attention forward the registry's
// `ForwardKolibri1ForCausalLM` hook delegates to. CPU queue only; the GPU arm
// is a separate owed row. See .agents/specs/kolibri-1-cpu.md.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"  // Dev, DBuf
#include "vllm/model_executor/models/kolibri1_dequant_cache.h"  // Cache
#include "vllm/model_executor/models/kolibri1_weights.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // PagedKvCache, ForwardLogits
#include "vt/ops.h"  // DType

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
    const std::vector<int32_t>& logits_indices,
    bool spec_verify_rows = false);

// ── The compute class (the row's ONE rule) ──────────────────────────────────
//
// The class a linear site's bytes ride: the bf16-dot GEMV (bf16 in, f32
// accumulate, the BFMMLA class) or the f32-widen MatmulBT. The plain bf16 arm
// and the fp8 arm's DECODED bytes read the SAME rule (spec
// .agents/specs/kolibri1-bf16-arm.md), so a change that moves one arm's class
// must move the other's.
//
// A numeric gate cannot see that move wherever the two classes produce the
// same bits, so the SELECTION is exposed here: the rule is a pure function of
// the site's facts, and the forward records what it selected.
enum class Kolibri1ComputeClass {
  kF32Widen = 0,  // the shared f32-widen vt::MatmulBT
  kBf16Dot = 1,   // the bf16-dot GEMV (vbfdotq_f32)
};

// The facts the rule reads, one per conjunct.
struct Kolibri1ClassFacts {
  bool weight_is_bf16 = false;   // the RESOLVED weight is bf16-resident
  int64_t t = 0;                 // the site's token count
  vt::DType out_dtype = vt::DType::kF32;  // the site's output dtype
  bool bf16dot_available = false;  // DetectArmBf16() proved FEAT_BF16
  bool cache_enabled = false;      // the dequant cache is enabled
  bool verify_rows = false;        // the spec-verify class pin is engaged
  // `verify_rows` rides beside `t`: the spec-verify step forwards its 1+k
  // rows in ONE t>1 call, and the exactness contract
  // (ISSUE-LOCAL-01M5H6AE8666C5SPQ3GGCMTB1M) requires that step to compute
  // every row in the SAME class plain greedy decode uses. The rule reads it
  // as the row-count tier, not as a separate class.
};

// THE ONE RULE. The bf16-dot class iff every conjunct holds.
Kolibri1ComputeClass SelectKolibri1ComputeClass(
    const Kolibri1ClassFacts& facts);

// What the forward selected, per site.
struct Kolibri1ClassRecord {
  uint64_t calls = 0;           // the linear calls at this site
  uint64_t bf16_dot_calls = 0;  // the calls that selected the bf16-dot class
  Kolibri1ClassFacts facts;     // the facts of the FIRST call at this site
};

// The recording is OFF by default, and the decode path pays one branch per
// linear call. `SetKolibri1ComputeClassRecording(true)` turns it on.
//
// It is a PRODUCTION instrument, not a test-only driver: the row's own
// profile switch (`VT_KOLIBRI1_PROFILE=1`) turns the recording on as well,
// and the profile report then prints the class table through
// `ReportKolibri1ComputeClasses` below.
void SetKolibri1ComputeClassRecording(bool on);
void ResetKolibri1ComputeClassRecords();
const std::map<std::string, Kolibri1ClassRecord>& Kolibri1ComputeClassRecords();

// The class table, one line per site: the linear calls at that site and the
// calls that selected the bf16-dot class. The profile report prints it.
void ReportKolibri1ComputeClasses();

}  // namespace vllm
