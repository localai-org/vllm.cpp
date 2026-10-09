// Shared Qwen3.5/3.6 registry-glue helper definitions (see qwen3_5_common.h).
// Extracted verbatim (behavior-preserving) from the former model_registry.cpp
// monolith so the dense and MoE variant TUs share one copy.
#include "vllm/model_executor/models/qwen3_5_common.h"

#include <cstdlib>
#include <optional>

#include "vllm/model_executor/layers/quantization/exl3_checkpoint.h"
#include "vllm/model_executor/models/qwen3_5.h"           // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_dense.h"     // DenseExecutionPrecision
#include "vllm/model_executor/models/qwen3_5_internal.h"  // ResolveMambaSsmCacheDType
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {

void ParseQwen3_5Config(const HfConfig& config) {
  // LoadHfConfig/HfConfigFromGguf already materialize the consumed Qwen fields.
  // This explicit per-family hook is where a family adds normalization or
  // validation without changing the registry/runner contract.
  (void)config;
}

ForwardLogits HostLogits(std::vector<float>&& host, int64_t vocab) {
  ForwardLogits logits;
  logits.vocab = vocab;
  logits.rows = vocab > 0 ? static_cast<int64_t>(host.size()) / vocab : 0;
  logits.host = std::move(host);
  return logits;
}

v1::KVCacheConfig MakeQwen3_5KVCache(const HfConfig& config, int block_size,
                                     int num_blocks) {
  return MakeQwen3_5KVCacheSpec(config, block_size, num_blocks, /*num_spec=*/0);
}

v1::KVCacheConfig MakeQwen3_5KVCacheSpec(const HfConfig& config, int block_size,
                                         int num_blocks, int num_spec,
                                         bool share_mtp_pages) {
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int head_dim = static_cast<int>(config.head_dim);
  const int num_value_heads = static_cast<int>(config.linear_num_value_heads);
  const int value_head_dim = static_cast<int>(config.linear_value_head_dim);
  const int key_head_dim = static_cast<int>(config.linear_key_head_dim);
  const int conv_kernel = static_cast<int>(config.linear_conv_kernel_dim);
  const int key_dim =
      static_cast<int>(config.linear_num_key_heads) * key_head_dim;
  const int value_dim = num_value_heads * value_head_dim;
  const int conv_dim = 2 * key_dim + value_dim;
  const bool gptq_f16 = config.torch_dtype == "float16" &&
      config.raw.contains("quantization_config") &&
      config.raw["quantization_config"].is_object() &&
      config.raw["quantization_config"].value("quant_method", std::string()) == "gptq";
  const bool exl3_f16 = IsExl3Checkpoint(config);
  const bool shared_mtp_pages = exl3_f16 && share_mtp_pages && num_spec > 0;
  // Target-only EXL3 has the same compact recurrent ownership as shared MTP.
  // It needs no draft pages, but must not reserve historical GDN identities.
  const bool aligned_exl3_pages = exl3_f16 && (num_spec == 0 || shared_mtp_pages);
  const bool explicit_f16 = gptq_f16 || exl3_f16;
  const auto precision =
      ResolveQwen3_5DensePrecision(config, gptq_f16, exl3_f16);

  // Diagnostic state-storage overrides belong to planning, not allocation:
  // the MambaSpec must describe the exact bytes the runner will consume.
  vt::DType conv_dtype = precision.gdn_conv_state;
  vt::DType ssm_dtype =
      explicit_f16 ? precision.gdn_recurrent_state
                   : detail::ResolveMambaSsmCacheDType(config, conv_dtype);
  if (const char* state_dtype = std::getenv("VT_GDN_STATE_BF16");
      state_dtype != nullptr && !explicit_f16) {
    if (state_dtype[0] == '0') {
      conv_dtype = vt::DType::kF32;
      ssm_dtype = vt::DType::kF32;
    } else if (state_dtype[0] == '1') {
      conv_dtype = vt::DType::kBF16;
      ssm_dtype = vt::DType::kBF16;
    }
  }

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  const vt::DType default_kv_dtype = v1::ResolveKvCacheDType();
  const vt::DType kv_dtype =
      explicit_f16 && default_kv_dtype != vt::DType::kF32
          ? precision.kv_auto : default_kv_dtype;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      // The spec is the SINGLE source of truth for the paged-KV storage dtype
      // and layout: the runner sizes the buffer from spec->page_size_bytes()
      // and builds its cache view from the spec's fields (MLA campaign W1).
      std::make_shared<v1::FullAttentionSpec>(
          block_size, num_kv_heads, head_dim, kv_dtype));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"gdn"},
      // SPEC-MTP I4: with k speculative tokens the conv row widens to
      // (K-1)+k taps (mamba_utils.py:226 `conv_kernel_size - 1 + num_spec`) so
      // the sliding window can be rewound to the accepted count, and
      // num_speculative_blocks = k gives MambaManager the k+1 SSM snapshot slots
      // per request (mamba/abstract.py:55-59). num_spec == 0 keeps the original
      // state shapes; EXL3 also uses bounded align ownership without a draft.
      std::make_shared<v1::MambaSpec>(
          block_size,
          std::vector<std::vector<int64_t>>{
              {conv_dim, conv_kernel - 1 + num_spec},
              {num_value_heads, value_head_dim, key_head_dim}},
          std::vector<vt::DType>{conv_dtype, ssm_dtype},
          /*page_size_padded=*/std::nullopt,
          // Native recurrence lives in request-owned compact rows. Align mode
          // reserves k+1 current-state identities plus one transition identity,
          // rather than charging a GDN page for every historical context page.
          /*mamba_cache_mode=*/aligned_exl3_pages ? "align" : "none",
          /*num_speculative_blocks=*/num_spec));
  // SPEC-MTP I5c: the MTP draft head is one extra full_attention decoder layer
  // (index num_hidden_layers upstream, qwen3_5_mtp.py:105-112) with its OWN paged
  // K/V — registered as a NEW attention KV layer whose draft names are all layer
  // names minus the target's (speculator.py:163-169). Sized exactly like a target
  // full-attn layer; it shares the target's block table / slot mapping. It exists
  // EXL3 registers it in the target group because the physical IDs are shared;
  // other checkpoints retain their separate draft group. It exists
  // ONLY when speculative decoding is on (num_spec > 0); num_spec == 0 (the
  // production default) emits only the two target groups, so the draft layer
  // is never allocated when speculation is off.
  if (num_spec > 0) {
    if (shared_mtp_pages) {
      kv.mtp_draft_shares_target_pages = true;
      kv.kv_cache_groups.front().layer_names.push_back("fa_draft");
    } else {
      kv.kv_cache_groups.emplace_back(
          std::vector<std::string>{"fa_draft"},
          std::make_shared<v1::FullAttentionSpec>(
              block_size, num_kv_heads, head_dim, kv_dtype));
    }
  }
  return kv;
}

}  // namespace vllm
