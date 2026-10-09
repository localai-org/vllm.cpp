// Ported from: vllm/v1/kv_cache_interface.py @ e24d1b24, with the DeepSeek-V4
// spec surface re-read at the parity pin 5559679229bc961848b121ccdeaa8fa5d79bec98.
//
// The page_size_bytes / real_page_size_bytes math for FullAttentionSpec,
// SlidingWindowSpec, MLAAttentionSpec, SlidingWindowMLASpec and MambaSpec. See
// kv_cache_interface.h for the formulas and deferred-spec notes.
#include "vllm/v1/kv_cache_interface.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace vllm::v1 {

namespace {

// Upstream KVQuantMode.is_per_token_head / is_nvfp4 (the modes whose page-size
// math is deferred here — see header).
bool needs_deferred_quant_math(KVQuantMode mode) {
  return mode != KVQuantMode::kNone;
}

// The `cache_dtype` string and the `model_version` that together select the
// DeepSeek-V4 page, written once. Upstream compares these literals in four
// places (`kv_cache_interface.py:398-405`, `:628-636`).
constexpr const char* kFp8DsMlaCacheDType = "fp8_ds_mla";
constexpr const char* kDeepseekV4ModelVersion = "deepseek_v4";

// 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token, upstream's own
// arithmetic in its own comment (`kv_cache_interface.py:399-401`, `:629-632`).
// We already hold this layout element for element: `Fp8DsMlaLayout` records
// nope 448 + rope 64*2 = 576 plus an 8-byte scale row
// (`include/vllm/model_executor/models/deepseek_v4_compressor.h:113-125`).
constexpr int64_t kFp8DsMlaV4TokenBytes = 584;

// V3.2 main MLA: a 656-byte custom layout over the 576-wide latent
// (kv_lora_rank 512 + qk_rope_head_dim 64), off BLOCK_SIZE and not
// storage_block_size (`kv_cache_interface.py:402-405`).
constexpr int64_t kFp8DsMlaV32TokenBytes = 656;

bool is_fp8_ds_mla(const std::optional<std::string>& cache_dtype_str) {
  return cache_dtype_str.has_value() &&
         *cache_dtype_str == kFp8DsMlaCacheDType;
}

bool is_deepseek_v4(const std::optional<std::string>& model_version) {
  return model_version.has_value() &&
         *model_version == kDeepseekV4ModelVersion;
}

}  // namespace

void CheckMlaCacheFields(int compress_ratio, std::optional<int> alignment) {
  VT_CHECK(compress_ratio >= 1,
           "kv cache spec: compress_ratio must be >= 1 (upstream's model path "
           "applies max(1, compress_ratios[layer_id]), attention.py:205-212); "
           "0 would divide by zero in storage_block_size()");
  VT_CHECK(!alignment.has_value() || *alignment > 0,
           "kv cache spec: alignment must be > 0 when set (upstream's None "
           "means no padding, kv_cache_interface.py:383)");
}

int64_t RoundUp(int64_t x, int64_t y) {
  VT_CHECK(y > 0, "RoundUp: the divisor must be > 0");
  return ((x + y - 1) / y) * y;
}

void ApplyAlignmentPadding(AttentionSpec& spec, std::optional<int> alignment) {
  // Upstream `_apply_alignment_padding` (kv_cache_interface.py:345-351).
  if (!alignment.has_value()) {
    return;
  }
  const int64_t actual_page_size = spec.real_page_size_bytes();
  const int64_t padded_page_size = RoundUp(actual_page_size, *alignment);
  // Upstream writes ONLY when the rounding changed the number, so an already
  // aligned page keeps `page_size_padded` unset rather than gaining a value
  // equal to the real page.
  if (padded_page_size != actual_page_size) {
    spec.page_size_padded = padded_page_size;
  }
}

int64_t AttentionSpec::real_page_size_bytes() const {
  if (needs_deferred_quant_math(kv_quant_mode)) {
    throw std::runtime_error(
        "AttentionSpec: kv_quant_mode != NONE page-size math is deferred (T1)");
  }
  // K + V: 2 * block_size * num_kv_heads * head_size * dtype_size.
  return 2LL * block_size * num_kv_heads * head_size *
         static_cast<int64_t>(vt::SizeOf(dtype));
}

int64_t AttentionSpec::page_size_bytes() const {
  const int64_t real_page_size = real_page_size_bytes();
  // (Per-token-head scale bytes for quantized KV cache are deferred; the NONE
  // path adds nothing here.)
  if (page_size_padded.has_value()) {
    if (*page_size_padded < real_page_size) {
      throw std::runtime_error(
          "AttentionSpec: page_size_padded must be >= real_page_size_bytes");
    }
    return *page_size_padded;
  }
  return real_page_size;
}

int64_t FullAttentionSpec::real_page_size_bytes() const {
  if (needs_deferred_quant_math(kv_quant_mode)) {
    throw std::runtime_error(
        "FullAttentionSpec: kv_quant_mode != NONE page-size math is deferred "
        "(T1)");
  }
  const int64_t last_dim =
      static_cast<int64_t>(head_size) + static_cast<int64_t>(head_size_v);
  return static_cast<int64_t>(block_size) * num_kv_heads * last_dim *
         static_cast<int64_t>(vt::SizeOf(dtype));
}

// Upstream kv_cache_interface.py:396-410 — the MLA page formula. ONE latent row
// per token (kv_lora_rank + qk_rope_head_dim wide), num_kv_heads == 1, and NO
// separate V: the factor 2 every other attention spec carries is absent.
int64_t MLAAttentionSpec::real_page_size_bytes() const {
  // THE BRANCH ORDER IS LOAD-BEARING, and it is the one thing a careless port
  // gets wrong. `DeepseekV4Attention.get_kv_cache_spec` builds its spec with
  // `kv_quant_mode=get_kv_quant_mode("fp8_ds_mla")`
  // (`vllm/models/deepseek_v4/attention.py:644`), and `get_kv_quant_mode` maps
  // ANY string starting with "fp8" to FP8_PER_TENSOR
  // (`kv_cache_interface.py:70-71`). So every real DeepSeek-V4 MLA spec carries
  // a non-NONE quant mode, and the guard below would throw on all of them.
  // Upstream returns from this branch (`:398-405`) before it ever reads
  // `kv_quant_mode`; so do we.
  if (is_fp8_ds_mla(cache_dtype_str)) {
    if (is_deepseek_v4(model_version)) {
      // head_size stays SEMANTIC (512); the bytes are determined here.
      return static_cast<int64_t>(storage_block_size()) * kFp8DsMlaV4TokenBytes;
    }
    // NOTE: block_size, not storage_block_size — upstream's V3.2 line has no
    // compression in it (`:404-405`).
    return static_cast<int64_t>(block_size) * kFp8DsMlaV32TokenBytes;
  }
  if (needs_deferred_quant_math(kv_quant_mode)) {
    // Upstream's INT4 per-token-head layout (`:406-407`) is still OUT OF SCOPE;
    // throw loudly rather than silently mis-size. Unchanged from before
    // KV-DSV4-MULTICACHE W1: a spec with no `fp8_ds_mla` marker takes this
    // path exactly as it did.
    throw std::runtime_error(
        "MLAAttentionSpec: kv_quant_mode != NONE page-size math (int4 "
        "per-token-head) is out of scope; the fp8_ds_mla layouts are selected "
        "by cache_dtype_str, not by the quant mode");
  }
  return static_cast<int64_t>(storage_block_size()) * num_kv_heads * head_size *
         static_cast<int64_t>(vt::SizeOf(dtype));
}

// Upstream kv_cache_interface.py:627-642. ONE vector instead of K + V
// (`vllm/models/deepseek_v4/compressor.py:193`), so `head_size` alone and NOT
// `head_size + head_size_v` — and off `storage_block_size`, not `block_size`.
int64_t SlidingWindowMLASpec::real_page_size_bytes() const {
  // Same branch order, and for the same reason, as MLAAttentionSpec above:
  // `DeepseekV4SWACache` passes `kv_quant_mode=get_kv_quant_mode(cache_dtype)`
  // (`vllm/v1/attention/backends/mla/sparse_swa.py:100`), which is
  // FP8_PER_TENSOR whenever this branch is the right one.
  if (is_deepseek_v4(model_version) && is_fp8_ds_mla(cache_dtype_str)) {
    return static_cast<int64_t>(storage_block_size()) * kFp8DsMlaV4TokenBytes;
  }
  // Upstream asserts the version set before the element formula (`:634-636`).
  VT_CHECK(!model_version.has_value() || is_deepseek_v4(model_version),
           "SlidingWindowMLASpec: unsupported model version");
  if (needs_deferred_quant_math(kv_quant_mode)) {
    // The FlashInfer contiguous bf16 arm reaches the element formula with
    // kv_quant_mode NONE. An fp8 arm that did NOT set cache_dtype_str would
    // need the per-token-head scale bytes upstream adds in
    // `unpadded_page_size_bytes` (`kv_cache_interface.py:202-211`), which this
    // port defers for every spec in this file. Refuse rather than mis-size.
    throw std::runtime_error(
        "SlidingWindowMLASpec: kv_quant_mode != NONE page-size math is "
        "deferred (T1) unless cache_dtype_str selects the fp8_ds_mla layout");
  }
  return static_cast<int64_t>(storage_block_size()) * num_kv_heads * head_size *
         static_cast<int64_t>(vt::SizeOf(dtype));
}

int64_t SlidingWindowSpec::real_page_size_bytes() const {
  if (needs_deferred_quant_math(kv_quant_mode)) {
    throw std::runtime_error(
        "SlidingWindowSpec: kv_quant_mode != NONE page-size math is deferred "
        "(T1)");
  }
  const int64_t last_dim =
      static_cast<int64_t>(head_size) + static_cast<int64_t>(head_size_v);
  return static_cast<int64_t>(block_size) * num_kv_heads * last_dim *
         static_cast<int64_t>(vt::SizeOf(dtype));
}

int SlidingWindowSpec::max_admission_blocks_per_request(
    int max_num_batched_tokens, int max_model_len) const {
  const int num_tokens = std::min(
      sliding_window - 1 + max_num_batched_tokens, max_model_len);
  return (num_tokens + block_size - 1) / block_size + 1;
}

int ChunkedLocalAttentionSpec::max_admission_blocks_per_request(
    int max_num_batched_tokens, int max_model_len) const {
  const int num_tokens = std::min(
      attention_chunk_size + max_num_batched_tokens, max_model_len);
  return (num_tokens + block_size - 1) / block_size;
}

int KVCacheSpec::max_num_blocks_per_req(int max_len) const {
  // kv_cache_interface.py:197-207 @ e126687a9a.
  return (max_len + block_size - 1) / block_size;
}

int MambaSpec::max_num_blocks_per_req(int max_len) const {
  // kv_cache_interface.py:896-905 @ e126687a9a. The runner passes
  // max_len == max_model_len (no encoder-decoder), so the `all` arm's
  // cdiv(max_memory_usage_bytes, page_size_bytes) =
  // cdiv(max_model_len, block_size) + k (:884-887) is the `align` arm's value.
  //
  // `none` (:893-894, :905) is upstream's 1 + k. That is correct upstream only
  // because upstream builds a `none` group at block_size = max_model_len
  // (model_executor/models/config.py:657), where MambaManager claims at most
  // cdiv(num_tokens + block_size * k, block_size) = 1 + k blocks. This tree
  // builds it at the attention block size (ISSUE-LOCAL-01M36XJNF0TRNZBH7GYCW756AQ),
  // where the same claim reaches cdiv(max_model_len, block_size) + k. The row
  // must hold the claim, so every mode returns cdiv(max_len, block_size) + k,
  // which is exactly upstream's 1 + k whenever block_size >= max_len.
  return (max_len + block_size - 1) / block_size + num_speculative_blocks;
}

int64_t MambaSpec::page_size_bytes() const {
  if (shapes.size() != dtypes.size()) {
    throw std::runtime_error(
        "MambaSpec: shapes and dtypes must have the same length");
  }
  int64_t page_size = 0;
  for (size_t i = 0; i < shapes.size(); ++i) {
    int64_t numel = 1;
    for (int64_t dim : shapes[i]) {
      numel *= dim;
    }
    page_size += numel * static_cast<int64_t>(vt::SizeOf(dtypes[i]));
  }
  if (page_size_padded.has_value()) {
    if (*page_size_padded < page_size) {
      throw std::runtime_error(
          "MambaSpec: page_size_padded must be >= computed page_size");
    }
    return *page_size_padded;
  }
  return page_size;
}

bool KVCacheConfig::has_mamba_layers() const {
  for (const auto& group : kv_cache_groups) {
    if (dynamic_cast<const MambaSpec*>(group.kv_cache_spec.get()) != nullptr) {
      return true;
    }
  }
  return false;
}

int64_t KVBytesPerBlock(const KVCacheConfig& config) {
  // Heterogeneous-KV path (Gemma-4 G1b): one attention spec per non-GDN layer,
  // GDN/linear layers left null. Mirror the runner's per-layer allocation.
  if (!config.per_layer_attn_specs.empty()) {
    int64_t bytes = 0;
    for (const auto& spec : config.per_layer_attn_specs) {
      if (spec != nullptr) {
        bytes += spec->page_size_bytes();
      }
    }
    return bytes;
  }
  // Uniform path: sum each attention group's page over the layers it covers.
  // Mamba/GDN groups do not scale with the block count in our runner, so they
  // are skipped (dynamic_cast to AttentionSpec fails for MambaSpec).
  int64_t bytes = 0;
  for (const auto& group : config.kv_cache_groups) {
    const auto* attn =
        dynamic_cast<const AttentionSpec*>(group.kv_cache_spec.get());
    if (attn == nullptr) {
      continue;
    }
    bytes += attn->page_size_bytes() *
             static_cast<int64_t>(group.layer_names.size());
  }
  return bytes;
}

// See kv_cache_interface.h for the argument. Body moved verbatim from
// `gpu/runner.cpp`'s `LayerIndexOfName`.
std::optional<int64_t> KVCacheLayerIndexOfName(std::string_view name) {
  constexpr std::string_view kSep = ".layers.";
  const size_t at = name.find(kSep);
  if (at == std::string_view::npos) return std::nullopt;
  size_t i = at + kSep.size();
  const size_t start = i;
  int64_t value = 0;
  while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
    value = value * 10 + (name[i] - '0');
    if (value > (1 << 20)) return std::nullopt;  // not a layer index
    ++i;
  }
  if (i == start) return std::nullopt;                         // ".layers.mixer"
  if (i < name.size() && name[i] != '.') return std::nullopt;   // ".layers.5x"
  return value;
}

void ResolveKVCacheGroupLayerNames(
    KVCacheConfig& config, int64_t num_hidden_layers,
    const std::vector<std::string>& layer_types) {
  // Nothing to name. A caller with no layer count cannot be given one here, and
  // inventing one is how the placeholder count became a layer count in the
  // first place.
  if (num_hidden_layers <= 0) return;

  // A registry that already publishes REAL names knows more than the fallback
  // classification below: NemotronH's `layer_types` is EMPTY and its MoE blocks
  // register no attention module at all, so overwriting it would re-introduce
  // exactly the 52-against-6 mis-classification #810 removed. One resolvable
  // name anywhere is the whole config's answer, which also makes this
  // idempotent.
  for (const KVCacheGroupSpec& group : config.kv_cache_groups) {
    for (const std::string& name : group.layer_names) {
      if (KVCacheLayerIndexOfName(name).has_value()) return;
    }
  }

  bool has_mamba_group = false;
  for (const KVCacheGroupSpec& group : config.kv_cache_groups) {
    if (group.kv_cache_spec != nullptr &&
        group.kv_cache_spec->kind() == KVCacheSpecKind::kMamba) {
      has_mamba_group = true;
    }
  }

  // The runner's own `is_gdn` fallback predicate, and nothing else. A dense
  // model (no Mamba group) and a hybrid whose config does not spell
  // `layer_types` both classify every layer as attention here, which is what the
  // runner does with them too.
  std::vector<std::string> recurrent;
  std::vector<std::string> attention;
  for (int64_t l = 0; l < num_hidden_layers; ++l) {
    const size_t idx = static_cast<size_t>(l);
    const bool is_gdn = has_mamba_group && idx < layer_types.size() &&
                        layer_types[idx] == "linear_attention";
    if (is_gdn) {
      recurrent.push_back("model.layers." + std::to_string(l) + ".linear_attn");
    } else {
      attention.push_back("model.layers." + std::to_string(l) +
                          ".self_attn.attn");
    }
  }

  bool target_named = false;
  bool draft_named = false;
  for (KVCacheGroupSpec& group : config.kv_cache_groups) {
    if (group.kv_cache_spec == nullptr) continue;
    const KVCacheSpecKind kind = group.kv_cache_spec->kind();
    if (kind == KVCacheSpecKind::kMamba) {
      group.layer_names = recurrent;
      continue;
    }
    if (dynamic_cast<const AttentionSpec*>(group.kv_cache_spec.get()) ==
        nullptr) {
      continue;  // not an attention group and not recurrent: leave it alone.
    }
    if (!group.is_eagle_group && !target_named) {
      group.layer_names = attention;
      if (config.mtp_draft_shares_target_pages) {
        group.layer_names.push_back("model.layers." +
                                   std::to_string(num_hidden_layers) +
                                   ".self_attn.attn");
      }
      target_named = true;
    } else if (!draft_named) {
      // The speculative draft layer. Upstream registers the MTP head as one
      // extra decoder layer at index num_hidden_layers
      // (`qwen3_5_mtp.py:105-112`), and the runner allocates exactly one buffer
      // for it before breaking out of its search.
      group.layer_names = {"model.layers." +
                           std::to_string(num_hidden_layers) +
                           ".self_attn.attn"};
      draft_named = true;
    } else {
      // The runner allocates NO buffer for a further attention group, so the
      // honest weight is zero. Unreachable for every registry shipping today.
      group.layer_names.clear();
    }
  }
}

namespace {

// The ONE arithmetic statement W3 makes about block sizing, written where it can
// be read beside the specs it rewrites: an fp8 KV element is 1 byte where bf16
// is 2 (`torch_utils.py:38-40` maps every fp8 CacheDType to `torch.uint8`), and
// `AttentionSpec::real_page_size_bytes` is linear in `vt::SizeOf(dtype)`, so the
// page halves. If this and the store ever disagree about the element size the
// result is wrong tokens rather than a crash, so the storage dtype is asserted
// here rather than assumed.
void RetypeAttentionSpec(AttentionSpec& spec, const ResolvedCacheDType& resolved,
                         float k_scale, float v_scale) {
  // KV-DSV4-MULTICACHE W1 (#1960) landed the fp8_ds_mla PAGE FORMULA on
  // `MLAAttentionSpec` and `SlidingWindowMLASpec`, and it landed nothing on the
  // store or the read. So the refusal stays and only its reason narrows:
  // accepting `--kv-cache-dtype fp8_ds_mla` here would size every MLA page at
  // 584 bytes per token while the attention block still writes a bf16 latent
  // into it, which is wrong tokens rather than a crash. `ParseCacheDType`
  // refuses the string one level up (`kv_cache_dtype.h:87-90`); this is the
  // second door. Owed to W5, the wave that lands the read and write side.
  //
  // THE MESSAGE USED TO SAY "requesting it here", AND ON THE PATH THAT ACTUALLY
  // FIRES, NOTHING REQUESTED ANYTHING (#2455). DeepSeek-V4 reaches this on
  // `--kv-cache-dtype auto` through `vllm-cli`, which has no such flag to pass:
  // `MakeDeepseekV4KVCache` PUBLISHES fp8_ds_mla specs at `vt::DType::kI8`,
  // mirroring upstream where `use_fp8_ds_mla_layout` is `ClassVar[bool] = True`
  // (`attention.py:140`), and the early-out above needs
  // `spec.dtype == resolved.storage`, which `auto` resolves to the model's bf16.
  // So the model's own factory declares a layout this function then refuses, and
  // an operator reading the old text went looking for a flag they never set.
  // Naming both possible causes is the difference between one grep and a
  // measured investigation.
  VT_CHECK(dynamic_cast<const MLAAttentionSpec*>(&spec) == nullptr,
           "cache_dtype: an MLA KV cache has its own quantized page formula "
           "upstream (fp8_ds_mla, kv_cache_interface.py:398-410). W1 landed "
           "that page formula but no fp8_ds_mla store or read, so a page sized "
           "for it would hold bytes nothing writes. EITHER --kv-cache-dtype "
           "asked for a non-auto dtype, OR (DeepSeek-V4, #2455) the model's own "
           "KV factory published fp8_ds_mla specs and no flag was passed at "
           "all: check the spec's cache_dtype_str before you check the command "
           "line. The fix for the second case is the fp8_ds_mla store and read "
           "(KV-DSV4-MULTICACHE W5), not a wider guard here -- this model's "
           "paged store writes f32 latents through vt::ConcatAndCacheMla, which "
           "is 3.5x the 584-byte page this layout declares");
  if (resolved.is_fp8) {
    VT_CHECK(resolved.storage == vt::DType::kI8,
             "cache_dtype: an fp8 KV cache stores 1 byte per element "
             "(vt::DType::kI8); any other storage dtype would size the block "
             "for one element width and store another");
    VT_CHECK(vt::SizeOf(resolved.storage) == 1,
             "cache_dtype: the fp8 KV storage dtype must be exactly 1 byte");
    VT_CHECK(resolved.fp8_kind == vt::Fp8KVCacheDataType::kFp8E4M3,
             "cache_dtype: only fp8_e4m3 is implemented on the KV store and "
             "read (fp8_e5m2 compute is a named later brick, KV-FP8 W5)");
    VT_CHECK(k_scale > 0.0F && v_scale > 0.0F,
             "cache_dtype: an fp8 KV cache needs k_scale/v_scale > 0");
  } else {
    // Explicit float overrides. bfloat16 is the storage dtype the KV path
    // already writes; float16 parses (the CacheDType surface is mirrored in
    // full) but no model's attention block casts K/V to f16 before the store,
    // so it is refused by name instead of reaching a dtype mismatch deep in an
    // op wrapper.
    VT_CHECK(resolved.storage == vt::DType::kBF16,
             "cache_dtype: only 'auto', 'bfloat16', 'fp8' and 'fp8_e4m3' are "
             "wired to the KV store; 'float16' parses but no attention block "
             "casts K/V to f16 before the store (KV-FP8 W3, owed)");
  }
  spec.dtype = resolved.storage;
  spec.fp8_kind = resolved.fp8_kind;
  spec.k_scale = k_scale;
  spec.v_scale = v_scale;
}

}  // namespace

void ApplyCacheDType(KVCacheConfig& config, const ResolvedCacheDType& resolved,
                     float k_scale, float v_scale) {
  const auto retype = [&](AttentionSpec& spec) {
    // `auto` MEANS "USE THE DTYPE THE MODEL RESOLVED", SO THERE IS NOTHING TO
    // APPLY TO ANY SPEC (KV-DSV4-MULTICACHE W8 slice 6, #2455).
    //
    // Upstream's `auto` is "if auto, use model data type" (`config/cache.py:76`),
    // and for DeepSeek-V4 the model's own `_resolve_dsv4_kv_cache_dtype` WRITES
    // `cache_config.cache_dtype = "fp8_ds_mla"` back onto the cache config and
    // returns `torch.uint8` (`vllm/models/deepseek_v4/attention.py:89-119`,
    // driven by `use_fp8_ds_mla_layout`, a `ClassVar[bool] = True` at `:140`).
    // So on `auto` the resolved cache dtype IS whatever the factory chose, and
    // rewriting it here is not resolution but an override the operator never
    // asked for.
    //
    // THIS SUBSUMES THE EXISTING EARLY-OUT BELOW RATHER THAN COMPETING WITH IT.
    // Every factory that builds its spec with `ResolveKvCacheDType()` already
    // satisfies `spec.dtype == resolved.storage` on `auto` and returned there,
    // so the ONLY specs that ever reached the retype on `auto` are the ones
    // whose factory deliberately chose a different dtype. DeepSeek-V4 publishes
    // THREE such shapes, and a narrower predicate keyed on `cache_dtype_str`
    // caught only the first:
    //   - the C4A/C128A latent and the 43-entry SWA group: `kI8` WITH
    //     `cache_dtype_str == "fp8_ds_mla"`;
    //   - the indexer key cache: `kI8` with NO `cache_dtype_str`, because
    //     upstream passes none (`attention.py:669-684`) and it takes the
    //     element formula at a byte-derived width of 132;
    //   - the three compressor state caches: `kF32` with no `cache_dtype_str`
    //     (`compressor.py:168-200`, where upstream ASSERTS f32).
    // The last two are `SlidingWindowMLASpec`/`MLAAttentionSpec` shapes that the
    // MLA refusal below either throws on or, worse, MISSES: a `kF32` compressor
    // state satisfies `storage == kBF16` in the float branch and was silently
    // retyped to bf16, halving a page whose allocation is already sized in f32.
    //
    // IT IS A CHANGE TO RESOLUTION, NOT A WIDER GUARD. `RetypeAttentionSpec` is
    // untouched and still refuses every EXPLICIT override: `--kv-cache-dtype
    // fp8` on this topology is refused exactly as before, because the operator
    // is then asking for a different page format instead of delegating the
    // choice. Widening that guard would instead let a packed 584-byte page be
    // written as though it were float — the 3.5x overrun this wave exists to
    // prevent.
    if (resolved.is_auto) {
      return;
    }
    // NOTHING TO APPLY, and this is the whole default path. "auto" resolves to
    // the model dtype, which is exactly what every KV-cache factory already
    // built the spec with (`ResolveKvCacheDType()`), so the write would set the
    // field to the value it holds. Returning first keeps the default load
    // byte-identical AND keeps the refusals below out of its way: an MLA model
    // on `--kv-cache-dtype auto`, and the `VT_KV_CACHE_F32=1` A/B cache, must
    // both keep loading, and neither is asking for anything to change.
    if (!resolved.is_fp8 && spec.dtype == resolved.storage &&
        spec.fp8_kind == vt::Fp8KVCacheDataType::kAuto) {
      return;
    }
    RetypeAttentionSpec(spec, resolved, k_scale, v_scale);
  };
  for (auto& group : config.kv_cache_groups) {
    auto* attn = dynamic_cast<AttentionSpec*>(group.kv_cache_spec.get());
    // MambaSpec (recurrent conv/SSM state) is NOT the KV cache: upstream sizes
    // it from its own `mamba_cache_dtype`/`mamba_ssm_cache_dtype` knobs
    // (`config/cache.py:131-138`) and `--kv-cache-dtype` never touches it.
    if (attn == nullptr) continue;
    retype(*attn);
  }
  // The heterogeneous-per-layer seam (Gemma-4) allocates from these instead of
  // the group spec, so a rewrite that skipped them would half-size the pool and
  // leave the layers writing full-width bytes into it.
  for (auto& spec : config.per_layer_attn_specs) {
    if (spec == nullptr) continue;
    retype(*spec);
  }
}

}  // namespace vllm::v1
