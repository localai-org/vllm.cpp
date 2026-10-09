// Ported from: vllm/v1/kv_cache_interface.py @ e24d1b24
//
// Scope (M1.3 Task 1): the KV-cache SPEC hierarchy + config wrappers the
// KVCacheManager (M1.3 Task 2-4) and the scheduler (M1.4) build on. The gate
// models interleave GDN (mamba-state) layers and full-attention layers, so the
// two T0 specs ported here are `FullAttentionSpec` (paged K+V blocks) and
// `MambaSpec` (the SSM+conv recurrent state). Behavioral only: no CUDA, no
// model — a spec is pure allocation metadata (block_size + page_size_bytes).
//
// HIERARCHY SHAPE: upstream is a frozen-dataclass hierarchy
//   KVCacheSpec (base) -> AttentionSpec -> FullAttentionSpec
//                                      -> SlidingWindowSpec
//                                      -> ChunkedLocalAttentionSpec
//                      -> MambaSpec
// mirrored here with a virtual base class + derived structs. `page_size_bytes`
// is the abstract per-page byte cost; `AttentionSpec` splits it into
// `real_page_size_bytes` (the raw K+V bytes) + a `page_size_padded` override,
// exactly as upstream, so FullAttentionSpec can override only the raw formula.
//
// PAGE-SIZE FORMULAS (kv_quant_mode == NONE, the only mode the gate models use):
//   AttentionSpec.real_page_size_bytes =
//       2 * block_size * num_kv_heads * head_size * dtype_size          (K + V)
//   FullAttentionSpec.real_page_size_bytes =
//       block_size * num_kv_heads * (head_size + head_size_v) * dtype_size
//       (identical to the base when head_size_v == head_size, the default; the
//        split exists so MLA/asymmetric-V layers can differ)
//   MLAAttentionSpec.real_page_size_bytes =
//       storage_block_size * num_kv_heads(1) * head_size * dtype_size
//       (NO factor 2, NO separate V — see the MLAAttentionSpec comment)
//   SlidingWindowSpec.real_page_size_bytes = the same asymmetric K+V formula;
//       its window changes allocation lifetime, not bytes per stored token.
//   ChunkedLocalAttentionSpec inherits AttentionSpec's symmetric K+V formula;
//       its fixed chunk changes allocation lifetime, not bytes per token.
//   MambaSpec.page_size_bytes =
//       sum_i( prod(shapes[i]) * dtype_size(dtypes[i]) )    (SSM + conv state)
//   page_size_bytes = page_size_padded if set (>= real), else real.
//
// FIELD/METHOD NAMES are kept EXACTLY as upstream (snake_case: block_size,
// num_kv_heads, head_size, head_size_v, page_size_bytes, real_page_size_bytes,
// page_size_padded, num_speculative_blocks, mamba_cache_mode, layer_names,
// kv_cache_spec, num_blocks, kv_cache_tensors, kv_cache_groups, shared_by,
// has_mamba_layers, needs_kv_cache_zeroing) — this overrides the repo's usual
// CamelCase convention per the plan's 1:1 mandate.
//
// DEFERRED (marked stubs / omissions; the gate models never exercise these, and
// later units fill them in without reshaping the base):
//   - SinkFullAttentionSpec / RSWASpec / EncoderOnlyAttentionSpec /
//     CrossAttentionSpec / UniformTypeKVCacheSpecs / TQFullAttentionSpec /
//     HiddenStateCacheSpec (T1/T2) — omitted. The base stays extensible
//     (virtual page_size_bytes/kind/storage_block_size).
//     `MLAAttentionSpec` LANDED (MLA campaign W1): allocation metadata only —
//     no MLA math, no MLA ops, no model consumes it yet.
//     `SlidingWindowMLASpec` LANDED (KV-DSV4-MULTICACHE W1, #1960), together
//     with the four DeepSeek-V4 fields on `MLAAttentionSpec` and
//     `ApplyAlignmentPadding`. Allocation metadata only: NOTHING outside
//     tests constructs either shape, because `gpu/runner.cpp:577-597` drops a
//     group whose kind matches no branch with no diagnostic, so publishing the
//     DeepSeek-V4 topology before the runner can carry it would allocate a
//     SUBSET of it in silence. W2 owns publication, W3 owns consumption.
//   - kv_quant_mode != NONE page-size math (per-token-head scale bytes, nvfp4 /
//     int4 packed layouts): the `kv_quant_mode` field is carried for fidelity
//     but any non-NONE mode throws in real_page_size_bytes (T1). The gate models
//     run unquantized KV cache.
//   - max_memory_usage_bytes(VllmConfig): needs the (not-yet-ported) VllmConfig
//     + parallel/cache configs. Omitted with this note (T1); no Task-1 test uses
//     it. page_size_bytes is the piece the KVCacheManager needs now.
//   - merge() / is_uniform_with_collection() / copy_with_new_block_size(): the
//     spec-grouping helpers (used when building KVCacheConfig from raw specs).
//     Omitted (T1); Task 1 constructs groups/configs directly, as upstream's
//     own tests do.
//   - MambaSpec.mamba_type (MambaAttentionBackendEnum): omitted — it selects a
//     CUDA attention backend, irrelevant to page-size accounting; porting the
//     backend enum is out of Task-1 scope.
#ifndef VLLM_V1_KV_CACHE_INTERFACE_H_
#define VLLM_V1_KV_CACHE_INTERFACE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/v1/kv_cache_dtype.h"
#include "vt/dtype.h"
#include "vt/fp8_kv.h"

namespace vllm::v1 {

class RecurrentPrefixSnapshotIndex;

// Upstream KVCacheSpecKind (str Enum). Ported as a plain enum; the T1/T2 kinds
// are listed for fidelity; kFullAttention / kSlidingWindow / kMamba are
// currently produced here.
enum class KVCacheSpecKind {
  kFullAttention,
  kMlaAttention,
  kSlidingWindow,
  kSlidingWindowMla,
  kMamba,
  kChunkedLocalAttention,
  kSinkFullAttention,
  kEncoderOnlyAttention,
  kCrossAttention,
  kUnknown,
};

// KVQuantMode moved to `kv_cache_dtype.h` (included above) with
// KV-NVFP4-TURBO W0 (#2620), which gave it the `get_kv_quant_mode` map it never
// had here. One definition, in the header that resolves the CacheDType string.
// Only kNone participates in the ported page-size math (see header note); the
// nvfp4 page formula is owed, `.agents/specs/nvfp4-kv-cache.md` `## Owed`.

// A base class for specifying the KV cache format of one layer.
// (Upstream: @dataclass(frozen=True) KVCacheSpec.)
struct KVCacheSpec {
  explicit KVCacheSpec(int block_size) : block_size(block_size) {}
  virtual ~KVCacheSpec() = default;

  // Number of tokens in a block.
  int block_size;

  // The size of a page with `block_size` tokens in bytes.
  virtual int64_t page_size_bytes() const = 0;

  // Upstream property `storage_block_size` (defaults to block_size).
  virtual int storage_block_size() const { return block_size; }

  // The spec kind (upstream: get_kv_cache_spec_kind(spec)).
  virtual KVCacheSpecKind kind() const = 0;

  // The number of block-table entries one request needs, i.e. the row length
  // of the worker's block table for this group. (Upstream
  // KVCacheSpec.max_num_blocks_per_req, kv_cache_interface.py:197-207 @
  // e126687a9a: cdiv(max_len, block_size).) AttentionSpec's override
  // (:432-435) divides by block_size * decode_context_parallel_size; this tree
  // has no DCP, so it is this base value.
  virtual int max_num_blocks_per_req(int max_len) const;
};

// (Upstream: @dataclass(frozen=True, kw_only=True) AttentionSpec.)
// Not instantiated directly (kind() stays pure); FullAttentionSpec is the T0
// concrete attention spec.
struct AttentionSpec : KVCacheSpec {
  AttentionSpec(int block_size, int num_kv_heads, int head_size, vt::DType dtype,
                KVQuantMode kv_quant_mode = KVQuantMode::kNone,
                std::optional<int64_t> page_size_padded = std::nullopt,
                bool indexes_kv_by_block_stride = false)
      : KVCacheSpec(block_size),
        num_kv_heads(num_kv_heads),
        head_size(head_size),
        dtype(dtype),
        kv_quant_mode(kv_quant_mode),
        page_size_padded(page_size_padded),
        indexes_kv_by_block_stride(indexes_kv_by_block_stride) {}

  int num_kv_heads;
  int head_size;
  vt::DType dtype;
  KVQuantMode kv_quant_mode;
  std::optional<int64_t> page_size_padded;
  bool indexes_kv_by_block_stride;

  // KV-FP8 W3 — the fp8 INTERPRETATION of a 1-byte (`vt::DType::kI8`) storage
  // dtype, plus the per-tensor dequant scales. ADDITIVE and default-inert: every
  // existing producer constructs a float spec and leaves these at kAuto/1.0, so
  // nothing about the bf16 default changes.
  //
  // WHY THE SPEC AND NOT A SIDE TABLE. `dtype` alone cannot answer "which fp8",
  // because upstream stores every fp8 flavour as `torch.uint8`
  // (`torch_utils.py:38-40`) and carries the flavour on the layer instead. Our
  // vt ops take the flavour and the scales as ARGUMENTS, and the runner builds
  // its `PagedKvCache` view from this spec and nothing else — the header's own
  // rule that "the KV cache SPEC is the single source of truth for the storage
  // dtype" (`kv_cache_dtype.h`). Splitting the interpretation away from the
  // storage dtype is exactly how a half-sized block and a full-sized store come
  // to disagree, which is silent corruption rather than a crash.
  //
  // Written ONCE, by `ApplyCacheDType` below, after the model's factory has
  // built the spec. Nothing else may set them.
  vt::Fp8KVCacheDataType fp8_kind = vt::Fp8KVCacheDataType::kAuto;
  float k_scale = 1.0F;
  float v_scale = 1.0F;

  int64_t page_size_bytes() const override;

  // The raw (unpadded) K+V bytes per page. Overridden by FullAttentionSpec.
  virtual int64_t real_page_size_bytes() const;
};

// The two DeepSeek-V4 fields whose zero value is a division rather than a
// behavior. `compress_ratio == 0` raises ZeroDivisionError upstream, and the
// model never produces it because `DeepseekV4Attention` applies
// `max(1, config.compress_ratios[layer_id])`
// (`vllm/models/deepseek_v4/attention.py:205-212`). Refuse both here rather
// than divide by zero in `storage_block_size()` or `RoundUp`.
void CheckMlaCacheFields(int compress_ratio, std::optional<int> alignment);

// Upstream `round_up` (vllm/utils/math_utils.py:20-22):
// `((x + y - 1) // y) * y`. Throws for `y <= 0` rather than dividing by zero.
int64_t RoundUp(int64_t x, int64_t y);

// Upstream `_apply_alignment_padding` (vllm/v1/kv_cache_interface.py:345-351).
// When `alignment` is set, round the spec's REAL page up to a multiple of it
// and store the result in `page_size_padded` — but only when the rounding
// actually changes the number, exactly as upstream's `if padded_page_size !=
// actual_page_size` does. `page_size_padded` is already the field
// `page_size_bytes()` returns when set, so upstream's field is our field.
//
// CALL IT FROM THE MOST-DERIVED CONSTRUCTOR BODY, never from a base one.
// Upstream runs this from `__post_init__`, where `self` is already the final
// type. C++ has no equivalent: a virtual call made while a BASE constructor is
// running dispatches to the base ([class.cdtor]/4), so `real_page_size_bytes()`
// would resolve to the wrong formula. Called from the body of the leaf class's
// own constructor the dynamic type is that class and the dispatch is right.
// A future subclass that overrides `real_page_size_bytes` must therefore call
// this again from its own constructor body; upstream's `HiddenStateCacheSpec`
// (`kv_cache_interface.py:452`) is that case, and it is not ported.
void ApplyAlignmentPadding(AttentionSpec& spec, std::optional<int> alignment);

// When the hybrid allocator is disabled and the model mixes full + sliding
// window attention, sliding window is treated as full attention here (blocks
// allocated for all tokens) — hence sliding_window / attention_chunk_size are
// recorded on the full-attention spec. (Upstream FullAttentionSpec.)
struct FullAttentionSpec : AttentionSpec {
  // head_size_v defaults to head_size (upstream __post_init__). Pass
  // std::nullopt (the default) to inherit head_size.
  FullAttentionSpec(int block_size, int num_kv_heads, int head_size,
                    vt::DType dtype,
                    std::optional<int> head_size_v = std::nullopt,
                    KVQuantMode kv_quant_mode = KVQuantMode::kNone,
                    std::optional<int64_t> page_size_padded = std::nullopt,
                    bool indexes_kv_by_block_stride = false,
                    std::optional<int> sliding_window = std::nullopt,
                    std::optional<int> attention_chunk_size = std::nullopt,
                    bool non_causal = false)
      : AttentionSpec(block_size, num_kv_heads, head_size, dtype, kv_quant_mode,
                      page_size_padded, indexes_kv_by_block_stride),
        head_size_v(head_size_v.value_or(head_size)),
        sliding_window(sliding_window),
        attention_chunk_size(attention_chunk_size),
        non_causal(non_causal) {}

  int head_size_v;
  std::optional<int> sliding_window;
  std::optional<int> attention_chunk_size;
  bool non_causal;

  int64_t real_page_size_bytes() const override;
  KVCacheSpecKind kind() const override {
    return KVCacheSpecKind::kFullAttention;
  }
};

// The compressed-latent (Multi-head Latent Attention) paged cache.
// (Upstream: vllm/v1/kv_cache_interface.py:363 MLAAttentionSpec(FullAttentionSpec),
// page formula :380-398.)
//
// MLA stores ONE latent row per token per layer — `kv_lora_rank +
// qk_rope_head_dim` elements (512 + 64 = 576 for every DeepSeek variant and for
// Kimi Linear's MLA layers) — with `num_kv_heads == 1` and **no V tensor at
// all**: V is reconstructed on the fly from the same latent via `W_UV`. So the
// page formula DROPS the factor 2 that every other attention spec carries:
//
//   real_page_size_bytes = storage_block_size * num_kv_heads * head_size * es
//
// mirroring upstream `kv_cache_interface.py:406-410`. The two `fp8_ds_mla`
// special cases upstream guards above that line (`:398-405`: DeepSeek-V4 =
// 584 B/token off `storage_block_size`, V3.2 = 656 B/token off `block_size`)
// ARE ported as of KV-DSV4-MULTICACHE W1 (#1960); INT4 per-token-head
// (`:406-407`) still throws via the shared kv_quant_mode guard rather than
// silently mis-sizing.
//
// MLA-ness is a page-SIZE and tensor-SHAPE concern ONLY: upstream maps
// `MLAAttentionSpec -> FullAttentionManager` with
// `uniform_type_base_spec=FullAttentionSpec`
// (`vllm/v1/core/single_type_kv_cache_manager.py:1539`), and
// `vllm/v1/worker/gpu_model_runner.py` has **no `use_mla` branch at all** — so
// the block table, prefix caching and eviction are untouched. We mirror that by
// deriving from FullAttentionSpec and registering the same manager kind
// (see kv_cache_spec_registry.cpp).
//
// Upstream also asserts MLA and full-attention layers can NEVER share a KV
// group (`kv_cache_interface.py:277-279` and `:400-403`, the two `merge`
// asserts). We have not ported `merge()` (see the DEFERRED note above); the
// distinct `kind()` below is what a future `merge()` will key that assert on.
struct MLAAttentionSpec : FullAttentionSpec {
  // num_kv_heads is accepted for field fidelity but is 1 for MLA — upstream
  // states it in three places (`mla_attention.py:390`, `:1004-1009`,
  // `vllm/config/model.py:1270-1274` "When using MLA during decode it becomes
  // MQA") and `MLACommonBackend.get_kv_cache_shape` accepts and IGNORES the
  // argument (`mla_attention.py:1219`).
  //
  // The four trailing parameters are upstream's DeepSeek-V4 fields
  // (`kv_cache_interface.py:381-388`), appended AFTER the existing ones and
  // defaulted to upstream's own defaults. Every call site in the tree passes
  // three positional arguments, so all of them keep building the spec they
  // built before this row: `compress_ratio == 1` makes `storage_block_size()`
  // return `block_size`, and an unset `alignment` writes no padding.
  MLAAttentionSpec(int block_size, int head_size, vt::DType dtype,
                   int num_kv_heads = 1,
                   KVQuantMode kv_quant_mode = KVQuantMode::kNone,
                   std::optional<int64_t> page_size_padded = std::nullopt,
                   bool indexes_kv_by_block_stride = false,
                   std::optional<std::string> cache_dtype_str = std::nullopt,
                   std::optional<int> alignment = std::nullopt,
                   int compress_ratio = 1,
                   std::optional<std::string> model_version = std::nullopt)
      : FullAttentionSpec(block_size, num_kv_heads, head_size, dtype,
                          /*head_size_v=*/head_size, kv_quant_mode,
                          page_size_padded, indexes_kv_by_block_stride),
        cache_dtype_str(std::move(cache_dtype_str)),
        alignment(alignment),
        compress_ratio(compress_ratio),
        model_version(std::move(model_version)) {
    CheckMlaCacheFields(this->compress_ratio, this->alignment);
    // Most-derived constructor body: see ApplyAlignmentPadding's contract.
    ApplyAlignmentPadding(*this, this->alignment);
  }

  // Upstream `kv_cache_interface.py:383-387`, name for name. Upstream's own
  // comment: "DeepseekV4 only fields. Non-DeepseekV4 MLA models leave these at
  // defaults."
  std::optional<std::string> cache_dtype_str;
  std::optional<int> alignment;  // None = no padding.
  int compress_ratio;            // 1 = no compression.
  std::optional<std::string> model_version;

  // Upstream `:393-395`. A compressed cache stores ONE row per
  // `compress_ratio` tokens, so a page holds `block_size / compress_ratio`
  // rows rather than `block_size`.
  int storage_block_size() const override { return block_size / compress_ratio; }

  int64_t real_page_size_bytes() const override;
  KVCacheSpecKind kind() const override {
    return KVCacheSpecKind::kMlaAttention;
  }
};

// Sliding-window paged K+V cache. The compute path still applies the local
// attention mask; this spec controls the recycling-aware cache allocation and
// prefix policy. (Upstream SlidingWindowSpec.)
struct SlidingWindowSpec : AttentionSpec {
  // head_size_v defaults to head_size (upstream __post_init__).
  SlidingWindowSpec(int block_size, int num_kv_heads, int head_size,
                    vt::DType dtype, int sliding_window,
                    std::optional<int> head_size_v = std::nullopt,
                    KVQuantMode kv_quant_mode = KVQuantMode::kNone,
                    std::optional<int64_t> page_size_padded = std::nullopt,
                    bool indexes_kv_by_block_stride = false)
      : AttentionSpec(block_size, num_kv_heads, head_size, dtype, kv_quant_mode,
                      page_size_padded, indexes_kv_by_block_stride),
        sliding_window(sliding_window),
        head_size_v(head_size_v.value_or(head_size)) {}

  int sliding_window;
  int head_size_v;

  int64_t real_page_size_bytes() const override;

  // Per-request startup-admission bound used by both pool sizing and the
  // runtime full-sequence fit check. The +1 covers a window that starts in the
  // middle of a block.
  int max_admission_blocks_per_request(int max_num_batched_tokens,
                                       int max_model_len) const;

  KVCacheSpecKind kind() const override {
    return KVCacheSpecKind::kSlidingWindow;
  }
};

// Sliding-window attention with the MLA cache format. (Upstream
// `kv_cache_interface.py:611-642` `SlidingWindowMLASpec(SlidingWindowSpec)`.)
//
// DeepSeek-V4 needs this class for 105 of its 167 cache entries: the
// per-attention-layer sliding-window cache (43 entries,
// `vllm/v1/attention/backends/mla/sparse_swa.py:86-101`) and both
// compressor-state populations (41 + 21 entries,
// `vllm/models/deepseek_v4/compressor.py:188-200`).
//
// ITS PAGE IS NOT THE PARENT'S. `SlidingWindowSpec` sums `head_size +
// head_size_v` because it stores K and V; this one holds ONE vector instead of
// K + V (`compressor.py:193` says exactly that beside the constructor) and
// multiplies `head_size` alone, off `storage_block_size` rather than
// `block_size` (`:637-642`).
//
// WHY THE SLIDING-WINDOW PRECEDENT IN THIS TREE POINTS THE WRONG WAY. No
// registry here has ever built a `SlidingWindowSpec`: Gemma-3 records that the
// window is masked at the attention kernel and that a smaller window-sized
// cache is "a memory-only vLLM optimization not needed for correctness"
// (`gemma3_registry.cpp:105-109`). That reasoning does NOT carry to
// DeepSeek-V4. V4's SWA cache is not a smaller copy of a full cache — it is the
// ONLY cache on layers 0 and 1, it holds a different 128-token window of a
// different quantity from the compressed latent, and its 64-token page size is
// fixed by physical tensor sharing with the C4A blocks (`sparse_swa.py:76-83`).
//
// head_size_v: upstream's `__post_init__` here does NOT call
// `SlidingWindowSpec.__post_init__`, so upstream leaves `head_size_v` at None;
// ours inherits `head_size` from the parent constructor. No formula on this
// class reads `head_size_v`, so the two are behaviorally identical. Recorded so
// a reader diffing the classes does not take it for a port defect.
struct SlidingWindowMLASpec : SlidingWindowSpec {
  SlidingWindowMLASpec(
      int block_size, int num_kv_heads, int head_size, vt::DType dtype,
      int sliding_window,
      std::optional<std::string> cache_dtype_str = std::nullopt,
      std::optional<int> alignment = std::nullopt, int compress_ratio = 1,
      std::optional<std::string> model_version = std::nullopt,
      KVQuantMode kv_quant_mode = KVQuantMode::kNone,
      std::optional<int64_t> page_size_padded = std::nullopt,
      bool indexes_kv_by_block_stride = false)
      : SlidingWindowSpec(block_size, num_kv_heads, head_size, dtype,
                          sliding_window, /*head_size_v=*/std::nullopt,
                          kv_quant_mode, page_size_padded,
                          indexes_kv_by_block_stride),
        cache_dtype_str(std::move(cache_dtype_str)),
        alignment(alignment),
        compress_ratio(compress_ratio),
        model_version(std::move(model_version)) {
    CheckMlaCacheFields(this->compress_ratio, this->alignment);
    // Most-derived constructor body: see ApplyAlignmentPadding's contract.
    ApplyAlignmentPadding(*this, this->alignment);
  }

  // Upstream `:614-618`, the same four fields as `MLAAttentionSpec`.
  std::optional<std::string> cache_dtype_str;
  std::optional<int> alignment;
  int compress_ratio;
  std::optional<std::string> model_version;

  // Upstream `:623-625`.
  int storage_block_size() const override { return block_size / compress_ratio; }

  int64_t real_page_size_bytes() const override;
  KVCacheSpecKind kind() const override {
    return KVCacheSpecKind::kSlidingWindowMla;
  }
};

// Fixed-chunk local-attention paged K+V cache. Tokens before the current
// attention chunk are represented by null logical blocks and their physical
// pages are recycled. (Upstream ChunkedLocalAttentionSpec.)
struct ChunkedLocalAttentionSpec : AttentionSpec {
  ChunkedLocalAttentionSpec(
      int block_size, int num_kv_heads, int head_size, vt::DType dtype,
      int attention_chunk_size,
      KVQuantMode kv_quant_mode = KVQuantMode::kNone,
      std::optional<int64_t> page_size_padded = std::nullopt,
      bool indexes_kv_by_block_stride = false)
      : AttentionSpec(block_size, num_kv_heads, head_size, dtype, kv_quant_mode,
                      page_size_padded, indexes_kv_by_block_stride),
        attention_chunk_size(attention_chunk_size) {}

  int attention_chunk_size;

  // During chunked prefill, physical KV holds at most the current fixed chunk
  // plus the newly scheduled batch, clamped by max_model_len.
  int max_admission_blocks_per_request(int max_num_batched_tokens,
                                       int max_model_len) const;

  KVCacheSpecKind kind() const override {
    return KVCacheSpecKind::kChunkedLocalAttention;
  }
};

// The mamba/GDN recurrent-state spec. `shapes`/`dtypes` describe each state
// tensor in upstream order (e.g. conv state, then temporal/SSM state);
// page_size_bytes is the sum of their byte sizes — NOT the K+V paged formula.
// (Upstream MambaSpec.)
struct MambaSpec : KVCacheSpec {
  MambaSpec(int block_size, std::vector<std::vector<int64_t>> shapes,
            std::vector<vt::DType> dtypes,
            std::optional<int64_t> page_size_padded = std::nullopt,
            std::string mamba_cache_mode = "none",
            int num_speculative_blocks = 0)
      : KVCacheSpec(block_size),
        shapes(std::move(shapes)),
        dtypes(std::move(dtypes)),
        page_size_padded(page_size_padded),
        mamba_cache_mode(std::move(mamba_cache_mode)),
        num_speculative_blocks(num_speculative_blocks) {}

  std::vector<std::vector<int64_t>> shapes;
  std::vector<vt::DType> dtypes;
  std::optional<int64_t> page_size_padded;
  std::string mamba_cache_mode;
  int num_speculative_blocks;

  int64_t page_size_bytes() const override;
  KVCacheSpecKind kind() const override { return KVCacheSpecKind::kMamba; }

  // Upstream MambaSpec.max_num_blocks_per_req (kv_cache_interface.py:896-905
  // @ e126687a9a), with the `none` deviation recorded in the .cpp and in
  // .agents/specs/block-table-row-width.md.
  int max_num_blocks_per_req(int max_len) const override;
};

// How the workers should initialize a KV cache tensor. (Upstream KVCacheTensor.)
struct KVCacheTensor {
  int64_t size;                      // size of the KV cache tensor in bytes
  std::vector<std::string> shared_by;  // layer names sharing this tensor
  int64_t offset = 0;                // byte offset within a contiguous block
  int64_t block_stride = 0;          // bytes per block in a packed layout (0 = unpacked)
};

// A group of model layers that share the same KV cache block table; treated as
// one layer by the KV cache manager. (Upstream KVCacheGroupSpec.)
struct KVCacheGroupSpec {
  KVCacheGroupSpec(std::vector<std::string> layer_names,
                   std::shared_ptr<KVCacheSpec> kv_cache_spec,
                   bool is_eagle_group = false)
      : layer_names(std::move(layer_names)),
        kv_cache_spec(std::move(kv_cache_spec)),
        is_eagle_group(is_eagle_group) {}

  std::vector<std::string> layer_names;
  std::shared_ptr<KVCacheSpec> kv_cache_spec;
  bool is_eagle_group = false;

  // The attention backend that SERVES this group, named by the model's KV
  // factory. Upstream's `AttentionGroupKey.attn_backend`
  // (`vllm/v1/worker/gpu_model_runner.py:7170`), whose value comes from the
  // LAYER: `layers[layer_name].get_attn_backend()` at `:7150`. DeepSeek-V4's
  // compressor, sparse-SWA and indexer layers each name their own
  // (`compressor.py:189-190`, `sparse_swa.py:116-118`, `indexer.py:183-196`),
  // and that is the whole reason upstream can publish 4- and 8-token groups
  // while three of its dense backends refuse anything that is not a multiple of
  // 16.
  //
  // EMPTY means "resolve through the platform selector", which is what every
  // model in this tree did before this field existed and still does: the runner
  // reads the field and falls back to its existing lazy dense/MLA resolution
  // when it is empty, so every existing topology resolves byte-identically.
  //
  // Keyed per GROUP rather than per LAYER, which is exact for every group any
  // factory here publishes because each one is homogeneous. A group that needed
  // two backends inside it would need the finer key; nothing here prevents that.
  std::string attn_backend;
};

// The KV cache configuration of a model. (Upstream KVCacheConfig.)
struct KVCacheConfig {
  int num_blocks;
  std::vector<KVCacheTensor> kv_cache_tensors;
  std::vector<KVCacheGroupSpec> kv_cache_groups;

  // OPTIONAL per-layer attention KV spec (index == model layer index). This is
  // NOT an upstream field; it is the additive seam that lets a HETEROGENEOUS
  // per-layer head_dim model (Gemma-4: sliding layers head_dim=256, global
  // layers global_head_dim=512, same num_kv_heads) allocate each non-GDN
  // layer's paged KV from its OWN head_size/num_kv_heads/page_size instead of
  // the single uniform group spec.
  //
  // BYTE-NEUTRALITY CONTRACT: EMPTY for every uniform-KV model (all existing
  // models leave it default-constructed), and when empty the runner falls back
  // to the single group spec for every layer — byte-identical allocation, view,
  // indexing and kernel dispatch to before this field existed. When NON-empty
  // it MUST have exactly `num_hidden_layers` entries; entry [l] is the attention
  // spec for non-GDN layer l (and is ignored for a GDN/linear-attention layer,
  // which is still sized from its MambaSpec). The block table / KV manager /
  // scheduler are head_dim-independent (they key on num_blocks + block_size,
  // uniform across layers), so a single group + this per-layer allocation is
  // sufficient — no per-group block table is introduced.
  //
  // The default member initializer keeps every existing positional aggregate
  // initialization of KVCacheConfig valid (the field is optional) AND suppresses
  // -Wmissing-field-initializers, so this addition is source-compatible too.
  std::vector<std::shared_ptr<AttentionSpec>> per_layer_attn_specs = {};

  // Optional bounded, immutable recurrent prefix rows shared by scheduler and runner.
  std::shared_ptr<RecurrentPrefixSnapshotIndex> recurrent_prefix_snapshots = {};

  // Native EXL3 MTP uses the target's physical page IDs for a separate draft
  // KV buffer. The draft layer therefore belongs to the target attention
  // group (and its layer count/byte budget), not a second page allocator.
  // False retains the existing topology for other checkpoints.
  bool mtp_draft_shares_target_pages = false;

  // Upstream property has_mamba_layers.
  bool has_mamba_layers() const;
  // Upstream property needs_kv_cache_zeroing.
  bool needs_kv_cache_zeroing() const { return has_mamba_layers(); }
};

// The marginal device bytes the paged KV pool grows by per additional block
// (ROAD-V1-MEM M2). This is exactly the runner's own per-block allocation math
// (gpu/runner.cpp: each non-GDN layer allocates `num_blocks * page_bytes`),
// summed over every attention layer:
//   - when `per_layer_attn_specs` is populated (heterogeneous-KV models such as
//     Gemma-4), the sum runs over those per-layer specs (null entries — the
//     GDN/linear layers — contribute nothing);
//   - otherwise it runs over `kv_cache_groups`, each attention-spec group
//     weighted by the number of layers it covers.
// GDN / Mamba state does NOT scale with the block count in our runner (it is
// sized per running sequence slot, not per block), so Mamba groups contribute
// zero here — matching what the block budget actually buys. The divisor is thus
// group-aware and correct for dense, MLA (no K+V factor 2), sliding-window,
// heterogeneous-per-layer, and hybrid architectures alike. Throws only if a
// spec's own `page_size_bytes()` throws (deferred quantized-KV math).
int64_t KVBytesPerBlock(const KVCacheConfig& config);

// The model layer index encoded in an upstream-style KV layer name, or nullopt
// when the name carries no layer identity.
//
// Upstream's `KVCacheGroupSpec.layer_names` holds real module paths
// ("model.layers.5.self_attn.attn", "backbone.layers.12.mixer"); this returns
// the integer of the `.layers.<N>.` segment. It deliberately returns nullopt
// for a PLACEHOLDER group name — "fa", "gdn", "mla", "kda", "fa_draft",
// "encoder" — because such a name names no layer at all, and that is the
// discriminator `ResolveKVCacheGroupLayerNames` and the runner's
// `GroupLayerMask` both key on.
//
// Lived in `gpu/runner.cpp`'s anonymous namespace until
// FIX-KV-GROUP-LAYER-COUNT needed the same parse in the sizing path. One
// function rather than two copies, because a second derivation of one rule is
// the thing that can disagree.
std::optional<int64_t> KVCacheLayerIndexOfName(std::string_view name);

// FIX-KV-GROUP-LAYER-COUNT (#1963, #1966). Replace PLACEHOLDER group names with
// the real per-layer names the runner's own allocation classification implies,
// so that every consumer weighting a group by `layer_names.size()` weights it
// by the layers the runner will actually allocate for.
//
// WHY THIS EXISTS. `layer_names` is upstream's per-layer name list
// (`kv_cache_utils.py:1208-1210` appends every layer sharing a spec object), and
// upstream bounds its allocation with `max(len(group.layer_names) ...)`
// (`:1399`) over that list, dividing the budget by the same count it multiplies
// the allocation by (`:1005-1008`, `:1409-1416`). Thirty-three of our
// thirty-four registries publish a single placeholder string instead, so
// `KVBytesPerBlock` divided a byte budget by ONE layer's page while the runner
// allocated one buffer PER layer — 8.5 GiB for a 1 GiB budget on the 27B — and
// the #371 recurrent-state OOM guard read 0.90 GiB against a 43.40 GiB
// allocation.
//
// THE CLASSIFICATION IS THE RUNNER'S OWN, not a second derivation of the
// model's shape:
//   - a layer is recurrent iff the config has a Mamba group AND `layer_types`
//     is non-empty AND `layer_types[l] == "linear_attention"` — the predicate at
//     `gpu/runner.cpp`'s `is_gdn` fallback;
//   - the TARGET attention group is the FIRST non-eagle attention-kind group,
//     which is the runner's own first-wins selection, and it covers every
//     non-recurrent layer;
//   - a SECOND attention group is the speculative draft layer: exactly ONE
//     layer, at index `num_hidden_layers` (upstream's MTP head index,
//     `qwen3_5_mtp.py:105-112`), because the runner allocates exactly one draft
//     buffer and breaks;
//   - a THIRD or later attention group gets an EMPTY list, because the runner
//     allocates no buffer for it at all. Zero is the honest count. No registry
//     emits one.
//
// A REGISTRY THAT ALREADY PUBLISHES REAL NAMES IS NEVER OVERWRITTEN. If any
// group carries a name `KVCacheLayerIndexOfName` resolves, this returns with the
// config untouched. `NemotronHForCausalLM` is that case and it knows more than
// the fallback can — its `layer_types` is empty and its MoE blocks cache
// nothing, which is exactly what #810 fixed.
//
// Idempotent: a second call finds real names and returns.
void ResolveKVCacheGroupLayerNames(KVCacheConfig& config,
                                   int64_t num_hidden_layers,
                                   const std::vector<std::string>& layer_types);

// KV-FP8 W3 — HALF-SIZED KV BLOCKS. Rewrite every ATTENTION spec in `config` to
// the resolved KV storage dtype, then hand the fp8 interpretation and the
// per-tensor scales to the same specs.
//
// This is the single place the resolved `cache_dtype` becomes bytes, and it
// mirrors where upstream does it: `GPUModelRunner.__init__` resolves
// `self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(cache_config.cache_dtype,
// model_config)` (`gpu_model_runner.py:484-486`) and every attention spec is
// then built with `dtype=self.kv_cache_dtype`. `AttentionSpec.
// real_page_size_bytes` is linear in `get_dtype_size(self.dtype)`
// (`kv_cache_interface.py:204-218`), so an fp8 store dtype (1 byte) against
// bf16 (2 bytes) halves the page and, at a fixed byte budget, doubles the block
// count. Nothing else in the sizing chain needs to know.
//
// A NO-OP on the default path, deliberately: `resolved.storage` for "auto" is
// the model dtype the factory already used, and the function returns before it
// touches a spec unless the caller named a non-auto dtype. `MambaSpec` is never
// rewritten — recurrent state is not the KV cache and upstream keeps it on its
// own `mamba_cache_dtype` knob (`config/cache.py:131-138`).
//
// Refuses BY NAME rather than mis-sizing: an MLA spec (upstream's fp8 arm there
// is `fp8_ds_mla`, a different page formula, `kv_cache_interface.py:398-410`), a
// storage dtype no store is wired for, and an fp8 kind the kernels refuse.
void ApplyCacheDType(KVCacheConfig& config, const ResolvedCacheDType& resolved,
                     float k_scale, float v_scale);

}  // namespace vllm::v1

#endif  // VLLM_V1_KV_CACHE_INTERFACE_H_
