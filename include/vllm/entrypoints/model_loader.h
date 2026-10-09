// ORIGINAL packaging helper (no upstream mirror). Factors the model-load +
// engine-stack wiring shared by the OpenAI HTTP server (examples/server/main.cpp)
// and the C ABI (src/capi/vllm_c.cpp) into one place, so both drive the same
// LLMEngine construction path. The wiring itself mirrors the M1.8 LLMEngine
// __init__ (vllm/v1/engine/llm_engine.py @ e24d1b24) and the test harness
// (tests/vllm/v1/test_llm_engine.cpp); this file only owns the pieces + their
// lifetimes so the LLMEngine's by-reference seams stay valid.
#ifndef VLLM_ENTRYPOINTS_MODEL_LOADER_H_
#define VLLM_ENTRYPOINTS_MODEL_LOADER_H_

#include <algorithm>  // std::find — skipped_towers() dedup (#607 L3)
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/config/device.h"
#include "vllm/config/kv_transfer.h"
#include "vllm/config/multimodal.h"
#include "vllm/config/scheduler.h"
#include "vllm/config/speculative.h"
#include "vllm/model_executor/models/interfaces.h"  // #607 L3 kVisionTowerStageName
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/model_executor/models/qwen3_vl_vision.h"  // LOAD-GGUF-MMPROJ tower
#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/model_executor/models/qwen3_dspark.h"  // SPEC-DSPARK W5 draft bundle
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/async_scheduler.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/core.h"
#include "vllm/v1/engine/async_llm.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/llm_engine.h"
#include "vllm/v1/engine/output_processor.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/config/offload.h"
#include "vllm/config/weight_residency.h"
#include "vllm/v1/kv_offload/kv_connector.h"
#include "vllm/v1/structured_output/manager.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/backend.h"

namespace vllm::entrypoints {

// SPEC-DFLASH D5: the separately-loaded DFlash draft. Unlike MTP (in-target
// mtp.*), the z-lab DFlash draft is its own checkpoint; FromModelDir loads it
// (host-resident bf16 weights + the target-shared embed/lm_head + the resolved
// draft config + k) and the LoadedEngine hands a borrow to the runner. Owned by
// LoadedEngine, declared before runner_ so the borrow outlives it.
struct DflashDraft {
  // The plain DFlash draft. EMPTY when `dspark` is set: a DSpark draft owns its
  // backbone inside Qwen3DSparkWeights, so there is exactly one copy either way.
  vllm::Qwen3DFlashWeights weights;
  // SPEC-DSPARK W5: set only for method=="dspark". DSpark IS a DFlash draft plus
  // a Markov head (Qwen3DSparkModel(DFlashQwen3Model)), so it rides the same
  // struct, the same loader seam and the same runner wiring; the two extra fields
  // are the head itself and the query-block layout the checkpoint selects.
  std::unique_ptr<vllm::Qwen3DSparkWeights> dspark;
  bool sample_from_anchor = false;
  vllm::HfConfig config;
  int k = 0;
};

// REBIND the draft's embedding lookup onto the TARGET's table, so the two share
// one OwnedTensor and therefore one device upload (#1946).
//
// THE DEFECT. The draft owns no table; the loader read the target's a second
// time into `draft.weights.embed_tokens`. `ResidentWeight` caches its
// host->device upload on the OwnedTensor (the `if (!w.d_dev)` guard in
// `include/vllm/model_executor/models/dense_attn_block.h::ResidentWeight`),
// so two tensors are two allocations of identical
// bytes no matter how the HOST bytes are shared — which is why W9's (#1849)
// borrow-first host read did not close this. On the 27B that is BF16
// [248320, 5120] = 2,542,796,800 B held twice, out of GB10's unified pool.
//
// UPSTREAM does the same rebind by reference, at
// vllm/v1/worker/gpu/spec_decode/dflash/utils.py:64-74 @
// b389ac29465b33f9e9c534df221ea3c129e9793f: `del draft_inner.embed_tokens;
// draft_inner.embed_tokens = target_embed`. `_should_share`
// (eagle/utils.py:12-25 @ the same head) shares UNCONDITIONALLY when the draft
// declares no own table, which is this lane, and only compares bytes when it
// does. So no byte comparison here either — provenance already says these are
// the same tensor of the same file, and comparing 2.5 GB at load would fault in
// every page of a table the load has otherwise never read.
//
// WHAT IT REFUSES TO BIND, and why the check is not decorative. A GGUF target
// may KEEP its table F16 in place (`LoadEmbedAndHead`'s kKeepF16 arm) while the
// draft's shared read always produces BF16. Those are different bytes and must
// never be aliased, so a dtype/rank/shape disagreement leaves both tables
// standing — today's behaviour, correct tokens, no dedup — and says so on
// stderr rather than failing the load. A DSpark draft is skipped outright: its
// backbone owns its table by value and is a separate lane.
//
// EXPORTED, not file-local, for the reason W9 gave when it exported
// `LoadDflashSharedEmbedBf16`: a deletion mutation can remove this lever
// silently, so the gate has to reach the exact function production calls.
// Returns true when it rebound. Called from the ONE LoadedEngine constructor
// that every draft loader funnels into.
bool BindDflashDraftSharedEmbed(DflashDraft& draft, const LoadedModel& target);

// vLLM CacheConfig.gpu_memory_utilization's own default (vllm/config/cache.py:68
// @ 555967922). Named because EngineParams carries the knob as an optional, so
// "the caller chose nothing" and "the caller chose 0.92" are different states
// and only one of them has a value to fall back to.
inline constexpr double kDefaultGpuMemoryUtilization = 0.92;

// Knobs that size the engine stack. Zero/negative fields fall back to the
// documented defaults (see below), so a default-constructed EngineParams is
// valid.
struct EngineParams {
  int block_size = 32;     // KV block size (tokens/block).
  // KV pool sizing (ROAD-V1-MEM). Precedence mirrors vLLM's cache knobs, applied
  // in ResolveNumBlocks (model_loader.cpp):
  //   1. num_blocks > 0            -> used verbatim (vLLM num_gpu_blocks_override)
  //   2. kv_cache_memory_bytes > 0 -> num_blocks = kv_cache_memory_bytes /
  //                                   KVBytesPerBlock(kv_cfg) (absolute pool size,
  //                                   IGNORES gpu_memory_utilization, cache.py:189)
  //   3. otherwise                 -> the gpu_memory_utilization profile path,
  //                                   which needs a device profile run (M3, not
  //                                   yet implemented) and so falls back to 256.
  // num_blocks now defaults to 0 ("auto") so a caller can reach knob 2/3 without
  // an explicit override; the resolved fallback when nothing sizes the pool is
  // still 256, so the default path is byte-identical to before this field's
  // default changed.
  int num_blocks = 0;      // 0 => auto (resolved: override > bytes > 256).
  // Fraction of free device memory the whole engine may consume (weights +
  // activations + KV), mirroring vLLM CacheConfig.gpu_memory_utilization
  // (cache.py:68). Still INERT: knob 3 needs the M3 profile run.
  //
  // TRI-STATE (FIX-GPU-MEM-UTIL-INERT, #1165), mirroring enable_prefix_caching
  // below. nullopt means the caller never named a fraction and resolves to
  // vLLM's kDefaultGpuMemoryUtilization; a value means the caller CHOSE one.
  // The distinction exists because the value is discarded: ResolveNumBlocks
  // warns on a chosen value, and stays silent on a default nobody set. Before
  // this was tri-state the field defaulted to 0.92 and no surface could tell
  // the two apart, so the engine accepted `--gpu-memory-utilization 0.85`,
  // sized nothing, and said nothing.
  std::optional<double> gpu_memory_utilization = std::nullopt;
  // Absolute KV-pool size in bytes (0 = unset). When > 0 it sizes the block
  // count directly and IGNORES gpu_memory_utilization, mirroring vLLM
  // CacheConfig.kv_cache_memory_bytes (cache.py:182,189).
  int64_t kv_cache_memory_bytes = 0;
  // KV-cache STORAGE dtype, mirroring vLLM CacheConfig.cache_dtype
  // (config/cache.py:19-36,76) and its `--kv-cache-dtype` flag. "auto" (the
  // default) uses the model dtype and is byte-identical to before this field
  // existed; "fp8"/"fp8_e4m3" stores 1-byte fp8 K/V, which HALVES the bytes per
  // KV block and therefore doubles the pool at a fixed --kv-cache-memory.
  //
  // TWO-STAGE, exactly as upstream. `FromModelDir` RESOLVES this string once
  // against the checkpoint's own `kv_cache_quant_algo` before anything reads it
  // (`vllm::ResolveKvCacheDTypeString`, mirroring `arg_utils.py:1915-1918`), so
  // every consumer downstream sees an already-resolved value and "auto" there
  // means "nothing declared it either". An explicit value is never overridden by
  // the checkpoint (`torch_utils.py:380-381`).
  std::string kv_cache_dtype = "auto";
  int max_model_len = 0;   // 0 => config.max_position_embeddings.
  // max concurrent sequences. vLLM's default is 1024 (EngineArgs.max_num_seqs);
  // ours was 8, which put c8 EXACTLY on the batch ceiling so the 8th stream
  // could not co-batch -- measured as a throughput ratio that stayed flat to c4
  // then collapsed at c8. Raising to 32 recovers it (c8 51.66 -> 64.41 tok/s,
  // +24.7%, ratio vs vLLM flat ~0.74x instead of degrading to 0.59x).
  // NOT vLLM's 1024: max_num_seqs scales KV demand, and GB10's unified memory
  // has a narrow usable band (see gb10 OOM/thrash notes). 32 is the value
  // MEASURED clean at --gpu-memory-utilization 0.55; higher is unverified.
  int max_num_seqs = 32;
  // Per-step token budget (the chunked-prefill knob). 0 => the bounded PER-ARCH
  // default (see LoadedEngine::ResolveMaxNumBatchedTokens): dense arch 2048 flat
  // (vLLM's DEFAULT_MAX_NUM_BATCHED_TOKENS, vllm/config/scheduler.py:42 @
  // e24d1b24); MoE arch 8192 at max_num_seqs >= 32 else 4096 (GB10-tuned). The
  // budget does NOT scale with max_num_seqs, so a long/many-request prefill is
  // split across steps (enable_chunked_prefill is always true) and the per-step
  // GDN chunked-scan activation stays bounded regardless of concurrency. This is
  // the fix for the 27B 8x1024 conc-8 OOM: the old max_model_len*max_num_seqs
  // product let an 8x1024 (8192-token) prefill run in ONE step, blowing the GDN
  // prefill activation.
  int max_num_batched_tokens = 0;
  // Mirrors vLLM's tri-state --[no-]enable-prefix-caching resolution. Hybrid
  // and attention-free generation models default OFF at the parity pin;
  // decoder-only models default ON. An explicit value overrides the model
  // capability default.
  std::optional<bool> enable_prefix_caching = std::nullopt;
  // Scheduling policy (mirrors SchedulerConfig.policy / the vLLM
  // --scheduling-policy flag). Default fcfs. Set kPriority to schedule by
  // (priority, arrival_time); requests then carry a `priority` field. kLPM is
  // SGLang's cache-aware longest-prefix-match admission ordering
  // (ENG-SGLANG-BEHAVIOR-FLAG SW1) — output-neutral, meaningful only with prefix
  // caching ON (resolves to fcfs otherwise). Exposed on the C ABI as the v9
  // string field vllm_model_params.scheduling_policy = "lpm".
  vllm::SchedulerPolicy policy = vllm::SchedulerPolicy::kFCFS;

  // Jump-forward decoding toggle (ENG-SGLANG-BEHAVIOR-FLAG SW3; the token-unique
  // forced-run subset — see structured_output/jump_forward.h). Tri-state,
  // resolved once at construction by vllm::v1::JumpForwardEnabled(this):
  //   * nullopt (default) => OFF unless the VT_ENABLE_JUMP_FORWARD env override
  //     turns it on (the byte-identical default);
  //   * true  => ON  (unless VT_ENABLE_JUMP_FORWARD is set, which then overrides);
  //   * false => OFF (likewise overridable by the env var).
  // The env var, WHEN SET, always wins (mirrors the VT_ASYNC_SCHED convention).
  // Exposed on the C ABI as vllm_model_params.enable_jump_forward (ABI v10) and
  // on the server as --enable-jump-forward / --disable-jump-forward.
  std::optional<bool> enable_jump_forward = std::nullopt;

  // Model-level sliding-window kill switch, mirroring vLLM's
  // `ModelConfig.disable_sliding_window` (`vllm/config/model.py:248` @ pin
  // `5559679229`). Tri-state on the same convention as the field above:
  //   * nullopt (default) => sliding window ENABLED, the byte-identical default
  //     and upstream's own `= False`;
  //   * true  => every model's MODEL-LEVEL window is disabled;
  //   * false => explicitly enabled.
  // ONE switch for every model, because upstream's is one field on ModelConfig.
  // It replaced `VT_GEMMA2_SLIDING` and `VT_GEMMA3_SLIDING`, which gave the
  // control to two of the five families that have a window and to none of the
  // rest. Applied through `vllm::SetDisableSlidingWindow`, which the five
  // forwards read via `ResolveAttentionWindow`. Exposed on the C ABI as
  // vllm_model_params.disable_sliding_window (ABI v26) and on the server as
  // --disable-sliding-window / --enable-sliding-window.
  std::optional<bool> disable_sliding_window = std::nullopt;

  // KV-EXTERNAL-CACHE (LMCache): opt-in external KV-cache connector selection.
  // Empty/absent (default) == NO connector == byte-identical production engine
  // (mirrors vLLM's --kv-transfer-config being unset). When set with a
  // kv_connector name, LoadedEngine builds the connector via KVConnectorFactory,
  // injects the runner's resolved full-attention KV geometry into its
  // extra_config, and wires it to BOTH the scheduler (prefix lookup / prefill
  // shortcut) and the runner (worker-side KV store/load).
  std::optional<vllm::KVTransferConfig> kv_transfer_config = std::nullopt;

  // ENG-WEIGHT-OFFLOAD W0b: opt-in WEIGHT-offload configuration (distinct from
  // kv_transfer_config above, which offloads KV blocks). Absent (default) == no
  // offloading == the byte-identical engine. Validated at construction; the
  // offloader that consumes it is W2/W5, so today this is recorded and inert.
  std::optional<vllm::OffloadConfig> offload_config = std::nullopt;

  // ENG-RESIDENCY-CONFIG (#1110): the host-RAM -> DISK residency tier, parsed out
  // of the `vllm_cpp` key of the SAME `--offload-config` document `offload_config`
  // above comes from. Absent (default) == every knob resolves from the environment
  // and its built-in default, i.e. the byte-identical engine.
  //
  // It is a SEPARATE field rather than an arm of `OffloadConfig` because that
  // struct is a 1:1 transcription of `vllm/config/offload.py` and upstream has no
  // disk tier — see include/vllm/config/weight_residency.h for the whole argument,
  // the schema, and the env-beats-config precedence.
  //
  // FromModelDir installs it in its FIRST statement block, ahead of every path,
  // config and weight operation, because each knob it feeds is read through a
  // static that latches on first use.
  std::optional<vllm::WeightResidencyConfig> weight_residency = std::nullopt;

  // SPEC-MTP I5d-pre: opt-in speculative-decoding configuration. Empty/absent
  // (default) == NO speculation == byte-identical production engine (mirrors
  // vLLM's --speculative-config being unset). When set to an MTP config,
  // FromModelDir additionally loads the checkpoint's `mtp.*` draft weights
  // (LoadQwen3_5MTP) and retains them on the loaded model so the runner has a
  // typed path to build the draft (LoadedModel::BuildMtpDraft). The verify /
  // propose runner loop and --speculative-config CLI parsing are I5d; this field
  // only enables the seam. Non-safetensors (GGUF) checkpoints lack `mtp.*`, so an
  // MTP config over a GGUF source is rejected.
  std::optional<vllm::SpeculativeConfig> speculative_config = std::nullopt;

  // ARCH-ONE-SURFACE fold ROW 8: explicit device selection, the mirror of
  // vLLM's DeviceConfig.device (vllm/config/device.py). kAuto (default) keeps
  // the accelerator-first probe that has always selected the queue — the
  // byte-identical default. kCPU forces the CPU queue without consulting the
  // probe. The INTERNAL value Device::kNamedPlatform is the tag for the stable
  // PUBLIC/WIRE request whose value and name remain 2="cuda"; it resolves that
  // canonical name through the platform registry and fails LOUD when CUDA is
  // absent (never a silent fallback — an explicit device is assigned verbatim
  // upstream, device.py:61-66). Exposed on the C ABI as
  // vllm_model_params.device (ABI v14: 0=auto, 1=cpu, 2=cuda) and on the
  // server as --device.
  vllm::Device device = vllm::Device::kAuto;

  // ENG-MM-INPUT-PIPELINE wave L2 (#607): the per-modality multimodal input
  // limits, mirroring vLLM threading MultiModalConfig onto the model config
  // (arg_utils.py:1691-1692 -> ModelConfig -> multimodal_config). Exposed on the
  // server as --language-model-only / --limit-mm-per-prompt and on the C ABI as
  // vllm_model_params.language_model_only / .limit_mm_per_prompt (ABI v19).
  //
  // The DEFAULT is the pre-L2 behaviour byte for byte: an empty map with the
  // flag off resolves to 999 per modality (multimodal.py:331-333), which no
  // real request reaches. It is deliberately a VALUE and not an optional — a
  // "no config" state distinct from "the default config" would be a second
  // spelling of the same thing, and upstream has one.
  vllm::MultiModalConfig multimodal;

  // ── The SECOND GGUF file: a `clip` multimodal projector (row
  // `LOAD-GGUF-MMPROJ`, issue #821) ─────────────────────────────────────────
  //
  // A GGUF multimodal model ships as TWO files, and until this field existed
  // the projector had nowhere to arrive: `ModelSource` carries a VECTOR of
  // safetensors shards and exactly one `GgufFile*`, and `GgufFile::Open`'s
  // shard merge (`DetectSplit`) is about shards of ONE split, not a second,
  // differently-architected file.
  //
  // The spelling is llama.cpp's user-facing one (`--mmproj`), because that is
  // the flag every user of these artifacts already types, and it is EXPLICIT on
  // purpose. Auto-discovery of a sibling `mmproj*.gguf` is deliberately NOT
  // implemented: a directory holding two unrelated models would then silently
  // fuse them, and the failure would be a wrong-shaped model rather than an
  // error.
  //
  // Empty (the default) is byte-identical to the pre-row behaviour: no second
  // file is opened and no vision tower is built. NON-EMPTY against anything
  // that is not a `.gguf` FILE is REFUSED BY NAME, not ignored: a safetensors
  // checkpoint carries its vision tower in its own shards, so accepting the
  // flag there and dropping it would load a tower the user did not ask for and
  // silently discard the one they named. The refusal fires in `FromModelDir`
  // before any path or config I/O, and its message begins `--mmproj: a
  // multimodal projector attaches to a .gguf language file`.
  std::string mmproj_path;
};

// The shared queue-selection seam used by every LoadedEngine construction
// path. Exposed from this internal header so the explicit named-platform path
// can be gated with a distinctive registered platform/backend rather than a
// parallel pure-policy copy.
vt::Queue SelectQueueForModel(std::string_view architecture,
                              vllm::Device device);

// The device type `SelectQueueForModel` will pick.
//
// It exists because the load-time GGUF device-fit refusal (issue #1123) has to
// know the target device BEFORE any weight I/O, and the load's own queue is not
// created until after the weights are loaded. Throws for an explicitly named
// device that this build/process cannot serve, exactly as the queue selector
// does; the auto arm falls back to `kCPU` instead of throwing, also exactly as it
// does.
//
// The two agree because both arms run one implementation, and on the AUTO arm
// that implementation CREATES A QUEUE and destroys it. #1136 measured why the
// cheaper version was wrong: `SelectQueueForModel`'s auto arm falls back to CPU
// when `CreateQueue()` throws, so a resolver that only asked `CurrentPlatform()`
// answered `'cuda'` on a box where the load would run on CPU, and the fit refusal
// then rejected a checkpoint by naming a device nothing was going to run on.
//
// The cost of agreeing is one extra stream created and destroyed, and it is bounded
// by where this function is called: the load-time GGUF fit check is the only caller
// outside `SelectQueueForModel` itself, so a safetensors load pays nothing, an
// explicitly named device pays nothing (that arm creates no queue here), and an
// auto-arm GGUF load pays one `CreateQueue`/`DestroyQueue` pair. That is not free,
// and it is smaller than removing a working load.
vt::DeviceType ResolveModelDeviceType(std::string_view architecture,
                                      vllm::Device device);

// Clamp enabled native EXL3 XPU image serving before runner/state allocation.
// Extra clients wait in the existing scheduler; other engine scopes pass through.
int ResolveNativeQwen3_5MaxNumSeqs(
    int resolved, const HfConfig& config, std::string_view architecture,
    vt::DeviceType device, const vllm::MultiModalConfig& multimodal);

// Adjust only the enabled native EXL3 Qwen3.5 image encoder. Capacity is
// bookkeeping for lazily resident outputs; it does not allocate the cache.
void ConfigureNativeQwen3_5EncoderBudget(
    vllm::SchedulerConfig& scheduler, const HfConfig& config,
    std::string_view architecture, vt::DeviceType device,
    const vllm::MultiModalConfig& multimodal);

// Owns the full V1 engine stack (config + weights + tokenizer + Scheduler +
// runner -> Executor -> EngineCore; Input/OutputProcessor -> LLMEngine) for a
// registered model. The concrete weights/forward are held behind LoadedModel;
// the Scheduler / Executor / EngineCore / processors are arch-agnostic (they
// touch the runner only through ModelRunnerBase).
// Members are declared in dependency order so the LLMEngine's by-reference
// collaborators stay valid for this object's lifetime. NON-COPYABLE /
// NON-MOVABLE (the internal references would dangle) — always heap-hold behind a
// unique_ptr and hand out engine() by reference.
class LoadedEngine {
 public:
  // Build the stack from already-loaded model pieces. This is the shared seam:
  // FromModelDir() loads config/tokenizer/weights from disk then calls this, and
  // tests construct it directly with synthetic in-memory weights (no disk). The
  // pieces are moved into members that outlive every collaborator that
  // references them.
  // `mtp_weights` is the in-memory mirror of FromModelDir's `maybe_attach_mtp`
  // (SPEC-MTP-K-GT-1, #81): with a speculative config of method "mtp", the
  // checkpoint path reads the `mtp.*` tensors off the same shards and attaches
  // them so the runner can build the draft. A caller holding weights in memory
  // had no way to supply that head, so a synthetic spec engine could only ever
  // run with a NULL drafter, which is exactly the state a depth gate must not
  // mistake for working speculation. Defaulted to nullopt, so every existing
  // construction is unchanged.
  LoadedEngine(HfConfig config, Qwen3_5MoeWeights weights,
               tok::Tokenizer tokenizer, const EngineParams& params,
               std::optional<Qwen3_5MTPWeights> mtp_weights = std::nullopt);

  // DENSE-arch overload (27B). Identical stack; the runner runs the dense
  // Qwen3_5DenseModel::Forward over the dense weights instead of the MoE forward.
  LoadedEngine(HfConfig config, Qwen3_5DenseWeights weights,
               tok::Tokenizer tokenizer, const EngineParams& params,
               std::optional<Qwen3_5MTPWeights> mtp_weights = std::nullopt);

  // SPEC-DFLASH2 W3 (#1314): the DFLASH counterpart of the `mtp_weights`
  // overload above, and it exists for the identical reason that one gives. A
  // caller holding weights in memory had no way to supply a DFlash/DFlash2
  // draft, so `dflash_draft_` was null on every synthetic engine, the runner's
  // `set_dflash_draft` was never called, and `propose_drafts_block` -- the
  // PRODUCTION site where the grouped convolution, the candidate selector and
  // the refusal all live -- was unreachable from any test in this repository.
  // That is what spec `## Owed` O5 and O7 record for W1 and W2: their production
  // call sites were mutation-proven UNGATED, and the stated reason was that a
  // gate would need an on-disk target plus draft driven through the loader. It
  // does not: it needs this overload, which is the same seam FromModelDir uses
  // (it builds a DflashDraft and hands it to the private constructor below).
  //
  // The draft is loaded by the CALLER, exactly as `mtp_weights` is, and
  // everything downstream -- ResolveSpecConfig, the aux-multi-tap refusal, the
  // set_dflash_draft wiring, the whole propose loop -- is the production code
  // path unchanged.
  LoadedEngine(HfConfig config, Qwen3_5DenseWeights weights,
               tok::Tokenizer tokenizer, const EngineParams& params,
               std::unique_ptr<DflashDraft> dflash_draft);

  LoadedEngine(const LoadedEngine&) = delete;
  LoadedEngine& operator=(const LoadedEngine&) = delete;
  LoadedEngine(LoadedEngine&&) = delete;
  LoadedEngine& operator=(LoadedEngine&&) = delete;

  // Load config.json + tokenizer.json + *.safetensors from `model_dir` and build
  // the stack. Throws std::runtime_error on any load failure (bad path, missing
  // shards, unparseable config).
  //
  // KV-FP8 W3: this is where `params.kv_cache_dtype` is RESOLVED against the
  // checkpoint's own `kv_cache_quant_algo` (mirroring `arg_utils.py:1915-1918`);
  // the direct constructors below take the field verbatim because they are
  // handed in-memory weights and have no checkpoint directory to ask.
  static std::unique_ptr<LoadedEngine> FromModelDir(
      const std::string& model_dir, const EngineParams& params_in);

  // ── The `clip` mmproj vision tower (row `LOAD-GGUF-MMPROJ`, issue #821) ───
  //
  // Non-null exactly when `EngineParams::mmproj_path` named a loadable
  // `qwen3vl_merger` projector beside a `.gguf` language file AND at least one
  // of {image, video} was above limit 0. Since #607 L3 a perfectly loadable
  // projector under `--language-model-only`, or under
  // `--limit-mm-per-prompt '{"image":0,"video":0}'`, leaves this NULL: the read
  // is what the skip removes. Null therefore no longer distinguishes "no
  // projector was named" from "the projector was deliberately not read" —
  // `mmproj_tower_skipped_` is what carries that difference, and it is why the
  // flag exists (model_loader.cpp, the `SkipTowerForModalities` guard).
  //
  // The tower is
  // host-side f32, the shared `multimodal::Qwen3VLVisionWeights` that
  // `multimodal::Qwen3VLVisionForward` consumes and that the safetensors
  // reader (`LoadQwen3VLVisionWeights`) and the MiniMax-H3 encoder reader
  // (`LoadQwen3VLVisionFromGguf`) also fill.
  //
  // It lives on the ENGINE rather than inside an architecture's weights struct
  // because that is where the tower already lives on the safetensors side —
  // `LoadQwen3_5MoeVision` is a separate reader over the same shards, and no
  // `Qwen3_5*Weights` has a vision member — and because the projector is a
  // separate FILE the engine was handed, not part of the model checkpoint.
  const multimodal::Qwen3VLVisionWeights* vision_tower() const {
    return vision_tower_.has_value() ? &*vision_tower_ : nullptr;
  }
  // The geometry read from the projector's own `clip.*` metadata. Populated
  // whenever a projector file was named and accepted, INCLUDING the zero-limit
  // load that leaves `vision_tower()` null: `ClipMmprojVisionConfig` runs above
  // the skip, which is the construct half of construct-without-initialise and
  // is what lets a refusal still name what is missing. Default-constructed, and
  // meaningless, only when no `--mmproj` was given or the file was refused.
  const multimodal::Qwen3VLVisionConfig& vision_config() const {
    return vision_config_;
  }

  // Resolve the per-step token budget (max_num_batched_tokens) for chunked
  // prefill. An explicit EngineParams override wins; otherwise a PER-ARCH
  // default (see the definition in model_loader.cpp for the measurements):
  //  * DENSE arch: 2048 flat — vLLM's own scheduler default
  //    (DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048, vllm/config/scheduler.py:42 @
  //    e24d1b24). The dense prefill is expensive per token; a bigger budget
  //    lets one giant mixed step run several full prompts' prefill and starves
  //    every decode stream behind it.
  //  * MoE arch: the GB10-tuned concurrency-aware budget (8192 at
  //    max_num_seqs >= 32, else 4096) — the cheap A3B expert prefill wants the
  //    bigger chunk.
  // Invariants (SchedulerConfig.verify_max_model_len,
  // vllm/config/scheduler.py:87): result >= max_num_seqs; the tiny-model
  // (max_model_len * max_num_seqs) ceiling is preserved. Exposed for testing
  // the default policy without a disk load.
  static int ResolveMaxNumBatchedTokens(const EngineParams& params,
                                        int max_model_len, bool is_dense_arch);
  // The serving `max_model_len`, resolved AGAINST the KV pool that will hold it.
  // Mirrors vllm/v1/core/kv_cache_utils.py:2160-2174 @ 555967922, which runs
  // both halves at engine init:
  //   - `params.max_model_len <= 0` (the caller did not pin a length) ->
  //     auto_fit_max_model_len (kv_cache_utils.py:1967-2027): serve the
  //     checkpoint's context, reduced to what the pool holds.
  //   - `params.max_model_len > 0` (pinned) -> check_enough_kv_cache_memory
  //     (kv_cache_utils.py:751-788): THROW std::invalid_argument when the pool
  //     cannot hold one sequence that long, naming the sizes and the flags.
  // Either way the post-condition is the one the scheduler and the admission
  // check both rely on: a request of max_model_len tokens fits in KV. Without
  // it an over-long prompt is admitted, never allocates, and the engine spins
  // at model_executed=0 with an idle GPU (issue #83 M4; external PR #227).
  // Exposed, like ResolveMaxNumBatchedTokens above, for testing the policy
  // without a disk load.
  //
  // FIX-KV-POOL-MIN-FIT: the need of that one request is the SUM over every KV
  // cache group (each group's block table draws from this one pool), and the
  // null block is not usable (kv_cache_utils.py:2029-2058, :2304-2327 @
  // e126687a9a). `is_dense_arch` selects the `ResolveMaxNumBatchedTokens`
  // budget the scheduler receives, which sliding-window counts depend on.
  static int ResolveMaxModelLen(const EngineParams& params,
                                const HfConfig& config,
                                const vllm::v1::KVCacheConfig& kv_cfg,
                                int block_size, bool is_dense_arch);
  // The serving `max_num_seqs`, resolved AGAINST the recurrent-state budget the
  // KV pool affords (issue #1983). `max_num_seqs` sizes no allocation anywhere
  // in vLLM; here it multiplied the GDN state pool
  // (`gdn_state_slots_ = max_num_reqs * (num_spec + 1)`, one conv and one SSM
  // buffer per GDN layer, zeroed at construction) on an axis no memory flag
  // bounded. Upstream unifies the mamba page with the attention page and draws
  // both from ONE budgeted pool, so its recurrent allocation is a function of
  // available memory and never of the concurrency cap; this resolves the same
  // property through `vllm::v1::ComputeHybridKvBudget`.
  //
  // Attention-only models and pure-recurrent models both pass the configured
  // value through unchanged (the budget reports `kStateSeqsUnbounded`), so every
  // non-hybrid engine is byte-identical to before this existed. A reduction is
  // LOGGED, never silent, and never refuses: refusing belongs to the #371 state
  // guard, which still runs afterwards against this resolved value.
  //
  // Exposed, like the two resolvers above, so the policy is gateable without a
  // disk load.
  //
  // MODEL-MM-QWEN4-EXP W5L (#2031): `serves_one_sequence_per_step` is the
  // model's own declaration that its forward refuses `num_reqs > 1`. It clamps
  // the resolved value to 1 AFTER the budget clamp, because the two bound
  // different things and the smaller must win: the budget says how many
  // sequences the KV pool can hold, this says how many the forward can read.
  // Passing `false` reproduces the pre-W5L result for every caller and every
  // model, which is what makes the parameter additive rather than a new policy.
  static int ResolveMaxNumSeqs(const EngineParams& params,
                               const vllm::v1::KVCacheConfig& kv_cfg,
                               bool serves_one_sequence_per_step);
  static bool ResolveEnablePrefixCaching(const EngineParams& params,
                                         const ModelInfo& model_info);
  // ARCH-ONE-SURFACE ROW 8: the EXPLICIT arms of the device-selection policy
  // behind SelectQueue, factored pure over the "is the CUDA platform
  // registered" probe answer so the CPU tier can gate the whole matrix without
  // registering fake global platforms:
  //   * kCPU  -> vt::DeviceType::kCPU unconditionally — an explicit CPU ask
  //     never consults the accelerator probe, even when CUDA is registered;
  //   * kNamedPlatform -> the DeviceType returned by the canonical-name
  //     platform lookup, else
  //     THROWS std::runtime_error naming the device (fail LOUD; the mirror of
  //     vLLM assigning an explicit device verbatim and never substituting
  //     another — vllm/config/device.py:61-66);
  //   * kAuto is NOT resolved here (it resolves through the accelerator-first
  //     probe inside SelectQueue, byte-identical to pre-ROW-8) and throws
  //     std::invalid_argument if passed.
  // SelectQueue routes its explicit arms through THIS function, so the gate on
  // it pins the production policy, not a parallel copy.
  static vt::DeviceType ResolveExplicitDeviceType(
      vllm::Device requested,
      std::optional<vt::DeviceType> named_platform_type);

  vllm::v1::LLMEngine& engine() { return engine_; }
  // The multimodal input limits this engine was loaded with (#607 L2). ONE
  // config object per engine, whether it was set by the server's
  // --language-model-only / --limit-mm-per-prompt or by the C ABI's
  // vllm_model_params, so the two entry points cannot resolve a limit
  // differently. It outlives every consumer that borrows it (declared before
  // input_processor_), which is what lets the OpenAI chat seam hold a reference.
  const vllm::MultiModalConfig& mm_config() const { return mm_config_; }
  // #607 L3, the TOWER SKIP made observable from a production entry point. The
  // stage names of the towers this engine's model constructed WITHOUT loading,
  // because every modality they serve was at limit 0 — upstream's
  // `_tower_model_names` + `StageMissingLayer(stage_name, ...)`
  // (interfaces.py:141,279-282,298). EMPTY on every text model and on every
  // multimodal model loaded with a non-zero limit, so a caller can tell
  // "--language-model-only actually freed the tower" from "the flag was
  // accepted and did nothing", which is the distinction L2 could not make.
  // The `--mmproj` projector is NOT part of `model_` — it is a second file the
  // engine was handed — so its skip is added here rather than inside the model.
  // Deduplicated: both would report the same stage name, and a caller counting
  // freed towers must not see one tower twice.
  std::vector<std::string> skipped_towers() const {
    std::vector<std::string> names = model_->skipped_towers();
    if (mmproj_tower_skipped_) {
      const std::string stage(vllm::kVisionTowerStageName);
      if (std::find(names.begin(), names.end(), stage) == names.end())
        names.push_back(stage);
    }
    return names;
  }
  // ARCH-ONE-SURFACE ROW 6: whether the loaded model registration declares the
  // POOLING task class (is_pooling_model). The entrypoints dispatch BY TASK on
  // this — text-generation refuses on a pooling engine (naming vllm_embed /
  // /v1/embeddings) and embed refuses on a text engine — the mirror of vLLM
  // validating runner_type against the model class (config/model.py:607-613).
  bool is_pooling_model() const {
    return model_->registration().info.is_pooling_model;
  }
  // ENG-MM-INPUT-PIPELINE (#2475): the architecture the model registry resolved
  // this engine's model to, and that architecture's own multimodal declaration.
  // The OpenAI server's multimodal chat install dispatches on these two, exactly
  // as upstream reads `get_model_architecture(model_config)` and
  // `model_config.is_multimodal_model` before it looks up a processor
  // (vllm/multimodal/registry.py:110,178-180 @ 5559679229bc). The install used
  // to branch on the presence of `preprocessor_config.json`, which is not an
  // architecture and gave a second multimodal model Qwen3-VL's processor.
  std::string_view architecture() const {
    return model_->registration().architecture;
  }
  bool is_multimodal_model() const {
    return model_->registration().info.supports_multimodal;
  }
  // Lazily start W2's EngineCoreProc + output-handler threads. Once created,
  // online/server callers use this frontend rather than the synchronous
  // LLMEngine over the same scheduler/executor.
  vllm::v1::AsyncLLM& async_engine();
  const tok::Tokenizer& tokenizer() const { return tokenizer_; }
  const HfConfig& config() const { return config_; }
  int max_model_len() const { return max_model_len_; }
  bool prefix_caching_enabled() const { return prefix_caching_enabled_; }
  // ENG-SGLANG-BEHAVIOR-FLAG SW3: the jump-forward enable, resolved ONCE at
  // construction from EngineParams::enable_jump_forward + the
  // VT_ENABLE_JUMP_FORWARD env override (vllm::v1::JumpForwardEnabled). Exposed so
  // the enablement gate can assert the C-ABI/C++/flag toggle took effect.
  bool jump_forward_enabled() const { return jump_forward_enabled_; }
  const vllm::v1::GPUModelRunner& runner() const { return runner_; }
  // MODEL-GLINER25 Phase 4: direct access to the type-erased LoadedModel so
  // custom inference paths (the GLiNER2 NER pipeline: encoder → boundary head
  // → candidate decoder) can reach architecture-specific weights without going
  // through the text-generation/embedding forward. The model outlives every
  // borrower because it is declared before runner_ and every consumer below.
  LoadedModel& loaded_model() { return *model_; }
  const LoadedModel& loaded_model() const { return *model_; }
  // KV-FP8 W3: the RESOLVED KV-cache config — the block count the sizing knobs
  // produced and the group specs carrying the storage dtype `--kv-cache-dtype`
  // selected. Exposed so a gate reads what the loader actually sized instead of
  // re-deriving the arithmetic it is supposed to be checking.
  const vllm::v1::KVCacheConfig& kv_cache_config() const { return kv_cfg_; }
  // The block size actually in force, which is the requested one raised to the
  // architecture's floor. Not `EngineParams::block_size`.
  int block_size() const { return block_size_; }
  // The RESOLVED serving concurrency: `--max-num-seqs` after the recurrent-state
  // budget clamp (issue #1983). Equal to the configured value on every
  // attention-only model.
  int max_num_seqs() const { return max_num_seqs_; }

  // SPEC-MTP I5d: the speculative config as RESOLVED against the checkpoint
  // (ResolveSpecConfig), or empty on the non-speculative default. Exposed for
  // #2770: the server's PrometheusStatLogger has to register the spec_decode
  // metric families with the same k the scheduler builds its per-position
  // vectors from, and `params.speculative_config` is not that value — the
  // loader re-resolves it against the checkpoint's mtp_num_hidden_layers.
  const std::optional<vllm::SpeculativeConfig>& speculative_config() const {
    return resolved_spec_config_;
  }

  // KV-EXTERNAL-CACHE (LMCache): the wired external KV connector, or null when
  // none was configured. Exposed so the output-invariance gate can read the
  // prefill-tokens-saved / chunks-stored counters. Non-owning.
  vllm::v1::kv_offload::KVConnector* kv_connector() const {
    return kv_connector_.get();
  }

  // The async-scheduling enable-flip, resolved ONCE at construction (W3
  // ENG-ASYNC-SCHED). `async_scheduling_enabled()` is
  // AsyncSchedulingEnabled(SchedulerConfig::ResolveAsyncScheduling(
  // runner_.runner_supports_async())) — i.e. vLLM's default-ON-when-compatible
  // resolution (vllm/config/vllm.py:990-1038) gated on the MRV2 runner
  // advertising the async device path (VT_ASYNC_RUNNER), with the house
  // VT_ASYNC_SCHED=0 rollback applied. When ON, scheduler() is an AsyncScheduler
  // and max_concurrent_batches() is 2 (depth-2 batch queue, step_with_batch_queue);
  // OFF keeps the byte-identical synchronous Scheduler + depth-1. Since the
  // 2026-07-17 flip the production default (no env) resolves ON (VT_ASYNC_RUNNER
  // default ON → AsyncScheduler + mcb=2), mirroring vLLM; VT_ASYNC_RUNNER=0 or
  // VT_ASYNC_SCHED=0 roll it back in the same binary.
  bool async_scheduling_enabled() const { return async_scheduling_enabled_; }
  int max_concurrent_batches() const { return max_concurrent_batches_; }
  // The engine's scheduler (Scheduler or, under async scheduling, AsyncScheduler).
  // Exposed for the construction-resolution gate; production drives it via
  // engine()/async_engine().
  const vllm::v1::Scheduler& scheduler() const { return *scheduler_; }

  // ResolveAsyncEnabled: the construction-time resolution, factored out so the
  // CPU construction-matrix test can assert it directly over the
  // runner_supports_async x VT_ASYNC_SCHED matrix without a disk load. Applies
  // SchedulerConfig::ResolveAsyncScheduling then the VT_ASYNC_SCHED rollback env.
  // `is_pooling_model` (ARCH-ONE-SURFACE ROW 6) resolves async OFF for pooling
  // models (mirror of vllm/config/vllm.py:1068-1073); default false is the
  // byte-identical text path. `spec_decode_incompatible` (SPEC-DFLASH2 W7,
  // #1824) resolves async OFF for a speculative method upstream refuses
  // (vllm/config/vllm.py:1076-1087 — anything outside the Eagle-type family /
  // ngram_gpu / dspark); an Eagle-type speculator passes false and keeps
  // async scheduling ON, exactly as upstream.
  static bool ResolveAsyncEnabled(const vllm::SchedulerConfig& scheduler_config,
                                  bool runner_supports_async,
                                  bool is_pooling_model = false,
                                  bool spec_decode_incompatible = false);
  // SPEC-MTP I5d: finalize the entrypoint's SpeculativeConfig against the loaded
  // checkpoint. params.speculative_config carries the CLI method + optional user
  // k; this re-runs SpeculativeConfig::ResolveMtp with the checkpoint's
  // mtp_num_hidden_layers (from config.raw text_config, default 1) so n_predict
  // and the resolved k are correct. Returns nullopt when no speculation is
  // configured (the byte-identical production path). Non-Qwen3.5 or non-mtp
  // methods that reached here throw.
  static std::optional<vllm::SpeculativeConfig> ResolveSpecConfig(
      const EngineParams& params, const HfConfig& config);
  // SPEC-DSPARK-BLOCK-SIZE-GUARD (#1225): public beside the other Resolve*
  // statics of this class, which are the pure config-resolution helpers the
  // constructor calls and which tests already drive directly
  // (tests/vllm/entrypoints/test_loaded_engine_dense.cpp:347 for
  // ResolveMaxNumBatchedTokens). ResolveSpecConfig is the same kind of function
  // and was private only by accident of where it was added. The DSpark block
  // floor has to be reachable from a test that enters the loader's resolution
  // path rather than calling SpeculativeConfig::ResolveDspark by hand, which is
  // exactly the distinction .agents/reachability.md draws.

 private:
  // Type-erased constructor used by FromModelDir and the concrete-weight
  // compatibility overloads above.
  LoadedEngine(HfConfig config, std::unique_ptr<LoadedModel> model,
               tok::Tokenizer tokenizer, const EngineParams& params,
               vt::Queue* preselected_queue = nullptr,
               std::unique_ptr<DflashDraft> dflash_draft = nullptr,
               std::optional<multimodal::Qwen3VLVisionWeights> vision_tower =
                   std::nullopt,
               multimodal::Qwen3VLVisionConfig vision_config = {},
               // #607 L3: the `--mmproj` projector was constructed (geometry
               // resolved, file validated) and deliberately NOT read, because
               // every modality it serves was at limit 0. Passed rather than
               // re-derived here: a second evaluation of the predicate would
               // report the skip even if the loader had stopped honouring it.
               bool mmproj_tower_skipped = false);

  vllm::SchedulerConfig MakeSchedulerConfig(
      int max_model_len, int max_num_seqs, int max_num_batched_tokens,
      vllm::SchedulerPolicy policy = vllm::SchedulerPolicy::kFCFS) const;
  // Construct the concrete scheduler for the resolved mode: an AsyncScheduler
  // (async scheduling ON) or the synchronous Scheduler. Mirrors upstream
  // get_scheduler_cls (scheduler.py:180-189) selecting AsyncScheduler when
  // async_scheduling. Both take the same ctor arguments.
  static std::unique_ptr<vllm::v1::Scheduler> MakeScheduler(
      bool async_enabled, vllm::SchedulerConfig scheduler_config,
      vllm::v1::KVCacheConfig kv_cache_config, int block_size,
      int hash_block_size, bool enable_caching,
      vllm::v1::StructuredOutputManager* structured_output_manager,
      std::optional<vllm::SpeculativeConfig> speculative_config = std::nullopt);
  // The (scheduler_block_size, hash_block_size) pair this engine schedules and
  // hashes at, derived from the BUILT KV cache config. Mirrors upstream
  // EngineCore.__init__: `cache_config.block_size = min(g.kv_cache_spec
  // .block_size for g in kv_cache_groups)` (engine/core.py:335-338 @
  // e126687a9a) followed by `resolve_kv_cache_block_sizes` (core.py:158-160).
  // Both steps are needed: the first because an architecture may page a group
  // smaller than the engine's configured block size, the second because the
  // scheduler's alignment invariant and the prefix-hash granularity are
  // different quantities once groups disagree.
  static std::pair<int, int> ResolveSchedulerBlockSizes(
      const vllm::v1::KVCacheConfig& kv_cfg, int block_size,
      bool enable_prefix_caching);
  // SPEC-MTP I5d: build the KV-cache spec, widened for speculation when a spec
  // config is set (the extra GDN k+1 state slots + widened conv row + the
  // `fa_draft` full-attn group, MakeQwen3_5KVCacheSpec num_spec>0). With no spec
  // config it is exactly ModelRegistry::MakeKVCache (byte-identical).
 public:
  //
  // PUBLIC because it is the only entry to the KV geometry the engine actually
  // uses, and a gate that enters below it cannot see the engine stop calling it.
  // Measured 2026-08-31: deleting the block-size resolution from this function's
  // body left test_deepseek_v4_scaffold, test_loaded_engine_dense and
  // test_kv_cache_fp8_wiring all green. It is a pure function of its arguments
  // and holds no engine state, so publishing it widens nothing else.
  static vllm::v1::KVCacheConfig MakeKVCacheMaybeSpec(
      const LoadedModel& model, const HfConfig& config, int block_size,
      int num_blocks, const std::optional<vllm::SpeculativeConfig>& spec);
 private:

  // ROAD-V1-MEM M1: resolve the KV block count from the sizing knobs against the
  // model's own per-block byte geometry. `probe` is a KVCacheConfig already
  // built for this model (its num_blocks is ignored; only the group/page
  // geometry is read). Precedence: num_blocks override > absolute
  // kv_cache_memory_bytes / KVBytesPerBlock(probe) > the gpu_memory_utilization
  // profile path (M3, not yet implemented) which falls back to 256. Throws
  // VLLM_ERR-shaped std::runtime_error when an absolute byte budget is smaller
  // than a single KV block.
  //
  // FIX-GPU-MEM-UTIL-INERT (#1165): the profile path also WARNS on stderr when
  // it reaches knob 3 with an explicitly chosen gpu_memory_utilization, because
  // that is the point at which the chosen fraction is discarded. Knobs 1 and 2
  // return first, so a caller who sized the pool with --num-blocks or
  // --kv-cache-memory is never warned -- and under knob 2 that also mirrors
  // vLLM, which ignores the fraction there (cache.py:189).
  static int ResolveNumBlocks(const EngineParams& params,
                              const vllm::v1::KVCacheConfig& probe);
  // ROAD-V1-MEM M1: MakeKVCacheMaybeSpec with the block count resolved from the
  // sizing knobs (builds a probe config to read the per-block geometry, then
  // rebuilds at the resolved count only when it differs — the geometry itself is
  // block-count-independent, so this is at most one extra metadata build).
  static vllm::v1::KVCacheConfig MakeKVCacheResolved(
      const LoadedModel& model, const HfConfig& config, int block_size,
      const EngineParams& params,
      const std::optional<vllm::SpeculativeConfig>& spec);
  // KV-FP8 W3: turn the (already checkpoint-resolved) `params.kv_cache_dtype`
  // into the KV specs' storage dtype, fp8 interpretation and per-tensor scales.
  // Runs on the PROBE config before ResolveNumBlocks reads its geometry, which
  // is what makes an fp8 cache double the block count rather than halve the
  // pool. A no-op on the "auto"/bf16 default.
  static void ApplyResolvedCacheDType(const EngineParams& params,
                                      vllm::v1::KVCacheConfig& cfg);
  // The `kv_cache.py:150-156` uncalibrated-scale warning, once per LOAD.
  static void WarnUncalibratedKvScales(const EngineParams& params);
  // Ensure NONE_HASH is initialized before the scheduler/hasher are built
  // (upstream global init). Idempotent; runs as the first member initializer.
  static bool EnsureNoneHash();
  // Mirrors kernel_warmup.py::flashinfer_autotune: before any async/server
  // frontend starts, one maximum-token synthetic request runs under the NVFP4
  // all-bucket autotune scope. CUDA dense W4A4 only; CPU/other models are no-op.
  void WarmupKernels();

  bool hash_ready_;  // declared first: forces EnsureNoneHash() ahead of the rest.
  HfConfig config_;
  // LOAD-GGUF-MMPROJ: the `clip` projector's tower + its geometry. Empty on
  // every load that named no --mmproj, which is every load that existed before
  // this row. Nothing below borrows them, so their declaration position is
  // free; they sit beside config_ because they are, like it, checkpoint
  // metadata resolved once at load.
  std::optional<multimodal::Qwen3VLVisionWeights> vision_tower_;
  multimodal::Qwen3VLVisionConfig vision_config_;
  // #607 L3: the projector above was skipped rather than absent. Read by
  // skipped_towers(), which is why the engine-held tower needs its own flag: it
  // is not part of `model_`, so `model_->skipped_towers()` cannot see it.
  bool mmproj_tower_skipped_ = false;
  // SPEC-MTP I5d: the finalized speculative config (method/k/n_predict), or
  // nullopt on the production default path. Declared before model_/kv_cfg_/runner_
  // because the KV-cache widening, the draft build, the scheduler lookahead, and
  // the engine-core draft pull all read it. nullopt ⇒ every spec path is inert
  // and the engine is byte-identical to the pre-spec engine.
  std::optional<vllm::SpeculativeConfig> resolved_spec_config_;
  // The selected main queue is owned by this engine, while runner/model state
  // borrows it. Declared before those owners so destruction releases it last,
  // including on a partially constructed engine.
  struct MainQueueOwner {
    vt::Queue queue{};
    ~MainQueueOwner();
  } main_queue_owner_;
  // Concrete weights and model-specific runtime state behind the central
  // registry contract. Declared before runner_ so its borrow remains live.
  //
  // #1946 moved it AHEAD of dflash_draft_, and the order is now load-bearing in
  // both directions. Construction: the draft's embedding table is rebound onto
  // this model's table in the initialiser list, so the model must already
  // exist. Destruction: members die in reverse declaration order, so the draft
  // -- which now holds a BORROWED pointer into this model's weights -- dies
  // first.
  //
  // The old order was not ITSELF a use-after-free, and #1946 first said it was.
  // `DflashDraft` has no user-declared destructor and destroying it never
  // DEREFERENCES `weights.shared_embed_tokens`, so the dangling pointer was
  // never read. What the old order left was a WINDOW: anything later added to
  // that teardown -- a device-residency release, a flush, a log of the table --
  // would have read freed memory. The new order closes the window; it did not
  // fix a live bug.
  std::unique_ptr<LoadedModel> model_;
  // SPEC-DFLASH D5: the owned DFlash draft (null unless method=="dflash").
  // Declared before runner_ so the borrow the runner holds stays live for the
  // runner's whole lifetime. Set in the ctor body via runner_.set_dflash_draft.
  std::unique_ptr<DflashDraft> dflash_draft_;
  // KV-EXTERNAL-CACHE (LMCache): the owned external KV connector (null unless
  // EngineParams::kv_transfer_config selects one). Declared BEFORE runner_ /
  // scheduler_ so it outlives the non-owning pointers they hold to it. Built in
  // the ctor body once runner_ geometry is known; see model_loader.cpp.
  std::unique_ptr<vllm::v1::kv_offload::KVConnector> kv_connector_;
  tok::Tokenizer tokenizer_;
  // #607 L2: the engine's multimodal input limits, copied from EngineParams.
  // Declared here — before input_processor_ and before anything that can borrow
  // it — because the OpenAI chat seam holds a reference for the process
  // lifetime. A default-constructed one is the pre-L2 behaviour exactly.
  vllm::MultiModalConfig mm_config_;
  // kv_cfg_ is declared BEFORE max_model_len_: the serving length is resolved
  // AGAINST the KV pool (ResolveMaxModelLen auto-fits it down to what the pool
  // holds, or refuses an explicit --max-model-len the pool cannot serve), and
  // every consumer below — max_num_batched_tokens_, runner_, scheduler_,
  // input_processor_ — takes the already-resolved value. MakeKVCacheResolved
  // depends only on model_/config_/resolved_spec_config_, all declared above.
  // THE block size this engine pages at, resolved ONCE against the model's
  // declared floor. Declared before `kv_cfg_` so it initializes first, and used
  // at every site that previously spelled `params.block_size > 0 ? ... : 32` --
  // the KV config, the max-model-len fit, and the scheduler. Those three must
  // agree: a pool paged at 256 with a block table indexing at 32 is silent
  // corruption, not a refusal.
  int block_size_;
  vllm::v1::KVCacheConfig kv_cfg_;
  int max_model_len_;
  // Declared AFTER kv_cfg_ (it is resolved against the KV pool's recurrent-state
  // budget) and BEFORE runner_ / scheduler_ / structured_output_manager_, every
  // one of which takes the already-resolved value. See ResolveMaxNumSeqs.
  int max_num_seqs_;
  int max_num_batched_tokens_;
  bool prefix_caching_enabled_;
  // The two block sizes the SCHEDULER works in, as distinct from `block_size_`
  // (which is the size the KV pool was BUILT at). Declared after `kv_cfg_` and
  // `prefix_caching_enabled_` because both are inputs, and before `scheduler_`
  // and `block_hasher_`, which are the two consumers. See
  // ResolveSchedulerBlockSizes.
  int scheduler_block_size_;
  int hash_block_size_;
  // ENG-SGLANG-BEHAVIOR-FLAG SW3: jump-forward enable, resolved once from
  // EngineParams::enable_jump_forward + the VT_ENABLE_JUMP_FORWARD env override.
  // Depends only on params + env (no member deps), so its init order is free.
  bool jump_forward_enabled_;
  // runner_ is declared BEFORE the scheduler (W3): the async-scheduling flip is
  // resolved from runner_.runner_supports_async(), so the runner must be fully
  // constructed before async_scheduling_enabled_ / scheduler_ are initialized.
  vllm::v1::GPUModelRunner runner_;
  // Resolved once, in dependency order after runner_ (see the accessors above).
  bool async_scheduling_enabled_;
  int max_concurrent_batches_;
  // The engine's StructuredOutputManager (upstream EngineCore constructs one,
  // core.py:134), wired with the NATIVE backend factory over tokenizer_ (which
  // outlives it — declared earlier) so response_format / C-ABI structured
  // constraints actually gate decoding. Declared before scheduler_ /
  // engine_core_, which hold a pointer to it.
  vllm::v1::StructuredOutputManager structured_output_manager_;
  // Polymorphic scheduler: a plain Scheduler by default, an AsyncScheduler when
  // async_scheduling_enabled_. Heap-held so the concrete class is chosen at
  // construction; engine_core_ / async_engine_ share this one instance by
  // reference (*scheduler_).
  std::unique_ptr<vllm::v1::Scheduler> scheduler_;
  vllm::v1::Executor executor_;
  vllm::v1::EngineCore engine_core_;
  vllm::v1::InputProcessor input_processor_;
  vllm::v1::OutputProcessor output_processor_;
  vllm::v1::BlockHasher block_hasher_;
  vllm::v1::LLMEngine engine_;
  std::mutex async_engine_mutex_;
  std::unique_ptr<vllm::v1::AsyncLLM> async_engine_;
};

}  // namespace vllm::entrypoints

#endif  // VLLM_ENTRYPOINTS_MODEL_LOADER_H_
