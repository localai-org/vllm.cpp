// Ported from: vllm/model_executor/models/registry.py
//               @ e24d1b24fe96a56ba8b0d653efa076d03eb95d6c
// (_ModelInfo:746-796, _ModelRegistry:998-1082,
//  resolve_model_cls:1244-1296, global registry:1396-1404).
//
// Python-only lazy imports, subprocess inspection/cache, dynamic Transformers,
// terratorch, and out-of-tree runtime loading intentionally have no C++ runtime
// analogue. The ordered lookup, capability metadata, implemented-model factory
// hooks, and unsupported-architecture messages mirror the pinned registry.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/config/multimodal.h"
#include "vllm/multimodal/inputs.h"  // MultiModalFeatureSpec (the P2 encoder seam)
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm {

class GgufFile;
class SafetensorsFile;
struct ForwardLogits;
// BACKEND-DISTRIBUTED-TP TP-W1: the per-rank tensor-parallel handle (comm +
// rank), defined in tensor_parallel.h. Forward-declared so the shared registry
// seam does not pull the TP/communicator headers; a LoadedModel carries an
// optional BORROWED pointer to it, set only when tp_size>1 (null = single-GPU,
// byte-identical).
struct TensorParallel;
struct GdnStateCache;
struct PagedKvCache;
// One owned host tensor (qwen3_5_weights.h). Forward-declared for the same
// reason every other weights type here is: `LoadedModel::shared_embed_tokens`
// hands out a BORROWED pointer to one and never needs its layout (#1946).
struct OwnedTensor;
struct Qwen3_5DenseWeights;
struct Qwen3_5MoeWeights;
// SPEC-MTP I5d-pre: the checkpoint-owned MTP draft weights, the drafter's
// hidden-state carrier, and the draft model wrapper (all defined in
// qwen3_5_mtp.h). Forward-declared here so the type-erased LoadedModel can
// expose a typed path to the draft (BuildMtpDraft) and a hidden-state tap
// out-field WITHOUT pulling the MTP header — or any concrete weight type — into
// this shared registry seam. Inert unless speculative decoding is configured.
struct Qwen3_5MTPWeights;
struct Qwen3_5MTPHiddenStates;
struct Qwen3_5AuxTaps;
class Qwen3_5MTPModel;

namespace v1 {
struct CommonAttentionMetadata;
struct GDNAttentionMetadata;
}  // namespace v1

// Consumed subset of upstream _ModelInfo. Extend this POD as task rows land;
// these fields are enough to choose generation vs pooling/scoring and hybrid vs
// full-attention construction without importing a Python model class.
struct ModelInfo {
  bool is_text_generation_model = false;
  bool is_pooling_model = false;
  bool is_hybrid = false;
  bool has_inner_state = false;
  bool supports_multimodal = false;
  // Mirror of the SupportsTranscription protocol
  // (vllm/model_executor/models/interfaces.py:1110-1118):
  // `supports_transcription` marks an ASR-capable arch;
  // `supports_transcription_only` marks one with NO text-generation path
  // (interfaces.py:1118 `supports_transcription_only: ClassVar[bool]`), which
  // the entrypoints use to refuse-by-task: LoadedEngine::FromModelDir rejects
  // such an arch with a message pointing at the transcription entry points
  // (vllm_transcribe / /v1/audio/transcriptions), mirroring how vLLM excludes
  // "generate" from supported_tasks for them.
  bool supports_transcription = false;
  bool supports_transcription_only = false;
  std::string_view score_type = "bi-encoder";
};

// Type-erased checkpoint source passed to a registration's weight loader.
struct ModelSource {
  enum class Kind { kSafetensors, kGguf };

  static ModelSource FromSafetensors(
      const std::vector<SafetensorsFile>& shards,
      vt::Queue* load_queue = nullptr);
  // Shares ownership of the mmap'd shards so a loader may keep them alive past
  // the load (e.g. the Qwen3.6-35B MoE deferred-expert streaming that
  // materializes routed experts per layer during PrepareMarlinResident). The
  // borrowing `safetensors` pointer is set to the owned vector.
  static ModelSource FromSafetensorsOwned(
      std::shared_ptr<const std::vector<SafetensorsFile>> shards,
      vt::Queue* load_queue = nullptr);
  // `device` is the device the ENGINE resolved for this load
  // (`ResolveModelDeviceType`), and it has NO DEFAULT. Every GGUF weight
  // loader reaches its residency policy through this value; the parameter is
  // required so a new GGUF entry point cannot fall back to the accelerator
  // probe by saying nothing. See `.agents/specs/gguf-residency-resolved-device.md`.
  static ModelSource FromGguf(const GgufFile& gguf, vt::DeviceType device);

  Kind kind = Kind::kSafetensors;
  const std::vector<SafetensorsFile>* safetensors = nullptr;
  // Non-null only for FromSafetensorsOwned: shared ownership a loader can retain.
  std::shared_ptr<const std::vector<SafetensorsFile>> safetensors_owned;
  const GgufFile* gguf = nullptr;
  // Non-owning queue selected by the entrypoint. Dense plain-BF16 loaders may
  // stage completed layers here and the runner then reuses the same queue.
  vt::Queue* load_queue = nullptr;
  // ENG-MM-INPUT-PIPELINE wave L3 (#607): the engine's multimodal input limits,
  // BORROWED for the duration of one load. A loader that owns a tower asks
  // `SkipTowerForModalities` (models/interfaces.h, the mirror of
  // interfaces.py:293) whether to read that tower's tensors at all.
  //
  // Upstream reaches the same value through `vllm_config.model_config
  // .multimodal_config` inside the model's own `__init__`. Our loader seam
  // (ModelWeightLoader below) carries no vllm_config, and ModelSource is
  // already the per-load CONTEXT rather than only the checkpoint — `load_queue`
  // beside it is an engine-selected execution resource, not a property of the
  // file — so this is where the borrow belongs. The two alternatives are
  // recorded in specs/multimodal-track.md §1.5 L3 with the reason each was
  // rejected: a third ModelWeightLoader parameter rewrites every architecture's
  // signature to thread a value nearly all of them ignore, and a process-global
  // (WeightOffloader's shape) has no upstream analogue for this config.
  //
  // NULL means "no limits configured for this load": load everything, which is
  // byte-identical to pre-L3 and is what every non-engine caller gets.
  const MultiModalConfig* multimodal = nullptr;
  // ENG-GGUF-RESIDENCY-RESOLVED-DEVICE: the device the ENGINE RESOLVED for this
  // load, which is what the GGUF residency policy must read.
  //
  // It is NOT `vllm::platforms::CurrentPlatform().device_type()`. That probe
  // answers `kCUDA` on any process where the CUDA platform registered, while
  // `LoadedEngine::ResolveExplicitDeviceType` returns `kCPU` for an explicit
  // `--device cpu` "even on a CUDA-capable build/process". The two therefore
  // disagreed by construction, and the GGUF registries — the only production
  // loaders with no policy argument of their own — took the probe's answer.
  // `--device cpu` on a CUDA box got the CUDA residency policy, and on
  // `qwen4_exp` that is a refusal whose own remedy text is `--device cpu`.
  //
  // It belongs on `ModelSource` for the reason `load_queue` and `multimodal`
  // do: this struct is already the per-load CONTEXT and not only the
  // checkpoint. `FromGguf` REQUIRES it. The struct default below is only ever
  // reached by a safetensors source, and no safetensors path reads it — the
  // residency policy is GGUF-only.
  vt::DeviceType device = vt::DeviceType::kCPU;
  // MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm W4 (#2411): the SECOND FILE
  // of a GGUF multimodal load -- the `clip`-architecture projector the user
  // named with `--mmproj`, BORROWED for the duration of one load.
  //
  // It belongs here for the reason `multimodal` and `load_queue` do: this struct
  // is already the per-load CONTEXT and not only the checkpoint, and a family's
  // on-disk name map belongs inside that family's `load_weights`. The
  // alternative, reading the projector in `model_loader.cpp` and handing the
  // result down, needs a CONCRETE tower type in the loader. That is what pinned
  // the existing Qwen3-VL arm to `Qwen3VLVisionWeights` and left it unable to
  // carry a second architecture's tower at all.
  //
  // NULL on every load that named no `--mmproj`, and on the Qwen3-VL arm, which
  // is still read in `model_loader.cpp`. So every other architecture is
  // byte-identical. A source that carries one may only be handed to a
  // registration whose `ModelFactory::consumes_mmproj` is true;
  // `ModelRegistry::Load` refuses it otherwise, so a projector paired with a
  // language model that cannot use it costs a message rather than being dropped
  // in silence.
  const GgufFile* mmproj = nullptr;
  // The path the user typed. Quoted back by every projector refusal, so a
  // message names WHICH file was wrong. Empty when `mmproj` is null.
  std::string mmproj_path;
};

struct ModelFactory;
struct ModelRegistration;

// Opaque, lifetime-owning (or test-only borrowing) concrete model. The runner
// sees only this base and dispatches through its registration's function table.
class LoadedModel {
 public:
  virtual ~LoadedModel();

  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
  LoadedModel(LoadedModel&&) = delete;
  LoadedModel& operator=(LoadedModel&&) = delete;

  const ModelRegistration& registration() const { return registration_; }
  // Runtime capability rather than architecture metadata: GGUF/synthetic
  // instances of a W4A4-capable family may contain only BF16 weights.
  virtual bool uses_nvfp4_w4a4() const { return false; }

  // ENG-MM-INPUT-PIPELINE wave L3 (#607): the mirror of
  // `SupportsMultiModal._tower_model_names` (interfaces.py:141,298) together
  // with the `stage_name` each skipped stage carries (`:279-282`). The stage
  // names of the towers this model CONSTRUCTED WITHOUT LOADING because every
  // modality they serve had limit 0 — upstream's
  // `isinstance(model.visual, StageMissingLayer)`, expressed on a type-erased
  // base.
  //
  // EMPTY on every text model, and on every multimodal model loaded with a
  // non-zero limit. It exists so the skip is observable from a PRODUCTION entry
  // point (LoadedEngine::skipped_towers) rather than only by downcasting to a
  // concrete model class inside a test.
  virtual std::vector<std::string> skipped_towers() const { return {}; }

  // ARCH-ONE-SURFACE ROW 6: the model-owned Pooler of a POOLING model — the
  // mirror of upstream `VllmModelForPooling.pooler` (as_embedding_model wires
  // `self.pooler = DispatchPooler.for_embedding(...)`, adapters.py:248-257).
  // Non-null iff the registration's info.is_pooling_model; the GPU runner
  // builds its PoolingRunner over exactly this pooler (the mirror of
  // gpu/model_runner.py:368-369 `PoolingRunner(self.model)`). Default null:
  // every text-generation model is byte-identical.
  virtual const class Pooler* pooler() const { return nullptr; }

  // ── SPEC-MTP I5d-pre: typed access to the MTP draft, without breaking the
  //    type-erasure of this base. Only the concrete Qwen3.5 dense/MoE
  //    LoadedModel (which owns the target Qwen3_5DenseWeights/Qwen3_5MoeWeights)
  //    can retain the loaded `mtp.*` weights and construct the draft that shares
  //    the target's embed_tokens/lm_head (qwen3_5_mtp.h:61-66). Every default
  //    below is the non-MTP behavior: no support, and BuildMtpDraft yields null.
  //    All of this is unreachable on the production default path (no
  //    SpeculativeConfig ⇒ FromModelDir never attaches MTP weights), so the
  //    engine is byte-identical when speculative decoding is off. ──────────────

  // Whether this concrete model can hold MTP draft weights and build the draft.
  virtual bool supports_mtp_draft() const { return false; }

  // Whether this concrete model can capture the DFlash/DSpark AUX MULTI-TAP —
  // the residual stream at the draft's `target_layer_ids`, written into
  // `ForwardInput::aux_tap` as [T, H x taps]. Both block drafters CONDITION on
  // that tap, so a target without it produces a speculative engine that dies
  // mid-run ("missing target aux multi-tap") rather than at load. Found exactly
  // that way: the first DSpark e2e ran against classic-dense `Qwen3ForCausalLM`,
  // which has no tap, and the engine threw on the first propose. Default false;
  // the Qwen3.5/3.6 dense + MoE forwards override it.
  virtual bool supports_aux_multi_tap() const { return false; }

  // This target's embedding table, for a block drafter that SHARES it rather
  // than owning one (#1946).
  //
  // WHY THE BASE CARRIES THIS. A DFlash/DFlash2 draft ships no embed_tokens; it
  // runs the target's table over its own hidden states, which is why upstream
  // rebinds the MODULE by reference — `del draft_inner.embed_tokens;
  // draft_inner.embed_tokens = target_embed`
  // (vllm/v1/worker/gpu/spec_decode/dflash/utils.py:64-74 @ b389ac29465b33f9e9c534df221ea3c129e9793f).
  // Our loader read the target's tensor a SECOND time into the draft's own
  // OwnedTensor instead, and `ResidentWeight` caches its device upload on the
  // OwnedTensor (dense_attn_block.h `if (!w.d_dev)`), so the 27B's BF16
  // [248320, 5120] table — 2,542,796,800 B — was uploaded twice. On GB10 a
  // device allocation IS host RAM, so that is 2.543 GB the KV pool cannot have.
  //
  // Null on every model that has no table to lend, which is every default. The
  // two Qwen3.5/3.6 forwards override it, and they are exactly the targets the
  // DFlash loader admits (it refuses anything without `supports_aux_multi_tap`
  // by name). BORROWED: it points into the model's own weights, so a holder must
  // not outlive them — which is why LoadedEngine declares `model_` ahead of
  // `dflash_draft_`.
  //
  // "THE MODEL'S OWN WEIGHTS" IS NOT ALWAYS THE MODEL'S OWN STORAGE, and the
  // ordering argument above only covers the case where it is. A model built
  // through a `BorrowedWeightsTag` constructor (qwen3_5_dense.cpp,
  // qwen3_5_moe.cpp; reached from GPUModelRunner's borrowing constructors,
  // src/vllm/v1/worker/gpu/runner.cpp) points at weights some CALLER owns, and
  // outliving the LoadedModel then says nothing about outliving the table. No
  // path constructs a DFlash draft over a borrowed target today — the engine
  // owns its model — so this is a constraint on a future caller rather than a
  // live defect: a borrowing target must keep its weights alive for as long as
  // any draft rebound onto them.
  virtual const OwnedTensor* shared_embed_tokens() const { return nullptr; }

  // Retain the checkpoint's loaded `mtp.*` draft weights (the 15/19 BF16 tensors
  // LoadQwen3_5MTP produced) inside the concrete model so the draft can be built
  // on demand with a stable lifetime. Default: unsupported (throws), because a
  // non-MTP architecture has nowhere valid to put them.
  virtual void AttachMtpDraftWeights(Qwen3_5MTPWeights weights);

  // Construct the Qwen3_5MTPModel draft from the retained MTP weights + this
  // model's own target weights. Returns null when no MTP weights were attached
  // or the architecture is not an MTP target. The returned draft borrows this
  // model's weights/config, so it must not outlive this LoadedModel.
  virtual std::unique_ptr<Qwen3_5MTPModel> BuildMtpDraft(
      const HfConfig& config) const;

  // ── BACKEND-DISTRIBUTED-TP TP-W1: per-rank tensor-parallel handle ──────────
  // The optional TP group (comm + rank) this model instance runs under. BORROWED
  // (owned by the executor/runner for this instance's lifetime), set only when
  // tp_size>1; null on the single-GPU path so tp-aware layers take the
  // whole-tensor, byte-identical branch (tensor_parallel.h: null ⇒ tp_size()==1
  // ⇒ every collective a no-op, parallel_state.py:638 bypass). Consumed by
  // TP-W2's forward/loader plumbing; here it is the carrier only.
  const TensorParallel* tensor_parallel() const { return tensor_parallel_; }
  void set_tensor_parallel(const TensorParallel* tp) { tensor_parallel_ = tp; }

 protected:
  explicit LoadedModel(const ModelRegistration& registration)
      : registration_(registration) {}

 private:
  const ModelRegistration& registration_;
  const TensorParallel* tensor_parallel_ = nullptr;
};

// ── The type-erasure seam's OTHER half (#775) ───────────────────────────────
// Every registered `prepare`/`forward` is handed a type-erased `LoadedModel&`
// and has to open it as its own concrete model. Doing that with a bare
// `static_cast` down the hierarchy is a PROMISE, not a check: on an object
// whose dynamic type is not that model, every member call through the resulting
// reference is undefined behaviour, and the compiler is entitled to act on the
// promise. UBSan reports it as "member call on address ... which does not point
// to an object of type 'X'". It is invisible without a sanitizer whenever the
// entry point was going to refuse anyway, which is precisely the state a model
// still missing its weight loader is in.
//
// `ModelAs` is the checked form: it establishes the dynamic type first and
// refuses BY NAME when the caller paired one architecture's registration with
// another architecture's model. The cost is one `dynamic_cast` per forward
// step (the seam is entered once per `ModelRegistry::Forward`, not per layer),
// against a step that is milliseconds of GEMMs.
//
// `architecture` is the caller's OWN architecture name, so the refusal says
// which entry point refused rather than only that something did.
[[noreturn]] void RaiseModelTypeMismatch(std::string_view architecture,
                                         const LoadedModel& model);

template <typename Model>
Model& ModelAs(LoadedModel& model, std::string_view architecture) {
  Model* typed = dynamic_cast<Model*>(&model);
  if (typed == nullptr) {
    RaiseModelTypeMismatch(architecture, model);
  }
  return *typed;
}

template <typename Model>
const Model& ModelAs(const LoadedModel& model, std::string_view architecture) {
  const Model* typed = dynamic_cast<const Model*>(&model);
  if (typed == nullptr) {
    RaiseModelTypeMismatch(architecture, model);
  }
  return *typed;
}

// MM-ENGINE-FORWARD: one multimodal (vision/audio-language) forward step's
// vision-conditioned inputs, carried as an OPTIONAL sub-field of ModelForwardInput
// (below). Set (non-nullopt) ONLY by the runner mm-path when the request carries
// Request.mm_features; nullopt on every text step, so the registered TEXT forwards
// never read it and the shared runner path stays byte-identical (the field is
// additive and default-nullopt — text inertness is by construction).
//
// ENG-MM-INPUT-PIPELINE P1 (#1358, #2300): THE HANDLES ARE DEVICE TENSORS, not
// host pointers. Every field is a BORROWED, non-owning `vt::Tensor` view of memory
// that lives on `ModelForwardInput::queue`'s device for the forward's duration; an
// absent channel is the default-constructed tensor, whose `data == nullptr`.
//
// WHY DEVICE, and why this had to move before the runner does. Upstream stages the
// merged embeddings in a PERSISTENT DEVICE buffer —
// `vllm/v1/worker/gpu_model_runner.py::GPUModelRunner._init_device_properties`
// allocates `self.inputs_embeds` (:798, the sole
// `self.inputs_embeds = self._make_buffer` in the tree) and the mm branch copies
// the merged result into it (:3607, `self.inputs_embeds.gpu[...].copy_(...)`) —
// precisely so the step can be CUDA-graph captured. A HOST handle here forced a
// `[T,H]` D2H+H2D round trip per step, made the mm arm permanently
// non-graph-capturable, and put a memory-format divergence in the seam that NO
// token gate can observe (`.agents/porting.md` "Mirror the memory format, not just
// the math"). The tokens matched; the bytes moved twice.
//
// The tower, the merge (`interfaces.py::SupportsMultiModal.embed_input_ids` :380 →
// `_merge_multimodal_embeddings` :411 → `utils.py::_merge_multimodal_embeddings`
// :658) and the MRoPE index math still run ABOVE this seam and fill these buffers,
// so the registered forward stays a pure per-step function of them:
//   * inputs_embeds — the ALREADY-MERGED [num_tokens, hidden] bf16 DEVICE
//     embeddings: embed(token_ids) with the projected vision features
//     masked-scattered into the placeholder rows. When set, the forward consumes
//     THESE instead of embedding token_ids. The forward COPIES it into its own
//     residual-stream buffer and never writes through this view, mirroring
//     upstream, whose decoder layers allocate rather than clobber the persistent
//     `inputs_embeds` buffer they are handed.
//   * positions3 — the 3-D MRoPE positions, [3, num_tokens] int32 DEVICE,
//     replacing the 1-D ModelForwardInput::positions for the vision-language
//     backbone.
//   * deepstack — the [levels, num_tokens, hidden] bf16 DEVICE DeepStack
//     multiscale tensor added after decoder layers 0..levels-1 (ABSENT on decode
//     steps and on models without DeepStack, e.g. the 27B GDN-hybrid VL path).
//   * deepstack_levels — `levels` (0 ⇒ no DeepStack inject).
//   * ple_token_ids — CLAIM-GEMMA4-MM-E2E: the Gemma-4 Per-Layer-Embedding (PLE)
//     token ids [num_tokens] int32 DEVICE, with the multimodal (image/audio) rows
//     masked to 0 and the vocab_size_per_layer_input range mask applied — mirror of
//     gemma4_mm.py embed_input_ids (`is_multimodal → 0`, :1962-1969) +
//     gemma4.py get_per_layer_inputs (`id < vocab_size_per_layer_input ? id : 0`,
//     :857-863). The Gemma-4 registered mm forward looks up embed_tokens_per_layer
//     from THESE ids (NOT the merged embeds). ABSENT for non-Gemma mm models
//     (Qwen3-VL never sets it). Gemma-4 also uses the 1-D
//     ModelForwardInput::positions (NOT positions3) and no DeepStack.
struct MultiModalForwardInput {
  vt::Tensor inputs_embeds;
  vt::Tensor positions3;
  vt::Tensor deepstack;
  int64_t deepstack_levels = 0;
  vt::Tensor ple_token_ids;
};

// ENG-MULTIKV-BYNAME: WHERE a published cache's payload lives.
//
// Upstream needs no such thing. `_reshape_kv_cache_tensors` builds ONE
// `kv_caches: dict[str, torch.Tensor]` (`vllm/v1/worker/gpu_model_runner.py:7354`
// @ pin 5559679229) and fills it from both arms of the same loop — the attention
// arm at `:7418-7427` and the `MambaSpec` arm at `:7429-7441`, whose own comment
// says "Keeping one tensor per layer lets the KV connector register it without
// special-casing Mamba" — and `bind_kv_cache`
// (`vllm/v1/worker/utils.py:450-465`) binds every layer out of that one dict.
// A Mamba page is a `torch.Tensor` like any other, so one dict is enough.
//
// This tree carries the two payloads in two typed containers instead:
// `ModelForwardInput::attn_kv` holds `PagedKvCache` and `::gdn_state` holds
// `GdnStateCache`. That is a C++ type distinction, not an addressing one, so
// this enum records the ONE fact the name cannot: which container the slot
// indexes. Every published name still lives in ONE list, in ONE order, and
// resolves through ONE `Find`.
enum class KvCachePayload : uint8_t { kPaged = 0, kRecurrent = 1 };

// KV-DSV4-MULTICACHE W3 (#2068): THE THIRD FORWARD CHANNEL — the name each
// published cache was addressed by, carried beside the cache itself.
//
// `ModelForwardInput::attn_kv` is POSITIONAL: entry `i` is the i-th
// full-attention layer. That is the only thing a position CAN be while a layer
// has at most one cache. DeepSeek-V4 gives one C4A layer FOUR — the compressed
// MLA latent, the sliding-window cache, the indexer key cache and two compressor
// states — and a position cannot say which of them it is.
//
// Upstream never had this problem, and what is mirrored here is its KEY rather
// than a new invention. Every cache upstream is an `AttentionLayerBase`
// registered under its own prefix in `compilation_config.static_forward_context`
// (`vllm/models/deepseek_v4/attention.py:315-321`, `:761-767`,
// `vllm/models/deepseek_v4/compressor.py:290-295`), and `get_kv_cache_spec()`
// returns a `dict[str, KVCacheSpec]` keyed by that prefix
// (`vllm/v1/worker/gpu_model_runner.py:7785-7801`). A cache is addressed BY NAME
// upstream, so this struct carries the name.
//
// ENG-MULTIKV-BYNAME: the five vectors below are PARALLEL to EACH OTHER and
// cover EVERY published cache — recurrent as well as attention. They are NOT
// parallel to `ModelForwardInput::attn_kv`, and W3's original contract that they
// were is what this row removes.
//
// WHY THAT CONTRACT COULD NOT STAND. W3 built these three off
// `attn_group_ids_`, which collects only the groups whose spec is an
// `AttentionSpec`, so a `MambaSpec` group's layers were never named at all:
// their states reached the forward through `gdn_state` POSITIONALLY, and
// `Find()` answered -1 for every one of them. DeepSeek-V4 hid this, because it
// publishes only MLA / SlidingWindowMLA specs and all 167 of its caches are
// attention ones. Both three-group hybrids do not: `qwen4_exp` and `glm5_next`
// each carry a `MambaSpec`, and #2343 measured the consequence as
// `22 KV cache(s) from 2 published group(s)` reported beside
// `block tables gathered for 3 of 3` — 34 recurrent states invisible while
// their group's block table was not.
//
// THE ORDER IS UPSTREAM'S INSERTION ORDER: published GROUP order, then that
// group's own `layer_names` order, which is exactly how
// `_reshape_kv_cache_tensors` fills its dict
// (`vllm/v1/worker/gpu_model_runner.py:7365-7372`,
// `for group in ...: for layer_name in group.layer_names`). A recurrent group
// between two attention groups therefore sits BETWEEN them here, and an entry's
// index is NOT its slot in either payload container. That is what
// `payload_kinds` / `payload_slots` are for: entry `i`'s cache is
// `attn_kv[(*payload_slots)[i]]` when its kind is `kPaged` and
// `gdn_state[(*payload_slots)[i]]` when it is `kRecurrent`.
//
// The vectors are owned by the runner and stay valid for the forward's
// duration.
//
// NULL on `ModelForwardInput` for every model whose topology the positional
// convention can express — every model in the tree except the multi-cache ones —
// so every existing forward is byte-identical by construction.
struct MultiKvCacheIndex {
  const std::vector<std::string>* layer_names = nullptr;
  const std::vector<int32_t>* group_ids = nullptr;
  const std::vector<int32_t>* layer_indices = nullptr;
  // ENG-MULTIKV-BYNAME: parallel to the three above. `payload_kinds` holds
  // `KvCachePayload` values widened to `uint8_t` so the header needs no
  // `std::vector<KvCachePayload>` instantiation on either side of the seam;
  // read them through `PayloadAt` / `Resolve` rather than by hand.
  const std::vector<uint8_t>* payload_kinds = nullptr;
  const std::vector<int32_t>* payload_slots = nullptr;

  // MODEL-MM-QWEN4-EXP W5c-2 ([#2249](https://github.com/mudler/vllm.cpp/issues/2249)
  // item 3): ONE GATHERED BLOCK TABLE PER PUBLISHED GROUP, indexed by GROUP ID
  // rather than parallel to `attn_kv` above.
  //
  // WHY THE CHANNEL NEEDED A FOURTH VECTOR. W3 made a cache addressable by NAME
  // and a runner allocate every published one. It did not make a cache
  // READABLE: a page-addressed cache needs the map from a sequence's LOGICAL
  // position to the PHYSICAL page holding it, and that map is the group's block
  // table. Before this, `GPUModelRunner::gather_block_table` was called for
  // exactly two group ids — the target attention group and the recurrent group
  // — so a third group's table never left the runner and its cache was
  // allocated and unread. `qwen4_exp` publishes exactly such a third group: the
  // QSA indexer side cache, an `MLAAttentionSpec` at `compress_ratio` 1 —
  // ONE ROW PER TOKEN. This line said 4 until W5h measured the group against
  // upstream's `Cache.update_indexer`, which concatenates one raw key per
  // token; the ratio is the SELECTION algorithm's, never the page geometry.
  //
  // UPSTREAM DOES NOT HAVE A "TWO NAMED GROUPS" SHAPE AT ALL. Its per-group
  // metadata loop runs over `enumerate(kv_cache_groups)` and hands every group
  // its own table — `cm.block_table_tensor = _get_block_table(kv_cache_gid)`
  // (`vllm/v1/worker/gpu_model_runner.py:2551-2567` @ pin 5559679229), with
  // `_get_block_table` (`:2318-2334`) being this tree's `gather_block_table`.
  // So this field is the mirror of an existing upstream loop, not a new channel.
  //
  // INDEXED BY GROUP ID, and that is deliberate: the other three vectors are
  // parallel to `attn_kv` because a CACHE is what they describe, while a block
  // table belongs to a GROUP and every layer in that group shares it — which is
  // upstream's own fan-out ("make layers in the same group share the same
  // metadata", `:2551-2552`). Entry `g` is group `g`'s row-major
  // `[num_reqs, group_block_table_cols[g]]` table. Both vectors are sized to the
  // published group count and owned by the runner, valid for the forward's
  // duration. NULL on every uniform topology, exactly like the three above.
  const std::vector<std::vector<int32_t>>* group_block_tables = nullptr;
  const std::vector<int32_t>* group_block_table_cols = nullptr;

  // How many caches arrived, paged and recurrent together. 0 when the channel is
  // empty.
  size_t size() const;
  // How many DISTINCT published groups they came from.
  int num_groups() const;
  // The first published name, for a diagnostic. Empty when the channel is empty.
  std::string_view first_name() const;
  // The FLAT index of the cache published under `layer_name`, or -1. Feed it to
  // `PayloadAt` to reach the cache itself; it is NOT a slot in either payload
  // container, and on any topology carrying a recurrent group it is not equal to
  // one either.
  // LINEAR: the list is 167 entries at DeepSeek-V4-Flash and a forward looks a
  // name up once per layer per role, so an index structure would be premature.
  // Recorded as a decision rather than left as an oversight.
  int64_t Find(std::string_view layer_name) const;

  // ENG-MULTIKV-BYNAME. How many of the published caches are paged, and how many
  // are recurrent. `num_paged() + num_recurrent() == size()`.
  int num_paged() const;
  int num_recurrent() const;

  // Where flat entry `index` lives. FALSE when `index` is out of range or the
  // channel carries no locators. `*kind` and `*slot` are written in EVERY case —
  // `kPaged` and -1 on the false answer — so a caller that forgets the bool
  // indexes out of range rather than reading a stale slot, which is the failure
  // mode `BlockTableForGroup` above already guards the same way.
  bool PayloadAt(int64_t index, KvCachePayload* kind, int32_t* slot) const;

  // `Find` and `PayloadAt` in one call, which is what a forward wants: it holds
  // a layer name and needs the cache. FALSE when nothing was published under
  // that name, with `*kind` / `*slot` written as above.
  bool Resolve(std::string_view layer_name, KvCachePayload* kind,
               int32_t* slot) const;

  // How many groups the model PUBLISHED, which is not `num_groups()`: that one
  // counts the distinct groups the caches in `attn_kv` came from, and a
  // recurrent group contributes no entry there. `qwen4_exp` publishes THREE and
  // `num_groups()` answers two, so a diagnostic that used it as the denominator
  // would report a full gather as partial. 0 when the channel is empty.
  int num_published_groups() const;

  // How many published groups actually carry a gathered block table. 0 when the
  // channel is empty AND when a step has not run yet, which is why the runner
  // publishes the pointers once and refills the vectors every step.
  int num_group_block_tables() const;

  // Group `group_id`'s row-major `[num_reqs, *num_cols]` block table, or
  // `nullptr` when the group has none (out of range, or no step has run).
  // `*num_cols` is written in EVERY case — 0 on the null answer — so a caller
  // that forgets to check the pointer cannot index a stale width.
  const std::vector<int32_t>* BlockTableForGroup(int group_id,
                                                 int* num_cols) const;
};

// KV-DSV4-MULTICACHE W5 (#2323): does a name-keyed cache set reaching this model
// have to be REFUSED?
//
// A PURE PREDICATE, deliberately, and this row has the precedent: W1 made the
// staging budget a pure function "so it is gateable without a device". The same
// applies here -- the decision is gateable without a model, a runner or a
// registry, so both polarities can be pinned directly instead of only through a
// full engine construction.
//
// True when a set arrived (`mk != nullptr`) and the registration has not claimed
// it. Note it reads NULLNESS for arrival: an EMPTY set that arrived is still an
// arrival, and the refusal's message reads the payload so it can say so.
bool MultiKvRefusalApplies(const MultiKvCacheIndex* mk, bool consumes_multi_kv);

// ENG-ASYNC-DEVICE-IDS-REFUSAL (#2710): must a step whose HOST token identifiers
// are stale be refused, because the forward it is bound for does not read the
// device ones?
//
// A PURE PREDICATE for the same reason its `MultiKvRefusalApplies` sibling above
// is one: the decision is gateable without a model, a runner or a registry, so
// both polarities can be pinned directly rather than only through a full engine.
// An inline copy inside `ModelRegistry::Forward` would be a second derivation,
// and a second derivation is the thing that can disagree with the one a test
// pins.
//
// TRUE when all three hold:
//
//   `device_token_ids != nullptr`   the mirror is live and the device buffer is
//                                   authoritative for this step.
//   `host_token_ids_stale`          and at least one row of this step actually
//                                   HAS a spliced identifier, so the host vector
//                                   and the device buffer disagree.
//   `!consumes_device_token_ids`    and the registered forward never reads the
//                                   device buffer.
//
// WHY THE MIDDLE TERM IS NOT REDUNDANT. Without it this reduces to nullness, and
// nullness refuses `ForwardLlamaModelEmbedding` — a pooling forward that reads
// host identifiers, never reads the pointer, and is CORRECT because every request
// it serves is prefill-only. A refusal that fires on a model which actually works
// is worse than the defect it prevents, so the guard turns on the DISAGREEMENT
// and not on the arrival.
//
// THE MIDDLE TERM IS A PER-STEP FACT AND MUST STAY ONE. `v1::AnyRowSplicedByCombine`
// computes it as an OR over every row, so a step that mixes prefill rows with one
// decode row refuses as a whole. Evaluating the same rule per REQUEST would let
// such a step through for its prefill rows while its decode row read identifiers
// the runner never wrote — a refusal disagreeing with its own route predicate,
// which is a failure this tree has shipped before.
//
// NOTE THE ASYMMETRY WITH THE SIBLING, which is why that one needs no such term:
// a multi-cache topology is a property of the whole model and the whole step, so
// its arrival IS its disagreement. Staleness here is a property of a ROW, so
// arrival and disagreement come apart and both have to be asked.
bool DeviceTokenIdsRefusalApplies(const int32_t* device_token_ids,
                                  bool host_token_ids_stale,
                                  bool consumes_device_token_ids);

// One MRV2 forward invocation. References stay valid for the duration of the
// registered forward hook; model-specific decode-graph state lives in the
// concrete LoadedModel rather than leaking concrete weight types into runner.
struct ModelForwardInput {
  const std::vector<int32_t>& token_ids;
  const std::vector<int32_t>& positions;
  const v1::CommonAttentionMetadata& attn_meta;
  const v1::GDNAttentionMetadata& gdn_meta;
  std::vector<PagedKvCache>& attn_kv;
  std::vector<GdnStateCache>& gdn_state;
  const HfConfig& config;
  vt::Queue& queue;
  const std::vector<int32_t>& logits_indices;
  int num_reqs = 0;
  int64_t gdn_state_slots = 0;
  bool pure_decode = false;
  // SPEC-DSPARK W8 (#442): the configured speculation width, so the decode-graph
  // gate can mirror vLLM's UNIFORM-decode predicate instead of "query_len == 1".
  // Upstream's captured decode length is `1 + num_speculative_tokens`
  // (cudagraph_dispatcher.py:37), which is why its T=1+k speculative VERIFY is
  // graph-captured and ours was not. DEFAULT 0 => the predicate reduces exactly
  // to today's pure-decode shape, so every non-spec caller is byte-identical.
  int64_t num_speculative_tokens = 0;
  // ENG-CUDAGRAPH-BREAK W6 (#1374): THE GRAPH-ELIGIBILITY PREDICATE, moved off
  // `pure_decode` and onto the step's ACTUAL uniform query length.
  //
  // The runner computes it once per step through
  // `v1::ActualUniformDecodeQueryLen` (`v1/worker/gpu/cudagraph_dispatch.h`) and
  // every model reads the answer. 0 means "no captured decode graph in this tree
  // can serve this step" -- prefill, mixed, ragged, or uniform at a length above
  // the configured `1 + num_speculative_tokens`. 1 is exactly `pure_decode`.
  // A value ABOVE 1 is a speculative VERIFY step at its actual draft depth,
  // which is the population [#1020] named and which the two Qwen3.5 drivers
  // serve.
  //
  // WHY `pure_decode` SURVIVES BESIDE IT. Seven of the nine decode drivers
  // capture a query_len == 1 shape and nothing else, and widening them here
  // would admit steps no driver can serve -- the exact failure the spec's
  // `## Work breakdown` W6 says to avoid by ordering this stage last. They keep
  // reading `pure_decode`, which is provably NARROWER than this field, so the
  // widening is opt-in per driver rather than imposed on all nine at once.
  int64_t uniform_query_len = 0;
  bool gather_logits = true;
  // SPEC-MTP I5d-pre hidden-state tap. When non-null (only the spec verify
  // forward sets it, I5d), the Qwen3.5 dense/MoE forward routes to
  // Qwen3_5{,Dense}Model::ForwardDeviceTap, which returns byte-identical logits
  // AND moves the full [num_actual_tokens, H] post-final-norm hidden into
  // `*hidden_tap` for the MTP drafter. When null (every spec-off run) the
  // forward takes the current ModelRegistry::Forward path and is byte-identical.
  // Models other than Qwen3.5 ignore this field.
  Qwen3_5MTPHiddenStates* hidden_tap = nullptr;
  // SPEC-DFLASH D1 (DF-AUX-TAPS) multi-layer aux hidden taps. When non-null (only
  // the DFlash spec verify forward sets it, D4), the Qwen3.5 dense/MoE forward
  // routes to Qwen3_5{,Dense}Model::ForwardDeviceMultiTap, which returns byte-
  // identical logits AND captures (hidden + residual) at each configured
  // target_layer_id boundary into `aux_tap->tensor` = [T, H×taps] for the DFlash
  // drafter. When null (every non-DFlash run) the forward takes the current path
  // and is byte-identical. Mutually exclusive with hidden_tap. Models other than
  // Qwen3.5 ignore this field.
  Qwen3_5AuxTaps* aux_tap = nullptr;
  // MM-ENGINE-FORWARD: the optional vision-language per-step inputs. nullopt on
  // every TEXT step (text models never read it ⇒ byte-identical shared-path). Set
  // only by the runner mm-path when the request carries Request.mm_features, and
  // consumed by the registered multimodal forward (Qwen3-VL) which routes the
  // tower/merge/MRoPE/DeepStack-conditioned decode. Default-nullopt keeps every
  // existing (text) call site byte-identical by construction.
  std::optional<MultiModalForwardInput> mm = std::nullopt;
  // ENG-ASYNC-SCHED W4: when non-null, the [token_ids.size()] input ids already
  // live in THIS device buffer and `token_ids` is stale for decode rows. The
  // async runner's device combine spliced each decode row's sampled token here
  // on the main queue, so the host copy never saw it — which is the whole point,
  // since materializing it on the host is the synchronize W4 removes.
  //
  // A model that honors this embeds from the device pointer instead of uploading
  // `token_ids`.
  //
  // WHAT THIS COMMENT USED TO SAY, AND WHY IT IS GONE (#2710). It said "a model
  // that ignores it is simply never given one (the runner only sets it on the
  // discrete-CUDA async path, which the Qwen3.5 gate vehicle owns)". THAT
  // SENTENCE WAS FALSE, and it is the sentence the defect hid behind for five
  // architectures. `GPUModelRunner::execute_model` assigns this field
  // UNCONDITIONALLY, for every registered forward, whenever the mirror is
  // engaged — and `async_device_mirror()` is the DEFAULT on CUDA, integrated
  // parts included, not a discrete-only opt-in. A model that ignored it was not
  // spared; it was handed identifiers it then failed to read.
  //
  // IT IS NO LONGER ADVISORY. A forward may only be handed a step whose host
  // identifiers are stale when its `ModelFactory::consumes_device_token_ids` is
  // true; `ModelRegistry::Forward` refuses the step otherwise, exactly as the
  // `multi_kv` channel below is refused. See `DeviceTokenIdsRefusalApplies`.
  //
  // Null on every non-mirror path, so every such forward is byte-identical.
  const int32_t* device_token_ids = nullptr;
  // ENG-ASYNC-DEVICE-IDS-REFUSAL (#2710): does `token_ids` actually DISAGREE with
  // `device_token_ids` for this step?
  //
  // The pointer above says "the device buffer is authoritative". This says "and
  // it differs", which is a different fact and is the one a refusal must turn on.
  // The runner's combine splices a row only when
  // `v1::CombineSplicesRow(seq_len, prefill_len)` holds, so on a step whose rows
  // are ALL prefill — every step of a pooling model, and the first step of every
  // ordinary one — nothing is spliced and the host vector is perfectly good.
  //
  // WITHOUT THIS FIELD THE GUARD WOULD BE WORSE THAN THE DEFECT. It would refuse
  // `ForwardLlamaModelEmbedding`, which reads host `token_ids`, never reads the
  // pointer, and is CORRECT because every request it serves is prefill-only.
  // Taking a working path away is a worse outcome than the silent staleness the
  // guard exists to stop.
  //
  // PER-STEP, computed by `v1::AnyRowSplicedByCombine` as an OR over the whole
  // batch. It is never a per-request answer; the header of that function says
  // what a per-request reading would let through. FALSE by default, so every
  // caller that is not the async runner is byte-identical.
  bool host_token_ids_stale = false;
  // KV-DSV4-MULTICACHE W3 (#2068): the THIRD cache channel. Non-null only when
  // the runner allocated a MULTI-CACHE topology — one whose published groups the
  // positional `attn_kv` convention cannot address. THREE architectures publish
  // one today (`DeepseekV4ForCausalLM`, `Qwen4ExpForConditionalGeneration` and
  // `Glm5NextForConditionalGeneration`); this line said "DeepSeek-V4 and nothing
  // else", which was true when W3 wrote it and stopped being true two rows
  // later. NULL on every other step, so every other forward is byte-identical.
  //
  // A forward may only READ it when its `ModelFactory::consumes_multi_kv` is
  // true; `ModelRegistry::Forward` refuses the step otherwise. Set after aggregate construction, like `device_token_ids`
  // above, so no positional initializer moves.
  const MultiKvCacheIndex* multi_kv = nullptr;
};

// ── ENG-MM-INPUT-PIPELINE P2 (#2379): the multimodal MODEL seam ─────────────
//
// Upstream's runner does not implement multimodal itself. It calls three MODEL
// methods through two protocols, and the runner code is generic over them:
//
//   SupportsMultiModal.embed_multimodal      -> the vision/audio tower
//   SupportsMultiModal.embed_input_ids       -> embed ids + splice the mm rows
//   SupportsMRoPE.get_mrope_input_positions  -> the 3-D prompt positions
//
// `ModelRegistry` is type-erased over `LoadedModel`, so a Python protocol
// becomes three optional function pointers on `ModelFactory`. All three default
// null, and `GPUModelRunner` runs its multimodal path only when the first two
// are set AND some request in the step carries mm_features — so a text model
// and a text step are byte-identical.
//
// WHY DEEPSTACK IS NOT IN THIS SEAM. `grep -c deepstack` over upstream's
// `gpu_model_runner.py` is 0. Qwen3-VL's multiscale features ride INSIDE the
// tower output and are unpacked model-side into a model-owned buffer
// (`qwen3_vl.py:1757 self.deepstack_input_embeds`, the ctor one of two,
// discriminated by its `# register buffer for deepstack` comment). A runner
// that knew about DeepStack would be a runner that knows about one
// architecture, so the runner here does not either: `MmEncoderOutput` carries
// exactly one tensor, and what a model chooses to stash beside it is its own.

// One multimodal item's encoder output — the mirror of one element of
// `MultiModalEmbeddings` (`embed_multimodal`'s return).
struct MmEncoderOutput {
  // Owns the device allocation `embeds` views. The runner keeps this alive for
  // as long as the mm_hash stays in its encoder cache, and drops it when the
  // scheduler reports the hash freed.
  std::shared_ptr<void> storage;
  // [num_embeds, hidden] in the model dtype.
  vt::Tensor embeds;
};

// What the runner hands the model's `embed_input_ids` mirror for ONE step.
// Every pointer is BORROWED and valid only for the duration of the call.
struct MmEmbedInputs {
  // [num_scheduled_tokens] the step's input ids, in batch (dense) order.
  const std::vector<int32_t>* token_ids = nullptr;
  // The gathered encoder-output ROW SLICES, in the order their `true` positions
  // appear in `is_mm_embed` — upstream's `multimodal_embeddings` list.
  const std::vector<vt::Tensor>* mm_embeds = nullptr;
  // [num_scheduled_tokens] upstream's `is_multimodal` bool mask. `char` rather
  // than `bool` because `std::vector<bool>` has no contiguous buffer to upload.
  const std::vector<char>* is_mm_embed = nullptr;
  // [3 * num_scheduled_tokens] row-major M-RoPE positions, or EMPTY when the
  // registration declares no `mrope_prompt_positions` (upstream's
  // `uses_mrope == False`, which is Gemma-4's multimodal arm: it reads the 1-D
  // `ModelForwardInput::positions` instead).
  const std::vector<int32_t>* mrope_positions = nullptr;
  // ENG-MM-EMBED-DEVICE-IDS ([#2730](https://github.com/mudler/vllm.cpp/issues/2730)):
  // the DEVICE twin of `token_ids`, and the reason this struct exists in the
  // shape it does.
  //
  // `token_ids` above is the host step vector, and the asynchronous runner
  // deliberately leaves it STALE for decode rows: its combine splices each decode
  // row's sampled token into the DEVICE buffer on the main queue and never writes
  // it back, because materialising it on the host is the synchronise that path
  // exists to remove. `token_ids_cpu` is zero-initialised, so a hook that embeds
  // the host vector alone builds `inputs_embeds` from TOKEN ID 0 on every decode
  // row -- at rc=0, with plausible-looking output. `batch_carries_mm()` returns
  // true on the decode steps of an image request BY DESIGN (see its own comment
  // in `runner.cpp`), so this is reached rather than theoretical, and
  // `async_device_mirror()` is the DEFAULT on CUDA, integrated parts included.
  //
  // The two fields carry exactly what `ModelForwardInput::device_token_ids` and
  // `ModelForwardInput::host_token_ids_stale` carry, set by the runner from the
  // SAME two values in the same step, so the embed's guard and the forward's
  // guard cannot disagree about one step.
  //
  // NOT ADVISORY. A hook may only be handed a step whose host identifiers are
  // stale when its `ModelFactory::embed_mm_consumes_device_token_ids` is true;
  // `ModelRegistry::EmbedMm` refuses the step otherwise, on the same
  // `DeviceTokenIdsRefusalApplies` predicate `ModelRegistry::Forward` uses.
  //
  // Null on every non-mirror path, so every such step is byte-identical.
  const int32_t* device_token_ids = nullptr;
  // Whether `token_ids` actually DISAGREES with `device_token_ids` for this step.
  // The pointer says the device buffer is authoritative; this says the two
  // differ, which is a different fact and the one a refusal must turn on. FALSE
  // on an image request's PREFILL step, where the combine splices no row -- and
  // every image request reaches its first token through such a step, so a guard
  // that ignored this term would take multimodal serving away entirely.
  bool host_token_ids_stale = false;
};

// The per-step device buffers the model staged, plus the seam view over them.
// `storage` keeps them alive until the runner's forward has returned.
struct MmForwardBuffers {
  std::vector<std::shared_ptr<void>> storage;
  MultiModalForwardInput mm;
};

// `get_mrope_input_positions` (`SupportsMRoPE`). Computed ONCE per request at
// admission (upstream `_init_mrope_positions`, gpu_model_runner.py:1654,
// grep -c == 1) because it needs the WHOLE prompt; the per-step slice and the
// synthesised completion part are the runner's.
struct MropePromptPositions {
  // [3 * num_prompt_tokens] row-major.
  std::vector<int32_t> positions;
  // The ONE scalar that carries M-RoPE across decode steps
  // (gpu_model_runner.py:2786, grep -c == 1): a completion position is
  // `context_len + i + delta` on all three axes.
  int64_t delta = 0;
};

using ModelEncodeMmFn = MmEncoderOutput (*)(
    LoadedModel& model, const HfConfig& config, vt::Queue& queue,
    const multimodal::MultiModalFeatureSpec& item);
using ModelEmbedMmFn = MmForwardBuffers (*)(LoadedModel& model,
                                            const HfConfig& config,
                                            vt::Queue& queue,
                                            const MmEmbedInputs& inputs);
using ModelMropePromptFn = MropePromptPositions (*)(
    LoadedModel& model, const HfConfig& config,
    const std::vector<int32_t>& prompt_token_ids,
    const std::vector<multimodal::MultiModalFeatureSpec>& mm_features);

using ModelConfigHook = void (*)(const HfConfig& config);
using ModelWeightLoader = std::unique_ptr<LoadedModel> (*)(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source);
using ModelPrepareFn = void (*)(LoadedModel& model, const HfConfig& config,
                                vt::Queue& queue);
using ModelForwardFn = ForwardLogits (*)(LoadedModel& model,
                                         const ModelForwardInput& input);
using KVCacheSpecBuilder = v1::KVCacheConfig (*)(const HfConfig& config,
                                                 int block_size,
                                                 int num_blocks);

// Per-family plug-in seam. load_weights contains that family's on-disk name map
// and construction path; forward remains type-erased over LoadedModel.
struct ModelFactory {
  ModelConfigHook parse_config = nullptr;
  ModelWeightLoader load_weights = nullptr;
  ModelPrepareFn prepare = nullptr;
  ModelForwardFn forward = nullptr;
  KVCacheSpecBuilder make_kv_cache = nullptr;
  // ENG-MM-INPUT-PIPELINE P2 (#2379). See the seam note above the typedefs.
  // Null on every text architecture, which is what keeps `GPUModelRunner`'s
  // multimodal path off for them entirely rather than merely inert.
  ModelEncodeMmFn encode_mm = nullptr;
  ModelEmbedMmFn embed_mm = nullptr;
  // OPTIONAL even for a multimodal model: null is upstream's `uses_mrope ==
  // False`, and the runner then leaves `MmEmbedInputs::mrope_positions` empty
  // and the model reads the ordinary 1-D positions.
  ModelMropePromptFn mrope_prompt_positions = nullptr;
  // The SMALLEST KV block size this architecture can be paged at, or 0 when it
  // has no constraint. Upstream DERIVES this geometry rather than taking it from
  // an operator: DeepSeek-V4 spells `[256 // compress_ratio, head_dim]`
  // throughout (`sparse_swa.py:76-83`, `compressor.py:174-178`), so a
  // `compress_ratio == 128` layer cannot be paged below 256 -- at the engine's
  // default of 32, `block_size / compress_ratio` is 0 and the page holds no
  // token.
  //
  // Declared here rather than left to a flag because `AGENTS.md` measures
  // reachability on the DEFAULT configuration. A model that loads only when an
  // operator happens to pass `--block-size 256` is not reachable by default, and
  // `vllm-cli` does not expose that flag at all.
  int kv_block_size_floor = 0;
  // Preserves the already-gated per-arch scheduler default. This is execution
  // policy, not an upstream _ModelInfo capability.
  bool is_dense_model = false;
  // ROW 7 (kimi-linear.md §20.3): the loader wants a `ModelSource::load_queue`
  // selected BEFORE the weights load — the GB10 recipe (CUDA context first, then
  // per-tensor stage-and-release; Kimi-Linear's 91.5 GiB bf16-resident loader).
  // Default false: every existing arch's engine load path is byte-identical.
  bool stage_on_load = false;
  // MODEL-MM-QWEN4-EXP W5j ([#2031](https://github.com/mudler/vllm.cpp/issues/2031),
  // [#2353](https://github.com/mudler/vllm.cpp/issues/2353)): whether THIS
  // model's forward RESOLVES its caches through `MultiKvCacheIndex` — by the
  // name each was published under — instead of reading `attn_kv` / `gdn_state`
  // positionally.
  //
  // THE DEFAULT IS FALSE AND THAT IS THE MECHANISM, the same polarity
  // `stage_on_load` and `supports_weight_offload` above already use.
  // `ModelRegistry::Forward` refuses a multi-cache topology for every model that
  // leaves it false, so a model that publishes three groups and then reads two
  // positional channels is stopped rather than served a cache set it silently
  // mis-indexes. On a topology whose groups the positional convention CAN
  // express the runner sends no channel at all (`multi_kv == nullptr`), so this
  // bit is inert for every uniform model whatever its value.
  //
  // WHY A BIT AND NOT AN ARCHITECTURE LIST IN THE GUARD. The fact is a property
  // of the forward and it lives beside the forward: the one translation unit
  // that resolves by name is the one that sets this. A list in
  // `ModelRegistry::Forward` would have to be edited by every model row, which
  // is the shared-file lock AGENTS.md `## Records` forbids, and it is the
  // construct #2288 has already driven stale six times on this row.
  //
  // IT IS NOT A LICENCE TO SERVE ANY SHAPE. It says the forward ASKS by name;
  // the forward still refuses a channel that answers wrongly — a name that
  // resolves to nothing, to the wrong payload kind, or to a group with no
  // gathered block table. Setting it true on a forward that reads positionally
  // would move a silent mis-index from the engine into the model, so it lands
  // WITH its first consumer and never before one.
  //
  // THE BIT ITSELF IS KV-DSV4-MULTICACHE W5's
  // ([#2323](https://github.com/mudler/vllm.cpp/issues/2323)), which introduced
  // it to turn `ModelRegistry::Forward`'s blanket refusal into a dispatch and
  // stated the reason this default cannot be flipped for convenience: the
  // failure it prevents is INVISIBLE. A model that discards a name-keyed set
  // decodes by recomputing the whole prefix, which keeps the tokens right
  // while doing asymptotically more work, so no token gate can see it. A
  // capability whose absence is invisible must be opt-in.
  bool consumes_multi_kv = false;
  // MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm W4 (#2411): does this
  // architecture's `load_weights` READ `ModelSource::mmproj`?
  //
  // Declared rather than inferred, and `ModelRegistry::Load` refuses a source
  // that carries a projector this registration would ignore. Without the
  // refusal the failure is silent and fluent: the user names `--mmproj`, the
  // load succeeds, no tower exists, and the first image request is answered as
  // text. That is the same class of defect `consumes_multi_kv` exists for, and
  // it is declared the same way.
  bool consumes_mmproj = false;
  // ENG-ASYNC-DEVICE-IDS-REFUSAL ([#2710](https://github.com/mudler/vllm.cpp/issues/2710)):
  // whether THIS model's registered forward READS
  // `ModelForwardInput::device_token_ids` rather than embedding from the host
  // `token_ids` vector the async runner deliberately leaves stale.
  //
  // THE DEFAULT IS FALSE AND THAT IS THE MECHANISM, the polarity
  // `consumes_multi_kv` above and `stage_on_load` already use. A model added
  // tomorrow that does not read the field is REFUSED on a step whose host
  // identifiers are stale, instead of being handed them and decoding from
  // whatever `token_ids_cpu` was zero-initialised to. Five architectures were
  // caught doing exactly that before this bit existed (#1305, #2496, #2544); the
  // sixth is the one this default is for.
  //
  // WHY A BIT AND NOT A LIST IN THE GUARD. The fact is a property of the forward
  // and it lives beside the forward: the translation unit that reads the field is
  // the one that sets this. A list inside `ModelRegistry::Forward` would have to
  // be edited by every model row, which is the shared-file lock AGENTS.md
  // `## Records` forbids.
  //
  // WHAT IT DOES NOT CLAIM. It says "this forward reads the field", not "on every
  // path it can take". A forward with one arm that reads the device identifiers
  // and another that does not is a defect in that forward, and this bit cannot
  // see it. Setting it is a statement about the code, not a warrant for it.
  //
  // AND IT IS A STATEMENT ABOUT THE REGISTRATION, NOT ONLY ABOUT THE FUNCTION
  // NAMED `forward` (ENG-MM-EMBED-DEVICE-IDS,
  // [#2730](https://github.com/mudler/vllm.cpp/issues/2730)). A multimodal
  // registration can resolve the step's identifiers in its `embed_mm` hook and
  // hand the forward merged embeds instead, in which case the forward reads no
  // identifier at all and the registration is still correct. `qwen3_vl` is that
  // case and is the only one today: its forward REFUSES a step without
  // `input.mm` by name, and the runner only ever supplies `mm` from the same
  // branch that just called `ModelRegistry::EmbedMm` with the same device
  // pointer, so every step it can reach came through the hook. Leaving this
  // comment saying "forward" while a registration sets it on the strength of its
  // embed is how a capability bit becomes a lie, so it says both. The HOOK's own
  // fact is `embed_mm_consumes_device_token_ids` below and the two are separate,
  // because `dots3_note` has one and not the other.
  bool consumes_device_token_ids = false;
  // ENG-MM-EMBED-DEVICE-IDS ([#2730](https://github.com/mudler/vllm.cpp/issues/2730)):
  // the same question asked of the MULTIMODAL hook -- whether THIS model's
  // `embed_mm` reads `MmEmbedInputs::device_token_ids` rather than gathering from
  // the host vector the async runner leaves stale for decode rows.
  //
  // WHY A SECOND BIT AND NOT THE ONE ABOVE. `dots3_note` is the counterexample
  // that makes one bit impossible: its `embed_mm` consumes the channel and its
  // `forward` does not -- it reaches
  // `Dots3NoteModel::ForwardDevice(input.token_ids, ...)` on a text-only batch,
  // which is class (b) of `.agents/specs/eng-async-device-ids-refusal.md` and
  // owed by [#2732](https://github.com/mudler/vllm.cpp/issues/2732). Collapsing
  // the two facts would either un-refuse its stale text step or refuse its
  // correct image step.
  //
  // THE DEFAULT IS FALSE AND THAT IS THE MECHANISM, the polarity every capability
  // bit beside it uses. A hook added tomorrow that ignores the channel is REFUSED
  // rather than served identifiers it never reads.
  bool embed_mm_consumes_device_token_ids = false;
  // MODEL-MM-QWEN4-EXP W5L ([#2031](https://github.com/mudler/vllm.cpp/issues/2031)):
  // whether THIS model's forward serves exactly ONE sequence per step, so the
  // engine must not schedule a batch it will refuse.
  //
  // THIS IS NOT A PREFERENCE, IT IS THE ONLY THING BETWEEN A SERVER AND A DEAD
  // ENGINE. A forward that refuses `num_reqs > 1` throws from inside the
  // EngineCore busy loop, and that loop treats a throw as FATAL: the process
  // keeps its socket open, every in-flight request gets a 500 naming the model,
  // and every later request gets the same. MEASURED on this architecture at
  // `--max-num-seqs 4` before this bit existed — three concurrent
  // `/v1/completions` calls returned
  // "EngineCore encountered an issue ... this forward serves ONE sequence per
  // call and the step carries 2" and the engine never recovered. The default
  // `max_num_seqs` is 128, so that was the OUT-OF-THE-BOX behaviour.
  //
  // `LoadedEngine::ResolveMaxNumSeqs` therefore clamps the resolved concurrency
  // to 1 for a model that sets it, and says so on stderr exactly as the
  // recurrent-state budget clamp beside it does. The refusal in the forward
  // STAYS: this bit stops the engine producing the batch, and the forward stops
  // anyone else's.
  //
  // THE DEFAULT IS FALSE AND MEANS "no opinion", not "batches fine": a model
  // that leaves it false is scheduled exactly as it was before this field
  // existed. It is a statement about the PORT and not about the architecture —
  // clearing it is what a wave that plumbs a ragged multi-request batch does,
  // and that wave is owed under `.agents/specs/qwen4-exp-flash-next.md`.
  bool serves_one_sequence_per_step = false;
  // ENG-WEIGHT-OFFLOAD: whether THIS model's loader asks
  // `WeightOffloader::ConsiderWeight` for each weight and honours the answer.
  //
  // THE DEFAULT IS FALSE, AND THAT IS THE MECHANISM. There is no single upload
  // helper in this tree: each model allocates its own device buffers, and
  // `load_stats::AddDeviceUpload` reaches only 9 call sites in 6 files, so
  // neither is a chokepoint that could enforce this. A model whose loader was
  // never wired would therefore accept a `cpu_offload_gb` and silently keep
  // every weight on the device, which is a memory bug with no error anywhere.
  //
  // Declaring the capability makes that case LOUD instead: the engine refuses a
  // configured offload against a model that does not claim support, naming the
  // architecture. A new model inherits false and is refused until someone wires
  // it, which is the same polarity as the `-Werror=switch` totality proof in
  // `gguf_keep_quant.cpp`, expressed at run time because there is no enum to
  // switch over.
  bool supports_weight_offload = false;
  // ENG-EXPERT-STREAM-DEVICE W0d (issue #1124): whether THIS model's forward
  // reads its routed-expert weights through the expert-stream slot seam
  // (`KqExpertSlice`), so the stacked `*_exps.weight` towers are served a slice
  // at a time out of the host slot store instead of being staged.
  //
  // THE DEFAULT IS FALSE FOR THE SAME REASON `supports_weight_offload`'s is, and
  // this one is load-bearing in the UNSAFE direction. The load-time fit bound
  // (`gguf_device_fit.h`) can drop a tensor set from what it charges the device,
  // and the loader identifies that set by NAME — `_exps.weight`, which is what a
  // llama.cpp MoE export writes for every MoE family it converts, not only the
  // ones this tree streams. `deepseek_v4_weights.cpp` and `laguna_weights.cpp`
  // both write that exact suffix, and neither model composes `RunMoeBlock`
  // (`deepseek_v2.cpp` says so at its head), so neither ever reaches
  // `KqExpertSlice`. Dropping their towers from the bound would remove a REFUSAL
  // THAT WAS CORRECT and put back the failure #1123 exists to prevent: a
  // 26-minute load and then `cudaMalloc: out of memory` on the first forward.
  //
  // WHY THE CAPABILITY AND NOT AN ARCHITECTURE LIST. The fact is a property of
  // the model's forward, and it lives beside the forward: `qwen3_5_moe.cpp` and
  // `qwen3_moe_registry.cpp` are the two translation units that route into
  // `RunMoeBlock`, and they are the two that set this. A list in the loader would
  // be a second description of the same fact, in a file that cannot see when the
  // first one changes — and a model whose forward stopped streaming would leave
  // the list saying it still does. Inheriting false is the safe answer: a new
  // architecture gets the whole bound and the #1123 refusal until somebody wires
  // the seam and says so here.
  bool streams_routed_experts = false;
  // MODEL-MM-GLM53-FLASH W5b-2c (#2348): whether THIS model's forward CONSUMES
  // `ModelForwardInput::multi_kv` -- the cache set keyed by published layer
  // name -- rather than ignoring it.
  //
  // THE DEFAULT IS FALSE, AND THE REFUSAL IN `ModelRegistry::Forward` IS WHAT
  // THE DEFAULT BUYS. KV-DSV4-MULTICACHE W3 (#2068) made the runner allocate
  // every published cache and hand them here by name, and refused the step
  // unconditionally because no forward knew what to do with them. Letting a
  // model through without that knowledge discards a correctly allocated KV
  // topology in silence and reports a decode rate for a full-recompute path --
  // and, for a paged model, re-attends an empty prefix on every step after the
  // first, which is fluent wrong text and not a crash.
  //
  // WHY THE CAPABILITY AND NOT AN ARCHITECTURE LIST, which is the same argument
  // `streams_routed_experts` above makes: the fact is a property of the
  // model's forward and it lives beside the forward. `glm5_next_registry.cpp`
  // is the translation unit that reads the channel and it is the one that sets
  // this. A new architecture inherits false and is refused until somebody
  // writes the consuming forward and says so here.
  //
  // DeepSeek-V4 keeps the default: `DeepseekV4Model::Forward` still opens with
  // `(void)attn_kv;` and row KV-DSV4-MULTICACHE W5 (#1925) owns its consuming
  // forward.
};

struct ModelRegistration {
  std::string_view architecture;
  const ModelFactory* factory = nullptr;
  ModelInfo info;
};

struct UnsupportedModelInfo {
  std::string_view architecture;
  std::string_view detail;  // previous version or plugin URL
};

// Self-registration seam (mirrors REGISTER_VLLM_MODEL in
// vllm/model_executor/models/registry.py:682-693 assembling `_VLLM_MODELS`).
// Each architecture registers itself from its OWN translation unit via a static
// `ModelRegistrar`, exactly like the RegisterOp/RegisterBackend/RegisterPlatform
// static-init idiom (src/vt/ops.cpp, src/vt/backend.cpp,
// src/vllm/platforms/platform.cpp). Adding a model = a new TU with one
// REGISTER_VLLM_MODEL line — ZERO edit to a shared array.
//
// The passed ModelRegistration is copied into the process-global registry (its
// string_view/factory members point at TU-static data with static lifetime, so
// the copy stays valid). Registration order across TUs is unspecified under C++
// static init, so the registry imposes a stable canonical sort by architecture
// name on first query (see model_registry.cpp) — resolution semantics (first
// config-architecture match) are order-independent and unchanged.
void RegisterModel(const ModelRegistration& registration);

// Internal adapter helper: returns the registry entry for an implemented
// architecture (used by the synthetic in-memory Make/Borrow adapters below).
const ModelRegistration& RegistrationFor(std::string_view architecture);

// Static-init helper whose constructor performs the self-registration; used
// only through the REGISTER_VLLM_MODEL macro.
struct ModelRegistrar {
  explicit ModelRegistrar(const ModelRegistration& registration) {
    RegisterModel(registration);
  }
};

// Registers one architecture's factory from its own TU. Place at namespace
// scope inside `namespace vllm { ... }`; `unique_tag` is any TU-unique token.
#define REGISTER_VLLM_MODEL(unique_tag, architecture_name, factory_ref,   \
                            info_val)                                      \
  namespace {                                                             \
  const ::vllm::ModelRegistrar vllm_model_registrar_##unique_tag(         \
      ::vllm::ModelRegistration{(architecture_name), &(factory_ref),      \
                                (info_val)});                             \
  } /* namespace */

class ModelRegistry {
 public:
  // Mirrors get_supported_archs() and resolve_model_cls(): registrations are
  // declaration-ordered and Resolve returns the first architecture-list match.
  static std::span<const ModelRegistration> Registrations();
  static std::vector<std::string_view> SupportedArchs();
  static const ModelRegistration& Resolve(
      std::span<const std::string> architectures);
  static const ModelRegistration& Resolve(const HfConfig& config);

  // Mirrors _raise_for_unsupported exactly. The explicit-supported overload is
  // used by the subset-oracle parity test and covers the upstream
  // registered-but-inspection-failed branch.
  [[noreturn]] static void RaiseForUnsupported(
      std::span<const std::string> architectures);
  [[noreturn]] static void RaiseForUnsupported(
      std::span<const std::string> architectures,
      std::span<const std::string_view> supported_architectures);

  static std::span<const UnsupportedModelInfo> PreviouslySupportedModels();
  static std::span<const UnsupportedModelInfo> OutOfTreeSupportedModels();

  // Type-erased construction and live runner hooks.
  static std::unique_ptr<LoadedModel> Load(const HfConfig& config,
                                           const ModelSource& source);
  static void Prepare(LoadedModel& model, const HfConfig& config,
                      vt::Queue& queue);
  static ForwardLogits Forward(LoadedModel& model,
                               const ModelForwardInput& input);
  static v1::KVCacheConfig MakeKVCache(const LoadedModel& model,
                                       const HfConfig& config, int block_size,
                                       int num_blocks);

  // The block size the engine must page this architecture at, given what the
  // caller asked for. Returns `requested` unchanged for an architecture that
  // declares no floor. THE single resolution point: `LoadedEngine` calls this
  // before `MakeKVCache`, and a gate calls it with `EngineParams{}.block_size`
  // to prove the model's DEFAULT configuration reaches a buildable geometry
  // (writing the arithmetic out a second time in a test would only transcribe
  // it, and could not detect it changing here).
  static int ResolveKVBlockSize(const ModelRegistration& reg, int requested);
  static bool IsDenseModel(const LoadedModel& model);

  // ── ENG-MM-INPUT-PIPELINE P2 (#2379): the multimodal model seam ──
  // SupportsMmInputs mirrors upstream's `self.supports_mm_inputs`
  // (gpu_model_runner.py:530, `grep -c 'self.supports_mm_inputs ='` == 1), which
  // is the predicate the whole mm arm of `execute_model` hangs on. Here it asks
  // whether the REGISTRATION declares the two required hooks, which is the same
  // question: a model that cannot run a tower and cannot embed with a splice has
  // no mm path to take.
  static bool SupportsMmInputs(const LoadedModel& model);
  static bool UsesMrope(const LoadedModel& model);
  static MmEncoderOutput EncodeMm(LoadedModel& model, const HfConfig& config,
                                  vt::Queue& queue,
                                  const multimodal::MultiModalFeatureSpec& item);
  static MmForwardBuffers EmbedMm(LoadedModel& model, const HfConfig& config,
                                  vt::Queue& queue, const MmEmbedInputs& inputs);
  static MropePromptPositions MropePromptPositionsFor(
      LoadedModel& model, const HfConfig& config,
      const std::vector<int32_t>& prompt_token_ids,
      const std::vector<multimodal::MultiModalFeatureSpec>& mm_features);
};

// Compatibility adapters for synthetic in-memory Qwen tests and callers that
// already own weights. Production disk loading uses ModelRegistry::Load.
std::unique_ptr<LoadedModel> MakeQwen3_5MoeLoadedModel(
    Qwen3_5MoeWeights weights);
std::unique_ptr<LoadedModel> MakeQwen3_5DenseLoadedModel(
    Qwen3_5DenseWeights weights);
// The same, under the registration `config` resolves to when that
// registration is an alias of the Qwen3.5 dense factory (Qwen3_5ForCausalLM,
// Tev1Model), so the synthetic engine carries the architecture a disk load of
// that config would. Any other config keeps Qwen3_5ForConditionalGeneration,
// which is what the one-argument form always uses.
std::unique_ptr<LoadedModel> MakeQwen3_5DenseLoadedModel(
    Qwen3_5DenseWeights weights, const HfConfig& config);
std::unique_ptr<LoadedModel> BorrowQwen3_5MoeLoadedModel(
    const Qwen3_5MoeWeights& weights);
std::unique_ptr<LoadedModel> BorrowQwen3_5DenseLoadedModel(
    const Qwen3_5DenseWeights& weights);

}  // namespace vllm
