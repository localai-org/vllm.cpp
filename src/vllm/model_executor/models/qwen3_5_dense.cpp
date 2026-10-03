// Qwen3.6 DENSE (27B) architecture registry TU. Self-registers
// "Qwen3_5ForConditionalGeneration" via REGISTER_VLLM_MODEL and owns the dense
// arch-specific entry points (LoadedModel subclass + load/prepare/forward
// wrappers + factory + synthetic Make/Borrow adapters). The heavy dense forward
// machinery (Qwen3_5DenseModel::/Qwen3_5DenseDecodeGraph::) lives in qwen3_5.cpp
// over the shared DevicePool/matmul/GDN helpers; this TU only wires it into the
// registry. Extracted verbatim (behavior-preserving) from the former
// model_registry.cpp monolith.
#include "vllm/v1/worker/gpu/cudagraph_dispatch.h"
#include "vllm/model_executor/models/model_registry.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/interfaces.h"  // #607 L3 kVisionTowerStageName

#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_common.h"  // kQwen3_5Info, helpers
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
#include "qwen3_5_internal.h"  // W4 DeviceTokenIdsScope
#include "vllm/model_executor/models/qwen3_5_mtp.h"  // SPEC-MTP I5d-pre draft
#include "vllm/platforms/interface.h"  // GetPlatform(device.type) memory-model seam
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"

namespace vllm {
namespace {

bool DenseDecodeGraphEnabled() {
  const char* value = std::getenv("VLLM_CPP_DENSE_DECODE_GRAPH");
  return value == nullptr || value[0] != '0';
}

class Qwen3_5DenseLoadedModel final : public LoadedModel {
 public:
  Qwen3_5DenseLoadedModel(const ModelRegistration& registration,
                          Qwen3_5DenseWeights weights)
      : LoadedModel(registration),
        owned_weights_(std::move(weights)),
        weights_(&*owned_weights_) {}
  Qwen3_5DenseLoadedModel(const ModelRegistration& registration,
                          const Qwen3_5DenseWeights& weights,
                          BorrowedWeightsTag)
      : LoadedModel(registration), weights_(&weights) {}

  const Qwen3_5DenseWeights& weights() const { return *weights_; }
  // #607 L3: non-empty only when the load deliberately left `model.visual.*`
  // unread because every modality the tower serves was at limit 0.
  std::vector<std::string> skipped_towers() const override {
    if (!weights_->vision_skipped) return {};
    return {std::string(kVisionTowerStageName)};
  }
  bool uses_nvfp4_w4a4() const override {
    return !weights_->layers.empty() &&
           weights_->layers.front().mlp.gate_proj_fp4.IsTrueW4A4();
  }
  std::unique_ptr<Qwen3_5DenseDecodeGraph>& decode_graph() {
    return decode_graph_;
  }

  // SPEC-MTP I5d-pre: retain the loaded `mtp.*` draft weights + build the draft
  // sharing this target's embed_tokens/lm_head. Inert unless FromModelDir
  // attached weights (i.e. unless a SpeculativeConfig is configured).
  bool supports_mtp_draft() const override { return true; }
  // SPEC-DFLASH / SPEC-DSPARK: this forward routes to ForwardDeviceMultiTap.
  bool supports_aux_multi_tap() const override { return true; }
  // #1946: LEND the embedding table to a block drafter that shares it, so the
  // DFlash/DFlash2 draft runs the target's own OwnedTensor and `ResidentWeight`
  // uploads it once instead of once per tensor. Null before a loader has filled
  // the table, so a half-built model lends nothing.
  const OwnedTensor* shared_embed_tokens() const override {
    return weights_->embed_tokens.Empty() ? nullptr : &weights_->embed_tokens;
  }
  void AttachMtpDraftWeights(Qwen3_5MTPWeights weights) override {
    mtp_draft_weights_ = std::move(weights);
  }
  std::unique_ptr<Qwen3_5MTPModel> BuildMtpDraft(
      const HfConfig& config) const override {
    if (!mtp_draft_weights_.has_value()) return nullptr;
    return std::make_unique<Qwen3_5MTPModel>(*mtp_draft_weights_, *weights_,
                                             config);
  }

 private:
  std::optional<Qwen3_5DenseWeights> owned_weights_;
  const Qwen3_5DenseWeights* weights_ = nullptr;
  std::unique_ptr<Qwen3_5DenseDecodeGraph> decode_graph_;
  // Retained draft weights (SPEC-MTP I5d-pre); empty on the production default.
  std::optional<Qwen3_5MTPWeights> mtp_draft_weights_;
};

std::unique_ptr<LoadedModel> LoadQwen3_5DenseModel(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind == ModelSource::Kind::kGguf) {
    // Dense-arch (`qwen35`, e.g. Qwen3.5-2B) GGUF: same GDN / full-attention
    // block loaders as the MoE path with a dense SwiGLU MLP per layer.
    if (source.gguf == nullptr) {
      throw std::runtime_error("GGUF model source is empty");
    }
    // ENG-GGUF-RESIDENCY-RESOLVED-DEVICE: the residency policy is built from
    // the device the ENGINE resolved for this load, never from
    // `platforms::CurrentPlatform()`. The two disagree on `--device cpu` on a
    // CUDA-capable process, and this hook is where the disagreement reached the
    // loader.
    const GgufLoadPolicy gguf_policy =
        GgufLoadPolicy::FromEnv(source.device, detail::ActDType(source.device));
    VT_CHECK(!gguf_policy.weight_value_dtype || config.torch_dtype.empty() ||
                 config.torch_dtype == "bfloat16" || config.torch_dtype == "bf16",
             "Qwen3.5 retained F16 GGUF: this forward resolves BF16 model values; "
             "an unsupported model dtype cannot silently select BF16 semantics");
    return std::make_unique<Qwen3_5DenseLoadedModel>(
        registration,
        LoadQwen3_5DenseFromGguf(*source.gguf, config, &gguf_policy));
  }
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture Qwen3_5ForConditionalGeneration does not support "
        "this weight source");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<Qwen3_5DenseLoadedModel>(
      registration,
      LoadQwen3_5Dense(*source.safetensors, config, source.load_queue,
                       source.multimodal));
}

void PrepareQwen3_5Dense(LoadedModel& model, const HfConfig& config,
                         vt::Queue& queue) {
  // MODEL-FP8-BLOCK-LINEAR (#1189 M4). FIRST, before any resident is built: the
  // dense forward READS `Fp8BlockWeight`s now, so what is refused here is a
  // device with no block-scaled GEMM rather than the weight itself. The
  // question is asked at `ModelRegistry::Prepare` — called by every runner
  // before the first forward and before graph capture — so a CUDA user is told
  // here rather than inside the first GEMM or, worse, inside a capture. Inert
  // on CPU and on every other checkpoint. M5 (`489a9a4c0`) narrowed rather than
  // removed it: the CUTLASS kernel covers `VT_CUTLASS_FP8_ARCHS` (12.0a, 12.1a)
  // only, and a CUDA arch outside that cell is still refused here by name.
  RefuseUnrunnableQwen3_5DenseFp8Block(
      ModelAs<Qwen3_5DenseLoadedModel>(model,
                                        "Qwen3_5ForConditionalGeneration")
          .weights(),
      queue.device.type);
  // PERF-27B-LMHEAD-FP4 (issue #213): build the packed lm_head's resident HERE —
  // on CUDA before the runner captures a decode graph, elsewhere before the first
  // forward pays the dequant. Inert on every BF16/FP8/GGUF/tied checkpoint.
  auto& qwen = ModelAs<Qwen3_5DenseLoadedModel>(model, "Qwen3_5ForConditionalGeneration");
  Qwen3_5DenseModel::PrepareLmHeadResident(qwen.weights(), queue);
  // PERF-27B-GDN-FP8-QKVZ: build the merged FP8 GDN [qkv;z] operand here, at
  // model prepare — before the first forward, so it can never allocate or copy
  // inside a CUDA-graph capture. No-op on CPU, on a non-FP8 owner, and when the
  // merge is rolled back (VT_GDN_MERGED_QKVZ_FP8=0).
  Qwen3_5DenseModel::PrepareGdnFp8Resident(qwen.weights(), config, queue);
}

ForwardLogits ForwardQwen3_5Dense(LoadedModel& model,
                                  const ModelForwardInput& input) {
  auto& qwen = ModelAs<Qwen3_5DenseLoadedModel>(model, "Qwen3_5ForConditionalGeneration");
  const Qwen3_5DenseWeights& weights = qwen.weights();

  // ENG-ASYNC-SCHED W4: publish the async runner's device-resident input ids for
  // the duration of THIS forward, so the embed at the top of every route below
  // (eager, gathered, tap, multi-tap, decode-graph replay) reads them instead of
  // uploading the host vector, which is stale for decode rows on that path.
  // Null on every other path, and RAII-scoped so it cannot outlive the call.
  const detail::DeviceTokenIdsScope device_ids_scope(
      input.device_token_ids, static_cast<int64_t>(input.token_ids.size()));

  // SPEC-MTP I5d-pre hidden-state tap. When the spec verify forward requests the
  // drafter's [T,H] post-final-norm hidden (I5d), route to the EXISTING
  // ForwardDeviceTap: byte-identical logits to ForwardDevice, plus the hidden
  // moved into *input.hidden_tap. Null (every spec-off run) falls through to the
  // unchanged path below, so the forward is byte-identical when spec is off.
  if (input.hidden_tap != nullptr) {
    return Qwen3_5DenseModel::ForwardDeviceTap(
        input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
        input.attn_kv, input.gdn_state, weights, input.config, input.queue,
        input.hidden_tap, input.logits_indices);
  }

  // SPEC-DFLASH D1 (DF-AUX-TAPS): non-null routes to ForwardDeviceMultiTap
  // (byte-identical logits + the [T,H×taps] aux capture); null is byte-identical to
  // the path below. Mutually exclusive with hidden_tap.

  // vLLM's CUDA-graph selection is independent of weight quantization. The
  // driver below already captures the shared dense forward, so restricting it
  // to the 27B FP4 checkpoint left ordinary BF16 Qwen3.5 decode eager.
  // #2812/#1625 gate: the ARCH-SCOPED overload, so the evidence families
  // (Qwen3.5-GDN included) capture ambient while every other family keeps
  // the explicit opt-in — the same shape as the qwen3 driver's gate.
  const bool graph_cuda =
      platforms::GetPlatform(input.queue.device.type).support_static_graph_mode() &&
      !platforms::GetPlatform(input.queue.device.type)
           .static_graph_requires_opt_in(input.config.architectures);
  constexpr int kMaxDecodeGraphBatch = 64;

  // SPEC-DSPARK W8 (#442): mirror vLLM's UNIFORM-decode predicate instead of
  // "query_len == 1". Upstream's captured decode length is
  // `1 + num_speculative_tokens` (cudagraph_dispatcher.py:37), so its T=1+k
  // speculative VERIFY is graph-captured; ours fell to the eager path EVERY step,
  // which the paired 35B measurement charged at ~4.8 ms/step (0.870x where
  // acceptance is high). With num_speculative_tokens == 0 this is exactly
  // `pure_decode`, so the non-spec path is unchanged.
  // DEFAULT ON. The staging defect that forced this off is fixed: the captured
  // graph reads gdn_spec_* / num_accepted, which StageSpecStepInputs now refills
  // per step, and the per-request arrays are sized by the REQUEST count rather
  // than the token count. Gated green with capture ON and OFF across the MTP,
  // DFlash, 35B and concurrent e2e suites, and measured +8.5% / +4.7% on the 35B
  // cells (0.870x -> 0.986x of the pinned graphed oracle on the high-acceptance
  // one). VT_SPEC_DECODE_GRAPH=0 restores the eager verify for an A/B.
  static const bool spec_graph = [] {
    const char* v = std::getenv("VT_SPEC_DECODE_GRAPH");
    return v == nullptr || (v[0] != '\0' && v[0] != '0');
  }();
  //
  // ENG-CUDAGRAPH-BREAK W6 (#1374): the predicate itself is the RUNNER's now.
  // This file used to re-derive uniformity, the GDN-prefill conjunct and the
  // configured-width comparison for itself, and `qwen3_5_dense.cpp` carried a
  // byte-identical copy of the same twenty lines -- one predicate written twice,
  // which is the shape this row exists to remove. `input.uniform_query_len` is
  // the answer, computed once in `GPUModelRunner::execute_model` through
  // `v1::ActualUniformDecodeQueryLen`.
  //
  // WHAT WIDENED, AND IT IS [#1020]. The old test demanded the batch's uniform
  // length equal `1 + num_speculative_tokens` EXACTLY, the width configured for
  // the engine's lifetime. The scheduler clamps drafts to the step's token
  // budget, so a step every request entered with the same SHORTER draft prefix
  // is uniform at a shorter length, is exactly the shape a graph can serve, and
  // got none. `uniform_query_len > 1` admits it at its actual depth. The driver
  // keys its slot ring on `(S, q, spec)` so the wider predicate cannot reach a
  // graph captured for a different shape.
  //
  // `> 1` rather than `>= 1` because the pure-decode arm is `input.pure_decode`
  // on the left: at q == 1 the two are the same population and `pure_decode` is
  // the one every other driver reads. A value above 1 also PROVES speculation is
  // configured, because the runner bounds the length by `1 + num_spec()`.
  const bool uniform_decode =
      input.pure_decode ||
      (spec_graph && input.uniform_query_len > 1 &&
       // The captured region emits EVERY row, so only a no-op gather may take it.
       // A spec verify needs all 1+k rows anyway, so this holds there.
       (input.logits_indices.empty() ||
        static_cast<int64_t>(input.logits_indices.size()) ==
            input.attn_meta.num_actual_tokens));
  if (std::getenv("VT_TT_ALLOC_TRACE") != nullptr)
    std::fprintf(stderr,
                 "[TT-FWD] route T=%zu num_reqs=%d pure_decode=%d "
                 "uniform_qlen=%lld gather=%d n_idx=%zu -> %s\n",
                 input.token_ids.size(), input.attn_meta.num_reqs,
                 input.pure_decode ? 1 : 0,
                 static_cast<long long>(input.uniform_query_len),
                 input.gather_logits ? 1 : 0, input.logits_indices.size(),
                 uniform_decode ? "graph" : "eager");
  if (DenseDecodeGraphEnabled() && uniform_decode && graph_cuda &&
      input.num_reqs <= kMaxDecodeGraphBatch) {
    if (!qwen.decode_graph()) {
      qwen.decode_graph() = std::make_unique<Qwen3_5DenseDecodeGraph>(
          weights, input.config, input.queue, input.gdn_state_slots);
    }
    return qwen.decode_graph()->Step(
        input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
        input.attn_kv, input.gdn_state, input.aux_tap);
  }

  // SPEC-DSPARK W8 (#442): the aux multi-tap forward is the DFlash/DSpark
  // VERIFY path, and it used to return BEFORE the decode-graph gate, which is
  // why the T=1+k verify never captured. The graph branch above now serves it
  // (writing the taps into the slot's persistent buffer); this remains the
  // eager fallback for every batch the graph declines.
  if (input.aux_tap != nullptr) {
    return Qwen3_5DenseModel::ForwardDeviceMultiTap(
        input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
        input.attn_kv, input.gdn_state, weights, input.config, input.queue,
        input.aux_tap, input.logits_indices);
  }
  if (input.gather_logits) {
    return Qwen3_5DenseModel::ForwardDevice(
        input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
        input.attn_kv, input.gdn_state, weights, input.config, input.queue,
        input.logits_indices);
  }
  return HostLogits(
      Qwen3_5DenseModel::Forward(
          input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
          input.attn_kv, input.gdn_state, weights, input.config, input.queue,
          input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kQwen3_5DenseFactory{
    .parse_config = &ParseQwen3_5Config,
    .load_weights = &LoadQwen3_5DenseModel,
    .prepare = &PrepareQwen3_5Dense,
    .forward = &ForwardQwen3_5Dense,
    .make_kv_cache = &MakeQwen3_5KVCache,
    .is_dense_model = true,
    .consumes_device_token_ids = true,
};

}  // namespace

std::unique_ptr<LoadedModel> MakeQwen3_5DenseLoadedModel(
    Qwen3_5DenseWeights weights) {
  return std::make_unique<Qwen3_5DenseLoadedModel>(
      RegistrationFor("Qwen3_5ForConditionalGeneration"), std::move(weights));
}

std::unique_ptr<LoadedModel> MakeQwen3_5DenseLoadedModel(
    Qwen3_5DenseWeights weights, const HfConfig& config) {
  const ModelRegistration* registration =
      &RegistrationFor("Qwen3_5ForConditionalGeneration");
  try {
    const ModelRegistration& resolved = ModelRegistry::Resolve(config);
    if (resolved.factory != nullptr &&
        resolved.factory->load_weights == kQwen3_5DenseFactory.load_weights) {
      registration = &resolved;
    }
  } catch (const std::exception&) {
    // An unregistered architecture keeps the default, as before.
  }
  return std::make_unique<Qwen3_5DenseLoadedModel>(*registration,
                                                   std::move(weights));
}

std::unique_ptr<LoadedModel> BorrowQwen3_5DenseLoadedModel(
    const Qwen3_5DenseWeights& weights) {
  return std::make_unique<Qwen3_5DenseLoadedModel>(
      RegistrationFor("Qwen3_5ForConditionalGeneration"), weights,
      BorrowedWeightsTag{});
}

REGISTER_VLLM_MODEL(qwen3_5_dense, "Qwen3_5ForConditionalGeneration",
                    kQwen3_5DenseFactory, kQwen3_5Info)

// TEXT-ONLY arm of the SAME backbone. Upstream's `Qwen3_5ForCausalLM` IS
// `Qwen3_5ForCausalLMBase` unchanged (`class Qwen3_5ForCausalLM(...): pass`,
// qwen3_5.py:439-440 @ `ad5d29db7`) and is registered against the same `qwen3_5`
// module (registry.py:202 @ `ad5d29db7`, PR #50210), so this is the SAME
// factory, additively: no forward, no KV-cache spec and no loader fork.
//
// AHEAD OF THE PIN, DELIBERATELY. `555967922` (.agents/upstream-sync.md) carries
// only the ForConditionalGeneration entries; the text-only arms landed upstream
// after it. This is a forward port of ONE upstream PR and does not advance the
// pin. See .agents/specs/qwen38-text-only.md §Gates for the owed run gate.
REGISTER_VLLM_MODEL(qwen3_5_dense_text, "Qwen3_5ForCausalLM",
                    kQwen3_5DenseFactory, kQwen3_5TextInfo)

}  // namespace vllm
