// Ported from:
//   vllm/tests/models/test_registry.py:31-103,151-159
//   vllm/tests/models/test_initialization.py:50-197
// at e24d1b24fe96a56ba8b0d653efa076d03eb95d6c.
// Negative cases additionally pin registry.py:_raise_for_unsupported:1051-1082,
// which has no direct upstream test at this commit.
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/opt.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/model_executor/models/qwen3_5_common.h"
#include "vllm/v1/kv_cache_interface.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using vllm::HfConfig;
using vllm::ModelRegistration;
using vllm::ModelRegistry;
using vllm::UnsupportedModelInfo;

namespace {

std::span<const std::string> ArchSpan(const std::vector<std::string>& archs) {
  return std::span<const std::string>(archs);
}

HfConfig Config(std::vector<std::string> architectures) {
  HfConfig config;
  config.architectures = std::move(architectures);
  return config;
}

}  // namespace

TEST_CASE("registry_imports: every registered architecture has a complete factory") {
  const auto registrations = ModelRegistry::Registrations();
  // 36 text archs + the 3 Parakeet transcription-only archs (ARCH-ONE-SURFACE
  // ROW 1: ParakeetForCTC/ForRNNT/ForTDT, SupportsTranscription mirror) + the
  // LlamaModel embedding arch (ARCH-ONE-SURFACE ROW 6, is_pooling_model).
  // 38 -> 40 on MODEL-QWEN38-TEXT-ONLY: `Qwen3_5ForCausalLM` and
  // `Qwen3_5MoeForCausalLM`, the TEXT-ONLY arms of the Qwen3.5 backbone
  // (upstream registry.py:202-203 @ `ad5d29db7`), registered against the
  // EXISTING dense and MoE factories.
  // 40 -> 41 on MODEL-MM-dots3-note (#699 W1): `Dots3NoteForCausalLM`, its own
  // additive TU. Upstream registers it in _MULTIMODAL_MODELS (registry.py:381
  // @ `c205726108df54bb6fbf15b19e725a4a3add2b18`, BEYOND our parity pin). Its
  // speculative head `Dots3NoteMTPModel` (registry.py:670) is INVENTORIED and
  // deliberately NOT registered, so it adds one entry and not two.
  // 42 -> 43 on MODEL-MM-GLM53-FLASH (#2067 W1):
  // `Glm5NextForConditionalGeneration`, its own additive TU. Registered by NO
  // vLLM revision -- absent from our parity pin AND from vLLM `main`, with
  // vllm#53906 open and therefore inadmissible -- so there is no upstream
  // registry line to cite here. It moves the count by ONE and not by two or
  // three: the checkpoint's 46th layer is an MTP block the transformers
  // reference discards rather than a second architecture string, and the
  // text-only `Glm5NextForCausalLM` is declared by no published artifact.
  // 44 -> 45 on MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm (W1,
  // ISSUE-LOCAL-01M2C04E8N4NTXYNS584M1TSFP): `DeepseekV41ForCausalLM`, its own
  // additive TU. Upstream registers it at `registry.py:379` @ `e77daef89e`,
  // BEYOND our parity pin, and registration claims no gate against the oracle
  // (the standing #490 gives `Qwen3_5ForCausalLM`). It moves the count by ONE
  // and not by two: upstream's text-only `DeepseekV41LLMForCausalLM` is
  // internal and declared by no published artifact, and the `DSparkV41DraftModel`
  // speculator (registry.py:647) is INVENTORIED and deliberately NOT registered.
  // 45 -> 46 on MODEL-GLINER25: `BoundaryExtractor`, its own additive TU. No
  // vLLM registration (vLLM has no DeBERTa support), ported from scratch against
  // the HuggingFace transformers + GLiNER2 library references. A POOLING
  // model (is_pooling_model=true) like `LlamaModel`, not a text-generation arch.
  // 46 -> 47 on MODEL-CUA-S1-FORMS: `CuaS1Forms`, a byte-level TinyTransformer
  // scorer ported from the `trycua/cua` reference. Also a POOLING model.
  // 47 -> 48 on MODEL-LAYA: `LayaModel`, a ModernBERT-large backbone with a
  // SystemOne decision head. Also a POOLING model (is_pooling_model=true).
  // 48 -> 49 on MODEL-KEV: `KevModel`, its own additive TU. A POOLING model
  // (is_pooling_model=true) built on a Qwen3.5 dense backbone + PointerHead
  // readout, ported from jaredpalmer/kev. No vLLM registration; ported from
  // scratch against the kev reference implementation.
  // 49 -> 50 on MODEL-CLM: `ClmModel`, its own additive TU. A POOLING model
  // (is_pooling_model=true) — bi-encoder (Qwen3-8B + dual MLP projection
  // heads) served via /v1/systemone. No vLLM registration; ported from
  // scratch against the Contrastive-LM/CLM reference implementation.
  // 50 -> 51 on MODEL-GLINER25-DECIDE: `SpanExtractor`, its own additive TU.
  // A POOLING model (is_pooling_model=true) — DeBERTa-v3 encoder +
  // classification head served via /v1/systemone. No vLLM registration;
  // ported from scratch.
  // 51 -> 52 on MODEL-XOR: `XorModel`, its own additive TU. A POOLING model
  // (is_pooling_model=true) — 35B MoE (Qwen3.6-35B-A3B) with
  // ForwardMoeHidden and forward+reverse calibration, served via
  // /v1/systemone. No vLLM registration; ported from scratch against the
  // juspay/xor reference.
  // 52 -> 53 on MODEL-TEV1: `Tev1Model`, its own additive TU. Unlike the
  // other SystemOne models, Tev1 is a text-generation model (standard
  // Qwen3.5-4B forward + lm_head, NOT pooling). Served via
  // /v1/chat/completions, NOT /v1/systemone. Ported from
  // togethercomputer/Tev1-4B-experimental.
  // 53 -> 54 on MODEL-TEXT-mimo-v2 (W1, ISSUE-LOCAL-01M3F82S8ZYCTTDSKPFF5PGAH7):
  // `MiMoV2ForCausalLM`, its own additive TU. A beyond-pin hybrid text-only
  // LLM (48 layers, full-attn every 6th + SWA, v_head_dim != head_dim, MoE)
  // with no vLLM registration. The weight loader (W2) and forward (W3) are
  // stubs; W1 gate is: config parses, KV-cache spec builds, model resolves.
  // 54 -> 56 on MODEL-MM-nano-nemotron-vl-nemotron-h-nano-vl-v2
  // (ISSUE-LOCAL-01M3RY6G385D41W5SNF1C85RRS): `NemotronH_Nano_VL_V2` and
  // `NemotronH_Nano_Omni_Reasoning_V3`, one additive TU
  // (nano_nemotron_vl_registry.cpp), upstream registry.py:512-513 @ e126687a9a.
  // 54 -> 55 on MODEL-TEXT-cohere2-moe (ISSUE-LOCAL-01M3S23H9B25EFPFJN75Y1XBWY):
  // `Cohere2MoeForCausalLM` (North), its own additive TU, registered upstream at
  // `registry.py:86` @ `a7c23ac96d`. Its EAGLE drafter is not registered.
  // Combined: 54 + 2 (Nemotron Nano VL/Omni) + 1 (Nimble) + 1 (Cohere2 North) = 58.
  // `NimbleModel` (MODEL-NIMBLE) is a POOLING decision model like ClmModel.
  // 58 -> 59 on MODEL-TEXT-kolibri-1 (W1,
  // ISSUE-LOCAL-01M41VCM4Z0JEFQHBFS5VT0GYA): `Kolibri1ForCausalLM`, its own
  // additive TU. A beyond-pin hybrid text-only LLM (50 layers, 4 SWA : 1
  // full RNoPE, MoE 384x6 + shared) with no vLLM registration. W1 gate is:
  // config parses, KV-cache spec builds, model resolves, FP8-block loader
  // reads; the forward refuses until its wave.
  REQUIRE(registrations.size() == 59);

  for (const ModelRegistration& registration : registrations) {
    CAPTURE(registration.architecture);
    REQUIRE(registration.factory != nullptr);
    CHECK(registration.factory->parse_config != nullptr);
    CHECK(registration.factory->load_weights != nullptr);
    CHECK(registration.factory->prepare != nullptr);
    CHECK(registration.factory->forward != nullptr);
    CHECK(registration.factory->make_kv_cache != nullptr);

    const HfConfig config = Config({std::string(registration.architecture)});
    CHECK(&ModelRegistry::Resolve(config) == &registration);
  }
}

TEST_CASE("self_registration: every arch self-registers from its own TU") {
  // With the fixed kRegistrations array replaced by REGISTER_VLLM_MODEL
  // static-init self-registration (qwen3_5_dense.cpp + qwen3_5_moe.cpp each
  // register themselves), the process-global registry must still contain both
  // implemented architectures with a complete factory — proving the static
  // Registrars ran and populated the shared ModelFactory registry.
  const auto registrations = ModelRegistry::Registrations();
  const auto has_arch = [&](std::string_view arch) {
    for (const ModelRegistration& r : registrations) {
      if (r.architecture == arch) return r.factory != nullptr;
    }
    return false;
  };
  CHECK(has_arch("Qwen3ForCausalLM"));
  CHECK(has_arch("Qwen3MoeForCausalLM"));
  // The CROSS-FAMILY canary: OPT is not a Qwen variant at all, and it still
  // needed nothing but its own TU + one REGISTER_VLLM_MODEL line.
  CHECK(has_arch("OPTForCausalLM"));
  CHECK(has_arch("Qwen3_5ForConditionalGeneration"));
  CHECK(has_arch("Qwen3_5MoeForConditionalGeneration"));
  // The text-only arms of that same backbone: two more REGISTER_VLLM_MODEL
  // lines against the SAME factories, ZERO shared-array edit.
  CHECK(has_arch("Qwen3_5ForCausalLM"));
  CHECK(has_arch("Qwen3_5MoeForCausalLM"));
  // MLA campaign W7: the DeepSeek-V2 MLA model, likewise one new TU + one
  // REGISTER_VLLM_MODEL line and ZERO edit to a shared array.
  CHECK(has_arch("DeepseekV2ForCausalLM"));
  // The CROSS-FAMILY dense bring-up: Llama reuses the shared dense forward
  // (qk-norm skipped + llama3 rope-scaling, both additive in the shared
  // AttnBlock) and adds only its own loader/registry TU + one REGISTER line.
  CHECK(has_arch("LlamaForCausalLM"));
  // The FIFTH family: Mistral is Llama minus llama3-rope (plain rope, untied
  // lm_head), landing as ONE loader/registry TU + one REGISTER line, ZERO
  // shared-array edit, reusing the shared dense forward verbatim.
  CHECK(has_arch("MistralForCausalLM"));
  // The FIRST GLM-family model (spike G2): GLM-4-9B-0414 dense, adding the
  // partial-interleaved-rope + sandwich-norm primitives as ONE forward/loader/
  // registry TU + one REGISTER line, ZERO shared-array edit.
  CHECK(has_arch("Glm4ForCausalLM"));
  // The FIRST Gemma-family model (sweep W2): Gemma-3 text (gemma-3-1b-it),
  // adding the GeGLU + dual-rope-theta + Gemma-RMSNorm-sandwich + embed-scale
  // primitives as ONE forward/loader/registry TU + one REGISTER line.
  CHECK(has_arch("Gemma3ForCausalLM"));
  // Sweep W4/W5: Gemma-2 (attn+final logit soft-cap, the inverse of Gemma-3's
  // QK-norm) and Gemma-1 (the plain two-fused-norm original), each one
  // forward/loader/registry TU + one REGISTER line, ZERO shared-array edit.
  CHECK(has_arch("Gemma2ForCausalLM"));
  CHECK(has_arch("GemmaForCausalLM"));
  // GLM-4.7-Flash (MLA + GLM MoE), likewise one TU + one REGISTER line.
  CHECK(has_arch("Glm4MoeLiteForCausalLM"));
  // The additive-family sweep since: each below is likewise ONE new
  // forward/loader/registry TU + one REGISTER_VLLM_MODEL line, ZERO shared-array
  // edit — the cross-family dense/MoE seam absorbing the recent-model breadth pass.
  CHECK(has_arch("CohereForCausalLM"));        // Cohere Command-R
  CHECK(has_arch("Cohere2MoeForCausalLM"));    // Cohere North (MoE, sliding window)
  CHECK(has_arch("GraniteForCausalLM"));       // IBM Granite-3
  CHECK(has_arch("InternLM2ForCausalLM"));     // InternLM2
  CHECK(has_arch("InternLM3ForCausalLM"));     // InternLM3 (reuses the Llama TU)
  CHECK(has_arch("MiniCPMForCausalLM"));       // MiniCPM
  CHECK(has_arch("MiniCPM3ForCausalLM"));      // MiniCPM3 (MLA)
  CHECK(has_arch("Olmo2ForCausalLM"));         // OLMo-2
  CHECK(has_arch("Olmo3ForCausalLM"));         // OLMo-3 (reuses the OLMo-2 TU)
  CHECK(has_arch("PhiForCausalLM"));           // Phi-1/1.5/2
  CHECK(has_arch("Phi3ForCausalLM"));          // Phi-3
  CHECK(has_arch("StableLmForCausalLM"));      // StableLM
  // Qwen3-VL (MM-ENGINE-FORWARD): the first NON-hybrid MULTIMODAL registration —
  // the vision-language model now driven by ModelRegistry::Forward, not a
  // standalone driver. One new registry TU + one REGISTER line, ZERO array edit.
  CHECK(has_arch("Qwen3VLForConditionalGeneration"));
  // Kimi-Linear-48B-A3B (CLAIM-KIMI-LINEAR-W1): the first text-only HYBRID
  // registration (20 KDA linear-attn + 7 NoPE-MLA layers), likewise ONE new
  // forward/loader/registry TU + one REGISTER line, ZERO shared-array edit.
  CHECK(has_arch("KimiLinearForCausalLM"));
  // GLM-5.3 (`MODEL-TEXT-deepseek-v2-glm-moe-dsa-for-causal-lm` W2, #2214): the
  // architecture vLLM routes into `deepseek_v2` at `registry.py:117` and that our
  // OWN DeepSeek-V2 loader refuses at its `index_topk` tripwire, so it registers
  // from its own TU with its own params struct rather than sharing one.
  CHECK(has_arch("GlmMoeDsaForCausalLM"));
  // GLiNER2.5 (MODEL-GLINER25): the first NER/extraction model — a POOLING
  // architecture (DeBERTa v2 encoder + GLiNER2 boundary head) with no vLLM
  // precedent, ported from scratch against HuggingFace + GLiNER2 library.
  CHECK(has_arch("BoundaryExtractor"));
  // MODEL-CLM: bi-encoder (Qwen3-8B + dual MLP heads), POOLING, /v1/systemone.
  CHECK(has_arch("ClmModel"));
  // MODEL-GLINER25-DECIDE: DeBERTa-v3 + classification head, POOLING.
  CHECK(has_arch("SpanExtractor"));
  // MODEL-XOR: 35B MoE (Qwen3.6-35B-A3B) + ForwardMoeHidden, POOLING.
  CHECK(has_arch("XorModel"));
  // MODEL-TEV1: Qwen3.5-4B SFT, text-generation (NOT pooling), /v1/chat/completions.
  CHECK(has_arch("Tev1Model"));
  CHECK(has_arch("NimbleModel"));

  // Registration arrival order across TUs is unspecified under C++ static init,
  // so the registry imposes a stable canonical sort by architecture name (byte
  // order — std::string operator<). This makes SupportedArchs()/error-message
  // order deterministic. Kept alias-for-alias with the enumerated set below and
  // with the kExampleConfigArchitectures ledger; adding a model appends its two
  // entries here.
  const std::vector<std::string_view> supported = ModelRegistry::SupportedArchs();
  REQUIRE(supported.size() == 59);
  CHECK(std::is_sorted(supported.begin(), supported.end()));
  // The full byte-order sequence. Note "MiniCPM3" < "MiniCPMF" and "Phi3" <
  // "PhiF" ('3' 0x33 < 'F' 0x46); "OPT" < "Olmo" ('P' 0x50 < 'l' 0x6C); and among
  // the Qwens "Qwen3F" < "Qwen3M" < "Qwen3V" < "Qwen3_"
  // ('F' 0x46 < 'M' 0x4D < 'V' 0x56 < '_' 0x5F). Within "Qwen3_5[Moe]For",
  // "Ca" < "Co" ('a' 0x61 < 'o' 0x6F), so each text-only ForCausalLM arm sorts
  // immediately BEFORE its ForConditionalGeneration wrapper.
  const std::vector<std::string_view> kSortedArchs{
      // 'B' 0x42 < 'C' 0x43: BoundaryExtractor sorts before CohereForCausalLM.
      "BoundaryExtractor",
      // "ClmModel" (Cl) < "CohereForCausalLM" (Co): l=0x6C < o=0x6F.
      "ClmModel",
      // "Cohere2Moe" < "CohereFor": they agree through "Cohere" and then
      // differ at '2' 0x32 against 'F' 0x46.
      "Cohere2MoeForCausalLM",
      "CohereForCausalLM",
      // 'C' 0x43: "CuaS1Forms" sorts after "CohereForCausalLM" ('u' 0x75 > 'o'
      // 0x6F) and before "DeepseekV2ForCausalLM" ('C' 0x43 < 'D' 0x44).
      "CuaS1Forms",
      "DeepseekV2ForCausalLM",
      // "DeepseekV41For" < "DeepseekV4For": they agree through "DeepseekV4" and
      // then differ at '1' 0x31 against 'F' 0x46, so the V4.1 arm sorts BEFORE
      // the V4 one rather than after it.
      "DeepseekV41ForCausalLM",
      "DeepseekV4ForCausalLM",
      "Dots3NoteForCausalLM",
      "Gemma2ForCausalLM",
      "Gemma3ForCausalLM",
      "Gemma4ForConditionalGeneration",
      "Gemma4UnifiedForConditionalGeneration",
      "GemmaForCausalLM",
      "Glm4ForCausalLM",
      "Glm4MoeLiteForCausalLM",
      "Glm5NextForConditionalGeneration",
      "GlmMoeDsaForCausalLM",
      "GraniteForCausalLM",
      "InternLM2ForCausalLM",
      "InternLM3ForCausalLM",
      "KevModel",
      "KimiK3ForConditionalGeneration",
      "KimiLinearForCausalLM",
      // "Kolibri1" sorts before "Laguna": 'K' (0x4B) < 'L' (0x4C).
      "Kolibri1ForCausalLM",
      "LagunaForCausalLM",
      "LayaModel",
      "LlamaForCausalLM",
      "LlamaModel",
      // "MiMoV2" sorts before "MiniCPM": 'M' (0x4D) < 'n' (0x6E) at index 2.
      "MiMoV2ForCausalLM",
      "MiniCPM3ForCausalLM",
      "MiniCPMForCausalLM",
      "MistralForCausalLM",
      "MuseGlimmerForCausalLM",
      "MuseGlimmerForConditionalGeneration",
      "NemotronHForCausalLM",
      // "NemotronHF" < "NemotronH_" ('F' 0x46 < '_' 0x5F), and "..._Nano_O" <
      // "..._Nano_V" ('O' 0x4F < 'V' 0x56).
      "NemotronH_Nano_Omni_Reasoning_V3",
      "NemotronH_Nano_VL_V2",
      "NimbleModel",
      "OPTForCausalLM",
      "Olmo2ForCausalLM",
      "Olmo3ForCausalLM",
      "ParakeetForCTC",
      "ParakeetForRNNT",
      "ParakeetForTDT",
      "Phi3ForCausalLM",
      "PhiForCausalLM",
      "Qwen3ForCausalLM",
      "Qwen3MoeForCausalLM",
      "Qwen3VLForConditionalGeneration",
      "Qwen3_5ForCausalLM",
      "Qwen3_5ForConditionalGeneration",
      "Qwen3_5MoeForCausalLM",
      "Qwen3_5MoeForConditionalGeneration",
      "Qwen4ExpForConditionalGeneration",
      // "SpanExtractor" (Sp) < "StableLmForCausalLM" (St): p=0x70 < t=0x74.
      "SpanExtractor",
      "StableLmForCausalLM",
      // "Tev1Model" (T) sorts after StableLmForCausalLM (S): T=0x54 > S=0x53.
      "Tev1Model",
      // "XorModel" (X) sorts after Tev1Model (T): X=0x58 > T=0x54.
      "XorModel",
  };
  REQUIRE(supported.size() == kSortedArchs.size());
  for (size_t i = 0; i < supported.size(); ++i) {
    CAPTURE(i);
    CHECK(supported[i] == kSortedArchs[i]);
  }

  // The dense/MoE scheduler policy split survives the per-variant TU move.
  std::unique_ptr<vllm::LoadedModel> dense =
      vllm::MakeQwen3_5DenseLoadedModel(vllm::Qwen3_5DenseWeights{});
  CHECK(ModelRegistry::IsDenseModel(*dense));
}

TEST_CASE("registry_model_property: Qwen registrations match pinned _ModelInfo") {
  for (const ModelRegistration& registration : ModelRegistry::Registrations()) {
    CAPTURE(registration.architecture);
    if (registration.architecture.substr(0, 8) == "Parakeet") {
      // ARCH-ONE-SURFACE ROW 1: the ASR family mirrors SupportsTranscription
      // (interfaces.py:1110-1118) — transcription-capable, transcription-ONLY,
      // NOT a text-generation model; the text entrypoints refuse it by task.
      CHECK_FALSE(registration.info.is_text_generation_model);
      CHECK(registration.info.supports_transcription);
      CHECK(registration.info.supports_transcription_only);
      CHECK_FALSE(registration.info.is_pooling_model);
      CHECK_FALSE(registration.info.is_hybrid);
      CHECK_FALSE(registration.info.supports_multimodal);
      continue;
    }
    if (registration.architecture == "LlamaModel") {
      // ARCH-ONE-SURFACE ROW 6: the EMBEDDING conversion of the Llama backbone
      // (_EMBEDDING_MODELS "LlamaModel" -> ("llama", "LlamaForCausalLM"),
      // registry.py:230, wrapped by as_embedding_model adapters.py:230) — a
      // POOLING model with NO text-generation path; the text entrypoints
      // refuse it by task and vllm_embed / /v1/embeddings serve it.
      CHECK(registration.info.is_pooling_model);
      CHECK_FALSE(registration.info.is_text_generation_model);
      CHECK_FALSE(registration.info.supports_transcription);
      CHECK_FALSE(registration.info.supports_transcription_only);
      CHECK_FALSE(registration.info.is_hybrid);
      CHECK_FALSE(registration.info.supports_multimodal);
      continue;
    }
    if (registration.architecture == "LayaModel") {
      // MODEL-LAYA: a POOLING model (ModernBERT-large backbone + SystemOne
      // decision head) with NO text-generation path. Served via the
      // /v1/systemone endpoint; the text entrypoints refuse by task. Ported
      // from the convaiinnovations/laya reference (no vLLM precedent).
      CHECK(registration.info.is_pooling_model);
      CHECK_FALSE(registration.info.is_text_generation_model);
      CHECK_FALSE(registration.info.supports_transcription);
      CHECK_FALSE(registration.info.supports_transcription_only);
      CHECK_FALSE(registration.info.is_hybrid);
      CHECK_FALSE(registration.info.supports_multimodal);
      continue;
    }
    if (registration.architecture == "BoundaryExtractor") {
      // MODEL-GLINER25: a POOLING model (DeBERTa v2 encoder + GLiNER2 boundary
      // head) with NO text-generation path — the same task-routing shape as
      // LlamaModel above. vllm_embed / /v1/embeddings serve it; the text
      // entrypoints refuse by task. No vLLM precedent (ported from scratch).
      CHECK(registration.info.is_pooling_model);
      CHECK_FALSE(registration.info.is_text_generation_model);
      CHECK_FALSE(registration.info.supports_transcription);
      CHECK_FALSE(registration.info.supports_transcription_only);
      CHECK_FALSE(registration.info.is_hybrid);
      CHECK_FALSE(registration.info.supports_multimodal);
      continue;
    }
    if (registration.architecture == "CuaS1Forms") {
      // MODEL-CUA-S1-FORMS: a POOLING model (byte-level TinyTransformer scorer
      // with cross-attention scoring head) with NO text-generation path. Served
      // via the /v1/score endpoint; the text entrypoints refuse by task. Ported
      // from the `trycua/cua` reference (no vLLM precedent).
      CHECK(registration.info.is_pooling_model);
      CHECK_FALSE(registration.info.is_text_generation_model);
      CHECK_FALSE(registration.info.supports_transcription);
      CHECK_FALSE(registration.info.supports_transcription_only);
      CHECK_FALSE(registration.info.is_hybrid);
      CHECK_FALSE(registration.info.supports_multimodal);
      continue;
    }
    if (registration.architecture == "KevModel") {
      // MODEL-KEV: a POOLING model (Qwen3.5 dense backbone + PointerHead
      // readout) with NO text-generation path — the same task-routing shape as
      // LlamaModel above. Served by /v1/systemone; the text entrypoints refuse
      // by task. No vLLM precedent (ported from scratch against jaredpalmer/kev).
      CHECK(registration.info.is_pooling_model);
      CHECK_FALSE(registration.info.is_text_generation_model);
      CHECK_FALSE(registration.info.supports_transcription);
      CHECK_FALSE(registration.info.supports_transcription_only);
      CHECK(registration.info.is_hybrid);
      CHECK_FALSE(registration.info.supports_multimodal);
      continue;
    }
    if (registration.architecture == "ClmModel" ||
        registration.architecture == "NimbleModel" ||
        registration.architecture == "SpanExtractor" ||
        registration.architecture == "XorModel") {
      // MODEL-CLM, MODEL-GLINER25-DECIDE, MODEL-XOR: pooling decision models
      // served via /v1/systemone. No text-generation path.
      CHECK(registration.info.is_pooling_model);
      CHECK_FALSE(registration.info.is_text_generation_model);
      CHECK_FALSE(registration.info.supports_transcription);
      CHECK_FALSE(registration.info.supports_transcription_only);
      continue;
    }
    CHECK(registration.info.is_text_generation_model);
    // No text arch is transcription-capable (SupportsTranscription default).
    CHECK_FALSE(registration.info.supports_transcription);
    CHECK_FALSE(registration.info.supports_transcription_only);
    CHECK_FALSE(registration.info.is_pooling_model);
    if (registration.architecture != "Tev1Model") {
      // MODEL-TEV1: has_inner_state=true (GDN recurrent state in Qwen3.5
      // backbone). All other text-generation models have has_inner_state=false.
      CHECK_FALSE(registration.info.has_inner_state);
    }
    CHECK(registration.info.score_type == "bi-encoder");
    if (registration.architecture == "Qwen3_5ForConditionalGeneration" ||
        registration.architecture == "Qwen3_5MoeForConditionalGeneration" ||
        registration.architecture == "Qwen4ExpForConditionalGeneration" ||
        registration.architecture == "Glm5NextForConditionalGeneration" ||
        registration.architecture == "KimiK3ForConditionalGeneration") {
      // The outer Qwen3.5 + Kimi-K3 multimodal wrappers inherit IsHybrid but not
      // HasInnerState; their inner language-model classes carry HasInnerState.
      // These ConditionalGeneration wrappers are the hybrid+multimodal
      // registrations (Kimi-K3 = KDA+MLA hybrid backbone + MoonViT-V2 vision).
      CHECK(registration.info.is_hybrid);
      CHECK(registration.info.supports_multimodal);
    } else if (registration.architecture == "Qwen3_5ForCausalLM" ||
               registration.architecture == "Qwen3_5MoeForCausalLM" ||
               registration.architecture == "Tev1Model") {
      // MODEL-QWEN38-TEXT-ONLY: upstream's `Qwen3_5ForCausalLMBase` inherits
      // HasInnerState + IsHybrid but NOT SupportsMultiModal
      // (qwen3_5.py:287-296 @ `ad5d29db7`) — these are the TEXT arms, and the
      // ConditionalGeneration wrappers above are the multimodal registrations.
      // has_inner_state stays false by the blanket assertion above, the same
      // kQwen3_5Info convention KimiLinearForCausalLM follows.
      //
      // MODEL-TEV1: same Qwen3.5 dense backbone, same hybrid+inner-state
      // shape, but is_text_generation_model=true and is_pooling_model=false
      // (served via /v1/chat/completions, NOT /v1/systemone).
      CHECK(registration.info.is_hybrid);
      CHECK_FALSE(registration.info.supports_multimodal);
    } else if (registration.architecture == "NemotronHForCausalLM") {
      // Nemotron-H (MODEL-NEMOTRON-H W3, #517): text-only HYBRID -- 23 Mamba2
      // layers carry a recurrent-state KV group alongside 6 GQA full-attention
      // layers. Upstream's class carries HasInnerState + IsHybrid
      // (nemotron_h.py:700-712); has_inner_state stays false here under the
      // blanket assertion above, following the kQwen3_5Info / kKimiLinearInfo
      // convention (our ModelInfo is a consumed subset whose only reader
      // short-circuits on is_hybrid). Text-only: no vision or audio tower.
      CHECK(registration.info.is_hybrid);
      CHECK_FALSE(registration.info.supports_multimodal);
    } else if (registration.architecture == "NemotronH_Nano_VL_V2" ||
               registration.architecture == "NemotronH_Nano_Omni_Reasoning_V3") {
      // The NemotronH hybrid language tower (a Mamba2 recurrent-state group)
      // WITH a vision tower: upstream's class carries HasInnerState, IsHybrid
      // and SupportsMultiModal (nano_nemotron_vl.py:899-901), and
      // kNanoNemotronVLFactory carries encode_mm/embed_mm, so the flag is
      // backed. Audio and video are refused by name at encode time.
      CHECK(registration.info.is_hybrid);
      CHECK(registration.info.supports_multimodal);
    } else if (registration.architecture == "KimiLinearForCausalLM") {
      // Kimi-Linear-48B-A3B: text-only HYBRID (20 KDA linear-attn layers ⇒ a GDN
      // recurrent-state KV group + 7 NoPE-MLA layers) — is_hybrid YES,
      // supports_multimodal NO (its K3 wrapper is the multimodal registration).
      // has_inner_state stays false (the blanket assertion above): our ModelInfo is
      // a consumed subset whose only reader short-circuits on is_hybrid, so we
      // follow the GDN-hybrid twin kQwen3_5Info convention.
      CHECK(registration.info.is_hybrid);
      CHECK_FALSE(registration.info.supports_multimodal);
    } else if (registration.architecture == "MiMoV2ForCausalLM" ||
               registration.architecture == "Kolibri1ForCausalLM") {
      // MODEL-TEXT-mimo-v2 W1 (ISSUE-LOCAL-01M3F82S8ZYCTTDSKPFF5PGAH7):
      // text-only HYBRID — 8 full-attention layers (every 6th) + 40 SWA
      // layers, each with a DIFFERENT KV geometry (4 vs 8 KV heads,
      // v_head_dim != head_dim). is_hybrid YES, supports_multimodal NO
      // (vision_config/audio_config are explicitly out of scope; a future
      // MODEL-MM-mimo-v2 row will cover the multimodal wrapper).
      CHECK(registration.info.is_hybrid);
      CHECK_FALSE(registration.info.supports_multimodal);
    } else if (registration.architecture == "Qwen3VLForConditionalGeneration" ||
               registration.architecture == "Dots3NoteForCausalLM" ||
               // MODEL-MM-deepseek-v4 W4 (#2411): `DeepseekV4ForCausalLM` names
               // BOTH the DeepSeek-V4 text checkpoint and the Flash-Vision one,
               // so the architecture cannot advertise this conditionally. It
               // joins this branch on the same terms as `Dots3NoteForCausalLM`
               // -- `kDeepseekV4Factory` carries `encode_mm` and `embed_mm`, so
               // a `deepseek4v` projector reaches the model forward -- and it
               // is NOT hybrid for the same reason: its MLA layers page one
               // cache and the sliding half is a window on it, not a recurrent
               // state. What keeps a TEXT checkpoint inert is
               // `DeepseekV4LoadedModel::has_vision()` being false, which is a
               // fact about a LOAD and not about a registration.
               registration.architecture == "DeepseekV4ForCausalLM" ||
               registration.architecture == "Gemma4ForConditionalGeneration" ||
               registration.architecture ==
                   "Gemma4UnifiedForConditionalGeneration" ||
               registration.architecture == "MuseGlimmerForCausalLM" ||
               registration.architecture ==
                   "MuseGlimmerForConditionalGeneration") {
      // Qwen3-VL (MM-ENGINE-FORWARD) + Gemma-4 (CLAIM-GEMMA4-MM-E2E) + Muse
      // Glimmer (CLAIM-MUSE-GLIMMER-W0): MULTIMODAL (a vision tower alongside
      // the text backbone) but the text backbone is dense attention → NOT
      // hybrid (no GDN state). Muse Glimmer's iRoPE split is sliding-vs-full
      // ATTENTION, which is not a recurrent lane, so it belongs here and not
      // with the hybrids; its vision tower is scaffolded, not yet forwarding.
      // The non-hybrid multimodal registrations.
      //
      // `Dots3NoteForCausalLM` LEFT THIS LIST AT W5 AND CAME BACK AT W6a, and
      // the round trip is the honest record rather than churn (#699, #2512).
      // W1 put it here because upstream registers it in `_MULTIMODAL_MODELS`
      // with image, video AND audio towers (multimodal.py:82-87, inside
      // `get_placeholder_str` at :80-88) — a statement about UPSTREAM. W5/W5c
      // made the released config loadable, at which point the flag became a
      // claim about THIS port that this port could not honour, so it moved to
      // the text-only branch below.
      //
      // W6a moved it back and BACKED it: `kDots3NoteFactory` carries `encode_mm`
      // and `embed_mm`, and `mm_chat_dots3note.cpp` registers this
      // architecture's own chat seam, so a served `image_url` request reaches
      // the model forward. It is NOT hybrid for the same reason the others here
      // are not: both its attention classes page one MLA cache, and the sliding
      // half is a window on it rather than a recurrent state. What is still
      // refused BY NAME is the pyramid MoE ViT (W6b), which is why the RELEASED
      // checkpoint's tower does not load — a flag about the ARCHITECTURE, not a
      // promise about every checkpoint of it.
      CHECK_FALSE(registration.info.is_hybrid);
      CHECK(registration.info.supports_multimodal);
    } else {
      // Pure text-only full-attention/MLA arch (dense OR MoE): NOT hybrid (no
      // GDN linear-attention state), NOT multimodal (no vision/audio tower).
      CHECK_FALSE(registration.info.is_hybrid);
      CHECK_FALSE(registration.info.supports_multimodal);
    }
  }
}

TEST_CASE("resolve_model_cls: Qwen3ForCausalLM resolves to the dense additive factory") {
  // W0 of the first additive-model bring-up: the new TU (qwen3_dense.cpp)
  // self-registers "Qwen3ForCausalLM" with ZERO edit to a shared array. It must
  // resolve to a complete factory whose KV-cache builder is present (the runner
  // W1 generalization consumes its full-attention-only KVCacheConfig).
  const HfConfig config = Config({"Qwen3ForCausalLM"});
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  CHECK(reg.architecture == "Qwen3ForCausalLM");
  REQUIRE(reg.factory != nullptr);
  CHECK(reg.factory->parse_config != nullptr);
  CHECK(reg.factory->load_weights != nullptr);
  CHECK(reg.factory->prepare != nullptr);
  CHECK(reg.factory->forward != nullptr);
  CHECK(reg.factory->make_kv_cache != nullptr);
  CHECK(reg.factory->is_dense_model);

  // The pure-dense KV-cache spec is full-attention-ONLY: exactly one FA group,
  // NO MambaSpec/GDN group (the topology that forces the runner generalization).
  HfConfig kv_config = config;
  kv_config.num_key_value_heads = 8;
  kv_config.head_dim = 128;
  const vllm::v1::KVCacheConfig kv =
      reg.factory->make_kv_cache(kv_config, /*block_size=*/16, /*num_blocks=*/8);
  REQUIRE(kv.kv_cache_groups.size() == 1);
  CHECK(kv.kv_cache_groups[0].kv_cache_spec->kind() ==
        vllm::v1::KVCacheSpecKind::kFullAttention);
}

TEST_CASE("resolve_model_cls: Gemma3ForCausalLM resolves to the Gemma dense factory") {
  // Sweep W2: the new TU (gemma3_registry.cpp) self-registers "Gemma3ForCausalLM"
  // with ZERO edit to a shared array. It must resolve to a complete dense factory
  // whose KV-cache builder yields the full-attention-only topology (sliding
  // layers masked at the kernel, not by a smaller cache).
  const HfConfig config = Config({"Gemma3ForCausalLM"});
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  CHECK(reg.architecture == "Gemma3ForCausalLM");
  REQUIRE(reg.factory != nullptr);
  CHECK(reg.factory->parse_config != nullptr);
  CHECK(reg.factory->load_weights != nullptr);
  CHECK(reg.factory->prepare != nullptr);
  CHECK(reg.factory->forward != nullptr);
  CHECK(reg.factory->make_kv_cache != nullptr);
  CHECK(reg.factory->is_dense_model);

  HfConfig kv_config = config;
  kv_config.num_key_value_heads = 1;
  kv_config.head_dim = 256;
  const vllm::v1::KVCacheConfig kv =
      reg.factory->make_kv_cache(kv_config, /*block_size=*/16, /*num_blocks=*/8);
  REQUIRE(kv.kv_cache_groups.size() == 1);
  CHECK(kv.kv_cache_groups[0].kv_cache_spec->kind() ==
        vllm::v1::KVCacheSpecKind::kFullAttention);
}

TEST_CASE("resolve_model_cls: OPTForCausalLM resolves to the cross-family dense factory") {
  // The CROSS-FAMILY additivity canary (registry.py:176). OPT shares no
  // attention preamble, no norm, no activation and no positional scheme with
  // any Qwen variant in the tree — and the model-registry seam still absorbed
  // it as ONE new TU carrying its own REGISTER_VLLM_MODEL line, with ZERO edit
  // to a shared registration array.
  const HfConfig config = Config({"OPTForCausalLM"});
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  CHECK(reg.architecture == "OPTForCausalLM");
  REQUIRE(reg.factory != nullptr);
  CHECK(reg.factory->parse_config != nullptr);
  CHECK(reg.factory->load_weights != nullptr);
  CHECK(reg.factory->prepare != nullptr);
  CHECK(reg.factory->forward != nullptr);
  CHECK(reg.factory->make_kv_cache != nullptr);
  CHECK(reg.factory->is_dense_model);

  // Pure full attention: one FA group, no MambaSpec/GDN group. OPT predates
  // GQA, so num_key_value_heads == num_attention_heads (12 x 64 for opt-125m).
  HfConfig kv_config = config;
  kv_config.num_key_value_heads = 12;
  kv_config.head_dim = 64;
  const vllm::v1::KVCacheConfig kv =
      reg.factory->make_kv_cache(kv_config, /*block_size=*/16, /*num_blocks=*/8);
  REQUIRE(kv.kv_cache_groups.size() == 1);
  CHECK(kv.kv_cache_groups[0].kv_cache_spec->kind() ==
        vllm::v1::KVCacheSpecKind::kFullAttention);
}

TEST_CASE("resolve_model_cls: LlamaForCausalLM resolves to the cross-family dense factory") {
  // The CROSS-FAMILY dense bring-up (registry.py). Llama shares the standard-dense
  // shape with Qwen3 but differs (no qk-norm, llama3 rope-scaling); it lands as ONE
  // new loader/registry TU + one REGISTER_VLLM_MODEL line, ZERO shared-array edit,
  // reusing the shared dense forward.
  const HfConfig config = Config({"LlamaForCausalLM"});
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  CHECK(reg.architecture == "LlamaForCausalLM");
  REQUIRE(reg.factory != nullptr);
  CHECK(reg.factory->parse_config != nullptr);
  CHECK(reg.factory->load_weights != nullptr);
  CHECK(reg.factory->prepare != nullptr);
  CHECK(reg.factory->forward != nullptr);
  CHECK(reg.factory->make_kv_cache != nullptr);
  CHECK(reg.factory->is_dense_model);

  // Pure-dense KV-cache spec: exactly one full-attention group (GQA 32/8,
  // head_dim 64 for Llama-3.2-1B), NO MambaSpec/GDN group.
  HfConfig kv_config = config;
  kv_config.num_key_value_heads = 8;
  kv_config.head_dim = 64;
  const vllm::v1::KVCacheConfig kv =
      reg.factory->make_kv_cache(kv_config, /*block_size=*/16, /*num_blocks=*/8);
  REQUIRE(kv.kv_cache_groups.size() == 1);
  CHECK(kv.kv_cache_groups[0].kv_cache_spec->kind() ==
        vllm::v1::KVCacheSpecKind::kFullAttention);
}

TEST_CASE("resolve_model_cls: MistralForCausalLM resolves to the fifth-family dense factory") {
  // The FIFTH family (registry.py:159). Mistral is Llama minus the llama3 rope
  // rescale (plain rope theta 1e6) with an untied lm_head and null sliding_window;
  // it lands as ONE new loader/registry TU + one REGISTER_VLLM_MODEL line, ZERO
  // shared-array edit, reusing the shared dense forward.
  const HfConfig config = Config({"MistralForCausalLM"});
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  CHECK(reg.architecture == "MistralForCausalLM");
  REQUIRE(reg.factory != nullptr);
  CHECK(reg.factory->parse_config != nullptr);
  CHECK(reg.factory->load_weights != nullptr);
  CHECK(reg.factory->prepare != nullptr);
  CHECK(reg.factory->forward != nullptr);
  CHECK(reg.factory->make_kv_cache != nullptr);
  CHECK(reg.factory->is_dense_model);

  // Pure-dense KV-cache spec: exactly one full-attention group (GQA 32/8, head_dim
  // 128 for Mistral-7B-v0.3), NO MambaSpec/GDN group, NO SlidingWindowSpec (v0.3
  // has sliding_window null).
  HfConfig kv_config = config;
  kv_config.num_key_value_heads = 8;
  kv_config.head_dim = 128;
  const vllm::v1::KVCacheConfig kv =
      reg.factory->make_kv_cache(kv_config, /*block_size=*/16, /*num_blocks=*/8);
  REQUIRE(kv.kv_cache_groups.size() == 1);
  CHECK(kv.kv_cache_groups[0].kv_cache_spec->kind() ==
        vllm::v1::KVCacheSpecKind::kFullAttention);
}

TEST_CASE("parse_config: OPT reads its family fields out of the raw config doc") {
  // OPT names its FFN width `ffn_dim`, not `intermediate_size`, and carries five
  // switches our typed HfConfig does not model — so its config hook reads
  // `HfConfig::raw`. That escape hatch is what let a new family land without
  // widening the shared HfConfig POD (spec "seams that held" S2).
  HfConfig config = Config({"OPTForCausalLM"});
  config.hidden_size = 768;
  config.num_attention_heads = 12;
  config.max_position_embeddings = 2048;
  config.raw = nlohmann::json{{"ffn_dim", 3072},
                              {"word_embed_proj_dim", 768},
                              {"do_layer_norm_before", true},
                              {"activation_function", "relu"}};
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  REQUIRE_NOTHROW(reg.factory->parse_config(config));

  const vllm::OPTConfigExtras extras = vllm::GetOPTConfigExtras(config);
  CHECK(extras.ffn_dim == 3072);
  CHECK(extras.word_embed_proj_dim == 768);
  CHECK(extras.do_layer_norm_before);
  // Keys facebook/opt-125m omits fall back to the transformers OPTConfig
  // defaults, not to zero.
  CHECK(extras.enable_bias);
  CHECK(extras.layer_norm_elementwise_affine);
  CHECK(extras.tie_word_embeddings);
  CHECK_FALSE(extras.remove_final_layer_norm);

  // A missing ffn_dim is a hard error rather than a silently-zero FFN.
  HfConfig bad = config;
  bad.raw.erase("ffn_dim");
  CHECK_THROWS(reg.factory->parse_config(bad));

  // The two upstream variants we have no checkpoint to gate are REJECTED
  // loudly rather than shipped untested (spec R2).
  HfConfig proj = config;
  proj.raw["word_embed_proj_dim"] = 512;
  CHECK_THROWS(reg.factory->parse_config(proj));
  HfConfig act = config;
  act.raw["activation_function"] = "gelu";
  CHECK_THROWS(reg.factory->parse_config(act));
}

TEST_CASE("resolve_model_cls: Qwen3MoeForCausalLM resolves to the full-attention MoE factory") {
  // W0 of the first full-attention MoE bring-up: the new TU
  // (qwen3_moe_registry.cpp) self-registers "Qwen3MoeForCausalLM" with ZERO edit
  // to a shared array. It must resolve to a complete factory whose KV-cache
  // builder is present and, unlike the 35B hybrid MoE, is full-attention ONLY.
  const HfConfig config = Config({"Qwen3MoeForCausalLM"});
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  CHECK(reg.architecture == "Qwen3MoeForCausalLM");
  REQUIRE(reg.factory != nullptr);
  CHECK(reg.factory->parse_config != nullptr);
  CHECK(reg.factory->load_weights != nullptr);
  CHECK(reg.factory->prepare != nullptr);
  CHECK(reg.factory->forward != nullptr);
  CHECK(reg.factory->make_kv_cache != nullptr);
  // MoE, not dense: the scheduler-policy split flags it non-dense.
  CHECK_FALSE(reg.factory->is_dense_model);

  // Full-attention MoE KV-cache spec: exactly one FA group, NO MambaSpec/GDN
  // group (Qwen3-Coder has no linear-attention layers — the runner's
  // full-attention-only path covers it with zero runner change).
  HfConfig kv_config = config;
  kv_config.num_key_value_heads = 4;
  kv_config.head_dim = 128;
  const vllm::v1::KVCacheConfig kv =
      reg.factory->make_kv_cache(kv_config, /*block_size=*/16, /*num_blocks=*/8);
  REQUIRE(kv.kv_cache_groups.size() == 1);
  CHECK(kv.kv_cache_groups[0].kv_cache_spec->kind() ==
        vllm::v1::KVCacheSpecKind::kFullAttention);
}

TEST_CASE("runtime quant capability distinguishes true W4A4 from dense BF16") {
  vllm::Qwen3_5DenseWeights bf16;
  std::unique_ptr<vllm::LoadedModel> bf16_model =
      vllm::MakeQwen3_5DenseLoadedModel(std::move(bf16));
  CHECK_FALSE(bf16_model->uses_nvfp4_w4a4());

  vllm::Qwen3_5DenseWeights w4a4;
  w4a4.layers.resize(1);
  w4a4.layers.front().mlp.gate_proj_fp4.alpha = 1.0F;
  std::unique_ptr<vllm::LoadedModel> w4a4_model =
      vllm::MakeQwen3_5DenseLoadedModel(std::move(w4a4));
  CHECK(w4a4_model->uses_nvfp4_w4a4());
}

TEST_CASE("safetensors model source propagates the selected load queue") {
  std::vector<vllm::SafetensorsFile> shards;
  vt::Queue queue;
  vllm::ModelSource borrowed =
      vllm::ModelSource::FromSafetensors(shards, &queue);
  CHECK(borrowed.safetensors == &shards);
  CHECK(borrowed.load_queue == &queue);

  auto owned_shards =
      std::make_shared<const std::vector<vllm::SafetensorsFile>>();
  vllm::ModelSource owned =
      vllm::ModelSource::FromSafetensorsOwned(owned_shards, &queue);
  CHECK(owned.safetensors == owned_shards.get());
  CHECK(owned.safetensors_owned == owned_shards);
  CHECK(owned.load_queue == &queue);
}

TEST_CASE("Qwen3.5 KV spec mirrors independent conv and SSM cache dtypes") {
  HfConfig config = Config({"Qwen3_5ForConditionalGeneration"});
  config.num_key_value_heads = 2;
  config.head_dim = 8;
  config.linear_num_key_heads = 2;
  config.linear_num_value_heads = 4;
  config.linear_key_head_dim = 8;
  config.linear_value_head_dim = 8;
  config.linear_conv_kernel_dim = 4;
  config.mamba_ssm_dtype = "float32";

  vllm::Qwen3_5DenseWeights weights;
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::MakeQwen3_5DenseLoadedModel(std::move(weights));
  const vllm::v1::KVCacheConfig kv =
      ModelRegistry::MakeKVCache(*model, config, /*block_size=*/16,
                                 /*num_blocks=*/8);
  REQUIRE(kv.kv_cache_groups.size() == 2);
  const auto* mamba = dynamic_cast<const vllm::v1::MambaSpec*>(
      kv.kv_cache_groups[1].kv_cache_spec.get());
  REQUIRE(mamba != nullptr);
  REQUIRE(mamba->shapes.size() == 2);
  REQUIRE(mamba->dtypes.size() == 2);

  // Upstream MambaStateShape/DtypeCalculator order is conv, then temporal.
  CHECK(mamba->shapes[0] == std::vector<int64_t>{64, 3});
  CHECK(mamba->shapes[1] == std::vector<int64_t>{4, 8, 8});
  CHECK(mamba->dtypes[0] == vt::DType::kBF16);
  CHECK(mamba->dtypes[1] == vt::DType::kF32);

  // The runner allocates exactly one conv row plus one temporal row per state
  // slot. Keep external-cache page planning byte-identical to that layout.
  const int64_t runtime_row_bytes =
      64 * 3 * static_cast<int64_t>(vt::SizeOf(vt::DType::kBF16)) +
      4 * 8 * 8 * static_cast<int64_t>(vt::SizeOf(vt::DType::kF32));
  CHECK(mamba->page_size_bytes() == runtime_row_bytes);
}

TEST_CASE("Qwen3.5 SSM cache dtype accepts upstream torch aliases exactly") {
  HfConfig config;
  const auto resolve = [&](const char* value) {
    config.mamba_ssm_dtype = value;
    return vllm::detail::ResolveMambaSsmCacheDType(
        config, vt::DType::kBF16);
  };

  CHECK(resolve("") == vt::DType::kBF16);
  CHECK(resolve("auto") == vt::DType::kBF16);
  CHECK(resolve("float32") == vt::DType::kF32);
  CHECK(resolve("float") == vt::DType::kF32);
  CHECK(resolve("float16") == vt::DType::kF16);
  CHECK(resolve("half") == vt::DType::kF16);
  CHECK(resolve("bfloat16") == vt::DType::kBF16);
  CHECK_THROWS_WITH_AS(resolve("fp16"),
                       doctest::Contains("unsupported mamba_ssm_dtype"),
                       std::runtime_error);
}

TEST_CASE("hf_registry_coverage: every registration has an example config fixture") {
  // C++ fixture registry for the currently implemented subset. Keep this list
  // alias-for-alias with the central ordered table, mirroring HF_EXAMPLE_MODELS.
  constexpr std::array<std::string_view, 59> kExampleConfigArchitectures{
      "BoundaryExtractor",
      // "ClmModel" (Cl) < "CohereForCausalLM" (Co): l=0x6C < o=0x6F.
      "ClmModel",
      // "Cohere2Moe" < "CohereFor": they agree through "Cohere" and then
      // differ at '2' 0x32 against 'F' 0x46.
      "Cohere2MoeForCausalLM",
      "CohereForCausalLM",
      "CuaS1Forms",
      "DeepseekV2ForCausalLM",
      // "DeepseekV41For" < "DeepseekV4For": they agree through "DeepseekV4" and
      // then differ at '1' 0x31 against 'F' 0x46, so the V4.1 arm sorts BEFORE
      // the V4 one rather than after it.
      "DeepseekV41ForCausalLM",
      "DeepseekV4ForCausalLM",
      "Dots3NoteForCausalLM",
      "Gemma2ForCausalLM",
      "Gemma3ForCausalLM",
      "Gemma4ForConditionalGeneration",
      "Gemma4UnifiedForConditionalGeneration",
      "GemmaForCausalLM",
      "Glm4ForCausalLM",
      "Glm4MoeLiteForCausalLM",
      "Glm5NextForConditionalGeneration",
      "GlmMoeDsaForCausalLM",
      "GraniteForCausalLM",
      "InternLM2ForCausalLM",
      "InternLM3ForCausalLM",
      "KevModel",
      "KimiK3ForConditionalGeneration",
      "KimiLinearForCausalLM",
      // "Kolibri1" sorts before "Laguna": 'K' (0x4B) < 'L' (0x4C).
      "Kolibri1ForCausalLM",
      "LagunaForCausalLM",
      "LayaModel",
      "LlamaForCausalLM",
      "LlamaModel",
      // "MiMoV2" sorts before "MiniCPM": 'M' (0x4D) < 'n' (0x6E) at index 2.
      "MiMoV2ForCausalLM",
      "MiniCPM3ForCausalLM",
      "MiniCPMForCausalLM",
      "MistralForCausalLM",
      "MuseGlimmerForCausalLM",
      "MuseGlimmerForConditionalGeneration",
      "NemotronHForCausalLM",
      // "NemotronHF" < "NemotronH_" ('F' 0x46 < '_' 0x5F), and "..._Nano_O" <
      // "..._Nano_V" ('O' 0x4F < 'V' 0x56).
      "NemotronH_Nano_Omni_Reasoning_V3",
      "NemotronH_Nano_VL_V2",
      "NimbleModel",
      "OPTForCausalLM",
      "Olmo2ForCausalLM",
      "Olmo3ForCausalLM",
      "ParakeetForCTC",
      "ParakeetForRNNT",
      "ParakeetForTDT",
      "Phi3ForCausalLM",
      "PhiForCausalLM",
      "Qwen3ForCausalLM",
      "Qwen3MoeForCausalLM",
      "Qwen3VLForConditionalGeneration",
      "Qwen3_5ForCausalLM",
      "Qwen3_5ForConditionalGeneration",
      "Qwen3_5MoeForCausalLM",
      "Qwen3_5MoeForConditionalGeneration",
      "Qwen4ExpForConditionalGeneration",
      // "SpanExtractor" (Sp) < "StableLmForCausalLM" (St): p=0x70 < t=0x74.
      "SpanExtractor",
      "StableLmForCausalLM",
      // "Tev1Model" (T) sorts after StableLmForCausalLM (S): T=0x54 > S=0x53.
      "Tev1Model",
      // "XorModel" (X) sorts after Tev1Model (T): X=0x58 > T=0x54.
      "XorModel",
  };
  const std::vector<std::string_view> supported = ModelRegistry::SupportedArchs();
  REQUIRE(supported.size() == kExampleConfigArchitectures.size());
  for (size_t i = 0; i < supported.size(); ++i) {
    CHECK(supported[i] == kExampleConfigArchitectures[i]);
  }
}

TEST_CASE("resolve_model_cls: architecture list order selects first registered entry") {
  HfConfig config = Config({"NotImplemented", "Qwen3_5MoeForConditionalGeneration",
                            "Qwen3_5ForConditionalGeneration"});
  CHECK(ModelRegistry::Resolve(config).architecture ==
        "Qwen3_5MoeForConditionalGeneration");

  config = Config({"Qwen3_5ForConditionalGeneration",
                   "Qwen3_5MoeForConditionalGeneration"});
  CHECK(ModelRegistry::Resolve(config).architecture ==
        "Qwen3_5ForConditionalGeneration");
}

TEST_CASE("resolve_model_cls: empty architecture list matches pinned oracle") {
  const HfConfig config = Config({});
  CHECK_THROWS_WITH_AS(ModelRegistry::Resolve(config),
                       "No model architectures are specified",
                       std::runtime_error);
}

TEST_CASE("raise_for_unsupported: registered inspection failure exact message") {
  const std::vector<std::string> architectures{
      "Qwen3_5ForConditionalGeneration"};
  const std::vector<std::string_view> supported{
      "Qwen3_5ForConditionalGeneration",
      "Qwen3_5MoeForConditionalGeneration"};
  CHECK_THROWS_WITH_AS(
      ModelRegistry::RaiseForUnsupported(
          ArchSpan(architectures), std::span<const std::string_view>(supported)),
      "Model architectures ['Qwen3_5ForConditionalGeneration'] failed to be "
      "inspected. Please check the logs for more details.",
      std::runtime_error);
}

TEST_CASE("raise_for_unsupported: previously-supported branch exact message") {
  const HfConfig config = Config({"MotifForCausalLM"});
  CHECK_THROWS_WITH_AS(
      ModelRegistry::Resolve(config),
      "Model architecture MotifForCausalLM was supported in vLLM until "
      "v0.10.2, and is not supported anymore. Please use an older version of "
      "vLLM if you want to use this model architecture.",
      std::runtime_error);
}

TEST_CASE("raise_for_unsupported: out-of-tree branch exact message") {
  const HfConfig config = Config({"BartModel"});
  CHECK_THROWS_WITH_AS(
      ModelRegistry::Resolve(config),
      "Model architecture BartModel is not supported in-tree anymore. Please "
      "install the plugin at https://github.com/vllm-project/bart-plugin if "
      "you want to use this model architecture.",
      std::runtime_error);
}

TEST_CASE("raise_for_unsupported: subset default message and order match oracle") {
  // Gemma4ForCausalLM is the still-unported Gemma dense arch (sweep W6 honesty
  // pass: MM-wrapped-only checkpoints + a PLE/YOCO/MoE campaign, gate-BLOCKED).
  // Gemma-1/2/3 are now SUPPORTED and appear in the sorted dict_keys below.
  const HfConfig unknown = Config({"Gemma4ForCausalLM"});
  CHECK_THROWS_WITH_AS(
      ModelRegistry::Resolve(unknown),
      "Model architectures ['Gemma4ForCausalLM'] are not supported for now. "
      "Supported architectures: "
      "dict_keys(['BoundaryExtractor', 'ClmModel', 'Cohere2MoeForCausalLM', 'CohereForCausalLM', 'CuaS1Forms', 'DeepseekV2ForCausalLM', "
      "'DeepseekV41ForCausalLM', "
      "'DeepseekV4ForCausalLM', 'Dots3NoteForCausalLM', 'Gemma2ForCausalLM', 'Gemma3ForCausalLM', "
      "'Gemma4ForConditionalGeneration', 'Gemma4UnifiedForConditionalGeneration', 'GemmaForCausalLM', "
      "'Glm4ForCausalLM', 'Glm4MoeLiteForCausalLM', "
      "'Glm5NextForConditionalGeneration', 'GlmMoeDsaForCausalLM', 'GraniteForCausalLM', "
      "'InternLM2ForCausalLM', 'InternLM3ForCausalLM', 'KevModel', "
      "'KimiK3ForConditionalGeneration', 'KimiLinearForCausalLM', "
      "'Kolibri1ForCausalLM', "
      "'LagunaForCausalLM', 'LayaModel', "
      "'LlamaForCausalLM', 'LlamaModel', 'MiMoV2ForCausalLM', "
      "'MiniCPM3ForCausalLM', 'MiniCPMForCausalLM', 'MistralForCausalLM', 'MuseGlimmerForCausalLM', 'MuseGlimmerForConditionalGeneration', "
      "'NemotronHForCausalLM', 'NemotronH_Nano_Omni_Reasoning_V3', 'NemotronH_Nano_VL_V2', 'NimbleModel', "
      "'OPTForCausalLM', 'Olmo2ForCausalLM', 'Olmo3ForCausalLM', "
      "'ParakeetForCTC', 'ParakeetForRNNT', 'ParakeetForTDT', "
      "'Phi3ForCausalLM', 'PhiForCausalLM', 'Qwen3ForCausalLM', "
      "'Qwen3MoeForCausalLM', 'Qwen3VLForConditionalGeneration', "
      "'Qwen3_5ForCausalLM', 'Qwen3_5ForConditionalGeneration', "
      "'Qwen3_5MoeForCausalLM', "
      "'Qwen3_5MoeForConditionalGeneration', 'Qwen4ExpForConditionalGeneration', 'SpanExtractor', 'StableLmForCausalLM', 'Tev1Model', 'XorModel'])",
      std::runtime_error);
}

TEST_CASE("previously-supported table is the complete pinned 32-entry table") {
  constexpr std::array<std::pair<std::string_view, std::string_view>, 32>
      kExpected{{
          {"MotifForCausalLM", "0.10.2"},
          {"Phi3SmallForCausalLM", "0.9.2"},
          {"Phi4FlashForCausalLM", "0.10.2"},
          {"Phi4MultimodalForCausalLM", "0.12.0"},
          {"JAISLMHeadModel", "0.22.0"},
          {"ErnieModel", "0.23.0"},
          {"ErnieForSequenceClassification", "0.23.0"},
          {"ErnieForTokenClassification", "0.23.0"},
          {"InternLM2VEForCausalLM", "0.23.0"},
          {"QWenLMHeadModel", "0.23.0"},
          {"QwenVLForConditionalGeneration", "0.23.0"},
          {"InternLMForCausalLM", "0.23.0"},
          {"DonutForConditionalGeneration", "0.10.2"},
          {"MllamaForConditionalGeneration", "0.10.2"},
          {"XverseForCausalLM", "0.23.0"},
          {"Dots1ForCausalLM", "0.23.0"},
          {"BambaForCausalLM", "0.23.0"},
          {"MiniMaxForCausalLM", "0.23.0"},
          {"MiniMaxText01ForCausalLM", "0.23.0"},
          {"MiniMaxM1ForCausalLM", "0.23.0"},
          {"MiniMaxVL01ForConditionalGeneration", "0.23.0"},
          {"BaiChuanForCausalLM", "0.23.0"},
          {"BaichuanForCausalLM", "0.23.0"},
          {"AquilaModel", "0.24.0"},
          {"AquilaForCausalLM", "0.24.0"},
          {"Grok1ModelForCausalLM", "0.24.0"},
          {"Grok1ForCausalLM", "0.24.0"},
          {"TarsierForConditionalGeneration", "0.24.0"},
          {"Tarsier2ForConditionalGeneration", "0.23.0"},
          {"MantisForConditionalGeneration", "0.24.0"},
          {"MusicFlamingoForConditionalGeneration", "0.24.0"},
          {"AyaVisionForConditionalGeneration", "0.24.0"},
      }};
  const std::span<const UnsupportedModelInfo> actual =
      ModelRegistry::PreviouslySupportedModels();
  REQUIRE(actual.size() == kExpected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    CHECK(actual[i].architecture == kExpected[i].first);
    CHECK(actual[i].detail == kExpected[i].second);
  }
}

TEST_CASE("out-of-tree table is the complete pinned four-entry table") {
  constexpr std::array<std::string_view, 4> kExpected{
      "BartModel", "BartForConditionalGeneration",
      "Florence2ForConditionalGeneration", "MBartForConditionalGeneration"};
  const std::span<const UnsupportedModelInfo> actual =
      ModelRegistry::OutOfTreeSupportedModels();
  REQUIRE(actual.size() == kExpected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    CHECK(actual[i].architecture == kExpected[i]);
    CHECK(actual[i].detail ==
          "https://github.com/vllm-project/bart-plugin");
  }
}

TEST_CASE("can_initialize SKIP: MODEL-FACTORY-registry: no second family yet" *
          doctest::skip()) {
  // Ported test_initialization.py:50-197. The registry seam and both existing
  // Qwen variants are exercised by the loaded-engine CPU suites; a genuinely
  // distinct second-family dummy initialization belongs to the Llama leaf.
  MESSAGE("SKIP MODEL-FACTORY-registry: no second family yet");
}

// ===========================================================================
// SPEC-MTP I4 — GDN speculative state-slot ALLOCATION. Mirrors
// vllm/model_executor/models/qwen3_5.py:524-543 (`num_spec` from the
// SpeculativeConfig) → mamba_utils.py:213-234 (conv row widened to
// `conv_kernel_size - 1 + num_spec`) → mamba/abstract.py:55-59
// (`MambaSpec.num_speculative_blocks = num_speculative_tokens`), which is what
// makes MambaManager allocate k+1 GDN state blocks per request
// (single_type_kv_cache_manager.py:1206-1215).
// ===========================================================================
TEST_CASE("Qwen3.5 KV-cache spec: num_spec widens the conv row and adds state blocks") {
  HfConfig cfg = Config({"Qwen3_5ForCausalLM"});
  // Real 35B GDN geometry (nvidia--Qwen3.6-35B-A3B-NVFP4 config.json).
  cfg.num_key_value_heads = 2;
  cfg.head_dim = 256;
  cfg.linear_num_key_heads = 16;
  cfg.linear_num_value_heads = 32;
  cfg.linear_key_head_dim = 128;
  cfg.linear_value_head_dim = 128;
  cfg.linear_conv_kernel_dim = 4;

  const int64_t conv_dim = 2 * 16 * 128 + 32 * 128;  // 8192

  // num_spec == 0 is the production default and must reproduce the registered
  // KVCacheSpecBuilder byte for byte.
  const vllm::v1::KVCacheConfig base =
      vllm::MakeQwen3_5KVCache(cfg, /*block_size=*/16, /*num_blocks=*/8);
  const vllm::v1::KVCacheConfig zero =
      vllm::MakeQwen3_5KVCacheSpec(cfg, /*block_size=*/16, /*num_blocks=*/8, /*num_spec=*/0);
  REQUIRE(base.kv_cache_groups.size() == 2);
  REQUIRE(zero.kv_cache_groups.size() == 2);
  const auto* base_mamba =
      dynamic_cast<const vllm::v1::MambaSpec*>(base.kv_cache_groups[1].kv_cache_spec.get());
  const auto* zero_mamba =
      dynamic_cast<const vllm::v1::MambaSpec*>(zero.kv_cache_groups[1].kv_cache_spec.get());
  REQUIRE(base_mamba != nullptr);
  REQUIRE(zero_mamba != nullptr);
  CHECK(base_mamba->num_speculative_blocks == 0);
  CHECK(base_mamba->shapes == zero_mamba->shapes);
  CHECK(base_mamba->dtypes == zero_mamba->dtypes);
  CHECK(base_mamba->num_speculative_blocks == zero_mamba->num_speculative_blocks);
  CHECK(base_mamba->page_size_bytes() == zero_mamba->page_size_bytes());
  // Conv row is the plain K-1 window with no speculation.
  CHECK(base_mamba->shapes[0] == std::vector<int64_t>{conv_dim, 3});

  for (const int num_spec : {1, 2, 3}) {
    CAPTURE(num_spec);
    const vllm::v1::KVCacheConfig kv =
        vllm::MakeQwen3_5KVCacheSpec(cfg, /*block_size=*/16, /*num_blocks=*/8, num_spec);
    const auto* mamba =
        dynamic_cast<const vllm::v1::MambaSpec*>(kv.kv_cache_groups[1].kv_cache_spec.get());
    REQUIRE(mamba != nullptr);
    // conv_kernel_size - 1 + num_spec (mamba_utils.py:226): the sliding window
    // the accepted count rewinds into. The conv needs NO extra slots.
    CHECK(mamba->shapes[0] == std::vector<int64_t>{conv_dim, 3 + num_spec});
    // The SSM state shape is unchanged; the k+1 snapshot slots come from the
    // per-request block count, not from a bigger row.
    CHECK(mamba->shapes[1] == std::vector<int64_t>{32, 128, 128});
    CHECK(mamba->num_speculative_blocks == num_spec);
    // The (target) full-attention group is untouched by speculation.
    CHECK(kv.kv_cache_groups[0].kv_cache_spec->page_size_bytes() ==
          base.kv_cache_groups[0].kv_cache_spec->page_size_bytes());

    // SPEC-MTP I5c: speculation on (num_spec > 0) adds ONE extra full-attention
    // KV group — the MTP draft head's own paged K/V layer (index num_hidden_layers
    // upstream, qwen3_5_mtp.py:105-112), sized exactly like a target full-attn
    // layer. num_spec == 0 (base) has NO such group (checked below).
    REQUIRE(kv.kv_cache_groups.size() == 3);
    CHECK(kv.kv_cache_groups[2].layer_names ==
          std::vector<std::string>{"fa_draft"});
    const auto* draft = dynamic_cast<const vllm::v1::FullAttentionSpec*>(
        kv.kv_cache_groups[2].kv_cache_spec.get());
    REQUIRE(draft != nullptr);
    CHECK(draft->page_size_bytes() ==
          base.kv_cache_groups[0].kv_cache_spec->page_size_bytes());
  }
  // num_spec == 0 (production default): the draft KV layer is never allocated —
  // exactly the two pre-I5c groups, byte for byte.
  CHECK(zero.kv_cache_groups.size() == 2);

  // MEASURED state cost of k=1 on this geometry, per GDN layer and per request:
  // one SSM slot is Hv*Dv*Dk*4B = 32*128*128*4 = 2 MiB, so k=1 (2 slots) doubles
  // it; the conv row grows by one tap per channel only.
  const vllm::v1::KVCacheConfig kv1 = vllm::MakeQwen3_5KVCacheSpec(cfg, 16, 8, 1);
  const auto* k1 =
      dynamic_cast<const vllm::v1::MambaSpec*>(kv1.kv_cache_groups[1].kv_cache_spec.get());
  REQUIRE(k1 != nullptr);
  const int64_t ssm_slot_bytes = 32 * 128 * 128 * 4;  // f32 SSM state
  CHECK(ssm_slot_bytes == 2 * 1024 * 1024);
  // page_size_bytes covers ONE block (conv row + ssm row); k=1 grows the conv
  // part by conv_dim * sizeof(bf16) and leaves the ssm part alone.
  CHECK(k1->page_size_bytes() - base_mamba->page_size_bytes() == conv_dim * 2);
}
