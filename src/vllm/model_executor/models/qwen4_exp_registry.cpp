// Qwen4-Exp registry TU — the ADDITIVE self-registration seam (W1 of
// MODEL-MM-QWEN4-EXP, #1981). Follows the dots3_note_registry.cpp /
// gemma4_registry.cpp seam exactly: a NEW translation unit with ONE
// REGISTER_VLLM_MODEL line and ZERO edit to any shared array.
//
// UPSTREAM. SUPERSEDED 2026-08-31 (#2489): vLLM registers this architecture at
// `e126687a9a`, 595 commits BEYOND our pin. Read live 2026-08-26 at `6a5e8f5979`:
// `qwen4*` path, no `registry.py` entry, and a repository-wide search for
// `qwen4` returns zero results; `vllm-omni` likewise. That is absence from
// vLLM `main` rather than staleness in our parity pin `555967922`, so this TU
// deliberately carries no pinned upstream module/class anchor, the convention
// `MODEL-TEXT-qwen3-5-qwen3-5-moe-for-causal-lm` follows for a beyond-pin arm.
// The ALGORITHM source is transformers **5.16.0**, the accepted lane pin; see
// `.agents/oracles/transformers.md` and `.agents/specs/qwen4-exp-flash-next.md`.
//
// The MTP head is deliberately NOT registered as a second architecture, and
// unlike dots3-note that is not a scheduling choice: upstream carries it as an
// `mtp` block INSIDE the same text config rather than as a separate registry
// entry, so there is no second architecture string to register. That is why
// this row moves the MODEL row ratchet by ONE and not by two.
//
// SCOPE HONESTY, RESTATED AT W5c-1 (#2031). Registering this arch makes it
// RESOLVE, parse and validate its config, LOAD a `qwen4exp` GGUF on a CPU
// device (W5a) and PUBLISH its three KV-cache groups (W5c-1). `make_kv_cache`
// no longer refuses; the sentence that said it did was true at W5a's parent and
// stopped being true one wave later, which is the drift this paragraph keeps
// being rewritten to remove.
//
// THREE THINGS STILL REFUSE, and the one that matters is the FORWARD: no token
// has been decoded by this architecture. The other two are load-time and are
// listed here so that the count is checkable rather than rhetorical -- the
// SAFETENSORS arm refuses unconditionally at the end of
// `LoadQwen4ExpForConditionalGeneration` (every published artifact is larger
// than every device this project owns), and the GGUF arm refuses a source that
// names the kind without carrying a file. Both are stated at their own sites.
//
// The forward's polarity matters more here than usual, because no oracle for
// this model runs on any hardware this project owns yet (`gateable = no`,
// blocked on memory rather than software), so there is no downstream token gate
// that would catch a forward returning plausible garbage. Refusing is the only
// safe default.
#include "vllm/model_executor/models/model_registry.h"

#include "vt/dtype.h"  // VT_CHECK

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits complete type
#include "vllm/model_executor/models/qwen3_5_internal.h"  // ResolveMambaSsmCacheDType
#include "vllm/model_executor/models/dense_attn_block.h"   // ResidentWeight
#include "vllm/model_executor/models/qwen4_exp.h"
#include "vllm/model_executor/models/qwen4_exp_forward.h"  // W5f: the layer loop
#include "vllm/model_executor/models/qwen4_exp_weights.h"
#include "vt/ops.h"
#include "vllm/v1/kv_cache_dtype.h"     // ResolveKvCacheDType
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {
namespace {

// Text generation, multimodal (image AND video: the published config carries
// `image_token_id`, `video_token_id` and a `vision_config`), and HYBRID —
// 36 of 48 layers are Gated DeltaNet carrying recurrent state, so this belongs
// with the hybrids and not with the pure-attention arms.
inline constexpr ModelInfo kQwen4ExpInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = true,
    // FALSE by the house convention the blanket assertion in
    // test_model_registry.cpp enforces: our ModelInfo is a consumed subset
    // whose only reader short-circuits on is_hybrid, so the GDN-hybrid
    // wrappers (kQwen3_5Info, kKimiLinearInfo) all leave this false even
    // though upstream's class carries HasInnerState.
    .has_inner_state = false,
    .supports_multimodal = true,
    .score_type = "bi-encoder",
};

// `Qwen4ExpLoadedModel` — the concrete model this hook produces — is declared in
// `qwen4_exp_weights.h` rather than here. That header says why: an anonymous
// type is unreachable by `dynamic_cast` from another translation unit, and a
// reachability case that cannot open the handle cannot tell a real load from a
// hook that returns `Qwen4ExpWeights{}`.

std::unique_ptr<LoadedModel> LoadQwen4ExpForConditionalGeneration(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind == ModelSource::Kind::kGguf) {
    // W5a (#2031) LOADS it. The GGUF k-quant arm is OWED, not optional
    // (AGENTS.md, porting-a-model.md §2), and for this row it is the ONLY arm
    // that fits a host we own: `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S is
    // 67.56 GiB of weights against ~119.6 GiB usable on GB10, where every
    // safetensors artifact (bf16 ~360 GB, FP8 ~180 GB, NVFP4 ~128 GB) does not.
    //
    // Both blockers W1 named have since landed. W6a (#1989) added the IQ4_NL
    // and Q5_0 reader arms, so the file opens at all, and made
    // `GgufTensorRole::kEmbeddingTable` keep-quant eligible with a dequantizing
    // gather behind it, so the 51.2 G-parameter n-gram table stays resident as
    // blocks instead of expanding to 102.4 GB of bf16.
    //
    // A null `gguf` reaches here from a caller that set the KIND without the
    // FILE. Refused by name rather than dereferenced: the alternative is a
    // segmentation fault inside a loader the reader is entitled to read as
    // "GGUF is not supported here".
    if (source.gguf == nullptr) {
      throw std::runtime_error(
          "Qwen4ExpForConditionalGeneration: the model source says GGUF but "
          "carries no file. See .agents/specs/qwen4-exp-flash-next.md and "
          "issue #2031.");
    }
    // ENG-GGUF-RESIDENCY-RESOLVED-DEVICE: `source.device`, which is the device
    // the ENGINE resolved for this load, and NOT
    // `platforms::CurrentPlatform().device_type()`.
    //
    // The probe answers `kCUDA` on any process where `platforms/cuda.cpp`'s
    // `Registrar` saw a usable GPU, while
    // `LoadedEngine::ResolveExplicitDeviceType` returns `kCPU` for an explicit
    // `--device cpu` "even on a CUDA-capable build/process". Reading the probe
    // here therefore handed the PLE guard below a device nothing was going to
    // run on: on a CUDA box, `--device cpu` got the CUDA refusal, whose own
    // remedy text is "Load this model with `--device cpu`".
    return std::make_unique<Qwen4ExpLoadedModel>(
        registration,
        LoadQwen4ExpFromGguf(*source.gguf, config, source.device));
  }
  (void)registration;
  (void)config;
  // The safetensors arm stays refused, and NOT because it is the harder one.
  // Every published safetensors artifact of this model is larger than every
  // device this project owns, so an arm that read them would be code nothing
  // could ever run. The spec's `## Owed` records it with that reason rather
  // than as an unqualified to-do.
  throw std::runtime_error(
      "Qwen4ExpForConditionalGeneration: the safetensors weight loader is not "
      "ported (every published safetensors artifact — bf16 ~360 GB, FP8 ~180 "
      "GB, NVFP4 ~128 GB — exceeds every device this project owns, so the GGUF "
      "arm is the supported one). See .agents/specs/qwen4-exp-flash-next.md and "
      "issue #1978.");
}

void PrepareQwen4ExpForConditionalGeneration(LoadedModel& model,
                                             const HfConfig& config,
                                             vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

// ─── THE PUBLISHED LAYER NAMES, DERIVED ONCE (W5j) ───────────────────────────
//
// `MakeQwen4ExpKVCache` PUBLISHES the caches under these names and
// `ForwardQwen4ExpForConditionalGeneration` RESOLVES them by the same names, so
// the two must agree or the model is refused at run time with no compile error
// to catch it. The spec's `## Owed` names the requirement in those words: "it
// must build its layer names through ONE builder shared with
// `MakeQwen4ExpKVCache` — two derivations of one name set is the shape that can
// disagree."
//
// A FILE-LOCAL HELPER IS ENOUGH AND A HEADER WOULD BE WORSE. Both the publisher
// and the consumer live in THIS translation unit, so there is no seam to widen
// and nothing outside needs the names; exporting them would invite a third
// derivation somewhere else, which is the failure this exists to remove.
//
// The suffixes are not free choices. `.linear_attn` is what
// `ResolveKVCacheGroupLayerNames` builds for a recurrent layer, so the runner's
// by-name membership sees the same string either way. `.self_attn.indexer.k_cache`
// mirrors upstream's own side-cache prefix
// (`vllm/models/deepseek_v4/attention.py:761-767` registers the indexer key cache
// under `...indexer.k_cache`); the runner parses the `.layers.<N>.` segment out of
// it, so the suffix is free to say WHICH cache it is.
std::string Qwen4ExpLayerPrefix(size_t layer) {
  return "model.layers." + std::to_string(layer) + ".";
}
std::string Qwen4ExpLinearAttnName(size_t layer) {
  return Qwen4ExpLayerPrefix(layer) + "linear_attn";
}
std::string Qwen4ExpQsaAttnName(size_t layer) {
  return Qwen4ExpLayerPrefix(layer) + "self_attn.attn";
}
std::string Qwen4ExpQsaIndexerName(size_t layer) {
  return Qwen4ExpLayerPrefix(layer) + "self_attn.indexer.k_cache";
}

// The step this hook serves is assembled below from EITHER the two POSITIONAL
// cache channels or the by-name index, plus — on the positional arm only — the
// three states that channel cannot carry.
//
// WHY THE SCRATCH IS BUILT HERE AND NOT IN THE LOOP. `Qwen4ExpTextModelForward`
// takes every cache as an operand and has no opinion about where they came
// from, which is what lets its gate drive it at any `past_len`. The RESTRICTION
// is the engine's, and only on the POSITIONAL arm: `ModelForwardInput` carries
// `attn_kv` and `gdn_state` and nothing else, so there the QSA indexer side cache
// and the PLE layer's conv ring and n-gram history have no home across steps. On
// the BY-NAME arm all three DO have one and a second step runs. Putting the
// scratch — and the refusal that makes it sound — at this boundary keeps the
// limit where it is true instead of baking it into the loop.

// ─────────────────────────────────────────────────────────────────────────────
// DECODEDIV (#2496) — THE PER-STEP STATE FINGERPRINT, DEFAULT OFF.
//
// #2496 is a whole-output symptom on the RELEASED artifact: `--device cuda`
// emits `11751 271 271 271 271 271 0 0` where `--device cpu` on the same tree
// and the same file emits `11751 13 15767 411 2029 11 1092 369`. Token 0 agrees,
// [KERNEL-GDN-CHUNKED-MIRROR, #2858: that CPU sequence was measured while the
// CPU GDN prefill ran the SEQUENTIAL recurrence, and this comment used to say
// the ids the new chunked default emits were EXPECTED TO DIFFER. THEY DO NOT.
// Re-measured on the same artifact on 2026-09-04 (docs/bench-evidence/
// qwen4exp-gdn-chunked-token-ids-20260904.md): the CPU default emits exactly
// the sequence quoted here, and `VT_GDN_CHUNKED=0` emits it too. The chunked
// arm IS running -- it moves decoder layer 0's GDN block output by 3.702e-04
// and lands 19.9x closer to CUDA than the sequential arm did -- and it flips
// none of the eight argmaxes. The paragraph is kept as written because it is
// the record of what #2496 was diagnosed from, and it is now also current.]
// so prefill is right; every decode token is wrong, and bit-stably so across
// builds and across `CUDA_LAUNCH_BLOCKING`. What is left is state the second
// step carries, computed differently on the two arms.
//
// The hermetic CPU-vs-CUDA comparison in `test_qwen4_exp_layer_loop.cpp` asks
// the same question on a fixture, and the fixture may not be able to hold the
// answer: its layer-3 activations sit near 2^18, where one bf16 ULP is ~1024 and
// the K/V store saturates (W5j measured 0 of 192 paged K/V words moving across
// two prompts). This is the arm that runs on the REAL weights, where that
// objection does not apply.
//
// IT PRINTS A SUMMARY, NOT A HASH, AND THAT IS THE POINT. The two arms are two
// processes and two orders of arithmetic, so a checksum disagrees on a CORRECT
// pair and says nothing. A count of non-finite words, a maximum magnitude, a sum
// of magnitudes and the first four values separate a state that is WRONG — zero,
// saturated, NaN, or a different tensor entirely — from one that differs in its
// last bits, which is the only distinction this defect needs.
//
// `VT_Q4EXP_STATE_FP=1` turns it on. Off it costs one `getenv` per step, read
// once. On it costs one device-to-host copy per state per step, which is a
// diagnostic price paid deliberately and never on a default run.
bool Q4ExpStateFpEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_Q4EXP_STATE_FP");
    return e != nullptr && e[0] == '1' && e[1] == '\0';
  }();
  return on;
}

void Q4ExpStateSummary(dense_attn::Dev d, long step, const std::string& what,
                       const vt::Tensor& t) {
  const int64_t n = t.Numel();
  if (t.data == nullptr || n <= 0) {
    std::cerr << "Q4EXP_FP step=" << step << " " << what << " ABSENT\n";
    return;
  }
  const size_t bytes = static_cast<size_t>(n) * vt::SizeOf(t.dtype);
  std::vector<unsigned char> host(bytes);
  // ONE entry point for both residencies. `Backend::Copy` infers its direction
  // from the pointer values, and `Synchronize` is a no-op on the host backend,
  // so the host arm pays nothing and the device arm cannot forget the drain.
  d.b.Copy(d.q, host.data(), t.data, bytes);
  d.b.Synchronize(d.q);
  const auto at = [&](int64_t i) -> double {
    switch (t.dtype) {
      case vt::DType::kF32:
        return static_cast<double>(reinterpret_cast<const float*>(host.data())[i]);
      case vt::DType::kBF16:
        return static_cast<double>(
            vt::BF16ToF32(reinterpret_cast<const uint16_t*>(host.data())[i]));
      case vt::DType::kF16:
        return static_cast<double>(
            vt::F16ToF32(reinterpret_cast<const uint16_t*>(host.data())[i]));
      case vt::DType::kI64:
        return static_cast<double>(reinterpret_cast<const int64_t*>(host.data())[i]);
      case vt::DType::kI32:
        return static_cast<double>(reinterpret_cast<const int32_t*>(host.data())[i]);
      default:
        // A block-quantized or otherwise unreadable state: report the byte at
        // this index rather than pretend to read a value, so the line is still
        // comparable between the two arms and is not silently empty.
        return static_cast<double>(host[static_cast<size_t>(i)]);
    }
  };
  int64_t nonfinite = 0, nonzero = 0;
  double maxabs = 0.0, sumabs = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    const double v = at(i);
    if (!std::isfinite(v)) { ++nonfinite; continue; }
    if (v != 0.0) ++nonzero;
    const double a = std::fabs(v);
    if (a > maxabs) maxabs = a;
    sumabs += a;
  }
  std::cerr << "Q4EXP_FP step=" << step << " " << what << " dtype="
            << vt::Name(t.dtype) << " n=" << n << " nonfinite=" << nonfinite
            << " nonzero=" << nonzero << " maxabs=" << maxabs
            << " sumabs=" << sumabs << " v=";
  for (int64_t i = 0; i < 4 && i < n; ++i) std::cerr << at(i) << (i + 1 < 4 ? "," : "");
  std::cerr << "\n";
}

ForwardLogits ForwardQwen4ExpForConditionalGeneration(
    LoadedModel& model, const ModelForwardInput& input) {
  // THE DOWNCAST COMES FIRST NOW, and the inversion this comment used to argue
  // for is retired with the refusal it protected. The house shape opens the
  // type-erased handle with `ModelAs<Qwen4ExpLoadedModel>` before doing anything
  // else, because a bare `static_cast` down the hierarchy is undefined behaviour
  // on an object that is not really that type (#775, #730). W5a made
  // `load_weights` produce a genuine `Qwen4ExpLoadedModel`, and this wave gives
  // the opened handle something to be used FOR, which is exactly the condition
  // the previous comment named for restoring it: "W5b restores `ModelAs` in the
  // same change that gives it something to read."
  Qwen4ExpLoadedModel& m =
      ModelAs<Qwen4ExpLoadedModel>(model, "Qwen4ExpForConditionalGeneration");
  Qwen4ExpWeights& w = m.weights();
  const Qwen4ExpParams& p = w.params;

  // ─── WHAT THIS HOOK STILL REFUSES, AND WHY IT IS NOT THE LOOP ──────────────
  //
  // THE REFUSAL NO LONGER NAMES THE LAYER LOOP, because W5f wrote it and a
  // refusal that enumerates finished work sends the next reader to rebuild it —
  // #2288, which this string has now produced eight times. What refuses is the
  // ENGINE's cache plumbing, and the boundary is exact:
  //
  //   * THE ENGINE'S `multi_kv` REFUSAL IS GONE FOR THIS ARCHITECTURE (W5j).
  //     `ModelRegistry::Forward` used to stop every multi-cache topology before
  //     this hook was entered; it now stops only a topology reaching a forward
  //     whose `ModelFactory::consumes_multi_kv` is false, and this factory sets
  //     it. So the three-group cache set the runner allocates REACHES here, and
  //     this hook resolves every member of it by the name
  //     `MakeQwen4ExpKVCache` published — including the recurrent ones, which
  //     `ENG-MULTIKV-BYNAME` made addressable and which the paragraph this
  //     replaces said "cannot be addressed at all".
  //   * THE PLE LAYER'S PAIR OF STATES NO LONGER REFUSES (W5k). This bullet said
  //     they did, "a DTYPE and a RESIDENCY, not a missing channel", and that
  //     "which side of each is wrong is a design call this row does not have the
  //     oracle to settle — the lane pin's `modeling_qwen4_exp.py` is not readable
  //     on this host". The pin is a RELEASE rather than a checkout, and W5k
  //     installed it (transformers 5.16.0, sha256 77fec77d…c459, verified by
  //     regenerating this row's committed forward golden byte-identically) and
  //     read the running model. Upstream types each cache slot from the tensor
  //     that first reaches it (`cache_utils.py:1019-1023`), so the ring carries
  //     the MODEL dtype and the history carries `input_ids.long()` on
  //     `input_ids.device`. The PUBLISHER was right on both counts; both
  //     requirements moved to `RunQwen4ExpPleBlock`, and both caches now persist
  //     in the engine's own recurrent group.
  //   * `RunQwen4ExpQsaBlockPaged` takes a `block_table` of i32 `[1, max_pages]`
  //     — ONE sequence per call — so `num_reqs > 1` is out of reach for the same
  //     seam work.
  //   * THE POSITIONAL ARM is still one shot, and that is a statement about the
  //     ARM rather than about the model: nothing publishes the PLE states there,
  //     so this hook allocates them per call and a per-call buffer is zeroed on
  //     entry. Refused on the same predicate that routes — see the PLE cache
  //     wiring below.
  //
  // So on the BY-NAME channel this hook serves a prefill AND a decode of one
  // sequence, over the engine's own persistent caches, and refuses everything
  // else BY NAME. The sentence this replaces ended "a decode needs a second step,
  // and the second step is what the two PLE states above still cannot carry" —
  // they carry it now, and `test_qwen4_exp_layer_loop.cpp` runs a `past_len = 6`
  // step that samples a token.
  VT_CHECK(input.num_reqs == 1,
           "Qwen4ExpForConditionalGeneration: this forward serves ONE sequence "
           "per call and the step carries " +
               std::to_string(input.num_reqs) +
               ". RunQwen4ExpQsaBlockPaged's block_table is i32 [1, max_pages], "
               "so a ragged multi-request batch needs query_start_loc plumbing "
               "no block on this row carries. Owned by W5f under #2031; see "
               ".agents/specs/qwen4-exp-flash-next.md.");
  const auto T = static_cast<int64_t>(input.token_ids.size());
  VT_CHECK(T > 0,
           "Qwen4ExpForConditionalGeneration: the step carries no tokens");

  // ─── THE IDENTIFIERS THIS STEP ACTUALLY RUNS ON (#2496) ───────────────────
  //
  // `ModelForwardInput::device_token_ids` says, in its own words, "the input ids
  // for this step are ALREADY on the device; the host vector is stale for decode
  // rows". The asynchronous runner's combine splices each decode row's sampled
  // token into that DEVICE buffer on the main queue and deliberately never
  // writes it back to `token_ids`, because materialising it on the host is the
  // synchronise that path exists to remove
  // (`src/vllm/v1/worker/gpu/runner.cpp`, the mirror arm).
  //
  // THIS HOOK READ THE STALE VECTOR, AND THAT IS #2496. On `--device cuda` the
  // mirror is the default, so every decode row embedded — and hashed into the
  // PLE n-gram context — whatever `token_ids_cpu` happened to hold, which for a
  // row the host never wrote is ZERO. Measured on `thor:gpu0` over the released
  // `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S artifact with `VT_Q4EXP_STATE_FP=1`:
  // the n-gram history rolls `[9338, 369] -> [369, 11751] -> [11751, 13] -> …` on
  // `--device cpu` and `[9338, 369] -> [369, 0] -> [0, 0] -> …` on `--device cuda`,
  // so the prefill is right and every decode step feeds the model token id 0.
  // The reported ids stay plausible because they are the argmax of each step's
  // logits; what is wrong is the FEEDBACK, which is why a token gate on the
  // prompt's first token could not see it. TWELVE other translation units already
  // consume this field — `grep -rl 'input\.device_token_ids\|in\.device_token_ids'
  // src/vllm/model_executor/models/` names them, `qwen3_moe_registry.cpp`,
  // `deepseek_v2_registry.cpp`, `nemotron_h_device.cpp` and
  // `kimi_linear_device.cpp` among them — under issue #1305, which is the same
  // defect on Qwen3-MoE. This architecture was simply never wired to it. The
  // count is COUNTED rather than remembered: an earlier draft of this comment
  // said "nine", which is the shape a number quoted often acquires.
  //
  // IT IS MATERIALISED ON THE HOST RATHER THAN SPLICED OVER A DEVICE BUFFER, and
  // that is forced rather than chosen. `detail::ApplyDeviceTokenIds` patches an
  // embed's device identifier buffer, which is enough for a model whose only
  // reader is the embed. This one has a second: `RunQwen4ExpPleBlock` hashes the
  // raw ids in HOST int64 arithmetic (`qwen4_exp::BuildNGramIds`, the port of
  // transformers 5.16.0 `_splitmix64`, :979-983), and the n-gram context it
  // advances is what the next step reads. A device splice would fix the embed and
  // leave the hash reading zeros.
  //
  // THE COPY IS ENQUEUED ON THIS STEP'S QUEUE, so it is ORDERED AFTER the
  // runner's combine rather than racing it, and the drain is the read's other
  // half — the same stage-then-synchronise shape `qwen4_exp_qsa_block.cpp`
  // establishes for its three host reads. One drain per step is the cost, and it
  // is the synchronise W4 removed; the way to get it back is a device-side n-gram
  // hash, which is what upstream does (`torch.ops.vllm.qwen4_exp_compute_ple_ngram_ids`,
  // `vllm/models/qwen4_exp/nvidia/ple_layer.py` @ 191cecd51e, a FORWARD reference
  // past this row's pin). The spec's `## Owed` carries it.
  //
  // NULL ON EVERY OTHER PATH, so `--device cpu` and any synchronous arm read the
  // host vector exactly as before and are byte-identical.
  std::vector<int32_t> resolved_token_ids;
  const std::vector<int32_t>* step_token_ids = &input.token_ids;
  if (input.device_token_ids != nullptr) {
    resolved_token_ids.assign(input.token_ids.size(), 0);
    vt::Backend& tok_backend = vt::GetBackend(input.queue.device.type);
    tok_backend.Copy(input.queue, resolved_token_ids.data(),
                     input.device_token_ids,
                     resolved_token_ids.size() * sizeof(int32_t));
    tok_backend.Synchronize(input.queue);
    step_token_ids = &resolved_token_ids;
  }
  VT_CHECK(input.attn_meta.seq_lens.size() == 1,
           "Qwen4ExpForConditionalGeneration: one sequence per call, so "
           "attn_meta.seq_lens must hold exactly one entry");
  const int64_t past_len =
      static_cast<int64_t>(input.attn_meta.seq_lens[0]) - T;
  VT_CHECK(past_len >= 0,
           "Qwen4ExpForConditionalGeneration: attn_meta.seq_lens[0] is smaller "
           "than this step's token count, so past_len is negative");
  // ─── WHY THE `past_len == 0` REFUSAL IS GONE (W5k, #2031) ─────────────────
  //
  // It said the PLE layer's pair of states "cannot persist", and named a dtype
  // and a residency as the reason: the recurrent group publishes the conv ring at
  // bf16 where `RunQwen4ExpPleBlock` required f32, and publishes the n-gram
  // history as a DEVICE i64 state where the same block required HOST residency.
  // It also said which side of each was wrong "is an oracle question this row
  // could not settle on this host". W5k settled it by installing the lane pin —
  // transformers 5.16.0, `modeling_qwen4_exp.py` sha256 77fec77d…c459, which
  // regenerates the committed forward golden byte-identically — and reading the
  // running model rather than the convention. Upstream was on the PUBLISHER's
  // side both times (see the PLE cache wiring below for the lines), so both
  // requirements moved to the block and both caches now persist.
  //
  // WHAT IS LEFT IS THE SCRATCH ARM, and it is a real limit rather than the same
  // one renamed. With `multi_kv == nullptr` nothing publishes the PLE states, the
  // hook allocates them per call, and a per-call buffer is zeroed on entry: every
  // step would re-seed the n-gram history with EOS and re-zero the ring, and the
  // model would decode as though each token were the first — a fluent wrong
  // answer with no error anywhere. So the positional arm still serves a
  // single-shot prefill, and it says so about ITSELF instead of about the model.
  // WHAT IS LEFT IS THE SCRATCH ARM, and it is a real limit rather than the same
  // one renamed. Where nothing publishes the PLE states the hook allocates them
  // per call, and a per-call buffer is zeroed on entry: every step would re-seed
  // the n-gram history with EOS and re-zero the ring, and the model would decode
  // as though each token were the first — a fluent wrong answer with no error
  // anywhere.
  //
  // THIS CHECK IS AN EARLY MESSAGE AND NOT THE DECIDING ONE. The predicate that
  // ROUTES is whether the recurrent group actually carries the four published
  // states, and it cannot be evaluated until they are resolved by name below; a
  // refusal on a DIFFERENT predicate from the route is how a per-call scratch
  // would serve a continuing step in silence. So the authoritative refusal lives
  // at the wiring site on that exact predicate, and this one only catches the
  // strictly weaker case — no channel at all — where the answer is already known
  // and the message can be plainer.
  VT_CHECK(past_len == 0 || input.multi_kv != nullptr,
           "Qwen4ExpForConditionalGeneration: this step continues a sequence at "
           "past_len " + std::to_string(past_len) +
               " and carries NO by-name cache index, so nothing published the "
               "PLE layer's conv ring or its n-gram history. On that arm this "
               "hook allocates both per call and a per-call buffer is zeroed on "
               "entry, so the history would be re-seeded with eos_token_id and "
               "the ring re-zeroed at every step: the model would decode as "
               "though each token were the first, and no shape or dtype error "
               "would say so. A continuing step must carry the three-group "
               "topology MakeQwen4ExpKVCache publishes. Owned by "
               "MODEL-MM-QWEN4-EXP; see .agents/specs/qwen4-exp-flash-next.md "
               "and issue #2031.");

  vt::Backend& backend = vt::GetBackend(input.queue.device.type);
  dense_attn::Dev d{backend, input.queue};

  // ─── the caches, in the loop's own three-vector shape ─────────────────────
  int64_t n_gdn = 0, n_qsa = 0;
  for (Qwen4ExpLayerKind k : p.layer_types) {
    if (k == Qwen4ExpLayerKind::kLinearAttention) {
      ++n_gdn;
    } else {
      ++n_qsa;
    }
  }
  // ─── WHICH CACHE IS WHICH: POSITIONALLY, OR BY NAME (W5j, #2031, #2353) ───
  //
  // This hook reads the caches TWO ways, because the engine's answer to "which
  // cache is this" changes shape the moment the model publishes a group the
  // positional convention cannot address.
  //
  //   POSITIONAL (`multi_kv == nullptr`). `attn_kv[i]` is the i-th
  //   qwen_sparse_attention layer's paged K/V and `gdn_state[i]` the i-th
  //   linear_attention layer's state. NOTHING publishes an indexer side cache on
  //   this arm, so the hook allocates the per-call scratch it always did. Every
  //   hand-built caller takes this arm, and it is byte-identical to W5i.
  //
  //   BY NAME (`multi_kv != nullptr`). The runner allocated all THREE published
  //   groups. `attn_kv` then holds `2 * n_qsa` entries — group 0's K/V AND
  //   group 2's indexer pages, one buffer per (attention group x layer), in
  //   PUBLICATION order (`runner.cpp`, the `for (int g : attn_group_ids_)` loop)
  //   — so a POSITIONAL read of it is off by a whole group from the second QSA
  //   layer onward and returns another layer's keys with no shape error. The
  //   assertion that used to stand here, `attn_kv.size() == n_qsa`, was the only
  //   thing between that and a wrong answer, and it would have fired on every
  //   real step: this is why lifting the engine guard without fixing the count
  //   would have changed a refusal into a different refusal and not into a
  //   token.
  //
  // The NAMES `MakeQwen4ExpKVCache` published are the only thing that says which
  // buffer is which, and `MultiKvCacheIndex::Resolve` is how they are asked. Both
  // sides build those names through the SAME three helpers above, which is what
  // the spec's `## Owed` requires: two derivations of one name set is the shape
  // that can disagree, and a disagreement between the publisher and the consumer
  // is a run-time refusal with no compile error behind it. A resolution failure
  // therefore means the ENGINE did not carry what this model published, which is
  // exactly what the refusal below says.
  const MultiKvCacheIndex* mk = input.multi_kv;

  std::vector<PagedKvCache> qsa_kv(static_cast<size_t>(n_qsa));
  // Group 2, one per QSA layer. EMPTY on the positional arm, where the hook
  // allocates scratch instead.
  std::vector<PagedKvCache> qsa_idx;
  std::vector<GdnStateCache> gdn(static_cast<size_t>(n_gdn));
  // Group 2's page map for THIS sequence, i32 `[cols]`. On the positional arm it
  // is the identity over a private buffer; on the by-name arm it is the row the
  // runner gathered.
  std::vector<int32_t> idx_bt;

  if (mk == nullptr) {
    VT_CHECK(static_cast<int64_t>(input.gdn_state.size()) == n_gdn,
             "Qwen4ExpForConditionalGeneration: the runner handed " +
                 std::to_string(input.gdn_state.size()) +
                 " recurrent state caches for " + std::to_string(n_gdn) +
                 " linear_attention layers");
    VT_CHECK(static_cast<int64_t>(input.attn_kv.size()) == n_qsa,
             "Qwen4ExpForConditionalGeneration: the runner handed " +
                 std::to_string(input.attn_kv.size()) +
                 " paged K/V caches for " + std::to_string(n_qsa) +
                 " qwen_sparse_attention layers, and no by-name cache index. "
                 "A step that publishes the QSA indexer side cache carries "
                 "multi_kv; a step without it carries exactly one paged cache "
                 "per QSA layer");
    for (int64_t i = 0; i < n_qsa; ++i)
      qsa_kv[static_cast<size_t>(i)] = input.attn_kv[static_cast<size_t>(i)];
    for (int64_t i = 0; i < n_gdn; ++i)
      gdn[static_cast<size_t>(i)] = input.gdn_state[static_cast<size_t>(i)];
  } else {
    qsa_idx.resize(static_cast<size_t>(n_qsa));
    VT_CHECK(mk->group_ids != nullptr && mk->group_ids->size() == mk->size(),
             "Qwen4ExpForConditionalGeneration: the by-name cache index carries "
             "no group ids, so the QSA indexer side cache's own block table "
             "cannot be found. See .agents/specs/qwen4-exp-flash-next.md and "
             "issues #2031 and #2249.");
    // Resolve ONE published name, refusing each of the three ways the answer can
    // be wrong SEPARATELY, because they mean different things: a name nothing
    // was published under is a publisher/consumer disagreement, a wrong payload
    // kind is a group classified as the other arm, and an out-of-range slot is a
    // channel whose locators disagree with the payload it describes.
    const auto locate = [&](const std::string& name, KvCachePayload want,
                            const char* role) {
      const int64_t flat = mk->Find(name);
      VT_CHECK(flat >= 0,
               std::string("Qwen4ExpForConditionalGeneration: the engine "
                           "published no KV cache under '") +
                   name + "', which is where this model keeps its " + role +
                   ". The channel carries " + std::to_string(mk->size()) +
                   " cache(s), first '" + std::string(mk->first_name()) +
                   "'. MakeQwen4ExpKVCache publishes that name; a step that "
                   "does not carry it is not this model's topology. See "
                   ".agents/specs/qwen4-exp-flash-next.md and issue #2031.");
      KvCachePayload kind = KvCachePayload::kPaged;
      int32_t slot = -1;
      VT_CHECK(mk->PayloadAt(flat, &kind, &slot),
               std::string("Qwen4ExpForConditionalGeneration: the by-name cache "
                           "index names '") +
                   name + "' but carries no payload locator for it");
      VT_CHECK(kind == want,
               std::string("Qwen4ExpForConditionalGeneration: '") + name +
                   "' is this model's " + role + ", which is a " +
                   (want == KvCachePayload::kPaged ? "PAGED" : "RECURRENT") +
                   " cache, and the engine published it as a " +
                   (kind == KvCachePayload::kPaged ? "PAGED" : "RECURRENT") +
                   " one. Reading it from the other payload container returns "
                   "an unrelated buffer with no shape error, so this refuses.");
      VT_CHECK(slot >= 0,
               std::string("Qwen4ExpForConditionalGeneration: '") + name +
                   "' resolved to no payload slot");
      return std::pair<int64_t, int32_t>(flat, slot);
    };

    int64_t qi = 0;
    int64_t gi = 0;
    int idx_group = -1;
    for (size_t l = 0; l < p.layer_types.size(); ++l) {
      const std::string idx = std::to_string(l);
      if (p.layer_types[l] == Qwen4ExpLayerKind::kLinearAttention) {
        const auto r = locate(Qwen4ExpLinearAttnName(l),
                              KvCachePayload::kRecurrent, "recurrent state");
        VT_CHECK(static_cast<size_t>(r.second) < input.gdn_state.size(),
                 "Qwen4ExpForConditionalGeneration: layer " + idx +
                     "'s recurrent state resolved to slot " +
                     std::to_string(r.second) + " of " +
                     std::to_string(input.gdn_state.size()) + " states");
        gdn[static_cast<size_t>(gi++)] =
            input.gdn_state[static_cast<size_t>(r.second)];
      } else {
        const auto a = locate(Qwen4ExpQsaAttnName(l), KvCachePayload::kPaged,
                              "paged K/V");
        const auto k = locate(Qwen4ExpQsaIndexerName(l), KvCachePayload::kPaged,
                              "QSA indexer side cache");
        VT_CHECK(static_cast<size_t>(a.second) < input.attn_kv.size() &&
                     static_cast<size_t>(k.second) < input.attn_kv.size(),
                 "Qwen4ExpForConditionalGeneration: layer " + idx +
                     " resolved to paged slots " + std::to_string(a.second) +
                     " and " + std::to_string(k.second) + " of " +
                     std::to_string(input.attn_kv.size()) + " paged caches");
        VT_CHECK(a.second != k.second,
                 "Qwen4ExpForConditionalGeneration: layer " + idx +
                     "'s paged K/V and its indexer side cache resolved to the "
                     "SAME slot " + std::to_string(a.second) +
                     "; they are two published groups and two buffers");
        qsa_kv[static_cast<size_t>(qi)] =
            input.attn_kv[static_cast<size_t>(a.second)];
        qsa_idx[static_cast<size_t>(qi)] =
            input.attn_kv[static_cast<size_t>(k.second)];
        // The GROUP the side cache came from, taken from the entry rather than
        // assumed to be 2: `MakeQwen4ExpKVCache` publishes it third today, and a
        // hard-coded id would keep answering after a reordering that this
        // resolution would otherwise survive.
        const int32_t g = (*mk->group_ids)[static_cast<size_t>(k.first)];
        VT_CHECK(idx_group < 0 || idx_group == g,
                 "Qwen4ExpForConditionalGeneration: the QSA indexer side caches "
                 "came from more than one published group (" +
                     std::to_string(idx_group) + " and " + std::to_string(g) +
                     "), so they do not share one block table");
        idx_group = g;
        ++qi;
      }
    }
    VT_CHECK(qi == n_qsa && gi == n_gdn,
             "Qwen4ExpForConditionalGeneration: resolved " +
                 std::to_string(qi) + " of " + std::to_string(n_qsa) +
                 " QSA layers and " + std::to_string(gi) + " of " +
                 std::to_string(n_gdn) + " linear_attention layers by name");

    // GROUP 2'S OWN GATHERED TABLE, which is the whole point of W5c-2's fourth
    // vector: group 0's table names group 0's physical pages in group 0's pool,
    // and reading the side cache through it returns another sequence's keys with
    // no shape error.
    int cols = 0;
    const std::vector<int32_t>* bt = mk->BlockTableForGroup(idx_group, &cols);
    VT_CHECK(bt != nullptr && cols > 0,
             "Qwen4ExpForConditionalGeneration: the engine gathered no block "
             "table for published group " + std::to_string(idx_group) +
                 ", the QSA indexer side cache's group; " +
                 std::to_string(mk->num_group_block_tables()) + " of " +
                 std::to_string(mk->num_published_groups()) +
                 " published group(s) carry one. A cache with no page map is "
                 "allocated and unreadable. See "
                 ".agents/specs/qwen4-exp-flash-next.md and issue #2249.");
    VT_CHECK(static_cast<int64_t>(bt->size()) >= cols,
             "Qwen4ExpForConditionalGeneration: group " +
                 std::to_string(idx_group) + "'s block table has " +
                 std::to_string(bt->size()) + " entries for a declared width of " +
                 std::to_string(cols));
    // ROW 0, because this hook already refused `num_reqs != 1` above; the table
    // is row-major `[num_reqs, cols]`.
    idx_bt.assign(bt->begin(), bt->begin() + cols);
  }

  Qwen4ExpForwardCaches caches;
  caches.gdn = gdn;

  // ONE block table and ONE slot mapping for the whole step, taken from the
  // runner's own metadata rather than rebuilt: `dense_attn::AttnBlock` does not
  // build them either (`StepInputs` carries the runner's own), and a locally
  // invented mapping would store this step's K/V at pages nothing else reads.
  const int64_t cols = input.attn_meta.block_table_num_cols;
  VT_CHECK(cols > 0 && static_cast<int64_t>(
                           input.attn_meta.block_table_tensor.size()) >= cols,
           "Qwen4ExpForConditionalGeneration: the step carries no block table");
  VT_CHECK(static_cast<int64_t>(input.attn_meta.slot_mapping.size()) == T,
           "Qwen4ExpForConditionalGeneration: the slot mapping has " +
               std::to_string(input.attn_meta.slot_mapping.size()) +
               " entries for " + std::to_string(T) + " tokens");
  std::vector<int32_t> bt(input.attn_meta.block_table_tensor.begin(),
                          input.attn_meta.block_table_tensor.begin() + cols);
  std::vector<int64_t> slots(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    slots[static_cast<size_t>(t)] = input.attn_meta.slot_mapping[static_cast<size_t>(t)];

  // The PLE scratch buffers live for exactly this call, which is what
  // `past_len == 0` buys and what the refusal above protects. The QSA indexer
  // scratch beside them is the POSITIONAL arm's only; see below.
  // POSITIONAL ARM ONLY: on the by-name arm the engine's own group-2 pages are
  // the buffer and nothing here is allocated.
  std::vector<std::vector<uint16_t>> index_keys(
      mk == nullptr ? static_cast<size_t>(n_qsa) : 0U);
  std::vector<std::vector<float>> ple_convs(p.ple.layer_ids_zero_based.size());
  std::vector<std::vector<int64_t>> ple_tokens(p.ple.layer_ids_zero_based.size());

  dense_attn::DBuf d_bt(d, vt::DType::kI32, {1, cols}, bt.data());
  dense_attn::DBuf d_slots(d, vt::DType::kI64, {T}, slots.data());

  // ─── THE QSA INDEXER SIDE CACHE, IN THE ENGINE'S OWN PAGED SHAPE (W5i) ────
  //
  // The block reads and writes group 2 through a block table now, so the scratch
  // this hook allocates is shaped the way the engine allocates group 2 —
  // `[num_pages, block_size, indexer_head_dim]`, the FUSED 3-dim MLA page — and
  // is addressed through a table. Nothing here is a second geometry invented for
  // the scratch: `page` is the group-0 page size the runner handed us, and
  // `MakeQwen4ExpKVCache` publishes group 2 at that same `block_size` with
  // `compress_ratio = 1`, which is W5h's capacity law (one indexer row per token
  // slot) written as a shape.
  //
  // THE TABLE IS THE IDENTITY, AND THAT IS WHAT A PRIVATE PER-CALL BUFFER MEANS,
  // NOT A SHORTCUT. There is no allocator behind this scratch, so there are no
  // physical pages to permute; logical page `i` IS physical page `i`. It is NOT
  // group 0's table: those are group 0's physical page ids in group 0's pool, and
  // adopting them here would name pages this buffer does not have. The
  // permutation the translation exists for is exercised by the block's own gate,
  // which runs a non-identity `{2, 6, 1}` against a logical `{0, 1, 2}` with a
  // partial final page (`test_qwen4_exp_qsa_block.cpp`, the W5i case).
  //
  // WHAT REPLACED IT (W5j). On the by-name arm these are group 2's OWN cache and
  // `group_block_tables[<group 2>]` — the vector W5c-2 already gathers — and
  // NOTHING in the block changed: `RunQwen4ExpQsaBlockPaged` takes the same
  // rank-3 page tensor and the same i32 `[1, pages]` table either way. That
  // substitution is the whole point of having paged it here.
  //
  // THE SCRATCH ARM SURVIVES, AND NOT AS A FALLBACK. It is what a caller with no
  // engine behind it gets — the block's own gate, and any hand-built step — and
  // on it the identity table is the CORRECT map, because a private buffer's
  // logical page `i` really is its physical page `i`. What it is NOT is the
  // production path any more: an engine step arrives with `multi_kv` and takes
  // the other arm, so the permutation is exercised by the allocator's pages and
  // not only by the block's `{2, 6, 1}` case.
  const int64_t id_dim = p.qsa.head_dim;
  int64_t idx_page = 0;
  int64_t idx_pages = 0;
  if (mk == nullptr) {
    idx_page = input.attn_kv[0].block_size > 0 ? input.attn_kv[0].block_size : T;
    idx_pages = (T + idx_page - 1) / idx_page;
    idx_bt.assign(static_cast<size_t>(idx_pages), 0);
    for (int64_t k = 0; k < idx_pages; ++k)
      idx_bt[static_cast<size_t>(k)] = static_cast<int32_t>(k);
  } else {
    // The geometry comes from the ENGINE's own view of group 2's buffer, never
    // from a second derivation here: `MLAAttentionSpec::real_page_size_bytes` is
    // `storage_block_size() * num_kv_heads * head_size`, and at
    // `compress_ratio == 1` and the `indexer_kv_heads == 1` upstream requires,
    // that is exactly `[num_blocks, block_size, indexer_head_dim]`.
    idx_page = qsa_idx[0].block_size;
    idx_pages = static_cast<int64_t>(idx_bt.size());
  }
  dense_attn::DBuf d_idx_bt(d, vt::DType::kI32, {1, idx_pages}, idx_bt.data());

  std::vector<dense_attn::DBuf> idx_bufs;
  idx_bufs.reserve(static_cast<size_t>(n_qsa));
  caches.qsa.resize(static_cast<size_t>(n_qsa));
  for (int64_t i = 0; i < n_qsa; ++i) {
    vt::Tensor index_key;
    if (mk == nullptr) {
      index_keys[static_cast<size_t>(i)].assign(
          static_cast<size_t>(idx_pages * idx_page * id_dim), 0);
      idx_bufs.emplace_back(d, vt::DType::kBF16,
                            std::vector<int64_t>{idx_pages, idx_page, id_dim},
                            index_keys[static_cast<size_t>(i)].data());
      index_key = idx_bufs.back().t();
    } else {
      const PagedKvCache& ic = qsa_idx[static_cast<size_t>(i)];
      VT_CHECK(ic.data != nullptr && ic.num_blocks > 0 && ic.block_size > 0,
               "Qwen4ExpForConditionalGeneration: the QSA indexer side cache "
               "for qwen_sparse_attention layer " + std::to_string(i) +
                   " is unallocated");
      VT_CHECK(ic.num_kv_heads == 1 && ic.head_size == id_dim,
               "Qwen4ExpForConditionalGeneration: the QSA indexer side cache is "
               "published as an MLAAttentionSpec of " +
                   std::to_string(ic.num_kv_heads) + " head(s) x " +
                   std::to_string(ic.head_size) +
                   ", and this model stores ONE raw key of indexer_head_dim " +
                   std::to_string(id_dim) +
                   " per token slot. upstream requires indexer_kv_heads == 1, "
                   "so any other view is a different cache");
      VT_CHECK(ic.block_size == idx_page,
               "Qwen4ExpForConditionalGeneration: the QSA indexer side caches "
               "do not share one page size (" + std::to_string(ic.block_size) +
                   " against " + std::to_string(idx_page) +
                   "), so one gathered block table cannot address them all");
      // The FUSED 3-dim MLA page, over the runner's own buffer. Not a copy: the
      // block WRITES this step's raw keys into it, and a copy would drop them.
      index_key = dense_attn::MakeTensor(ic.data, ic.dtype, input.queue.device,
                                         {ic.num_blocks, ic.block_size, id_dim});
    }
    caches.qsa[static_cast<size_t>(i)].kv = qsa_kv[static_cast<size_t>(i)];
    caches.qsa[static_cast<size_t>(i)].block_table = d_bt.t();
    caches.qsa[static_cast<size_t>(i)].slot_mapping = d_slots.t();
    caches.qsa[static_cast<size_t>(i)].index_key = index_key;
    caches.qsa[static_cast<size_t>(i)].index_block_table = d_idx_bt.t();
  }

  // ─── THE PLE LAYER'S TWO STATES (W5k, #2031) ──────────────────────────────
  //
  // These are the pair that refused `past_len > 0` for four waves, and the reason
  // was never a missing channel: `MakeQwen4ExpKVCache` has published both as the
  // recurrent group's third and fourth states since W5c, and W5j made the group
  // resolvable by name. What refused was a DTYPE and a RESIDENCY, and W5k settled
  // both against the running lane oracle (transformers 5.16.0, sha256
  // 77fec77d…c459) rather than against this tree's convention:
  //
  //   * THE RING'S DTYPE IS THE MODEL'S. `cache_utils.py:1019-1023` allocates
  //     each slot as `torch.zeros(..., dtype=conv_states.dtype,
  //     device=conv_states.device)` — PER SLOT, from the tensor that first
  //     reaches it — and the tensor reaching this one is `hidden_states`
  //     (`modeling_qwen4_exp.py:1157-1159`). Observed: at `dtype=torch.bfloat16`
  //     the oracle reports `conv_states[1] dtype=torch.bfloat16`. So the
  //     PUBLISHER was right at bf16 and `RunQwen4ExpPleBlock`'s f32 requirement
  //     was the wrong side; it now requires the ring to EQUAL the stream dtype.
  //   * THE HISTORY IS DEVICE-RESIDENT. `:1070` takes `input_ids.long()` and
  //     `:1089-1091` hands exactly that to `update_conv_state(..., state_idx=2)`,
  //     so the slot's device is `input_ids.device` — the compute device. The
  //     publisher was right here too, and the block now stages the row around its
  //     host splitmix64 instead of refusing the buffer.
  //
  // SO THIS TAKES THE ENGINE'S OWN BUFFERS ON THE BY-NAME ARM, which is what
  // makes a second step possible: a per-call scratch is zeroed on entry, so every
  // step would re-seed the history with EOS and re-zero the ring, and the model
  // would decode as though each token were the first. The scratch remains on the
  // positional arm, where nothing publishes these states at all.
  //
  // THE STATE ORDER IS THE ONE `MakeQwen4ExpKVCache` PUBLISHES —
  // `[gdn_conv, temporal, ple_conv, ngram]` — read from `GdnStateCache::states`,
  // the complete ordered list `ENG-RECURRENT-MULTISTATE` (#2131) added. It is NOT
  // read positionally out of anything else: `conv_state` and `ssm_state` are
  // `states[0]` and `states[1]` under different names, and slots 2 and 3 have no
  // named field to be confused with.
  std::vector<dense_attn::DBuf> ple_conv_bufs;
  ple_conv_bufs.reserve(p.ple.layer_ids_zero_based.size());
  caches.ple.resize(p.ple.layer_ids_zero_based.size());
  for (size_t i = 0; i < p.ple.layer_ids_zero_based.size(); ++i) {
    const int64_t stream = p.stream_width();
    const int64_t state_len = p.ple.short_conv_state_len();
    const int64_t ctx = p.ple.ngram_size - 1;
    // WHICH RECURRENT SLOT THIS PLE LAYER OWNS. The PLE layer is a
    // linear_attention layer (upstream refuses PLE on a sparse one), so its state
    // is the `gdn` entry at its RANK AMONG LINEAR LAYERS — never at its decoder
    // layer index, which counts the sparse layers too.
    const int64_t ple_layer = p.ple.layer_ids_zero_based[i];
    int64_t rank = -1, seen = 0;
    for (size_t l = 0; l < p.layer_types.size(); ++l) {
      if (p.layer_types[l] != Qwen4ExpLayerKind::kLinearAttention) continue;
      if (static_cast<int64_t>(l) == ple_layer) { rank = seen; break; }
      ++seen;
    }
    VT_CHECK(rank >= 0 && rank < static_cast<int64_t>(gdn.size()),
             "Qwen4ExpForConditionalGeneration: PLE layer " +
                 std::to_string(ple_layer) +
                 " is not a linear_attention layer, so it owns no recurrent "
                 "state slot. See .agents/specs/qwen4-exp-flash-next.md.");
    const GdnStateCache& g = gdn[static_cast<size_t>(rank)];

    // The by-name arm carries the published four; a hand-built cache sets only the
    // two named fields and leaves `states` EMPTY, which `qwen3_5.h` documents and
    // which is the scratch arm below.
    const bool published = g.states.size() >= 4;
    if (published) {
      caches.ple[i].conv_state = g.states[2];
      caches.ple[i].tokens = g.states[3];
      // `MambaSpec` shapes carry the slot dim prepended, so the history arrives
      // [num_slots, ngram_size-1] and the ring [num_slots, stream, state_len] —
      // exactly the two shapes `RunQwen4ExpPleBlock` checks.
      VT_CHECK(caches.ple[i].tokens.rank == 2 &&
                   caches.ple[i].tokens.shape[1] == ctx,
               "Qwen4ExpForConditionalGeneration: the recurrent group's fourth "
               "state is this model's n-gram history and must be [slots," +
                   std::to_string(ctx) + "]");
      VT_CHECK(caches.ple[i].conv_state.rank == 3 &&
                   caches.ple[i].conv_state.shape[1] == stream &&
                   caches.ple[i].conv_state.shape[2] == state_len,
               "Qwen4ExpForConditionalGeneration: the recurrent group's third "
               "state is this model's PLE conv ring and must be [slots," +
                   std::to_string(stream) + "," + std::to_string(state_len) + "]");
    } else {
      // ─── THE AUTHORITATIVE CONTINUING-STEP REFUSAL, ON THE ROUTING PREDICATE ──
      // `published` is what decides which buffers this step runs on, so it is what
      // must decide whether the step may continue a sequence. The early check at
      // the top of this hook tests `multi_kv != nullptr`, which is STRICTLY
      // WEAKER: a channel can be present and still carry a recurrent group whose
      // `states` list was never filled (every hand-built `GdnStateCache` in this
      // tree sets the two named fields and leaves it empty — `qwen3_5.h` says so).
      // Refusing on the weaker predicate would let exactly that case through onto
      // a zeroed per-call scratch, which produces a fluent wrong answer and no
      // error at all. Same predicate, same decision.
      VT_CHECK(past_len == 0,
               "Qwen4ExpForConditionalGeneration: this step continues a sequence "
               "at past_len " + std::to_string(past_len) +
                   " and the recurrent group carries " +
                   std::to_string(g.states.size()) +
                   " state(s), not the four MakeQwen4ExpKVCache publishes "
                   "([gdn_conv, temporal, ple_conv, ngram]). Without the third "
                   "and fourth this hook allocates the PLE conv ring and the "
                   "n-gram history per call, and a per-call buffer is zeroed on "
                   "entry: the history would be re-seeded with eos_token_id and "
                   "the ring re-zeroed at every step, so the model would decode "
                   "as though each token were the first with no shape or dtype "
                   "error to say so. Owned by MODEL-MM-QWEN4-EXP; see "
                   ".agents/specs/qwen4-exp-flash-next.md and issue #2031.");
      // THE SCRATCH ARM, byte-identical to W5j apart from the ring's dtype. It is
      // zeroed on entry, so it serves a `past_len == 0` call and nothing else,
      // which is what the refusal directly above enforces.
      ple_tokens[i].assign(static_cast<size_t>(ctx), 0);
      // THE SCRATCH RING CARRIES THE STREAM DTYPE, not f32. It was f32 while the
      // loop ran bf16, which is the widening the block's new equality check
      // refuses — and which no golden could ever have seen, because on a
      // `past_len == 0` call the ring is zeroed on entry and only WRITTEN at the
      // end, so its dtype cannot move a single output value. It moves bytes and
      // it would have moved the answer the moment a second step read it.
      // Sized in FLOATS and viewed at the stream dtype, so the storage is >= the
      // bytes any dtype up to f32 needs and is zero-filled either way (all-zero
      // bits is +0.0 in f32 and bf16 alike). Over-allocating a per-call scratch
      // by a factor of two is not worth a second length expression to get wrong.
      ple_convs[i].assign(static_cast<size_t>(stream * state_len), 0.0F);
      ple_conv_bufs.emplace_back(d, kQwen4ExpStreamDType,
                                 std::vector<int64_t>{1, stream, state_len},
                                 ple_convs[i].data());
      caches.ple[i].conv_state = ple_conv_bufs.back().t();
      caches.ple[i].tokens = dense_attn::MakeTensor(
          ple_tokens[i].data(), vt::DType::kI64,
          vt::Device{vt::DeviceType::kCPU, 0}, {1, ctx});
    }
    // ONE SEQUENCE PER CALL, so the state row is the one this step's recurrent
    // metadata names. `gdn_meta.non_spec_state_indices_tensor` is the runner's own
    // per-request slot assignment and is what every other recurrent consumer in
    // this tree reads; defaulting to 0 would put a second sequence's state on the
    // first sequence's row with no shape error.
    int64_t row = 0;
    if (published && input.gdn_meta.non_spec_state_indices_tensor.has_value() &&
        !input.gdn_meta.non_spec_state_indices_tensor->empty()) {
      row = (*input.gdn_meta.non_spec_state_indices_tensor)[0];
    }
    caches.ple[i].state_row = row;
  }

  const Qwen4ExpTextModelOutput hidden = Qwen4ExpTextModelForward(
      d, w, input.config, *step_token_ids, input.positions, input.attn_meta,
      input.gdn_meta, caches, past_len);

  // DECODEDIV (#2496): the per-step state fingerprint, default OFF. See
  // `Q4ExpStateSummary` above for why it prints a summary and not a hash, and
  // why it is the arm that runs on the REAL weights.
  if (Q4ExpStateFpEnabled()) {
    static long q4exp_fp_step = 0;
    const long step = q4exp_fp_step++;
    std::cerr << "Q4EXP_FP step=" << step << " BEGIN T=" << T
              << " past_len=" << past_len << " device="
              << static_cast<int>(input.queue.device.type) << "\n";
    for (size_t i = 0; i < caches.gdn.size(); ++i) {
      const GdnStateCache& g = caches.gdn[i];
      Q4ExpStateSummary(d, step, "gdn[" + std::to_string(i) + "].conv", g.conv_state);
      Q4ExpStateSummary(d, step, "gdn[" + std::to_string(i) + "].ssm", g.ssm_state);
      for (size_t k = 2; k < g.states.size(); ++k)
        Q4ExpStateSummary(d, step, "gdn[" + std::to_string(i) + "].state" +
                                       std::to_string(k), g.states[k]);
    }
    for (size_t i = 0; i < caches.ple.size(); ++i) {
      Q4ExpStateSummary(d, step, "ple[" + std::to_string(i) + "].ring",
                        caches.ple[i].conv_state);
      Q4ExpStateSummary(d, step, "ple[" + std::to_string(i) + "].ngram",
                        caches.ple[i].tokens);
    }
    for (size_t i = 0; i < caches.qsa.size() && i < 2; ++i)
      Q4ExpStateSummary(d, step, "qsa[" + std::to_string(i) + "].index_key",
                        caches.qsa[i].index_key);
    Q4ExpStateSummary(d, step, "hidden", hidden.tensor);
  }

  // ─── the lm_head, which is `Qwen4ExpForCausalLM` and not the text model ────
  // `Qwen4ExpTextModel` has NO final RMSNorm (the mixer's `hc_norm` is the last
  // normalization in the model), so the tail is the loop's output straight into
  // one `kMatmulBT`. The head is TIED to the embedding table when the file
  // carries no `output.weight`, which is read off the FILE by the loader.
  const OwnedTensor& head =
      w.tied_word_embeddings ? w.embed_tokens : w.lm_head;
  VT_CHECK(head.rank == 2 && head.shape[0] == p.vocab_size &&
               head.shape[1] == p.hidden_size,
           "Qwen4ExpForConditionalGeneration: the lm_head must be [vocab, "
           "hidden_size]");

  // ONLY THE ROWS THE SAMPLER ASKED FOR, which on a prefill is the last token
  // of each sequence. Computing the full [T, vocab] and discarding it is the
  // shape every other forward in this tree avoids.
  std::vector<int32_t> rows = input.logits_indices;
  if (rows.empty()) rows.push_back(static_cast<int32_t>(T - 1));
  const auto R = static_cast<int64_t>(rows.size());
  for (int32_t r : rows) {
    VT_CHECK(r >= 0 && static_cast<int64_t>(r) < T,
             "Qwen4ExpForConditionalGeneration: logits index " +
                 std::to_string(r) + " is outside this step's " +
                 std::to_string(T) + " tokens");
  }
  dense_attn::DBuf gathered(d, hidden.tensor.dtype, {R, p.hidden_size});
  {
    dense_attn::DBuf d_rows(d, vt::DType::kI32, {R}, rows.data());
    vt::Tensor g = gathered.t();
    vt::IndexSelect(input.queue, g, hidden.tensor, d_rows.t());
  }
  dense_attn::DBuf logits(d, vt::DType::kF32, {R, p.vocab_size});
  {
    vt::Tensor o = logits.t();
    vt::MatmulBT(input.queue, o, gathered.t(),
                 dense_attn::ResidentWeight(d, head, {p.vocab_size, p.hidden_size}));
  }

  // The head's output is the model's answer, and the `LayerFp` tap stops at
  // `out`, so this is the only observation of the `lm_head` decode. A no-op
  // unless `VT_Q4EXP_LOGITS` is set, and it reads through the same Copy +
  // Synchronize pair the layer tap uses, so both arms are read the same way.
  detail::Qwen4ExpLogitsTap(d, logits.t());

  ForwardLogits r;
  r.rows = R;
  r.vocab = p.vocab_size;
  r.device_tensor = logits.t();
  r.device_storage = logits.ReleaseShared();
  return r;
}

// ─── The KV-cache spec (W5c, #2031) ──────────────────────────────────────────
//
// THREE published groups, and the shape of them is the decision this function
// exists to record:
//
//   0. the QSA layers' paged K+V              `FullAttentionSpec`
//   1. EVERY linear-attention layer's state   `MambaSpec`, N states
//   2. the QSA layers' indexer side cache     `MLAAttentionSpec`, ONE ROW PER TOKEN
//
// ONE UNIFORM RECURRENT GROUP, NOT ONE PER LAYER, AND THE COST IS DELIBERATE.
// Only ONE linear-attention layer carries the PLE conv and the n-gram token
// history (`ple_layer_ids` selects 0-based layer 1 on the published
// checkpoint), so a per-layer spec would give 35 of the 36 recurrent layers a
// smaller state set. Upstream cannot express that and does not try:
// `get_mamba_state_shape_from_config` is a CLASSMETHOD taking only the config
// (`vllm/model_executor/models/interfaces.py:809-812` at the pin
// `5559679229`), all 18 implementations of it declare ONE shape model-wide and
// not one of them takes a `layer_idx`, and `get_mamba_groups`
// (`vllm/v1/worker/mamba_utils.py:441`) asserts
// `all(mamba_specs[0] == spec for spec in mamba_specs)` — every recurrent spec
// in the model equal, field for field. Upstream pays the uniform cost by
// PADDING rather than by splitting (`vllm/v1/core/kv_cache_utils.py:1101-1109`
// sets `page_size_padded=max_page_size` on the smaller `MambaSpec`).
//
// The cost here, derived from the published shapes rather than measured: the
// PLE conv is `10240 x 9` at bf16 = 184320 B and the n-gram history is 2 int64
// = 16 B, so 184336 B per sequence on each of the 35 linear layers that do not
// use them. At the default `max_num_seqs` of 8 that is 49.2 MiB — 0.09% of the
// GB10 headroom the row's `## Hardware` section accounts. Splitting the group
// to recover it would need a SECOND recurrent group, which
// `.agents/specs/recurrent-multistate.md` records as owed generic engine debt
// and which this topology does not need.
//
// STATE ORDER IS A DELIBERATE DIVERGENCE FROM UPSTREAM'S LIST ORDER, and it is
// the same bytes either way. Upstream keeps the three CONV states adjacent
// (`number_of_conv_states = 3`: GDN conv, PLE conv, n-gram history) with the
// temporal state after them. This tree publishes
// `[gdn_conv, temporal, ple_conv, ngram]` because `GdnStateCache` exposes
// `conv_state = states[0]` and `ssm_state = states[1]` as NAMED fields that
// THREE model families already read (`qwen3_5.cpp`, `kimi_linear_device.cpp`
// and the `nemotron_h` pair `nemotron_h_device.cpp` / `nemotron_h_forward.h`),
// and moving the temporal state off slot 1 would silently re-point every one
// of them. Recorded in the row spec.
//
// THE COUNT IS THREE, NOT FOUR (#2203). `gemma4_mm.cpp` was named here and in
// `.agents/specs/recurrent-multistate.md` as a fourth reader, and it reads
// NEITHER field: its only two mentions of the type are an include comment and
// `std::vector<GdnStateCache> no_gdn_state;` (`gemma4_mm.cpp:221`), passed
// EMPTY, which is the file proving Gemma-4 has no recurrent arm.
// `muse_glimmer_mm.cpp:340` and `qwen3_vl.cpp:621` carry that same empty-vector
// shape. Grepping the FIELD name over-counts in the other direction:
// `glm5_next_kda.cpp` matches `conv_state` 13 times on
// `Glm5NextKdaCache::conv_state`, a `std::vector<float>` KDA sequence state
// (`glm5_next_kda.h:314`), where this one is a `vt::Tensor` (`qwen3_5.h:111`),
// and that file has zero occurrences of `GdnStateCache`. Grep the TYPE.
//
// REAL PER-LAYER NAMES, NEVER PLACEHOLDERS. `ResolveKVCacheGroupLayerNames`
// (`src/vllm/v1/kv_cache_interface.cpp`) rewrites a placeholder group set into
// per-layer names, but its fallback classification can name only a TARGET
// attention group and one `fa_draft` slot: a third attention group gets
// `layer_names.clear()` and an unnamed group is then refused by the runner's
// multi-cache admission check, because its names "do not all resolve to
// distinct in-range layer indices". Publishing the real names also makes the
// rewrite a no-op by its own idempotence guard, so what the runner allocates is
// what this function said.
//
// GROUP 2 IS AN `MLAAttentionSpec` AND THAT IS LOAD-BEARING. `MLAAttentionSpec`
// is not an MLA claim — it is the KEY-ONLY page budget, one vector per stored
// row instead of a K+V pair. A `FullAttentionSpec` here would be absorbed by the
// runner as the single `fa_draft` draft-KV slot instead (`gpu/runner.cpp`, the
// `draft_slot_taken` arm of the leftover scan), `multi_cache_topology` would
// stay false, and the side cache would be published and never allocated — in
// silence.
//
// ITS `compress_ratio` IS 1 AND NOT `indexer_compress_ratio` (W5h). This
// paragraph used to say the ratio "is what makes a state cover four tokens",
// which is what `compress_ratio` MEANS
// (`vllm/v1/kv_cache_interface.py:386`, `:393-395`) and is not what this cache
// STORES. The oracle is unambiguous — transformers 5.16.0
// `models/qwen4_exp/modeling_qwen4_exp.py`, the lane pin recorded in
// `.agents/oracles/transformers.md`:
//
//   :650      `q, raw_keys = q.reshape(*hidden_shape),
//              token_k.reshape(*hidden_shape).squeeze(2)` — one UN-normed,
//              UN-roped key per TOKEN. (W5i re-derived this at the pin: the
//              anchor read `:645-646`, which is inside the `torch.split(` call
//              that FEEDS the line, not the line quoted here.)
//   :654-655  `raw_keys = past_key_values.update_indexer(raw_keys,
//              self.layer_idx)` — and `cache_utils.py:340` calls that "update
//              the indexer key cache by concatenation", returning
//              `[batch_size, total_len, index_head_dim]` (`:346`). Those two are
//              the DOCSTRING; the executing line is
//              `self.indexer_keys = torch.cat([self.indexer_keys,
//              indexer_key_states], dim=1)` (`:350-351`). The static arm returns
//              `[batch_size, max_cache_len, index_head_dim]` (`:672`)
//   :679-682  the POOLED block keys are rebuilt from those raw keys on EVERY
//              step (`raw_keys[batch_idx].index_select(0, ...)` at `:679` then
//              `.float().mean(dim=1)` at `:681` and `k_layernorm` at `:682`).
//              Nothing caches a pooled key. (W5i re-derived this too: the anchor
//              read `:678-681` and `:678` is blank.)
//
// So `indexer_compress_ratio` belongs to the indexer's ALGORITHM — `block_topk =
// token_budget // compress_ratio` (`:622`) and `complete_keys = (kv_len / CR) *
// CR` in `Qwen4ExpQsaIndex` — and never to this cache's page geometry. Our own
// consumer already said so, and since W5i it says it in the ENGINE's own shape:
// `Qwen4ExpQsaPagedCaches::index_key` is the fused MLA page
// `[num_pages, block_size, indexer_head_dim]`, ONE ROW PER TOKEN SLOT, written
// at the physical rows of logical positions `[past_len, past_len + T)` and read
// over the physical rows of `[0, kv_len)`. Both go through `IndexerRows` in
// `qwen4_exp_qsa_block.cpp` — see that helper for the translation, and see
// `Qwen4ExpQsaPagedCaches` for why group 2 carries its own block table.
//
// WHAT THE FOUR COST, AND WHY IT WAS WORSE THAN A SHORT CACHE. The allocation is
// `num_blocks * page_size_bytes()` and `MLAAttentionSpec::real_page_size_bytes`
// takes `storage_block_size()` (`src/vllm/v1/kv_cache_interface.cpp:151-152`),
// while the `PagedKvCache` VIEW the runner hands the forward carries
// `kv.block_size = fa_dims[i].block_size` — the spec's own `block_size`
// (`src/vllm/v1/worker/gpu/runner.cpp:1532` filled at `:1333`). At ratio 4 the
// view claimed 16 rows per page over an allocation of 4, so a consumer that
// trusted the view read four times past the buffer. The two agree exactly when
// `storage_block_size() == block_size`, which is what ratio 1 makes true. It was
// unreachable only because `ModelRegistry::Forward` refuses a multi-cache
// topology before any of it runs.
v1::KVCacheConfig MakeQwen4ExpKVCache(const HfConfig& config, int block_size,
                                      int num_blocks) {
  // The row's own resolve-and-validate, not a second reading of the raw config.
  // It is what rewrites `full_attention` into `qwen_sparse_attention`, so the
  // classification below is upstream's post-`__post_init__` one.
  const Qwen4ExpParams p = ParseQwen4ExpParams(config);

  VT_CHECK(block_size > 0,
           "qwen4_exp KV spec: block_size must be positive, got " +
               std::to_string(block_size));

  std::vector<std::string> qsa_layers;
  std::vector<std::string> qsa_indexer_layers;
  std::vector<std::string> linear_layers;
  // THE SAME THREE BUILDERS THE FORWARD RESOLVES THROUGH. See their definition
  // above for why they exist and why they are file-local.
  for (size_t l = 0; l < p.layer_types.size(); ++l) {
    if (p.layer_types[l] == Qwen4ExpLayerKind::kLinearAttention) {
      linear_layers.push_back(Qwen4ExpLinearAttnName(l));
    } else {
      qsa_layers.push_back(Qwen4ExpQsaAttnName(l));
      qsa_indexer_layers.push_back(Qwen4ExpQsaIndexerName(l));
    }
  }

  VT_CHECK(!qsa_layers.empty(),
           "qwen4_exp KV spec: the config declares no qwen_sparse_attention "
           "layer, so there is no attention KV to publish. See "
           ".agents/specs/qwen4-exp-flash-next.md and issue #2031.");
  VT_CHECK(!linear_layers.empty(),
           "qwen4_exp KV spec: the config declares no linear_attention layer, "
           "so there is no recurrent state to publish. See "
           ".agents/specs/qwen4-exp-flash-next.md and issue #2031.");

  // QSA is optional as a WHOLE in the config layer (all five `indexer_*` fields
  // or none), while the `full_attention` -> `qwen_sparse_attention` rewrite is
  // unconditional. So a config CAN declare sparse layers and no indexer, and
  // that combination has no side cache to size. Refuse rather than publish two
  // groups where the model needs three.
  VT_CHECK(p.qsa.compress_ratio > 0 && p.qsa.head_dim > 0 &&
               p.qsa.kv_heads > 0,
           "qwen4_exp KV spec: the config declares " +
               std::to_string(qsa_layers.size()) +
               " qwen_sparse_attention layer(s) but no `indexer_*` group, so "
               "the QSA indexer side cache cannot be sized. See "
               ".agents/specs/qwen4-exp-flash-next.md and issue #2031.");

  // W5h: THE BLOCK-SIZE DIVISIBILITY REFUSAL IS GONE, BECAUSE THE COMPRESSION IT
  // GUARDED WAS NEVER THIS CACHE'S. It read `block_size % indexer_compress_ratio
  // == 0` and explained that `storage_block_size()` truncates under integer
  // division (`vllm/v1/kv_cache_interface.py:393-395`). Every word of that was
  // true of a COMPRESSED page and this page is not one — see the group-2
  // construction below for the oracle that decides it — so at `compress_ratio`
  // 1 there is no division and nothing to truncate. Deleting a guard needs an
  // argument rather than a green run, and the argument is that its replacement
  // is STRICTLY STRONGER: `test_qwen4_exp_kv_cache.cpp`'s W5h case asserts the
  // side cache's row capacity EQUALS the paged K/V group's token capacity, at
  // both a dividing and a non-dividing block size. The old refusal could only
  // see one arithmetic accident; the capacity law sees any spec that cannot hold
  // what the model stores.

  // The recurrent state set, in the order stated above.
  //
  // The two dtypes come from the SAME resolver every other hybrid in this tree
  // uses, rather than a second reading of `mamba_ssm_dtype`; its refusal
  // message is spelled `qwen3_5:` because that is where the one copy lives.
  const vt::DType conv_dtype = vt::DType::kBF16;
  const vt::DType ssm_dtype =
      detail::ResolveMambaSsmCacheDType(config, conv_dtype);

  std::vector<std::vector<int64_t>> state_shapes{
      // GDN conv: the concatenated q|k|v stream, `conv_kernel - 1` taps.
      {p.linear_conv_dim(), p.linear_conv_kernel_dim - 1},
      // GDN temporal.
      {p.linear_num_value_heads, p.linear_value_head_dim,
       p.linear_key_head_dim},
  };
  std::vector<vt::DType> state_dtypes{conv_dtype, ssm_dtype};

  // `number_of_conv_states` is 3 exactly when the model has a PLE layer, and 1
  // otherwise (`Qwen4ExpParams::number_of_conv_states`, mirroring upstream).
  // The two extra conv states are the PLE conv and the n-gram token history,
  // which upstream keeps in the linear-attention cache beside the GDN conv
  // because the state manipulations are identical (`modular_qwen4_exp.py`
  // :178-180).
  if (p.number_of_conv_states() == 3) {
    // The PLE conv is DILATED by `ngram_size`, so its state is
    // `(kernel - 1) * ngram_size` = 9 columns deep, not `kernel - 1`, and it
    // runs over the FULL hyper-connection stream width.
    state_shapes.push_back(
        {p.stream_width(), p.ple.short_conv_state_len()});
    state_dtypes.push_back(conv_dtype);
    // TOKEN IDS, and `kI64` is not a widening. The history holds
    // `input_ids.long()` and feeds a `uint64_t` hash multiply; storing it in a
    // float dtype rounds a token id, which the row spec records as one of the
    // three silent divergence sites. `ENG-RECURRENT-MULTISTATE` (#2131) is what
    // made an integer recurrent state expressible at all.
    state_shapes.push_back({p.ple.ngram_size - 1});
    state_dtypes.push_back(vt::DType::kI64);
  }

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::move(qsa_layers),
      std::make_shared<v1::FullAttentionSpec>(
          block_size, static_cast<int>(p.num_key_value_heads),
          static_cast<int>(p.head_dim), v1::ResolveKvCacheDType()));
  kv.kv_cache_groups.emplace_back(
      std::move(linear_layers),
      std::make_shared<v1::MambaSpec>(block_size, std::move(state_shapes),
                                      std::move(state_dtypes)));
  kv.kv_cache_groups.emplace_back(
      std::move(qsa_indexer_layers),
      std::make_shared<v1::MLAAttentionSpec>(
          block_size, static_cast<int>(p.qsa.head_dim),
          v1::ResolveKvCacheDType(), static_cast<int>(p.qsa.kv_heads),
          v1::KVQuantMode::kNone, /*page_size_padded=*/std::nullopt,
          /*indexes_kv_by_block_stride=*/false,
          /*cache_dtype_str=*/std::nullopt, /*alignment=*/std::nullopt,
          // ONE ROW PER TOKEN — `compress_ratio` 1, NOT
          // `indexer_compress_ratio`. See the "GROUP 2" paragraph above this
          // function for the oracle lines that decide it.
          /*compress_ratio=*/1,
          /*model_version=*/std::nullopt));
  return kv;
}

const ModelFactory kQwen4ExpFactory{
    .parse_config = &ParseQwen4ExpConfig,
    .load_weights = &LoadQwen4ExpForConditionalGeneration,
    .prepare = &PrepareQwen4ExpForConditionalGeneration,
    .forward = &ForwardQwen4ExpForConditionalGeneration,
    .make_kv_cache = &MakeQwen4ExpKVCache,
    .is_dense_model = false,
    // W5j (#2031, #2353): THE DECLARATION, landed with its first consumer.
    // `ForwardQwen4ExpForConditionalGeneration` resolves every published cache
    // through `MultiKvCacheIndex::Resolve` and reads group 2's own gathered
    // block table, so `ModelRegistry::Forward` lets this architecture's
    // three-group topology past its guard. Setting this on a forward that read
    // `attn_kv` positionally would move a silent mis-index out of the engine and
    // into the model: `attn_kv` carries 2 x n_qsa entries on this topology.
    .consumes_multi_kv = true,
    // W5L (#2031): THE DECLARATION THAT KEEPS A SERVER ALIVE. This forward's
    // first check refuses `num_reqs > 1` (see the `input.num_reqs == 1`
    // VT_CHECK above), and a refusal thrown inside the EngineCore busy loop is
    // FATAL: it does not fail one request, it ends the engine. Measured on
    // `examples/server` at `--max-num-seqs 4` — three overlapping
    // `/v1/completions` calls all returned 500 with this hook's own message and
    // nothing served afterwards. `LoadedEngine::ResolveMaxNumSeqs` reads this
    // bit and clamps the scheduler to one sequence, which is why the same
    // binary at the DEFAULT `max_num_seqs` of 128 now serves those three in
    // sequence instead. It is cleared by the wave that plumbs the ragged batch,
    // not by anything smaller.
    .consumes_device_token_ids = true,
    .serves_one_sequence_per_step = true,
};

}  // namespace

REGISTER_VLLM_MODEL(qwen4_exp, "Qwen4ExpForConditionalGeneration",
                    kQwen4ExpFactory, kQwen4ExpInfo)

}  // namespace vllm
