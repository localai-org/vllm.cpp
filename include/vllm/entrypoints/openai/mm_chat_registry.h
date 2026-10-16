// The per-ARCHITECTURE multimodal chat seam registry (ENG-MM-INPUT-PIPELINE,
// issue #2475).
//
// Ported from: vllm/multimodal/registry.py @ 5559679229bc — the mechanism vLLM
// uses to resolve a multimodal processor for a model:
//   * :81-95   `_ProcessorFactories` + `build_processor`, the per-model record;
//   * :142-172 `register_processor`, the decorator that attaches that record TO
//              THE MODEL CLASS, i.e. keys it on the architecture;
//   * :176-186 `_get_model_cls`, which resolves
//              `get_model_architecture(model_config)` and RAISES BY NAME —
//              "Model class {name} has no registered multimodal processor" —
//              rather than substituting another model's factory;
//   * :211-230 `create_processor`, the dispatch;
//   * :103-139 `supports_multimodal_inputs`, the predicate: the model is
//              multimodal AND a processor is registered for it. Its
//              missing-processor arm logs `warning_once` naming the model and
//              runs text-only.
// Qwen3-VL's own registration sits on the model in one place and edits no shared
// table (`qwen3_vl.py:1673-1678`).
//
// WHAT THIS REPLACES. The server used to decide whether it could serve images by
// asking whether `<model_dir>/preprocessor_config.json` existed, and then
// constructed `Qwen3VLImageProcessor` + `MakeQwen3VLImageChatFn`
// unconditionally: no branch on the architecture anywhere in the block.
// `preprocessor_config.json` is not a Qwen3-VL marker, so a second multimodal
// checkpoint got Qwen3-VL's processor read against its own `vision_config`
// shape, or threw into a swallowed `catch` and left the server silently on the
// text-only path. A multimodal model that answers an image request as though it
// were text looks like a working server, which is why the silent arm is the
// worse one (#2475).
//
// WHY HERE AND NOT ON `ModelFactory`. `ModelFactory` already carries this
// model's mm-FORWARD hooks (`encode_mm` / `embed_mm`,
// model_registry.h:761-767), so a chat-seam field beside them is the obvious
// mirror. It is the wrong layer: the artefact a chat factory produces is
// `MultiModalChatFn`, and building one needs `ChatPromptRenderFn`,
// `ImageCodecFn`, `ChatMessage` and `BaseProcessingInfo`, all of which live in
// `entrypoints/openai`. Putting them on `ModelFactory` makes `model_executor`
// depend on `entrypoints`. Upstream has the same split for the same reason: the
// registered artefact is a `BaseMultiModalProcessor` from `vllm/multimodal/`,
// and `serving_chat.py` carries no per-model registration at all. So the
// registry lives beside what it produces and is keyed on the same architecture
// string `ModelRegistry` resolves.
//
// The self-registration idiom is `REGISTER_VLLM_MODEL`'s
// (model_registry.h:939-976), deliberately: a TU-static registrar whose
// constructor registers, reached through the `vllm` static library's INTERFACE
// `--whole-archive` (CMakeLists.txt:1446-1453). Adding a model is a new
// translation unit with one line and ZERO edits to a shared array.
#ifndef VLLM_ENTRYPOINTS_OPENAI_MM_CHAT_REGISTRY_H_
#define VLLM_ENTRYPOINTS_OPENAI_MM_CHAT_REGISTRY_H_

#include <iosfwd>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/config/multimodal.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/entrypoints/openai/chat_mm.h"
#include "vllm/entrypoints/openai/serving_chat.h"

namespace vllm {
namespace tok {
class Tokenizer;  // vllm/tokenizer/tokenizer.h
}  // namespace tok
}  // namespace vllm

namespace vllm::entrypoints::openai {

// Everything an architecture's factory is given. This is our
// `InputProcessingContext` (registry.py:188-195: the model config plus the
// tokenizer) with the two serving collaborators upstream's processor does not
// need, because upstream renders the chat template and decodes images
// elsewhere.
struct MultiModalChatContext {
  // The architecture the model registry resolved, and the registry key. OWNED,
  // like every other string here: the two current callers pass static-lifetime
  // data (`LoadedEngine::architecture()` points into the registration, a test
  // passes a literal), but a `string_view` field on a struct a caller fills in
  // field by field dangles the moment somebody assigns a temporary to it, and
  // the copy costs one allocation once per server start.
  std::string architecture;
  // The checkpoint directory. A factory reads its own processor config out of
  // it BY NAME; the directory is never the thing that decides whether the seam
  // is installed at all.
  std::string model_dir;
  // `<model_dir>/config.json`, resolved by the caller because the server
  // already holds it.
  std::string config_path;
  std::string served_model_name;
  // Borrowed for the process lifetime, exactly like the seam that consumes it.
  const vllm::tok::Tokenizer* tokenizer = nullptr;
  // The server's real chat-template renderer.
  ChatPromptRenderFn prompt_fn;
  // The server's image codec. It lives on the CONTEXT and not inside a factory
  // on purpose: the raw-RGB-only limit is a NAMED MM-SERVE residual belonging to
  // the server, not to an architecture, and every future factory consumes the
  // one codec the server supplies rather than growing its own.
  ImageCodecFn codec;
  // Where `--limit-mm-per-prompt` / `--language-model-only` landed. Borrowed:
  // `BaseProcessingInfo` holds it by reference (context.h:105).
  const vllm::MultiModalConfig* mm_config = nullptr;
  // The engine's RESOLVED model config, borrowed. This is the half of
  // upstream's `InputProcessingContext` (registry.py:188-195) that
  // `config_path` above cannot supply: a `.gguf` checkpoint has NO
  // `config.json` beside it, and its `vocab_size`, `hidden_size` and the rest
  // are read out of the file's own metadata by the loader. A factory whose
  // processor is keyed on a model number -- DeepSeek-V4 spells every image
  // position `vocab_size + type`, so a wrong one puts the sentinels INSIDE the
  // vocabulary -- reads it here rather than guessing a default.
  // Null for a caller that has none; a factory that needs it refuses by name.
  const vllm::HfConfig* config = nullptr;
  // The `--mmproj` SECOND FILE the engine was given (`EngineParams::
  // mmproj_path`), empty when none. It is the only thing at install time that
  // says whether a two-file vehicle actually arrived with its vision half:
  // `DeepseekV4ForCausalLM` names both the TEXT checkpoint and the Flash-Vision
  // one, so the architecture cannot answer it and neither can `config.json`.
  //
  // WHY THE ANSWER IS NEEDED HERE AND NOT IN THE ENGINE. A tower-free load
  // refuses inside `encode_mm`, which runs in the engine's busy loop; throwing
  // there stops `AsyncLLM` and turns every LATER request, TEXT ONES INCLUDED,
  // into a 500. That was measured on the dots3-note row before its seam asked
  // the same question at install (`mm_chat_dots3note.cpp`). Refusing here
  // installs a REFUSING seam instead: HTTP 400 naming the architecture, text
  // path untouched.
  std::string mmproj_path;
  // The ENGINE's resolved max_model_len, 0 when the installer did not supply
  // it. A processor whose token budget is defined against it reads it here:
  // Nemotron Nano VL's dynamic tiler sizes every image from
  // `max_model_len - text_len - 4` (processors/nano_nemotron_vl.py:330-331).
  int64_t max_model_len = 0;
  // Resolved execution device. Native device-specific processors refuse at
  // install when it is absent or unsupported, before entering the busy loop.
  std::optional<vt::Device> device;
};

// What a factory returns. `chat_fn` OWNS whatever processor state it needs —
// the factory captures it — so an architecture cannot get its lifetime wrong by
// forgetting to add a local to `main`.
struct MultiModalChatSeam {
  MultiModalChatFn chat_fn;
  // `BaseProcessingInfo::AllowedMmLimits()` for this seam, so the server can
  // print the per-modality request ceiling without knowing the architecture.
  std::map<std::string, int> allowed_limits;
  // One line naming what was installed and from where, for the startup log.
  std::string detail;
  // Optional additive request-aware seam. Factories choose exactly one of the
  // two callback forms; the installer owns and forwards the request context.
  MultiModalRequestChatFn request_chat_fn;
};

// A plain function pointer, exactly like every hook on `ModelFactory`.
using MultiModalChatFactoryFn =
    MultiModalChatSeam (*)(const MultiModalChatContext&);

struct MultiModalChatRegistration {
  std::string_view architecture;
  MultiModalChatFactoryFn make_seam = nullptr;
};

// Copies the registration into the process-global registry. Its members point at
// TU-static data with static lifetime, so the copy stays valid.
void RegisterMultiModalChat(const MultiModalChatRegistration& registration);

// Static-init helper; used only through REGISTER_VLLM_MM_CHAT.
struct MultiModalChatRegistrar {
  explicit MultiModalChatRegistrar(
      const MultiModalChatRegistration& registration) {
    RegisterMultiModalChat(registration);
  }
};

// Registers one architecture's multimodal chat factory from its own translation
// unit. Place at namespace scope inside `namespace vllm::entrypoints::openai`;
// `unique_tag` is any TU-unique token.
#define REGISTER_VLLM_MM_CHAT(unique_tag, architecture_name, factory_fn)      \
  namespace {                                                                 \
  const ::vllm::entrypoints::openai::MultiModalChatRegistrar                  \
      vllm_mm_chat_registrar_##unique_tag(                                    \
          ::vllm::entrypoints::openai::MultiModalChatRegistration{            \
              (architecture_name), (factory_fn)});                            \
  } /* namespace */

class MultiModalChatRegistry {
 public:
  // Sorted by architecture on first query, because registration order across
  // translation units is unspecified under C++ static init (the same guarantee
  // model_registry.cpp gives).
  static std::span<const MultiModalChatRegistration> Registrations();
  static std::vector<std::string_view> SupportedArchs();

  // The registration for `architecture`, or null. Null is the ONLY answer for an
  // unregistered architecture: there is no default seam and no first-entry
  // fallback, which is the whole of #2475.
  static const MultiModalChatRegistration* Find(std::string_view architecture);

  // Mirrors `_get_model_cls`'s raise (registry.py:182-185): names the
  // architecture and the missing part, and lists what IS registered.
  [[noreturn]] static void RaiseForUnregistered(std::string_view architecture);

  // Mirrors `create_processor` (registry.py:211-230): resolve by architecture,
  // refuse by name, then build. Throws `std::runtime_error` when nothing is
  // registered for `ctx.architecture`; propagates whatever the factory throws;
  // and throws when a registered factory RETURNS a seam whose `chat_fn` is
  // empty. That last arm is the symmetric half of the null-`make_seam` check:
  // an empty `std::function` installs as nothing, so the model would answer
  // image requests from the text path under a "seam wired" success log. Every
  // seam this returns therefore has a callable `chat_fn`.
  static MultiModalChatSeam MakeSeam(const MultiModalChatContext& ctx);
};

// A chat function that REFUSES every multimodal request, naming the architecture
// and why. Throws `vllm::v1::InputValidationError`, which api_server.cpp:357-360
// maps to HTTP 400 — the same mapping the `--language-model-only` refusal lands
// on, because the condition is the same class of answer: this server does not
// accept images for this model. Returns nullopt for a request carrying no
// multimodal part, so the text path is untouched.
MultiModalChatFn MakeRefusingMultiModalChatFn(std::string architecture,
                                              std::string reason);

// What `InstallMultiModalChatSeam` did. Mirrors the three outcomes upstream
// spreads across `supports_multimodal_inputs` and `create_processor`.
enum class MultiModalChatInstall {
  // The architecture declares no multimodal support. NOTHING is installed and
  // the chat path stays byte-identical to the text-only server
  // (registry.py:110-111).
  kTextOnlyModel,
  // The architecture's registered factory built a seam and it is installed.
  kInstalled,
  // The architecture declares multimodal support but has no registered factory,
  // or its factory threw, or its factory returned a seam with no callable
  // `chat_fn`. A REFUSING seam is installed: an mm request gets HTTP 400 naming
  // the architecture, and never a silent text answer. The third arm matters as
  // much as the first two: an empty `chat_fn` installs as NOTHING, so without it
  // a success log would sit over the text path answering image requests.
  kRefusing,
};

// THE PRODUCTION INSTALL. `server_main.cpp` and the reachability gate call this
// one function; nothing else may call `set_multimodal_chat_fn` for a model.
//
// `is_multimodal_model` is the architecture's own declaration —
// `ModelInfo::supports_multimodal`, reachable as
// `LoadedEngine::is_multimodal_model()` — and mirrors
// `model_config.is_multimodal_model` (registry.py:110). It is NOT the presence
// of a file.
//
// Every outcome is announced on `log`. The previous code caught the factory's
// exception, printed one line, and left the seam unset; that silence is the
// defect this replaces, so there is no arm here that installs nothing on a
// model that says it is multimodal.
MultiModalChatInstall InstallMultiModalChatSeam(
    OpenAIServingChat& chat, bool is_multimodal_model,
    const MultiModalChatContext& ctx, std::ostream& log);

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_MM_CHAT_REGISTRY_H_
