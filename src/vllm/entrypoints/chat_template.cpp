// Ported from: vllm/entrypoints/chat_utils.py @ e24d1b24 (see chat_template.h
// for the deviation note). vLLM delegates chat templating to transformers'
// full CPython Jinja2 (`apply_chat_template`), which we cannot depend on at
// runtime. This file is the ADAPTER over the vendored google/minja Jinja
// engine (third_party/minja/, the same engine llama.cpp historically vendored):
// minja renders the tokenizer's Jinja `chat_template` with transformers'
// whitespace policy (trim_blocks=True, lstrip_blocks=True, keep_trailing_newline
// =False) and supports the full construct surface real templates use
// (namespace(), macros, filters, is-tests, slicing, ...), not the reduced
// subset the previous hand-written renderer accepted.
#include "vllm/entrypoints/chat_template.h"

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/v1/engine/validation_error.h"  // refused kwarg -> HTTP 400

#include <algorithm>
#include <atomic>
#include <functional>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <minja/minja.hpp>

namespace vllm::entrypoints {
namespace {

std::atomic<std::uint64_t> g_chat_template_parse_count{0};

// The ONE place a chat template reaches minja's parser, with transformers'
// whitespace policy (trim_blocks=True, lstrip_blocks=True,
// keep_trailing_newline=False).
std::shared_ptr<minja::TemplateNode> ParseChatTemplate(
    const std::string& template_str) {
  g_chat_template_parse_count.fetch_add(1, std::memory_order_relaxed);
  return minja::Parser::parse(
      template_str, minja::Options{/*trim_blocks=*/true,
                                   /*lstrip_blocks=*/true,
                                   /*keep_trailing_newline=*/false});
}

nlohmann::ordered_json DecodeHistoricalToolArguments(
    const std::string& encoded, size_t tool_call_index,
    const std::string& function_name) {
  if (encoded.empty()) return nlohmann::ordered_json::object();

  nlohmann::ordered_json decoded;
  try {
    decoded = nlohmann::ordered_json::parse(encoded);
  } catch (const nlohmann::json::parse_error&) {
    // Do not echo arbitrary client-supplied argument contents into server logs.
    throw ChatTemplateError(
        "assistant tool_calls[" + std::to_string(tool_call_index) +
        "].function.arguments for function '" + function_name +
        "' is not valid JSON");
  }
  return decoded.is_null() ? nlohmann::ordered_json::object()
                           : std::move(decoded);
}

// Build the `messages` variable minja exposes to the template, mirroring what
// transformers passes to `apply_chat_template`: a list of {role, content} maps,
// carrying `tool_calls` (OpenAI shape) when a ChatMessage has them. ordered_json
// keeps key order stable so `{{ message | tojson }}`-style branches are
// deterministic.
nlohmann::ordered_json BuildMessages(
    const std::vector<openai::ChatMessage>& messages) {
  nlohmann::ordered_json arr = nlohmann::ordered_json::array();
  for (const openai::ChatMessage& m : messages) {
    nlohmann::ordered_json o = nlohmann::ordered_json::object();
    o["role"] = m.role;
    if (m.content.has_value()) {
      o["content"] = *m.content;
    } else {
      o["content"] = nullptr;
    }
    // Multi-turn tool identity + prior reasoning: exposed only when present so
    // `{% if message.tool_call_id %}`-style template gates stay falsy on plain
    // turns (an always-present null would still be falsy in Jinja, but absent
    // matches transformers' dict shape exactly).
    if (m.tool_call_id.has_value()) o["tool_call_id"] = *m.tool_call_id;
    if (m.name.has_value()) o["name"] = *m.name;
    if (m.reasoning.has_value()) o["reasoning"] = *m.reasoning;
    if (m.tool_calls.has_value()) {
      nlohmann::ordered_json calls = nlohmann::ordered_json::array();
      for (size_t i = 0; i < m.tool_calls->size(); ++i) {
        const openai::ToolCall& tc = (*m.tool_calls)[i];
        nlohmann::ordered_json fn = nlohmann::ordered_json::object();
        fn["name"] = tc.function.name;
        // Keep FunctionCall::arguments as an OpenAI JSON string on the wire,
        // but mirror pinned vLLM's _postprocess_messages contract at the
        // protocol-to-template boundary: historical assistant tool arguments
        // are structured values in Jinja, empty/null become {}, and malformed
        // JSON fails before generation. Non-assistant messages stay strings.
        fn["arguments"] =
            m.role == "assistant"
                ? DecodeHistoricalToolArguments(tc.function.arguments, i,
                                                tc.function.name)
                : nlohmann::ordered_json(tc.function.arguments);
        nlohmann::ordered_json call = nlohmann::ordered_json::object();
        call["id"] = tc.id;
        call["type"] = tc.type;
        call["function"] = std::move(fn);
        calls.push_back(std::move(call));
      }
      o["tool_calls"] = std::move(calls);
    }
    arr.push_back(std::move(o));
  }
  return arr;
}

// Rebuild the OpenAI tool JSON objects the request carried, exposed to the
// template as `tools` exactly as transformers' apply_chat_template(tools=...)
// sees them:
//   {"type": <t>, "function": {"name": .., "description"?: .., "parameters"?: ..}}
nlohmann::ordered_json BuildTools(
    const std::vector<openai::ChatCompletionToolsParam>& tools) {
  nlohmann::ordered_json arr = nlohmann::ordered_json::array();
  for (const openai::ChatCompletionToolsParam& t : tools) {
    nlohmann::ordered_json fn = nlohmann::ordered_json::object();
    fn["name"] = t.function.name;
    if (t.function.description.has_value()) {
      fn["description"] = *t.function.description;
    }
    if (t.function.parameters.has_value()) {
      // parameters is a plain nlohmann::json (unordered). Round-trip through a
      // string to land it in ordered_json without an implicit cross-container
      // conversion.
      fn["parameters"] =
          nlohmann::ordered_json::parse(t.function.parameters->dump());
    }
    nlohmann::ordered_json tool = nlohmann::ordered_json::object();
    tool["type"] = t.type;
    tool["function"] = std::move(fn);
    arr.push_back(std::move(tool));
  }
  return arr;
}

// A template parsed ONCE, or the reason it would not parse. The failure is
// kept, not thrown, because transformers compiles inside apply_chat_template
// and a broken template is therefore a per-request error upstream (api_server
// answers it with 400); holding the message lets every render report exactly
// what a fresh parse would have.
//
// Sharing `root` across the server's HTTP worker threads is safe because a
// minja render never writes to the tree (third_party/minja/minja.hpp):
// TemplateNode::do_render and Expression::do_evaluate are `const` (:865, :656),
// no node or expression has a `mutable` member, and every name a render binds
// goes into the per-call Context or into a Value that Context owns -- SetNode
// :1149 (namespace write) and :1152, ForNode :1010-1013, MacroNode :1106,
// SetTemplateNode :1166, CallNode :1736. LiteralExpr (:1190) holds only the
// scalars parseConstant produces (:1891-1912) and returns them by value, so a
// template cannot `.append()` into a node's own storage; list and dict literals
// build a fresh Value per evaluation (ArrayExpr :1204, DictExpr :1219).
// Context::builtins() builds a new globals object per call (:2818), and the
// only statics are function-local const regexes, initialised once
// thread-safely and only read.
struct ParsedChatTemplate {
  std::shared_ptr<const minja::TemplateNode> root;  // null when parse failed
  std::string parse_error;                          // the parser's what()
};

ParsedChatTemplate ParseChatTemplateKeepingError(
    const std::string& template_str) {
  ParsedChatTemplate parsed;
  try {
    parsed.root = ParseChatTemplate(template_str);
  } catch (const std::exception& e) {
    parsed.parse_error = e.what();
  }
  return parsed;
}

std::string RenderChatTemplate(
    const ParsedChatTemplate& parsed,
    const std::vector<openai::ChatMessage>& messages, bool add_generation_prompt,
    const std::string& bos_token, const std::string& eos_token,
    const std::vector<openai::ChatCompletionToolsParam>& tools,
    const nlohmann::ordered_json& chat_template_kwargs) {
  try {
    // Raised inside the try, ahead of everything a render validates, so the
    // arm below wraps it into the same ChatTemplateError, in the same order
    // against a refused kwarg, that parsing here used to produce.
    if (!parsed.root) throw std::runtime_error(parsed.parse_error);
    const minja::TemplateNode& root = *parsed.root;

    nlohmann::ordered_json top = nlohmann::ordered_json::object();
    top["messages"] = BuildMessages(messages);
    top["add_generation_prompt"] = add_generation_prompt;
    // The ENGINE's own names, held so the kwargs loop below can ask what they
    // are. minja resolves a global, a filter and an is-test through ONE Context
    // chain: Context::make parents the render context on Context::builtins()
    // and set() writes into the CHILD, so any key bound below would shadow all
    // 31 of them (minja.hpp). Passing the same parent explicitly changes
    // nothing about the render; it only makes the set queryable.
    std::shared_ptr<minja::Context> builtins = minja::Context::builtins();
    std::shared_ptr<minja::Context> context =
        minja::Context::make(minja::Value(top), builtins);
    // tojson with jinja2's DEFAULT policy: CPython Jinja sorts object keys
    // (jinja2.defaults.DEFAULT_POLICIES["json.dumps_kwargs"] =
    // {"sort_keys": True}), and that is the tojson every transformers /
    // vLLM-served template runs under -- the Kolibri tool preamble renders
    // `{{ tool | tojson }}` and byte-matches the jinja2 references only with
    // the sort. minja's builtin dumps in insertion order, so this child-scope
    // global shadows it (same signature: value + optional indent; set() on
    // the child cannot touch the shared builtins).
    context->set(
        "tojson",
        minja::Value::callable(
            [](const std::shared_ptr<minja::Context>&,
               minja::ArgumentsValue& args) -> minja::Value {
              minja::Value value = args.args.at(0);
              int64_t indent = -1;
              for (const auto& [name, v] : args.kwargs) {
                if (name == "indent") indent = v.get<int64_t>();
              }
              std::function<minja::Value(minja::Value)> sort_keys =
                  [&](minja::Value v) -> minja::Value {
                if (v.is_object()) {
                  std::vector<std::string> keys;
                  for (const auto& k : v.keys()) keys.push_back(k.get<std::string>());
                  std::sort(keys.begin(), keys.end());
                  minja::Value out = minja::Value::object();
                  for (const auto& k : keys) {
                    out.set(minja::Value(k), sort_keys(v.at(minja::Value(k))));
                  }
                  return out;
                }
                if (v.is_array()) {
                  minja::Value out = minja::Value::array();
                  for (std::size_t i = 0; i < v.size(); ++i) {
                    out.push_back(sort_keys(v.at(minja::Value(static_cast<int64_t>(i)))));
                  }
                  return out;
                }
                return v;
              };
              return minja::Value(sort_keys(value).dump(indent,
                                                       /*to_json=*/true));
            }));
    context->set("bos_token", minja::Value(bos_token));
    context->set("eos_token", minja::Value(eos_token));
    context->set("tools", minja::Value(BuildTools(tools)));
    // The caller's chat_template_kwargs, and ONLY those. A key nobody supplied
    // is left unbound, so `{% if enable_thinking is undefined %}` sees what
    // transformers shows it (vllm/renderers/hf.py:777-783 forwards the resolved
    // kwargs and nothing else). Binding `enable_thinking` unconditionally --
    // which this function used to do -- makes that test permanently false and
    // silently flips a model's own reasoning default (#1681).
    if (chat_template_kwargs.is_object()) {
      for (auto it = chat_template_kwargs.begin();
           it != chat_template_kwargs.end(); ++it) {
        const std::string& key = it.key();
        // (1) apply_chat_template's OWN parameters. resolve_chat_template_kwargs
        // RAISES on them before anything renders, rather than dropping them
        // (vllm/renderers/hf.py:639-648 @ 555967922; `raise_on_unexpected`
        // defaults to True and its only call site takes the default,
        // hf.py:731-735).
        if (key == "chat_template" || key == "tokenize") {
          throw vllm::v1::InputValidationError(
              "Found unexpected chat template kwargs from request: {'" + key +
              "'}");
        }
        // (2) The names the RENDERER supplies. resolve_chat_template_kwargs
        // KEEPS these two -- `messages` is in
        // jinja2.meta.find_undeclared_variables of every real chat template and
        // `tools` is an apply_chat_template parameter as well -- and
        // transformers then dies on the duplicate keyword before it renders:
        //   tokenizer.apply_chat_template(conversation=..., tools=tools,
        //       chat_template=..., tokenize=..., **resolved_kwargs)
        //     -> TypeError: got multiple values for keyword argument 'tools'
        //   compiled_template.render(messages=chat, tools=..., **kwargs)
        //     -> TypeError: got multiple values for keyword argument 'messages'
        //   (transformers 5.3.0, utils/chat_template_utils.py)
        // Measured on the pin against tests/fixtures/qwen38_chat_template.jinja.
        // So upstream has NO path on which a request replaces the conversation.
        // Binding them here did have one, and it was silent: the request log
        // line, `usage`, `ToolsEnabled` and any policy layer reading
        // request.messages all described a conversation the model never got.
        // InputValidationError, not ChatTemplateError, because this is a CLIENT
        // mistake: api_server maps it to 400 exactly as upstream's ValueError /
        // TypeError reach create_error_response's BadRequestError default
        // (serve/utils/error_response.py:16-21), and the C ABI maps it to
        // VLLM_ERR_INVALID_ARGUMENT.
        if (key == "messages" || key == "tools") {
          throw vllm::v1::InputValidationError(
              "chat template kwargs from request may not set '" + key +
              "': the renderer supplies it");
        }
        // (3) `add_generation_prompt` and `continue_final_message` are the two
        // renderer-owned names upstream neither raises on nor honours.
        // build_chat_params puts the request's OWN field of each name in
        // `extra_kwargs`, the OVERRIDE side of merge_kwargs, so the field has
        // already replaced the kwarg before resolve_chat_template_kwargs ever
        // sees it (vllm/entrypoints/openai/chat_completion/protocol.py:530-544,
        //  merge_kwargs at vllm/renderers/params.py:28-40). This function's
        // `add_generation_prompt` parameter IS that field; the chat path has no
        // `continue_final_message` field yet, and binding the kwarg would show
        // a template a value upstream can never show it.
        if (key == "add_generation_prompt" || key == "continue_final_message") {
          continue;
        }
        // (4) A name minja's own builtins layer supplies. CPython jinja2
        // resolves a filter through env.filters, a test through env.tests and a
        // global through env.globals, and none of those is the variable
        // namespace jinja2.meta.find_undeclared_variables reports on -- so
        // upstream's accept_vars can never keep one, and it renders 200 for a
        // request that sends one. minja has a single namespace, so binding it
        // shadowed the engine: `{"namespace": 1}` broke line 1 of the shipped
        // Qwen3.8 template as a client-triggerable 500.
        //
        // `raise_exception` is the one exception, and it is upstream's.
        // transformers adds it to the environment AFTER
        // _resolve_chat_template_kwargs has parsed with a fresh env of its own
        // (hf.py:598-606), so jinja2 reports it undeclared, accept_vars keeps
        // it, and the request value shadows the global at render. Measured on
        // jinja2 3.1.2 + transformers 5.3.0 over
        // tests/fixtures/qwen38_chat_template.jinja: of minja's 31 builtins,
        // `template_vars | hf_base_params` keeps `raise_exception` and nothing
        // else.
        //
        // One residual, one-sided and deliberate. jinja2's filter and test
        // namespaces are separate from its variable namespace, so a template MAY
        // read `{{ items }}` as an ordinary variable while `| items` still
        // resolves as a filter, and upstream then keeps that kwarg. minja has
        // one namespace and cannot hold both, so this keeps the built-in: a
        // dropped kwarg renders a working template, and the other choice answers
        // 500 for every template that uses the filter.
        if (key != "raise_exception" && builtins->contains(minja::Value(key))) {
          continue;
        }
        // Everything else binds, `bos_token` / `eos_token` included, and that
        // matches upstream in both directions: a template that NAMES either has
        // it in find_undeclared_variables, so the request value survives the
        // filter and transformers lets it win over the tokenizer's special
        // tokens (`template_kwargs = {**self.special_tokens_map, **kwargs}`,
        // PythonBackend.apply_chat_template); a template that names neither
        // drops it upstream and cannot observe it here.
        //
        // A name the template never uses is likewise unobservable, which is why
        // this mirrors upstream's accept_vars filter by REFUSING the adapter's
        // names and SKIPPING the engine's, rather than reproducing
        // find_undeclared_variables: minja exposes no AST walk, and for every
        // remaining name "bound but never read" and "dropped" render the same
        // bytes.
        context->set(key, minja::Value(it.value()));
      }
    }
    const auto now = std::chrono::system_clock::now();
    context->set(
        "strftime_now",
        minja::Value::callable([now](const std::shared_ptr<minja::Context>&,
                                     minja::ArgumentsValue& args) {
          args.expectArgs("strftime_now", {1, 1}, {0, 0});
          const auto format = args.args[0].get<std::string>();
          const std::time_t t = std::chrono::system_clock::to_time_t(now);
          std::tm local_time{};
#if defined(_WIN32)
          localtime_s(&local_time, &t);
#else
          localtime_r(&t, &local_time);
#endif
          std::ostringstream ss;
          ss << std::put_time(&local_time, format.c_str());
          return minja::Value(ss.str());
        }));

    return root.render(context);
  } catch (const ChatTemplateError&) {
    throw;
  } catch (const vllm::v1::InputValidationError&) {
    // A refused kwarg is a CLIENT error, not a render failure. Rethrown before
    // the generic arm so it stays a 400 / VLLM_ERR_INVALID_ARGUMENT instead of
    // being rewrapped as a ChatTemplateError the server reports as a 500.
    throw;
  } catch (const std::exception& e) {
    throw ChatTemplateError(std::string("chat template render failed: ") +
                            e.what());
  }
}

}  // namespace

// ─── Public API ──────────────────────────────────────────────────────────────
std::string apply_chat_template(
    const std::string& template_str,
    const std::vector<openai::ChatMessage>& messages, bool add_generation_prompt,
    const std::string& bos_token, const std::string& eos_token,
    const std::vector<openai::ChatCompletionToolsParam>& tools,
    const nlohmann::ordered_json& chat_template_kwargs) {
  return RenderChatTemplate(ParseChatTemplateKeepingError(template_str),
                            messages, add_generation_prompt, bos_token,
                            eos_token, tools, chat_template_kwargs);
}

std::uint64_t ChatTemplateParseCountForTesting() {
  return g_chat_template_parse_count.load(std::memory_order_relaxed);
}

openai::ChatPromptFn MakeChatTemplatePromptFn(
    std::string template_str, std::string bos_token, std::string eos_token,
    nlohmann::ordered_json default_chat_template_kwargs) {
  // Parsed here, once per prompt fn, and shared by every request the fn
  // renders -- including concurrent ones (see ParsedChatTemplate). Upstream
  // compiles once as well: transformers' _compile_jinja_template is
  // lru_cache'd, and vLLM resolves the template once per model
  // (vllm/renderers/hf.py). Re-parsing per request cost 45 ms of an 8952-char
  // Qwen3.8 template against 0.1 ms to render it
  // (ISSUE-LOCAL-01M2EAQ6R63BSRF1GVZ3JAR4A4).
  auto parsed = std::make_shared<const ParsedChatTemplate>(
      ParseChatTemplateKeepingError(template_str));
  return [parsed = std::move(parsed), bos = std::move(bos_token),
          eos = std::move(eos_token),
          defaults = std::move(default_chat_template_kwargs)](
             const std::vector<openai::ChatMessage>& messages,
             bool add_generation_prompt,
             const std::vector<openai::ChatCompletionToolsParam>& tools,
             const nlohmann::ordered_json& request_kwargs) {
    // merge_kwargs (vllm/renderers/params.py:28-40 @ 555967922), reached as
    // ChatParams.with_defaults(default_chat_template_kwargs) (params.py:93-122)
    // from vllm/entrypoints/openai/chat_completion/serving.py:208:
    //   defaults | {k: v for k, v in overrides.items()
    //               if v not in unset_values}      unset_values = (None, "auto")
    // A shallow merge in which the request's keys win over the server defaults,
    // EXCEPT that an override valued null or "auto" means "the client did not
    // set this" and leaves the server default standing. Without that exception
    // a request null defeated `--no-enable-thinking`.
    // (`multimodal/media/base.py:53-67`, cited here before the #1681 review, is
    //  MediaIO.merge_kwargs -- the media-io path, not this one.)
    nlohmann::ordered_json merged =
        defaults.is_object() ? defaults : nlohmann::ordered_json::object();
    if (request_kwargs.is_object()) {
      for (auto it = request_kwargs.begin(); it != request_kwargs.end(); ++it) {
        if (it.value().is_null()) continue;
        if (it.value().is_string() && it.value().get<std::string>() == "auto") {
          continue;
        }
        merged[it.key()] = it.value();
      }
    }
    return RenderChatTemplate(*parsed, messages, add_generation_prompt, bos,
                              eos, tools, merged);
  };
}

nlohmann::ordered_json DefaultChatTemplateKwargs(
    std::optional<bool> enable_thinking) {
  nlohmann::ordered_json kwargs = nlohmann::ordered_json::object();
  if (enable_thinking.has_value()) kwargs["enable_thinking"] = *enable_thinking;
  return kwargs;
}

std::string LoadChatTemplateFromConfig(
    const std::string& tokenizer_config_path) {
  std::ifstream f(tokenizer_config_path, std::ios::binary);
  if (!f) {
    throw ChatTemplateError("cannot open tokenizer_config.json: " +
                            tokenizer_config_path);
  }
  nlohmann::json doc;
  try {
    f >> doc;
  } catch (const std::exception& e) {
    throw ChatTemplateError(std::string("failed to parse tokenizer_config.json: ") +
                            e.what());
  }
  auto it = doc.find("chat_template");
  if (it != doc.end() && !it->is_null()) {
    if (it->is_string()) return it->get<std::string>();
    // List-of-{name,template} form: pick "default", else the first.
    if (it->is_array()) {
      const nlohmann::json* chosen = nullptr;
      for (const auto& entry : *it) {
        if (entry.is_object() && entry.value("name", std::string()) == "default") {
          chosen = &entry;
          break;
        }
      }
      if (!chosen && !it->empty()) chosen = &it->front();
      if (chosen && chosen->contains("template") &&
          (*chosen)["template"].is_string()) {
        return (*chosen)["template"].get<std::string>();
      }
    }
    throw ChatTemplateError("unrecognized 'chat_template' shape in " +
                            tokenizer_config_path);
  }

  // Sibling chat_template.jinja (HF layout for Gemma4 / many multimodal models).
  std::string dir = tokenizer_config_path;
  const auto slash = dir.find_last_of("/\\");
  if (slash != std::string::npos) dir.resize(slash + 1);
  else dir.clear();
  const std::string jinja_path = dir + "chat_template.jinja";
  std::ifstream jf(jinja_path, std::ios::binary);
  if (jf) {
    std::ostringstream ss;
    ss << jf.rdbuf();
    std::string tmpl = ss.str();
    if (!tmpl.empty()) return tmpl;
  }

  throw ChatTemplateError("tokenizer_config.json has no 'chat_template' and no "
                          "sibling chat_template.jinja: " +
                          tokenizer_config_path);
}

std::string LoadChatTemplateFromGguf(const std::string& gguf_path) {
  try {
    const vllm::GgufFile gguf = vllm::GgufFile::Open(gguf_path);
    const vllm::GgufValue* kv = gguf.FindKv("tokenizer.chat_template");
    if (kv == nullptr) {
      throw ChatTemplateError("gguf has no 'tokenizer.chat_template': " +
                              gguf_path);
    }
    const std::string* tmpl = std::get_if<std::string>(&kv->v);
    if (tmpl == nullptr || tmpl->empty()) {
      throw ChatTemplateError(
          "gguf 'tokenizer.chat_template' is not a non-empty string: " +
          gguf_path);
    }
    return *tmpl;
  } catch (const ChatTemplateError&) {
    throw;
  } catch (const std::exception& e) {
    // GgufFile::Open throws std::runtime_error on any malformation.
    throw ChatTemplateError(std::string("cannot read gguf chat template: ") +
                            e.what());
  }
}

std::string LoadChatTemplateForModel(const std::string& tokenizer_config_path,
                                     const std::string& model_dir,
                                     std::string& source) {
  namespace fs = std::filesystem;
  // A .gguf model has no tokenizer_config.json — its chat template lives in
  // the GGUF metadata under tokenizer.chat_template. Try the config path first
  // (covers safetensors dirs and --tokenizer-config overrides), then fall back
  // to the GGUF itself before giving up on the template.
  try {
    std::string tmpl =
        LoadChatTemplateFromConfig(tokenizer_config_path);
    source = tokenizer_config_path + " or sibling chat_template.jinja";
    return tmpl;
  } catch (const ChatTemplateError&) {
    const fs::path dir(model_dir);
    if (fs::is_regular_file(dir) && dir.extension() == ".gguf") {
      std::string tmpl = LoadChatTemplateFromGguf(model_dir);
      source = "GGUF metadata (tokenizer.chat_template) in " + model_dir;
      return tmpl;
    }
    throw;
  }
}

}  // namespace vllm::entrypoints
