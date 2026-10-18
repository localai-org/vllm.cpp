// Unit tests for the minja-subset chat-template renderer (M3.1 Task 3 / M3.2).
//
// The KEY oracle (cases below marked "GOLDEN") is a transformers
// `apply_chat_template` dump of the pinned Qwen3 chat_template
// (models--Qwen--Qwen3-30B-A3B tokenizer_config.json — the Qwen3-family
// template the gate model Qwen3.6 shares) for a no-tools/no-thinking
// conversation. For that path the full upstream template is byte-identical to
// the subset template exercised here; the expected strings were captured with:
//   AutoTokenizer.from_pretrained(qwen3).apply_chat_template(
//       msgs, tokenize=False, add_generation_prompt=...)
// and re-verified to render identically from the subset template via jinja2
// configured exactly as transformers does (trim_blocks=True, lstrip_blocks=True).
#include "vllm/entrypoints/chat_template.h"

#include "../gguf_builder.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/v1/engine/validation_error.h"

using vllm::entrypoints::apply_chat_template;
using vllm::entrypoints::ChatTemplateError;
using vllm::entrypoints::MakeChatTemplatePromptFn;
using vllm::entrypoints::openai::ChatCompletionToolsParam;
using vllm::entrypoints::openai::ChatMessage;

namespace {

// The minja-subset Qwen chat template. Uses only supported constructs (for / if
// / interpolation / `+` concat / member access / `-` whitespace trim) and
// reproduces the full upstream Qwen3 template's output on the no-tools path.
const char* kQwenTemplate =
    "{%- for message in messages %}\n"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' "
    "+ '\\n' }}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "{{- '<|im_start|>assistant\\n' }}\n"
    "{%- endif %}";

std::vector<ChatMessage> SystemUser() {
  return {ChatMessage{"system", std::string("You are a helpful assistant.")},
          ChatMessage{"user", std::string("Hello, who are you?")}};
}

std::vector<ChatMessage> MultiTurn() {
  return {ChatMessage{"system", std::string("You are a helpful assistant.")},
          ChatMessage{"user", std::string("Hello, who are you?")},
          ChatMessage{"assistant", std::string("I am Qwen.")},
          ChatMessage{"user", std::string("What can you do?")}};
}

}  // namespace

// ─── (b) GOLDEN: [system,user], add_generation_prompt=true ───────────────────
TEST_CASE("chat_template: Qwen3 [system,user] with generation prompt (GOLDEN)") {
  const std::string expected =
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\nHello, who are you?<|im_end|>\n"
      "<|im_start|>assistant\n";
  CHECK(apply_chat_template(kQwenTemplate, SystemUser(),
                            /*add_generation_prompt=*/true) == expected);
}

// ─── (d) add_generation_prompt=false drops the assistant header ──────────────
TEST_CASE("chat_template: Qwen3 [system,user] without generation prompt (GOLDEN)") {
  const std::string expected =
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\nHello, who are you?<|im_end|>\n";
  CHECK(apply_chat_template(kQwenTemplate, SystemUser(),
                            /*add_generation_prompt=*/false) == expected);
}

// ─── (c) GOLDEN: multi-turn [system,user,assistant,user] ─────────────────────
TEST_CASE("chat_template: Qwen3 multi-turn conversation (GOLDEN)") {
  const std::string expected =
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\nHello, who are you?<|im_end|>\n"
      "<|im_start|>assistant\nI am Qwen.<|im_end|>\n"
      "<|im_start|>user\nWhat can you do?<|im_end|>\n"
      "<|im_start|>assistant\n";
  CHECK(apply_chat_template(kQwenTemplate, MultiTurn(),
                            /*add_generation_prompt=*/true) == expected);
}

// ─── ChatPromptFn adapter drives the same render through Task 2's seam ───────
TEST_CASE("chat_template: MakeChatTemplatePromptFn adapts to the ChatPromptFn seam") {
  auto fn = MakeChatTemplatePromptFn(kQwenTemplate);
  const std::string expected =
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\nHello, who are you?<|im_end|>\n"
      "<|im_start|>assistant\n";
  CHECK(fn(SystemUser(), /*add_generation_prompt=*/true, /*tools=*/{},
           /*chat_template_kwargs=*/nlohmann::ordered_json::object()) ==
        expected);
}

// ─── (a) interpolation + `+` concat ──────────────────────────────────────────
TEST_CASE("chat_template: interpolation and string concat") {
  CHECK(apply_chat_template("{{ 'a' + 'b' }}", {}, false) == "ab");
  CHECK(apply_chat_template("{{ 'x' ~ 'y' ~ 'z' }}", {}, false) == "xyz");
  CHECK(apply_chat_template("[{{ bos_token }}]", {}, false, "<s>", "</s>") ==
        "[<s>]");
}

// ─── (a) for loop + member access + loop vars ────────────────────────────────
TEST_CASE("chat_template: for loop with loop.first/last/index0") {
  const std::string t =
      "{% for m in messages %}{{ loop.index0 }}:{{ m.role }}"
      "{% if not loop.last %},{% endif %}{% endfor %}";
  CHECK(apply_chat_template(t, SystemUser(), false) == "0:system,1:user");
}

// ─── (a) if / elif / else ────────────────────────────────────────────────────
TEST_CASE("chat_template: if / elif / else branches") {
  const std::string t =
      "{% for m in messages %}"
      "{% if m.role == 'system' %}S{% elif m.role == 'user' %}U"
      "{% else %}?{% endif %}"
      "{% endfor %}";
  std::vector<ChatMessage> msgs = {
      ChatMessage{"system", std::string("")}, ChatMessage{"user", std::string("")},
      ChatMessage{"assistant", std::string("")}};
  CHECK(apply_chat_template(t, msgs, false) == "SU?");
}

// ─── (a) membership `in` / `not in` ──────────────────────────────────────────
TEST_CASE("chat_template: membership tests") {
  CHECK(apply_chat_template("{% if 'im' in 'system' %}Y{% else %}N{% endif %}",
                            {}, false) == "N");
  CHECK(apply_chat_template("{% if 'sys' in 'system' %}Y{% else %}N{% endif %}",
                            {}, false) == "Y");
  CHECK(apply_chat_template(
            "{% if 'x' not in 'system' %}Y{% else %}N{% endif %}", {}, false) ==
        "Y");
}

// ─── (a) set + strip ─────────────────────────────────────────────────────────
TEST_CASE("chat_template: set binding and .strip()") {
  CHECK(apply_chat_template("{% set g = 'hi' %}{{ g }}!", {}, false) == "hi!");
  CHECK(apply_chat_template("{{ '  pad  '.strip() }}", {}, false) == "pad");
  CHECK(apply_chat_template("{{ '\\n\\nx\\n'.strip('\\n') }}", {}, false) == "x");
}

// ─── (a) whitespace control: `-` trim markers ────────────────────────────────
TEST_CASE("chat_template: whitespace trim control") {
  // No trim: the surrounding newlines/spaces are preserved.
  CHECK(apply_chat_template("A\n  {{ 'x' }}  \nB", {}, false) == "A\n  x  \nB");
  // Left+right trim collapses the surrounding whitespace.
  CHECK(apply_chat_template("A\n  {{- 'x' -}}  \nB", {}, false) == "AxB");
  // Block-tag trim_blocks/lstrip_blocks: the leading line-ws + trailing newline
  // around a block tag are removed even without explicit `-`.
  CHECK(apply_chat_template("A\n  {% set z = '1' %}\nB", {}, false) == "A\nB");
}

// ─── (e) a malformed template throws ChatTemplateError ───────────────────────
// The engine is now the full google/minja renderer, so constructs the old
// hand-written subset rejected (filters, macros, namespace(), is-tests, slicing)
// render fine (see the "minja engine renders" case below). Only a genuinely
// malformed template (a parse/eval error) still throws, and it is surfaced as a
// ChatTemplateError so the capi probe-and-fallback safety net keeps working.
TEST_CASE("chat_template: malformed templates throw ChatTemplateError") {
  // Unterminated tag.
  CHECK_THROWS_AS(apply_chat_template("{{ 'x'", {}, false), ChatTemplateError);
  // Unbalanced endfor.
  CHECK_THROWS_AS(apply_chat_template("{% endfor %}", {}, false),
                  ChatTemplateError);
  // Unterminated block body.
  CHECK_THROWS_AS(apply_chat_template("{% for m in messages %}", {}, false),
                  ChatTemplateError);
}

// ─── the vendored minja engine renders constructs the old subset rejected ────
// namespace()/macros/filters are exactly what the real Qwen3.5 template needs;
// the previous subset renderer threw on them. Assert they now render.
TEST_CASE("chat_template: minja engine renders namespace() and macros") {
  // namespace() + a mutating loop (the Qwen tool/thinking idiom).
  const char* ns =
      "{%- set found = namespace(value=false) %}"
      "{%- for m in messages %}{%- if m.role == 'user' %}"
      "{%- set found.value = true %}{%- endif %}{%- endfor %}"
      "{{- found.value }}";
  CHECK(apply_chat_template(ns, SystemUser(), false) == "True");

  // A macro definition + call.
  const char* macro =
      "{%- macro tag(role) -%}<|{{ role }}|>{%- endmacro -%}"
      "{{- tag('assistant') }}";
  CHECK(apply_chat_template(macro, {}, false) == "<|assistant|>");

  // A filter the subset never supported.
  CHECK(apply_chat_template("{{ 'hi' | upper }}", {}, false) == "HI");
}

// ─── #1681: the arity-0 Jinja2 built-in tests ────────────────────────────────
// The vendored minja implemented twelve of Jinja2's built-in tests and threw on
// every other name. `undefined` was missing, and it is the one real chat
// templates reach for, so the whole Qwen3.8 family answered HTTP 500. The rest
// of the arity-0 set is here for the same reason: a template is entitled to any
// name jinja2/tests.py defines, and finding out one is missing costs a 500 in
// production. Each expectation below is what CPython jinja2 3.1 returns.
namespace {
// Render one `{{ x is <test> }}` with `x` bound through chat_template_kwargs,
// which is also how the production request path binds it.
std::string IsTest(const std::string& test_name, nlohmann::ordered_json value) {
  nlohmann::ordered_json kwargs = nlohmann::ordered_json::object();
  kwargs["x"] = std::move(value);
  return apply_chat_template("{{ x is " + test_name + " }}", {}, false, "", "",
                             {}, kwargs);
}
}  // namespace

TEST_CASE("chat_template: `is undefined` answers for a variable nobody bound") {
  // THE defect. An unbound name is undefined and not defined; a bound one is
  // the other way round.
  CHECK(apply_chat_template("{{ nobody_bound_this is undefined }}", {},
                            false) == "True");
  CHECK(apply_chat_template("{{ nobody_bound_this is defined }}", {}, false) ==
        "False");
  CHECK(IsTest("undefined", 1) == "False");
  CHECK(IsTest("defined", 1) == "True");
  // `is not undefined` negates, so both spellings of the question agree.
  CHECK(apply_chat_template("{{ nobody_bound_this is not undefined }}", {},
                            false) == "False");

  // `undefined` is the exact complement of minja's `defined` on EVERY value,
  // including an explicitly bound null. That is a divergence from CPython
  // jinja2, which calls a bound None defined-and-not-undefined -- but the
  // divergence is `defined`'s, it shipped years ago, and the alternative is two
  // built-in tests that contradict each other on the same value. Pinned here so
  // the coupling is a decision and not an accident.
  CHECK(IsTest("undefined", nullptr) == "True");
  CHECK(IsTest("defined", nullptr) == "False");
  CHECK(IsTest("none", nullptr) == "True");

  // The exact construct that 500ed, in the shape the Qwen3.8 template uses it:
  // unsupplied renders the reasoning branch, supplied-false does not.
  const char* kGate =
      "{%- if enable_thinking is undefined or enable_thinking is true %}ON"
      "{%- else %}OFF{%- endif %}";
  CHECK(apply_chat_template(kGate, {}, false) == "ON");
  nlohmann::ordered_json off = nlohmann::ordered_json::object();
  off["enable_thinking"] = false;
  CHECK(apply_chat_template(kGate, {}, false, "", "", {}, off) == "OFF");
  nlohmann::ordered_json on = nlohmann::ordered_json::object();
  on["enable_thinking"] = true;
  CHECK(apply_chat_template(kGate, {}, false, "", "", {}, on) == "ON");
}

TEST_CASE("chat_template: the remaining arity-0 Jinja2 built-in tests") {
  // even / odd: jinja2 tests.py `value % 2 == 0` / `== 1`. Negatives included,
  // because C++ `%` truncates toward zero and a naive `== 1` gets them wrong.
  CHECK(IsTest("even", 4) == "True");
  CHECK(IsTest("even", 3) == "False");
  CHECK(IsTest("odd", 3) == "True");
  CHECK(IsTest("odd", 4) == "False");
  CHECK(IsTest("odd", -3) == "True");
  CHECK(IsTest("even", -4) == "True");
  CHECK_THROWS_AS(IsTest("even", "nope"), ChatTemplateError);

  // #1681 second review F5. `value % 2` is Python's, not C++'s, so the whole
  // numeric tower answers and the first implementation truncated it away with
  // `get<int64_t>()`. Measured on CPython jinja2 3.1.2, which is the standard
  // this block states:
  //   4.5 -> even False, odd False   (4.5 % 2 == 0.5, which is neither)
  //   4.0 -> even True               (a float that IS integral still counts)
  //   True -> odd True               (bool is an int in Python)
  // A non-integral float truncated to 4 answered `even True` here, and a bool
  // is not is_number() in minja so it threw where jinja2 answers.
  CHECK(IsTest("even", 4.5) == "False");
  CHECK(IsTest("odd", 4.5) == "False");
  CHECK(IsTest("even", -4.5) == "False");
  CHECK(IsTest("odd", -4.5) == "False");
  CHECK(IsTest("even", 4.0) == "True");
  CHECK(IsTest("odd", 4.0) == "False");
  CHECK(IsTest("odd", 3.0) == "True");
  CHECK(IsTest("odd", -3.0) == "True");
  CHECK(IsTest("even", -4.0) == "True");
  CHECK(IsTest("even", true) == "False");
  CHECK(IsTest("odd", true) == "True");
  CHECK(IsTest("even", false) == "True");
  CHECK(IsTest("odd", false) == "False");
  // Still a TypeError on everything that is not a number, as jinja2 raises.
  CHECK_THROWS_AS(IsTest("odd", "nope"), ChatTemplateError);
  CHECK_THROWS_AS(IsTest("even", nullptr), ChatTemplateError);

  // lower / upper: str(value).islower() / .isupper(). Python needs at least one
  // cased character, so a digit string is neither.
  CHECK(IsTest("lower", "abc") == "True");
  CHECK(IsTest("lower", "aBc") == "False");
  CHECK(IsTest("upper", "ABC") == "True");
  CHECK(IsTest("upper", "AbC") == "False");
  CHECK(IsTest("lower", "123") == "False");
  CHECK(IsTest("upper", "123") == "False");
  CHECK(IsTest("lower", "a1!") == "True");

  // escaped: hasattr(value, "__html__"). minja has no Markup type, so nothing
  // it can hold is escaped. Constant by construction, not a stub.
  CHECK(IsTest("escaped", "abc") == "False");
}

TEST_CASE("chat_template: an unknown `is` test still throws, so the list stays "
          "closed") {
  // The arity-1 tests are NOT implemented (minja parses the right side of `is`
  // as a bare identifier) and neither are `filter`/`test`. They must refuse
  // loudly rather than answer something plausible; that refusal is what turned
  // #1681 into a report instead of a silently wrong prompt.
  CHECK_THROWS_AS(apply_chat_template("{{ 4 is divisibleby }}", {}, false),
                  ChatTemplateError);
  CHECK_THROWS_AS(apply_chat_template("{{ 'x' is filter }}", {}, false),
                  ChatTemplateError);
  CHECK_THROWS_AS(apply_chat_template("{{ 'x' is callable }}", {}, false),
                  ChatTemplateError);
  CHECK_THROWS_AS(apply_chat_template("{{ 'x' is not_a_jinja_test }}", {},
                                      false),
                  ChatTemplateError);
}

// ─── #1681: chat_template_kwargs binding rules ───────────────────────────────
TEST_CASE("chat_template: chat_template_kwargs bind only the keys supplied") {
  nlohmann::ordered_json kwargs = nlohmann::ordered_json::object();
  kwargs["reasoning_effort"] = "low";
  kwargs["depth"] = 3;
  CHECK(apply_chat_template("{{ reasoning_effort }}/{{ depth }}", {}, false, "",
                            "", {}, kwargs) == "low/3");
  // A key NOT supplied stays undefined rather than becoming a bound null.
  CHECK(apply_chat_template("{{ depth is undefined }}", {}, false, "", "", {},
                            kwargs) == "False");
  CHECK(apply_chat_template("{{ other is undefined }}", {}, false, "", "", {},
                            kwargs) == "True");
}

// ─── #1681 review F1/F2: what a request may NOT put in chat_template_kwargs ──
// Measured against the pinned oracle (vLLM `555967922`, transformers 5.3.0) on
// tests/fixtures/qwen38_chat_template.jinja, by re-executing
// resolve_chat_template_kwargs and the transformers call it feeds:
//   template_vars   = {add_generation_prompt, add_vision_id, content,
//                      enable_thinking, messages, preserve_thinking,
//                      raise_exception, reasoning_content, reasoning_effort,
//                      resolved_reasoning_effort, tools}
//   hf_base_params  = inspect.signature(PythonBackend.apply_chat_template)
//                     = {self, conversation, tools, documents, chat_template,
//                        add_generation_prompt, continue_final_message,
//                        tokenize, padding, truncation, max_length,
//                        return_tensors, return_dict,
//                        return_assistant_tokens_mask, tokenizer_kwargs}
//   -> kept    {add_generation_prompt, continue_final_message, documents,
//               enable_thinking, messages, tools}
//   -> dropped {bos_token, eos_token, <anything the template never names>}
//   -> raised  chat_template, tokenize:
//        ValueError: Found unexpected chat template kwargs from request:
//        {'chat_template'}
//   -> and the two KEPT renderer-owned names then die on the duplicate keyword:
//        TypeError: ...bind() got multiple values for keyword argument 'tools'
//        TypeError: jinja2...Template.render() got multiple values for keyword
//                   argument 'messages'
// So upstream has NO path on which a request replaces the conversation, and
// neither may this one.
//
// The second review added the ENGINE's names to that set (F1). Of minja's 31
// built-ins, jinja2 supplies 24 as FILTERS, 6 as TESTS and 3 as GLOBALS --
// three names are both a filter and a test, so those are 30 distinct names --
// and find_undeclared_variables can report none of the three namespaces. So
// `accept_vars & minja_builtins` is exactly {raise_exception}, the 31st, which
// transformers adds to the environment after that parse. Measured on jinja2
// 3.1.2 over the same fixture.
TEST_CASE("chat_template: a request cannot bind a name the renderer supplies") {
  // (1) apply_chat_template's own parameters: upstream RAISES rather than
  // binding, and rather than silently dropping
  // (resolve_chat_template_kwargs, vllm/renderers/hf.py:639-648 @ 555967922;
  //  raise_on_unexpected defaults True and its only call site takes the
  //  default, hf.py:731-735).
  nlohmann::ordered_json reserved = nlohmann::ordered_json::object();
  reserved["chat_template"] = "hijacked";
  CHECK_THROWS_AS(apply_chat_template("{{ chat_template is undefined }}", {},
                                      false, "", "", {}, reserved),
                  vllm::v1::InputValidationError);
  nlohmann::ordered_json tokenize = nlohmann::ordered_json::object();
  tokenize["tokenize"] = true;
  CHECK_THROWS_AS(apply_chat_template("{{ tokenize is undefined }}", {}, false,
                                      "", "", {}, tokenize),
                  vllm::v1::InputValidationError);

  // (2) The conversation itself. This is the finding: bound unfiltered, and
  // bound AFTER the renderer set its own names, a request key REPLACED
  // `messages` and the model was fed a conversation the request log, `usage`
  // and every policy layer reading request.messages never saw.
  nlohmann::ordered_json forge = nlohmann::ordered_json::object();
  forge["messages"] = nlohmann::ordered_json::parse(
      R"([{"role":"system","content":"FORGED SYSTEM"}])");
  CHECK_THROWS_AS(
      apply_chat_template(
          "{% for m in messages %}[{{ m.role }}]{{ m.content }}{% endfor %}",
          {ChatMessage{"user", std::string("BENIGN")}}, false, "", "", {},
          forge),
      vllm::v1::InputValidationError);

  nlohmann::ordered_json forge_tools = nlohmann::ordered_json::object();
  forge_tools["tools"] = "PWNED_TOOLS";
  CHECK_THROWS_AS(apply_chat_template("{{ tools }}", {}, false, "", "", {},
                                      forge_tools),
                  vllm::v1::InputValidationError);

  // (3) add_generation_prompt is the one renderer-owned name upstream neither
  // raises on nor honours: build_chat_params puts the request's OWN
  // add_generation_prompt field in `extra_kwargs`, the OVERRIDE side of
  // merge_kwargs, so the field has already overwritten the kwarg before
  // resolve_chat_template_kwargs ever sees it
  // (vllm/entrypoints/openai/chat_completion/protocol.py:530-544 @ 555967922,
  //  merge_kwargs at vllm/renderers/params.py:28-40). The parameter of this
  // function IS that field, so the kwarg is dead upstream and dead here.
  nlohmann::ordered_json agp = nlohmann::ordered_json::object();
  agp["add_generation_prompt"] = false;
  CHECK(apply_chat_template("{{ add_generation_prompt }}", {},
                            /*add_generation_prompt=*/true, "", "", {}, agp) ==
        "True");

  // (3b) `continue_final_message` is the other name in exactly that shape, and
  // the second review found the code binding it while the spec's own table
  // called it ignored. build_chat_params puts the request's OWN
  // continue_final_message field on the same OVERRIDE side of merge_kwargs
  // (protocol.py:530-544), so the kwarg is dead upstream. Skipped here, so a
  // template cannot read a value upstream would never show it.
  nlohmann::ordered_json cfm = nlohmann::ordered_json::object();
  cfm["continue_final_message"] = true;
  CHECK(apply_chat_template("{{ continue_final_message is undefined }}", {},
                            false, "", "", {}, cfm) == "True");

  // (4) A name the ENGINE supplies. minja resolves a global, a filter and an
  // is-test through the same Context chain, and `set()` writes into the child
  // of Context::builtins(), so before the second review ANY of its 31 names
  // could be shadowed by a request key. jinja2 keeps all three kinds out of
  // the variable namespace, so find_undeclared_variables never reports one and
  // upstream's accept_vars drops the kwarg: a 200 there, a 500 here.
  nlohmann::ordered_json ns = nlohmann::ordered_json::object();
  ns["namespace"] = 1;
  CHECK(apply_chat_template("{%- set c = namespace(value=0) %}"
                            "{%- set c.value = 7 %}{{ c.value }}",
                            {}, false, "", "", {}, ns) == "7");
  nlohmann::ordered_json up = nlohmann::ordered_json::object();
  up["upper"] = 1;
  CHECK(apply_chat_template("{{ 'hi' | upper }}", {}, false, "", "", {}, up) ==
        "HI");
  // `select` resolves its test BY NAME through the same Context
  // (`context->get(args.args[1])`, minja.hpp select_or_reject), so the
  // registry names are shadowable too.
  nlohmann::ordered_json eq = nlohmann::ordered_json::object();
  eq["equalto"] = 1;
  CHECK(apply_chat_template(
            "{{ [1,2,1] | select('equalto', 1) | list | length }}", {}, false,
            "", "", {}, eq) == "2");

  // The ONE exception, and it is upstream's. `raise_exception` is the only
  // minja builtin jinja2 supplies nowhere: transformers adds it to the
  // environment AFTER _resolve_chat_template_kwargs parses with its own env
  // (hf.py:598-606), so it lands in find_undeclared_variables, upstream keeps
  // it, and the request value shadows the global at render. Binding it here is
  // therefore the mirror, and the shadow is observable the same way.
  nlohmann::ordered_json re = nlohmann::ordered_json::object();
  re["raise_exception"] = 1;
  CHECK_THROWS_WITH_AS(apply_chat_template("{{ raise_exception('x') }}", {},
                                           false, "", "", {}, re),
                       doctest::Contains("not callable"), ChatTemplateError);

  // (5) bos_token / eos_token DO bind, and that is upstream's behaviour, not a
  // hole. A template that names either has it in
  // find_undeclared_variables(chat_template), so upstream keeps the request's
  // value and transformers lets it win over the tokenizer's special tokens
  // (`template_kwargs = {**self.special_tokens_map, **kwargs}`,
  //  PythonBackend.apply_chat_template). Verified on the oracle:
  //   render("{{ bos_token }}|...", bos_token="REQ_BOS") -> "REQ_BOS|BENIGN".
  // A template that names neither drops them upstream and cannot observe them
  // here either way.
  nlohmann::ordered_json tokens = nlohmann::ordered_json::object();
  tokens["bos_token"] = "REQ_BOS";
  CHECK(apply_chat_template("{{ bos_token }}", {}, false, "MODEL_BOS", "", {},
                            tokens) == "REQ_BOS");
}

TEST_CASE("chat_template: the request kwargs win over the server defaults") {
  // merge_kwargs (vllm/renderers/params.py:28-40 @ 555967922), reached as
  // ChatParams.with_defaults(default_chat_template_kwargs) (params.py:93-122)
  // from vllm/entrypoints/openai/chat_completion/serving.py:208:
  //   defaults | {k: v for k, v in overrides.items() if v not in (None, "auto")}
  // (`multimodal/media/base.py:53-67`, cited here before the #1681 review, is
  //  MediaIO.merge_kwargs -- the media-io path, not this one.)
  nlohmann::ordered_json defaults = nlohmann::ordered_json::object();
  defaults["enable_thinking"] = false;
  defaults["reasoning_effort"] = "low";
  auto fn = MakeChatTemplatePromptFn(
      "{{ enable_thinking }}/{{ reasoning_effort }}", "", "", defaults);

  CHECK(fn({}, false, {}, nlohmann::ordered_json::object()) == "False/low");

  nlohmann::ordered_json request = nlohmann::ordered_json::object();
  request["enable_thinking"] = true;
  CHECK(fn({}, false, {}, request) == "True/low");

  // ...EXCEPT that `unset_values = (None, "auto")`: an override valued null or
  // "auto" means "the client did not set this", and the SERVER default stands.
  // Without this, a request null defeated `--no-enable-thinking` on the very
  // field this row adds (#1681 review F3).
  nlohmann::ordered_json unset = nlohmann::ordered_json::object();
  unset["enable_thinking"] = nullptr;
  unset["reasoning_effort"] = "auto";
  CHECK(fn({}, false, {}, unset) == "False/low");

  // `false` and `""` are NOT unset: Python's `v not in (None, "auto")` keeps
  // them, and only a null or the exact string "auto" drops out. Driven against
  // a server default of TRUE so that "kept" and "dropped" differ.
  nlohmann::ordered_json on = nlohmann::ordered_json::object();
  on["enable_thinking"] = true;
  on["reasoning_effort"] = "low";
  auto fn_on = MakeChatTemplatePromptFn(
      "{{ enable_thinking }}/{{ reasoning_effort }}", "", "", on);
  nlohmann::ordered_json falsey = nlohmann::ordered_json::object();
  falsey["enable_thinking"] = false;
  falsey["reasoning_effort"] = "";
  CHECK(fn_on({}, false, {}, falsey) == "False/");
  CHECK(fn_on({}, false, {}, unset) == "True/low");

  // No server default at all leaves the name undefined, which is upstream's
  // own default (--default-chat-template-kwargs is None).
  auto bare = MakeChatTemplatePromptFn("{{ enable_thinking is undefined }}");
  CHECK(bare({}, false, {}, nlohmann::ordered_json::object()) == "True");
}

// The rule behind --enable-thinking / --no-enable-thinking, which server_main
// calls with its tri-state flag. Driven here rather than through the binary
// because the server resolves its chat template only after a real tokenizer
// loads, and that needs a checkpoint no CPU gate has.
TEST_CASE("chat_template: DefaultChatTemplateKwargs keeps unset apart from "
          "explicitly false") {
  using vllm::entrypoints::DefaultChatTemplateKwargs;
  CHECK(DefaultChatTemplateKwargs(std::nullopt).empty());
  CHECK(DefaultChatTemplateKwargs(false).dump() ==
        "{\"enable_thinking\":false}");
  CHECK(DefaultChatTemplateKwargs(true).dump() == "{\"enable_thinking\":true}");

  // What the three states DO to the construct that 500ed. Neither flag is not
  // the same answer as --no-enable-thinking, which is the defect this repairs.
  const char* kGate =
      "{%- if enable_thinking is undefined or enable_thinking is true %}ON"
      "{%- else %}OFF{%- endif %}";
  const nlohmann::ordered_json kNone = nlohmann::ordered_json::object();
  CHECK(MakeChatTemplatePromptFn(
            kGate, "", "", DefaultChatTemplateKwargs(std::nullopt))(
            {}, false, {}, kNone) == "ON");
  CHECK(MakeChatTemplatePromptFn(kGate, "", "",
                                 DefaultChatTemplateKwargs(false))(
            {}, false, {}, kNone) == "OFF");
  CHECK(MakeChatTemplatePromptFn(kGate, "", "",
                                 DefaultChatTemplateKwargs(true))(
            {}, false, {}, kNone) == "ON");
}

// ─── M3.3 Task 3: tools rendered into the prompt via the tool branch ─────────
namespace {
// A minja-subset tool template mirroring the Qwen3.6/Hermes tool-system-prompt
// shape (the full qwen35.jinja tool branch uses namespace/macros/is-tests/`[::-1]`
// slicing beyond the subset; this reproduces its tool-schema injection using the
// constructs the engine supports + the M3.3 `tojson` filter).
const char* kToolTemplate =
    "{%- if tools %}"
    "<|im_start|>system\n# Tools\n<tools>"
    "{%- for tool in tools %}\n{{ tool | tojson }}{%- endfor %}"
    "\n</tools><|im_end|>\n"
    "{%- endif %}"
    "{%- for message in messages %}"
    "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
    "{%- endfor %}";

std::vector<ChatCompletionToolsParam> WeatherTool() {
  ChatCompletionToolsParam t;
  t.type = "function";
  t.function.name = "get_weather";
  t.function.description = "Get the weather for a city.";
  t.function.parameters = nlohmann::json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"}}})");
  return {t};
}
}  // namespace

TEST_CASE("chat_template: tools render the function schemas into the prompt") {
  const std::vector<ChatMessage> msgs = {
      ChatMessage{"user", std::string("weather?")}};
  const std::string out = apply_chat_template(
      kToolTemplate, msgs, /*add_generation_prompt=*/false, /*bos=*/"",
      /*eos=*/"", WeatherTool());

  // The tool system prompt + the schema JSON (name/description/parameters) are
  // present, and the user turn follows. minja's `tojson` uses Python/Jinja2
  // json.dumps separators (`": "`, `", "`) rather than the deleted subset's
  // compact nlohmann dump, so the key/value pairs carry a space after the colon.
  CHECK(out.find("# Tools") != std::string::npos);
  CHECK(out.find("<tools>") != std::string::npos);
  CHECK(out.find("\"name\": \"get_weather\"") != std::string::npos);
  CHECK(out.find("\"description\": \"Get the weather for a city.\"") !=
        std::string::npos);
  CHECK(out.find("\"parameters\"") != std::string::npos);
  CHECK(out.find("\"type\": \"function\"") != std::string::npos);
  CHECK(out.find("<|im_start|>user\nweather?<|im_end|>") != std::string::npos);
}

TEST_CASE("chat_template: no tools leaves the tool branch out (falsy tools)") {
  const std::vector<ChatMessage> msgs = {
      ChatMessage{"user", std::string("hi")}};
  const std::string out =
      apply_chat_template(kToolTemplate, msgs, /*add_generation_prompt=*/false);
  CHECK(out.find("# Tools") == std::string::npos);
  // The `{%- endfor %}` left-trim drops the loop body's trailing newline.
  CHECK(out == "<|im_start|>user\nhi<|im_end|>");
}

// `tojson` serializes a nested tool schema faithfully.
TEST_CASE("chat_template: tojson filter serializes the tool object") {
  const std::string out =
      apply_chat_template("{{ tools[0] | tojson }}", {}, false, "", "",
                          WeatherTool());
  const nlohmann::json j = nlohmann::json::parse(out);
  CHECK(j.at("type") == "function");
  CHECK(j.at("function").at("name") == "get_weather");
  CHECK(j.at("function").at("parameters").at("properties").at("city").at("type") ==
        "string");
}

// ─── The REAL Qwen3.5 chat template (namespace()/macros/is-tests/<tool_call>) ─
// Extracted from the Qwen3.5-2B GGUF (tokenizer.chat_template) into
// tests/fixtures/qwen35_chat_template.jinja. The old hand-written subset threw
// on its namespace()/macro/is-test constructs; the vendored minja engine renders
// it. This is the real-world driver for the whole vendoring effort.
namespace {
std::string ReadFixture(const std::string& name) {
  std::ifstream f(std::string(VLLM_TEST_FIXTURES_DIR) + "/" + name,
                  std::ios::binary);
  if (!f) throw std::runtime_error("cannot open fixture: " + name);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

ChatMessage AssistantToolCall(std::string arguments,
                              std::string function_name = "terminal") {
  ChatMessage assistant;
  assistant.role = "assistant";
  vllm::entrypoints::openai::ToolCall call;
  call.id = "call_1";
  call.function.name = std::move(function_name);
  call.function.arguments = std::move(arguments);
  assistant.tool_calls =
      std::vector<vllm::entrypoints::openai::ToolCall>{std::move(call)};
  return assistant;
}

// ─── tojson guards: the SHARED seam stays on the pinned renderer ────────────
// The serving reference for `{{ tool | tojson }}` is the PINNED transformers
// 5.14.1 renderer's tojson override (utils/chat_template_utils.py:481:
// sort_keys=False, ensure_ascii=False, no HTML escaping) — NOT plain jinja2's
// builtin (which sorts keys and escapes HTML). The expected bytes below were
// rendered by the pin itself (transformers 5.14.1 + jinja2 3.1.6 through
// render_jinja_template, 2026-10-09; see docs/bench-evidence/
// kolibri1-serve-20261008.md "Review repair"), so any drift of this adapter's
// tojson away from the pinned renderer fails here — on NON-kolibri templates
// no kolibri1 row owns.
// An insertion-ordered (NOT alphabetical) tool at every level, with Unicode
// and HTML in the leaves: the case that separates the pinned renderer from
// plain jinja2's built-in tojson.
std::vector<ChatCompletionToolsParam> UnsortedUnicodeTool() {
  ChatCompletionToolsParam t;
  t.type = "function";
  t.function.name = "get_weather";
  t.function.description = "Wetter für <b>München</b> & ☕";
  t.function.parameters = nlohmann::ordered_json::parse(
      R"({"type":"object","properties":{"zeta":{"type":"string","description":"<hr> München"},"alpha":{"type":"string"}},"required":["zeta","alpha"],"x_f":0.5})");
  return {t};
}

// The schema is alphabetical at EVERY level (properties < required < type;
// the single-property schema needs no order), so the sorted-dump override the
// adapter carried BEFORE the review repair and the pinned renderer's
// insertion-order tojson produce the SAME bytes for it. The tool WRAPPER's
// field order (type/function, name/description/parameters) is the pinned
// renderer's fixed model_dump order — NOT alphabetical — so the byte-identity
// claim can only be demonstrated on the schema itself, which is what the
// case below renders.
std::vector<ChatCompletionToolsParam> AlphabeticalTool() {
  ChatCompletionToolsParam t;
  t.type = "function";
  t.function.name = "get_weather";
  t.function.description = "Get the weather for a city.";
  t.function.parameters = nlohmann::ordered_json::parse(
      R"({"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"})");
  return {t};
}

// The pinned renderer's insertion-ordered tojson of UnsortedUnicodeTool(),
// shared by the tests below.
const char* kUnsortedToolJson =
    "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", "
    "\"description\": \"Wetter für <b>München</b> & ☕\", \"parameters\": "
    "{\"type\": \"object\", \"properties\": {\"zeta\": {\"type\": \"string\", "
    "\"description\": \"<hr> München\"}, \"alpha\": {\"type\": \"string\"}}, "
    "\"required\": [\"zeta\", \"alpha\"], \"x_f\": 0.5}}}";
}  // namespace

TEST_CASE("chat_template: tojson keeps the pinned renderer's insertion order") {
  const std::vector<ChatMessage> msgs = {
      ChatMessage{"user", std::string("weather?")}};
  const std::string out = apply_chat_template(
      kToolTemplate, msgs, /*add_generation_prompt=*/false, /*bos=*/"",
      /*eos=*/"", UnsortedUnicodeTool());
  // Byte-identical to the pinned transformers 5.14.1 renderer's
  // `{{ tool | tojson }}`: insertion key order at every level, Unicode raw,
  // HTML unescaped.
  CHECK(out ==
        std::string("<|im_start|>system\n# Tools\n<tools>") +
            kUnsortedToolJson + "</tools><|im_end|>" +
            "<|im_start|>user\nweather?<|im_end|>");
}

TEST_CASE("chat_template: tojson is byte-stable for already-ordered schemas") {
  // The repair must not alter a rendering where the pinned renderer and the
  // old sorted override already agreed: for this alphabetically-ordered
  // schema the sorted and insertion-order dumps are the same bytes, so this
  // case stays green even under a sorted tojson — the byte-identity-before/
  // after claim is real. The schema is rendered alone because the tool
  // wrapper's field order (type/function, name/description/parameters) is the
  // pinned model_dump order, which is NOT alphabetical.
  const std::string out = apply_chat_template(
      "{{ tools[0].function.parameters | tojson }}", {},
      /*add_generation_prompt=*/false, /*bos=*/"", /*eos=*/"",
      AlphabeticalTool());
  CHECK(out ==
        "{\"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": "
        "[\"city\"], \"type\": \"object\"}");
}

TEST_CASE("chat_template: tojson options match the pinned renderer") {
  // The four options the pinned tojson accepts (chat_template_utils.py:481),
  // each pinned byte-for-byte against the pin.
  const auto render = [](const std::string& call) {
    return apply_chat_template("{{ tools[0] | " + call + " }}", {}, false, "",
                               "", UnsortedUnicodeTool());
  };
  // indent=2: newline + 2 spaces per level, (",", ": ") separators.
  CHECK(render("tojson(indent=2)") ==
        "{\n"
        "  \"type\": \"function\",\n"
        "  \"function\": {\n"
        "    \"name\": \"get_weather\",\n"
        "    \"description\": \"Wetter für <b>München</b> & ☕\",\n"
        "    \"parameters\": {\n"
        "      \"type\": \"object\",\n"
        "      \"properties\": {\n"
        "        \"zeta\": {\n"
        "          \"type\": \"string\",\n"
        "          \"description\": \"<hr> München\"\n"
        "        },\n"
        "        \"alpha\": {\n"
        "          \"type\": \"string\"\n"
        "        }\n"
        "      },\n"
        "      \"required\": [\n"
        "        \"zeta\",\n"
        "        \"alpha\"\n"
        "      ],\n"
        "      \"x_f\": 0.5\n"
        "    }\n"
        "  }\n"
        "}");
  // sort_keys=True: recursive key sort (byte order == code-point order).
  CHECK(render("tojson(sort_keys=True)") ==
        "{\"function\": {\"description\": \"Wetter für <b>München</b> & ☕\", "
        "\"name\": \"get_weather\", \"parameters\": {\"properties\": {\"alpha\": "
        "{\"type\": \"string\"}, \"zeta\": {\"description\": \"<hr> "
        "München\", \"type\": \"string\"}}, \"required\": [\"zeta\", "
        "\"alpha\"], \"type\": \"object\", \"x_f\": 0.5}}, \"type\": "
        "\"function\"}");
  // ensure_ascii=True: non-ASCII escapes as \uXXXX (literal backslash-u in
  // the rendered bytes, hence the doubled backslashes here).
  CHECK(render("tojson(ensure_ascii=True)") ==
        "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", "
        "\"description\": \"Wetter f\\u00fcr <b>M\\u00fcnchen</b> & "
        "\\u2615\", \"parameters\": {\"type\": \"object\", \"properties\": "
        "{\"zeta\": {\"type\": \"string\", \"description\": \"<hr> "
        "M\\u00fcnchen\"}, \"alpha\": {\"type\": \"string\"}}, \"required\": "
        "[\"zeta\", \"alpha\"], \"x_f\": 0.5}}}");
  // separators: compact (item ",", key ":").
  CHECK(render("tojson(separators=(',', ':'))") ==
        "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
        "\"description\":\"Wetter für <b>München</b> & ☕\",\"parameters\":"
        "{\"type\":\"object\",\"properties\":{\"zeta\":{\"type\":\"string\""
        ",\"description\":\"<hr> München\"},\"alpha\":{\"type\":\"string\"}},"
        "\"required\":[\"zeta\",\"alpha\"],\"x_f\":0.5}}}");
}

TEST_CASE("chat_template: the real Qwen3.5 template's tojson matches the pin") {
  // A second, non-kolibri template (the real Qwen3.5 fixture) rendering the
  // unsorted tool: the pinned renderer's insertion-ordered, raw-Unicode,
  // unescaped-HTML tojson bytes must appear in the prompt.
  const std::string tmpl = ReadFixture("qwen35_chat_template.jinja");
  const std::vector<ChatMessage> msgs = {
      ChatMessage{"user", std::string("weather?")}};
  std::string out;
  REQUIRE_NOTHROW(out = apply_chat_template(tmpl, msgs,
                                            /*add_generation_prompt=*/true,
                                            /*bos=*/"", /*eos=*/"<|im_end|>",
                                            UnsortedUnicodeTool()));
  CHECK(out.find(kUnsortedToolJson) != std::string::npos);
}

TEST_CASE("chat_template: real Qwen3.5 template renders a plain conversation") {
  const std::string tmpl = ReadFixture("qwen35_chat_template.jinja");
  std::string out;
  REQUIRE_NOTHROW(out = apply_chat_template(tmpl, SystemUser(),
                                            /*add_generation_prompt=*/true,
                                            /*bos=*/"", /*eos=*/"<|im_end|>"));
  // The chatml turns are present and the assistant generation header is added.
  CHECK(out.find("<|im_start|>system\nYou are a helpful assistant.<|im_end|>") !=
        std::string::npos);
  CHECK(out.find("<|im_start|>user\nHello, who are you?<|im_end|>") !=
        std::string::npos);
  CHECK(out.find("<|im_start|>assistant") != std::string::npos);
  // With no tools, the tools system prompt must NOT appear.
  CHECK(out.find("<tools>") == std::string::npos);
}

TEST_CASE("chat_template: real Qwen3.5 template renders the tools branch") {
  const std::string tmpl = ReadFixture("qwen35_chat_template.jinja");
  const std::vector<ChatMessage> msgs = {
      ChatMessage{"user", std::string("what is the weather?")}};
  std::string out;
  REQUIRE_NOTHROW(out = apply_chat_template(
                      tmpl, msgs, /*add_generation_prompt=*/true, /*bos=*/"",
                      /*eos=*/"<|im_end|>", WeatherTool()));
  // The tool system prompt, the <tool_call> instruction surface, and the tool
  // name are all injected by the template's tool branch.
  CHECK(out.find("# Tools") != std::string::npos);
  CHECK(out.find("<tools>") != std::string::npos);
  CHECK(out.find("<tool_call>") != std::string::npos);
  CHECK(out.find("get_weather") != std::string::npos);
  CHECK(out.find("<|im_start|>user\nwhat is the weather?<|im_end|>") !=
        std::string::npos);
}

// ─── LoadChatTemplateFromGguf (ABI v3) ───────────────────────────────────────
// GGUF models carry their chat template as `tokenizer.chat_template` metadata
// (no tokenizer_config.json exists). Build minimal synthetic GGUFs and assert
// the loader extracts the template / errors loudly.
TEST_CASE("chat_template: LoadChatTemplateFromGguf extracts tokenizer.chat_template") {
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", "qwen35"));
  b.AddKv(gguf_test::StrKv("tokenizer.chat_template",
                           "{{ messages[0].content }}"));
  gguf_test::TempFile f(b.Build());
  CHECK(vllm::entrypoints::LoadChatTemplateFromGguf(f.path()) ==
        "{{ messages[0].content }}");
}

TEST_CASE("chat_template: LoadChatTemplateFromGguf errors without the key") {
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", "qwen35"));
  gguf_test::TempFile f(b.Build());
  CHECK_THROWS_AS(vllm::entrypoints::LoadChatTemplateFromGguf(f.path()),
                  vllm::entrypoints::ChatTemplateError);
}

TEST_CASE("chat_template: LoadChatTemplateFromGguf errors on an unreadable file") {
  CHECK_THROWS_AS(
      vllm::entrypoints::LoadChatTemplateFromGguf("/nonexistent/x.gguf"),
      vllm::entrypoints::ChatTemplateError);
}

// ─── LoadChatTemplateForModel (server selection path, #2077) ────────────────
// The server's chat-template SELECTION path: try tokenizer_config.json first,
// then fall back to GGUF metadata. A .gguf model has no tokenizer_config.json,
// so the config path throws and the GGUF supplies the template. This is the
// path the server wires (server_main.cpp), and the existing
// LoadChatTemplateFromGguf unit tests only cover the helper, not this
// selection. A regression here is the model looping on `\nassistant:\n` with
// zero useful output (#2077).
TEST_CASE("chat_template: LoadChatTemplateForModel loads from GGUF when no tokenizer_config.json") {
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", "qwen35"));
  b.AddKv(gguf_test::StrKv("tokenizer.chat_template",
                           "{{ messages[0].content }}"));
  gguf_test::TempFile f(b.Build());
  // The server derives tokenizer_config_path as <model_dir>/tokenizer_config.json.
  // For a .gguf file that path does not exist (the GGUF is a file, not a dir),
  // so LoadChatTemplateFromConfig throws and the selection falls back to the GGUF.
  const std::string config_path = f.path() + "/tokenizer_config.json";
  std::string source;
  std::string tmpl = vllm::entrypoints::LoadChatTemplateForModel(
      config_path, f.path(), source);
  // The GGUF template is loaded, NOT the role-join fallback.
  CHECK(tmpl == "{{ messages[0].content }}");
  CHECK(!tmpl.empty());
  // The source names the GGUF, not the non-existent tokenizer_config_path.
  CHECK(source.find("GGUF") != std::string::npos);
  CHECK(source.find(f.path()) != std::string::npos);
  CHECK(source.find("tokenizer_config") == std::string::npos);
}

TEST_CASE("chat_template: LoadChatTemplateForModel prefers tokenizer_config.json over GGUF") {
  // A safetensors dir has a real tokenizer_config.json. The config path wins
  // even when the model_dir happens to be a .gguf file, because the config
  // load is tried first.
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", "qwen35"));
  b.AddKv(gguf_test::StrKv("tokenizer.chat_template", "GGUF_TEMPLATE"));
  gguf_test::TempFile f(b.Build());
  // Write a tokenizer_config.json beside the GGUF with a different template.
  const std::string config_path = f.path() + "_tokenizer_config.json";
  {
    std::ofstream out(config_path);
    out << R"({"chat_template":"CONFIG_TEMPLATE"})";
  }
  std::string source;
  std::string tmpl = vllm::entrypoints::LoadChatTemplateForModel(
      config_path, f.path(), source);
  CHECK(tmpl == "CONFIG_TEMPLATE");
  CHECK(source.find(config_path) != std::string::npos);
  CHECK(source.find("GGUF") == std::string::npos);
  std::filesystem::remove(config_path);
}

TEST_CASE("chat_template: LoadChatTemplateForModel throws for a non-GGUF dir with no config") {
  // A non-.gguf model dir with no tokenizer_config.json has no fallback, so the
  // selection throws and the server falls back to the role-join prompt.
  std::string source;
  CHECK_THROWS_AS(
      vllm::entrypoints::LoadChatTemplateForModel(
          "/nonexistent/tokenizer_config.json", "/nonexistent/model_dir", source),
      vllm::entrypoints::ChatTemplateError);
}

// Review-driven round trip: user -> assistant tool_call -> tool result ->
// next generation prompt, rendered through the REAL Qwen3.5 template. The
// tool turn's identity fields must reach the template context (protocol
// from_json + the minja adapter), or the second turn is malformed.
TEST_CASE("chat_template: multi-turn tool conversation renders through the Qwen3.5 fixture") {
  const std::string tmpl = ReadFixture("qwen35_chat_template.jinja");

  ChatMessage user{"user", std::string("What is the weather in Rome?")};
  ChatMessage assistant;
  assistant.role = "assistant";
  vllm::entrypoints::openai::ToolCall call;
  call.id = "call_1";
  call.function.name = "get_weather";
  call.function.arguments = "{\"city\": \"Rome\"}";
  assistant.tool_calls = std::vector<vllm::entrypoints::openai::ToolCall>{call};
  ChatMessage tool;
  tool.role = "tool";
  tool.tool_call_id = "call_1";
  tool.name = "get_weather";
  tool.content = "{\"temp\": 21}";

  const std::string out = vllm::entrypoints::apply_chat_template(
      tmpl, {user, assistant, tool}, /*add_generation_prompt=*/true, "", "",
      WeatherTool());
  // The rendered prompt must carry the tool call AND the tool result, and end
  // with the assistant generation header.
  CHECK(out.find("get_weather") != std::string::npos);
  CHECK(out.find("<parameter=city>\nRome\n</parameter>") !=
        std::string::npos);
  CHECK(out.find("{\"temp\": 21}") != std::string::npos);
  CHECK(out.rfind("<|im_start|>assistant") != std::string::npos);
}

// Issue #526: OpenAI carries historical function.arguments as a JSON string,
// but pinned vLLM decodes that string before handing messages to a tokenizer
// chat template. Gemma 4 requires the decoded mapping. The fixture is byte-exact
// to the live FP8 model's chat_template.jinja (sha256 aa3185df...), derived from
// pinned vLLM's canonical example with its raise branch adapted for minja.
TEST_CASE("chat_template: Gemma4 decodes OpenAI tool arguments before rendering") {
  const std::string tmpl = ReadFixture("gemma4_tool_chat_template.jinja");

  ChatMessage user{"user", std::string("Run the date command.")};
  ChatMessage assistant = AssistantToolCall(R"({"command":"date"})");

  std::string out;
  REQUIRE_NOTHROW(out = apply_chat_template(
                      tmpl, {user, assistant},
                      /*add_generation_prompt=*/false));
  CHECK(out.find("<|tool_call>call:terminal{command:<|\"|>date<|\"|>}<tool_call|>") !=
        std::string::npos);
  CHECK(out.find("call:terminal{\"command\"") == std::string::npos);
}

TEST_CASE("chat_template: Gemma4 renders nested decoded tool arguments") {
  const std::string tmpl = ReadFixture("gemma4_tool_chat_template.jinja");
  const ChatMessage assistant = AssistantToolCall(
      R"({"text":"hi","ok":true,"n":7,"items":[1,"x"],"obj":{"k":null},"nil":null})");
  const std::string out =
      apply_chat_template(tmpl, {assistant}, /*add_generation_prompt=*/false);
  CHECK(out.find(
            "call:terminal{items:[1,<|\"|>x<|\"|>],n:7,nil:null,"
            "obj:{k:null},ok:true,text:<|\"|>hi<|\"|>}") !=
        std::string::npos);
}

TEST_CASE("chat_template: Gemma4 maps empty and JSON null tool arguments to objects") {
  const std::string tmpl = ReadFixture("gemma4_tool_chat_template.jinja");
  for (const std::string& arguments : {std::string(), std::string("null")}) {
    const std::string out = apply_chat_template(
        tmpl, {AssistantToolCall(arguments)}, /*add_generation_prompt=*/false);
    CHECK(out.find("<|tool_call>call:terminal{}<tool_call|>") !=
          std::string::npos);
  }
}

TEST_CASE("chat_template: malformed historical tool arguments fail closed") {
  const std::string malformed = R"({"command":)";
  try {
    (void)apply_chat_template(ReadFixture("gemma4_tool_chat_template.jinja"),
                              {AssistantToolCall(malformed)}, false);
    FAIL("expected ChatTemplateError");
  } catch (const ChatTemplateError& e) {
    const std::string what = e.what();
    CHECK(what.find("assistant tool_calls[0].function.arguments") !=
          std::string::npos);
    CHECK(what.find("terminal") != std::string::npos);
    CHECK(what.find(malformed) == std::string::npos);
  }
}

TEST_CASE("chat_template: argument normalization leaves OpenAI wire encoding intact") {
  const ChatMessage assistant = AssistantToolCall(R"({"command":"date"})");
  const nlohmann::json wire = assistant;
  CHECK(wire["tool_calls"][0]["function"]["arguments"] ==
        R"({"command":"date"})");
  CHECK(wire["tool_calls"][0]["function"]["arguments"].is_string());
}

TEST_CASE("chat_template: argument normalization is assistant-history only") {
  const std::string malformed = R"({"command":)";
  ChatMessage user = AssistantToolCall(malformed);
  user.role = "user";

  std::string out;
  REQUIRE_NOTHROW(out = apply_chat_template(
                      "{{ messages[0].tool_calls[0].function.arguments }}",
                      {user}, /*add_generation_prompt=*/false));
  CHECK(out == malformed);
}

// ─── ISSUE-LOCAL-01M2EAQ6R63BSRF1GVZ3JAR4A4: parse once, render per request ──
// The chat path used to hand the template STRING to apply_chat_template on
// every request, so minja re-parsed all 8952 chars of the Qwen3.8 template
// (45 ms on a desktop core, against 0.1 ms to render) before each render.
// Upstream compiles once: transformers' _compile_jinja_template is lru_cache'd,
// and vLLM resolves the template once per model (vllm/renderers/hf.py).
namespace {
using vllm::entrypoints::ChatTemplateParseCountForTesting;

// One request the prompt fn can be handed: the messages, the tools, the
// per-request chat_template_kwargs, and add_generation_prompt.
struct TemplateRequest {
  std::vector<ChatMessage> messages;
  std::vector<ChatCompletionToolsParam> tools;
  nlohmann::ordered_json kwargs = nlohmann::ordered_json::object();
  bool add_generation_prompt = true;
};

nlohmann::ordered_json Kwargs(const char* json) {
  return nlohmann::ordered_json::parse(json);
}

// The shapes the chat endpoint actually sends to the Qwen3.8 target: reasoning
// left to the template's default, forced on, forced off with an effort, tools
// present, and a multi-turn history carrying a tool call, its result and prior
// reasoning.
std::vector<TemplateRequest> Qwen38Requests() {
  std::vector<TemplateRequest> out;
  out.push_back({SystemUser(), {}, nlohmann::ordered_json::object(), true});
  out.push_back({{ChatMessage{"user", std::string("Janet's ducks lay 16 eggs "
                                                  "per day. How much?")}},
                 {},
                 Kwargs(R"({"enable_thinking": true})"),
                 true});
  out.push_back({SystemUser(), {},
                 Kwargs(R"({"enable_thinking": false, "reasoning_effort": "low"})"),
                 true});
  out.push_back({{ChatMessage{"user", std::string("what is the weather?")}},
                 WeatherTool(),
                 nlohmann::ordered_json::object(),
                 true});

  ChatMessage user{"user", std::string("What is the weather in Rome?")};
  ChatMessage assistant = AssistantToolCall(R"({"city": "Rome"})", "get_weather");
  assistant.reasoning = std::string("The user wants the weather in Rome.");
  ChatMessage tool;
  tool.role = "tool";
  tool.tool_call_id = "call_1";
  tool.name = "get_weather";
  tool.content = "{\"temp\": 21}";
  ChatMessage followup{"user", std::string("And tomorrow?")};
  out.push_back({{ChatMessage{"system", std::string("Be brief.")}, user,
                  assistant, tool, followup},
                 WeatherTool(),
                 Kwargs(R"({"enable_thinking": true, "reasoning_effort": "medium"})"),
                 true});
  out.push_back({MultiTurn(), {}, nlohmann::ordered_json::object(), false});
  return out;
}

std::string Render(const vllm::entrypoints::openai::ChatPromptFn& fn,
                   const TemplateRequest& r) {
  return fn(r.messages, r.add_generation_prompt, r.tools, r.kwargs);
}

// The uncached path: a fresh parse of the template string for this one render.
std::string RenderUncached(const std::string& tmpl, const TemplateRequest& r,
                           const std::string& bos, const std::string& eos) {
  return apply_chat_template(tmpl, r.messages, r.add_generation_prompt, bos,
                             eos, r.tools, r.kwargs);
}
}  // namespace

TEST_CASE("chat_template: the prompt fn parses its template once, not per "
          "request") {
  const std::string tmpl = ReadFixture("qwen38_chat_template.jinja");
  const std::vector<TemplateRequest> requests = Qwen38Requests();

  const std::uint64_t before = ChatTemplateParseCountForTesting();
  auto fn = MakeChatTemplatePromptFn(tmpl, "", "<|im_end|>");
  for (int round = 0; round < 3; ++round) {
    for (const TemplateRequest& r : requests) (void)Render(fn, r);
  }
  // 18 renders through one prompt fn, one parse. A per-request parse reads 18.
  CHECK(ChatTemplateParseCountForTesting() - before == 1);
}

TEST_CASE("chat_template: the cached prompt fn renders the bytes a fresh parse "
          "renders") {
  const std::string qwen38 = ReadFixture("qwen38_chat_template.jinja");
  auto fn = MakeChatTemplatePromptFn(qwen38, "", "<|im_end|>");
  for (const TemplateRequest& r : Qwen38Requests()) {
    const std::string cached = Render(fn, r);
    CHECK(cached == RenderUncached(qwen38, r, "", "<|im_end|>"));
    CHECK(cached.find("<|im_start|>") != std::string::npos);
  }

  // The reasoning default really is a different prompt per shape, so the
  // equality above is not comparing one constant string with itself.
  const std::vector<TemplateRequest> reqs = Qwen38Requests();
  CHECK(Render(fn, reqs[0]) != Render(fn, reqs[2]));

  // The existing fixtures and the in-file templates.
  const std::string qwen35 = ReadFixture("qwen35_chat_template.jinja");
  auto fn35 = MakeChatTemplatePromptFn(qwen35, "", "<|im_end|>");
  TemplateRequest tools_req{{ChatMessage{"user", std::string("weather?")}},
                            WeatherTool(),
                            nlohmann::ordered_json::object(),
                            true};
  TemplateRequest plain_req{SystemUser(), {}, nlohmann::ordered_json::object(),
                            true};
  CHECK(Render(fn35, tools_req) ==
        RenderUncached(qwen35, tools_req, "", "<|im_end|>"));
  CHECK(Render(fn35, plain_req) ==
        RenderUncached(qwen35, plain_req, "", "<|im_end|>"));

  const std::string gemma4 = ReadFixture("gemma4_tool_chat_template.jinja");
  auto fn_g4 = MakeChatTemplatePromptFn(gemma4);
  TemplateRequest g4_req{{ChatMessage{"user", std::string("Run date.")},
                          AssistantToolCall(R"({"command":"date"})")},
                         {},
                         nlohmann::ordered_json::object(),
                         false};
  CHECK(Render(fn_g4, g4_req) == RenderUncached(gemma4, g4_req, "", ""));

  for (const char* t : {kQwenTemplate, kToolTemplate}) {
    auto small = MakeChatTemplatePromptFn(t, "<s>", "</s>");
    CHECK(Render(small, tools_req) == RenderUncached(t, tools_req, "<s>", "</s>"));
    CHECK(Render(small, plain_req) == RenderUncached(t, plain_req, "<s>", "</s>"));
  }
}

TEST_CASE("chat_template: an unparseable template still fails per request, "
          "with the error a fresh parse reports") {
  // transformers compiles lazily inside apply_chat_template, so a broken
  // template is a per-request error there, and api_server maps it to 400. The
  // cached path keeps that: building the prompt fn does not throw, and every
  // call throws the same ChatTemplateError a fresh parse throws.
  for (const char* broken : {"{{ 'x'", "{% endfor %}", "{% for m in messages %}"}) {
    std::optional<vllm::entrypoints::openai::ChatPromptFn> fn;
    REQUIRE_NOTHROW(fn.emplace(MakeChatTemplatePromptFn(broken)));
    std::string fresh;
    try {
      (void)apply_chat_template(broken, {}, false);
    } catch (const ChatTemplateError& e) {
      fresh = e.what();
    }
    REQUIRE_FALSE(fresh.empty());
    for (int i = 0; i < 2; ++i) {
      std::string cached;
      try {
        (void)(*fn)({}, false, {}, nlohmann::ordered_json::object());
      } catch (const ChatTemplateError& e) {
        cached = e.what();
      }
      CHECK(cached == fresh);
    }
  }
}

TEST_CASE("chat_template: one prompt fn renders correctly from many threads at "
          "once") {
  // The server renders on its HTTP worker threads, all through the ONE prompt
  // fn server_main builds, so the shared parsed template must be safe to render
  // concurrently. Each thread renders a different request, so a render that
  // leaked state into the shared tree would show up as a wrong prompt.
  const std::string tmpl = ReadFixture("qwen38_chat_template.jinja");
  auto fn = MakeChatTemplatePromptFn(tmpl, "", "<|im_end|>");
  const std::vector<TemplateRequest> requests = Qwen38Requests();
  std::vector<std::string> reference;
  for (const TemplateRequest& r : requests) reference.push_back(Render(fn, r));

  constexpr int kThreads = 8;
  constexpr int kRounds = 25;
  std::vector<int> mismatches(kThreads, 0);
  std::vector<int> errors(kThreads, 0);
  std::vector<std::thread> workers;
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&, t] {
      for (int round = 0; round < kRounds; ++round) {
        const size_t i = static_cast<size_t>(t + round) % requests.size();
        try {
          if (Render(fn, requests[i]) != reference[i]) ++mismatches[t];
        } catch (const std::exception&) {
          ++errors[t];
        }
      }
    });
  }
  for (std::thread& w : workers) w.join();
  for (int t = 0; t < kThreads; ++t) {
    CHECK(mismatches[t] == 0);
    CHECK(errors[t] == 0);
  }
}
