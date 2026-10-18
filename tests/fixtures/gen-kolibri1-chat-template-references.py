#!/usr/bin/env python3
"""Reference capture for the Kolibri-1 chat template — through the PINNED
Transformers renderer, not plain Jinja2.

Renders the chat_template committed at
tests/fixtures/kolibri1-chat-template-tokenizer_config.json (the
chat_template key of /mnt/models/Aleph-Alpha/Kolibri-1/tokenizer_config.json,
checkpoint pin of oracle aleph-alpha-inference 049a6a7bd240) with
transformers' own chat-template path:
transformers.utils.chat_template_utils.render_jinja_template, the function
apply_chat_template delegates to after kwarg resolution. Its
_compile_jinja_template installs the tojson override
(chat_template_utils.py:481, pin transformers 5.14.1):

    def tojson(x, ensure_ascii=False, indent=None, separators=None,
               sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

so the references preserve INSERTION key order, keep Unicode raw
(ensure_ascii=False) and never HTML-escape — plain jinja2's built-in tojson
sorts keys and escapes HTML, which is NOT the serving reference. Measured
against the pin on 2026-10-09 with a probe over an insertion-ordered
{"z": 1, "a": 2}, nested unsorted tool schemas and Unicode/HTML leaves; see
docs/bench-evidence/kolibri1-serve-20261008.md ("Review repair").

The output is the fixture the C++ rendering gate compares against
 byte-for-byte. Both the input and the output live in this directory, so the
generator runs on a clean checkout with no model directory and no /tmp
destination.

Run:  python3 tests/fixtures/gen-kolibri1-chat-template-references.py
(the script asserts the pinned transformers==5.14.1 and records jinja2's
version in the provenance).
"""

import hashlib
import json
import pathlib
import sys

import jinja2
import transformers
from transformers.utils.chat_template_utils import render_jinja_template

# The oracle pin .agents/oracles/transformers.md records inside the pinned
# vLLM environment. Refuse to capture against anything else: the references
# are only a reference if they come from the pinned renderer.
PINNED_TRANSFORMERS = "5.14.1"
if transformers.__version__ != PINNED_TRANSFORMERS:
    sys.exit(
        f"refusing to capture references with transformers "
        f"{transformers.__version__}: the pin is {PINNED_TRANSFORMERS} "
        f"(.agents/oracles/transformers.md)"
    )

HERE = pathlib.Path(__file__).resolve().parent
TEMPLATE_CONFIG_PATH = HERE / "kolibri1-chat-template-tokenizer_config.json"
OUT_PATH = HERE / "kolibri1_chat_template_references.json"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

# Deliberately UNSORTED at every level (type before function; name before
# description before parameters; type before properties before required;
# zeta_city before alpha_unit before city) with Unicode and HTML characters
# in the leaves: this is the case that distinguishes the pinned renderer
# (insertion order, raw UTF-8, no HTML escaping) from plain jinja2's
# built-in tojson (sorted keys, \uXXXX escaping, HTML escaped).
TOOLS_UNSORTED = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Wetter für <b>München</b> & <i>Berlin</i> ☕ "
                           '"quoted" \'apos\'',
            "parameters": {
                "type": "object",
                "properties": {
                    "zeta_city": {"type": "string",
                                  "description": "Zeta first <hr>"},
                    "alpha_unit": {"type": "string",
                                   "enum": ["celsius", "fahrenheit ☀"]},
                    "city": {"type": "string",
                             "description": "line1\nline2 <tag>"},
                },
                "required": ["zeta_city", "city"],
            },
        },
    }
]

# A MINIMAL tool: no description. Pinned vLLM renders it with
# `"description": null` (model_dump keeps absent fields), so this scenario
# pins that edge of the pinned renderer's tool shape.
TOOLS_MINIMAL = [
    {
        "type": "function",
        "function": {
            "name": "ping",
            "parameters": {
                "type": "object",
                "properties": {
                    "zeta": {"type": "string"},
                    "alpha": {"type": "integer"},
                },
                "required": ["alpha"],
            },
        },
    }
]

USER = {"role": "user", "content": "What is the weather in Berlin?"}
USER_UNICODE = {
    "role": "user",
    "content": "Wetter in München? <b>Berlin</b> & 'quotes' \"dquotes\" ☕ äöü",
}
ASSISTANT_REASONING = {
    "role": "assistant",
    "content": "It is sunny.",
    "reasoning": "The user asks about Berlin weather.",
}
ASSISTANT_THINK_IN_CONTENT = {
    "role": "assistant",
    "content": "<think>\nBerlin is in Germany.\n</think>\nIt is sunny.",
}
TOOL_MSG = {"role": "tool", "content": "22C, clear"}
# Historical assistant tool call whose arguments ride as the OpenAI wire
# STRING, unsorted and carrying Unicode/HTML. The generator parses the string
# to a dict before rendering, mirroring pinned vLLM's _postprocess_messages
# (chat_utils.py:2032-2075: json.loads, so the template's
# `tool_call.arguments | tojson` branch sees a structured value and the
# pinned renderer re-dumps it in document order).
ASSISTANT_TOOL_CALL = {
    "role": "assistant",
    "content": "Let me check the weather.",
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"zeta_city\": \"München <b>rocks</b> & more\", "
                             "\"alpha_unit\": \"celsius ☀\", "
                             "\"city\": \"Berlin\"}",
            },
        }
    ],
}

SCENARIOS = [
    # (name, messages, add_generation_prompt, kwargs, tools)
    ("default_user_only", [USER], True, {}, None),
    ("enable_thinking_false", [USER], True, {"enable_thinking": False}, None),
    ("reasoning_effort_none", [USER], True, {"reasoning_effort": "none"}, None),
    ("reasoning_effort_low", [USER], True, {"reasoning_effort": "low"}, None),
    ("reasoning_effort_medium", [USER], True, {"reasoning_effort": "medium"}, None),
    ("reasoning_effort_high", [USER], True, {"reasoning_effort": "high"}, None),
    # Effort wins over an explicit enable_thinking:false (reasoning.py:44-48).
    (
        "effort_low_overrides_enable_thinking_false",
        [USER],
        True,
        {"reasoning_effort": "low", "enable_thinking": False},
        None,
    ),
    ("effort_none_overrides_enable_thinking_default", [USER], True,
     {"reasoning_effort": "none", "enable_thinking": True}, None),
    ("with_tools", [USER], True, {}, TOOLS),
    ("with_tools_thinking_off", [USER], True, {"enable_thinking": False}, TOOLS),
    ("system_and_user", [
        {"role": "system", "content": "You are helpful."}, USER], True, {}, None),
    ("assistant_turn_preserved", [USER, ASSISTANT_REASONING, USER], False, {},
     None),
    ("assistant_think_in_content", [USER, ASSISTANT_THINK_IN_CONTENT, USER],
     False, {}, None),
    ("tool_response_turn", [USER, ASSISTANT_REASONING, TOOL_MSG], True, {},
     None),
    ("no_generation_prompt", [USER], False, {}, None),
    # Review-repair additions (PR #3422 review P1): the tojson comparison set
    # carries unsorted nested tool schemas and Unicode/HTML characters.
    ("with_tools_unsorted_unicode", [USER_UNICODE], True, {}, TOOLS_UNSORTED),
    (
        "with_tools_unsorted_unicode_thinking_off",
        [USER_UNICODE],
        True,
        {"enable_thinking": False},
        TOOLS_UNSORTED,
    ),
    ("with_tools_minimal_no_description", [USER], True, {}, TOOLS_MINIMAL),
    (
        "assistant_tool_call_unsorted_arguments",
        [USER, ASSISTANT_TOOL_CALL, USER],
        True,
        {},
        None,
    ),
    ("unicode_html_user_message", [USER_UNICODE], True, {}, None),
]


def model_dump_tool(tool):
    """Mirror pinned vLLM's `[tool.model_dump() for tool in request.tools]`
    (online_renderer.py:178): pydantic field order type/function,
    name/description/parameters, with `description` and `parameters` present
    as null when the request omitted them (only strict/defer_loading are
    popped by the model serializer). `parameters` keeps the request
    document's key order."""
    fn = tool.get("function", {})
    return {
        "type": tool.get("type", "function"),
        "function": {
            "name": fn.get("name"),
            "description": fn.get("description"),
            "parameters": fn.get("parameters"),
        },
    }


def render(template, messages, add_generation_prompt, kwargs, tools):
    # The pinned renderer's own path: render_jinja_template compiles the
    # template through _compile_jinja_template (tojson override installed)
    # and renders with transformers' whitespace policy
    # (trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False),
    # the policy src/vllm/entrypoints/chat_template.cpp mirrors.
    rendered, _ = render_jinja_template(
        [messages],
        tools=[model_dump_tool(t) for t in tools] if tools else None,
        chat_template=template,
        add_generation_prompt=add_generation_prompt,
        # The C++ adapter binds bos_token/eos_token (empty for this gate);
        # the template references neither, so binding them is unobservable.
        bos_token="",
        eos_token="",
        **kwargs,
    )
    return rendered[0]


def main():
    config = json.loads(TEMPLATE_CONFIG_PATH.read_text())
    template = config["chat_template"]
    sha = hashlib.sha256(template.encode()).hexdigest()
    recorded = config.get("fixture_provenance", {}).get("chat_template_sha256")
    if recorded is not None and recorded != sha:
        sys.exit(
            f"template sha256 {sha} does not match the fixture's recorded "
            f"{recorded}: recapture the fixture config first"
        )
    cases = []
    for name, messages, agp, kwargs, tools in SCENARIOS:
        # Mirror pinned vLLM's _postprocess_messages: assistant tool-call
        # arguments ride as the wire string; the template must see the parsed
        # dict (chat_utils.py:2032-2075).
        render_messages = json.loads(json.dumps(messages))
        for m in render_messages:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls", []):
                    args = tc.get("function", {}).get("arguments")
                    if isinstance(args, str):
                        tc["function"]["arguments"] = json.loads(args)
        cases.append({
            "name": name,
            "messages": messages,
            "add_generation_prompt": agp,
            "chat_template_kwargs": kwargs,
            "tools": tools,
            "expected": render(template, render_messages, agp, kwargs, tools),
        })
    OUT_PATH.write_text(json.dumps({
        "provenance": (
            "chat_template from tests/fixtures/"
            "kolibri1-chat-template-tokenizer_config.json (sha256 " + sha +
            " of the template string; originally the chat_template key of "
            "/mnt/models/Aleph-Alpha/Kolibri-1/tokenizer_config.json, "
            "checkpoint pin of oracle aleph-alpha-inference 049a6a7bd240); "
            "rendered through the PINNED transformers " +
            PINNED_TRANSFORMERS + " renderer "
            "(transformers.utils.chat_template_utils.render_jinja_template, "
            "the function apply_chat_template delegates to; its "
            "_compile_jinja_template installs the tojson override at "
            "chat_template_utils.py:481 with sort_keys=False and "
            "ensure_ascii=False — insertion key order, raw UTF-8, no HTML "
            "escaping), with jinja2 " + jinja2.__version__ + "; captured by "
            "tests/fixtures/gen-kolibri1-chat-template-references.py, which "
            "asserts the transformers pin; the C++ gate compares minja's "
            "rendering (src/vllm/entrypoints/chat_template.cpp) "
            "byte-for-byte against these references"
        ),
        "cases": cases,
    }, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_PATH} with {len(cases)} cases "
          f"(transformers {transformers.__version__}, jinja2 {jinja2.__version__})")


if __name__ == "__main__":
    main()
