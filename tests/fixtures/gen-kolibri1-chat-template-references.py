#!/usr/bin/env python3
"""Reference capture for the Kolibri-1 chat template.

Renders the chat_template shipped in
/mnt/models/Aleph-Alpha/Kolibri-1/tokenizer_config.json with CPython Jinja2
under transformers' whitespace policy (trim_blocks=True, lstrip_blocks=True,
keep_trailing_newline=False), which is the same policy the vendored minja
engine mirrors (src/vllm/entrypoints/chat_template.cpp). The output is the
fixture the C++ rendering gate compares against.

Provenance: template text from the pinned checkpoint tokenizer_config.json;
behavior oracle aleph-alpha-inference @ 049a6a7bd240 (the plugin serves this
checkpoint; reasoning.py:36 thinking_enabled documents the same switch the
template encodes).
"""

import json
import pathlib

import jinja2

TEMPLATE_PATH = pathlib.Path(
    "/mnt/models/Aleph-Alpha/Kolibri-1/tokenizer_config.json"
)
OUT_PATH = pathlib.Path(
    "/tmp/vllm-kolibri-serve/tests/fixtures/kolibri1_chat_template_references.json"
)

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

USER = {"role": "user", "content": "What is the weather in Berlin?"}
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
]


def render(template, messages, add_generation_prompt, kwargs, tools):
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                             keep_trailing_newline=False)
    context = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
        "bos_token": "",
        "eos_token": "",
        "tools": tools if tools else [],
    }
    context.update(kwargs)
    return env.from_string(template).render(**context)


def main():
    template = json.loads(TEMPLATE_PATH.read_text())["chat_template"]
    cases = []
    for name, messages, agp, kwargs, tools in SCENARIOS:
        cases.append({
            "name": name,
            "messages": messages,
            "add_generation_prompt": agp,
            "chat_template_kwargs": kwargs,
            "tools": tools,
            "expected": render(template, messages, agp, kwargs, tools),
        })
    OUT_PATH.write_text(json.dumps({
        "provenance": (
            "chat_template from /mnt/models/Aleph-Alpha/Kolibri-1/"
            "tokenizer_config.json (checkpoint pin of oracle "
            "aleph-alpha-inference 049a6a7bd240); rendered with CPython jinja2 "
            "3.1.6 under transformers' whitespace policy "
            "(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline="
            "False), the policy src/vllm/entrypoints/chat_template.cpp "
            "mirrors; captured by tests/fixtures/"
            "gen-kolibri1-chat-template-references.py"
        ),
        "cases": cases,
    }, indent=1) + "\n")
    print(f"wrote {OUT_PATH} with {len(cases)} cases")


if __name__ == "__main__":
    main()
