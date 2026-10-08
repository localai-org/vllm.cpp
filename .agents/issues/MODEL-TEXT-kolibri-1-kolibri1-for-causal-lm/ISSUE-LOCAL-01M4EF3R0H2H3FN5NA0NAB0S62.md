ID: ISSUE-LOCAL-01M4EF3R0H2H3FN5NA0NAB0S62
Title: kolibri1 serving completion: chat template rendering plus kolibri1 reasoning and tool-call parsers
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: OPEN
Kind: enhancement
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-08
Updated: 2026-10-08
Closed: -

## Problem

The kolibri1 CPU arm reaches the model forward but not serving: the OpenAI chat path has no kolibri1 reasoning parser, no kolibri1 tool-parser alias, and no template-detection rows, so the model-author plugin's serving recipe (aleph-alpha-inference pin 049a6a7bd240: reasoning parser kolibri1 = Qwen3 grammar with the template's thinking switch derived from chat_template_kwargs reasoning_effort/enable_thinking; tool parser kolibri1 = the Hermes <tool_call> format; chat template shipped in tokenizer_config.json) cannot be served end to end. Scope: port the thinking_enabled switch and the engine-backed kolibri1 reasoning adapter, alias the kolibri1 tool parser to hermes, add template-marker detection rows, gate template rendering against jinja2 reference outputs on kolibri1 goldens. No tokenizer change, no behavior change to other models' parsers.

## Resolution

-
