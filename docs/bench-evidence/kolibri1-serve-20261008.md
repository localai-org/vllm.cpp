# kolibri1 serving completion — evidence (2026-10-08)

Row `MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm`, branch `row/kolibri-serve`
(base `bf7b654ae`). Issue: ISSUE-LOCAL-01M4EF3R0H2H3FN5NA0NAB0S62.

## Oracle and what was mirrored

Primary serving oracle: `aleph-alpha-inference`, pin `049a6a7bd240`
(PyPI v1.0.0), read from a git clone of
`https://github.com/Aleph-Alpha/aleph-alpha-inference` checked out at the
pin (`git log -1` = `049a6a7 chore(main): release 1.0.0 (#5)`).

| Plugin anchor | What it defines | Local mirror |
|---|---|---|
| `aleph_alpha_inference/reasoning.py:36-48` (`thinking_enabled`) | the thinking switch: a non-None `reasoning_effort` wins, only `"none"` disables; else a literal `enable_thinking == False` disables | `Kolibri1ThinkingEnabled` in `src/vllm/entrypoints/openai/reasoning_parsers/kolibri1.cpp` |
| `reasoning.py:51-58` (`Kolibri1Parser`) | the Qwen3 grammar with the starting state chosen like the template | `Kolibri1ParserReasoningAdapter` over `vllm::parser::Qwen3Parser` (`pe::qwen3_config`), state derived per request from `chat_template_kwargs` |
| `reasoning.py:61` (`Kolibri1ParserReasoningAdapter`) | the engine-backed reasoning face | same-named adapter class, registered under `"kolibri1"` in `reasoning_parsers/abstract.cpp` |
| `__init__.py:50-54` | tool parser `kolibri1` = `vllm.tool_parsers.hermes_tool_parser.Hermes2ProToolParser` verbatim ("shares the Hermes `<tool_call>` format") | alias branch `"kolibri1" -> HermesToolParser` in `tool_parsers/abstract.cpp` + name in `tool_parser_names()` |
| `__init__.py:44-48` | reasoning parser registered under the name `kolibri1` | name in `reasoning_parser_names()`; detection rows resolve to it |
| `README.md:32-40` | the serving recipe (`--reasoning-parser kolibri1 --tool-call-parser kolibri1`) | both names now resolve; `--tool-call-parser auto` also detects kolibri1 from the template |

The plugin ships NO chat template of its own: the template rides in the
checkpoint's `tokenizer_config.json` (`/mnt/models/Aleph-Alpha/Kolibri-1`),
and `reasoning.py:3-26` documents the exact switch it encodes. There is no
plugin-vs-HF template discrepancy to record — the plugin serves the HF
template.

## Reference capture

- Script: `tests/fixtures/gen-kolibri1-chat-template-references.py`.
- Fixture: `tests/fixtures/kolibri1_chat_template_references.json` (15
  scenarios: default, `enable_thinking:false`, `reasoning_effort`
  none/low/medium/high/minimal/xhigh/max, effort-over-`enable_thinking`
  both ways, tools on/off, system turn, preserved assistant reasoning,
  `</think>`-embedded content, tool-response turn, no generation prompt).
- Method: CPython jinja2 3.1.6 (system python3) rendering the checkpoint's
  template text under transformers' whitespace policy (trim_blocks=True,
  lstrip_blocks=True, keep_trailing_newline=False) — the policy the minja
  adapter mirrors. The plugin itself could not be executed: it imports vLLM
  at registration, and the oracle file records `gateable = no` (the GPU
  measurement is owed); the plugin's PARSER behavior is mirrored from its
  source, and the template behavior from the jinja2 stack it serves through.

## Renderer divergence found and fixed

`{{ tool | tojson }}` rendered with minja's insertion-order dump; the jinja2
references sort keys (jinja2 `DEFAULT_POLICIES["json.dumps_kwargs"]` =
`{"sort_keys": True}` — the tojson every transformers/vLLM-served template
runs under). Fixed adapter-side in `src/vllm/entrypoints/chat_template.cpp`:
a child-scope `tojson` global (same `value`/`indent` signature) that sorts
object keys recursively before dumping. No vendor change; request kwargs can
still not shadow it (the builtins set already refuses `tojson`).

## Red-first

Captured before any implementation existed (build of the three new test
targets against the unmodified tree):

- `test_reasoning_kolibri1`: 7/7 cases FAILED (registry had no
  `kolibri1` reasoning parser).
- `test_tool_parser_kolibri1`: 3/3 FAILED (no `kolibri1` tool name).
- `test_kolibri1_chat_template`: detection resolved `think_auto` (reasoning)
  and `hermes` (tool) off the real template; the two rendering cases that
  involve `tojson` failed byte-compares. Non-tool rendering cases already
  matched jinja2, which isolated the divergence to `tojson`.

## Gates (this tree, CPU-only build, /tmp/build-kolibri-serve)

- `test_reasoning_kolibri1`: 43 assertions, PASS.
- `test_tool_parser_kolibri1`: 19 assertions, PASS.
- `test_kolibri1_chat_template`: 35 assertions, PASS (byte-exact vs jinja2 on
  all 15 scenarios; plus switch-boundary and detection cases).
- `test_reasoning_parser_detect`: 75 assertions PASS; `test_tool_parser_detect`:
  361 assertions PASS; registry count pins moved 12->13 (reasoning) and
  42->43 (tool) with dated comments.
- `test_reasoning_qwen3` 164 PASS, `test_openai_tool_parsers` 64 PASS,
  `test_chat_template` 196 PASS (no behavior change to other parsers).
- Row battery (ctest, this host): `test_kolibri1`, `test_kolibri1_dequant`,
  `test_kolibri1_dequant_cache`, `test_kolibri1_w2`, `test_kolibri1_w3`,
  `test_kolibri1_decode_bench`, `test_kolibri1_moe_glue` — all PASS
  (test_kolibri1_tt / _b2i / _b2bi PASS in the same run).
- W3 rerun in a verified quiet window (free > 110 GB, no other
  test_kolibri1_w3 process), `VLLM_CPP_CPU_THREADS=8`: PASS.

Pre-existing, not touched by this change (verified on the same tree):
`ctest`-cwd artifacts from `-ffile-prefix-map` (`test_linear_scaling_rope`
passes when run from the source dir), the `test_safetensors` RSS-mapping
assertion under host memory pressure, and the gliner fixture loads
(`test_gliner2_e2e`, `test_capi` v27).

## Out of scope

- The oracle gateability measurement (`vllm serve` with the plugin on a GPU
  lease) — owed by the oracle file, unchanged.
- chat_template_kwargs threading for the OTHER engine-backed reasoning
  parsers (the pre-existing W4 note in `reasoning_parsers/abstract.cpp`);
  kolibri1 carries its own per-request derivation because the plugin's
  parser is defined by it.
- fp8 KV cache / 1M-context serving recipe (loader/config rows).
