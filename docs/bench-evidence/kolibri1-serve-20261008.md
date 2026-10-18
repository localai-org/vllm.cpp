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

**SUPERSEDED 2026-10-09 by the "Review repair" section below:** the references
were captured with plain jinja2, whose built-in tojson sorts keys and escapes
HTML — NOT the pinned Transformers renderer's tojson, which preserves
insertion order and never HTML-escapes. The regenerated references (20
scenarios, through the pinned renderer) are the ones the gate compares
against.

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

**SUPERSEDED 2026-10-09 by the "Review repair" section below.** The claim that
"the tojson every transformers/vLLM-served template runs under" sorts keys was
measured against plain jinja2, not the pinned Transformers renderer — and the
pinned renderer overrides Jinja's tojson with `sort_keys=False`
(transformers 5.14.1 `utils/chat_template_utils.py:481`), so the override
below rendered prompt bytes no serving reference produces. See "Review
repair" for the measurement, the corrected adapter and the regenerated
fixtures.

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
- Row battery (ctest, this host): `test_kolibri1`, `test_kolibri1_tt`,
  `test_kolibri1_tt_b2i`, `test_kolibri1_tt_b2bi`, `test_kolibri1_dequant`,
  `test_kolibri1_dequant_cache`, `test_kolibri1_moe_glue`, `test_kolibri1_w2`,
  `test_kolibri1_decode_bench` — all PASS.
- `test_kolibri1_w3` PASS in a verified quiet window: 2026-10-09 01:07 CEST,
  211 GB free at start, `ps`-verified zero other `test_kolibri1` processes,
  `VLLM_CPP_CPU_THREADS=8`, 539.95 s. Two earlier attempts were ABORTED BY
  CONTENTION, not by the change: the first fired while a peer was building,
  the second was OOM-killed at 75977088 kB anon RSS after a peer's
  `test_kolibri1_decode_bench` started mid-run (dmesg
  `oom-kill ... task=test_kolibri1_w`). The quiet-window check is what
  serializes W3 on this host.

Pre-existing, not touched by this change (verified on the same tree):
`ctest`-cwd artifacts from `-ffile-prefix-map` (`test_linear_scaling_rope`
passes when run from the source dir), the `test_safetensors` RSS-mapping
assertion under host memory pressure, and the gliner fixture loads
(`test_gliner2_e2e`, `test_capi` v27).

## Review repair (2026-10-09, PR #3422 review blockers P1/P2)

The review (localai-org-maint-bot) blocked the landing with two findings:

- **P1:** the adapter's global `tojson` override sorted object keys for every
  model, but the pinned Transformers renderer overrides Jinja's filter with
  `tojson(..., sort_keys=False)` and installs it in `_compile_jinja_template`
  (transformers 5.14.1 `utils/chat_template_utils.py:481`), so plain Jinja's
  default is not the serving behavior — and the fixture generator used plain
  `jinja2.Environment`, certifying the same incorrect reference.
- **P2:** `tests/vllm/entrypoints/test_kolibri1_chat_template.cpp` loaded
  `/mnt/models/Aleph-Alpha/Kolibri-1/tokenizer_config.json` in both the
  rendering and detection cases, and the generator hard-coded its output under
  `/tmp/vllm-kolibri-serve` — a clean checkout could not run the gate.

### What the pinned renderer actually does (MEASURED)

The pinned oracle is `transformers` 5.14.1 (`.agents/oracles/transformers.md`,
the version the pinned vLLM environment resolves), run in a venv with
`jinja2` 3.1.6. Probe: `transformers.utils.chat_template_utils.
render_jinja_template` (the function `apply_chat_template` delegates to, whose
`_compile_jinja_template` installs the override) over an insertion-ordered
`{"z": 1, "a": 2}`, nested unsorted tool schemas, and Unicode/HTML leaves.
Command: `/tmp/tfprobe-venv/bin/python /tmp/probe_tojson.py` (and
`/tmp/probe_tojson2.py` for the option/edge semantics).

- **Default:** `{"z": 1, "a": 2}` — INSERTION order (`sort_keys=False`), raw
  UTF-8 (`ensure_ascii=False`), NO HTML escaping (`<b>`, `&` stay raw — Jinja's
  builtin escapes them as `\u003c`/`\u0026`, which is exactly what the
  transformers override exists to avoid), Python `json.dumps` separators
  (`", "` / `": "`), control characters as `\u00xx`, `"`/`\` escaped, `/` raw.
- **Options (all four accepted):** `indent=2` pretty-prints (`{\n  "z": 1,\n
  "a": 2\n}`, empty containers stay `{}`/`[]`, `indent=0`/`-1` newline with no
  spaces, a string indent is the per-level prefix); `sort_keys=True` sorts
  recursively; `ensure_ascii=True` escapes non-ASCII as `\uXXXX` with surrogate
  pairs; `separators=(',', ':')` replaces the defaults (still between the
  newlines when indent is set).
- Plain jinja2's built-in tojson (the OLD reference): sorted keys, `\uXXXX`,
  HTML-escaped — different bytes on every multi-key object — and it rejects the
  four options (`TypeError: do_tojson() got an unexpected keyword argument`).
- minja's builtin tojson matches the pinned DEFAULT byte-for-byte (ordered_map
  objects, nlohmann dump: raw UTF-8, no HTML escape, Python's separators and
  indent shape) but accepts only `indent`; the other three options raise
  "Unknown argument".

Pinned vLLM's tool shape was also measured (replication of
`online_renderer.py:178` `[tool.model_dump() for tool in request.tools]` with
the pinned `ChatCompletionToolsParam`/`FunctionDefinition` pydantic models):
fixed field order `type`/`function`, `name`/`description`/`parameters`, with
`description` and `parameters` present as **null** when the request omitted
them, and `parameters` keeping the request document's key order.

### What changed

- `src/vllm/entrypoints/chat_template.cpp`: the sorted-dump `tojson` override
  is replaced by a port of the pinned filter's FULL signature
  (`ensure_ascii`/`indent`/`separators`/`sort_keys`, CPython `json.dumps`
  semantics — see the `JsonDumps` helpers and the filter's comment). The
  default now keeps insertion order; the four options render the same bytes
  as the pin. `BuildTools` mirrors the measured `model_dump` shape (null
  `description`/`parameters` when absent) and assigns `parameters` directly.
- `include/vllm/entrypoints/openai/protocol.h` + `protocol.cpp`:
  `FunctionDefinition::parameters` is `nlohmann::ordered_json`
  (order-preserving; `nlohmann::json` sorts at parse), with an ordered
  `from_json` overload and `RestoreToolSchemaOrder`, which re-reads `tools`
  from an order-preserving body parse.
- Entry points call it after their regular parse: `api_server.cpp`
  (`handle_chat_completions`), `src/capi/vllm_c.cpp` (`ParseChatRequest`,
  both `vllm_chat` and `vllm_chat_stream`), `run_batch.cpp`. Tool-less requests
  pay nothing; the parsers that only look keys up are untouched (behavior
  identical).
- `src/vllm/entrypoints/openai/tool_parsers/step3.cpp`: one pointer type
  follows the field (`const nlohmann::ordered_json*`).
- Fixtures REGENERATED through the pinned renderer:
  `tests/fixtures/gen-kolibri1-chat-template-references.py` now drives
  `render_jinja_template` (asserting `transformers==5.14.1`), reads the
  template from the committed fixture config, mirrors vLLM's
  `_postprocess_messages` (assistant tool-call arguments parsed to dicts) and
  `model_dump` tool shape, and writes in-tree next to itself. The reference
  fixture has 20 scenarios (was 15): the 15 original plus
  `with_tools_unsorted_unicode`, `with_tools_unsorted_unicode_thinking_off`
  (unsorted nested tool schemas + Unicode/HTML), `with_tools_minimal_no_description`
  (pins `"description": null`), `assistant_tool_call_unsorted_arguments`
  (the template's second tojson site, `tool_call.arguments | tojson`, with
  unsorted Unicode/HTML arguments), and `unicode_html_user_message`.
- P2 portability: the template input is committed at
  `tests/fixtures/kolibri1-chat-template-tokenizer_config.json` (the
  checkpoint's `chat_template`, sha256
  `9ba35d4bd6baa26b66aa75d03a922dfee98b16bb1fa37481b195d247267b0f97` recorded in
  the fixture's `fixture_provenance`). The test loads it (and the references)
  from `KOLIBRI1_TEMPLATE_FIXTURE_DIR` (= `tests/fixtures` at compile time);
  no `/mnt` path remains in any load path — proven by running the test from
  `/` with both fixtures copied to `/tmp/p2-proof`: 61/61 PASS with the model
  directory absent.
- Non-kolibri guard (`tests/vllm/entrypoints/test_chat_template.cpp`, +8
  assertions): a non-kolibri tool template (the Hermes/Qwen-style tool branch
  AND the real Qwen3.5 fixture template) renders an insertion-ordered,
  Unicode/HTML tool byte-identical to the pinned renderer; the four tojson
  options are pinned byte-for-byte against the pin; an already-alphabetical
  schema renders byte-identical before/after the repair (the change alters
  only what the pinned renderer actually differs on).

### Red-first (under the corrected reference, before the fix)

`test_kolibri1_chat_template` against the regenerated 20-scenario fixture with
the UNMODIFIED adapter: 6 of 20 rendering scenarios FAILED — `with_tools`,
`with_tools_thinking_off`, `with_tools_unsorted_unicode`,
`with_tools_unsorted_unicode_thinking_off`, `with_tools_minimal_no_description`,
`assistant_tool_call_unsorted_arguments` (40 assertions, 34 passed / 6
failed). The rendered bytes were fully sorted
(`{"function": {"description": ..., "name": ..., "parameters": {"properties":
..., "required": ..., "type": ...}}, "type": "function"}`) where the pinned
renderer emits insertion order. Non-tool scenarios already matched, which
isolated the divergence to `tojson` (and, for the minimal tool, the
model_dump null shape).

### Gates (this tree, after the repair)

- `test_reasoning_kolibri1` 43, `test_tool_parser_kolibri1` 19,
  `test_kolibri1_chat_template` 61 (20 scenarios + switch-boundary +
  detection + fixture REQUIREs), `test_chat_template` 204 (196 pre-existing +
  8 new guard assertions), `test_reasoning_parser_detect` 75,
  `test_tool_parser_detect` 361, `test_reasoning_qwen3` 164,
  `test_openai_tool_parsers` 64 — all PASS.
- `test_kolibri1` 27/234 PASS; `test_kolibri1_decode_bench` default config
  PASS (anchor chain `101807, 109726, …`, last token 109726);
  `test_kolibri1_dequant` 10, `test_kolibri1_dequant_cache` 52,
  `test_kolibri1_moe_glue` 21, `test_kolibri1_w2` 1608 — all PASS.
- Protocol/serving suites green: `test_openai_serving` 1365,
  `test_openai_api_server` 1517, `test_openai_conformance` 252,
  `test_parser_engine_assembly` 5038,
  `test_openai_api_server_dots3_mm_forward` 16499, `test_openai_run_batch` 16,
  `test_input_batch` 232; the parameters-reading tool-parser suites
  (`test_deepseek_v32` 143, `test_glm47` 60, `test_minimax_m2_tool` 56,
  `test_tool_parser_step3` 31, `test_tool_parser_step3p5` 155,
  `test_tool_parser_qwen3_coder` 166, `test_tool_parser_minicpm5` 119,
  `test_tool_parser_hy_v3` 40, `test_tool_parser_poolside_v1` 25,
  `test_tool_parser_gemma4` 161) all PASS — the ordered field is
  behavior-preserving for key-lookup readers.
- `test_capi` 710/711: the one failure is `capi v27` (gliner fixture load),
  the pre-existing host failure recorded above, untouched by this change.
- W3 (`test_kolibri1_w3`) NOT rerun: no forward change; the serving path
  renders prompts that feed the bench, and the bench's own gate is green.
- Full tree builds clean (881 targets) with the `protocol.h` change;
  `scripts/check-agent-record.py` OK (three `engine-matrix.md` api_server.cpp
  anchors shifted by the +5-line `RestoreToolSchemaOrder` call were repaired:
  1354→1359, 1365→1370, 1614→1619).

## Re-review repair (2026-10-09, PR #3422 scoped re-review + the operator's clean-head check)

The operator's clean-head re-verification at the review-repair head
`d93d9d492` found the repair itself RED, and the "Gates (this tree, after
the repair)" list above is corrected by this section: **`test_openai_run_batch`
16 was a STALE BINARY.** At `d93d9d492` the suite fails 3 of 7 cases —
`test_run_batch.cpp:451`, `:507`, `:601` all throw
`[json.exception.type_error.302] type must be string, but is object` — while
the doctest summary still reads `assertions: 16 | 16 passed`, which is what
the repair's gate report counted. The PR's CI lane (build-test-cpu runs the
full ctest) would be red. ISSUE-LOCAL-01M4FR20MES4HQVBRWJBJ2AVCN.

**The crash.** The repair wired `RestoreToolSchemaOrder(body, request)` into
`RunBatch::DispatchChat` (run_batch.cpp:110), but the seam's first parameter
is the body TEXT (`const std::string&`, protocol.h:605) and `body` is the
parsed `nlohmann::json` object; the implicit `get<std::string>()` conversion
throws on every object body, and `DispatchChat` has no try/catch around the
call, so every object-body chat line through `RunBatch` throws. The other two
call sites pass the raw body string and were already correct
(api_server.cpp:400, vllm_c.cpp:518).

**The fix** (commit `a6679a82a`). The original request text IS available:
`RunLine(const std::string& request_json)` parses it and dropped the text.
`RunLine` now re-serializes the chat body (the line's top-level `body`
member) from an ORDER-PRESERVING `nlohmann::ordered_json` parse of the
original line text and threads it through
`DispatchChat(custom_id, body, body_json)`; `DispatchChat` calls
`RestoreToolSchemaOrder(body_json, request)`. `ordered_json::dump` keeps
insertion order, so the re-read restores the request document's schema key
order into the prompt, matching the other entry points. A first in-flow
attempt (`body.dump()`) was REVERTED: `body` is the key-sorted
`nlohmann::json`, so its dump is already sorted and the order restoration
reads sorted text — the crash goes away but the seam no-ops and the batch
path loses the document's schema key order.

**Red-first / mutation evidence** (`/tmp/build-kolibri-serve`, this host):
- Committed throwing call + the new F1 test: 4 of 8 cases throw
  type_error.302 (the 3 pre-existing + the new case); assertions read
  `16 | 16 passed` while 4 cases fail — the stale-binary signature.
- `body.dump()` mutation: the 3 pre-existing cases go GREEN but the new F1
  case fails BOTH key-order assertions (the prompt carries the key-sorted
  schema) — the seam no-op is detected.
- The fix: 8/8 cases, 89/89 assertions, SUCCESS; the F1 case's prompt carries
  the document-ordered schema.

**F1 (coverage gap)** — ISSUE-LOCAL-01M4FR1D7RH7CN5X2R2CAR6N61: the
ordered-parameters seam was detected by NO committed test. Two entry-point
tests now pin it (commit `2f535c002a`):
- `test_run_batch.cpp`, new case "run_batch: a tools line keeps the
  document's schema key order in the prompt": drives `RunBatch::RunLine` with
  a RAW JSONL line whose nested chat body is non-alphabetical at every level
  (the raw string is load-bearing — `BatchLine()` would re-serialize through
  the key-sorting `nlohmann::json` and destroy the order under test). Asserts
  (a) no exception + a 200 row and (b) the prompt seam receives the schema in
  the request document's key order, not the sorted form, captured via a
  `CapturingToolsPrompt` seam (the capture pattern the api_server harness
  uses).
- `test_api_server.cpp`, new case "api_server: an unsorted tool schema keeps
  its document key order in the rendered prompt": posts an unsorted-schema
  tools request through the PRODUCTION `/v1/chat/completions` dispatch with
  the real Qwen3.8 fixture template and asserts the rendered prompt bytes.
  Mutation-proven: with `RestoreToolSchemaOrder` no-op'd at api_server.cpp:400
  the case fails both key-order assertions (RED); restored, green.

**F2 (mislabeled guard)** — same issue: the "byte-stable for already-ordered
schemas" case's `AlphabeticalTool` fixture was NOT fully alphabetical, so
its before/after byte-identity claim was not demonstrated. The fixture is now
genuinely alphabetical at every level (`properties` < `required` < `type`;
single-property schema) and the case renders the schema alone
(`{{ tools[0].function.parameters | tojson }}`) because the tool wrapper's
`model_dump` field order is not alphabetical. Mutation-proven: with the
`tojson` default mutated to sorted keys (the override the repair removed) the
insertion-order case goes RED while this byte-stable case stays GREEN — the
agreement it claims to pin. `test_chat_template` 45/45 cases, 204/204
assertions.

**Gates (this tree, at the FIXED head, CPU-only build,
/tmp/build-kolibri-serve, 2026-10-09):** `test_openai_run_batch` 8/8 cases /
89/89 assertions (7 pre-existing + the new F1 case); `test_openai_api_server`
104/104 / 1521/1521 (1517 pre-existing + 4 new); `test_openai_serving`
48/48 / 1365/1365; `test_chat_template` 45/45 / 204/204;
`test_kolibri1_chat_template` 4/4 / 61/61; `test_reasoning_kolibri1` 7/7 /
43/43; `test_tool_parser_kolibri1` 3/3 / 19/19;
`test_reasoning_parser_detect` 8/8 / 75/75; `test_tool_parser_detect` 16/16 /
361/361; `test_kolibri1` 27/27 / 234/234; `test_kolibri1_decode_bench` 1/1 /
2/2 (anchor chain `101807, 109726, …`, last token 109726) — all PASS.
W3 (`test_kolibri1_w3`) NOT rerun: no forward change.
`scripts/check-agent-record.py` OK (ANCHOR-ROT=0);
`scripts/agent-preflight.sh --staged` exits 0.

## Out of scope

- The oracle gateability measurement (`vllm serve` with the plugin on a GPU
  lease) — owed by the oracle file, unchanged.
- chat_template_kwargs threading for the OTHER engine-backed reasoning
  parsers (the pre-existing W4 note in `reasoning_parsers/abstract.cpp`);
  kolibri1 carries its own per-request derivation because the plugin's
  parser is defined by it.
- fp8 KV cache / 1M-context serving recipe (loader/config rows).
