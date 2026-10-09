ID: ISSUE-LOCAL-01M4FR1D7RH7CN5X2R2CAR6N61
Title: Scoped re-review of PR #3422's repair: F1 the ordered-parameters seam (RestoreToolSchemaOrder) is detected by no committed test; F2 the 'byte-stable for already-ordered schemas' guard uses a non-alphabetical fixture
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: 2026-10-09

## Problem

Two findings from the scoped re-review of the PR #3422 review repair (branch row/kolibri-serve, head d93d9d492). F1 (coverage gap): RestoreToolSchemaOrder — the ordered-parameters seam called at src/vllm/entrypoints/openai/api_server.cpp:400 (handle_chat_completions), src/capi/vllm_c.cpp:518 (ParseChatRequest) and src/vllm/entrypoints/openai/run_batch.cpp:110 (DispatchChat) — is detected by NO committed test: a no-op passes every gate. The kolibri1 chat-template gate drives apply_chat_template directly over C++-constructed tools and never crosses the entry-point parse, so the seam that re-reads tools order-preserving after the key-sorting nlohmann::json parse is unguarded. OWED: a test that posts an unsorted-schema tools request through a chat entry point and asserts the rendered prompt bytes keep the request document's key order. F2 (mislabeled guard case): the 'byte-stable for already-ordered schemas' case in tests/vllm/entrypoints/test_chat_template.cpp uses an AlphabeticalTool fixture that is NOT fully alphabetical — the rendered tool is insertion-ordered non-alphabetically at type/function, name/description/parameters and type/properties/required — so under a sorted mutation the case goes red too and the 'alphabetical-schema byte-identity before/after' sub-claim is not demonstrated. OWED: make the fixture genuinely alphabetical at every level (or relabel the case to what it actually pins; prefer genuinely alphabetical so the byte-identity-before/after claim is real) and correct the comment.

The same re-review surfaced a CONFIRMED crash the repair introduced on the batch path (RestoreToolSchemaOrder(body, request) throws type_error.302 on every object body) — filed separately as ISSUE-LOCAL-01M4FR20MES4HQVBRWJBJ2AVCN, since the crash and the coverage gap are distinct defects with distinct fixes.

## Resolution

Both findings landed on row/kolibri-serve 2026-10-09 (test commit 2f535c002a1b15c8717a53a5e8103258ad5fd229, landed with the F2 guard and
these records in the follow-up records commit — see commit bodies).

F1 — two entry-point tests now detect a no-op RestoreToolSchemaOrder, one per affected entry point:
- tests/vllm/entrypoints/openai/test_run_batch.cpp, new case 'run_batch: a tools line keeps the document's schema key order in the prompt': drives RunBatch::RunLine (the batch entry point) with a RAW JSONL line whose nested chat body is non-alphabetical at every level (built as a raw string — BatchLine() would re-serialize through the key-sorting nlohmann::json and destroy the order under test). It asserts (a) no exception and a 200 row — the crash fix, ISSUE-LOCAL-01M4FR20MES4HQVBRWJBJ2AVCN — and (b) the prompt seam receives the schema in the request document's key order (captured via a CapturingToolsPrompt seam over the parsed tools, the same capture pattern the api_server harness uses), and NOT the key-sorted form. Mutation-proven in both directions: with the committed throwing call restored the case throws type_error.302 (RED); with the call's argument mutated to body.dump() — the reverted first repair attempt — the case fails both key-order assertions (the sorted dump no-ops the seam); at the fix it passes.
- tests/vllm/entrypoints/openai/test_api_server.cpp, new case 'api_server: an unsorted tool schema keeps its document key order in the rendered prompt': posts an unsorted-schema tools request through the PRODUCTION /v1/chat/completions dispatch (handle_chat_completions) with the real Qwen3.8 fixture template and asserts the rendered prompt bytes carry the document order. Mutation-proven: with RestoreToolSchemaOrder no-op'd at api_server.cpp:400 the case fails both key-order assertions (RED); restored, green.
Gates: test_openai_run_batch 8/8 cases / 89/89 assertions; test_openai_api_server 104/104 cases / 1521/1521 assertions (1517 pre-existing + 4 new).

F2 — the 'byte-stable for already-ordered schemas' case (tests/vllm/entrypoints/test_chat_template.cpp) now uses a genuinely alphabetical fixture at every level: the schema is {"properties": {"city": ...}, "required": [...], "type": "object"} (properties < required < type; the single-property schema needs no order), and the case renders the SCHEMA alone ({{ tools[0].function.parameters | tojson }}) because the tool wrapper's field order (type/function, name/description/parameters) is the pinned model_dump order, which is not alphabetical — so the byte-identity claim is now real and the comment says so. Mutation-proven: with the tojson default mutated to sorted keys (the override the repair removed), the insertion-order case goes RED while this byte-stable case stays GREEN — exactly the before/after agreement it claims to pin. Gate: test_chat_template 45/45 cases / 204/204 assertions.
