ID: ISSUE-LOCAL-01M4FR20MES4HQVBRWJBJ2AVCN
Title: run_batch: RestoreToolSchemaOrder(body, request) passes a nlohmann::json where a string is expected — every chat batch line throws json.exception.type_error.302 (test_openai_run_batch red at the PR #3422 repair head)
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

Found by the operator's clean-head re-verification of PR #3422 (branch
row/kolibri-serve, head d93d9d492), after the scoped re-review F1 finding
(ISSUE-LOCAL-01M4FR1D7RH7CN5X2R2CAR6N61) showed the ordered-parameters seam
was detected by no committed test. The repair commit 5983c6918 wired
RestoreToolSchemaOrder into RunBatch::DispatchChat as
RestoreToolSchemaOrder(body, request) (src/vllm/entrypoints/openai/run_batch.cpp:110),
but the function's first parameter is const std::string&
(include/vllm/entrypoints/openai/protocol.h:605) and body is a
const nlohmann::json&. The call compiles only through nlohmann's implicit
operator ValueType() -> get<std::string>(), which throws
json.exception.type_error.302 ('type must be string, but is object') for every
object body — and DispatchChat has no try/catch around the call, so EVERY
/v1/chat/completions line dispatched through RunBatch throws an uncaught
exception. Operator's clean-head evidence at d93d9d492:
/tmp/build-kolibri-serve/tests/test_openai_run_batch reports 'test cases: 7 |
4 passed | 3 failed', 'assertions: 16 | 16 passed', Status: FAILURE — the three
chat-dispatch cases (test_run_batch.cpp:451, :507, :601 at that head) all throw
type_error.302. The repair's recorded gate 'test_openai_run_batch 16'
(docs/bench-evidence/kolibri1-serve-20261008.md "Review repair"; the PR #3422
body; commit 5983c6918's message) counted the ASSERTION total (16/16) and
missed the three throwing test cases — a stale binary, so the PR's CI lane
(build-test-cpu runs the full ctest) would be red. The other two call sites
pass the raw body string and are correct (api_server.cpp:400
handle_chat_completions(request_body); vllm_c.cpp:518
ParseChatRequest(request_json)).

A first in-flow repair attempt (passing body.dump()) is WRONG and was
reverted: body is the key-sorted nlohmann::json (std::map), so its dump is
already alphabetically sorted and the order restoration re-reads sorted text —
the crash goes away but the seam no-ops, and the request document's tool-schema
key order is lost on the batch path (the pinned renderer dumps schemas into
the prompt with sort_keys=False, so the prompt bytes would silently change
order). Mutation-proven below.

## Resolution

Landed on row/kolibri-serve 2026-10-09 (fix commit a6679a82a3e6edc7bbc71a09585a5944bcd91810; test commit
2f535c002a1b15c8717a53a5e8103258ad5fd229 — see commit bodies). The original request text IS available:
RunBatch::RunLine(const std::string& request_json) parses it at run_batch.cpp:159
and drops the text. The fix threads it through: RunLine re-serializes the chat
body — the line's top-level "body" member — from an ORDER-PRESERVING
(nlohmann::ordered_json) parse of the original line text, and passes it to
DispatchChat(custom_id, body, body_json); DispatchChat calls
RestoreToolSchemaOrder(body_json, request). ordered_json::dump keeps insertion
order, so body_json carries the request document's key order and the re-read
restores it into request.tools, exactly as the api_server and C-ABI entry
points already did with their raw body strings. The shared seam's signature is
unchanged.

Red-first and mutation evidence (all at /tmp/build-kolibri-serve, this host):
- Committed throwing call (head d93d9d492 plus the new F1 test): 4 of 8 cases
  throw type_error.302 — the 3 pre-existing cases (test_run_batch.cpp:484/:540/:634
  in the edited file) plus the new F1 case; assertions read '16 | 16 passed'
  while 4 cases fail — the stale-binary signature.
- body.dump() mutation (the reverted first attempt): the 3 pre-existing cases
  go GREEN (crash silenced) but the new F1 case fails BOTH key-order assertions
  — the prompt carries the key-sorted schema, proving the seam no-ops.
- The fix: test_openai_run_batch 8/8 cases, 89/89 assertions, Status: SUCCESS;
  the F1 case's prompt carries the document-ordered schema.
Full battery re-verified at the fixed head: test_openai_api_server 1521,
test_openai_serving 1365, test_chat_template 204, test_kolibri1_chat_template
61, test_reasoning_kolibri1 43, test_tool_parser_kolibri1 19,
test_reasoning_parser_detect 75, test_tool_parser_detect 361, test_kolibri1
27/234, test_kolibri1_decode_bench anchor 109726 — all PASS; W3 not rerun (no
forward change). The repair's 'test_openai_run_batch 16' gate line in
docs/bench-evidence/kolibri1-serve-20261008.md is falsified by this evidence and
corrected in that document's re-review section.
