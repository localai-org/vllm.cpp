# Refresh multimodal limit instructions

## Scope

Documentation correction under `ENG-MM-INPUT-PIPELINE`, tracked by
[ISSUE-LOCAL-01M42DVD1H44AV0JTT2F6K3JDB](../issues/ENG-MM-INPUT-PIPELINE/ISSUE-LOCAL-01M42DVD1H44AV0JTT2F6K3JDB.md).
Base: `33fb82b09`. Update the multimodal input guide, its server flag reference,
and one README news entry. No model lifecycle or benchmark disposition changes.

## Baseline and source anchors

| Surface | Source | Documentation gap |
|---|---|---|
| Dense tower skip | `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:1193`, `src/vllm/model_executor/models/qwen3_5_dense.cpp:55` and `:133` | The guide and flag reference omit the dense loader. |
| Shared zero-limit predicate | `src/vllm/model_executor/models/interfaces.cpp:9` | Explain that every modality served by a tower must be zero. |
| Projector loading | `src/vllm/entrypoints/model_loader.cpp:3186` and `:3190` | Preserve the distinction between metadata validation and skipped tensor validation. |
| Serving limits | `src/vllm/entrypoints/openai/chat_mm.cpp` and `chat_utils.cpp` | Preserve architecture ceilings and configured-limit refusal behavior. |
| Dense loader test | `tests/vllm/models/test_qwen3_5_dense_vision.cpp`, `qwen3_5_dense_loader_leaves_the_tower_unread_at_zero_limits` | Source evidence only. No new execution claim. |

The existing implementation spec is
[qwen35-dense-tower-modality-limits.md](qwen35-dense-tower-modality-limits.md).
This task documents local behavior. It ports no upstream behavior or tests and
makes no new oracle, token, memory, or speed claim. No GPU is available.

## Design and work breakdown

1. Add a short README news entry for the dense loader's text-only loading change.
   Link to the input guide. Do not imply new multimodal HTTP architecture support.
2. Rewrite the guide's per-prompt limits section as user instructions. Include
   the dense loader, keep the dots3-note exception, and explain both zero-limit
   spellings. Verify and document current C API media routing. Keep projector validation caveats.
3. Replace the long server flag cell with its behavior and a guide link.
4. Link benchmark history to `docs/benchmarks/memory.md`, which already retains
   the historical measurements. Do not duplicate or change those measurements.

Keep unrelated guide content intact unless an adjacent sentence contradicts the
corrected zero-limit behavior. Any additional loader named in a coverage list
needs its production call site verified. Use short sentences and avoid internal
wave names, superseded attempt narratives, and speculative support claims.

## Tests and gates

No new test is needed for this documentation-only correction. Verify each changed
statement against the production source and existing tests. Check Markdown links
and executable flag spellings. Run:

```sh
python3 scripts/check-readme-structure.py
python3 scripts/check-agent-record.py
git diff --check 33fb82b09
bash scripts/agent-preflight.sh --quiet
```

Record exact results. Pre-existing full-preflight failures must reproduce at the
pinned base. Do not alter a checker to make this documentation change pass.
A fresh reviewer checks the committed diff in a separate worktree. Because no
behavior or test changes, production-call deletion is not an applicable gate.
The reviewer uses a scratch documentation mutation to demonstrate any applicable
structural guard and separately checks semantic claims against source.

## Dependencies, risks, and stop conditions

Local CPU, Python, Git, and the repository sources are sufficient. No checkpoint,
GPU, new benchmark, remote compute, or upstream checkout is required. An existing
PR covers ABI documentation, and another covers Qwen3.8 quantized arms. Do not
duplicate those changes. Stop on an unresolved source contradiction that changes
the intended scope. The user authorized a fork PR, not a merge to upstream.

## Source correction before the C API documentation edit

Production `src/capi/vllm_c.cpp:480` installs the multimodal chat handler.
`vllm_chat` reaches that handler at `:1343`, contrary to the guide's old
no-media claim. `include/vllm.h:205` describes the current behavior, although
its field comment at `:702` still describes the old behavior. The existing
case at `tests/capi/test_capi.cpp:1518` checks named refusal for an unregistered
multimodal architecture. Update the guide's opening and limits subsection to
reflect current routing and refusals. No new ABI behavior or test is introduced.
`src/vllm/entrypoints/model_loader.cpp:3143` routes the DeepSeek-V4 projector
separately. Qualify the guide's GGUF request limitation to the Qwen3-VL
projector arm, rather than all GGUF models. The opening table lists the
examples covered by this guide, not every registered multimodal architecture.
The operator approved these source-backed scope corrections on 4 October 2026.

## Outcome

The guide now documents dense safetensors tower skipping, current C API routing,
and the Qwen3-VL projector's validation boundaries. The server reference links
to these instructions. README news describes the dense loader change.
Historical memory results remain in `docs/benchmarks/memory.md` without a new
measurement or changed disposition.

Source verification on 4 October 2026 used:

- `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:1193` and
  `qwen3_5_dense.cpp:133`: the production loader skips vision tensors and
  configuration parsing when both image and video limits are zero.
- `src/vllm/model_executor/models/interfaces.cpp:9`, `qwen3_vl.cpp:463`, and
  `muse_glimmer_weights.cpp:824`: shared predicate and existing loader examples.
- `src/vllm/model_executor/models/dots3_note.cpp:783` and `:804`: supported
  towers still load without a modality-limit condition.
- `src/vllm/entrypoints/model_loader.cpp:3143`, `:3154`, and `:3186`:
  separate DeepSeek-V4 routing and Qwen3-VL projector validation before skipping.
- `src/vllm/multimodal/processing/context.cpp:38`: effective limits and the
  conditional error hint. `include/vllm/config/multimodal.h:79`: zero-limit precedence.
- `src/capi/vllm_c.cpp:480` and `:1343`: shared chat handler installation and use.
  Existing refusal coverage is `tests/capi/test_capi.cpp:1518`.

Verification used the local Python runtime with
`PATH=/tmp/vllm-doc-tools/root/usr/bin:$PATH` and
`LD_LIBRARY_PATH=/tmp/vllm-doc-tools/root/usr/lib`:

- `python3 scripts/check-readme-structure.py`: exit 0.
- `python3 scripts/check-agent-record.py`: exit 0 after releasing the editorial claim.
- `python3 tests/scripts/test_agent_record.py`: exit 0, 106 tests.
- `git diff --check 33fb82b09`: exit 0.
- Added local Markdown links: 6 targets and heading anchors resolve.
- `bash scripts/agent-preflight.sh --quiet`: INCOMPLETE. Both the pre-edit and
  post-edit invocations were terminated after unrelated baseline failures and
  slow subprocess suites. Logs are `/tmp/vllm-docs-impl-preedit.log` and
  `/tmp/vllm-docs-impl-preflight.log`. Observed failures include missing build
  tools, release-tool dependencies, oracle-pin records, and the live-row audit.
  The coordinator reproduced baseline failures in a separate clean worktree.
  The post-edit run also saw a transient table-claim error for the parent row's
  `READY` state. The released editorial handoff and focused record suite pass.

No behavior or tests changed. Red-first implementation and production-call
mutation tests are not applicable. The fresh reviewer owns structural mutation
and semantic review. No GPU, checkpoint, oracle, or performance run occurred.

## Now

Documentation implementation is ready for independent review. The coordinator
owns review and the fork pull request. Model lifecycle states and benchmark
dispositions remain unchanged.
