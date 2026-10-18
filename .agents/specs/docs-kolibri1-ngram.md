# Document Kolibri-1 n-gram decoding

## Scope

Update README News and the n-gram section of docs/SPECULATIVE-DECODING.md
for the existing Kolibri-1 CPU path. This is a documentation correction,
not a feature or lifecycle transition. Existing PRs cover basic Kolibri
usage and ABI versions. Do not duplicate those changes.

## Source and baseline

The implementation is src/vllm/entrypoints/model_loader.cpp,
LoadedEngine::MakeKVCacheMaybeSpec (line 1742), with config resolution in
ResolveSpecConfig. Public entry points accept --speculative-config and
vllm_model_params.speculative_config. The CPU model spec is
.agents/specs/kolibri-1-cpu.md, whose current n-gram evidence starts at
line 370. tests/vllm/models/test_kolibri1_ngram_spec.cpp contains the
checkpoint-free cache test and the checkpoint-dependent engine comparison.
The public guide currently describes only a generic n-gram invocation.

## Design and port map

Add one short October news item and a Kolibri-1 subsection to the n-gram
guide. Show the tested configuration: method ngram, five speculative
tokens, lookup minimum 1 and maximum 4, greedy CPU execution. Explain
that no draft weights are needed and omitting the config disables it.
Ground CLI spelling in examples/cli/main.cpp and examples/server/main.cpp.
Link existing checkpoint instructions, source tests, and the recorded
comparison. Separate spec-on/off token identity from oracle parity.
Do not publish a new performance result or repeat raw speed ratios.
Replace the generic `server` executable with the actual built path.

## Gates and tests to port

No upstream code or tests are ported. All edits are prose and command
examples. Validate every new option against source, parse the JSON, check
relative links and Markdown anchors, and run the relevant CPU document
checkers. Run preflight and distinguish existing failures or unavailable
tools from regressions. Use a fresh implementer and fresh reviewer.
No GPU, model download, model execution, or new benchmark is required.

## Dependencies, risks, and stop conditions

Base: 50d425bec. New evidence must not upgrade model, backend, or speed
claims. Batch size greater than one and real-traffic acceptance remain
unverified. A source mismatch is a review finding to repair before push.
Only README.md and docs/SPECULATIVE-DECODING.md are implementation scope.
The local issue and this spec carry the documentation record.

## Work breakdown

1. Commit this scope and the local issue.
2. Delegate the two-file documentation edit and source checks.
3. Review the immutable commit independently and rerun the document gates.
4. Open one upstream PR from the existing fork, without merging.

## Owed

- ISSUE-LOCAL-01M4W5TSTGNQY9WVBDVHK1PXQF: publish the source-checked guide.
