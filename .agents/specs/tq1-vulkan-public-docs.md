# TQ1_0 Vulkan public documentation

## Now

Editorial work for `QUANT-GGUF-TQ1_0`, tracked by
`ISSUE-LOCAL-01M3NHV68SVMBBCBPZY8N47PFY`. The implementation inventory remains
`INVENTORIED`. Base: `d15b1cc09`. One PR carries the spec and documentation.
No model, backend, or benchmark lifecycle changes are in scope.

## Scope and inventory

| Surface | Gap | Source | Planned correction |
|---|---|---|---|
| README news | The newly merged TQ1_0 Vulkan kernels are absent | `src/vt/vulkan/vulkan_ops.cpp`, merge `8c93754bf` | Brief kernel announcement with the GGUF loading limitation |
| `docs/FEATURES.md` | Ternary kernel coverage is absent | Vulkan dispatch, `src/vt/dtype.cpp`, `src/vt/cpu/cpu_quant_dot.cpp` | Compact partial-support description, without competitor claims |
| `docs/BUILD.md` | Vulkan coverage omits ternary kernels | `CMakeLists.txt`, `src/vt/vulkan/vulkan_ops.cpp` | Explain the existing Vulkan build includes the kernels |
| `docs/USAGE.md` | Users cannot distinguish kernel availability from loadable models | `src/vllm/model_executor/model_loader/gguf_reader.cpp:201` and `:522` | State that stock TQ1_0/TQ2_0 GGUF loading is unavailable; no fabricated command |

## Design and upstream chain

This is a documentation correction, not a kernel or loader port. The existing
[vulkan kernel spec](vulkan-tq1_0-keep-quant.md) owns the implementation design
and its upstream llama.cpp references. The current reader rejects unknown GGML
type IDs. Its traits switch has no ternary entries. Do not infer usable GGUF
model support from internal `vt::DType` registration.

Read the actual loader, Vulkan dispatch, CPU reference, shaders, and tests
before describing the feature. Distinguish committed kernel code from measured
GPU correctness. The existing spec reports Maple results, but those results do
not prove a production load path in this base. Publish no model claim or speed
number from that report. The broader loader obligation already belongs to
`QUANT-GGUF-TQ1_0` and `ISSUE-GH-331`.

Keep additions short, factual, and linked. Reuse the existing Vulkan build
recipe. Do not invent model artifacts, hashes, flags, or new usage commands.
Do not modify benchmark dispositions, ABI documentation, or other open PR work.

## Tests and gates

No upstream tests to port: executable behavior is unchanged. Source inspection
fully verifies the proposed documentation. No GPU, model download, external
compute, runtime benchmark, or new mirrored-prose test is required.

Run these CPU-only checks on the base and implementation:

```sh
python3 scripts/check-agent-record.py
python3 scripts/check-readme-structure.py
python3 scripts/check-supported-models.py
python3 scripts/check-quickstart-recipes.py
python3 -m unittest discover -s tests/scripts -p 'test_check_readme_structure.py'
git diff --check
```

Run `scripts/agent-preflight.sh --quiet` and compare failures with the unchanged
base. Preserve logs outside the source tree. Existing failures are not a pass
and do not authorize checker edits. The scoped gate must pass. Independent
review checks every added source claim at the immutable head. There are no
new executable guarantees to mutate. The reviewer still runs the existing
README mutation suite. The operator independently reruns the focused gate.

## Work breakdown and authority

1. Operator records the local issue and commits this spec before documentation.
2. A fresh implementer uses an isolated worktree to edit the four public
   documents, this spec's outcome, and the scoped claim record.
3. A fresh reviewer inspects an immutable head in a separate worktree.
4. Operator reruns the gate, pushes to the maintenance fork, and opens a PR
   against `mudler/vllm.cpp:main`. Merge is not authorized.

## Risks and stop conditions

Missing loader support makes a generic ternary-model announcement false.
Shader catalog checks are not GPU numerical tests. Stop if the source cannot
justify a sentence. Do not repair code, unrelated records, checkers, or baseline
failures. Keep broader implementation gaps open under their existing owner.

## Outcome

29 September 2026: the four public documents distinguish ternary kernel code
from model loading. The existing Vulkan build recipe needs no new option.
The BUILD note links its kernel coverage and states the loader limitation.
No code, model lifecycle, benchmark number, or default changes.

Source checks at base `d15b1cc09`:

- `src/vt/vulkan/vulkan_ops.cpp:2178`, `:2287`, and `:2451` dispatch the
  matrix, grouped expert, and fused gate/up/SwiGLU kernels.
- `src/vt/vulkan/shaders/vt_matmul_bt_tq1_0.comp:74` reads compressed blocks.
  `src/vt/cpu/cpu_quant_dot.cpp:1145` supplies the CPU dot implementations.
- `src/vllm/model_executor/model_loader/gguf_reader.cpp:200` omits ternary
  type traits. `:522` rejects unknown tensor types.
- `tests/vt/test_vulkan_backend.cpp:318` checks TQ1_0 shader metadata.
  `:4253` checks CPU decoding, without executing a GPU kernel.
- `CMakeLists.txt:1731` includes the Vulkan operations and committed SPIR-V
  under the existing `VLLM_CPP_VULKAN` option.

The four focused checkers pass. The existing README mutation suite passes
19 tests. `git diff --check` passes. Commands are listed under Tests and gates.
The implementation log is `/tmp/vllm-doc-tools/impl-focused.log`.
No new test or negative mutation is needed for these editorial changes.

The initial claim table failed record validation because the implementation row
is `INVENTORIED`. The operator selected narrative editorial ownership instead,
with draft PR 3349 as the live claim. No checker or matrix changed.
The corrected record gate passes.

Full preflight was attempted with Bash. Its log at
`/tmp/vllm-doc-tools/impl-preflight.log` includes the provisional claim failure
and cannot establish final-head verification. The operator compares broad
failures against the unchanged base and owns final verification.
The initial `sh` invocation exited 2 because preflight requires Bash.
No GPU or oracle execution was attempted. No numerical correctness, model
loading, or performance claim follows from this documentation review.
