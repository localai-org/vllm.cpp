# Qwen3.8-27B documentation correction

**Row:** `DOCS-QWEN38-ARMS`

## Scope

Correct the public description of existing quantized Qwen3.8-27B paths.
This is documentation maintenance, with no model lifecycle transition.
Base: `9157ba9c855c1b8a1c7240a2affa56c85da046b2`.

| Surface | Current gap | Required correction |
|---|---|---|
| `docs/models/README.md` | Calls block-wise FP8 CPU-only | Name CPU and supported CUDA targets |
| `docs/models/qwen3-8-27b.md` | Implies all running arms have a correctness result | Separate loading, generation, and recorded token gates by artifact |
| `README.md` News | Omits the published block-wise FP8 correctness gate | Add a dated, scoped entry with no speed claim |

Open documentation PRs were inspected before selection. ABI 30, HTTP routes,
Hub downloads, and EXL3 updates are owned elsewhere and excluded here.

## Source and evidence

| Claim | Source | Evidence |
|---|---|---|
| CUDA block-scaled FP8 | `src/vt/cuda/cuda_matmul_fp8_block_cutlass.cu` | `tests/vt/test_ops_matmul_fp8_block_cuda.cpp` |
| Block shape and activation constraints | `src/vllm/model_executor/layers/quantization/fp8_block_quant.cpp` | `tests/vt/test_fp8_block_scaled_dispatch.cpp` |
| Dense model routing | `include/vllm/model_executor/models/dense_fp8_block_gemm.h` | `.agents/specs/gate-qwen38-27b-fp8-block.md` |
| Mixed NVFP4 artifact limits | `tests/vllm/models/test_qwen38_27b_nvfp4_arm.cpp` | `.agents/specs/qwen38-27b-quant-arms.md` |
| GGUF token mismatch | `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp` | `docs/bench-evidence/qwen38-27b-q4km-token-gate-20260823.md` |

The upstream chain and measured historical oracle revision belong to the
linked implementation and gate specs. This task ports no implementation and
runs no oracle. It must preserve their evidence scope, including six strict
prompts and one adjudicated near-tie for block-wise FP8.

## Design

Use a compact artifact table, then explain the two FP8 layouts and their
limits. Link the detailed gate record rather than repeating its chronology.
Preserve checkpoint revisions, meaningful errors, and reproduction references.
Do not claim that generic NVFP4 support proves this model's NVFP4 artifacts.
Keep the documented shape restriction as an open implementation gap, without
calling it impossible to fix. Preserve existing section anchors where useful.

## Tests and gates

No upstream tests to port: no runtime behavior changes. Verify each changed
claim against local source and existing evidence. Run the README structure
checker, its mutation suite, and local Markdown link checks. Run preflight
before edits and before submission, reporting baseline failures separately.
A fresh reviewer independently checks source, links, and claim scope.

## Dependencies, risks, and stop conditions

CPU-only. No GPU, model download, remote compute, service change, or new
benchmark. Unsupported claims must be removed or qualified, not measured here.
Do not change matrices, historical evidence, or benchmark dispositions.
One PR carries this committed spec followed by the documentation change.

## Work breakdown

1. Commit this scope and canonical local issue.
2. Fresh implementer edits only the three public files above.
3. Fresh reviewer audits the immutable implementation commit.
4. Run the scoped checks and submit to upstream from the maintenance fork.

## Now

Documentation implemented in `0ba4305cc`, with prose repair `0ae333669`.
Independent source and prose review passed at `0ae333669`.
The local issue remains open until the correction lands upstream.

## Outcome

The guide now distinguishes artifact loading from historical correctness
results. It retains checkpoint pins and links detailed evidence instead of
repeating the gate chronology. The index names the implemented CUDA targets,
and README News describes six exact prompts plus one accepted near-tie.
No new performance result or model lifecycle claim is introduced.

Scoped verification: README structure passes, its 19 mutation tests pass,
and all 40 local Markdown links in the two model pages resolve. Commit style
and trailer checks pass. No GPU, model, or oracle execution was needed.

Full preflight is not green on the base: stale record anchors, undocumented
Tenstorrent variables, gate-record drift, and unavailable build dependencies
are outside this documentation task. The broader runs were stopped after
these failures, so the complete suite remains unverified. The PR reports
these limitations.

## Owed

Upstream review and merge of the documentation correction.
