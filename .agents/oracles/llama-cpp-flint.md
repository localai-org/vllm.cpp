# Oracle: `ggml-org/llama.cpp` PR #30928, the only llama.cpp that knows `flint`

A scoped, second llama.cpp record, admitted for one narrow reason, in the
same shape as [`llama-cpp-qwen4exp`](llama-cpp-qwen4exp.md) and
[`llama-cpp-glm5next`](llama-cpp-glm5next.md). It is the only llama.cpp that
converts or loads the `flint` architecture (`FlintForCausalLM`), and the
`FLINT-CPU-PARITY` gate needs a llama.cpp denominator on the identical GGUF
artifact. It does not replace the [`llama-cpp`](llama-cpp.md) oracle, it does
not outrank vLLM, and it is never a mirror source.

## Why the `llama-cpp` file cannot carry this pin

`scripts/check-oracle-pins.py` admits exactly one ` ```oracle-pin ` block per
file, and the `llama-cpp` pin is deliberately stock release `b10451`, which
predates Flint entirely. Folding a PR head into that record would change what
the floor means for every measurement already taken against it.

## Provenance

PR ggml-org/llama.cpp#30928 "models: Add Flint model (FlintForCausalLM)",
author alexey-rsi (RecursiveSI — the model author), head `1b44e3077`.
Checked 2026-10-21 against the real files: stock master and the
`unslothai` fork both carry zero flint hits. CI on the PR is green on
ubuntu / windows / webgpu / vulkan / cuda / metal / model-naming / lint.
The converter records `expert_shared_count` =
`shared_expert_intermediate_size // moe_intermediate_size` (= 2 for flint:
one shared expert of double width stored as two single-expert slots in one
rank-2 `ffn_*_shexp` matrix — never two shared experts), `leading_dense_block_count` = 2,
`expert_gating_func` = SIGMOID, per-layer `attention.sliding_window_pattern`
and `attention.rope_pattern`, and the per-head `attn_sinks`.

A local build of this exact head exists at `/tmp/llamacpp-flint-oracle`
(git log head `1b44e3077`), and an external session measured it serving
the Flint I-Mini IQ-class artifact at tg200 24.99 / tg400 24.33 tok/s
(untracked log, not a record). That demonstrates the head builds and runs;
the tracked gateability measurement on the shipped Q4_K_M artifact is what
this row owes.

```oracle-pin
id = llama-cpp-flint
role = secondary
upstream = https://github.com/ggml-org/llama.cpp
scope = the flint architecture, its GGUF conversion, its graph, and the CPU GGUF floor on a flint checkpoint, which no released llama.cpp defines
pin = 1b44e30775dcb2f93952116b05f89e5bd797fb00
pin_label = pr-30928
pinned_on = 2026-10-28
gateable = yes
evidence = docs/bench-evidence/flint-cpu-parity-20261029/report.md
```

Gateability evidence (2026-10-29, row FLINT-CPU-PARITY,
ISSUE-LOCAL-01M61G9R7D4EAM54RKXMES0Z6J): build 1b44e3077 served the shipped
flint-Q4_K_M.gguf (sha256
`20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e`, the SAME
file both sides) on the shared aarch64 box: greedy token goldens on three
fixed prompts (`golden-sayhi-raw.log`, `golden-count-raw.log`,
`golden-200tok-raw.log`, temperature 0, seed 42, finish reason length) and
llama.cpp tg rates, 3-rep medians (`llama-bench-tg.log`: tg200 26.29 / tg400
25.66 tok/s at 16 threads, tg200 26.38 / tg400 24.69 tok/s at 32 threads,
`taskset -c 32-63`). One noted harness quirk, recorded in the evidence
report: the container's chat-template metadata crashes template init (a
`\x00` escape the lexer rejects), so the golden legs pass an explicit
raw-passthrough chat template with `--jinja` — numerics are unaffected, the
graph is the same.

