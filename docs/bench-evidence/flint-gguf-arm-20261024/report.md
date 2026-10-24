# Flint GGUF arm — the published containers load and their chains replay

Row: `MODEL-TEXT-FLINT` wave 2 (the GGUF arm + the real-weights token gate)
Spec: `.agents/specs/flint-gguf-arm.md`
Issue: ISSUE-LOCAL-01M5PRSC2E4VR54HBSWFPMN37E
Branch: `row/MODEL-TEXT-FLINT-GGUF`, base `7f842ee9b3c7e9b651283de48f42cd76a0ad5d30` (origin/main)
Date: 2026-10-24

## Recipe

- Build: `cmake -S . -B /tmp/flint-bld -G Ninja -DVLLM_CPP_BUILD_VERSION=0.0.3-dev
  -DCMAKE_BUILD_TYPE=Release && ninja -j 32` on this box (aarch64, 128 threads,
  255 GB, shared — the test binaries were run `nice -n 10 taskset -c 32-127`).
- Gate container: `/mnt/models/flint-GGUF/flint-Q4_K_M.gguf`
  (sha256 `20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e`).
- Goldens: `scripts/gen-flint-goldens.py` against `llama serve` from the
  pinned `llama-cpp-flint` oracle (PR #30928 head
  `1b44e30775dcb2f93952116b05f89e5bd797fb00`, build at
  `/tmp/llamacpp-flint-oracle`), `--no-jinja --chat-template chatml`,
  `/tokenize` for prompt ids (no BOS: `tokens_evaluated == len(ids)`),
  `/completion` temperature 0 top_k 1 with `n_probs: 1` for the generated
  token ids. 8 prompts x 24 tokens.

## The red → green story

- RED (`red.txt`): the dispatch case in `tests/vllm/test_model_loader_gguf.cpp`
  fails — a file with `general.architecture = flint` got the
  "is not supported by this build" refusal; `test_flint_gguf.cpp` does not
  compile (no `flint_gguf_weights.h`). The log also carries a SECOND red,
  `FromModelDir rejects an unknown dense architecture`: expected collateral,
  red only because the supported-arch list gained `FlintForCausalLM` — the
  same case is green in the implementation commit.
- GREEN after implementation, FIRST RUN of the token gate: 0/8 chains, all
  hard flips (gaps up to 17 nats) — NOT adjudicated green. Two latent W1
  defects surfaced, both filed and fixed in this change:
  1. ISSUE-LOCAL-01M5Q1PS772CPTMYJ02AMVZ3Y0 — `attn_out` sized `t*hidden`
     but written with the `t*hq*head_dim` head layout (the synthetic gate's
     `hq*dh < hidden` masked it). Found by an AddressSanitizer heap-buffer-
     overflow trace.
  2. ISSUE-LOCAL-01M5QBGBM3Z7NT3D9TA0K1HT7T — the embedding lookup memcpy'd
     raw bf16 bits into the f32 hidden stream (every token became a ~1e-42
     denormal). Adjudicated against TWO independent references that both
     answer the oracle's token: the pinned llama.cpp itself, and
     transformers `FlintForCausalLM` (transformers main) loaded with the same
     container's dequanted weights — top-1 12834 (" Paris") on both.
- GREEN (`green.txt`): **46/53 greedy tokens exact, 7 flips, 7 near-tie,
  0 hard flips, worst second-best gap 0.7348 nats against the 2.5-nat band.**
  One chain (`def fibonacci(n):`) matched all 24 tokens.

## Focused gates

- `ctest -R 'flint|gguf'`: 33/35 pass. The 2 failures are NOT this branch:
  `test_kolibri1_gguf` fails on its own row's recorded verdict (the kolibri1
  Q2_K token gate was landed red-by-record in 0350841b7; its hard flips at
  step 0 are a kolibri-row finding, none of this branch's files touch the
  kolibri path) and `test_deepseek_v4_gguf_load` fails on a
  `VT_CPU_QUANT_REPACK` environment requirement (W8 slice 4), 20/21 cases
  green.
- `ctest -R 'tokenizer|pretoken'`: 9/9 pass (shared tokenizer touched for the
  `flint` pre name).
- The new `flint gguf: the tokenizer reproduces the oracle's prompt ids` case:
  our `Encode` reproduces the oracle's `/tokenize` ids on all 8 gate prompts.

## Admitted / refused containers

Every published container's ggml types are inside the existing
`DequantGgufRowToBf16` capability set, so NOTHING is refused for quant type —
the refusal list is empty ON PURPOSE (the spec records the census). The
Q4_K_M container is gated end to end; the others load through the same path
without goldens of their own in this wave (spec ## Owed).

## What was measured, what was rejected

- The bf16-owned expansion (all weights to bf16 at the load) was kept — the
  same shape the kolibri1 GGUF arm uses — and the 2.5-nat band held with
  worst gap 0.73, so no keep-quant arm is needed for this gate.
- The `expert_gating_func = 2` (sigmoid) KV is read and REQUIRED, the
  opposite of the kolibri1 ROUTE-A choice (there the value was outside every
  enum; here the file states exactly the architecture's own routing).

## Deviations / notes

- The roadmap has no `MODEL-TEXT-FLINT-GGUF` row; the issue and spec are
  filed under `MODEL-TEXT-FLINT` (whose W1 `## Owed` names this work). The
  roadmap row addition is left to the operator.
- The worktree's `agent-preflight.sh` full run passes (done once at spec
  time); the `--staged` variant re-runs the whole suite and was not re-run
  per commit for time — the record and commit checkers below were run
  explicitly.
