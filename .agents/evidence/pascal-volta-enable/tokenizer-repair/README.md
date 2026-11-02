# Published GGUF tokenizer repair

Issue: `ISSUE-LOCAL-01M63W5W3MTWY1KXRRD77FBG5D`.
Date: 29 October 2026. Spec-first commit: `6a50938f0`.

Pinned llama.cpp `62318cd57dc63900ecb153272c0b10f78550855d` supplies the
executing tokenizer oracle. `src/llama-vocab.cpp:111-246` defines score BPE;
`:362-371` defines SmolLM splitting; `:2097-2107,2564-2595` defines SPM
metadata defaults; `:3579-3633` defines prefix and special-token behavior.
The score merge scan uses the same highest-score and leftmost order as the
upstream queue. Initial symbols remain UTF-8 characters until final byte
fallback. Existing HF tokenizer behavior remains on its previous branch.
The primary vLLM plugin pin is `d4c1f0d082fc7cd4350da56689109a01c1f29d6c`;
it delegates GGUF tokenizer conversion to Transformers. Pascal cannot run
primary vLLM, so the registered pinned llama.cpp supplies executable evidence.

## Tests

The focused `tests/vllm/test_bpe.cpp` first fails both new public loader cases
with the exact unsupported `smollm` and `llama` diagnostics. See `red-run.log`.
The complete binary after repair passes 32 cases and 1027 assertions.
See `green-final.log`. It also preserves the old unsupported-family test by
changing its sentinel from the now-supported `llama` to `unimplemented`.
Synthetic cases pin score priority and leftmost ties, unconditional SPM prefix,
unknown/control token splitting, UTF-8 byte fallback after merging, raw invalid
byte decoding, BOS/EOS flags, malformed score metadata, and SmolLM digit
isolation before mergeable GPT-2 tokens. These adapt the pinned vocabulary
algorithm to zero-tensor GGUF fixtures, not a different tokenizer algorithm.

Eight independent scratch source mutations compile and link successfully.
Each focused run exits 1: remove SmolLM digit isolation, invert score priority,
remove unconditional prefix, remove unknown-token classification, replace raw
byte decoding, remove default BOS, remove EOS flags, and ignore prefix flags.
`mutations.log` records the results. Production sources remain unchanged during
mutations. The original full green binary remains intact and passes again.

## Actual checkpoints

The operator approved and downloaded these immutable artifacts:

- `bartowski/SmolLM2-135M-Instruct-GGUF` revision
  `09816acd5d99df7be770d85ea30822623dab342c`,
  `SmolLM2-135M-Instruct-Q4_K_M.gguf`, 105454432 bytes,
  SHA256 `2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d`.
- `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` revision
  `52e7645ba7c309695bec7ac98f4f005b139cf465`,
  `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`, 668788096 bytes,
  SHA256 `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`.

`corpus.json` contains 25 texts. For each checkpoint, `actual.cpp` calls
`Tokenizer::FromGguf`, plain encode, special-token encode, and decode.
`oracle.cpp` loads only the pinned llama.cpp vocabulary and calls tokenize
with parse_special=true, both add_special modes, then detokenize with
remove_special=false and unparse_special=true. Both checkpoint results match
all 25 entries exactly across all three outputs. Committed oracle JSONL files
record every ID and decoded string. The corpus covers empty input, whitespace,
contractions, Unicode numbers, combining marks, multilingual text, byte
fallback, control/unknown tokens, and the actual inference prompt.
The existing upstream tokenizer executable reads checkpoint-specific external
input/output files; this adaptation uses the same library API directly and a
small committed JSON corpus so no large weights enter Git.

Build `actual.cpp` against this source's tokenizer object and production
archive. Build `oracle.cpp` against the pinned llama.cpp `include` and
`ggml/include`, linking `llama`, `ggml`, and `ggml-base`. The precise local
commands and logs remain under `/tmp/tokenizer-scratch`. No GPU is used.

## Limits

`Tokenizer` private state grows. Recompile every holder and caller against the
updated header before model integration. The operator owns that full build.
The bounded startup preflight reaches repository checks and is interrupted at
45 seconds; it is not a green full preflight. After updating seven directly
invalidated tokenizer anchors, `scripts/check-agent-record.py` passes with
`ANCHOR-ROT=0`; `git diff --check` also passes. Independent review and real-model inference remain required. Issues
remain open until landing. No merge, remote write, or model forward occurs.
