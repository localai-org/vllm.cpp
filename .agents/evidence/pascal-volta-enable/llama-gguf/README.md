# Llama GGUF adapter evidence

Issue: `ISSUE-LOCAL-01M63VJCK05BTSAMDXVWT60E2T`.
Spec first: `9934d78dd`. Quant seam dependencies: original `8f4ef1005`,
`d0d054456`, locally integrated as `5b305425b`, `b4c9389d7`.

The original public CUDA C API refuses both approved checkpoints before loading:
`GGUF architecture 'llama' is not supported by this build`. The new focused
test first fails compilation because the Llama GGUF header is absent.
`header-red.txt` preserves this result. The operator records public entry-point
red and final real-model token gates independently.

The adapter reuses the existing Llama registry and forward. Its config maps the
GGUF `llama` metadata family. The shared dense GGUF helper retains separate
quantized Q/K/V and gate/up ownership; the old Qwen3 wrapper still selects its
original concatenation branch. Q/K inverse permutation copies complete packed
rows with scales unchanged. No Q/K normalization is introduced.

Pinned source: llama.cpp `62318cd57`, `conversion/llama.py:175-181,255-264`;
vLLM `a7c23ac96`, `vllm/model_executor/models/llama.py`. The pinned first-party
GGUF plugin `d4c1f0d`, `weights_adapter/transformers.py:27-58` selects its generic
HF model adapter and GGUF tensor-name map. The local inverse supplies the
existing HF NeoX RoPE path with its original row layout.

Targeted Release C++20 compilation uses the complete build's exact compile and
link recipe. Scratch `/tmp/llama-unit-build.py` rebuilds adapter, shared GGUF
loader, and focused tests, links the existing production archive after removing
its old shared loader object. This is focused adapter evidence, not a full
container-layout rebuild. The quant-seam header is selected from its committed
dependency. `test_llama_gguf_weights` passes 4 cases and 171 assertions.

Mutation replaces the actual production Q/K source row expression with identity
in a scratch adapter source. Compilation and link succeed, then the focused
suite fails its independent F32-row and mixed Q4_0/Q8_0 packed-byte checks.
`inverse-mutation.txt` preserves the observed failure. Production source is
unchanged by mutation and the focused suite passes again after restoration.

The directly shifted `ENG-MOE-LOADSTREAM` anchor moves from line 2580 to 2582.
All unrelated keyed record bytes remain equal. No checker semantics change.

## Pending

The operator owns full-build and public entry-point execution after integrating
the separately implemented tokenizer and quantized-forward seams. No scratch
old-layout archive establishes full model correctness. Public-call deletion,
real-checkpoint token parity, traces, and performance remain pending that run.
The implementation issue remains OPEN until landing. Unsupported scaled/partial
GGUF RoPE, sidecars, bias, normalization variants and asymmetric value heads
are explicitly refused and remain owed.
