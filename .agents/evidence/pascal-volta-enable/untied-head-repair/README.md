# Untied NK dense output head repair

Issue: `ISSUE-LOCAL-01M63XH2ED7MGFNBNWG8BKEEB6`.
Spec-first commit: `0f8631672`.

The real TinyLlama Q4_K_M public load succeeds, then completion fails in
`vt::Matmul` with inner dimensions mismatch. GDB reaches that function from
`ForwardLayers`, `Qwen3DenseModel::ForwardDevice`, Llama's registered forward,
and `ModelRegistry::Forward`. The sparse primary vLLM pin defines the untied
`ParallelLMHead(vocab_size, hidden_size, quant_config)` in
`vllm/model_executor/models/llama.py:502-513` and forwards it through the logits
processor at lines 546-550. Pinned llama.cpp `62318cd57` implements its output
projection as `build_lora_mm(model.output, cur, model.output_s)` in
`src/models/llama.cpp:239-240`.

Local `qwen3_gguf_weights.cpp:237-256` preserves quantized and expanded NK
weights in `[N,K]` layout. `LoadEmbedAndHead` loads the untied head through
that function. The shared forward instead assumes every untied head is
transposed Matmul-B. Honor its `nk` flag and use the existing registered
`MatmulBT` operation. Tied heads keep their existing branch. Untied non-NK HF
heads keep `Matmul`. No storage conversion, new forward, or shape workaround
is introduced.

The small synthetic public fixture has an untied Q4_0 head `[64,32]`. Tests
assert the NK layout, dtype, and dimensions before public load/completion.
All four generated steps produce finite complete logits. The full Llama
adapter host suite passes 5 cases and 180 assertions. Removing the new NK
condition in a scratch production TU reproduces the exact public completion
failure. Both real TinyLlama and the synthetic fixture detect this mutation.

Real pinned TinyLlama generates 16 tokens for the existing 16-token prompt.
The tied SmolLM2 checkpoint still generates 4 tokens as a control. Exact token
IDs and elapsed times are retained in the logs. These are correctness and
reachability observations, not accepted performance measurements or oracle
parity. The parent owns those gates and artifact provenance.

GPU commands use `flock /home/user/gpu.lock`. Scratch compile replaces only
model `qwen3.cpp.o` in a separate copied complete archive; the tool-parser
member with the same basename remains. An initial scratch link duplicated
that parser member, then removed the redundant extra object. Another initial
link included the doctest main in a standalone CAPI executable and was fixed.
Neither linker error is counted as a model failure. Fixture tokenizer metadata
was completed before recording the intended forward red.

Original `/tmp/lumen-build-legacy` was never manually edited or rebuilt here.
The complete full build, independent review, real oracle parity, and all
performance axes remain pending. Startup preflight was invoked and interrupted;
it is not claimed green. `git diff --check` and record checker pass.
