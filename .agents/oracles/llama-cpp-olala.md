# Oracle: `OVHai-LLM/llama.cpp` `olala-master` — the only llama.cpp that knows `olala`

A scoped, second llama.cpp record, in the same shape as
[`llama-cpp-flint.md`](llama-cpp-flint.md). The published
`OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF` files state they need this fork: upstream
llama.cpp, Ollama and LM Studio refuse `unknown model architecture: 'olala'`.
It supplies the GGUF quant-matched token-exact floor (CPU) for the port's
q5_k_m / q6_k / q8_0-safe2 arms — the same role the flint PR played for flint.
It does not replace the stock [`llama-cpp`](llama-cpp.md) floor and is never a
mirror source.

## Provenance

Fork `OVHai-LLM/llama.cpp`, branch `olala-master` (docs: `docs/olala.md`,
`docs/olala-setup.md`; model source `src/models/olala.cpp`; fused kernels
`MAMBA3_MIMO` and `GEODESIC` on CPU/Metal/CUDA). The fork README documents the
safe quantization overrides (`ffn_latent_up=bf16`, `ffn_up_exps=q8_0`) that the
published q6_k/q5_k_m/q8_0-safe2 files use — matching our W1 dtype census
(ffn_latent_up BF16, ffn_up_exps Q8_0 in the q6_k arm). Upstream ggml-org
llama.cpp at the stock pin carries zero olala hits (checked 2026-11-21 via the
`olala` string in src/models; the ollama issue #19777 confirms refusal).

```oracle-pin
id = llama-cpp-olala
role = secondary
upstream = https://github.com/OVHai-LLM/llama.cpp
scope = the olala GGUF architecture, its conversion, its graph, and the CPU GGUF floor on an olala checkpoint — the quant-matched denominator for the port's k-quant arms
pin = bd20ca9b471fdcb3f03b75f82d3cf575414bc573
pin_label = olala-master
pinned_on = 2026-11-21
gateable = no
evidence = #3582
```

Gateability owed: build `bd20ca9b` on this host and serve the pinned
`/mnt/models/olala-v1-mini/Olala-v1.0-mini-7A1B-q6_k.gguf` (sha256 recorded in
the W1 evidence), greedy goldens on fixed prompts. Not yet measured.
