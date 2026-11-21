ID: ISSUE-LOCAL-01M7ZAHMYW87MPTAW4J78HYMB7
Title: NEW-CANDIDATE-PORTS W2 (olala): pin the two secondary oracles (author transformers code + OVHai-LLM/llama.cpp fork) and write the architecture map; then the forward port waves
Row: NEW-CANDIDATE-PORTS
State: OPEN
Kind: task
GitHub: 3582
Mirror: SYNCED
Availability: FULL
Created: 2026-11-21
Updated: 2026-11-21
Closed: -

## Problem

W1 landed (GGUF census-validated loader arm; forward refuses by name). W2 needs the oracle decision recorded: vLLM registers no olala arm at any pin, so per AGENTS.md the port runs against secondary oracles: (1) the model author's own HF transformers custom code OVHaiLLM/Olala-v1.0-mini-7A1B @ f781f12498adc9bdd71b7dd59d1311434369c684 (modeling_olala.py 2350 lines + configuration_olala.py; Apache-2.0; 6.77B params, bf16 model.safetensors 13.6 GB) — the behavior reference; (2) the author's llama.cpp fork OVHai-LLM/llama.cpp branch olala-master @ bd20ca9b471fdcb3f03b75f82d3cf575414bc573 (src/models/olala.cpp; CPU/Metal/CUDA kernels MAMBA3_MIMO + GEODESIC) — the GGUF quant-matched floor for the q5_k_m/q6_k/q8_0-safe2 arms (the published GGUFs REQUIRE this fork; upstream llama.cpp refuses olala). Both get .agents/oracles records with gateable=no and the owing measurement issue. W2 spec maps the architecture from modeling_olala.py: OlalaMonoBlock per layer = mixer (M: OlalaMamba3MimoFast mamba3-mimo with ssm_* tensors; V: OlalaDifferentialTensorProductAttentionV2 with attn_* tensors, sliding window 32768, softcap 150) + OlalaGeodesicNorm (geo_* scalars) + OlalaMoE (sigmoid gate ffn_gate_inp+exp_probs_b, top-6 of 256, latent experts ffn_latent_* + ffn_*_exps, shared ffn_*_shexp); globals token_embd/output [151936,1536]. Forward port waves follow the qwen3.5 shape; gates = unit parity vs the transformers reference per op, then a greedy token gate vs the llama.cpp fork on the same GGUF.

## Resolution

-
