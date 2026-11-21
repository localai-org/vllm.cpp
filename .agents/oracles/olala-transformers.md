# Oracle: `OVHaiLLM/Olala-v1.0-mini-7A1B` (the model author's own transformers custom code)

A scoped secondary oracle for the `olala` architecture, which vLLM registers at
no pin. The model author (OVHcloud OVHai LLM) publishes a HF transformers
custom-code implementation (`modeling_olala.py`, 2350 lines, +
`configuration_olala.py`) under Apache-2.0; per the oracle registry's
`transformers` row, the model author's reference implementation is admissible
where vLLM implements nothing. It is never a mirror source and never outranks
vLLM; it answers "what correct olala output looks like".

## Provenance

Repo `OVHaiLLM/Olala-v1.0-mini-7A1B`, single bf16 `model.safetensors`
(13,597,677,592 B; 6,773,124,144 params: 6,756,461,568 BF16 + 16,662,576 F32),
`auto_map` -> `configuration_olala.OlalaConfig` / `modeling_olala.OlalaModel` /
`modeling_olala.OlalaForCausalLM`. Architecture: 36 layers = 29 Mamba-3-MIMO +
7 differential-attention (V) layers, each followed by a MoE block (256 routed
+ 1 shared expert, top-6); the class map is in
`docs/bench-evidence/new-candidate-ports-olala-20261120/olala-namemap.md`
(W1 census of the sibling GGUF container).

```oracle-pin
id = olala-transformers
role = secondary
upstream = https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B
scope = the olala architecture's reference forward (Mamba3-MIMO mixer, differential tensor-product attention, geodesic norm, latent MoE) — behavior oracle for the port, runnable on CPU from the safetensors checkpoint
pin = f781f12498adc9bdd71b7dd59d1311434369c684
pin_label = hf-f781f124
pinned_on = 2026-11-21
gateable = no
evidence = #3582
```

Gateability owed: run `OlalaForCausalLM` from this revision on CPU (host has
255 GiB RAM; the bf16 checkpoint is 13.6 GB) on fixed greedy prompts and
record token ids as the behavior golden. Not yet measured — no transformers
install with this custom code exists on this host yet.
