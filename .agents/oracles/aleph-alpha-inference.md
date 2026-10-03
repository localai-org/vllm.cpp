# `aleph-alpha-inference` — Aleph Alpha's vLLM plugin for Kolibri

The model-author's own out-of-tree vLLM plugin
([general_plugins](https://github.com/Aleph-Alpha/aleph-alpha-inference) entry
point, PyPI `aleph-alpha-inference`): it registers `Kolibri1ForCausalLM`
(`model_type kolibri1`) plus the `kolibri1` reasoning and tool-call parsers, and
ships the serving recipe (fp8 KV cache, 1M-context override). Upstream
[vLLM](https://github.com/vllm-project/vllm) implements nothing for this
architecture — the plugin is where vLLM implements it, so where the plugin
serves, the answer it produces is vLLM's answer for Kolibri.

```oracle-pin
id = aleph-alpha-inference
role = primary-for-kolibri1
upstream = https://github.com/Aleph-Alpha/aleph-alpha-inference
scope = the kolibri1 (Kolibri1ForCausalLM) architecture only: 50-layer hybrid sliding/full attention (4:1, window 513), MoE 384 experts x 6 active + shared expert, hidden 2560, vocab 128000, 262k (extendable 1M) context, English-German; includes the kolibri1 reasoning/tool-call parsers and the chat template
pin = 049a6a7bd240
pinned_on = 2026-10-03
gateable = no
```

## Gateability

`gateable = no` until the plugin demonstrably builds and runs
`Aleph-Alpha/Kolibri-1` (78.9 GB bf16, 32 shards — the weights are mirrored at
`/mnt/models/Aleph-Alpha/Kolibri-1`). The measurement owed: `vllm serve` with
the plugin on a GPU lease, greedy decode of a fixed prompt set, recorded per
[the oracle rules](../../AGENTS.md). The plugin also ships fp8 variants; the
bf16 artifact is the gate denominator.

## Notes for the port

- The architecture is structurally close to surfaces this tree already serves:
  the hybrid sliding/full KV-group pattern mirrors MiMoV2's two-group spec, and
  the MoE shape matches the existing router machinery. The new parts are the
  kolibri1 tokenizer scheme (no BOS, pad 127901, eos 127906) and the
  sliding-window geometry (513).
- The Kolibri tech report
  ([PDF](https://aleph-alpha.com/downloads/tech-report.pdf)) is the architecture
  reference: 78.1B total / 3.46B active, 24T tokens, German >20% of the mix.
- Quantized arms are a standing requirement: the repo and plugin ship bf16 and
  fp8; the GGUF k-quant conversion is part of the port row, with the checkpoint
  revision + sha256 pinned in docs/USAGE.md.
