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
role = secondary
upstream = https://github.com/Aleph-Alpha/aleph-alpha-inference
scope = the kolibri1 (Kolibri1ForCausalLM) architecture only: 50-layer hybrid sliding/full attention (4:1, window 513), MoE 384 experts x 6 active + shared expert, hidden 2560, vocab 128000, 262k (extendable 1M) context, English-German; includes the kolibri1 reasoning/tool-call parsers and the chat template
pin = 049a6a7bd240
pin_label = PyPI v1.0.0 (049a6a7bd240)
pinned_on = 2026-10-03
gateable = yes
evidence = docs/bench-evidence/kolibri1-readj/shadow-recon-bisect-20261015.md
```

## Gateability

`gateable = yes` on CPU/aarch64 (2026-10-15, host thalia): `aleph-alpha-inference`
1.0.0 (049a6a7bd240) via vLLM 0.31.0+cpu (built from source), greedy decode of the
8-prompt battery, TP=4 (`distributed_executor_backend=mp`), bf16. The primary
goldens are committed at `tests/vllm/models/kolibri1_goldens_primary.json`
(generator `scripts/gen-kolibri1-primary-goldens.py`, smoke test
`scripts/aleph-alpha-inference-smoke.py`; venv recipe torch==2.13.0+cpu from the
PyTorch cpu wheel — NOT plain PyPI — plus `VLLM_TARGET_DEVICE=cpu` source build,
~6 min at -j64).

Two facts the GPU-lease framing had wrong: (1) the fp8 checkpoint is UNUSABLE on
CPU outright — vLLM's CPU backend has no fp8 ScaledMM kernel — so the bf16
dequant was always mandatory, not a preference; (2) the first bf16 conversion
shipped an internally contradictory index (fp8 `weight_scale_inv` siblings left
in place) that routed vLLM's MoE to its fp8 path — fixed by the landed
`scripts/gen-kolibri1-bf16-reference.py` (e7bba5335); use the
`Kolibri-1-bf16-clean` directory. `gpu_memory_utilization` is a memory FRACTION
(re-tune per run on a shared box; 0.74 worked where 0.85/0.88 did not), and TP=4
is required because the bf16 tower (151 GB) exceeds one 63.5 GiB node.

**The transcription caveat**: the pre-existing `tests/vllm/models/kolibri1_goldens.json`
is a PURE-TORCH TRANSCRIPTION of the plugin's math, not plugin output — and the
plugin (the primary oracle) disagrees with it on 6/8 first tokens and 3/8 full
sequences (measured 2026-10-15, both artifacts committed). Gates scored against
the transcription are calibrated to a proxy; the schema decision (generated-ids
across all three implementations vs exporting logits from the Python side) is
recorded as open in the primary goldens' `vs_transcription_goldens` field.

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

## Kolibri-1.1 and plugin v2.0.0 (recorded 2026-11-23, NOT a pin advance)

Aleph Alpha shipped Kolibri-1.1 (`Kolibri1_1ForCausalLM`, `model_type
kolibri1_1`) and plugin v2.0.0 the same day. The pin above stays v1.0.0
(`049a6a7bd240`): v2.0.0 changes the supported vLLM minor (0.29 → 0.33), needs
transformers ≥ 5.16.1, and its Kolibri-1.1 kernels are CUDA-only (SM90/SM100,
NVFP4 KV cache + FlashAttention 4, TP=1), so the v1 CPU gateability evidence
does not carry. Wheel sha256
`9c98c2610be03dce0c541b40461a4a4ef232cb52a9c5bee7e5201e6b1c3c9102`
(PyPI, uploaded 2026-11-23 14:45 UTC); the GitHub revision for v2.0.0 was not
recorded at write time and is owed with the pin advance. Until that advance
(a CUDA-lease measurement, likely under row
`MODEL-TEXT-kolibri-1-1-kolibri1-1-for-causal-lm`), kolibri1_1 has NO gateable
plugin oracle and v1.0.0 remains the record for kolibri-1. Assessment detail:
docs/bench-evidence/kolibri11-assessment-20261123/ASSESSMENT.md.
