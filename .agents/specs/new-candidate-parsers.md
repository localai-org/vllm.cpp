# SPEC — `NEW-CANDIDATE-PARSERS` — the CPU-side config parsers for the three new candidates: olala, hy_v4_flash, Maxwell

- Owning row: `NEW-CANDIDATE-PARSERS` (record ID `BACKEND-NEW-CANDIDATE-PARSERS`)
- Trigger: the developer downloaded three new candidate artifacts (all pinned, all byte-ready); none of them parses today. This row delivers the PARSER layer only — the config builders and the loader rows so each artifact PARSES and refuse-by-names at the forward. The arch ports (forwards, kernels) are their own rows and are explicitly OUT.

## The three artifacts (downloaded, pinned)

1. **Olala-v1.0-mini-7A1B**: `/mnt/models/olala-v1-mini/` — GGUF v3, `general.architecture = "olala"` (CUSTOM — a new arch string), 989 KVs, quants q8_0-safe2/q6_k/q5_k_m/bf16. Tags: mamba/hybrid MoE, 7A1B. The mamba-hybrid shape is near our qwen3.5 GDN path but THIS ROW DOES NOT PORT IT — parsers only.
2. **Hy4-flash**: `/mnt/models/hy4-flash-gguf/` — GGUF v3, `general.architecture = "hy_v4_flash"` (CUSTOM), 1415 KVs, Q4_K_M 2 shards + mmproj. The tencent config (tencent/Hy4-flash) carries Engram n-gram tables, a compressed indexer, a DFLY 5-layer MTP module, a vision tower — the PARSER maps what the GGUF KVs carry; the semantics are the arch-port row's.
3. **Maxwell-V1-Flash**: `/mnt/models/maxwell-v1-flash/snapshot/` — **safetensors** (35 shards, 73.6 GB) + config.json with **remote code** (`auto_map`: MaxwellMoeConfig/MaxwellMoeForCausalLM), 36.8B params. Parsed from the config.json + the safetensors header (the enumerate path), NOT the remote code (no trust-remote-code execution in the parser).

## The deliverable (per artifact)

Following the loader's additive arch-table pattern (src/vllm/entrypoints/model_loader.cpp:1223 — ONE row per arch, the config builder in its own TU, the refusal names the supported set BY READING the table):

1. **The W0 enumeration committed**: the full KV/shape inventory per artifact (the GGUF header walk for olala/hy_v4_flash — the parser pattern in the dispatch's probes; the safetensors header walk + config.json field map for Maxwell) into docs/bench-evidence/new-candidates-parsers-20261119/ (FLAT).
2. **The config builders**: `OlalaHfConfigFromGguf`, `HyV4FlashHfConfigFromGguf`, `MaxwellConfigFromJson` — each synthesizing an HF-shaped config the loader's downstream can consume, mapping the artifact's KVs/fields by name. Unmapped fields: carried verbatim (the glm-dsa builder precedent — the builder hands the parsed params to the same validator a config.json descends through).
3. **The loader rows**: ONE row per arch in the additive table, each naming its config builder.
4. **Refuse-by-name at the forward**: after the parser, the model forwards refuse by name (the parsers do not implement the arch port) — the refusal names the missing row/forward so a user knows the parser vs forward boundary.
5. **CPU-only tests**: each parser parses its pinned artifact end-to-end on CPU (the three pinned paths are the test fixtures), the unmapped-field carry-verbatim assertions, the refuse-by-name assertions. NO device, NO forwards.
6. **docs/FEATURES.md/USAGE.md**: the three arches' parse-status entries (parses: yes; forwards: no — the arch-port rows named).

## Scope

IN: the three config parsers/builders, the loader rows, the enumerations, the CPU tests, the docs. OUT: the arch ports (forwards/kernels), the GPU paths, the tokenizers beyond what the parser carries verbatim, any device code.

## Risks/decisions

- The olala/hy_v4_flash GGUFs carry 989/1415 KVs — far more than our existing rows handle. The builders map the STANDARD fields by name and carry the rest verbatim; the W0 enumeration records the unmapped set so the arch-port rows know what they inherit.
- Maxwell's remote code is NOT executed (no trust-remote-code in the parser) — the config.json's explicit fields are the source; fields the remote code would default are UNKNOWN and recorded as such.
- Hy4's 1415 KVs include the vision tower — the parser carries vision fields verbatim like the rest; the vision port is out of scope.

## Gates

CPU-only: each parser parses its pinned artifact (the three paths as fixtures), the refuse-by-name forward assertions, `check-agent-record.py`, preflight --staged DETACHED (known gaps: dotnet, test_check_test_registration). NO device legs.

## Git integration

One pull request; worktree `/tmp/tt-parsers`; push to `mine`.
