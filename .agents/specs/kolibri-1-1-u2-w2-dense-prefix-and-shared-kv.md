# SPEC — Kolibri-1.1 W2 (U2+U4): dense prefix and shared-KV layer plumbing

- Owning row: `MODEL-TEXT-kolibri-1-1-kolibri1-1-for-causal-lm`.
  Issue: ISSUE-LOCAL-01M861BE22E3VJNM13ZFW19NX1.
- Coordinator: oc-41 go-ahead 2026-11-24. Worktree `/tmp/vllm-kolibri11-u2`,
  branch `row/MODEL-TEXT-kolibri-1-1-U2`, pinned base
  `92ea7d3a8306ec70e82512a25ff6668519e706b7` (current main, includes U1 #3596).
  One PR carrying spec and implementation.
- Evidence inputs: U1's four LFS-verified arms and header/tensor walks in
  `docs/bench-evidence/kolibri11-u1-20261123/`, and the accepted assessment
  `docs/bench-evidence/kolibri11-assessment-20261123/ASSESSMENT.md`.

## Trigger / problem

Kolibri-1.1 reuses the 1.0 geometry but changes the forward's layer contract
in two ways that the current tree cannot represent:

1. Layers 0-1 use a dense FFN (`mlp_only_layers=[0,1]`,
   `mlp_only_intermediate_size=3584`) instead of the 1.0 MoE block. Our
   `Kolibri1Params` comment still says "every layer is MoE (no dense layers)"
   and `EnumerateKolibri1Tensors` expects eight MoE tensors on every layer.
2. `layer_types` contains a third kind, `indexed_attention`, and
   `kv_source_layers` makes six layers read another layer's K/V. Today
   `ParseKolibri1Params` refuses any kind other than sliding/full, and
   `IsSlidingLayer()` would silently classify `indexed_attention` as full.

U1 already refuses 1.1 GGUFs by marker KV. This wave makes the *remaining*
1.1 shape representable in config, tensor accounting and the KV-cache spec,
without pretending the U3 sparse kernel exists.

## Scope

IN:
1. Extend `Kolibri1Params` with `index_layers`, `kv_source_layers`,
   `mlp_only_layers`, `mlp_only_intermediate_size`, and a third layer kind
   `indexed_attention`. Parsing accepts them only when the 1.1 marker KVs are
   present; a 1.0 config keeps its exact current behaviour (byte-identical
   key set and defaults). Unknown kinds still refuse by name.
2. Tensor-map/accounting: accept the dense-prefix tensor names on layers
   0-1 and the absence of `attn_k` / `attn_v` / `attn_k_norm` on the six
   shared-KV layers (9, 14, 34, 39, 44, 49). Accounting must not report
   legitimate 1.1 omissions as missing or unaccounted.
3. KV-cache spec: assign layers by effective KV source so a consumer and its
   source share one group, keep the 40 sliding layers in the SWA group, and
   put NoPE indexed layers in a named full-span group. Add a named refusal
   at the forward boundary for the still-unimplemented sparse mask, so no
   layer silently runs with the wrong attention semantics.
4. Align the guard marker set with the six keys from U1 and burn-vllm PR #8
   (notably `kv_source_map` and the indexer KVs). No guard weakening.
5. Hermetic tests over synthetic configs/GGUF headers and tensor-name lists;
   the four real walks are evidence for the field values, not a test fixture.

OUT: U3's indexed-attention kernel, candidate mask/pool, top-2048 selection,
NVFP4 KV cache dtype, paged serving integration (U5), quantized-arm gates
(U6), any token-parity or throughput claim, and any public benchmark ID.

## Upstream anchors

- `Aleph-Alpha/Kolibri-1.1/config.json` (HF main at study time): arch
  `Kolibri1_1ForCausalLM`, 50 layers, hidden 2560, 48/4 heads, head_dim 128,
  384 experts top-6, sliding 513, `mlp_only_layers=[0,1]`,
  `mlp_only_intermediate_size=3584`, `index_layers=[4,14,19,29]`, 50-entry
  `kv_source_layers`, `index_topk=2048`, `index_pool_size=16384`,
  `index_compression=16`, `index_num_sink_tokens=8`,
  `index_num_local_tokens=128`, `kv_cache_qat=true`, `head_dtype=float32`.
- llama.cpp PR #32891 (OPEN, head `5f4d222898dd`): converter and sparse
  attention; unmerged, so it is a mechanism reference, not an oracle.
- Local anchors: `include/vllm/model_executor/models/kolibri1.h:25-70`,
  `src/vllm/model_executor/models/kolibri1_registry.cpp:62-82` (layer kinds),
  `:188-225` (`MakeKolibri1KVCache`),
  `src/vllm/model_executor/models/kolibri1_weights.cpp:252-273` (accounting).

## Design

- Parse 1.1 fields into params with explicit defaults for 1.0; validate
  lengths: `kv_source_layers.size()==num_hidden_layers`,
  `mlp_only_layers` within `[0, num_hidden_layers)`,
  `index_layers` subset of layers whose `layer_types` is `indexed_attention`.
- `IsSlidingLayer()` stays exact (`== "sliding_attention"`); add a separate
  `IsIndexedLayer()` rather than widening the predicate, so 1.0's grouping
  cannot change.
- KV groups are built from the effective source: a layer with
  `kv_source_layers[i] != i` is added to the source layer's group. Indexed
  layers get their own named group with a full-span, RoPE-free spec; the
  sparse mask remains U3 and the forward refuses by name until then.
- Tensor accounting takes an optional 1.1 shape: dense prefix tensors on
  0-1, and no K/V/norm tensors on shared-KV layers. Refusals name the row.

## Tests

1. Parse: 1.0 config unchanged; 1.1 fields parsed and validated; each bad
   length/range refused by name.
2. Tensor accounting: synthetic 1.1 name list accepted; a 1.0 list with 1.1
   fields absent unchanged; genuinely missing tensors still refused.
3. KV groups: 40 sliding layers in the SWA group, 50-source groups for the
   six shared-KV layers, indexed layers in their own named group; no layer
   duplicated or dropped. Mutation: reverting the group assignment must
   turn the test red.
4. Reachability: drive the config/KV path through the production loader seam
   so deleting the registry hook turns the gate red.
5. RED-first for every new guarantee; restore scratch mutations byte-for-byte.

## Gates / evidence

- `scripts/agent-preflight.sh --staged`, `check-agent-record.py --report` rot 0,
  commit style/trailers, `agent-pr-body.py --pr <N>` exit 0, focused ctest.
- Evidence lives in a flat `docs/bench-evidence/<run-id>/` dir; the U1 walks
  are cited, not copied.

## Dependencies / risks

- No matrix row exists for this model row in `.agents/model-matrix.md` (same
  gap U1 landed with). Record as owed; if adding one, make a scoped edit with
  unrelated rows byte-identical, since that file is a shared lock.
- The KV-cache group contract must stay valid for 1.0; a regression there is
  a stop condition, not a judgement call.
- No CPU-gateable oracle: U2 gates are parse/accounting/group unit tests.

## Stop conditions

`NEEDS_CONTEXT` if the pinned base or the 1.1 config fields are unavailable;
`NEEDS_DECISION` if a group assignment would change 1.0 behaviour. Do not stop
at a correctable failing test. Never claim token parity or publish a number.

## Git integration

One PR on `row/MODEL-TEXT-kolibri-1-1-U2` to `localai-org/vllm.cpp:main`, push
only to `mine`. Trailers/PR body per AGENTS.md. Fresh reviewer must PASS before
merge-ready. Never merge or close a PR here.

## Now

SPEC — no implementation yet. Issue ISSUE-LOCAL-01M861BE22E3VJNM13ZFW19NX1
filed; base pinned; awaiting spec commit.

## Owed

- U3 indexed-attention kernel + retrieval cache and its goldens.
- U5 paged integration/serving and U6 quantized-arm gates.
- A matrix row for MODEL-TEXT-kolibri-1-1-kolibri1-1-for-causal-lm.
- A gateable oracle before any token-parity claim.
