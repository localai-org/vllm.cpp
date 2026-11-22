# SPEC — `NEW-CANDIDATE-PORTS` W2 olala: the forward port against the author's two oracles

- Owning row: `NEW-CANDIDATE-PORTS`. Issue: ISSUE-LOCAL-01M7ZAHMYW87MPTAW4J78HYMB7.
- W1 landed: GGUF census-validated loader arm (`olala_gguf_weights.{h,cpp}`,
  989 tensors mapped, `ValidateOlalaGguf` on the production load path,
  forward still refuses by name). Evidence:
  `docs/bench-evidence/new-candidate-ports-olala-20261120/`.

## Oracles (pinned 2026-11-21, both `gateable = no` until measured — the issue owes it)

- **`olala-transformers`**: `OVHaiLLM/Olala-v1.0-mini-7A1B` @
  `f781f12498adc9bdd71b7dd59d1311434369c684` — the author's transformers custom
  code (`modeling_olala.py` 2350 lines, `configuration_olala.py`), Apache-2.0,
  bf16 safetensors 13.6 GB (fits this host's 255 GiB RAM on CPU). Behavior
  oracle: per-op reference values and greedy token ids.
  Record: `.agents/oracles/olala-transformers.md`.
- **`llama-cpp-olala`**: `OVHai-LLM/llama.cpp` branch `olala-master` @
  `bd20ca9b471fdcb3f03b75f82d3cf575414bc573` (`src/models/olala.cpp`, fused
  `MAMBA3_MIMO`/`GEODESIC` kernels) — the only llama.cpp that loads the
  published GGUFs; quant-matched token-exact floor for the k-quant arms.
  Record: `.agents/oracles/llama-cpp-olala.md`.
- vLLM registers no `olala` arm at any pin — primary oracle absent by
  construction; these secondaries are admissible per AGENTS.md §"When vLLM has
  no implementation". They never outrank vLLM and never become the mirror
  source; the port mirrors the AUTHOR's code, cited by file:line.

## Artifacts (pinned)

- GGUF container (W1 fixtures): `/mnt/models/olala-v1-mini/`, HF
  `OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF` — bf16 (13.59 GB, sha in W1 evidence),
  q6_k, q5_k_m, q8_0-safe2. Arch `olala`, 989 tensors, 52 KVs.
- HF safetensors (oracle checkpoint): `OVHaiLLM/Olala-v1.0-mini-7A1B` @
  `f781f124…`, `model.safetensors` 13,597,677,592 B — same weights, HF names.

## Architecture map (from `modeling_olala.py` @ f781f124; GGUF names from the W1 census)

Config cross-check: author `config.json` has `layers_config =
MMMMVMMMMVMMMMVMMMMVMMMMVMMMMVMMMMVM` (36 chars, V at 4,9,14,19,24,29,34) —
AGREES with both the GGUF `olala.layer_types` KV and the W1 tensor walk
(attn_q presence). GGUF `attention.sliding_window=32768` = config
`slw_wsize` (config also carries `sliding_window_size=65536` — the V layer uses
slw_wsize; recorded).

Per-layer `OlalaMonoBlock.forward` (modeling_olala.py:1735-1812), with
`geodesic_update=True`, `layer_norm_scaling=False` (lns=1), `mixer_gn=False`,
`input_norm`/`postmixer_norm` = Identity:

```
residual = x
y = mixer(x)                      # M: OlalaMamba3MimoFast; V: OlalaDifferentialTensorProductAttentionV2
y = mixer_proj(y)                 # attn_output.weight ([1536, 3072] M / [1536, 4608] V)
x = geodesic_mixer(residual, y)   # OlalaGeodesicNorm, geo_mixer_{scale,bias}
residual = x
y = mlp(x)                        # OlalaMoE (every layer; feed_forward_length=0)
x = geodesic_mlp(residual, y)     # OlalaGeodesicNorm, geo_mlp_{scale,bias}
```

`OlalaGeodesicNorm.forward` (modeling_olala.py:1596-1617): sphere projection —
`proj = <x,g>/||x||^2; grad = g - proj*x; theta = clamp(||grad||/||x||, pi/4);
theta = clamp((theta*scale + bias)/(layer_idx+1), pi/4);
out = x*cos(theta) + (grad/||grad||)*||x||*sin(theta)`. Scalars are the [1]
F32 geo_* tensors.

**V layer** (`OlalaDifferentialTensorProductAttentionV2`, :498-764): heads 48
(signal 36 + noise 12 = num_signal_heads_diff 36), kv heads 12, head_dim 128;
q = attn_q [6144,1536] (48*128), q_norm (attn_q_norm [128]), k/v via
attn_wa/wb_k/attn_wa/wb_v + attn_shift_k/v + attn_lambda (differential
attention: lambda*(...) pairing), per-head attn_softmax_scaler [48], softcap
150 (attn_logit_softcapping), sliding window 32768, RoPE with
`num_rope_angles=32` and `p_rotary=0.75` (apply_p_rotary_emb :424), gate:
`gate_attn=True`, `gate_type=elementwise`, `gate_act=silu`,
`zero_centered_gate=True` (bias 1.15), gate = attn_gate [4608,1536] applied as
`y *= silu(gate_proj(x) + 1.15)` BEFORE mixer_proj... (gate_before_norm=True,
:1781-1794). KV cache only on V layers (7 of 36).

**M layer** (`OlalaMamba3MimoFast`, :933-1406): mamba3 with MIMO —
`ssm_in` [6288,1536] and `ssm_in_dyn` [1056,1536] input projections (6288 =
2*(inner 3072)? + dt etc. — exact split per :1248 `_preprocess`),
`ssm_mimo_{x,z,o}` [48,4,64] (mimo_dim 4, headdim 64, 48 heads),
`ssm_b/c_bias` [48,4,128] + `ssm_b/c_norm` [128] (state 128),
`ssm_d`/`ssm_dt_bias` [48], `ssm_m3_*` (m3 trapezoid/rope extras:
mamba3_add_trapezoid=True, mamba3_rope=True, tpa_rank=4, num_rope_angles=32).
Recurrent state f32 (the fork README warns bf16 state degrades).

**MoE** (`OlalaMoE.forward`, :1512-1552): router logits = x @ ffn_gate_inp^T
([256,1536] F32) + exp_probs_b [256]; sigmoid; top-6; probs normalized,
`* moe_routed_scaling_factor 2.45`; routed path: down = ffn_latent_down
[384,1536] → experts (activation relu(x)^2, up/gate = ffn_up_exps
[256,768,384], down = ffn_down_exps [256,384,768]) → up = ffn_latent_up
[1536,384]; shared = ffn_up_shexp/down_shexp [1536,1536] (plain MLP, no gate —
moe_shared_expert_gate=False); out = shared + routed.

Globals: `token_embd.weight` [151936,1536], `output.weight` same (untied,
tie_word_embeddings=False); `final_norm=False` (no final norm — final_norm KV
False agrees).

## Waves

- **W2a (this PR)**: oracle records + AGENTS.md registry rows + this spec +
  issue. No product code.
- **W2b**: model TU skeleton — registry row for `OlalaForCausalLM`,
  `OlalaWeights` filled from the census-validated GGUF (W1 map), forward
  dispatching per layer kind with each op refusing BY NAME (mamba3-mimo,
  differential-attention, geodesic, latent-MoE), reachable from the loader
  (the blanket refusal moves from load to first forward call). CPU tests on
  the pinned GGUF: weight census already green; skeleton refuses with the
  named pieces.
- **W2c+**: port ops one at a time against `olala-transformers` per-op unit
  parity (f32 tolerance vs the torch reference), then a greedy token gate vs
  `llama-cpp-olala` on the same q6_k GGUF (quant-matched), then bf16 arm vs
  the transformers oracle on the HF checkpoint.

## Gates

- CPU-only until a device arm exists: preflight sweep, record rot 0,
  TT-surface checker unaffected, the W1 census test stays green, pr-size.
- W2b: loader reaches the model; refusal messages name the missing op.
- W2c+: per-op unit tests vs the author's code (file:line anchors); token gate
  vs the fork on the identical GGUF (sha-pinned, both sides).

## Risks/decisions

- The GGUF and HF checkpoints are the same weights under two name maps; the
  W1 name-map is written against the GGUF — the safetensors name map (for the
  transformers-oracle gates) is a separate small map, owed in W2c.
- bf16 state for the SSM recurrence is required (fork README: bf16 state
  loops); keep M-layer state f32.
- `sliding_window_size` vs `slw_wsize` discrepancy recorded above; the V layer
  uses slw_wsize=32768 (GGUF KV agrees).
- ik_llama.cpp PR #2956 also adds olala — noted as a third fork, not needed.

## Git integration

One PR per wave (row claim answer recorded 2026-11-20 for NEW-CANDIDATE-PORTS).

## Now

W2a in progress on branch `row/NEW-CANDIDATE-PORTS`, worktree `/tmp/port-olala2`,
base `cb0da7d66`.

## Now (W2c/W2d)

2026-11-21: W2a/W2b landed (PR #3583). W2c-1 (geodesic + latent MoE), W2c-2
(V differential attention), W2c-3 (Mamba3-MIMO step) and W2d-1 (the assembled
full forward, assembly smoke on the pinned bf16 artifact) are on branch
`row/NEW-CANDIDATE-PORTS` in PR #3585, worktree `/tmp/olala-w2c`. The smoke
run caught three assembly defects the per-op units missed (n_gen+1 append,
GQA k/v indexing, empty ymix scratch) — all fixed in W2d-1. W2d-2 landed
(2026-11-22): the step host writes the post-shift post-norm k/v into the
cache row at `pos`, so later tokens attend cross-token keys; suite 5/5 cases
253/253 green in Release and bounds-checked Debug. Next: the token gate vs
`llama-cpp-olala` on q6_k; the `olala-transformers` oracle run stays owed
(gateable=no, evidence #3582). The quantized arms
(q6_k/q5_k_m/q8_0-safe2 keep-quant decode) remain refused by name.
