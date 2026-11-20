# SPEC — `NEW-CANDIDATE-PORTS` — W1 olala: the GGUF name-map + census-validating weights arm (forward waves follow)

- Owning row: `NEW-CANDIDATE-PORTS`
- Trigger: row `NEW-CANDIDATE-PARSERS` landed (parse + refuse-by-name for olala / hy_v4_flash / Maxwell) and named the arch ports as owed to this row. Developer direction 2026-11-20: start with olala; W1 = loader arm + census, forward in later waves.

## Artifact (pinned)

`/mnt/models/olala-v1-mini/` — HF `OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF`, snapshot `b9f5d20a` (verify against the HF cache ref at commit time), arms: `Olala-v1.0-mini-7A1B-bf16.gguf` (13,594,188,096 B), `-q6_k.gguf`, `-q5_k_m.gguf`, `-q8_0-safe2.gguf`, plus the chat template. GGUF v3, `general.architecture = "olala"` (custom), 989 tensors, 52 KVs (parser-row walk: `docs/bench-evidence/new-candidates-parsers-20261119/olala-kvs.txt`).

## W1 enumeration (committed evidence)

`docs/bench-evidence/new-candidates-ports-olala-20261120/` (FLAT):

- `olala-tensor-walk-bf16.txt`, `olala-tensor-walk-q6k.txt` — full 989-tensor `dump_container` manifests (name, dtype, shape, bytes, 64KiB-prefix sha256).
- `olala-namemap.md` — the aggregated census: 2 globals (`token_embd.weight`, `output.weight` [151936,1536]); 36 blocks, each with attn_output / exp_probs_b / ffn_gate_inp / ffn_up_exps [256,768,384] / ffn_down_exps [256,384,768] / ffn_up_shexp / ffn_down_shexp [1536,1536] / ffn_latent_up [1536,384] / ffn_latent_down [384,1536] / geo_{mixer,mlp}_{bias,scale} [1]; 29 M-layers add the ssm_* set (ssm_in [6288,1536], ssm_in_dyn [1056,1536], ssm_b/c_bias [48,4,128], ssm_b/c_norm [128], ssm_d/dt_bias [48], ssm_m3_* , ssm_mimo_{x,z,o} [48,4,64]); 7 V-layers (4,9,14,19,24,29,34) add attn_{q [6144,1536], gate [4608,1536], output [1536,4608], k_norm, q_norm, lambda, shift_k, shift_v, softmax_scaler, wa_k, wa_v, wb_k, wb_v}. Dtype census: bf16 = 375 BF16 + 614 F32; q6_k = 303 Q6_K + 614 F32 + 36 Q8_0 (ffn_up_exps) + 36 BF16 (ffn_latent_up). Cross-check: `olala.layer_types` read directly from the file is 36 chars with V at [4,9,14,19,24,29,34], agreeing with the tensor walk (attn_q presence).

## W1 deliverables

1. `include/vllm/model_executor/models/olala_gguf_weights.h` + `src/vllm/model_executor/models/olala_gguf_weights.cpp` (new TU, added to the vllm source list in CMakeLists.txt beside the other `*_gguf_weights.cpp`):
   - the pattern→role name-map (regex per role: globals, per-layer common, M-only ssm_*, V-only attn_*);
   - `struct OlalaGgufCensus { ... }` + `OlalaGgufCensus CensusOlalaGguf(const GgufFile&)` — classifies every tensor in `gguf.Tensors()`, counts per role, records unmapped names;
   - `void ValidateOlalaGguf(const GgufFile&, const HfConfig&)` — throws `std::runtime_error` naming the offenders when: any tensor is unmapped; counts mismatch (36 blocks, 29 M + 7 V, 256 experts, per-layer tensor counts); shapes mismatch the config (token_embd [vocab, hidden], output [vocab, hidden], ffn_*_exps [experts, latent, ff], attn_q [heads*key_length, hidden] on V layers, attn_output [hidden, inner_size] on M / [hidden, 4608] on V, ssm shapes vs ssm KVs carried in the config's verbatim map);
   - dtype policy recorded, not enforced yet (forward wave): bf16 arm = BF16/F32 as shipped; q6_k arm keeps Q6_K/Q8_0 quantized (the keep-quant path, `gguf_keep_quant.h` precedent) and F32/BF16 as-is.
2. Loader wiring (`src/vllm/entrypoints/model_loader.cpp`): on the `IsOlalaGguf` path, call `ValidateOlalaGguf(gguf, config)` right after `HfConfigFromGgufDispatch` and BEFORE the existing `OlalaForwardRefusal()` throw — the census is reached from the production entry point, and the forward boundary message is unchanged.
3. CPU test `tests/vllm/models/test_olala_gguf_weights.cpp` (registered in tests/CMakeLists.txt beside test_new_candidate_parsers), gated on the pinned path existing (skip 77 when absent, the established checkpoint-gated pattern): opens the bf16 and q6_k artifacts, asserts 989 tensors, 0 unmapped, per-role counts (2 globals; 36× common set; 29× ssm set; 7× attn set), the dtype histograms above, and the shape-vs-config assertions via `OlalaHfConfigFromGguf` + `ValidateOlalaGguf` (no throw), plus a negative case (a synthetic census with one tensor renamed → unmapped listed).
4. docs/USAGE.md: olala parse-status entry updated from "parses: yes; forwards: no" to add "GGUF loader arm: census-validated (W1); forwards: no — W2+".

## Out of scope (W2+, named so they are not discovered later)

- The forward itself: M-layer SSM semantics (mimo/tpa/geodesic — `geo_*`/`ssm_m3_*` sets are custom, no upstream mirror; secondary oracle TBD per the oracle registry), V-layer attention (sliding window 32768, softcapping 150, per-head gates), MoE router (sigmoid, exp_probs_b + expert_weights_scale 2.45, top-6 of 256) + latent experts (384) + shared expert, tokenizer (gpt2/qwen2 pre-tokenizer — the qwen2 split pattern exists).
- Quantized-arm decode kernels beyond what the keep-quant path already provides (Q6_K/Q8_0 CPU+CUDA exist; q5_k_m needs Q5_K).
- Hy4 and Maxwell ports (their own waves on this row).

## Gates

- CPU-only: the new test passes against both pinned artifacts (skip 77 when absent); `check-agent-record.py --report` rot 0; preflight sweep green (known environmental gaps: dotnet, test_check_test_registration); the touched TUs compile in the CPU build dir; `check-tt-public-surface.py` unaffected (no vt::tenstorrent refs added).
- The loader path change is compile-verified and the census is exercised by the test through the same `GgufFile::Open` + dispatch functions the loader uses.

## Risks/decisions

- The name-map is regex-based over `blk.N.` prefixes; a future olala checkpoint with per-expert tensor suffixes (e.g. `ffn_up_exps.N.weight`) would need the map extended — the census's unmapped list makes that loud, not silent.
- layer classification: tensor presence (attn_q) and the layer_types KV agree on the pinned artifact (36 chars, V at 4,9,14,19,24,29,34); W1 uses tensor presence, W2 should assert both.
- No upstream (vLLM/llama.cpp) carries `olala` — per AGENTS.md the port needs a secondary oracle; candidate: the publisher's own inference code if present in the HF repo, else llama.cpp PR if any. That decision is W2's, recorded here as owed.

## Git integration

One pull request (row claim answer recorded 2026-11-20 in developer-preferences.md ## Git integration).

## Now

W1 in progress on branch `row/NEW-CANDIDATE-PORTS`, worktree `/tmp/port-olala`, base `6cbb0194e`.
