ID: ISSUE-GH-941
Title: `.agents/specs/nemotron-h-abi-e2e.md` §2 named `dense_attn::AttnBlock` as NemotronH's device attention seam, and that block cannot serve this architecture: it applies `vt::RopeNeox` unconditionally (`include/vllm/model_executor/models/dense_attn_block.h:497` @ `10002648199` — the `:496` cited in-tree at `nemotron_h_device.cpp:56` is a comment line, so the anchor is stale as well as the claim), while Nemotron-H has NO RoPE at all: a case-insensitive grep for `rotary`, `rope`, `q_norm`, `k_norm` over `vllm/model_executor/models/nemotron_h.py` at the pinned oracle `5559679229bc` returns ZERO hits, and its four-line attention forward (`:474-483`) sends `q` and `k` straight into `self.attn` with no positional transform. It also reads `cfg.rms_norm_eps`, which `src/vllm/transformers_utils/hf_config.cpp:551` defaults to `0.0` for a checkpoint that ships `layer_norm_epsilon` and `norm_eps` and no `rms_norm_eps`. There is NO rope-free entry point: the header's whole public surface is enumerated in the A2-P spec §2.3, `:490-493` selects only WHICH rope implementation, and `rotary_dim == 0` ABORTS at `src/vt/ops.cpp:1427-1429` rather than bypassing — so neither failure mode announces itself as one. Same shape as [#810](https://github.com/mudler/vllm.cpp/issues/810): a shared function reconstructing behaviour from HF-config fields the model does not ship, and two of its three failure modes are silent. PARTLY FIXED IN FLOW by the A2-P spec: item 1 (correct the seam claim so no later implementer is sent that way) is answered by [`specs/nemotron-h-a2p-paged-forward.md`](../../specs/nemotron-h-a2p-paged-forward.md) §2.3 plus the correction block appended to the governing spec's §1; item 2 (a model-local block in the `granite.cpp:84` / `gemma4.cpp:206` idiom) was already answered by A2-R at `598226e96` and is extended by A2-P. STILL OPEN and deliberately not fixed here: item 3, whether defaulting `rms_norm_eps` to `0.0` rather than refusing is right in general — that is tree-wide, needs its own red-before, and is listed under `## Owed` in the A2-P spec
Row: MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm
State: UNKNOWN
Kind: bug
GitHub: 941
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:320`

### Frozen archive evidence

> | [#941](https://github.com/mudler/vllm.cpp/issues/941) | `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` | `.agents/specs/nemotron-h-abi-e2e.md` §2 named `dense_attn::AttnBlock` as NemotronH's device attention seam, and that block cannot serve this architecture: it applies `vt::RopeNeox` unconditionally (`include/vllm/model_executor/models/dense_attn_block.h:497` @ `10002648199` — the `:496` cited in-tree at `nemotron_h_device.cpp:56` is a comment line, so the anchor is stale as well as the claim), while Nemotron-H has NO RoPE at all: a case-insensitive grep for `rotary`, `rope`, `q_norm`, `k_norm` over `vllm/model_executor/models/nemotron_h.py` at the pinned oracle `5559679229bc` returns ZERO hits, and its four-line attention forward (`:474-483`) sends `q` and `k` straight into `self.attn` with no positional transform. It also reads `cfg.rms_norm_eps`, which `src/vllm/transformers_utils/hf_config.cpp:551` defaults to `0.0` for a checkpoint that ships `layer_norm_epsilon` and `norm_eps` and no `rms_norm_eps`. There is NO rope-free entry point: the header's whole public surface is enumerated in the A2-P spec §2.3, `:490-493` selects only WHICH rope implementation, and `rotary_dim == 0` ABORTS at `src/vt/ops.cpp:1427-1429` rather than bypassing — so neither failure mode announces itself as one. Same shape as [#810](https://github.com/mudler/vllm.cpp/issues/810): a shared function reconstructing behaviour from HF-config fields the model does not ship, and two of its three failure modes are silent. PARTLY FIXED IN FLOW by the A2-P spec: item 1 (correct the seam claim so no later implementer is sent that way) is answered by [`specs/nemotron-h-a2p-paged-forward.md`](../../specs/nemotron-h-a2p-paged-forward.md) §2.3 plus the correction block appended to the governing spec's §1; item 2 (a model-local block in the `granite.cpp:84` / `gemma4.cpp:206` idiom) was already answered by A2-R at `598226e96` and is extended by A2-P. STILL OPEN and deliberately not fixed here: item 3, whether defaulting `rms_norm_eps` to `0.0` rather than refusing is right in general — that is tree-wide, needs its own red-before, and is listed under `## Owed` in the A2-P spec | bug |

## Resolution

-
