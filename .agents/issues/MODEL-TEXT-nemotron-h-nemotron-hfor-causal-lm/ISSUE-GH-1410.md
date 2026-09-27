ID: ISSUE-GH-1410
Title: `scripts/check-runner-routing-consistency.py` resolves the registry hook's delegate across translation units only for the `Class::ForwardDevice` shape (`_DELEGATE`, `:142-145`) or a helper defined in the registry TU itself (`classify_with_helpers`, `:246-271`). NemotronH's production device forward is a FREE function in another TU — `ForwardNemotronHForCausalLM` (`nemotron_h_registry.cpp`) calls `NemotronHPagedForward` (`nemotron_h_device.cpp`) — so the hop finds nothing and the hook's own host-reference fall-through, which A2-P deliberately keeps below the paged fold as the numeric gate's operand, classifies the model HOST. Measured on `row/A2-Q2b-lmhead-nvfp4` after A2-Q2b put `lm_head` on the device: `NemotronHPagedForward` assigns both `fl.device_tensor` and `fl.device_storage = dlogits.ReleaseShared()`, which IS `_DEVICE_SEAM` (`:125-128`), and the checker still names the model. A FALSE RED, so the safe direction — but it holds an allowlist entry open for a clause that is MET, and the allowlist is what a reader trusts to know what is still unrouted; it is latent the other way for any future model whose device logits come from a cross-TU free function. The checker already builds the `free_fn_file` map the fix needs (invariant (b) uses it). NOT fixed in flow: it CHANGES CHECKER SEMANTICS, which `AGENTS.md` `## Changing the rules or a checker` routes to the normal row, spec and fresh-review path, and widening a classification to turn a red gate green is exactly the move that section slows down — it needs its own red-before in `tests/scripts/test_check_runner_routing_consistency.py`. Owned by row `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm`, listed under `## 5. Owed` in [`nemotron-h-a2q2b-realckpt-lmhead.md`](../../specs/nemotron-h-a2q2b-realckpt-lmhead.md)
Row: MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm
State: UNKNOWN
Kind: bug
GitHub: 1410
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:482`

### Frozen archive evidence

> | [#1410](https://github.com/mudler/vllm.cpp/issues/1410) | `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` | `scripts/check-runner-routing-consistency.py` resolves the registry hook's delegate across translation units only for the `Class::ForwardDevice` shape (`_DELEGATE`, `:142-145`) or a helper defined in the registry TU itself (`classify_with_helpers`, `:246-271`). NemotronH's production device forward is a FREE function in another TU — `ForwardNemotronHForCausalLM` (`nemotron_h_registry.cpp`) calls `NemotronHPagedForward` (`nemotron_h_device.cpp`) — so the hop finds nothing and the hook's own host-reference fall-through, which A2-P deliberately keeps below the paged fold as the numeric gate's operand, classifies the model HOST. Measured on `row/A2-Q2b-lmhead-nvfp4` after A2-Q2b put `lm_head` on the device: `NemotronHPagedForward` assigns both `fl.device_tensor` and `fl.device_storage = dlogits.ReleaseShared()`, which IS `_DEVICE_SEAM` (`:125-128`), and the checker still names the model. A FALSE RED, so the safe direction — but it holds an allowlist entry open for a clause that is MET, and the allowlist is what a reader trusts to know what is still unrouted; it is latent the other way for any future model whose device logits come from a cross-TU free function. The checker already builds the `free_fn_file` map the fix needs (invariant (b) uses it). NOT fixed in flow: it CHANGES CHECKER SEMANTICS, which `AGENTS.md` `## Changing the rules or a checker` routes to the normal row, spec and fresh-review path, and widening a classification to turn a red gate green is exactly the move that section slows down — it needs its own red-before in `tests/scripts/test_check_runner_routing_consistency.py`. Owned by row `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm`, listed under `## 5. Owed` in [`nemotron-h-a2q2b-realckpt-lmhead.md`](../../specs/nemotron-h-a2q2b-realckpt-lmhead.md) | bug |

## Resolution

-
