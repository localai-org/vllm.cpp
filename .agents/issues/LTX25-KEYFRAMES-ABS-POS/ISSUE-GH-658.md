ID: ISSUE-GH-658
Title: Neither shipped LTX-2.5 DiT could be loaded inside the contract, refused from OPPOSITE directions, and both refusals were correct while `keyframes_abs_pos_embedding` was unported: the vonkaiser FP8 tower carries a TRAINED `F8_E4M3 [1,4096]` + scalar `F32` scale (4096/4096 bytes non-zero), and the first-party NVFP4 DiT declares `use_keyframes_abs_pos_embedding=true` while carrying no tensor. Upstream adds a learned `[1, inner_dim]` per-token bias to the first latent frame UNCONDITIONALLY (`model.py:217-219`, `transformer_args.py:23-43` at `:269`, `tools.py:186-196`), so every render this port could produce was missing a trained term. The NVFP4 arm is settled by execution and reproducible through `scripts/measure-ltx2-keyframes-meta.py`: the parameter stays on `meta` before AND after `load_state_dict(..., assign=True)` while a neighbour materialises, so that arm loads and applies NOTHING rather than refusing or synthesising a zero. Spec [`ltx25-keyframes-abs-pos.md`](../../specs/ltx25-keyframes-abs-pos.md), campaign #644
Row: LTX25-KEYFRAMES-ABS-POS
State: UNKNOWN
Kind: bug
GitHub: 658
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:216`

### Frozen archive evidence

> | [#658](https://github.com/mudler/vllm.cpp/issues/658) | `LTX25-KEYFRAMES-ABS-POS` | Neither shipped LTX-2.5 DiT could be loaded inside the contract, refused from OPPOSITE directions, and both refusals were correct while `keyframes_abs_pos_embedding` was unported: the vonkaiser FP8 tower carries a TRAINED `F8_E4M3 [1,4096]` + scalar `F32` scale (4096/4096 bytes non-zero), and the first-party NVFP4 DiT declares `use_keyframes_abs_pos_embedding=true` while carrying no tensor. Upstream adds a learned `[1, inner_dim]` per-token bias to the first latent frame UNCONDITIONALLY (`model.py:217-219`, `transformer_args.py:23-43` at `:269`, `tools.py:186-196`), so every render this port could produce was missing a trained term. The NVFP4 arm is settled by execution and reproducible through `scripts/measure-ltx2-keyframes-meta.py`: the parameter stays on `meta` before AND after `load_state_dict(..., assign=True)` while a neighbour materialises, so that arm loads and applies NOTHING rather than refusing or synthesising a zero. Spec [`ltx25-keyframes-abs-pos.md`](../../specs/ltx25-keyframes-abs-pos.md), campaign #644 | bug |

## Resolution

-
