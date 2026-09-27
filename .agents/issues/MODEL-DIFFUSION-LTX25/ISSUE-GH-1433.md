ID: ISSUE-GH-1433
Title: `Lightricks/LTX-2` is the repository the WHOLE LTX-2.5 lane mirrors — ten `.agents/specs/ltx25-*.md` files and `.agents/porting-inventory.md` carry the revision string `fd4ded7f`, and every `file:line` upstream anchor in that lane resolves there — and it has NO registry row in `AGENTS.md`, NO `.agents/oracles/<id>.md` file, and NO recorded pin. Measured at `2da236832`: `scripts/check-oracle-pins.py` reports `oracle-pins ok (9 oracles pinned)` — `diffusers`, `llama-cpp`, `llama-cpp-unsloth`, `sglang`, `sglang-omni`, `transformers`, `tt-forge`, `vllm`, `vllm-omni` — and it reads `.agents/oracles/*.md` with a GLOB, so an oracle with no file is INVISIBLE to it. The green line is what an unpinned upstream looks like from inside the gate, which is the failure `AGENTS.md` "Pin every oracle" names: "An unpinned upstream is a moving target, not an oracle. Its measurements are not reproducible." `vllm-omni` does NOT already cover it: `.agents/oracles/vllm-omni.md:5` names LTX-2.5 in scope and is the right home for the MIRRORING relationship (`vllm-project/vllm-omni` is the structure this project mirrors), but it is not the repository the specs READ — that is a THIRD repository — and `vllm-omni` is itself `pin = UNPINNED` under [#633](https://github.com/mudler/vllm.cpp/issues/633), so the lane's actual reference source is unpinned twice over, once knowingly and once with no record at all. PRE-EXISTING and NOT caused by [#1428](https://github.com/mudler/vllm.cpp/pull/1428): the `fd4ded7f` anchoring dates to the original port `cefacd2d0` (2026-08-13, #435 / #641), first carried by `f193eaf75`, `742e38acf` and `0541cbeaa`. NOT FIXED IN FLOW because adding a row to the `AGENTS.md` oracle table is a POLICY edit, which `AGENTS.md` "Changing the rules or a checker" routes to its own row, spec and fresh review. Found while repairing the fresh review of #1428. Owned by row `MODEL-DIFFUSION-LTX25`; spec [`ltx-2-5.md`](../../specs/ltx-2-5.md)
Row: MODEL-DIFFUSION-LTX25
State: UNKNOWN
Kind: gap
GitHub: 1433
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:491`

### Frozen archive evidence

> | [#1433](https://github.com/mudler/vllm.cpp/issues/1433) | `MODEL-DIFFUSION-LTX25` | `Lightricks/LTX-2` is the repository the WHOLE LTX-2.5 lane mirrors — ten `.agents/specs/ltx25-*.md` files and `.agents/porting-inventory.md` carry the revision string `fd4ded7f`, and every `file:line` upstream anchor in that lane resolves there — and it has NO registry row in `AGENTS.md`, NO `.agents/oracles/<id>.md` file, and NO recorded pin. Measured at `2da236832`: `scripts/check-oracle-pins.py` reports `oracle-pins ok (9 oracles pinned)` — `diffusers`, `llama-cpp`, `llama-cpp-unsloth`, `sglang`, `sglang-omni`, `transformers`, `tt-forge`, `vllm`, `vllm-omni` — and it reads `.agents/oracles/*.md` with a GLOB, so an oracle with no file is INVISIBLE to it. The green line is what an unpinned upstream looks like from inside the gate, which is the failure `AGENTS.md` "Pin every oracle" names: "An unpinned upstream is a moving target, not an oracle. Its measurements are not reproducible." `vllm-omni` does NOT already cover it: `.agents/oracles/vllm-omni.md:5` names LTX-2.5 in scope and is the right home for the MIRRORING relationship (`vllm-project/vllm-omni` is the structure this project mirrors), but it is not the repository the specs READ — that is a THIRD repository — and `vllm-omni` is itself `pin = UNPINNED` under [#633](https://github.com/mudler/vllm.cpp/issues/633), so the lane's actual reference source is unpinned twice over, once knowingly and once with no record at all. PRE-EXISTING and NOT caused by [#1428](https://github.com/mudler/vllm.cpp/pull/1428): the `fd4ded7f` anchoring dates to the original port `cefacd2d0` (2026-08-13, #435 / #641), first carried by `f193eaf75`, `742e38acf` and `0541cbeaa`. NOT FIXED IN FLOW because adding a row to the `AGENTS.md` oracle table is a POLICY edit, which `AGENTS.md` "Changing the rules or a checker" routes to its own row, spec and fresh review. Found while repairing the fresh review of #1428. Owned by row `MODEL-DIFFUSION-LTX25`; spec [`ltx-2-5.md`](../../specs/ltx-2-5.md) | gap |

## Resolution

-
