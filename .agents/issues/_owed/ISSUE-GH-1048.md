ID: ISSUE-GH-1048
Title: LTX-2.5 ships with NO checkpoint pin. `docs/USAGE.md` names six LTX-2.5 artifacts by bare file name and gives no HuggingFace repo, no revision and no sha256 for any of them: `:663-670` and `:2183-2188` on `origin/main` at `d1b0ea3a8`, plus the text-to-audio recipe [#1005](https://github.com/mudler/vllm.cpp/issues/1005) added at `:853-857`. AGENTS.md § *Say which weights, and from where* requires file name, size, exact repo AND revision, grouped by arm, with a sha256 for a quantized artifact. Campaign-wide and PRE-EXISTING rather than introduced by that row, measured rather than asserted: `grep -n sha256 docs/USAGE.md` returns two checkpoint hashes and BOTH belong to MiniMax-Music3 (`:3127`, `:3269`), while MiniMax-H3 (`:1950-1993`) and MiniMax-Music3 (`:3123-3149`) each carry a full repo + revision + sha256 table and LTX-2.5 carries none anywhere. RECORDED AND DELIBERATELY NOT FABRICATED: no LTX-2.5 row in the campaign claims a render on real weights — `dgx.casa` is down and every LTX-2.5 gate runs on a reduced fixture — so there is no checkpoint any of them was gated against to pin, and inventing a repo id would be worse than the gap. The real-checkpoint render owed by [#644](https://github.com/mudler/vllm.cpp/issues/644) is what closes it. Two smaller things fall out of the same gap and belong to the same change: the t2a recipe names `ltx-2.5-dit.safetensors`, a file name appearing nowhere else in the tree, where every other LTX-2.5 recipe names `ltx-2.5-22b-distilled-fp8.safetensors` plus the `--dit-config` its missing `__metadata__` requires; and whether that recipe runs at all without `--dit-config` is unverified. Found repairing the fresh review of [#1039](https://github.com/mudler/vllm.cpp/issues/1039) on PR [#1032](https://github.com/mudler/vllm.cpp/pull/1032). Listed under `## Owed` in [`ltx25-t2a-one-stage.md`](../../specs/ltx25-t2a-one-stage.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1048
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:304`

### Frozen archive evidence

> | [#1048](https://github.com/mudler/vllm.cpp/issues/1048) | — | LTX-2.5 ships with NO checkpoint pin. `docs/USAGE.md` names six LTX-2.5 artifacts by bare file name and gives no HuggingFace repo, no revision and no sha256 for any of them: `:663-670` and `:2183-2188` on `origin/main` at `d1b0ea3a8`, plus the text-to-audio recipe [#1005](https://github.com/mudler/vllm.cpp/issues/1005) added at `:853-857`. AGENTS.md § *Say which weights, and from where* requires file name, size, exact repo AND revision, grouped by arm, with a sha256 for a quantized artifact. Campaign-wide and PRE-EXISTING rather than introduced by that row, measured rather than asserted: `grep -n sha256 docs/USAGE.md` returns two checkpoint hashes and BOTH belong to MiniMax-Music3 (`:3127`, `:3269`), while MiniMax-H3 (`:1950-1993`) and MiniMax-Music3 (`:3123-3149`) each carry a full repo + revision + sha256 table and LTX-2.5 carries none anywhere. RECORDED AND DELIBERATELY NOT FABRICATED: no LTX-2.5 row in the campaign claims a render on real weights — `dgx.casa` is down and every LTX-2.5 gate runs on a reduced fixture — so there is no checkpoint any of them was gated against to pin, and inventing a repo id would be worse than the gap. The real-checkpoint render owed by [#644](https://github.com/mudler/vllm.cpp/issues/644) is what closes it. Two smaller things fall out of the same gap and belong to the same change: the t2a recipe names `ltx-2.5-dit.safetensors`, a file name appearing nowhere else in the tree, where every other LTX-2.5 recipe names `ltx-2.5-22b-distilled-fp8.safetensors` plus the `--dit-config` its missing `__metadata__` requires; and whether that recipe runs at all without `--dit-config` is unverified. Found repairing the fresh review of [#1039](https://github.com/mudler/vllm.cpp/issues/1039) on PR [#1032](https://github.com/mudler/vllm.cpp/pull/1032). Listed under `## Owed` in [`ltx25-t2a-one-stage.md`](../../specs/ltx25-t2a-one-stage.md) | bug |

## Resolution

-
