ID: ISSUE-GH-1308
Title: ENV-AGNOSTIC: the campaign's derivation query `git grep -cIE 'dgx\.casa|nas_share|192\.168\.|thor:gpu0'` names a HOST, and a hard-coded default names a PATH, so the query cannot see the defect [#1190](https://github.com/mudler/vllm.cpp/issues/1190) was opened to remove. Measured at `5c8671c50` over `scripts/` and `tools/`, which is `ENV-AGNOSTIC-W1-TOOLING`'s ownership: the four campaign patterns match 19 files and 22 hits, and widening to `/home/mudler`, `~/venvs/vllm-oracle`, `~/work/vllm.cpp`, `cutlass-4.5.0` and `cutlass_probe` matches 73 files and 174 hits. Twelve of those lines in six shell scripts are the exact `${KEY:-<one operator's path>}` shape the campaign's worked example removed from `scripts/dgx-bringup.sh`, and `scripts/upstream-inventory.py:38-40` carries the Python form `os.environ.get("VLLM_SOURCE", str(Path.home() / "_git/vllm"))`. `ENV-AGNOSTIC-W1-TOOLING` converts `scripts/regen-triton-aot.sh`, because the campaign row wrote that one debt into the file's own text at `:20-23` rather than into a count, and leaves the rest: two of them read as provenance on a first pass and must not be swept blind, since `scripts/cpu-x86-llamacpp-floor.sh:33` states that "every recorded leg used these defaults verbatim" and `scripts/dgx-gdn-packed-bridge-ab.sh:4` dates its prerequisites, which is the same instrument-identity argument that keeps `scripts/mtp-k-gt-1-neartie-gap.py` literal. The blind spot is campaign-level: `ENV-AGNOSTIC-W3-CODE` and `ENV-AGNOSTIC-W4-RECORDS` derive their sets from the same query, so the campaign's claim that its five waves partition 227 files is a claim about the query's 227 and not about the tree. Owed by [`env-agnostic-w1-tooling.md`](../../specs/env-agnostic-w1-tooling.md) under `## Owed`
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1308
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:432`

### Frozen archive evidence

> | [#1308](https://github.com/mudler/vllm.cpp/issues/1308) | — | ENV-AGNOSTIC: the campaign's derivation query `git grep -cIE 'dgx\.casa\|nas_share\|192\.168\.\|thor:gpu0'` names a HOST, and a hard-coded default names a PATH, so the query cannot see the defect [#1190](https://github.com/mudler/vllm.cpp/issues/1190) was opened to remove. Measured at `5c8671c50` over `scripts/` and `tools/`, which is `ENV-AGNOSTIC-W1-TOOLING`'s ownership: the four campaign patterns match 19 files and 22 hits, and widening to `/home/mudler`, `~/venvs/vllm-oracle`, `~/work/vllm.cpp`, `cutlass-4.5.0` and `cutlass_probe` matches 73 files and 174 hits. Twelve of those lines in six shell scripts are the exact `${KEY:-<one operator's path>}` shape the campaign's worked example removed from `scripts/dgx-bringup.sh`, and `scripts/upstream-inventory.py:38-40` carries the Python form `os.environ.get("VLLM_SOURCE", str(Path.home() / "_git/vllm"))`. `ENV-AGNOSTIC-W1-TOOLING` converts `scripts/regen-triton-aot.sh`, because the campaign row wrote that one debt into the file's own text at `:20-23` rather than into a count, and leaves the rest: two of them read as provenance on a first pass and must not be swept blind, since `scripts/cpu-x86-llamacpp-floor.sh:33` states that "every recorded leg used these defaults verbatim" and `scripts/dgx-gdn-packed-bridge-ab.sh:4` dates its prerequisites, which is the same instrument-identity argument that keeps `scripts/mtp-k-gt-1-neartie-gap.py` literal. The blind spot is campaign-level: `ENV-AGNOSTIC-W3-CODE` and `ENV-AGNOSTIC-W4-RECORDS` derive their sets from the same query, so the campaign's claim that its five waves partition 227 files is a claim about the query's 227 and not about the tree. Owed by [`env-agnostic-w1-tooling.md`](../../specs/env-agnostic-w1-tooling.md) under `## Owed` | bug |

## Resolution

-
