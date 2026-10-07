ID: ISSUE-LOCAL-01M3JPQHH0EBZHCWK50NYPCQW0
Title: docs/BENCHMARKS.md carries a frozen second copy of nine benchmark pages that has diverged from the detail files, and nothing gates it
Row: ENG-PUBLIC-DOC-PROJECTIONS
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-27
Updated: 2026-09-27
Closed: -

## Problem

docs/BENCHMARKS.md holds a second, frozen copy of nine benchmark detail pages, and the two copies have diverged. Nothing gates that, because check-benchmark-index only reads the index table and never looks at the inline sections.

1db7e59cf ("docs: retire shared status and split benchmark details", #1714) was supposed to turn docs/BENCHMARKS.md into "one compact index" and move its ten logical sections into docs/benchmarks/. The checker and the detail files landed. The inline sections were never removed, and they have since stopped receiving updates, so the public page now publishes a condensed older fork of content that also exists, in a fuller and newer form, in the detail files.

The inline sections are not copies. Measured at 1de097c46, first column of every table row, comparing the inline section against its detail file:

| page | inline rows | detail rows | only in inline |
|---|---|---|---|
| vllm-online-serving | 59 | 96 | 17 |
| open-gaps | 38 | 75 | 2 |
| llama-cpp-cpu | 11 | 19 | 1 |
| memory | 5 | 10 | 0 |
| mlx-lm-apple-m4 | 4 | 4 | 0 |
| dwarfstar-gguf | 4 | 4 | 0 |
| speculative-decoding | 6 | 10 | 0 |
| how-we-measure | 0 | 0 | 0 |
| reproduce | 7 | 12 | 0 |

The detail side is richer in every case and newer in every case: last commit 2026-09-06 for vllm-online-serving and speculative-decoding, 2026-09-05 for at-a-glance, how-we-measure and reproduce, 2026-08-31 for open-gaps, 2026-08-30 for memory, 2026-08-28 for llama-cpp-cpu, 2026-08-22 for mlx-lm-apple-m4 and dwarfstar-gguf. The two most recent commits touching docs/benchmarks/open-gaps.md are record commits (511115a1b, b426de5ac) while the most recent commit touching docs/BENCHMARKS.md is 82c81b376, a CPU docs change. Contributors write to the detail files; the page froze.

The concrete loss risk, which is why the inline sections are NOT removed by the change that adds the index: open-gaps carries two rows that exist ONLY inline, "SGLang floor arms" and "Vulkan vs llama.cpp Vulkan (BENCH-VK-LLAMA)", and vllm-online-serving carries 17 first-column entries absent from its detail file, several of which are table sub-headers ("**Ratio POST-LEVER (BINDING, main @`348c265d`)**", "**vllm.cpp** tok/s", "**vllm.cpp** tok/s (canonical 2026-08-10)") and need reading individually to say whether each is a real row or a label. Deleting the inline sections on the strength of "the detail files are newer" would drop the only copy of anything that is genuinely only there.

Two decisions are owed and neither is taken here:

1. Which side is authoritative. The evidence says the detail files, but "newer and richer" is not the same as "supersedes", and a fork gets richer by accumulating.
2. What happens to the rows that exist only inline. They are either content the detail files lost, or rows the detail files replaced, and the two answers need different actions -- port them across, or record that they were retired.

Once those are answered, the consolidation is mechanical: delete the nine inline sections, keep the index table, and the page becomes the entry point 1db7e59cf intended.

## Resolution

-

## Executed verification, 2026-10-02 (Linux x86_64)

Requested by the mudler-agent review (focused checker + mutation).

- `python3 scripts/check-benchmark-index.py`: `benchmark index OK: every ID
  owns exactly one detail file` (was 16 orphan errors on main).
- `python3 tests/scripts/test_check_benchmark_index.py`: 6/6 OK --
  `test_shipped_index_passes` green for the first time since the checker
  landed.
- Mutation: deleting the `at-a-glance` index row makes the gate rc=1 with
  `benchmark index error: orphan benchmark detail file at-a-glance.md`;
  restoring the row returns rc=0. The table is load-bearing.

