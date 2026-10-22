# Kolibri-1 CPU: the dequant-cache default measured on node 3 — decode,
# prefill and the n-gram spec-decode serving number (2026-10-20/21)

Row: `MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm`
Issue: `ISSUE-LOCAL-01M5E7ZJW0YXR2QZXQESJEP0MX` (kind gap — the public
performance record the row did not have).
Branch: `row/KOLIBRI1-CPU-BENCH-RECORDS` (measured on the base `origin/main`
`891c4f27d`; the record was rebased onto `3ce14f35c`, which carries the
landed default).
Date: 2026-10-20/21.
No product code changed for this record.

## What this is

The public performance record for the Kolibri-1 CPU path, for the
configuration the tree now ships: the CPU **dequant cache ON by default**.
That default is PR #3488
(`row/KOLIBRI1-DEQUANT-CACHE-DEFAULT`, head `e7d7704d3`, commits `84d98a9e6`
→ `d47173c51` → `e7d7704d3`), which was **OPEN at the time of writing** and
has since **LANDED** on `main` as `d9e669582` (the flip), `e451be705` (the
sampling-moment repair and the knee floor) and `3ce14f35c` (the knee
refusal). Every "default" number below is that PR's configuration measured
on the box that runs this row's gates, and it is the configuration `main`
ships today. The cache-off leg is the baseline `main` shipped before the
default landed, and the first table says what that costs.

The record answers three questions the tree could not answer before:

1. What does the shipped default actually deliver on CPU decode, prefill and
   the n-gram spec-decode serving path?
2. Is the campaign's **12 tok/s** target met, and on which configuration?
3. Which budget and tuning choices were measured and **rejected**?

## Host and window

- aarch64, 128 cores, 255 GB RAM, **node 3**, governor `performance` at
  **3.4 GHz**.
- 32 worker threads (`VLLM_CPP_CPU_THREADS=32`), the process pinned with
  `numactl --cpunodebind=3 --interleave=all`.
- The file mutex `$HOME/gpu.lock` was held for the window and the box was
  quiet: no other model was resident, and the legs of one table were run
  back to back inside one hold.
- One binary per window, rebuilt from the committed head of the branch under
  test. A leg is one command; the runs of a leg are invocations of it.

## The shapes, and which harness measured each

Three harnesses measured three shapes. The shapes are not interchangeable,
so no cell moves between them.

| harness | shape | used for |
|---|---|---|
| `tests/vllm/models/test_kolibri1_decode_bench.cpp` (in repo) | 128-token prompt, 64 greedy tokens out | the decode table (§1) |
| `/tmp/kolibri-scale/bench-run` (session-local copy of the same bench, with `VT_KB_PROMPT` and `VT_KB_GEN`) | empty prompt, 32 generated tokens; and a 512-token prompt, 8 generated tokens | the thread-scaling and prefill legs (§2, §3) |
| `tests/vllm/models/test_kolibri1_spec_probe.cpp` (in repo) | the plain leg against the n-gram spec-decode leg | the serving number (§4) |

The session driver is a **session artifact, not an in-repo harness**: it is
rebuilt from the branch head, and it is not reproducible from a clean
checkout. The in-repo production-shape bench and the in-repo spec probe are
the reproducible pair, and the spec probe carries the campaign's serving
number.

## The measurement sets

Every leg in this record comes from ONE set, named here with its file and
its date. The session took other sets for the same cells; they are listed in
§7 and are not mixed into the tables.

| leg | set | file | when | binary |
|---|---|---|---|---|
| cache OFF, decode | S1 | `bench_repair.log` (`/tmp/build-dq`) | 2026-10-21 02:07-02:09 | rebuilt from `d47173c51` |
| default (repaired), decode | S1 | `bench_repair.log` | 2026-10-21 02:03-02:05 | rebuilt from `d47173c51` |
| explicit 16 GiB, decode | S1 | `bench_repair.log` | 2026-10-21 02:05-02:07 | rebuilt from `d47173c51` |
| default, first implementation (sub-knee), decode | S2 | `prof-default2.log` (`/tmp/build-dq`) | 2026-10-21 01:45 | the pre-repair binary |
| prefill p512, thread scaling, spin pair | S3 | `final_sweep.log` (`/tmp/kolibri-scale`) | 2026-10-20 23:09-23:28 | the session driver |
| spec-decode probe | S4 | `spec_repair.log` (`/tmp/build-dq`) | 2026-10-21 04:41-05:21 | rebuilt from `d47173c51` |

S1 is the set the tree's own commit `d47173c51` quotes, and S1 is the set
the public page publishes. S2 is the single run that commit quotes for the
sub-knee leg, with the cache counters that name the leg's budget and hit
rate. S3 and S4 are the session's own logs.

**The commit bodies are a transcription check, not an independent
measurement.** `d47173c51` quotes S1's numbers, so its body confirms that
the transcription into this record is right; it does not confirm the
measurement a second time. The public page therefore states the comparison
between the sub-knee leg and the cache-off leg from the cells, and not from
the commit body.

## The noise floor

Two runs of the same cell — same binary, same weights, same configuration,
same host state — differed by 3.5% in the t32 cache-off leg (4.02 against
4.16) and by 4.7% in the decode table's cache-off leg (3.62 against 3.79).
That is the floor this record is read at:

- **No cell below is quoted more finely than 4%.** Two legs that differ by
  less than 4% are a **tie**, not a win.
- A single run is an anecdote. Every leg in the tables below is three runs
  unless the table says otherwise, and a range is reported as a range, never
  as a precise figure.
- The floor is a property of this box and this window, not of the engine.

## The weights

| Field | Value |
|---|---|
| Local path | `/mnt/models/Aleph-Alpha/Kolibri-1` |
| Upstream repo | [`Aleph-Alpha/Kolibri-1`](https://huggingface.co/Aleph-Alpha/Kolibri-1) |
| Revision measured | `5bb7075ba22db733e3834b1c71d2245537cc14ed` |
| Quantization | FP8 block-quantized, `weight_block_size [128, 128]`, dynamic activation scheme; the `mlp.gate` routers are bf16 |
| Size | 32 safetensors shards, 78,827,029,120 bytes of tensors (74 GiB on disk) |
| Checksums | **No shard SHA-256 is recorded in this tree.** The manifest (`tests/vllm/models/kolibri1_manifest.inc`) carries names, dtypes and shapes only, and the public recipe says so: "A complete artifact checksum record remains unavailable" ([`docs/models/kolibri-1.md`](../../models/kolibri-1.md)) |
| Other arm (not measured here) | [`Aleph-Alpha/Kolibri-1-BF16`](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16) @ `cc6d729e6a024c1e59ad48887fdef95ff37b534d` (149 GiB) is the **plain bf16 arm**, and this report does NOT measure it: every number here is the FP8 arm's decode. The arm landed after this measurement ([spec](../../../.agents/specs/kolibri1-bf16-arm.md)), so the loader no longer refuses a checkpoint without an FP8 `quantization_config` — it reads it as the plain bf16 arm. The upstream artifact's own gate is OWED ([evidence](../kolibri1-bf16-arm-20261021.md)) |

## 1. Decode, the in-repo production-shape bench (128 in, 64 out)

Set S1, three runs per leg, tok/s. The sub-knee leg is S2's single run.

| leg | runs | median | note |
|---|---|---|---|
| **cache OFF** (`VT_KOLIBRI1_DEQUANT_CACHE_MB=0`) — the configuration `main` shipped before the default landed | 3.55 / 3.62 / 3.79 | **3.62** | the baseline every other leg is read against |
| default, **first implementation** (8.25 GiB budget, 33% hit rate, 16068 evictions) | 3.84 | 3.84 | **worthless**: +6.1% over the cache-off median and +1.3% over the cache-off leg's best run, for 8.25 GiB of memory and a 33% hit rate |
| default, **repaired** (26.5 GiB budget from a 212 GiB process-start reading; hits 72190-72200, misses 17746-17765, evictions 7967-8027, **80% hit rate**) | 7.39 / 7.45 / 7.47 | **7.45** | the shipped default; **2.06x** the cache-off median |
| **explicit 16 GiB** (`VT_KOLIBRI1_DEQUANT_CACHE_MB=16384`) | 6.99 / 7.05 / 7.06 | 7.05 | 1.95x the cache-off median |

The full-precision cells of S1, in the order the log recorded them:

| leg | run 1 | run 2 | run 3 | counters of run 1 |
|---|---|---|---|---|
| cache OFF | 3.62428 | 3.78799 | 3.54865 | `enabled=0`, no cache traffic |
| default (repaired) | 7.44833 | 7.39141 | 7.46827 | `budget_mib=27157`, `budget_start_mib=217262`, hits 72190, misses 17756, evictions 7993 |
| explicit 16 GiB | 7.05777 | 7.05438 | 6.98659 | `budget_mib=16384`, hits 71097, misses 18849, evictions 13396 |

S2's single run reads `budget_mib=8454` from `budget_source_mib=67638`,
hits 9097, misses 18349, evictions 16068, and 3.83784 tok/s. The hit rate is
9097/(9097+18349) = 33%, against 80% for the repaired default
(72190/(72190+17756) = 80%).

**Read the explicit-16-GiB leg with the margin in view.** In this set it
reads 6.99-7.06 tok/s against 7.39-7.47 for the derived default: 7.45/7.05 =
1.057, so +5.7%. That is above the 4% noise floor and a small margin for
10.5 GiB more cache, so the record does not claim the default's margin over an
explicit budget as a reason to prefer one over the other. What it does claim
is the ~2x of the default over the cache-off configuration (7.45 against
3.62, 2.06x). Across the session's cache-off readings that ratio runs
1.97x-3.04x (§7), so ~2x is its conservative end.

## 2. Prefill, `p512` (512-token prompt), cache OFF

Set S3, two runs per leg, tok/s.

| threads | 8 | 16 | 32 |
|---|---:|---:|---:|
| prefill tok/s | 12.27-12.30 | 23.78-23.89 | **43.02-43.33** |

Prefill is the arm the cache does not help (it is one wide forward, not a
reused decode), and it is quoted with the cache **OFF** on purpose: that is
the configuration `main` shipped before the default landed, and it is the
baseline every cache-on number is read against.

## 3. Thread scaling, decode, cache OFF, empty prompt

Set S3, three runs per leg, tok/s.

| threads | 8 | 16 | 32 | 8→16 | 16→32 | 8→32 |
|---|---:|---:|---:|---:|---:|---:|
| decode tok/s | 1.38 | 2.59 | 4.02 | **1.88x** | **1.55x** | **2.91x** |

The ratios are this row's own: 2.59/1.38 = 1.88, 4.02/2.59 = 1.55,
4.02/1.38 = 2.91. The knee is between 16 and 32 threads: the first doubling
scales almost linearly, the second returns 1.55x, not 2x. 32 threads on one
NUMA node is the configuration every other number in this record is taken
at, and it is the configuration the 12 tok/s target is claimed at.

The full-precision medians of S3 are 1.382377 (t8), 2.587428 (t16) and
4.019148 (t32), from the runs 1.406050 / 1.382377 / 1.379258,
2.587428 / 2.569467 / 2.590473 and 4.019148 / 3.949273 / 4.158811.

## 4. The spin budget is not a lever here

Set S3, the same decode cell, median of three runs per arm.

| threads | default | `VT_CPU_SPIN_ROUNDS=0` | read |
|---|---:|---:|---|
| 32 | 4.019 | 4.193 | +4.3%, at the noise floor |
| 16 | 2.587 | 2.581 | -0.2%, the other way |

**Not adopted: spin tuning.** The two paired readings do not agree in sign
(+4.3% at 32 threads, -0.2% at 16), so the pair does not separate the arms
and no spin setting is recommended. One paired reading on one host is not a
settled lever question.

## 5. The spec-decode probe — the campaign's serving number

Set S4, the in-repo n-gram spec probe, plain against spec-decode, tok/s,
three runs per leg.

| leg | plain | spec decode |
|---|---|---|
| cache OFF | 3.90-3.92 | 6.27-6.30 |
| **default (repaired)** | 8.63-9.36 | **12.74-13.14** |
| explicit 16 GiB | 8.33-8.73 | 10.42-10.55 |

The full-precision cells of S4 are 3.92253 / 3.91669 / 3.90194 plain and
6.30373 / 6.27138 / 6.26552 spec for the cache-off leg; 9.36388 / 8.63662 /
8.63118 plain and 13.1397 / 12.9146 / 12.7398 spec for the default; and
8.5415 / 8.73088 / 8.32622 plain and 10.4702 / 10.5469 / 10.4178 spec for
the explicit 16 GiB leg.

## 6. The verdict on the 12 tok/s target

**The campaign's 12 tok/s target is MET, by the shipped default, with the
dequant cache ON.** The default leg's spec decode reads **12.74-13.14
tok/s** against the target's 12. The cache-off leg reads 6.27-6.30, so the
target is **not** met without the cache. The default's spec leg is 2.03x the
cache-off leg's (12.74/6.27).

A second, independent reading clears it at the engine level rather than the
harness level: the row's `## Now` records the engine-level n-gram spec decode
at **4.53 → 13.78 tok/s** on the bench prompt (3.04x, t32 with a 32 GiB
cache, 2026-10-12), and the spec probe's own harness reference at
4.92 → 14.51.

**The caveats travel with the number, and they are load-bearing:**

- The 12 tok/s figure **needs the dequant cache ON**, which is the PR #3488
  default and is now **on `main`** (`d9e669582`, `e451be705`, `3ce14f35c`).
  With the cache off — the configuration `main` shipped before the default
  landed — the same probe reads 6.27-6.30.
- The figure is **host-specific**: 32 threads, one NUMA node, 3.4 GHz,
  aarch64, `governor=performance`, on node 3 of the shared box. It is not a
  portable tok/s figure and it is not a vLLM comparison — vLLM does not
  serve this model on this path, and the author's plugin oracle is still
  unverified as a runnable denominator
  ([`docs/models/kolibri-1.md`](../../models/kolibri-1.md)).
- The box's **noise floor is ~4%**, so 12.74 against a target of 12 is a
  **+6.2% margin, about 1.5 noise floors**. It clears the target as
  measured, and the margin is not wide.

## 7. The other sets the session took, and what they are worth

The session measured the same cells more than once. These sets are recorded
here so the record is complete; they are not mixed into the tables above.

| set | file | when | binary | what it read |
|---|---|---|---|---|
| S5 | `bench_window.log` (`/tmp/build-dq`) | 2026-10-21 01:29-02:03 | the pre-repair binary | cache OFF 3.52 / 3.62 / 3.62; default 3.53 / 3.57 / 3.62; explicit 16 GiB 6.38 / 6.83 / 6.95; the spec probe's first pass (default spec 7.86 / 11.43 / 11.46) |
| S6 | `mymeasure.log` (`/tmp/build-dq`) | 2026-10-21 02:09-02:14 | rebuilt from `d47173c51` | cache OFF 2.45 / 3.49 / 3.58; default 7.36 / 7.50 / 7.58; explicit 16 GiB 7.14 / 7.15 / 7.23 — the set `e7d7704d3` quotes |
| S7 | `final_sweep.log` (`/tmp/kolibri-scale`) | 2026-10-20 23:22-23:23 | the session driver | the explicit 16 GiB leg in the empty-prompt shape: 9.03 / 9.10 tok/s (and 8.86 / 9.03 with the spin budget off) |

S5's default leg is the **pre-repair** default, so it is the sub-knee
configuration measured three times: 3.53 / 3.57 / 3.62 tok/s, which is the
cache-off level. S6's cache-off leg carries a **2.45 tok/s outlier**, one run
below its own other two (3.49, 3.58).

Across S5, S6 and S1 the cache-off baseline of the decode table's shape runs
2.45-3.79 tok/s, and the default's 7.45 median is 1.97x-3.04x that band
(7.45/3.79 and 7.45/2.45). In the empty-prompt shape the cache-off readings
are 3.95-4.16 tok/s (S3), and the default was not measured in that shape, so
no ratio is quoted there.

S7 shows that the empty-prompt shape is a different shape and not a second
reading of the decode table's cells: the same explicit 16 GiB leg reads
9.03-9.10 tok/s there against 6.99-7.06 in the decode table's shape.

## 8. Every derived number, with its arithmetic

| derived number | arithmetic | cells |
|---|---|---|
| default vs cache off, 2.06x | 7.45/3.62 = 2.058 | §1 medians |
| explicit 16 GiB vs cache off, 1.95x | 7.05/3.62 = 1.947 | §1 medians |
| sub-knee vs the cache-off median, +6.1% | 3.84/3.62 = 1.061 | §1, S2 against S1 |
| sub-knee vs the cache-off best run, +1.3% | 3.84/3.79 = 1.013 | §1, S2 against S1 |
| default vs the explicit 16 GiB leg, +5.7% | 7.45/7.05 = 1.057 | §1 medians |
| default vs the cache-off band, 1.97x-2.10x | 7.45/3.79 = 1.97 and 7.45/3.55 = 2.10 | §1 medians and the leg's own min and max |
| thread scaling 8→16, 1.88x | 2.59/1.38 = 1.877 | §3 |
| thread scaling 16→32, 1.55x | 4.02/2.59 = 1.552 | §3 |
| thread scaling 8→32, 2.91x | 4.02/1.38 = 2.913 | §3 |
| spin budget at 32 threads, +4.3% | 4.193/4.019 = 1.043 | §4 medians |
| spin budget at 16 threads, -0.2% | 2.581/2.587 = 0.998 | §4 medians |
| 12 tok/s margin, +6.2% | 12.74/12 = 1.062 | §5 default spec leg, against the target |
| the default's spec leg over the cache-off leg, 2.03x | 12.74/6.27 = 2.032 | §5 |
| the repaired default's hit rate, 80% | 72190/(72190+17756) = 0.803 | §1 counters |
| the sub-knee default's hit rate, 33% | 9097/(9097+18349) = 0.331 | §1 counters, S2 |
| the engine-level reading, 3.04x | 13.78/4.53 = 3.042 | the row spec's `## Now` |
| the cross-session cache-off band, 1.97x-3.04x | 7.45/3.79 = 1.97 and 7.45/2.45 = 3.04 | §7 |
| the noise floor's t32 example, 3.5% | 4.16/4.02 = 1.035 | §3 |
| the noise floor's decode-table example, 4.7% | 3.79/3.62 = 1.047 | §1 |

## 9. What was rejected, and why

| rejected | measured | why it is rejected |
|---|---|---|
| the **sub-knee budget** (the default's first implementation: 8.25 GiB, 33% hit rate, 16068 evictions) | 3.84 tok/s, +6.1% over the cache-off median and +1.3% over its best run | a budget below the knee costs the memory and buys nothing: the leg holds 8.25 GiB at a 33% hit rate and leaves the 2x of the repaired default on the table. This is why the sampling moment is part of the policy and why the policy now refuses a sub-knee budget outright (`e7d7704d3`) |
| the **explicit 16 GiB budget** as the recommended setting | 6.99-7.06 tok/s in the published set, +5.7% below the derived default | the margin is a small one for 10.5 GiB more cache, and pinning 16 GiB by hand hard-codes a number the host's own memory report should decide |
| **spin tuning** (`VT_CPU_SPIN_ROUNDS`) | t32 4.019 vs 4.193, t16 2.587 vs 2.581 | the two paired readings do not agree in sign, so the pool is insensitive to the spin budget on this box; a wash, not a lever |
| **more threads** past 32 | the 16→32 step returns 1.55x, not 2x | the knee is inside the 32-thread node; the record keeps the node pinned rather than claiming a scaling law |

## Provenance and limits

- The numbers in §1-§5 are the operator's session report for the 2026-10-20/21
  node-3 window, transcribed into this record. The **commands** are the ones
  the tree already carries (PR #3488's body, and the two repair commits
  quoted per table), so a reader can re-run them.
- The per-leg raw logs are session artifacts and are **not retained in this
  tree**. Each set is named above with its file, its date, and the binary it
  ran, and the values of S1 and S6 are quoted verbatim in the tree's own
  commit bodies (`d47173c51` and `e7d7704d3`), so a reader can check the
  transcription against the repository's history.
- The decode-bench driver (`/tmp/kolibri-scale/bench-run`) is a **session
  artifact, not an in-repo harness**: it is rebuilt from the branch head, and
  it is not reproducible from a clean checkout. The in-repo production-shape
  bench and the in-repo spec probe are the reproducible pair, and the spec
  probe carries the campaign's serving number above.
- No number in this record is a vLLM comparison, and none is a parity claim.
  The Kolibri-1 CPU correctness record is the row's token gate, not this
  file.

## References

- PR #3488 `row/KOLIBRI1-DEQUANT-CACHE-DEFAULT`: `84d98a9e6` (the default
  flip), `d47173c51` (the sampling-moment repair and the knee floor),
  `e7d7704d3` (the knee refusal) — the policy this record measures. It
  landed on `main` as `d9e669582`, `e451be705` and `3ce14f35c`.
- `.agents/specs/kolibri-1-cpu.md` — the row's spec, its `## Now` and its
  `## Outcome`.
- `docs/benchmarks/kolibri1-cpu-dequant-cache-default.md` — the public
  benchmark page for this measurement.
- `docs/bench-evidence/kolibri1-spec-probe-20261011.md` — the probe that
  first projected the 12 tok/s target.
- `docs/bench-evidence/kolibri1-bf16dot-adoption-20261019/report.md` — the
  bf16-dot tier this record's runs inherit (default on a FEAT_BF16 host).
