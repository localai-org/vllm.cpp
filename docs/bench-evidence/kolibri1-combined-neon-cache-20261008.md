# Kolibri-1 CPU: the combined NEON-GEMM + dequant-cache production figure
# (MODEL-TEXT-kolibri-1)

Date: 2026-10-08. Host: aarch64, 128 cores, 255 GB RAM, Linux (the same box as
the 2026-10-07 records). Tree: pristine `origin/main` **5d2f5d31a** — the
default order-preserving NEON GEMM tier PLUS the relanded fp8 dequant cache
with the PR #3414 review repairs (identity, lease, portability) — worktree
`/tmp/vllm-kolibri-combined`, branch `row/kolibri-combined`, build
`/tmp/vllm-kolibri-build`, Release with `-DVLLM_CPP_CUDA=OFF
-DVLLM_CPP_TENSTORRENT=OFF -DVLLM_CPP_SERVER=OFF -DVLLM_CPP_BUILD_EXAMPLES=OFF`.
The build of main was clean: **no source changes were needed or made.** All
runs `VLLM_CPP_CPU_THREADS=8`. Real checkpoint
`/mnt/models/Aleph-Alpha/Kolibri-1` (74 GB fp8, 32 shards).

## The question

Both levers are now on main and each was measured alone: the NEON GEMM tier
(2.07× decode tok/s vs the true scalar, default ON,
[kolibri1-linear-gemm-neon-20261007.md](kolibri1-linear-gemm-neon-20261007.md))
and the opt-in dequant cache (`VT_KOLIBRI1_DEQUANT_CACHE_MB`, 2.50× at 16 GiB
on the pre-repair tree,
[kolibri1-dequant-cache-reland-20261007.md](kolibri1-dequant-cache-reland-20261007.md)).
This unit measures the COMBINED figure on the post-everything tree — the
publishable production number. Baseline anchor: cache-off `decode_bench`
1.267-1.275 tok/s, last token **109726**, chain alternating
`101807, 109726, …`.

## Quiet-window discipline

Every W3 and bench leg waited (poll every 20 s) for `free -m` available
> 110000 MB AND zero `test_kolibri1_*` processes, per the task brief and
`.agents/environment.md` (a contended W3 OOM-kills this host). All eight
model-running legs passed the window at ~212-214 GB available with zero kolibri
processes; the unit gates ran after W3, also with no kolibri process live.
Window log: `/tmp/kolibri-runs/quiet-window.log`.

## A/B: the combined table (production-shape decode bench, 8 threads, this tree)

`tests/vllm/models/test_kolibri1_decode_bench`: 128-token prefill, then 63
incremental t=1 greedy steps (64 greedy tokens), one persistent paged KV
topology. Wall excludes the checkpoint load (timed inside); process wall
includes it (~20 s). The 16 GiB legs and the extra 8 GiB leg ran with
`VT_KOLIBRI1_PROFILE=1` to capture the cache counters (a per-stage chrono
accumulator, negligible); the OFF and first two 8 GiB legs ran without it.

| leg | wall s | prefill s | decode s | decode tok/s | vs OFF |
|---|---|---|---|---|---|
| cache OFF #1 | 63.99 | 12.97 | 51.01 | 1.235 | 1.00× |
| cache OFF #2 | 63.94 | 12.91 | 51.03 | 1.235 | 1.00× |
| 8 GiB #1 | 36.85 | 15.52 | 21.33 | 2.954 | 2.39× |
| 8 GiB #2 | 35.68 | 15.44 | 20.24 | 3.113 | 2.52× |
| 8 GiB #3 (profiled) | 36.39 | 15.72 | 20.67 | 3.048 | 2.47× |
| 16 GiB #1 (profiled) | 36.51 | 15.95 | 20.56 | 3.064 | 2.48× |
| 16 GiB #2 (profiled) | 35.92 | 16.02 | 19.90 | 3.165 | 2.56× |

Means: OFF decode 51.02 s / 1.235 tok/s; 8 GiB decode 20.75 s / 3.038 tok/s
(**2.46×**); 16 GiB decode 20.23 s / 3.115 tok/s (**2.52×**). Whole-run wall:
OFF 63.96 s → 8 GiB 36.31 s (1.76×) → 16 GiB 36.22 s (1.77×). Prefill rises
12.9 → 16.0 s (cold-miss decode plus cache fill), as on the pre-repair tree.

Token identity held on every leg: the anchored last token **109726** passed
the `CHECK_EQ` on all seven runs, and the CHAIN line is byte-identical across
all seven (md5 `8de1463acdeca696f84728b2e3f98831` of the chain string), chain
alternating `101807, 109726, …`, first prefill token 101807.

Cache counters (`VT_KOLIBRI1_PROFILE=1`, final report):

| leg | hits | misses | evictions | decode_calls |
|---|---|---|---|---|
| 8 GiB | 69377 | 20497 | 18321 | 89874 |
| 16 GiB #1 | 71157 | 18717 | 13264 | 89874 |
| 16 GiB #2 | 71157 | 18717 | 13264 | 89874 |

The 8 GiB and 16 GiB counters are **byte-identical to the pre-repair tree's
record** (69377/20497/18321 and 71157/18717/13264) — the PR #3414 repairs are
behavior- and perf-neutral on the bench shape, and the two 16 GiB legs are
deterministic in the counters.

## W3 hitting-config gate (16 GiB, the configuration that hits)

`test_kolibri1_w3` with `VT_KOLIBRI1_DEQUANT_CACHE_MB=16384`,
`VT_KOLIBRI1_PROFILE=1`, 8 threads — the row's TOKEN GATE against the real
checkpoint and the golden fingerprints:

- **900/900 assertions PASS** (`[doctest] Status: SUCCESS!`, exit 0).
- `ARGMAX CHAIN: 141/145 compared positions match the golden greedy decode
  (4 flips: 4 near-tie, 0 hard)` — identical to the landed baseline.
- `FINAL-STEP FINGERPRINT: worst topk logit diff 2.18646, worst sum diff
  26763` — identical to the landed baseline.
- Counters: `hits=348418 misses=519935 evictions=514482 decode_calls=868353`
  — byte-identical to the 2026-10-07 reland record.
- Wall 8:32.20 (512.2 s), max RSS 98,323,328 KB (96.0 GiB; the 16 GiB cache on
  top of the same working set). Prior record: 8:43.87 wall, 96.5 GB RSS.

## Unit gates (same build)

| gate | result |
|---|---|
| `test_kolibri1` | 27/27 cases, 186/186 assertions PASS |
| `test_kolibri1_dequant_cache` | 7/7 cases, 52/52 assertions PASS |
| `test_kolibri1_dequant` | 3/3 cases, 10/10 assertions PASS |

## Reading: do the two levers compose multiplicatively or overlap?

**They compose multiplicatively, because they act on disjoint stages of the
decode wall.** The cache-off baseline on this tree (1.235 tok/s) already
carries the default NEON GEMM tier, so the 16 GiB cache multiplies THAT
baseline by 2.52× (decode 51.0 → 20.2 s). The NEON tier itself was recorded at
2.07× decode tok/s over the true scalar (0.547 tok/s, forced ref tier). Stacked,
the production figure is **3.06-3.17 tok/s at 16 GiB**: 2.52× the NEON-on
cache-off baseline, 2.44-2.50× the recorded 1.267-1.275 tok/s anchor band, and
~5.7× the true-scalar-everything decode — matching the product of the two lever
ratios (2.07 × 2.52 ≈ 5.2×) within cross-run noise. The stage profile shows the
disjointness directly. Per forward, cache OFF (NEON doc, 68-forward block) vs
cache ON 16 GiB (this tree, 60-forward block): `dequant_fp8_block` falls
0.62 → 0.14 s (4.3× — the cache serves owned-byte hits instead of re-decoding
fp8 blocks), `moe_glue` (which nests the per-expert dequant) falls 0.375 →
0.25 s, while `linear_gemm` stays flat at ~0.25-0.30 s and `attn_core`/`lm_head`
are unchanged — the cache does not touch the GEMM, and the NEON tier has
already accelerated it. So the cache removes the re-decode work that every
decode step would otherwise pay, and the NEON GEMM then runs over whatever GEMM
work remains: each lever scales a different term of the decode wall, and the
ratios multiply rather than overlap. Had both levers attacked the same stage,
the stacked ratio would fall short of the product; it does not. The combined
16 GiB figure (3.06-3.17 tok/s) also matches the pre-repair tree's cache-on
number (3.185 tok/s) within run noise, confirming the repairs cost nothing.
The publishable production figure on this host, 8 threads, is therefore
**≈3.1 decode tok/s at a 16 GiB dequant-cache budget (2.5× the NEON-only
baseline, ~5.7× the true scalar floor)**; the cache stays opt-in (default 0)
until a production budget policy is chosen.

## Records

- Evidence doc only; no `src/`, `include/`, `tests/` change in this unit.
- Gates: `scripts/check-agent-record.py` rc=0 (ENGINE=179 MODEL=384 QUANT=87
  KERNEL=60 BACKEND=90 ANCHOR-ROT=0); `scripts/agent-preflight.sh` rc=0
  (background, /tmp; no new failures).
- Row: `MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm`, spec
  `.agents/specs/kolibri-1-cpu.md`. Run logs: `/tmp/kolibri-runs/`.
