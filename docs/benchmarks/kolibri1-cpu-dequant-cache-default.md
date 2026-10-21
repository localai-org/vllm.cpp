# Kolibri-1 on CPU: the dequant-cache default

The Kolibri-1 CPU decode, prefill and n-gram spec-decode numbers for the
configuration the tree now ships: the **FP8 dequant cache ON by default**.
PR #3488 has landed (`d9e669582` the flip, `e451be705` the sampling-moment
repair and the knee floor, `3ce14f35c` the knee refusal), so the numbers
below are the shipped configuration's numbers. Measured 2026-10-20/21 on
node 3 (aarch64, 128 cores, 255 GB RAM, governor `performance` at 3.4 GHz),
32 threads, one NUMA node, the file mutex held, the box quiet.

This is a **host-specific** record, not a parity claim: vLLM does not serve
this model on this path, and the author's plugin is still unverified as a
runnable oracle. The correctness record is the row's token gate, not this
page.

## Results

Three harnesses carry the numbers, and they measure three different shapes.
Each table below names the harness, the shape, and the session log its cells
came from. A cell never moves between shapes, and every ratio below is
derived from the cells printed next to it.

### 1. Decode, the in-repo production-shape bench

`tests/vllm/models/test_kolibri1_decode_bench.cpp`: a 128-token prompt and
64 greedy tokens out, 32 threads. The cache leg is set by
`VT_KOLIBRI1_DEQUANT_CACHE_MB`: `0` disables it, `16384` is the explicit
16 GiB, and an unset value is the derived default. Three runs per leg,
except the sub-knee leg, which has one.

| leg | runs (tok/s) | median | vs cache off |
|---|---|---:|---:|
| **cache OFF** — what `main` shipped before the default landed | 3.55 / 3.62 / 3.79 | **3.62** | 1.00x |
| default, first implementation (8.25 GiB budget, 33% hit rate) | 3.84 | 3.84 | 1.06x |
| **default, repaired** (26.5 GiB budget, 80% hit rate) | 7.39 / 7.45 / 7.47 | **7.45** | **2.06x** |
| explicit 16 GiB budget | 6.99 / 7.05 / 7.06 | 7.05 | 1.95x |

The `vs cache off` column is the ratio to the cache-off median, 3.62:
7.45/3.62 = 2.06, 7.05/3.62 = 1.95, 3.84/3.62 = 1.06.

Every cell in this table is one measurement set: the session's
`bench_repair.log` (2026-10-21 02:03-02:09, `/tmp/build-dq`), the binary
rebuilt from the committed head `d47173c51` of PR #3488 — the head the
measurement was taken on — one hold, three invocations per leg. That PR has
since landed on `main` as `d9e669582` (the flip), `e451be705` (the
sampling-moment repair and the knee floor) and `3ce14f35c` (the knee
refusal), so the default this table measures is the configuration `main`
now ships. The sub-knee leg is the single run its own policy commit quotes
(`prof-default2.log`, 2026-10-21 01:45: `budget_mib=8454`, `hits=9097`,
`misses=18349`, `evictions=16068`).

### 2. Prefill and thread scaling, the session driver

`/tmp/kolibri-scale/bench-run`, the session's copy of the same bench with
two knobs (`VT_KB_PROMPT`, `VT_KB_GEN`): an empty prompt and 32 generated
tokens for the decode legs, a 512-token prompt and 8 generated tokens for
the prefill legs. The cache is off.

| threads | 8 | 16 | 32 |
|---|---:|---:|---:|
| prefill tok/s (p512, 2 runs per leg) | 12.27-12.30 | 23.78-23.89 | **43.02-43.33** |

| threads | 8 | 16 | 32 | 8→16 | 16→32 | 8→32 |
|---|---:|---:|---:|---:|---:|---:|
| decode tok/s (empty prompt, 32 generated, 3 runs) | 1.38 | 2.59 | 4.02 | 1.88x | 1.55x | 2.91x |

The ratios are this row's own: 2.59/1.38 = 1.88, 4.02/2.59 = 1.55, and
4.02/1.38 = 2.91. The knee is between 16 and 32 threads: the first doubling
scales almost linearly, the second returns 1.55x, not 2x. The cells come
from one set, the session's `final_sweep.log` (2026-10-20 23:09-23:28,
`/tmp/kolibri-scale`), one sweep, cache off. The t32 cell's three runs are
3.95 / 4.02 / 4.16, and the spin-budget arm of the same cell reads 3.96 /
4.19 / 4.43 with `VT_CPU_SPIN_ROUNDS=0`.

### 3. The n-gram spec-decode probe, the campaign's serving number

`tests/vllm/models/test_kolibri1_spec_probe.cpp`, the in-repo probe: the
plain leg against the n-gram spec-decode leg, 32 threads, three runs per
leg.

| leg | plain | spec decode |
|---|---|---|
| cache OFF | 3.90-3.92 | 6.27-6.30 |
| **default (repaired)** | 8.63-9.36 | **12.74-13.14** |
| explicit 16 GiB | 8.33-8.73 | 10.42-10.55 |

The cells come from one set, the session's `spec_repair.log` (2026-10-21
04:41-05:21, `/tmp/build-dq`), the same rebuilt binary, one hold, three
runs per leg.

**The campaign's 12 tok/s target is MET** by the shipped default: the spec
decode leg reads 12.74-13.14 tok/s. It is **not** met with the cache off
(6.27-6.30). The default's spec leg is **2.03x** the cache-off leg's
(12.74/6.27 = 2.03). A second, independent reading clears the target at the
engine level rather than the harness level: **4.53 -> 13.78 tok/s**, or
**3.04x** (2026-10-12, t32 with a 32 GiB cache; the row spec's `## Now`).
The caveats are in [Limitations](#limitations) and they travel with the
number.

## The weights

| Field | Value |
|---|---|
| Local path | `/mnt/models/Aleph-Alpha/Kolibri-1` |
| Upstream repo | [`Aleph-Alpha/Kolibri-1`](https://huggingface.co/Aleph-Alpha/Kolibri-1) |
| Revision measured | `5bb7075ba22db733e3834b1c71d2245537cc14ed` |
| Quantization | FP8 block-quantized, `weight_block_size [128, 128]`, dynamic activation scheme (the `mlp.gate` routers are bf16) |
| Size | 32 safetensors shards, 78,827,029,120 bytes of tensors (74 GiB on disk) |
| Checksums | **No shard SHA-256 is recorded in this tree.** The manifest carries names, dtypes and shapes only, and the model recipe says so: "A complete artifact checksum record remains unavailable" ([`docs/models/kolibri-1.md`](../models/kolibri-1.md)) |
| **Refused arm** | [`Aleph-Alpha/Kolibri-1-BF16`](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16) @ `cc6d729e6a024c1e59ad48887fdef95ff37b534d` (149 GiB) is **not an implemented arm** — the loader refuses a checkpoint without the FP8 `quantization_config`: "a bf16 Kolibri-1 artifact is not an implemented arm" (`src/vllm/model_executor/models/kolibri1_weights.cpp:72`). It is named here as refused; **no bf16 measurement exists** |

The public recipe still records the short revision `e52eb462` for this
checkpoint ([`docs/models/kolibri-1.md`](../models/kolibri-1.md),
[`docs/USAGE.md`](../USAGE.md)); the full revision of the artifact these
numbers were measured on is the one in the table above.

## Reproduce

The decode legs run the in-repo bench under the node pinning and the mutex.
`VT_KOLIBRI1_PROFILE=1` prints the cache counters that name each leg's
budget and hit rate:

```sh
export TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal-pin
export TT_METAL_RUNTIME_ROOT=/home/lu_zero/Sources/tt/tt-metal-pin
BIN=tests/test_kolibri1_decode_bench
BR=/tmp/kolibri-scale/bench-run

# decode, 32 threads: the cache-off leg
numactl --cpunodebind=3 --interleave=all env VT_KOLIBRI1_PROFILE=1 \
  VLLM_CPP_CPU_THREADS=32 VT_KOLIBRI1_DEQUANT_CACHE_MB=0 "$BIN"

# decode, 32 threads: the derived default (the variable unset)
numactl --cpunodebind=3 --interleave=all env VT_KOLIBRI1_PROFILE=1 \
  VLLM_CPP_CPU_THREADS=32 "$BIN"

# decode, 32 threads: the explicit 16 GiB budget
numactl --cpunodebind=3 --interleave=all env VT_KOLIBRI1_PROFILE=1 \
  VLLM_CPP_CPU_THREADS=32 VT_KOLIBRI1_DEQUANT_CACHE_MB=16384 "$BIN"

# prefill p512, 8 generated tokens, 32 threads, cache off
numactl --cpunodebind=3 --interleave=all env VT_KB_PROMPT=512 \
  VT_KB_GEN=8 VLLM_CPP_CPU_THREADS=32 \
  VT_KOLIBRI1_DEQUANT_CACHE_MB=0 "$BR"

# decode, empty prompt, 32 generated tokens, 32 threads, cache off
numactl --cpunodebind=3 --interleave=all env VT_KB_PROMPT=0 \
  VT_KB_GEN=32 VLLM_CPP_CPU_THREADS=32 \
  VT_KOLIBRI1_DEQUANT_CACHE_MB=0 "$BR"

# the serving number: the in-repo n-gram spec probe
VLLM_CPP_CPU_THREADS=32 tests/test_kolibri1_spec_probe
```

The mutex is held for the whole series:

```sh
flock -x "$HOME/gpu.lock" bash -c '...the legs above...'
```

The prefill and thread-scaling legs are driven by a **session-local bench
driver** (`$BR`, the two knobs `VT_KB_PROMPT` and `VT_KB_GEN`), rebuilt
from the branch head under test. It is **not** an in-repo harness and it is
not reproducible from a clean checkout; the in-repo pair — the
production-shape bench and the spec-decode probe — is, and the spec-decode
probe carries the serving number this page publishes.

## Limitations

- **The noise floor on this box is ~4%.** Two runs of the same cell (same
  binary, weights and configuration) differed by 3.5% in the t32 cache-off
  leg (4.02 against 4.16) and by 4.7% in the decode table's cache-off leg
  (3.62 against 3.79). **No cell on this page is quoted more finely than
  that**, and two legs that differ by less than 4% are a tie, not a win.
- **The default's margin over an explicit 16 GiB budget is 5.7%, and it is
  not why the default ships.** In the one set this page publishes, the
  explicit leg reads 6.99-7.06 tok/s (median 7.05) against 7.39-7.47 for
  the derived default (median 7.45): 7.45/7.05 = 1.057. That is above the
  4% noise floor, but it is a small margin for 10.5 GiB more cache. The
  default ships because the policy derives the budget from the host's own
  memory report; a reader who pins 16 GiB by hand gets a defensible number
  and takes that report out of a decision the report makes better. What
  this set agrees on is the **~2x of the default over the cache-off
  configuration** (7.45 against 3.62, 2.06x).
- **The 12 tok/s margin is 6.2% over the target, about 1.5 noise floors**
  (12.74 against 12). It clears the target as measured, and the margin is
  not wide.
- **The numbers are host-specific**: 32 threads, one NUMA node, 3.4 GHz,
  aarch64, `governor=performance`, on node 3 of a shared box. They are not a
  portable tok/s figure.
- **The default is on `main` now, and the numbers are its measured
  behaviour.** PR #3488 landed as `d9e669582`, `e451be705` and `3ce14f35c`,
  so the configuration `main` ships is the cache-ON one this page measures.
  The cache-off leg is the baseline `main` shipped before the default
  landed, and the same probe reads 6.27-6.30 tok/s with it off.
- **A sub-knee budget buys nothing**, and the measurement says so: the
  default's first implementation (8.25 GiB, 33% hit rate, 16068 evictions)
  read 3.84 tok/s. That is +6.1% over the cache-off median (3.84/3.62 =
  1.061) and +1.3% over the cache-off leg's best run (3.84/3.79 = 1.013),
  so under the rule above it is **not a tie** to the cache-off median. It is
  not a win either: the leg pays 8.25 GiB of memory for a 33% hit rate and
  16068 evictions, and it leaves the 2x of the repaired default (7.45 tok/s
  at an 80% hit rate) on the table. The policy now refuses a sub-knee budget
  outright.
- **The spin budget is not adopted on this evidence**: `VT_CPU_SPIN_ROUNDS=0`
  moved the same cell from 4.019 to 4.193 tok/s at 32 threads (+4.3%, at the
  noise floor) and from 2.587 to 2.581 at 16 threads (-0.2%, the other way).
  The pair does not separate the arms, so no spin setting is recommended.
  One paired reading on one host is not a settled lever question.
- **The cache-off baseline is a band, not a point.** The cache-off leg reads
  3.55-3.79 tok/s across its three runs, and the default's 7.45 median is
  1.97x-2.10x that band (7.45/3.79 and 7.45/3.55).
- **No bf16 arm was measured**, and none can be: the loader refuses it.
- **No vLLM or plugin denominator.** These are absolute tok/s on one host.

## Evidence

The full record — host state, the commands, the box-quiet window, the noise
floor, the per-cell readings with their source log, the other measurement
sets the session took for the same cells, and the arithmetic behind every
derived number — is
[`docs/bench-evidence/kolibri1-cpu-dequant-cache-default-20261021/report.md`](../bench-evidence/kolibri1-cpu-dequant-cache-default-20261021/report.md).
The policy the default is measured against is PR #3488
(`84d98a9e6` the flip, `d47173c51` the sampling-moment repair and the knee
floor, `e7d7704d3` the knee refusal), which landed on `main` as
`d9e669582`, `e451be705` and `3ce14f35c`; the row's own record is
[`.agents/specs/kolibri-1-cpu.md`](../../.agents/specs/kolibri-1-cpu.md).
