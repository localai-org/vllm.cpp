# Kolibri-1 on CPU: the dequant-cache default

The Kolibri-1 CPU decode, prefill and n-gram spec-decode numbers for the
configuration the tree ships once PR #3488 lands: the **FP8 dequant cache ON
by default**. Measured 2026-10-20/21 on node 3 (aarch64, 128 cores, 255 GB
RAM, governor `performance` at 3.4 GHz), 32 threads, one NUMA node, the file
mutex held, the box quiet.

This is a **host-specific** record, not a parity claim: vLLM does not serve
this model on this path, and the author's plugin is still unverified as a
runnable oracle. The correctness record is the row's token gate, not this
page.

## Results

Decode, empty prompt, 32 generated tokens, 32 threads, three runs per leg:

| leg | runs (tok/s) | median | vs cache off |
|---|---|---:|---:|
| **cache OFF** — what `main` ships today | 3.55 / 3.62 / 3.79 | **3.62** | 1.00x |
| default, first implementation (8.25 GiB budget, 33% hit rate) | 3.84 | 3.84 | 1.06x, a tie |
| **default, repaired** (26.5 GiB budget, 80% hit rate) | 7.39 / 7.45 / 7.47 | **7.45** | **2.06x** |
| explicit 16 GiB budget | 6.38 / 6.95 / 6.99 | 6.95 | 1.92x |

Prefill, 512-token prompt, cache off, one leg per thread count:

| threads | 8 | 16 | 32 |
|---|---:|---:|---:|
| prefill tok/s | 12.30 | 23.89 | **43.33** |

Decode thread scaling, cache off, empty prompt:

| threads | 8 | 16 | 32 | 8→16 | 16→32 |
|---|---:|---:|---:|---:|---:|
| decode tok/s | 1.38 | 2.59 | 4.02 | 1.89x | 1.36x |

The n-gram spec-decode probe, the campaign's **serving** number:

| leg | plain | spec decode |
|---|---|---|
| cache OFF | 3.90-3.92 | 6.27-6.30 |
| **default (repaired)** | 8.63-9.36 | **12.74-13.14** |
| explicit 16 GiB | 8.33-8.73 | 10.42-10.55 |

**The campaign's 12 tok/s target is MET** by the shipped default: the spec
decode leg reads 12.74-13.14 tok/s. It is **not** met with the cache off
(6.27-6.30). The caveats are in [Limitations](#limitations) and they travel
with the number.

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

The decode and prefill legs run the session's bench driver, rebuilt from the
branch head, under the node pinning and the mutex:

```sh
export TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal-pin
export TT_METAL_RUNTIME_ROOT=/home/lu_zero/Sources/tt/tt-metal-pin
BR=/tmp/kolibri-scale/bench-run

# empty-prompt decode, 32 tokens, 32 threads: cache off
numactl --cpunodebind=3 --interleave=all env VT_KB_PROMPT=0 \
  VT_KB_GEN=32 VLLM_CPP_CPU_THREADS=32 \
  VT_KOLIBRI1_DEQUANT_CACHE_MB=0 "$BR"

# the same with the DEFAULT (the variable unset -> the derived budget)
numactl --cpunodebind=3 --interleave=all env VT_KB_PROMPT=0 \
  VT_KB_GEN=32 VLLM_CPP_CPU_THREADS=32 "$BR"

# the same with the explicit 16 GiB budget
numactl --cpunodebind=3 --interleave=all env VT_KB_PROMPT=0 \
  VT_KB_GEN=32 VLLM_CPP_CPU_THREADS=32 \
  VT_KOLIBRI1_DEQUANT_CACHE_MB=16384 "$BR"

# prefill p512, 8 tokens, 32 threads, cache off
numactl --cpunodebind=3 --interleave=all env VT_KB_PROMPT=512 \
  VT_KB_GEN=8 VLLM_CPP_CPU_THREADS=32 \
  VT_KOLIBRI1_DEQUANT_CACHE_MB=0 "$BR"

# the in-repo production-shape bench (the token anchor gate), and the
# in-repo spec-decode probe (the serving number)
VLLM_CPP_CPU_THREADS=32 /tmp/build-dq/tests/test_kolibri1_decode_bench
VLLM_CPP_CPU_THREADS=32 /tmp/build-dq/tests/test_kolibri1_spec_probe
```

The mutex is held for the whole series:

```sh
flock -x "$HOME/gpu.lock" bash -c '...the legs above...'
```

The decode and prefill legs are driven by a **session-local bench driver**
(`$BR`, the two knobs `VT_KB_PROMPT` and `VT_KB_GEN`), rebuilt from the
branch head under test. It is **not** an in-repo harness and it is not
reproducible from a clean checkout; the in-repo pair — the production-shape
bench and the spec-decode probe — is, and the spec-decode probe carries the
serving number this page publishes.

## Limitations

- **The noise floor on this box is ~4%.** Two single runs of the same cell
  (same binary, weights and configuration) differed by ~4% (8.45 against 8.11
  tok/s). **No cell on this page is quoted more finely than that**, and two
  legs that differ by less than 4% are a tie, not a win.
- **The default's margin over an explicit 16 GiB budget is not claimed as a
  win.** In this shape the explicit leg reads 6.38-7.06 tok/s across the
  session's readings of that cell, against 7.39-7.47 for the derived
  default: a ~6% gap, above the 4% noise floor but inside the explicit leg's
  own spread. What every reading agrees on is the **~2x of the default over
  the cache-off configuration**.
- **The 12 tok/s margin is about one noise floor** (12.74 against 12, ~6%).
  It clears the target as measured.
- **The numbers are host-specific**: 32 threads, one NUMA node, 3.4 GHz,
  aarch64, `governor=performance`, on node 3 of a shared box. They are not a
  portable tok/s figure.
- **The default is not on `main` yet.** PR #3488 was open at the time of
  writing; on `main` the shipped configuration is the cache-off one, and the
  same probe reads 6.27-6.30 tok/s.
- **A sub-knee budget is worthless**, and the measurement says so: the
  default's first implementation (8.25 GiB, 33% hit rate) delivered 3.84
  tok/s, a 1.3% difference from the cache-off median and inside the ~4%
  noise floor. The policy now refuses a sub-knee budget outright.
- **The spin budget is not adopted on this evidence**: `VT_CPU_SPIN_ROUNDS=0`
  moved the same cell from 4.019 to 4.193 tok/s at 32 threads and from 2.587
  to 2.581 at 16, so the paired reading does not separate the arms beyond the
  noise floor. One paired reading on one host is not a settled lever question
  (the session's t32 cache-off readings ranged 3.49-4.02 tok/s across the
  day).
- **The cache-off baseline is a band, not a point.** One cache-off run in
  the session's other shape read 2.45 tok/s against 3.49 and 3.58, and the
  t32 leg of this record reads 4.02, so the cache-off band runs 2.45-4.02
  tok/s across the session's readings; the default's advantage over it is
  1.85x-3.0x depending on which baseline run is picked.
- **No bf16 arm was measured**, and none can be: the loader refuses it.
- **No vLLM or plugin denominator.** These are absolute tok/s on one host.

## Evidence

The full record — host state, the commands, the box-quiet window, the noise
floor, the per-cell readings, and the values the tree's own commits quote for
the same window — is
[`docs/bench-evidence/kolibri1-cpu-dequant-cache-default-20261021/report.md`](../bench-evidence/kolibri1-cpu-dequant-cache-default-20261021/report.md).
The policy the default is measured against is PR #3488
(`84d98a9e6` the flip, `d47173c51` the sampling-moment repair and the knee
floor, `e7d7704d3` the knee refusal); the row's own record is
[`.agents/specs/kolibri-1-cpu.md`](../../.agents/specs/kolibri-1-cpu.md).
