# SPEC — `QWEN38-P150-REBENCH` — the qwen3.8-27B P150 re-bench: the first current-head measurement since the 09-14 end-to-end pass

- Owning row: `QWEN38-P150-REBENCH`
- Kind: benchmark (measurement; the first re-measurement of qwen3.8-27B on the P150 since 2026-09-14)
- Issue: created in the same flow (row-scoped local issue)

## Why now

Qwen3.8-27B (Q4_K_M keep-quant) completed end-to-end on the P150 on
2026-09-14 (RC=0, 2 requests × 64 tokens, TTFT 1389 s, TPOT 29.7 s,
0.03 tok/s output, 0.14 tok/s total, JIT cache 99.6%) — the only
measurement that exists. **Every performance fix since then applies to
that path and none has been re-measured on it**: the pool-match fix
(4.6x on the CPU keep-quant arm — the same threadpool pathology applies
to the TT host side), the NEON q4_K and q6_K dots (the routed experts
and lm_head), the sliding-attention decomposition, the router
total-order, and the slot-serve cure (allocation-order insensitivity —
the P150's band gate was a victim of exactly this class).

The corrected budget fits a single P150: ~22–23 GB of 32 GB.

## Deliverable

1. **The re-bench**: the same bench (the 2026-09-14 campaign's
   `vllm-bench` invocation via the production AsyncLLM path, recipe in
   `docs/bench-evidence/` from row BACKEND-TT-KEEPQUANT-W3 and the
   session records), current head, with `VT_TT_KEEPQUANT_INT8DOT=1`
   MANDATORY (the default lm_head arm OOMs: 636 MB dequant chunk).
   2 requests × 64 generated tokens minimum — matching the 09-14
   measurement for comparability. Record TTFT, TPOT, output tok/s,
   total tok/s, duration, JIT hit rate.
2. **The governor sweep** (the anchoring method): the same bench under
   `performance` AND `schedutil`, 3-rep medians each — the kolibri1
   anchoring session found the 10-21 vs 10-28 numbers were REAL CODE
   DELTA (the governor was only +2.8%), but qwen3.8-27B's 09-14 run was
   a single un-swept measurement and the governor question is open for
   it.
3. **The comparison record**: the 09-14 numbers vs the current
   numbers, with the intervening fixes named and each lever's expected
   effect on the TT path stated (pool match: host-side, applies; NEON
   dots: CPU-side compute — do they apply to the TT device path's
   host-side legs? measure; sliding decomposition: applies if the
   attention runs the same paged kernel; router total-order:
   determinism only).
4. **Public docs**: docs/BENCHMARKS.md + docs/benchmarks/ (a qwen3.8
   P150 serving entry if a number is quotable) + docs/bench-evidence/
   qwen38-p150-rebench-20261102/.

## The device recipe (mandatory, every device leg)

Mutex exec 9>$HOME/gpu.lock; flock -x 9; reset
~/Sources/tt/luwen/target/release/reset + sleep 15 + rm -f
/dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory; env
LD_LIBRARY_PATH=/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium
TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/libexec/tt-metalium
TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal-pin; build taskset -c
64-127 -j 64; bench legs taskset -c 32-63; idle gate load < 16 + ps
aux --sort=-%cpu before every measured leg; governor logged per leg;
NO timeout guards anywhere. Substrate (/tmp/umdtrial-install)
unmodified.

## Gates

check-agent-record, check-env-doc, style, trailers, preflight (named
host dotnet failure), claims file RUNNING→DONE.

## Non-goals

- No CPU-path change (the row measures; fixes discovered file their
  own rows). No TT kernel change. No new quant arms.

## Stop conditions

- The bench cannot complete (OOM at INT8DOT=1, a capture-time failure
  surviving the known fix set) → record the failure class + the
  backtrace, STOP: that is a regression finding, not a bench failure.
- The comparison shows a REGRESSION vs 09-14 beyond run variance →
  STOP and report the bisect bracket (the commit range between the
  09-14-era HEAD and current main) — a regression finding.

## Git integration

One pull request, `row/QWEN38-P150-REBENCH`, spec committed first.
