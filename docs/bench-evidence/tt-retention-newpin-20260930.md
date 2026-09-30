# Engine-side DRAM-retention re-check on the NEW tt-metal pin (2026-09-30)

Verdict: **REFUTED** — the per-new-request retention staircase (settled free
DRAM dropping ~948 MiB/request, tt-metal#57970 as filed 2026-09-25) does NOT
reproduce on the new pin `vllm-cpp-pin/20260923-adv` @ `6449cf13f7b`, on the
27B eager decode arm in a multi-request engine composition.

## Context

Original measurement: `tt-ttm-retention-rootcause-20260925.md` (old pin,
eager arm, 48 GDN layers × (16,777,216 B + 3,932,160 B) tilize outputs
retained per request). The standalone repro remains a flat negative control
(verified 2026-09-30); only the engine composition showed the staircase. At
HEAD the 27B serve defaults to whole-graph decode capture (the #1625 flip),
which dies at `end_mesh_trace` OOM before serving, so the eager arm was
selected with the existing toggle — no new enabler was needed.

Capture toggle: `VT_TT_DECODE_CAPTURE=0` — `DecodeCaptureEnabled()`,
`src/vt/tenstorrent/tenstorrent_device.h:57`; wired at
`src/vllm/platforms/tenstorrent.cpp:100`. The log confirms `cap=0` on every
forward (no trace capture, fully eager).

## Leg

Host thalia (personal P150a, file mutex `$HOME/gpu.lock`), luwen reset +
sleep 15 first. Env pinned directly at the pin tree (the stale
`env-tt-common.sh` runtime tree lacks the pinned `_ttnncpp.so` symbol
`ttnn::transformer::chunk_gated_delta_rule`, which aborted the first attempt
with a symbol-lookup error — pin `LD_LIBRARY_PATH` at
`~/Sources/tt/tt-metal-pin/build_release_script/lib64`, not
`~/Sources/tt/tt-metal/build_Release/lib64`):

```sh
flock /home/lu_zero/gpu.lock bash -c 'for i in 1 2 3; do ~/Sources/tt/luwen/target/release/reset; [ $? -ne 101 ] && break; sleep 15; done; sleep 15;
export TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal-pin
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
export PYTHON_ENV_DIR=/home/lu_zero/Sources/tt/tt-metal/python_env
export LD_LIBRARY_PATH=$TT_METAL_HOME/build_release_script/lib64:$PYTHON_ENV_DIR/lib
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
cd /tmp/vllm-region-capture-spec/build2 && VT_TT_DECODE_CAPTURE=0 VT_TT_KEEPQUANT_INT8DOT=0 VT_TT_ALLOC_TRACE=1 \
  timeout -k 10 9000 ./examples/vllm-bench \
  --model /mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf \
  --num-prompts 8 --input-len 128 --output-len 32 --concurrency 4 \
  --num-blocks 64 --max-num-batched-tokens 64 --seed 0; echo BENCH_EXIT=$?'
```

BENCH_EXIT=**124** (the 9000 s wrapper timeout cut the tail of the second
wave; see the honesty note below). No `bank_manager` OOM, no
out-of-memory of any kind, in either leg. A first leg with the same workload
minus the ledger (`VT_TT_TRACE_DEBUG=1`, timeout 5400 s, BENCH_EXIT=124)
also served through with no OOM — the wrapper exit means nothing here.

Free-DRAM telemetry: the dram-ledger instrument `VT_TT_ALLOC_TRACE`
(`[TT-ALLOC]` lines, `src/vt/tenstorrent/tenstorrent_capture.cpp:392`),
read at the per-forward settled boundary `label=block/0/pre`
(`src/vllm/model_executor/models/qwen3_5.cpp:10133`).

## Result: settled free DRAM per bank (8 banks, P150)

Wave 1 (requests 1–4 joined; 32-token outputs, batch 4):

| forward | snapshot | free/bank (B) | note |
|---|---|---|---|
| 1 (prefill) | #73 | 3,892,482,432 | post-load steady state 34.18 GB/bank minus KV/weights staging |
| 2 | #6063 | 315,611,520 | first decode settles the transient floor |
| 3 | #11539 | 315,529,600 | |
| 4 | #17015 | 315,353,536 | |
| 5 | #22491 | 314,853,824 | |
| 6 | #27967 | 312,975,040 | warm-up drift, ~2.6 MB/bank total |
| 7 | #33443 | 302,011,968 | settles |
| 8–33 | #38919…#175819 | **301,930,048** | **EXACTLY FLAT for 26 consecutive forwards** |

Wave 2 (requests 5–8 joined at the wave boundary, forward 34):

| forward | snapshot | free/bank (B) | note |
|---|---|---|---|
| 34 | #181295 | 301,973,056 | +43 KB vs wave 1 floor (new prefill reclaims) |
| 35 | #186771 | 294,643,264 | transient dip, then recovers |
| 36–39 | #192247…#214151 | 293.1–294.3 MB | bounded drift |
| 40–62+ | #219627…#334623 | **300,485,696** | **EXACTLY FLAT for 23 consecutive forwards** |

Per-new-request step: wave 1 → wave 2 settled floor moved
301,930,048 − 300,485,696 = **1,444,352 B/bank = 11.6 MiB across 4 new
requests ≈ 2.9 MiB/request** — versus the old pin's 948 MiB/request. The
staircase is gone; settled free DRAM holds a stable ~2.4 GB total floor
across both request joins with zero monotonic decline. No request died; the
run was still serving when the wrapper timeout fired.

## Honesty notes

- The leg was cut at ~forward 62 of ~66 by the 9000 s wrapper timeout
  (eager 27B on P150 is ~90 s/forward under the ledger instrument). Both
  request joins were fully observed; only the final few decode steps of
  requests 7/8 were not. Nothing about the staircase signature depends on
  them: a 948 MiB/request step would have drained the observed 300 MB/bank
  floor within one request.
- The `[TT-ALLOC]` per-op deltas inside a forward still show the transient
  churn (e.g. −297,722,880 B `kq-decode/repair` spikes that recover
  +217,945,600/+109,025,280 within the same forward); the claim here is
  about the SETTLED per-forward free, which is flat.
- The PA device-decode path in this build falls back
  (`EnsurePagedKvTtnn: contiguous rank-4 NHD cache` refusal at
  `tenstorrent_paged.cpp:445`, per-step `[TT-TRACE]` in leg 1) — decode
  serves through the host-backed arm. This is the same arm shape the
  original staircase was measured on (eager, GDN scatter web active).

## Interpretation for the upstream reply

tt-metal#57970 as filed (the engine-side 948 MiB/request retention
staircase) no longer reproduces on pin `vllm-cpp-pin/20260923-adv`
(`6449cf13f7b`). One of the ~576 commits between the old pin and this one
drained the deferred-reader retention (or the to_layout chain now owns its
transients). The standalone repro stays a negative control. Reply held
pending this result; nothing was posted upstream from this session.

Raw logs: `/home/lu_zero/.local/logs/maki/CeRgUcXiFK5bYWGSPS4sy/
monitor-1790769562-3eed/stdout.log` (leg 1, TRACE_DEBUG) and
`.../monitor-1790775009-43f2/stdout.log` (leg 2, ALLOC_TRACE ledger,
~340k snapshots).
