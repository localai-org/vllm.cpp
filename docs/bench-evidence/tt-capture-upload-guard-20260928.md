# tt capture-scope upload guard — the leg that falsifies the inline-upload attribution (2026-09-28)

Worktree `row/tt-27b-region-capture-spec`, fixes `286947603` (red test) +
`39e2ca8ef` (guard + broadcast-operand cache), audit input
`docs/bench-evidence/tt-trace-record-audit-20260928.md`. Legs:
`~/.local/logs/maki/CeRgUcXiFK5bYWGSPS4sy/monitor-1790617827-8e22/stdout.log`
(the full suite), `.../monitor-1790618190-57bb/{stdout,stderr}.log` (the 27B
whole-graph c1 leg), and the in-tree region-handoff test run.

## What landed

1. **The guard.** `UploadRows` and `UploadRowsBf16`
   (`src/vt/tenstorrent/tenstorrent_residency.cpp`) refuse any H2D upload with
   `tt_capture_active()` set, by name, after the required
   `VT_TT_TRACE_DEBUG` print on the route. `AddKernel`'s broadcast operand
   (`src/vt/tenstorrent/tenstorrent_ops.cpp`) moves behind a cache keyed by
   host pointer, geometry, and an FNV-1a hash of the operand's values: the
   eager pass uploads once, the capture pass serves the resident copy, and a
   capture-scope miss refuses by name (no silent inline, no value staleness).
2. **The red-first test** ("kTENSTORRENT capture-scope upload refuses and the
   warmed capture records the 2 KB floor"): red on `286947603`'s parent — the
   unwarmed capture-scope upload fired and died on tt-metal's own
   `TT_FATAL fd_mesh_command_queue.cpp:826 !trace_id_.has_value()` (no named
   refusal) after printing `[TT-UP] UploadRowsBf16 from_span WRITE during
   capture rows=1024 cols=1508`. Green after the fix: named refusal + the
   warmed capture records **1,024 B**.

## The money leg and what it actually showed

27B whole-graph c1 (`VT_TT_TRACE_DEBUG=1 VT_TT_KEEPQUANT_INT8DOT=0`,
`--num-prompts 2 --input-len 128 --output-len 32 --concurrency 1`):
**BENCH_EXIT=1. No TPOT table — the leg died at the same capture end.**
The fatal is byte-identical to the pre-row one: `end_trace_capture` asks for
**3,153,969,152 B** against 298,568,896 B free
(`bank_manager.cpp` OOM, `assert.hpp:104`).

But the census around it is decisive:

- **Zero** `[TT-UP]` lines in the whole leg: no upload route (staging,
  broadcast-Add, rope cache, ids) attempted an H2D write under capture. The
  doctrine's eager pass already covers every site — the guard never had to
  fire.
- **Zero** `[TT-KQ]` keep-quant word-shadow refusals: that route was already
  refusing/covered.
- The 6 `EnsureHostBytes DURING CAPTURE` readbacks fired as before (known
  sync hazard, zero trace bytes, still owed).

So on this pin, **the ~3.15 GB demand persists with zero capture-scope
uploads — the audit's model C (inline H2D payload) is falsified for the
default whole-graph arm.** The discriminating experiment the audit itself
proposed settles it at op scale: the region-handoff test with the MatmulBT
fully warmed closes region 1 at exactly **3,088,384 B** — the same close the
audit attributed to a ~3.086 MB inline upload. That payload is NOT an upload;
it is the recorded per-program command stream of the MatmulBT program class
(a warmed RmsNorm region still closes at 2,048 B, so binaries-by-relay holds
for small programs — the ~3 MB rides with the quant-matmul program's launch
record, mechanism unattributed at dispatch level).

## Verdict and next hypothesis

- The guard and the warmable broadcast operand are correct hardening and stay
  (they convert the pre-fix raw TT_FATAL into a named refusal and remove the
  per-call broadcast re-upload in both passes), but they do not shrink the
  whole-graph record, because there was nothing left to shrink from the
  upload side.
- The 27B decode-trace DRAM-fit site closes only on a tt-metal-side
  attribution: dump the dispatch command stream
  (`tt::LogDispatch` trace level) for one captured `MatmulBTQuantGrouped`
  launch and find the ~3 MB of `bypass_data` words. Next candidates: the
  program's kernel-binary relay pages being re-relayed per launch for
  programs that miss the 1,024 KB prefetch ringbuffer
  (`fd_mesh_command_queue.cpp:453` fit decision), or per-launch config-page
  writes scaling with the quant-program's CB/RTA footprint.
- Region arm unchanged: `VLLM_CPP_REGION_CAPTURE=1` stays env-gated; the
  64 live regions summing to the same ~3.15 GB (previous evidence) is now
  doubly explained — the demand is per-program, so segmentation cannot help
  either.

## Suite

98 cases, 526,778 assertions: **526,777 passed, 1 failed** — the pre-recorded
batched-RAC residual flake (96/128 K/V, user-1 second head,
`test_tenstorrent_backend.cpp:2261`), the exact owed failure the issue
already records. No new failures from the guard.
