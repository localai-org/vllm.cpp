# TT-EMBED-DUAL-RESIDENCY — W3 STOP lifted: the frozen recorded copy, root-caused and repaired

Evidence for row `TT-EMBED-DUAL-RESIDENCY` (issue
`ISSUE-LOCAL-01M7CFF6ESGHZ2WQDA74PG4VDY`). Implementation heads: W1
`881752c9b`, W2 `be416cc5a`, W3 `95ff6d146`, repair on top (see
`git log --grep TT-EMBED-DUAL-RESIDENCY`). Build: `/tmp/build-embed` (same
recipe as `tt-staging-footprint-20261113/README.md`). Recipe per leg: mutex
`${GPU_LOCK:-$HOME/gpu.lock}`, luwen reset + 15 s + `/dev/shm` cleanup,
idle gate load<16, `taskset -c 32-63` runs / `64-127` builds, governor/
env/ps-top logged (`*-env.txt`, `ps-top-*.txt`), bench run FROM THE REPO
ROOT. Gate script: `run-unlock-gate.sh` (the 13:44-13:49 `q6kl-cap-*` cwd
failures in the W3 STOP record remain residue).

## Root cause: a recorded `ttnn::copy` FROM the row shadow replays capture-time bytes

The bisection in the W3 STOP record stands: the captured quant-row branch
introduced in W2 `be416cc5a` broke BOTH artifacts identically (`a0033dd8…`),
and the W3 dtype fix neither caused nor repaired it. The trace work
instrumented the embed site (`VT_TT_TRACE_DEBUG`, `[TT-EMB-DBG]` lines) and
settled the recorded hypotheses:

1. **The refresh is correct and reaches the pinned address.** Per replay
   step `WarmEmbedQuantRows` decodes this step's rows on host (per-step
   `[TT-EMB-DBG] RefreshQuantEmbedRows HIT` hash CHANGES with the step id)
   and H2D-copies into the SAME buffer address the capture baked
   (`shadow_addr=240586496` at the dummy run, at capture, and at every
   refresh). The eager arm — which decodes through the same
   `RefreshQuantEmbedRows` — is byte-identical at every head, so the host
   decode itself is exact.
2. **The read side is frozen.** Poison probe (`VT_TT_EMB_POISON`, mechanics
   test only — the `VT_TT_NO_IDS_WARM` precedent): zeroing the ENTIRE row
   shadow after the refresh, before every replay, left the captured stream
   bit-for-bit unchanged (`220,16,16,220,16,…`, the attractor). A replayed
   read of that buffer would have seen zeros.
3. **Refinement of H1.** It is not the W3 `typecast`/`to_layout` ops alone:
   the pure same-dtype copy variant (bf16-staged shadow, no conversion op in
   the region at all) diverges IDENTICALLY. The capture pins the STAGED
   buffer's CONTENT, not just its address, for the high-level `ttnn::copy`
   (the #2812 pin-bytes class). The recorded ops that ARE replay-live in
   this trace are the custom kernels' reads (`WarmRacIdx`/`WarmPaMeta`) and
   `ttnn::embedding`'s recorded ids read (`WarmDecodeIds`) — which is why
   the W1-era captured arm passed and the quant row branch (a copy) did not.

Bisection artifacts kept: `bisect-q4km*`, `q4km-cap-1-token-ids.json`
(W3-era divergence), `q6kl-cap-pure-1`, `q6kl-cap-sync-1`.

## Fix (stage-B contract: the representation change lands OUTSIDE the region)

- NEW `vt::tenstorrent::EmbedQuantRowArm(table_host, n, h)`
  (`tenstorrent_capture.cpp`, declared in `tenstorrent_device.h`): the same
  map match `EmbedDeviceIdsInto`'s quant arm uses, exposed to the drivers.
- Drivers `qwen3.cpp` (replay + capture) and `qwen3_5.cpp` (replay +
  capture): when the embed is the quant row arm, the in-region
  `EmbedDeviceIdsInto` is SKIPPED (the trace records no embed) and the
  commit runs OUTSIDE the region per replay step, right beside
  `WarmEmbedQuantRows` — allocation-free (the shadow and the hidden slot
  are persistent; the copy program is warmed by the R4 dummy run at
  capture). A float table keeps the in-region recorded `ttnn::embedding`
  (its ids read is replay-live; its twin never changes — that path passed
  before W2 and is untouched).
- `EmbedDeviceIdsInto`'s quant arm now REFUSES a hidden slot whose storage
  does not match the shadow's dtype+layout, by name: any in-region
  conversion replays frozen capture-time bytes (the W3 lesson, made loud).
- The row shadow is staged BF16 (the twin's exact bits; `from_vector`'s
  f32->bf16 upload cast is the identity on bf16-rounded values), so the
  eager f32-out arm typecasts after the refresh, outside any trace.
- qwen3_5.cpp's capture step gained the missing pre-mirror
  `WarmEmbedQuantRows` (qwen3.cpp:1022-1025 already had it): without it the
  capture would pin the previous eager step's rows.
- Instrumentation kept as bisection knobs: `VT_TT_TRACE_DEBUG`
  (`[TT-EMB-DBG]` prints), `VT_TT_EMB_POISON` (mechanics test only).

## Gate results (all on the repaired tree, 3 reps each, 200 tokens, seed 0)

| Gate | Result |
|---|---|
| Q4_K_M captured, 64 blocks ×3 | `d49c53d2…` 3/3 — byte-identical to the recorded pass |
| Q4_K_M eager, 64 blocks | `d49c53d2…` — unchanged |
| UD-Q6_K_XL captured, KV 8 ×3 | `b36bc61a…` 3/3 — byte-identical to the recorded eager stream |
| UD-Q6_K_XL eager, KV 8 | `b36bc61a…` — unchanged |
| TPOT captured vs eager (UD-Q6_K_XL, KV 8) — W3(b), now on the CORRECT stream | captured 5329.15 / 5319.70 / 5318.76 ms vs eager 5341.31 ms — capture costs nothing measurable |
| Largest fitting KV pool — W3(c) | KV 16 fits (no OOM), deterministic 2/2, sha `5b34331d…` (`q6kl-cap16-*`) |
| DRAM ledger — W3(d) | `VT_TT_ALLOC_TRACE`, `q6kl-cap-1.stderr.log`: 1743 snapshots, tightest `total_free=2,634,189,824 B` (~329 MB/bank × 8 banks) — the ~2.5 GB freed twin stays freed; the ~80 MB trace region allocates |
| Focused | `test_gguf_keep_quant` 63/63 cases, 12205/12205 assertions (`/tmp/kq2.log`) |

Q4_K_M per-leg wall time in `*-env.txt`; per-leg environment, governor,
load and ps-top snapshots beside every leg.

## Owed (row stays OPEN)

- Fresh reviewer cycle on the repair (mutation of the guarantee: gate the
  out-of-region commit off and the captured stream must re-diverge; delete
  the driver call site and the focused gate must catch it).
- `docs/USAGE.md` weights-and-checkpoint line for the row's gate artifacts,
  at row DONE.
