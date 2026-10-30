# SPEC — `FLINT-KQ-CPU-DISPATCH` — the CPU paged arm consumes the keep-quant block tree; the 8x lever under the llama.cpp floor

- Owning row: `FLINT-KQ-CPU-DISPATCH`
- Kind: feature (the named follow-up row from FLINT-CPU-PARITY)
- Issue: created in the same flow (row-scoped local issue)

## Goal

Make the sentence "Flint runs quantized on CPU" literally true: the paged
serving arm accepts the keep-quant block tree (`VT_FLINT_KEEP_QUANT=1`) and
serves it, replacing the named refusal that landed with FLINT-CPU-PARITY
(`flint.cpp:868-881`). The measured target is the gated llama.cpp floor:
**26.29 tok/s at 16 threads** on `flint-Q4_K_M.gguf` (sha `20c89707...927e`,
goldens + raw logs committed in `docs/bench-evidence/flint-cpu-parity-20261029/`).

## What already exists (do not rebuild it)

- The keep-quant LOADER is landed (#3514 lineage): admission by name, tower
  slicer, block staging, per-dtype dispatch (`MatmulWeight` → `vt::MatmulBT`
  in the TT path), `gguf_dequant.cpp` knows every encoding the artifact uses.
- The llama-cpp-flint oracle is gated: floor numbers, token goldens, raw
  logs, pin — all committed on main.
- The shexp mapping is verified correct (one double-width shared expert as
  two slots of 512).
- The named refusal at `FlintForwardPaged` is the red-first marker: the arm
  flips from refusal to serving, and the refusal's test vehicle
  (`keepquant-refusal.log` shape) flips to a serving gate.

## Design intent (the implementer owns the details)

The CPU paged forward consumes the block tree through the same per-dtype
dispatch seam the TT row built — the shared seam, not a parallel path. The
open engineering question is which compute class each dtype gets on CPU
(dequant-to-bf16 GEMM vs int8-dot vs a NEON quantized dot), decided by
MEASUREMENT against the floor, not by preference. The parity row's kernel
map from mistral.rs is reference context: their 63→23 ms routed drop came
from Q8_K block-dot gathers. Start with the simplest correct arm (dequant +
existing bf16/int8 dot path), measure, then descend into quantized dots only
where the floor demands it.

## Non-goals

- No TT-path change.
- No new GGUF encodings.
- No public benchmark ID until the arm has a recorded serving number
  (docs/BENCHMARKS.md row lands with the number, in this row's final commit
  if the number is quotable).

## Red-first

The refusal test flips: `VT_FLINT_KEEP_QUANT=1` through the paged serving
path must go from "refuses by name" to "serves and matches the oracle
golden within the established band" (the `test_flint_gguf` contract shape:
46/53 exact, 0 hard flips, in-band at landing — the bar is that band, not
bit-exactness; llama.cpp uses integer block dots and we may not).

## Gates

- The keep-quant tree serves end-to-end through the paged arm; the golden
  cross-check holds in-band.
- Throughput at 16 and 32 threads recorded next to the floor, same zone
  (32-63), same governor state logged, idle-gated per the benchmarking
  guide; 3-rep medians; the claims-file discipline.
- All existing flint suites green; the refusal-flip test inverted.
- check-agent-record, style, trailers, env-doc if any new env, preflight
  (named host dotnet failure).

## Stop conditions

- If the keep-quant tree cannot be made to serve without a loader change
  beyond dispatch (i.e. the block shapes themselves don't fit the paged
  forward), STOP and report — that is a material discovery, not a detail.
- If in-band token parity cannot be reached with any compute class, STOP
  and report the closest band achieved — a numerics gap is a finding, not
  something to widen.

## Git integration

One pull request, `row/FLINT-KQ-CPU-DISPATCH`, spec committed first.
