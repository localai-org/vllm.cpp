# SPEC — `FLINT-CPU-PARITY` — the llama.cpp floor on flint Q4_K_M, measured like-for-like, and the gap our keepquant arm must close

- Owning row: `FLINT-CPU-PARITY`
- Kind: performance (parity measurement + gap record)
- Issue: created in the same flow (row-scoped local issue)

## Scope

Three deliverables, in order:

1. **Oracle gateability**: build and run llama.cpp PR #30928 head
   `1b44e3077` on `flint-Q4_K_M.gguf`, capture a token golden and the tg
   rates into `docs/bench-evidence/`, flip
   `.agents/oracles/llama-cpp-flint.md` to `gateable = yes`.
2. **Like-for-like floor**: the llama.cpp tok/s on the identical artifact,
   same box, same core count and thread count as our serving harness —
   this number is the row's target and nothing else is quotable as the
   floor.
3. **Our side + the gap**: vllm.cpp keepquant Q4_K_M serving throughput
   under the same core/thread budget, and a recorded gap statement. If a
   named lever is identified, it files its OWN row; this row does not
   implement kernel work.

No product code change lands in this row unless the gap measurement
exposes a defect (a wrong number, not a slow one) — a defect is fixed
here, a slowness is filed.

## Why llama.cpp is the peer

The developer set it on 2026-10-28: llama.cpp is the peer to at least
reach for Flint CPU. It is a **registered** secondary oracle for the GGUF
arm (`llama-cpp-flint`, this row pins it), which mistral.rs is not.
Reference context, not denominators: mistral.rs serves the same Q4_K_M at
18.7 tok/s (16 threads, external session 2026-10-27); llama.cpp I-Mini
(IQ-class) measured 24.99/24.33 tok/s — the I-Mini number is NOT the Q4_K_M
floor and must never be quoted as one.

## Like-for-like contract (what makes the comparison valid)

- Identical artifact: `flint-Q4_K_M.gguf`, byte-identical file on both
  sides (sha256 pinned in the evidence doc).
- Same host (the shared aarch64 box), same NUMA node, same core count and
  thread count as our serving harness's configuration, same governor
  (`schedutil`, untouched), same box-claims discipline
  (`~/.config/box-claims.txt`), idle gates per `.agents/benchmarking.md`.
- Greedy decode, temperature 0, finish reason length, 200 and 400
  completion windows; quote steady-state windows only.
- vllm.cpp side: keepquant arm (`VT_FLINT_KEEP_QUANT=1`), the arm that
  consumes the same GGUF bytes.

## Ground truth

- vllm.cpp reference point: Flint CPU decode 1.26 tok/s (TPOT 797 ms,
  32 threads) on the paged bf16 arm — pre-parity, not comparable to a
  quantized serving rate. The keepquant arm has no throughput number yet;
  this row produces the first one.
- llama.cpp PR #30928: Qwen2-MoE-derived graph with head-dim q/k RMSNorm,
  ISWA sliding/full split, per-head `attn_sinks`, sigmoid gating, and the
  Q8-activation amax rescale for the MoE FFN (`build_amax_src1_f16`).
  Its expert path runs packed indexed GEMV on k-quant blocks.
- Our keepquant arm: resident packed slots + per-dtype dispatch
  (int8-dot / dequant paths per dtype). Different kernel strategy from
  llama.cpp's packed GEMV — the gap decomposition names where they
  diverge.

## Risks

- The llama.cpp PR is unmerged upstream; its head could rebase. The pin
  records the exact SHA; a rebase that changes numerics re-opens the pin,
  not silently.
- `expert_shared_count` = 2 (one double-width shared expert as two slots
  in one rank-2 matrix): our loader's shexp slicing must be verified
  against the golden in deliverable 3 before any throughput number is
  quoted. A mismatch here is a correctness defect and is fixed in this
  row.
- Shared-box contention: all legs idle-gated and claimed; a contaminated
  leg is discarded, not averaged.

## Tests

- Gateability: llama.cpp serves the artifact, greedy tokens captured;
  `gateable` flips with the evidence path in the pin block.
- Golden cross-check: vllm.cpp keepquant tokens vs the llama.cpp golden
  on the same prompt — the arms are different kernels, so exact match is
  NOT expected (llama.cpp does integer block dots; our rel_rms contract
  governs). Divergence beyond the established band is a defect finding,
  not a perf finding.
- Throughput: 3-run medians per side, per the benchmarking guide; the row
  reports one floor number, one vllm.cpp number, one gap ratio.

## Gates

- `python3 scripts/check-oracle-pins.py` green with the new oracle file
  (the AGENTS.md registry row for `llama-cpp-flint` lands in the same
  change).
- `python3 scripts/check-agent-record.py`, commit style, trailers,
  preflight (known host dotnet failure named).
- Device-free: nothing in this row needs the P150.

## Evidence

- External session 2026-10-27: mistral.rs Q4_K_M 18.7 tok/s; llama.cpp
  I-Mini 24.99/24.33 tok/s (`/tmp/grok-llama-imini/bench.log`, untracked).
- Oracle build present at `/tmp/llamacpp-flint-oracle` @ `1b44e3077`.
- vllm.cpp keepquant arm: #3514, landed 2026-10-28 as `3284412c0`.

## Stop conditions

- If the llama.cpp PR head no longer builds or the artifact stops
  loading, report `REMOTE_UNVERIFIED`-style: the row stalls on an
  external blocker, pin stays `gateable = no`.
- If our keepquant tokens diverge from the golden beyond the band, STOP
  the perf legs: correctness first, file the defect, fix, then measure.

## Git integration

One pull request, `row/FLINT-CPU-PARITY`, spec committed first. The
oracle file, the AGENTS.md registry row, the evidence docs, and any
defect fix land together; kernel-speed work does not.
