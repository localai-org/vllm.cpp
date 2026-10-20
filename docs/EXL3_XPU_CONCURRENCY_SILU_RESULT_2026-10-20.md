# Experimental concurrency coverage for FP16 SiLU on B70

`VT_XPU_SILU_SMALL_DECODE=2` extends the existing width-17408 table route to
rows 1, 2, 3, 4, 8, 12 and 16. The default stays `1` (rows 1/4); `0` keeps its
existing rollback meaning. Wrong widths/dtypes/layouts retain the existing fallback or refusal.
Exact 0/1/2 parsing and graph-policy identity cover the new mode. The same
immutable 131,072-byte table and kernel are reused, with no new per-token
allocation or synchronization.

The arithmetic remains `SiLU(gate) -> FP16 -> multiply(up) -> FP16`.
GDN's FP32-intermediate SiLU is a separate contract and is unchanged.

## Attribution and complete-call control

Original C4-4K/16K graph-replay profiles contain 67 MLP SiLU calls per steady
cycle: 65 generic calls and two table calls. SiLU accounts for 2.622/2.632 ms,
about 4.07%/3.20% of the recorded GPU interval union. Those fractions do not
represent the entire native/Python gap, and the profiling overhead is not
scored request latency.

Transparent same-queue copies capture the actual M16 gate/up values before
consumption in both target graph ring slots. Three distinct, finite replay
frames per workload are retained. The operator control uses each workload's
last frame and prefix rows for M1/2/3/4/8/12/16; prefix slices are not independent
new C2/C3 trajectories. Mode1 versus mode2 outputs match exact FP16 bytes,
including protected guards and unchanged inputs. The profiler-free complete
public-call screen includes dispatch and completion, two warmups, seven samples
per arm, one A/A pair and three alternating A/B pairs per row. M16 median-call
reductions are 82.3-82.6% at 4K and 81.8-83.6% at 16K; unchanged M1/M4 routes
remain controls. These are complete-operator measurements, not serving gains.

The separately profiled complete target-MLP GPU chain (transforms, gate/up GEMM,
SiLU, down GEMM and intervening gaps; 64 target MLPs per cycle) falls from
23.834 to 21.533 ms at 4K and from 23.911 to 21.535 ms at 16K, about 9.7%/9.9%.
This saves roughly 2.3-2.4 ms, or about 3% of the recorded GPU cycle; it does
not establish 10% higher request throughput. The three additional draft MLPs
are accounted for separately in the raw profiles.

## Original serving screen

One frozen executable runs both arms with request-stable auto alignment and
long-Q1 mode2. Mixed-prefill stays off. All event/host/graph profilers and route
traces are off in scored workers. The original HTTP fixtures, seeds,
temperature1/top-p0.95/top-k20, 1,024-token caps, model/cache/queue settings and
conditioning lifecycle are preserved. Every arm uses a fresh worker.

The declared screen is two adjacent C4-4K A/A workers, three alternating A/B
pairs for each C4 workload and one C1-4K sentinel pair. The A/A fully overlapping
aggregate decode changes from 140.154 to 133.377 tok/s (-4.835%); its median
TTFT changes +0.697% and E2E +2.622%. This is one observed noise estimate,
not a population bound or a statistical significance claim.

| Case | Pair | Reference tok/s | Candidate tok/s | Decode change | Median TTFT change | Median E2E change | Accepted ref/cand | Proposed ref/cand |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c4-4k | 1 | 140.716 | 144.578 | +2.744% | +0.124% | -1.550% | 2566/2569 | 4599/4578 |
| c4-4k | 2 | 137.678 | 143.402 | +4.158% | -0.097% | -2.658% | 2540/2560 | 4662/4605 |
| c4-4k | 3 | 138.759 | 145.441 | +4.816% | +0.118% | -2.185% | 2555/2571 | 4626/4584 |
| c4-16k | 1 | 123.899 | 129.298 | +4.357% | -0.063% | -1.446% | 2610/2603 | 4473/4488 |
| c4-16k | 2 | 125.344 | 127.793 | +1.953% | +0.156% | -0.440% | 2603/2615 | 4488/4449 |
| c4-16k | 3 | 124.366 | 124.490 | +0.099% | +0.006% | +0.033% | 2615/2574 | 4449/4581 |

C4 is the fully overlapping aggregate counter/window metric. Median paired
changes are +4.158% decode / +0.118% TTFT / -2.185% E2E at 4K and +1.953% /
+0.006% / -0.440% at 16K. The smallest 16K throughput difference is +0.099%.
Acceptance/proposal counts and stochastic output hashes vary, including in A/A;
no claim of identical complete-request outputs or new strict task-quality PASS
follows from the exact same-input SiLU control. The known strict closing-fence
failure remains a separate limit.

All 16 predeclared workers complete successfully. The single C1-4K sentinel
pair gives 57.344 -> 57.331 client post-first-token tok/s
(-0.021%), median TTFT +0.006% and E2E +0.020%; its exact output hashes match.
This is a sentinel pair, not a separate performance qualification. C4 workers
compute exactly 16,292/65,460 cold prompt tokens, and C1 computes 4,091; every
scored worker has zero cached tokens and zero preemptions. Host cgroup memory
peaks are retained; they are not VRAM measurements.

## Detecting focused checks and identity

- Policy RED: seven new-row admission assertions fail before the extension.
  GREEN: two cases, 55 assertions pass.
- GPU operator/fallback/graph ownership: three cases, 1,005 assertions pass.
  All 65,536 half gate patterns are exercised against ten selected up patterns
  at each admitted row count. Finite values, infinities and signed zeros retain
  exact bits; NaN classification is preserved, with payload selection excluded
  from that IEEE arithmetic contract. Wrong width, BF16/F32 fallback, strided
  refusal, shifted aliases, cold queue capture, queue joins and changed-input
  consumed graph outputs are covered.
- Captured-input complete-call screens: one case / 237 assertions per context,
  profiler disabled; actual finite input/output and guard bytes match exactly.
- Graph-policy seam: two cases / 112 assertions pass, including mode2 identity
  and restoring the default identity. No full test suite was run.

Measured server SHA-256: `2e783827d43fb22fb0bee58e9f3dd099f39f544f9f034213ff3a575632f61b20`.

Product library SHA-256: `268c73652a15528ee535fa6ead0caa0e1429ea178ca9662db473d7526ae0bf2a`.

The source is base commit `db6ef69e9e0fc609900869eb471bd16112d9114e` plus the
five exact SiLU source/test files in the retained source manifest and snapshots.
The measured tests and server are frozen independently of the build cache.
Evidence directories are `phase-next-pro-silu-v1`,
`phase-next-pro-silu-operands-v1`, `phase-next-pro-silu-profile-v1` and
`phase-next-pro-silu-serving-v1`; raw pairs, settings, MTP/work/cache counters,
outputs, receipts and identities remain available there.

Disposition: retain experimental mode2; do not change the default. Complete
operator and MLP costs improve, but the declared small serving screen does not
separate a general serving benefit from the observed A/A variation. No new full
native/Python matrix is justified by this unpromoted candidate alone.
