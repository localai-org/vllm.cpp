# How we measure


Record dates are CI-guarded: structured state event timestamps and ordered
indexes are validated by `check-state-record`, so scoreboard stamps remain
traceable. The
review protocol behind these numbers is guarded the same way: the reviewer and
implementer sub-agent prompts are tracked artifacts checked by
`check-protocol-consistency` (orchestration harness step 5/5), and
`check-gate-commands` pins the record rows that name a gate command able to
FAIL. That pin is exact, not shrink-only: gaining a gate command reddens it too,
so the set is never re-pinned silently in either direction (#621). Since 2026-08-07,
a PR verified green merges in that same session (disposition rule).

**Hardware.** NVIDIA GB10 / DGX Spark (sm_121a) for CUDA, `dgx.casa` aarch64 for
CPU, Apple M4 for Metal. GB10's 119 GiB pool is unified, so host and device
memory compete; end-to-end wall-clock on a cold page cache is unusable there,
and steady-state per-step timing or `nsys` GPU-busy is the anchor. The
2026-08-06 #77-slip tree-revert changed no benchmark content or number.

**Oracle pin.** vLLM <!--pin:label-->0.3.0.dev267<!--/pin--> (<!--pin:commit-->`a7c23ac96d`<!--/pin-->) since 2026-09-26, with
FlashInfer `0.6.18` and CUTLASS DSL `4.6.2`.

**EVERY BINDING RATIO ON THESE PAGES WAS MEASURED AGAINST THE PREVIOUS PIN**,
vLLM 0.26.0.dev0 (`55596792`) plus transformers 5.14.1, built from source for
sm_121a, the running oracle reporting `0.23.1rc1.dev1511+g555967922` with
FlashInfer `0.6.15.post1`, selected by explicit path and asserted per leg. The
pin advanced before those baselines were re-measured, by a developer ruling that
inverts the sync cycle's own step order (`.agents/upstream-sync.md`), because the
committed harness refuses to run at any revision but the pinned one. **The
re-measurement is owed against the new pin
([#2818](https://github.com/mudler/vllm.cpp/issues/2818)), it covers five rows
across `vllm-online-serving` and `speculative-decoding`, and a red there requires
reverting the pin rather than re-arguing the rows.** No gate of any kind has yet
run at the new pin: the declared token-exact gate is owed by
[#2794](https://github.com/mudler/vllm.cpp/issues/2794) and every committed
golden predates the advance. Speed figures labelled 0.25.0 ran the ROLLBACK the
harness enforced until 2026-08-12 and are SUPERSEDED, never binding (#520).
Correctness re-validated bit-identical across the advance, zero golden drift.
The llama.cpp oracle is stock `b10451` since 2026-08-16, `gateable = no` until
someone builds it (#857). Every llama.cpp figure here is SUPERSEDED and owed a
re-take (#1003), and a sweep finds it ran one of **three** revisions, none of
them the pin. The Reproduce table names all three.

**Protocol.** Greedy, closed loop, three interleaved repetitions per point, one
`flock` across the whole series, same-binary A/B for every lever, cold legs
discarded. Workload equivalence between arms is audited, not assumed: batch cap,
token budget, context, corpus bytes, KV and SSM dtypes, kernel family, and
graphed decode all match, and the audit is
[recorded](../../.agents/specs/benchmark-equivalence-audit-2026-07-15.md). The
2026-08-04/08 governance checkpoints (record/CI substrate, anchor backfill,
operator/helper W0-W5, upstream/device inventory, onboarding probe, the
review-hardened `agent-start.py` entrypoint, and CI-bound review-until-PASS
policy) touched no engine code and moved no number: **NOT APPLICABLE**, nothing
to reproduce.

The PR #28 sanitizer repair is also NOT APPLICABLE to performance: both full
333-test CPU detector lanes pass after merging upstream `main`, while the
ASan+UBSan build footprint falls from 93 GiB to 5.7 GiB and TSan occupies
1.9 GiB. Reproduce with the sanitizer
CTest commands preserved in the structured state evidence. The 2026-08-06 live-state audit and the 2026-08-08 state-record migration plus range-gate and stale-reference repair are likewise NOT APPLICABLE: bookkeeping, record checkers, and prose. No engine code, kernel, or number on this page changed.

**Vocabulary.** *Token-exact* means our output ids equal the reference's, byte
for byte. *Near-tie* means the reference's own greedy decode is not deterministic
at this precision, so the gate is distributional: our output must fall inside the
set the reference produces across K runs. *Tie* means the difference is inside
the measured run-to-run noise band, which is 0.5% on GB10 and 0.12% to 0.34% on
M4. We never publish a partial, contended, or stale-denominator number as
binding, and when a denominator turns out to be wrong we correct every ratio
built on it rather than keeping the flattering one.

**CPU elementwise GEMM, wide x86 tiers (2026-08-07).** INDICATIVE ONLY, not binding: the x86 dev box is VOID for timing per `CLAIM-KERNEL-CPU-ELEM-GEMM-1`. The AVX-512 tier measures 1.56x to 2.83x over SSE2 on the elementwise micro-kernels, byte-identically. A binding number needs a qualified x86 host, which the project does not have.

**CPU elementwise GEMM, transpose-free `[K,N]` path (2026-08-07).** On dgx aarch64 the `[K,N]` path beats `[N,K]` by 1.16x to 1.30x, byte-identically. The x86 arm is INDICATIVE ONLY, not binding: that box is VOID for timing per `CLAIM-KERNEL-CPU-ELEM-GEMM-1`. `VT_CPU_MATMUL_STEAL` ships default OFF and is NOT measured; it must justify itself by measurement and may measure neutral.
