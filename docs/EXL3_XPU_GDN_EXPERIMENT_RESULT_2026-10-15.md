# B70 speculative GDN experiments, 15 October 2026

Wave2 initially rejected three bounded experiments under the bit-exact contract.
The developer subsequently authorized investigating small rounding differences
for the faster subgroup candidate. The developer then authorized adopting the subgroup candidate and deferring
the longer quality campaign. The guarded subgroup path now defaults ON;
`VT_XPU_GDN_SPEC_SUBGROUP=0` retains the diagnostic control. No existing exact
assertion or numerical tolerance is changed; tests specifically qualifying the
original SLM route select that route explicitly.

Before reopening, the restored server was byte-identical to the qualified Wave1
binary (`99d4c6f6a93cb374f62af0d36c3956f65a5a36bfd9f72c82696433c9257a2660`).
Those restored exact snapshot/graph controls passed2 cases /990 assertions.

## Actual input and timing contract

An external diagnostic driver captured the first actual eager C1/MTP3 verifier
invocation after the fixed4096-token prefill. It delegates to the unchanged
native operator and saves inputs before execution and outputs afterwards.
Capture is diagnostic only; its provider is never installed in serving.

The seam is normalized FP16 Q/K `[4,16,128]`, FP16 V/output `[4,48,128]`,
derived FP32 g/beta `[4,48]`, and full FP32 state `[4,48,128,128]` (12 MiB).
Offsets are `[0,4]`, slots `[0,1,2,3]`, accepted selector1, and scale
0.0883883461356163. The initial slot contains the real nonzero prefill state.
No second normalization/sigmoid or FP16 state shortcut is used.

Each replay restores the entire captured initial state and output before timing.
Balanced trials record both the complete operator and the actual recurrence
event. Output and every snapshot byte are checked after timing. Input transfers,
state restoration and result downloads are excluded from these kernel/operator
timings. These isolated diagnostics are not end-to-end serving measurements.

| Experiment | Control kernel µs | Candidate kernel µs | Complete operator ms, control/candidate | Decision |
|---|---:|---:|---:|---|
| Local snapshot-retirement fence | 103.073 | 103.021 | 0.12896 /0.12875 | No measurable benefit |
| 32 versus64 value rows/WG | 103.021 | 103.438 | 0.12921 /0.12885 | No measurable benefit |
| Four keys/lane, subgroup reductions | 103.125 | 23.281 | 0.12929 /0.04972 | Exact numerical contract fails |

Both order-preserving candidates pass1212 focused assertions, including all
selectors, null/permuted/repeated slots, aliasing, graph metadata and12-packet
own-state continuation. Actual replay bytes also match. They were rejected
before serving measurements because there is no demonstrated local speed gain.

## Faster subgroup candidate: initial exact-contract rejection

The experimental subgroup path uses256-thread workgroups, one value per
32-lane subgroup, four FP32 state elements per lane, and every required snapshot.
It changes the association of dot-product sums. The existing unmodified exact
SLM tests fail:2 cases,84 passed /2 failed assertions. No test was relaxed.

On the actual invocation:

- FP16 output:6 of24,576 elements differ, each by1 ULP; max absolute error
  4.76837158203125e-7, relative L2 3.17019e-8.
- FP32 snapshots:1,124,353 of3,145,728 elements differ; max absolute error
  7.62939453125e-6, relative L2 7.98219e-8.
- Snapshot relative L2 values are6.09e-8,7.53e-8,8.53e-8 and9.11e-8 across
  the four sequentially produced states. No control state is injected between
  these tokens.
- The first mismatch is1 FP32 ULP. The largest ULP distance is1,015,808 near
  cancellation: reference−6.359186954796314e-9 versus−5.908077582716942e-9.
  Large ULP counts near zero must not be read as large absolute errors.

At this initial gate the candidate was rejected before multi-cycle or serving
tests. The measured error is not a model answer-error rate. The follow-up below
records the subsequent investigation requested by the developer.

## Intel resource evidence and remaining scope

A separate Level Zero tracer queries kernel properties through API callbacks.
Both typed controls and subgroup candidates report0 private and0 spill bytes,
required subgroup32. The query does not expose GRF counts or dynamic SLM usage;
zero reported static local memory is not proof of zero dynamic SLM. Source
local-accessor reservations are32 KiB/16 KiB for64/32-row candidates; the subgroup
experiment has no local accessor. Instrumented runs are excluded from timing
selection.

External evidence lives under `pro-optimization-2026-10-15/wave2` in the local
recovery artifact root: capture/replay sources, complete raw inputs/snapshots,
hash manifest, exact build/runtime commands, numerical analysis, resource logs,
and frozen source/binaries for each rejected variant. Setup/launcher failures
are retained separately. The final restored-control test passes, and production
remains stopped.

## Initial developer-directed subgroup follow-up

The unchanged CPU comparison and teacher-forced MTP1/MTP3 committed-prefix tests
pass with the subgroup candidate:2 cases /356 assertions. Their existing
numerical tolerances remain unchanged. This is distinct from the failed bit-exact
SLM comparison above.

An external continuation replay runs256 verifier calls per selector pattern,
1024 evaluated token rows. Each arm starts once from the same real nonzero state
and then carries its own state; the control is never injected into the candidate.
It exercises selectors1,2,3,4 and a1–4 cycle, including rejection of uncommitted
snapshots. Both arms reuse the same captured normalized operands on every call.
This is a bounded arithmetic/selector diagnostic, not1024 naturally generated
tokens or a broad model-quality qualification.

At checkpoints after1,4,16,64 and256 calls:

- Every result is finite; immutable input bytes remain unchanged.
- The first control output and all four snapshots match the original real
  capture byte-for-byte in every pattern.
- The largest FP16 output distance is1 ULP. Largest absolute output distance
  across these checkpoints is0.0009765625, illustrating the dependence of a
  FP16 step on value magnitude. Largest relative output L2 is5.12528e-5.
- At256 calls the relative snapshot L2 ranges from7.24707e-7 to2.27941e-6;
  largest absolute snapshot difference is0.0001926422119140625.
- At256 calls the relative output L2 ranges from1.64043e-8 to5.48394e-7.

These are checkpoint measurements, not an error bound for arbitrary inputs or
long generations. Full final output/state pairs and the executable diagnostic
source are retained under `wave2/subgroup-qualification/sequence-v1`. The
existing exact SLM tests are preserved. At that checkpoint the subgroup default remained off pending
serving and wider state/graph qualification. Long C1 attention remains explicitly
disabled in both serving comparison arms to isolate the GDN change.


The first trace-free serving pair (one fresh process per arm, C1/P4096/O256/MTP3,
four measured responses, final three warm medians) gives:

| Metric | Typed SLM control | Subgroup | Change |
|---|---:|---:|---:|
| Decode tokens/s | 58.9064 | 63.0011 | +6.95% |
| Decode seconds | 4.32890 | 4.04755 | -6.50% |
| Prefill seconds | 1.84591 | 1.86814 | +1.20% |
| Total server seconds | 6.17588 | 5.91475 | -4.23% |

All four public token representations, decoded texts and usage records agree
between arms and with the frozen same-engine baseline. Prompt IDs and all MTP
acceptance counters agree (335 draft steps,1005 draft tokens,730 accepted,
including primer). This is an encouraging pilot, not the three balanced blocks
required for a performance selection. It does not establish quality on different
prompts or contexts. At that checkpoint C4, null/alias slots and graph replay were still pending.
The following adoption step closes those focused ownership checks.


## Approved default adoption

On15 October the developer approved integrating the subgroup path now and doing
longer quality testing later. The default is shared between dispatch and the
Qwen dense XPU graph key. Switching its diagnostic override retires the captured
policy. The previous typed SLM route remains the fallback for ineligible operands
or devices without WG256. The long-context attention experiment stays off.

The added default-selection test fails at the prior default:16 passed /1 failed
assertion, first differing snapshot byte at3145736. With the new default:

- Focused GPU:7 cases /50,948 assertions pass. This includes the unchanged
  numerical CPU/teacher-forced MTP prefix checks and the original exact SLM
  tests, explicitly selecting the original route.
- New subgroup checks compare the default with explicit1 and preserve the
  explicit0 control, including F32 fallback. C1 and ragged C4 exercise all four
  accepted selectors, null/repeated/permuted slots, untouched poisoned storage,
  output aliasing and eager-versus-graph bytes. Replay sees changed live metadata
  and rejects invalid metadata without modifying output/state.
- Host graph-policy checks:2 cases /94 assertions pass; changing the subgroup
  setting changes the graph key.
- The existing Wave3 planner test now includes `<initializer_list>` explicitly
  and passes1 case /82 assertions. An initial wrong filter selected no tests;
  that result is retained and is not counted as a passed test.
- The server builds successfully. No new end-to-end timing is claimed for this
  default-adoption build; the preceding pilot measured the same subgroup math
  with explicit1. The profile label is now `gdn_spec_decode_subgroup`.

The longer natural-generation/quality campaign and balanced serving confirmation
remain open by developer direction. Neither answer quality nor arbitrary-context
parity follows from these bounded checks. Production remains stopped.
