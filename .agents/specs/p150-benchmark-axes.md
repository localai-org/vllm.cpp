# SPEC — P150 benchmark axes: the gated container for the reference map

- Owning row: `QWEN38-DFLASH2-STAGE-C`. Issue: ISSUE-LOCAL-01M82MQAJYM6ZAAZC7APD8063T.
- Input: `docs/bench-evidence/p150-reference-map-20261122/reference-map.md`
  (PR #3593). That document is analysis; this spec is the plan that turns its
  axis table into publishable, gated numbers.
- Developer authorization: "draft the benchmark row spec from the map".
  Spec-first applies: this spec lands before any benchmark ID or number.
- Status: SPEC ONLY. No `docs/BENCHMARKS.md` row, no `docs/benchmarks/*.md`
  detail page, and no benchmark number is added by this change. Those are owed
  by the benchmark unit when an axis actually reaches PUBLISHABLE.

## Trigger / problem

The developer asked to continue until benchmarks can be compared with other
implementations. A verified local copy of
`Lottolabs/Inkling-2-Mini-TT-Mixed-BFP4-BFP8-P150` @ `b0ad7060…` now exists
(482/482 SHA256 OK) and the reference map records what its numbers mean. What
is missing is the container: which benchmark IDs will be published, what an
identical workload is for each axis, which correctness gate must be EXACT
before a throughput claim, and what makes an axis PENDING, FAILING or
PUBLISHABLE. Without that, numbers can be published with an invalid denominator
or before their gate closes.

## Upstream chain and reference status

- The binding denominator is the pinned vLLM production configuration,
  quant-matched, never `--enforce-eager`, per AGENTS.md §Gates and
  `.agents/benchmarking.md` §The denominator. Current vLLM parity pin:
  `a7c23ac96d` (0.3.0.dev267) from `.agents/upstream-sync.md`.
- The Lottolabs package is a **published reference**, not an oracle. It is not
  in the AGENTS.md oracle registry and has no oracle file, so its numbers can
  never discharge a vLLM correctness gate. A comparison to it is either a
  reproduction of its exact recipe on its shipped runtime image (a measured
  arm) or a citation of its published numbers labelled as reference-only.
- Where vLLM has no Inkling-2 path, the comparison is reference-only and the
  vLLM axis stays an OPEN GAP, not a waived gate. A secondary oracle may be
  used only if registered and pinned in `.agents/oracles/` first.

## Benchmark IDs this unit will publish

One detail page per subject, one evidence detail file per axis. Nothing is
created until its first axis reaches PUBLISHABLE.

| Benchmark ID | Detail page | Subject | Axis detail files |
|---|---|---|---|
| `p150-inkling2-native` | `docs/benchmarks/p150-inkling2-native.md` | Inkling-2-Mini on one P150 against the Lottolabs GPTQ-BFP4 package | A1-A3, A6-A9 under `docs/bench-evidence/<run-id>/` |
| `p150-dflash2-decode` | `docs/benchmarks/p150-dflash2-decode.md` | Qwen3.8-27B DFlash2 decode/spec on the P150, drafter ON/OFF and DNR arms | A1-A5, A8, A10 |
| `p150-olala-bf16` | `docs/benchmarks/p150-olala-bf16.md` | olala bf16 forward after the W2e arms, token gate already EXACT | A1-A3, A8 |

Publishing an ID owns one `docs/BENCHMARKS.md` index row and its detail file;
`scripts/check-benchmark-index.py` requires the ID to equal the file stem and
every detail file to be indexed exactly once. Public document triggers follow
AGENTS.md §Public documents: `FEATURES.md` changes only when the shipped
capability changes, `USAGE.md` only when a command or config key changes.

## Identical-workload definition

Every arm of an axis must use the same model artifact and hash, prompts and
token counts, output length, batch size, concurrency, sampling parameters,
context profile, KV dtype profile, prefill chunk schedule, and trace state.
The two things most likely to break identity here:

1. **Trace vs eager.** W28's struck leg was fully eager (`capture_hit = 0`).
   An eager leg must never be compared with a traced Lottolabs number. Each
   result records `trace=on|off`, bucket boundaries and the position-to-bucket
   rule.
2. **Arms must BE arms.** Prefer a same-binary runtime switch
   (`VT_DFLASH_DRAFT`, `VT_TT_WEIGHT_RESIDENCY`, `VT_TT_BFP8_RESTAGE`). If two
   artifacts are built, prove with `scripts/ab-arms-differ.py` that at least
   one behavioural control moved; equal hashes or equal call counts are
   `FATAL`, not a result.

## Axis plan

Columns: **Precondition** uses `BLOCKED-ON-W33`, `PUBLISHABLE-AFTER-W2E-ARMS`
or `W2E-EQUIVALENT-ARMS` as the state that must close before a number may be
published. W33 is the stage-C wave that resolves the W31 writer mechanism;
W2e is the olala full-block-forward gate wave after W2d assembly.

| Axis | Metric / identical workload | Denominator | Precondition | PENDING | FAILING | PUBLISHABLE |
|---|---|---|---|---|---|---|
| A1 traced decode step / bucket ladder | Traced ms per token at 4096, 8192, 16384, 32768, 65536, 131072; same prompt, bucket rule, KV profile, no readback differences | vLLM production config at the same context; Lottolabs 68.4-94.1 ms is reference-only | DFlash2 `BLOCKED-ON-W33`; olala `PUBLISHABLE-AFTER-W2E-ARMS`; inkling2 `W2E-EQUIVALENT-ARMS` | No bucket-aware traced path, or trace/eager mismatch | Token gate not EXACT once the precondition closes | Token EXACT, same-binary A/B, idle-host 2-3 legs, values and ratios recorded |
| A2 decode vs context ladder | Aggregate tok/s and TTFT at 0/4K/16K/32K/64K/128K; batch 1, same lmx recipe | vLLM production, quant-matched; Lottolabs matrix is reference-only | same as A1 | Model cannot fit a required context (open capacity gap, e.g. Q6_K_XL drafter-ON) | Token gate fails on a context that does run | All runnable contexts pass token gate; blocked cells stay named gaps |
| A3 TTFT / prefill | TTFT at 128/2048/8192/32768 and prefill tok/s; same chunk schedule | vLLM production, same prompt and chunking | same as A1 | No P150 prefill harness for the model | Non-deterministic prefill with no named cause | Token EXACT at the continuation boundary; noise band calibrated |
| A4 speculative speedup + acceptance | tok/s, accepted draft tokens/step, both drafter arms, both DNR arms | vLLM production spec decode; Lottolabs 1.24x is reference-only | DFlash2 `BLOCKED-ON-W33` + acceptance gates; olala `PUBLISHABLE-AFTER-W2E-ARMS` | W33 writer mechanism unresolved; acceptance gate not green | Acceptance gate red or token gate not EXACT | Both arms on one binary, acceptance reported separately from tok/s |
| A5 token identity spec vs plain | Tokens and logits on the same golden, near-tie margins recorded | Golden parity test, not a throughput ratio | same as A4 | Golden not available for the arm | Any unexplained divergence | Bitwise or ratified tie-band identity with margins |
| A6 sampled decoding exactness | Seeded T/top_k/top_p grid, spec on/off, distributional test | vLLM sampler semantics; Lottolabs device sampling is reference-only | `W2E-EQUIVALENT-ARMS` | Device full-vocab sampler not reachable | Seeded runs differ or distribution fails | Distribution and seeded identity pass on the same grid |
| A7 long context | Passkey 3/3 at 131K/262K/524K/1M, TTFT, decode tok/s | vLLM production at the same context or explicit OPEN GAP | `W2E-EQUIVALENT-ARMS` | Artifact or memory plan cannot serve 1M | Passkey misses | Passkeys pass with the control prompt before == after |
| A8 weight residency memory | Device bytes by dtype/expert role, embedding placement, peak RSS | Same weights, same batch/context; Lottolabs package is a different provenance class | same as A1 | Residency arm unavailable | Bytes exceed the card budget without a named plan | Dtype, calibration and placement stated; values and ratios recorded |
| A9 quality vs BF16 | KL / top-1 / ΔPPL on the identical eval set | CPU BF16 reference of the same checkpoint | `W2E-EQUIVALENT-ARMS` | Eval set or reference unavailable | KL above the ratified band | Same eval set; reference noise floor reported with the result |
| A10 suite-state correctness | Writer/RAC/PA strike absence under the full suite, not a headline number | W28/W30/W31 evidence arms on the same binary | DFlash2 `BLOCKED-ON-W33`; both PR #3592 falsification experiments | W32/W33 instrument not run | Any strike, hang, or non-empty `new_low_blocks` | All arms clean across the full suite; this is a gate, not a published score |

No ceiling may be recorded. An apparent same-architecture limit is an
unresolved implementation difference and names the next traceable hypothesis.

## Correctness precondition (binding)

Token gate EXACT (or an explicitly ratified distributional gate) is established
**before** any throughput or latency number is claimed for that axis. Specific
states:

- **olala bf16**: token gate already EXACT; its axes are
  `PUBLISHABLE-AFTER-W2E-ARMS` — W2e must close its forward arms first.
- **DFlash2 axes**: `BLOCKED-ON-W33` — the W31 writer mechanism must be root
  caused and the acceptance gates green. Until then W30's ~3.73x is a repair
  multiple, not a performance result.
- **inkling2 axes**: `W2E-EQUIVALENT-ARMS` — the parse-only loader refuses at
  the forward boundary today; a full forward with the same arm structure as
  W2e is required before any number.

## Idle-host A/B and same-binary discipline

- One P150 job at a time. Claim the device with `rc run` / `rc hold`; never
  `ssh` to a fleet device or use a second mutex. On a non-fleet GPU take
  `${GPU_LOCK:-$HOME/gpu.lock}`.
- Calibrate the noise band from repeated identical legs before interpreting a
  delta; paired, order-alternated A/B with a majority rule. Discard a cold leg
  only for a named cause. Reproduce the accepted result 2-3x on an idle host.
- Record clock state (including boot id and throttle reasons), contention
  state, build and run recipe, revisions, model hashes, and the same-binary A/B
  with at least one behavioural control. Follow `.agents/environment.md` for
  the fleet and the local lock; resolve values from the shared `.env`, never
  from a document that names another developer's path.
- Evidence is flat: `docs/bench-evidence/<run-id>/<file>.{json,log,csv,txt,md}`.
  Binary dumps stay untracked. No measurement is stored inside a shared record
  file.

## Work breakdown

1. W33 resolves the writer; close both PR #3592 falsification experiments.
2. W2e closes the olala forward arms; define the equivalent inkling2 arms.
3. One benchmark row per ID above, claiming `row/<ROW-ID>` in its own
   worktree, spec already committed here. Each row ports the applicable
   upstream/serving tests, runs the axes, and publishes only PUBLISHABLE cells.
4. Publication change owns `docs/BENCHMARKS.md` index row and the
   `docs/benchmarks/<id>.md` detail page in the same PR as the number.

## Dependencies and risks

- P150 access under an `rc` lease; the Lottolabs runtime image is 5.3 GiB and
  needs Docker plus hugepage mounts, so a measured reference arm is a real
  cost, not a citation.
- Model artifacts: pinned HF revision `b0ad7060…` for the reference; the exact
  GGUF/safetensors hash for our side. No artifact may be substituted silently.
- The reference package is not an oracle; a missing vLLM path is an OPEN GAP.
- Near-tie divergence can move tokens without moving speed; A5 must be its own
  gate.

## Stop conditions

Stop and report `NEEDS_CONTEXT` if the device lease, model hash or denominator
is unavailable; `NEEDS_DECISION` if a correctness gate cannot be ratified.
Never stop at a correctable failing test. An axis below its floor stays an open
gap and the next traceable hypothesis is named; it is not rounded into a pass.

## Git integration

One PR for spec and implementation per benchmark row. Push only to `mine` on
`row/*`; never force-push `main`. Commit trailers and the validated PR body
follow AGENTS.md. This records-only PR carries the issue and this spec first.

## Now

SPEC — no benchmark published. A1-A10 are planned; DFlash2 axes are
`BLOCKED-ON-W33`, olala bf16 is `PUBLISHABLE-AFTER-W2E-ARMS`, inkling2 axes
need `W2E-EQUIVALENT-ARMS`. Reference map and verified package are the inputs.

## Owed

- W33 writer mechanism; W2e olala arms; inkling2 equivalent arms.
- A registered and pinned oracle if any axis needs a secondary denominator.
- Public benchmark IDs and detail pages when the first axis reaches
  PUBLISHABLE.
