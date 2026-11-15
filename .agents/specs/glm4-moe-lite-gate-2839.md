# GLM-4.7-Flash — the only vs-vLLM assertion cannot fail, and the licensed bar is not the one asserted

**Row:** `MODEL-TEXT-GLM4-MOE-LITE-GATE-2839` — new row, `ACTIVE`.
**Issue:** [#2839](https://github.com/mudler/vllm.cpp/issues/2839).
**Date:** 2026-09-04. **Base:** `3e246b34f`.
**Predecessor row:** `MODEL-TEXT-glm4-moe-lite-glm4-moe-lite-for-causal-lm`.

## Scope

`tests/vllm/models/test_glm4_moe_lite_paged_engine.cpp` is the SACRED
correctness gate for `Glm4MoeLiteForCausalLM`. This row makes it assert the bar
`CLAUDE.md` §Gates licenses, and makes the committed artifacts gateable without
the 31.2B checkpoint.

Out of scope: the forward itself, regenerating any golden (both need a GPU and a
`zai-org/GLM-4.7-Flash` snapshot, and neither is present on this host), and the
`Glm4MoeLiteForCausalLM` speed axis.

## What is true at `3e246b34f`

Three defects, all measured rather than argued.

**D1 — the only vs-vLLM assertion has no failure mode.** The gate reads
`neartie_gap_mnats.npy` and tests `mn > kNearTieMnats` with
`kNearTieMnats = 500`. The committed array is `(8,16) int32`, `min 0 max 0`, all
128 values exactly `0`, so `prompt_ok` is true for all eight prompts
unconditionally and `CHECK_MESSAGE(prompt_ok, ...)` cannot trip.

**D2 — the licensed bar is STRICT, and nothing asserts it.** The gate computes
vLLM's own self-determinism from `greedy_dist.npy` over K=5 and `CHECK`s that it
is zero, then prints `DETERMINISTIC -> the STRICT bar is well-posed` — and
proceeds to apply the near-tie band anyway. `CLAUDE.md` §Gates permits a
distributional gate "only when the oracle's greedy decode is non-deterministic".
Against the committed `greedy_ids.npy`, `our_ids.npy` matches **69 of 128
positions (53.9%)** and **1 of 8 prompts exactly**, per-prompt mismatches
`[2, 6, 6, 12, 12, 11, 10, 0]`.

**D3 — the case is a skip wearing a pass.** With no checkpoint under
`~/.cache/huggingface/hub/`, the body emits `MESSAGE("SKIP ...")` and `return`s.
doctest counts that as a **passed** test case with zero assertions, so on every
host without a 31.2B snapshot — which is every CI runner — the SACRED gate
reports a pass having measured nothing. The same shape hides the second early
`return` (goldens absent) and the third (teacher-forced goldens absent), and both
of those conditions are now impossible: all five artifacts are committed.

The header's own definition of a near-tie has a second conjunct, "AND our token
is inside vLLM's top-K", which is not implemented anywhere: `greedy_dist.npy` is
read only for the self-determinism count.

## The artifact is very likely not a measurement, and this row does not gate on that

At a prompt's FIRST divergent position our prefix is identical to the oracle's,
so vLLM's teacher-forced argmax there IS `greedy_ids[i,j]` by construction and
our token is a different one. The gap
`max(0, arg_lp - our_lp)` must then be strictly positive. It is `0` at all seven
first divergences (prompts 0..6, positions 14, 6, 6, 4, 4, 4, 6). The script also
writes a `99_999_000` sentinel when our token falls outside vLLM's top-20, and
there is not one. So every one of our divergent tokens was inside vLLM's top-20
AND within 0.0005 nats of its argmax, seven times out of seven at the one
position where the prefixes provably agree.

This is recorded as a finding, not gated. A gate that asserts "the gap at a first
divergence is positive" could in principle be defeated by genuine sub-millinat
ties, and the STRICT bar below settles the same question without that risk.

## Design

**A — the skip becomes a skip.** The checkpoint predicate moves into
`doctest::skip(...)`, so an absent snapshot is reported as `skipped` rather than
`passed`. The two golden-absence `return`s become `REQUIRE`s, because those
artifacts are committed and their absence is a defect rather than an
environment. The `VT_GLM_DUMP_IDS` bootstrap arm stays, because it is an explicit
opt-in that writes an artifact rather than a silent pass.

**B — the bar follows the measurement that selects it.** `greedy_dist.npy`
becomes REQUIRED, and its verdict decides the bar in code rather than in a
`MESSAGE`. Zero multi-valued cells means STRICT: every token must equal the
oracle's. A non-zero count is the only thing that admits the near-tie band, and
that arm now implements BOTH conjuncts the file's own header states — within
`kNearTieMnats`, AND our token inside vLLM's top-K from `greedy_dist.npy`.

**C — the committed artifacts become gateable with no checkpoint.** A new case
reads only `tests/parity/goldens/glm4_moe_lite_greedy/`, re-derives the
self-determinism verdict, and applies the licensed bar to `our_ids.npy`, which
the engine case already REQUIREs the live engine to reproduce token for token
(`:283-287`). It needs no 31.2B snapshot and no GPU, so it runs everywhere the
suite runs.

## Tests and expected verdicts

- **T1** (checkpoint-free) the committed `greedy_dist.npy` has zero multi-valued
  cells. Expected GREEN — this is what licenses STRICT.
- **T2** (checkpoint-free) under STRICT, `our_ids.npy` equals `greedy_ids.npy` at
  every position. **Expected RED at 69/128.** This is the row's result: the gate
  fails because it finally measures something.
- **T3** (checkpoint-gated) the engine case applies the same bar. Not runnable
  here; no `zai-org/GLM-4.7-Flash` snapshot exists on this host and none is on
  the NAS.

## Gates

```sh
cmake --build build -j 3 --target test_glm4_moe_lite_paged_engine
./build/tests/test_glm4_moe_lite_paged_engine
scripts/agent-preflight.sh
```

## Stop conditions

Stop and report `NEEDS_DECISION` before weakening T2. The two admissible exits
are a forward repair that makes 128/128 true, or a re-capture of
`neartie_gap_mnats.npy` on the checkpoint plus an explicit ratification argument
for a distributional bar over a deterministic oracle. Neither is available to a
host with no GPU and no snapshot, and neither is this row's work.

## 2026-11-12 — the gap array's resolution cannot carry this argument

Recorded from the committed artifacts alone: no GPU, no checkpoint, no rebuild.
See ISSUE-LOCAL-01M7GNV6F01WRYT1RCDS766BG7.

`neartie_gap_mnats.npy` is `int32` MILLINATS. `scripts/glm4-moe-lite-neartie-gap.py:92`
floors every cell at zero (`max(0.0, arg_lp - lp(our_token))`) and `:93` rounds
`gap * 1000`. A committed `0` therefore means "under half a millinat", which is
not the same claim as "exactly tied". The section above reads the all-zero array
as the stronger one, so it proves less than it says.

Four measurements over the committed files, each one command and no model:

- `greedy_ids == greedy_dist[:,:,0]` is true, and all five K channels are
  byte-identical, so the K=5 self-determinism result stands unchanged.
- `neartie_gap_mnats` holds exactly ONE distinct value, `0`, across all 128 cells.
- No cell carries the `99_999_000` sentinel, so OUR token sat inside vLLM's
  top-20 at every position, including all seven first divergences.
- Prompt 7 is exact at 16/16. The others agree for 4 to 14 tokens before
  diverging, at positions `[14, 6, 6, 4, 4, 4, 6]`.

The last two are the finding. At a first divergence our prefix IS the oracle's,
so vLLM's teacher-forced argmax there is `greedy_ids[i,j]` by construction and
`gap` must be positive. Measured, `gap` is under half a millinat and the argmax
it was measured against is OUR token. That is vLLM's greedy-decode path and its
prefill `prompt_logprobs` path disagreeing with each other at sub-millinat
resolution, not evidence about this forward.

Two consequences, both of which change the order of the campaign:

1. The STRICT bar was selected by five runs on ONE numeric path, so that
   experiment cannot detect prefill-vs-decode disagreement and does not, by
   itself, establish that the forward is what is wrong.
2. A structural defect — wrong RoPE, wrong expert set, wrong norm epsilon —
   diverges at the first token. Agreement for 4 to 14 tokens with one exact
   prompt is the shape of accumulated bf16 noise.

The next measurements are therefore not a kernel bisection:

- **K1.** Re-capture with the gap stored in micro-nats or as raw float. The
  half-millinat bucket is the only thing between "tied" and "0.0004 behind".
- **K2.** Teacher-force vLLM twice on the same prompt and prefix: greedy decode
  against prefill `prompt_logprobs`, and at `max_num_seqs` 1 and 4. If the argmax
  flips, the exit is a ratified distributional bar over the sub-millinat band,
  argued from that flip rather than from convenience.

Only if K1 and K2 show real gaps above `kNearTieMnats` does the forward become
the suspect. Then the two paths this port actually exercises are the `noaux_tc`
router — `p.router_dtype_is_f32` (`glm4_moe_lite_registry.cpp:75`), top-4-of-64
selection with `e_score_correction_bias` (`deepseek_v2.cpp:376`) — and the MLA
`q_lora` branch (`q_a_layernorm`, `deepseek_v2.cpp:269`). This port carries no
DSA indexer, so that is not a suspect.

## Owed

- O1. T2 is RED as landed, and it must stay red until one of the two exits
  above. It is the honest state of `Glm4MoeLiteForCausalLM` and it is why
  `docs/FEATURES.md` already carries `the shipped 8/8 is not a measurement`.
- O2. The engine case (T3) is unrun. It compiles here; nothing on this host can
  execute it. Owed to whoever next holds a GPU and the snapshot.
- O4. The zero-assertion skip is a REPOSITORY-WIDE shape, not this file's.
  `MESSAGE(...SKIP...); return;` occurs at 228 sites in 72 files under `tests/`,
  and `test_deepseek_v2_paged_engine` — the SACRED DeepSeek-V2 gate — reports
  `1 passed | 0 failed`, `assertions: 0`, `Status: SUCCESS!` on this host at
  `3f12c617d`. Repairing 228 sites is a campaign, so it is filed as
  [#2918](https://github.com/mudler/vllm.cpp/issues/2918) and owned here rather
  than fixed in this flow. This row repairs the one instance it touches.
- O3. `docs/USAGE.md` carries no checkpoint row for `Glm4MoeLiteForCausalLM` —
  no revision and no sha256 — which `CLAUDE.md` §"Say which weights, and from
  where" requires. Not repaired here because the values must be read off the
  artifact this host does not have.
- O5. The resolution above and the two measurements it names (K1, K2) were
  recorded on 2026-11-12 without a GPU. Both need the snapshot and a lease, so the
  exit is still owned here and still needs the decision this row already names.
