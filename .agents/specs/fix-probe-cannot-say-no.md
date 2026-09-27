# FIX-PROBE-CANNOT-SAY-NO — the second always-true `TensorExists`, and the shape that produced both

**Issue:** [#1258](https://github.com/mudler/vllm.cpp/issues/1258).
**Predecessors:** [#1256](https://github.com/mudler/vllm.cpp/issues/1256) named two
stubs; [#1257](https://github.com/mudler/vllm.cpp/pull/1257) (`MODEL-FP8-BLOCK-LINEAR`,
merged `281b4bc76`) fixed one.
**Kind:** loader repair plus one shared guard. No kernel, no forward-pass numerics,
no measurement.

## Now

`origin/main` at `281b4bc76` carries exactly one always-true presence probe:

```
$ grep -n 'return true; }' src/vllm/model_executor/models/qwen3_5_dense_weights.cpp
794:  const TensorExists has = [] (const std::string&) { return true; };
```

It is the resolver-only overload of `LoadQwen3_5DenseLayer`. Its live caller is
`tests/parity/test_op_parity.cpp:1042`, which replays
`tests/parity/goldens/qwen36_gdn_layer_27b` — `"layer_type": "linear_attention"` —
through it. That routes to `LoadGdnDense`, whose first act is
`IsFp8BlockProjection(has, la + "in_proj_qkv", ...)`, so the stub reports a
`weight_scale_inv` the checkpoint has never carried and the load is refused.

The golden is `SKIP`ped when the pinned `unsloth/Qwen3.6-27B-NVFP4` snapshot is
absent, which is the whole reason no CPU lane has shown it. The seam is broken for
its documented purpose either way: `qwen3_5_dense.h:204-215` still describes the
constant `true` as a deliberate design, and since #1189 M3 that design cannot load
a non-block checkpoint at all.

## Scope

In scope:

- `dense_loaders::ProbeThroughResolver` — one implementation of "ask the
  resolver", used by both resolver-only seams. #1257's inline copy collapses onto
  it, so the tree carries one copy of a subtle `try`/`catch` rather than two.
- `dense_loaders::CheckProbeCanAnswerNo` — refuse, by name, a presence predicate
  that answers `true` for a name no checkpoint carries.
- The three sites in `qwen3_5_dense_weights.cpp` where a probe is built or
  received before a guard consults it.
- Red-first cases in `tests/vllm/models/test_qwen27_dense_forward.cpp`.
- The [#1258](https://github.com/mudler/vllm.cpp/issues/1258) row appended to
  [`issue-index.md`](../completed/issue-index.md).

Out of scope, each for a stated reason:

- `src/vllm/model_executor/models/qwen3_weights.cpp`. Audited, and it does not
  carry the defect. See below.
- `src/vllm/model_executor/models/laguna_weights.cpp:408`. Map-backed, same
  audit.
- The routing decisions themselves. `IsFp8BlockProjection`,
  `IsNvfp4Projection` and the arm order are correct and are not touched. #1256's
  guard is the thing that found this; weakening it was never on the table.
- Any change to `TensorExists`'s type. Argued and rejected below.

## The audit #1256 asked for, answered rather than assumed

Every line number in this section is read at `281b4bc76`, this row's base.

| File | Probes | Shape | Verdict |
|---|---|---|---|
| `qwen3_5_dense_weights.cpp` | 3 | one map-backed (`:820`), one resolver-derived (`:689`, #1257), one **constant `true`** (`:794`) | the row's work |
| `qwen3_weights.cpp` | 1, at `:151` | `where.find(name) != where.end()` | clean — no public resolver-only seam exists, so there is nowhere for a stub to live |
| `laguna_weights.cpp` | 1, at `:408` | `where->find(name) != where->end()` | clean |

Repo-wide, `grep -rn '\[] (const std::string&) { return true; }' src include tests`
returns the one line above and nothing else.

## The general question, and why this answer

Two instances of one defect, in one file, both found by one guard. The type is a
bare `std::function<bool(const std::string&)>`, so a constant satisfies it. Three
remedies were considered.

**Rejected — make `TensorExists` a distinct type with only safe factories.** It
does prevent construction from a constant, and it is the only option that is
airtight at compile time. It is rejected on measured cost against measured reach:
18 spellings of `std::function<bool(const std::string&)>` across 8 files, 12
`TensorExists&` parameters, two headers and three test fixtures. That buys
hardening for exactly one predicate in a tree that has many, and it forecloses a
legitimate future probe shape (a GGUF name list, a deliberately restricted test
probe) behind a new factory each time. The defect is not "this type is too weak";
it is "a predicate was allowed to be incapable of its negative answer".

**Rejected — delete the resolver-only overloads so every caller supplies a real
probe.** Cheap: two call sites. It is rejected because #1257 has just reaffirmed
the opposite contract one function away, deleting a seam teaches the next probe
nothing, and both callers would immediately hand-write the probe the helper below
already owns — three copies instead of one.

**Chosen — require a probe to prove it can answer `no`.** `CheckProbeCanAnswerNo`
asks for a name no checkpoint carries and refuses a `true`. It is independent of
how the probe was built, so it covers a future stub, a future map that was
populated wrong, and a probe that a refactor detaches from its data. It is a
`VT_CHECK` and not an `assert` on purpose: Release defines `NDEBUG`, and a guard
that evaporates in the configuration everything ships in is not a guard. Cost is
one map miss per layer load.

It cannot catch a probe that answers `true` for everything *except* the sentinel.
That is a contrived adversary, not the failure observed twice, and it is recorded
here rather than defended against.

## Design

`include/vllm/model_executor/models/dense_weight_loaders.h`, beside the other
shared loader helpers:

- `kAbsentProbeSentinel` — the name asked about.
- `ProbeThroughResolver(get)` — `TensorResolver` returns a **reference**, so
  absence has exactly one representation in its contract: it throws.
  `SafetensorsFile::Get` documents that (`safetensors_reader.h:52-53`) and every
  fixture resolver in the tree mirrors it. The resolver is captured **by value**,
  not by reference, so a returned probe cannot outlive its resolver — the
  by-reference form is the next subtle thing to get wrong in a helper written
  because a subtle thing was got wrong.
- `CheckProbeCanAnswerNo(has, seam)` — `VT_CHECK` naming the seam, the sentinel
  and both issues.

Three call sites in `qwen3_5_dense_weights.cpp`: after the map-backed probe in
`LoadQwen3_5Dense`, after the derived probe in `LoadQwen3_5DenseGdn`, and on
entry to the explicit-probe `LoadQwen3_5DenseLayer`, which is the one overload
every other path funnels through.

## Risks

| Risk | Control |
|---|---|
| the derived probe changes routing for the 27B golden | the golden's projections are compressed-tensors, so `has(weight_packed)` is genuinely `true`; T1 pins bf16 routing on a synthetic layer and the existing 9 cases pin the rest |
| a resolver signals absence by something other than an exception | `catch (const std::exception&)` mirrors #1257; the contract is documented at `safetensors_reader.h:52-53` and every fixture resolver in the tree throws |
| the returned probe outlives its resolver | captured by value |
| the sentinel collides with a real tensor name | it is not a valid module path and carries a `__vllm_cpp` prefix; a collision refuses loudly rather than silently |
| the new guard is dead code | T2 mutates the probe back to a constant and asserts the refusal by name |

## Tests

All in `tests/vllm/models/test_qwen27_dense_forward.cpp`, already registered at
`tests/CMakeLists.txt:449` and re-run under `VT_GLUE_FUSE=0` at `:461`.

- **T1** — the resolver-only `LoadQwen3_5DenseLayer` loads an all-BF16
  `linear_attention` layer without refusing, and routes bf16: `in_proj_qkvz` and
  `gate_up_proj` populated, every fp8-block slot empty. RED before this row with
  the #1256 message.
- **T2** — the explicit-probe overload, handed a constant-`true` predicate, is
  refused by name. This is the mutation of the chosen guard, run as a test rather
  than by hand.
- **T3** — the negative control: `ProbeThroughResolver` answers `false` for an
  absent name and `true` for a present one, and a map-backed probe passes
  `CheckProbeCanAnswerNo`.

## Gates

1. `ctest -R 'test_qwen27_dense_forward'` — both the plain and `_glue_fuse_off`
   registrations, read at `Status:`, not at `assertions:`. This bug reports
   `assertions: N | N passed | 0 failed` on a failing case because it throws
   before asserting.
2. Full `ctest` on a clean CPU build.
3. `scripts/agent-preflight.sh --fail-on-skip`, read at the verdict line.

## Owed

Nothing. The audit is complete for this predicate across `src/` and the two
rejected remedies are recorded above rather than deferred.

## Stop conditions

- If T1 is green before the change, the premise is wrong: stop and re-derive.
- If the derived probe moves any existing assertion in the 9 pre-existing cases,
  stop — the seam's routing was load-bearing in a way this spec did not predict.
