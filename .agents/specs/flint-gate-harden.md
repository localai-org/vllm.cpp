# Flint gate hardening: regression cases the wave-1 synthetic gate lacked

Row: MODEL-TEXT-FLINT
Issue: ISSUE-LOCAL-01M5SQG0389SAHK9NY9ENR4AHP
Status: ACTIVE (2026-10-25)

## Scope

Add two assertions to `tests/vllm/models/test_flint_forward.cpp` — the
synthetic forward gate — that fail on the wave-1 pre-fix code and pass on
current `main`. No production change: both defects are already fixed in
`src/vllm/model_executor/models/flint.cpp` (attn_out sized `t * hq * dh`,
embedding lookup widened with `Bf16()`). This unit pays back the tests that
would have caught them at wave 1.

## The two cases

1. BUFFER-CONTRACT (`flint: attn_out is sized for the head layout, not the
   hidden stream`): a synthetic config with `hq * dh = 16 > hidden = 4` —
   the exact shape that overflowed when `attn_out` was sized `t * h`
   (ISSUE-LOCAL-01M5Q1PS772CPTMYJ02AMVZ3Y0). The gate asserts the causal
   prefix invariant: row `i` of a `t`-token forward must equal the forward
   of the first `i+1` tokens alone, bitwise, because every per-row stage
   (RMSNorm, matmul, per-row attention over the causal prefix) is
   row-local. Under the `t*h` sizing, token 0's heads 1..3 are written past
   their slots and the tail of `attn_out` is heap garbage, so row 0 of the
   2-token forward diverges from the 1-token forward. RED on the mutation,
   GREEN on `main`.

2. EMBEDDING-DTYPE (`flint: the embedding lookup widens bf16 bits, it does
   not memcpy them`): an embedding table with bf16-exact values
   `[1.0, 0.5, 0.25, 0.125]` for the probed token, a 1-layer model whose
   attention (q=v=0) and dense MLP contribute exactly zero, so the logits
   are a pure function of the embedding stream. The test recomputes the
   expected logits in the test file — bf16 widen, RMSNorm, bf16 matmul
   against `lm_head` — and asserts bitwise equality. Under the memcpy
   defect the bf16 bits land in the low half of the f32 word
   (~1e-41 denormals), the epsilon-dominated norm rescales them to
   ~1e-38, and the logits diverge from the reference (ISSUE-LOCAL-
   01M5QBGBM3Z7NT3D9TA0K1HT7T). RED on the mutation, GREEN on `main`.

## Red-first proof

`main` already carries the fixes, so red is proven by mutation in a
scratch copy of the worktree: (a) resize `attn_out` back to `t * h`;
(b) replace the embedding widen with a raw memcpy. Each mutation must turn
its own case red while the other case stays green; the tree is restored
byte-for-byte afterward. Evidence: `docs/bench-evidence/flint-gate-harden-20261025/report.md`.

## Tests

`ctest -R test_flint_forward` (the fast set; `test_flint_gguf` is skipped,
785 s, main's Q4_K_M evidence stands). Commit-style and trailer checkers
over the commit range, and `check-agent-record.py`.

## Stop conditions

- If the synthetic config cannot express `hq * dh > hidden` without a
  production change, report `NEEDS_CONTEXT` instead of widening the
  production parser.
- If the mutation does not turn a case red, the case does not detect the
  defect; rework or drop the case and say so in the report.

## Owed

Nothing. Both arms ship reached through `vllm::FlintModel::Forward`, the
same entry point the existing wave-1 cases use.

## Outcome

(filled at DONE)
