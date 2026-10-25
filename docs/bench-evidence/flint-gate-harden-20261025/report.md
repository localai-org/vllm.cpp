# Flint gate hardening — red-first mutation evidence (2026-10-25)

Row: MODEL-TEXT-FLINT · Issue: ISSUE-LOCAL-01M5SQG0389SAHK9NY9ENR4AHP
Spec: `.agents/specs/flint-gate-harden.md`
Base: `97dbf7d09` (origin/main) · Branch: `row/FLINT-GATE-HARDEN`

`main` already carries both fixes, so red is proven by mutation: each fix
site in `src/vllm/model_executor/models/flint.cpp` is reverted in a scratch
edit, the tree rebuilt, the gate rerun, and the tree restored byte-for-byte
with `git checkout --`.

## New cases (in `tests/vllm/models/test_flint_forward.cpp`)

- `flint: attn_out is sized for the head layout, not the hidden stream` —
  config `hq*dh = 16 > hidden = 4`; asserts bitwise causal-prefix equality
  (row i of a t-token forward == forward of the first i+1 tokens).
- `flint: the embedding lookup widens bf16 bits, it does not memcpy them` —
  bf16-exact embedding row `[1.0, 0.5, 0.25, 0.125]`, a zero-contribution
  layer, logits compared bitwise against an in-test reference (widen →
  RMSNorm → bf16 matmul).

## Green (unmutated `main`, commit 97dbf7d09 + test commit)

`ctest -R test_flint_forward`: 100% tests passed (2 runs, before and after
the mutations).

## Mutation A — attn_out resized back to `t * h` (pre-fix sizing)

- Edit: `std::vector<float> attn_out(static_cast<size_t>(t) * hq * dh)` →
  `t * h` at `flint.cpp:408`.
- Result: RED. The buffer case aborts with SIGABRT (glibc heap-corruption
  check fires on the out-of-bounds writes), before the remaining cases run.
- Log: `red-mutation-A-attn-out-t-h.log`

## Mutation B — embedding widen replaced with a raw memcpy

- Edit: the per-element `Bf16(emb[tok*h+d])` widen at `flint.cpp:354-355`
  → `std::memcpy` of the bf16 bits into the f32 stream.
- Result: RED. Only the embedding case fails — 16/16 logit CHECKs mismatch
  (the widened reference is O(1); the memcpy path yields the epsilon-
  rescaled denormal pattern). The buffer case stays green under B, so each
  mutation is caught by exactly its own case.
- Log: `red-mutation-B-embedding-memcpy.log`

## Green baselines

Three `ctest -R test_flint_forward` runs — before mutation A, between the
mutations, and after mutation B — each 100% passed
(`green-baseline.log`). The unmutated path was additionally checked with
AddressSanitizer (an ASAN-instrumented driver against the -O3
`libvllm.a`): no heap error.

## Note: a stale-build false alarm during this evidence run

The first mutation cycle produced an intermittent SIGABRT on the GREEN
code. Root cause: after `git checkout --` restored `flint.cpp`, ninja
did not recompile it (the restore did not advance the mtime past the
mutated object), so `libvllm.a` kept carrying the mutated object and the
binary flipped between mutated and fixed behaviour across rebuilds. Every
rebuild in the final cycle above is preceded by an explicit `touch` of
`flint.cpp`. A builder that restores source bytes must force the
recompile; timestamp-only rebuild detection is unsound for byte-identical
restores.


## Verdict

Both cases are red on the pre-fix code and green on `main`. The wave-1 gap
is closed: a regression of either fix site now fails `ctest -R
test_flint_forward`.

`test_flint_gguf` (785 s) was skipped per the unit's scope; main's
Q4_K_M band evidence stands.
