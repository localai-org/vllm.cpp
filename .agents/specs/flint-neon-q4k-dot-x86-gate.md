# SPEC — `FLINT-NEON-Q4K-DOT-X86-GATE` — the dispatch test's auto-arm needs a DotProd host

- Owning row: `FLINT-NEON-Q4K-DOT-X86-GATE`
- Kind: bug (a test-only fix: the NEON q4_K dispatch test's auto-arm assertion fails on x86 CI lanes that have no NEON tier)
- Issue: created in the same flow (row-scoped local issue)

## The failure

`test_neon_q4k_dot.cpp:246` on the x86 CI lanes
(`build-test-cpu`, ubuntu-latest, no NEON tier):

```
CHECK( vt::cpu::SelectQuantQ4KVecDot(portable) == vt::cpu::QuantQ4KNeonVecDot() )
values: CHECK( 1 == nullptr )
```

`QuantQ4KNeonVecDot()` IS the HWCAP dotprod probe — it returns `nullptr`
on a non-dotprod host. The auto-arm assertion asserts that auto mode
resolves to the NEON kernel — meaningless on a host where the NEON tier
does not exist. The test's own tier-availability case one TEST_CASE up
(`:149`, "NEON Q4_K dot tier not live on this host; bit-identity
skipped") already guards its body with exactly this check; the dispatch
case's auto-arm lacks the same guard.

This is not theoretical: the FLINT-NEON-Q4K-DOT PR (#3537 lineage)
passed on the arm64 authoring host and failed on the x86 CI lane, reding
`build-test-cpu` on the merged main.

## The fix (one line + comment)

Guard the auto-arm on the NEON tier's availability:

```cpp
if (!forced_portable && vt::cpu::QuantQ4KNeonVecDot() != nullptr) {
  CHECK(vt::cpu::SelectQuantQ4KVecDot(portable) == vt::cpu::QuantQ4KNeonVecDot());
}
```

On a DotProd host the arm runs as before (the auto-mode assertion);
on a non-DotProd host the arm is skipped — the portable resolution is
already asserted by the first CHECK in the same TEST_CASE, so nothing
is lost.

## Red-first

The failure IS the red: `1 == nullptr` on x86 CI (run 42172515466, job
127743391331, `test_neon_q4k_dot.cpp:246`). Green-after: the suite
green on x86 (no NEON tier → the guarded arm skipped, everything else
passes) AND green on a DotProd host (this aarch64 box → the arm runs
and passes). Both recorded.

## Gates

check-agent-record; test_neon_q4k_dot on x86 (green — the fix); on a
DotProd host (green — the arm runs); commit style; trailers; preflight
(named host dotnet failure).

## Git integration

One pull request, `row/FLINT-NEON-Q4K-DOT-X86-GATE`, spec committed
first.
