# Independent review repairs

Date: 29 October 2026
Issue: `ISSUE-LOCAL-01M63MYTWBYSRRCEMAY20QR08M`
Reviewed base: `bc9354cc300aa875ec8a8d9faf0962bb004504a2`

The original C11 and C++20 callers included both headers. Removing the forwarding
include from a scratch `lumenengine.h` left both syntax checks green. Each new
caller now includes `lumenengine.h` alone. Separate legacy callers include
`vllm.h` alone. The package gate exercises both languages for both names.

Commands: `cc -std=c11 -Werror -fsyntax-only -I<headers> tests/branding/consumer.c`
and the C++20 equivalent. The reviewed callers exited 0 against the broken
scratch header. The repaired callers both exited 1. The production header stayed
unchanged. Both repaired callers exited 0 against production headers.

The original command checker compared the two version outputs without a pin.
Both scratch server binaries replaced `vllm.cpp` with `wrongver`. The original
checker exited 0. The repaired checker exited 1 with `legacy server version
changed`. The repaired checker exited 0 on unchanged production commands.
The historical output comes from the existing tracked evidence artifact.

Mutation binaries and scratch headers were separate copies. No production
artifact or header was modified. The operator authorized fast-forwarding the branding task branch to the repair.
`python3 tests/branding/run_gate.py /tmp/lumen-brand-build` then exited 0:
8 build-tree and 7 relocated installed tests passed. Both production command
checks passed. The root build compiled all 4 independent header callers at
`-j4`. Root CTest passed these 4 callers and the command checker, 5/5.
See `repair-full-gate.log` and `repair-root.log` for exact commands and results.

`python3 scripts/check-test-registration.py` and
`python3 scripts/check-agent-record.py` exited 0. The full repository preflight
started before edits and remains PENDING at handoff. Existing baseline failures
remain documented in the owning report. No entire-preflight success is claimed.
