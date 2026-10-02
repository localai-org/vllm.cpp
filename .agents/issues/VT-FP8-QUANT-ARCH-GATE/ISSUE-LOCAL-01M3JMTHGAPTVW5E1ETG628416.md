ID: ISSUE-LOCAL-01M3JMTHGAPTVW5E1ETG628416
Title: check-cuda-op-arch-gate compares a native-separator path against a POSIX literal, so it reports a correct tree as having no registration AND a duplicate one
Row: VT-FP8-QUANT-ARCH-GATE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-27
Updated: 2026-09-27
Closed: -

## Problem

scripts/check-cuda-op-arch-gate.py is red on a tree that satisfies every one of its four assertions, and it is red because the checker compares a native-separator path against a POSIX literal.

The registrations are correct. src/vt/cuda/cuda_quant_fp8.cu:200 registers kQuantFp8Static for kCUDA and :202 registers kQuantFp8Group, both inside Registrar::Registrar() at preprocessor-conditional depth 0, and the file is in the unconditional target_sources(vllm PRIVATE ...) list under if(VLLM_CPP_CUDA). The checker's own --report says so: registrations {'src\vt\cuda\cuda_quant_fp8.cu': [(200, 0)]} and {'src\vt\cuda\cuda_quant_fp8.cu': [(202, 0)]}, unconditional CUDA sources 35, home in that set True.

The failure is the dict key. cuda_registrations() builds it with str(path.relative_to(root)), which on Windows yields backslashes, while REQUIRED spells the home "src/vt/cuda/cuda_quant_fp8.cu" with forward slashes -- a spelling that comes from the CMake source list, which the same checker parses. So regs.get(home, []) misses the file the checker just found, and the gate reports BOTH halves of the contradiction at once: "expected exactly ONE live RegisterOp ... found 0" from the home lookup, and "is ALSO registered for kCUDA here" from the exclusivity loop, which sees the home file as some other file. One defect, two errors per op, four errors in total, and both messages name the file whose registration the checker has already located and measured.

The same expression, str(path.relative_to(...)), is the shape to look for elsewhere: scripts/check-agent-record.py:84, check-attention-rung-consistency.py:222, check-oracle-denominator-flags.py:188, check-oracle-pins.py:427 and check-test-registration.py:631. Only sites that compare the result against a POSIX literal are affected; the rest are display-only and stay as they are.

Measured on Windows, Python 3.12, at upstream/main 1de097c46. The gate has never been red for this reason on a POSIX host, where str() already yields forward slashes, which is why CI does not see it.

## Resolution

-

## Executed verification, 2026-10-02 (Linux x86_64)

Requested by the mudler-agent review (executed Windows-path focused test and
mutation). The native host is POSIX, so the Windows spelling was emulated
faithfully: `Path.relative_to` was monkeypatched to return `PureWindowsPath`,
which makes `str()` produce backslashes exactly as a Windows host does, and
the checker was driven through its real `check()` entry point.

- Pre-fix checker (`HEAD^` revision) under Windows-spelled paths: 4 errors --
  for each of kQuantFp8Static and kQuantFp8Group, `expected exactly ONE live
  RegisterOp ... found 0` together with `src\vt\cuda\cuda_quant_fp8.cu ...
  is ALSO registered for kCUDA here`. Both halves of the reported
  contradiction, with the backslash key visible verbatim.
- Fixed checker under the same emulation: 0 errors.
- POSIX host, unmodified run: `check-cuda-op-arch-gate: OK (2 op(s) pinned to
  an unconditional CUDA TU)`; `tests/scripts/test_check_cuda_op_arch_gate.py`
  14/14 OK, including the four gate mutations.

