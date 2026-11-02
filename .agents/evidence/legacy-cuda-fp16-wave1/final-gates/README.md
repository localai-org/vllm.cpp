# Wave 1 final verification disposition

Integration source: `2f195a4c3`. This record changes no product or test code.
The broad independent review at `6c785c4a8` found a cached RoPE dtype-matrix
gap and a stale comment. It reported no numerical implementation defect.
The scoped review of repair head `4c416296c` passes and resolves both findings.
The operator independently reran the repaired scalar suite: 14 cases and 3428 assertions pass.

## Results and limits

- Independent numerical, architecture and mutation evidence passes within Wave1 scope.
  The original broad FAIL remains preserved; the repair PASS resolves its P2/P3 findings.
- The original full preflight against Wave1 head `3c2111d30` is not green.
  It was intentionally interrupted
  before the default eight-worker compile stage. Its exact FAIL labels are
  test_check_gate_commands, test_rocprof_attach_preflight, tools suites and
  check-device-leakage.py. Its initial role was UNDECLARED; the later helper
  claim does not retroactively change that run.
- The branding baseline and exact DSR registrar accounting were independently
  repaired and integrated after the original source base. Current operator
  reruns pass 61 branding tests and DSR 32 equal to baseline 32. These local passes
  do not erase the original failed run.
- The rocprof invalid-page failure remains failing. The aggregate tools suite
  also contains a unittest error whose traceback was truncated by preflight.
  The retained log does not establish its individual test identity; it remains failing.
- The unchanged standalone tree checker then completed with two workers, exit1:
  44 of 829 translation units fail in 981.2 seconds. Most failures are CUDA12.4
  rejecting the default compute121a target; the scratch nvlink object failure
  is also retained. Actual sm60/61/70 builds are separate passing evidence.
- Full inherited commit trailers fail on eight named historical commits.
  The new implementation/evidence range passes. No existing refs were rewritten.
  Full-range writing style passes.
- Canonical record checks pass. This final record also receives its own offline check.

Primary FP16 model execution and automatic dtype propagation remain unreached,
owned by Wave4. Quant projection and attention/KV waves remain pending.
This record claims no model token parity, performance floor, modern/Volta
hardware runtime, complete preflight PASS, or lifecycle DONE transition.

The JSON manifest hashes every retained source log/report and the integrated
source/test files. The complete original preflight remains at its recorded/tmp
path; the tracked excerpt preserves failure and skip names without its large
ELF diagnostics. The full standalone compile log is retained because it is small.

Reproduction commands: `bash scripts/agent-preflight.sh` for the standard lane;
`python3 scripts/check-tree-compiles.py --base db1d71c9b7352a07983b408b2e755e853b4534f0 --jobs 2`
for the bounded-worker unchanged checker; `flock /home/user/gpu.lock SCALAR_BINARY`
for the scalar gate, using the source and binary manifests in the parent evidence. The standalone log also records one unstaged file;
its source-state report is preserved rather than described as a clean immutable gate.
These are provenance recipes, not fresh execution claims in this documentation task.
