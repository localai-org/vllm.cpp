# Legacy CUDA metadata regression repair

Issue: `ISSUE-LOCAL-01M63NZRF5VEMZHJ9F3NQM27FF`.
Spec-first commit: `be9784105`.

The repair changes tests only. It invokes the registered `CUDA_LEGACY`
selector with valid and invalid geometry, storage, and request modes. The
production descriptor and newer GPU selection remain unchanged.

The inherited suite passes 25 cases and 148 assertions after adding 3 cases.
Independent review found five mutations that escaped the earlier 22-case
suite. The new tests reject those mutations: widened head geometry, arbitrary
block geometry, accepted KV connectors, removed bfloat16 KV acceptance, and
removed invalid cache-block refusal. Additional descriptor and inherited-mode
mutations pin the other assertions in the new cases.

The scratch harness compiles the actual production platform and descriptor
objects, plus the complete host selector suite. It links a frozen production
archive with those replacement objects. No GPU workload runs in this host gate.
The full Release build and public CUDA gate remain separate operator gates.

The first mutation sweep exhausted scratch storage because each mutation kept
a complete archive. It stopped before producing a false result. The retained
sweep reuses one working archive and executable. Log files retain each result.
This bounded scratch harness does not change the repository test harness.

The owning issue remains open until landing. Real-model correctness,
performance, P100 runtime, and Volta hardware validation remain pending.

## Results

Run `python3 /tmp/metadata-verify.py` with `scratch-lib.py` materialized as
`/tmp/metadata-lib.py` and the declared frozen archive/build inputs. The first
sweep reaches the five requested failures and the extra declaration controls.
An encoder mutation initially failed to compile because its removed predicate
left an unused parameter under warnings-as-errors. The retained recipe casts
that parameter to void. The resumed sweep tests encoder and nine inherited
mode predicates, then restores the production descriptor.

The complete host suite passes 25 cases and 148 assertions before and after
20 deliberate predicate mutations. Each mutant exits 1 because the test detects
the changed behavior. See the individual `*-run.log` files. No compile failure
is counted as a detected mutation. The retained verifier expresses the complete
corrected sweep; the raw logs come from the initial and resumed runs.

`python3 scripts/check-agent-record.py` and `git diff --check` exit 0.
The full Release CUDA build remains the operator's pending gate.
